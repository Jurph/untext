#!/usr/bin/env python3
"""Text hunting sandbox for detecting text watermarks based on color analysis.

This script implements a color-based approach to detect text watermarks by:
1. Finding text regions using OCR
2. Analyzing color patterns in those regions
3. Creating masks for potential text colors
4. Validating which colors actually contain text
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Any
import logging
import sys
from untext.detector import TextDetector
import argparse
from untext.preprocessor import preprocess_image_array
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
Color = Tuple[int, int, int]  # RGB color tuple

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def get_largest_text_region(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the largest text region in the image using DocTR.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Tuple of (x, y, width, height) for the largest text region
    """
    logger.info("Detecting text regions in image...")
    detector = TextDetector()
    mask, detections = detector.detect(image)
    
    if not detections:
        logger.warning("No text regions found, using failover region")
        # Use lower-right corner: 1/4 width, 1/16 height
        h, w = image.shape[:2]
        failover_w = w // 4
        failover_h = h // 16
        return (w - failover_w, h - failover_h, failover_w, failover_h)
    
    # Find largest region by area
    # Each detection has a 'geometry' key with polygon points
    largest_region = max(detections, key=lambda d: cv2.contourArea(d['geometry']))
    
    # Convert polygon to bounding box
    x, y, w, h = cv2.boundingRect(largest_region['geometry'].astype(np.int32))
    return (x, y, w, h)

def dilate_bbox(bbox: Tuple[int, int, int, int], dilation: int = 4, image_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """Dilate a bounding box by the specified amount, clamping to image bounds if provided.
    
    Args:
        bbox: Tuple of (x, y, width, height)
        dilation: Number of pixels to dilate by
        image_shape: Optional tuple of (height, width) to clamp coordinates
        
    Returns:
        Dilated bounding box as (x, y, width, height)
    """
    x, y, w, h = bbox
    x1 = x - dilation
    y1 = y - dilation
    x2 = x + w + dilation
    y2 = y + h + dilation
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        h, w = image_shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
    
    return (x1, y1, x2 - x1, y2 - y1)

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color (#RRGGBB) to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def html_to_bgr(html_color: str) -> Tuple[int, int, int]:
    """Convert HTML color name to BGR tuple."""
    # Convert HTML name to hex using PIL
    from PIL import ImageColor
    rgb = ImageColor.getrgb(html_color)
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two BGR colors."""
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

def get_most_common_color(image: np.ndarray, bbox: Tuple[int, int, int, int], num_colors: int = 12, target_color: Optional[Tuple[int, int, int]] = None) -> Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    """Get the most common color in the specified region, using color quantization.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height)
        num_colors: Number of colors to quantize to (default: 12)
        target_color: Optional target BGR color to match against
        
    Returns:
        Tuple of:
        - Representative BGR color tuple (the quantized color)
        - List of original BGR colors that were assigned to this quantized color
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    # Reshape ROI to 2D array of pixels
    pixels = roi.reshape(-1, 3)
    
    # Convert to float32 for k-means
    pixels_float = np.float32(pixels)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels_float, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to uint8
    centers = np.uint8(centers)
    
    # Count occurrences of each color
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Get top 12 colors by frequency
    top_indices = np.argsort(counts)[-12:]
    top_colors = centers[unique_labels[top_indices]]
    top_counts = counts[top_indices]
    
    # Calculate variance between R,G,B channels for each color
    variances = np.var(top_colors, axis=1)
    
    # Log the top 12 colors and their properties
    logger.info("\nTop 12 colors by frequency:")
    logger.info("Rank  BGR Color    Variance    Count    % of Pixels")
    logger.info("------------------------------------------------")
    for i, (color, var, count) in enumerate(zip(top_colors, variances, top_counts)):
        percent = (count / len(pixels)) * 100
        logger.info(f"{i+1:2d}    {tuple(color)}    {var:8.2f}    {count:6d}    {percent:6.1f}%")
    
    if target_color is not None:
        # Find which of the top 12 colors is closest to the target
        target_float = np.float32([target_color])
        distances = np.linalg.norm(top_colors - target_float, axis=1)
        closest_idx = np.argmin(distances)
        selected_color = tuple(top_colors[closest_idx])
        logger.info(f"\nTarget color: {target_color}")
        logger.info(f"Closest match: {selected_color} (distance: {distances[closest_idx]:.2f})")
    else:
        # Find the "grayest" color among top 12
        grayest_idx = np.argmin(variances)
        selected_color = tuple(top_colors[grayest_idx])
        logger.info(f"\nSelected color {selected_color} with variance {variances[grayest_idx]:.2f}")
    
    # Get all original colors that were assigned to this quantized color
    selected_label = unique_labels[top_indices[closest_idx if target_color is not None else grayest_idx]]
    original_colors = pixels[labels.flatten() == selected_label]
    
    # Count occurrences of each original color
    unique_colors, color_counts = np.unique(original_colors, axis=0, return_counts=True)
    
    # Filter out colors that appear in fewer than 5 pixels
    MIN_PIXELS = 2
    mask = color_counts >= MIN_PIXELS
    filtered_colors = unique_colors[mask]
    filtered_counts = color_counts[mask]
    
    # Log color filtering results
    logger.info(f"Found {len(unique_colors)} original colors")
    logger.info(f"After filtering colors with < {MIN_PIXELS} pixels: {len(filtered_colors)} colors remain")
    
    # Log the distribution of remaining colors
    logger.info("\nColor frequency distribution after filtering:")
    logger.info("Count Range    Number of Colors")
    logger.info("-------------------------------")
    ranges = [(10, 50), (51, 100), (101, 500), (501, 1000), (1001, None)]
    for start, end in ranges:
        if end is None:
            count = np.sum(filtered_counts >= start)
            logger.info(f"{start:4d}+          {count:4d}")
        else:
            count = np.sum((filtered_counts >= start) & (filtered_counts < end))
            logger.info(f"{start:4d}-{end:4d}     {count:4d}")
    
    return selected_color, [tuple(color) for color in filtered_colors]

def create_color_mask(image: ImageArray, target_colors: List[Color], tolerance: int = 2) -> MaskArray:
    """Create a binary mask for pixels matching any of the target colors.
    
    Args:
        image: Input image in BGR format
        target_colors: List of BGR color tuples to match
        tolerance: Color matching tolerance
        
    Returns:
        Binary mask as H×W uint8 numpy array
    """
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # For each target color, create a mask and combine
    for target_color in target_colors:
        # Create color bounds
        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])
        
        # Create mask for this color
        color_mask = cv2.inRange(image, lower, upper)
        
        # Combine with main mask
        mask = cv2.bitwise_or(mask, color_mask)
    
    return mask

def contains_text(mask: MaskArray) -> bool:
    """Check if a binary mask contains text using DocTR.
    
    Args:
        mask: Binary mask as H×W uint8 numpy array
        
    Returns:
        True if text is detected, False otherwise
    """
    logger.info("Testing color mask for text content...")
    
    # Initialize detector
    detector = TextDetector()
    
    # Convert binary mask to BGR image
    bgr_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Get detections
    _, detections = detector.detect(bgr_image)
    
    # Check if any text was detected
    return len(detections) > 0

def create_final_mask(image: ImageArray, text_colors: Set[Color]) -> MaskArray:
    """Create final mask using detected text colors.
    
    Args:
        image: Input image as H×W×3 RGB uint8 numpy array
        text_colors: Set of RGB colors that contain text
        
    Returns:
        Binary mask as H×W uint8 numpy array (0 for non-text colors, 255 for text colors)
    """
    logger.info("Creating final mask from detected text colors...")
    
    # Create mask where pixels match any text color
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for color in text_colors:
        color_mask = np.all(image == color, axis=2)
        mask[color_mask] = 255
        
    return mask

def anchor_connected_components(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Filter connected components based on their distance from the bounding box center.
    Uses the bbox height to determine the vertical threshold, allowing full horizontal freedom.
    
    Args:
        mask: Binary mask where white pixels (255) represent potential text regions
        bbox: Tuple of (x, y, width, height) from original text detection
        
    Returns:
        Filtered binary mask
    """
    # Create a copy of the mask to modify
    filtered_mask = np.zeros_like(mask)
    
    # Get the vertical center and height from the bbox
    center_y = bbox[1] + bbox[3] // 2
    box_height = bbox[3]
    
    # Allow half a box height above and below the box
    vertical_threshold = box_height // 2
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask)
    
    # For each component, check if it's within the vertical threshold
    for label in range(1, num_labels):
        # Get all pixels with this label
        component_pixels = np.where(labels == label)
        
        # Calculate vertical distances from each pixel to the center
        vertical_distances = np.abs(component_pixels[0] - center_y)
        
        # Keep component if any pixel is within vertical threshold
        # (no horizontal restriction)
        if np.any(vertical_distances <= vertical_threshold):
            filtered_mask[labels == label] = 255
    
    return filtered_mask

def clean_up_mask(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Clean up the mask using morphological operations.
    
    Args:
        mask: Binary mask (H×W uint8)
        bbox: Tuple (x, y, w, h) from the original text detection
    """
    # 1. Morphological opening with 2x2 ellipse kernel
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Morphological closing with 3x3 kernel
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. First dilation by 6px
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    mask = cv2.dilate(mask, kernel_dilate)
    
    # 4. Erode by 3px
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel_erode)
    
    # 5. Blur with 3x3 kernel
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 6. Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(mask, (0, 0), 3.0)
    mask = cv2.addWeighted(mask, 1.5, gaussian, -0.5, 0)
    
    # 7. Final dilation by 6px
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 8. Filter out components far from the text region
    mask = anchor_connected_components(mask, bbox)
    
    return mask

def apply_telea_inpainting(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply TELEA inpainting to the image."""
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def apply_lama_inpainting(image: np.ndarray, mask: np.ndarray, subregion: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Apply LaMa inpainting to the image."""
    from untext.lama_inpainter import LamaInpainter
    inpainter = LamaInpainter()
    return inpainter.inpaint(image, mask, subregion)

def process_single_image(input_path: Path, output_dir: Path, target_color: Optional[Tuple[int, int, int]] = None) -> None:
    """Process a single image file."""
    logger.info(f"Processing {input_path.name}...")
    
    # Load and preprocess image
    logger.info("Applying preprocessing...")
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    preprocessed = preprocess_image_array(image)
    if preprocessed is None:
        raise ValueError("Preprocessing failed")
    
    # Detect text regions
    logger.info("Detecting text regions in image...")
    bbox = get_largest_text_region(preprocessed)
    logger.info(f"Text region: {bbox}")
    bbox = dilate_bbox(bbox, 4, image.shape[:2])
    logger.info(f"Dilated region: {bbox}")
    
    # Get target color and create mask - use original image for color analysis
    target_color, original_colors = get_most_common_color(image, bbox, target_color=target_color)
    logger.info(f"Target color: {target_color}")
    logger.info(f"Found {len(original_colors)} original colors to mask")
    
    # Create and clean up the mask
    mask = create_color_mask(image, original_colors)
    mask = clean_up_mask(mask, bbox)
    
    # Get bounding box of white region in mask
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("No white pixels found in mask")
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    mask_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Dilate the mask bbox by 64px, clamping to image bounds
    mask_bbox = dilate_bbox(mask_bbox, 64, image.shape[:2])
    
    # Convert to subregion format (x1, y1, x2, y2)
    x, y, w, h = mask_bbox
    subregion = (x, y, x + w, y + h)
    logger.info(f"Using subregion for inpainting: {subregion}")
    
    # Save mask for debugging
    mask_path = output_dir / f"{input_path.stem}_mask{input_path.suffix}"
    cv2.imwrite(str(mask_path), mask)
    logger.info(f"Saved mask to {mask_path}")
    
    # Apply inpainting
    logger.info("Applying inpainting...")
    inpainted_lama = apply_lama_inpainting(image, mask, subregion=subregion)
    
    # Save results
    output_path = output_dir / f"{input_path.stem}_lama{input_path.suffix}"
    cv2.imwrite(str(output_path), inpainted_lama)
    logger.info(f"Saved result to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Remove text from images using color-based masking and inpainting.")
    parser.add_argument("input_path", help="Path to input image or directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save output images")
    parser.add_argument("--hex-answer", help="Target color in hex format (e.g., #112233)")
    parser.add_argument("--html-answer", help="Target color as HTML color name (e.g., ghostwhite)")
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert target color if provided
    target_color = None
    if args.hex_answer:
        target_color = hex_to_bgr(args.hex_answer)
    elif args.html_answer:
        target_color = html_to_bgr(args.html_answer)
    
    if input_path.is_file():
        process_single_image(input_path, output_dir, target_color)
        num_images = 1
    else:
        # Process all images in directory
        image_files = [f for f in input_path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
        logger.info(f"Found {len(image_files)} images to process")
        num_images = len(image_files)
        
        for image_file in image_files:
            try:
                process_single_image(image_file, output_dir, target_color)
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue
    
    # Calculate and log timing information
    total_time = time.time() - start_time
    avg_time = total_time / num_images
    logger.info("\nProcessing complete:")
    logger.info(f"Total elapsed time: {total_time:.1f} seconds")
    logger.info(f"Average time per image: {avg_time:.1f} seconds")
    logger.info(f"Images processed: {num_images}")

if __name__ == "__main__":
    main() 