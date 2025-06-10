"""Color analysis module for finding text colors in images.

This module analyzes colors within detected text regions to identify the colors
that represent text watermarks. It uses k-means quantization to reduce the
color space and then selects the most appropriate colors based on either
target matching or "grayness" criteria.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .utils import ImageArray, BBox, Color, setup_logger, color_distance

logger = setup_logger(__name__)

def find_text_colors(
    image: ImageArray, 
    bbox: BBox, 
    target_color: Optional[Color] = None
) -> Tuple[Color, List[Color]]:
    """Find text colors within a bounding box using color quantization.
    
    This is the main entry point for color analysis. It quantizes colors
    within the bounding box and selects the most appropriate color based
    on either target matching or grayness criteria.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height) to analyze
        target_color: Optional target BGR color to match against
        
    Returns:
        Tuple of:
        - Representative BGR color tuple (the quantized color)
        - List of original BGR colors that were assigned to this quantized color
    """
    # TODO: Make number of quantized colors configurable (currently hardcoded to 12)
    return _get_most_common_color(image, bbox, num_colors=12, target_color=target_color)

def hex_to_bgr(hex_color: str) -> Color:
    """Convert hex color string to BGR tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        BGR color tuple
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Parse hex values
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to BGR
    return (rgb[2], rgb[1], rgb[0])

def html_to_bgr(html_color: str) -> Color:
    """Convert HTML color name to BGR tuple.
    
    Args:
        html_color: HTML color name (e.g., "red", "ghostwhite")
        
    Returns:
        BGR color tuple
    """
    # Convert HTML name to hex using PIL
    from PIL import ImageColor
    rgb = ImageColor.getrgb(html_color)
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def _get_most_common_color(
    image: ImageArray, 
    bbox: BBox, 
    num_colors: int = 12, 
    target_color: Optional[Color] = None
) -> Tuple[Color, List[Color]]:
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
    
    # Get top colors by frequency (up to num_colors)
    top_indices = np.argsort(counts)[-num_colors:]
    top_colors = centers[unique_labels[top_indices]]
    top_counts = counts[top_indices]
    
    # Calculate variance between R,G,B channels for each color
    variances = np.var(top_colors, axis=1)
    
    # Log the top colors and their properties
    logger.info(f"\nTop {len(top_colors)} colors by frequency:")
    logger.info("Rank  BGR Color    Variance    Count    % of Pixels")
    logger.info("------------------------------------------------")
    for i, (color, var, count) in enumerate(zip(top_colors, variances, top_counts)):
        percent = (count / len(pixels)) * 100
        logger.info(f"{i+1:2d}    {tuple(color)}    {var:8.2f}    {count:6d}    {percent:6.1f}%")
    
    if target_color is not None:
        # Find which of the top colors is closest to the target
        target_float = np.float32([target_color])
        distances = np.linalg.norm(top_colors - target_float, axis=1)
        closest_idx = np.argmin(distances)
        selected_color = tuple(top_colors[closest_idx])
        selected_label_idx = closest_idx
        logger.info(f"\nTarget color: {target_color}")
        logger.info(f"Closest match: {selected_color} (distance: {distances[closest_idx]:.2f})")
    else:
        # Find the "grayest" color among top colors
        grayest_idx = np.argmin(variances)
        selected_color = tuple(top_colors[grayest_idx])
        selected_label_idx = grayest_idx
        logger.info(f"\nSelected color {selected_color} with variance {variances[grayest_idx]:.2f}")
    
    # Get all original colors that were assigned to this quantized color
    selected_label = unique_labels[top_indices[selected_label_idx]]
    original_colors = pixels[labels.flatten() == selected_label]
    
    # Count occurrences of each original color
    unique_colors, color_counts = np.unique(original_colors, axis=0, return_counts=True)
    
    # Filter out colors that appear in fewer than MIN_PIXELS pixels
    # TODO: Make minimum pixel threshold configurable
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