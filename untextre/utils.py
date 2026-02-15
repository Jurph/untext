"""Shared utilities and type definitions for untextre."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import logging

# Type aliases for clarity
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
Color = Tuple[int, int, int]  # BGR color tuple
BBox = Tuple[int, int, int, int]  # (x, y, width, height)
ImagePath = Union[str, Path]

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ---------------------------------------------------------------------------
# Confidence threshold constants
# ---------------------------------------------------------------------------
# There are three distinct confidence values in this system.  They are
# intentionally different and serve different purposes.
#
# MODEL_CONFIDENCE_FLOOR (0.1):
#     The internal threshold baked into the DocTR model at initialization.
#     Set deliberately low so that ALL plausible detections survive the
#     model's own filtering.  The *real* user-facing threshold is applied
#     as a post-filter at detection time -- this lets users adjust the
#     slider without re-initializing the model (which wastes VRAM and
#     takes 30+ seconds).
#
# CLI_DEFAULT_CONFIDENCE (0.3):
#     Conservative default for CLI / batch processing where there is no
#     human in the loop.  A higher threshold reduces false positives at
#     the cost of potentially missing faint watermarks.
#
# WEB_DEFAULT_CONFIDENCE (0.025):
#     Aggressive default for the Streamlit web UI where the user can see
#     results and adjust interactively.  Because consensus detection
#     requires 2+ detectors to agree, even very low thresholds rarely
#     produce false positives.  Values as low as 0.03 have been found
#     effective in practice.
# ---------------------------------------------------------------------------
MODEL_CONFIDENCE_FLOOR: float = 0.1
CLI_DEFAULT_CONFIDENCE: float = 0.3
WEB_DEFAULT_CONFIDENCE: float = 0.025

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(level)
    return logger

def load_image(image_path: ImagePath) -> ImageArray:
    """Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array in BGR format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

def save_image(image: ImageArray, output_path: ImagePath, quality: int = 97) -> None:
    """Save an image to file with quality control.
    
    Args:
        image: Image array in BGR format
        output_path: Path where to save the image
        
    Raises:
        ValueError: If image cannot be saved
    """
    output_path = Path(output_path)
    
    # Set compression parameters based on file extension
    if output_path.suffix.lower() in {'.jpg', '.jpeg'}:
        # JPEG with specified quality
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_path.suffix.lower() == '.png':
        # PNG with high compression (0-9, where 9 is max compression)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 8]  # High compression for smaller files
    else:
        # Default parameters for other formats
        params = []
    
    success = cv2.imwrite(str(output_path), image, params)
    if not success:
        raise ValueError(f"Could not save image to: {output_path}")

def get_image_files(path: Path) -> List[Path]:
    """Get list of image files from path (file or directory).
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of image file paths
    """
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        else:
            return []
    
    return [f for f in path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]

def clamp_bbox_to_image(bbox: BBox, image_shape: Tuple[int, int]) -> BBox:
    """Clamp bounding box coordinates to stay within image bounds.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        image_shape: Image shape as (height, width)
        
    Returns:
        Clamped bounding box
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape
    
    # Clamp coordinates
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # Adjust width and height to stay in bounds
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    return (x, y, w, h)

def dilate_bbox(bbox: BBox, dilation: int, image_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """Dilate a bounding box by the specified amount.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        dilation: Number of pixels to dilate by
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Dilated bounding box
    """
    x, y, w, h = bbox
    x1 = x - dilation
    y1 = y - dilation
    x2 = x + w + dilation
    y2 = y + h + dilation
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        img_h, img_w = image_shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
    
    return (x1, y1, x2 - x1, y2 - y1)


def pad_bbox_to_multiple(bbox: BBox, multiple: int = 4, image_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """Pad a bounding box to ensure width and height are divisible by a given multiple.
    
    This prevents pixel alignment issues in neural networks that require
    dimensions divisible by specific values (e.g., 4, 8, 16, 32).
    Padding is applied symmetrically when possible.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        multiple: Value that dimensions must be divisible by (default: 4)
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Padded bounding box with width and height divisible by multiple
    """
    if multiple <= 0:
        raise ValueError(f"Multiple must be positive, got {multiple}")
    
    x, y, w, h = bbox
    
    # Calculate padding needed to make dimensions divisible by multiple
    w_pad = (multiple - (w % multiple)) % multiple  # 0 if already divisible
    h_pad = (multiple - (h % multiple)) % multiple
    
    # Apply padding symmetrically when possible
    w_pad_left = w_pad // 2
    w_pad_right = w_pad - w_pad_left
    h_pad_top = h_pad // 2
    h_pad_bottom = h_pad - h_pad_top
    
    # Calculate new coordinates
    new_x = x - w_pad_left
    new_y = y - h_pad_top
    new_w = w + w_pad
    new_h = h + h_pad
    
    # Clamp to image bounds if provided
    # Track which edges touch the image boundary - we cannot lose pixels at those edges
    # and we cannot extend past those edges
    touches_left = False
    touches_right = False
    touches_top = False
    touches_bottom = False
    
    if image_shape is not None:
        img_h, img_w = image_shape
        
        # Adjust if padding extends beyond image bounds
        if new_x < 0:
            # Shift right padding to compensate
            w_pad_right += abs(new_x)
            new_x = 0
            new_w = w + w_pad  # Recalculate width with adjusted padding
            
        if new_y < 0:
            # Shift bottom padding to compensate  
            h_pad_bottom += abs(new_y)
            new_y = 0
            new_h = h + h_pad  # Recalculate height with adjusted padding
            
        if new_x + new_w > img_w:
            new_w = img_w - new_x
            touches_right = True  # Right edge is at image boundary
            
        if new_y + new_h > img_h:
            new_h = img_h - new_y
            touches_bottom = True  # Bottom edge is at image boundary
        
        # Check if bbox is at left/top edge AFTER all clamping
        # (may have started at 0, or been clamped to 0)
        touches_left = (new_x == 0)
        touches_top = (new_y == 0)
    
    # Ensure final dimensions are divisible by multiple
    # CRITICAL: If bbox touches image edge, we must NOT round down (would lose edge pixels)
    # Instead, round UP and shift x/y left/up to compensate - BUT only if there's room
    
    w_remainder = new_w % multiple
    h_remainder = new_h % multiple
    
    if w_remainder != 0:
        if touches_right:
            # Can't extend right, so try to extend left
            w_needed = multiple - w_remainder
            if touches_left:
                # Can't extend left either! Bbox spans full image width.
                # Accept non-mod-8 width; SimpleLama will pad internally.
                pass
            else:
                # Extend left: shift x and increase width
                new_x = max(0, new_x - w_needed)
                new_w = new_w + w_needed
        else:
            # Safe to round down (right side is padding area, not real content)
            new_w = (new_w // multiple) * multiple
    
    if h_remainder != 0:
        if touches_bottom:
            # Can't extend down, so try to extend up
            h_needed = multiple - h_remainder
            if touches_top:
                # Can't extend up either! Bbox spans full image height.
                # Accept non-mod-8 height; SimpleLama will pad internally.
                pass
            else:
                # Extend up: shift y and increase height
                new_y = max(0, new_y - h_needed)
                new_h = new_h + h_needed
        else:
            # Safe to round down (bottom side is padding area, not real content)
            new_h = (new_h // multiple) * multiple
    
    # Ensure dimensions are at least the minimum
    final_w = max(multiple, new_w)
    final_h = max(multiple, new_h)
    
    return (new_x, new_y, final_w, final_h)


def dilate_by_pixels(image: ImageArray, bbox: BBox, pixels: int) -> BBox:
    """Dilate a bounding box by a specific number of pixels.
    
    Args:
        image: Input image to get dimensions from
        bbox: Bounding box as (x, y, width, height)
        pixels: Number of pixels to dilate by
        
    Returns:
        Dilated bounding box clamped to image bounds
    """
    return dilate_bbox(bbox, pixels, image.shape[:2])

def dilate_by_percent(image: ImageArray, bbox: BBox, percent: float) -> BBox:
    """Dilate a bounding box by a percentage of its current size.
    
    Args:
        image: Input image to get dimensions from
        bbox: Bounding box as (x, y, width, height)
        percent: Percentage to dilate by (e.g., 0.2 for 20%)
        
    Returns:
        Dilated bounding box clamped to image bounds
    """
    x, y, w, h = bbox
    
    # Calculate dilation amount based on bbox dimensions
    dilation_w = int(w * percent / 2)  # Divide by 2 since we dilate in both directions
    dilation_h = int(h * percent / 2)
    
    # Use the average of width and height dilation for uniform expansion
    dilation = (dilation_w + dilation_h) // 2
    
    return dilate_bbox(bbox, dilation, image.shape[:2])

def color_distance(color1: Color, color2: Color) -> float:
    """Calculate Euclidean distance between two BGR colors.
    
    Args:
        color1: First color as BGR tuple
        color2: Second color as BGR tuple
        
    Returns:
        Euclidean distance between colors
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2))))

def calculate_bbox_superset(bboxes: List[BBox], image_shape: Optional[Tuple[int, int]] = None) -> Optional[BBox]:
    """Calculate the superset bounding box that contains all input boxes.
    
    The superset box uses:
    - Leftmost left coordinate (minimum x)
    - Uppermost top coordinate (minimum y)
    - Rightmost right coordinate (maximum x + width)
    - Bottommost bottom coordinate (maximum y + height)
    
    Args:
        bboxes: List of bounding boxes as (x, y, width, height) tuples
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Superset bounding box, or None if bboxes is empty
    """
    if not bboxes:
        return None
    
    # Find the bounding box that contains all boxes
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
    max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
    
    superset = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        superset = clamp_bbox_to_image(superset, image_shape)
    
    return superset 