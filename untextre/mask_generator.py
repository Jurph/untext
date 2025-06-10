"""Mask generation module for creating binary masks from detected text colors.

This module takes detected text colors and generates binary masks that cover
the text regions. It includes morphological operations to clean up and
enhance the masks for better inpainting results.
"""

import cv2
import numpy as np
from typing import List, Tuple

from .utils import ImageArray, MaskArray, Color, BBox, setup_logger

logger = setup_logger(__name__)

def generate_mask(image: ImageArray, text_colors: List[Color], bbox: BBox) -> MaskArray:
    """Generate a binary mask for text regions based on detected colors.
    
    This is the main entry point for mask generation. It creates a color-based
    mask and then cleans it up using morphological operations.
    
    Args:
        image: Input image in BGR format
        text_colors: List of BGR color tuples that represent text
        bbox: Bounding box from original text detection for anchoring
        
    Returns:
        Binary mask as H×W uint8 numpy array (255 = text, 0 = background)
    """
    # Create initial color-based mask
    mask = create_color_mask(image, text_colors)
    
    # Clean up mask with morphological operations
    mask = clean_up_mask(mask, bbox)
    
    logger.info(f"Generated mask from {len(text_colors)} text colors")
    return mask

def create_color_mask(image: ImageArray, target_colors: List[Color], tolerance: int = 2) -> MaskArray:
    """Create a binary mask for pixels matching any of the target colors.
    
    Args:
        image: Input image in BGR format
        target_colors: List of BGR color tuples to match
        tolerance: Color matching tolerance (default: 2)
        
    Returns:
        Binary mask as H×W uint8 numpy array
    """
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # For each target color, create a mask and combine
    for target_color in target_colors:
        # Create color bounds with tolerance
        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])
        
        # Create mask for this color
        color_mask = cv2.inRange(image, lower, upper)
        
        # Combine with main mask
        mask = cv2.bitwise_or(mask, color_mask)
    
    return mask

def clean_up_mask(mask: MaskArray, bbox: BBox) -> MaskArray:
    """Clean up the mask using morphological operations.
    
    This function applies a series of morphological operations to:
    1. Remove small artifacts (opening)
    2. Fill gaps (closing) 
    3. Expand text regions (dilation)
    4. Smooth edges (erosion + blur)
    5. Final expansion and smoothing
    6. Remove disconnected components far from text region
    
    Args:
        mask: Binary mask (H×W uint8)
        bbox: Tuple (x, y, w, h) from the original text detection
        
    Returns:
        Cleaned binary mask
    """
    # TODO: Make morphological operation parameters configurable
    
    # 1. Morphological opening with 3x3 ellipse kernel
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Morphological closing with 4x4 kernel
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
    
    # 7. Final dilation by 6px (with iterations and blur)
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 8. Filter out components far from the text region
    mask = anchor_connected_components(mask, bbox)
    
    return mask

def anchor_connected_components(mask: MaskArray, bbox: BBox) -> MaskArray:
    """Filter connected components based on their distance from the bounding box center.
    
    Uses the bbox height to determine the vertical threshold, allowing full horizontal freedom.
    This removes isolated noise while preserving text that extends beyond the detected region.
    
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
    # TODO: Make this threshold configurable
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