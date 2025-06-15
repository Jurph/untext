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
    
    # Save raw mask before cleanup for debugging
    cv2.imwrite("raw_color_mask.png", mask)
    logger.debug(f"Saved raw color mask to raw_color_mask.png")
    
    # Clean up mask with morphological operations
    mask = clean_up_mask(mask, bbox)
    
    logger.info(f"Generated mask from {len(text_colors)} text colors")
    return mask

def create_color_mask(image: ImageArray, target_colors: List[Color]) -> MaskArray:
    """Create a binary mask for pixels exactly matching any of the target colors.
    
    Args:
        image: Input image in BGR format
        target_colors: List of BGR color tuples to match exactly
        
    Returns:
        Binary mask as H×W uint8 numpy array
    """
    logger.debug(f"Creating mask for {len(target_colors)} colors (exact matching)")
    
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert target colors to set for O(1) lookup
    target_set = set(target_colors)
    
    # Process image in chunks to avoid memory explosion
    h, w = image.shape[:2]
    chunk_size = 1000  # Process 1000 rows at a time
    
    total_matched = 0
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk = image[start_row:end_row]
        
        # Reshape chunk to (chunk_pixels, 3)
        chunk_reshaped = chunk.reshape(-1, 3)
        
        # Check each pixel against the target set
        chunk_matches = np.array([tuple(pixel) in target_set for pixel in chunk_reshaped], dtype=bool)
        
        # Reshape back and update mask
        chunk_mask = (chunk_matches.astype(np.uint8) * 255).reshape(end_row - start_row, w)
        mask[start_row:end_row] = chunk_mask
        
        chunk_matched = np.sum(chunk_mask == 255)
        total_matched += chunk_matched
        
        if chunk_matched > 0:
            logger.debug(f"  Rows {start_row}-{end_row}: {chunk_matched} pixels matched")
    
    logger.debug(f"Total pixels matched (exact): {total_matched}")
    
    return mask

def morph_clean_mask(mask: MaskArray, bbox: BBox) -> MaskArray:
    """Apply morphological operations to clean up a binary mask.
    
    This function applies a simplified series of morphological operations to:
    1. Fill gaps and connect text fragments (closing)
    2. Light expansion for inpainting coverage (dilation)
    3. Smooth edges (blur + threshold)
    
    Args:
        mask: Binary mask (H×W uint8)
        bbox: Tuple (x, y, w, h) from the original text detection
        
    Returns:
        Cleaned binary mask
    """
    initial_white_pixels = np.sum(mask == 255)
    logger.debug(f"Starting morphological cleanup with {initial_white_pixels} white pixels")
    
    # Simplified parameters for detail preservation
    close_kernel_size = 11
    dilate_size = 13
    blur_size = 9
    
    # 1. Morphological closing to fill gaps and connect text fragments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    logger.debug(f"After closing: {np.sum(mask == 255)} white pixels")
    
    # 2. Light dilation to ensure good inpainting coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask = cv2.dilate(mask, kernel_dilate)
    logger.debug(f"After dilation: {np.sum(mask == 255)} white pixels")
    
    # 3. Light Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    logger.debug(f"After blur: {np.sum(mask == 255)} white pixels")
    
    # 4. Re-threshold to binary
    mask = (mask > 127).astype(np.uint8) * 255
    
    return mask

def morph_expand_mask(mask: MaskArray, expansion_px: int = 8) -> MaskArray:
    """Expand a binary mask by a specified number of pixels.
    
    Args:
        mask: Binary mask (H×W uint8)
        expansion_px: Number of pixels to expand the mask
        
    Returns:
        Expanded binary mask
    """
    if expansion_px <= 0:
        return mask
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expansion_px * 2 + 1, expansion_px * 2 + 1))
    expanded_mask = cv2.dilate(mask, kernel)
    
    logger.debug(f"Expanded mask by {expansion_px}px: {np.sum(mask == 255)} -> {np.sum(expanded_mask == 255)} pixels")
    return expanded_mask

def morph_smooth_mask(mask: MaskArray, blur_size: int = 5) -> MaskArray:
    """Smooth a binary mask using Gaussian blur and re-threshold.
    
    Args:
        mask: Binary mask (H×W uint8)
        blur_size: Size of the Gaussian blur kernel (must be odd)
        
    Returns:
        Smoothed binary mask
    """
    if blur_size <= 1:
        return mask
        
    # Ensure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1
        
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Re-threshold to binary
    smoothed_mask = (blurred > 127).astype(np.uint8) * 255
    
    logger.debug(f"Smoothed mask with {blur_size}x{blur_size} kernel")
    return smoothed_mask

def clean_up_mask(mask: MaskArray, bbox: BBox) -> MaskArray:
    """Clean up the mask using morphological operations.
    
    This function applies a simplified series of morphological operations to:
    1. Fill gaps and connect text fragments (closing)
    2. Light expansion for inpainting coverage (dilation)
    3. Smooth edges (blur + threshold)
    
    Args:
        mask: Binary mask (H×W uint8)
        bbox: Tuple (x, y, w, h) from the original text detection
        
    Returns:
        Cleaned binary mask
    """
    return morph_clean_mask(mask, bbox)

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