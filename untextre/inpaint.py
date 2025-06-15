"""Image inpainting module using LaMa and TELEA.

This module provides functionality to inpaint masked regions in images using
either the LaMa (Large Mask Inpainting) algorithm or OpenCV's TELEA method.
It supports subregion processing for efficiency and maintains compatibility 
with the existing codebase.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Literal

from .utils import ImageArray, MaskArray, BBox, setup_logger, dilate_bbox


try:
    from .lama_inpainter import LamaInpainter
except ImportError:
    LamaInpainter = None

logger = setup_logger(__name__)

# Global LaMa model cache for persistent loading
_lama_inpainter = None

def initialize_lama_model(device: str = "cuda") -> None:
    """Initialize and cache the LaMa model for persistent use.
    
    Args:
        device: Device to load the model on ("cuda" or "cpu")
    """
    global _lama_inpainter
    
    if LamaInpainter is None:
        logger.warning("LaMa inpainter is not available - skipping initialization")
        return
    
    if _lama_inpainter is not None:
        logger.info("LaMa model already initialized")
        return
    
    logger.info(f"Initializing LaMa model on {device}...")
    _lama_inpainter = LamaInpainter(device=device)
    logger.info("LaMa model initialized and cached")

def get_lama_inpainter() -> Optional[LamaInpainter]:
    """Get the cached LaMa inpainter instance.
    
    Returns:
        The cached LaMa inpainter or None if not initialized
    """
    return _lama_inpainter

InpaintMethod = Literal["lama", "telea"]

def inpaint_image(
    image: ImageArray, 
    mask: MaskArray, 
    bbox: Optional[BBox] = None,
    method: InpaintMethod = "lama"
) -> ImageArray:
    """Inpaint masked regions in an image.
    
    This is the main entry point for inpainting. It supports both LaMa and
    TELEA inpainting methods, with automatic fallback if needed.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask (255 = regions to inpaint, 0 = keep original)
        bbox: Optional bounding box to guide subregion calculation
        method: Inpainting method to use ("lama" or "telea")
        
    Returns:
        Inpainted image in BGR format
        
    Raises:
        RuntimeError: If the specified method is not available
        ValueError: If method parameter is invalid
    """
    if method not in ["lama", "telea"]:
        raise ValueError(f"Invalid inpainting method: {method}. Must be 'lama' or 'telea'")
    
    # Check if there are any pixels to inpaint
    if not _has_pixels_to_inpaint(mask):
        logger.info("No pixels to inpaint found in mask - returning original image unchanged")
        return image.copy()
    
    if method == "lama":
        return _inpaint_with_lama(image, mask, bbox)
    else:  # method == "telea"
        return _inpaint_with_telea(image, mask, bbox)

def _inpaint_with_lama(
    image: ImageArray, 
    mask: MaskArray, 
    bbox: Optional[BBox] = None
) -> ImageArray:
    """Inpaint using LaMa algorithm.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask (255 = regions to inpaint, 0 = keep original)
        bbox: Optional bounding box to guide subregion calculation
        
    Returns:
        Inpainted image in BGR format
        
    Raises:
        RuntimeError: If LaMa is not available or inpainting fails
    """
    if LamaInpainter is None:
        raise RuntimeError("LaMa inpainter is not available. Please check installation.")
    
    # Get cached LaMa inpainter
    inpainter = get_lama_inpainter()
    if inpainter is None:
        raise RuntimeError("LaMa model not initialized. Call initialize_lama_model() first.")
    
    # Calculate subregion for efficient processing
    subregion = _calculate_inpainting_subregion(mask, bbox, image.shape[:2])
    
    # If no subregion found (no pixels to inpaint), return original image
    if subregion is None:
        logger.info("No subregion to inpaint - returning original image unchanged")
        return image.copy()
    
    # Perform inpainting using cached model
    logger.info(f"Applying LaMa inpainting (subregion: {subregion})")
    result = inpainter.inpaint(image, mask, subregion=subregion)
    
    logger.info("LaMa inpainting completed successfully")
    return result

def _inpaint_with_telea(
    image: ImageArray, 
    mask: MaskArray, 
    bbox: Optional[BBox] = None
) -> ImageArray:
    """Inpaint using OpenCV's TELEA algorithm.
    
    TELEA (Fast Marching Method) is faster but may produce lower quality results
    compared to LaMa, especially for large regions or complex textures.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask (255 = regions to inpaint, 0 = keep original)
        bbox: Optional bounding box (not used for TELEA, kept for API compatibility)
        
    Returns:
        Inpainted image in BGR format
        
    Raises:
        RuntimeError: If TELEA inpainting fails
    """
    try:
        # Ensure mask is single channel and uint8
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        
        # Check if there are any pixels to inpaint in the processed mask
        if not np.any(mask > 0):
            logger.info("No pixels to inpaint in processed mask - returning original image unchanged")
            return image.copy()
        
        # Apply TELEA inpainting with radius of 3
        # TODO: Make inpainting radius configurable
        logger.info("Applying TELEA inpainting")
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        logger.info("TELEA inpainting completed successfully")
        return result
        
    except Exception as e:
        raise RuntimeError(f"TELEA inpainting failed: {e}")

def _has_pixels_to_inpaint(mask: MaskArray) -> bool:
    """Check if the mask contains any pixels to inpaint.
    
    Args:
        mask: Binary mask array
        
    Returns:
        True if there are pixels to inpaint (white pixels), False otherwise
    """
    return np.any(mask > 0)

def _calculate_inpainting_subregion(
    mask: MaskArray, 
    bbox: Optional[BBox] = None,
    image_shape: Tuple[int, int] = None
) -> Optional[Tuple[int, int, int, int]]:
    """Calculate optimal subregion for inpainting based on mask and bbox.
    
    Args:
        mask: Binary mask array
        bbox: Optional bounding box from text detection
        image_shape: Shape of the image (height, width)
        
    Returns:
        Subregion as (x1, y1, x2, y2) or None for full image processing
    """
    # Find bounding box of white pixels in mask
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        logger.warning("No pixels to inpaint found in mask")
        return None
    
    # Get mask bounding box
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    mask_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Log mask statistics for debugging
    total_white_pixels = len(xs)
    mask_area = mask_bbox[2] * mask_bbox[3]
    coverage_percent = (total_white_pixels / mask_area) * 100 if mask_area > 0 else 0
    
    logger.info(f"Mask analysis:")
    logger.info(f"  White pixels found: {total_white_pixels:,}")
    logger.info(f"  Mask bounding box: ({mask_bbox[0]}, {mask_bbox[1]}) size {mask_bbox[2]}×{mask_bbox[3]}")
    logger.info(f"  Mask area: {mask_area:,} pixels")
    logger.info(f"  Coverage density: {coverage_percent:.1f}%")
    
    if image_shape is not None:
        total_image_pixels = image_shape[0] * image_shape[1]
        image_coverage_percent = (total_white_pixels / total_image_pixels) * 100
        logger.info(f"  Image coverage: {image_coverage_percent:.2f}% of total image")
    
    # Dilate the mask bbox by 64px for better context
    # TODO: Make dilation amount configurable
    dilation_amount = 64
    original_bbox = mask_bbox
    if image_shape is not None:
        mask_bbox = dilate_bbox(mask_bbox, dilation_amount, image_shape)
    
    # Log dilation results
    logger.info(f"After {dilation_amount}px dilation:")
    logger.info(f"  Original bbox: ({original_bbox[0]}, {original_bbox[1]}) size {original_bbox[2]}×{original_bbox[3]}")
    logger.info(f"  Dilated bbox: ({mask_bbox[0]}, {mask_bbox[1]}) size {mask_bbox[2]}×{mask_bbox[3]}")
    
    # Convert to subregion format (x1, y1, x2, y2)
    x, y, w, h = mask_bbox
    subregion = (x, y, x + w, y + h)
    
    logger.info(f"Final inpainting subregion: {subregion}")
    return subregion 