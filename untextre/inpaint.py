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
    from untext.lama_inpainter import LamaInpainter
except ImportError:
    LamaInpainter = None

logger = setup_logger(__name__)

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
    
    # Calculate subregion for efficient processing
    subregion = _calculate_inpainting_subregion(mask, bbox, image.shape[:2])
    
    # Initialize LaMa inpainter
    inpainter = LamaInpainter()
    
    # Perform inpainting
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
        
        # Apply TELEA inpainting with radius of 3
        # TODO: Make inpainting radius configurable
        logger.info("Applying TELEA inpainting")
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        logger.info("TELEA inpainting completed successfully")
        return result
        
    except Exception as e:
        raise RuntimeError(f"TELEA inpainting failed: {e}")

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
    
    # Dilate the mask bbox by 64px for better context
    # TODO: Make dilation amount configurable
    dilation_amount = 64
    if image_shape is not None:
        mask_bbox = dilate_bbox(mask_bbox, dilation_amount, image_shape)
    
    # Convert to subregion format (x1, y1, x2, y2)
    x, y, w, h = mask_bbox
    subregion = (x, y, x + w, y + h)
    
    logger.debug(f"Calculated inpainting subregion: {subregion}")
    return subregion 