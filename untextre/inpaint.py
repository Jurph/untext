"""Image inpainting module using LaMa.

This module provides functionality to inpaint masked regions in images using
the LaMa (Large Mask Inpainting) algorithm. It supports subregion processing
for efficiency and maintains compatibility with the existing codebase.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from .utils import ImageArray, MaskArray, BBox, setup_logger, dilate_bbox

# TODO: Add support for other inpainting methods (TELEA, EAST)
try:
    from untext.lama_inpainter import LamaInpainter
except ImportError:
    LamaInpainter = None

logger = setup_logger(__name__)

def inpaint_image(
    image: ImageArray, 
    mask: MaskArray, 
    bbox: Optional[BBox] = None
) -> ImageArray:
    """Inpaint masked regions in an image.
    
    This is the main entry point for inpainting. It calculates an optimal
    subregion around the mask and applies LaMa inpainting for best results.
    
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
    
    logger.info("Inpainting completed successfully")
    return result

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
    if image_shape is not None:
        mask_bbox = dilate_bbox(mask_bbox, 64, image_shape)
    
    # Convert to subregion format (x1, y1, x2, y2)
    x, y, w, h = mask_bbox
    subregion = (x, y, x + w, y + h)
    
    logger.debug(f"Calculated inpainting subregion: {subregion}")
    return subregion 