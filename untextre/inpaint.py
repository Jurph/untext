"""Image inpainting module using LaMa and TELEA.

This module provides functionality to inpaint masked regions in images using
either the LaMa (Large Mask Inpainting) algorithm or OpenCV's TELEA method.
It supports subregion processing for efficiency and maintains compatibility 
with the existing codebase.
"""

import cv2
import gc
import numpy as np
import torch
from typing import Optional, Tuple, Literal

from .utils import ImageArray, MaskArray, BBox, setup_logger, dilate_bbox, pad_bbox_to_multiple


try:
    from .lama_inpainter import LamaInpainter
except ImportError:
    LamaInpainter = None

logger = setup_logger(__name__)

# Global LaMa model cache for persistent loading
_lama_inpainter = None
_lama_device = None
_lama_init_failed = False

def is_lama_available() -> bool:
    """Check if LaMa inpainter is available for import.
    
    Returns:
        True if LaMa can be imported, False otherwise
    """
    return LamaInpainter is not None

def is_lama_initialized() -> bool:
    """Check if LaMa model is currently initialized.
    
    Returns:
        True if LaMa model is loaded and ready, False otherwise
    """
    global _lama_inpainter
    return _lama_inpainter is not None

def is_lama_healthy() -> bool:
    """Check if LaMa model is healthy and responsive.
    
    Performs a quick test with a small dummy image to verify the model
    is working correctly.
    
    Returns:
        True if LaMa is healthy, False otherwise
    """
    global _lama_inpainter
    
    if _lama_inpainter is None:
        return False
    
    try:
        # Create a small test image and mask
        test_image = np.zeros((32, 32, 3), dtype=np.uint8)
        test_mask = np.zeros((32, 32), dtype=np.uint8)
        test_mask[10:22, 10:22] = 255  # Small square to inpaint
        
        # Try a quick inpaint operation
        result = _lama_inpainter.inpaint(test_image, test_mask)
        
        # Basic validation of result
        if result is None or result.shape != test_image.shape:
            return False
            
        return True
        
    except Exception as e:
        logger.warning(f"LaMa health check failed: {e}")
        return False

def get_lama_status() -> dict:
    """Get comprehensive status information about LaMa.
    
    Returns:
        Dictionary with status information including:
        - available: Whether LaMa can be imported
        - initialized: Whether model is loaded
        - healthy: Whether model passes health check
        - device: Device the model is loaded on
        - init_failed: Whether initialization previously failed
    """
    return {
        "available": is_lama_available(),
        "initialized": is_lama_initialized(),
        "healthy": is_lama_healthy(),
        "device": _lama_device,
        "init_failed": _lama_init_failed
    }

def reset_lama_model() -> None:
    """Reset the LaMa model state, clearing any cached instances.
    
    This is useful for forcing a fresh initialization after errors.
    """
    global _lama_inpainter, _lama_init_failed
    
    logger.info("Resetting LaMa model state")
    
    # Drop the model reference before asking CUDA to release cached blocks.
    # Emptying the cache while the model is still strongly referenced cannot
    # free its tensors and can make OOM recovery reinitialize on top of itself.
    old_inpainter = _lama_inpainter
    was_cuda = (
        old_inpainter is not None
        and hasattr(old_inpainter, 'device')
        and old_inpainter.device.type == 'cuda'
    )
    _lama_inpainter = None
    _lama_init_failed = False

    try:
        del old_inpainter
        gc.collect()
        if was_cuda:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Error during GPU cleanup: {e}")

    logger.info("LaMa model state reset")

def initialize_lama_model(device: str = "cuda", force_reinit: bool = False) -> bool:
    """Initialize and cache the LaMa model for persistent use.
    
    Args:
        device: Device to load the model on ("cuda" or "cpu")
        force_reinit: Whether to force reinitialization even if already loaded
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    global _lama_inpainter, _lama_device, _lama_init_failed
    
    if LamaInpainter is None:
        logger.warning("LaMa inpainter is not available - skipping initialization")
        _lama_init_failed = True
        return False
    
    if _lama_inpainter is not None and not force_reinit:
        logger.info("LaMa model already initialized")
        return True
    
    if force_reinit:
        reset_lama_model()
    
    try:
        logger.info(f"Initializing LaMa model on {device}...")
        _lama_inpainter = LamaInpainter(device=device)
        _lama_device = device
        _lama_init_failed = False
        logger.info("LaMa model initialized and cached")
        
        # Perform initial health check
        if is_lama_healthy():
            logger.info("LaMa model passed initial health check")
            return True
        else:
            logger.error("LaMa model failed initial health check")
            reset_lama_model()
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize LaMa model: {e}")
        _lama_init_failed = True
        reset_lama_model()
        return False

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
    method: InpaintMethod = "lama",
    auto_retry: bool = True
) -> ImageArray:
    """Inpaint masked regions in an image.
    
    This is the main entry point for inpainting. It supports both LaMa and
    TELEA inpainting methods, with automatic fallback if needed.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask (255 = regions to inpaint, 0 = keep original)
        bbox: Optional bounding box to guide subregion calculation
        method: Inpainting method to use ("lama" or "telea")
        auto_retry: Whether to automatically retry with reinitialization on failure
        
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
        return _inpaint_with_lama(image, mask, bbox, auto_retry=auto_retry)
    else:  # method == "telea"
        return _inpaint_with_telea(image, mask, bbox)

def _inpaint_with_lama(
    image: ImageArray, 
    mask: MaskArray, 
    bbox: Optional[BBox] = None,
    auto_retry: bool = True
) -> ImageArray:
    """Inpaint using LaMa algorithm with automatic retry on failure.
    
    Args:
        image: Input image in BGR format
        mask: Binary mask (255 = regions to inpaint, 0 = keep original)
        bbox: Optional bounding box to guide subregion calculation
        auto_retry: Whether to automatically retry with reinitialization on failure
        
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
        if auto_retry:
            logger.warning("LaMa model not initialized. Attempting auto-initialization...")
            if initialize_lama_model(device=_lama_device or "cuda"):
                inpainter = get_lama_inpainter()
            else:
                raise RuntimeError("Failed to auto-initialize LaMa model.")
        else:
            raise RuntimeError("LaMa model not initialized. Call initialize_lama_model() first.")
    
    # Calculate subregion for efficient processing
    subregion = _calculate_inpainting_subregion(mask, bbox, image.shape[:2])
    
    # If no subregion found (no pixels to inpaint), return original image
    if subregion is None:
        logger.info("No subregion to inpaint - returning original image unchanged")
        return image.copy()
    
    try:
        # Perform inpainting using cached model
        logger.info(f"Applying LaMa inpainting (subregion: {subregion})")
        result = inpainter.inpaint(image, mask, subregion=subregion)
        logger.info("LaMa inpainting completed successfully")
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"LaMa inpainting failed due to CUDA out of memory: {e}")
        raise RuntimeError(f"CUDA out of memory during LaMa inpainting: {e}") from e

    except Exception as e:
        logger.error(f"LaMa inpainting failed: {e}")
        
        if auto_retry:
            logger.warning("Attempting LaMa recovery by reinitializing model...")
            try:
                # Try to reinitialize LaMa
                if initialize_lama_model(device=_lama_device or "cuda", force_reinit=True):
                    inpainter = get_lama_inpainter()
                    if inpainter is not None:
                        logger.info("LaMa reinitialized, retrying inpainting...")
                        result = inpainter.inpaint(image, mask, subregion=subregion)
                        logger.info("LaMa inpainting succeeded after recovery")
                        return result
                        
            except Exception as retry_error:
                logger.error(f"LaMa recovery attempt failed: {retry_error}")
        
        # If we get here, LaMa failed even after retry
        raise RuntimeError(f"LaMa inpainting failed: {e}")

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
    # NOTE: +1 because max is inclusive (pixel at max_x IS part of the mask)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    mask_bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    
    # Log mask statistics for debugging
    total_white_pixels = len(xs)
    mask_area = mask_bbox[2] * mask_bbox[3]
    coverage_percent = (total_white_pixels / mask_area) * 100 if mask_area > 0 else 0

    logger.debug("Mask analysis:")
    logger.debug(f"  White pixels found: {total_white_pixels:,}")
    logger.debug(f"  Mask bounding box: ({mask_bbox[0]}, {mask_bbox[1]}) size {mask_bbox[2]}x{mask_bbox[3]}")
    logger.debug(f"  Mask area: {mask_area:,} pixels")
    logger.debug(f"  Coverage density: {coverage_percent:.1f}%")
    
    if image_shape is not None:
        total_image_pixels = image_shape[0] * image_shape[1]
        image_coverage_percent = (total_white_pixels / total_image_pixels) * 100
        logger.debug(f"  Image coverage: {image_coverage_percent:.2f}% of total image")
    
    # Dilate the mask bbox by 64px for better context
    # TODO: Make dilation amount configurable
    dilation_amount = 64
    original_bbox = mask_bbox
    if image_shape is not None:
        mask_bbox = dilate_bbox(mask_bbox, dilation_amount, image_shape)
    
    # Log dilation results
    logger.debug(f"After {dilation_amount}px dilation:")
    logger.debug(f"  Original bbox: ({original_bbox[0]}, {original_bbox[1]}) size {original_bbox[2]}x{original_bbox[3]}")
    logger.debug(f"  Dilated bbox: ({mask_bbox[0]}, {mask_bbox[1]}) size {mask_bbox[2]}x{mask_bbox[3]}")
    
    # Ensure dimensions are compatible with neural networks (LaMa requires mod-4, but may internally pad to mod-8 or mod-16)
    # Use mod-8 padding to be safe for most neural network architectures
    if image_shape is not None:
        # Apply mod-8 padding for better neural network compatibility
        mod8_bbox = pad_bbox_to_multiple(mask_bbox, multiple=8, image_shape=image_shape)
        logger.debug(f"  Mod-8 padded bbox: ({mod8_bbox[0]}, {mod8_bbox[1]}) size {mod8_bbox[2]}x{mod8_bbox[3]}")
        mask_bbox = mod8_bbox
    
    # Convert to subregion format (x1, y1, x2, y2)
    x, y, w, h = mask_bbox
    subregion = (x, y, x + w, y + h)
    
    logger.info(
        f"Inpainting subregion: {subregion} "
        f"(mask pixels={total_white_pixels:,}, density={coverage_percent:.1f}%)"
    )
    return subregion
