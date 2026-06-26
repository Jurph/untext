"""Known-mask image processing using ORB template matching."""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .orb_matcher import find_known_mask_in_image
from .utils import load_image, save_image, setup_logger

logger = setup_logger(__name__)

def process_with_known_mask(
    image_path: Path,
    output_dir: Path,
    known_mask_rgba: np.ndarray,
    keep_masks: bool = False,
    method: str = "lama",
    min_matches: int = 6,
    dilation_pixels: int = 15,
) -> Optional[dict]:
    """Process a single image using known-mask ORB feature matching.
    
    This bypasses consensus detection entirely and uses ORB to find a known
    watermark/logo template in the image, then inpaints it.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        known_mask_rgba: Known watermark as RGBA image (alpha channel is mask)
        keep_masks: Whether to save debug masks
        method: Inpainting method ("lama" or "telea")
        min_matches: Minimum ORB matches required
        dilation_pixels: Pixels to dilate mask for safety margin
        
    Returns:
        Dictionary with timing details, or None if processing failed
    """
    start_time = time.time()
    timings = {
        'image': image_path.name,
        'load_time': 0,
        'orb_time': 0,
        'inpaint_time': 0,
        'total_time': 0,
        'mask_found': False,
    }
    
    # Load image
    load_start = time.time()
    try:
        image = load_image(image_path)
    except ValueError:
        logger.error(f"Failed to load image: {image_path}")
        return None
    timings['load_time'] = time.time() - load_start
    logger.info(f"Loaded image: {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Find known mask using ORB
    orb_start = time.time()
    match_result = find_known_mask_in_image(
        image,
        known_mask_rgba,
        min_matches=min_matches,
        dilation_pixels=dilation_pixels,
    )
    timings['orb_time'] = time.time() - orb_start
    
    if match_result is None:
        logger.warning(f"Known mask not found in {image_path.name} - saving original")
        # Save original image unchanged
        output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
        save_image(image, output_path, source_path=image_path)
        timings['total_time'] = time.time() - start_time
        return timings
    
    mask, bbox, _inliers = match_result
    timings['mask_found'] = True
    
    # Save mask if requested
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to {mask_path}")
    
    # Inpaint
    from .inpaint import inpaint_image

    inpaint_start = time.time()
    result = inpaint_image(image, mask, bbox=bbox, method=method)
    timings['inpaint_time'] = time.time() - inpaint_start
    
    # Save result
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path, source_path=image_path)
    logger.info(f"Saved cleaned image to {output_path}")
    
    timings['total_time'] = time.time() - start_time
    return timings
