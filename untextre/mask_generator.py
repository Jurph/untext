"""Mask generation module for morphological cleanup of binary masks.

This module provides morphological operations to clean up and enhance
binary masks for better inpainting results.
"""

import cv2
import numpy as np

from .utils import MaskArray, setup_logger

logger = setup_logger(__name__)


def morph_clean_mask(
    mask: MaskArray,
    *,
    cleanup_close_px: int = 11,
    cleanup_dilate_px: int = 13,
) -> MaskArray:
    """Apply morphological operations to clean up a binary mask.

    This function applies a series of morphological operations to:
    1. Fill gaps and connect text fragments (closing)
    2. Light expansion for inpainting coverage (dilation)
    3. Smooth edges (blur + threshold)

    IMPORTANT: Operations are performed on a padded mask to avoid edge artifacts.
    Without padding, Gaussian blur treats out-of-bounds pixels as zero, causing
    white pixels at the mask edge to drop below threshold and create a "rim".

    Args:
        mask: Binary mask (H×W uint8)

    Returns:
        Cleaned binary mask
    """
    initial_white_pixels = np.sum(mask == 255)
    logger.debug(f"Starting morphological cleanup with {initial_white_pixels} white pixels")

    close_kernel_size = int(cleanup_close_px)
    dilate_size = int(cleanup_dilate_px)
    blur_size = 9  # EMPIRICAL — not yet validated; chosen to match kernel scale

    # Pad mask to avoid edge artifacts during morphological operations
    # Use BORDER_REFLECT to mirror the pattern near edges (more realistic than REPLICATE)
    pad_size = max(close_kernel_size, dilate_size, blur_size)
    padded_mask = cv2.copyMakeBorder(
        mask, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_REFLECT
    )

    # 1. Morphological closing to fill gaps and connect text fragments
    if close_kernel_size > 0:
        close_kernel_size = max(1, close_kernel_size)
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )
        padded_mask = cv2.morphologyEx(padded_mask, cv2.MORPH_CLOSE, kernel_close)
        logger.debug(f"After closing: {np.sum(padded_mask == 255)} white pixels (padded)")

    # 2. Light dilation to ensure good inpainting coverage
    if dilate_size > 0:
        dilate_size = max(1, dilate_size)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        padded_mask = cv2.dilate(padded_mask, kernel_dilate)
        logger.debug(f"After dilation: {np.sum(padded_mask == 255)} white pixels (padded)")

    # 3. Light Gaussian blur for smooth edges
    padded_mask = cv2.GaussianBlur(padded_mask, (blur_size, blur_size), 0)
    logger.debug(f"After blur: {np.sum(padded_mask == 255)} white pixels (padded)")

    # 4. Re-threshold to binary
    padded_mask = (padded_mask > 127).astype(np.uint8) * 255

    # Crop back to original size
    h, w = mask.shape[:2]
    mask = padded_mask[pad_size:pad_size+h, pad_size:pad_size+w]

    logger.debug(f"Final mask: {np.sum(mask == 255)} white pixels")
    return mask
