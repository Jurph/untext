"""Mask generation module for morphological cleanup of binary masks.

This module provides morphological operations to clean up and enhance
binary masks for better inpainting results.
"""

import cv2
import numpy as np

from .utils import MaskArray, BBox, setup_logger

logger = setup_logger(__name__)


def morph_clean_mask(mask: MaskArray, bbox: BBox) -> MaskArray:
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

    # Pad mask to avoid edge artifacts during morphological operations
    # Use BORDER_REFLECT to mirror the pattern near edges (more realistic than REPLICATE)
    pad_size = max(close_kernel_size, dilate_size, blur_size)
    padded_mask = cv2.copyMakeBorder(
        mask, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_REFLECT
    )

    # 1. Morphological closing to fill gaps and connect text fragments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    padded_mask = cv2.morphologyEx(padded_mask, cv2.MORPH_CLOSE, kernel_close)
    logger.debug(f"After closing: {np.sum(padded_mask == 255)} white pixels (padded)")

    # 2. Light dilation to ensure good inpainting coverage
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
