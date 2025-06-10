"""Image preprocessing module for OCR enhancement.

This module provides image preprocessing functions optimized through grid search
testing to improve OCR performance with the fixed DocTR implementation.

The main preprocessing pipeline includes:
- CLAHE contrast enhancement (clip_limit=2.0, tile_size=(4,4))
- Bilateral filtering for noise reduction while preserving edges
- No thresholding (grayscale works better than binary for neural OCR)

These settings were determined through comprehensive grid search testing that
evaluated hundreds of parameter combinations against ground truth text on
real test images, achieving a Figure of Merit of -3.1942.
"""

import cv2
import numpy as np
import logging
from typing import Optional
from pathlib import Path

from .utils import ImageArray, setup_logger

logger = setup_logger(__name__)

def preprocess_image(image: ImageArray) -> Optional[ImageArray]:
    """Apply optimized preprocessing pipeline to an image array.
    
    This function applies a preprocessing pipeline optimized through grid search
    testing to improve OCR performance with the fixed DocTR implementation.
    
    Optimized pipeline steps (based on grid search results):
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement (clip_limit=2.0, tile_size=(4,4))
    3. Apply bilateral filtering for noise reduction while preserving edges
    4. Keep as grayscale (no thresholding - neural networks work better with grayscale)
    5. Convert back to RGB format for compatibility
    
    Args:
        image: Input image as numpy array (BGR format expected)
        
    Returns:
        Preprocessed image as RGB numpy array, or None if processing fails
    """
    
    # First apply gray-to-black conversion if needed
    # TODO: Add support for customizable gray range parameters
    result_img = _preprocess_gray_to_black(image, gray_min=124, gray_max=132)
    if result_img is None:
        result_img = image  # Fall back to original if gray-to-black fails
    else:
        # Convert back to BGR for the rest of the pipeline
        if result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    
        # Apply optimized CLAHE for contrast enhancement (grid search winner)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Apply optimized bilateral filtering for noise reduction (grid search winner)
        # Bilateral filter reduces noise while preserving edges - better for text than Gaussian
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=100, sigmaSpace=75)
        
        # Keep as grayscale - no thresholding (grid search found this works best)
        # Neural OCR networks work better with grayscale than binary images
        
        # Convert back to RGB format for compatibility with OCR pipeline
        preprocessed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        return preprocessed
        
    except Exception as e:
        logger.error(f"Failed to preprocess image array: {e}")
        return None

def _preprocess_gray_to_black(
    img_array: ImageArray, 
    gray_min: int = 124, 
    gray_max: int = 132,
    chunk_size: int = 512
) -> Optional[ImageArray]:
    """Convert gray watermarks to black for better OCR detection.
    
    This preprocessor specifically targets gray watermarks by converting pixels
    in a narrow gray range to black, making them much more detectable by OCR.
    Perfect for watermarks around (128, 128, 128) that are often barely visible.
    
    Memory-efficient implementation that processes images in chunks to avoid
    large memory allocations on high-resolution images.
    
    Args:
        img_array: Input image as numpy array (BGR format expected)
        gray_min: Minimum gray value to convert (inclusive)
        gray_max: Maximum gray value to convert (inclusive)
        chunk_size: Size of chunks to process at once (pixels)
        
    Returns:
        Preprocessed image as RGB numpy array with gray watermarks converted to black
    """
    try:
        # Create a copy to avoid modifying the original (but do it efficiently)
        result = img_array.copy()
        converted_pixels = 0
        
        if len(result.shape) == 3:
            h, w, c = result.shape
            
            # Process image in chunks to avoid large memory allocations
            for y in range(0, h, chunk_size):
                for x in range(0, w, chunk_size):
                    # Get chunk boundaries
                    y_end = min(y + chunk_size, h)
                    x_end = min(x + chunk_size, w)
                    
                    # Extract chunk
                    chunk = result[y:y_end, x:x_end, :]
                    
                    # Check if all channels are in gray range (much smaller memory footprint)
                    is_gray = (
                        (chunk[:, :, 0] >= gray_min) & (chunk[:, :, 0] <= gray_max) &
                        (chunk[:, :, 1] >= gray_min) & (chunk[:, :, 1] <= gray_max) &
                        (chunk[:, :, 2] >= gray_min) & (chunk[:, :, 2] <= gray_max)
                    )
                    
                    # Convert gray pixels to black in-place
                    if np.any(is_gray):
                        chunk[is_gray] = [0, 0, 0]
                        converted_pixels += np.sum(is_gray)
                    
                    # Update result with modified chunk
                    result[y:y_end, x:x_end, :] = chunk
        else:
            # Grayscale image - process in chunks
            h, w = result.shape
            
            for y in range(0, h, chunk_size):
                for x in range(0, w, chunk_size):
                    y_end = min(y + chunk_size, h)
                    x_end = min(x + chunk_size, w)
                    
                    chunk = result[y:y_end, x:x_end]
                    is_gray = (chunk >= gray_min) & (chunk <= gray_max)
                    
                    if np.any(is_gray):
                        chunk[is_gray] = 0
                        converted_pixels += np.sum(is_gray)
                    
                    result[y:y_end, x:x_end] = chunk
        
        logger.debug(f"Gray-to-black: Converted {converted_pixels} pixels from gray to black")
        
        # Convert to RGB format if needed
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Convert BGR to RGB for consistency with other preprocessors
        if result.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
        return result
            
    except Exception as e:
        logger.error(f"Failed to apply gray-to-black preprocessing: {e}")
        return None 