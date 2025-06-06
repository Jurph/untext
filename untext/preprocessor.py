#!/usr/bin/env python3
"""Image preprocessing module for OCR enhancement.

This module provides image preprocessing functions that can improve OCR performance
by enhancing image quality, reducing noise, and normalizing illumination.

Note: The preprocessing functions in this module were originally optimized for
DocTR OCR, but those optimization results were based on a broken DocTR configuration
with dimension squashing bugs. With the DocTR fixes applied, these preprocessing
steps may or may not improve OCR performance and should be re-evaluated.

The preprocessing functions are kept for:
- Compatibility with existing code
- Use with other OCR engines (Tesseract, etc.)
- General image enhancement tasks
- Future re-optimization with working DocTR

Example:
    >>> from untext.preprocessor import preprocess_image_array
    >>> processed = preprocess_image_array(image_array)
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """Preprocess an image file for potential OCR improvement.
    
    Applies a standard preprocessing pipeline that may help with OCR
    performance, particularly for images with poor contrast or noise.
    
    Note: The effectiveness of this preprocessing depends on the OCR engine
    and specific image characteristics. Test with your specific use case.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Preprocessed image as RGB numpy array, or None if loading fails
    """
    return preprocess_image_standard(image_path)


def preprocess_image_standard(image_path: str) -> Optional[np.ndarray]:
    """Apply standard preprocessing pipeline to an image file.
    
    This pipeline includes:
    - Grayscale conversion
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Minimal Gaussian blur for noise reduction
    - Adaptive thresholding for binarization
    - Convert back to RGB format
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Preprocessed image as RGB numpy array, or None if loading fails
    """
    logger.debug(f"Preprocessing image with standard pipeline: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    return preprocess_image_array(img)


def preprocess_image_array(img_array: np.ndarray) -> Optional[np.ndarray]:
    """Apply standard preprocessing pipeline to an image array.
    
    This function applies the same preprocessing pipeline but takes
    a numpy array as input instead of a file path.
    
    Pipeline steps:
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Apply minimal Gaussian blur for noise reduction  
    4. Apply adaptive thresholding for binarization
    5. Convert back to RGB format
    
    Args:
        img_array: Input image as numpy array (BGR format expected)
        
    Returns:
        Preprocessed image as RGB numpy array, or None if processing fails
    """
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Apply minimal Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Apply adaptive thresholding for binarization
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 1
        )
        
        # Convert back to RGB format
        preprocessed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return preprocessed
        
    except Exception as e:
        logger.error(f"Failed to preprocess image array: {e}")
        return None


def preprocess_image_legacy(image_path: str) -> Optional[np.ndarray]:
    """Legacy preprocessing function with broader parameters.
    
    This is the original preprocessing approach with more aggressive
    blur and threshold settings. May work better for some images.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Preprocessed image as RGB numpy array, or None if loading fails
    """
    logger.debug(f"Preprocessing image with legacy pipeline: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply broader Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding with broader parameters
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to RGB format
    preprocessed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return preprocessed


def preprocess_image_minimal(image_path: str) -> Optional[np.ndarray]:
    """Minimal preprocessing - just convert to RGB without modification.
    
    This function simply loads the image and ensures it's in RGB format
    without applying any enhancement. Useful as a baseline or for images
    that are already well-prepared for OCR.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Image as RGB numpy array, or None if loading fails
    """
    logger.debug(f"Loading image with minimal preprocessing: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Just convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img 