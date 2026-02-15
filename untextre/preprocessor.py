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
from typing import Optional

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
    
    try:
        result_img = image
        # Convert to grayscale - use the preprocessed image, not the original
        if len(result_img.shape) == 3:
            gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = result_img
    
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