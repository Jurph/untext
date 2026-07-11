"""Image preprocessing for consensus text detection.

This module provides image preprocessing functions optimized through grid search
testing for the shared EAST, EasyOCR, and YOLO11x consensus path.

The main preprocessing pipeline includes:
- CLAHE contrast enhancement (clip_limit=2.0, tile_size=(4,4))
- Bilateral filtering for noise reduction while preserving edges
- No thresholding (grayscale gave better detector agreement than binary)

These settings were determined through comprehensive grid search testing that
evaluated hundreds of parameter combinations against ground truth text on
real test images. Grid search winner at FOM = -3.1942 (lower is better; see
experiments/grid_search.py for scale, direction, and dataset details).
"""

import cv2
from typing import Optional

from .utils import ImageArray, setup_logger

logger = setup_logger(__name__)

def preprocess_image(image: ImageArray) -> Optional[ImageArray]:
    """Apply optimized preprocessing pipeline to an image array.
    
    This function applies a preprocessing pipeline optimized through grid search
    testing for the shared consensus detector path.
    
    Optimized pipeline steps (based on grid search results):
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement (clip_limit=2.0, tile_size=(4,4))
    3. Apply bilateral filtering for noise reduction while preserving edges
    4. Keep as grayscale (no thresholding - detector agreement was better with grayscale)
    5. Convert back to a 3-channel image for detector compatibility
    
    Args:
        image: Input image as numpy array (BGR format expected)
        
    Returns:
        3-channel grayscale image (BGR-convention array with R == G == B) ready
        for the detector paths, or None if processing fails
    """
    
    try:
        # Convert to grayscale before contrast/noise processing.
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
        
        # Keep as grayscale - no thresholding (grid search found this works best).
        
        # Replicate gray into 3 channels for detector APIs that expect color input.
        # GRAY2BGR to match the codebase-wide BGR convention (it is the same OpenCV
        # enum as GRAY2RGB, and the output has R == G == B either way).
        preprocessed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        return preprocessed
        
    except Exception:
        logger.exception("Failed to preprocess image array")
        return None 
