#!/usr/bin/env python3
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

Alternative preprocessing functions are also provided for:
- Compatibility with existing code
- Use with other OCR engines (Tesseract, etc.)
- Different image enhancement strategies

Example:
    >>> from untext.preprocessor import preprocess_image_array
    >>> processed = preprocess_image_array(image_array)  # Uses optimized pipeline
"""

import cv2
import numpy as np
import logging
from typing import Optional
from pathlib import Path

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
        img_array: Input image as numpy array (BGR format expected)
        
    Returns:
        Preprocessed image as RGB numpy array, or None if processing fails
    """
    
    # First apply gray-to-black conversion if needed
    result_img = preprocess_image_gray_to_black(img_array, gray_min=124, gray_max=132)
    if result_img is None:
        result_img = img_array  # Fall back to original if gray-to-black fails
    else:
        # Convert back to BGR for the rest of the pipeline
        if result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
    
        
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


def preprocess_image_gray_to_black(
    img_array: np.ndarray, 
    gray_min: int = 124, 
    gray_max: int = 132,
    chunk_size: int = 512,
    debug_output_path: Optional[str] = None
) -> Optional[np.ndarray]:
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
        debug_output_path: Optional path to save debug mask showing detected gray pixels
        
    Returns:
        Preprocessed image as RGB numpy array with gray watermarks converted to black
    """
    try:
        # Create a copy to avoid modifying the original (but do it efficiently)
        result = img_array.copy()
        converted_pixels = 0
        
        if len(result.shape) == 3:
            h, w, c = result.shape
            
            # Create binary mask for pure gray pixels
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            
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
                    
                    # Update binary mask
                    binary_mask[y:y_end, x:x_end] = np.where(is_gray, 0, 255)
                    
                    # Convert gray pixels to black in-place
                    if np.any(is_gray):
                        chunk[is_gray] = [0, 0, 0]
                        converted_pixels += np.sum(is_gray)
                    
                    # Update result with modified chunk
                    result[y:y_end, x:x_end, :] = chunk
        else:
            # Grayscale image - process in chunks
            h, w = result.shape
            
            # Create binary mask for pure gray pixels
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            
            for y in range(0, h, chunk_size):
                for x in range(0, w, chunk_size):
                    y_end = min(y + chunk_size, h)
                    x_end = min(x + chunk_size, w)
                    
                    chunk = result[y:y_end, x:x_end]
                    is_gray = (chunk >= gray_min) & (chunk <= gray_max)
                    
                    # Update binary mask
                    binary_mask[y:y_end, x:x_end] = np.where(is_gray, 0, 255)
                    
                    if np.any(is_gray):
                        chunk[is_gray] = 0
                        converted_pixels += np.sum(is_gray)
                    
                    result[y:y_end, x:x_end] = chunk
        
        logger.info(f"Gray-to-black: Converted {converted_pixels} pixels from gray to black")
        
        # Convert to RGB format if needed
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Convert BGR to RGB for consistency with other preprocessors
        if result.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        debug_output_path = Path("X:/Movies/fast quarantine/testmasks/")
        # Save debug outputs if requested
        if debug_output_path:
            debug_path = Path(debug_output_path)
            debug_path.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_path / "binary_mask.png"), binary_mask)
            cv2.imwrite(str(debug_path / "processed.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
        return result
            
    except Exception as e:
        logger.error(f"Failed to apply gray-to-black preprocessing: {e}")
        return None


def _create_binary_mask(image: np.ndarray) -> np.ndarray:
    """Create a binary mask where pure gray pixels are black and everything else is white."""
    # Create a binary mask where pure gray pixels (R=G=B) are True
    gray_mask = (image[:, :, 0] == image[:, :, 1]) & (image[:, :, 1] == image[:, :, 2])
    
    # Create binary image (0 for gray pixels, 255 for everything else)
    binary = np.zeros(image.shape[:2], dtype=np.uint8)
    binary[~gray_mask] = 255
    
    return binary


def process(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Process the image to enhance text detection.
    
    Args:
        image: Input image as numpy array
        debug: If True, saves debug visualizations
        
    Returns:
        Processed image as numpy array
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    # Create binary mask for pure gray pixels
    binary_mask = _create_binary_mask(image)
    
    # Convert pure gray pixels to black
    gray_mask = (image[:, :, 0] == image[:, :, 1]) & (image[:, :, 1] == image[:, :, 2])
    image[gray_mask] = [0, 0, 0]
    
    if debug:
        # Save binary mask visualization
        debug_path = Path("debug_output")
        debug_path.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_path / "binary_mask.png"), binary_mask)
        
        # Save processed image
        cv2.imwrite(str(debug_path / "processed.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
    return image 