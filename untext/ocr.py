"""Standardized OCR functionality using fixed DocTR preprocessors.

This module provides the single source of truth for all OCR operations in the codebase.
All other modules should import and use functions from this module rather than
implementing their own OCR logic.

The core issue with DocTR is that the default preprocessors have incorrect output_size
parameters that cause dimension errors. This module fixes those issues.

Example:
    >>> from untext.ocr import extract_text_from_image, extract_text_from_array
    >>> text = extract_text_from_image("image.jpg")
    >>> text = extract_text_from_array(img_array)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def create_fixed_ocr_predictor():
    """Create OCR predictor with fixed preprocessors.
    
    This function creates a DocTR OCR predictor with corrected preprocessor
    configurations that fix the dimension issues in the default implementation.
    
    Returns:
        DocTR OCR predictor with fixed detection and recognition preprocessors
        
    Note:
        The key fixes are:
        - Detection preprocessor: output_size=(1024, 1024) instead of (3, 1024)
        - Recognition preprocessor: output_size=(32, 128) instead of (3, 32)
    """
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Create OCR predictor (will have broken preprocessors)
    ocr = ocr_predictor(pretrained=True)
    
    # Create FIXED detection preprocessor
    detection_preprocessor = PreProcessor(
        output_size=(1024, 1024),  # FIX: was (3, 1024) 
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    
    # Create FIXED recognition preprocessor  
    recognition_preprocessor = PreProcessor(
        output_size=(32, 128),  # FIX: was (3, 32)
        batch_size=128,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    
    # Replace broken preprocessors with fixed ones
    ocr.det_predictor.pre_processor = detection_preprocessor
    ocr.reco_predictor.pre_processor = recognition_preprocessor
    
    return ocr


def extract_text_from_image(image_path: Union[str, Path]) -> str:
    """Extract text from an image file using fixed DocTR OCR.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Extracted text as a string, empty string if no text found or error occurs
        
    Example:
        >>> text = extract_text_from_image("document.jpg")
        >>> print(f"Found text: {text}")
    """
    # Load and convert image (BGR -> RGB as required by DocTR)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.error(f"Failed to load image: {image_path}")
        return ""
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return extract_text_from_array(img_rgb)


def extract_text_from_array(image_array: np.ndarray) -> str:
    """Extract text from an image array using fixed DocTR OCR.
    
    Args:
        image_array: Input image as H×W×3 RGB uint8 numpy array
        
    Returns:
        Extracted text as a string, empty string if no text found or error occurs
        
    Example:
        >>> img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
        >>> text = extract_text_from_array(img)
    """
    if not isinstance(image_array, np.ndarray):
        logger.error("Input must be a numpy array")
        return ""
        
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        logger.error(f"Image must be H×W×3 RGB, got shape: {image_array.shape}")
        return ""
    
    try:
        # Create fixed OCR predictor
        ocr = create_fixed_ocr_predictor()
        
        # Run OCR with fixed preprocessors
        results = ocr([image_array])
        
        # Extract text following the correct DocTR hierarchy: page -> block -> line -> word
        extracted_text = ""
        for page in results.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        extracted_text += text + " "
        
        return extracted_text.strip()
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return ""


def extract_text_from_detections(image_array: np.ndarray, detections: list) -> list:
    """Extract text from specific detected regions in an image.
    
    Args:
        image_array: Input image as H×W×3 RGB uint8 numpy array 
        detections: List of detection dictionaries with 'geometry' keys
        
    Returns:
        List of dictionaries with 'box' and 'text' keys for each detection
        
    Example:
        >>> detections = detector.detect(image)[1]  # Get detections
        >>> results = extract_text_from_detections(image, detections)
        >>> for result in results:
        ...     print(f"Text '{result['text']}' at {result['box']}")
    """
    if not detections:
        logger.info("No detections to process for OCR")
        return []
        
    try:
        # Create fixed OCR predictor
        ocr = create_fixed_ocr_predictor()
        
        ocr_results = []
        for det in detections:
            # Compute bounding rectangle from the detected geometry
            geometry = det.get("geometry")
            if geometry is None:
                continue
                
            pts = np.array(geometry, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            
            # Crop the detected region from the image (convert to BGR for cropping)
            img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            cropped_bgr = img_bgr[y:y+h, x:x+w]
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            
            # Run OCR on the cropped region
            text = extract_text_from_array(cropped_rgb)
            
            ocr_results.append({
                "box": (x, y, w, h), 
                "text": text
            })
            
        return ocr_results
        
    except Exception as e:
        logger.error(f"Detection-based OCR failed: {e}")
        return [] 