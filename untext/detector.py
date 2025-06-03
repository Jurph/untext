"""Module for detecting text in images using DocTR.

This module provides functionality to detect text in images using the DocTR library.
It supports both detection-only and OCR pipelines.

WARNING: DO NOT ASSUME DocTR's OUTPUT FORMAT! 
The format is documented in their codebase at docs/doctr/doctr/io/elements.py.
The predictor returns a Document object with a nested structure:
- Document has pages
- Pages have blocks
- Blocks have lines
- Lines have words
- Each word has value, confidence, geometry, objectness_score, and crop_orientation

Example:
    >>> from untext.detector import TextDetector
    >>> detector = TextDetector()
    >>> image = cv2.imread('image.jpg')
    >>> detections = detector.detect(image)
    >>> print(f"Found {len(detections)} text regions")
"""

import warnings
warnings.filterwarnings("ignore", message="defusedxml.cElementTree is deprecated")

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImagePath = Union[str, Path]
ImageArray = np.ndarray  # H×W×3 BGR uint8
Detection = Dict[str, np.ndarray]  # {'geometry': points, 'confidence': score}
MaskArray = np.ndarray  # H×W uint8

class TextDetector:
    """Class for detecting text in images using DocTR."""
    
    def __init__(self, confidence_threshold: float = 0.3, min_text_size: int = 10, mask_dilation: int = 2) -> None:
        """Initialize the TextDetector with a pre-trained model.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            min_text_size: Minimum size of text regions to detect
            mask_dilation: Number of pixels to dilate masks by
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_text_size <= 0:
            raise ValueError("min_text_size must be positive")
        if mask_dilation < 0:
            raise ValueError("mask_dilation must be non-negative")
            
        self.confidence_threshold = confidence_threshold
        self.min_text_size = min_text_size
        self.mask_dilation = mask_dilation
        try:
            # Initialize detection model
            self.model = detection.db_resnet50(pretrained=True)
            self.pre_processor = PreProcessor(
                output_size=(1024, 1024),
                batch_size=1,
                mean=(0.798, 0.785, 0.772),
                std=(0.264, 0.2749, 0.287)
            )
            self.predictor = DetectionPredictor(
                pre_processor=self.pre_processor,
                model=self.model
            )
            logger.info("Initialized TextDetector with DB ResNet50 model")
        except Exception as e:
            logger.error(f"Failed to initialize TextDetector: {e}")
            raise RuntimeError("Failed to initialize TextDetector") from e
    
    def detect(
        self,
        image: ImageArray
    ) -> Tuple[MaskArray, List[Detection]]:
        # Note to LLMs: this function returns a 2D mask, not a 3D mask
        # The 2D mask is a binary mask where 255 = text region and 0 = background
        # There is no reason to change this back to a 3D mask!!
        """Detect text regions in an image.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            Tuple of:
            - Binary mask as H×W uint8 numpy array (255 = text region)
            - List of detection dictionaries, each containing:
                - 'geometry': numpy array of polygon points
                - 'confidence': confidence score
        
        Raises:
            ValueError: If image is invalid
            RuntimeError: If detection fails
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be H×W×3 BGR")
        
        try:
            # Run detection
            raw = self.predictor([image])
            if not isinstance(raw, list) or len(raw) == 0:
                logger.warning("No text detected")
                return np.zeros(image.shape[:2], dtype=np.uint8), []
            
            # Convert DocTR output to our format
            detections = self._parse_doctr_output(raw[0], image.shape[:2])
            
            # Filter detections
            detections = self._filter_detections(detections, image.shape[:2])
            
            # Create mask from detections
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for det in detections:
                det_mask = self._geometry_to_mask(det['geometry'], image.shape[:2])
                mask = np.maximum(mask, det_mask)
            
            # Apply dilation if needed
            if self.mask_dilation > 0:
                kernel = np.ones((self.mask_dilation * 2 + 1, self.mask_dilation * 2 + 1), np.uint8)
                mask = cv2.dilate(mask, kernel)
            
            # Return 2D mask instead of 3D
            logger.info(f"Detected {len(detections)} text regions")
            return mask, detections
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise RuntimeError("Text detection failed") from e

    def _parse_doctr_output(self, page: Dict, image_shape: Tuple[int, int]) -> List[Detection]:
        """Convert DocTR output to our detection format.
        
        Args:
            page: Dictionary from DocTR containing predictions
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of detection dictionaries in our format
        """
        if not isinstance(page, dict):
            logger.warning("Invalid page format from DocTR")
            return []
            
        detections = []
        # DocTR returns predictions as numpy arrays with shape (*, 5) or (*, 6)
        # where each row is [x1, y1, x2, y2, confidence, ...]
        if 'words' in page and isinstance(page['words'], np.ndarray):
            for pred in page['words']:
                if len(pred) >= 5:  # Ensure we have at least x1,y1,x2,y2,conf
                    x1, y1, x2, y2, conf = pred[:5]
                    # Convert normalized coordinates to pixel coordinates
                    h, w = image_shape
                    points = np.array([
                        [x1 * w, y1 * h],
                        [x2 * w, y1 * h],
                        [x2 * w, y2 * h],
                        [x1 * w, y2 * h]
                    ], dtype=np.float32)
                    
                    detections.append({
                        'geometry': points,
                        'confidence': float(conf)
                    })
                
        return detections

    def _geometry_to_mask(self, geometry: Union[np.ndarray, List, Tuple], image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert geometry points to binary mask.
        
        Args:
            geometry: Array of polygon points (numpy array, list, or tuple)
            image_shape: Shape of the image (height, width)
            
        Returns:
            Binary mask as H×W uint8 numpy array (1 = text region)
        """
        # Convert input to numpy array if needed
        if not isinstance(geometry, np.ndarray):
            geometry = np.array(geometry, dtype=np.float32)
            
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [geometry.astype(np.int32)], 1)
        return mask

    def _filter_detections(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Filter detections based on size and position.
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the image (height, width)
            
        Returns:
            Filtered list of detections
        """
        if not isinstance(image_shape, tuple) or len(image_shape) != 2 or any(d <= 0 for d in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers")
            
        filtered = []
        for det in detections:
            if not isinstance(det, dict) or 'geometry' not in det:
                continue
                
            geometry = det['geometry']
            if not isinstance(geometry, np.ndarray) or geometry.shape[0] < 3:
                continue
                
            # Calculate width and height
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            # Check size
            if width < self.min_text_size or height < self.min_text_size:
                continue
                
            # Check position (not too close to edges)
            margin = self.min_text_size // 2
            if (np.min(x_coords) < margin or 
                np.max(x_coords) > image_shape[1] - margin or
                np.min(y_coords) < margin or
                np.max(y_coords) > image_shape[0] - margin):
                continue
                
            filtered.append(det)
            
        return filtered

class WordMaskGenerator:
    """Class for generating word-level masks from text detection results."""
    
    def __init__(self, mode: str = "word") -> None:
        """Initialize the WordMaskGenerator.
        
        Args:
            mode: Mask generation mode ("word" or "line")
        """
        self.mode = mode
        self.detector = TextDetector() 