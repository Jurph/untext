"""Text detection module using DocTR.

This module provides functionality to detect text regions in images using the DocTR library.
It includes preprocessing integration and returns bounding boxes for detected text regions.

WARNING: DO NOT ASSUME DocTR's OUTPUT FORMAT! 
The format is documented in their codebase at docs/doctr/doctr/io/elements.py.
"""

import warnings
warnings.filterwarnings("ignore", message="defusedxml.cElementTree is deprecated")

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor

from .utils import ImageArray, BBox, setup_logger
from .preprocessor import preprocess_image

logger = setup_logger(__name__)

# Type alias for detection results
Detection = Dict[str, np.ndarray]  # {'geometry': points, 'confidence': score}

def detect_text_regions(image: ImageArray) -> List[BBox]:
    """Detect text regions in an image and return bounding boxes.
    
    This is the main entry point for text detection. It applies preprocessing,
    runs DocTR detection, and returns a list of bounding boxes for detected text.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        
    Returns:
        List of bounding boxes as (x, y, width, height) tuples
        
    Raises:
        ValueError: If image is invalid
        RuntimeError: If detection fails
    """
    detector = TextDetector()
    detections = detector.detect(image)
    
    # Convert detections to bounding boxes
    bboxes = []
    for det in detections:
        bbox = _geometry_to_bbox(det['geometry'])
        bboxes.append(bbox)
    
    logger.info(f"Detected {len(bboxes)} text regions")
    return bboxes

def get_largest_text_region(image: ImageArray) -> BBox:
    """Get the largest detected text region.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        
    Returns:
        Bounding box of the largest text region as (x, y, width, height)
        
    Raises:
        ValueError: If no text is detected
    """
    bboxes = detect_text_regions(image)
    
    if not bboxes:
        # TODO: Add fallover to corner-based region selection
        raise ValueError("No text regions detected")
    
    # Find largest bbox by area
    largest_bbox = max(bboxes, key=lambda bbox: bbox[2] * bbox[3])
    return largest_bbox

def _geometry_to_bbox(geometry: np.ndarray) -> BBox:
    """Convert geometry points to bounding box.
    
    Args:
        geometry: Array of polygon points
        
    Returns:
        Bounding box as (x, y, width, height)
    """
    x_coords = geometry[:, 0]
    y_coords = geometry[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

class TextDetector:
    """Text detector using DocTR with optimized preprocessing."""
    
    def __init__(
        self, 
        confidence_threshold: float = 0.3, 
        min_text_size: int = 10
    ) -> None:
        """Initialize the TextDetector with a pre-trained model.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            min_text_size: Minimum size of text regions to detect
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_text_size <= 0:
            raise ValueError("min_text_size must be positive")
            
        self.confidence_threshold = confidence_threshold
        self.min_text_size = min_text_size
        
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
    
    def detect(self, image: ImageArray) -> List[Detection]:
        """Detect text regions in an image.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            List of detection dictionaries, each containing:
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
            # Apply preprocessing
            logger.debug("Applying optimized preprocessing before text detection")
            processed_image = preprocess_image(image)
            if processed_image is None:
                logger.warning("Preprocessing failed, using original image")
                processed_image = image
            else:
                # Convert back to BGR if needed (preprocessor returns RGB)
                if processed_image.shape[2] == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            # Run detection
            raw = self.predictor([processed_image])
            if not isinstance(raw, list) or len(raw) == 0:
                logger.warning("No text detected")
                return []
            
            # Convert DocTR output to our format
            detections = self._parse_doctr_output(raw[0], image.shape[:2])
            
            # Filter detections
            detections = self._filter_detections(detections, image.shape[:2])
            
            logger.debug(f"Detected {len(detections)} text regions")
            return detections
            
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
                    
                    # Skip low confidence detections
                    if conf < self.confidence_threshold:
                        continue
                    
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

    def _filter_detections(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Filter detections based on size and position.
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the image (height, width)
            
        Returns:
            Filtered list of detections
        """
        if not isinstance(image_shape, tuple) or len(image_shape) != 2:
            raise ValueError("image_shape must be a tuple of two integers")
            
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
                
            # TODO: Add smarter position-based filtering (corner preference)
            
            filtered.append(det)
            
        return filtered 