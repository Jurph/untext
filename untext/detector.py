"""Text detection module using DocTR's DBNet++ implementation."""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from doctr.io import DocumentFile
from doctr.models import db_resnet50
from doctr.models.detection.predictor import DetectionPredictor


class TextDetector:
    """Detects text in images using DBNet++."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        mask_dilation: int = 2,
        min_text_size: int = 10,
    ):
        """Initialize the text detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            mask_dilation: Number of pixels to dilate the text mask
            min_text_size: Minimum size (in pixels) for text to be considered
        """
        self.model = db_resnet50(pretrained=True)
        self.predictor = DetectionPredictor(self.model)
        self.confidence_threshold = confidence_threshold
        self.mask_dilation = mask_dilation
        self.min_text_size = min_text_size

    def _geometry_to_mask(
        self, geometry: List[Tuple[float, float]], image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Convert geometry points to a binary mask.
        
        Args:
            geometry: List of (x, y) points defining the text polygon
            image_shape: (height, width) of the image
            
        Returns:
            Binary mask where text regions are 1
        """
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert points to integer coordinates
        points = np.array(geometry, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [points], 1)
        
        # Dilate mask to ensure we catch all text pixels
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask

    def _filter_detections(
        self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Filter detections based on size and other criteria.
        
        Args:
            detections: List of detection dictionaries
            image_shape: (height, width) of the image
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for det in detections:
            # Get text region size
            points = np.array(det['geometry'])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min
            
            # Filter based on size
            if width < self.min_text_size or height < self.min_text_size:
                continue
                
            # Filter based on position (optional: remove text too close to edges)
            margin = 10
            if (x_min < margin or y_min < margin or
                x_max > image_shape[1] - margin or
                y_max > image_shape[0] - margin):
                continue
                
            filtered.append(det)
            
        return filtered

    def detect(self, image_path: str) -> Tuple[np.ndarray, List[dict]]:
        """Detect text in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
            - Binary mask of detected text regions
            - List of detection results with confidence scores
        """
        # Load and preprocess the image
        doc = DocumentFile.from_images(image_path)
        image = doc[0]
        image_shape = (image.shape[0], image.shape[1])
        
        # Run detection
        result = self.predictor(doc)
        
        # Extract detections above threshold
        detections = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.confidence >= self.confidence_threshold:
                            detections.append({
                                'geometry': word.geometry,
                                'confidence': word.confidence,
                                'text': word.value
                            })
        
        # Filter detections
        detections = self._filter_detections(detections, image_shape)
        
        # Create binary mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        for det in detections:
            word_mask = self._geometry_to_mask(det['geometry'], image_shape)
            mask = np.logical_or(mask, word_mask)
        
        # Convert to uint8
        mask = mask.astype(np.uint8)
        
        return mask, detections 