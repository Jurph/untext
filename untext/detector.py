"""Text detection module using DocTR's DBNet++ implementation.

This module provides a high-level interface for detecting text regions in images using
DocTR's DBNet++ model. It handles image preprocessing, text detection, and mask generation
for text regions.

The module requires DocTR version >= 0.11.0 for compatibility with the DBNet++ model
and its API structure.

Note: DocTR's detection results are returned as a list of dictionaries, where each
dictionary contains a 'words' key with a list of numpy arrays. Each array has shape (5,)
containing [x1, y1, x2, y2, score] where coordinates are normalized to [0,1].
"""

from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import cv2
import packaging.version
import os

# Import and version check
try:
    import doctr
    from doctr.io import DocumentFile
    from doctr.models import db_resnet50
    from doctr.models.detection.predictor import DetectionPredictor
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Verify minimum version
    if packaging.version.parse(doctr.__version__) < packaging.version.parse("0.11.0"):
        raise ImportError(
            f"DocTR version {doctr.__version__} is not supported. "
            "Please upgrade to version >= 0.11.0"
        )
except ImportError as e:
    raise ImportError(
        "Failed to import DocTR or its dependencies. "
        "Please ensure doctr>=0.11.0 is installed."
    ) from e


class TextDetector:
    """Detects text in images using DBNet++.
    
    This class provides a high-level interface for text detection using DocTR's DBNet++
    model. It handles the entire pipeline from image loading to mask generation.
    
    The class expects DocTR's detection results to be a list of dictionaries, where each
    dictionary contains a 'words' key with a list of numpy arrays. Each array has shape (5,)
    containing [x1, y1, x2, y2, score] where coordinates are normalized to [0,1].
    
    Attributes:
        model: The DBNet++ model instance
        pre_processor: PreProcessor instance for image normalization
        predictor: DetectionPredictor instance for running inference
        confidence_threshold: Minimum confidence score for detections (0-1)
        mask_dilation: Number of pixels to dilate the text mask
        min_text_size: Minimum size (in pixels) for text to be considered
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        mask_dilation: int = 2,
        min_text_size: int = 10,
    ) -> None:
        """Initialize the text detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            mask_dilation: Number of pixels to dilate the text mask
            min_text_size: Minimum size (in pixels) for text to be considered
            
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        # Validate parameters
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if mask_dilation < 0:
            raise ValueError("mask_dilation must be non-negative")
        if min_text_size < 1:
            raise ValueError("min_text_size must be positive")
            
        # Initialize model and pre-processor with correct parameters from DBNet config
        self.model = db_resnet50(pretrained=True)
        self.pre_processor = PreProcessor(
            output_size=(1024, 1024),  # From default_cfgs in DBNet
            batch_size=1,
            mean=(0.798, 0.785, 0.772),  # From default_cfgs in DBNet
            std=(0.264, 0.2749, 0.287)   # From default_cfgs in DBNet
        )
        self.predictor = DetectionPredictor(pre_processor=self.pre_processor, model=self.model)
        self.confidence_threshold = confidence_threshold
        self.mask_dilation = mask_dilation
        self.min_text_size = min_text_size

    def _box_to_mask(
        self, box: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Convert box coordinates to a binary mask.
        
        Args:
            box: Numpy array of shape (5,) containing [x1, y1, x2, y2, score]
                where coordinates are normalized to [0,1]
            image_shape: (height, width) of the target image
            
        Returns:
            Binary mask where text region is 1, background is 0
            
        Raises:
            ValueError: If box is invalid or image_shape is invalid
        """
        # Validate inputs
        if not isinstance(box, np.ndarray):
            raise ValueError("Box must be a numpy array")
        if box.shape != (5,):
            raise ValueError("Box must have shape (5,) containing [x1, y1, x2, y2, score]")
        if len(image_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers")
            
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image_shape
        x1, y1, x2, y2, _ = box
        points = np.array([
            [int(x1 * w), int(y1 * h)],
            [int(x2 * w), int(y1 * h)],
            [int(x2 * w), int(y2 * h)],
            [int(x1 * w), int(y2 * h)]
        ], dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [points], 1)
        
        # Dilate mask to ensure we catch all text pixels
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask

    def _filter_detections(
        self, words: List[np.ndarray], image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Filter detections based on size and other criteria.
        
        Args:
            words: List of numpy arrays, each with shape (5,) containing [x1, y1, x2, y2, score]
            image_shape: (height, width) of the image
            
        Returns:
            Filtered list of detection results, each containing:
            - 'geometry': Numpy array of box coordinates [x1, y1, x2, y2, score]
            - 'confidence': Detection confidence score (0-1)
            - 'text': Placeholder for OCR text (not available in detection)
            
        Raises:
            ValueError: If image_shape is invalid
        """
        # Validate inputs
        if len(image_shape) != 2 or not all(isinstance(x, int) and x > 0 for x in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers")
            
        filtered = []
        h, w = image_shape
        
        for word in words:
            # Get text region size in pixels
            x1, y1, x2, y2, score = word
            width = int((x2 - x1) * w)
            height = int((y2 - y1) * h)
            
            # Filter based on size
            if width < self.min_text_size or height < self.min_text_size:
                continue
                
            # Filter based on position (optional: remove text too close to edges)
            margin = 10
            if (x1 * w < margin or y1 * h < margin or
                x2 * w > w - margin or y2 * h > h - margin):
                continue
                
            # Filter based on confidence
            if score < self.confidence_threshold:
                continue
                
            filtered.append({
                'geometry': word,
                'confidence': float(score),
                'text': ''  # Placeholder for OCR text
            })
            
        return filtered

    def detect(self, image_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Detect text in an image.
        
        This method runs the complete text detection pipeline:
        1. Loads and preprocesses the image
        2. Runs text detection using DBNet++
        3. Filters detections based on size and confidence
        4. Generates a binary mask of text regions
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
            - Binary mask of detected text regions (numpy.ndarray)
            - List of detection results, each containing:
              - 'geometry': Numpy array of box coordinates [x1, y1, x2, y2, score]
              - 'confidence': Detection confidence score (0-1)
              - 'text': Placeholder for OCR text (not available in detection)
              
        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If image cannot be loaded or is invalid
            RuntimeError: If detection fails
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load and preprocess the image
        try:
            doc = DocumentFile.from_images(image_path)
            if not doc or len(doc) == 0:
                raise ValueError("Failed to load image")
            image = doc[0]
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                raise ValueError("Invalid image format")
            image_shape = (image.shape[0], image.shape[1])
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}") from e
        
        # Run detection
        try:
            result = self.predictor(doc)
            if not result:
                return np.zeros(image_shape, dtype=np.uint8), []
                
            # Get words from first page result
            if not isinstance(result[0], dict) or 'words' not in result[0]:
                raise RuntimeError("Invalid detection result format")
            words = result[0]['words']
        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}") from e
        
        # Filter detections
        filtered_words = self._filter_detections(words, image_shape)
        
        # Create binary mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        for word in filtered_words:
            word_mask = self._box_to_mask(word['geometry'], image_shape)
            mask = np.logical_or(mask, word_mask)
        
        # Convert to uint8
        mask = mask.astype(np.uint8)
        
        return mask, filtered_words 