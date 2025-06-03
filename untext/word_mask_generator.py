"""Module for generating binary masks for detected text regions in images.

This module provides functionality to detect text in images using the DocTR library
and generate binary masks for the detected text regions.

Example:
    >>> from untext.word_mask_generator import WordMaskGenerator
    >>> generator = WordMaskGenerator()
    >>> mask_paths = generator.generate_masks(['image1.jpg', 'image2.jpg'])
    >>> print(f"Generated masks: {mask_paths}")
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor
from .detector import TextDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImagePath = Union[str, Path]
MaskPath = Union[str, Path]
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8

class WordMaskGenerator:
    """Class for generating binary masks for detected text regions in images."""
    
    def __init__(self, mode: str = "box") -> None:
        """Initialize the WordMaskGenerator.
        
        Args:
            mode: Mask generation mode ("box" or "letters")
        """
        if mode not in ["box", "letters"]:
            raise ValueError("mode must be either 'box' or 'letters'")
        self.mode = mode
        self.detector = TextDetector()
        logger.info(f"Initialized WordMaskGenerator with mode={mode}")
    
    def generate_masks(
        self,
        image_paths: List[ImagePath],
        output_dir: Optional[ImagePath] = None
    ) -> Dict[Path, Path]:
        """Generate binary masks for detected text regions in images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Optional directory to save masks. If None, will save in
                       the same directory as input images.
        
        Returns:
            Dictionary mapping input image paths to their mask paths
        
        Raises:
            ValueError: If any input path is invalid
            RuntimeError: If no masks are generated
        """
        # Convert paths to Path objects
        image_paths = [Path(p) for p in image_paths]
        
        # Validate input paths
        for path in image_paths:
            if not path.exists():
                raise ValueError(f"Input image not found: {path}")
        
        # Set up output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_paths = {}
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                
                # Detect text
                logger.info(f"Detecting text in {image_path}")
                _, detections = self.detector.detect(image)
                
                if not detections:
                    logger.warning(f"No text detected in {image_path}")
                    continue
                
                # Generate mask
                mask = self._create_mask(image.shape[:2], detections)
                
                # Save mask
                if output_dir is not None:
                    mask_path = output_dir / f"{image_path.stem}_mask.png"
                else:
                    mask_path = image_path.parent / f"{image_path.stem}_mask.png"
                
                cv2.imwrite(str(mask_path), mask)
                mask_paths[image_path] = mask_path
                logger.info(f"Generated mask for {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        if not mask_paths:
            raise RuntimeError("No masks were generated")
        
        logger.info(f"Generated {len(mask_paths)} masks")
        return mask_paths
    
    def _create_mask(
        self,
        image_shape: Tuple[int, int],
        detections: List[Dict]
    ) -> MaskArray:
        """Create a binary mask for detected text regions.
        
        Args:
            image_shape: Shape of the image (height, width)
            detections: List of detection dictionaries from TextDetector
            
        Returns:
            Binary mask as H×W uint8 numpy array (255 = text region)
        """
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Fill detected regions
        for det in detections:
            points = det['geometry'].astype(np.int32)
            if self.mode == "box":
                # Use bounding box
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            else:  # letters mode
                # Use exact polygon
                cv2.fillPoly(mask, [points], 255)
        
        return mask 