"""Module for generating word masks using DocTR text detection.

This module provides functionality to detect text in images and generate binary masks
for the detected text regions using the DocTR library.

Example:
    >>> from untext.word_mask_generator import WordMaskGenerator
    >>> generator = WordMaskGenerator()
    >>> mask_paths = generator.generate_masks(['image1.jpg', 'image2.png'])
    >>> print(f"Generated masks: {mask_paths}")
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union
from doctr.io import DocumentFile
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WordMaskGenerator:
    """Class to generate word masks for images using DocTR text detection."""

    def __init__(self) -> None:
        """Initialize the WordMaskGenerator with DocTR's text detection model."""
        try:
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
            logger.info("Successfully initialized WordMaskGenerator")
        except Exception as e:
            logger.error("Failed to initialize WordMaskGenerator: %s", str(e))
            raise RuntimeError(f"Failed to initialize WordMaskGenerator: {str(e)}") from e

    def generate_masks(
        self,
        image_paths: List[Union[str, Path]]
    ) -> Dict[Path, Path]:
        """Generate binary masks for detected text in images.
        
        Args:
            image_paths: List of paths to images to process. Images should be in a
                        format supported by OpenCV (e.g., PNG, JPEG).
        
        Returns:
            Dictionary mapping input image paths to their corresponding mask paths.
        
        Raises:
            ValueError: If any input path is invalid.
            FileNotFoundError: If any input image cannot be found.
            RuntimeError: If mask generation fails.
        """
        # Convert paths to Path objects
        image_paths = [Path(p) for p in image_paths]
        
        # Validate input files
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
        
        mask_paths = {}
        for image_path in image_paths:
            try:
                logger.info("Processing image: %s", image_path)
                
                # Load the image
                doc = DocumentFile.from_images(str(image_path))
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                # Run prediction
                result = self.predictor(doc)
                if not result:
                    logger.warning("No text detected in %s", image_path)
                    continue

                # Process results
                first_result = result[0]
                if 'words' not in first_result:
                    logger.warning("No words found in detection result for %s", image_path)
                    continue

                # Create a single mask for all words in the image
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                words = first_result['words']
                
                for word in words:
                    if not isinstance(word, (list, np.ndarray)) or len(word) != 5:
                        logger.warning("Invalid word format detected in %s", image_path)
                        continue
                        
                    h, w = image.shape[:2]
                    x1, y1, x2, y2, score = word
                    cv2.rectangle(
                        mask,
                        (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)),
                        255,
                        -1
                    )

                # Save the mask
                mask_path = image_path.with_suffix('').with_name(
                    f"{image_path.stem}_mask.jpg"
                )
                cv2.imwrite(str(mask_path), mask)
                mask_paths[image_path] = mask_path
                logger.info("Generated mask for %s: %s", image_path, mask_path)

            except Exception as e:
                logger.error("Failed to process %s: %s", image_path, str(e))
                continue

        if not mask_paths:
            raise RuntimeError("No masks were generated for any images")
            
        return mask_paths 