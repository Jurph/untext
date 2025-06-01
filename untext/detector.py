"""Text detection module using DocTR."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from doctr.io import DocumentFile
from doctr.models import detection

logger = logging.getLogger(__name__)

class TextDetector:
    """Class for detecting text in images using DocTR."""
    
    def __init__(self) -> None:
        """Initialize the text detector."""
        self.model = detection.db_resnet50(pretrained=True)
        logger.info("Initialized text detector")
    
    def detect_text(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Detect text in an image.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path to save the visualization
            
        Returns:
            Tuple of (image with bounding boxes, list of bounding boxes)
        """
        # Load image
        doc = DocumentFile.from_images(image_path)
        
        # Run detection
        result = self.model(doc)
        
        # Extract bounding boxes
        boxes = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        boxes.append(word.geometry)
        
        # Draw boxes on image
        image = cv2.imread(image_path)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save visualization if requested
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info("Saved detection visualization to %s", output_path)
        
        return image, boxes 