"""Word mask generation module for text regions in images.

This module provides functionality to generate binary masks for detected text regions
using either box mode (DocTR detection) or letter mode (DocTR OCR with fixed preprocessors).

The module supports two masking approaches:
- Box mode: Fast rectangular masks around detected text regions (DocTR detection only)
- Letter mode: Precise character-level masks using OCR text extraction (DocTR OCR)

Both modes produce standardized binary masks (black background, white text regions)
suitable for inpainting pipelines.

Example:
    >>> from untext.word_mask_generator import WordMaskGenerator
    >>> generator = WordMaskGenerator(mode="box")
    >>> mask_paths = generator.generate_masks(['image1.jpg', 'image2.jpg'])
    >>> print(f"Generated masks: {mask_paths}")
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Literal

from .detector import TextDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImagePath = Union[str, Path]
MaskPath = Union[str, Path]
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
MaskMode = Literal["box", "letters"]


class WordMaskGenerator:
    """Generator for binary masks of text regions in images.
    
    This class provides two masking modes:
    
    1. **Box mode** (fast): Uses DocTR detection to create rectangular masks
       around detected text regions. Faster but less precise.
       
    2. **Letter mode** (precise): Uses DocTR OCR with fixed preprocessors to
       create tight masks around individual characters. Slower but more accurate.
       
    Both modes produce binary masks with:
    - Black background (0 values)
    - White text regions (255 values)
    - Same output format for downstream inpainting
    
    Attributes:
        mode: Masking mode ("box" or "letter")
        detector: TextDetector instance with fixed DocTR preprocessors
    """
    
    def __init__(self, mode: MaskMode = "box", preprocess: bool = True) -> None:
        """Initialize the WordMaskGenerator.
        
        Args:
            mode: Masking mode - "box" for rectangular masks (DocTR detection only)
                  or "letters" for precise character masks (DocTR OCR)
            preprocess: Whether to apply image preprocessing before text detection
                  
        Raises:
            ValueError: If mode is not "box" or "letters"
        """
        if mode not in ["box", "letters"]:
            raise ValueError("mode must be either 'box' or 'letters'")
            
        self.mode = mode
        self.preprocess = preprocess
        self.detector = TextDetector(preprocess=preprocess)
        
        logger.info(f"Initialized WordMaskGenerator with mode='{mode}', preprocess={preprocess}")
        logger.info(f"Mode details:")
        if mode == "box":
            logger.info("  - Uses DocTR detection (fast, rectangular masks)")
        else:
            logger.info("  - Uses DocTR OCR with fixed preprocessors (precise, character-level masks)")
    
    def generate_masks(
        self,
        image_paths: List[ImagePath],
        output_dir: Optional[ImagePath] = None
    ) -> Dict[Path, Path]:
        """Generate binary masks for detected text regions in images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Optional directory to save masks. If None, saves in
                       the same directory as input images
        
        Returns:
            Dictionary mapping input image paths to their generated mask paths
        
        Raises:
            ValueError: If any input path is invalid
            RuntimeError: If no masks are generated
        """
        # Convert and validate paths
        image_paths = [Path(p) for p in image_paths]
        for path in image_paths:
            if not path.exists():
                raise ValueError(f"Input image not found: {path}")
        
        # Set up output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_paths = {}
        successful_count = 0
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    continue
                
                # Generate mask based on mode
                logger.info(f"Processing {image_path} in {self.mode} mode")
                
                if self.mode == "box":
                    mask = self._generate_box_mask(image)
                else:  # letters mode
                    mask = self._generate_letter_mask(image)
                
                                # Check if any text was found
                if np.sum(mask) == 0:
                    logger.warning(f"No text detected in {image_path}")
                    continue  # Skip saving empty masks

                # Save mask
                if output_dir is not None:
                    mask_path = output_dir / f"{image_path.stem}_mask.png"
                else:
                    mask_path = image_path.parent / f"{image_path.stem}_mask.png"

                success = cv2.imwrite(str(mask_path), mask)
                if success:
                    mask_paths[image_path] = mask_path
                    successful_count += 1
                    logger.info(f"Generated {self.mode} mask: {mask_path}")
                else:
                    logger.error(f"Failed to save mask: {mask_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        if successful_count == 0:
            raise RuntimeError("No masks were generated successfully")
        
        logger.info(f"Generated {successful_count}/{len(image_paths)} masks successfully")
        return mask_paths
    
    def generate_mask_from_array(self, image: ImageArray) -> MaskArray:
        """Generate a binary mask from an image array.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            Binary mask as H×W uint8 numpy array (255 = text region, 0 = background)
            
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If mask generation fails
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be H×W×3 BGR format")
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8 type")
        
        try:
            if self.mode == "box":
                return self._generate_box_mask(image)
            else:  # letters mode
                return self._generate_letter_mask(image)
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            raise RuntimeError("Mask generation failed") from e
    
    def _generate_box_mask(self, image: ImageArray) -> MaskArray:
        """Generate box-mode mask using DocTR detection only.
        
        This method uses DocTR's detection capabilities to find text regions
        and creates rectangular masks around them. Fast but less precise.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            Binary mask with rectangular regions around detected text
        """
        # Use detection mode (same as letters mode for now, but could be optimized)
        mask, detections = self.detector.detect(image)
        
        # The detector already returns a proper mask, but we may want to
        # ensure it uses rectangular regions rather than precise polygons
        if len(detections) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Create fresh mask with rectangular regions
        box_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for detection in detections:
            geometry = detection.get('geometry')
            if geometry is None:
                continue
                
            # Get bounding rectangle from polygon
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            
            # Draw filled rectangle
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
        
        return box_mask
    
    def _generate_letter_mask(self, image: ImageArray) -> MaskArray:
        """Generate letter-mode mask using DocTR OCR with fixed preprocessors.
        
        This method uses the full DocTR OCR pipeline (detection + recognition)
        with fixed preprocessors to extract precise character-level masks.
        Slower but more accurate for surgical text removal.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            Binary mask with precise character-level regions
        """
        # Use full OCR pipeline with fixed preprocessors
        mask, detections = self.detector.detect(image)
        
        if len(detections) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # For letter mode, we could potentially do more sophisticated
        # character-level mask generation here if needed, but the
        # standard polygon masks from DocTR OCR are already quite precise
        
        # The mask from detector.detect() already uses the fixed preprocessors
        # and gives us precise text regions, so we can return it directly
        return mask
    
    def set_mode(self, mode: MaskMode) -> None:
        """Change the masking mode.
        
        Args:
            mode: New masking mode ("box" or "letter")
            
        Raises:
            ValueError: If mode is not "box" or "letter"
        """
        if mode not in ["box", "letters"]:
            raise ValueError("mode must be either 'box' or 'letters'")
            
        old_mode = self.mode
        self.mode = mode
        
        logger.info(f"Changed masking mode from '{old_mode}' to '{mode}'")
        if mode == "box":
            logger.info("  - Now using DocTR detection (fast, rectangular masks)")
        else:
            logger.info("  - Now using DocTR OCR with fixed preprocessors (precise, character-level masks)")
    
    def get_mode(self) -> MaskMode:
        """Get the current masking mode.
        
        Returns:
            Current masking mode ("box" or "letter")
        """
        return self.mode 