"""Module for masking and patching images using DocTR and Deep Image Prior.

This module provides functionality to detect text in images, generate masks for
the detected text regions, and patch the images using Deep Image Prior.

Example:
    >>> from untext.mask_image import process_images
    >>> image_paths = ['image1.jpg', 'image2.jpg']
    >>> output_dir = 'output'
    >>> process_images(image_paths, output_dir)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from .word_mask_generator import WordMaskGenerator
from .image_patcher import ImagePatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImagePath = Union[str, Path]
MaskPath = Union[str, Path]
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8

def process_images(
    image_paths: List[ImagePath],
    output_dir: Optional[ImagePath] = None,
    mask_dir: Optional[ImagePath] = None,
    patched_dir: Optional[ImagePath] = None,
    method: str = "lama"
) -> Dict[Path, Path]:
    """Process a list of images to detect text, generate masks, and patch them.
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save all outputs (masks and patched images)
        mask_dir: Optional separate directory for masks
        patched_dir: Optional separate directory for patched images
        method: Inpainting method to use ('lama', 'telea', or 'dip')
        
    Returns:
        Dictionary mapping input image paths to their patched image paths
        
    Raises:
        ValueError: If any input path is invalid
        RuntimeError: If no masks are generated or no images are patched
    """
    # Convert paths to Path objects
    image_paths = [Path(p) for p in image_paths]
    
    # Validate input paths
    for path in image_paths:
        if not path.exists():
            raise ValueError(f"Input image not found: {path}")
    
    # Set up output directories
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if mask_dir is None:
            mask_dir = output_dir / "masks"
        if patched_dir is None:
            patched_dir = output_dir / "patched"
    
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
    
    if patched_dir is not None:
        patched_dir = Path(patched_dir)
        patched_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate masks
    logger.info("Generating masks for %d images...", len(image_paths))
    mask_generator = WordMaskGenerator()
    mask_paths = mask_generator.generate_masks(image_paths, mask_dir)
    
    if not mask_paths:
        raise RuntimeError("No masks were generated")
    
    logger.info("Generated %d masks", len(mask_paths))
    
    # Patch images
    logger.info("Patching %d images...", len(mask_paths))
    patcher = ImagePatcher()
    patched_paths = patcher.patch_images(mask_paths, patched_dir)
    
    if not patched_paths:
        raise RuntimeError("No images were successfully patched")
    
    logger.info("Successfully patched %d images", len(patched_paths))
    return patched_paths 