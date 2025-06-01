"""Module for masking and patching images using DocTR and Deep Image Prior.

This module provides functionality to detect text in images, generate masks for the detected
text, and use Deep Image Prior to inpaint the masked regions.

Example:
    >>> from untext.mask_image import process_images
    >>> results = process_images(['image1.jpg', 'image2.png'], output_dir='output')
    >>> print(f"Generated masks: {results['masks']}")
    >>> print(f"Patched images: {results['patched']}")
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from .word_mask_generator import WordMaskGenerator
from .image_patcher import ImagePatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_images(
    image_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    device: Literal['cuda', 'cpu'] = 'cpu'
) -> Dict[str, Dict[Path, Path]]:
    """Process a list of images by generating masks and patching them.
    
    Args:
        image_paths: List of paths to images to process. Images should be in a format
                    supported by OpenCV (e.g., PNG, JPEG).
        output_dir: Directory to save output files. If None, will save in the same
                   directory as input images.
        device: Device to run Deep Image Prior on ('cuda' or 'cpu').
    
    Returns:
        Dictionary containing paths to generated files:
            - 'masks': dict mapping input paths to mask paths
            - 'patched': dict mapping input paths to patched image paths
    
    Raises:
        ValueError: If any input path is invalid or if device is not 'cuda' or 'cpu'.
        FileNotFoundError: If any input image cannot be found.
        RuntimeError: If image processing fails.
    """
    # Validate inputs
    if not image_paths:
        raise ValueError("No image paths provided")
    
    if device not in ('cuda', 'cpu'):
        raise ValueError("Device must be either 'cuda' or 'cpu'")
    
    # Convert paths to Path objects
    image_paths = [Path(p) for p in image_paths]
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input files
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
    
    try:
        # Generate masks
        logger.info("Generating masks for %d images", len(image_paths))
        generator = WordMaskGenerator()
        mask_paths = generator.generate_masks(image_paths)
        
        if not mask_paths:
            raise RuntimeError("No masks were generated")
        
        # Patch images using the masks
        logger.info("Patching %d images", len(mask_paths))
        patcher = ImagePatcher(device=device)
        patched_paths = patcher.patch_images(mask_paths, output_dir)
        
        if not patched_paths:
            raise RuntimeError("No images were patched")
        
        logger.info("Successfully processed %d images", len(patched_paths))
        return {
            'masks': mask_paths,
            'patched': patched_paths
        }
        
    except Exception as e:
        logger.error("Failed to process images: %s", str(e))
        raise RuntimeError(f"Image processing failed: {str(e)}") from e 