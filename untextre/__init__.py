"""Untextre: Clean, efficient text watermark removal tool.

This package provides a streamlined pipeline for detecting and removing
text-based watermarks from images using OCR detection and inpainting.
"""

__version__ = "0.1.0"
__author__ = "Untextre Team"

# Main pipeline components
from .preprocessor import preprocess_image
from .detector import detect_text_regions
from .find_text_colors import find_text_colors
from .mask_generator import generate_mask
from .inpaint import inpaint_image

# CLI entry point
from .cli import main

__all__ = [
    "preprocess_image",
    "detect_text_regions", 
    "find_text_colors",
    "generate_mask",
    "inpaint_image",
    "main",
] 