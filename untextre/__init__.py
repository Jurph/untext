"""Untextre: Clean, efficient text watermark removal tool.

This package provides a streamlined pipeline for detecting and removing
text-based watermarks from images using OCR detection and inpainting.
"""

__version__ = "0.1.0"
__author__ = "Untextre Team"

# Main pipeline components
from .preprocessor import preprocess_image
from .detector import detect_text_regions, TextDetector, initialize_models
from .find_text_colors import find_text_colors, find_mask_by_spatial_tf_idf
from .mask_generator import (
    generate_mask, 
    create_color_mask, 
    clean_up_mask,
    morph_clean_mask,
    morph_expand_mask, 
    morph_smooth_mask,
    anchor_connected_components
)
from .inpaint import inpaint_image
from .lama_inpainter import LamaInpainter
from .utils import load_image, save_image, setup_logger

# CLI entry point
from .cli import main

__all__ = [
    "TextDetector",
    "initialize_models",
    "preprocess_image",
    "detect_text_regions", 
    "find_text_colors",
    "find_mask_by_spatial_tf_idf",
    "generate_mask",
    "create_color_mask",
    "clean_up_mask",
    "morph_clean_mask",
    "morph_expand_mask",
    "morph_smooth_mask",
    "anchor_connected_components",
    "inpaint_image",
    "LamaInpainter",
    "load_image",
    "save_image",
    "setup_logger",
    "main",
] 