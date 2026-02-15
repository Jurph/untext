"""Untextre: Clean, efficient text watermark removal tool.

This package provides a streamlined pipeline for detecting and removing
text-based watermarks from images using OCR detection and inpainting.

Heavy ML modules (detector, consensus, inpaint, lama_inpainter, etc.)
are imported lazily so that lightweight entry points like ``--help`` and
watermark-only CLI runs don't pay the TensorFlow / PyTorch startup cost.
"""

__version__ = "0.1.0"
__author__ = "Untextre Team"

__all__ = [
    "TextDetector",
    "initialize_models",
    "cleanup_vram",
    "preprocess_image",
    "detect_text_regions",
    "find_mask_by_spatial_tf_idf",
    "hex_to_bgr",
    "html_to_bgr",
    "morph_clean_mask",
    "inpaint_image",
    "LamaInpainter",
    "load_image",
    "save_image",
    "setup_logger",
    "MODEL_CONFIDENCE_FLOOR",
    "CLI_DEFAULT_CONFIDENCE",
    "WEB_DEFAULT_CONFIDENCE",
    "main",
]


def __getattr__(name: str):
    """Lazy-load submodule symbols on first access."""
    # Lightweight utils — always safe to load immediately
    if name in (
        "load_image", "save_image", "setup_logger",
        "MODEL_CONFIDENCE_FLOOR", "CLI_DEFAULT_CONFIDENCE",
        "WEB_DEFAULT_CONFIDENCE",
    ):
        from .utils import (
            load_image, save_image, setup_logger,
            MODEL_CONFIDENCE_FLOOR, CLI_DEFAULT_CONFIDENCE,
            WEB_DEFAULT_CONFIDENCE,
        )
        return locals()[name]

    # Heavy modules — only load when someone asks for them
    if name in ("TextDetector", "initialize_models", "cleanup_vram", "detect_text_regions"):
        from . import detector
        return getattr(detector, name)

    if name == "preprocess_image":
        from .preprocessor import preprocess_image
        return preprocess_image

    if name in ("find_mask_by_spatial_tf_idf", "hex_to_bgr", "html_to_bgr"):
        from . import find_text_colors
        return getattr(find_text_colors, name)

    if name == "morph_clean_mask":
        from .mask_generator import morph_clean_mask
        return morph_clean_mask

    if name == "inpaint_image":
        from .inpaint import inpaint_image
        return inpaint_image

    if name == "LamaInpainter":
        from .lama_inpainter import LamaInpainter
        return LamaInpainter

    if name == "main":
        from .cli import main
        return main

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
