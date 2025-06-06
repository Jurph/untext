"""untext - Remove text-based watermarks from images using Deep Image Prior.

This package provides tools for detecting text in images and generating masks
for subsequent inpainting to remove watermarks and other text overlays.

Key components:
- TextDetector: DocTR-based text detection and OCR with fixed preprocessors
- WordMaskGenerator: Binary mask generation for box or letter-level text removal
- DeepImagePriorInpainter: Neural network-based inpainting for watermark removal

DocTR Fixes Applied:
This package includes critical fixes for DocTR's broken preprocessor configurations
that cause dimension squashing issues. See docs/DocTR-format.md for details.

Example:
    >>> from untext import TextDetector, WordMaskGenerator
    >>> 
    >>> # Box mode: fast rectangular masks (DocTR detection only)
    >>> detector = TextDetector()
    >>> mask_gen = WordMaskGenerator(mode="box")
    >>> 
    >>> # Letter mode: precise character masks (DocTR OCR with fixes)
    >>> mask_gen_precise = WordMaskGenerator(mode="letter")
"""

__version__ = "0.2.0"

from untext.detector import TextDetector
from untext.word_mask_generator import WordMaskGenerator
from untext.inpainter import DeepImagePriorInpainter

__all__ = [
    "TextDetector", 
    "WordMaskGenerator", 
    "DeepImagePriorInpainter",
    "__version__"
] 