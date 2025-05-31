"""untext - Remove text-based watermarks from images using Deep Image Prior."""

__version__ = "0.1.0"

from untext.detector import TextDetector
from untext.inpainter import DeepImagePriorInpainter

__all__ = ["TextDetector", "DeepImagePriorInpainter"] 