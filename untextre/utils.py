"""Shared utilities and type definitions for untextre."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import logging

# Type aliases for clarity
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
Color = Tuple[int, int, int]  # BGR color tuple
BBox = Tuple[int, int, int, int]  # (x, y, width, height)
ImagePath = Union[str, Path]

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(level)
    return logger

def load_image(image_path: ImagePath) -> ImageArray:
    """Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array in BGR format
        
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

def save_image(image: ImageArray, output_path: ImagePath) -> None:
    """Save an image to file.
    
    Args:
        image: Image array in BGR format
        output_path: Path where to save the image
        
    Raises:
        ValueError: If image cannot be saved
    """
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise ValueError(f"Could not save image to: {output_path}")

def get_image_files(path: Path) -> List[Path]:
    """Get list of image files from path (file or directory).
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of image file paths
    """
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        else:
            return []
    
    return [f for f in path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]

def clamp_bbox_to_image(bbox: BBox, image_shape: Tuple[int, int]) -> BBox:
    """Clamp bounding box coordinates to stay within image bounds.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        image_shape: Image shape as (height, width)
        
    Returns:
        Clamped bounding box
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape
    
    # Clamp coordinates
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    # Adjust width and height to stay in bounds
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    return (x, y, w, h)

def dilate_bbox(bbox: BBox, dilation: int, image_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """Dilate a bounding box by the specified amount.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        dilation: Number of pixels to dilate by
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Dilated bounding box
    """
    x, y, w, h = bbox
    x1 = x - dilation
    y1 = y - dilation
    x2 = x + w + dilation
    y2 = y + h + dilation
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        img_h, img_w = image_shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
    
    return (x1, y1, x2 - x1, y2 - y1)

def color_distance(color1: Color, color2: Color) -> float:
    """Calculate Euclidean distance between two BGR colors.
    
    Args:
        color1: First color as BGR tuple
        color2: Second color as BGR tuple
        
    Returns:
        Euclidean distance between colors
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))) 