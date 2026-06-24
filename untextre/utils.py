"""Shared utilities and type definitions for untextre."""

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import logging
from PIL import Image, JpegImagePlugin, PngImagePlugin
import struct

# Type aliases for clarity
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
Color = Tuple[int, int, int]  # BGR color tuple
BBox = Tuple[int, int, int, int]  # (x, y, width, height)
ImagePath = Union[str, Path]

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ---------------------------------------------------------------------------
# Confidence threshold constants
# ---------------------------------------------------------------------------
# There are three distinct confidence values in this system.  They are
# intentionally different and serve different purposes.
#
# MODEL_CONFIDENCE_FLOOR (0.1):
#     The internal threshold baked into the DocTR model at initialization.
#     Set deliberately low so that ALL plausible detections survive the
#     model's own filtering.  The *real* user-facing threshold is applied
#     as a post-filter at detection time -- this lets users adjust the
#     slider without re-initializing the model (which wastes VRAM and
#     takes 30+ seconds).
#
# CLI_DEFAULT_CONFIDENCE (0.3):
#     Conservative default for CLI / batch processing where there is no
#     human in the loop.  A higher threshold reduces false positives at
#     the cost of potentially missing faint watermarks.
#
# WEB_DEFAULT_CONFIDENCE (0.025):
#     Aggressive default for the Streamlit web UI where the user can see
#     results and adjust interactively.  Because consensus detection
#     requires 2+ detectors to agree, even very low thresholds rarely
#     produce false positives.  Values as low as 0.03 have been found
#     effective in practice.
# ---------------------------------------------------------------------------
MODEL_CONFIDENCE_FLOOR: float = 0.1
CLI_DEFAULT_CONFIDENCE: float = 0.3
WEB_DEFAULT_CONFIDENCE: float = 0.025

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

def _as_pillow_image(image: ImageArray, output_suffix: str) -> Image.Image:
    """Convert an OpenCV-style array to a Pillow image."""
    if image.ndim == 2:
        return Image.fromarray(image)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape for save: {image.shape}")

    if image.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if image.shape[2] == 4:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        if output_suffix in {'.jpg', '.jpeg'}:
            return pil_image.convert("RGB")
        return pil_image

    raise ValueError(f"Unsupported channel count for save: {image.shape[2]}")


def _safe_dpi(info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return a Pillow-compatible DPI tuple when the source provides one."""
    dpi = info.get("dpi")
    if not dpi or len(dpi) < 2:
        return None

    try:
        x_dpi = float(dpi[0])
        y_dpi = float(dpi[1])
    except (TypeError, ValueError):
        return None

    if x_dpi <= 0 or y_dpi <= 0:
        return None
    return (x_dpi, y_dpi)


def _png_color_fallback_chunks(info: Dict[str, Any]) -> Optional[PngImagePlugin.PngInfo]:
    """Carry PNG color-rendering chunks when no ICC profile is available."""
    pnginfo = PngImagePlugin.PngInfo()
    added = False

    srgb = info.get("srgb")
    if srgb is not None:
        try:
            pnginfo.add(b"sRGB", bytes([int(srgb) & 0xFF]))
            added = True
        except (TypeError, ValueError):
            pass

    gamma = info.get("gamma")
    if gamma is not None:
        try:
            pnginfo.add(b"gAMA", int(round(float(gamma) * 100000)).to_bytes(4, "big"))
            added = True
        except (OverflowError, TypeError, ValueError):
            pass

    chromaticity = info.get("chromaticity")
    if chromaticity and len(chromaticity) == 8:
        try:
            values = [int(round(float(value) * 100000)) for value in chromaticity]
            pnginfo.add(b"cHRM", struct.pack(">8I", *values))
            added = True
        except (struct.error, TypeError, ValueError):
            pass

    return pnginfo if added else None


def _pillow_save_kwargs(source_path: ImagePath, output_suffix: str, quality: int) -> Dict[str, Any]:
    """Build a whitelisted rendering-metadata bundle from the source image."""
    kwargs: Dict[str, Any] = {}
    with Image.open(source_path) as source:
        icc_profile = source.info.get("icc_profile")
        dpi = _safe_dpi(source.info)

        if output_suffix in {'.jpg', '.jpeg'}:
            kwargs["quality"] = quality
            sampling = JpegImagePlugin.get_sampling(source)
            if sampling in {0, 1, 2}:
                kwargs["subsampling"] = sampling
        elif output_suffix == '.png':
            kwargs["compress_level"] = 8
            if not icc_profile:
                pnginfo = _png_color_fallback_chunks(source.info)
                if pnginfo is not None:
                    kwargs["pnginfo"] = pnginfo
        elif output_suffix == '.webp':
            kwargs["quality"] = quality

        if icc_profile:
            kwargs["icc_profile"] = icc_profile
        if dpi:
            kwargs["dpi"] = dpi

    return kwargs


def _save_image_with_source_metadata(
    image: ImageArray,
    output_path: Path,
    quality: int,
    source_path: ImagePath,
) -> bool:
    """Save with selected source rendering metadata; return False if unsupported."""
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}:
        return False

    pil_image = _as_pillow_image(image, output_suffix)
    save_kwargs = _pillow_save_kwargs(source_path, output_suffix, quality)
    pil_image.save(output_path, **save_kwargs)
    return True


def save_image(
    image: ImageArray,
    output_path: ImagePath,
    quality: int = 97,
    source_path: Optional[ImagePath] = None,
) -> None:
    """Save an image to file with quality control.
    
    Args:
        image: Image array in BGR format
        output_path: Path where to save the image
        source_path: Optional original image path whose rendering metadata
            (ICC profile, density, and PNG color fallback chunks) should be
            carried forward. Full EXIF/XMP is intentionally not copied.
        
    Raises:
        ValueError: If image cannot be saved
    """
    output_path = Path(output_path)

    if source_path is not None:
        try:
            if _save_image_with_source_metadata(image, output_path, quality, source_path):
                return
        except Exception as exc:  # pragma: no cover - exercised through caller failure path
            raise ValueError(f"Could not save image to: {output_path}") from exc
    
    # Set compression parameters based on file extension
    if output_path.suffix.lower() in {'.jpg', '.jpeg'}:
        # JPEG with specified quality
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_path.suffix.lower() == '.png':
        # PNG with high compression (0-9, where 9 is max compression)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 8]  # High compression for smaller files
    else:
        # Default parameters for other formats
        params = []
    
    success = cv2.imwrite(str(output_path), image, params)
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


def pad_bbox_to_multiple(bbox: BBox, multiple: int = 4, image_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """Pad a bounding box to ensure width and height are divisible by a given multiple.
    
    This prevents pixel alignment issues in neural networks that require
    dimensions divisible by specific values (e.g., 4, 8, 16, 32).
    Padding is applied symmetrically when possible.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        multiple: Value that dimensions must be divisible by (default: 4)
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Padded bounding box with width and height divisible by multiple
    """
    if multiple <= 0:
        raise ValueError(f"Multiple must be positive, got {multiple}")
    
    x, y, w, h = bbox
    
    # Calculate padding needed to make dimensions divisible by multiple
    w_pad = (multiple - (w % multiple)) % multiple  # 0 if already divisible
    h_pad = (multiple - (h % multiple)) % multiple
    
    # Apply padding symmetrically when possible
    w_pad_left = w_pad // 2
    w_pad_right = w_pad - w_pad_left
    h_pad_top = h_pad // 2
    h_pad_bottom = h_pad - h_pad_top
    
    # Calculate new coordinates
    new_x = x - w_pad_left
    new_y = y - h_pad_top
    new_w = w + w_pad
    new_h = h + h_pad
    
    # Clamp to image bounds if provided
    # Track which edges touch the image boundary - we cannot lose pixels at those edges
    # and we cannot extend past those edges
    touches_left = False
    touches_right = False
    touches_top = False
    touches_bottom = False
    
    if image_shape is not None:
        img_h, img_w = image_shape
        
        # Adjust if padding extends beyond image bounds
        if new_x < 0:
            # Shift right padding to compensate
            w_pad_right += abs(new_x)
            new_x = 0
            new_w = w + w_pad  # Recalculate width with adjusted padding
            
        if new_y < 0:
            # Shift bottom padding to compensate  
            h_pad_bottom += abs(new_y)
            new_y = 0
            new_h = h + h_pad  # Recalculate height with adjusted padding
            
        if new_x + new_w > img_w:
            new_w = img_w - new_x
            touches_right = True  # Right edge is at image boundary
            
        if new_y + new_h > img_h:
            new_h = img_h - new_y
            touches_bottom = True  # Bottom edge is at image boundary
        
        # Check if bbox is at left/top edge AFTER all clamping
        # (may have started at 0, or been clamped to 0)
        touches_left = (new_x == 0)
        touches_top = (new_y == 0)
    
    # Ensure final dimensions are divisible by multiple
    # CRITICAL: If bbox touches image edge, we must NOT round down (would lose edge pixels)
    # Instead, round UP and shift x/y left/up to compensate - BUT only if there's room
    
    w_remainder = new_w % multiple
    h_remainder = new_h % multiple
    
    if w_remainder != 0:
        if touches_right:
            # Can't extend right, so try to extend left
            w_needed = multiple - w_remainder
            if touches_left:
                # Can't extend left either! Bbox spans full image width.
                # Accept non-mod-8 width; SimpleLama will pad internally.
                pass
            else:
                # Extend left: shift x and increase width
                new_x = max(0, new_x - w_needed)
                new_w = new_w + w_needed
        else:
            # Safe to round down (right side is padding area, not real content)
            new_w = (new_w // multiple) * multiple
    
    if h_remainder != 0:
        if touches_bottom:
            # Can't extend down, so try to extend up
            h_needed = multiple - h_remainder
            if touches_top:
                # Can't extend up either! Bbox spans full image height.
                # Accept non-mod-8 height; SimpleLama will pad internally.
                pass
            else:
                # Extend up: shift y and increase height
                new_y = max(0, new_y - h_needed)
                new_h = new_h + h_needed
        else:
            # Safe to round down (bottom side is padding area, not real content)
            new_h = (new_h // multiple) * multiple
    
    # Ensure dimensions are at least the minimum
    final_w = max(multiple, new_w)
    final_h = max(multiple, new_h)
    
    return (new_x, new_y, final_w, final_h)


def dilate_by_pixels(image: ImageArray, bbox: BBox, pixels: int) -> BBox:
    """Dilate a bounding box by a specific number of pixels.
    
    Args:
        image: Input image to get dimensions from
        bbox: Bounding box as (x, y, width, height)
        pixels: Number of pixels to dilate by
        
    Returns:
        Dilated bounding box clamped to image bounds
    """
    return dilate_bbox(bbox, pixels, image.shape[:2])

def color_distance(color1: Color, color2: Color) -> float:
    """Calculate Euclidean distance between two BGR colors.
    
    Args:
        color1: First color as BGR tuple
        color2: Second color as BGR tuple
        
    Returns:
        Euclidean distance between colors
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2))))

def calculate_bbox_superset(bboxes: List[BBox], image_shape: Optional[Tuple[int, int]] = None) -> Optional[BBox]:
    """Calculate the superset bounding box that contains all input boxes.
    
    The superset box uses:
    - Leftmost left coordinate (minimum x)
    - Uppermost top coordinate (minimum y)
    - Rightmost right coordinate (maximum x + width)
    - Bottommost bottom coordinate (maximum y + height)
    
    Args:
        bboxes: List of bounding boxes as (x, y, width, height) tuples
        image_shape: Optional image shape to clamp coordinates
        
    Returns:
        Superset bounding box, or None if bboxes is empty
    """
    if not bboxes:
        return None
    
    # Find the bounding box that contains all boxes
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
    max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
    
    superset = (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Clamp to image bounds if provided
    if image_shape is not None:
        superset = clamp_bbox_to_image(superset, image_shape)
    
    return superset
