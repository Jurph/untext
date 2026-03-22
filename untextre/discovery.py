"""Watermark auto-discovery for -U mode.

Discovers a watermark template from a directory of consistently-watermarked
images and returns RGBA crop(s) suitable for the -K / ORB pipeline.
"""

import logging
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import load_image, setup_logger, IMAGE_EXTENSIONS

logger = setup_logger(__name__)

# Variance threshold: population variance of a pair in [0,1] luminance space.
# Corresponds to ~20% per-pixel luminance difference.
VARIANCE_THRESHOLD = 0.01

# Minimum blob area as a fraction of total image area.
MIN_BLOB_AREA_FRACTION = 0.0005  # 0.05%

# Morphological kernel sizes (match morph_clean_mask for consistency).
CLOSE_KERNEL_SIZE = 11
DILATE_SIZE = 13

# Border added around each BGRA crop (pixels).
CROP_BORDER_PX = 8

# Zone grid: long edge into thirds, short edge into halves → 6 zones.
ZONE_LONG_DIVISIONS = 3
ZONE_SHORT_DIVISIONS = 2

# Convergence parameters.
MAX_DRAWS = 50
STABLE_STREAK_REQUIRED = 10

# Cross-bucket IoU threshold for family membership.
CROSS_BUCKET_IOU_THRESHOLD = 0.50


def bucket_images_by_size(
    image_paths: List[Path],
) -> Dict[Tuple[int, int], List[Path]]:
    """Group image paths by exact pixel dimensions (W, H).

    Unreadable images are skipped and logged.

    Args:
        image_paths: List of candidate image paths.

    Returns:
        Dict mapping (width, height) → list of paths in that bucket.
    """
    buckets: Dict[Tuple[int, int], List[Path]] = {}
    for path in image_paths:
        try:
            img = load_image(path)
            h, w = img.shape[:2]
            key = (w, h)
            buckets.setdefault(key, []).append(path)
        except (IOError, cv2.error, ValueError) as e:
            logger.warning(f"Skipping unreadable image {path.name}: {e}")
    return buckets
