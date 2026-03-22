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


def compute_pair_variance(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Compute per-pixel population variance of a 2-image luminance pair.

    Uses np.var(stack, axis=0) where stack shape is (2, H, W).
    Population variance of two values equals ((a - b) / 2)^2.

    Args:
        img_a: First image (H×W×3 BGR uint8).
        img_b: Second image (H×W×3 BGR uint8), same shape as img_a.

    Returns:
        Per-pixel variance map (H×W float32), values in [0, 0.25].
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    stack = np.stack([gray_a, gray_b], axis=0)
    return np.var(stack, axis=0).astype(np.float32)


def extract_blobs(
    variance_map: np.ndarray,
    image_area: int,
) -> List[Tuple[int, int]]:
    """Extract 8-connected low-variance blobs from a variance map.

    The variance map is the signal: pixels below VARIANCE_THRESHOLD are
    watermark candidates. No morphological operations are applied — the
    convergence loop (requiring blobs to appear across multiple independent
    draws) is the noise filter.

    Args:
        variance_map: Per-pixel variance (H×W float32).
        image_area: Total image area in pixels (H * W), used for min-area check.

    Returns:
        List of (cx, cy) blob centroids for blobs that exceed the minimum area.
    """
    min_area = max(1, int(image_area * MIN_BLOB_AREA_FRACTION))

    # Threshold: 255 where variance is LOW (candidate watermark pixels)
    binary = (variance_map < VARIANCE_THRESHOLD).astype(np.uint8) * 255

    # Find 8-connected contiguous blobs directly — no morphological expansion
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    result = []
    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = int(centroids[label][0]), int(centroids[label][1])
            result.append((cx, cy))
    return result


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
