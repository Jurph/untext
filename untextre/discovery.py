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


def assign_zone(
    cx: int, cy: int, img_w: int, img_h: int
) -> Tuple[int, int]:
    """Assign a blob centroid to a grid zone.

    The long image edge is divided into ZONE_LONG_DIVISIONS (3) segments;
    the short edge into ZONE_SHORT_DIVISIONS (2) segments.  This yields
    6 zones that capture typical corner-marking strategies.

    Returns:
        (col, row) zero-indexed zone coordinates.
        For landscape (w >= h): col is along width (0-2), row along height (0-1).
        For portrait (h > w):   col is along width (0-1), row along height (0-2).
    """
    if img_w >= img_h:  # landscape or square
        col_divs, row_divs = ZONE_LONG_DIVISIONS, ZONE_SHORT_DIVISIONS
    else:               # portrait
        col_divs, row_divs = ZONE_SHORT_DIVISIONS, ZONE_LONG_DIVISIONS

    col = min(int(cx / img_w * col_divs), col_divs - 1)
    row = min(int(cy / img_h * row_divs), row_divs - 1)
    return (col, row)


def discover_zones(
    bucket_paths: List[Path],
    img_w: int,
    img_h: int,
) -> Dict[Tuple[int, int], List[np.ndarray]]:
    """Run the random-pair convergence loop to find occupied zones.

    Draws random pairs with replacement, computes variance, extracts blobs,
    assigns them to zones.  Stops when the occupied zone set is stable for
    STABLE_STREAK_REQUIRED consecutive draws, or after MAX_DRAWS total.

    Zero-blob draws count toward the stability streak (they do not change the
    set, so they count as stable).

    Convergence is tracked globally across the full set of occupied zones
    (not per-zone).

    At MAX_DRAWS without convergence, accepts current occupied set and logs
    a warning.

    Args:
        bucket_paths: Paths to all images in this bucket (≥ 3 required).
        img_w: Image width (all images in bucket are this size).
        img_h: Image height.

    Returns:
        Dict mapping zone (col, row) → list of variance maps that contributed
        a blob to that zone.  Empty dict if no blobs found at all.
    """
    image_area = img_w * img_h
    # Cache loaded images to avoid repeated disk reads
    images: Dict[Path, np.ndarray] = {}
    for p in bucket_paths:
        try:
            images[p] = load_image(p)
        except Exception as e:
            logger.warning(f"Could not load {p.name} for discovery: {e}")

    paths = list(images.keys())
    if len(paths) < 2:
        logger.warning("Not enough loadable images for pair stacking")
        return {}

    zone_variance_maps: Dict[Tuple[int, int], List[np.ndarray]] = {}
    occupied: set = set()
    stable_streak = 0
    draw_count = 0

    while draw_count < MAX_DRAWS:
        # Draw pair with replacement
        a_path, b_path = random.choices(paths, k=2)
        img_a, img_b = images[a_path], images[b_path]

        var_map = compute_pair_variance(img_a, img_b)
        centroids = extract_blobs(var_map, image_area)

        prev_occupied = frozenset(occupied)
        for (cx, cy) in centroids:
            zone = assign_zone(cx, cy, img_w, img_h)
            occupied.add(zone)
            zone_variance_maps.setdefault(zone, []).append(var_map)

        draw_count += 1
        if frozenset(occupied) == prev_occupied:
            stable_streak += 1
        else:
            stable_streak = 0

        if stable_streak >= STABLE_STREAK_REQUIRED:
            logger.debug(f"Zone convergence after {draw_count} draws")
            break
    else:
        if occupied:
            logger.warning(
                f"Zone set did not converge after {MAX_DRAWS} draws; "
                f"accepting current set: {occupied}"
            )

    return zone_variance_maps


def crop_zone_to_bgra(
    mean_image: np.ndarray,
    blob_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Crop a blob region from the mean image and produce a tight BGRA array.

    The alpha channel is 255 inside the blob and 0 in the transparent border.
    Channel order is BGRA to match cv2's native format and what
    find_known_mask_in_image expects.

    Note on disk I/O: cv2.imwrite handles BGRA→PNG correctly.
    Use cv2.imread(..., cv2.IMREAD_UNCHANGED) to reload; do NOT use
    load_image() from utils (it drops the alpha channel).

    Args:
        mean_image: Pixel-wise mean of all images in the bucket (H×W×3 BGR).
        blob_mask: Binary mask (H×W uint8, 255 = blob).

    Returns:
        BGRA crop (H'×W'×4 uint8) with transparent border, or None if
        blob_mask is empty.
    """
    ys, xs = np.where(blob_mask == 255)
    if len(ys) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Crop with border, clamped to image bounds
    h, w = mean_image.shape[:2]
    cy0 = max(0, y0 - CROP_BORDER_PX)
    cy1 = min(h, y1 + CROP_BORDER_PX)
    cx0 = max(0, x0 - CROP_BORDER_PX)
    cx1 = min(w, x1 + CROP_BORDER_PX)

    bgr_crop = mean_image[cy0:cy1, cx0:cx1].copy()
    mask_crop = blob_mask[cy0:cy1, cx0:cx1].copy()

    # Produce BGRA — keep BGR channel order from source image so that
    # find_known_mask_in_image (which expects BGRA) gets the correct order.
    b_ch, g_ch, r_ch = cv2.split(bgr_crop)
    bgra = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
    return bgra


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
