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


def compute_alpha_iou(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """Compute IoU on alpha channels after resizing the smaller crop to the larger.

    "Larger" is defined by pixel area (H * W).

    Args:
        crop_a: BGRA array (H×W×4).
        crop_b: BGRA array (H×W×4).

    Returns:
        IoU float in [0, 1].
    """
    area_a = crop_a.shape[0] * crop_a.shape[1]
    area_b = crop_b.shape[0] * crop_b.shape[1]

    if area_a >= area_b:
        large, small = crop_a, crop_b
    else:
        large, small = crop_b, crop_a

    target_h, target_w = large.shape[:2]
    small_resized = cv2.resize(
        small, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )

    alpha_large = (large[:, :, 3] > 127).astype(np.uint8)
    alpha_small = (small_resized[:, :, 3] > 127).astype(np.uint8)

    intersection = np.sum(alpha_large & alpha_small)
    union = np.sum(alpha_large | alpha_small)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def select_best_family(candidates: List[np.ndarray]) -> List[np.ndarray]:
    """Group candidates into families by IoU similarity; return one rep per family.

    Within each family, the largest crop (by pixel area) is the representative.
    Families are ordered by descending pixel area of their representative
    (largest first, for Phase 4 processing order).

    Args:
        candidates: List of BGRA crops.

    Returns:
        List of representative BGRA crops, one per family, largest first.
    """
    if not candidates:
        return []

    families: List[List[np.ndarray]] = []

    sorted_candidates = sorted(candidates, key=lambda c: c.shape[0] * c.shape[1], reverse=True)
    for crop in sorted_candidates:
        assigned = False
        for family in families:
            rep = max(family, key=lambda c: c.shape[0] * c.shape[1])
            if compute_alpha_iou(crop, rep) >= CROSS_BUCKET_IOU_THRESHOLD:
                family.append(crop)
                assigned = True
                break
        if not assigned:
            families.append([crop])

    representatives = [
        max(family, key=lambda c: c.shape[0] * c.shape[1])
        for family in families
    ]
    representatives.sort(key=lambda c: c.shape[0] * c.shape[1], reverse=True)
    return representatives


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


def _make_zone_mask(
    shape: Tuple[int, int],
    zone: Tuple[int, int],
) -> np.ndarray:
    """Return a binary mask (255) for the pixels belonging to a given zone."""
    h, w = shape
    col, row = zone

    if w >= h:
        col_divs, row_divs = ZONE_LONG_DIVISIONS, ZONE_SHORT_DIVISIONS
    else:
        col_divs, row_divs = ZONE_SHORT_DIVISIONS, ZONE_LONG_DIVISIONS

    x0 = int(w * col / col_divs)
    x1 = int(w * (col + 1) / col_divs)
    y0 = int(h * row / row_divs)
    y1 = int(h * (row + 1) / row_divs)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _aspect_ratio(crop: np.ndarray) -> float:
    """Return width/height ratio of a crop array (H×W×4)."""
    h, w = crop.shape[:2]
    return w / h if h > 0 else 1.0


def discover_watermark_candidates(
    image_paths: List[Path],
) -> List[np.ndarray]:
    """Discover watermark template(s) from a batch of consistently-watermarked images.

    Buckets images by exact dimensions, runs the pair-stacking convergence loop
    per qualifying bucket (≥ 3 images), crops BGRA candidates, cross-validates
    across buckets, and returns one representative BGRA crop per watermark family
    in descending pixel-area order.

    Args:
        image_paths: All image paths in the input directory (pre-frozen list).

    Returns:
        List of BGRA crops (H×W×4 uint8), largest family first.
        Empty list if no candidates discovered.
    """
    buckets = bucket_images_by_size(image_paths)
    if not buckets:
        logger.warning("No loadable images found for discovery")
        return []

    all_candidates: List[np.ndarray] = []

    for (img_w, img_h), paths in buckets.items():
        if len(paths) < 3:
            logger.info(
                f"Bucket {img_w}×{img_h}: only {len(paths)} image(s) — "
                f"skipping self-discovery, will use cross-bucket template if available"
            )
            continue

        logger.info(f"Discovering watermark in bucket {img_w}×{img_h} ({len(paths)} images)")
        zone_maps = discover_zones(paths, img_w, img_h)

        if not zone_maps:
            logger.warning(f"Bucket {img_w}×{img_h}: no low-variance blobs found")
            continue

        # Compute mean image once per bucket (used for all zone crops)
        loaded = []
        for p in paths:
            try:
                loaded.append(load_image(p).astype(np.float32))
            except Exception as e:
                logger.warning(f"Could not reload {p.name} for mean image: {e}")
        if not loaded:
            continue
        mean_img = np.mean(loaded, axis=0).astype(np.uint8)

        for zone, var_maps in zone_maps.items():
            # Build union blob mask from all variance maps for this zone
            union_low_var = np.zeros(mean_img.shape[:2], dtype=np.uint8)
            for vmap in var_maps:
                binary = (vmap < VARIANCE_THRESHOLD).astype(np.uint8) * 255
                union_low_var = cv2.bitwise_or(union_low_var, binary)

            # Restrict to the blob's grid zone for clean cropping.
            # No morphological operations — the union of low-variance pixels
            # across draws IS the signal. Do not inflate or connect blobs.
            zone_mask = _make_zone_mask(union_low_var.shape, zone)
            zoned_mask = cv2.bitwise_and(union_low_var, zone_mask)

            bgra = crop_zone_to_bgra(mean_img, zoned_mask)
            if bgra is not None:
                all_candidates.append(bgra)
                logger.info(
                    f"Bucket {img_w}×{img_h} zone {zone}: "
                    f"candidate {bgra.shape[1]}×{bgra.shape[0]} px"
                )

    if not all_candidates:
        logger.warning("No watermark candidates discovered across all buckets")
        return []

    # Per-bucket aspect-ratio dedup (spec Phase 2 Step 5):
    # If multiple candidates have similar aspect ratios (symmetric relative
    # difference < 10%), keep only the largest.
    deduped: List[np.ndarray] = []
    for crop in sorted(all_candidates, key=lambda c: c.shape[0] * c.shape[1], reverse=True):
        r1 = _aspect_ratio(crop)
        if not any(
            2 * abs(r1 - _aspect_ratio(kept)) / (r1 + _aspect_ratio(kept)) < 0.10
            for kept in deduped
        ):
            deduped.append(crop)
    all_candidates = deduped

    qualifying_count = sum(1 for paths in buckets.values() if len(paths) >= 3)
    if qualifying_count <= 1:
        logger.info("Only one qualifying bucket — cross-validation unavailable")

    families = select_best_family(all_candidates)
    logger.info(f"Discovery complete: {len(families)} watermark family/families found")
    return families
