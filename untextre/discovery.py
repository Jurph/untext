"""Watermark auto-discovery for -U mode.

Discovers a watermark template from a directory of consistently-watermarked
images and returns BGRA crop(s) suitable for the -K / ORB pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import load_image, setup_logger, IMAGE_EXTENSIONS

logger = setup_logger(__name__)

# Population variance threshold: pixels at or below this value are considered
# identical across the full image stack.  Near-zero (rather than exactly zero)
# handles tiny floating-point rounding differences in float32 arithmetic.
POPULATION_VARIANCE_THRESHOLD = 1e-6

# Minimum blob area as a fraction of total image area.
MIN_BLOB_AREA_FRACTION = 0.0005  # 0.05%

# Border added around each BGRA crop (pixels).
CROP_BORDER_PX = 8

# Zone grid: long edge into thirds, short edge into halves → 6 zones.
ZONE_LONG_DIVISIONS = 3
ZONE_SHORT_DIVISIONS = 2

# Cross-bucket IoU threshold for family membership.
CROSS_BUCKET_IOU_THRESHOLD = 0.50


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

    Buckets images by exact dimensions, computes population variance across all
    images in each qualifying bucket (≥ 3), finds near-zero-variance blobs
    (pixels identical across the full stack), assigns them to zones, and returns
    one representative BGRA crop per watermark family in descending pixel-area order.

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

        # Load all images once: grayscale for variance, BGR for mean crop
        gray_stack: List[np.ndarray] = []
        bgr_stack: List[np.ndarray] = []
        for p in paths:
            try:
                img = load_image(p)
                bgr_stack.append(img.astype(np.float32))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                gray_stack.append(gray)
            except Exception as e:
                logger.warning(f"Could not load {p.name}: {e}")

        if len(gray_stack) < 3:
            logger.warning(f"Bucket {img_w}×{img_h}: fewer than 3 loadable images, skipping")
            continue

        # Population variance across all N images.  A watermark pixel — stamped
        # identically onto every image — has near-zero variance.  Varying image
        # content produces nonzero variance even when visually similar.
        pop_variance = np.var(np.stack(gray_stack, axis=0), axis=0).astype(np.float32)
        mean_img = np.mean(np.stack(bgr_stack, axis=0), axis=0).astype(np.uint8)

        # Threshold: keep only pixels that are effectively identical across all images
        image_area = img_w * img_h
        min_area = max(1, int(image_area * MIN_BLOB_AREA_FRACTION))
        binary = (pop_variance <= POPULATION_VARIANCE_THRESHOLD).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Assign qualifying blobs to zones; accumulate per-zone pixel masks
        zone_masks: Dict[Tuple[int, int], np.ndarray] = {}
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            cx = int(centroids[label][0])
            cy = int(centroids[label][1])
            zone = assign_zone(cx, cy, img_w, img_h)
            if zone not in zone_masks:
                zone_masks[zone] = np.zeros((img_h, img_w), dtype=np.uint8)
            zone_masks[zone][labels == label] = 255

        if not zone_masks:
            logger.warning(f"Bucket {img_w}×{img_h}: no consistent blobs found above minimum size")
            continue

        for zone, zone_mask in zone_masks.items():
            bgra = crop_zone_to_bgra(mean_img, zone_mask)
            if bgra is not None:
                all_candidates.append(bgra)
                logger.info(
                    f"Bucket {img_w}×{img_h} zone {zone}: "
                    f"candidate {bgra.shape[1]}×{bgra.shape[0]} px"
                )

    if not all_candidates:
        logger.warning("No watermark candidates discovered across all buckets")
        return []

    # Aspect-ratio dedup: if multiple candidates have similar aspect ratios
    # (symmetric relative difference < 10%), keep only the largest.
    deduped: List[np.ndarray] = []
    for crop in sorted(all_candidates, key=lambda c: c.shape[0] * c.shape[1], reverse=True):
        r1 = _aspect_ratio(crop)
        if not any(
            2 * abs(r1 - _aspect_ratio(kept)) / (r1 + _aspect_ratio(kept)) < 0.10
            for kept in deduped
        ):
            deduped.append(crop)
    all_candidates = deduped

    qualifying_count = sum(1 for p in buckets.values() if len(p) >= 3)
    if qualifying_count <= 1:
        logger.info("Only one qualifying bucket — cross-validation unavailable")

    families = select_best_family(all_candidates)
    logger.info(f"Discovery complete: {len(families)} watermark family/families found")
    return families
