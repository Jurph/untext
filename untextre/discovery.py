"""Watermark auto-discovery for -U mode.

Discovers a watermark template from a directory of consistently-watermarked
images and returns BGRA crop(s) suitable for the -K / ORB pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import load_image, setup_logger

logger = setup_logger(__name__)


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



def _aspect_ratio(crop: np.ndarray) -> float:
    """Return width/height ratio of a crop array (H×W×4)."""
    h, w = crop.shape[:2]
    return w / h if h > 0 else 1.0


def _precision_outlier_threshold(var_gray: np.ndarray) -> float:
    """Find the variance threshold separating outlier-precision (watermark) pixels.

    Precision = 1/variance.  For watermark pixels, precision is astronomical;
    for normal image content, log-precision = -log(var) follows approximately
    a Gaussian distribution (since variance of i.i.d. samples is chi-squared
    and log-chi-squared is approximately normal).  The watermark is therefore
    a handful of extreme outliers in the upper tail of the log-precision
    distribution.

    The Tukey upper fence — Q3 + 3 × IQR — excludes > 99.994% of a Gaussian
    distribution; anything above it is a genuine statistical outlier, not a
    threshold we guessed.  k=3 is a standard statistical constant for
    "extreme outliers" (Tukey, 1977) with a precise probabilistic meaning.

    Args:
        var_gray: H×W float32 per-pixel population variance.

    Returns:
        Variance threshold.  Pixels with var_gray ≤ this value are statistical
        outliers on the precision axis — watermark candidates.
    """
    flat = var_gray.flatten().astype(np.float64)
    if flat.size == 0:
        return 0.0

    log_prec = -np.log10(flat + 1e-8)
    q1, q3 = float(np.percentile(log_prec, 25)), float(np.percentile(log_prec, 75))
    fence = q3 + 3.0 * (q3 - q1)      # Tukey extreme-outlier upper fence
    return float(10.0 ** (-fence))     # convert back to variance threshold


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize arr to [0, 1] float32. Returns zeros if range is degenerate."""
    a = arr.astype(np.float64)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def compute_stack_statistics(paths: List[Path]) -> Optional[Dict[str, np.ndarray]]:
    """Compute per-pixel statistics across a set of same-size images.

    Uses Welford's online algorithm so memory scales with image size, not
    batch size.  Computes grayscale variance, BGR mean, and the gradient
    magnitude of the mean gray image — the inputs needed for the composite
    watermark score.

    Args:
        paths: Paths to images that must all be the same pixel dimensions.

    Returns:
        Dict with keys:
            mean_bgr  H×W×3 uint8
            var_gray  H×W float32 population variance
        or None if fewer than 3 images loaded.
    """
    n = 0
    gray_mean: Optional[np.ndarray] = None
    gray_M2: Optional[np.ndarray] = None
    bgr_mean: Optional[np.ndarray] = None

    for p in paths:
        try:
            img = load_image(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            bgr = img.astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not load {p.name}: {e}")
            continue

        n += 1
        if gray_mean is None:
            gray_mean = gray.copy()
            gray_M2 = np.zeros_like(gray)
            bgr_mean = bgr.copy()
        else:
            delta = gray - gray_mean
            gray_mean += delta / n
            gray_M2 += delta * (gray - gray_mean)
            bgr_mean += (bgr - bgr_mean) / n

    if n < 3:
        return None

    mean_bgr = bgr_mean.astype(np.uint8)
    var_gray = (gray_M2 / n).astype(np.float32)

    return {
        "n_loaded": n,
        "mean_bgr": mean_bgr,
        "var_gray": var_gray,
    }


def build_watermark_score(
    stats: Dict[str, np.ndarray],
    var_norm: np.ndarray,
) -> np.ndarray:
    """Build a composite watermark-likelihood score from stack statistics.

    score = low_var × var_boundary

    low_var      = normalize( −log10(var_gray + ε) )        stable pixels → 1
    var_boundary = normalize( gradient_magnitude(var_norm) )  step edges → 1

    The key insight: the variance field has a sharp step edge at the watermark
    boundary — near-zero variance inside the overlay, elevated variance outside.
    That step is large regardless of the watermark's opacity.

    The previous approach (gradient of the mean image within a stable mask) fails
    for semi-transparent watermarks because:
      - Boundary pixels blend with the varying background → elevated variance →
        excluded from the stable mask → structure = 0 there.
      - Interior pixels are stable but have zero gradient → score = 0 there too.

    Using the gradient of the variance map solves both:
      - The step in var_norm at the watermark boundary is the LARGEST such step
        in the image (from ≈0 inside to content-level outside), so it dominates
        even a globally-normalized structure signal.
      - The score is highest exactly where the watermark boundary lies, not where
        some incidentally-stable scene edge happens to be.

    Returns:
        H×W float32 in [0, 1].
    """
    var_gray = stats["var_gray"].astype(np.float64)

    low_var = _normalize_01(-np.log10(var_gray + 1e-8))

    # Gradient of the variance map: large where variance transitions sharply.
    # The watermark boundary creates the steepest step (var ≈ 0 inside →
    # content-level variance outside), so it dominates after normalization.
    vn = var_norm.astype(np.float32)
    gx = cv2.Sobel(vn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(vn, cv2.CV_32F, 0, 1, ksize=3)
    var_boundary = _normalize_01(np.sqrt(gx ** 2 + gy ** 2))

    score = (low_var.astype(np.float32) * var_boundary).astype(np.float32)
    return score


def _trim_to_budget(
    component_mask: np.ndarray,
    seed_mask: np.ndarray,
    max_area: int,
) -> np.ndarray:
    """Keep only the max_area pixels closest to the score seed.

    When a grown low-variance component exceeds the size budget (10% of image),
    prioritise the pixels nearest to the composite-score high-confidence region.
    Pixels that were pulled in far from any confirmed watermark structure are
    dropped first.

    Args:
        component_mask: Full grown component (H×W uint8, 255 = included).
        seed_mask:      High-confidence score pixels that anchored the growth.
        max_area:       Maximum number of pixels to keep.

    Returns:
        Trimmed H×W uint8 mask with at most max_area pixels set to 255.
    """
    area = int(np.sum(component_mask == 255))
    if area <= max_area:
        return component_mask

    # Distance transform: for each pixel, how far is the nearest seed pixel?
    # cv2.distanceTransform measures distance FROM 0-pixels TO nearest 255-pixel,
    # so we invert the seed mask (seed pixels become 0 in the input).
    inv_seed = cv2.bitwise_not(seed_mask)
    dist = cv2.distanceTransform(inv_seed, cv2.DIST_L2, maskSize=5)

    comp_ys, comp_xs = np.where(component_mask == 255)
    distances = dist[comp_ys, comp_xs]

    # Keep the max_area pixels with the smallest distance to seed
    order = np.argsort(distances, kind="stable")
    keep_ys = comp_ys[order[:max_area]]
    keep_xs = comp_xs[order[:max_area]]

    result = np.zeros_like(component_mask, dtype=np.uint8)
    result[keep_ys, keep_xs] = 255
    return result


def score_to_mask(
    score: np.ndarray,
    var_norm: np.ndarray,
    precision_outlier_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Produce the candidate mask from the composite score and stable-pixel region.

    The function has two responsibilities:

    1. Build mask_raw — Otsu threshold on the score image followed by a
       morphological close.  This is a high-confidence seed: it marks where
       the score is strongest.  It is returned so that select_candidate_components
       can use it as a size-budget seed if needed.

    2. Build mask_clean — the full stable-pixel region from which candidates
       are drawn.  When precision_outlier_mask is supplied (the Tukey-fence
       derived stable mask from the pooled distribution), that is used
       directly.  Otherwise Otsu on var_norm provides a fallback.

    Returns:
        Dict with mask_raw, mask_clean, score_closed, num_labels, labels,
        stats, and centroids from connectedComponentsWithStats on mask_clean.
    """
    score_u8 = (score * 255).clip(0, 255).astype(np.uint8)
    _, mask_raw = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    score_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)

    # Grow region: use the caller-supplied precision-outlier mask when available.
    # The Tukey fence identifies ~6-7% of pixels as statistically unusually stable
    # (the watermark and near-zero-variance content).  Growing into this region
    # keeps the candidate bounded by the actual statistical evidence, not an
    # arbitrary Otsu split on the log-normalized variance which gave 25%+ of pixels.
    if precision_outlier_mask is not None:
        low_var_mask = precision_outlier_mask
    else:
        _, low_var_mask = cv2.threshold(
            var_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    lv_pct = float(np.mean(low_var_mask > 0) * 100)
    logger.debug(f"Stable mask: {lv_pct:.2f}% of pixels")

    # ALL connected components of the precision-outlier stable mask become
    # candidates.  Do not filter by score overlap here.
    #
    # The score (gradient of var_norm) fires more strongly at isolated noise
    # spots — which have sharp variance boundaries — than at semi-transparent
    # watermarks, which have gradual boundaries.  A score-based seed filter
    # would therefore exclude the watermark while retaining noise.
    #
    # The watermark IS the largest coherent stable region; noise spots are
    # isolated and individually small.  Area-based selection in
    # select_candidate_components handles the distinction correctly.
    mask_clean = low_var_mask

    num_labels, labels, cc_stats, centroids = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )

    return {
        "mask_raw": mask_raw,
        "mask_clean": mask_clean,
        "score_closed": score_closed,
        "num_labels": num_labels,
        "labels": labels,
        "stats": cc_stats,
        "centroids": centroids,
    }


def select_candidate_components(
    mask_data: Dict,
    score: np.ndarray,
    img_w: int,
    img_h: int,
    max_zones: int = 3,
) -> List[np.ndarray]:
    """Group stable components by spatial zone; return zone unions ranked by total stable area.

    A watermark like "PLAYBOYPLUS.COM + logo" consists of many medium-sized
    connected components (each letter, punctuation, logo element) all clustered
    in one image zone.  Individual letter areas overlap with accidentally-stable
    noise spots that appear scattered elsewhere.

    Individual component area cannot distinguish a letter from a noise spot.
    Spatial density can: the watermark zone accumulates the most total stable
    area because it holds 12+ components vs. 0-2 per noise zone.

    No size threshold is needed — we ask which zone has the most stable pixels
    and return the union of all components in that zone as the template.

    Returns:
        List of H×W uint8 union masks, one per selected zone, highest-area zone first.
    """
    num_labels = mask_data["num_labels"]
    labels = mask_data["labels"]
    cc_stats = mask_data["stats"]
    cc_centroids = mask_data["centroids"]

    zone_area: Dict[Tuple[int, int], int] = {}
    zone_components: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for label in range(1, num_labels):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        cx = int(round(cc_centroids[label][0]))
        cy = int(round(cc_centroids[label][1]))
        zone = assign_zone(cx, cy, img_w, img_h)

        blob = np.zeros(labels.shape, dtype=np.uint8)
        blob[labels == label] = 255

        zone_area[zone] = zone_area.get(zone, 0) + area
        zone_components.setdefault(zone, []).append(blob)

    if not zone_area:
        return []

    ranked_zones = sorted(zone_area.items(), key=lambda x: x[1], reverse=True)
    zone_strs = ", ".join(f"z{z}:{a}px" for z, a in ranked_zones[:6])
    logger.debug(f"Zone total stable area: [{zone_strs}]")

    candidates: List[np.ndarray] = []
    for zone, total_area in ranked_zones[:max_zones]:
        union = np.zeros(labels.shape, dtype=np.uint8)
        for blob in zone_components[zone]:
            union = cv2.bitwise_or(union, blob)
        candidates.append(union)
        logger.debug(
            f"Zone {zone}: {len(zone_components[zone])} component(s), "
            f"{total_area} stable px total"
        )

    return candidates


def discover_watermark_candidates(
    image_paths: List[Path],
    debug_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """Discover watermark template(s) from a batch of consistently-watermarked images.

    Two-pass algorithm:

    Pass 1 — pool statistics across all qualifying buckets and compute a single
    Tukey-fence stable-pixel threshold from the combined log-precision
    distribution.  Pooling means small buckets and watermark-free buckets both
    contribute to the baseline, so the threshold is calibrated against the true
    normal-content distribution rather than a single bucket's noise.

    Pass 2 — for each qualifying bucket, apply the global threshold, build the
    composite score, grow candidates, and emit BGRA crops.

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

    # ── Pass 1: accumulate per-bucket stats; pool log-precision values ────────
    bucket_data: Dict[tuple, tuple] = {}   # (w, h) → (paths, stats, var_norm)
    all_log_prec: List[np.ndarray] = []

    for (img_w, img_h), paths in buckets.items():
        if len(paths) < 3:
            logger.info(
                f"Bucket {img_w}×{img_h}: only {len(paths)} image(s) — "
                f"skipping self-discovery, will use cross-bucket template if available"
            )
            continue

        logger.info(f"Computing statistics for bucket {img_w}×{img_h} ({len(paths)} images)")
        stats = compute_stack_statistics(paths)
        if stats is None:
            logger.warning(f"Bucket {img_w}×{img_h}: fewer than 3 loadable images, skipping")
            continue

        var_gray = stats["var_gray"]
        log_var = np.log10(var_gray.astype(np.float64) + 1e-8)
        var_norm = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        bucket_data[(img_w, img_h)] = (paths, stats, var_norm)
        all_log_prec.append(-log_var.flatten())

    if not bucket_data:
        logger.warning("No qualifying buckets for discovery")
        return []

    # Compute the global Tukey extreme-outlier fence from the POOLED distribution.
    # All non-watermark pixels across all buckets contribute to Q1/Q3/IQR, making
    # the estimate robust against any single bucket's idiosyncrasies.
    pooled = np.concatenate(all_log_prec)
    q1_p, q3_p = float(np.percentile(pooled, 25)), float(np.percentile(pooled, 75))
    global_stable_threshold = float(10.0 ** (-(q3_p + 3.0 * (q3_p - q1_p))))

    qualifying_count = len(bucket_data)
    if qualifying_count > 1:
        logger.info(
            f"Pooled Tukey fence across {qualifying_count} bucket(s): "
            f"stable threshold = {global_stable_threshold:.2e} "
            f"(from {pooled.size:,} pixels)"
        )
    else:
        logger.info("Single qualifying bucket — pooled threshold equals per-bucket threshold")

    # ── Pass 2: process each bucket using the shared global threshold ─────────
    all_candidates: List[np.ndarray] = []

    for (img_w, img_h), (paths, stats, var_norm) in bucket_data.items():
        logger.info(f"Discovering watermark in bucket {img_w}×{img_h} ({len(paths)} images)")

        mean_img = stats["mean_bgr"]
        var_gray = stats["var_gray"]

        if debug_dir is not None:
            stem = f"{img_w}x{img_h}"
            cv2.imwrite(str(debug_dir / f"debug_variance_{stem}.png"), var_norm)
            cv2.imwrite(str(debug_dir / f"debug_mean_{stem}.png"), mean_img)

        # Precision-outlier stable mask: pixels whose stability is a
        # statistical anomaly (Tukey upper fence on log-precision).
        # Used as the grow region in score_to_mask so we never expand
        # beyond the truly stable evidence.
        precision_mask = (var_gray <= global_stable_threshold).astype(np.uint8) * 255
        stable_pct = float(np.mean(precision_mask > 0) * 100)
        logger.debug(
            f"Bucket {img_w}×{img_h}: precision-outlier stable mask "
            f"= {stable_pct:.2f}% of pixels (threshold {global_stable_threshold:.2e})"
        )

        # Composite score: variance-field gradient × stability
        score = build_watermark_score(stats, var_norm)

        # Phase 4: Otsu on score → morph close → seed-grow into precision-outlier mask
        mask_data = score_to_mask(score, var_norm, precision_outlier_mask=precision_mask)

        if debug_dir is not None:
            stem = f"{img_w}x{img_h}"
            # Stable mask overlay debug image
            cv2.imwrite(str(debug_dir / f"debug_stable_mask_{stem}.png"), precision_mask)
            # Score with NORM_MINMAX so max always maps to 255 regardless of amplitude
            score_vis = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(str(debug_dir / f"debug_score_{stem}.png"), score_vis)
            cv2.imwrite(str(debug_dir / f"debug_mask_raw_{stem}.png"), mask_data["mask_raw"])
            cv2.imwrite(str(debug_dir / f"debug_mask_clean_{stem}.png"), mask_data["mask_clean"])
            logger.info(f"Debug images saved for bucket {stem}")

        filtered_candidates = select_candidate_components(
            mask_data, score, img_w, img_h
        )

        if not filtered_candidates:
            logger.warning(f"Bucket {img_w}×{img_h}: no candidate components found")
            continue

        logger.debug(
            f"Bucket {img_w}×{img_h}: {len(filtered_candidates)} candidate(s) selected"
        )

        # Safety backstop: with the Tukey-fence stable mask the zone union
        # should always be well under 10% of the image.  This trim fires only
        # if a pathological bucket still produces an oversized candidate.
        image_area = img_w * img_h
        max_wm_area = int(image_area * 0.10)
        score_seed = mask_data["score_closed"]
        trimmed_candidates = []
        for mask in filtered_candidates:
            area = int(np.sum(mask == 255))
            if area > max_wm_area:
                frac = area / image_area
                logger.debug(
                    f"Bucket {img_w}×{img_h}: trimming candidate "
                    f"({area:,} px, {frac:.1%}) → {max_wm_area:,} px budget"
                )
                mask = _trim_to_budget(mask, score_seed, max_wm_area)
            trimmed_candidates.append(mask)
        filtered_candidates = trimmed_candidates

        for zone_mask in filtered_candidates:
            ys, xs = np.where(zone_mask == 255)
            if len(ys) == 0:
                continue
            cx = int(np.round(xs.mean()))
            cy = int(np.round(ys.mean()))
            zone = assign_zone(cx, cy, img_w, img_h)
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

    if qualifying_count <= 1:
        logger.info("Only one qualifying bucket — cross-validation unavailable")

    families = select_best_family(all_candidates)
    logger.info(f"Discovery complete: {len(families)} watermark family/families found")
    return families
