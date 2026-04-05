"""Watermark auto-discovery for -U mode.

Discovers a watermark template from a directory of consistently-watermarked
images and returns BGRA crop(s) suitable for the -K / ORB pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import load_image, setup_logger
from .watermark_consensus import (
    CandidateMetadata,
    build_candidate_record,
    build_final_templates,
    split_candidate_bgra,
)
from .orb_prep import count_candidate_orb_keypoints

logger = setup_logger(__name__)


# Border added around each BGRA crop (pixels).
CROP_BORDER_PX = 8

# Zone grid: long edge into thirds, short edge into halves → 6 zones.
ZONE_LONG_DIVISIONS = 3
ZONE_SHORT_DIVISIONS = 2

# Cross-bucket IoU threshold for family membership.
CROSS_BUCKET_IOU_THRESHOLD = 0.50

# Structural lower bounds for candidates entering graph consensus.
MIN_CONSENSUS_CANDIDATE_AREA_PX = 64
MIN_CONSENSUS_CANDIDATE_SHORT_SIDE_PX = 10
MIN_CONSENSUS_CANDIDATE_LONG_SIDE_PX = 14
MIN_CONSENSUS_CANDIDATE_EDGE_PX = 64
MIN_CONSENSUS_CANDIDATE_FILL_RATIO = 0.01
MIN_CONSENSUS_CANDIDATE_ORB_KEYPOINTS = 6


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
    wm_color: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Crop a blob region and produce a tight BGRA array.

    The alpha channel is 255 inside the blob and 0 in the transparent border.
    Channel order is BGRA to match cv2's native format and what
    find_known_mask_in_image expects.

    When wm_color is provided (from extract_watermark_colors), it is used for
    the BGR channels instead of mean_image.  The transparency-corrected colors
    have background bleed removed; mean_image is the fallback when correction
    is unavailable.

    Note on disk I/O: cv2.imwrite handles BGRA→PNG correctly.
    Use cv2.imread(..., cv2.IMREAD_UNCHANGED) to reload; do NOT use
    load_image() from utils (it drops the alpha channel).

    Args:
        mean_image: Pixel-wise mean of all images in the bucket (H×W×3 BGR).
        blob_mask: Binary mask (H×W uint8, 255 = blob).
        wm_color: Optional H×W×3 uint8 transparency-corrected watermark colors.
                  When supplied, used for BGR channels instead of mean_image.

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

    color_source = wm_color if wm_color is not None else mean_image
    bgr_crop = color_source[cy0:cy1, cx0:cx1].copy()
    mask_crop = blob_mask[cy0:cy1, cx0:cx1].copy()

    # Produce BGRA — keep BGR channel order from source image so that
    # find_known_mask_in_image (which expects BGRA) gets the correct order.
    b_ch, g_ch, r_ch = cv2.split(bgr_crop)
    bgra = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
    return bgra


def _candidate_meets_consensus_minimums(
    bgra: np.ndarray,
) -> Tuple[bool, int, int, int, int, float, int]:
    """Return whether a BGRA candidate is large enough for graph consensus."""
    alpha = bgra[:, :, 3] > 0
    area = int(alpha.sum())
    if area == 0:
        return False, 0, 0, 0, 0, 0.0, 0

    ys, xs = np.where(alpha)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    short_side = min(bbox_h, bbox_w)
    long_side = max(bbox_h, bbox_w)
    bbox_area = max(bbox_h * bbox_w, 1)
    fill_ratio = float(area) / float(bbox_area)
    edge_px = int(cv2.Canny(alpha.astype(np.uint8) * 255, 50, 150).astype(bool).sum())

    structural_valid = (
        area >= MIN_CONSENSUS_CANDIDATE_AREA_PX
        and short_side >= MIN_CONSENSUS_CANDIDATE_SHORT_SIDE_PX
        and long_side >= MIN_CONSENSUS_CANDIDATE_LONG_SIDE_PX
        and edge_px >= MIN_CONSENSUS_CANDIDATE_EDGE_PX
        and fill_ratio >= MIN_CONSENSUS_CANDIDATE_FILL_RATIO
    )
    if not structural_valid:
        return False, area, bbox_w, bbox_h, edge_px, fill_ratio, 0

    orb_keypoints = count_candidate_orb_keypoints(bgra, nfeatures=1000)
    is_valid = orb_keypoints >= MIN_CONSENSUS_CANDIDATE_ORB_KEYPOINTS
    return is_valid, area, bbox_w, bbox_h, edge_px, fill_ratio, orb_keypoints


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
    family_reps: List[np.ndarray] = []   # parallel list: family_reps[i] is the rep for families[i]

    sorted_candidates = sorted(candidates, key=lambda c: c.shape[0] * c.shape[1], reverse=True)
    for crop in sorted_candidates:
        assigned = False
        for i, rep in enumerate(family_reps):
            if compute_alpha_iou(crop, rep) >= CROSS_BUCKET_IOU_THRESHOLD:
                families[i].append(crop)
                assigned = True
                break
        if not assigned:
            families.append([crop])
            family_reps.append(crop)   # first (and so far largest) crop in this new family

    representatives = list(family_reps)
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
    q1, q3 = np.percentile(log_prec, [25, 75]).tolist()
    fence = q3 + 3.0 * (q3 - q1)      # Tukey extreme-outlier upper fence
    return float(10.0 ** (-fence))     # convert back to variance threshold


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize arr to [0, 1] float32. Returns zeros if range is degenerate."""
    a = arr.astype(np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


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
            var_gray  H×W float32 sample variance (Bessel-corrected, denominator n−1)
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
    var_gray = (gray_M2 / (n - 1)).astype(np.float32) if n > 1 else np.zeros_like(gray_M2, dtype=np.float32)

    return {
        "n_loaded": n,
        "mean_bgr": mean_bgr,
        "var_gray": var_gray,
    }


def compute_median_gradient(paths: List[Path]) -> Optional[np.ndarray]:
    """Compute the per-pixel median gradient magnitude across an image stack.

    For each image, compute the Sobel gradient magnitude on grayscale.  Return
    the per-pixel median across all successfully-loaded images.

    This implements the core signal from Dekel et al. (CVPR 2017, §3):
    scene background gradients are statistically independent across varying
    images and converge toward zero under the median; the watermark gradient
    (identical position in every image) persists robustly.  The result is a
    map that is high at watermark edges and near-zero at most background regions.

    For consistently-lit or static backgrounds, the background gradients do NOT
    cancel under the median — they persist just like the watermark.  This signal
    is therefore complementary to the variance-based score rather than a complete
    replacement: it adds discriminating power over varied backgrounds while the
    variance-based score handles the static-background failure mode.

    Memory note: all N gradient-magnitude frames (H×W float32 each) are held
    in memory simultaneously for the median computation.  For large batches of
    high-resolution images this may be significant.

    Args:
        paths: Paths to images that must all be the same pixel dimensions.

    Returns:
        H×W float32 median gradient magnitude, or None if fewer than 3 images
        could be loaded.
    """
    frames = []
    for p in paths:
        try:
            img = load_image(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        except Exception as e:
            logger.warning(f"compute_median_gradient: could not load {p.name}: {e}")
            continue

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        frames.append(np.sqrt(gx ** 2 + gy ** 2))

    if len(frames) < 3:
        return None

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


def build_watermark_score(
    stats: Dict[str, np.ndarray],
    var_norm: np.ndarray,
    median_grad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a composite watermark-likelihood score from stack statistics.

    score = low_var × var_boundary × median_grad_norm  (if median_grad provided)
    score = low_var × var_boundary                     (fallback without it)

    low_var          = normalize( −log10(var_gray + ε) )       stable pixels → 1
    var_boundary     = normalize( |∇ var_norm| )               step edges → 1
    median_grad_norm = normalize( median_k(|∇J_k|) )           persistent structure → 1

    The var_boundary term fires at the outer edge of the watermark region (where
    variance transitions sharply from ≈0 inside to background-level outside).

    The median_grad_norm term fires at pixels that have consistent image-space
    gradients across the stack — i.e., watermark strokes and edges whose structure
    persists regardless of the background underneath them.  Background gradients
    from varying scenes cancel toward zero under the median; watermark gradients
    do not.  This adds discriminating power over varied backgrounds, and is
    complementary to var_boundary (which handles the watermark boundary itself).

    The product requires all active factors to be high simultaneously, which makes
    the seed score sparse.  This is intentional: score_to_mask grows from high-
    confidence seed pixels into the full stable (low-variance) region, so the
    score only needs to fire somewhere within the watermark, not everywhere.

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

    score = low_var.astype(np.float32) * var_boundary

    if median_grad is not None:
        score = score * _normalize_01(median_grad)

    return score.astype(np.float32)


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


def _deblot_mask_by_area(mask: np.ndarray) -> np.ndarray:
    """Remove satellite noise components using Otsu on the log(component area) distribution.

    The component area distribution of a watermark candidate is typically bimodal:
    signal components (text glyphs, graphic elements) cluster at 100–2000 px;
    noise blotches cluster at 1–50 px.  Otsu on log(area) finds the valley between
    these two clusters as the split threshold — no pre-specified size constant needed.

    If the distribution is not clearly bimodal (la_max − la_min < 0.5 in log space,
    meaning all components are nearly the same size), the mask is returned unchanged,
    because Otsu would split a uniform distribution arbitrarily.

    Otsu requires at least 2 samples per class; if there are fewer than 4 components
    total, the mask is returned unchanged.

    The safety fallback — return the original if zero components survive — prevents
    deblotching from returning an empty mask.

    Args:
        mask: H×W uint8 binary mask (255 = candidate pixel).

    Returns:
        Cleaned mask with noise-sized components removed, or the original if the
        distribution is unimodal or the result would be empty.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 2:   # background + at most one foreground component
        return mask

    areas = np.array(
        [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)], dtype=np.float64
    )
    if len(areas) < 4:
        # Need at least 2 samples per class for Otsu to have any meaning.
        return mask

    log_areas = np.log(areas + 1.0)
    la_min, la_max = float(log_areas.min()), float(log_areas.max())
    if la_max - la_min < 0.5:
        # All components nearly the same size — no meaningful gap to cut.
        return mask

    # Normalise to [0, 255] and find the Otsu threshold on the 1-D distribution.
    # Otsu maximises between-class variance, placing the threshold at the valley
    # of a bimodal distribution — exactly the gap between glyphs and blotches.
    la_u8 = ((log_areas - la_min) / (la_max - la_min) * 255).clip(0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(la_u8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_log = la_min + float(thresh_val) / 255.0 * (la_max - la_min)
    min_area = float(np.exp(threshold_log))

    clean = np.zeros_like(mask)
    kept = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
            kept += 1

    # Fallback: if the threshold eliminated everything, return the original.
    # A single surviving component is a valid result (one clean glyph cluster).
    if kept < 1:
        return mask

    return clean


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
    median_grad: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Group stable components by spatial zone; return zone unions ranked by gradient-weighted area.

    A watermark like "PLAYBOYPLUS.COM + logo" consists of many medium-sized
    connected components (each letter, punctuation, logo element) all clustered
    in one image zone.  Individual letter areas overlap with accidentally-stable
    noise spots that appear scattered elsewhere.

    Ranking by raw stable area fails when a large consistently-lit background
    region (e.g., a white wall always in frame) accumulates more stable pixels
    than the watermark zone.  Weighting by gradient-consistency solves this:

        zone_score = total_stable_area × mean( median_grad_norm[stable_pixels_in_zone] )

    Flat uniform regions (a white wall, clear sky) have near-zero median gradient
    and score ≈ 0 regardless of how many stable pixels they contain.  A text or
    logo watermark has persistent structure at stroke edges and scores high.

    Falls back to raw stable-area ranking when median_grad is None, preserving
    the previous behaviour for callers that do not provide gradient evidence.

    Returns:
        List of H×W uint8 union masks, one per selected zone, highest-score zone first.
    """
    num_labels = mask_data["num_labels"]
    labels = mask_data["labels"]
    cc_stats = mask_data["stats"]
    cc_centroids = mask_data["centroids"]

    # Normalise median gradient to [0, 1] once so all zone means are comparable.
    grad_norm: Optional[np.ndarray] = None
    if median_grad is not None:
        grad_norm = _normalize_01(median_grad)

    zone_area: Dict[Tuple[int, int], int] = {}
    zone_grad_sum: Dict[Tuple[int, int], float] = {}
    zone_labels: Dict[Tuple[int, int], List[int]] = {}

    for label in range(1, num_labels):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        left = int(cc_stats[label, cv2.CC_STAT_LEFT])
        top = int(cc_stats[label, cv2.CC_STAT_TOP])
        width = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        height = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        cx = int(round(cc_centroids[label][0]))
        cy = int(round(cc_centroids[label][1]))
        zone = assign_zone(cx, cy, img_w, img_h)

        zone_area[zone] = zone_area.get(zone, 0) + area
        zone_labels.setdefault(zone, []).append(label)

        if grad_norm is not None:
            label_window = labels[top:top + height, left:left + width]
            grad_window = grad_norm[top:top + height, left:left + width]
            zone_grad_sum[zone] = zone_grad_sum.get(zone, 0.0) + float(
                grad_window[label_window == label].sum()
            )

    if not zone_area:
        return []

    # Compute ranking key per zone.
    if grad_norm is not None:
        # Gradient-consistency weight: mean normalised gradient over stable pixels.
        # zone_score = total_stable_area × mean_grad_in_zone
        # Flat regions: many stable pixels × ≈0 gradient → low score.
        # Text watermark: moderate stable pixels × high gradient → high score.
        def zone_score(zone: Tuple[int, int]) -> float:
            area = zone_area[zone]
            mean_grad = zone_grad_sum.get(zone, 0.0) / area
            return area * mean_grad
    else:
        def zone_score(zone: Tuple[int, int]) -> float:
            return float(zone_area[zone])

    ranked_zones = sorted(zone_area.keys(), key=zone_score, reverse=True)

    if grad_norm is not None:
        zone_strs = ", ".join(
            f"z{z}:{zone_area[z]}px×{zone_grad_sum.get(z,0)/zone_area[z]:.2f}g"
            for z in ranked_zones[:6]
        )
        logger.debug(f"Zone gradient-weighted scores: [{zone_strs}]")
    else:
        zone_strs = ", ".join(f"z{z}:{zone_area[z]}px" for z in ranked_zones[:6])
        logger.debug(f"Zone total stable area: [{zone_strs}]")

    candidates: List[np.ndarray] = []
    for zone in ranked_zones[:max_zones]:
        union = np.zeros(labels.shape, dtype=np.uint8)
        for label in zone_labels[zone]:
            left = int(cc_stats[label, cv2.CC_STAT_LEFT])
            top = int(cc_stats[label, cv2.CC_STAT_TOP])
            width = int(cc_stats[label, cv2.CC_STAT_WIDTH])
            height = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
            label_window = labels[top:top + height, left:left + width]
            union_window = union[top:top + height, left:left + width]
            union_window[label_window == label] = 255
        candidates.append(union)
        logger.debug(
            f"Zone {zone}: {len(zone_labels[zone])} component(s), "
            f"{zone_area[zone]} stable px, score={zone_score(zone):.1f}"
        )

    return candidates


def extract_watermark_colors(
    mean_bgr: np.ndarray,
    var_gray: np.ndarray,
    precision_mask: np.ndarray,
    inpaint_radius: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract true watermark colors by inverting the transparency mixing model.

    A semi-transparent watermark composites as:
        observed_i = α·WM_color + (1−α)·background_i

    The pixel-wise mean image at stable (watermark) pixels is contaminated:
        E[observed] = α·WM_color + (1−α)·E[background]

    This function removes the background term by estimating it from spatial
    neighbors (inpainting) and recovers WM_color:
        WM_color = (E[observed] − (1−α)·bg_estimate) / α

    Alpha is estimated per-pixel from the variance ratio (no guessed constant):
        Var[observed] = (1−α)² · Var[background]
        → α = 1 − sqrt(Var_observed / Var_background)

    Background mean and variance at watermark pixels are estimated by inpainting
    through precision_mask from surrounding non-watermark pixels.

    Args:
        mean_bgr: H×W×3 uint8 per-pixel mean image.
        var_gray: H×W float32 per-pixel grayscale population variance.
        precision_mask: H×W uint8, 255 at stable (watermark candidate) pixels.
        inpaint_radius: Pixel radius for TELEA inpainting neighborhood.

    Returns:
        wm_color: H×W×3 uint8 estimated true watermark color at every pixel.
        alpha_hat: H×W float32 per-pixel opacity estimate, clamped to [0.05, 1.0].
    """
    # ── Background mean estimation ─────────────────────────────────────────
    # Inpaint mean_bgr through the watermark mask so that watermark pixels are
    # replaced by spatial interpolation of the surrounding non-watermark mean.
    bg_mean_inpainted = cv2.inpaint(
        mean_bgr, precision_mask, inpaint_radius, cv2.INPAINT_TELEA
    )

    # ── Background variance estimation ────────────────────────────────────
    # Inpaint in log-variance space (uint8 proxy preserves relative scale).
    log_var = np.log10(var_gray.astype(np.float64) + 1e-8).astype(np.float32)
    log_min = float(log_var.min())
    log_max = float(log_var.max())

    if log_max - log_min > 1e-6:
        log_var_u8 = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        log_var_u8 = np.zeros_like(log_var, dtype=np.uint8)

    log_var_u8_inpainted = cv2.inpaint(
        log_var_u8, precision_mask, inpaint_radius, cv2.INPAINT_TELEA
    )

    # Invert normalisation to recover float log-variance, then exponentiate.
    log_var_inpainted = (
        log_var_u8_inpainted.astype(np.float32) / 255.0 * (log_max - log_min) + log_min
    )
    bg_var_inpainted = (10.0 ** log_var_inpainted.astype(np.float64)).astype(np.float32)

    # ── Per-pixel alpha estimation ─────────────────────────────────────────
    # Var[observed] = (1−α)² · Var[background]  →  α = 1 − sqrt(ratio)
    var_ratio = np.clip(var_gray / (bg_var_inpainted + 1e-8), 0.0, 1.0).astype(np.float32)
    alpha_hat = np.clip(1.0 - np.sqrt(var_ratio), 0.05, 1.0).astype(np.float32)

    # ── WM color extraction ────────────────────────────────────────────────
    # Rearranging the mixing model: WM = (observed − (1−α)·bg) / α
    alpha_3ch = alpha_hat[:, :, np.newaxis]   # broadcast over BGR
    wm_f = (mean_bgr.astype(np.float32) - (1.0 - alpha_3ch) * bg_mean_inpainted.astype(np.float32)) / alpha_3ch
    wm_color = np.clip(wm_f, 0, 255).astype(np.uint8)

    return wm_color, alpha_hat


def _top_n_alpha_components(alpha: np.ndarray, n: int = 1) -> np.ndarray:
    """Return alpha retaining only the n largest connected components by area.

    Used to focus phase-correlation geometry on the dominant watermark
    structure and suppress scattered noise pixels that would otherwise
    dominate the geometry field.

    Args:
        alpha: H×W uint8 alpha channel.
        n: Number of largest components to keep.

    Returns:
        H×W uint8 alpha with only the top-n components; all other pixels
        are set to 0.  Returns alpha unchanged if no foreground exists.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8), connectivity=8
    )
    if stats.shape[0] <= 1:  # only background label
        return alpha
    fg_ids = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:n] + 1
    out = np.zeros_like(alpha)
    for cid in fg_ids:
        out[labels == cid] = alpha[labels == cid]
    return out


def _consensus_vote(
    zone_data: List[Tuple[np.ndarray, int, int, Tuple[int, int], np.ndarray, np.ndarray]],
    debug_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """Convert discovery candidates into graph-consensus templates."""
    records = []
    record_index = 0
    for index, (zone_mask, img_w, img_h, zone, mean_img, wm_color) in enumerate(zone_data):
        bgra = crop_zone_to_bgra(mean_img, zone_mask, wm_color=wm_color)
        if bgra is None:
            continue
        subcrops = split_candidate_bgra(bgra)
        if len(subcrops) > 1:
            logger.info(
                f"Consensus: split candidate from {img_w}x{img_h} zone {zone} into {len(subcrops)} sub-candidates"
            )
        for split_index, subcrop in enumerate(subcrops):
            is_valid, area, bbox_w, bbox_h, edge_px, fill_ratio, orb_keypoints = _candidate_meets_consensus_minimums(subcrop)
            if not is_valid:
                logger.info(
                    f"Consensus: skipping low-signal candidate from {img_w}x{img_h} zone {zone} "
                    f"(split {split_index + 1}/{len(subcrops)}, {area} px, bbox {bbox_w}x{bbox_h}, "
                    f"edges {edge_px}, fill {fill_ratio:.4f}, orb_kp {orb_keypoints})"
                )
                continue
            records.append(
                build_candidate_record(
                    subcrop,
                    CandidateMetadata(
                        family_key=(img_w, img_h),
                        zone=zone,
                        candidate_index=record_index,
                        source_kind="discovery",
                    ),
                )
            )
            record_index += 1

    if not records:
        return []

    kept_buckets = {record.metadata.family_key for record in records}
    logger.info(
        "Consensus prep complete: %d zone candidate(s) -> %d kept sub-candidates across %d bucket(s)",
        len(zone_data),
        len(records),
        len(kept_buckets),
    )

    templates = build_final_templates(
        records,
        debug_dir=None if debug_dir is None else str(debug_dir),
    )
    logger.info(
        f"Consensus: {len(records)} candidate(s) -> {len(templates)} graph-derived template(s)"
    )
    return [template.bgra for template in templates]


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
    q1_p, q3_p = np.percentile(pooled, [25, 75]).tolist()
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
    all_zone_data: List[Tuple[np.ndarray, int, int, Tuple[int, int], np.ndarray, np.ndarray]] = []
    # Each entry: (zone_mask, img_w, img_h, zone, mean_bgr, wm_color)

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

        # Cross-sub-sample validation: intersect stable masks from two independent
        # halves of the image population.  A pixel that is a Tukey-fence outlier in
        # BOTH halves is a genuine watermark pixel; incidentally-stable noise that
        # barely passes the fence in the full population often fails one half.
        # Requires ≥ 6 images per bucket so each half has ≥ 3 images (Welford minimum).
        if len(paths) >= 6:
            rng = np.random.RandomState(seed=img_w ^ (img_h * 1031))
            shuffled = list(paths)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            stats_a = compute_stack_statistics(shuffled[:half])
            stats_b = compute_stack_statistics(shuffled[half:])
            if stats_a is not None and stats_b is not None:
                mask_a = (stats_a["var_gray"] <= global_stable_threshold).astype(np.uint8) * 255
                mask_b = (stats_b["var_gray"] <= global_stable_threshold).astype(np.uint8) * 255
                precision_mask = cv2.bitwise_and(precision_mask, cv2.bitwise_and(mask_a, mask_b))
                xval_pct = float(np.mean(precision_mask > 0) * 100)
                logger.debug(
                    f"Bucket {img_w}×{img_h}: after cross-sub-sample validation: "
                    f"{xval_pct:.2f}% of pixels stable in both halves"
                )

        # Transparency-model inversion: estimate true WM color and per-pixel
        # alpha by removing background bleed from the mean image.
        # observed = α·WM + (1−α)·background  →  background estimated by
        # inpainting, α from variance ratio; no constants needed.
        wm_color, alpha_hat = extract_watermark_colors(mean_img, var_gray, precision_mask)

        # Median gradient: persistent image-space gradients across the stack.
        # Dekel et al. (CVPR 2017): background gradients cancel under the median;
        # watermark gradients (fixed position) survive.
        median_grad = compute_median_gradient(paths)
        if median_grad is None:
            logger.warning(
                f"Bucket {img_w}×{img_h}: median gradient computation failed, "
                f"falling back to two-factor score"
            )

        # Composite score: stability × variance-boundary × persistent-gradient
        score = build_watermark_score(stats, var_norm, median_grad=median_grad)

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
            cv2.imwrite(str(debug_dir / f"debug_wm_color_{stem}.png"), wm_color)
            alpha_vis = (alpha_hat * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"debug_alpha_hat_{stem}.png"), alpha_vis)
            logger.info(f"Debug images saved for bucket {stem}")

        filtered_candidates = select_candidate_components(
            mask_data, score, img_w, img_h, median_grad=median_grad
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

        # Deblot: remove satellite noise components using Otsu on the log(area)
        # distribution.  The bimodal gap between glyph-sized components and tiny
        # blotches is found from the data; no size threshold is hard-coded.
        deblotted = [_deblot_mask_by_area(m) for m in filtered_candidates]
        filtered_candidates = [m for m in deblotted if np.any(m)]
        if not filtered_candidates:
            logger.warning(f"Bucket {img_w}×{img_h}: all candidates empty after deblotching")
            continue

        if debug_dir is not None:
            stem = f"{img_w}x{img_h}"
            for idx, m in enumerate(filtered_candidates):
                suffix = "" if idx == 0 else f"_{idx}"
                cv2.imwrite(str(debug_dir / f"debug_deblotted{suffix}_{stem}.png"), m)

        for zone_mask in filtered_candidates:
            ys, xs = np.where(zone_mask == 255)
            if len(ys) == 0:
                continue
            cx = int(np.round(xs.mean()))
            cy = int(np.round(ys.mean()))
            zone = assign_zone(cx, cy, img_w, img_h)
            all_zone_data.append((zone_mask, img_w, img_h, zone, mean_img, wm_color))
            logger.info(
                f"Bucket {img_w}×{img_h} zone {zone}: "
                f"{int(np.sum(zone_mask == 255)):,} stable px → queued for consensus vote"
            )

    if not all_zone_data:
        logger.warning("No watermark candidates discovered across all buckets")
        return []

    if qualifying_count <= 1:
        logger.info("Only one qualifying bucket — cross-validation unavailable")

    logger.info(
        f"Consensus vote across {len(all_zone_data)} zone candidate(s) "
        f"from {qualifying_count} bucket(s)"
    )
    result = _consensus_vote(all_zone_data, debug_dir=debug_dir)
    logger.info(f"Discovery complete: {len(result)} watermark crop(s) after consensus vote")
    return result
