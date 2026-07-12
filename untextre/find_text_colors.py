"""Spatial TF-IDF color analysis for text watermark detection.

This module provides spatial TF-IDF analysis to identify text-like colors
in detected text regions. The main function creates binary masks by:
1. Clustering colors in the text region and surrounding background
2. Computing a Figure of Merit (FOM) for each cluster based on:
   - TF-IDF score (color distinctiveness)
   - Border ratio (text is underrepresented on bbox edges)
   - Largest connected component fraction (text is fragmented, not solid)
3. Selecting clusters with high FOM and rejecting solid blobs
4. Applying morphological cleanup for final mask

This approach automatically identifies text colors without needing to specify
target colors, handling complex scenarios like multi-colored text and varying
backgrounds.
"""

import cv2
import numpy as np
from typing import Optional

from .utils import ImageArray, BBox, Color, image_hw, setup_logger

logger = setup_logger(__name__)


def hex_to_bgr(hex_color: str) -> Color:
    """Convert hex color string to BGR tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        BGR color tuple
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Parse hex values
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to BGR
    return (rgb[2], rgb[1], rgb[0])


def html_to_bgr(html_color: str) -> Color:
    """Convert HTML color name to BGR tuple.
    
    Args:
        html_color: HTML color name (e.g., "red", "ghostwhite")
        
    Returns:
        BGR color tuple
    """
    # Convert HTML name to hex using PIL
    from PIL import ImageColor
    rgb = ImageColor.getrgb(html_color)
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # Convert RGB to BGR


def compute_cluster_fom(
    tf_idf_normalized: float,
    border_ratio: float,
    largest_cc_fraction: float
) -> float:
    """Compute Figure of Merit for a cluster's text-likelihood.
    
    Weights empirically determined from experiments on 18,000+ synthetic
    watermark samples across K=4,5,6,7,9.
    
    Args:
        tf_idf_normalized: TF-IDF score normalized to [0, 255]
        border_ratio: Ratio of cluster's border presence to its bbox presence
                     (< 1 means underrepresented on border = text-like)
        largest_cc_fraction: Fraction of cluster pixels in largest connected component
                            (< 1 means fragmented = text-like)
    
    Returns:
        FOM in [0, 1] where higher = more likely text
    """
    tf_score = min(tf_idf_normalized / 255.0, 1.0)
    border_score = max(0.0, 1.0 - border_ratio)
    cc_score = max(0.0, 1.0 - largest_cc_fraction)
    
    # Weights: border_ratio is the strongest predictor (63%)
    return 0.07 * tf_score + 0.63 * border_score + 0.30 * cc_score


def grabcut_refine(
    image_roi: ImageArray,
    fom_mask: np.ndarray,
    iterations: int = 5,
    dilation_px: int = 21,
    debug: bool = False,
) -> np.ndarray:
    """Refine a binary FOM mask using GrabCut for spatially coherent edges.

    Seeds GrabCut with the FOM mask: high-confidence text pixels become
    ``GC_FGD``, high-confidence background becomes ``GC_BGD``, and a
    border of ambiguity around each is marked probable.

    After GrabCut produces a tight glyph-level mask, an aggressive
    dilation is applied so that the final mask gives LaMa plenty of
    background context around each text stroke.

    Args:
        image_roi: The BGR image region (same dimensions as *fom_mask*).
        fom_mask: Binary mask (uint8, 0/255) from FOM analysis.
        iterations: Number of GrabCut iterations (default 5).
        dilation_px: Diameter of the elliptical dilation kernel applied
                    after GrabCut (default 21).  Larger values give the
                    inpainter more surrounding context.
        debug: If True, log diagnostic info.

    Returns:
        Refined binary mask (uint8, 0/255), same shape as *fom_mask*.
    """
    h, w = fom_mask.shape[:2]

    # GrabCut needs at least a 3×3 region to be meaningful
    if h < 3 or w < 3:
        if debug:
            logger.debug("GrabCut: region too small, returning FOM mask unchanged")
        return fom_mask

    # Build the 4-value initialization mask that GrabCut expects:
    #   GC_BGD (0)     — definite background
    #   GC_FGD (1)     — definite foreground
    #   GC_PR_BGD (2)  — probable background
    #   GC_PR_FGD (3)  — probable foreground
    #
    # Strategy: erode the FOM foreground to get "definite foreground",
    # erode the FOM background to get "definite background", and mark
    # the remaining fringe as probable.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_eroded = cv2.erode(fom_mask, kern, iterations=1)
    bg_mask = cv2.bitwise_not(fom_mask)
    bg_eroded = cv2.erode(bg_mask, kern, iterations=1)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[bg_eroded == 255] = cv2.GC_BGD
    gc_mask[fom_mask == 255] = cv2.GC_PR_FGD
    gc_mask[fg_eroded == 255] = cv2.GC_FGD

    if debug:
        counts = {
            "GC_BGD": int(np.sum(gc_mask == cv2.GC_BGD)),
            "GC_FGD": int(np.sum(gc_mask == cv2.GC_FGD)),
            "GC_PR_BGD": int(np.sum(gc_mask == cv2.GC_PR_BGD)),
            "GC_PR_FGD": int(np.sum(gc_mask == cv2.GC_PR_FGD)),
        }
        logger.debug(f"GrabCut init mask: {counts}")

    # GrabCut requires the image to be uint8 BGR (already is) and
    # allocates two 13-component GMM arrays internally.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_roi,
            gc_mask,
            (0, 0, 0, 0),  # rect unused when mask is provided (GC_INIT_WITH_MASK)
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as e:
        if debug:
            logger.warning(f"GrabCut failed ({e}), returning FOM mask unchanged")
        return fom_mask

    # Extract foreground: definite + probable foreground
    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    if debug:
        fom_px = int(np.sum(fom_mask == 255))
        gc_px = int(np.sum(refined == 255))
        logger.debug(f"GrabCut refinement: {fom_px} -> {gc_px} foreground pixels "
                    f"({gc_px - fom_px:+d})")

    # Dilate the tight glyph mask so the inpainter gets generous
    # background context around every text stroke.
    if dilation_px > 0:
        dilate_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px, dilation_px)
        )
        refined = cv2.dilate(refined, dilate_kern)
        if debug:
            dilated_px = int(np.sum(refined == 255))
            logger.debug(f"GrabCut post-dilation ({dilation_px}px): "
                        f"{gc_px} -> {dilated_px} foreground pixels")

    return refined


def color_guided_expand(
    image: ImageArray,
    bbox: BBox,
    confirmed_mask: np.ndarray,
    centers: np.ndarray,
    top_id: int,
    bot_id: int,
    color_radius: float,
    bg_radius: float,
    expand_factor: float = 2.0,
    min_cc_px: int = 10,
    color_radius_multiplier: float = 1.0,
    bg_radius_multiplier: float = 1.0,
    iterations: int = 5,
    zero_bbox_cluster_ids: tuple = (),
    debug: bool = False,
) -> np.ndarray:
    """Expand a confirmed text mask using adaptive color matching and GrabCut.

    Uses the bbox-derived K-means centroids to find pixels genuinely similar
    to the watermark color — bounded to an expanded bbox, not the full image.

    1. Expands the detection bbox by expand_factor (clamped to image bounds)
    2. Within that ROI, measures each pixel's distance to the top-FOM centroid
    3. Pixels within color_radius → GC_FGD; within bg_radius of bottom
       centroid → GC_BGD; everything else → GC_PR_BGD
    4. Applies morphological closing (3×3) + CC area filter to the FGD mask
    5. Runs GrabCut on the expanded bbox ROI
    6. Returns confirmed_mask OR'd with the GrabCut result

    Args:
        image: Full image (H×W×3 BGR uint8).
        bbox: Original detection bounding box (x, y, w, h).
        confirmed_mask: Full-image binary mask (H×W uint8, 0/255) from
                        bbox FOM analysis. Returned unchanged on failure.
        centers: K-means cluster centroids (K×3 float32, RGB order).
        top_id: Index of the highest-FOM cluster → GC_FGD seed.
        bot_id: Index of the lowest-FOM cluster → GC_BGD seed.
        color_radius: Max distance (RGB) from top centroid for FGD candidates.
                      Set adaptively to mean + 1σ of top-cluster bbox pixels.
        bg_radius: Max distance (RGB) from bottom centroid for BGD pixels.
        expand_factor: How much to expand the bbox (default 2.0).
        min_cc_px: Minimum CC area to keep in FGD mask (default 10).
        iterations: GrabCut iterations (default 5).
        debug: If True, log diagnostic info.

    Returns:
        Updated full-image binary mask (H×W uint8, 0/255), superset of
        confirmed_mask.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    # --- Expand bbox, clamped to image bounds ---
    cx, cy = x + w // 2, y + h // 2
    ew, eh = int(w * expand_factor), int(h * expand_factor)
    x1 = max(0, cx - ew // 2)
    y1 = max(0, cy - eh // 2)
    x2 = min(img_w, x1 + ew)
    y2 = min(img_h, y1 + eh)

    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return confirmed_mask

    roi = image[y1:y2, x1:x2]
    roi_flat = roi.reshape(-1, 3).astype(np.float32)[:, [2, 1, 0]]  # BGR→RGB

    # --- Distance-threshold color matching (not nearest-centroid) ---
    color_radius = float(color_radius) * float(color_radius_multiplier)
    bg_radius = float(bg_radius) * float(bg_radius_multiplier)
    d_top = np.sqrt(((roi_flat - centers[top_id]) ** 2).sum(axis=1))
    d_bot = np.sqrt(((roi_flat - centers[bot_id]) ** 2).sum(axis=1))
    roi_h, roi_w = roi.shape[:2]
    d_top = d_top.reshape(roi_h, roi_w)
    d_bot = d_bot.reshape(roi_h, roi_w)

    # --- Build FGD mask; apply closing then CC filter ---
    fgd_mask = (d_top <= color_radius).astype(np.uint8) * 255
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgd_mask = cv2.morphologyEx(fgd_mask, cv2.MORPH_CLOSE, kern)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        fgd_mask, connectivity=8
    )
    for cc_id in range(1, num_labels):
        if stats[cc_id, cv2.CC_STAT_AREA] < min_cc_px:
            fgd_mask[cc_labels == cc_id] = 0

    if debug:
        n_fgd = int(np.sum(fgd_mask == 255))
        logger.debug(
            f"Color expand: ROI={roi_w}x{roi_h}, color_radius={color_radius:.1f}, "
            f"{n_fgd} FGD pixels after closing+CC filter"
        )

    if int(np.sum(fgd_mask == 255)) == 0:
        if debug:
            logger.debug("Color expand: no FGD pixels survived, returning confirmed_mask")
        return confirmed_mask

    # --- Build GrabCut init mask ---
    #   Within color_radius of top centroid  → GC_FGD
    #   Within bg_radius of bottom centroid  → GC_BGD
    #   Nearest centroid has zero bbox pixels → GC_BGD  (exclusive background)
    #   Everything else                      → GC_PR_BGD
    #   FGD always wins ties
    gc_init = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_init[d_bot <= bg_radius] = cv2.GC_BGD

    if zero_bbox_cluster_ids:
        # Assign each ROI pixel to its nearest centroid; pixels whose nearest
        # centroid had strictly zero presence inside the detection bbox are
        # definitionally background — lock them as GC_BGD.
        n_centroids = len(centers)
        all_dists = np.stack(
            [np.linalg.norm(roi_flat - centers[cid].astype(np.float32), axis=1)
             for cid in range(n_centroids)],
            axis=1,
        )  # (N_pixels, K)
        nearest = np.argmin(all_dists, axis=1).reshape(roi_h, roi_w)
        for cid in zero_bbox_cluster_ids:
            gc_init[nearest == cid] = cv2.GC_BGD

    gc_init[fgd_mask == 255] = cv2.GC_FGD  # FGD wins all ties

    if debug:
        n_fgd_gc = int(np.sum(gc_init == cv2.GC_FGD))
        n_bgd_gc = int(np.sum(gc_init == cv2.GC_BGD))
        n_pr_bgd  = int(np.sum(gc_init == cv2.GC_PR_BGD))
        n_excl    = int(np.sum(nearest[..., np.newaxis] == np.array(zero_bbox_cluster_ids)).item()) if zero_bbox_cluster_ids else 0
        logger.debug(
            f"Color expand: GC_FGD={n_fgd_gc}, GC_BGD={n_bgd_gc} "
            f"(excl-bg clusters={list(zero_bbox_cluster_ids)}, excl-bg px≈{n_excl}), "
            f"GC_PR_BGD={n_pr_bgd}"
        )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(roi, gc_init, (0, 0, 0, 0), bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        if debug:
            logger.warning(f"Color expand GrabCut failed ({e}), returning confirmed_mask")
        return confirmed_mask

    gc_result = np.where(
        (gc_init == cv2.GC_FGD) | (gc_init == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    if debug:
        orig_px = int(np.sum(confirmed_mask[y1:y2, x1:x2] == 255))
        new_px = int(np.sum(gc_result == 255)) - orig_px
        logger.debug(f"Color expand: confirmed={orig_px} -> +{max(new_px, 0)} new pixels")

    # OR with confirmed_mask: refining, not replacing
    result = confirmed_mask.copy()
    result[y1:y2, x1:x2] = cv2.bitwise_or(result[y1:y2, x1:x2], gc_result)
    return result


def _expanded_bbox_roi(
    image_shape: tuple[int, int],
    bbox: BBox,
    expand_factor: float,
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    ew, eh = int(w * expand_factor), int(h * expand_factor)
    x1 = max(0, cx - ew // 2)
    y1 = max(0, cy - eh // 2)
    x2 = min(img_w, x1 + ew)
    y2 = min(img_h, y1 + eh)
    return x1, y1, x2, y2


def _bbox_geometry_cost(
    bbox: BBox,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    long_weight: float,
    short_weight: float,
    short_axis_power: float,
) -> np.ndarray:
    x, y, w, h = bbox
    x2 = x + w - 1
    y2 = y + h - 1

    left = np.maximum(x - xx, 0)
    right = np.maximum(xx - x2, 0)
    above = np.maximum(y - yy, 0)
    below = np.maximum(yy - y2, 0)

    outside_x = np.maximum(left, right).astype(np.float32)
    outside_y = np.maximum(above, below).astype(np.float32)
    norm_w = max(float(w), 1.0)
    norm_h = max(float(h), 1.0)

    if w >= h:
        long_distance = outside_x / norm_w
        short_distance = outside_y / norm_h
    else:
        long_distance = outside_y / norm_h
        short_distance = outside_x / norm_w

    return (
        1.0
        + long_weight * long_distance
        + short_weight * np.power(short_distance, short_axis_power)
    )


def _remove_small_components(mask: np.ndarray, min_cc_px: int) -> np.ndarray:
    if min_cc_px <= 1 or int(np.sum(mask > 0)) == 0:
        return mask
    cleaned = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), connectivity=8
    )
    for cc_id in range(1, num_labels):
        if stats[cc_id, cv2.CC_STAT_AREA] < min_cc_px:
            cleaned[labels == cc_id] = 0
    return cleaned


def geometry_budgeted_expand(
    image: ImageArray,
    bbox: BBox,
    confirmed_mask: np.ndarray,
    centers: np.ndarray,
    top_id: int,
    bot_id: int,
    color_radius: float,
    bg_radius: float,
    expand_factor: float = 2.0,
    inside_rejection_credit: float = 1.0,
    max_budget_fraction: float = 0.50,
    long_weight: float = 1.0,
    short_weight: float = 4.0,
    short_axis_power: float = 1.5,
    connectivity_dilation_px: int = 3,
    min_cc_px: int = 10,
    color_radius_multiplier: float = 1.0,
    bg_radius_multiplier: float = 1.0,
    iterations: int = 5,
    debug: bool = False,
) -> np.ndarray:
    """Expand a confirmed mask by spending a geometry budget on outside pixels.

    Pixels are proposed by color matching plus GrabCut in an expanded bbox ROI,
    but pixels outside the detector bbox are admitted cheapest-first by distance
    from the bbox. Expansion along the bbox long axis is cheaper than
    perpendicular swelling.
    """
    x, y, w, h = bbox
    bbox_area = max(w * h, 0)
    x1, y1, x2, y2 = _expanded_bbox_roi(image_hw(image), bbox, expand_factor)
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < 3 or roi_h < 3:
        return confirmed_mask

    roi = image[y1:y2, x1:x2]
    roi_flat = roi.reshape(-1, 3).astype(np.float32)[:, [2, 1, 0]]
    color_radius = float(color_radius) * float(color_radius_multiplier)
    bg_radius = float(bg_radius) * float(bg_radius_multiplier)
    d_top = np.sqrt(((roi_flat - centers[top_id]) ** 2).sum(axis=1)).reshape(roi_h, roi_w)
    d_bot = np.sqrt(((roi_flat - centers[bot_id]) ** 2).sum(axis=1)).reshape(roi_h, roi_w)

    candidate_foreground = d_top <= color_radius
    definite_background = d_bot <= bg_radius

    gc_init = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_init[definite_background] = cv2.GC_BGD
    gc_init[candidate_foreground] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            roi,
            gc_init,
            (0, 0, 0, 0),  # rect unused when mask is provided (GC_INIT_WITH_MASK)
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as e:
        if debug:
            logger.warning(f"Budgeted expand GrabCut failed ({e}), returning confirmed_mask")
        return confirmed_mask

    grabcut_foreground = (gc_init == cv2.GC_FGD) | (gc_init == cv2.GC_PR_FGD)

    bbox_roi_mask = np.zeros((roi_h, roi_w), dtype=bool)
    bx1 = max(x, x1) - x1
    by1 = max(y, y1) - y1
    bx2 = min(x + w, x2) - x1
    by2 = min(y + h, y2) - y1
    if bx2 > bx1 and by2 > by1:
        bbox_roi_mask[by1:by2, bx1:bx2] = True

    inside_color_candidates = candidate_foreground & bbox_roi_mask
    inside_rejected = inside_color_candidates & ~grabcut_foreground
    budget = float(np.sum(inside_rejected)) * inside_rejection_credit
    budget = min(budget, max_budget_fraction * bbox_area)
    if budget <= 0:
        return confirmed_mask.copy()

    outside_candidates = grabcut_foreground & candidate_foreground & ~bbox_roi_mask
    if not np.any(outside_candidates):
        return confirmed_mask.copy()

    confirmed_roi = confirmed_mask[y1:y2, x1:x2] > 0
    dilation_size = max(1, int(connectivity_dilation_px))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    reachable = cv2.dilate(confirmed_roi.astype(np.uint8) * 255, kern, iterations=1) > 0
    candidate_u8 = outside_candidates.astype(np.uint8)
    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    connected_candidates = np.zeros_like(outside_candidates)
    for cc_id in range(1, num_labels):
        cc_mask = labels == cc_id
        if np.any(cc_mask & reachable):
            connected_candidates |= cc_mask

    if not np.any(connected_candidates):
        return confirmed_mask.copy()

    local_yy, local_xx = np.nonzero(connected_candidates)
    global_yy = local_yy + y1
    global_xx = local_xx + x1
    costs = _bbox_geometry_cost(
        bbox,
        global_yy,
        global_xx,
        long_weight=long_weight,
        short_weight=short_weight,
        short_axis_power=short_axis_power,
    )
    order = np.argsort(costs, kind="stable")
    sorted_costs = costs[order]
    cumulative = np.cumsum(sorted_costs)
    keep_count = int(np.searchsorted(cumulative, budget, side="right"))
    if keep_count <= 0:
        return confirmed_mask.copy()

    selected = np.zeros((roi_h, roi_w), dtype=np.uint8)
    keep = order[:keep_count]
    selected[local_yy[keep], local_xx[keep]] = 255
    selected = _remove_small_components(selected, min_cc_px)

    result = confirmed_mask.copy()
    result[y1:y2, x1:x2] = cv2.bitwise_or(result[y1:y2, x1:x2], selected)
    return result


def find_mask_by_spatial_tf_idf(
    image: ImageArray,
    bbox: BBox,
    num_clusters: int = 4,
    debug: bool = False,
    target_color: Optional[Color] = None,
    # EMPIRICAL: FOM weights and this 0.30 threshold were calibrated together
    # against the 18K watermark/color-cluster experiment set.
    fom_threshold: float = 0.30,
    cc_guard: float = 0.85,
    cleanup_close_px: int = 11,
    cleanup_dilate_px: int = 13,
    use_grabcut: bool = False,
    return_cluster_data: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Create a binary mask using Figure of Merit analysis.
    
    This approach identifies text colors by computing a Figure of Merit (FOM)
    for each color cluster, combining multiple signals:
    - TF-IDF: colors distinctive to text region vs background
    - Border ratio: text is underrepresented at bbox edges
    - Connected components: text is fragmented, not one solid blob
    
    Process:
    1. Extract the text bbox region (the "document")
    2. Extract a surrounding region with same pixel count (the "corpus")
    3. K-means cluster all colors together
    4. For each cluster, compute:
       - TF-IDF score (color distinctiveness)
       - Border ratio (edge underrepresentation)
       - Largest CC fraction (fragmentation)
    5. Compute FOM from weighted combination of metrics
    6. Select clusters with FOM >= threshold AND largest_cc <= guard
    7. Apply morphological operations for cleanup
    
    Args:
        image: Input image in BGR format  
        bbox: Bounding box (x, y, width, height) to analyze
        num_clusters: Number of color clusters for K-means (default: 4)
        debug: If True, prints diagnostic information
        target_color: Optional BGR color tuple - if provided, forces selection 
                     of the color cluster containing that exact color
        fom_threshold: Minimum Figure of Merit to accept cluster (default: 0.30)
        cc_guard: Maximum largest_cc_fraction to accept (default: 0.85)
                 Rejects solid blobs that would over-mask for inpainting
        use_grabcut: If True, refine the FOM mask with cv2.grabCut before
                    morphological cleanup (default: False)
        return_cluster_data: If True, return a tuple (cleaned_mask, cluster_data)
                    where cluster_data is a dict with keys 'centers', 'top_id',
                    'bot_id', 'color_radius', 'bg_radius' — the K-means centroids,
                    the highest- and lowest-FOM cluster indices, and the adaptive
                    color radii for each. Used to seed color_guided_expand.
                    (default: False)

    Returns:
        cleaned_mask (uint8, 255=text) normally; or (cleaned_mask, cluster_data)
        when return_cluster_data=True.
    """
    from .mask_generator import morph_clean_mask
    
    x, y, w, h = bbox
    image_h, image_w = image.shape[:2]
    
    # Calculate expanded region (1.414x to double the area)
    scale_factor = 1.414  # sqrt(2)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Center the expanded region on the original bbox center
    bbox_center_x = x + w // 2
    bbox_center_y = y + h // 2
    
    # Calculate ideal expanded bbox position
    ideal_new_x = bbox_center_x - new_w // 2
    ideal_new_y = bbox_center_y - new_h // 2
    
    # Adjust for image boundaries
    new_x = max(0, min(ideal_new_x, image_w - new_w))
    new_y = max(0, min(ideal_new_y, image_h - new_h))
    
    # Ensure we don't exceed image bounds
    actual_new_w = min(new_w, image_w - new_x)
    actual_new_h = min(new_h, image_h - new_y)
    
    if debug:
        logger.debug(f"Spatial TF-IDF: bbox=({x}, {y}, {w}, {h}), expanded=({new_x}, {new_y}, {actual_new_w}, {actual_new_h})")
    
    # Extract regions
    bbox_region = image[y:y+h, x:x+w]  # The "document"
    expanded_region = image[new_y:new_y+actual_new_h, new_x:new_x+actual_new_w]
    
    # Create mask for the original bbox within the expanded region
    bbox_mask = np.zeros((actual_new_h, actual_new_w), dtype=bool)
    
    # Calculate bbox position within expanded region
    bbox_in_expanded_x = x - new_x
    bbox_in_expanded_y = y - new_y
    
    # Ensure bbox fits within expanded region
    bbox_end_x = min(bbox_in_expanded_x + w, actual_new_w)
    bbox_end_y = min(bbox_in_expanded_y + h, actual_new_h)
    bbox_start_x = max(0, bbox_in_expanded_x)
    bbox_start_y = max(0, bbox_in_expanded_y)
    
    bbox_mask[bbox_start_y:bbox_end_y, bbox_start_x:bbox_end_x] = True
    
    # Extract surrounding region (corpus) by excluding bbox area
    # 2D boolean indexing into the 3-channel array selects rows directly,
    # producing shape (N, 3) without an intermediate 3-channel mask allocation.
    surrounding_region = expanded_region[~bbox_mask]
    
    # Combine both regions for k-means clustering
    bbox_pixels = bbox_region.reshape(-1, 3)
    surrounding_pixels = surrounding_region
    all_pixels = np.vstack([bbox_pixels, surrounding_pixels])
    
    # Convert to RGB for k-means
    all_pixels_rgb = all_pixels[:, [2, 1, 0]]  # BGR to RGB
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        all_pixels_rgb.astype(np.float32), 
        num_clusters, 
        np.zeros((all_pixels_rgb.shape[0], 1), dtype=np.int32),  # bestLabels ignored (KMEANS_PP_CENTERS)
        criteria, 
        10, 
        cv2.KMEANS_PP_CENTERS
    )
    
    # Split labels back into bbox and surrounding regions
    bbox_labels = labels[:len(bbox_pixels)]
    surrounding_labels = labels[len(bbox_pixels):]
    
    # Calculate TF (Term Frequency) for bbox region
    bbox_cluster_counts = np.bincount(bbox_labels.flatten(), minlength=num_clusters)
    bbox_total = len(bbox_pixels)
    tf_scores = bbox_cluster_counts / bbox_total if bbox_total > 0 else np.zeros(num_clusters)
    
    # Calculate IDF (Inverse Document Frequency) for surrounding region
    surrounding_cluster_counts = np.bincount(surrounding_labels.flatten(), minlength=num_clusters)
    surrounding_total = len(surrounding_pixels)
    
    # IDF calculation: log(total_pixels / (cluster_count + 1))
    # Adding 1 to avoid division by zero
    idf_scores = np.log(surrounding_total / (surrounding_cluster_counts + 1))
    
    # Calculate TF-IDF scores
    tf_idf_scores = tf_scores * idf_scores
    
    if debug:
        logger.debug(f"TF-IDF scores range: {np.min(tf_idf_scores):.4f} to {np.max(tf_idf_scores):.4f}")
        positive_scores = np.sum(tf_idf_scores > 0)
        logger.debug(f"Clusters with positive TF-IDF: {positive_scores}/{num_clusters}")
    
    # Normalize TF-IDF scores to 0-255 range
    min_score = np.min(tf_idf_scores)
    max_score = np.max(tf_idf_scores)
    
    if max_score > min_score:
        normalized_scores = ((tf_idf_scores - min_score) / (max_score - min_score) * 255).astype(np.uint8)
    else:
        # If all scores are the same, set to middle gray
        normalized_scores = np.full(num_clusters, 128, dtype=np.uint8)
    
    if debug:
        logger.debug(f"Normalized scores range: {np.min(normalized_scores)} to {np.max(normalized_scores)}")
    
    # Use actual bbox_region dimensions, not original h,w (in case of boundary clipping)
    actual_h, actual_w = bbox_region.shape[:2]
    bbox_labels_2d = bbox_labels.reshape(actual_h, actual_w)
    total_bbox_pixels = actual_h * actual_w
    
    # Create border mask (2px ring around the edge of the bbox)
    # IMPORTANT: Exclude bbox edges that coincide with image edges
    # Background colors naturally extend to image edges, so those shouldn't count
    border_width = 2
    border_mask = np.zeros((actual_h, actual_w), dtype=bool)
    
    # Check which bbox edges touch the image boundary
    touches_top = (y == 0)
    touches_bottom = (y + h >= image_h)
    touches_left = (x == 0)
    touches_right = (x + w >= image_w)
    
    # Only include internal borders (not touching image edge)
    if not touches_top:
        border_mask[:border_width, :] = True
    if not touches_bottom:
        border_mask[-border_width:, :] = True
    if not touches_left:
        border_mask[:, :border_width] = True
    if not touches_right:
        border_mask[:, -border_width:] = True
    
    total_border_pixels = np.sum(border_mask)
    
    if debug and (touches_top or touches_bottom or touches_left or touches_right):
        excluded = []
        if touches_top:
            excluded.append("top")
        if touches_bottom:
            excluded.append("bottom")
        if touches_left:
            excluded.append("left")
        if touches_right:
            excluded.append("right")
        logger.debug(f"Border mask excludes image edges: {', '.join(excluded)}")
    
    # Compute per-cluster metrics and Figure of Merit
    cluster_foms = np.zeros(num_clusters)
    cluster_accepted = np.zeros(num_clusters, dtype=bool)
    
    for cluster_id in range(num_clusters):
        cluster_mask = (bbox_labels_2d == cluster_id)
        cluster_pixel_count = np.sum(cluster_mask)
        
        if cluster_pixel_count == 0:
            continue
        
        # Border ratio: (cluster's % of border) / (cluster's % of bbox)
        # < 1 means underrepresented on border = text-like
        cluster_border_pixels = np.sum(cluster_mask & border_mask)
        cluster_bbox_share = cluster_pixel_count / total_bbox_pixels
        cluster_border_share = cluster_border_pixels / total_border_pixels if total_border_pixels > 0 else 0.0
        border_ratio = cluster_border_share / cluster_bbox_share if cluster_bbox_share > 0 else 1.0
        
        # Largest connected component fraction
        # Text is fragmented (many small CCs), background is unified (one big CC)
        cluster_mask_uint8 = cluster_mask.astype(np.uint8)
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cluster_mask_uint8, connectivity=8)
        if num_labels > 1:
            # stats[0] is background, stats[1:] are components
            component_sizes = stats[1:, cv2.CC_STAT_AREA]
            largest_cc_size = np.max(component_sizes) if len(component_sizes) > 0 else 0
        else:
            largest_cc_size = 0
        largest_cc_fraction = largest_cc_size / cluster_pixel_count if cluster_pixel_count > 0 else 0.0
        
        # Compute Figure of Merit
        fom = compute_cluster_fom(normalized_scores[cluster_id], border_ratio, largest_cc_fraction)
        cluster_foms[cluster_id] = fom
        
        # Accept if FOM meets threshold AND not a solid blob
        is_text_like = fom >= fom_threshold
        is_not_solid_blob = largest_cc_fraction <= cc_guard
        cluster_accepted[cluster_id] = is_text_like and is_not_solid_blob
        
        if debug:
            logger.debug(f"Cluster {cluster_id}: TF-IDF={normalized_scores[cluster_id]}, "
                       f"border_ratio={border_ratio:.3f}, largest_cc={largest_cc_fraction:.3f}, "
                       f"FOM={fom:.3f}, accepted={cluster_accepted[cluster_id]}")
    
    # Create binary mask from accepted clusters
    binary_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
    for cluster_id in range(num_clusters):
        if cluster_accepted[cluster_id]:
            cluster_mask = (bbox_labels_2d == cluster_id)
            binary_mask[cluster_mask] = 255
    
    if debug:
        accepted_count = np.sum(cluster_accepted)
        logger.debug(f"Accepted {accepted_count}/{num_clusters} clusters (FOM>={fom_threshold}, CC<={cc_guard})")
    
    # Target color override: if target_color is specified, force inclusion of the color cluster containing it
    if target_color is not None:
        if debug:
            logger.debug(f"Target color override: forcing inclusion of color cluster containing {target_color}")
        
        # Find which cluster the target color belongs to
        target_color_rgb = np.array([target_color[2], target_color[1], target_color[0]])  # Convert BGR to RGB for comparison with centers
        
        # Calculate distances from target color to all cluster centers
        distances = np.linalg.norm(centers - target_color_rgb, axis=1)
        target_cluster_id = np.argmin(distances)
        
        if debug:
            closest_center = centers[target_cluster_id]
            distance = distances[target_cluster_id]
            logger.debug(f"Target color {target_color} (RGB: {target_color_rgb}) closest to cluster {target_cluster_id}")
            logger.debug(f"Cluster center: {closest_center}, distance: {distance:.2f}")
        
        # Create mask for all pixels in the target cluster
        target_cluster_mask = (bbox_labels_2d == target_cluster_id)
        target_pixel_count = np.sum(target_cluster_mask)
        
        if target_pixel_count > 0:
            if debug:
                logger.debug(f"Forcing inclusion of {target_pixel_count} pixels in target color cluster {target_cluster_id}")
                original_tf_idf = tf_idf_scores[target_cluster_id]
                logger.debug(f"Original TF-IDF score for target cluster: {original_tf_idf:.4f}")
            
            # Add target cluster pixels to the existing TF-IDF mask (combine both)
            target_mask = target_cluster_mask.astype(np.uint8) * 255
            binary_mask = cv2.bitwise_or(binary_mask, target_mask)
            
            if debug:
                combined_pixels = np.sum(binary_mask == 255)
                logger.debug(f"Combined mask (TF-IDF + target color): {combined_pixels} pixels")
        else:
            if debug:
                logger.debug("No pixels found in target color cluster, using TF-IDF only")
    
    if debug:
        if target_color is not None:
            logger.debug(f"Using FOM>={fom_threshold}, CC<={cc_guard} + target color override")
        else:
            logger.debug(f"Using FOM>={fom_threshold}, CC<={cc_guard}")
        mask_pixels_before_morph = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        logger.debug(f"Mask coverage before morphology: {mask_pixels_before_morph}/{total_pixels} pixels ({100*mask_pixels_before_morph/total_pixels:.1f}%)")

    # Optional GrabCut refinement: use FOM mask to seed GrabCut for
    # spatially coherent edges before morphological cleanup.
    if use_grabcut:
        binary_mask = grabcut_refine(bbox_region, binary_mask, debug=debug)

    # Apply morphological operations to clean up the mask
    cleaned_mask = morph_clean_mask(
        binary_mask,
        cleanup_close_px=cleanup_close_px,
        cleanup_dilate_px=cleanup_dilate_px,
    )

    if debug:
        mask_pixels_after_morph = np.sum(cleaned_mask == 255)
        logger.debug(f"Mask coverage after morphology: {mask_pixels_after_morph}/{total_pixels} pixels ({100*mask_pixels_after_morph/total_pixels:.1f}%)")
        pixel_change = mask_pixels_after_morph - mask_pixels_before_morph
        logger.debug(f"Morphological operations changed mask by {pixel_change:+d} pixels ({100*pixel_change/total_pixels:+.1f}%)")

    if return_cluster_data:
        accepted_ids = [cid for cid in range(num_clusters) if cluster_accepted[cid]]
        top_id = max(accepted_ids, key=lambda cid: cluster_foms[cid]) if accepted_ids else int(np.argmax(cluster_foms))
        bot_id = int(np.argmin(cluster_foms))

        # Adaptive color radius: mean + 1σ of distances from bbox top-cluster
        # pixels to their centroid. Captures the natural spread of that color.
        bbox_pixels_rgb = bbox_pixels[:, [2, 1, 0]].astype(np.float32)
        top_bbox_pixels = bbox_pixels_rgb[bbox_labels.flatten() == top_id]
        bot_bbox_pixels = bbox_pixels_rgb[bbox_labels.flatten() == bot_id]
        if len(top_bbox_pixels) > 0:
            d = np.linalg.norm(top_bbox_pixels - centers[top_id], axis=1)
            color_radius = float(np.mean(d) + np.std(d))
        else:
            logger.warning(
                "No bbox pixels found for selected foreground color cluster "
                f"(top_id={top_id}, bbox={bbox}); using empirical fallback color_radius=30.0"
            )
            # EMPIRICAL: degenerate fallback retained for defensive behavior.
            color_radius = 30.0
        if len(bot_bbox_pixels) > 0:
            d = np.linalg.norm(bot_bbox_pixels - centers[bot_id], axis=1)
            bg_radius = float(np.mean(d) + np.std(d))
        else:
            logger.warning(
                "No bbox pixels found for selected background color cluster "
                f"(bot_id={bot_id}, bbox={bbox}); using empirical fallback bg_radius=30.0"
            )
            # EMPIRICAL: degenerate fallback retained for defensive behavior.
            bg_radius = 30.0

        return cleaned_mask, {
            "centers": centers,
            "top_id": top_id,
            "bot_id": bot_id,
            "color_radius": color_radius,
            "bg_radius": bg_radius,
            "zero_bbox_cluster_ids": tuple(
                int(cid) for cid in range(num_clusters)
                if bbox_cluster_counts[cid] == 0
            ),
        }
    return cleaned_mask 
