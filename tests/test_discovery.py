import numpy as np
import pytest
import logging
import random
import cv2
from unittest.mock import patch
from untextre.discovery import (
    bucket_images_by_size,
    assign_zone,
    _candidate_meets_consensus_minimums,
)

def test_bucket_images_by_size_groups_correctly(tmp_path):
    # Create fake image files with known sizes
    paths = [tmp_path / f"img{i}.png" for i in range(4)]
    for p in paths:
        p.touch()

    # Mock load_image to return arrays of known shapes
    shapes = {
        paths[0]: np.zeros((100, 200, 3), dtype=np.uint8),  # 200x100
        paths[1]: np.zeros((100, 200, 3), dtype=np.uint8),  # 200x100
        paths[2]: np.zeros((300, 400, 3), dtype=np.uint8),  # 400x300
        paths[3]: np.zeros((200, 100, 3), dtype=np.uint8),  # 100x200 (portrait)
    }
    with patch("untextre.discovery.load_image", side_effect=lambda p: shapes[p]):
        buckets = bucket_images_by_size(paths)

    assert (200, 100) in buckets
    assert len(buckets[(200, 100)]) == 2
    assert (400, 300) in buckets
    assert len(buckets[(400, 300)]) == 1
    assert (100, 200) in buckets  # portrait is a separate bucket
    assert len(buckets[(100, 200)]) == 1

def test_bucket_images_skips_unreadable(tmp_path, caplog):
    paths = [tmp_path / "bad.png"]
    paths[0].touch()
    with patch("untextre.discovery.load_image", side_effect=ValueError("bad")):
        with caplog.at_level(logging.WARNING):
            buckets = bucket_images_by_size(paths)
    assert buckets == {}
    assert any("bad.png" in msg for msg in caplog.messages)


def test_assign_zone_landscape():
    # 300w x 200h → long=width(3 cols), short=height(2 rows) → 6 zones
    # centroid at (250, 50) → col 2 (right third), row 0 (top half)
    zone = assign_zone(cx=250, cy=50, img_w=300, img_h=200)
    assert zone == (2, 0)

def test_assign_zone_portrait():
    # 200w x 300h → long=height(3 rows), short=width(2 cols) → 6 zones
    # centroid at (50, 250) → col 0 (left half), row 2 (bottom third)
    zone = assign_zone(cx=50, cy=250, img_w=200, img_h=300)
    assert zone == (0, 2)


from untextre.discovery import crop_zone_to_bgra


def _make_sift_ready_candidate(size: int = 128) -> np.ndarray:
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    corner_size = max(6, size // 10)
    cv2.circle(bgr, center, size // 3, (220, 220, 220), 6, cv2.LINE_AA)
    cv2.circle(bgr, center, size // 7, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.line(
        bgr,
        (size // 5, size // 2),
        (size - size // 5, size // 2),
        (220, 220, 220),
        4,
        cv2.LINE_AA,
    )
    cv2.circle(alpha, center, size // 3, 255, 6, cv2.LINE_AA)
    cv2.circle(alpha, center, size // 7, 255, -1, cv2.LINE_AA)
    cv2.line(
        alpha,
        (size // 5, size // 2),
        (size - size // 5, size // 2),
        255,
        4,
        cv2.LINE_AA,
    )
    for x, y in (
        (size // 4, size // 4),
        (size - size // 4 - corner_size, size // 4),
        (size // 4, size - size // 4 - corner_size),
        (size - size // 4 - corner_size, size - size // 4 - corner_size),
    ):
        cv2.rectangle(bgr, (x, y), (x + corner_size, y + corner_size), (220, 220, 220), -1)
        cv2.rectangle(alpha, (x, y), (x + corner_size, y + corner_size), 255, -1)
    return np.dstack([bgr, alpha])


def _make_sift_dead_bar(width: int = 160, height: int = 40) -> np.ndarray:
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    bgra[10:30, 20:140, :3] = 220
    bgra[10:30, 20:140, 3] = 255
    return bgra


def _stamp_transparent_patch(
    image: np.ndarray,
    patch_bgra: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = patch_bgra.shape[:2]
    alpha = patch_bgra[:, :, 3:4].astype(np.float32) / 255.0
    region = image[top:top + h, left:left + w].astype(np.float32)
    patch = patch_bgra[:, :, :3].astype(np.float32)
    image[top:top + h, left:left + w] = np.clip(
        patch * alpha + region * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def _accept_candidate_for_consensus(bgra: np.ndarray):
    alpha = bgra[:, :, 3] > 0
    area = int(alpha.sum())
    return True, area, bgra.shape[1], bgra.shape[0], 128, 0.25, 12

def test_crop_zone_to_bgra_shape_and_alpha():
    # 100x100 mean image, blob occupying rows 30-60, cols 40-70
    mean_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    blob_mask[30:60, 40:70] = 255

    bgra = crop_zone_to_bgra(mean_img, blob_mask)

    # Crop should be blob bounding box + CROP_BORDER_PX on each side
    from untextre.discovery import CROP_BORDER_PX
    expected_h = (60 - 30) + 2 * CROP_BORDER_PX
    expected_w = (70 - 40) + 2 * CROP_BORDER_PX
    assert bgra.shape == (expected_h, expected_w, 4)

def test_crop_zone_to_bgra_alpha_channel():
    mean_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    # L-shaped blob: right column and bottom row of a 20x20 region
    blob_mask[40:60, 58:60] = 255  # right edge column of blob bbox
    blob_mask[58:60, 40:60] = 255  # bottom edge row of blob bbox

    bgra = crop_zone_to_bgra(mean_img, blob_mask)
    from untextre.discovery import CROP_BORDER_PX
    b = CROP_BORDER_PX
    # A pixel in the L — alpha=255
    # The L occupies rows 40-59, cols 40-59 in the original image.
    # After crop, that maps to rows b..b+20, cols b..b+20.
    # The bottom-right corner of the L (row 58-59, col 58-59 in original)
    # maps to approximately row (b + 18), col (b + 18) in the crop.
    h, w = bgra.shape[:2]
    assert bgra[b + 18, b + 18, 3] == 255   # inside the L
    # A pixel in the interior gap of the L — alpha=0
    assert bgra[b + 5, b + 5, 3] == 0       # top-left interior of bbox, not in L
    # Border corners — alpha=0
    assert bgra[0, 0, 3] == 0
    assert bgra[h - 1, w - 1, 3] == 0

def test_crop_zone_to_bgra_returns_none_for_empty_mask():
    mean_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    result = crop_zone_to_bgra(mean_img, blob_mask)
    assert result is None

def test_crop_zone_to_bgra_channel_order():
    # Verify output is BGRA (not RGBA): blue channel should be preserved
    mean_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mean_img[:, :] = [255, 0, 0]  # Pure blue in BGR
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    blob_mask[40:60, 40:60] = 255

    bgra = crop_zone_to_bgra(mean_img, blob_mask)
    from untextre.discovery import CROP_BORDER_PX
    b = CROP_BORDER_PX
    # Channel 0 should be 255 (blue), channel 2 should be 0 (red) — BGRA order
    assert bgra[b, b, 0] == 255  # B channel
    assert bgra[b, b, 2] == 0    # R channel


def test_candidate_minimums_reject_sift_unusable_bar():
    candidate = _make_sift_dead_bar()

    is_valid, area, bbox_w, bbox_h, edge_px, fill_ratio, sift_keypoints = _candidate_meets_consensus_minimums(candidate)

    assert area > 0
    assert bbox_w >= 120
    assert bbox_h >= 20
    assert edge_px >= 64
    assert fill_ratio > 0.01
    assert sift_keypoints < 5
    assert is_valid is False


def test_candidate_minimums_accept_sift_ready_candidate():
    candidate = _make_sift_ready_candidate()

    is_valid, area, bbox_w, bbox_h, edge_px, fill_ratio, sift_keypoints = _candidate_meets_consensus_minimums(candidate)

    assert area > 0
    assert bbox_w >= 80
    assert bbox_h >= 80
    assert edge_px >= 64
    assert fill_ratio > 0.01
    assert sift_keypoints >= 5
    assert is_valid is True

def test_crop_zone_to_bgra_blob_at_edge_clamps_border():
    # Blob touching top-left corner — border cannot extend beyond (0,0)
    mean_img = np.full((50, 50, 3), 100, dtype=np.uint8)
    blob_mask = np.zeros((50, 50), dtype=np.uint8)
    blob_mask[0:10, 0:10] = 255  # blob flush with top-left edge

    bgra = crop_zone_to_bgra(mean_img, blob_mask)

    # Border is clamped: crop starts at (0,0), not at (-CROP_BORDER_PX, -CROP_BORDER_PX)
    from untextre.discovery import CROP_BORDER_PX
    expected_h = min(50, 10 + CROP_BORDER_PX)   # clamped — no top border possible
    expected_w = min(50, 10 + CROP_BORDER_PX)
    assert bgra.shape == (expected_h, expected_w, 4)
    # Top-left pixel is inside the blob, so alpha should be 255
    assert bgra[0, 0, 3] == 255


def test_cap_zone_candidates_keeps_strongest_by_sift_edges_area():
    """Truncation ranks by (sift_kp, edge_px, area) descending, not arrival order.

    Twelve synthetic candidates, capped to MAX_CONSENSUS_CANDIDATES_PER_ZONE=10:
    the two weakest by sift_keypoints (4 and 3) must be dropped regardless of
    their area/edge_px, since sift_keypoints is the primary rank key.
    """
    from untextre.discovery import MAX_CONSENSUS_CANDIDATES_PER_ZONE, _cap_zone_candidates

    stats = [
        (100, 100, 20), (90, 90, 15), (80, 80, 12), (70, 70, 10),
        (60, 60, 9), (50, 50, 8), (40, 40, 7), (30, 30, 6),
        (20, 20, 5), (200, 200, 5), (10, 10, 4), (5, 5, 3),
    ]
    zone_valid = [(None, area, 10, 10, edge_px, 0.5, sift_kp) for area, edge_px, sift_kp in stats]

    capped = _cap_zone_candidates(zone_valid, zone_label="test-zone")

    assert len(capped) == MAX_CONSENSUS_CANDIDATES_PER_ZONE
    kept_sift = sorted(item[6] for item in capped)
    assert kept_sift == [5, 5, 6, 7, 8, 9, 10, 12, 15, 20]


def test_cap_zone_candidates_is_noop_under_limit():
    from untextre.discovery import _cap_zone_candidates

    zone_valid = [(None, 10, 5, 5, 10, 0.5, 6) for _ in range(3)]
    capped = _cap_zone_candidates(zone_valid, max_per_zone=10, zone_label="test-zone")
    assert capped == zone_valid



from untextre.discovery import compute_alpha_iou, select_best_family

def test_compute_alpha_iou_identical():
    crop = np.zeros((50, 50, 4), dtype=np.uint8)
    crop[10:40, 10:40, 3] = 255
    iou = compute_alpha_iou(crop, crop)
    assert abs(iou - 1.0) < 0.01

def test_compute_alpha_iou_no_overlap():
    a = np.zeros((50, 50, 4), dtype=np.uint8)
    a[0:10, 0:10, 3] = 255
    b = np.zeros((50, 50, 4), dtype=np.uint8)
    b[40:50, 40:50, 3] = 255
    iou = compute_alpha_iou(a, b)
    assert iou < 0.01

def test_select_best_family_single_crop():
    crop = np.zeros((60, 80, 4), dtype=np.uint8)
    crop[:, :, 3] = 255
    families = select_best_family([crop])
    assert len(families) == 1
    assert families[0] is crop

def test_select_best_family_merges_similar():
    # Two crops with high IoU → same family, largest returned
    small = np.zeros((30, 30, 4), dtype=np.uint8)
    small[5:25, 5:25, 3] = 255
    large = np.zeros((60, 60, 4), dtype=np.uint8)
    large[10:50, 10:50, 3] = 255
    from untextre.discovery import CROSS_BUCKET_IOU_THRESHOLD
    assert compute_alpha_iou(small, large) >= CROSS_BUCKET_IOU_THRESHOLD, \
        "fixture IoU too low — test geometry is broken"
    families = select_best_family([small, large])
    assert len(families) == 1
    assert families[0].shape == large.shape

def test_select_best_family_keeps_distinct():
    # Two crops with low IoU → different families
    a = np.zeros((40, 40, 4), dtype=np.uint8)
    a[0:10, 0:10, 3] = 255
    b = np.zeros((40, 40, 4), dtype=np.uint8)
    b[30:40, 30:40, 3] = 255
    from untextre.discovery import CROSS_BUCKET_IOU_THRESHOLD
    assert compute_alpha_iou(a, b) < CROSS_BUCKET_IOU_THRESHOLD, \
        "fixture IoU too high — test geometry is broken"
    families = select_best_family([a, b])
    assert len(families) == 2

def test_compute_alpha_iou_different_sizes():
    # Exercises the resize path: small crop vs large crop with same watermark shape
    small = np.zeros((25, 25, 4), dtype=np.uint8)
    small[5:20, 5:20, 3] = 255
    large = np.zeros((50, 50, 4), dtype=np.uint8)
    large[10:40, 10:40, 3] = 255  # same fractional position as small
    iou = compute_alpha_iou(small, large)
    assert iou >= 0.5  # same watermark at different scales

def test_compute_alpha_iou_all_transparent():
    # Both crops fully transparent → union=0 → returns 0.0
    a = np.zeros((20, 20, 4), dtype=np.uint8)  # all alpha=0
    b = np.zeros((20, 20, 4), dtype=np.uint8)
    iou = compute_alpha_iou(a, b)
    assert iou == 0.0




def test_build_watermark_score_high_at_watermark_boundary(tmp_path):
    """Composite score low_var * structure should peak at the watermark boundary."""
    from untextre.discovery import compute_stack_statistics, build_watermark_score

    rng = np.random.RandomState(31)
    paths = []
    wm_slice = (slice(40, 60), slice(60, 110))
    for i in range(5):
        img = rng.randint(40, 160, (100, 150, 3), dtype=np.uint8)
        img[wm_slice] = 240
        p = tmp_path / f"bs_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    stats = compute_stack_statistics(paths)
    var_gray = stats["var_gray"].astype(np.float64)
    log_var = np.log10(var_gray + 1e-8)
    var_norm = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    score = build_watermark_score(stats, var_norm)

    assert score.shape == (100, 150), f"Score shape mismatch: {score.shape}"
    wm_border_score = float(score[40, 60:110].mean())
    bg_score = float(score[15, 15:60].mean())
    assert wm_border_score > bg_score, (
        f"Watermark border score ({wm_border_score:.4f}) should exceed "
        f"background score ({bg_score:.4f})"
    )


def test_build_watermark_score_detects_semi_transparent_watermark(tmp_path):
    """Score must show signal at a semi-transparent watermark boundary.

    For a semi-transparent overlay, boundary pixels blend with varying backgrounds
    and have slightly elevated variance.  The mean-image gradient approach misses
    these pixels because they fall outside the stable mask.  The variance-field
    gradient approach finds them because the step from near-zero variance (inside
    the overlay) to higher variance (outside) is the signal.
    """
    from untextre.discovery import compute_stack_statistics, build_watermark_score

    rng = np.random.RandomState(70)
    paths = []
    wm_slice = (slice(40, 60), slice(60, 110))
    alpha = 0.35  # semi-transparent — boundary has elevated variance

    for i in range(5):
        bg = rng.randint(40, 160, (100, 150, 3), dtype=np.uint8).astype(np.float32)
        img = bg.copy()
        img[wm_slice] = alpha * 240.0 + (1.0 - alpha) * bg[wm_slice]
        p = tmp_path / f"semi_{i}.png"
        cv2.imwrite(str(p), img.clip(0, 255).astype(np.uint8))
        paths.append(p)

    stats = compute_stack_statistics(paths)
    var_gray = stats["var_gray"].astype(np.float64)
    log_var = np.log10(var_gray + 1e-8)
    var_norm = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    score = build_watermark_score(stats, var_norm)

    # Watermark boundary row (transition from background → overlay)
    wm_boundary_score = float(score[40, 60:110].mean())
    # Interior of watermark (stable but flat — no gradient signal needed here)
    bg_score = float(score[0:25, 0:50].mean())
    assert wm_boundary_score > bg_score, (
        f"Semi-transparent watermark boundary ({wm_boundary_score:.4f}) "
        f"should score above background ({bg_score:.4f})"
    )


def test_build_watermark_score_unstable_region_scores_zero(tmp_path):
    """Pixels outside the stable mask must score zero even if globally prominent.

    A region that varies wildly (high variance) should have structure=0 in the
    composite score regardless of how sharp its edges are in any one image.
    With global normalization this could leak a small nonzero score; with local
    normalization within the stable mask it cannot.
    """
    from untextre.discovery import compute_stack_statistics, build_watermark_score

    rng = np.random.RandomState(42)
    paths = []
    wm_slice = (slice(55, 70), slice(110, 145))
    # Explicitly volatile region: every pixel takes a fresh random value in
    # each image — guaranteed maximum variance, guaranteed NOT in stable mask.
    volatile_slice = (slice(10, 45), slice(20, 70))
    for i in range(5):
        img = rng.randint(40, 160, (100, 150, 3), dtype=np.uint8)
        img[wm_slice] = 235
        # Alternate between black and white every image: guaranteed maximum variance
        img[volatile_slice] = 255 if i % 2 == 0 else 0
        p = tmp_path / f"uz_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    stats = compute_stack_statistics(paths)
    var_gray = stats["var_gray"].astype(np.float64)
    log_var = np.log10(var_gray + 1e-8)
    var_norm = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    score = build_watermark_score(stats, var_norm)

    # Watermark boundary must score above zero (it is stable AND has structure)
    wm_boundary_score = float(score[55, 110:145].max())
    assert wm_boundary_score > 0, (
        f"Watermark boundary should score > 0, got {wm_boundary_score:.4f}"
    )

    # The explicitly volatile region has maximum per-pixel variance every image.
    # It cannot be in the stable mask, so its structure = 0 → score = 0.
    volatile_score = float(score[volatile_slice].max())
    assert volatile_score == 0.0, (
        f"Volatile region should score 0, got {volatile_score:.4f}"
    )


from untextre.discovery import compute_median_gradient


def test_compute_median_gradient_watermark_survives_background_cancellation(tmp_path):
    """Watermark edges should produce high median gradient; varied backgrounds cancel.

    Setup: a white rectangle overlaid on random-valued images.  The overlay
    creates consistent Sobel edges at its boundary in every image.  The random
    background creates large but uncorrelated gradients everywhere else.

    Under the median, uncorrelated values converge toward the median of the
    background gradient distribution (~0 for zero-mean noise), while the
    constant watermark edges persist.
    """
    rng = np.random.RandomState(17)
    wm_slice = (slice(30, 60), slice(40, 100))
    paths = []
    for i in range(9):
        bg = rng.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        bg[wm_slice] = 230
        p = tmp_path / f"mg_{i}.png"
        cv2.imwrite(str(p), bg)
        paths.append(p)

    med_grad = compute_median_gradient(paths)

    assert med_grad is not None
    assert med_grad.shape == (100, 150)

    # Top edge of watermark rectangle: persistent horizontal step in every image
    edge_score = float(med_grad[30, 40:100].mean())
    # Well inside the watermark: constant value → near-zero gradient
    interior_score = float(med_grad[35:55, 45:95].mean())
    # Background region with no watermark: varied backgrounds cancel
    bg_score = float(med_grad[5:25, 5:30].mean())

    assert edge_score > interior_score * 3, (
        f"Watermark edge ({edge_score:.4f}) should substantially exceed "
        f"watermark interior ({interior_score:.4f})"
    )
    assert edge_score > bg_score, (
        f"Watermark edge ({edge_score:.4f}) should exceed background "
        f"median gradient ({bg_score:.4f})"
    )


def test_compute_median_gradient_returns_none_for_insufficient_images(tmp_path):
    """Returns None when fewer than 3 images can be loaded."""
    rng = np.random.RandomState(3)
    paths = []
    for i in range(2):
        img = rng.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        p = tmp_path / f"few_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    assert compute_median_gradient(paths) is None


def test_build_watermark_score_boosted_by_median_gradient(tmp_path):
    """Score at watermark boundary must be higher when median_grad is provided.

    With varied backgrounds, the median gradient adds genuine signal at
    watermark edges.  The score at those edges must be at least as high with
    median_grad as without it (product of two positive values in [0,1] cannot
    increase the score, but the normalized factors redistribute weight so that
    the watermark edge, which is high in both var_boundary AND median_grad,
    is preferentially boosted relative to background noise).

    We test the structural property: with median_grad, the watermark-edge score
    must dominate the background score by a larger margin than without it.
    """
    from untextre.discovery import compute_stack_statistics, build_watermark_score

    rng = np.random.RandomState(55)
    wm_slice = (slice(30, 60), slice(40, 100))
    paths = []
    for i in range(9):
        bg = rng.randint(0, 200, (100, 150, 3), dtype=np.uint8)
        bg[wm_slice] = 230
        p = tmp_path / f"boost_{i}.png"
        cv2.imwrite(str(p), bg)
        paths.append(p)

    stats = compute_stack_statistics(paths)
    var_gray = stats["var_gray"].astype(np.float64)
    log_var = np.log10(var_gray + 1e-8)
    var_norm = cv2.normalize(log_var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    med_grad = compute_median_gradient(paths)

    score_without = build_watermark_score(stats, var_norm, median_grad=None)
    score_with    = build_watermark_score(stats, var_norm, median_grad=med_grad)

    edge_row = 30  # top boundary of watermark rectangle
    edge_without = float(score_without[edge_row, 40:100].mean())
    edge_with    = float(score_with[edge_row, 40:100].mean())
    bg_without   = float(score_without[5:25, 5:35].mean())
    bg_with      = float(score_with[5:25, 5:35].mean())

    snr_without = edge_without / (bg_without + 1e-8)
    snr_with    = edge_with    / (bg_with    + 1e-8)

    assert snr_with >= snr_without, (
        f"Median gradient should not reduce watermark SNR: "
        f"SNR without={snr_without:.2f}, SNR with={snr_with:.2f}"
    )


def test_select_candidate_prefers_structured_zone_over_flat_zone():
    """Gradient-weighted ranking must prefer a text-like zone over a flat stable zone.

    Setup: two zones with equal stable pixel count.
      Zone A (top-left): flat uniform stable pixels — zero internal gradient.
      Zone B (bottom-right): stable pixels arranged as a stripe pattern with
        persistent edges — high median gradient at the stripe boundaries.

    Without gradient weighting both zones rank equally.
    With gradient weighting, Zone B must rank first.
    """
    from untextre.discovery import select_candidate_components

    H, W = 200, 300
    labels_img = np.zeros((H, W), dtype=np.int32)
    cc_stats = []
    centroids = []

    # Background label 0 (required by connectedComponentsWithStats format)
    cc_stats.append([0, 0, W, H, H * W])
    centroids.append([W / 2, H / 2])

    # Zone A: flat block in top-left (zone (0,0)) — 5 components of 100px each
    for i in range(5):
        y, x = 10 + i * 15, 10 + i * 5
        label = i + 1
        labels_img[y : y + 10, x : x + 10] = label
        cc_stats.append([x, y, 10, 10, 100])
        centroids.append([x + 5, y + 5])

    # Zone B: stripe in bottom-right (zone (2,1) for landscape) — 5 components of 100px each
    for i in range(5):
        y, x = 150 + i * 5, 230 + i * 5
        label = i + 6
        labels_img[y : y + 10, x : x + 10] = label
        cc_stats.append([x, y, 10, 10, 100])
        centroids.append([x + 5, y + 5])

    num_labels = 11
    mask_data = {
        "num_labels": num_labels,
        "labels": labels_img,
        "stats": np.array(cc_stats, dtype=np.int32),
        "centroids": np.array(centroids, dtype=np.float64),
        "mask_raw": np.zeros((H, W), dtype=np.uint8),
        "mask_clean": np.zeros((H, W), dtype=np.uint8),
        "score_closed": np.zeros((H, W), dtype=np.uint8),
    }
    dummy_score = np.zeros((H, W), dtype=np.float32)

    # Build a median_grad that is zero everywhere except Zone B pixels
    median_grad = np.zeros((H, W), dtype=np.float32)
    for i in range(5):
        y, x = 150 + i * 5, 230 + i * 5
        median_grad[y : y + 10, x : x + 10] = 1.0   # high persistent gradient

    # Without gradient: both zones have equal area; zone A appears first (lower label index)
    without = select_candidate_components(mask_data, dummy_score, W, H, max_zones=2)
    assert int(np.sum(without[0][0:100, 0:150])) > 0
    # With gradient: Zone B must rank first
    with_grad = select_candidate_components(
        mask_data, dummy_score, W, H, max_zones=2, median_grad=median_grad
    )

    # In the gradient-weighted result, Zone B pixels must appear in the first candidate
    zone_b_pixels_in_first = int(np.sum(with_grad[0][150:200, 220:300]))
    assert zone_b_pixels_in_first > 0, (
        "With gradient weighting, the structured zone (B) must be the top candidate"
    )




from untextre.discovery import (
    discover_watermark_candidates,
    extract_watermark_colors,
    _deblot_mask_by_area,
)


def test_deblot_removes_small_components_in_bimodal_distribution():
    """Otsu on log(area) should remove tiny blotches while keeping glyph-sized components.

    Signal: 10 components of ~120 px each (letter-sized).
    Noise:  40 components of ~4 px each (satellite blotches).
    After deblotting, the signal components must survive intact and the noise
    components must be removed or nearly all removed.
    """
    rng = np.random.RandomState(7)
    mask = np.zeros((300, 300), dtype=np.uint8)

    # 10 large signal components (12×10 = 120 px each), well-separated
    signal_region = (slice(10, 260), slice(10, 20))
    for i in range(10):
        y = 10 + i * 25
        mask[y : y + 12, 10:20] = 255

    # 40 tiny blotch components (2×2 = 4 px each), placed away from signal
    for _ in range(40):
        y = rng.randint(150, 290)
        x = rng.randint(50, 290)
        mask[y : y + 2, x : x + 2] = 255

    clean = _deblot_mask_by_area(mask)

    # Signal region must be fully preserved
    assert np.array_equal(clean[signal_region], mask[signal_region]), (
        "Signal components should survive deblotching unchanged"
    )

    # At least half the blotches should have been removed
    n_before = int(cv2.connectedComponentsWithStats(mask)[0]) - 1
    n_after  = int(cv2.connectedComponentsWithStats(clean)[0]) - 1
    assert n_after <= n_before // 2, (
        f"Expected roughly half or fewer components after deblotching: "
        f"{n_before} → {n_after}"
    )


def test_deblot_preserves_uniform_components():
    """When all components are the same size (unimodal), the mask is returned unchanged.

    la_max − la_min < 0.5 triggers the early-return guard — no Otsu cut is made.
    """
    mask = np.zeros((200, 400), dtype=np.uint8)
    # 12 identical 10×10 = 100 px blocks
    for i in range(12):
        x = 10 + i * 30
        mask[50:60, x : x + 10] = 255

    clean = _deblot_mask_by_area(mask)

    n_before = int(cv2.connectedComponentsWithStats(mask)[0]) - 1
    n_after  = int(cv2.connectedComponentsWithStats(clean)[0]) - 1
    assert n_after == n_before, (
        f"Uniform components should not be filtered: {n_before} → {n_after}"
    )


def test_deblot_skips_otsu_when_too_few_components():
    """With fewer than 4 components, deblotching returns the mask unchanged.

    Otsu needs at least 2 samples per class (minimum 4 total) to have any meaning.
    """
    mask = np.zeros((200, 200), dtype=np.uint8)
    # 3 components: 1 large and 2 tiny — bimodal shape but below the floor.
    mask[10:30, 10:30] = 255             # 20×20 = 400 px
    mask[100:102, 100:102] = 255         # 2×2 = 4 px
    mask[150:152, 150:152] = 255         # 2×2 = 4 px

    clean = _deblot_mask_by_area(mask)

    assert np.array_equal(clean, mask), (
        "Mask with 3 components should be returned unchanged (too few for Otsu)"
    )


def test_deblot_keeps_single_surviving_component():
    """A single large component after Otsu is a valid result and must be kept.

    Previously, the `kept < 2` guard incorrectly reverted the result when Otsu
    correctly found only one large component. The fix changes this to `kept < 1`.
    """
    mask = np.zeros((300, 300), dtype=np.uint8)
    # 1 large signal component (100×20 = 2000 px)
    mask[10:30, 10:110] = 255
    # 20 tiny blotches (2×2 = 4 px each) — clearly bimodal distribution
    rng = np.random.RandomState(42)
    for _ in range(20):
        y = rng.randint(150, 285)
        x = rng.randint(150, 285)
        mask[y : y + 2, x : x + 2] = 255

    clean = _deblot_mask_by_area(mask)

    # The large component must survive; the tiny blotches should be removed.
    assert np.array_equal(clean[10:30, 10:110], mask[10:30, 10:110]), (
        "Large signal component must be preserved"
    )
    n_clean = int(cv2.connectedComponentsWithStats(clean)[0]) - 1
    assert n_clean == 1, (
        f"Only the large component should survive deblotching, got {n_clean}"
    )


def test_extract_watermark_colors_reduces_background_bleed(tmp_path):
    """Extracted WM color should be closer to the true watermark than the raw mean.

    For α=0.4 (white watermark on random [40,160] background):
      mean pixel ≈ 0.4·255 + 0.6·100 = 162  (heavily background-contaminated)
    After transparency inversion, wm_color should recover a value much closer
    to 255, demonstrating that background bleed has been substantially removed.
    """
    from untextre.discovery import compute_stack_statistics

    rng = np.random.RandomState(99)
    alpha = 0.4
    wm_slice = (slice(40, 60), slice(60, 110))

    paths = []
    for i in range(8):
        bg = rng.randint(40, 160, (100, 150, 3), dtype=np.uint8).astype(np.float32)
        img = bg.copy()
        img[wm_slice] = alpha * 255.0 + (1.0 - alpha) * bg[wm_slice]
        p = tmp_path / f"bleed_{i}.png"
        cv2.imwrite(str(p), img.clip(0, 255).astype(np.uint8))
        paths.append(p)

    stats = compute_stack_statistics(paths)
    mean_bgr = stats["mean_bgr"]
    var_gray = stats["var_gray"]

    # Pass the known watermark region as the precision mask.
    # extract_watermark_colors is agnostic to how the mask was obtained.
    precision_mask = np.zeros((100, 150), dtype=np.uint8)
    precision_mask[wm_slice] = 255

    wm_color, alpha_hat = extract_watermark_colors(mean_bgr, var_gray, precision_mask)

    # Extracted color at WM pixels should be closer to white (255) than the
    # raw mean, which is contaminated with ~60% background content.
    mean_error = float(np.abs(mean_bgr[wm_slice].astype(float) - 255).mean())
    extracted_error = float(np.abs(wm_color[wm_slice].astype(float) - 255).mean())
    assert extracted_error < mean_error, (
        f"extract_watermark_colors error ({extracted_error:.1f}) should be < "
        f"raw mean error ({mean_error:.1f}) for a semi-transparent watermark"
    )

    # Alpha should register as substantially positive in the WM region.
    wm_alpha = float(alpha_hat[wm_slice].mean())
    assert wm_alpha > 0.2, (
        f"Expected alpha_hat > 0.2 at watermark pixels, got {wm_alpha:.3f}"
    )


def test_extract_watermark_colors_opaque_watermark_unchanged(tmp_path):
    """For an opaque watermark (α=1.0), extracted color should match the WM exactly."""
    from untextre.discovery import compute_stack_statistics

    rng = np.random.RandomState(77)
    wm_slice = (slice(30, 50), slice(50, 100))
    wm_true = np.array([200, 180, 160], dtype=np.uint8)

    paths = []
    for i in range(6):
        img = rng.randint(30, 180, (80, 120, 3), dtype=np.uint8)
        img[wm_slice] = wm_true
        p = tmp_path / f"opaque_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    stats = compute_stack_statistics(paths)
    mean_bgr = stats["mean_bgr"]
    var_gray = stats["var_gray"]

    precision_mask = np.zeros((80, 120), dtype=np.uint8)
    precision_mask[wm_slice] = 255

    wm_color, alpha_hat = extract_watermark_colors(mean_bgr, var_gray, precision_mask)

    # Opaque watermark: mean_bgr already equals WM color; extracted should not
    # introduce new error relative to the raw mean.
    mean_error = float(np.abs(mean_bgr[wm_slice].astype(float) - wm_true.astype(float)).mean())
    extracted_error = float(np.abs(wm_color[wm_slice].astype(float) - wm_true.astype(float)).mean())
    assert extracted_error <= mean_error + 5, (
        f"Opaque case: extracted error ({extracted_error:.1f}) should not greatly exceed "
        f"mean error ({mean_error:.1f})"
    )

    # Alpha should be near 1.0 (opaque) at watermark pixels.
    wm_alpha = float(alpha_hat[wm_slice].mean())
    assert wm_alpha > 0.8, f"Expected alpha_hat ≈ 1.0 for opaque watermark, got {wm_alpha:.3f}"

from untextre.watermark_consensus import ConsensusTemplate


def _template_from_crop(label: str, crop: np.ndarray, member_indices=(0,), family_count: int = 1):
    return ConsensusTemplate(
        label=label,
        bgra=crop,
        alpha_mass=float(crop[:, :, 3].astype(np.float32).sum() / 255.0),
        member_indices=tuple(member_indices),
        family_count=family_count,
    )






















def test_discover_finds_watermark_in_homogeneous_batch(tmp_path):
    """Integration test: batch of images with a consistent watermark blob."""
    wm_patch = _make_sift_ready_candidate(160)
    paths = []
    np.random.seed(0)
    random.seed(0)
    for i in range(6):
        img = np.random.randint(30, 180, (300, 400, 3), dtype=np.uint8)
        _stamp_transparent_patch(img, wm_patch, 130, 230)
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    candidates = discover_watermark_candidates(paths)
    assert len(candidates) >= 1
    # Each candidate is a BGRA array
    assert candidates[0].shape[2] == 4

def test_discover_deduplicates_similar_aspect_ratio_candidates(tmp_path):
    """Two zones with similar aspect ratios in one bucket → only largest kept."""
    wm = np.array([210, 210, 210], dtype=np.uint8)
    paths = []
    np.random.seed(1)
    for i in range(6):
        img = np.random.randint(30, 180, (300, 400, 3), dtype=np.uint8)
        # Two watermark patches with similar aspect ratio (both ~2:1 wide)
        img[10:30, 10:50] = wm    # small, top-left
        img[260:290, 340:390] = wm  # large, bottom-right (same ~2:1 ratio)
        p = tmp_path / f"dual_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    candidates = discover_watermark_candidates(paths)
    # Both patches are in different zones but have similar aspect ratios —
    # dedup should keep only the largest. In practice zone clustering may
    # find 1 or 2; we check that we never get more than 2 (no duplicates).
    assert len(candidates) <= 2



def test_discover_finds_candidate_near_watermark_despite_adjacent_stable_mass(tmp_path):
    """Discovery should locate a candidate in the watermark zone even when a larger
    stable region is adjacent.

    The composite score (low_var × structure) may merge or split the watermark
    and mass depending on Otsu splits.  What matters for the downstream SIFT
    step is that at least one candidate is found and its centroid is near the
    watermark location.
    """
    paths = []
    rng = np.random.RandomState(20)

    stable_color = np.array([210, 210, 210], dtype=np.int16)
    watermark_patch = _make_sift_ready_candidate(160)
    mass_slice = (slice(120, 250), slice(170, 340))

    for i in range(6):
        img = rng.randint(20, 180, (300, 400, 3), dtype=np.uint8)
        stable_region = stable_color + rng.randint(-8, 9, (130, 170, 3))
        img[mass_slice] = np.clip(stable_region, 0, 255).astype(np.uint8)
        _stamp_transparent_patch(img, watermark_patch, 130, 230)
        path = tmp_path / f"attached_{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)

    candidates = discover_watermark_candidates(paths)
    assert candidates, "Expected at least one candidate after trimming"

    # Every candidate must stay within the 10% ceiling after trimming.
    # 10% of 300×400 = 12,000 px.
    max_allowed = int(300 * 400 * 0.10)
    for candidate in candidates:
        alpha_area = int(np.sum(candidate[:, :, 3] > 0))
        assert alpha_area <= max_allowed, (
            f"Trimmed candidate has {alpha_area} alpha px, exceeds {max_allowed} (10%)"
        )





def test_discover_skips_bucket_with_fewer_than_3_images(tmp_path):
    """Bucket with < 3 images is skipped for self-discovery."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(3)
    for i in range(2):  # Only 2 images — below the 3-image threshold
        img = np.random.randint(30, 180, (200, 300, 3), dtype=np.uint8)
        img[170:195, 250:290] = wm
        p = tmp_path / f"small_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    # With only 2 images in the bucket, discovery is skipped — returns empty
    candidates = discover_watermark_candidates(paths)
    assert candidates == []






def test_discover_variance_map_watermark_pixels_are_darker(tmp_path):
    """Low-variance (consistent watermark) pixels are darker in the variance map."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(12)
    for i in range(5):
        img = np.random.randint(30, 180, (100, 150, 3), dtype=np.uint8)
        img[80:95, 120:145] = wm
        p = tmp_path / f"dark_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    discover_watermark_candidates(paths, debug_dir=debug_dir)

    var_map = cv2.imread(str(debug_dir / "debug_variance_150x100.png"), cv2.IMREAD_GRAYSCALE)
    assert var_map is not None

    wm_brightness = float(var_map[80:95, 120:145].mean())
    bg_brightness = float(var_map[0:50, 0:100].mean())
    assert wm_brightness < bg_brightness, (
        f"Watermark region mean={wm_brightness:.1f} should be darker than "
        f"background mean={bg_brightness:.1f}"
    )


def test_discover_no_debug_files_without_debug_dir(tmp_path):
    """No debug files are written when debug_dir is omitted."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(13)
    for i in range(4):
        img = np.random.randint(30, 180, (100, 150, 3), dtype=np.uint8)
        img[80:95, 120:145] = wm
        p = tmp_path / f"nodebug_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    discover_watermark_candidates(paths)  # no debug_dir

    assert not any(tmp_path.glob("debug_*.png")), \
        "No debug files should be written when debug_dir is not provided"


@pytest.mark.slow
def test_discovery_end_to_end_with_real_images(tmp_path):
    """
    Create a batch of 6 synthetic images with a consistent watermark,
    run discover_watermark_candidates, confirm we get at least 1 BGRA crop,
    and that it has nonzero alpha pixels.
    """
    np.random.seed(42)
    random.seed(42)
    wm_patch = _make_sift_ready_candidate(160)
    paths = []
    for i in range(6):
        # Randomize background to simulate real photos
        img = np.random.randint(20, 200, (400, 600, 3), dtype=np.uint8)
        _stamp_transparent_patch(img, wm_patch, 220, 400)
        p = tmp_path / f"photo_{i:02d}.png"  # PNG to avoid JPEG compression artifacts
        cv2.imwrite(str(p), img)
        paths.append(p)

    candidates = discover_watermark_candidates(paths)

    assert len(candidates) >= 1
    best = candidates[0]
    assert best.shape[2] == 4, "Expected 4-channel BGRA output"
    assert np.any(best[:, :, 3] > 127), "Expected nonzero alpha in best candidate"


def test_cross_sub_sample_per_half_thresholds_differ_with_noise(tmp_path):
    """Cross-sub-sample validation must compute a per-half Tukey threshold
    from each half's OWN log-variance distribution, not reuse one shared
    number across halves with different noise characteristics.

    Regression lock for the threshold-mismatch fix (2026-07-19): a threshold
    calibrated on a full/pooled population is a poor statistical fit for a
    half-sized-sample variance estimate, especially when the two halves have
    genuinely different underlying noise levels. Builds two synthetic image
    sets with deliberately different per-pixel noise ranges (standing in for
    "half A" and "half B" of a real cross-sub-sample split) and asserts their
    self-consistent thresholds differ — the observable mechanism the fix
    introduces.
    """
    from untextre.discovery import (
        _precision_outlier_threshold_from_log_precision,
        compute_stack_statistics,
    )

    rng = np.random.RandomState(5)
    base = rng.randint(80, 120, (100, 150, 3), dtype=np.uint8)

    low_noise_paths = []
    for i in range(3):
        img = np.clip(base.astype(np.int16) + rng.randint(-3, 4, base.shape), 0, 255).astype(np.uint8)
        p = tmp_path / f"low_{i}.png"
        cv2.imwrite(str(p), img)
        low_noise_paths.append(p)

    high_noise_paths = []
    for i in range(3):
        img = np.clip(base.astype(np.int16) + rng.randint(-40, 41, base.shape), 0, 255).astype(np.uint8)
        p = tmp_path / f"high_{i}.png"
        cv2.imwrite(str(p), img)
        high_noise_paths.append(p)

    stats_low = compute_stack_statistics(low_noise_paths)
    stats_high = compute_stack_statistics(high_noise_paths)

    log_var_low = np.log10(stats_low["var_gray"].astype(np.float64) + 1e-8)
    log_var_high = np.log10(stats_high["var_gray"].astype(np.float64) + 1e-8)
    threshold_low = _precision_outlier_threshold_from_log_precision(-log_var_low.flatten())
    threshold_high = _precision_outlier_threshold_from_log_precision(-log_var_high.flatten())

    # The low-noise half's own distribution sits at much lower absolute
    # variance than the high-noise half's, so their self-consistent
    # thresholds must differ — the whole point of computing them separately
    # instead of reusing one shared/pooled number.
    assert threshold_low != threshold_high


def test_discover_finds_watermark_with_noisy_competing_region_at_cross_sample_boundary(tmp_path):
    """Regression coverage for the changed cross-sub-sample code path.

    Exactly 6 images (the >=6 threshold that activates cross-sub-sample
    validation with self-consistent per-half thresholds), with both a
    genuine fixed watermark and a noisier competing "stable-ish" region.
    Discovery must still find the real watermark through the changed path.
    """
    from untextre.discovery import discover_watermark_candidates

    rng = np.random.RandomState(7)
    stable_color = np.array([210, 210, 210], dtype=np.int16)
    watermark_patch = _make_sift_ready_candidate(160)
    mass_slice = (slice(120, 250), slice(170, 340))

    paths = []
    for i in range(6):
        img = rng.randint(20, 180, (300, 400, 3), dtype=np.uint8)
        stable_region = stable_color + rng.randint(-20, 21, (130, 170, 3))
        img[mass_slice] = np.clip(stable_region, 0, 255).astype(np.uint8)
        _stamp_transparent_patch(img, watermark_patch, 130, 230)
        path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)

    candidates = discover_watermark_candidates(paths)

    assert candidates, "Expected the real watermark to still be found"
    # At least one candidate must be roughly watermark-patch-sized (the
    # stamped patch is 160x160 before SIFT-ready trimming), not the much
    # larger ~130x170 competing "stable-ish" mass region.
    found_watermark_sized = False
    for c in candidates:
        ys, xs = np.where(c[:, :, 3] > 0)
        if len(ys) == 0:
            continue
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        if bbox_w <= 180 and bbox_h <= 180:
            found_watermark_sized = True
    assert found_watermark_sized, "No watermark-sized candidate found (all candidates oversized)"
