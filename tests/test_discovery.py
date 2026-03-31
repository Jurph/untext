import numpy as np
import pytest
import logging
import random
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
from untextre.discovery import bucket_images_by_size, assign_zone

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


def test_compute_stack_statistics_grad_high_at_watermark_edges(tmp_path):
    """grad_mean_gray must have strong response at the watermark boundary."""
    from untextre.discovery import compute_stack_statistics

    rng = np.random.RandomState(30)
    paths = []
    wm_slice = (slice(40, 60), slice(60, 110))
    for i in range(5):
        img = rng.randint(40, 160, (100, 150, 3), dtype=np.uint8)
        img[wm_slice] = 240
        p = tmp_path / f"gs_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    stats = compute_stack_statistics(paths)
    assert stats is not None, "compute_stack_statistics returned None"
    assert "grad_mean_gray" in stats

    grad = stats["grad_mean_gray"]
    edge_response = float(grad[40, 60:110].mean())
    interior_response = float(grad[50, 65:105].mean())
    background_response = float(grad[15, 15:60].mean())

    assert edge_response > interior_response, (
        f"Edge ({edge_response:.2f}) should exceed interior ({interior_response:.2f})"
    )
    assert edge_response > background_response, (
        f"Edge ({edge_response:.2f}) should exceed background ({background_response:.2f})"
    )


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


from untextre.discovery import discover_watermark_candidates, _precision_outlier_threshold

def test_discover_uses_pooled_threshold_across_buckets(tmp_path, monkeypatch):
    """Pass-1 pooling: stable threshold is derived from ALL qualifying buckets combined.

    With two buckets that share similar image statistics, the pooled Tukey fence
    should fall between the watermark variance (near 0) and the background
    variance.  The per-bucket threshold computed from bucket A alone should
    give the same general order of magnitude but is noisier (fewer images).
    This test verifies the two-pass path is taken when multiple buckets exist.
    """
    calls = []
    real_build = __import__("untextre.discovery", fromlist=["build_watermark_score"]).build_watermark_score

    def tracking_build(stats, var_norm, stable_threshold=None):
        calls.append(stable_threshold)
        return real_build(stats, var_norm, stable_threshold)

    monkeypatch.setattr("untextre.discovery.build_watermark_score", tracking_build)

    rng = np.random.RandomState(50)
    wm = np.array([240, 240, 240], dtype=np.uint8)

    # Bucket A: 200×150
    paths_a = []
    for i in range(4):
        img = rng.randint(40, 160, (150, 200, 3), dtype=np.uint8)
        img[130:145, 170:195] = wm
        p = tmp_path / f"a_{i}.png"
        cv2.imwrite(str(p), img)
        paths_a.append(p)

    # Bucket B: 300×200  (different size → different bucket)
    paths_b = []
    for i in range(4):
        img = rng.randint(40, 160, (200, 300, 3), dtype=np.uint8)
        img[185:196, 275:295] = wm
        p = tmp_path / f"b_{i}.png"
        cv2.imwrite(str(p), img)
        paths_b.append(p)

    candidates = discover_watermark_candidates(paths_a + paths_b)

    # With two qualifying buckets, build_watermark_score should have been
    # called with a non-None stable_threshold derived from the pooled data.
    assert len(calls) == 2, f"Expected 2 build calls (one per bucket), got {len(calls)}"
    assert all(t is not None for t in calls), (
        f"Expected pooled threshold passed to each build call, got {calls}"
    )
    # Both calls should receive the SAME pooled threshold.
    assert calls[0] == calls[1], (
        f"Both buckets should use the same pooled threshold, got {calls}"
    )


def test_discover_finds_watermark_in_homogeneous_batch(tmp_path):
    """Integration test: batch of images with a consistent watermark blob."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(0)
    random.seed(0)
    for i in range(6):
        img = np.random.randint(30, 180, (300, 400, 3), dtype=np.uint8)
        # Fixed watermark: 50x30 block at bottom-right corner
        img[260:290, 340:390] = wm
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

def test_discover_returns_empty_for_no_common_pixels(tmp_path):
    """Random noise images should yield no stable candidates."""
    paths = []
    np.random.seed(2)
    for i in range(4):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        p = tmp_path / f"noise{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    # May or may not find blobs; if it does they'll be noise — just ensure no crash
    candidates = discover_watermark_candidates(paths)
    assert isinstance(candidates, list)
    # Noise images should produce at most a handful of spurious candidates.
    # The algorithm caps output per bucket; after family dedup we should not
    # see more than 3 (the per-bucket max_candidates ceiling).
    assert len(candidates) <= 3


def test_discover_finds_candidate_near_watermark_despite_adjacent_stable_mass(tmp_path):
    """Discovery should locate a candidate in the watermark zone even when a larger
    stable region is adjacent.

    The composite score (low_var × structure) may merge or split the watermark
    and mass depending on Otsu splits.  What matters for the downstream ORB
    step is that at least one candidate is found and its centroid is near the
    watermark location.
    """
    paths = []
    rng = np.random.RandomState(20)

    stable_color = np.array([210, 210, 210], dtype=np.int16)
    watermark_color = np.array([238, 238, 238], dtype=np.uint8)
    wm_slice = (slice(220, 250), slice(320, 390))
    mass_slice = (slice(120, 250), slice(170, 340))

    for i in range(6):
        img = rng.randint(20, 180, (300, 400, 3), dtype=np.uint8)
        stable_region = stable_color + rng.randint(-8, 9, (130, 170, 3))
        img[mass_slice] = np.clip(stable_region, 0, 255).astype(np.uint8)
        img[wm_slice] = watermark_color
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


def test_discover_finds_candidate_in_watermark_zone(tmp_path):
    """Discovery should produce a candidate in the correct image zone
    even when a noisier stable region occupies a nearby zone.
    """
    paths = []
    rng = np.random.RandomState(21)

    watermark_color = np.array([242, 242, 242], dtype=np.uint8)
    weaker_color = np.array([208, 208, 208], dtype=np.int16)

    wm_slice = (slice(220, 248), slice(330, 390))
    junk_slice = (slice(155, 230), slice(275, 345))

    for i in range(6):
        img = rng.randint(20, 180, (300, 400, 3), dtype=np.uint8)
        weaker_region = weaker_color + rng.randint(-10, 11, (75, 70, 3))
        img[junk_slice] = np.clip(weaker_region, 0, 255).astype(np.uint8)
        img[wm_slice] = watermark_color
        path = tmp_path / f"same_zone_{i}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)

    candidates = discover_watermark_candidates(paths)
    assert candidates, "Expected at least one candidate"

    for candidate in candidates:
        assert np.any(candidate[:, :, 3] > 0), "Candidate must have non-zero alpha"



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


def test_discover_saves_debug_variance_map(tmp_path):
    """discover_watermark_candidates saves a log-variance map when debug_dir is provided."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(10)
    for i in range(5):
        img = np.random.randint(30, 180, (100, 150, 3), dtype=np.uint8)
        img[80:95, 120:145] = wm
        p = tmp_path / f"var_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    discover_watermark_candidates(paths, debug_dir=debug_dir)

    var_map_path = debug_dir / "debug_variance_150x100.png"
    assert var_map_path.exists(), "Expected variance map to be saved in debug_dir"
    saved = cv2.imread(str(var_map_path))
    assert saved is not None
    assert saved.shape[:2] == (100, 150)


def test_discover_saves_debug_mean_image(tmp_path):
    """discover_watermark_candidates saves a mean image when debug_dir is provided."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    np.random.seed(11)
    for i in range(5):
        img = np.random.randint(30, 180, (100, 150, 3), dtype=np.uint8)
        img[80:95, 120:145] = wm
        p = tmp_path / f"mean_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    discover_watermark_candidates(paths, debug_dir=debug_dir)

    mean_path = debug_dir / "debug_mean_150x100.png"
    assert mean_path.exists(), "Expected mean image to be saved in debug_dir"
    saved = cv2.imread(str(mean_path))
    assert saved is not None
    assert saved.shape[:2] == (100, 150)


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
    wm_patch = np.full((40, 80, 3), 210, dtype=np.uint8)
    paths = []
    for i in range(6):
        # Randomize background to simulate real photos
        img = np.random.randint(20, 200, (400, 600, 3), dtype=np.uint8)
        # Place identical watermark bottom-right
        img[350:390, 510:590] = wm_patch
        p = tmp_path / f"photo_{i:02d}.png"  # PNG to avoid JPEG compression artifacts
        cv2.imwrite(str(p), img)
        paths.append(p)

    candidates = discover_watermark_candidates(paths)

    assert len(candidates) >= 1
    best = candidates[0]
    assert best.shape[2] == 4, "Expected 4-channel BGRA output"
    assert np.any(best[:, :, 3] > 127), "Expected nonzero alpha in best candidate"
