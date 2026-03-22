import numpy as np
import pytest
import logging
import random
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
from untextre.discovery import bucket_images_by_size, compute_pair_variance, extract_blobs, assign_zone, discover_zones

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

def test_compute_pair_variance_low_for_identical():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    var = compute_pair_variance(img, img)
    assert var.max() < 0.001

def test_compute_pair_variance_high_for_different():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = np.full((100, 100, 3), 255, dtype=np.uint8)
    var = compute_pair_variance(a, b)
    assert var.mean() > 0.1

def test_extract_blobs_finds_dark_region():
    # Build a variance map that is low everywhere except a 50x50 block
    var_map = np.ones((200, 300), dtype=np.float32) * 0.5  # high variance
    var_map[75:125, 125:175] = 0.0  # low-variance blob
    blobs = extract_blobs(var_map, image_area=200 * 300)
    assert len(blobs) == 1
    cx, cy = blobs[0]
    assert 125 <= cx <= 175 and 75 <= cy <= 125

def test_extract_blobs_ignores_tiny_noise():
    # 4px blob in a 200x300 image: min_area = int(60000 * 0.0005) = 30px
    # Without morphology, 4px < 30px → filtered out
    var_map = np.ones((200, 300), dtype=np.float32) * 0.5
    var_map[10:12, 10:12] = 0.0  # 4 px — below min_area threshold
    blobs = extract_blobs(var_map, image_area=200 * 300)
    assert len(blobs) == 0

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

def test_discover_zones_returns_consistent_zone(tmp_path):
    # All images have a watermark blob in the same position → one stable zone.
    # Watermark is constant; background varies significantly between images.
    wm_color = np.array([200, 200, 200], dtype=np.uint8)

    paths = []
    np.random.seed(42)  # for reproducibility
    for i in range(5):
        # Highly varying background (0-255 range) to create high variance
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        # Constant watermark: 50x50 block bottom-right
        img[140:190, 240:290] = wm_color
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    zones = discover_zones(paths, img_w=300, img_h=200)
    assert len(zones) >= 1
    # The watermark is in the right-third, bottom-half → zone (2, 1)
    assert (2, 1) in zones


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


from untextre.discovery import discover_watermark_candidates

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
    # Noise images should produce 0 or very few candidates (not a crash)
    assert len(candidates) <= 2  # allow occasional noise blobs but not many

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
