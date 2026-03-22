import numpy as np
import pytest
import logging
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
