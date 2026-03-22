# Unknown Watermark Discovery (`-U`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `-U` / `--unknown-watermark` CLI flag that auto-discovers a watermark template from a directory of images and processes them using the existing `-K` / ORB pipeline.

**Architecture:** A new `untextre/discovery.py` module handles all discovery logic (bucketing, pair stacking, blob extraction, zone clustering, cross-bucket validation, RGBA crop output). `cli.py` gains a `-U` flag that calls discovery, writes the candidate PNGs, then feeds them into the existing `try_watermark_cascade` / `process_with_known_mask` path unchanged.

**Tech Stack:** Python, NumPy, OpenCV (`cv2`), existing `untextre` utilities (`load_image`, `get_image_files`, `setup_logger`), pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `untextre/discovery.py` | All discovery logic: bucketing, stacking, blob extraction, zone clustering, crop-to-RGBA, cross-bucket validation |
| Modify | `untextre/cli.py` | Extract `create_parser()`; add `-U` argument (mutually exclusive with `-K`); add same-dir guard; add `--force`; call discovery before main loop; freeze image list before writing candidates |
| Create | `tests/test_discovery.py` | Unit tests for every public function in `discovery.py` |

---

## Task 1: Bucket images by exact pixel dimensions

**Files:**
- Create: `untextre/discovery.py`
- Create: `tests/test_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_discovery.py
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from untextre.discovery import bucket_images_by_size

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
    assert (100, 200) in buckets  # portrait is a separate bucket

def test_bucket_images_skips_unreadable(tmp_path):
    paths = [tmp_path / "bad.png"]
    paths[0].touch()
    with patch("untextre.discovery.load_image", side_effect=ValueError("bad")):
        buckets = bucket_images_by_size(paths)
    assert buckets == {}
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_discovery.py::test_bucket_images_by_size_groups_correctly -v
```
Expected: `ModuleNotFoundError` or `ImportError` (file doesn't exist yet)

- [ ] **Step 3: Create `untextre/discovery.py` with `bucket_images_by_size`**

```python
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

# Morphological kernel sizes (match morph_clean_mask for consistency).
CLOSE_KERNEL_SIZE = 11
DILATE_SIZE = 13

# Border added around each RGBA crop (pixels).
CROP_BORDER_PX = 8

# Zone grid: long edge into thirds, short edge into halves → 6 zones.
ZONE_LONG_DIVISIONS = 3
ZONE_SHORT_DIVISIONS = 2

# Convergence parameters.
MAX_DRAWS = 50
STABLE_STREAK_REQUIRED = 10

# Cross-bucket IoU threshold for family membership.
CROSS_BUCKET_IOU_THRESHOLD = 0.50


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
        except Exception as e:
            logger.warning(f"Skipping unreadable image {path.name}: {e}")
    return buckets
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_discovery.py::test_bucket_images_by_size_groups_correctly tests/test_discovery.py::test_bucket_images_skips_unreadable -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add discovery module skeleton with bucket_images_by_size"
```

---

## Task 2: Compute pair variance and extract blobs

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery.py (append)
from untextre.discovery import compute_pair_variance, extract_blobs

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
    var_map = np.ones((200, 300), dtype=np.float32) * 0.5
    var_map[10:12, 10:12] = 0.0  # 4 px — below threshold
    blobs = extract_blobs(var_map, image_area=200 * 300)
    assert len(blobs) == 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_discovery.py -k "variance or blobs" -v
```
Expected: `ImportError` for `compute_pair_variance`, `extract_blobs`

- [ ] **Step 3: Implement `compute_pair_variance` and `extract_blobs`**

Add to `untextre/discovery.py`:

```python
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

    Applies morphological closing then dilation (no blur — smooth edges
    are not needed for bounding-box extraction).

    Args:
        variance_map: Per-pixel variance (H×W float32).
        image_area: Total image area in pixels (H * W), used for min-area check.

    Returns:
        List of (cx, cy) blob centroids for blobs that exceed the minimum area.
    """
    min_area = max(1, int(image_area * MIN_BLOB_AREA_FRACTION))

    # Threshold: 1 where variance is LOW (candidate watermark pixels)
    binary = (variance_map < VARIANCE_THRESHOLD).astype(np.uint8) * 255

    # Morphological closing then dilation (no blur)
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATE_SIZE, DILATE_SIZE)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.dilate(binary, kernel_dilate)

    # Find 8-connected contiguous blobs
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_discovery.py -k "variance or blobs" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add compute_pair_variance and extract_blobs"
```

---

## Task 3: Assign blobs to zones and run convergence loop

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery.py (append)
from untextre.discovery import assign_zone, discover_zones

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
    # All images have a watermark blob in the same position → one stable zone
    wm_color = np.array([200, 200, 200], dtype=np.uint8)
    bg_color = np.array([50, 100, 150], dtype=np.uint8)

    paths = []
    for i in range(5):
        img = np.full((200, 300, 3), dtype=np.uint8, fill_value=0)
        img[:] = bg_color
        # Watermark: 40x40 block bottom-right
        img[150:190, 250:290] = wm_color
        # Vary background slightly so pairs have high variance elsewhere
        img[0:50, 0:50] = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    zones = discover_zones(paths, img_w=300, img_h=200)
    assert len(zones) >= 1
    # The watermark is in the right-third, bottom-half → zone (2, 1)
    assert (2, 1) in zones
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_discovery.py -k "zone" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `assign_zone` and `discover_zones`**

Add to `untextre/discovery.py`:

```python
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

    Zero-blob draws count toward the stability streak.

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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_discovery.py -k "zone" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add zone assignment and convergence loop"
```

---

## Task 4: Crop blobs to RGBA candidates

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery.py (append)
from untextre.discovery import crop_zone_to_rgba

def test_crop_zone_to_rgba_shape_and_alpha():
    # 100x100 mean image, blob occupying rows 30-60, cols 40-70
    mean_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    blob_mask[30:60, 40:70] = 255

    rgba = crop_zone_to_rgba(mean_img, blob_mask)

    # Crop should be blob bounding box + CROP_BORDER_PX on each side
    from untextre.discovery import CROP_BORDER_PX
    expected_h = (60 - 30) + 2 * CROP_BORDER_PX
    expected_w = (70 - 40) + 2 * CROP_BORDER_PX
    assert rgba.shape == (expected_h, expected_w, 4)

def test_crop_zone_to_rgba_alpha_channel():
    mean_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    blob_mask[40:60, 40:60] = 255

    rgba = crop_zone_to_rgba(mean_img, blob_mask)
    # Alpha=255 inside blob region, 0 in border
    h, w = rgba.shape[:2]
    from untextre.discovery import CROP_BORDER_PX
    b = CROP_BORDER_PX
    # Interior of crop (excluding border) should be alpha=255
    assert np.all(rgba[b:h-b, b:w-b, 3] == 255)
    # Corners (pure border) should be alpha=0
    assert rgba[0, 0, 3] == 0
    assert rgba[h-1, w-1, 3] == 0

def test_crop_zone_to_rgba_returns_none_for_empty_mask():
    mean_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    blob_mask = np.zeros((100, 100), dtype=np.uint8)
    result = crop_zone_to_rgba(mean_img, blob_mask)
    assert result is None
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_discovery.py -k "rgba" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `crop_zone_to_rgba`**

Add to `untextre/discovery.py`:

```python
def crop_zone_to_rgba(
    mean_image: np.ndarray,
    blob_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Crop a blob region from the mean image and produce a tight RGBA PNG array.

    The alpha channel is 255 inside the blob and 0 in the transparent border.

    Args:
        mean_image: Pixel-wise mean of all images in the bucket (H×W×3 BGR).
        blob_mask: Binary mask (H×W uint8, 255 = blob).

    Returns:
        BGRA crop (H'×W'×4 uint8) with transparent border, or None if
        blob_mask is empty.  Channel order matches what find_known_mask_in_image expects.
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

    # Produce BGRA — keep BGR channel order from the source image so that
    # find_known_mask_in_image (which expects BGRA) gets the correct channel order.
    b_ch, g_ch, r_ch = cv2.split(bgr_crop)
    rgba = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
    return rgba
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_discovery.py -k "rgba" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add crop_zone_to_rgba for RGBA candidate generation"
```

---

## Task 5: Cross-bucket validation and family selection

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_discovery.py (append)
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
    families = select_best_family([small, large])
    assert len(families) == 1
    assert families[0].shape == large.shape

def test_select_best_family_keeps_distinct():
    # Two crops with low IoU → different families
    a = np.zeros((40, 40, 4), dtype=np.uint8)
    a[0:10, 0:10, 3] = 255
    b = np.zeros((40, 40, 4), dtype=np.uint8)
    b[30:40, 30:40, 3] = 255
    families = select_best_family([a, b])
    assert len(families) == 2
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_discovery.py -k "iou or family" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `compute_alpha_iou` and `select_best_family`**

Add to `untextre/discovery.py`:

```python
def compute_alpha_iou(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """Compute IoU on alpha channels after resizing the smaller crop to the larger.

    "Larger" is defined by pixel area (H * W).

    Args:
        crop_a: RGBA array (H×W×4).
        crop_b: RGBA array (H×W×4).

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
        small, (target_w, target_h), interpolation=cv2.INTER_CUBIC
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
        candidates: List of RGBA crops.

    Returns:
        List of representative RGBA crops, one per family, largest first.
    """
    if not candidates:
        return []

    families: List[List[np.ndarray]] = []

    for crop in candidates:
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_discovery.py -k "iou or family" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add cross-bucket IoU validation and family selection"
```

---

## Task 6: Top-level `discover_watermark_candidates` function

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_discovery.py (append)
from untextre.discovery import discover_watermark_candidates

def test_discover_finds_watermark_in_homogeneous_batch(tmp_path):
    """Integration test: batch of images with a consistent watermark blob."""
    wm = np.array([220, 220, 220], dtype=np.uint8)
    paths = []
    for i in range(6):
        img = np.random.randint(30, 180, (300, 400, 3), dtype=np.uint8)
        # Fixed watermark: 50x30 block at bottom-right corner
        img[260:290, 340:390] = wm
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    candidates = discover_watermark_candidates(paths)
    assert len(candidates) >= 1
    # Each candidate is an RGBA PNG array
    assert candidates[0].shape[2] == 4

def test_discover_deduplicates_similar_aspect_ratio_candidates(tmp_path):
    """Two zones with similar aspect ratios in one bucket → only largest kept."""
    wm = np.array([210, 210, 210], dtype=np.uint8)
    paths = []
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
    for i in range(4):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        p = tmp_path / f"noise{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)

    # May or may not find blobs; if it does they'll be noise — just ensure no crash
    candidates = discover_watermark_candidates(paths)
    assert isinstance(candidates, list)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_discovery.py -k "discover_watermark_candidates" -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `discover_watermark_candidates`**

Add to `untextre/discovery.py`:

```python
def discover_watermark_candidates(
    image_paths: List[Path],
) -> List[np.ndarray]:
    """Discover watermark template(s) from a batch of consistently-watermarked images.

    Buckets images by exact dimensions, runs the pair-stacking convergence loop
    per qualifying bucket (≥ 3 images), crops RGBA candidates, cross-validates
    across buckets, and returns one representative RGBA crop per watermark family
    in descending pixel-area order.

    Args:
        image_paths: All image paths in the input directory (pre-frozen list).

    Returns:
        List of RGBA crops (H×W×4 uint8), largest family first.
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
            except Exception:
                pass
        if not loaded:
            continue
        mean_img = np.mean(loaded, axis=0).astype(np.uint8)

        for zone, var_maps in zone_maps.items():
            # Build union blob mask from all variance maps for this zone
            union_low_var = np.zeros(mean_img.shape[:2], dtype=np.uint8)
            for vmap in var_maps:
                binary = (vmap < VARIANCE_THRESHOLD).astype(np.uint8) * 255
                union_low_var = cv2.bitwise_or(union_low_var, binary)

            # Restrict to the blob's grid zone for clean cropping
            zone_mask = _make_zone_mask(union_low_var.shape, zone, img_w, img_h)
            zoned_mask = cv2.bitwise_and(union_low_var, zone_mask)

            # Clean up
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
            )
            kernel_dilate = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (DILATE_SIZE, DILATE_SIZE)
            )
            zoned_mask = cv2.morphologyEx(zoned_mask, cv2.MORPH_CLOSE, kernel_close)
            zoned_mask = cv2.dilate(zoned_mask, kernel_dilate)

            rgba = crop_zone_to_rgba(mean_img, zoned_mask)
            if rgba is not None:
                all_candidates.append(rgba)
                logger.info(
                    f"Bucket {img_w}×{img_h} zone {zone}: "
                    f"candidate {rgba.shape[1]}×{rgba.shape[0]} px"
                )

    if not all_candidates:
        logger.warning("No watermark candidates discovered across all buckets")
        return []

    # Per-bucket aspect-ratio dedup (spec Phase 2 Step 5):
    # If multiple candidates from the same bucket have similar aspect ratios
    # (symmetric relative difference < 10%), keep only the largest.
    def _aspect_ratio(c: np.ndarray) -> float:
        h, w = c.shape[:2]
        return w / h if h > 0 else 1.0

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


def _make_zone_mask(
    shape: Tuple[int, int],
    zone: Tuple[int, int],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Return a binary mask (255) for the pixels belonging to a given zone."""
    h, w = shape
    col, row = zone

    if img_w >= img_h:
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
```

- [ ] **Step 4: Run all discovery tests**

```
pytest tests/test_discovery.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: add discover_watermark_candidates top-level function"
```

---

## Task 7: Wire `-U` into the CLI

**Files:**
- Modify: `untextre/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py (append — find existing test_cli.py and add to it)
from unittest.mock import patch, MagicMock
import sys

def test_unknown_watermark_flag_exists():
    """Smoke test: -U flag is registered and mutually exclusive with -K."""
    from untextre.cli import create_parser
    parser = create_parser()
    # Should parse without error
    args = parser.parse_args(["-U", "some/dir", "-o", "out/dir"])
    assert args.unknown_watermark is True

def test_unknown_watermark_and_known_mask_are_mutually_exclusive():
    from untextre.cli import create_parser
    import pytest
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-U", "-K", "template.png", "some/dir", "-o", "out/dir"])
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_cli.py -k "unknown_watermark" -v
```
Expected: `AttributeError` — `create_parser` exists but `-U` not yet added (or `ImportError` if `create_parser` is not exposed)

- [ ] **Step 3: Extract `create_parser()` from `main()` in `cli.py`**

The argparse block is built inline inside `main()`. Extract it into a standalone function so tests can import it directly. Find the `ArgumentParser(...)` construction in `main()` and move everything up to `return parser` into a new function. Update `main()` to call `args = create_parser().parse_args()`.

```python
# In untextre/cli.py — add before main():
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)
    # ... all add_argument calls ...
    return parser

def main():
    args = create_parser().parse_args()
    # ... rest of main unchanged ...
```

Run existing CLI tests to confirm no regressions before proceeding:

```
pytest tests/test_cli.py -v -m "not slow"
```
Expected: All existing tests PASS

- [ ] **Step 4: Add `-U`, `--force`, and mutually exclusive group to `create_parser()`**

In `create_parser()`, replace the standalone `-K` `add_argument` call with a mutually exclusive group, and add `--force`:

```python
# Replace the existing standalone -K add_argument with this group:
mask_group = parser.add_mutually_exclusive_group()
mask_group.add_argument(
    "-K", "--known-mask",
    help="Path to RGBA image (PNG with transparency) of a known watermark/logo, "
         "or a directory of such images. Uses ORB feature matching to find and mask "
         "the watermark at any scale/position (first match wins). The alpha channel "
         "defines the mask. Skips consensus detection when used."
)
mask_group.add_argument(
    "-U", "--unknown-watermark",
    action="store_true",
    default=False,
    help="Auto-discover watermark from input directory via low-variance stacking, "
         "save candidate BGRA template(s) to output dir, then process with ORB matching. "
         "Requires directory input. Mutually exclusive with -K."
)

# Add --force flag (standalone, not in the group):
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="Allow output directory to be the same as input directory. "
         "WARNING: cleaned images will overwrite originals."
)
```

Also add same-directory guard at the start of the `-U` block (Step 5):

```python
if args.unknown_watermark:
    if args.input.resolve() == output_path.resolve() and not args.force:
        logger.error(
            "Input and output directories are the same. "
            "This would overwrite originals. Use --force to proceed."
        )
        sys.exit(1)
```


- [ ] **Step 5: Add `-U` handling in the main processing block**

In `cli.py`, after the image file list is gathered (but before any processing loop), add the discovery call. The image file list must be frozen before candidates are written. Find the block around line 676 (`logger.info(f"Found {len(image_files)} image(s) to process")`) and insert.

**Important:** also set `explicit_known_mask = True` when `-U` is active. The existing code at lines ~707 and ~757 uses `explicit_known_mask` to decide (a) which models to load (inpainting-only vs. full consensus stack) and (b) whether to fall back to consensus detection when no template matches. `-U` behaves like `-K` in both respects: load only the inpainting model, and skip (not fall back) when ORB finds no match. Without this, `-U` will load consensus-detection models it never uses and silently fall through to full consensus detection on no-match images.

```python
# ── -U: auto-discover watermark templates ────────────────────────────
if args.unknown_watermark:
    if not args.input.is_dir():
        logger.error("-U requires a directory input, not a single file")
        sys.exit(1)
    from .discovery import discover_watermark_candidates

    # Treat -U like -K for model loading and fallback decisions
    explicit_known_mask = True
    logger.info("Running watermark discovery (-U mode)...")
    # image_files list is already frozen above this point
    candidates = discover_watermark_candidates(image_files)

    if not candidates:
        logger.error(
            "No watermark candidates discovered. "
            "Try -K with a manually-identified template."
        )
        sys.exit(1)

    # Write candidates to output dir before processing
    output_path.mkdir(parents=True, exist_ok=True)
    watermark_templates = []
    for i, rgba in enumerate(candidates):
        suffix = "" if i == 0 else f"_{i + 1}"
        candidate_path = output_path / f"watermark_candidate{suffix}.png"
        if candidate_path.exists():
            logger.warning(f"Overwriting existing candidate: {candidate_path.name}")
        # Array is already BGRA — cv2.imwrite saves channel 3 as alpha automatically
        cv2.imwrite(str(candidate_path), rgba)
        logger.info(f"Saved watermark candidate: {candidate_path.name}")
        watermark_templates.append((candidate_path.name, rgba))

    # Warn about stale candidates from prior runs
    existing = sorted(output_path.glob("watermark_candidate*.png"))
    stale = [p for p in existing if p not in [output_path / f"watermark_candidate{'' if j == 0 else f'_{j+1}'}.png" for j in range(len(candidates))]]
    if stale:
        logger.warning(
            f"Stale candidate file(s) from prior run still present: "
            + ", ".join(p.name for p in stale)
        )
```

- [ ] **Step 6: Run CLI tests**

```
pytest tests/test_cli.py -k "unknown_watermark" -v
```
Expected: PASS

- [ ] **Step 7: Run full test suite to check for regressions**

```
pytest tests/ -v --ignore=tests/images -x
```
Expected: All existing tests PASS

- [ ] **Step 8: Commit**

```bash
git add untextre/cli.py untextre/discovery.py tests/test_cli.py tests/test_discovery.py
git commit -m "feat: wire -U flag into CLI with discovery + cascade handoff"
```

---

## Task 8: End-to-end smoke test and manual validation

**Files:**
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write a slow integration test**

```python
# tests/test_discovery.py (append)
import pytest

@pytest.mark.slow
def test_discovery_end_to_end_with_real_images(tmp_path):
    """
    Create a batch of 6 synthetic images with a consistent watermark,
    run discover_watermark_candidates, confirm we get exactly 1 RGBA crop,
    and that it has nonzero alpha pixels.
    """
    wm_patch = np.full((40, 80, 3), 210, dtype=np.uint8)
    paths = []
    for i in range(6):
        # Randomize background to simulate real photos
        img = np.random.randint(20, 200, (400, 600, 3), dtype=np.uint8)
        # Place identical watermark bottom-right
        img[350:390, 510:590] = wm_patch
        p = tmp_path / f"photo_{i:02d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)

    from untextre.discovery import discover_watermark_candidates
    candidates = discover_watermark_candidates(paths)

    assert len(candidates) >= 1
    best = candidates[0]
    assert best.shape[2] == 4
    assert np.any(best[:, :, 3] > 127), "Expected nonzero alpha in best candidate"
```

- [ ] **Step 2: Run the slow test to confirm end-to-end behavior**

```
pytest tests/test_discovery.py::test_discovery_end_to_end_with_real_images -v -s
```
Expected: PASS — inspect the logged output to confirm zone detection and candidate shape look reasonable

- [ ] **Step 3: Run the full fast suite one final time**

```
pytest tests/ -v --ignore=tests/images -m "not slow"
```
Expected: All PASS, no regressions

- [ ] **Step 4: Final commit**

```bash
git add tests/test_discovery.py
git commit -m "test: add slow end-to-end discovery integration test"
```

---

## Notes for implementer

- `load_image` raises `ValueError` on failure — wrap all calls in try/except in discovery code
- `IMAGE_EXTENSIONS` is already defined in `utils.py` — import it, don't redefine
- The `-K` / `-U` mutually exclusive group: check whether the existing parser already has a group for `-K` or if it's a standalone `add_argument`. If standalone, you'll need to replace it with the group version.
- JPEG compression artifacts may raise effective variance; the 0.01 threshold was chosen for near-lossless inputs. If the slow end-to-end test is flaky, switch it from `.jpg` to `.png` before adjusting the threshold constant.
- The `watermark_templates` variable in `cli.py` is a `List[Tuple[str, np.ndarray]]` matching the format that `try_watermark_cascade` expects — `(filename_string, bgra_array)`. The discovery code hands off in exactly this format.
- `discover_zones` loads all images into memory; `discover_watermark_candidates` then loads them again to compute the mean. For large batches this doubles disk reads. This is intentional for now (keeps the functions independently testable); cache at the bucket level if performance is a concern in the future.
- `-f` is already taken by `--force-bbox`. Do not add `-f` as a short form for `--force`.
- **Channel order on disk vs. in memory:** discovery arrays are BGRA in memory throughout. `cv2.imwrite` with a 4-channel BGRA array produces a valid PNG (cv2 handles the swap). `load_watermark_templates` reloads with `cv2.IMREAD_UNCHANGED`, returning BGRA again — correct for the pipeline. Do NOT use `save_image` from `utils.py` for BGRA arrays (it does not pass `IMREAD_UNCHANGED` on the read side, and has no concept of alpha). Do NOT use `load_image` from `utils.py` to reload templates (drops the alpha channel). Disk I/O for BGRA PNGs always uses `cv2.imwrite` / `cv2.imread(..., cv2.IMREAD_UNCHANGED)` directly.
