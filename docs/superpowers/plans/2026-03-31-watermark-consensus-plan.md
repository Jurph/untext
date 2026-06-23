Using the writing-plans skill format for this implementation plan. The skill calls for a fixed header, a file-structure section before tasks, checkbox-tracked bite-sized tasks, and a default save path under `docs/superpowers/plans/...`. ([GitHub][1])

# Watermark Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bolt-on consensus stage that takes multiple noisy BGRA watermark candidates and returns one tighter, cleaner BGRA watermark crop without assuming the watermark is text.

**Architecture:** Keep the existing discovery path intact. Add a new sibling module that operates only on already-extracted BGRA candidates: prepare each candidate, align them into one canvas, estimate a shared soft alpha via iterative reweighted consensus, estimate RGB from the aligned candidates, then crop by support mass instead of blob heuristics. Expose the new behavior through an additive wrapper in `discovery.py` so callers can opt in without breaking the old function.

**Tech Stack:** Python, NumPy, OpenCV, pytest

---

## File Structure

* Create: `watermark_consensus.py`

  * Single responsibility: take `list[np.ndarray]` BGRA candidates and return one consensus BGRA.
  * Contains candidate preparation, alignment, alpha consensus, RGB consensus, support-mass crop, and debug image writing.
* Modify: `discovery.py`

  * Single responsibility change: add one new opt-in wrapper that calls the existing discovery routine, then runs consensus on its BGRA outputs.
* Create: `tests/test_watermark_consensus.py`

  * Single responsibility: deterministic synthetic tests for the new consensus module only.
* Create: `tests/test_discovery_consensus_integration.py`

  * Single responsibility: verify the new wrapper in `discovery.py` calls the consensus stage correctly and preserves fallback behavior.

This keeps the new logic isolated, testable, and removable. It also avoids refactoring the existing discovery algorithm while the consensus stage is still stabilizing.

---

### Task 1: Create the additive consensus module scaffold

**Files:**

* Create: `watermark_consensus.py`

* Create: `tests/test_watermark_consensus.py`

* [ ] **Step 1: Write the failing tests**

Add this file exactly as shown:

```python
# tests/test_watermark_consensus.py
import numpy as np

from watermark_consensus import build_consensus_watermark


def make_single_blob_bgra(height: int = 32, width: int = 48) -> np.ndarray:
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    bgra[10:22, 14:38, :3] = (120, 160, 200)
    bgra[10:22, 14:38, 3] = 255
    return bgra


def test_build_consensus_watermark_returns_none_for_empty_input() -> None:
    assert build_consensus_watermark([]) is None


def test_build_consensus_watermark_returns_tight_single_candidate_crop() -> None:
    bgra = make_single_blob_bgra()
    result = build_consensus_watermark([bgra])

    assert result is not None
    assert result.shape == (12, 24, 4)
    assert np.all(result[:, :, 3] == 255)
    assert tuple(result[0, 0, :3]) == (120, 160, 200)
```

* [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_watermark_consensus.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'watermark_consensus'`.

* [ ] **Step 3: Write the minimal implementation**

Create this file exactly as shown:

```python
# watermark_consensus.py
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def alpha_to_soft_mask(alpha_u8: np.ndarray) -> np.ndarray:
    if alpha_u8.ndim != 2:
        raise ValueError("alpha_to_soft_mask expects an HxW alpha channel")
    return alpha_u8.astype(np.float32) / 255.0


def signed_distance_from_alpha(alpha: np.ndarray) -> np.ndarray:
    fg = (alpha > 0.0).astype(np.uint8)
    bg = 1 - fg
    dist_fg = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    return dist_fg - dist_bg


def support_mass_bbox(alpha: np.ndarray, coverage: float = 0.995) -> tuple[int, int, int, int]:
    row_mass = alpha.sum(axis=1)
    col_mass = alpha.sum(axis=0)

    row_total = float(row_mass.sum())
    col_total = float(col_mass.sum())

    if row_total <= 1e-8 or col_total <= 1e-8:
        h, w = alpha.shape
        return (0, 0, w, h)

    low = (1.0 - coverage) / 2.0
    high = 1.0 - low

    row_cum = np.cumsum(row_mass)
    col_cum = np.cumsum(col_mass)

    y0 = int(np.searchsorted(row_cum, low * row_total))
    y1 = int(np.searchsorted(row_cum, high * row_total)) + 1
    x0 = int(np.searchsorted(col_cum, low * col_total))
    x1 = int(np.searchsorted(col_cum, high * col_total)) + 1

    return (x0, y0, x1, y1)


def crop_bgra(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    rgb_crop = rgb[y0:y1, x0:x1]
    alpha_crop = alpha_u8[y0:y1, x0:x1]
    return np.dstack(
        [
            np.clip(rgb_crop[:, :, 0] * 255.0, 0, 255).astype(np.uint8),
            np.clip(rgb_crop[:, :, 1] * 255.0, 0, 255).astype(np.uint8),
            np.clip(rgb_crop[:, :, 2] * 255.0, 0, 255).astype(np.uint8),
            alpha_crop,
        ]
    )


def build_consensus_watermark(
    candidates_bgra: list[np.ndarray],
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    del debug_dir  # reserved for later tasks

    if not candidates_bgra:
        return None

    if len(candidates_bgra) == 1:
        bgra = candidates_bgra[0]
        rgb = bgra[:, :, :3].astype(np.float32) / 255.0
        alpha = alpha_to_soft_mask(bgra[:, :, 3])
        bbox = support_mass_bbox(alpha)
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        return crop_bgra(rgb, alpha_u8, bbox)

    raise NotImplementedError("Consensus path is added in later tasks")
```

* [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_watermark_consensus.py -q
```

Expected: `2 passed`.

* [ ] **Step 5: Commit**

```bash
git add watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: scaffold additive watermark consensus module"
```

---

### Task 2: Add alignment on a common canvas using signed-distance matching

**Files:**

* Modify: `watermark_consensus.py`

* Modify: `tests/test_watermark_consensus.py`

* [ ] **Step 1: Write the failing test**

Append this test code to `tests/test_watermark_consensus.py`:

```python
import cv2

from watermark_consensus import align_candidates_to_reference


def make_rect_mask(height: int = 96, width: int = 160) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[28:68, 42:118] = 255
    return mask


def render_bgra_from_mask(mask: np.ndarray, color: tuple[int, int, int] = (180, 180, 180)) -> np.ndarray:
    bgra = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    bgra[mask > 0, :3] = color
    bgra[:, :, 3] = mask
    return bgra


def scale_and_shift_mask(mask: np.ndarray, scale: float, dx: int, dy: int) -> np.ndarray:
    resized = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    out = np.zeros_like(mask)

    h, w = resized.shape
    y0 = max(0, dy)
    x0 = max(0, dx)
    y1 = min(out.shape[0], dy + h)
    x1 = min(out.shape[1], dx + w)

    sy0 = max(0, -dy)
    sx0 = max(0, -dx)
    sy1 = sy0 + (y1 - y0)
    sx1 = sx0 + (x1 - x0)

    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = resized[sy0:sy1, sx0:sx1]

    return out


def alpha_iou(a: np.ndarray, b: np.ndarray) -> float:
    am = a > 0.2
    bm = b > 0.2
    inter = np.logical_and(am, bm).sum()
    union = np.logical_or(am, bm).sum()
    return float(inter) / float(union)


def test_align_candidates_to_reference_recovers_shift_and_scale() -> None:
    ref = render_bgra_from_mask(make_rect_mask())
    moved = render_bgra_from_mask(scale_and_shift_mask(make_rect_mask(), scale=0.92, dx=9, dy=-6))

    aligned_rgbs, aligned_alphas, weights = align_candidates_to_reference([ref, moved])

    assert len(aligned_rgbs) == 2
    assert len(aligned_alphas) == 2
    assert weights.shape == (2,)
    assert alpha_iou(aligned_alphas[0], aligned_alphas[1]) > 0.80
```

* [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_align_candidates_to_reference_recovers_shift_and_scale -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `align_candidates_to_reference`.

* [ ] **Step 3: Write minimal implementation**

Append these functions to `watermark_consensus.py` above `build_consensus_watermark`:

```python
def _resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _place_on_canvas(img: np.ndarray, canvas_shape: tuple[int, int], x: int, y: int) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    if img.ndim == 3:
        out = np.zeros((canvas_h, canvas_w, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((canvas_h, canvas_w), dtype=img.dtype)

    h, w = img.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas_w, x + w)
    y1 = min(canvas_h, y + h)

    sx0 = max(0, -x)
    sy0 = max(0, -y)
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]

    return out


def _alpha_mass(bgra: np.ndarray) -> int:
    return int((bgra[:, :, 3] > 0).sum())


def align_candidates_to_reference(
    candidates_bgra: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if not candidates_bgra:
        return [], [], np.zeros((0,), dtype=np.float32)

    reference_index = int(np.argmax([_alpha_mass(c) for c in candidates_bgra]))
    reference = candidates_bgra[reference_index]
    ref_alpha = alpha_to_soft_mask(reference[:, :, 3])
    ref_sdt = signed_distance_from_alpha(ref_alpha)

    max_h = max(candidate.shape[0] for candidate in candidates_bgra)
    max_w = max(candidate.shape[1] for candidate in candidates_bgra)
    canvas_shape = (max_h * 3, max_w * 3)
    anchor_x = max_w
    anchor_y = max_h

    ref_sdt_canvas = _place_on_canvas(ref_sdt.astype(np.float32), canvas_shape, anchor_x, anchor_y)

    aligned_rgbs: list[np.ndarray] = []
    aligned_alphas: list[np.ndarray] = []

    for candidate in candidates_bgra:
        rgb = candidate[:, :, :3].astype(np.float32) / 255.0
        alpha = alpha_to_soft_mask(candidate[:, :, 3])
        sdt = signed_distance_from_alpha(alpha)

        best_score = None
        best_rgb = None
        best_alpha = None

        for scale in np.linspace(0.85, 1.15, 13):
            scaled_rgb = _resize_image(rgb, float(scale))
            scaled_alpha = _resize_image(alpha, float(scale))
            scaled_sdt = _resize_image(sdt, float(scale)).astype(np.float32)

            sdt_canvas = _place_on_canvas(scaled_sdt, canvas_shape, anchor_x, anchor_y)
            (shift_x, shift_y), response = cv2.phaseCorrelate(ref_sdt_canvas, sdt_canvas)

            dx = anchor_x + int(round(shift_x))
            dy = anchor_y + int(round(shift_y))

            placed_rgb = _place_on_canvas(scaled_rgb, canvas_shape, dx, dy)
            placed_alpha = _place_on_canvas(scaled_alpha, canvas_shape, dx, dy)

            score = float(response)
            if best_score is None or score > best_score:
                best_score = score
                best_rgb = placed_rgb
                best_alpha = placed_alpha

        aligned_rgbs.append(best_rgb)
        aligned_alphas.append(best_alpha)

    weights = np.ones((len(aligned_rgbs),), dtype=np.float32) / len(aligned_rgbs)
    return aligned_rgbs, aligned_alphas, weights
```

Also replace the body of `build_consensus_watermark` with this version so multi-candidate input no longer raises:

```python
def build_consensus_watermark(
    candidates_bgra: list[np.ndarray],
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    del debug_dir  # reserved for later tasks

    if not candidates_bgra:
        return None

    if len(candidates_bgra) == 1:
        bgra = candidates_bgra[0]
        rgb = bgra[:, :, :3].astype(np.float32) / 255.0
        alpha = alpha_to_soft_mask(bgra[:, :, 3])
        bbox = support_mass_bbox(alpha)
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        return crop_bgra(rgb, alpha_u8, bbox)

    aligned_rgbs, aligned_alphas, _ = align_candidates_to_reference(candidates_bgra)
    mean_rgb = np.mean(np.stack(aligned_rgbs, axis=0), axis=0)
    mean_alpha = np.mean(np.stack(aligned_alphas, axis=0), axis=0)
    bbox = support_mass_bbox(mean_alpha)
    alpha_u8 = np.clip(mean_alpha * 255.0, 0, 255).astype(np.uint8)
    return crop_bgra(mean_rgb, alpha_u8, bbox)
```

* [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_align_candidates_to_reference_recovers_shift_and_scale -q
```

Expected: `1 passed`.

* [ ] **Step 5: Commit**

```bash
git add watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: align watermark candidates on common canvas"
```

---

### Task 3: Replace naive averaging with iterative reweighted alpha consensus

**Files:**

* Modify: `watermark_consensus.py`

* Modify: `tests/test_watermark_consensus.py`

* [ ] **Step 1: Write the failing test**

Append this test code to `tests/test_watermark_consensus.py`:

```python
from watermark_consensus import estimate_consensus_alpha


def make_irregular_truth_mask(height: int = 120, width: int = 220) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[35:85, 45:95] = 255
    mask[48:72, 95:170] = 255
    return mask


def make_partial(mask: np.ndarray, side: str) -> np.ndarray:
    out = np.zeros_like(mask)
    if side == "left":
        out[:, : mask.shape[1] // 2] = mask[:, : mask.shape[1] // 2]
    elif side == "right":
        out[:, mask.shape[1] // 2 :] = mask[:, mask.shape[1] // 2 :]
    else:
        raise ValueError("side must be 'left' or 'right'")
    return out


def add_blob(mask: np.ndarray, top: int, left: int, size: int = 12) -> np.ndarray:
    out = mask.copy()
    out[top : top + size, left : left + size] = 255
    return out


def test_estimate_consensus_alpha_keeps_shared_truth_and_suppresses_one_off_blob() -> None:
    truth = make_irregular_truth_mask()
    candidate_a = make_partial(truth, "left").astype(np.float32) / 255.0
    candidate_b = make_partial(truth, "right").astype(np.float32) / 255.0
    candidate_c = add_blob(truth, top=8, left=185, size=14).astype(np.float32) / 255.0

    consensus, weights = estimate_consensus_alpha([candidate_a, candidate_b, candidate_c], max_iters=12)

    truth_support = float(consensus[truth > 0].mean())
    blob_support = float(consensus[8:22, 185:199].mean())

    assert weights.shape == (3,)
    assert truth_support > 0.45
    assert blob_support < 0.25
```

* [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_estimate_consensus_alpha_keeps_shared_truth_and_suppresses_one_off_blob -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `estimate_consensus_alpha`.

* [ ] **Step 3: Write minimal implementation**

Append these functions to `watermark_consensus.py` above `build_consensus_watermark`:

```python
def _robust_global_error(alpha_i: np.ndarray, alpha_ref: np.ndarray) -> float:
    diff = alpha_i - alpha_ref
    abs_diff = np.abs(diff)
    delta = 0.10
    quad = np.minimum(abs_diff, delta)
    lin = abs_diff - quad
    return float((0.5 * quad**2 + delta * lin).mean())


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 1e-8:
        return np.ones_like(weights) / len(weights)
    return weights / total


def estimate_consensus_alpha(
    aligned_alphas: list[np.ndarray],
    max_iters: int = 10,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if not aligned_alphas:
        raise ValueError("estimate_consensus_alpha requires at least one aligned alpha map")

    weights = np.ones((len(aligned_alphas),), dtype=np.float32) / len(aligned_alphas)
    consensus = np.average(np.stack(aligned_alphas, axis=0), axis=0, weights=weights)

    for _ in range(max_iters):
        errors = np.array(
            [_robust_global_error(alpha_i, consensus) for alpha_i in aligned_alphas],
            dtype=np.float32,
        )
        new_weights = _normalize_weights(1.0 / (errors + eps))
        updated = np.average(np.stack(aligned_alphas, axis=0), axis=0, weights=new_weights)
        updated = cv2.GaussianBlur(updated, (0, 0), sigmaX=1.0)

        if float(np.mean(np.abs(updated - consensus))) < 1e-4:
            consensus = updated
            weights = new_weights
            break

        consensus = updated
        weights = new_weights

    return consensus, weights
```

Now replace the multi-candidate branch in `build_consensus_watermark` with this version:

```python
    aligned_rgbs, aligned_alphas, _ = align_candidates_to_reference(candidates_bgra)
    consensus_alpha, _ = estimate_consensus_alpha(aligned_alphas)
    mean_rgb = np.mean(np.stack(aligned_rgbs, axis=0), axis=0)
    bbox = support_mass_bbox(consensus_alpha)
    alpha_u8 = np.clip(consensus_alpha * 255.0, 0, 255).astype(np.uint8)
    return crop_bgra(mean_rgb, alpha_u8, bbox)
```

* [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_estimate_consensus_alpha_keeps_shared_truth_and_suppresses_one_off_blob -q
```

Expected: `1 passed`.

Also run the full unit file:

```bash
pytest tests/test_watermark_consensus.py -q
```

Expected: all tests in that file pass.

* [ ] **Step 5: Commit**

```bash
git add watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: add iterative reweighted alpha consensus"
```

---

### Task 4: Add robust RGB consensus and support-mass cropping to the final builder

**Files:**

* Modify: `watermark_consensus.py`

* Modify: `tests/test_watermark_consensus.py`

* [ ] **Step 1: Write the failing test**

Append this test code to `tests/test_watermark_consensus.py`:

```python
def make_colored_bgra_from_mask(
    mask: np.ndarray,
    color: tuple[int, int, int],
    extra_blob: tuple[int, int, int] | None = None,
) -> np.ndarray:
    bgra = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    bgra[mask > 0, :3] = color
    bgra[:, :, 3] = mask

    if extra_blob is not None:
        top, left, size = extra_blob
        bgra[top : top + size, left : left + size, :3] = (20, 240, 20)
        bgra[top : top + size, left : left + size, 3] = 255

    return bgra


def test_build_consensus_watermark_combines_partial_truth_and_removes_blob() -> None:
    truth = make_irregular_truth_mask()

    candidate_a = make_colored_bgra_from_mask(make_partial(truth, "left"), (130, 140, 150))
    candidate_b = make_colored_bgra_from_mask(make_partial(truth, "right"), (132, 138, 152))
    candidate_c = make_colored_bgra_from_mask(truth, (128, 142, 148), extra_blob=(10, 188, 16))

    result = build_consensus_watermark([candidate_a, candidate_b, candidate_c])

    assert result is not None
    assert result.shape[0] < truth.shape[0]
    assert result.shape[1] < truth.shape[1]
    assert float(result[:, :, 3].mean()) > 40.0
    assert float(result[0:10, -20:, 3].mean()) < 20.0
```

* [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_build_consensus_watermark_combines_partial_truth_and_removes_blob -q
```

Expected: FAIL because the simple mean RGB path and raw crop are still keeping too much junk.

* [ ] **Step 3: Write minimal implementation**

Append this function to `watermark_consensus.py` above `build_consensus_watermark`:

```python
def estimate_consensus_rgb(
    aligned_rgbs: list[np.ndarray],
    aligned_alphas: list[np.ndarray],
    candidate_weights: np.ndarray,
) -> np.ndarray:
    if not aligned_rgbs:
        raise ValueError("estimate_consensus_rgb requires at least one aligned RGB image")

    h, w, _ = aligned_rgbs[0].shape
    numerator = np.zeros((h, w, 3), dtype=np.float32)
    denominator = np.zeros((h, w, 1), dtype=np.float32)

    for rgb, alpha, weight in zip(aligned_rgbs, aligned_alphas, candidate_weights):
        local_weight = (weight * alpha)[..., None]
        numerator += rgb * local_weight
        denominator += local_weight

    return numerator / np.clip(denominator, 1e-6, None)
```

Now replace the multi-candidate branch in `build_consensus_watermark` with this version:

```python
    aligned_rgbs, aligned_alphas, _ = align_candidates_to_reference(candidates_bgra)
    consensus_alpha, candidate_weights = estimate_consensus_alpha(aligned_alphas)
    consensus_rgb = estimate_consensus_rgb(aligned_rgbs, aligned_alphas, candidate_weights)

    bbox = support_mass_bbox(consensus_alpha)
    alpha_u8 = np.clip(consensus_alpha * 255.0, 0, 255).astype(np.uint8)
    return crop_bgra(consensus_rgb, alpha_u8, bbox)
```

* [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_watermark_consensus.py::test_build_consensus_watermark_combines_partial_truth_and_removes_blob -q
```

Expected: `1 passed`.

Then run the whole unit file:

```bash
pytest tests/test_watermark_consensus.py -q
```

Expected: all tests in that file pass.

* [ ] **Step 5: Commit**

```bash
git add watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: add rgb consensus and support-mass crop"
```

---

### Task 5: Add debug output and an opt-in wrapper in discovery.py

**Files:**

* Modify: `watermark_consensus.py`

* Modify: `discovery.py`

* Create: `tests/test_discovery_consensus_integration.py`

* [ ] **Step 1: Write the failing integration tests**

Create this file exactly as shown:

```python
# tests/test_discovery_consensus_integration.py
from pathlib import Path

import numpy as np

import discovery


def make_bgra_rect() -> np.ndarray:
    bgra = np.zeros((40, 80, 4), dtype=np.uint8)
    bgra[12:28, 20:64, :3] = (150, 150, 150)
    bgra[12:28, 20:64, 3] = 255
    return bgra


def test_discover_consensus_watermark_candidates_returns_single_consensus(monkeypatch) -> None:
    candidates = [make_bgra_rect(), make_bgra_rect()]

    monkeypatch.setattr(discovery, "discover_watermark_candidates", lambda image_paths: candidates)

    result = discovery.discover_consensus_watermark_candidates([Path("a.png"), Path("b.png")])

    assert len(result) == 1
    assert result[0].shape[2] == 4


def test_discover_consensus_watermark_candidates_falls_back_when_consensus_is_none(monkeypatch) -> None:
    candidates = [make_bgra_rect(), make_bgra_rect()]

    monkeypatch.setattr(discovery, "discover_watermark_candidates", lambda image_paths: candidates)
    monkeypatch.setattr(discovery, "build_consensus_watermark", lambda candidates_bgra, debug_dir=None: None)

    result = discovery.discover_consensus_watermark_candidates([Path("a.png"), Path("b.png")])

    assert result == candidates
```

* [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_discovery_consensus_integration.py -q
```

Expected: FAIL with `AttributeError` because `discover_consensus_watermark_candidates` does not exist yet.

* [ ] **Step 3: Write minimal implementation**

First, append this debug helper to `watermark_consensus.py` above `build_consensus_watermark`:

```python
from pathlib import Path


def _write_debug_images(
    debug_dir: str,
    aligned_alphas: list[np.ndarray],
    consensus_alpha: np.ndarray,
    consensus_rgb: np.ndarray,
    candidate_weights: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> None:
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, alpha in enumerate(aligned_alphas):
        cv2.imwrite(
            str(out_dir / f"aligned_alpha_{index:02d}.png"),
            np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
        )

    cv2.imwrite(
        str(out_dir / "consensus_alpha.png"),
        np.clip(consensus_alpha * 255.0, 0, 255).astype(np.uint8),
    )

    cv2.imwrite(
        str(out_dir / "consensus_rgb.png"),
        np.clip(consensus_rgb * 255.0, 0, 255).astype(np.uint8),
    )

    with open(out_dir / "weights.txt", "w", encoding="utf-8") as handle:
        for index, weight in enumerate(candidate_weights.tolist()):
            handle.write(f"{index}\t{weight:.6f}\n")

    with open(out_dir / "bbox.txt", "w", encoding="utf-8") as handle:
        x0, y0, x1, y1 = bbox
        handle.write(f"{x0},{y0},{x1},{y1}\n")
```

Now replace the full body of `build_consensus_watermark` in `watermark_consensus.py` with this final version:

```python
def build_consensus_watermark(
    candidates_bgra: list[np.ndarray],
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not candidates_bgra:
        return None

    if len(candidates_bgra) == 1:
        bgra = candidates_bgra[0]
        rgb = bgra[:, :, :3].astype(np.float32) / 255.0
        alpha = alpha_to_soft_mask(bgra[:, :, 3])
        bbox = support_mass_bbox(alpha)
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        result = crop_bgra(rgb, alpha_u8, bbox)
        if debug_dir is not None:
            _write_debug_images(debug_dir, [alpha], alpha, rgb, np.array([1.0], dtype=np.float32), bbox)
        return result

    aligned_rgbs, aligned_alphas, _ = align_candidates_to_reference(candidates_bgra)
    consensus_alpha, candidate_weights = estimate_consensus_alpha(aligned_alphas)
    consensus_rgb = estimate_consensus_rgb(aligned_rgbs, aligned_alphas, candidate_weights)
    bbox = support_mass_bbox(consensus_alpha)

    if debug_dir is not None:
        _write_debug_images(
            debug_dir=debug_dir,
            aligned_alphas=aligned_alphas,
            consensus_alpha=consensus_alpha,
            consensus_rgb=consensus_rgb,
            candidate_weights=candidate_weights,
            bbox=bbox,
        )

    alpha_u8 = np.clip(consensus_alpha * 255.0, 0, 255).astype(np.uint8)
    return crop_bgra(consensus_rgb, alpha_u8, bbox)
```

Then modify `discovery.py` by adding this import near the top with the other imports:

```python
from watermark_consensus import build_consensus_watermark
```

And append this new wrapper at the bottom of `discovery.py`:

```python
def discover_consensus_watermark_candidates(
    image_paths: List[Path],
    debug_dir: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Run the existing candidate discovery pipeline, then collapse the returned
    BGRA candidates into one consensus BGRA candidate. Falls back to the
    original candidate list if consensus fails.
    """
    candidates = discover_watermark_candidates(image_paths)
    if not candidates:
        return []

    consensus = build_consensus_watermark(candidates, debug_dir=debug_dir)
    if consensus is None:
        return candidates

    return [consensus]
```

* [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_discovery_consensus_integration.py -q
```

Expected: `2 passed`.

Then run the whole new test surface:

```bash
pytest tests/test_watermark_consensus.py tests/test_discovery_consensus_integration.py -q
```

Expected: all tests pass.

* [ ] **Step 5: Commit**

```bash
git add watermark_consensus.py discovery.py tests/test_discovery_consensus_integration.py tests/test_watermark_consensus.py
git commit -m "feat: wire consensus watermark builder into discovery wrapper"
```

---

## Self-Review

* Spec coverage:

  * Multiple noisy BGRA candidates combined into one latent watermark: covered in Tasks 2–4.
  * No text-specific assumptions: all synthetic tests use generic geometric masks, not glyph logic.
  * Threshold-light behavior: support mass crop and inverse-error weighting are data-driven; no “largest text-like blob” logic appears anywhere.
  * Bolt-on integration with unseen codebase: covered by additive `watermark_consensus.py` and the new wrapper in `discovery.py`.
  * Debug visibility: covered in Task 5 with aligned-alpha, consensus-alpha, RGB, weights, and bbox artifacts.

* Placeholder scan:

  * No `TODO`, `TBD`, or “implement later” content remains in the task steps.
  * Every code-changing step includes concrete code.
  * Every test-running step includes the exact command and expected result.

* Type consistency:

  * `build_consensus_watermark(candidates_bgra, debug_dir=None)` is used consistently in tests and integration.
  * `align_candidates_to_reference` always returns `(aligned_rgbs, aligned_alphas, weights)`.
  * `estimate_consensus_alpha` always returns `(consensus_alpha, weights)`.

Recommended save path for this one: `docs/superpowers/plans/2026-03-31-watermark-consensus.md`. I have not saved it here because you asked for it in chat. ([GitHub][1])