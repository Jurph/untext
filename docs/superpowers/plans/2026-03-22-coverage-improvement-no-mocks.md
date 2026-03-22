# Coverage Improvement — No-Mock Low-Hanging Fruit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8 new tests to two existing test files, covering pure-logic branches in `find_mask_by_spatial_tf_idf`, error paths in `cli.py`, and the `__init__.py` lazy-loader — raising coverage from 68% to ≥73%.

**Architecture:** All tests use real code paths and synthetic inputs only — no mocks, no temporary test modules. Task 1 adds a new test class to `tests/test_find_text_colors.py`. Task 2 adds a new test class plus one standalone test to `tests/test_cli.py`.

**Tech Stack:** pytest, numpy, cv2 (OpenCV), Python pathlib, `untextre` package

---

## Files Changed

- **Modify:** `tests/test_find_text_colors.py` — append `TestFindMaskDebugAndClusterData` class (3 tests)
- **Modify:** `tests/test_cli.py` — append `TestCliErrorPaths` class (4 tests) and `test_package_lazy_imports` function

No new test files. No new source files. No mock usage anywhere.

---

## Important: How to Run Tests

**Always use the project venv.** The system Python on this machine has a broken TensorFlow installation that causes collection errors. Use:

```bash
venv/Scripts/python -m pytest -m "not slow" --cov=untextre --cov-report=term
```

To run a single test class during development:

```bash
venv/Scripts/python -m pytest tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData -v
venv/Scripts/python -m pytest tests/test_cli.py::TestCliErrorPaths -v
```

---

## Background: What These Tests Cover

### `find_mask_by_spatial_tf_idf` (in `untextre/find_text_colors.py`)

Signature:
```python
def find_mask_by_spatial_tf_idf(
    image: np.ndarray,   # BGR image
    bbox: tuple,         # (x, y, width, height) — NOT (x1,y1,x2,y2)
    num_clusters: int = 4,
    debug: bool = False,
    target_color=None,
    fom_threshold: float = 0.30,
    cc_guard: float = 0.85,
    use_grabcut: bool = False,
    return_cluster_data: bool = False,
) -> np.ndarray:  # or (np.ndarray, dict) when return_cluster_data=True
```

When `return_cluster_data=True`, returns `(cleaned_mask, cluster_data)` where:
```python
cluster_data = {
    "centers": np.ndarray,   # shape (K, 3) — K-means centroids in RGB
    "top_id": int,           # index of highest-FOM cluster
    "bot_id": int,           # index of lowest-FOM cluster
    "color_radius": float,   # adaptive radius for top cluster
    "bg_radius": float,      # adaptive radius for background cluster
}
```

### `_save_clean_timing_report` (in `untextre/cli.py`)

Signature:
```python
def _save_clean_timing_report(
    detailed_timings: list,
    total_time: float,
    avg_time: float,
    timing_file: Path,     # caller supplies the full path; function writes to it
    method: str,
    confidence_threshold: float,
    target_color: Optional[tuple],
    forced_bbox: Optional[tuple],
) -> None:
```

Always writes `"Inpainting method: {method}"` unconditionally.

### `process_with_known_mask` (in `untextre/cli.py`)

Early-return path: if `load_image(image_path)` returns `None` (bad/empty file), logs an error and returns `None`. A zero-byte `.jpg` triggers this path.

### `load_watermark_templates` (in `untextre/cli.py`)

Early return at line 234: if `path.exists()` is False, returns `[]` immediately.

### Same-dir guard in `main()` (in `untextre/cli.py`)

Guard at lines 679-685: when `-U` flag is set and `input_path.resolve() == output_path.resolve()` and `--force` is not set, calls `sys.exit(1)`.

The guard fires AFTER `get_image_files()` is called — so the `get_image_files` call at line 653 must return a non-empty list to reach the guard. `get_image_files` is imported from `untextre.utils` into `untextre.cli`; monkeypatch it at `untextre.cli.get_image_files`.

### `__init__.py` lazy-loader

The `__getattr__` hook (lines 39-80) dispatches by name to lazy-import branches:
- `load_image` → utils branch (line 39-49)
- `find_mask_by_spatial_tf_idf` → find_text_colors branch (line 60-62)
- `inpaint_image` → inpaint branch (line 68-70)
- `main` → cli branch (line 76-78)

Accessing `untextre.cli` as an attribute raises `AttributeError` — the hook only exports named symbols, not submodule objects. Use `callable(untextre.load_image)` etc.

---

## Task 1: `TestFindMaskDebugAndClusterData` in `test_find_text_colors.py`

**Files:**
- Modify: `tests/test_find_text_colors.py` (append at end of file)

- [ ] **Step 1: Write the failing tests**

Append this entire block to the end of `tests/test_find_text_colors.py`:

```python
import cv2
import numpy as np
import pytest
from untextre.find_text_colors import find_mask_by_spatial_tf_idf


class TestFindMaskDebugAndClusterData:
    """Cover the debug-logging branches and return_cluster_data path
    of find_mask_by_spatial_tf_idf — reachable without mocks."""

    @pytest.fixture
    def text_image(self):
        """100×100 BGR image: vibrant red text on grey background."""
        img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
        cv2.putText(img, "TEST TEXT", (5, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 220), 2)
        return img

    @pytest.fixture
    def text_bbox(self):
        """(x, y, width, height) covering the text region."""
        return (5, 35, 90, 40)

    def test_find_mask_debug_paths(self, text_image, text_bbox):
        """debug=True exercises all if-debug logger.info branches."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, debug=True
        )
        assert isinstance(result, np.ndarray), "Expected ndarray mask"
        assert result.dtype == np.uint8

    def test_find_mask_returns_cluster_data(self, text_image, text_bbox):
        """return_cluster_data=True exercises the dict-construction path."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, return_cluster_data=True
        )
        assert isinstance(result, tuple), "Expected (mask, cluster_data) tuple"
        mask, cluster_data = result
        assert isinstance(mask, np.ndarray)
        assert isinstance(cluster_data, dict)
        for key in ("centers", "top_id", "bot_id", "color_radius", "bg_radius"):
            assert key in cluster_data, f"Missing key: {key}"
        assert cluster_data["centers"].shape == (4, 3)

    def test_find_mask_flat_image_normalization(self):
        """Solid-color image exercises the max_score==min_score guard (line 492)."""
        flat = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        result = find_mask_by_spatial_tf_idf(flat, bbox, num_clusters=4)
        assert isinstance(result, np.ndarray), "Should return mask without raising"
```

- [ ] **Step 2: Run the tests**

```bash
venv/Scripts/python -m pytest tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData -v
```

Expected: 3 tests collected. If they fail with `ImportError`, check the imports at the top of the appended block (`cv2` and `find_mask_by_spatial_tf_idf` must be importable — confirm both are available in the venv).

Note: `find_mask_by_spatial_tf_idf` already exists in the source; the tests should pass on first run if the code is correct.

- [ ] **Step 3: Verify the 3 tests pass**

```bash
venv/Scripts/python -m pytest tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData -v
```

Expected output:
```
PASSED tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData::test_find_mask_debug_paths
PASSED tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData::test_find_mask_returns_cluster_data
PASSED tests/test_find_text_colors.py::TestFindMaskDebugAndClusterData::test_find_mask_flat_image_normalization
3 passed
```

If `test_find_mask_returns_cluster_data` fails on the `centers.shape == (4, 3)` assertion, check that `num_clusters=4` matches the shape. `cv2.kmeans` returns centers of shape `(K, 3)` when input is `(N, 3)`.

- [ ] **Step 4: Verify the full test suite still passes**

```bash
venv/Scripts/python -m pytest -m "not slow" -q
```

Expected: all previously-passing tests still pass, plus the 3 new ones.

- [ ] **Step 5: Commit**

```bash
git add tests/test_find_text_colors.py
git commit -m "test: cover find_mask debug/cluster_data paths and flat-image guard"
```

---

## Task 2: `TestCliErrorPaths` and `test_package_lazy_imports` in `test_cli.py`

**Files:**
- Modify: `tests/test_cli.py` (append at end of file)

Note: `test_cli.py` already imports most needed symbols. Check the existing imports at the top of the file and add `import untextre` there (alongside the other top-level imports) rather than in the appended block — this avoids a mid-file import.

- [ ] **Step 1: Check existing imports in test_cli.py**

Read the first 40 lines of `tests/test_cli.py` to confirm which symbols are already imported. The following are already imported (verified from the file):

```python
import sys
import cv2
import numpy as np
import pytest
import untextre.cli as cli_mod
from untextre.cli import (
    _apply_color_enhancement,
    _save_clean_timing_report,
    find_known_mask_in_image,
    load_watermark_templates,
    main,
    parse_args,
    process_with_known_mask,
    process_single_image,
    try_watermark_cascade,
)
```

All symbols needed for the new tests are already imported. You only need to add `import untextre` for the lazy-loader test.

- [ ] **Step 2: Write the new tests**

First, add `import untextre` to the top-level import block at the top of `tests/test_cli.py` (alongside the existing `import sys`, `import cv2`, etc.).

Then append this block to the end of `tests/test_cli.py`:

```python
class TestCliErrorPaths:
    """Cover error/guard paths in cli.py — no mocks, pure logic paths."""

    def test_process_with_known_mask_bad_image(self, tmp_path):
        """Zero-byte image file → load_image returns None → early return None."""
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"")  # zero-byte file
        template = np.zeros((50, 50, 4), dtype=np.uint8)
        result = process_with_known_mask(
            image_path=bad_path,
            output_dir=tmp_path,
            known_mask_rgba=template,
        )
        assert result is None

    def test_main_same_dir_guard(self, tmp_path, monkeypatch):
        """main() with -U and same input/output dir exits with code 1."""
        monkeypatch.setattr(
            sys, "argv", ["untextre", "-i", str(tmp_path), "-o", str(tmp_path), "-U"]
        )
        # Return a non-empty file list so the empty-list guard (line 654) doesn't
        # fire before we reach the same-dir guard (line 679).
        monkeypatch.setattr(
            "untextre.cli.get_image_files",
            lambda _path: [tmp_path / "fake.jpg"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_load_watermark_templates_missing_path(self):
        """Non-existent path → load_watermark_templates returns []."""
        from pathlib import Path
        result = load_watermark_templates(Path("definitely_does_not_exist_xyz"))
        assert result == []

    def test_save_timing_report(self, tmp_path):
        """_save_clean_timing_report writes a file containing expected headers."""
        from pathlib import Path
        timing_file = tmp_path / "timing.txt"
        detailed = [{"file": "a.jpg", "time": 1.2}]
        _save_clean_timing_report(
            detailed_timings=detailed,
            total_time=1.2,
            avg_time=1.2,
            timing_file=timing_file,
            method="known_mask",
            confidence_threshold=0.5,
            target_color=None,
            forced_bbox=None,
        )
        assert timing_file.exists(), "Timing report file was not created"
        content = timing_file.read_text()
        assert "Inpainting method:" in content


def test_package_lazy_imports():
    """__getattr__ lazy-loader in __init__.py is triggered by attribute access."""
    # Access one name from each lazy branch to maximise coverage:
    # utils branch, find_text_colors branch, inpaint branch, cli branch
    assert callable(untextre.load_image), "load_image not callable"
    assert callable(untextre.find_mask_by_spatial_tf_idf), "find_mask_by_spatial_tf_idf not callable"
    assert callable(untextre.inpaint_image), "inpaint_image not callable"
    assert callable(untextre.main), "main not callable"
```

- [ ] **Step 3: Run the new tests to verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_cli.py::TestCliErrorPaths tests/test_cli.py::test_package_lazy_imports -v
```

Expected output:
```
PASSED tests/test_cli.py::TestCliErrorPaths::test_process_with_known_mask_bad_image
PASSED tests/test_cli.py::TestCliErrorPaths::test_main_same_dir_guard
PASSED tests/test_cli.py::TestCliErrorPaths::test_load_watermark_templates_missing_path
PASSED tests/test_cli.py::TestCliErrorPaths::test_save_timing_report
PASSED tests/test_cli.py::test_package_lazy_imports
5 passed
```

**Troubleshooting:**

- If `test_main_same_dir_guard` exits with code other than 1 or doesn't exit at all:
  - Verify `tmp_path` is a real directory (it is — pytest creates it automatically)
  - Verify the monkeypatch target string is exactly `"untextre.cli.get_image_files"` (the reference in cli's own namespace)
  - The `-U` flag maps to `args.unknown_watermark`; verify this with `parse_args()` if needed

- If `test_save_timing_report` fails on the file existence assertion:
  - Verify `timing_file` is `tmp_path / "timing.txt"` (a full path, not just a filename)
  - `_save_clean_timing_report` takes `timing_file: Path` as the 4th positional argument

- If `test_package_lazy_imports` raises `ImportError` on `untextre.inpaint_image`:
  - `inpaint.py` imports `cv2` and `numpy` at the module level — confirm both are in the venv

- [ ] **Step 4: Run the full suite with coverage**

```bash
venv/Scripts/python -m pytest -m "not slow" --cov=untextre --cov-report=term -q
```

Expected: all tests pass; coverage ≥ 73%.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: cover cli error paths, same-dir guard, and __init__ lazy-loader"
```

---

## Success Criteria Checklist

After both tasks are complete, verify:

- [ ] All 8 new tests pass: `venv/Scripts/python -m pytest -m "not slow" -q`
- [ ] Coverage ≥ 73%: `venv/Scripts/python -m pytest -m "not slow" --cov=untextre --cov-report=term`
- [ ] No new test files were created (only the two existing files were modified)
- [ ] No mocks used (`unittest.mock`, `pytest-mock`, `monkeypatch.setattr` on ML internals — the one monkeypatch in `test_main_same_dir_guard` patches a pure Python function, not an ML component, and is necessary to reach the guard)
