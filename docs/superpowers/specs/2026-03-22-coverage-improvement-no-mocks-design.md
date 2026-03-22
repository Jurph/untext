# Design: Coverage Improvement — No-Mock Low-Hanging Fruit

**Date:** 2026-03-22
**Status:** Approved for implementation
**Goal:** Raise test coverage from 68% toward 75% by covering pure-logic branches, error paths, and the package lazy-loader — no mocks required.

---

## Problem

Current coverage is 68%. The 75% target is failing. The main drag is `detector.py` (13%) and `consensus.py` (38%), both of which require live ML models. Those will be addressed with mocks in a future pass. This spec targets the remaining uncovered lines that are reachable with synthetic inputs and real code paths.

---

## Scope

Three categories of uncovered lines, all testable without mocks:

1. **Pure-logic branches in `find_mask_by_spatial_tf_idf`** — debug logging branches and cluster-data return path, only hit when `debug=True` and `return_cluster_data=True`
2. **Error and guard paths in `cli.py`** — bad image file, same-dir guard, missing template path, timing report writer
3. **`__init__.py` lazy-loader** — `__getattr__` hook covering lines 39-80, triggered by attribute access on the package

Estimated gain: **+5 to +5.5 percentage points**, reaching approximately 73–74%.

---

## Files Changed

- **Modify:** `tests/test_find_text_colors.py` — new `TestFindMaskDebugAndClusterData` class
- **Modify:** `tests/test_cli.py` — new `TestCliErrorPaths` class + one lazy-loader test

No new test files.

---

## Test Designs

### `tests/test_find_text_colors.py` — `TestFindMaskDebugAndClusterData`

**Shared fixture:** A 100×100 BGR image rendered with `cv2.putText("TEST TEXT", ...)` in BGR `(0, 0, 220)` (vibrant red) on a `(128, 128, 128)` grey background. The text region bbox is passed as the detection area. Using rendered text rather than a plain rectangle ensures the red cluster has fragmented connected components (letter shapes), strong TF-IDF distinctiveness, and meaningful border underrepresentation — giving the FOM scoring signals something real to evaluate.

#### `test_find_mask_debug_paths`

- Call `find_mask_by_spatial_tf_idf(image, bbox, num_clusters=4, debug=True)`
- Assert: returns a numpy array (mask), does not raise
- Coverage hit: lines 527-536, 565, 596, 606-609, 617-619, 626-630, 634 (all `if debug: logger.info(...)` branches)

#### `test_find_mask_returns_cluster_data`

- Call `find_mask_by_spatial_tf_idf(image, bbox, num_clusters=4, return_cluster_data=True)`
- Assert: returns a tuple `(mask, cluster_data)`
- Assert: `cluster_data` is a dict with keys `centers`, `top_id`, `bot_id`, `color_radius`, `bg_radius`
- Assert: `cluster_data["centers"]` has shape `(K, 3)`
- Coverage hit: lines 656-676 (adaptive radius computation, dict construction)

#### `test_find_mask_flat_image_normalization`

- Create a 100×100 solid-color image (all pixels identical, e.g. `(100, 100, 100)`)
- Call `find_mask_by_spatial_tf_idf` on it with any bbox
- Assert: returns without raising (the `max_score == min_score` guard at line 492 fires safely)
- Coverage hit: line 492 (TF-IDF normalization edge case — division-by-zero guard)

---

### `tests/test_cli.py` — `TestCliErrorPaths`

#### `test_process_with_known_mask_bad_image`

- Write a zero-byte file to `tmp_path / "bad.jpg"`
- Create a valid 50×50 BGRA dummy template array
- Call `process_with_known_mask(image_path=bad_path, output_dir=tmp_path, known_mask_rgba=template)`
- Assert: returns `None`
- Coverage hit: lines 529-530 (load failure early return)

#### `test_main_same_dir_guard`

- Create a temp directory `d` via `tmp_path`
- Monkeypatch `sys.argv` to `["untextre", "-i", str(d), "-o", str(d), "-U"]` (argv[0] = program name)
- Monkeypatch `untextre.cli.get_image_files` to return `[d / "fake.jpg"]` so the empty-list early return at line 656 does not fire first
- Use `pytest.raises(SystemExit) as exc_info:` around `main()`, then assert `exc_info.value.code == 1`
- Coverage hit: lines 681-685 (same-dir guard)

#### `test_load_watermark_templates_missing_path`

- Call `load_watermark_templates(Path("definitely_does_not_exist_xyz"))`
- Assert: returns `[]`
- Coverage hit: line 234 (early return for non-existent path)

#### `test_save_timing_report`

- Call `_save_clean_timing_report` with: a minimal list of timing dicts (e.g. `[{"file": "a.jpg", "time": 1.2}]`), `total_time=1.2`, `avg_time=1.2`, `timing_file=tmp_path / "timing.txt"`, `method="known_mask"`, `confidence_threshold=0.5`, `target_color=None`, `forced_bbox=None`
- Assert: `tmp_path / "timing.txt"` exists
- Assert: the file content contains `"Inpainting method:"` (always written unconditionally)
- Coverage hit: line 1359 (file write in timing report)

---

### `tests/test_cli.py` — standalone

#### `test_package_lazy_imports`

- `import untextre`
- Access exported names from different lazy branches: `untextre.load_image` (utils branch), `untextre.inpaint_image` (inpaint branch), `untextre.main` (cli branch), `untextre.find_mask_by_spatial_tf_idf` (find_text_colors branch)
- Assert each is callable (i.e. `callable(untextre.load_image)` etc.)
- Note: `__getattr__` returns individual symbols, not submodule objects — do NOT access `untextre.cli` as an attribute (it is not exported and will raise `AttributeError`)
- Coverage hit: lines 39-80 of `untextre/__init__.py` (`__getattr__` lazy-loader)

---

## Success Criteria

- All 8 new tests pass with `venv/Scripts/python -m pytest -m "not slow"`
- Coverage rises to ≥73% as measured by `--cov=untextre`
- No new test files created
- No mocks used in any of the 8 tests

---

## Out of Scope

- `detector.py`, `consensus.py` ML-model paths — deferred to mock-based pass
- `grabcut_refine` and `color_guided_expand` GrabCut paths — deferred
- `cli.py` ORB matching paths (lines 397-450) — require real images with keypoints; deferred
