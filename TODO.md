# Vision 

The goal of this tool is fast and accurate watermark removal. We detect text-based watermarks in images, we identify regions of that color, generate a mask that exactly matches those pixelated regions, and then inpaint only the masked area. 


# TODO

## Completed Features

- Text detection approach with pixel-accurate masking (spatial TF-IDF / FOM)
- Extensive testing (246 tests across 14 test modules)
- Full-featured CLI pipeline with all options
- Smart sub-region selection with padding and mod-4/mod-8 alignment
- `--maskfile` option for mask editing and retry
- LaMa (default) vs TELEA inpainting, both implemented
- EAST added to detection suite — consensus of 3 detectors (EAST+DocTR+EasyOCR)
- Local web service with drag-and-drop (Streamlit app)
- Batch processing for directories
- Automatic granularity optimization (g=4 default, auto-retry at g=8)
- CLI bbox expansion along long axis for missed text
- VRAM management for steady-state batch processing
- Known-mask mode (`-K`) for ORB-based template matching of consistent watermarks
- `@pytest.mark.slow` on model-loading tests; fast suite runs in ~18s
- Streamlit `main()` decomposed: pure helpers extracted, section comments, constants
- `mask_generator.py` pruned to single production function (`morph_clean_mask`)
- `cli.py` mask+inpaint closure extracted to module-level `_generate_masks_and_inpaint`

## Future Ideas

- **`--letters` mode**: Fill detected regions with approximate matching letters (font rendering + OCR text extraction). Lower priority — current inpainting works well.
- **End-to-end pipeline tests**: Full image-in / image-out tests with SSIM/PSNR quality gates.
- **Shape-metric fusion**: Combine FOM with per-cluster blackhat_energy and edge_row_energy for even better text/background separation. Experiment infrastructure exists in `experiments/cluster_shape_experiment.py`.

## Quality Debt

*Findings from a three-pass code review (2026-06-23): senior-engineer findings, PM cost/benefit response, and engineer rebuttal. Scratch artifacts have been consolidated here and removed. Items are ordered by priority within each tier.*

### Must Fix

- [x] **Dual model-cache VRAM hazard**: `consensus.py` and `detector.py` each maintain independent singletons for DocTR, EasyOCR, and EAST — three models each loaded twice. Refactor `consensus.py` to use `detector.py`'s `initialize_models()` and singleton accessors. Real OOM risk on GPUs with <8GB VRAM. (`consensus.py` lines 18–20; `detector.py` lines 32–34)
- [x] **`import easyocr` at `detector.py` module top-level defeats lazy loading**: Unconditional top-level import adds 2–5s startup cost to every invocation, including `--help`. Move to inside the functions that use it. (`detector.py` line 19)
- [x] **`MAX_STRETCH = 1.25` affine guard is undocumented and probably too tight**: SVD ordering means `scale_major >= scale_minor` by construction, so 1.25:1 (a 5:4 axis ratio) rejects any mildly off-axis or curved-surface watermark silently. Add `# EMPIRICAL — not validated` comment with derivation, or calibrate against the actual watermark corpus. (`cli.py` lines 437, 607; project rules: `CLAUDE.md`)
- [x] **Rotation-failover coordinate inverse transform has a live uncertainty comment**: The comment at `cli.py` line 1272 reads "Wait, let me think about this differently…" and the formula `y_orig = h - x_rot - w_rot` mixes rotated-frame coordinates in a suspicious way. Write a unit test: rotate a known image 90° clockwise, detect a known bbox, apply the inverse, verify coordinates against hand-calculated expected values.
- [x] **`fom_threshold = 0.30` gates the entire spatial TF-IDF color path without derivation**: FOM weights cite 18,000+ samples; the acceptance threshold does not. Add `# EMPIRICAL — not yet validated against a diverse dataset` comment. (`find_text_colors.py` ~line 341; project rules: `CLAUDE.md`)

### Should Fix

- [x] **Streamline fixture tests and local developer tooling**: Mark long-running real-image fixture tests as `@pytest.mark.slow` or split them out of the default fast suite; add Ruff configuration and checks; migrate project metadata/dependency management from `setup.py` + `requirements*.txt` to `pyproject.toml` + `uv.lock`. Keep this as local quality-of-life and existing-CircleCI maintenance only: no PyPI publishing, GitHub Actions migration, CD pipeline, branch protections, or remote CUDA testing.
- [ ] **Investigate `_parse_doctr_output` for silent zero-detection**: The `isinstance(page['words'], np.ndarray)` guard at `detector.py` line 598 returns `[]` with no log if DocTR returns a dict-of-dicts (v0.6+ format). Confirm whether DocTR detection is currently live; add a per-image detection-count log to make silent failures visible.
- [x] **Delete dead `_find_known_mask_in_image_single_variant`**: Zero callers outside the definition file; 185 lines of logic ~85% duplicated from `find_known_mask_in_image`. Safe delete. (`cli.py` lines 324–509)
- [x] **Replace two inline bbox-superset calculations with `utils.calculate_bbox_superset`**: Inline copies at `cli.py` lines 267–271 and 1347–1351 diverge from the utility function that already handles the empty-list case.
- [x] **Fix `timing_data["total_time"] = 0` in template-match path**: Hardcoded zero misleads users comparing ORB-match vs. consensus-detection performance. Record actual elapsed time. (`cli.py` line 1041)
- [ ] **Add `# EMPIRICAL` to `overlap_threshold = 0.1` in `consensus.py`**: Controls whether two detectors "agree"; no documented derivation. (`consensus.py` lines 244, 397)
- [ ] **Log EAST NMS fallback threshold**: The `+0.1` fallback at `detector.py` line 292 announces the fallback but not the effective threshold, making intermittent detection failures undiagnosable. Log the value and add `# EMPIRICAL` comment.
- [x] **Delete `get_largest_text_region` and `_merge_bboxes` and their unit tests**: Confirmed dead production code; tests written for them have no diagnostic value. (`detector.py` lines 385–477)
- [ ] **Fix `auto_retry` OOM loop in `inpaint.py`**: Catching all exceptions and calling `initialize_lama_model(force_reinit=True)` on `OutOfMemoryError` tries to reload a model into already-full VRAM, making OOM worse. Catch `torch.cuda.OutOfMemoryError` explicitly and skip the retry. (`inpaint.py` lines 279–292)
- [ ] **Downgrade `_calculate_inpainting_subregion` logs from INFO to DEBUG**: Eight `logger.info` calls per subregion = 8,000 log lines per 1,000-image batch. Keep one INFO summary per region. (`inpaint.py`)
- [ ] **Add `logger.propagate = False` to `setup_logger`**: Prevents double-logging when the root logger already has handlers (common in both CLI and Streamlit contexts). (`utils.py`)
- [ ] **Change `logger.error()` to `logger.exception()` in `preprocessor.py` bare except**: Preserves stack trace when preprocessing fails. One-character-ish change, zero risk. (`preprocessor.py` line 66)
- [x] **Remove `nfeatures` parameter from `count_candidate_orb_keypoints`**: Parameter is immediately `del`d; callers who pass it get a 5000-feature ORB regardless. Deceptive API. (`orb_prep.py` lines 159–163)
- [x] **Remove deprecated `confidence_threshold` from `consensus.initialize_consensus_models`**: Documented as "ignored"; callers believe they are configuring behavior when they are not.
- [x] **Remove `debug_dir` from `build_final_templates` signature**: Accepted then immediately `del`d; replace with a TODO comment if the feature is planned. (`watermark_consensus.py` lines 1443–1447)
- [ ] **Add `# EMPIRICAL` to `color_distance_floor=24.0`**: Per project rules. (`watermark_consensus._filter_scoring_alpha` ~line 161)
- [ ] **Add `# EMPIRICAL` and `logger.warning` to `color_radius = 30.0` fallback**: Fallback fires when bbox contains no valid pixels — a suspicious degenerate condition that should surface in logs. (`find_text_colors.py` ~line 671)
- [ ] **Add runtime `logger.warning` to `compute_median_gradient` for large batches**: Docstring acknowledges O(N×H×W) RAM; a warning when N exceeds a documented threshold (e.g., N > 50 at estimated 4K) prevents silent OOM. (`discovery.py` lines 386–402)

### Backlog

- [ ] **Consolidate `cli.py` into smaller modules** (`orb_matcher.py`, `pipeline.py`): Right long-term direction; defer until test coverage is comprehensive enough to support a safe refactor of the lazy-loading import graph.
- [ ] **Switch to `cv2.KMEANS_PP_CENTERS`**: Better initialization stability improves run-to-run reproducibility. (`find_text_colors.py`)
- [x] **Delete other confirmed dead code**: `dilate_by_percent` in `utils.py`; inline Tukey fence in `discover_watermark_candidates` duplicates `_precision_outlier_threshold` in `discovery.py`.
- [ ] **Switch `np.random.RandomState` to `np.random.default_rng`**: Avoids XOR seed collisions for same-aspect-ratio images. (`discovery.py` line 1023)
- [ ] **Add `O(N³)` complexity comment to `find_consensus_boxes`**: Not a real bottleneck at current N, but worth documenting. (`consensus.py` lines 275–318)
- [ ] **Fix `__init__.py` `locals()[name]` pattern**: Replace with `getattr(module, name)` to give `AttributeError` instead of `KeyError` on maintenance mistakes. (`__init__.py` lines 44–49)


# Architecture

## Pipeline Flow

### 1. `preprocessor.py` — Image Enhancement
- CLAHE contrast enhancement
- Bilateral filtering for noise reduction while preserving edges
- Grayscale → RGB conversion for detector compatibility

### 2. `consensus.py` — Consensus Detection
- Run all 3 detectors simultaneously: EAST, DocTR, EasyOCR
- Find regions where 2+ detectors agree (configurable overlap threshold)
- Calculate hybrid confidence: 1 − (1−c₁)×(1−c₂)×…×(1−cₙ)
- Merge overlapping detections into consensus bounding boxes
- Pad boxes by 20% and align to mod-4
- Failover cascade if no consensus:
  1. Rotation (90° clockwise)
  2. Target color enhancement (if user specified)
  3. Generic gray enhancement (#808080)
  4. White enhancement (#FFFFFF)
  5. Fall back to standard watermark regions (bottom corners)

### 3. `find_text_colors.py` — Figure of Merit Analysis
- For each consensus region:
  - Extract region + surrounding area (1.414× expanded for equal pixel count)
  - K-means cluster colors (default: 4 clusters, auto-retry at 8 if remnants detected)
  - For each cluster, compute TF-IDF, border ratio, and largest CC fraction
  - Compute weighted FOM = 0.07 × tf_idf + 0.63 × border + 0.30 × cc
  - Accept clusters with FOM ≥ 0.30 and largest CC < 85%
  - Optional: force inclusion of target color cluster
- Combine all regional masks into final mask

### 4. `mask_generator.py` — Morphological Cleanup
- Closing to fill gaps and connect fragments (11×11 kernel)
- Dilation for inpainting coverage (13×13 ellipse)
- Gaussian blur for smooth edges (9×9)
- Re-threshold to binary mask

### 5. `inpaint.py` — LaMa/TELEA Inpainting
- Calculate optimal subregion from mask bounds
- Dilate subregion by 64 pixels for context
- Pad to mod-8 for neural network compatibility
- Crop image and mask to subregion
- Run LaMa (GPU-accelerated) or TELEA (fast CPU fallback)
- Paste result back into full image
- Save cleaned image and optional mask 
