# TODO

This file tracks current problems in the codebase. It should not duplicate the
README or preserve old completed review notes.

## Must Fix Before v1.0

- [ ] **WatermarkTemplate iteration in log**: Change `name for name, _ in watermark_templates`
  to `t.name for t in watermark_templates` to stop depending on the backward-compat
  `__iter__` shim. `(cli.py:162)`

- [ ] **EAST download reliability**: Add timeout, file-size check, and a human-readable
  error to the EAST model download; prevent silent HTML-404 corruption of the cached
  `.pb` file. If automatic download fails, print a long-term reliable manual-download
  URL and tell the user where to place the `.pb` file. `(detector.py:244-245)`

---

## Should Fix

- [ ] **Remove WatermarkTemplate shim**: Delete `__iter__` / `__getitem__` from
  `WatermarkTemplate` and the legacy tuple branch from `try_watermark_cascade` after
  the `cli.py:162` fix lands; the shim has no remaining legitimate use.
  `(orb_matcher.py:25-31, 291-298)`

- [ ] **Lowe ratio-test citation**: Add a Lowe (2004) citation comment to the `0.75`
  ratio-test constant. `(orb_matcher.py:135)`

- [ ] **50% coverage guard comment**: Add one sentence explaining why a watermark
  covering more than 50% of the image is treated as spurious. `(orb_matcher.py:252-256)`

- [ ] **Private re-exports in cli**: Remove or explicitly promote the `_private`
  re-exports from `cli.py` lines 25-28; do not let them ship as implicit public API.
  `(cli.py:22-28)`

- [ ] **LaMa paste-back coordinates**: Replace the redundant `subregion[0] + edge_pad_*`
  recomputation at paste-back with the already-adjusted `x1, y1, x2, y2` local
  variables — two sources of truth in a correctness-critical path.
  `(lama_inpainter.py:292-295)`

- [ ] **edge_pad_size empirical label**: Add `# EMPIRICAL` label (or architectural
  derivation) to `edge_pad_size = 32`. `(lama_inpainter.py:193-196)`

- [ ] **Formula weight labels**: Add `# EMPIRICAL` labels to the floating-point
  weights in `_component_shape_weight` and `compatibility`.
  `(watermark_consensus.py:624-625, 780-787)`

- [ ] **Minimum-n label**: Add `# EMPIRICAL — minimum n for Welford variance; broader
  validation pending` to the `n < 3` guard. `(discovery.py:349)`

- [ ] **metrics.py constant cross-references**: Add `granularity_experiment.py`
  cross-reference directly to each threshold constant's comment (not only the module
  docstring). `(metrics.py:27-32)`

- [ ] **preprocessor.py dead alias**: Remove `result_img = image` alias; write
  `cv2.cvtColor(image, ...)` directly. `(preprocessor.py:44)`

- [ ] **preprocessor.py CLAHE/bilateral citation**: Update the module docstring and
  inline comments to name the dataset and define the FOM metric for the grid-search
  result. `(preprocessor.py:52-57)`

- [ ] **morph_clean_mask dead bbox param**: Remove the unused `bbox: BBox` parameter
  from `morph_clean_mask` and update all call sites. `(mask_generator.py:27)`

- [ ] **mask_generator kernel labels**: Add `# EMPIRICAL` labels to
  `close_kernel_size = 11`, `dilate_size = 13`, `blur_size = 9`.
  `(mask_generator.py:37-39)`

- [ ] **Remove initialize_models**: Delete the dead `initialize_models` function from
  `detector.py` and remove it from `__init__.py`'s `__all__`.
  `(detector.py:86-107, __init__.py:14)`

- [ ] **IoU threshold empirical note**: Update the IoU comment to read "EMPIRICAL —
  validated on N=400 has-text-2 samples; broader validation pending".
  `(consensus.py:289)`

- [ ] **Consensus padding label**: Add `# EMPIRICAL` label to the 10%-per-side
  padding constant. `(consensus.py:397-417)`

- [ ] **process_with_known_mask API**: Document (or add a `WatermarkTemplate`
  overload to) `process_with_known_mask` so batch callers are not forced into
  O(N×templates) ORB extraction. `(orb_matcher.py:93)`

- [ ] **TELEA radius configurable**: Decide whether the TELEA inpaint radius (`3`)
  should be a CLI option, pipeline parameter, or documented constant with an
  `# EMPIRICAL` label. `(inpaint.py:338)`

- [ ] **Inpainting-context dilation**: Document why `64px` is the right dilation
  for LaMa context, or make it configurable. Current comment is a TODO with no
  derivation. `(inpaint.py:404)`

---

## Backlog

- [ ] **Defensive copy in preprocessing**: Add a comment documenting the no-mutation
  assumption in `preprocess_image`; add `.copy()` at next refactor if the chain
  gains any in-place step. `(pipeline.py:329)`

- [ ] **Triple color-enhance consolidation**: Consolidate the three sequential
  `_try_color_enhanced_detection` invocations into a single helper that tries all
  colors in one pass with early exit. `(pipeline.py:87-121)`

- [ ] **Color enhancement comment accuracy**: Rewrite the "converting [color] to
  black" comment to accurately describe CLAHE-on-contrast-enhanced-grayscale as
  the working mechanism. `(pipeline.py:359-366)`

- [ ] **coverage_limit call-site label**: Add a brief `# EMPIRICAL` inline note at
  the `coverage_limit=0.06` default argument. `(pipeline.py:123-225)`

- [ ] **force_output double-load**: When `image is not None` in the `force_output`
  path, use the already-loaded image instead of reloading from disk. Requires a
  conditional (not a one-line swap — `image` is `None` in the no-templates path).
  `(cli.py:254-260)`

- [ ] **8-parameter report signature**: Refactor `_save_clean_timing_report` to take
  a config dict for its optional `target_color` and `forced_bbox` parameters.
  `(reports.py:51)`

- [ ] **Detector position filtering**: Convert the position-based filtering TODO to
  a tracked issue and remove the in-code placeholder. Either implement with tests
  and documented evidence, or remove it. `(detector.py:685)`

- [ ] **EAST inner loop vectorization**: Vectorize the pure-Python trig loop in
  `_decode_east_predictions` using numpy slice operations. `(detector.py:438-473)`

- [ ] **LaMa health check cost**: Replace the full forward-pass health check with a
  cheaper attribute-existence check. `(inpaint.py:47-78)`

- [ ] **Inlier count in timing dict**: Propagate the ORB inlier count into the timing
  dict so reports can distinguish marginal from strong matches. `(known_mask.py:78)`

- [ ] **load_checkpoint no-op**: Replace `load_checkpoint = load_checkpoint` with an
  explanatory comment. `(lama_inpainter.py:43)`

- [ ] **Defer CUDA synchronize to batch loop**: Move `torch.cuda.synchronize()` from
  per-image to per-batch in `cli.py`. `(lama_inpainter.py:317-320)`

- [ ] **3D boolean mask allocation**: Replace the H×W×3 boolean mask in
  surrounding-region extraction with index-based logic. `(find_text_colors.py:440-442)`

- [ ] **Double ORB extraction in discovery**: Change `_candidate_meets_consensus_minimums`
  to return pre-built variants instead of discarding them. `(orb_prep.py:159-161)`

- [ ] **Silent metadata drop on save**: Add `logger.debug` when `save_image` falls
  through from Pillow to `cv2.imwrite`, noting that ICC profile and DPI are dropped.
  `(utils.py:321-332)`

- [ ] **build_final_templates output size**: Confirm that `O(clusters) + O(records)`
  output in `build_final_templates` is intentional (cascade design) and add a comment
  explaining the intent. `(watermark_consensus.py:1442-1451)`

- [ ] **Shape metrics are measured but not part of production FOM**: `metrics.py`
  provides `measure_blackhat_energy()` and `measure_edge_row_energy()`, but
  `find_mask_by_spatial_tf_idf()` still selects clusters using TF-IDF, border ratio,
  and connected-component fraction only. Evaluate whether these metrics improve mask
  quality before adding them to the scoring path.

- [ ] **Output-quality regression coverage is still indirect**: The suite has strong
  unit/workflow coverage but few image-in/image-out assertions for final visual quality.
  Add focused regression tests around known failure cases.

---

## Repo Hygiene

- [ ] **Ignored experiment artifacts need review.**
  `experiments/` contains local research inputs, outputs, and troubleshooting
  folders. Decide what should be archived, documented, promoted into tests, or
  deleted.

- [ ] **Tracked docs should stay authoritative.**
  Keep research/problem docs that describe current reasoning, but do not check in
  agent transcripts, implementation plans, or speculative scratch notes as docs.
  Those belong outside the repository or in ignored local scratch space.

- [ ] **Local run outputs should stay ignored.**
  Keep batch outputs such as `cleaned-*`, `scratch/`, `.codex-*`, and generated
  experiment output out of version control so agents do not confuse local artifacts
  with source-of-truth project files.
