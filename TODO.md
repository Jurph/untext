# TODO

This file tracks current problems in the codebase. It should not duplicate the
README or preserve old completed review notes.

## Should Fix

- [x] **Lowe ratio-test citation**: Add a Lowe (2004) citation comment to the `0.75`
  ratio-test constant. `(orb_matcher.py:135)`

- [x] **50% coverage guard comment**: Add one sentence explaining why a watermark
  covering more than 50% of the image is treated as spurious. `(orb_matcher.py:252-256)`

- [x] **LaMa paste-back coordinates**: Extracted `paste_subregion(full_image, patch,
  x1, y1, x2, y2)` as the single source of truth for paste-back geometry (owns the
  resize-on-mismatch); the method now passes its existing edge-pad-adjusted
  `x1,y1,x2,y2`, and the duplicate `subregion[0] + edge_pad_*` recompute is deleted.
  Guarded by `TestPasteSubregion` (placement + resize). `(lama_inpainter.py)`

- [x] **edge_pad_size empirical label**: Add `# EMPIRICAL` label (or architectural
  derivation) to `edge_pad_size = 32`. `(lama_inpainter.py:193-196)`

- [x] **Formula weight labels**: Add `# EMPIRICAL` labels to the floating-point
  weights in `_component_shape_weight` and `compatibility`.
  `(watermark_consensus.py:624-625, 780-787)`

- [x] **Minimum-n label**: Add `# EMPIRICAL — minimum n for Welford variance; broader
  validation pending` to the `n < 3` guard. `(discovery.py:349)`

- [x] **metrics.py constant cross-references**: Add `granularity_experiment.py`
  cross-reference directly to each threshold constant's comment (not only the module
  docstring). `(metrics.py:27-32)`

- [x] **preprocessor.py dead alias**: Remove `result_img = image` alias; write
  `cv2.cvtColor(image, ...)` directly. `(preprocessor.py:44)`

- [ ] **preprocessor.py CLAHE/bilateral citation**: Update the module docstring and
  inline comments to name the dataset and define the FOM metric for the grid-search
  result. `(preprocessor.py:52-57)`

- [x] **morph_clean_mask dead bbox param**: Remove the unused `bbox: BBox` parameter
  from `morph_clean_mask` and update all call sites. `(mask_generator.py:27)`

- [x] **mask_generator kernel labels**: Add `# EMPIRICAL` labels to
  `close_kernel_size = 11`, `dilate_size = 13`, `blur_size = 9`.
  `(mask_generator.py:37-39)`

- [x] **Remove initialize_models**: Delete the dead `initialize_models` function from
  `detector.py` and remove it from `__init__.py`'s `__all__`.
  `(detector.py:86-107, __init__.py:14)`

- [x] **IoU threshold empirical note**: Update the IoU comment to read "EMPIRICAL —
  validated on N=400 has-text-2 samples; broader validation pending".
  `(consensus.py:289)`

- [x] **Consensus padding label**: Add `# EMPIRICAL` label to the 10%-per-side
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

- [ ] **Direct CRAFT detector diversification**: Consider adding a direct CRAFT pass
  alongside EasyOCR. EasyOCR already uses OCR-oriented detection internally, but a
  standalone CRAFT detector may expose different boxes/thresholds for cursive,
  curved, or fragmented text and reduce detector misses.

- [ ] **DB++ detector diversification**: Consider adding DB++ via PaddleOCR or an
  available DocTR variant alongside the current DocTR DBNet detector. Validate that
  its errors are independent enough to improve consensus before accepting the
  dependency/runtime cost.

- [ ] **YOLOv8 watermark-detector tryout**: Benchmark
  [`mnemic/watermarks_yolov8`](https://huggingface.co/mnemic/watermarks_yolov8)
  against the current EAST/DocTR/EasyOCR panel on (a) the zero-corpus FP baseline
  (`tests/images/zero_detector_fp_baseline/`) and (b) the synthetic watermark
  benchmark, measuring recall, FP rate, and per-image latency. Unlike the panel,
  it is watermark-specific rather than text-generic, so it may catch logo marks
  the text detectors miss. If it dominates the slowest/least-accurate panel
  member, give it that seat; validate error independence before changing the
  2-of-3 consensus rule.

- [ ] **Replace DocTR with YOLO11x in production consensus panel**: Ready-to-implement
  ticket at `docs/superpowers/specs/2026-07-07-replace-doctr-with-yolo11x.md`. Known-truth
  simulation of the actual production `find_consensus_boxes` algorithm (415-pair corpus)
  shows the swap improves recall 92.3%->96.1%, cuts false-positive incidents 3.6%->2.7%,
  cuts masked FP area 0.018%->0.015%, and cuts detector wall-clock ~5x (DocTR is 68x
  slower than YOLO11x per image). Every axis improves; no tradeoff. Reproduce with
  `scripts/simulate_production_consensus_swap.py`.

- [ ] **DocTR receives BGR input on the normal path**: `detector.py:599` feeds
  `load_image`'s BGR array straight into the DocTR predictor (trained on RGB);
  EAST (`swapRB=True`) and EasyOCR (`BGR2RGB`) both convert explicitly. A/B test
  (2026-07-06, 6 synthetic vivid-text cases + 6 real flagged photos, script at
  `.codex-tmp/doctr_ab/run_ab.py`) showed detection differences within noise:
  6/6 truth hits both arms, confidence deltas < 0.03, one threshold-straddling
  box flicker. DELIBERATELY left unchanged to keep the frozen FP baseline
  consistent (a KNOWN QUIRK comment now marks the call site). Revisit if DocTR
  is ever used for recognition (color-sensitive) or when thresholds are next
  recalibrated. The misleading signposts are fixed (2026-07-06): the
  preprocessor docstring now states the BGR-convention R==G==B contract, and
  the GRAY2RGB/GRAY2BGR alias confusion is documented at the conversion site.

- [ ] **CLIP/ViT semantic bbox adjudication**: Consider scoring detector-proposed
  bbox crops with a lightweight OpenCLIP ViT (`ViT-B-32`/`ViT-B-16`) against prompt
  axes such as text/logo/URL vs natural texture/clothing/grass/backdrop. Use the
  scores as voting/deconfliction features, not as a standalone detector, and validate
  whether they separate true overlay text from structured-background false positives.

- [ ] **Output-quality regression coverage is still indirect**: The suite has strong
  unit/workflow coverage but few image-in/image-out assertions for final visual quality.
  Add focused regression tests around known failure cases.

- [ ] **Promote has-text-2 bbox harvest into bbox/blob-selection tests**:
  `tests/images/has_text_2_pipeline_bbox_harvest/` contains a completed 405-image
  full-pipeline bbox-of-record harvest with review slots and overlays. It has strong
  candidates for bbox optimization and blob selection: 145 ambiguous cases, 58
  near/full-width expansions, 43 cases with >=4x bbox growth, 73 multi-bbox cases,
  16 failover-only bbox hits, and 201 unresolved controls. Use `review_template.json`
  to add human or multimodal ground-truth watermark bboxes before turning selected
  cases into regression fixtures.

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
