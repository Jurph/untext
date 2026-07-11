# TODO

This file tracks current problems in the codebase. It should not duplicate the
README or preserve old completed review notes.

**Source of truth:** the GitHub issue tracker (epic #31) is authoritative for
scope, priority, and status. Entries below that map to an open issue carry an
`(#N)` reference — check the issue, not this file, for current status. When an
issue closes, remove or check off the matching entry here in the same commit
so this file never drifts from the tracker.

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

- [ ] **process_with_known_mask API** (#8): Document (or add a `WatermarkTemplate`
  overload to) `process_with_known_mask` so batch callers are not forced into
  O(N×templates) ORB extraction. `(orb_matcher.py:93)`

---

## Backlog

- [ ] **Triple color-enhance consolidation** (#11): Consolidate the three sequential
  `_try_color_enhanced_detection` invocations into a single helper that tries all
  colors in one pass with early exit. `(pipeline.py:87-121)`

- [ ] **force_output double-load** (#10): When `image is not None` in the `force_output`
  path, use the already-loaded image instead of reloading from disk. Requires a
  conditional (not a one-line swap — `image` is `None` in the no-templates path).
  `(cli.py:254-260)`

- [ ] **LaMa health check cost** (#17): Replace the full forward-pass health check with a
  cheaper attribute-existence check. `(inpaint.py:47-78)`

- [ ] **Inlier count in timing dict** (#13): Propagate the ORB inlier count into the timing
  dict so reports can distinguish marginal from strong matches. `(known_mask.py:78)`

- [ ] **3D boolean mask allocation** (#15): Replace the H×W×3 boolean mask in
  surrounding-region extraction with index-based logic. `(find_text_colors.py:440-442)`

- [ ] **Double ORB extraction in discovery** (#14): Change `_candidate_meets_consensus_minimums`
  to return pre-built variants instead of discarding them. `(orb_prep.py:159-161)`

- [ ] **Direct CRAFT detector diversification** (#25): Consider adding a direct CRAFT pass
  alongside EasyOCR. EasyOCR already uses OCR-oriented detection internally, but a
  standalone CRAFT detector may expose different boxes/thresholds for cursive,
  curved, or fragmented text and reduce detector misses.

- [ ] **YOLOv8 watermark-detector tryout** (#22): Benchmark
  [`mnemic/watermarks_yolov8`](https://huggingface.co/mnemic/watermarks_yolov8)
  against the current EAST/EasyOCR/YOLO11x panel on (a) the zero-corpus FP baseline
  (`tests/images/zero_detector_fp_baseline/`) and (b) the synthetic watermark
  benchmark, measuring recall, FP rate, and per-image latency. Unlike the panel,
  it is watermark-specific rather than text-generic, so it may catch logo marks
  the text detectors miss. If it dominates the slowest/least-accurate panel
  member, give it that seat; validate error independence before changing the
  2-of-N consensus rule.

- [ ] **CLIP/ViT semantic bbox adjudication** (#28): Consider scoring detector-proposed
  bbox crops with a lightweight OpenCLIP ViT (`ViT-B-32`/`ViT-B-16`) against prompt
  axes such as text/logo/URL vs natural texture/clothing/grass/backdrop. Use the
  scores as voting/deconfliction features, not as a standalone detector, and validate
  whether they separate true overlay text from structured-background false positives.

- [ ] **Output-quality regression coverage is still indirect** (#21): The suite has strong
  unit/workflow coverage but few image-in/image-out assertions for final visual quality.
  Add focused regression tests around known failure cases.

- [ ] **Promote has-text-2 bbox harvest into bbox/blob-selection tests** (#19):
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
