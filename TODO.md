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
  ratio-test constant. `(sift_matcher.py:18)`

- [x] **50% coverage guard comment**: Add one sentence explaining why a watermark
  covering more than 50% of the image is treated as spurious. `(sift_matcher.py:329-334)`

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

- [x] **process_with_known_mask API** (#8): Added reusable target-SIFT feature
  preparation so multi-template known-mask cascades extract the target image’s
  SIFT keypoints/descriptors once per image instead of once per template.

---

## Backlog

- [ ] **Triple color-enhance consolidation** (#11): Consolidate the three sequential
  `_try_color_enhanced_detection` invocations into a single helper that tries all
  colors in one pass with early exit. `(pipeline.py:87-121)`

- [x] **force_output double-load** (#10): `force_output` now reuses the already-loaded
  `image` array from the template-cascade path instead of reloading from disk; falls
  back to `load_image(image_path)` only when no image was loaded (no-templates path).
  `(cli.py:239)`

- [ ] **LaMa health check cost** (#17): Replace the full forward-pass health check with a
  cheaper attribute-existence check. `(inpaint.py:47-78)`

- [x] **Inlier count in timing dict** (#13): Report the generic `feature_inliers`
  timing field so reports can distinguish marginal from strong template matches.
  `(reports.py, cli.py)`

- [ ] **3D boolean mask allocation** (#15): Replace the H×W×3 boolean mask in
  surrounding-region extraction with index-based logic. `(find_text_colors.py:440-442)`

- [ ] **Double feature extraction in discovery** (#14): Avoid computing SIFT
  features once for candidate viability and again when exported candidates are
  loaded for matching; reuse prepared candidate features where practical.
  `(sift_prep.py, discovery.py, reports.py)`

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

- [ ] **Post-`-U` candidate consolidation and corpus recall pass**: After unknown-mode
  discovery emits raw candidates, cluster candidates by detected position, geometry,
  and visual similarity; build one or more consensus candidates; then run a `-K`-style
  pass over the source corpus with those candidates, compare bbox placements, and
  overlay/align hits to see whether the consolidated candidate set can approach
  full capture.

- [ ] **Reject texture-rich stable-background false positives in discovery** (measured
  on `has-TYM-logo`, 2026-07-19): #1 (split-merge ratio tuning) and #3
  (`MAX_CONSENSUS_CANDIDATES_PER_ZONE`) were implemented and validated against SQ
  (13->14 templates, comparable quality) but had **zero effect** on TYM — both before
  and after tuning, discovery produced identically "16 zone candidate(s) -> 42 kept
  sub-candidates" -> 861 pairs. Root cause: TYM's candidate explosion is NOT
  glyph-fragmentation (what #1 targets). A visual audit of all 44 emitted TYM
  candidates found only 2 real watermark hits (`watermark_candidate_12.png` /
  `_39.png`, "THIS YEARS MODEL.COM", both matched with 70+ inliers in production);
  the other ~41 are large diffuse skin-tone/fabric-fold blobs and thin
  frame/border edges — stable, texture-rich, non-watermark regions that clear the
  SIFT-keypoint floor (`MIN_CONSENSUS_CANDIDATE_SIFT_KEYPOINTS`) because natural
  texture legitimately produces >5 keypoints. The discovery module's own docstring
  already predicts this failure mode ("consistently-lit or static backgrounds ...
  do NOT cancel under the median"). Needs a rejector for this specific case, but
  two candidate metrics were tried and **falsified against an independent set**
  (2026-07-19) — do not re-attempt either without new evidence:
  - `stroke_width_cv` (coefficient of variation of `cv2.distanceTransform` under
    the alpha mask; hypothesis: text/logo strokes have tight width distribution,
    texture blobs don't). Result: full overlap. Known-good marks in `watermarks/`
    span 0.487-0.847; TYM's true hit scored 0.68-0.69, inside that range but so
    did >30 of TYM's 42 false-positive candidates (0.17-1.06). Some texture blobs
    scored *lower* CV than real letterforms — letters have corners/serifs/junctions
    that add width variance; smooth diagonal skin/fabric blobs taper gradually and
    don't.
  - `color_std` under an eroded alpha mask (hypothesis: printed ink is flat-color,
    natural texture has real photographic shading). Result: `TYM-4.png` — an
    already-curated, hand-verified production template — scored `color_std=97.8`,
    statistically identical to the discovered true positive (104-108) and to
    several of TYM's false-positive candidates. A "reject high color_std" rule
    would reject a known-good template. Also found the metric itself is unreliable
    on sparse/scattered alpha masks: an isolated-speckle noise candidate scored
    `color_std=108` purely because its scattered specks sampled unrelated
    background colors, not because of any real ink/texture property.
  - Calibration data: `watermarks/*.png` (independent curated ground truth,
    25 marks) vs. all 44 TYM discovered candidates (hand-labeled from visual
    audit) vs. SQ's discovered candidates. Neither metric separated cleanly on
    the independent `watermarks/` set even before touching the TYM/SQ data —
    that's what killed both, not sample-size noise.
  - Next attempt should probably not be simple per-pixel/per-shape statistics;
    consider a proper texture descriptor (LBP/GLCM contrast+homogeneity,
    frequency-domain high/low energy ratio) or a small learned classifier over
    labeled discovered-candidate examples, and validate the SAME way — against
    an independent known-good set, not just the diagnostic sample that motivated
    the idea.
  `(discovery.py: _candidate_meets_consensus_minimums, extract_watermark_colors)`

- [ ] **Cheap pairwise pre-filter before full scale-ladder search**: `score_candidate_pair`
  (`watermark_consensus.py`) runs a full coarse+refine multi-scale search
  (`_stable_scale_ladder` + top-3 refine) for every candidate pair, both directions,
  with no cheap aspect/area-ratio bailout first. #1/#3 (above) did not reduce TYM's
  42-candidate/861-pair case, so this O(n^2) stage is still the bottleneck on noisy
  galleries (TYM: ~47 min end-to-end, dominated by the graph stage) until the
  texture-rejection item above lands. Add an early skip for pairs whose aspect
  ratio / area ratio is wildly incompatible before the ladder search runs.
  **Perceptual-hash Hamming distance was tried as this cheap pre-filter and
  falsified against real data (2026-07-19)** — see the standalone TODO item
  below for the full result; do not re-attempt hash-distance pruning without
  new evidence, since the failure mode (no scale/crop invariance) applies to
  any single fixed-size hash of the raw crop, not just pHash specifically.
  `(watermark_consensus.py:848-970)`

- [ ] **Cross-bucket recurrence as a weak-candidate admission signal**: Currently
  `_cap_zone_candidates` and the SIFT/structural floor in `_candidate_meets_consensus_minimums`
  judge each candidate in isolation. A candidate too weak to admit alone (low sift_kp,
  small area) but whose zone/bbox recurs at a consistent relative position across
  multiple independent buckets is likely a genuine — if small — piece of the same
  watermark (e.g. a period, a thin monogram stroke). `CROSS_BUCKET_IOU_THRESHOLD`
  already does this kind of cross-bucket matching, but only after the O(n^2) graph
  runs. Consider a lightweight pre-pass that lets recurrence override the per-candidate
  floor before the graph stage, operationalizing the "consistently placed, same
  watermark" hypothesis directly instead of relying purely on a per-candidate score.
  `(discovery.py: _consensus_vote, _cap_zone_candidates)`

- [ ] **Perceptual-hash-based (not random) split for cross-sub-sample validation**:
  `discover_watermark_candidates`'s cross-sub-sample check (`discovery.py:1109-1124`,
  `if len(paths) >= 6`) currently splits a bucket's images into two RANDOM halves
  and requires a pixel to be a Tukey-fence stable outlier in both halves before
  trusting it. Random splitting can still land two near-duplicate/highly-correlated
  images (same photoshoot, same pose, same background) in the same half, letting
  a coincidentally-stable background element (skin tone, fabric, floor) pass the
  cross-validation just because its correlated source images weren't separated.
  Instead, compute a perceptual hash (`cv2.img_hash`) per image in the bucket
  and split by hash DISSIMILARITY instead of random.

  **Tested against real data (2026-07-19), corrected 2026-07-19 after finding a
  hand-labeling error (see below) — leaning positive for `greedy_diverse`, not
  negative as first reported.** Compared random (production's exact seeded
  split) vs. k-means-stratified vs. greedy max-min-diversity splits, on 5 real
  buckets across TYM/SQ/PB with known watermark zones, holding one fixed Tukey
  threshold per bucket so only the split construction varied. Scored by
  stable-pixel mass inside the known zone vs. everywhere else:
  - `PB 4002x6000` (n=18): greedy_diverse won — out-zone noise 23046->17167
    (~25% down) while true-zone mass held (62204->62178).
  - `SQ 2832x4240` (n=19), `SQ 4240x2832` (n=9): no meaningful difference
    between strategies, or a slight regression for greedy_diverse.
  - `TYM 5760x3840` (n=6): no noise present to reject either way; k-means
    degenerated outright (SKIP) — k-means has a real robustness problem at
    small n, a hazard for a pipeline that must handle whatever bucket sizes
    a directory hands it.
  - `PB 5464x8192` (n=12): **originally reported as a severe loss — that was
    wrong, caused by a hand-labeling error** (the "known-good zone" was
    entered as (0,0) when the real watermark, verified by cropping the same
    pixel region from all 12 source images and visually confirming the same
    Playboy bunny silhouette in every one, is actually zone (1,2); (0,0) is
    the noise). With the corrected zone: `greedy_diverse` in-zone fraction
    was 74.8%, `kmeans` 51.8%, `random` 26.7% — a real, monotonic win for
    the more-diverse splits in this bucket, not a loss.
  So: 2 of 5 buckets show a genuine, meaningful win for `greedy_diverse`
  (PB's both tested buckets); 2 show no difference or a mild regression
  (SQ); 1 has no noise to discriminate on (TYM). `greedy_diverse` no longer
  looks like a high-variance coin flip — it was consistently at least as
  good as random and sometimes much better, once the ground truth was
  fixed. `kmeans`' small-n degeneration is still a real, separate
  robustness gap to fix before adopting either alternative by default.
  Harness for this test: `experiments/split_strategy_step{1,2,3}_*.py`
  (manifest -> hash+split -> score); `experiments/crop_zone_across_bucket.py`
  is the direct-visual-verification tool that caught the labeling error —
  reuse it to sanity-check any "known good zone" ground truth before trusting
  a comparison built on it.

- [x] **IMPLEMENTED (2026-07-20): self-consistent per-half Tukey threshold is
  now the production cross-sub-sample path.** SG falsification check ran
  first (4 more buckets, 3rd independent gallery): self-consistent threshold
  matched or beat the shared/pooled one in every SG bucket, including the
  largest single-bucket win seen yet (`SG 6000x4000`: 69.0-84.9% ->
  99.3-100% in-zone concentration across all three split strategies). Did
  not falsify — shipped. `discovery.py:1109-1138` now computes
  `threshold_a`/`threshold_b` from each half's own log-variance distribution
  instead of reusing `global_stable_threshold`. Verified: unit test
  (`test_cross_sub_sample_per_half_thresholds_differ_with_noise`) locks the
  mechanism; integration test
  (`test_discover_finds_watermark_with_noisy_competing_region_at_cross_sample_boundary`)
  covers the changed code path end-to-end; full suite (375 passed, 2
  skipped) shows no regression; real (unmonkeypatched) CLI `-U` smoke test
  on `SG 6000x4000` confirmed the new debug log line
  (`self-consistent thresholds a=1.98e-04, b=1.12e-03` — genuinely
  different values) and produced a single, exceptionally clean candidate
  matching all 7 images at 192-329 SIFT inliers each (well above the
  typical range).

- [x] **Cross-sub-sample threshold mismatch: global threshold applied to
  half-sized samples — bigger effect than split strategy, and a CLEAN win
  where tested** (measured 2026-07-19, corrected 2026-07-19 after the same
  hand-labeling fix as above). `discovery.py`'s real cross-sub-sample
  validation (`discovery.py:1109`) applies `global_stable_threshold` —
  computed once in Pass 1 from POOLED, FULL-image-count per-pixel
  log-variance across all qualifying buckets — unchanged to
  `stats_a`/`stats_b`, which are HALF-sized-sample variance estimates.
  Re-ran the split-strategy comparison with each half instead getting its
  OWN self-consistent Tukey fence (from its own log-variance distribution)
  instead of the one shared/mismatched number:
  - `PB 4002x6000`: ALL THREE split strategies converged to ~100%
    pct-in-zone (out-zone noise 17167-23046px -> 0-13px) with true-zone
    signal essentially unchanged (62178-62207 -> 61182-61750px).
  - `PB 5464x8192`: with the corrected zone (1,2), self-consistent
    thresholds ALSO achieve exactly 100% pct-in-zone in all three
    strategies (real-zone mass 18731-18940px fully preserved, the noise
    zone (0,0) reduced to 0px). Originally reported as the opposite
    (collapsing the "true" zone) — that reversal was entirely the
    hand-labeling error, not a real effect.
  - `SQ` (both buckets): modest, roughly symmetric reduction in both signal
    and noise; pct-in-zone stays ~71-73% either way, no dramatic effect.
  Conclusion, corrected: this mismatch is a real, LARGE effect, and in
  both PB buckets tested — the only two with meaningful out-zone noise to
  reject — a self-consistent per-half threshold achieved PERFECT (100%)
  separation regardless of split strategy. This is a much stronger, cleaner
  result than first reported, and is now the highest-priority item in this
  cluster: fixing the threshold-calibration mismatch looks like it could
  matter more than the split-strategy question, and split-strategy
  differences in the fixed-threshold runs may partly be explained by this
  confound. Still only 2 buckets with real signal to test against (SQ's
  buckets don't discriminate either way) — validate on more real galleries,
  and re-verify zone ground truth by direct crop inspection every time
  before trusting a result, before changing production behavior.
  Harness: `experiments/split_strategy_step3_score.py` (reports both
  `fixed_thr` and `self_thr` variants per strategy).
  `(discovery.py:1109-1124, _precision_outlier_threshold_from_log_precision)`

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
