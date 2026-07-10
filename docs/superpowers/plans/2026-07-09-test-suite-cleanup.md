# Test Suite Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove low-value tests from `untext` and preserve only behavior-level coverage that protects real user-facing or regression-critical contracts.

**Architecture:** Execute a risk-ranked cleanup pass. Delete plumbing/shape/internal-choreography tests first, rewrite only when deleting would leave an external contract unprotected, and verify each cluster with targeted pytest commands before moving on. Keep strong E2E, corpus, geometry, replay, and artifact-generation tests intact.

**Tech Stack:** Python, pytest, uv-managed virtualenv, OpenCV/numpy image-processing test suite.

## Global Constraints

- Run commands from `untext/` repo root.
- Use `uv run pytest`, not system Python.
- Default bias is **aggressive deletion**.
- Do **not** rewrite a weak test unless deleting it would remove the last protection for a real external contract.
- Do **not** broaden production-code scope unless a replacement behavior test reveals a real gap.
- After each task, run only the touched test modules.
- Final verification must include all touched test modules together.
- Keep strong E2E / corpus / fixture / replay tests unless the plan explicitly deletes one.

---

### Task 1: Clean up inpainting internals

**Files:**
- Modify: `tests/test_inpaint.py`
- Modify: `tests/test_lama_inpainter.py`
- Modify: `tests/test_model_lifecycle.py`

**Interfaces:**
- Consumes: Existing public seams `inpaint_image(...)`, `LamaInpainter.inpaint(...)`, `paste_subregion(...)`
- Produces: Inpainting suite focused on public behavior; internal cache/init choreography tests removed or compressed

- [ ] **Step 1: Delete pure internal-helper tests from `tests/test_inpaint.py`**

Delete these groups entirely:

```text
TestHasPixelsToInpaint
  - test_empty_mask_returns_false
  - test_non_empty_mask_returns_true
  - test_single_white_pixel

TestCalculateInpaintingSubregion
  - test_empty_mask_returns_none
  - test_centered_mask_returns_valid_subregion
  - test_subregion_within_image_bounds
  - test_subregion_dimensions_are_mod8
  - test_mask_near_edge_clamps_to_bounds
  - test_mask_near_origin_clamps_to_zero
  - test_full_image_mask

TestInpaintWithTelea
  - test_basic_inpainting
  - test_3channel_mask_handled

TestInpaintWithLamaRecovery
  - test_uninitialized_and_no_retry_raises
  - test_retry_reinitialize_then_succeeds
  - test_retry_reinitialize_failure_raises

TestLamaHealthAndReset
  - all tests

TestInitializeLamaModel
  - all tests

TestInpaintWithLamaBranches
  - all tests

TestInpaintWithTeleaBranches
  - test_empty_processed_mask_returns_copy
```

- [ ] **Step 2: Trim plumbing-only status tests from `tests/test_inpaint.py`**

Delete these tests:

```text
- test_is_lama_available_returns_bool
- test_is_lama_initialized_returns_bool
- test_get_lama_status_returns_dict
- test_status_keys_consistent_with_helpers
- test_telea_output_shape_matches_input
```

Keep these behavior tests:

```text
- test_invalid_method_raises
- test_empty_mask_returns_copy
- test_telea_inpainting_modifies_masked_region
- test_telea_preserves_unmasked_region
- test_telea_restores_shrub
- test_telea_preserves_unmasked
- test_lama_restores_shrub
- test_lama_preserves_unmasked
- test_both_methods_beat_stamped_baseline
```

- [ ] **Step 3: Trim `tests/test_lama_inpainter.py` down to public behavior**

Delete these low-value tests:

```text
- test_inpainter_initialization
- test_3d_mask_auto_converts
- test_basic_inpaint_returns_correct_shape
- test_subregion_inpaint_returns_full_size
- test_subregion_touching_top_left_edge_pads
- test_subregion_touching_bottom_right_edge_pads
- test_subregion_touching_all_edges
- test_simplelama_output_cropped_when_padded
- test_subregion_output_size_mismatch_resized
```

Keep these behavior tests:

```text
- test_inpaint_single_image
- test_inpaint_with_subregion
- test_inpaint_with_invalid_input
- test_inpaint_with_invalid_subregion
- all `TestSelectDevice` tests
- constructor-guard tests
- validation tests that raise on bad public inputs
- test_inpaint_error_raises_and_cleans_up
- paste_subregion tests
```

- [ ] **Step 4: Delete model-lifecycle choreography tests**

Delete all tests from `tests/test_model_lifecycle.py`:

```text
- test_initialize_consensus_models_keeps_existing_yolo11x_instances
- test_initialize_consensus_models_uses_yolo11x_detector_cache
- test_reset_lama_model_drops_model_before_cuda_cache_cleanup
```

They pin internal cache orchestration, not external behavior.

- [ ] **Step 5: Run targeted verification**

```bash
uv run pytest tests/test_inpaint.py tests/test_lama_inpainter.py tests/test_model_lifecycle.py -q
```

Expected: pass with the reduced suite. Any failure means a deleted test was supporting a real contract indirectly; add a public-seam replacement before continuing.

- [ ] **Step 6: Commit**

```bash
git add tests/test_inpaint.py tests/test_lama_inpainter.py tests/test_model_lifecycle.py
git commit -m "test: prune inpainting internal and plumbing coverage"
```

---

### Task 2: Clean up detection internals and smoke tests

**Files:**
- Modify: `tests/test_detection.py`
- Modify: `tests/test_detector_unit.py`
- Modify: `tests/test_detector_lazy_import.py`
- Modify: `tests/test_east_model_download.py`
- Modify: `tests/test_detector_east_nms.py`
- Modify: `tests/test_detector_parse.py`
- Keep mostly intact: `tests/test_detector_pair_harvest.py`

**Interfaces:**
- Consumes: public seams `detect_with_doctr`, `detect_with_easyocr`, `detect_with_east`, `detect_text_regions`, harvest script outputs
- Produces: detector suite centered on region behavior and evidence-row outputs, not loader/cache/logging plumbing

- [ ] **Step 1: Delete detector-internal loader/cache tests**

Delete all tests from these files:

```text
tests/test_detector_lazy_import.py
tests/test_east_model_download.py
tests/test_detector_east_nms.py
tests/test_detector_parse.py
```

These are mostly private-loader or internal-call choreography tests.

- [ ] **Step 2: Reduce `tests/test_detector_unit.py` to real public-input validation only**

Delete these tests:

```text
- test_default_construction
- test_returns_list
- test_detection_dict_has_required_keys
- test_blank_image_produces_few_detections
- test_confidence_values_in_range
- test_returns_list_of_tuples
- test_blank_image_returns_few_boxes
- test_does_not_raise
```

Keep these:

```text
- test_confidence_out_of_range_raises
- test_negative_confidence_raises
- test_non_positive_min_text_size_raises
```

- [ ] **Step 3: Trim `tests/test_detection.py` smoke/plumbing tests**

Delete these tests:

```text
- test_color_enhanced_detection_pipeline
- test_doctr_detector_on_null_image
- test_easyocr_detector_on_null_image
- test_east_detector_on_null_image
- test_consensus_detection_on_null_image
- test_consensus_detection_on_test_image
- test_find_consensus_boxes_function
- test_all_detectors_on_test_image
```

Keep these:

```text
- all color-enhancement behavior tests at the top of the file
- test_detection_confidence_thresholds
- test_detector_consistency_across_runs
```

- [ ] **Step 4: Preserve `tests/test_detector_pair_harvest.py` except for no-op if all assertions are still behavior-level**

Do not change that file unless you find a clearly internal-only test during execution review. It was the strongest part of the detection cluster.

- [ ] **Step 5: Run targeted verification**

```bash
uv run pytest tests/test_detection.py tests/test_detector_unit.py tests/test_detector_pair_harvest.py -q
```

Expected: pass. If a removed internal file was the last guard for a public behavior, add a new behavior test in `test_detection.py` or `test_detector_pair_harvest.py`, not a replacement private-helper test.

- [ ] **Step 6: Commit**

```bash
git add tests/test_detection.py tests/test_detector_unit.py tests/test_detector_pair_harvest.py tests/test_detector_lazy_import.py tests/test_east_model_download.py tests/test_detector_east_nms.py tests/test_detector_parse.py
git commit -m "test: remove detector internal and smoke-only coverage"
```

---

### Task 3: Rewrite the discovery `_consensus_vote` block around returned behavior

**Files:**
- Modify: `tests/test_discovery.py`

**Interfaces:**
- Consumes: discovery public/observable outputs from `_consensus_vote` and `discover(...)`
- Produces: discovery tests that assert returned crops/templates, not internal `CandidateMetadata` packaging

- [ ] **Step 1: Delete the internal-record-packaging tests**

Delete or replace these tests called out in the review:

```text
- test_select_candidate_components_avoids_full_frame_blob_per_component
- test_consensus_vote_delegates_cross_bucket_candidates_to_graph_builder
- test_consensus_vote_same_bucket_different_zone_remains_separate_input
- test_consensus_vote_singleton_returned_through_graph_builder
- test_consensus_vote_returns_stingy_and_generous_outputs
- test_consensus_vote_skips_tiny_candidates_before_graph_builder
- test_consensus_vote_skips_sparse_spray_candidates_before_graph_builder
- test_consensus_vote_skips_orb_unusable_candidate_before_graph_builder
- test_consensus_vote_splits_one_raw_candidate_into_multiple_records
- test_consensus_vote_logs_prep_cutline_before_graph_scoring
- test_discover_two_pass_processes_all_qualifying_buckets
```

- [ ] **Step 2: Add behavior-level replacements only where coverage would otherwise vanish**

Add or rewrite tests so they assert outcomes like this instead of internal packaging:

```python
def test_consensus_vote_returns_two_crops_for_two_supported_regions():
    crops = _consensus_vote(...)
    assert len(crops) == 2
    assert all(crop.shape[2] == 4 for crop in crops)
    assert all(np.any(crop[:, :, 3] > 0) for crop in crops)


def test_consensus_vote_drops_tiny_candidate_from_output():
    crops = _consensus_vote(...)
    assert len(crops) == 1
    assert np.any(crops[0][:, :, 3] > 0)
```

Use real returned crops; do not patch `build_final_templates` or inspect internal record metadata.

- [ ] **Step 3: Delete pure plumbing checks in the same file when covered elsewhere**

Delete these if still present unchanged:

```text
- test_compute_stack_statistics_returns_required_keys
- test_discover_finds_watermark_in_homogeneous_batch (if still shape-only)
- test_discover_returns_empty_for_no_common_pixels (if still only isinstance/loose ceiling)
- test_discover_finds_candidate_in_watermark_zone (if still only any-alpha)
- test_discover_saves_debug_variance_map (if still only file/shape)
- test_discover_saves_debug_mean_image (if still only file/shape)
```

- [ ] **Step 4: Run targeted verification**

```bash
uv run pytest tests/test_discovery.py -q
```

Expected: pass. If replacing `_consensus_vote` wiring tests reveals missing coverage, keep the replacement strictly at the returned-crop seam.

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery.py
git commit -m "test: replace discovery wiring assertions with behavior tests"
```

---

### Task 4: Prune CLI/parser/logging/config plumbing

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_logging.py`
- Modify: `tests/test_streamlit_helpers.py`
- Modify: `tests/test_generated_text_benchmark.py`
- Modify: `tests/test_mask_experiments.py`

**Interfaces:**
- Consumes: CLI/user-facing parser behavior, output-artifact behavior, streamlit helper geometry, generated-text benchmark entry points
- Produces: fewer config/topology tests; preserved file/output/geometry behavior coverage

- [ ] **Step 1: Trim parser-default/config registration tests from `tests/test_cli.py`**

Delete these tests:

```text
- test_required_args_present
- test_defaults
- test_mask_mode_choices
- test_invalid_mask_mode_rejected
- test_confidence_threshold_default
- test_paint_choices
- test_invalid_paint_choice_rejected
- test_granularity_parsed_as_int
- test_force_bbox_is_raw_string
- test_boolean_flags
- test_force_output_default_false
- test_force_output_flag
- test_known_mask_parsed
- test_maskfile_is_long_only
- test_unknown_watermark_flag_exists
- test_package_lazy_imports
```

Keep these:

```text
- force-bbox validation tests
- main same-dir guard
- timing/report artifact tests
- real output-copy behavior tests
```

- [ ] **Step 2: Delete logging-topology tests**

Delete all tests from `tests/test_logging.py`.

If you feel one ASCII-output check must survive, replace the whole file with a single public-seam smoke test, not handler-topology assertions.

- [ ] **Step 3: Trim streamlit helper plumbing**

Delete these tests from `tests/test_streamlit_helpers.py`:

```text
- test_returns_dict_with_objects
- test_rect_type
```

Keep the coordinate-conversion and roundtrip behavior tests.

- [ ] **Step 4: Delete config-constant plumbing in benchmark helpers**

Delete:

```text
tests/test_generated_text_benchmark.py
- test_generated_text_modes_match_pipeline_semantics

tests/test_mask_experiments.py
- test_preset_configs_have_stable_ids_and_ordering   (only if not treated as external API)
- test_mask_experiment_config_serializes_all_grid_dials
- test_public_mask_mode_choices_remain_unchanged
```

- [ ] **Step 5: Run targeted verification**

```bash
uv run pytest tests/test_cli.py tests/test_streamlit_helpers.py tests/test_generated_text_benchmark.py tests/test_mask_experiments.py tests/test_mask_experiment_scripts.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_cli.py tests/test_logging.py tests/test_streamlit_helpers.py tests/test_generated_text_benchmark.py tests/test_mask_experiments.py tests/test_mask_experiment_scripts.py
git commit -m "test: prune cli logging and config plumbing coverage"
```

---

### Task 5: Remove arithmetic tautologies and low-value helper smoke from healthy clusters

**Files:**
- Modify: `tests/test_find_text_colors.py`
- Modify: `tests/test_consensus.py`
- Modify: `tests/test_mask_experiments.py`
- Modify: `tests/test_preprocessor.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_mask_generator.py`
- Modify: `tests/test_watermark_consensus.py`
- Modify: `tests/test_inmemory_watermark_benchmark.py`
- Modify: `tests/test_generated_text_cases.py`

**Interfaces:**
- Consumes: healthy pure-function and benchmark clusters
- Produces: same strong coverage with tautologies and helper/config barnacles removed

- [ ] **Step 1: Replace or delete the tautological arithmetic tests**

In `tests/test_find_text_colors.py`, rewrite these to independent literals or delete if redundant:

```text
- test_all_zeros_gives_maximum
- test_perfect_text_cluster
- test_solid_blob_rejected
- test_border_heavy_cluster
- test_border_ratio_above_one_clamped
- test_cc_fraction_above_one_clamped
```

In `tests/test_consensus.py`, delete or rewrite these with invariant-based assertions instead of formula restatement:

```text
- test_two_high_confidences
- test_two_low_confidences
- test_three_detectors
```

In `tests/test_mask_experiments.py`, rewrite:

```text
- test_truth_target_mask_uses_deterministic_elliptical_dilation
```

using a hand-worked expected mask, not `cv2.dilate(...)` in the test.

- [ ] **Step 2: Delete low-value helper/config smoke tests**

Delete these tests if they still exist unchanged:

```text
tests/test_preprocessor.py
- test_bgr_input_returns_rgb
- test_grayscale_input_returns_rgb
- test_output_not_same_object

tests/test_metrics.py
- test_returns_float (blackhat)
- test_non_negative (blackhat)
- test_returns_float (edge_row)
- test_bounded_zero_to_one
- test_returns_tuple_of_four
- threshold-constant sanity tests

tests/test_mask_generator.py
- test_output_is_binary
- test_preserves_shape

tests/test_watermark_consensus.py
- test_candidate_record_separates_pixels_from_metadata
- test_consensus_module_logger_is_configured_for_info_output
- test_geometry_helpers_return_nonempty_fields_for_simple_blob
- test_build_candidate_record_constructs_geometry
- test_build_candidate_graph_logs_pair_workload_and_progress

tests/test_inmemory_watermark_benchmark.py
- test_run_inmemory_watermark_benchmark_cli_writes_json_artifacts

tests/test_generated_text_cases.py
- helper-coupled metadata tests that only pin `_choose_*` internals
```

- [ ] **Step 3: Preserve the strong tests explicitly**

Do **not** delete these categories:

```text
- ORB / bbox worked-example geometry tests
- benchmark determinism / replay tests
- mask ranking / scoring behavior tests
- watermark consensus fixture and E2E tests
- corpus-wide generated-text / in-memory benchmark behavior tests
```

- [ ] **Step 4: Run targeted verification**

```bash
uv run pytest tests/test_find_text_colors.py tests/test_consensus.py tests/test_mask_experiments.py tests/test_preprocessor.py tests/test_metrics.py tests/test_mask_generator.py tests/test_watermark_consensus.py tests/test_inmemory_watermark_benchmark.py tests/test_generated_text_cases.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_find_text_colors.py tests/test_consensus.py tests/test_mask_experiments.py tests/test_preprocessor.py tests/test_metrics.py tests/test_mask_generator.py tests/test_watermark_consensus.py tests/test_inmemory_watermark_benchmark.py tests/test_generated_text_cases.py
git commit -m "test: remove arithmetic tautologies and helper smoke coverage"
```

---

### Task 6: Final verification and cleanup summary

**Files:**
- Modify: `cardinal/reviews/pocock-unit-test-review/00-summary.md` (optional only if you want to append execution notes)
- No required source changes

**Interfaces:**
- Consumes: all prior cleanup commits
- Produces: verified reduced suite and a human-readable summary of what was removed vs rewritten

- [ ] **Step 1: Run the full touched-module verification sweep**

```bash
uv run pytest \
  tests/test_inpaint.py \
  tests/test_lama_inpainter.py \
  tests/test_model_lifecycle.py \
  tests/test_detection.py \
  tests/test_detector_unit.py \
  tests/test_detector_pair_harvest.py \
  tests/test_discovery.py \
  tests/test_cli.py \
  tests/test_streamlit_helpers.py \
  tests/test_generated_text_benchmark.py \
  tests/test_mask_experiments.py \
  tests/test_mask_experiment_scripts.py \
  tests/test_find_text_colors.py \
  tests/test_consensus.py \
  tests/test_preprocessor.py \
  tests/test_metrics.py \
  tests/test_mask_generator.py \
  tests/test_watermark_consensus.py \
  tests/test_inmemory_watermark_benchmark.py \
  tests/test_generated_text_cases.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Produce a short execution summary for Jurph**

Write a summary in the PR/hand-off notes with this structure:

```text
Removed:
- internal helper tests
- parser/config/logger plumbing
- arithmetic tautologies

Rewritten:
- discovery `_consensus_vote` assertions (if changed)
- any public-seam replacements added during cleanup

Preserved intentionally:
- E2E / corpus / replay / ORB geometry / benchmark artifact tests
```

- [ ] **Step 3: Final commit**

```bash
git add tests
git commit -m "test: excise low-value coverage across suite"
```

---

## Self-Review

**Spec coverage:**
- Aggressive deletion bias: covered in all tasks
- Risk-ranked ordering: Tasks 1–5 follow approved order
- Rewrite only when contract would be lost: repeated in Global Constraints and each task
- Preserve strong E2E/corpus/geometry/benchmark tests: called out explicitly in Tasks 1, 2, 4, and 5

**Placeholder scan:**
- No `TODO` / `TBD`
- Each task has exact file paths and named tests to delete/keep/rewrite
- Each task has an explicit verification command and commit step

**Type / interface consistency:**
- Public seams referenced consistently: `inpaint_image`, `LamaInpainter.inpaint`, detector wrappers, `find_consensus_boxes`, `process_image_array`, `process_single_image`
- No later task depends on renamed interfaces from earlier tasks
