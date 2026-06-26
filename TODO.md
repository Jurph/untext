# TODO

This file tracks current problems in the codebase. It should not duplicate the
README or preserve old completed review notes.

## Code Problems

- **TELEA radius is hardcoded.**
  `_inpaint_with_telea()` in `untextre/inpaint.py` always calls OpenCV TELEA with
  radius `3`. That may be fine, but callers cannot tune it for small text,
  large logos, or high-resolution masks. Decide whether this should become a CLI
  option, a pipeline parameter, or a documented constant.

- **Inpainting-context dilation is hardcoded.**
  `_calculate_inpainting_subregion()` in `untextre/inpaint.py` expands mask bounds
  by `64px`. That is a real algorithm choice hidden inside a helper. Large images,
  tiny crops, and dense masks may want different context sizes. Make it explicit
  or document why `64px` is the right invariant.

- **Detector position filtering is still a placeholder.**
  `untextre/detector.py` has a TODO for smarter position-based filtering/corner
  preference. Either implement it with tests and documented evidence, or remove
  the placeholder if position filtering is no longer part of the detector layer.

- **Shape metrics are measured but not part of production FOM.**
  `untextre/metrics.py` provides `measure_blackhat_energy()` and
  `measure_edge_row_energy()`, but `find_mask_by_spatial_tf_idf()` still selects
  clusters using TF-IDF, border ratio, and connected-component fraction only.
  Evaluate whether these metrics improve mask quality before adding them to the
  scoring path.

- **Output-quality regression coverage is still indirect.**
  The suite has strong unit/workflow coverage, but there are few image-in/image-out
  assertions for final visual quality. Add focused regression tests around known
  failure cases rather than broad snapshot tests.

## Repo Hygiene

- **Root-level utility scripts should not accumulate.**
  Standalone workflow helpers should live under a deliberate `tools/` directory
  or be deleted. The repository root should stay limited to package entrypoints,
  project metadata, and high-level docs.

- **Ignored experiment artifacts need review.**
  `experiments/` contains local research inputs, outputs, and troubleshooting
  folders. Decide what should be archived, documented, promoted into tests, or
  deleted.

- **Historical plans are stale by design.**
  Files under `docs/superpowers/plans/` and related specs are useful history, but
  many references predate the module split. Do not treat them as current
  architecture without checking the code.
