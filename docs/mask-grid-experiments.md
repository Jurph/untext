# Mask Grid Experiments

This harness tunes the experimental `budgeted-regional` mask path against generated text fixtures without changing CLI or web defaults.

## Quick Verification

Run focused tests for the new harness and nearby mask behavior:

```powershell
uv run --extra web --with pytest pytest tests/test_mask_experiments.py tests/test_mask_experiment_scripts.py tests/test_pipeline.py tests/test_find_text_colors.py -q
```

Run the broad non-slow suite:

```powershell
uv run --extra web --with pytest pytest -m "not slow" -q
```

Run the focused generated-text slow checks:

```powershell
uv run --extra web --with pytest pytest tests/test_generated_text_benchmark.py::test_generated_text_modes_match_pipeline_semantics -q
uv run --extra web --with pytest pytest tests/test_generated_text_cases.py::test_budgeted_regional_generated_text_fixture_metrics -q
```

## Fast Smoke Grid

Use `--limit-configs` while checking the plumbing:

```powershell
New-Item -ItemType Directory -Force -Path experiments\mask-grid | Out-Null

uv run python scripts\run_mask_grid.py `
  --preset local-cleanup `
  --limit-configs 3 `
  --out experiments\mask-grid\local-cleanup-smoke.jsonl

uv run python scripts\summarize_mask_grid.py `
  experiments\mask-grid\local-cleanup-smoke.jsonl `
  --out experiments\mask-grid\local-cleanup-smoke.csv `
  --top-json experiments\mask-grid\local-cleanup-smoke-top.json
```

Optional masks for visual inspection:

```powershell
uv run python scripts\run_mask_grid.py `
  --preset local-cleanup `
  --limit-configs 3 `
  --out experiments\mask-grid\local-cleanup-smoke.jsonl `
  --save-images experiments\mask-grid\local-cleanup-smoke-masks
```

## Phased Sweep

Run phases separately rather than one large Cartesian sweep:

```powershell
uv run python scripts\run_mask_grid.py --preset local-cleanup --out experiments\mask-grid\local-cleanup.jsonl
uv run python scripts\summarize_mask_grid.py experiments\mask-grid\local-cleanup.jsonl --out experiments\mask-grid\local-cleanup.csv --top-json experiments\mask-grid\local-cleanup-top.json

uv run python scripts\run_mask_grid.py --preset color-proposal --out experiments\mask-grid\color-proposal.jsonl
uv run python scripts\summarize_mask_grid.py experiments\mask-grid\color-proposal.jsonl --out experiments\mask-grid\color-proposal.csv --top-json experiments\mask-grid\color-proposal-top.json

uv run python scripts\run_mask_grid.py --preset geometry-budget --out experiments\mask-grid\geometry-budget.jsonl
uv run python scripts\summarize_mask_grid.py experiments\mask-grid\geometry-budget.jsonl --out experiments\mask-grid\geometry-budget.csv --top-json experiments\mask-grid\geometry-budget-top.json
```

Then run inpaint evaluation on a top-config JSON:

```powershell
uv run python scripts\run_inpaint_eval.py `
  --configs experiments\mask-grid\geometry-budget-top.json `
  --method telea `
  --out experiments\mask-grid\geometry-budget-inpaint-telea.jsonl
```

LaMa can be run the same way if the local model path and dependencies are ready:

```powershell
uv run python scripts\run_inpaint_eval.py `
  --configs experiments\mask-grid\geometry-budget-top.json `
  --method lama `
  --out experiments\mask-grid\geometry-budget-inpaint-lama.jsonl
```

## How It Works

`scripts/run_mask_grid.py` loads `tests/images/generated_text_watermarks/manifest.json` by default. For each manifest case and each config in the selected preset, it:

1. Reads the watermarked image and truth mask.
2. Uses the fixture bbox as `forced_bbox`, so the experiment measures mask quality rather than detector recall.
3. Calls `process_image_array(..., use_budgeted_expand=True, mask_config=...)`.
4. Scores the predicted mask against `truth_mask` dilated by 2px with an elliptical kernel.
5. Writes one JSONL row per case/config.

The mask metrics include:

- `target_iou`, `target_precision`, `target_recall`
- `coverage`
- `overmask_ratio = predicted_px / target_px`
- `fp_inside_bbox`, `fp_outside_bbox`, `fp_outside_fraction`
- `weighted_precision`, where false positives farther from `truth+2px` cost more
- `score = target_recall^4 * weighted_precision^3 / sqrt(overmask_ratio)`

`scripts/summarize_mask_grid.py` groups JSONL rows by `config_id`, applies hard filters, ranks by mean score, writes a CSV summary, and writes the top configs to JSON.

Hard filters:

- mean recall >= 0.98
- min recall >= 0.95
- max coverage <= 0.06

`scripts/run_inpaint_eval.py` currently evaluates oracle `truth+2px` masks for top-config rows. It reports local-window SSIM and LAB-MAE gain ratios using TELEA or LaMa.

## Config Dials

The harness uses `MaskExperimentConfig` in `untextre/mask_experiments.py`. Config values are passed through the pipeline only when the experiment scripts supply `mask_config`; existing defaults remain reachable for production calls.

The current presets are:

- `local-cleanup`: cleanup dilation/closing, FOM threshold, CC guard
- `color-proposal`: foreground/background radius multipliers
- `geometry-budget`: bbox expansion, inside rejection credit, budget cap, long/short axis costs, connectivity dilation, min component size
- `final-neighborhood`: placeholder neighborhood generator around the first few local-cleanup configs; replace this with top-config neighborhoods before relying on it for final selection

## Outputs

By default the scripts produce structured artifacts only:

- per-case JSONL from `run_mask_grid.py`
- ranked CSV from `summarize_mask_grid.py`
- top-config JSON from `summarize_mask_grid.py`
- inpaint JSONL from `run_inpaint_eval.py`

Images are written only when `run_mask_grid.py --save-images <dir>` is provided.

## Notes

- `MASK_MODE_CHOICES` intentionally remains public-default only: `regional`, `local-shape`, `local-color`.
- `budgeted-regional` is available as an internal pipeline mode for tests/experiments.
- The generated fixture files are currently local/untracked in this checkout; keep the manifest and images together when moving machines.
