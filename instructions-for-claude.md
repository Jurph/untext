# Mask Grid Overnight Instructions

This repo has a deterministic, LLM-free mask grid harness for probing `budgeted-regional` mask parameters against the generated-text fixtures.

## Current Goal

Create high-value overnight artifacts for a later high-end model to inspect:

- complete `local-cleanup` sweep
- complete `color-proposal` control sweep
- broad deterministic strided sample of `geometry-budget`
- summaries ranked by the existing hard filters and scoring
- optional TELEA eval for the top ranked configs

The main runner writes one JSONL row per fixture/config pair. There are 17 fixtures, so a shard with `N` configs should produce `17 * N` rows.

## Important Files

- `scripts/run_mask_grid.py`: runs a preset slice.
- `scripts/summarize_mask_grid.py`: aggregates JSONL rows into ranked CSV and top JSON.
- `scripts/run_inpaint_eval.py`: runs TELEA/LaMa eval from a top-config JSON.
- `scripts/launch_mask_overnight.ps1`: starts background PowerShell lane workers.
- `scripts/collect_mask_overnight.ps1`: audits shards, combines JSONL, summarizes, and optionally runs TELEA.
- `docs/mask-grid-experiments.md`: feature documentation and CLI examples.

`run_mask_grid.py` supports deterministic sharding:

```powershell
uv run python scripts\run_mask_grid.py `
  --preset geometry-budget `
  --config-start 1024 `
  --config-step 16 `
  --limit-configs 64 `
  --out experiments\mask-grid\some-run\shards\geometry-budget-01024-step16.jsonl
```

`--config-start` and `--config-step` slice the canonical preset order after config generation. `config_id` values are preserved.

## Recommended Overnight Launch

From repo root:

```powershell
.\scripts\launch_mask_overnight.ps1 -Lanes 8 -GeometryConfigs 13824 -GeometryStep 16
```

This launches 8 hidden PowerShell processes and returns immediately. It creates:

- `experiments\mask-grid\overnight-YYYYMMDD-HHMMSS\manifest.json`
- `experiments\mask-grid\overnight-YYYYMMDD-HHMMSS\processes.json`
- `experiments\mask-grid\overnight-YYYYMMDD-HHMMSS\logs\*.log`
- `experiments\mask-grid\overnight-YYYYMMDD-HHMMSS\shards\*.jsonl`

The default plan covers:

- `color-proposal`: 12 configs, expected 204 rows
- `local-cleanup`: 288 configs, expected 4,896 rows
- `geometry-budget`: 13,824 strided configs, expected 235,008 rows

This is a broad strided sample, not the full geometry Cartesian space. The default geometry jobs use starts `0..7` with `--config-step 16`, sampling half of the 27,648-config geometry grid across the full `itertools.product` order while preserving canonical `config_id` values.

## Monitor Progress

Set `$runDir` to the directory printed by the launcher:

```powershell
$runDir = "experiments\mask-grid\overnight-YYYYMMDD-HHMMSS"
Get-Content "$runDir\processes.json" | ConvertFrom-Json | Format-Table
Get-Process -Id ((Get-Content "$runDir\processes.json" | ConvertFrom-Json).pid) -ErrorAction SilentlyContinue
Get-Content "$runDir\logs\lane-00.log" -Tail 20
```

Check row counts and find partial shards:

```powershell
$manifest = Get-Content "$runDir\manifest.json" -Raw | ConvertFrom-Json
$manifest.jobs | ForEach-Object {
  $rows = if (Test-Path $_.out) { (Get-Content $_.out | Measure-Object -Line).Lines } else { 0 }
  [pscustomobject]@{
    name = $_.name
    preset = $_.preset
    rows = $rows
    expected = $_.expected_rows
    complete = ($rows -eq $_.expected_rows)
  }
} | Sort-Object complete, preset, name | Format-Table -AutoSize
```

If a shard is partial, rerun just that shard with the same `--preset`, `--config-start`, `--config-step`, `--limit-configs`, and `--out`. The runner overwrites its output file, so move or delete a bad partial first only if you want to preserve it for inspection.

## Combine And Summarize

After all lane processes exit, prefer the collector:

```powershell
$runDir = "experiments\mask-grid\overnight-YYYYMMDD-HHMMSS"
.\scripts\collect_mask_overnight.ps1 -RunDir $runDir -TopN 20 -RunTelea
```

The collector writes `shard-audit.csv` and refuses to summarize if any shard is missing rows.

Manual combine commands are:

```powershell
$runDir = "experiments\mask-grid\overnight-YYYYMMDD-HHMMSS"
Get-Content "$runDir\shards\color-proposal-*.jsonl" | Set-Content -Encoding UTF8 "$runDir\color-proposal.jsonl"
Get-Content "$runDir\shards\local-cleanup-*.jsonl" | Set-Content -Encoding UTF8 "$runDir\local-cleanup.jsonl"
Get-Content "$runDir\shards\geometry-budget-*.jsonl" | Set-Content -Encoding UTF8 "$runDir\geometry-budget-strided.jsonl"

uv run python scripts\summarize_mask_grid.py "$runDir\color-proposal.jsonl" --out "$runDir\color-proposal.csv" --top-json "$runDir\color-proposal-top.json" --top-n 10
uv run python scripts\summarize_mask_grid.py "$runDir\local-cleanup.jsonl" --out "$runDir\local-cleanup.csv" --top-json "$runDir\local-cleanup-top.json" --top-n 10
uv run python scripts\summarize_mask_grid.py "$runDir\geometry-budget-strided.jsonl" --out "$runDir\geometry-budget-strided.csv" --top-json "$runDir\geometry-budget-strided-top.json" --top-n 20
```

If a strict summary is empty, the hard filters eliminated every config. Keep the raw JSONL; it is still useful. A later model can make an unfiltered summary by calling `summarize_grid_rows()` and sorting by `mean_score` without `rank_mask_summaries()`.

## TELEA Eval For Top Configs

Run this only after summaries exist:

```powershell
uv run python scripts\run_inpaint_eval.py --configs "$runDir\color-proposal-top.json" --method telea --out "$runDir\color-proposal-top-telea.jsonl"
uv run python scripts\run_inpaint_eval.py --configs "$runDir\local-cleanup-top.json" --method telea --out "$runDir\local-cleanup-top-telea.jsonl"
uv run python scripts\run_inpaint_eval.py --configs "$runDir\geometry-budget-strided-top.json" --method telea --out "$runDir\geometry-budget-strided-top-telea.jsonl"
```

Current caveat: `run_inpaint_eval.py` evaluates oracle `truth+2px` masks rather than config-specific predicted masks, so TELEA rows are mostly a ceiling/control until that script is extended.

## Other Exploration Runs

If the first strided half finishes early, run the complementary half:

```powershell
.\scripts\launch_mask_overnight.ps1 -RunName overnight-geometry-odd-strata -Lanes 8 -GeometryStart 8 -GeometryConfigs 13824 -GeometryStep 16 -IncludeLocalCleanup 0 -IncludeColorProposal 0
```

That command covers starts `8..15` with the same step. Together with the default run, it completes all 27,648 geometry configs.

## Questions Worth Answering From Artifacts

- Do any local cleanup settings pass the hard filters, or is local cleanup alone too weak?
- Does geometry budget tuning improve weighted precision without losing recall?
- Are top geometry configs reducing far-away false positives, or only shrinking near-text collars?
- Which knobs appear in top configs repeatedly: `short_weight`, `connectivity_dilation_px`, `min_cc_px`, or `max_budget_fraction`?
- Are best configs stable across all 17 cases, or do they win by overfitting a few fixtures?

## Safety Notes

- Do not change CLI/web production defaults based only on these runs.
- Use forced manifest bboxes; this experiment measures mask generation, not detector recall.
- Preserve raw shard JSONL files even when summaries are empty.
- Strided geometry shards are not lexicographically contiguous. Sort or group by `config_id` during analysis, not by filename range assumptions.
- Partial shards should be detected by row count, exit code, and lane logs together. Do not rely on file presence alone.
- Background processes are ordinary PowerShell processes. Stop them with `Stop-Process -Id <pid>` only if you intentionally want to abort the run.
