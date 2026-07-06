# Detector Pair Harvest Runbook

Use this when you want the first full detector-pair harvest over the verified-clean `tests/images/zero` corpus. The expensive work is detector inference; build the paired corpus once, harvest each detector independently, then rerun analysis offline as thresholds or consensus rules change.

## Inputs

- Repo root: `C:/Users/Jurph/Documents/Python Scripts/untext`
- Python: `C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe`
- Clean corpus: `tests/images/zero`
- Harvest root: `tests/images/detector_pair_harvest`
- Corpus seed: `20260706`
- Detector harvest floor: `0.01` capture setting only; do not treat it as the final analysis threshold.

## Build paired corpus

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/build_detector_pair_corpus.py tests/images/zero --out-root tests/images/detector_pair_harvest --seed 20260706 --resume
```

Expected output:

- `tests/images/detector_pair_harvest/manifest.json`
- `tests/images/detector_pair_harvest/pairs/pair_manifest.jsonl`
- `tests/images/detector_pair_harvest/pairs/synthetic_twins/*.jpg`
- `tests/images/detector_pair_harvest/pairs/truth_masks/*.png`

## Harvest detectors

Run each detector independently so one backend failure does not lose progress from the others. Keep `--resume` on reruns; the script skips existing `(pair_id, state)` rows.

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors east --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors doctr --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors easyocr --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors yolo11x --floor 0.01 --resume
```

YOLO uses `.codex-tmp/yolo_eval/weights/yolo11x-train28-best.pt` by default. Override with `--yolo-weights <path>` if the checkpoint lives elsewhere.

Expected evidence files:

- `tests/images/detector_pair_harvest/evidence/east.jsonl`
- `tests/images/detector_pair_harvest/evidence/doctr.jsonl`
- `tests/images/detector_pair_harvest/evidence/easyocr.jsonl`
- `tests/images/detector_pair_harvest/evidence/yolo11x.jsonl`

Each evidence file should contain two rows per pair: one `clean`, one `twin`.

## Analyze

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py tests/images/detector_pair_harvest
```

Expected analysis files:

- `tests/images/detector_pair_harvest/analysis/per_detector_metrics.csv`
- `tests/images/detector_pair_harvest/analysis/pairwise_overlap.csv`
- `tests/images/detector_pair_harvest/analysis/twin_box_metrics.jsonl`
- `tests/images/detector_pair_harvest/analysis/combination_grid.csv`

## Smoke workflow

Use this before a full run or after script changes.

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/build_detector_pair_corpus.py tests/images/zero --out-root .codex-tmp/detector_pair_harvest_smoke5 --seed 20260706 --limit 5
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke5 --clean-dir tests/images/zero --detectors yolo11x --floor 0.01 --limit 5 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke5 --detectors yolo11x
```

Smoke acceptance:

- `pairs/pair_manifest.jsonl` has 5 rows.
- `evidence/yolo11x.jsonl` has 10 rows.
- `analysis/per_detector_metrics.csv` exists and has one `yolo11x` row.
- `analysis/twin_box_metrics.jsonl` exists.

## Interpreting outputs

- `per_detector_metrics.csv`: detector-level clean fire count, twin fire count, box volume, confidence, and best-IoU summary.
- `pairwise_overlap.csv`: overlap of clean/twin fire sets between detector pairs. Use it to see whether two detectors make the same mistakes.
- `twin_box_metrics.jsonl`: one row per twin evidence row with best IoU and center-distance metrics against synthetic truth.
- `combination_grid.csv`: simulated detector-combination counts for 2-of-N through 4-of-N consensus rules.

The harvest floor is intentionally permissive. Choose production thresholds and hit/overlap rules during offline analysis, not during GPU inference.
