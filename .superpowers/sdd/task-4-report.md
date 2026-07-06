# Task 4 Report: Offline analysis script

Status: DONE_WITH_CONTROLLER_FALLBACK

Subagent attempt:
- Not dispatched after Task 2/3 provider rate limits; controller implemented Task 4 inline under TDD.

Files changed:
- `untextre/detector_pair_harvest.py`
- `tests/test_detector_pair_harvest.py`
- `scripts/analyze_detector_pair_harvest.py`

TDD evidence:
1. RED command:
   ```bash
   "C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
   ```
   Observed result: expected collection failure because `pairwise_fire_overlap` did not exist.
2. GREEN helper command:
   ```bash
   "C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
   ```
   Observed result: `13 passed in 1.98s`.
3. Final focused command after adding the analyzer CLI:
   ```bash
   "C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
   ```
   Observed result: `13 passed in 2.00s`.

Smoke evidence:
```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke --detectors yolo11x
```
Observed result: printed `.codex-tmp\detector_pair_harvest_smoke\analysis`.

Artifacts confirmed:
- `.codex-tmp/detector_pair_harvest_smoke/analysis/per_detector_metrics.csv` contains one `yolo11x` row with `pair_count=3`, `clean_fired_count=0`, `twin_fired_count=3`.
- `.codex-tmp/detector_pair_harvest_smoke/analysis/twin_box_metrics.jsonl` contains 3 rows with best-IoU metrics.

Concerns:
- None for Task 4 scope.
