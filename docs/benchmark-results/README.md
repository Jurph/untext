# Benchmark Results

This directory stores selected benchmark summaries that are useful for later
analysis, README charts, and regression calibration. Large raw outputs and
ad-hoc inspection images belong in ignored local output directories.

## Generated Text Monte Carlo - 2026-06-28

Command:

```powershell
uv run --extra dev --extra web pytest tests/test_generated_text_benchmark.py -v -s --generated-text-cases=1000 --generated-text-color-sensitivities=3 --generated-text-report tests/images/output/generated_text_1000_cli_matched.jsonl
```

Runtime:

- Wall clock: 3:33:31
- Mean case time: about 12.8 seconds

Overall fallout:

| Outcome | Count |
| --- | ---: |
| repaired | 695 |
| coverage_rejected | 162 |
| miss | 116 |
| detected_no_mask | 20 |
| false_positive | 7 |

Outcome definitions:

| Outcome | Meaning |
| --- | --- |
| repaired | The pipeline detected the generated watermark region, produced a mask under the coverage guardrail, and ran inpainting. |
| miss | No consensus detection survived the full detector/failover path. |
| detected_no_mask | Detection overlapped the known generated watermark bbox, but spatial TF-IDF/GrabCut produced an empty mask. |
| false_positive | The pipeline detected a region, but it had zero overlap with the generated watermark bbox. |
| coverage_rejected | The pipeline produced a mask, but the mask covered more than the configured coverage guardrail (`6%`), so inpainting was skipped. |

Mode split:

| Mode | Count | Repaired | Coverage rejected | Miss |
| --- | ---: | ---: | ---: | ---: |
| local-color | 312 | 63.1% | 20.8% | 13.1% |
| local-shape | 316 | 61.7% | 25.3% | 10.1% |
| regional | 372 | 81.5% | 4.6% | 11.6% |

Notes:

- Generated text width targeted roughly one third of image width.
- Synthetic text colors sampled black, white, mid-gray, light-gray, and random RGB.
- Mode names use the current CLI vocabulary. The original raw run used
  pre-rename internal labels; the table above has been translated to the
  current `local-color`, `local-shape`, and `regional` names.
- No inspection PNGs are stored here; regenerate them with `--save-test-images`
  when visual review is needed.
- Raw JSONL rows are intentionally not committed. They may be stored locally in
  this directory or another ignored output directory; regenerate them with
  `--generated-text-report <path>` when charting or deeper analysis is needed.
- A Sankey diagram would be a useful future visualization: generated case ->
  detected nothing / detected something -> overlapped truth / false positive ->
  mask empty / coverage rejected / repaired.
