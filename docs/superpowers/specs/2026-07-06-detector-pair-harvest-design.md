# Detector Pair Harvest Design

## Goal
Build a replayable detector-evaluation corpus over the verified-clean `zero` images and one synthetic watermarked twin per clean image, so expensive detector inference runs once and later threshold, consensus, and overlap analysis runs offline.

## Problem
We now have a known-clean image set and a synthetic watermark generator with a measured visibility floor. We want to compare four detectors:

- EAST
- DocTR
- EasyOCR
- YOLO11-x

The current replay tooling is built around the production consensus pipeline. That is good for end-to-end threshold sweeps, but it is the wrong abstraction for detector-isolated science. We need to answer questions such as:

- What is each detector's false-positive rate on the clean corpus?
- What is each detector's recall on matched synthetic twins?
- Do two detectors agree on the same false positives, or produce separate false positives?
- If we replace one detector or move to four-detector consensus, what changes?

Those questions require raw per-detector evidence, not only the production bbox-of-record.

## Scope
This design covers:

1. generating one deterministic synthetic twin for each clean image
2. harvesting raw detector outputs for each detector on both clean and twin images
3. storing enough metadata to replay threshold and consensus experiments offline
4. analyzing detector precision, recall, overlap, and combination performance

This design does **not** cover:

- mask generation or inpainting evaluation
- changing production detector thresholds
- changing production consensus rules
- scaling beyond the first `N=415` paired corpus
- adding more synthetic variants per image in the first pass

## Design Summary
Use a matched-pair replay harvest.

For each clean image, generate one synthetic twin. For each detector, run inference independently on both the clean image and the twin. Save raw detector outputs once. Perform all later threshold, overlap, and consensus analysis offline from the saved evidence.

This gives us a stable experimental object:

> one paired corpus, one expensive detector harvest, many cheap analyses

## Corpus Shape
For each clean image:

- `clean`: human-verified watermark-free source image
- `twin`: one deterministic synthetic watermark overlay produced from `clean`

Each pair has:

- `pair_id`
- `clean_relative_path`
- `twin_relative_path`
- `truth_bbox`
- `truth_mask`
- synthetic metadata from the current generator, including visibility metrics

The first experiment uses **exactly one** twin per clean image. That keeps the paired clean-vs-marked analysis simple and interpretable. If the machinery proves useful, later runs may add multiple watermark variants per image without changing the harvest schema.

## Artifact Layout
Write the experiment under a new root:

```text
tests/images/detector_pair_harvest/
  manifest.json
  pairs/
    pair_manifest.jsonl
    synthetic_twins/
  evidence/
    east.jsonl
    doctr.jsonl
    easyocr.jsonl
    yolo11x.jsonl
  analysis/
    per_detector_metrics.csv
    pairwise_overlap.csv
    combination_grid.csv
    confusion_tables.json
```

### `pairs/pair_manifest.jsonl`
One row per clean/twin pair. Fields:

- `pair_id`
- `clean_relative_path`
- `twin_relative_path`
- `base_width`
- `base_height`
- `truth_bbox`
- `truth_bbox_coverage`
- `truth_alpha_coverage`
- `measured_visibility_delta_e`
- `visibility_attempts`
- `visibility_fallback`
- synthetic watermark metadata needed for later interpretation and replay

### `evidence/<detector>.jsonl`
One row per detector per image-state. Fields:

- `pair_id`
- `state`: `clean` or `twin`
- `detector`: `east`, `doctr`, `easyocr`, `yolo11x`
- `image_relative_path`
- `width`
- `height`
- `harvest_floor`
- `elapsed_ms`
- `boxes`: array of normalized detection records

Each normalized detection record contains:

- `xywh`
- `confidence`
- `label`
- `raw_payload`: detector-specific fields preserved without interpretation

The common fields make cross-detector analysis easy. The raw payload keeps us from throwing away detector-specific information we may want later.

## Expensive vs Replayable Work
### One-time expensive work

1. generate the 415 synthetic twins
2. run all four detectors on both clean and twin images
3. write pair manifests and detector evidence JSONL files

### Cheap replayable work

- threshold sweeps per detector
- per-detector precision and recall summaries
- geometric comparisons to truth bbox
- pairwise overlap and orthogonality analysis
- simulated consensus across any subset of detectors
- replacement studies such as "swap YOLO for detector X"

The design goal is simple: no later question about thresholds or consensus should require rerunning detector inference.

## Detector Harvest Rules
### Detector set
The first pass uses:

- EAST
- DocTR
- EasyOCR
- YOLO11-x

YOLO11-x is included because the zero-corpus specificity probe showed it is competitive with the current panel and has a meaningfully different false-positive set. It is therefore a plausible fourth detector or replacement candidate.

### Harvest floors
Each detector is harvested at a permissive detector-internal floor. These floors are capture settings, not analytical thresholds. They exist only to keep output finite and useful.

The floors must be recorded in `manifest.json` and in each evidence row. Offline analysis may later impose stricter thresholds without rerunning inference.

This design intentionally does **not** define decision thresholds such as "IoU >= X means a hit." Those belong to later analysis, not to the harvest phase.

## Metrics to Capture, Not Judge
The first pass should collect continuous values and simple counts, not force early pass/fail boundaries.

### Clean-image side
Per detector, per image:

- fired or not
- number of boxes
- max confidence
- total detected area fraction
- box geometry summaries

These support image-level false-positive analysis and texture-vs-text failure inspection.

### Twin-image side
Per detector, per image:

- number of boxes
- max confidence
- best IoU to truth bbox
- center-distance from detector box to truth box
- truth-center-contained boolean
- detector area / truth area
- detector width / truth width
- detector height / truth height
- optional top-k overlaps and confidences when a detector sprays multiple boxes

These measurements support later recall definitions without baking them into the expensive harvest.

## Overlap and Orthogonality Analysis
The first analysis pass should produce two kinds of orthogonality measurements.

### False-positive orthogonality on the clean corpus
For each detector pair:

- both fired
- only A fired
- only B fired
- neither fired
- Jaccard overlap of FP-image sets

This tells us whether two detectors make the same mistakes.

### Twin-side complementarity on the marked corpus
For each detector pair, using later-defined matching rules over the stored geometry:

- both hit
- only A hit
- only B hit
- both missed

This tells us whether a new detector adds recall or only duplicates existing coverage.

## Consensus Replay
Once the evidence exists, the analysis layer should replay at least these rule families:

- all `4 choose 2` detector pairs
- all `4 choose 3` detector triplets
- full `4-of-4`
- replacement studies:
  - current production triplet
  - current triplet with YOLO11-x replacing EAST
  - current triplet with YOLO11-x replacing DocTR
  - current triplet with YOLO11-x replacing EasyOCR
- asymmetric studies such as:
  - YOLO plus any one text detector
  - two text detectors or YOLO alone at high confidence

The replay engine must be box-aware. Image-level co-firing is not enough; later consensus requires reasoning about whether two boxes plausibly refer to the same mark.

## Manifest and Provenance
Write a top-level `manifest.json` with:

- source clean corpus path
- image count
- twin-generation seed and relevant config
- detector names and versions
- detector harvest floors
- start and finish timestamps
- environment notes needed for reproduction
- known quirks, for example:
  - DocTR BGR-input quirk left unchanged during the frozen baseline period
  - stem-collision caveat if the replay naming bug is still unfixed

This keeps the experiment citable and reproducible.

## Error Handling
- If one pair fails to generate, record the failure in the pair manifest and continue.
- If one detector fails on one image, write an evidence row with the failure metadata and continue.
- If a detector harvest is interrupted, reruns should resume without recomputing finished rows.
- Analysis must tolerate missing detector rows and report coverage gaps explicitly.

## Success Criteria
The first implementation is successful if it produces:

1. one deterministic synthetic twin for each clean image
2. one evidence JSONL file per detector over both clean and twin images
3. one analysis script that emits:
   - per-detector clean/twin summaries
   - pairwise overlap matrices
   - a combination scoreboard for detector subsets
4. a workflow where changing thresholds or consensus rules does not require rerunning GPU detector inference

## Recommended Implementation Order
1. generate the 415 twins and `pair_manifest.jsonl`
2. harvest all four detectors into `evidence/*.jsonl`
3. build the offline analysis script
4. inspect the first `N=415` results before scaling up

## Tradeoff Summary
This design costs a little more engineering now than a fixed-threshold shootout. In return, it gives us the exact experimental object we need: a replayable, detector-isolated evidence corpus over matched clean/marked pairs.

That trade is worth it. It buys future threshold studies, detector replacement studies, and four-detector consensus experiments without paying the GPU tax again.
