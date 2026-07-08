# Ticket: Replace DocTR with YOLO11x in the production detector panel

Status: READY TO IMPLEMENT. Not started.

## One-paragraph summary

Production's 3-detector consensus panel is EAST + DocTR + EasyOCR. Replace DocTR
with a YOLO11x model fine-tuned specifically on watermarks. Evidence below (from
a 415-pair known-truth corpus, replayed through the *actual* production consensus
algorithm) shows the swap improves watermark recall, reduces false-positive
incidents, reduces false-positive masked area, AND cuts detector wall-clock time
by ~5x. This is not a tradeoff — every measured axis improves. The only real work
is plumbing: adding a `detect_with_yolo11x()` function that matches the existing
`detect_with_east`/`detect_with_doctr`/`detect_with_easyocr` shape, wiring it into
`run_consensus_detection()` and `initialize_consensus_models()` in place of DocTR,
adding `ultralytics` as a real dependency, and giving the weights file a permanent
home with an auto-download path (mirroring how EAST already does this).

## Why (evidence)

Source data: `tests/images/detector_pair_harvest/` — 415 clean/synthetic-twin
image pairs with known-truth watermark bboxes, built via
`scripts/build_detector_pair_corpus.py` against `tests/images/zero`. Raw
detector evidence for east/doctr/easyocr/yolo11x already harvested at
`evidence/*.jsonl` (floor=0.01 capture, so all confidence-threshold policies can
be replayed offline without re-running any model).

Reproduce with: `scripts/simulate_production_consensus_swap.py` (replays the
**actual** `untextre.consensus.find_consensus_boxes()` algorithm — IoU>=0.1
grouping requiring >=2 different detectors, `CLI_DEFAULT_CONFIDENCE=0.3` gate,
10%-per-side padding, mod-4 alignment — not a hypothetical rule).

| metric | current (east+doctr+easyocr) | swapped (east+easyocr+yolo11x) |
|---|---|---|
| watermark recall (IoU>=0.3 vs truth) | 383/415 = 92.3% | 399/415 = **96.1%** |
| clean images incorrectly masked | 15/415 = 3.6% | 11/415 = **2.7%** |
| mean masked-area fraction (all clean images) | 0.018% | **0.015%** |
| detector wall-clock (has-text-2, 312 images, steady-state) | 13.26 s/image (4-detector reference incl. yolo11x already run) | **2.66 s/image** (east+easyocr+yolo11x only) |

Per-detector solo throughput measured on has-text-2 (312 images,
`tests/images/has_text_2_full_review/`): east 1.95 s/img, doctr **10.60 s/img**
(68x slower than yolo11x, not explained by image size — has-text-2 images are
smaller on average than the zero corpus), easyocr 0.55 s/img, yolo11x
**0.155 s/img**.

Per-detector solo accuracy (strong hit, IoU>=0.3, over the 415-pair corpus):
DocTR F1=0.867, EAST F1=0.846, EasyOCR F1=0.947, **YOLO11x F1=0.947**. DocTR and
EAST are both sensitive to `mid_gray` watermark color and low-contrast text
(53-76% hit rate near the visibility floor); EasyOCR/YOLO11x are contrast-robust
(93-100% across the full range). Full breakdown:
`.codex-tmp/failure_mode_analysis.py` output (session-local; rerun to regenerate,
this file is not committed).

DocTR is also the single largest source of persistent clean-image false alarms:
656 clean-image FP boxes across the corpus (vs EAST 481, YOLO11x 316, EasyOCR
70), 94.8% of which recur at the same location on that image's twin (i.e. these
are stable, scene-content-pinned false triggers — lace, jewelry, texture — not
detector noise). Removing DocTR removes the single biggest source of that noise.

## What NOT to change

- Do not touch the consensus algorithm itself (`find_consensus_boxes`,
  `overlap_threshold=0.1`, the 2-detector-minimum requirement, the 10%-padding +
  mod-4 alignment). It's already the right mechanism; this ticket only changes
  which three detectors feed it.
- Do not add YOLO11x as a 4th detector alongside DocTR. The evidence above is
  for a **3-detector swap** (east+easyocr+yolo11x), not a 4-detector panel. A
  4th detector was evaluated separately and found to have diminishing/negative
  returns (see `docs/superpowers/plans/2026-07-06-detector-pair-harvest-design.md`
  and the P/R combination-grid analysis in this session's transcript — best
  overall F1 came from a 3-detector `>=2-of-3` rule using east+easyocr+yolo11x,
  and a naive all-4 majority vote was *worse* than the 2-detector easyocr+yolo11x
  pair alone).
- Do not change `CLI_DEFAULT_CONFIDENCE` (0.3) or `overlap_threshold` (0.1) as
  part of this ticket — those are separate tuning questions with their own
  evidence requirements. This ticket is detector-set-only.

## Prerequisite decision: where do model weights live?

Weights currently sit at `.codex-tmp/yolo_eval/weights/yolo11x-train28-best.pt`
(114,512,018 bytes / ~114.5 MB), downloaded from:

```
https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt
```

`.codex-tmp/` is scratch space and may not persist. **Before writing code**,
follow the EAST pattern exactly (`untextre/detector.py:_get_east_model_path`,
`_download_east_model`, `_load_east_model`, `docs/detector-models.md`):

1. Add `_get_yolo11x_model_path()` returning
   `Path.home() / ".untextre" / "models" / "yolo11x-train28-best.pt"`.
2. Add `_download_yolo11x_model()`: atomic download (`.tmp` then `.replace()`),
   size validation (expect exactly 114,512,018 bytes, or at minimum reject
   anything under, say, 50 MB as truncated/error-page), same retry-free
   fail-loud behavior as `_download_east_model`.
3. Add a `YOLO11X_MODEL_URL` constant with the HF URL above, and a
   `YOLO11X_MODEL_DOCS = "docs/detector-models.md"` constant.
4. Add a `## YOLO11x` section to `docs/detector-models.md` (match the existing
   EAST/DocTR/EasyOCR section format exactly): model used, "invented by"
   (fine-tuned by `fancyfeast`; no separate paper — this is a fine-tune of
   Ultralytics YOLO11x, not a novel architecture), how we ingest it, canonical
   model location (the HF Space URL above; **note: no LICENSE file is present
   on that Space as of 2026-07-07** — flag this explicitly in the doc and get
   an explicit go/no-go on redistribution terms before shipping to any
   external users, even though local/internal use is presumably fine), manual
   fallback instructions, primary references (Ultralytics YOLO11:
   https://github.com/ultralytics/ultralytics).

**Open question the implementer must resolve, not guess:** is committing a
114.5 MB binary into git acceptable for this repo, or does it need Git LFS /
external hosting? Check `.gitattributes` for existing LFS config before
deciding; if none exists, default to the EAST pattern (auto-download to
`~/.untextre/models/`, nothing committed) since that's the established
convention and avoids the repo-size question entirely.

## Implementation steps

### 1. Dependency

Add `ultralytics` as a real project dependency (currently only ad-hoc installed
via `uv pip install --python .venv/Scripts/python.exe ultralytics --no-deps`
during this session's experiments — not tracked in `pyproject.toml` or
`uv.lock`). Run `uv add ultralytics` from the repo root, then `uv lock`. Verify
it doesn't force an unwanted torch/torchvision upgrade — the existing pinned
`torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` must keep working for
DocTR/EasyOCR; use `uv add ultralytics --no-deps` and manually verify
`ultralytics`'s runtime imports resolve against the already-pinned torch stack
if `uv add` wants to bump anything.

### 2. `untextre/detector.py` — model loading

Add, mirroring `get_east_net()` (lines ~79-88) and its supporting
`_load_east_model`/`_download_east_model`/`_validate_east_model_file` quartet:

```python
_yolo11x_model = None  # module-level singleton, alongside _east_net etc.

def get_yolo11x_model():
    """Return the shared YOLO11x watermark detector."""
    global _yolo11x_model
    if _yolo11x_model is None:
        logger.info("Initializing YOLO11x watermark detector...")
        _yolo11x_model = _load_yolo11x_model()
        logger.info("YOLO11x model ready")
    return _yolo11x_model
```

`_load_yolo11x_model()` follows `_load_east_model()`'s shape: resolve
`_get_yolo11x_model_path()`, download if missing/invalid via
`_download_yolo11x_model()`, then `from ultralytics import YOLO; return
YOLO(str(model_path))`.

### 3. `untextre/consensus.py` — detection function

Add `detect_with_yolo11x`, matching the `detect_with_east`/`detect_with_doctr`
signature and return contract exactly — `List[Tuple[x, y, w, h, confidence_pct]]`
where confidence is on a **0-100 scale** (matches the other three; see
`find_consensus_boxes`'s `conf / 100.0 if conf > 1.0 else conf` normalization).
Reference implementation for the YOLO call itself:
`scripts/run_detector_pair_harvest.py:run_yolo11x()` (uses
`model.predict(str(image_path), conf=floor, verbose=False, device=0)`, xyxy ->
xywh conversion). Note that function takes an `image_path`, not just the BGR
array — Ultralytics' `.predict()` accepts either a path or an array; prefer the
array form (`model.predict(image, conf=..., verbose=False, device=0)`) inside
`consensus.py` to match the other three detectors' signature
(`detect_with_X(image: np.ndarray, confidence_threshold: float)`) and avoid a
redundant disk read, since production always has the image in memory already.

```python
def detect_with_yolo11x(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int, float]]:
    """Run YOLO11x watermark detection with configurable confidence threshold."""
    try:
        model = detector_mod.get_yolo11x_model()
        results = model.predict(image, conf=MODEL_CONFIDENCE_FLOOR, verbose=False, device=0)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf * 100))
        return detections
    except Exception as e:
        logger.warning(f"YOLO11x detection failed: {e}")
        return []
```

(Follow the DocTR pattern of running the model at `MODEL_CONFIDENCE_FLOOR` and
post-filtering by `confidence_threshold`, so users can re-threshold without
re-running the model — same rationale as the existing detectors' comments.)

### 4. `untextre/consensus.py` — wire into the pipeline

In `run_consensus_detection()` (currently lines ~360-382), replace the `doctr`
try/except block with a `yolo11x` block calling `detect_with_yolo11x`. Keep the
`detections['yolo11x'] = []` fallback-on-exception pattern identical to the
other three.

In `initialize_consensus_models()` (currently lines ~432-441), replace the
DocTR preload block with a YOLO11x preload block calling
`detector_mod.get_yolo11x_model()`.

Update the module docstring (lines 1-5): "EAST, DocTR, and EasyOCR" ->
"EAST, EasyOCR, and YOLO11x".

### 5. Remove DocTR from the live path (do not delete the code)

`detect_with_doctr()`, `get_doctr_detector()`, and `detector.py`'s
`detect_text_regions(method="doctr")` branch should stay in the codebase
(useful for offline research, matches how `scripts/run_detector_pair_harvest.py`
already treats all 4 detectors as available adapters) — just stop calling
`detect_with_doctr` from `run_consensus_detection`/`initialize_consensus_models`.
Do not delete DocTR support entirely; this is a production-path swap, not a
deprecation of DocTR as a research tool.

### 6. Tests

- Update/add `tests/test_pipeline.py` and `tests/test_detector*.py` coverage
  that currently asserts on the 3-detector set by name (`east`, `doctr`,
  `easyocr`) — grep for `"doctr"` string literals in `tests/` before starting;
  several will need renaming to `"yolo11x"`.
- Add a focused unit test for `detect_with_yolo11x` mirroring the existing
  `detect_with_east`/`detect_with_doctr` test shape (mock or stub the model,
  assert on the `(x, y, w, h, conf_pct)` tuple contract and the
  confidence_threshold post-filter behavior).
- Add/update `initialize_consensus_models` test coverage to expect YOLO11x
  preload logging instead of DocTR.

### 7. Verification (run before declaring done)

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_pipeline.py tests/test_detector_unit.py tests/test_detector_parse.py -q
```

Then re-run the known-truth simulation to confirm the *actual* wired-up
production code (not the offline harvest replay) reproduces the numbers in
this ticket — build a small paired corpus via
`scripts/build_detector_pair_corpus.py tests/images/zero --out-root
.codex-tmp/verify_swap --limit 40 --seed 20260706`, then call
`untextre.consensus.run_consensus_detection()` directly (not the harvest
adapters) against both clean and twin images in that sample, and confirm
recall/FP direction matches (don't expect exact percentage match — the
`--limit 40` sample is small and `run_consensus_detection` runs each detector
at its live confidence path rather than post-hoc-filtering a floor=0.01
harvest, per the caveat below).

## Known caveats (carry these into the PR description, don't silently drop them)

1. **DocTR confidence-threshold caveat**: this ticket's 92.3%/3.6% *current*
   baseline numbers were computed by filtering a floor=0.01 harvest up to
   confidence>=0.3 post-hoc, not by running DocTR live at 0.3.
   `get_doctr_detector()`'s docstring says it "upgrades to a more permissive
   config" at lower thresholds, which could mean live candidate generation at
   0.3 differs slightly from post-hoc-filtered floor=0.01 candidates. This
   doesn't threaten the swap decision (DocTR is being removed either way), but
   don't cite the *current*-baseline numbers as an exact production replay in
   the PR — cite them as "closely approximates."
2. **Corpus prevalence**: the 415-pair corpus is exactly 50/50 clean/twin by
   construction. Recall (92.3% -> 96.1%) transfers directly to any real
   deployment mix. The area-FP-rate percentages will look *even better* in a
   real deployment with higher watermark prevalence (fewer negative images to
   generate FP area from) — don't undersell the swap by quoting only the
   50/50 numbers if real deployment prevalence is known to be higher.
3. **Bbox-level, not pixel-level, area**: the "masked area" metric above is
   bounding-box area, not the actual GrabCut/inpaint pixel footprint inside
   the box (mask_generator.py does further pixel-level refinement inside each
   box). Real painted-pixel area will be somewhat smaller than what's reported
   here.
4. **License**: no LICENSE file exists on the source HF Space as of
   2026-07-07. Confirm redistribution/production-use terms are acceptable
   before this ships anywhere outside local/internal use — don't silently
   assume permissive licensing from the Space being public.

## Files touched (expected)

- `pyproject.toml`, `uv.lock` — add `ultralytics` dependency
- `untextre/detector.py` — add YOLO11x model loader (mirror EAST's pattern)
- `untextre/consensus.py` — add `detect_with_yolo11x`, wire into
  `run_consensus_detection`/`initialize_consensus_models`, update docstring
- `docs/detector-models.md` — add YOLO11x bibliography section
- `tests/test_pipeline.py`, `tests/test_detector_unit.py`,
  `tests/test_detector_parse.py` (or wherever `"doctr"` string literals live —
  grep first) — rename/update detector-set assertions, add YOLO11x unit test

## Files NOT touched

- `scripts/run_detector_pair_harvest.py`, `scripts/analyze_detector_pair_harvest.py`,
  `scripts/simulate_production_consensus_swap.py`, `untextre/detector_pair_harvest.py`
  — these are the offline research harness and already support all 4 detectors
  as adapters; no change needed for this ticket.
