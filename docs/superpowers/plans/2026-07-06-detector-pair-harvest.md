# Detector Pair Harvest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable, detector-isolated harvest over the verified-clean `zero` corpus and one synthetic watermarked twin per image, so EAST, DocTR, EasyOCR, and YOLO11-x can be compared offline for precision, recall, and overlap.

**Architecture:** Put reusable pure helpers in `untextre/detector_pair_harvest.py`; keep experiment orchestration in scripts under `scripts/`. The expensive workflow has two phases: build paired images/manifest, then harvest raw detector evidence. The analysis phase reads only JSONL evidence and emits CSV/JSON summaries without touching GPU inference.

**Tech Stack:** Python 3.10+, OpenCV, numpy, Pillow, existing `untextre.synthetic_text_benchmark`, existing detector wrappers in `untextre.detector`, ultralytics for YOLO11-x, pytest.

## Global Constraints

- Use the project venv for all commands: `"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe"`.
- Do not change production detector defaults or production consensus rules.
- Do not bake analytical IoU or recall thresholds into the harvest phase; save continuous measurements and let analysis sweep thresholds offline.
- One synthetic twin per clean image for the first pass (`N≈415`).
- Evidence capture must be resumable and must not recompute finished rows.
- Use extension-aware image IDs; do not key by stem only. The `EzlMcCy.jpeg` / `EzlMcCy.jpg` collision proved stem-only IDs are wrong.
- Large experiment outputs belong under `tests/images/detector_pair_harvest/` and remain local artifacts unless Jurph explicitly chooses to track them.

---

## File Structure

- Create: `untextre/detector_pair_harvest.py`
  - Pure helpers: extension-aware IDs, JSONL IO, bbox geometry, truth-relative continuous metrics, detector-normalized box records.
- Create: `scripts/build_detector_pair_corpus.py`
  - Generates one deterministic synthetic twin per clean image and writes `pairs/pair_manifest.jsonl`.
- Create: `scripts/run_detector_pair_harvest.py`
  - Runs detector adapters independently on clean/twin images and writes `evidence/<detector>.jsonl`.
- Create: `scripts/analyze_detector_pair_harvest.py`
  - Reads pair manifest + evidence files and writes analysis tables.
- Create: `tests/test_detector_pair_harvest.py`
  - Focused unit tests for pure helper behavior and analysis math.
- Optional modify: `TODO.md`
  - Add any discovered follow-up, especially if YOLO install/runtime has environment caveats.

---

### Task 1: Pure harvest helpers

**Files:**
- Create: `untextre/detector_pair_harvest.py`
- Test: `tests/test_detector_pair_harvest.py`

**Interfaces:**
- Produces:
  - `image_key(path: Path) -> str`
  - `bbox_area(box: Sequence[float]) -> float`
  - `bbox_iou(a: Sequence[float], b: Sequence[float]) -> float`
  - `bbox_center(box: Sequence[float]) -> tuple[float, float]`
  - `center_distance(a: Sequence[float], b: Sequence[float]) -> float`
  - `contains_point(box: Sequence[float], point: tuple[float, float]) -> bool`
  - `box_metrics_against_truth(boxes: list[dict], truth_bbox: Sequence[float]) -> dict`
  - `append_jsonl(path: Path, row: dict) -> None`
  - `load_jsonl(path: Path) -> list[dict]`
- Consumes: none.

- [ ] **Step 1: Write tests for IDs and bbox geometry**

Add to `tests/test_detector_pair_harvest.py`:

```python
from pathlib import Path

import pytest

from untextre.detector_pair_harvest import (
    bbox_area,
    bbox_center,
    bbox_iou,
    center_distance,
    contains_point,
    image_key,
)


def test_image_key_preserves_extension_to_avoid_stem_collision():
    assert image_key(Path("EzlMcCy.jpeg")) == "EzlMcCy__jpeg"
    assert image_key(Path("EzlMcCy.jpg")) == "EzlMcCy__jpg"


def test_bbox_iou_reports_continuous_overlap_without_decision_threshold():
    assert bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)
    assert bbox_iou([0, 0, 10, 10], [20, 20, 3, 3]) == 0.0


def test_bbox_center_and_distance_are_float_metrics():
    assert bbox_center([10, 20, 30, 40]) == (25.0, 40.0)
    assert center_distance([0, 0, 10, 10], [3, 4, 10, 10]) == pytest.approx(5.0)


def test_contains_point_is_geometry_only_not_a_hit_rule():
    assert contains_point([10, 10, 5, 5], (12, 14)) is True
    assert contains_point([10, 10, 5, 5], (15.1, 14)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
```

Expected: import failure because `untextre.detector_pair_harvest` does not exist.

- [ ] **Step 3: Implement pure helpers**

Create `untextre/detector_pair_harvest.py`:

```python
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence


def image_key(path: Path | str) -> str:
    """Return a filename-safe ID that preserves extension identity."""
    p = Path(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem).strip("_") or "image"
    suffix = re.sub(r"[^A-Za-z0-9]+", "", p.suffix.lower().lstrip(".")) or "nosuffix"
    return f"{stem}__{suffix}"


def bbox_area(box: Sequence[float]) -> float:
    _x, _y, w, h = [float(v) for v in box]
    return max(0.0, w) * max(0.0, h)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in box]
    return (x + w / 2.0, y + h / 2.0)


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def contains_point(box: Sequence[float], point: tuple[float, float]) -> bool:
    x, y, w, h = [float(v) for v in box]
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def box_metrics_against_truth(boxes: list[dict], truth_bbox: Sequence[float]) -> dict:
    truth_area = bbox_area(truth_bbox)
    truth_w = max(float(truth_bbox[2]), 1.0)
    truth_h = max(float(truth_bbox[3]), 1.0)
    truth_center = bbox_center(truth_bbox)
    metrics = []
    for box in boxes:
        xywh = box["xywh"]
        metrics.append({
            "xywh": xywh,
            "confidence": float(box.get("confidence", 0.0)),
            "iou": bbox_iou(xywh, truth_bbox),
            "center_distance": center_distance(xywh, truth_bbox),
            "truth_center_contained": contains_point(xywh, truth_center),
            "area_ratio": bbox_area(xywh) / truth_area if truth_area > 0 else 0.0,
            "width_ratio": float(xywh[2]) / truth_w,
            "height_ratio": float(xywh[3]) / truth_h,
        })
    metrics.sort(key=lambda item: (item["iou"], item["confidence"]), reverse=True)
    return {
        "box_count": len(boxes),
        "max_confidence": max((float(b.get("confidence", 0.0)) for b in boxes), default=0.0),
        "best_iou": metrics[0]["iou"] if metrics else 0.0,
        "best_center_distance": metrics[0]["center_distance"] if metrics else None,
        "best_truth_center_contained": metrics[0]["truth_center_contained"] if metrics else False,
        "top_boxes": metrics[:5],
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: Add tests for truth-relative metrics and JSONL**

Append to `tests/test_detector_pair_harvest.py`:

```python
from untextre.detector_pair_harvest import append_jsonl, box_metrics_against_truth, load_jsonl


def test_box_metrics_against_truth_preserves_continuous_values():
    boxes = [
        {"xywh": [40, 40, 10, 10], "confidence": 0.9},
        {"xywh": [0, 0, 10, 10], "confidence": 0.2},
    ]
    metrics = box_metrics_against_truth(boxes, [0, 0, 10, 10])

    assert metrics["box_count"] == 2
    assert metrics["max_confidence"] == 0.9
    assert metrics["best_iou"] == 1.0
    assert metrics["best_truth_center_contained"] is True
    assert metrics["top_boxes"][0]["confidence"] == 0.2


def test_jsonl_helpers_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"b": 2, "a": 1})
    append_jsonl(path, {"c": [3]})
    assert load_jsonl(path) == [{"a": 1, "b": 2}, {"c": [3]}]
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add untextre/detector_pair_harvest.py tests/test_detector_pair_harvest.py
git commit -m "Add detector pair harvest helpers"
```

---

### Task 2: Build paired corpus generator

**Files:**
- Create: `scripts/build_detector_pair_corpus.py`
- Test: `tests/test_detector_pair_harvest.py`

**Interfaces:**
- Consumes:
  - `image_key(path: Path) -> str`
  - `append_jsonl(path: Path, row: dict) -> None`
  - `generate_synthetic_text_case(clean_bgr, rng, font_dirs=None) -> SyntheticTextCase`
- Produces:
  - CLI script that writes `pairs/pair_manifest.jsonl`, `pairs/synthetic_twins/*.jpg`, and `pairs/truth_masks/*.png`.

- [ ] **Step 1: Add a testable plan builder helper**

Add to `untextre/detector_pair_harvest.py`:

```python
def pair_id_for_path(path: Path | str) -> str:
    return image_key(path)


def build_pair_row(pair_id: str, clean_relative_path: str, twin_relative_path: str, case_metadata: dict, truth_bbox: Sequence[int], width: int, height: int) -> dict:
    return {
        "pair_id": pair_id,
        "clean_relative_path": clean_relative_path,
        "twin_relative_path": twin_relative_path,
        "base_width": int(width),
        "base_height": int(height),
        "truth_bbox": [int(v) for v in truth_bbox],
        "truth_bbox_coverage": case_metadata.get("truth_bbox_coverage"),
        "truth_alpha_coverage": case_metadata.get("truth_alpha_coverage"),
        "measured_visibility_delta_e": case_metadata.get("measured_visibility_delta_e"),
        "visibility_attempts": case_metadata.get("visibility_attempts"),
        "visibility_fallback": case_metadata.get("visibility_fallback"),
        "synthetic_metadata": dict(case_metadata),
    }
```

- [ ] **Step 2: Add tests for pair rows**

Append to `tests/test_detector_pair_harvest.py`:

```python
from untextre.detector_pair_harvest import build_pair_row, pair_id_for_path


def test_pair_id_for_path_preserves_extension():
    assert pair_id_for_path(Path("foo bar.jpeg")) == "foo_bar__jpeg"


def test_build_pair_row_carries_visibility_and_truth_metadata():
    row = build_pair_row(
        "img__jpg",
        "img.jpg",
        "pairs/synthetic_twins/img__jpg.jpg",
        {"measured_visibility_delta_e": 12.5, "visibility_attempts": 1, "visibility_fallback": False, "color_class": "white"},
        [1, 2, 30, 4],
        640,
        480,
    )
    assert row["pair_id"] == "img__jpg"
    assert row["truth_bbox"] == [1, 2, 30, 4]
    assert row["measured_visibility_delta_e"] == 12.5
    assert row["synthetic_metadata"]["color_class"] == "white"
```

- [ ] **Step 3: Run tests to verify the new helper passes**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
```

Expected: pass.

- [ ] **Step 4: Implement the corpus builder script**

Create `scripts/build_detector_pair_corpus.py`:

```python
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
from PIL import Image

from untextre.detector_pair_harvest import append_jsonl, build_pair_row, pair_id_for_path
from untextre.synthetic_text_benchmark import generate_synthetic_text_case, iter_base_images
from untextre.utils import load_image


def save_bgr_jpeg(image_bgr, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, quality=quality)


def save_mask_png(mask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired clean/synthetic detector harvest corpus.")
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("tests/images/detector_pair_harvest"))
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root
    pairs_dir = out_root / "pairs"
    twins_dir = pairs_dir / "synthetic_twins"
    masks_dir = pairs_dir / "truth_masks"
    manifest_path = pairs_dir / "pair_manifest.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    images = iter_base_images(args.clean_dir)
    if args.limit is not None:
        images = images[: args.limit]

    started = time.time()
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()

    completed = {row["pair_id"] for row in []}
    if args.resume and manifest_path.exists():
        import json
        completed = {json.loads(line)["pair_id"] for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    rows_written = 0
    for index, image_path in enumerate(images):
        pair_id = pair_id_for_path(image_path.relative_to(args.clean_dir))
        if pair_id in completed:
            continue
        clean = load_image(image_path)
        rng = random.Random(f"{args.seed}:{index}:{image_path.name}")
        case = generate_synthetic_text_case(clean, rng)
        twin_rel = Path("pairs") / "synthetic_twins" / f"{pair_id}.jpg"
        mask_rel = Path("pairs") / "truth_masks" / f"{pair_id}.png"
        save_bgr_jpeg(case.watermarked, out_root / twin_rel, quality=args.quality)
        save_mask_png(case.truth_mask, out_root / mask_rel)
        row = build_pair_row(
            pair_id,
            str(image_path.relative_to(args.clean_dir)).replace("\\", "/"),
            twin_rel.as_posix(),
            {**case.metadata, "truth_mask_relative_path": mask_rel.as_posix()},
            case.truth_bbox,
            clean.shape[1],
            clean.shape[0],
        )
        append_jsonl(manifest_path, row)
        rows_written += 1
        print(f"[{index + 1}/{len(images)}] wrote {pair_id}", flush=True)

    top_manifest = {
        "clean_dir": str(args.clean_dir),
        "out_root": str(out_root),
        "seed": args.seed,
        "image_count": len(images),
        "rows_written_this_run": rows_written,
        "elapsed_seconds": time.time() - started,
        "schema": "detector_pair_harvest.v1",
    }
    (out_root / "manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke-generate 3 pairs**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/build_detector_pair_corpus.py tests/images/zero --out-root .codex-tmp/detector_pair_harvest_smoke --limit 3
```

Expected: `.codex-tmp/detector_pair_harvest_smoke/pairs/pair_manifest.jsonl` has 3 rows; synthetic twin images and truth masks exist.

- [ ] **Step 6: Commit**

```bash
git add untextre/detector_pair_harvest.py tests/test_detector_pair_harvest.py scripts/build_detector_pair_corpus.py
git commit -m "Add detector pair corpus builder"
```

---

### Task 3: Detector evidence adapters and harvest runner

**Files:**
- Modify: `untextre/detector_pair_harvest.py`
- Create: `scripts/run_detector_pair_harvest.py`
- Test: `tests/test_detector_pair_harvest.py`

**Interfaces:**
- Consumes:
  - pair manifest rows from Task 2
  - `append_jsonl`, `load_jsonl`, `image_key`
- Produces:
  - `normalize_detection_box(xywh, confidence, label, raw_payload=None) -> dict`
  - evidence rows under `evidence/<detector>.jsonl`

- [ ] **Step 1: Add normalized detection box tests**

Append to `tests/test_detector_pair_harvest.py`:

```python
from untextre.detector_pair_harvest import normalize_detection_box


def test_normalize_detection_box_rounds_geometry_but_keeps_raw_payload():
    box = normalize_detection_box([1.234, 2.5, 30.0, 4.0], 0.98765, "watermark", {"source": "unit"})
    assert box["xywh"] == [1.2, 2.5, 30.0, 4.0]
    assert box["confidence"] == 0.9877
    assert box["label"] == "watermark"
    assert box["raw_payload"] == {"source": "unit"}
```

- [ ] **Step 2: Implement normalized box helper**

Add to `untextre/detector_pair_harvest.py`:

```python
def normalize_detection_box(xywh: Sequence[float], confidence: float, label: str, raw_payload: dict | None = None) -> dict:
    return {
        "xywh": [round(float(v), 1) for v in xywh],
        "confidence": round(float(confidence), 4),
        "label": str(label),
        "raw_payload": dict(raw_payload or {}),
    }
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
```

Expected: pass.

- [ ] **Step 4: Implement harvest runner skeleton with fake adapter support**

Create `scripts/run_detector_pair_harvest.py` with real CLI plus adapter functions. Include a `--detectors` flag so smoke tests can use one detector and production runs can use all four.

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from untextre.detector_pair_harvest import append_jsonl, load_jsonl, normalize_detection_box
from untextre.detector import (
    _detect_with_easyocr,
    _detect_with_east,
    get_doctr_detector,
    get_east_net,
    get_easyocr_reader,
)
from untextre.utils import load_image


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def geometry_to_xywh(points) -> list[float]:
    import numpy as np
    pts = np.asarray(points, dtype="float32").reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts.astype("int32"))
    return [x, y, w, h]


def run_doctr(image_bgr, floor: float) -> list[dict]:
    detector = get_doctr_detector(confidence_threshold=floor)
    rows = []
    for det in detector.detect(image_bgr):
        rows.append(normalize_detection_box(
            geometry_to_xywh(det["geometry"]),
            det.get("confidence", 0.0),
            "text",
            {"geometry": det["geometry"].tolist() if hasattr(det["geometry"], "tolist") else det["geometry"]},
        ))
    return rows


def run_easyocr(image_bgr, floor: float) -> list[dict]:
    reader = get_easyocr_reader()
    rows = []
    for det in _detect_with_easyocr(image_bgr, reader, confidence_threshold=floor):
        rows.append(normalize_detection_box(
            geometry_to_xywh(det["geometry"]),
            det.get("confidence", 0.0),
            "text",
            {"geometry": det["geometry"].tolist() if hasattr(det["geometry"], "tolist") else det["geometry"]},
        ))
    return rows


def run_east(image_bgr, floor: float) -> list[dict]:
    net = get_east_net()
    rows = []
    for det in _detect_with_east(image_bgr, net, min_confidence=floor):
        rows.append(normalize_detection_box(
            geometry_to_xywh(det["geometry"]),
            det.get("confidence", 0.0),
            "text",
            {"geometry": det["geometry"].tolist() if hasattr(det["geometry"], "tolist") else det["geometry"]},
        ))
    return rows


def run_yolo11x(image_path: Path, model, floor: float) -> list[dict]:
    results = model.predict(str(image_path), conf=floor, verbose=False, device=0)
    r = results[0]
    rows = []
    for box in r.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cls_index = int(box.cls[0])
        rows.append(normalize_detection_box(
            [x1, y1, x2 - x1, y2 - y1],
            float(box.conf[0]),
            r.names.get(cls_index, str(cls_index)),
            {"class_index": cls_index},
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest raw detector evidence for paired corpus.")
    parser.add_argument("harvest_root", type=Path)
    parser.add_argument("--clean-dir", type=Path, default=Path("tests/images/zero"))
    parser.add_argument("--detectors", nargs="+", default=["east", "doctr", "easyocr", "yolo11x"])
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument("--yolo-weights", type=Path, default=Path(".codex-tmp/yolo_eval/weights/yolo11x-train28-best.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pair_rows = load_jsonl(args.harvest_root / "pairs" / "pair_manifest.jsonl")
    if args.limit is not None:
        pair_rows = pair_rows[: args.limit]

    yolo_model = None
    if "yolo11x" in args.detectors:
        from ultralytics import YOLO
        yolo_model = YOLO(str(args.yolo_weights))

    for detector in args.detectors:
        out_path = args.harvest_root / "evidence" / f"{detector}.jsonl"
        done = set()
        if args.resume and out_path.exists():
            done = {(row["pair_id"], row["state"]) for row in load_jsonl(out_path)}
        for index, pair in enumerate(pair_rows, start=1):
            for state, rel in [("clean", pair["clean_relative_path"]), ("twin", pair["twin_relative_path"])]:
                if (pair["pair_id"], state) in done:
                    continue
                image_path = args.clean_dir / rel if state == "clean" else args.harvest_root / rel
                started = time.time()
                try:
                    if detector == "yolo11x":
                        boxes = run_yolo11x(image_path, yolo_model, args.floor)
                        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    else:
                        image = load_image(image_path)
                        if detector == "east":
                            boxes = run_east(image, args.floor)
                        elif detector == "doctr":
                            boxes = run_doctr(image, args.floor)
                        elif detector == "easyocr":
                            boxes = run_easyocr(image, args.floor)
                        else:
                            raise ValueError(f"unknown detector {detector}")
                    row = {
                        "pair_id": pair["pair_id"],
                        "state": state,
                        "detector": detector,
                        "image_relative_path": rel,
                        "width": int(image.shape[1]) if image is not None else None,
                        "height": int(image.shape[0]) if image is not None else None,
                        "harvest_floor": args.floor,
                        "elapsed_ms": round(1000 * (time.time() - started), 1),
                        "boxes": boxes,
                    }
                except Exception as exc:
                    row = {
                        "pair_id": pair["pair_id"],
                        "state": state,
                        "detector": detector,
                        "image_relative_path": rel,
                        "harvest_floor": args.floor,
                        "elapsed_ms": round(1000 * (time.time() - started), 1),
                        "boxes": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                append_jsonl(out_path, row)
            print(f"{detector} [{index}/{len(pair_rows)}] {pair['pair_id']}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke-harvest YOLO over 3 pairs**

Run after Task 2 smoke corpus exists:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke --clean-dir tests/images/zero --detectors yolo11x --floor 0.01 --limit 3 --resume
```

Expected: `.codex-tmp/detector_pair_harvest_smoke/evidence/yolo11x.jsonl` has 6 rows.

- [ ] **Step 6: Commit**

```bash
git add untextre/detector_pair_harvest.py tests/test_detector_pair_harvest.py scripts/run_detector_pair_harvest.py
git commit -m "Add detector pair evidence harvester"
```

---

### Task 4: Offline analysis script

**Files:**
- Modify: `untextre/detector_pair_harvest.py`
- Create: `scripts/analyze_detector_pair_harvest.py`
- Test: `tests/test_detector_pair_harvest.py`

**Interfaces:**
- Consumes:
  - `pairs/pair_manifest.jsonl`
  - `evidence/*.jsonl`
  - `box_metrics_against_truth`
- Produces:
  - `analysis/per_detector_metrics.csv`
  - `analysis/pairwise_overlap.csv`
  - `analysis/twin_box_metrics.jsonl`
  - `analysis/combination_grid.csv`

- [ ] **Step 1: Add tests for per-detector summary math**

Append to `tests/test_detector_pair_harvest.py`:

```python
from untextre.detector_pair_harvest import summarize_detector_rows


def test_summarize_detector_rows_counts_clean_fires_and_twin_geometry():
    pairs = {
        "a": {"truth_bbox": [0, 0, 10, 10]},
        "b": {"truth_bbox": [50, 50, 10, 10]},
    }
    rows = [
        {"pair_id": "a", "state": "clean", "boxes": []},
        {"pair_id": "a", "state": "twin", "boxes": [{"xywh": [0, 0, 10, 10], "confidence": 0.9}]},
        {"pair_id": "b", "state": "clean", "boxes": [{"xywh": [1, 1, 2, 2], "confidence": 0.5}]},
        {"pair_id": "b", "state": "twin", "boxes": [{"xywh": [0, 0, 10, 10], "confidence": 0.4}]},
    ]
    summary = summarize_detector_rows("fake", pairs, rows)
    assert summary["detector"] == "fake"
    assert summary["pair_count"] == 2
    assert summary["clean_fired_count"] == 1
    assert summary["twin_fired_count"] == 2
    assert summary["median_best_iou"] == 0.5
```

- [ ] **Step 2: Implement summary helper**

Add to `untextre/detector_pair_harvest.py`:

```python
def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_detector_rows(detector: str, pairs: dict[str, dict], rows: list[dict]) -> dict:
    clean_rows = [row for row in rows if row.get("state") == "clean"]
    twin_rows = [row for row in rows if row.get("state") == "twin"]
    twin_ious = []
    for row in twin_rows:
        pair = pairs.get(row["pair_id"])
        if pair is None:
            continue
        twin_ious.append(box_metrics_against_truth(row.get("boxes", []), pair["truth_bbox"])["best_iou"])
    return {
        "detector": detector,
        "pair_count": len(pairs),
        "clean_row_count": len(clean_rows),
        "twin_row_count": len(twin_rows),
        "clean_fired_count": sum(1 for row in clean_rows if row.get("boxes")),
        "twin_fired_count": sum(1 for row in twin_rows if row.get("boxes")),
        "clean_mean_boxes": sum(len(row.get("boxes", [])) for row in clean_rows) / max(len(clean_rows), 1),
        "twin_mean_boxes": sum(len(row.get("boxes", [])) for row in twin_rows) / max(len(twin_rows), 1),
        "median_best_iou": _median(twin_ious),
        "max_best_iou": max(twin_ious, default=0.0),
    }
```

- [ ] **Step 3: Add tests for pairwise overlap**

Append to `tests/test_detector_pair_harvest.py`:

```python
from untextre.detector_pair_harvest import pairwise_fire_overlap


def test_pairwise_fire_overlap_reports_sets_without_decision_claims():
    a = {"img1", "img2"}
    b = {"img2", "img3"}
    result = pairwise_fire_overlap("a", a, "b", b, universe={"img1", "img2", "img3", "img4"})
    assert result == {
        "left": "a",
        "right": "b",
        "both": 1,
        "left_only": 1,
        "right_only": 1,
        "neither": 1,
        "jaccard": 1 / 3,
    }
```

- [ ] **Step 4: Implement pairwise overlap helper**

Add to `untextre/detector_pair_harvest.py`:

```python
def pairwise_fire_overlap(left_name: str, left: set[str], right_name: str, right: set[str], universe: set[str]) -> dict:
    both = left & right
    left_only = left - right
    right_only = right - left
    neither = universe - (left | right)
    denom = len(left | right)
    return {
        "left": left_name,
        "right": right_name,
        "both": len(both),
        "left_only": len(left_only),
        "right_only": len(right_only),
        "neither": len(neither),
        "jaccard": len(both) / denom if denom else 0.0,
    }
```

- [ ] **Step 5: Implement analysis script**

Create `scripts/analyze_detector_pair_harvest.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from untextre.detector_pair_harvest import (
    box_metrics_against_truth,
    load_jsonl,
    pairwise_fire_overlap,
    summarize_detector_rows,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detector pair harvest evidence offline.")
    parser.add_argument("harvest_root", type=Path)
    parser.add_argument("--detectors", nargs="+", default=["east", "doctr", "easyocr", "yolo11x"])
    args = parser.parse_args()

    pair_rows = load_jsonl(args.harvest_root / "pairs" / "pair_manifest.jsonl")
    pairs = {row["pair_id"]: row for row in pair_rows}
    universe = set(pairs)
    out_dir = args.harvest_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = {}
    summaries = []
    twin_metric_rows = []
    clean_fire_sets = {}
    twin_fire_sets = {}

    for detector in args.detectors:
        rows = load_jsonl(args.harvest_root / "evidence" / f"{detector}.jsonl")
        evidence[detector] = rows
        summaries.append(summarize_detector_rows(detector, pairs, rows))
        clean_fire_sets[detector] = {row["pair_id"] for row in rows if row.get("state") == "clean" and row.get("boxes")}
        twin_fire_sets[detector] = {row["pair_id"] for row in rows if row.get("state") == "twin" and row.get("boxes")}
        for row in rows:
            if row.get("state") != "twin" or row.get("pair_id") not in pairs:
                continue
            metrics = box_metrics_against_truth(row.get("boxes", []), pairs[row["pair_id"]]["truth_bbox"])
            twin_metric_rows.append({
                "pair_id": row["pair_id"],
                "detector": detector,
                "box_count": metrics["box_count"],
                "max_confidence": metrics["max_confidence"],
                "best_iou": metrics["best_iou"],
                "best_center_distance": metrics["best_center_distance"],
                "best_truth_center_contained": metrics["best_truth_center_contained"],
            })

    write_csv(out_dir / "per_detector_metrics.csv", summaries)

    overlap_rows = []
    for left, right in combinations(args.detectors, 2):
        fp = pairwise_fire_overlap(left, clean_fire_sets[left], right, clean_fire_sets[right], universe)
        fp["state"] = "clean"
        overlap_rows.append(fp)
        tp = pairwise_fire_overlap(left, twin_fire_sets[left], right, twin_fire_sets[right], universe)
        tp["state"] = "twin"
        overlap_rows.append(tp)
    write_csv(out_dir / "pairwise_overlap.csv", overlap_rows)

    with (out_dir / "twin_box_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in twin_metric_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # This initial combination grid is image-fire only, explicitly not a recall claim.
    combo_rows = []
    for size in (2, 3, 4):
        for combo in combinations(args.detectors, size):
            clean_union = set().union(*(clean_fire_sets[name] for name in combo))
            twin_union = set().union(*(twin_fire_sets[name] for name in combo))
            clean_all = set.intersection(*(clean_fire_sets[name] for name in combo)) if combo else set()
            twin_all = set.intersection(*(twin_fire_sets[name] for name in combo)) if combo else set()
            combo_rows.append({
                "detectors": "+".join(combo),
                "rule": "any_fire",
                "clean_images_fired": len(clean_union),
                "twin_images_fired": len(twin_union),
            })
            combo_rows.append({
                "detectors": "+".join(combo),
                "rule": "all_fire",
                "clean_images_fired": len(clean_all),
                "twin_images_fired": len(twin_all),
            })
    write_csv(out_dir / "combination_grid.csv", combo_rows)

    print(out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py -q
```

Expected: pass.

- [ ] **Step 7: Smoke-analyze YOLO smoke output**

Run after Task 3 smoke:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke --detectors yolo11x
```

Expected: `.codex-tmp/detector_pair_harvest_smoke/analysis/per_detector_metrics.csv` exists and has one `yolo11x` row.

- [ ] **Step 8: Commit**

```bash
git add untextre/detector_pair_harvest.py tests/test_detector_pair_harvest.py scripts/analyze_detector_pair_harvest.py
git commit -m "Add detector pair harvest analysis"
```

---

### Task 5: Full N=415 runbook and verification

**Files:**
- Create: `docs/superpowers/plans/2026-07-06-detector-pair-harvest-runbook.md`

**Interfaces:**
- Consumes scripts from Tasks 2-4.
- Produces documented commands for the first full run.

- [ ] **Step 1: Write runbook**

Create `docs/superpowers/plans/2026-07-06-detector-pair-harvest-runbook.md`:

```markdown
# Detector Pair Harvest Runbook

## Build paired corpus

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/build_detector_pair_corpus.py tests/images/zero --out-root tests/images/detector_pair_harvest --seed 20260706 --resume
```

## Harvest detectors

Run each detector independently so a failure does not lose all progress:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors east --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors doctr --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors easyocr --floor 0.01 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py tests/images/detector_pair_harvest --clean-dir tests/images/zero --detectors yolo11x --floor 0.01 --resume
```

## Analyze

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py tests/images/detector_pair_harvest
```

## Expected artifacts

- `pairs/pair_manifest.jsonl`: one row per clean/twin pair
- `evidence/east.jsonl`: two rows per pair
- `evidence/doctr.jsonl`: two rows per pair
- `evidence/easyocr.jsonl`: two rows per pair
- `evidence/yolo11x.jsonl`: two rows per pair
- `analysis/per_detector_metrics.csv`
- `analysis/pairwise_overlap.csv`
- `analysis/twin_box_metrics.jsonl`
- `analysis/combination_grid.csv`
```

- [ ] **Step 2: Smoke the full workflow on `--limit 5`**

Run:

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/build_detector_pair_corpus.py tests/images/zero --out-root .codex-tmp/detector_pair_harvest_smoke5 --seed 20260706 --limit 5
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/run_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke5 --clean-dir tests/images/zero --detectors yolo11x --floor 0.01 --limit 5 --resume
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" scripts/analyze_detector_pair_harvest.py .codex-tmp/detector_pair_harvest_smoke5 --detectors yolo11x
```

Expected: 5 pair rows, 10 YOLO evidence rows, analysis CSVs exist.

- [ ] **Step 3: Run focused tests**

```bash
"C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" -m pytest tests/test_detector_pair_harvest.py tests/test_generated_text_cases.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-06-detector-pair-harvest-runbook.md
git commit -m "Add detector pair harvest runbook"
```

---

## Self-Review Checklist

- Spec coverage: Tasks 1-5 cover paired generation, raw detector evidence, provenance-friendly JSONL, offline analysis, and resumable run commands.
- Placeholder scan: no `TBD`, `TODO`, or unspecified edge handling remains in this plan.
- Type consistency: `pair_id`, `state`, `detector`, `xywh`, `confidence`, and `truth_bbox` names are consistent across helpers, scripts, and tests.
- Threshold discipline: the plan uses detector harvest floors as capture settings only. It does not define a hard IoU hit threshold.
