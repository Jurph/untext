from __future__ import annotations

import json
import math
import re
from statistics import median
from pathlib import Path
from typing import Sequence


def image_key(path: Path | str) -> str:
    """Return a filename-safe ID that preserves extension identity."""
    p = Path(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem).strip("_") or "image"
    suffix = re.sub(r"[^A-Za-z0-9]+", "", p.suffix.lower().lstrip(".")) or "nosuffix"
    return f"{stem}__{suffix}"


def normalize_detection_box(
    xywh: Sequence[float],
    confidence: float,
    label: str,
    raw_payload: dict | None = None,
) -> dict:
    return {
        "xywh": [round(float(value), 1) for value in xywh],
        "confidence": round(float(confidence), 4),
        "label": str(label),
        "raw_payload": dict(raw_payload or {}),
    }


def bbox_area(box: Sequence[float]) -> float:
    _x, _y, width, height = [float(value) for value in box]
    return max(0.0, width) * max(0.0, height)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(value) for value in a]
    bx, by, bw, bh = [float(value) for value in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    x, y, width, height = [float(value) for value in box]
    return (x + width / 2.0, y + height / 2.0)


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def contains_point(box: Sequence[float], point: tuple[float, float]) -> bool:
    x, y, width, height = [float(value) for value in box]
    px, py = point
    return x <= px <= x + width and y <= py <= y + height


def box_metrics_against_truth(boxes: list[dict], truth_bbox: Sequence[float]) -> dict:
    truth_area = bbox_area(truth_bbox)
    truth_width = max(float(truth_bbox[2]), 1.0)
    truth_height = max(float(truth_bbox[3]), 1.0)
    truth_center = bbox_center(truth_bbox)
    metrics = []
    for box in boxes:
        xywh = box["xywh"]
        metrics.append(
            {
                "xywh": xywh,
                "confidence": float(box.get("confidence", 0.0)),
                "iou": bbox_iou(xywh, truth_bbox),
                "center_distance": center_distance(xywh, truth_bbox),
                "truth_center_contained": contains_point(xywh, truth_center),
                "area_ratio": bbox_area(xywh) / truth_area if truth_area > 0 else 0.0,
                "width_ratio": float(xywh[2]) / truth_width,
                "height_ratio": float(xywh[3]) / truth_height,
            }
        )
    metrics.sort(key=lambda item: (item["iou"], item["confidence"]), reverse=True)
    return {
        "box_count": len(boxes),
        "max_confidence": max((float(box.get("confidence", 0.0)) for box in boxes), default=0.0),
        "best_iou": metrics[0]["iou"] if metrics else 0.0,
        "best_center_distance": metrics[0]["center_distance"] if metrics else None,
        "best_truth_center_contained": metrics[0]["truth_center_contained"] if metrics else False,
        "top_boxes": metrics[:5],
    }


def summarize_detector_rows(detector: str, pairs: dict[str, dict], rows: list[dict]) -> dict:
    """Summarize one detector's clean/twin evidence without choosing hit thresholds."""
    clean_rows = [row for row in rows if row.get("state") == "clean"]
    twin_rows = [row for row in rows if row.get("state") == "twin"]
    clean_fires: set[str] = set()
    twin_fires: set[str] = set()
    twin_ious: list[float] = []
    twin_confidences: list[float] = []
    clean_confidences: list[float] = []

    for row in clean_rows:
        pair_id = row.get("pair_id")
        boxes = row.get("boxes") or []
        if not boxes:
            continue
        clean_fires.add(pair_id)
        clean_confidences.append(max((float(box.get("confidence", 0.0)) for box in boxes), default=0.0))

    for row in twin_rows:
        pair_id = row.get("pair_id")
        boxes = row.get("boxes") or []
        if boxes:
            twin_fires.add(pair_id)
            twin_confidences.append(max((float(box.get("confidence", 0.0)) for box in boxes), default=0.0))
        pair = pairs.get(pair_id, {})
        truth_bbox = pair.get("truth_bbox")
        if truth_bbox is not None:
            twin_ious.append(box_metrics_against_truth(boxes, truth_bbox)["best_iou"])

    pair_count = len(pairs)
    return {
        "detector": detector,
        "pair_count": pair_count,
        "clean_row_count": len(clean_rows),
        "twin_row_count": len(twin_rows),
        "clean_fired_count": len(clean_fires),
        "twin_fired_count": len(twin_fires),
        "clean_mean_boxes": sum(len(row.get("boxes", [])) for row in clean_rows) / max(len(clean_rows), 1),
        "twin_mean_boxes": sum(len(row.get("boxes", [])) for row in twin_rows) / max(len(twin_rows), 1),
        "clean_fire_rate": len(clean_fires) / pair_count if pair_count else 0.0,
        "twin_fire_rate": len(twin_fires) / pair_count if pair_count else 0.0,
        "median_best_iou": float(median(twin_ious)) if twin_ious else 0.0,
        "max_best_iou": max(twin_ious, default=0.0),
        "max_clean_confidence": max(clean_confidences, default=0.0),
        "max_twin_confidence": max(twin_confidences, default=0.0),
        "clean_fire_pair_ids": sorted(clean_fires),
        "twin_fire_pair_ids": sorted(twin_fires),
    }


def pairwise_fire_overlap(
    left: str,
    left_fires: set[str],
    right: str,
    right_fires: set[str],
    universe: set[str],
) -> dict:
    """Compare detector fire sets without implying whether the fires are correct."""
    both = left_fires & right_fires
    left_only = left_fires - right_fires
    right_only = right_fires - left_fires
    union = left_fires | right_fires
    return {
        "left": left,
        "right": right,
        "both": len(both),
        "left_only": len(left_only),
        "right_only": len(right_only),
        "neither": len(universe - union),
        "jaccard": len(both) / len(union) if union else 0.0,
    }



def pair_id_for_path(path: Path | str) -> str:
    return image_key(path)


def build_pair_row(
    pair_id: str,
    clean_relative_path: str,
    twin_relative_path: str,
    case_metadata: dict,
    truth_bbox: Sequence[int],
    width: int,
    height: int,
) -> dict:
    return {
        "pair_id": pair_id,
        "clean_relative_path": clean_relative_path,
        "twin_relative_path": twin_relative_path,
        "base_width": int(width),
        "base_height": int(height),
        "truth_bbox": [int(value) for value in truth_bbox],
        "truth_bbox_coverage": case_metadata.get("truth_bbox_coverage"),
        "truth_alpha_coverage": case_metadata.get("truth_alpha_coverage"),
        "measured_visibility_delta_e": case_metadata.get("measured_visibility_delta_e"),
        "visibility_attempts": case_metadata.get("visibility_attempts"),
        "visibility_fallback": case_metadata.get("visibility_fallback"),
        "synthetic_metadata": dict(case_metadata),
    }

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
