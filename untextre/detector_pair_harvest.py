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
