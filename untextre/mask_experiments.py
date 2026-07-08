"""Deterministic experiment helpers for generated-text mask sweeps."""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_MANIFEST = Path("tests/images/generated_text_watermarks/manifest.json")


@dataclass(frozen=True)
class MaskExperimentConfig:
    preset: str = "custom"
    config_id: str = "custom"
    cleanup_dilate_px: int = 13
    cleanup_close_px: int = 11
    fom_threshold: float = 0.30
    cc_guard: float = 0.85
    color_radius_multiplier: float = 1.0
    bg_radius_multiplier: float = 1.0
    expand_factor: float = 2.0
    inside_rejection_credit: float = 1.0
    max_budget_fraction: float = 0.25
    long_weight: float = 0.5
    short_weight: float = 4.0
    short_axis_power: float = 1.5
    connectivity_dilation_px: int = 3
    min_cc_px: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MaskExperimentConfig":
        allowed = cls.__dataclass_fields__.keys()
        values = {key: data[key] for key in allowed if key in data}
        return cls(**values)


PRESET_GRIDS = {
    "local-cleanup": {
        "cleanup_dilate_px": [0, 1, 2, 3, 5, 7],
        "cleanup_close_px": [0, 1, 3, 5],
        "fom_threshold": [0.20, 0.25, 0.30, 0.35],
        "cc_guard": [0.65, 0.75, 0.85],
    },
    "color-proposal": {
        "color_radius_multiplier": [0.75, 1.0, 1.25, 1.5],
        "bg_radius_multiplier": [0.75, 1.0, 1.25],
    },
    "geometry-budget": {
        "expand_factor": [1.25, 1.5, 2.0, 2.5],
        "inside_rejection_credit": [0.5, 1.0, 2.0, 4.0],
        "max_budget_fraction": [0.10, 0.25, 0.50, 0.75],
        "long_weight": [0.25, 0.5, 1.0],
        "short_weight": [2.0, 4.0, 8.0, 12.0],
        "short_axis_power": [1.25, 1.5, 2.0],
        "connectivity_dilation_px": [1, 3, 5],
        "min_cc_px": [1, 5, 10, 25],
    },
}


def iter_preset_configs(preset: str) -> Iterable[MaskExperimentConfig]:
    if preset == "final-neighborhood":
        base = list(itertools.islice(iter_preset_configs("local-cleanup"), 5))
        for idx, cfg in enumerate(base):
            yield MaskExperimentConfig(
                **{
                    **cfg.to_dict(),
                    "preset": preset,
                    "config_id": f"{preset}-{idx:06d}",
                }
            )
        return
    if preset not in PRESET_GRIDS:
        raise ValueError(f"Unknown preset: {preset}")

    grid = PRESET_GRIDS[preset]
    keys = list(grid.keys())
    for idx, values in enumerate(itertools.product(*(grid[key] for key in keys))):
        data = dict(zip(keys, values))
        yield MaskExperimentConfig(preset=preset, config_id=f"{preset}-{idx:06d}", **data)


def truth_target_mask(truth_mask: np.ndarray, dilation_px: int = 2) -> np.ndarray:
    mask = _binary(truth_mask)
    if dilation_px <= 0:
        return mask
    size = dilation_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel)


def weighted_precision(predicted: np.ndarray, target: np.ndarray) -> float:
    pred = _binary_bool(predicted)
    tgt = _binary_bool(target)
    tp = int(np.sum(pred & tgt))
    fp = pred & ~tgt
    if tp == 0 and not np.any(fp):
        return 1.0
    if not np.any(fp):
        return 1.0 if tp > 0 else 0.0
    distance = cv2.distanceTransform((~tgt).astype(np.uint8), cv2.DIST_L2, 3)
    fp_cost = np.sum(1.0 + 0.15 * np.power(distance[fp], 1.5))
    return float(tp / (tp + fp_cost)) if (tp + fp_cost) > 0 else 0.0


def compute_mask_metrics(
    predicted: np.ndarray,
    truth_mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    target_dilation_px: int = 2,
) -> dict:
    target = truth_target_mask(truth_mask, target_dilation_px)
    pred = _binary_bool(predicted)
    tgt = _binary_bool(target)
    tp = int(np.sum(pred & tgt))
    fp = pred & ~tgt
    fn = int(np.sum(~pred & tgt))
    pred_px = int(np.sum(pred))
    target_px = int(np.sum(tgt))
    union = int(np.sum(pred | tgt))

    bbox_mask = np.zeros(tgt.shape, dtype=bool)
    x, y, w, h = bbox
    bbox_mask[y:y + h, x:x + w] = True
    fp_inside = int(np.sum(fp & bbox_mask))
    fp_outside = int(np.sum(fp & ~bbox_mask))
    recall = tp / target_px if target_px else 1.0
    precision = tp / pred_px if pred_px else 1.0
    overmask_ratio = pred_px / target_px if target_px else 0.0
    coverage = pred_px / pred.size if pred.size else 0.0
    w_precision = weighted_precision(predicted, target)
    score = (recall ** 4) * (w_precision ** 3) / math.sqrt(max(overmask_ratio, 1e-9))

    return {
        "target_iou": tp / union if union else 1.0,
        "target_precision": precision,
        "target_recall": recall,
        "weighted_precision": w_precision,
        "coverage": coverage,
        "overmask_ratio": overmask_ratio,
        "fp_inside_bbox": fp_inside,
        "fp_outside_bbox": fp_outside,
        "fp_outside_fraction": fp_outside / (fp_inside + fp_outside)
        if (fp_inside + fp_outside)
        else 0.0,
        "predicted_px": pred_px,
        "target_px": target_px,
        "tp": tp,
        "fn": fn,
        "score": score,
    }


def load_manifest_cases(manifest_path: Path) -> list[dict]:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = data.get("cases", data if isinstance(data, list) else [])
    normalized = []
    for case in cases:
        image_rel = case.get("image") or case.get("watermarked")
        mask_rel = case.get("truth_mask") or case.get("mask")
        if image_rel is None or mask_rel is None:
            raise ValueError(f"Manifest case is missing image/truth_mask: {case}")
        bbox = case.get("truth_bbox") or case.get("bbox") or case.get("bbox_xywh")
        if bbox is None:
            raise ValueError(f"Manifest case is missing bbox/truth_bbox/bbox_xywh: {case}")
        normalized.append(
            {
                **case,
                "id": case.get("id") or case.get("slug") or Path(image_rel).stem,
                "image_path": manifest_path.parent / image_rel,
                "truth_mask_path": manifest_path.parent / mask_rel,
                "clean_path": manifest_path.parent / case["clean"] if case.get("clean") else None,
                "truth_bbox": tuple(bbox),
            }
        )
    return normalized


def summarize_grid_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)

    summaries = []
    for config_id, items in grouped.items():
        first = items[0]
        summaries.append(
            {
                "config_id": config_id,
                "preset": first.get("preset", ""),
                "case_count": len(items),
                "mean_target_recall": _mean(item["target_recall"] for item in items),
                "min_target_recall": min(item["target_recall"] for item in items),
                "mean_weighted_precision": _mean(item["weighted_precision"] for item in items),
                "mean_overmask_ratio": _mean(item["overmask_ratio"] for item in items),
                "max_coverage": max(item["coverage"] for item in items),
                "mean_score": _mean(item["score"] for item in items),
                "config": first.get("config", {}),
            }
        )
    return summaries


def rank_mask_summaries(summaries: list[dict]) -> list[dict]:
    filtered = [
        row for row in summaries
        if row["mean_target_recall"] >= 0.98
        and row["min_target_recall"] >= 0.95
        and row["max_coverage"] <= 0.06
    ]
    return sorted(filtered, key=lambda row: (-row["mean_score"], row["config_id"]))


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "preset",
        "case_count",
        "mean_target_recall",
        "min_target_recall",
        "mean_weighted_precision",
        "mean_overmask_ratio",
        "max_coverage",
        "mean_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def _binary_bool(mask: np.ndarray) -> np.ndarray:
    return _binary(mask) > 0


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
