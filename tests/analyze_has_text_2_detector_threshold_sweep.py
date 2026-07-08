from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from harvest_has_text_2_pipeline_bboxes import area, is_ambiguous, pad_consensus_bbox
from untextre.consensus import find_consensus_boxes
from untextre.metrics import expand_bbox_along_long_axis
from untextre.utils import load_image, pad_bbox_to_multiple


DEFAULT_THRESHOLDS = [0.30, 0.15, 0.10, 0.05, 0.025]
IMAGE_CACHE = {}
EXPANSION_CACHE = {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay detector-specific threshold policies from the loosest has-text-2 sweep records."
    )
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-threshold", type=float, default=0.025)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source_dir = args.sweep_root / threshold_id(args.source_threshold)
    source_jsonl = source_dir / "pipeline_bboxes.jsonl"
    if not source_jsonl.exists():
        raise SystemExit(f"missing source JSONL: {source_jsonl}")

    rows = [
        json.loads(line)
        for line in source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    policies = []
    for east, doctr, easyocr in product(args.thresholds, repeat=3):
        policies.append(score_policy(rows, args.input_dir, {"east": east, "doctr": doctr, "easyocr": easyocr}))

    policies.sort(
        key=lambda row: (
            -row["bbox_rate"],
            row["ambiguous_rate"],
            row["width_ge_050_count"],
            row["coverage_ge_006_count"],
            row["east_threshold"],
            row["doctr_threshold"],
            row["easyocr_threshold"],
        )
    )

    out_path = args.out or (args.sweep_root / "detector_specific_policy_grid.csv")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policies[0].keys()))
        writer.writeheader()
        writer.writerows(policies)

    (out_path.with_suffix(".json")).write_text(
        json.dumps(policies, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def score_policy(rows: list[dict], input_dir: Path, thresholds: dict[str, float]) -> dict:
    bbox_rows = 0
    ambiguous_rows = 0
    unresolved_rows = 0
    multi_bbox_rows = 0
    width_ge_033_rows = 0
    width_ge_050_rows = 0
    growth_ge_4_rows = 0
    coverage_ge_006_rows = 0
    max_total_area_fraction = 0.0
    failover_counts: dict[str, int] = {}

    for row in rows:
        image_path = input_dir / row["relative_path"]
        image = None
        h = int(row.get("height", 0))
        w = int(row.get("width", 0))
        bbox_records = []
        failover_type = "unresolved"
        for stage in row.get("stages", []):
            boxes = replay_stage(stage, thresholds, (h, w))
            if not boxes:
                continue
            failover_type = stage.get("stage", "unknown")
            if failover_type == "normal":
                failover_type = "none"
            if image is None and image_path.exists():
                image = cached_image(image_path)
            for bbox in boxes:
                expanded = cached_expansion(row["relative_path"], image, bbox) if image is not None else bbox
                bbox_records.append(
                    {
                        "raw_or_failover_bbox": list(map(int, bbox)),
                        "expanded": list(map(int, expanded)),
                        "expanded_changed": expanded != bbox,
                        "area_fraction": area(expanded) / max(w * h, 1),
                        "width_fraction": expanded[2] / w if w else 0.0,
                        "height_fraction": expanded[3] / h if h else 0.0,
                        "growth_ratio": area(expanded) / max(area(bbox), 1),
                    }
                )
            break

        failover_counts[failover_type] = failover_counts.get(failover_type, 0) + 1
        if not bbox_records:
            unresolved_rows += 1
            continue
        bbox_rows += 1
        if len(bbox_records) >= 2:
            multi_bbox_rows += 1
        total_area_fraction = sum(area(tuple(box["expanded"])) for box in bbox_records) / max(w * h, 1)
        max_total_area_fraction = max(max_total_area_fraction, total_area_fraction)
        if is_ambiguous(bbox_records, total_area_fraction):
            ambiguous_rows += 1
        if any(box["width_fraction"] >= 0.33 for box in bbox_records):
            width_ge_033_rows += 1
        if any(box["width_fraction"] >= 0.5 for box in bbox_records):
            width_ge_050_rows += 1
        if any(box["growth_ratio"] >= 4.0 for box in bbox_records):
            growth_ge_4_rows += 1
        if total_area_fraction >= 0.06:
            coverage_ge_006_rows += 1

    image_count = max(len(rows), 1)
    return {
        "east_threshold": thresholds["east"],
        "doctr_threshold": thresholds["doctr"],
        "easyocr_threshold": thresholds["easyocr"],
        "image_count": len(rows),
        "bbox_image_count": bbox_rows,
        "bbox_rate": bbox_rows / image_count,
        "unresolved_count": unresolved_rows,
        "unresolved_rate": unresolved_rows / image_count,
        "ambiguous_count": ambiguous_rows,
        "ambiguous_rate": ambiguous_rows / image_count,
        "multi_bbox_count": multi_bbox_rows,
        "width_ge_033_count": width_ge_033_rows,
        "width_ge_050_count": width_ge_050_rows,
        "growth_ge_4_count": growth_ge_4_rows,
        "coverage_ge_006_count": coverage_ge_006_rows,
        "max_total_area_fraction": max_total_area_fraction,
        "failover_counts": json.dumps(failover_counts, sort_keys=True),
    }


def replay_stage(stage: dict, thresholds: dict[str, float], image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    detections = {}
    for name, threshold in thresholds.items():
        detections[name] = [
            tuple(box["bbox"] + [box["confidence"]])
            for box in stage.get("detectors", {}).get(name, [])
            if box.get("confidence", 0.0) >= threshold * 100.0
        ]
    consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
    boxes = []
    for item in consensus:
        padded = pad_consensus_bbox(tuple(item["bbox"]), image_shape)
        boxes.append(pad_bbox_to_multiple(padded, multiple=4, image_shape=image_shape))
    return boxes


def cached_image(image_path: Path):
    key = str(image_path)
    if key not in IMAGE_CACHE:
        IMAGE_CACHE[key] = load_image(image_path)
    return IMAGE_CACHE[key]


def cached_expansion(relative_path: str, image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    key = (relative_path, tuple(map(int, bbox)))
    if key not in EXPANSION_CACHE:
        EXPANSION_CACHE[key] = expand_bbox_along_long_axis(image, bbox)
    return EXPANSION_CACHE[key]


def threshold_id(threshold: float) -> str:
    return f"threshold_{int(round(threshold * 1000)):03d}"


if __name__ == "__main__":
    main()
