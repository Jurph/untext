from __future__ import annotations

import argparse
import csv
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

try:
    from harvest_pipeline_bboxes import area, area as bbox_area, is_ambiguous, pad_consensus_bbox
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest
    from scripts.has_text_2.harvest_pipeline_bboxes import area, area as bbox_area, is_ambiguous, pad_consensus_bbox
from untextre.consensus import find_consensus_boxes
from untextre.metrics import expand_bbox_along_long_axis
from untextre.utils import load_image, pad_bbox_to_multiple

DEFAULT_THRESHOLDS = [0.30, 0.10, 0.05, 0.025]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the current merge policy against a detector replay matrix.")
    parser.add_argument(
        "matrix_jsonl",
        type=Path,
        default=Path("tests/images/has_text_2_detector_threshold_sweep/replay_matrix/replay_matrix.jsonl"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Detector thresholds to replay across east/yolo11x/easyocr.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/images/has_text_2_detector_threshold_sweep/replay_policy_eval"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.matrix_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    image_cache, skip_records = build_image_cache(rows)
    if skip_records:
        skip_path = args.out_dir / "replay_policy_skips.jsonl"
        with skip_path.open("w", encoding="utf-8") as handle:
            for item in skip_records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    combos = list(itertools.product(args.thresholds, repeat=3))
    evaluations = []
    for east_threshold, yolo11x_threshold, easyocr_threshold in combos:
        evaluations.append(
            evaluate_combo(
                rows,
                image_cache=image_cache,
                east_threshold=east_threshold,
                yolo11x_threshold=yolo11x_threshold,
                easyocr_threshold=easyocr_threshold,
            )
        )

    evaluations.sort(
        key=lambda row: (
            -row["recall"],
            row["ambiguous_rate"],
            row["coverage_ge_006_rate"],
            row["width_ge_050_rate"],
            row["east_threshold"],
            row["yolo11x_threshold"],
            row["easyocr_threshold"],
        )
    )

    csv_path = args.out_dir / "merge_policy_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluations[0].keys()))
        writer.writeheader()
        writer.writerows(evaluations)

    top = evaluations[:20]
    summary = {
        "matrix_rows": len(rows),
        "combo_count": len(evaluations),
        "skip_count": len(skip_records),
        "top_rows": top,
        "best_combo": top[0] if top else None,
        "current_policy_rows": {
            "0.30/0.30/0.30": find_combo(evaluations, 0.30, 0.30, 0.30),
            "0.10/0.10/0.10": find_combo(evaluations, 0.10, 0.10, 0.10),
            "0.05/0.05/0.05": find_combo(evaluations, 0.05, 0.05, 0.05),
            "0.025/0.025/0.025": find_combo(evaluations, 0.025, 0.025, 0.025),
        },
    }
    (args.out_dir / "merge_policy_grid.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_image_cache(rows: list[dict]) -> tuple[dict[str, np.ndarray], list[dict]]:
    cache: dict[str, np.ndarray] = {}
    skips: list[dict] = []
    for row in rows:
        image, reason = load_exact_image(row)
        if image is None:
            skips.append(
                {
                    "image_id": row.get("image_id"),
                    "path": row.get("path"),
                    "relative_path": row.get("relative_path"),
                    "reason": reason,
                }
            )
            continue
        image_id = row.get("image_id")
        if image_id:
            cache[str(image_id)] = image
    return cache, skips


def evaluate_combo(
    rows: list[dict],
    *,
    image_cache: dict[str, np.ndarray],
    east_threshold: float,
    yolo11x_threshold: float,
    easyocr_threshold: float,
) -> dict:
    geometry_image_count = 0
    geometry_bbox_image_count = 0
    geometry_ambiguous_count = 0
    geometry_multi_bbox_count = 0
    geometry_coverage_ge_006_count = 0
    geometry_width_ge_050_count = 0
    geometry_width_ge_033_count = 0
    geometry_growth_ge_4_count = 0
    geometry_total_area_fraction_sum = 0.0
    geometry_total_box_count = 0

    expanded_image_count = 0
    expanded_bbox_image_count = 0
    expanded_ambiguous_count = 0
    expanded_multi_bbox_count = 0
    expanded_coverage_ge_006_count = 0
    expanded_width_ge_050_count = 0
    expanded_width_ge_033_count = 0
    expanded_growth_ge_4_count = 0
    expanded_total_area_fraction_sum = 0.0
    expanded_total_box_count = 0

    for row in rows:
        cell_east = row["thresholds"].get(threshold_id(east_threshold))
        cell_yolo11x = row["thresholds"].get(threshold_id(yolo11x_threshold))
        cell_easyocr = row["thresholds"].get(threshold_id(easyocr_threshold))
        if cell_east is None or cell_yolo11x is None or cell_easyocr is None:
            continue
        width = row.get("width")
        height = row.get("height")
        if not width or not height:
            continue

        geometry_image_count += 1
        detections = {
            "east": [tuple(box["bbox"] + [box["confidence"]]) for box in cell_east["detectors"].get("east", []) if box["confidence"] >= east_threshold * 100.0],
            "yolo11x": [tuple(box["bbox"] + [box["confidence"]]) for box in cell_yolo11x["detectors"].get("yolo11x", []) if box["confidence"] >= yolo11x_threshold * 100.0],
            "easyocr": [tuple(box["bbox"] + [box["confidence"]]) for box in cell_easyocr["detectors"].get("easyocr", []) if box["confidence"] >= easyocr_threshold * 100.0],
        }
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        geometry_boxes = []
        for item in consensus:
            padded = pad_consensus_bbox(tuple(item["bbox"]), (height, width))
            mod4 = pad_bbox_to_multiple(padded, multiple=4, image_shape=(height, width))
            geometry_boxes.append(
                {
                    "raw_union": list(item["bbox"]),
                    "expanded": list(mod4),
                    "width_fraction": mod4[2] / width if width else 0.0,
                    "height_fraction": mod4[3] / height if height else 0.0,
                    "growth_ratio": bbox_area(mod4) / max(bbox_area(tuple(item["bbox"])), 1),
                }
            )

        if geometry_boxes:
            geometry_bbox_image_count += 1
            geometry_total_box_count += len(geometry_boxes)
            geometry_total_area_fraction = sum(area(tuple(box["expanded"])) for box in geometry_boxes) / max(width * height, 1)
            geometry_total_area_fraction_sum += geometry_total_area_fraction
            if is_ambiguous(geometry_boxes, geometry_total_area_fraction):
                geometry_ambiguous_count += 1
            if len(geometry_boxes) >= 2:
                geometry_multi_bbox_count += 1
            if geometry_total_area_fraction >= 0.06:
                geometry_coverage_ge_006_count += 1
            if any(box["width_fraction"] >= 0.5 for box in geometry_boxes):
                geometry_width_ge_050_count += 1
            if any(box["width_fraction"] >= 0.33 for box in geometry_boxes):
                geometry_width_ge_033_count += 1
            if any(box["growth_ratio"] >= 4.0 for box in geometry_boxes):
                geometry_growth_ge_4_count += 1

        image_id = str(row.get("image_id", ""))
        image = image_cache.get(image_id)
        if image is None:
            continue

        expanded_image_count += 1
        expanded_boxes = []
        for box in geometry_boxes:
            try:
                expanded = expand_bbox_along_long_axis(image, tuple(box["expanded"]))
            except Exception:
                expanded_boxes = []
                break
            expanded_boxes.append(
                {
                    "raw_union": box["raw_union"],
                    "expanded": list(expanded),
                    "width_fraction": expanded[2] / width if width else 0.0,
                    "height_fraction": expanded[3] / height if height else 0.0,
                    "growth_ratio": bbox_area(tuple(expanded)) / max(bbox_area(tuple(box["raw_union"])), 1),
                }
            )

        if not expanded_boxes:
            continue

        expanded_bbox_image_count += 1
        expanded_total_box_count += len(expanded_boxes)
        expanded_total_area_fraction = sum(area(tuple(box["expanded"])) for box in expanded_boxes) / max(width * height, 1)
        expanded_total_area_fraction_sum += expanded_total_area_fraction
        if is_ambiguous(expanded_boxes, expanded_total_area_fraction):
            expanded_ambiguous_count += 1
        if len(expanded_boxes) >= 2:
            expanded_multi_bbox_count += 1
        if expanded_total_area_fraction >= 0.06:
            expanded_coverage_ge_006_count += 1
        if any(box["width_fraction"] >= 0.5 for box in expanded_boxes):
            expanded_width_ge_050_count += 1
        if any(box["width_fraction"] >= 0.33 for box in expanded_boxes):
            expanded_width_ge_033_count += 1
        if any(box["growth_ratio"] >= 4.0 for box in expanded_boxes):
            expanded_growth_ge_4_count += 1

    geometry_recall = geometry_bbox_image_count / max(geometry_image_count, 1)
    geometry_clean_image_count = geometry_bbox_image_count - geometry_ambiguous_count
    geometry_precision_proxy = geometry_clean_image_count / max(geometry_bbox_image_count, 1)
    geometry_clean_rate = geometry_clean_image_count / max(geometry_image_count, 1)

    recall = expanded_bbox_image_count / max(expanded_image_count, 1)
    clean_image_count = expanded_bbox_image_count - expanded_ambiguous_count
    precision_proxy = clean_image_count / max(expanded_bbox_image_count, 1)
    clean_rate = clean_image_count / max(expanded_image_count, 1)

    return {
        "east_threshold": east_threshold,
        "yolo11x_threshold": yolo11x_threshold,
        "easyocr_threshold": easyocr_threshold,
        "image_count": geometry_image_count,
        "valid_rows": geometry_image_count,
        "geometry_image_count": geometry_image_count,
        "geometry_bbox_image_count": geometry_bbox_image_count,
        "geometry_recall": geometry_recall,
        "geometry_clean_image_count": geometry_clean_image_count,
        "geometry_clean_rate": geometry_clean_rate,
        "geometry_precision_proxy": geometry_precision_proxy,
        "geometry_ambiguous_count": geometry_ambiguous_count,
        "geometry_ambiguous_rate": geometry_ambiguous_count / max(geometry_image_count, 1),
        "geometry_multi_bbox_count": geometry_multi_bbox_count,
        "geometry_multi_bbox_rate": geometry_multi_bbox_count / max(geometry_image_count, 1),
        "geometry_width_ge_033_count": geometry_width_ge_033_count,
        "geometry_width_ge_033_rate": geometry_width_ge_033_count / max(geometry_image_count, 1),
        "geometry_width_ge_050_count": geometry_width_ge_050_count,
        "geometry_width_ge_050_rate": geometry_width_ge_050_count / max(geometry_image_count, 1),
        "geometry_growth_ge_4_count": geometry_growth_ge_4_count,
        "geometry_growth_ge_4_rate": geometry_growth_ge_4_count / max(geometry_image_count, 1),
        "geometry_coverage_ge_006_count": geometry_coverage_ge_006_count,
        "geometry_coverage_ge_006_rate": geometry_coverage_ge_006_count / max(geometry_image_count, 1),
        "geometry_mean_total_area_fraction": geometry_total_area_fraction_sum / max(geometry_bbox_image_count, 1),
        "geometry_mean_boxes_per_image": geometry_total_box_count / max(geometry_image_count, 1),
        "expanded_image_count": expanded_image_count,
        "expanded_bbox_image_count": expanded_bbox_image_count,
        "recall": recall,
        "clean_image_count": clean_image_count,
        "clean_rate": clean_rate,
        "precision_proxy": precision_proxy,
        "ambiguous_count": expanded_ambiguous_count,
        "ambiguous_rate": expanded_ambiguous_count / max(expanded_image_count, 1),
        "multi_bbox_count": expanded_multi_bbox_count,
        "multi_bbox_rate": expanded_multi_bbox_count / max(expanded_image_count, 1),
        "width_ge_033_count": expanded_width_ge_033_count,
        "width_ge_033_rate": expanded_width_ge_033_count / max(expanded_image_count, 1),
        "width_ge_050_count": expanded_width_ge_050_count,
        "width_ge_050_rate": expanded_width_ge_050_count / max(expanded_image_count, 1),
        "growth_ge_4_count": expanded_growth_ge_4_count,
        "growth_ge_4_rate": expanded_growth_ge_4_count / max(expanded_image_count, 1),
        "coverage_ge_006_count": expanded_coverage_ge_006_count,
        "coverage_ge_006_rate": expanded_coverage_ge_006_count / max(expanded_image_count, 1),
        "mean_total_area_fraction": expanded_total_area_fraction_sum / max(expanded_bbox_image_count, 1),
        "mean_boxes_per_image": expanded_total_box_count / max(expanded_image_count, 1),
    }


def load_exact_image(row: dict) -> tuple[np.ndarray | None, str | None]:
    path_value = row.get("path")
    if not path_value:
        return None, "missing_path"

    image_path = Path(path_value)
    if not image_path.exists():
        return None, "missing_file"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            image = load_image(image_path)
        except Warning as exc:  # pragma: no cover - defensive
            return None, f"warning: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return None, f"load_error: {type(exc).__name__}: {exc}"

    height = row.get("height")
    width = row.get("width")
    if height and width and tuple(image.shape[:2]) != (int(height), int(width)):
        return None, f"shape_mismatch: got={tuple(image.shape[:2])} expected={(int(height), int(width))}"

    return image, None


def find_combo(rows: list[dict], east_threshold: float, yolo11x_threshold: float, easyocr_threshold: float) -> dict | None:
    key = (round(east_threshold, 3), round(yolo11x_threshold, 3), round(easyocr_threshold, 3))
    for row in rows:
        if (
            round(row["east_threshold"], 3),
            round(row["yolo11x_threshold"], 3),
            round(row["easyocr_threshold"], 3),
        ) == key:
            return row
    return None


def threshold_id(threshold: float) -> str:
    return f"threshold_{int(round(threshold * 1000)):03d}"


if __name__ == "__main__":
    main()
