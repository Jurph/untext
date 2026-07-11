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

DEFAULT_DETECTORS = ["east", "easyocr", "yolo11x"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze detector pair harvest evidence offline.")
    parser.add_argument("harvest_root", type=Path)
    parser.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = load_jsonl(args.harvest_root / "pairs" / "pair_manifest.jsonl")
    pairs = {row["pair_id"]: row for row in pair_rows}
    universe = set(pairs)
    out_dir = args.harvest_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    twin_metric_rows: list[dict] = []
    clean_fire_sets: dict[str, set[str]] = {}
    twin_fire_sets: dict[str, set[str]] = {}

    for detector in args.detectors:
        rows = load_jsonl(args.harvest_root / "evidence" / f"{detector}.jsonl")
        summaries.append(summarize_detector_rows(detector, pairs, rows))
        clean_fire_sets[detector] = {
            row["pair_id"] for row in rows if row.get("state") == "clean" and row.get("boxes")
        }
        twin_fire_sets[detector] = {
            row["pair_id"] for row in rows if row.get("state") == "twin" and row.get("boxes")
        }

        for row in rows:
            if row.get("state") != "twin" or row.get("pair_id") not in pairs:
                continue
            metrics = box_metrics_against_truth(row.get("boxes", []), pairs[row["pair_id"]]["truth_bbox"])
            twin_metric_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "detector": detector,
                    "box_count": metrics["box_count"],
                    "max_confidence": metrics["max_confidence"],
                    "best_iou": metrics["best_iou"],
                    "best_center_distance": metrics["best_center_distance"],
                    "best_truth_center_contained": metrics["best_truth_center_contained"],
                }
            )

    write_csv(out_dir / "per_detector_metrics.csv", summaries)

    overlap_rows: list[dict] = []
    for left, right in combinations(args.detectors, 2):
        fp = pairwise_fire_overlap(
            left,
            clean_fire_sets[left],
            right,
            clean_fire_sets[right],
            universe=universe,
        )
        fp["state"] = "clean"
        overlap_rows.append(fp)
        tp = pairwise_fire_overlap(
            left,
            twin_fire_sets[left],
            right,
            twin_fire_sets[right],
            universe=universe,
        )
        tp["state"] = "twin"
        overlap_rows.append(tp)
    write_csv(out_dir / "pairwise_overlap.csv", overlap_rows)

    with (out_dir / "twin_box_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in twin_metric_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    combo_rows: list[dict] = []
    for size in range(2, min(4, len(args.detectors)) + 1):
        for combo in combinations(args.detectors, size):
            clean_union = set().union(*(clean_fire_sets[name] for name in combo))
            twin_union = set().union(*(twin_fire_sets[name] for name in combo))
            clean_all = set.intersection(*(clean_fire_sets[name] for name in combo))
            twin_all = set.intersection(*(twin_fire_sets[name] for name in combo))
            combo_rows.append(
                {
                    "detectors": "+".join(combo),
                    "rule": "any_fire",
                    "clean_images_fired": len(clean_union),
                    "twin_images_fired": len(twin_union),
                }
            )
            combo_rows.append(
                {
                    "detectors": "+".join(combo),
                    "rule": "all_fire",
                    "clean_images_fired": len(clean_all),
                    "twin_images_fired": len(twin_all),
                }
            )
    write_csv(out_dir / "combination_grid.csv", combo_rows)

    print(out_dir)


if __name__ == "__main__":
    main()
