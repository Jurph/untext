from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from harvest_has_text_2_pipeline_bboxes import safe_id
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest
    from tests.harvest_has_text_2_pipeline_bboxes import safe_id
from untextre.utils import IMAGE_EXTENSIONS


DEFAULT_THRESHOLDS = [0.30, 0.10, 0.05, 0.025]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a replay matrix from has-text-2 detector threshold runs.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("tests/images/has_text_2_detector_threshold_sweep"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Detector confidence thresholds to include in the matrix.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/images/has_text_2_detector_threshold_sweep/replay_matrix"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows = {
        threshold_id(threshold): load_threshold_rows(args.sweep_root / threshold_id(threshold))
        for threshold in args.thresholds
    }

    image_paths = [
        path
        for path in sorted(args.input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    rows = []
    missing_by_threshold = {threshold_id(threshold): 0 for threshold in args.thresholds}
    for image_path in image_paths:
        image_id = safe_id(image_path, args.input_dir)
        row = {
            "image_id": image_id,
            "relative_path": str(image_path.relative_to(args.input_dir)),
            "path": str(image_path),
            "thresholds": {},
        }
        width = None
        height = None
        for threshold in args.thresholds:
            key = threshold_id(threshold)
            source_row = threshold_rows[key].get(image_id)
            if source_row is None:
                row["thresholds"][key] = None
                missing_by_threshold[key] += 1
                continue

            width = source_row.get("width", width)
            height = source_row.get("height", height)
            stage0 = first_stage(source_row)
            row["thresholds"][key] = {
                "confidence": source_row.get("confidence"),
                "failover_type": source_row.get("failover_type"),
                "detectors": {name: [dict(box) for box in boxes] for name, boxes in stage0.items()},
            }

        row["width"] = width
        row["height"] = height
        row["complete"] = all(row["thresholds"][threshold_id(threshold)] is not None for threshold in args.thresholds)
        rows.append(row)

    out_jsonl = args.out_dir / "replay_matrix.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    complete_count = sum(1 for row in rows if row["complete"])
    summary = {
        "image_count": len(rows),
        "complete_count": complete_count,
        "complete_rate": complete_count / max(len(rows), 1),
        "thresholds": [threshold_id(threshold) for threshold in args.thresholds],
        "missing_by_threshold": missing_by_threshold,
    }
    (args.out_dir / "replay_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_threshold_rows(threshold_dir: Path) -> dict[str, dict]:
    records_dir = threshold_dir / "records"
    rows = {}
    if not records_dir.exists():
        return rows
    for path in records_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if "image_id" in row:
            rows[row["image_id"]] = row
    return rows


def first_stage(row: dict) -> dict[str, list]:
    stages = row.get("stages", [])
    if not stages:
        return {"east": [], "doctr": [], "easyocr": []}
    detectors = stages[0].get("detectors", {})
    return {
        "east": detectors.get("east", []),
        "doctr": detectors.get("doctr", []),
        "easyocr": detectors.get("easyocr", []),
    }


def threshold_id(threshold: float) -> str:
    return f"threshold_{int(round(threshold * 1000)):03d}"


if __name__ == "__main__":
    main()
