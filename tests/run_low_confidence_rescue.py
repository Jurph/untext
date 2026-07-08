from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harvest_has_text_2_pipeline_bboxes import draw_overlay, empty_review, run_one, safe_id, write_summary
from untextre.utils import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run low-confidence bbox rescue for previously unresolved records.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/images/has_text_2_low_confidence_rescue"))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--color-sensitivity", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    baseline_rows = [
        json.loads(line)
        for line in args.baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unresolved = [row for row in baseline_rows if row.get("failover_type") == "unresolved"]
    if args.limit is not None:
        unresolved = unresolved[: args.limit]

    records_dir = args.out_dir / "records"
    overlays_dir = args.out_dir / "overlays"
    records_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, baseline in enumerate(unresolved, start=1):
        image_path = args.input_dir / baseline["relative_path"]
        image_id = safe_id(image_path, args.input_dir)
        record_path = records_dir / f"{image_id}.json"
        if args.resume and record_path.exists():
            rows.append(json.loads(record_path.read_text(encoding="utf-8")))
            print(f"[{index}/{len(unresolved)}] resume {ascii(baseline['relative_path'])}", flush=True)
            continue

        if not image_path.exists():
            row = {
                "image_id": image_id,
                "path": str(image_path),
                "relative_path": baseline["relative_path"],
                "baseline_failover_type": baseline.get("failover_type"),
                "error": "missing from input_dir",
                "bbox_records": [],
                "bbox_count": 0,
                "failover_type": "missing",
                "skipped": True,
                "review": empty_review(),
            }
            record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(row)
            print(f"[{index}/{len(unresolved)}] missing {ascii(baseline['relative_path'])}", flush=True)
            continue

        start = time.time()
        print(f"[{index}/{len(unresolved)}] rescue {ascii(baseline['relative_path'])}", flush=True)
        row = run_one(image_path, args.input_dir, args.confidence, args.color_sensitivity)
        row["baseline_failover_type"] = baseline.get("failover_type")
        row["baseline_confidence"] = baseline.get("confidence")
        row["rescue_confidence"] = args.confidence
        row["review"] = empty_review()
        row["elapsed_sec"] = time.time() - start
        row["overlay_relpath"] = str(Path("overlays") / f"{image_id}.jpg") if args.save_overlays and row.get("bbox_records") else None
        record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)

        if args.save_overlays and row.get("bbox_records"):
            image = load_image(image_path)
            overlay = draw_overlay(image, row)
            import cv2

            cv2.imwrite(str(overlays_dir / f"{image_id}.jpg"), overlay)

    write_summary(args.out_dir, rows)


if __name__ == "__main__":
    main()
