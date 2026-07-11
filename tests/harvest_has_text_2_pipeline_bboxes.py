from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from untextre.consensus import (
    detect_with_yolo11x,
    detect_with_east,
    detect_with_easyocr,
    find_consensus_boxes,
)
from untextre.metrics import expand_bbox_along_long_axis
from untextre.pipeline import _apply_color_enhancement, _translate_rotated_bbox_to_original
from untextre.preprocessor import preprocess_image
from untextre.utils import CLI_DEFAULT_CONFIDENCE, IMAGE_EXTENSIONS, load_image, pad_bbox_to_multiple


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest full-pipeline bbox-of-record data for has-text-2 images."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/images/has_text_2_pipeline_bbox_harvest"),
    )
    parser.add_argument("--confidence", type=float, default=CLI_DEFAULT_CONFIDENCE)
    parser.add_argument("--color-sensitivity", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    image_paths = [
        path
        for path in sorted(args.input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    records_dir = args.out_dir / "records"
    overlays_dir = args.out_dir / "overlays"
    records_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        display_name = ascii(image_path.name)
        image_id = safe_id(image_path, args.input_dir)
        record_path = records_dir / f"{image_id}.json"
        overlay_relpath = str(Path("overlays") / f"{image_id}.jpg") if args.save_overlays else None
        if args.resume and record_path.exists():
            row = json.loads(record_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] resume {display_name}", flush=True)
            continue

        start = time.time()
        print(f"[{index}/{len(image_paths)}] detect {display_name}", flush=True)
        row = run_one(image_path, args.input_dir, args.confidence, args.color_sensitivity)
        row["image_id"] = image_id
        row["overlay_relpath"] = overlay_relpath if row.get("bbox_records") else None
        row["review"] = empty_review()
        row["elapsed_sec"] = time.time() - start
        record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)

        if args.save_overlays and row.get("bbox_records"):
            image = load_image(image_path)
            overlay = draw_overlay(image, row)
            cv2.imwrite(str(overlays_dir / f"{image_id}.jpg"), overlay)

    write_summary(args.out_dir, rows)


def run_one(image_path: Path, input_dir: Path, confidence: float, color_sensitivity: int) -> dict:
    try:
        image = load_image(image_path)
    except ValueError as exc:
        return {
            "image_id": safe_id(image_path, input_dir),
            "path": str(image_path),
            "relative_path": str(image_path.relative_to(input_dir)),
            "error": str(exc),
            "bbox_records": [],
            "bbox_count": 0,
            "failover_type": "load_error",
            "skipped": True,
        }

    preprocessed = preprocess_image(image)
    if preprocessed is None:
        preprocessed = image

    stages = []
    normal_stage = run_detection_stage("normal", preprocessed, image.shape[:2], confidence)
    stages.append(normal_stage)
    consensus_boxes = [tuple(box["mod4_bbox"]) for box in normal_stage["boxes"]]
    failover_type = "none" if consensus_boxes else "unresolved"

    if not consensus_boxes:
        h, w = preprocessed.shape[:2]
        rotated_image = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)
        rotated_stage = run_detection_stage("rotation", rotated_image, rotated_image.shape[:2], confidence)
        stages.append(rotated_stage)
        rotated_boxes = [tuple(box["mod4_bbox"]) for box in rotated_stage["boxes"]]
        if rotated_boxes:
            consensus_boxes = [
                _translate_rotated_bbox_to_original(bbox, (h, w))
                for bbox in rotated_boxes
            ]
            failover_type = "rotation"

    if not consensus_boxes:
        gray_stage = run_color_enhanced_stage(
            "gray_enhancement",
            image,
            confidence,
            "#808080",
            color_sensitivity,
        )
        stages.append(gray_stage)
        consensus_boxes = [tuple(box["mod4_bbox"]) for box in gray_stage["boxes"]]
        if consensus_boxes:
            failover_type = "gray_enhancement"

    if not consensus_boxes:
        white_stage = run_color_enhanced_stage(
            "white_enhancement",
            image,
            confidence,
            "#FFFFFF",
            color_sensitivity,
        )
        stages.append(white_stage)
        consensus_boxes = [tuple(box["mod4_bbox"]) for box in white_stage["boxes"]]
        if consensus_boxes:
            failover_type = "white_enhancement"

    h, w = image.shape[:2]
    bbox_records = []
    for bbox in consensus_boxes:
        expanded = expand_bbox_along_long_axis(image, bbox)
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

    total_area_fraction = sum(area(tuple(box["expanded"])) for box in bbox_records) / max(w * h, 1)
    return {
        "image_id": safe_id(image_path, input_dir),
        "path": str(image_path),
        "relative_path": str(image_path.relative_to(input_dir)),
        "width": w,
        "height": h,
        "confidence": confidence,
        "color_sensitivity": color_sensitivity,
        "failover_type": failover_type,
        "skipped": not bool(consensus_boxes),
        "bbox_count": len(consensus_boxes),
        "stages": stages,
        "bbox_records": bbox_records,
        "total_area_fraction": total_area_fraction,
        "ambiguous": is_ambiguous(bbox_records, total_area_fraction),
    }


def run_color_enhanced_stage(
    stage_name: str,
    image,
    confidence: float,
    target_hex: str,
    sensitivity: int,
) -> dict:
    enhanced = _apply_color_enhancement(image, target_hex, sensitivity)
    preprocessed = preprocess_image(enhanced)
    if preprocessed is None:
        preprocessed = enhanced
    stage = run_detection_stage(stage_name, preprocessed, image.shape[:2], confidence)
    stage["target_hex"] = target_hex
    stage["color_sensitivity"] = sensitivity
    return stage


def run_detection_stage(stage_name: str, detector_image, original_shape: tuple[int, int], confidence: float) -> dict:
    detections = {
        "east": detect_with_east(detector_image, confidence),
        "yolo11x": detect_with_yolo11x(detector_image, confidence),
        "easyocr": detect_with_easyocr(detector_image, confidence),
    }
    consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
    boxes = []
    for item in consensus:
        padded = pad_consensus_bbox(tuple(item["bbox"]), original_shape)
        mod4 = pad_bbox_to_multiple(padded, multiple=4, image_shape=original_shape)
        boxes.append(
            {
                "consensus_bbox": list(map(int, item["bbox"])),
                "padded_bbox": list(map(int, padded)),
                "mod4_bbox": list(map(int, mod4)),
                "detectors": item["detectors"],
                "detector_count": item["detector_count"],
                "confidence": item["confidence"],
                "original_confidences": [float(value) for value in item["original_confidences"]],
            }
        )
    return {
        "stage": stage_name,
        "detector_image_width": int(detector_image.shape[1]),
        "detector_image_height": int(detector_image.shape[0]),
        "raw_detector_count": sum(len(items) for items in detections.values()),
        "detectors": {
            name: [box_to_record(box) for box in boxes]
            for name, boxes in detections.items()
        },
        "consensus_count": len(consensus),
        "boxes": boxes,
    }


def pad_consensus_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape
    x, y, box_w, box_h = bbox
    pad_w = int(box_w * 0.1)
    pad_h = int(box_h * 0.1)
    padded_x = max(0, x - pad_w)
    padded_y = max(0, y - pad_h)
    padded_w = min(w - padded_x, box_w + 2 * pad_w)
    padded_h = min(h - padded_y, box_h + 2 * pad_h)
    return padded_x, padded_y, padded_w, padded_h


def is_ambiguous(bbox_records: list[dict], total_area_fraction: float) -> bool:
    return (
        len(bbox_records) >= 2
        or total_area_fraction >= 0.06
        or any(
            box["width_fraction"] >= 0.5
            or box["height_fraction"] >= 0.5
            or box["growth_ratio"] >= 4.0
            for box in bbox_records
        )
    )


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    with (out_dir / "pipeline_bboxes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    failovers = Counter(row.get("failover_type", "unknown") for row in rows)
    ambiguous = [row for row in rows if row.get("ambiguous")]
    stats = {
        "image_count": len(rows),
        "bbox_image_count": sum(1 for row in rows if row.get("bbox_count", 0) > 0),
        "skipped_count": sum(1 for row in rows if row.get("skipped")),
        "ambiguous_count": len(ambiguous),
        "failover_counts": dict(sorted(failovers.items())),
        "max_bbox_count": max((row.get("bbox_count", 0) for row in rows), default=0),
        "max_total_area_fraction": max((row.get("total_area_fraction", 0.0) for row in rows), default=0.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    ambiguous_rows = sorted(
        ambiguous,
        key=lambda row: (-row.get("total_area_fraction", 0.0), -row.get("bbox_count", 0), row["relative_path"]),
    )
    with (out_dir / "ambiguous.csv").open("w", encoding="utf-8") as handle:
        handle.write("relative_path,failover_type,bbox_count,total_area_fraction,max_width_fraction,max_growth_ratio\n")
        for row in ambiguous_rows:
            boxes = row.get("bbox_records", [])
            max_width_fraction = max((box["width_fraction"] for box in boxes), default=0.0)
            max_growth_ratio = max((box["growth_ratio"] for box in boxes), default=0.0)
            handle.write(
                f"{csv_escape(row['relative_path'])},{row.get('failover_type', '')},"
                f"{row.get('bbox_count', 0)},{row.get('total_area_fraction', 0.0):.6f},"
                f"{max_width_fraction:.6f},{max_growth_ratio:.3f}\n"
            )

    review_template = [
        {
            "image_id": row.get("image_id"),
            "relative_path": row.get("relative_path"),
            "overlay_relpath": row.get("overlay_relpath"),
            "pipeline_bboxes": [box.get("expanded") for box in row.get("bbox_records", [])],
            "review": empty_review(),
        }
        for row in rows
    ]
    (out_dir / "review_template.json").write_text(
        json.dumps(review_template, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def draw_overlay(image, row: dict):
    overlay = image.copy()
    for index, box in enumerate(row.get("bbox_records", []), start=1):
        raw = tuple(box["raw_or_failover_bbox"])
        exp = tuple(box["expanded"])
        cv2.rectangle(overlay, (raw[0], raw[1]), (raw[0] + raw[2], raw[1] + raw[3]), (255, 0, 0), 2)
        cv2.rectangle(overlay, (exp[0], exp[1]), (exp[0] + exp[2], exp[1] + exp[3]), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{index} {row.get('failover_type', '')}",
            (exp[0] + 4, max(20, exp[1] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    return overlay


def area(box: tuple[int, int, int, int]) -> int:
    return max(int(box[2]), 0) * max(int(box[3]), 0)


def box_to_record(box) -> dict:
    x, y, w, h, confidence = box
    return {
        "bbox": [int(x), int(y), int(w), int(h)],
        "confidence": float(confidence),
        "area": int(w * h),
    }


def empty_review() -> dict:
    return {
        "status": "unreviewed",
        "reviewer": None,
        "reviewed_at": None,
        "actual_watermark_bboxes": [],
        "actual_bbox_format": "xywh_pixels",
        "notes": None,
    }


def safe_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "__".join(rel.with_suffix("").parts).replace(" ", "_")


def csv_escape(value: str) -> str:
    if any(ch in value for ch in ',\"\n'):
        return '"' + value.replace('"', '""')
    return value


if __name__ == "__main__":
    main()
