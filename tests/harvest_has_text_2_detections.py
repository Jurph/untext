from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from untextre.consensus import (
    detect_with_doctr,
    detect_with_east,
    detect_with_easyocr,
    find_consensus_boxes,
)
from untextre.metrics import expand_bbox_along_long_axis
from untextre.preprocessor import preprocess_image
from untextre.utils import CLI_DEFAULT_CONFIDENCE, pad_bbox_to_multiple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest detector-only boxes for has-text-2 images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/images/has_text_2_detector_harvest"),
    )
    parser.add_argument("--confidence", type=float, default=CLI_DEFAULT_CONFIDENCE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    image_paths = [
        path
        for path in sorted(args.input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    records_dir = args.out_dir / "records"
    overlays_dir = args.out_dir / "overlays_ambiguous"
    records_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        display_name = ascii(image_path.name)
        image_id = safe_id(image_path, args.input_dir)
        record_path = records_dir / f"{image_id}.json"
        if args.resume and record_path.exists():
            row = json.loads(record_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] resume {display_name}")
            continue

        start = time.time()
        print(f"[{index}/{len(image_paths)}] detect {display_name}", flush=True)
        row = run_one(image_path, args.input_dir, args.confidence)
        row["elapsed_sec"] = time.time() - start
        record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)

        if row["ambiguous"]:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                overlay = draw_overlay(image, row)
                cv2.imwrite(str(overlays_dir / f"{image_id}.jpg"), overlay)

    write_summary(args.out_dir, rows)


def run_one(image_path: Path, input_dir: Path, confidence: float) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "path": str(image_path),
            "relative_path": str(image_path.relative_to(input_dir)),
            "error": "cv2.imread returned None",
            "detectors": {},
            "consensus": [],
            "final_boxes": [],
            "ambiguous": False,
        }

    preprocessed = preprocess_image(image)
    if preprocessed is None:
        preprocessed = image

    detections = {
        "east": detect_with_east(preprocessed, confidence),
        "doctr": detect_with_doctr(preprocessed, confidence),
        "easyocr": detect_with_easyocr(preprocessed, confidence),
    }
    consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
    h, w = image.shape[:2]
    final_boxes = []
    for item in consensus:
        x, y, box_w, box_h = item["bbox"]
        pad_w = int(box_w * 0.1)
        pad_h = int(box_h * 0.1)
        padded_x = max(0, x - pad_w)
        padded_y = max(0, y - pad_h)
        padded_w = min(w - padded_x, box_w + 2 * pad_w)
        padded_h = min(h - padded_y, box_h + 2 * pad_h)
        padded = (padded_x, padded_y, padded_w, padded_h)
        mod4 = pad_bbox_to_multiple(padded, multiple=4, image_shape=(h, w))
        expanded = expand_bbox_along_long_axis(image, mod4)
        final_boxes.append(
            {
                "detectors": item["detectors"],
                "confidence": item["confidence"],
                "raw_union": list(item["bbox"]),
                "padded": list(padded),
                "mod4": list(mod4),
                "expanded": list(expanded),
                "area_fraction": area_fraction(expanded, w, h),
                "width_fraction": expanded[2] / w if w else 0.0,
                "height_fraction": expanded[3] / h if h else 0.0,
                "growth_ratio": area(expanded) / max(area(item["bbox"]), 1),
            }
        )

    total_area_fraction = sum(area(tuple(box["expanded"])) for box in final_boxes) / max(w * h, 1)
    ambiguous = (
        len(final_boxes) >= 2
        or total_area_fraction >= 0.06
        or any(box["width_fraction"] >= 0.5 or box["height_fraction"] >= 0.5 for box in final_boxes)
    )
    return {
        "path": str(image_path),
        "relative_path": str(image_path.relative_to(input_dir)),
        "width": w,
        "height": h,
        "confidence": confidence,
        "detectors": {
            name: [box_to_record(box) for box in boxes]
            for name, boxes in detections.items()
        },
        "raw_detector_count": sum(len(boxes) for boxes in detections.values()),
        "consensus_count": len(consensus),
        "consensus": [
            {
                "bbox": list(item["bbox"]),
                "detectors": item["detectors"],
                "confidence": item["confidence"],
                "detector_count": item["detector_count"],
            }
            for item in consensus
        ],
        "final_boxes": final_boxes,
        "total_area_fraction": total_area_fraction,
        "ambiguous": ambiguous,
    }


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    summary_jsonl = out_dir / "detections.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    ambiguous = [row for row in rows if row.get("ambiguous")]
    stats = {
        "image_count": len(rows),
        "ambiguous_count": len(ambiguous),
        "no_consensus_count": sum(1 for row in rows if row.get("consensus_count", 0) == 0),
        "max_total_area_fraction": max((row.get("total_area_fraction", 0.0) for row in rows), default=0.0),
        "max_consensus_count": max((row.get("consensus_count", 0) for row in rows), default=0),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    ambiguous_rows = sorted(
        ambiguous,
        key=lambda row: (-row.get("total_area_fraction", 0.0), -row.get("consensus_count", 0), row["relative_path"]),
    )
    with (out_dir / "ambiguous.csv").open("w", encoding="utf-8") as handle:
        handle.write("relative_path,consensus_count,raw_detector_count,total_area_fraction,max_width_fraction,max_growth_ratio\n")
        for row in ambiguous_rows:
            final_boxes = row.get("final_boxes", [])
            max_width_fraction = max((box["width_fraction"] for box in final_boxes), default=0.0)
            max_growth_ratio = max((box["growth_ratio"] for box in final_boxes), default=0.0)
            handle.write(
                f"{csv_escape(row['relative_path'])},{row.get('consensus_count', 0)},"
                f"{row.get('raw_detector_count', 0)},{row.get('total_area_fraction', 0.0):.6f},"
                f"{max_width_fraction:.6f},{max_growth_ratio:.3f}\n"
            )


def draw_overlay(image, row: dict):
    overlay = image.copy()
    colors = {
        "raw_union": (255, 0, 0),
        "expanded": (0, 0, 255),
    }
    for index, box in enumerate(row.get("final_boxes", []), start=1):
        raw = tuple(box["raw_union"])
        exp = tuple(box["expanded"])
        cv2.rectangle(overlay, (raw[0], raw[1]), (raw[0] + raw[2], raw[1] + raw[3]), colors["raw_union"], 2)
        cv2.rectangle(overlay, (exp[0], exp[1]), (exp[0] + exp[2], exp[1] + exp[3]), colors["expanded"], 2)
        cv2.putText(
            overlay,
            f"{index} {','.join(box['detectors'])}",
            (exp[0] + 4, max(20, exp[1] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colors["expanded"],
            2,
        )
    return overlay


def box_to_record(box) -> dict:
    x, y, w, h, confidence = box
    return {
        "bbox": [int(x), int(y), int(w), int(h)],
        "confidence": float(confidence),
        "area": int(w * h),
    }


def area(box: tuple[int, int, int, int]) -> int:
    return max(int(box[2]), 0) * max(int(box[3]), 0)


def area_fraction(box: tuple[int, int, int, int], width: int, height: int) -> float:
    return area(box) / max(width * height, 1)


def safe_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "__".join(rel.with_suffix("").parts).replace(" ", "_")


def csv_escape(value: str) -> str:
    if any(ch in value for ch in ',\"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


if __name__ == "__main__":
    main()
