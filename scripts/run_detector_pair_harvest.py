from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from untextre.detector_pair_harvest import append_jsonl, load_jsonl, normalize_detection_box
from untextre.utils import load_image

DEFAULT_DETECTORS = ["east", "doctr", "easyocr", "yolo11x"]
SUPPORTED_DETECTORS = [*DEFAULT_DETECTORS, "fake"]


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def geometry_to_xywh(points: Any) -> list[float]:
    import numpy as np

    pts = np.asarray(points, dtype="float32").reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts.astype("int32"))
    return [x, y, w, h]


def serializable_geometry(geometry: Any) -> Any:
    return geometry.tolist() if hasattr(geometry, "tolist") else geometry


def run_doctr(image_bgr, floor: float) -> list[dict]:
    from untextre.detector import get_doctr_detector

    detector = get_doctr_detector(confidence_threshold=floor)
    rows = []
    for det in detector.detect(image_bgr):
        rows.append(
            normalize_detection_box(
                geometry_to_xywh(det["geometry"]),
                det.get("confidence", 0.0),
                "text",
                {"geometry": serializable_geometry(det["geometry"])},
            )
        )
    return rows


def run_easyocr(image_bgr, floor: float) -> list[dict]:
    from untextre.detector import _detect_with_easyocr, get_easyocr_reader

    reader = get_easyocr_reader()
    rows = []
    for det in _detect_with_easyocr(image_bgr, reader, confidence_threshold=floor):
        rows.append(
            normalize_detection_box(
                geometry_to_xywh(det["geometry"]),
                det.get("confidence", 0.0),
                "text",
                {"geometry": serializable_geometry(det["geometry"])},
            )
        )
    return rows


def run_east(image_bgr, floor: float) -> list[dict]:
    from untextre.detector import _detect_with_east, get_east_net

    net = get_east_net()
    rows = []
    for det in _detect_with_east(image_bgr, net, min_confidence=floor):
        rows.append(
            normalize_detection_box(
                geometry_to_xywh(det["geometry"]),
                det.get("confidence", 0.0),
                "text",
                {"geometry": serializable_geometry(det["geometry"])},
            )
        )
    return rows


def run_fake(image_bgr, floor: float) -> list[dict]:
    height, width = image_bgr.shape[:2]
    return [
        normalize_detection_box(
            [0, 0, width, height],
            max(floor, 1.0),
            "fake",
            {"adapter": "fake"},
        )
    ]


def run_yolo11x(image_path: Path, model, floor: float) -> list[dict]:
    results = model.predict(str(image_path), conf=floor, verbose=False, device=0)
    result = results[0]
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        cls_index = int(box.cls[0])
        rows.append(
            normalize_detection_box(
                xyxy_to_xywh(x1, y1, x2, y2),
                float(box.conf[0]),
                result.names.get(cls_index, str(cls_index)),
                {"class_index": cls_index},
            )
        )
    return rows


def load_yolo_model(weights_path: Path):
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest raw detector evidence for paired corpus.")
    parser.add_argument("harvest_root", type=Path)
    parser.add_argument("--clean-dir", type=Path, default=Path("tests/images/zero"))
    parser.add_argument("--detectors", nargs="+", choices=SUPPORTED_DETECTORS, default=DEFAULT_DETECTORS)
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path(".codex-tmp/yolo_eval/weights/yolo11x-train28-best.pt"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def detector_boxes(detector: str, image_path: Path, image_bgr, yolo_model, floor: float) -> list[dict]:
    if detector == "yolo11x":
        return run_yolo11x(image_path, yolo_model, floor)
    if detector == "east":
        return run_east(image_bgr, floor)
    if detector == "doctr":
        return run_doctr(image_bgr, floor)
    if detector == "easyocr":
        return run_easyocr(image_bgr, floor)
    if detector == "fake":
        return run_fake(image_bgr, floor)
    raise ValueError(f"unknown detector {detector}")


def image_path_for_state(harvest_root: Path, clean_dir: Path, pair: dict, state: str) -> tuple[Path, str]:
    if state == "clean":
        rel = pair["clean_relative_path"]
        return clean_dir / rel, rel
    rel = pair["twin_relative_path"]
    return harvest_root / rel, rel


def evidence_row(
    pair: dict,
    state: str,
    detector: str,
    rel: str,
    floor: float,
    elapsed_ms: float,
    boxes: list[dict],
    image_bgr,
) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "state": state,
        "detector": detector,
        "image_relative_path": rel,
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "harvest_floor": floor,
        "elapsed_ms": elapsed_ms,
        "boxes": boxes,
    }


def error_row(
    pair: dict,
    state: str,
    detector: str,
    rel: str,
    floor: float,
    elapsed_ms: float,
    exc: Exception,
) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "state": state,
        "detector": detector,
        "image_relative_path": rel,
        "harvest_floor": floor,
        "elapsed_ms": elapsed_ms,
        "boxes": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    args = parse_args()

    pair_rows = load_jsonl(args.harvest_root / "pairs" / "pair_manifest.jsonl")
    if args.limit is not None:
        pair_rows = pair_rows[: args.limit]

    yolo_model = None
    if "yolo11x" in args.detectors:
        yolo_model = load_yolo_model(args.yolo_weights)

    for detector in args.detectors:
        out_path = args.harvest_root / "evidence" / f"{detector}.jsonl"
        if not args.resume and out_path.exists():
            out_path.unlink()

        done = set()
        if args.resume and out_path.exists():
            done = {(row["pair_id"], row["state"]) for row in load_jsonl(out_path)}

        for index, pair in enumerate(pair_rows, start=1):
            for state in ("clean", "twin"):
                if (pair["pair_id"], state) in done:
                    continue

                image_path, rel = image_path_for_state(args.harvest_root, args.clean_dir, pair, state)
                started = time.time()
                try:
                    image = load_image(image_path)
                    boxes = detector_boxes(detector, image_path, image, yolo_model, args.floor)
                    row = evidence_row(
                        pair,
                        state,
                        detector,
                        rel,
                        args.floor,
                        round(1000 * (time.time() - started), 1),
                        boxes,
                        image,
                    )
                except Exception as exc:
                    row = error_row(
                        pair,
                        state,
                        detector,
                        rel,
                        args.floor,
                        round(1000 * (time.time() - started), 1),
                        exc,
                    )
                append_jsonl(out_path, row)
            print(f"{detector} [{index}/{len(pair_rows)}] {pair['pair_id']}", flush=True)


if __name__ == "__main__":
    main()
