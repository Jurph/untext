"""Simulate the ACTUAL production consensus/masking policy (untextre.consensus.
find_consensus_boxes, overlap_threshold=0.1, CLI_DEFAULT_CONFIDENCE=0.3, plus the
10%-per-side padding + mod-4 alignment run_consensus_detection applies) against
the known-truth detector-pair-harvest corpus, for two detector sets:

  CURRENT:  east + doctr + easyocr   (production today)
  SWAPPED:  east + easyocr + yolo11x (doctr relegated, yolo11x promoted)

Reports, per set:
  - recall: fraction of twins where a final (padded) consensus box lands on
    the true watermark region (IoU>=0.3)
  - image-level FP rate: fraction of clean images that get ANY consensus box
  - area-level FP rate: fraction of clean-image pixel area that ends up inside
    a consensus box (the actual "unwatermarked area masked-and-painted" question)

Prerequisite: a detector-pair-harvest corpus must exist with raw evidence for
all four detectors (east, doctr, easyocr, yolo11x). See
docs/superpowers/plans/2026-07-06-detector-pair-harvest-runbook.md to build one
against tests/images/zero (415 images as of 2026-07-06) or any other known-clean
corpus. Point --harvest-root at it; defaults to the 2026-07-06 zero-corpus run.

Usage:
    "C:/Users/Jurph/Documents/Python Scripts/untext/.venv/Scripts/python.exe" \
        scripts/simulate_production_consensus_swap.py

Reference run (2026-07-06, tests/images/detector_pair_harvest, N=415):
    CURRENT (east+doctr+easyocr):  recall=92.3% (383/415)  image-FP=3.6% (15/415)  area-FP=0.018%
    SWAPPED (east+easyocr+yolo11x): recall=96.1% (399/415)  image-FP=2.7% (11/415)  area-FP=0.015%
"""
from __future__ import annotations

import argparse
from pathlib import Path

from untextre.consensus import find_consensus_boxes
from untextre.utils import pad_bbox_to_multiple, CLI_DEFAULT_CONFIDENCE
from untextre.detector_pair_harvest import load_jsonl, bbox_iou, bbox_area

DETECTORS_AVAILABLE = ["east", "doctr", "easyocr", "yolo11x"]


def load_evidence(root: Path) -> dict:
    """evidence[det][pair_id][state] = list of raw boxes (unfiltered detector floor)."""
    evidence: dict = {det: {} for det in DETECTORS_AVAILABLE}
    for det in DETECTORS_AVAILABLE:
        rows = load_jsonl(root / "evidence" / f"{det}.jsonl")
        for row in rows:
            evidence[det].setdefault(row["pair_id"], {})[row["state"]] = row.get("boxes", [])
    return evidence


def production_consensus_boxes(
    evidence: dict, pair_id: str, state: str, detector_set: list[str], image_w: int, image_h: int
):
    """Reproduce run_consensus_detection's box-finding + padding, for a chosen detector set."""
    detections = {}
    for det in detector_set:
        raw = evidence[det].get(pair_id, {}).get(state, [])
        # apply CLI_DEFAULT_CONFIDENCE gate, same as production
        tuples = [
            (b["xywh"][0], b["xywh"][1], b["xywh"][2], b["xywh"][3], b["confidence"])
            for b in raw
            if b["confidence"] >= CLI_DEFAULT_CONFIDENCE
        ]
        detections[det] = tuples

    consensus_boxes = find_consensus_boxes(detections, overlap_threshold=0.1)
    padded = []
    for c in consensus_boxes:
        x, y, box_w, box_h = c["bbox"]
        pad_w = int(box_w * 0.1)
        pad_h = int(box_h * 0.1)
        px, py = max(0, x - pad_w), max(0, y - pad_h)
        pw = min(image_w - px, box_w + 2 * pad_w)
        ph = min(image_h - py, box_h + 2 * pad_h)
        mod4 = pad_bbox_to_multiple((px, py, pw, ph), multiple=4, image_shape=(image_h, image_w))
        padded.append(mod4)
    return padded


def evaluate(evidence: dict, pairs: dict, pair_ids: list[str], detector_set: list[str], label: str):
    n = len(pair_ids)
    twin_hits = 0
    clean_fp_images = 0
    clean_fp_area_fracs = []

    for pid in pair_ids:
        pair = pairs[pid]
        w, h = pair["base_width"], pair["base_height"]
        truth_bbox = pair["truth_bbox"]

        twin_boxes = production_consensus_boxes(evidence, pid, "twin", detector_set, w, h)
        if any(bbox_iou(list(b), truth_bbox) >= 0.3 for b in twin_boxes):
            twin_hits += 1

        clean_boxes = production_consensus_boxes(evidence, pid, "clean", detector_set, w, h)
        if clean_boxes:
            clean_fp_images += 1
        total_area = w * h
        fp_area = sum(bbox_area(list(b)) for b in clean_boxes)
        clean_fp_area_fracs.append(min(1.0, fp_area / total_area) if total_area else 0.0)

    recall = twin_hits / n
    fp_image_rate = clean_fp_images / n
    mean_fp_area = sum(clean_fp_area_fracs) / n
    nonzero = [a for a in clean_fp_area_fracs if a > 0]
    mean_fp_area_when_present = sum(nonzero) / len(nonzero) if nonzero else 0.0

    print(f"=== {label}: detectors={detector_set} ===")
    print(f"  recall (twin, IoU>=0.3 vs truth): {twin_hits}/{n} = {recall:.4f}")
    print(f"  clean image-level FP rate:        {clean_fp_images}/{n} = {fp_image_rate:.4f}")
    print(f"  clean mean masked-area fraction (all images):        {mean_fp_area:.5f} ({mean_fp_area*100:.3f}%)")
    print(f"  clean mean masked-area fraction (only flagged imgs): {mean_fp_area_when_present:.5f} ({mean_fp_area_when_present*100:.3f}%)")
    print()
    return recall, fp_image_rate, mean_fp_area


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harvest-root",
        type=Path,
        default=Path("tests/images/detector_pair_harvest"),
        help="Root of a detector-pair-harvest run (must contain pairs/pair_manifest.jsonl and evidence/*.jsonl)",
    )
    args = parser.parse_args()

    pairs_rows = load_jsonl(args.harvest_root / "pairs" / "pair_manifest.jsonl")
    pairs = {r["pair_id"]: r for r in pairs_rows}
    pair_ids = sorted(pairs.keys())
    evidence = load_evidence(args.harvest_root)

    evaluate(evidence, pairs, pair_ids, ["east", "doctr", "easyocr"], "CURRENT production (east+doctr+easyocr)")
    evaluate(evidence, pairs, pair_ids, ["east", "easyocr", "yolo11x"], "PROPOSED swap (east+easyocr+yolo11x)")


if __name__ == "__main__":
    main()
