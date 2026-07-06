from argparse import Namespace
from pathlib import Path

import pytest

from untextre.detector_pair_harvest import (
    append_jsonl,
    bbox_area,
    bbox_center,
    bbox_iou,
    box_metrics_against_truth,
    build_pair_row,
    center_distance,
    contains_point,
    image_key,
    load_jsonl,
    normalize_detection_box,
    pair_id_for_path,
    pairwise_fire_overlap,
    summarize_detector_rows,
)

from scripts import run_detector_pair_harvest as harvest_script


class _StubImage:
    shape = (24, 36, 3)


def _write_single_pair_manifest(harvest_root: Path) -> None:
    append_jsonl(
        harvest_root / "pairs" / "pair_manifest.jsonl",
        {
            "pair_id": "sample__jpg",
            "clean_relative_path": "sample.jpg",
            "twin_relative_path": "pairs/sample__jpg.jpg",
        },
    )


def _stub_harvest_args(monkeypatch, tmp_path: Path, harvest_root: Path, detectors: list[str]) -> None:
    monkeypatch.setattr(
        harvest_script,
        "parse_args",
        lambda: Namespace(
            harvest_root=harvest_root,
            clean_dir=tmp_path / "clean",
            detectors=detectors,
            floor=0.5,
            yolo_weights=tmp_path / "missing-yolo.pt",
            limit=None,
            resume=False,
        ),
    )


def test_image_key_preserves_extension_to_avoid_stem_collision():
    assert image_key(Path("EzlMcCy.jpeg")) == "EzlMcCy__jpeg"
    assert image_key(Path("EzlMcCy.jpg")) == "EzlMcCy__jpg"


def test_bbox_iou_reports_continuous_overlap_without_decision_threshold():
    assert bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)
    assert bbox_iou([0, 0, 10, 10], [20, 20, 3, 3]) == 0.0


def test_bbox_center_and_distance_are_float_metrics():
    assert bbox_center([10, 20, 30, 40]) == (25.0, 40.0)
    assert center_distance([0, 0, 10, 10], [3, 4, 10, 10]) == pytest.approx(5.0)


def test_contains_point_is_geometry_only_not_a_hit_rule():
    assert contains_point([10, 10, 5, 5], (12, 14)) is True
    assert contains_point([10, 10, 5, 5], (15.1, 14)) is False


def test_box_metrics_against_truth_preserves_continuous_values():
    boxes = [
        {"xywh": [40, 40, 10, 10], "confidence": 0.9},
        {"xywh": [0, 0, 10, 10], "confidence": 0.2},
    ]
    metrics = box_metrics_against_truth(boxes, [0, 0, 10, 10])

    assert metrics["box_count"] == 2
    assert metrics["max_confidence"] == 0.9
    assert metrics["best_iou"] == 1.0
    assert metrics["best_truth_center_contained"] is True
    assert metrics["top_boxes"][0]["confidence"] == 0.2


def test_jsonl_helpers_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"b": 2, "a": 1})
    append_jsonl(path, {"c": [3]})
    assert load_jsonl(path) == [{"a": 1, "b": 2}, {"c": [3]}]


def test_pair_id_for_path_preserves_extension():
    assert pair_id_for_path(Path("foo bar.jpeg")) == "foo_bar__jpeg"


def test_build_pair_row_carries_visibility_and_truth_metadata():
    row = build_pair_row(
        "img__jpg",
        "img.jpg",
        "pairs/synthetic_twins/img__jpg.jpg",
        {
            "measured_visibility_delta_e": 12.5,
            "visibility_attempts": 1,
            "visibility_fallback": False,
            "color_class": "white",
        },
        [1, 2, 30, 4],
        640,
        480,
    )
    assert row["pair_id"] == "img__jpg"
    assert row["truth_bbox"] == [1, 2, 30, 4]
    assert row["measured_visibility_delta_e"] == 12.5
    assert row["synthetic_metadata"]["color_class"] == "white"



def test_normalize_detection_box_rounds_geometry_but_keeps_raw_payload():
    box = normalize_detection_box([1.234, 2.5, 30.0, 4.0], 0.98765, "watermark", {"source": "unit"})
    assert box["xywh"] == [1.2, 2.5, 30.0, 4.0]
    assert box["confidence"] == 0.9877
    assert box["label"] == "watermark"
    assert box["raw_payload"] == {"source": "unit"}


def test_yolo_load_failure_is_isolated_to_yolo_evidence_rows(tmp_path, monkeypatch):
    harvest_root = tmp_path / "harvest"
    _write_single_pair_manifest(harvest_root)
    _stub_harvest_args(monkeypatch, tmp_path, harvest_root, ["yolo11x", "fake"])
    monkeypatch.setattr(harvest_script, "load_image", lambda _path: _StubImage())

    def fail_load_yolo_model(_weights_path: Path):
        raise RuntimeError("missing yolo")

    monkeypatch.setattr(harvest_script, "load_yolo_model", fail_load_yolo_model)

    harvest_script.main()

    fake_rows = load_jsonl(harvest_root / "evidence" / "fake.jsonl")
    assert [row["state"] for row in fake_rows] == ["clean", "twin"]
    assert all(row["detector"] == "fake" for row in fake_rows)
    assert all("error" not in row for row in fake_rows)

    yolo_rows = load_jsonl(harvest_root / "evidence" / "yolo11x.jsonl")
    assert [row["state"] for row in yolo_rows] == ["clean", "twin"]
    assert all(row["boxes"] == [] for row in yolo_rows)
    assert all(row["error"] == "RuntimeError: missing yolo" for row in yolo_rows)
    assert all(row["width"] == 36 for row in yolo_rows)
    assert all(row["height"] == 24 for row in yolo_rows)


def test_detector_error_rows_include_dimensions_after_image_load(tmp_path, monkeypatch):
    harvest_root = tmp_path / "harvest"
    _write_single_pair_manifest(harvest_root)
    _stub_harvest_args(monkeypatch, tmp_path, harvest_root, ["fake"])
    monkeypatch.setattr(harvest_script, "load_image", lambda _path: _StubImage())

    def fail_detector_boxes(*_args, **_kwargs):
        raise ValueError("detector down")

    monkeypatch.setattr(harvest_script, "detector_boxes", fail_detector_boxes)

    harvest_script.main()

    rows = load_jsonl(harvest_root / "evidence" / "fake.jsonl")
    assert [row["state"] for row in rows] == ["clean", "twin"]
    assert all(row["error"] == "ValueError: detector down" for row in rows)
    assert all(row["width"] == 36 for row in rows)
    assert all(row["height"] == 24 for row in rows)


def test_summarize_detector_rows_counts_clean_fires_and_twin_geometry():
    pairs = {
        "a": {"truth_bbox": [0, 0, 10, 10]},
        "b": {"truth_bbox": [50, 50, 10, 10]},
    }
    rows = [
        {"pair_id": "a", "state": "clean", "boxes": []},
        {"pair_id": "a", "state": "twin", "boxes": [{"xywh": [0, 0, 10, 10], "confidence": 0.9}]},
        {"pair_id": "b", "state": "clean", "boxes": [{"xywh": [1, 1, 2, 2], "confidence": 0.5}]},
        {"pair_id": "b", "state": "twin", "boxes": [{"xywh": [0, 0, 10, 10], "confidence": 0.4}]},
    ]
    summary = summarize_detector_rows("fake", pairs, rows)
    assert summary["detector"] == "fake"
    assert summary["pair_count"] == 2
    assert summary["clean_fired_count"] == 1
    assert summary["twin_fired_count"] == 2
    assert summary["clean_row_count"] == 2
    assert summary["twin_row_count"] == 2
    assert summary["clean_mean_boxes"] == 0.5
    assert summary["twin_mean_boxes"] == 1.0
    assert summary["max_best_iou"] == 1.0
    assert summary["median_best_iou"] == 0.5


def test_pairwise_fire_overlap_reports_sets_without_decision_claims():
    a = {"img1", "img2"}
    b = {"img2", "img3"}
    result = pairwise_fire_overlap("a", a, "b", b, universe={"img1", "img2", "img3", "img4"})
    assert result == {
        "left": "a",
        "right": "b",
        "both": 1,
        "left_only": 1,
        "right_only": 1,
        "neither": 1,
        "jaccard": 1 / 3,
    }