from pathlib import Path

import pytest

from untextre.detector_pair_harvest import (
    append_jsonl,
    bbox_area,
    bbox_center,
    bbox_iou,
    box_metrics_against_truth,
    center_distance,
    contains_point,
    image_key,
    load_jsonl,
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
