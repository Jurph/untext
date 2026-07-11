from __future__ import annotations

import cv2
import numpy as np

from scripts.has_text_2.build_replay_matrix import first_stage, threshold_id
from scripts.has_text_2.evaluate_replay_policy import build_image_cache, find_combo, load_exact_image


def test_threshold_id_is_stable() -> None:
    assert threshold_id(0.30) == "threshold_300"
    assert threshold_id(0.05) == "threshold_050"
    assert threshold_id(0.025) == "threshold_025"


def test_first_stage_extracts_detector_lists() -> None:
    row = {
        "stages": [
            {
                "detectors": {
                    "east": [{"bbox": [1, 2, 3, 4], "confidence": 91.0}],
                    "yolo11x": [{"bbox": [5, 6, 7, 8], "confidence": 88.0}],
                    "easyocr": [],
                }
            }
        ]
    }
    detectors = first_stage(row)
    assert detectors["east"][0]["bbox"] == [1, 2, 3, 4]
    assert detectors["yolo11x"][0]["bbox"] == [5, 6, 7, 8]
    assert detectors["easyocr"] == []


def test_find_combo_matches_threshold_triplet() -> None:
    rows = [
        {"east_threshold": 0.3, "yolo11x_threshold": 0.1, "easyocr_threshold": 0.05, "recall": 0.5},
        {"east_threshold": 0.025, "yolo11x_threshold": 0.025, "easyocr_threshold": 0.025, "recall": 0.6},
    ]
    assert find_combo(rows, 0.025, 0.025, 0.025)["recall"] == 0.6
    assert find_combo(rows, 0.3, 0.1, 0.05)["recall"] == 0.5


def test_load_exact_image_handles_unicode_path_and_missing_file(tmp_path) -> None:
    image_path = tmp_path / "тестовый_снимок.png"
    expected = np.zeros((6, 5, 3), dtype=np.uint8)
    expected[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".png", expected)
    assert ok
    encoded.tofile(image_path)

    loaded, reason = load_exact_image(
        {
            "path": str(image_path),
            "height": 6,
            "width": 5,
        }
    )
    assert reason is None
    np.testing.assert_array_equal(loaded, expected)

    missing, reason = load_exact_image(
        {
            "path": str(tmp_path / "missing.png"),
            "height": 6,
            "width": 5,
        }
    )
    assert missing is None
    assert reason == "missing_file"


def test_build_image_cache_skips_shape_mismatch(tmp_path) -> None:
    image_path = tmp_path / "mismatch.png"
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", expected)
    assert ok
    encoded.tofile(image_path)

    cache, skips = build_image_cache(
        [
            {
                "image_id": "ok",
                "path": str(image_path),
                "height": 4,
                "width": 4,
            },
            {
                "image_id": "bad",
                "path": str(image_path),
                "height": 5,
                "width": 4,
            },
        ]
    )
    assert "ok" in cache
    assert "bad" not in cache
    assert any(item["reason"].startswith("shape_mismatch") for item in skips)
