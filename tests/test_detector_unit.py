"""Unit tests for untextre.detector public seams.

These tests keep detector.py free of optional heavyweight imports at module
import time and pin the supported single-detector adapters after DocTR removal.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from untextre import detector as detector_mod
from untextre.detector import _decode_east_predictions, cleanup_vram, detect_text_regions


def test_doctr_detector_is_not_public_api():
    assert not hasattr(detector_mod, "TextDetector")
    assert not hasattr(detector_mod, "get_doctr_detector")


def test_detect_text_regions_rejects_removed_doctr_method():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported detection method: doctr"):
        detect_text_regions(image, method="doctr")


def test_detect_text_regions_uses_east_adapter(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    net = object()
    monkeypatch.setattr(detector_mod, "get_east_net", Mock(return_value=net))
    monkeypatch.setattr(
        detector_mod,
        "_detect_with_east",
        Mock(
            return_value=[
                {
                    "geometry": np.array(
                        [[1, 2], [5, 2], [5, 8], [1, 8]],
                        dtype=np.float32,
                    ),
                    "confidence": 0.9,
                }
            ]
        ),
    )

    assert detect_text_regions(image, method="east", confidence_threshold=0.4) == [(1, 2, 4, 6)]
    detector_mod._detect_with_east.assert_called_once_with(image, net, min_confidence=0.4)


def test_detect_text_regions_uses_easyocr_adapter(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    reader = object()
    monkeypatch.setattr(detector_mod, "get_easyocr_reader", Mock(return_value=reader))
    monkeypatch.setattr(
        detector_mod,
        "_detect_with_easyocr",
        Mock(
            return_value=[
                {
                    "geometry": np.array(
                        [[3, 4], [13, 4], [13, 9], [3, 9]],
                        dtype=np.float32,
                    ),
                    "confidence": 0.8,
                }
            ]
        ),
    )

    assert detect_text_regions(image, method="easyocr", confidence_threshold=0.25) == [(3, 4, 10, 5)]
    detector_mod._detect_with_easyocr.assert_called_once_with(
        image,
        reader,
        confidence_threshold=0.25,
    )


def test_detect_text_regions_uses_yolo11x_adapter(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    model = object()
    monkeypatch.setattr(detector_mod, "get_yolo11x_model", Mock(return_value=model))
    monkeypatch.setattr(
        detector_mod,
        "_detect_with_yolo11x",
        Mock(
            return_value=[
                {
                    "geometry": np.array(
                        [[2, 3], [12, 3], [12, 11], [2, 11]],
                        dtype=np.float32,
                    ),
                    "confidence": 0.7,
                }
            ]
        ),
    )

    assert detect_text_regions(image, method="yolo11x", confidence_threshold=0.5) == [(2, 3, 10, 8)]
    detector_mod._detect_with_yolo11x.assert_called_once_with(
        image,
        model,
        confidence_threshold=0.5,
    )


def test_cleanup_vram_does_not_crash_on_cpu_only(monkeypatch):
    monkeypatch.setattr(detector_mod.torch.cuda, "is_available", Mock(return_value=False))

    cleanup_vram()


def test_decode_east_predictions_computes_rotation_aware_bbox():
    """Single passing cell, zero rotation: exact geometry math, no trig surprises."""
    scores = np.full((1, 1, 2, 3), 0.1, dtype=np.float32)
    scores[0, 0, 1, 2] = 0.8
    geometry = np.zeros((1, 5, 2, 3), dtype=np.float32)
    # top, right, bottom, left distances + angle=0 at the passing cell (y=1, x=2)
    geometry[0, 0, 1, 2] = 5.0
    geometry[0, 1, 1, 2] = 7.0
    geometry[0, 2, 1, 2] = 6.0
    geometry[0, 3, 1, 2] = 4.0
    geometry[0, 4, 1, 2] = 0.0

    rectangles, confidences = _decode_east_predictions(scores, geometry, min_confidence=0.5)

    assert rectangles == [(4, -1, 11, 11)]
    assert confidences == pytest.approx([0.8])


def test_decode_east_predictions_filters_below_threshold():
    scores = np.full((1, 1, 3, 3), 0.2, dtype=np.float32)
    geometry = np.zeros((1, 5, 3, 3), dtype=np.float32)

    rectangles, confidences = _decode_east_predictions(scores, geometry, min_confidence=0.5)

    assert rectangles == []
    assert confidences == []


def test_decode_east_predictions_preserves_row_major_order():
    """Multiple passing cells come back in (y, then x) order, matching the
    row-outer/col-inner scan this function replaced."""
    scores = np.zeros((1, 1, 2, 2), dtype=np.float32)
    scores[0, 0, 0, 1] = 0.9  # y=0, x=1
    scores[0, 0, 1, 0] = 0.7  # y=1, x=0
    scores[0, 0, 1, 1] = 0.6  # y=1, x=1
    geometry = np.zeros((1, 5, 2, 2), dtype=np.float32)

    _, confidences = _decode_east_predictions(scores, geometry, min_confidence=0.5)

    assert confidences == pytest.approx([0.9, 0.7, 0.6])
