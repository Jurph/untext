import numpy as np
import logging

from untextre.detector import TextDetector


def _detector_without_model(confidence_threshold=0.3, min_text_size=3):
    detector = object.__new__(TextDetector)
    detector.confidence_threshold = confidence_threshold
    detector.min_text_size = min_text_size
    return detector


def test_parse_doctr_output_accepts_dict_words_format():
    """DocTR dict-of-words output should not silently parse as zero detections."""
    detector = _detector_without_model()
    page = {
        "words": {
            "word-1": {
                "geometry": ((0.10, 0.20), (0.40, 0.50)),
                "confidence": 0.91,
            },
            "word-2": {
                "geometry": ((0.60, 0.70), (0.65, 0.75)),
                "confidence": 0.10,
            },
        }
    }

    detections = detector._parse_doctr_output(page, (100, 200))

    assert len(detections) == 1
    np.testing.assert_allclose(
        detections[0]["geometry"],
        np.array(
            [
                [20.0, 20.0],
                [80.0, 20.0],
                [80.0, 50.0],
                [20.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )
    assert detections[0]["confidence"] == 0.91


def test_detect_logs_detection_count_at_info(caplog):
    detector = _detector_without_model()
    detector.predictor = lambda _images: [{
        "words": np.array([[0.10, 0.20, 0.40, 0.50, 0.91]], dtype=np.float32)
    }]
    detector._filter_detections = lambda detections, _shape: detections

    image = np.zeros((100, 200, 3), dtype=np.uint8)

    with caplog.at_level(logging.INFO):
        detections = detector.detect(image)

    assert len(detections) == 1
    assert "DocTR detected 1 text region(s)" in caplog.text
