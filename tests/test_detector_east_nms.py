import numpy as np

from untextre import detector


class FakeEastNet:
    def setInput(self, blob):
        self.blob = blob

    def forward(self, layer_names):
        scores = np.zeros((1, 1, 1, 1), dtype=np.float32)
        geometry = np.zeros((1, 5, 1, 1), dtype=np.float32)
        return scores, geometry


def test_east_nms_receives_xywh_boxes(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        detector,
        "_decode_east_predictions",
        lambda scores, geometry, min_confidence: (
            [(10, 20, 30, 40), (12, 22, 30, 40)],
            [0.9, 0.8],
        ),
    )

    def fake_nms_boxes(boxes, confidences, min_confidence, nms_threshold):
        captured["boxes"] = boxes
        return np.array([0])

    monkeypatch.setattr(detector.cv2.dnn, "NMSBoxes", fake_nms_boxes)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detector._detect_with_east(image, FakeEastNet(), width=64, height=64)

    assert captured["boxes"] == [[10, 20, 30, 40], [12, 22, 30, 40]]


def test_east_nms_fallback_suppresses_overlapping_boxes(monkeypatch):
    monkeypatch.setattr(
        detector,
        "_decode_east_predictions",
        lambda scores, geometry, min_confidence: (
            [(10, 10, 20, 20), (12, 12, 20, 20), (70, 70, 10, 10)],
            [0.9, 0.8, 0.7],
        ),
    )

    def failing_nms_boxes(boxes, confidences, min_confidence, nms_threshold):
        raise RuntimeError("simulated OpenCV NMS binding failure")

    monkeypatch.setattr(detector.cv2.dnn, "NMSBoxes", failing_nms_boxes)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector._detect_with_east(
        image,
        FakeEastNet(),
        min_confidence=0.3,
        nms_threshold=0.4,
        width=100,
        height=100,
    )

    boxes = [tuple(detection["geometry"][0].astype(int)) for detection in detections]
    assert boxes == [(10, 10), (70, 70)]
