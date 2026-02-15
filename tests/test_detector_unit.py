"""Direct unit tests for untextre.detector module.

The consensus tests exercise detectors *through* the consensus API,
but these tests pin the detector module's own public surface directly:
    - ``TextDetector``  – initialization, parameter validation, detect() return format
    - ``get_largest_text_region()``  – selection + merge logic
    - ``detect_text_regions()``  – module-level entry point
    - ``cleanup_vram()``  – should not crash regardless of GPU availability

Heavy model-loading happens once via the autouse fixture.
"""

import numpy as np
import cv2
import pytest

# Every test in this module loads ML models (DocTR).
pytestmark = pytest.mark.slow

from untextre.detector import (
    TextDetector,
    detect_text_regions,
    get_largest_text_region,
    cleanup_vram,
    _merge_bboxes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    """Create a TextDetector once for all tests in this module."""
    return TextDetector(confidence_threshold=0.1, min_text_size=3)


@pytest.fixture
def image_with_text():
    """200×300 white image with large black text (detectable by DocTR)."""
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "HELLO WORLD", (10, 130), font, 1.5, (0, 0, 0), 3)
    return image


@pytest.fixture
def blank_image():
    """200×200 white image with no text."""
    return np.ones((200, 200, 3), dtype=np.uint8) * 255


# =========================================================================
# TextDetector.__init__
# =========================================================================

class TestTextDetectorInit:
    """Verify constructor validation and basic properties."""

    def test_default_construction(self, detector):
        assert detector is not None
        assert hasattr(detector, "confidence_threshold")
        assert hasattr(detector, "min_text_size")

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            TextDetector(confidence_threshold=1.5)

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            TextDetector(confidence_threshold=-0.1)

    def test_non_positive_min_text_size_raises(self):
        with pytest.raises(ValueError, match="min_text_size"):
            TextDetector(min_text_size=0)


# =========================================================================
# TextDetector.detect
# =========================================================================

class TestTextDetectorDetect:
    """Verify detect() return format on synthetic images."""

    def test_returns_list(self, detector, image_with_text):
        result = detector.detect(image_with_text)
        assert isinstance(result, list)

    def test_detection_dict_has_required_keys(self, detector, image_with_text):
        result = detector.detect(image_with_text)
        if not result:
            pytest.skip("DocTR found no detections on synthetic image (model-dependent)")

        for det in result:
            assert "geometry" in det, "Detection should have 'geometry' key"
            assert "confidence" in det, "Detection should have 'confidence' key"
            assert isinstance(det["geometry"], np.ndarray)
            assert det["geometry"].ndim == 2
            assert det["geometry"].shape[1] == 2  # (N, 2) polygon points

    def test_blank_image_produces_few_detections(self, detector, blank_image):
        result = detector.detect(blank_image)
        # A blank image should produce few or zero detections
        assert len(result) <= 3, f"Too many detections ({len(result)}) on blank image"

    def test_confidence_values_in_range(self, detector, image_with_text):
        result = detector.detect(image_with_text)
        for det in result:
            conf = det["confidence"]
            assert 0 <= conf <= 1, f"Confidence {conf} out of [0, 1]"


# =========================================================================
# detect_text_regions  (module-level entry point)
# =========================================================================

class TestDetectTextRegions:
    """Verify the module-level detection wrapper."""

    def test_returns_list_of_tuples(self, image_with_text):
        bboxes = detect_text_regions(image_with_text, method="doctr", confidence_threshold=0.1)
        assert isinstance(bboxes, list)
        for bbox in bboxes:
            assert len(bbox) == 4, "Each bbox should be (x, y, w, h)"
            x, y, w, h = bbox
            assert w > 0 and h > 0

    def test_blank_image_returns_few_boxes(self, blank_image):
        bboxes = detect_text_regions(blank_image, method="doctr", confidence_threshold=0.3)
        assert len(bboxes) <= 3


# =========================================================================
# get_largest_text_region
# =========================================================================

class TestGetLargestTextRegion:
    """Verify the selection + merge logic."""

    def test_returns_bbox_on_text_image(self, image_with_text):
        bbox = get_largest_text_region(image_with_text, method="doctr", confidence_threshold=0.1)
        x, y, w, h = bbox
        assert w > 0 and h > 0

    def test_no_text_raises_value_error(self, blank_image):
        """Blank image with high threshold should raise ValueError."""
        with pytest.raises(ValueError, match="No text regions detected"):
            get_largest_text_region(blank_image, method="doctr", confidence_threshold=0.99)


# =========================================================================
# _merge_bboxes  (internal but important for correctness)
# =========================================================================

class TestMergeBboxes:
    """Verify bounding box merge logic."""

    def test_single_bbox_unchanged(self):
        assert _merge_bboxes([(10, 20, 30, 40)]) == (10, 20, 30, 40)

    def test_two_adjacent_horizontal(self):
        # Two boxes side by side: (0,0,50,20) and (50,0,50,20)
        merged = _merge_bboxes([(0, 0, 50, 20), (50, 0, 50, 20)])
        assert merged == (0, 0, 100, 20)

    def test_two_overlapping(self):
        merged = _merge_bboxes([(10, 10, 40, 20), (30, 10, 40, 20)])
        assert merged == (10, 10, 60, 20)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _merge_bboxes([])

    def test_three_scattered_boxes(self):
        merged = _merge_bboxes([(0, 0, 10, 10), (50, 50, 10, 10), (90, 0, 10, 10)])
        assert merged == (0, 0, 100, 60)


# =========================================================================
# cleanup_vram
# =========================================================================

class TestCleanupVram:
    """Verify cleanup_vram doesn't crash (regardless of GPU availability)."""

    def test_does_not_raise(self):
        # Should be safe to call even without a GPU
        cleanup_vram()
