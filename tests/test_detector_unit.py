"""Direct unit tests for untextre.detector module.

The consensus tests exercise detectors *through* the consensus API,
but these tests pin the detector module's own public surface directly:
    - ``TextDetector``  – initialization, parameter validation, detect() return format
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
    cleanup_vram,
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







# =========================================================================
# detect_text_regions  (module-level entry point)
# =========================================================================





# =========================================================================
# cleanup_vram
# =========================================================================


