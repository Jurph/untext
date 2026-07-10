"""Tests for untextre.preprocessor module.

Covers:
    - ``preprocess_image()`` — BGR input, grayscale input, exception handling
"""

import cv2
import logging
import numpy as np

from untextre.preprocessor import preprocess_image


class TestPreprocessImage:
    """Test the preprocessing pipeline."""




    def test_exception_returns_none(self, monkeypatch, caplog):
        """Internal failure returns None instead of propagating."""
        # Force cvtColor to raise
        monkeypatch.setattr(cv2, "cvtColor", lambda *a, **kw: (_ for _ in ()).throw(cv2.error("boom")))
        image = np.ones((40, 40, 3), dtype=np.uint8) * 128
        with caplog.at_level(logging.ERROR):
            result = preprocess_image(image)
        assert result is None
        assert any(record.exc_info for record in caplog.records)
        assert "Failed to preprocess image array" in caplog.text
