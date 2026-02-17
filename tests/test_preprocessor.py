"""Tests for untextre.preprocessor module.

Covers:
    - ``preprocess_image()`` — BGR input, grayscale input, exception handling
"""

import cv2
import numpy as np
import pytest

from untextre.preprocessor import preprocess_image


class TestPreprocessImage:
    """Test the preprocessing pipeline."""

    def test_bgr_input_returns_rgb(self):
        """Standard 3-channel BGR input should produce 3-channel RGB output."""
        bgr = np.ones((60, 80, 3), dtype=np.uint8) * 128
        result = preprocess_image(bgr)
        assert result is not None
        assert result.shape == bgr.shape
        assert result.dtype == np.uint8

    def test_grayscale_input_returns_rgb(self):
        """Single-channel grayscale input should still produce 3-channel output."""
        gray = np.ones((60, 80), dtype=np.uint8) * 128
        result = preprocess_image(gray)
        assert result is not None
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.shape[:2] == gray.shape

    def test_output_not_same_object(self):
        """Preprocessing should not return the original array."""
        image = np.ones((40, 40, 3), dtype=np.uint8) * 200
        result = preprocess_image(image)
        assert result is not image

    def test_exception_returns_none(self, monkeypatch):
        """Internal failure returns None instead of propagating."""
        # Force cvtColor to raise
        monkeypatch.setattr(cv2, "cvtColor", lambda *a, **kw: (_ for _ in ()).throw(cv2.error("boom")))
        image = np.ones((40, 40, 3), dtype=np.uint8) * 128
        result = preprocess_image(image)
        assert result is None
