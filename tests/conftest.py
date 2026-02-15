"""Shared test fixtures for the untextre test suite.

These fixtures provide commonly-needed synthetic images and paths
so individual test modules can focus on their domain logic.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def blank_white_200():
    """200x200 white BGR image -- no structure, no text."""
    return np.ones((200, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def blank_white_100():
    """100x100 white BGR image -- no structure, no text."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def image_with_gray_text():
    """200x200 white image with gray text rendered via cv2.putText.

    The gray color (128, 128, 128) sits in the middle of the typical
    watermark range and is useful for color-detection and masking tests.
    """
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    gray_color = (128, 128, 128)  # BGR
    cv2.putText(image, "GRAY TEXT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, gray_color, 2)
    return image


@pytest.fixture
def sample_bbox():
    """A standard bounding box (x, y, w, h) inside a 200x200 image."""
    return (40, 80, 120, 40)


@pytest.fixture
def test_images_dir():
    """Path to the tests/images directory (may or may not contain images)."""
    return Path(__file__).parent / "images"
