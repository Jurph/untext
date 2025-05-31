"""Common test fixtures."""

import os
import pytest
import numpy as np
from pathlib import Path
import cv2


@pytest.fixture
def test_image_dir():
    """Return the path to the test images directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_image(test_image_dir):
    """Create a sample test image with a watermark."""
    # Create a simple test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    # Add some text watermark
    cv2.putText(
        image,
        "TEST",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    
    # Save it
    os.makedirs(test_image_dir, exist_ok=True)
    image_path = test_image_dir / "test_image.jpg"
    cv2.imwrite(str(image_path), image)
    
    return image_path 