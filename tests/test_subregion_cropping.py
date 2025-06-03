"""Tests for the subregion cropping functionality."""

import cv2
import numpy as np
import pytest
from pathlib import Path
from untext.image_patcher import ImagePatcher

def create_test_image(size=(100, 100), text="Test"):
    """Create a test image with text.
    
    Args:
        size: Image size (height, width)
        text: Text to write on image
        
    Returns:
        Tuple of (image array, mask array)
    """
    # Create white background
    image = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 0)  # Black text
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position to center text
    x = (size[1] - text_width) // 2
    y = (size[0] + text_height) // 2
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    # Create mask (same size as image)
    mask = np.zeros(size, dtype=np.uint8)
    
    # Fill text region in mask
    cv2.putText(mask, text, (x, y), font, font_scale, 255, thickness)
    
    return image, mask

def test_calculate_subregion():
    """Test subregion calculation."""
    # Create test image
    image, mask = create_test_image((200, 200), "Test")
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Test with detections
    detections = [{
        'geometry': np.array([
            [50, 50],
            [150, 50],
            [150, 150],
            [50, 150]
        ], dtype=np.float32),
        'confidence': 0.9
    }]
    
    subregion = patcher.calculate_subregion(detections, image.shape[:2])
    assert subregion is not None
    assert len(subregion) == 4
    assert all(isinstance(x, (int, float)) for x in subregion)
    assert subregion[0] <= subregion[2]  # x1 <= x2
    assert subregion[1] <= subregion[3]  # y1 <= y2
    
    # Test with mask
    subregion = patcher.calculate_subregion([], image.shape[:2], mask=mask)
    assert subregion is not None
    assert len(subregion) == 4
    assert all(isinstance(x, (int, float)) for x in subregion)
    assert subregion[0] <= subregion[2]  # x1 <= x2
    assert subregion[1] <= subregion[3]  # y1 <= y2
    
    # Test with no detections and no mask
    subregion = patcher.calculate_subregion([], image.shape[:2])
    assert subregion is not None
    assert subregion == (0, 0, image.shape[1], image.shape[0])  # Full image

def test_subregion_bounds():
    """Test that subregion is within image bounds."""
    # Create test image
    image, _ = create_test_image((200, 200), "Test")
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Test with detections near edges
    detections = [{
        'geometry': np.array([
            [0, 0],
            [50, 0],
            [50, 50],
            [0, 50]
        ], dtype=np.float32),
        'confidence': 0.9
    }]
    
    subregion = patcher.calculate_subregion(detections, image.shape[:2])
    assert subregion is not None
    assert subregion[0] >= 0  # x1 >= 0
    assert subregion[1] >= 0  # y1 >= 0
    assert subregion[2] <= image.shape[1]  # x2 <= width
    assert subregion[3] <= image.shape[0]  # y2 <= height

def test_subregion_scaling():
    """Test subregion scaling with different scale factors."""
    # Create test image
    image, _ = create_test_image((200, 200), "Test")
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Test with different scale factors
    detections = [{
        'geometry': np.array([
            [50, 50],
            [100, 50],
            [100, 100],
            [50, 100]
        ], dtype=np.float32),
        'confidence': 0.9
    }]
    
    # Test with scale_factor=2
    subregion = patcher.calculate_subregion(detections, image.shape[:2], scale_factor=2)
    assert subregion is not None
    width = subregion[2] - subregion[0]
    height = subregion[3] - subregion[1]
    assert width > 50  # Should be larger than detection width
    assert height > 50  # Should be larger than detection height
    
    # Test with scale_factor=4
    subregion = patcher.calculate_subregion(detections, image.shape[:2], scale_factor=4)
    assert subregion is not None
    width = subregion[2] - subregion[0]
    height = subregion[3] - subregion[1]
    assert width > 100  # Should be even larger
    assert height > 100  # Should be even larger

def test_subregion_min_margin():
    """Test subregion minimum margin."""
    # Create test image
    image, _ = create_test_image((200, 200), "Test")
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Test with small detection
    detections = [{
        'geometry': np.array([
            [50, 50],
            [60, 50],
            [60, 60],
            [50, 60]
        ], dtype=np.float32),
        'confidence': 0.9
    }]
    
    # Test with min_margin=20
    subregion = patcher.calculate_subregion(detections, image.shape[:2], min_margin=20)
    assert subregion is not None
    width = subregion[2] - subregion[0]
    height = subregion[3] - subregion[1]
    assert width >= 40  # Should be at least 2 * min_margin
    assert height >= 40  # Should be at least 2 * min_margin 