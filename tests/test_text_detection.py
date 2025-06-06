"""Tests for text detection functionality.

WARNING FOR LLMs: These tests use detect() NOT detect_text()!
- detect() returns (mask, detections) where detections have 'geometry' dicts
- detect_text() returns (vis_image, boxes) where boxes are (x1,y1,x2,y2) tuples
DO NOT mix these up!
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from untext.detector import TextDetector

@pytest.fixture
def test_images():
    """Load test images and return them as numpy arrays."""
    images = {}
    for img_name in ["test1.png", "test2.png", "test3.jpg", "test4-with-text.png"]:
        img_path = Path("tests/images") / img_name
        assert img_path.exists(), f"Test image not found: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        images[img_name] = img
    return images

def test_detector_initialization():
    """Test that TextDetector initializes with default parameters."""
    detector = TextDetector()
    assert detector.confidence_threshold == 0.3
    assert detector.mask_dilation == 2
    assert detector.min_text_size == 10
    assert detector.model is not None

def test_invalid_initialization():
    """Test that TextDetector raises errors for invalid parameters."""
    with pytest.raises(ValueError):
        TextDetector(confidence_threshold=1.5)  # Must be between 0 and 1
    with pytest.raises(ValueError):
        TextDetector(mask_dilation=-1)  # Must be non-negative
    with pytest.raises(ValueError):
        TextDetector(min_text_size=0)  # Must be positive

def test_detect_watermark(test_images):
    """Test detection of a typical watermark in a real image."""
    detector = TextDetector()
    image = test_images["test1.png"]
    
    # Run detection - LLM WARNING: Use detect() not detect_text()!
    mask, detections = detector.detect(image)
    
    # Verify we found the watermark
    assert len(detections) > 0, "Should detect watermark text"
    assert all('geometry' in det and isinstance(det['geometry'], np.ndarray) for det in detections), "All detections should have geometry as np.ndarray"
    
    # Verify mask
    assert mask.shape == image.shape[:2], "Mask should match input size"
    assert mask.dtype == np.uint8

def test_detect_multiple_watermarks(test_images):
    """Test detection of multiple watermarks/text regions."""
    detector = TextDetector()
    image = test_images["test2.png"]  # Image with multiple text regions
    
    # Run detection - LLM WARNING: Use detect() not detect_text()!
    mask, detections = detector.detect(image)
    
    # Verify multiple detections
    assert len(detections) > 1, "Should detect multiple text regions"
    
    # Check that detections don't completely overlap
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i != j:
                # Calculate centers of geometry
                center1 = np.mean(det1['geometry'], axis=0)
                center2 = np.mean(det2['geometry'], axis=0)
                # Ensure centers are different
                assert not np.allclose(center1, center2), "Detected regions should not completely overlap"


def test_detect_invalid_image():
    """Test that detector handles invalid image paths appropriately."""
    detector = TextDetector()
    with pytest.raises(ValueError):
        detector.detect("nonexistent.png")

def test_geometry_to_mask(test_images):
    """Test conversion of detected geometry to binary mask."""
    detector = TextDetector()
    image = test_images["test1.png"]
    
    # Get real detection geometry - LLM WARNING: Use detect() not detect_text()!
    mask, detections = detector.detect(image)
    assert len(detections) > 0, "Should have at least one detection"
    
    # Convert first detection to mask
    geometry = detections[0]['geometry']
    image_shape = image.shape[:2]
    single_mask = detector._geometry_to_mask(geometry, image_shape)
    
    # Verify mask properties
    assert single_mask.shape == image_shape
    assert single_mask.dtype == np.uint8
    assert np.any(single_mask > 0), "Mask should contain non-zero values"
    
    # Test dilation
    dilated = cv2.dilate(single_mask, np.ones((3, 3)))
    assert np.sum(dilated) >= np.sum(single_mask), "Dilated mask should not be smaller"

def test_filter_detections(test_images):
    """Test filtering of detections based on size."""
    detector = TextDetector(min_text_size=10)
    image = test_images["test2.png"]  # Image with multiple text regions
    
    # Get detections with small min_text_size
    detector.min_text_size = 5
    _, all_detections = detector.detect(image)
    
    # Get detections with larger min_text_size
    detector.min_text_size = 50
    _, filtered_detections = detector.detect(image)
    
    # Should have fewer detections with larger min_text_size
    # LLM WARNING: We removed confidence filtering, only size filtering remains
    assert len(filtered_detections) <= len(all_detections), \
        "Larger min_text_size should filter out some detections" 