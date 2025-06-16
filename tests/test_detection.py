"""Tests for consensus text detection functionality.

This module tests the three text detectors (EAST, DocTR, EasyOCR) and the consensus
detection system that combines their results to find high-confidence text regions.

These tests verify that:
1. All three detectors can be initialized and run
2. Detectors return reasonable results on null images (few/no detections)
3. Detectors find similar text regions on real test images
4. Consensus detection properly combines results from multiple detectors
5. Color-based enhancement failover modes work correctly
"""

import numpy as np
import cv2
import pytest
from pathlib import Path
from typing import List, Tuple, Dict

# Import our consensus detection functions from CLI
from untextre.cli import (
    _detect_with_doctr_configurable,
    _detect_with_easyocr_configurable, 
    _detect_with_east_configurable,
    run_consensus_detection,
    find_consensus_boxes,
    initialize_consensus_models,
    _apply_color_enhancement,
    _try_color_enhanced_detection
)
from untextre.utils import load_image


@pytest.fixture(autouse=True)
def setup_models():
    """Initialize all detection models once before running tests."""
    # Initialize consensus models to avoid repeated loading
    try:
        initialize_consensus_models(confidence_threshold=0.3, device="cuda")
    except Exception:
        # Fall back to CPU if CUDA is not available
        initialize_consensus_models(confidence_threshold=0.3, device="cpu")


@pytest.fixture
def null_image():
    """Create a blank image with no text."""
    # White background, no text
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    return image


@pytest.fixture
def gray_text_image():
    """Create an image with gray text that might be hard to detect."""
    # White background
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Add gray text in the target range (#7E7E7E to #828282)
    gray_color = (128, 128, 128)  # BGR format, middle of target range
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "GRAY TEXT", (50, 100), font, 1, gray_color, 2)
    
    return image


@pytest.fixture
def white_text_image():
    """Create an image with near-white text on a colored background."""
    # Light gray background
    image = np.ones((200, 200, 3), dtype=np.uint8) * 200
    
    # Add near-white text in the target range (#FCFCFC to #FFFFFF)
    white_color = (254, 254, 254)  # BGR format, near-white
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "WHITE TEXT", (40, 100), font, 1, white_color, 2)
    
    return image


@pytest.fixture
def test_images():
    """Load available test images."""
    images = {}
    test_dir = Path("tests/images")
    
    # Load test images that should exist
    for img_name in ["test1.png", "test2.png"]:
        img_path = test_dir / img_name
        if img_path.exists():
            image = load_image(img_path)
            if image is not None:
                images[img_name] = image
    
    return images


def test_color_enhancement_gray(gray_text_image):
    """Test gray color enhancement functionality."""
    # Apply gray enhancement using hex color
    enhanced = _apply_color_enhancement(gray_text_image, "#808080", sensitivity=3)
    
    # Enhanced image should be different from original
    assert not np.array_equal(enhanced, gray_text_image), "Enhanced image should be different from original"
    
    # Should have some black pixels (converted gray text)
    black_pixels = np.sum(np.all(enhanced == [0, 0, 0], axis=2))
    assert black_pixels > 0, "Enhanced image should have black pixels from converted gray text"
    
    # Original image should not have black pixels
    original_black_pixels = np.sum(np.all(gray_text_image == [0, 0, 0], axis=2))
    assert black_pixels > original_black_pixels, "Enhanced image should have more black pixels than original"


def test_color_enhancement_white(white_text_image):
    """Test white color enhancement functionality."""
    # Apply white enhancement using hex color
    enhanced = _apply_color_enhancement(white_text_image, "#FFFFFF", sensitivity=3)
    
    # Enhanced image should be different from original
    assert not np.array_equal(enhanced, white_text_image), "Enhanced image should be different from original"
    
    # Should have some black pixels (converted white text)
    black_pixels = np.sum(np.all(enhanced == [0, 0, 0], axis=2))
    assert black_pixels > 0, "Enhanced image should have black pixels from converted white text"


def test_color_enhancement_custom_color():
    """Test color enhancement with custom colors and sensitivity."""
    # Create image with specific blue color
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    blue_color = (255, 100, 100)  # BGR: bright blue
    cv2.rectangle(image, (30, 30), (70, 70), blue_color, -1)
    
    # Apply enhancement targeting that blue color
    enhanced = _apply_color_enhancement(image, "#6464FF", sensitivity=5)  # RGB 100,100,255 -> hex
    
    # Should have black pixels where blue was
    black_pixels = np.sum(np.all(enhanced == [0, 0, 0], axis=2))
    assert black_pixels > 0, "Should have black pixels from converted blue region"


def test_color_enhancement_sensitivity():
    """Test different sensitivity values."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    gray_color = (128, 128, 128)  # BGR: medium gray
    cv2.rectangle(image, (30, 30), (70, 70), gray_color, -1)
    
    # Test with different sensitivities
    enhanced_low = _apply_color_enhancement(image, "#808080", sensitivity=1)
    enhanced_high = _apply_color_enhancement(image, "#808080", sensitivity=10)
    
    # Higher sensitivity should affect more pixels (if there are nearby colors)
    black_pixels_low = np.sum(np.all(enhanced_low == [0, 0, 0], axis=2))
    black_pixels_high = np.sum(np.all(enhanced_high == [0, 0, 0], axis=2))
    
    # Both should convert the exact match
    assert black_pixels_low > 0, "Low sensitivity should convert exact color match"
    assert black_pixels_high >= black_pixels_low, "Higher sensitivity should convert at least as many pixels"


def test_color_enhancement_invalid_hex():
    """Test that invalid hex formats raise errors."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Test various invalid formats
    with pytest.raises(ValueError, match="Invalid hex color format"):
        _apply_color_enhancement(image, "808080")  # Missing #
    
    with pytest.raises(ValueError, match="Invalid hex color format"):
        _apply_color_enhancement(image, "#80")  # Too short
    
    with pytest.raises(ValueError, match="Invalid hex color format"):
        _apply_color_enhancement(image, "#GGGGGG")  # Invalid hex characters


def test_color_enhanced_detection_pipeline(gray_text_image):
    """Test the full color enhancement detection pipeline."""
    # This test simulates what happens when normal detection fails
    # and the system tries color enhancement
    
    # Try color enhanced detection with hex color
    consensus_boxes = _try_color_enhanced_detection(gray_text_image, 0.3, "#808080", sensitivity=3)
    
    # Should return a list (may be empty if detectors don't find the synthetic text)
    assert isinstance(consensus_boxes, list), "Should return a list of consensus boxes"
    
    # If boxes are found, they should be properly formatted
    for box in consensus_boxes:
        assert len(box) == 4, "Each consensus box should have 4 elements (x, y, w, h)"
        x, y, w, h = box
        assert w > 0 and h > 0, "Consensus box should have positive dimensions"


def test_doctr_detector_on_null_image(null_image):
    """Test DocTR detector on image with no text."""
    detections = _detect_with_doctr_configurable(null_image, confidence_threshold=0.3)
    
    # Should return a list
    assert isinstance(detections, list)
    
    # Should have few or no detections on blank image
    assert len(detections) <= 5, f"Too many detections ({len(detections)}) on blank image"
    
    # Each detection should be properly formatted
    for detection in detections:
        assert len(detection) == 5  # (x, y, w, h, confidence)
        x, y, w, h, conf = detection
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))
        assert isinstance(conf, (int, float))
        assert 0 <= conf <= 1, f"Confidence {conf} should be between 0 and 1"


def test_easyocr_detector_on_null_image(null_image):
    """Test EasyOCR detector on image with no text."""
    detections = _detect_with_easyocr_configurable(null_image, confidence_threshold=0.3)
    
    # Should return a list
    assert isinstance(detections, list)
    
    # Should have few or no detections on blank image
    assert len(detections) <= 5, f"Too many detections ({len(detections)}) on blank image"
    
    # Each detection should be properly formatted
    for detection in detections:
        assert len(detection) == 5  # (x, y, w, h, confidence)
        x, y, w, h, conf = detection
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))
        assert isinstance(conf, (int, float))
        assert 0 <= conf <= 1, f"Confidence {conf} should be between 0 and 1"


def test_east_detector_on_null_image(null_image):
    """Test EAST detector on image with no text."""
    detections = _detect_with_east_configurable(null_image, confidence_threshold=0.3)
    
    # Should return a list
    assert isinstance(detections, list)
    
    # Should have few or no detections on blank image
    assert len(detections) <= 5, f"Too many detections ({len(detections)}) on blank image"
    
    # Each detection should be properly formatted
    for detection in detections:
        assert len(detection) == 5  # (x, y, w, h, confidence)
        x, y, w, h, conf = detection
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))
        assert isinstance(conf, (int, float))
        assert 0 <= conf <= 1, f"Confidence {conf} should be between 0 and 1"


def test_all_detectors_on_test_image(test_images):
    """Test all three detectors on real test images and compare results."""
    if not test_images:
        pytest.skip("No test images available")
    
    # Test with first available image
    image_name, image = next(iter(test_images.items()))
    print(f"\nTesting detectors on {image_name}")
    
    # Run all three detectors
    doctr_detections = _detect_with_doctr_configurable(image, confidence_threshold=0.3)
    easyocr_detections = _detect_with_easyocr_configurable(image, confidence_threshold=0.3) 
    east_detections = _detect_with_east_configurable(image, confidence_threshold=0.3)
    
    # All should return lists
    assert isinstance(doctr_detections, list)
    assert isinstance(easyocr_detections, list)
    assert isinstance(east_detections, list)
    
    # Print results for debugging
    print(f"DocTR found {len(doctr_detections)} detections")
    print(f"EasyOCR found {len(easyocr_detections)} detections") 
    print(f"EAST found {len(east_detections)} detections")
    
    # At least one detector should find something (test images have text)
    total_detections = len(doctr_detections) + len(easyocr_detections) + len(east_detections)
    assert total_detections > 0, f"No detections found by any detector on {image_name}"
    
    # Validate detection formats
    for detections, detector_name in [
        (doctr_detections, "DocTR"),
        (easyocr_detections, "EasyOCR"), 
        (east_detections, "EAST")
    ]:
        for i, detection in enumerate(detections):
            assert len(detection) == 5, f"{detector_name} detection {i} should have 5 elements"
            x, y, w, h, conf = detection
            assert w > 0 and h > 0, f"{detector_name} detection {i} should have positive dimensions"
            assert 0 <= conf <= 1, f"{detector_name} detection {i} confidence should be 0-1"


def test_consensus_detection_on_null_image(null_image):
    """Test consensus detection on image with no text."""
    consensus_boxes = run_consensus_detection(null_image, confidence_threshold=0.3)
    
    # Should return a list of bounding boxes
    assert isinstance(consensus_boxes, list)
    
    # Should have few or no consensus regions on blank image
    assert len(consensus_boxes) <= 2, f"Too many consensus regions ({len(consensus_boxes)}) on blank image"
    
    # Each consensus box should be properly formatted
    for box in consensus_boxes:
        assert len(box) == 4  # (x, y, w, h)
        x, y, w, h = box
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))
        assert w > 0 and h > 0, "Consensus box should have positive dimensions"


def test_consensus_detection_on_test_image(test_images):
    """Test consensus detection on real test images."""
    if not test_images:
        pytest.skip("No test images available")
    
    # Test with first available image
    image_name, image = next(iter(test_images.items()))
    print(f"\nTesting consensus detection on {image_name}")
    
    consensus_boxes = run_consensus_detection(image, confidence_threshold=0.3)
    
    # Should return a list
    assert isinstance(consensus_boxes, list)
    
    print(f"Found {len(consensus_boxes)} consensus regions")
    
    # Print consensus boxes for debugging
    for i, box in enumerate(consensus_boxes):
        x, y, w, h = box
        print(f"Consensus box {i+1}: ({x}, {y}) size {w}×{h}")
    
    # Validate consensus box format
    for i, box in enumerate(consensus_boxes):
        assert len(box) == 4, f"Consensus box {i} should have 4 elements (x, y, w, h)"
        x, y, w, h = box
        assert w > 0 and h > 0, f"Consensus box {i} should have positive dimensions"
        
        # Boxes should be within image bounds
        img_h, img_w = image.shape[:2]
        assert 0 <= x < img_w, f"Consensus box {i} x-coordinate out of bounds"
        assert 0 <= y < img_h, f"Consensus box {i} y-coordinate out of bounds"
        assert x + w <= img_w, f"Consensus box {i} extends beyond image width"
        assert y + h <= img_h, f"Consensus box {i} extends beyond image height"


def test_find_consensus_boxes_function():
    """Test the consensus box finding algorithm directly."""
    # Create mock detection results
    detections = {
        "doctr": [
            (100, 100, 50, 20, 0.8),  # x, y, w, h, confidence
            (200, 150, 60, 25, 0.7)
        ],
        "easyocr": [
            (105, 105, 45, 18, 0.9),  # Overlaps with first DocTR detection
            (300, 200, 40, 15, 0.6)   # Different location
        ],
        "east": [
            (98, 102, 52, 22, 0.75),  # Overlaps with first DocTR detection
            (205, 148, 55, 28, 0.65)  # Overlaps with second DocTR detection
        ]
    }
    
    consensus_boxes = find_consensus_boxes(detections, overlap_threshold=0.1)
    
    # Should return a list of consensus boxes
    assert isinstance(consensus_boxes, list)
    
    # Should find at least one consensus (first detection overlaps across detectors)
    assert len(consensus_boxes) >= 1, "Should find at least one consensus region"
    
    # Each consensus box should have the required fields
    for box in consensus_boxes:
        assert isinstance(box, dict)
        assert "bbox" in box
        assert "detectors" in box  
        assert "confidence" in box
        
        # Validate bbox format
        bbox = box["bbox"]
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w > 0 and h > 0
        
        # Should have detections from multiple detectors for consensus
        assert len(box["detectors"]) >= 2, "Consensus requires multiple detectors"
        
        # Confidence should be reasonable
        assert 0 <= box["confidence"] <= 1


def test_detection_confidence_thresholds(test_images):
    """Test that confidence thresholds work properly."""
    if not test_images:
        pytest.skip("No test images available")
    
    # Test with first available image
    image_name, image = next(iter(test_images.items()))
    
    # Test with high confidence threshold - should get fewer detections
    high_threshold_detections = _detect_with_doctr_configurable(image, confidence_threshold=0.8)
    
    # Test with low confidence threshold - should get more detections  
    low_threshold_detections = _detect_with_doctr_configurable(image, confidence_threshold=0.1)
    
    print(f"\nDocTR on {image_name}:")
    print(f"High threshold (0.8): {len(high_threshold_detections)} detections")
    print(f"Low threshold (0.1): {len(low_threshold_detections)} detections")
    
    # Low threshold should generally find more or equal detections
    assert len(low_threshold_detections) >= len(high_threshold_detections), \
        "Lower confidence threshold should find more or equal detections"
    
    # All high-confidence detections should actually have high confidence
    for detection in high_threshold_detections:
        _, _, _, _, conf = detection
        assert conf >= 0.8, f"High threshold detection has confidence {conf} < 0.8"


def test_detector_consistency_across_runs(test_images):
    """Test that detectors give consistent results across multiple runs."""
    if not test_images:
        pytest.skip("No test images available")
    
    # Test with first available image
    image_name, image = next(iter(test_images.items()))
    
    # Run DocTR detector twice
    detections1 = _detect_with_doctr_configurable(image, confidence_threshold=0.3)
    detections2 = _detect_with_doctr_configurable(image, confidence_threshold=0.3) 
    
    # Should get same number of detections
    assert len(detections1) == len(detections2), \
        f"Inconsistent results: {len(detections1)} vs {len(detections2)} detections"
    
    # If there are detections, they should be very similar
    if detections1:
        # Compare first detection as sanity check
        det1 = detections1[0]
        det2 = detections2[0]
        
        # Coordinates should be identical or very close
        for i in range(4):  # x, y, w, h
            diff = abs(det1[i] - det2[i])
            assert diff <= 1, f"Detection coordinate {i} differs by {diff} between runs"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

