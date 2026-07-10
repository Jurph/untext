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

# Every test in this module loads ML models (DocTR, EasyOCR, EAST).
pytestmark = pytest.mark.slow

# Import consensus detection functions
from untextre.consensus import (
    detect_with_doctr,
    detect_with_easyocr, 
    detect_with_east,
    run_consensus_detection,
    find_consensus_boxes,
)
from untextre.pipeline import (
    initialize_consensus_models as init_all_models,
    _apply_color_enhancement,
    _try_color_enhanced_detection
)
from untextre.utils import load_image


@pytest.fixture(autouse=True)
def setup_models():
    """Initialize all detection models once before running tests."""
    # Initialize consensus models to avoid repeated loading
    try:
        init_all_models(device="cuda")
    except Exception:
        # Fall back to CPU if CUDA is not available
        init_all_models(device="cpu")


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


















def test_detection_confidence_thresholds(test_images):
    """Test that confidence thresholds work properly."""
    if not test_images:
        pytest.skip("No test images available")
    
    # Test with first available image
    image_name, image = next(iter(test_images.items()))
    
    # Test with high confidence threshold - should get fewer detections
    high_threshold_detections = detect_with_doctr(image, confidence_threshold=0.8)
    
    # Test with low confidence threshold - should get more detections  
    low_threshold_detections = detect_with_doctr(image, confidence_threshold=0.1)
    
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
    detections1 = detect_with_doctr(image, confidence_threshold=0.3)
    detections2 = detect_with_doctr(image, confidence_threshold=0.3) 
    
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
