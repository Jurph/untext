"""Tests for the text detector module.

This module contains unit tests for the TextDetector class, verifying its functionality
for text detection, mask generation, and detection filtering. The tests are designed
to be thorough and handle edge cases.
"""

import os
from typing import List, Tuple, Dict, Any, Type
import numpy as np
import cv2
import pytest
from untext.detector import TextDetector


def test_detector_initialization() -> None:
    """Test that the detector initializes correctly with default and custom parameters.
    
    This test verifies that:
    1. Default parameters are set correctly
    2. Custom parameters override defaults properly
    3. All required attributes are initialized
    4. Invalid parameters raise appropriate exceptions
    """
    # Test default initialization
    detector = TextDetector()
    assert detector.confidence_threshold == 0.3
    assert detector.mask_dilation == 2
    assert detector.min_text_size == 10
    
    # Test custom initialization
    detector = TextDetector(
        confidence_threshold=0.5,
        mask_dilation=3,
        min_text_size=20
    )
    assert detector.confidence_threshold == 0.5
    assert detector.mask_dilation == 3
    assert detector.min_text_size == 20
    
    # Test invalid parameters
    with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
        TextDetector(confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
        TextDetector(confidence_threshold=1.1)
    with pytest.raises(ValueError, match="mask_dilation must be non-negative"):
        TextDetector(mask_dilation=-1)
    with pytest.raises(ValueError, match="min_text_size must be positive"):
        TextDetector(min_text_size=0)


def test_geometry_to_mask() -> None:
    """Test conversion of geometry points to binary masks.
    
    This test verifies that:
    1. Geometry points are correctly converted to a binary mask
    2. The mask has the correct shape and type
    3. The mask correctly represents the polygon region
    4. Dilation increases the mask size appropriately
    5. Invalid inputs raise appropriate exceptions
    """
    detector = TextDetector()
    
    # Create a test image
    image_shape: Tuple[int, int] = (100, 100)
    
    # Test single polygon
    geometry = np.array([
        [10, 10], [30, 10], [30, 30], [10, 30]
    ], dtype=np.float32)
    mask = detector._geometry_to_mask(geometry, image_shape)
    
    # Check mask properties
    assert mask.shape == image_shape
    assert mask.dtype == np.uint8
    assert np.sum(mask) > 0  # Should have some white pixels
    
    # Check polygon location
    assert mask[20, 20] == 1  # Inside polygon
    assert mask[5, 5] == 0    # Outside polygon
    
    # Test dilation
    detector.mask_dilation = 2
    mask_dilated = detector._geometry_to_mask(geometry, image_shape)
    assert np.sum(mask_dilated) > np.sum(mask)  # Dilated mask should be larger
    
    # Test invalid inputs
    with pytest.raises(ValueError, match="Geometry must be a numpy array"):
        detector._geometry_to_mask([(0, 0), (1, 1), (2, 2)], image_shape)
    with pytest.raises(ValueError, match="Geometry must contain at least 3 points"):
        detector._geometry_to_mask(np.array([[0, 0], [1, 1]], dtype=np.float32), image_shape)
    with pytest.raises(ValueError, match="image_shape must be a tuple of two positive integers"):
        detector._geometry_to_mask(geometry, (0, 100))
    with pytest.raises(ValueError, match="Geometry points must be within image bounds"):
        detector._geometry_to_mask(
            np.array([[-1, -1], [101, 101], [50, 50]], dtype=np.float32),
            image_shape
        )


def test_filter_detections() -> None:
    """Test filtering of detections based on various criteria.
    
    This test verifies that:
    1. Words that are too small are filtered out
    2. Words too close to image edges are filtered out
    3. Valid words are preserved with correct attributes
    4. Invalid inputs raise appropriate exceptions
    """
    detector = TextDetector(
        min_text_size=10
    )
    
    image_shape: Tuple[int, int] = (100, 100)
    
    # Test cases
    words = [
        np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32),  # Good word
        np.array([[10, 10], [15, 10], [15, 15], [10, 15]], dtype=np.float32),  # Too small
        np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32),      # Too close to edge
        np.array([[80, 80], [100, 80], [100, 100], [80, 100]], dtype=np.float32), # Too close to edge
    ]
    
    filtered = detector._filter_detections(words, image_shape)
    
    # Should only keep the first word
    assert len(filtered) == 1
    assert np.array_equal(filtered[0]['geometry'], words[0])
    assert filtered[0]['confidence'] == 1.0
    assert filtered[0]['text'] == ''
    
    # Test invalid inputs
    with pytest.raises(ValueError, match="image_shape must be a tuple of two positive integers"):
        detector._filter_detections(words, (0, 100))


def test_detection_output_format() -> None:
    """Test the format and content of detection results.
    
    This test verifies that:
    1. The detector returns a valid mask and detection list
    2. The mask has the correct shape and type
    3. Detections have the correct format and valid values
    4. Geometry points and confidence scores are properly formatted
    5. Invalid inputs raise appropriate exceptions
    """
    # Create a test image with text
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.putText(image, "TEST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite("images/test_image.jpg", image)
    
    try:
        # Run detection
        detector = TextDetector()
        mask, detections = detector.detect("images/test_image.jpg")
        
        # Check mask
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert np.sum(mask) > 0  # Should detect some text
        
        # Check detections
        assert isinstance(detections, list)
        if detections:  # If any text was detected
            det = detections[0]
            assert isinstance(det, dict)
            assert 'geometry' in det
            assert 'confidence' in det
            assert 'text' in det
            
            # Check geometry format
            geometry = det['geometry']
            assert isinstance(geometry, np.ndarray)
            assert geometry.dtype == np.float32
            assert geometry.shape[1] == 2  # Each point should have x,y coordinates
            
            # Check confidence
            assert isinstance(det['confidence'], float)
            assert det['confidence'] == 1.0  # DocTR detection doesn't provide confidence scores
            
            # Check text
            assert isinstance(det['text'], str)
            assert det['text'] == ''  # DocTR detection doesn't provide text
            
        # Test invalid inputs
        with pytest.raises(FileNotFoundError):
            detector.detect("nonexistent.jpg")
            
        # Create an invalid image
        invalid_image = np.ones((100, 100), dtype=np.uint8)  # 2D instead of 3D
        cv2.imwrite("images/invalid_image.jpg", invalid_image)
        try:
            with pytest.raises(ValueError, match="Invalid image format"):
                detector.detect("images/invalid_image.jpg")
        finally:
            if os.path.exists("images/invalid_image.jpg"):
                os.remove("images/invalid_image.jpg")
    finally:
        # Clean up
        if os.path.exists("images/test_image.jpg"):
            os.remove("images/test_image.jpg") 