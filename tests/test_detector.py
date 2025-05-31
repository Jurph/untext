"""Tests for the text detector module."""

import os
import pytest
import numpy as np
import cv2
from pathlib import Path
from untext.detector import TextDetector


def test_detector_initialization():
    """Test that the detector initializes correctly."""
    detector = TextDetector(
        confidence_threshold=0.3,
        mask_dilation=2,
        min_text_size=10
    )
    assert detector.confidence_threshold == 0.3
    assert detector.mask_dilation == 2
    assert detector.min_text_size == 10
    assert detector.model is not None
    assert detector.predictor is not None


def test_geometry_to_mask():
    """Test conversion of geometry to mask."""
    detector = TextDetector()
    image_shape = (100, 100)
    
    # Test with a simple rectangle
    geometry = [(10, 10), (30, 10), (30, 30), (10, 30)]
    mask = detector._geometry_to_mask(geometry, image_shape)
    
    assert isinstance(mask, np.ndarray)
    assert mask.shape == image_shape
    assert mask.dtype == np.uint8
    assert np.sum(mask) > 0  # Should have some white pixels
    
    # Test dilation
    detector.mask_dilation = 2
    dilated_mask = detector._geometry_to_mask(geometry, image_shape)
    assert np.sum(dilated_mask) > np.sum(mask)  # Dilated mask should be larger


def test_filter_detections():
    """Test filtering of detections."""
    detector = TextDetector(min_text_size=10)
    image_shape = (100, 100)
    
    # Create test detections
    detections = [
        {
            'geometry': [(5, 5), (15, 5), (15, 15), (5, 15)],  # Too small
            'confidence': 0.9,
            'text': 'small'
        },
        {
            'geometry': [(10, 10), (30, 10), (30, 30), (10, 30)],  # Good size
            'confidence': 0.9,
            'text': 'good'
        },
        {
            'geometry': [(0, 0), (20, 0), (20, 20), (0, 20)],  # Too close to edge
            'confidence': 0.9,
            'text': 'edge'
        }
    ]
    
    filtered = detector._filter_detections(detections, image_shape)
    assert len(filtered) == 1  # Only the 'good' detection should remain
    assert filtered[0]['text'] == 'good'


def test_detection_output_format(sample_image):
    """Test that detection returns the expected output format."""
    detector = TextDetector()
    mask, detections = detector.detect(str(sample_image))
    
    # Check mask
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.ndim == 2  # Should be 2D (height, width)
    
    # Check detections
    assert isinstance(detections, list)
    if detections:
        assert all(isinstance(d, dict) for d in detections)
        assert all('geometry' in d and 'confidence' in d and 'text' in d for d in detections)
        
        # Check that mask covers detected regions
        for det in detections:
            points = np.array(det['geometry'], dtype=np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            region = mask[y_min:y_max, x_min:x_max]
            assert np.any(region)  # Should have some white pixels in the region 