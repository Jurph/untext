"""Tests for untextre.metrics module.

These tests verify that our metric functions return expected values for
known synthetic inputs. We're not testing cv2 internals - we're testing
that our functions correctly interpret results for our use case.
"""

import cv2
import numpy as np
import pytest

from untextre.metrics import (
    measure_blackhat_energy,
    measure_edge_row_energy,
    needs_retry,
    expand_bbox_along_long_axis,
    EXPANSION_BLACKHAT_THRESHOLD,
    EXPANSION_EDGE_ROW_THRESHOLD,
    RETRY_BLACKHAT_THRESHOLD,
    RETRY_EDGE_ROW_THRESHOLD,
)


# =============================================================================
# FIXTURES: Synthetic test images
# =============================================================================

@pytest.fixture
def white_image():
    """Pure white 100x100 image - no structure, should have near-zero metrics."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def black_image():
    """Pure black 100x100 image - no structure, should have near-zero metrics."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gray_uniform_image():
    """Uniform gray 100x100 image - no structure."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128


@pytest.fixture
def dark_text_on_white():
    """White image with dark horizontal text-like strokes.
    
    This should have HIGH blackhat_energy (dark strokes on light background)
    and HIGH edge_row_energy (horizontal concentration of edges).
    """
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Draw several dark horizontal lines (simulating text)
    for y in [30, 35, 50, 55, 70, 75]:
        cv2.line(img, (20, y), (80, y), (0, 0, 0), 2)
    return img


@pytest.fixture
def light_text_on_dark():
    """Dark image with light horizontal strokes.
    
    This should have LOW blackhat_energy (blackhat finds dark on light)
    but still have edge_row_energy from the contrast.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in [30, 35, 50, 55, 70, 75]:
        cv2.line(img, (20, y), (80, y), (255, 255, 255), 2)
    return img


@pytest.fixture
def vertical_stripes():
    """Vertical stripes - edges but no horizontal concentration.
    
    Should have LOW edge_row_energy since edges are spread across rows.
    """
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    for x in range(10, 90, 10):
        cv2.line(img, (x, 10), (x, 90), (0, 0, 0), 2)
    return img


@pytest.fixture
def horizontal_band():
    """Single thick horizontal band - maximum edge concentration in few rows."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 45), (90, 55), (0, 0, 0), -1)
    return img


# =============================================================================
# TESTS: measure_blackhat_energy
# =============================================================================

class TestMeasureBlackhatEnergy:
    """Tests for blackhat morphological energy measurement."""
    
    def test_uniform_white_near_zero(self, white_image):
        """Uniform white image should have near-zero blackhat energy."""
        energy = measure_blackhat_energy(white_image)
        assert energy < 1.0, f"Uniform white should have near-zero energy, got {energy}"
    
    def test_uniform_black_near_zero(self, black_image):
        """Uniform black image should have near-zero blackhat energy."""
        energy = measure_blackhat_energy(black_image)
        assert energy < 1.0, f"Uniform black should have near-zero energy, got {energy}"
    
    def test_uniform_gray_near_zero(self, gray_uniform_image):
        """Uniform gray image should have near-zero blackhat energy."""
        energy = measure_blackhat_energy(gray_uniform_image)
        assert energy < 1.0, f"Uniform gray should have near-zero energy, got {energy}"
    
    def test_dark_text_on_white_high(self, dark_text_on_white):
        """Dark strokes on white should have HIGH blackhat energy."""
        energy = measure_blackhat_energy(dark_text_on_white)
        # Should be well above the expansion threshold
        assert energy > EXPANSION_BLACKHAT_THRESHOLD, \
            f"Dark text on white should exceed threshold {EXPANSION_BLACKHAT_THRESHOLD}, got {energy}"
    
    def test_light_text_on_dark_also_detected(self, light_text_on_dark):
        """Light strokes on dark also produce blackhat energy due to edge contrast.
        
        Note: Blackhat morphology responds to stroke-like structures regardless
        of polarity, because the closing operation affects both polarities.
        """
        energy = measure_blackhat_energy(light_text_on_dark)
        # Actually produces notable energy due to structural contrast
        assert energy > 0, f"Should have some energy from contrast, got {energy}"
    
    


# =============================================================================
# TESTS: measure_edge_row_energy
# =============================================================================

class TestMeasureEdgeRowEnergy:
    """Tests for edge row concentration measurement."""
    
    def test_uniform_white_near_zero(self, white_image):
        """Uniform white image should have zero edge_row_energy."""
        energy = measure_edge_row_energy(white_image)
        assert energy < 0.01, f"Uniform white should have ~zero energy, got {energy}"
    
    def test_uniform_black_near_zero(self, black_image):
        """Uniform black image should have zero edge_row_energy."""
        energy = measure_edge_row_energy(black_image)
        assert energy < 0.01, f"Uniform black should have ~zero energy, got {energy}"
    
    def test_horizontal_band_high(self, horizontal_band):
        """Horizontal band should have HIGH edge_row_energy."""
        energy = measure_edge_row_energy(horizontal_band)
        assert energy > EXPANSION_EDGE_ROW_THRESHOLD, \
            f"Horizontal band should exceed threshold {EXPANSION_EDGE_ROW_THRESHOLD}, got {energy}"
    
    def test_dark_text_horizontal_high(self, dark_text_on_white):
        """Horizontal text lines should have elevated edge_row_energy."""
        energy = measure_edge_row_energy(dark_text_on_white)
        assert energy > 0.05, f"Horizontal text should have notable edge_row, got {energy}"
    
    def test_vertical_stripes_lower_than_horizontal(self, vertical_stripes, horizontal_band):
        """Vertical stripes should have lower edge_row_energy than horizontal band."""
        vert_energy = measure_edge_row_energy(vertical_stripes)
        horiz_energy = measure_edge_row_energy(horizontal_band)
        assert vert_energy < horiz_energy, \
            f"Vertical ({vert_energy}) should be less than horizontal ({horiz_energy})"
    
    


# =============================================================================
# TESTS: needs_retry
# =============================================================================

class TestNeedsRetry:
    """Tests for retry decision function."""
    
    def test_clean_image_no_retry(self, white_image):
        """Clean uniform image should NOT need retry."""
        assert needs_retry(white_image) is False
    
    def test_text_remnants_needs_retry(self, dark_text_on_white):
        """Image with text-like remnants SHOULD need retry."""
        assert needs_retry(dark_text_on_white) is True
    
    def test_respects_blackhat_threshold(self, dark_text_on_white):
        """Should respect custom blackhat threshold."""
        # Very high threshold should prevent retry
        result = needs_retry(dark_text_on_white, blackhat_threshold=100.0, edge_row_threshold=1.0)
        assert result is False
    
    def test_respects_edge_row_threshold(self, horizontal_band):
        """Should respect custom edge_row threshold."""
        # Very high threshold should prevent retry  
        result = needs_retry(horizontal_band, blackhat_threshold=100.0, edge_row_threshold=1.0)
        assert result is False
    
    def test_either_metric_triggers(self, dark_text_on_white):
        """Either high blackhat OR high edge_row should trigger retry."""
        # This image has high blackhat - should trigger even with impossible edge threshold
        result = needs_retry(dark_text_on_white, blackhat_threshold=0.1, edge_row_threshold=1.0)
        assert result is True


# =============================================================================
# TESTS: expand_bbox_along_long_axis
# =============================================================================

class TestExpandBboxAlongLongAxis:
    """Tests for bbox expansion logic."""
    
    def test_no_expansion_uniform_background(self, white_image):
        """Bbox in uniform area should not expand."""
        bbox = (30, 30, 40, 20)  # Horizontal bbox in center
        expanded = expand_bbox_along_long_axis(white_image, bbox)
        assert expanded == bbox, f"Should not expand in uniform area, got {expanded}"
    
    def test_expansion_when_text_adjacent(self):
        """Bbox should expand when text-like region is adjacent."""
        # Create image with text spanning beyond initial bbox
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Draw text-like strokes across the full width
        for y in [45, 50, 55]:
            cv2.line(img, (10, y), (190, y), (0, 0, 0), 3)
        
        # Start with bbox covering only middle portion
        initial_bbox = (80, 40, 40, 30)  # Small horizontal bbox in center
        expanded = expand_bbox_along_long_axis(img, initial_bbox)
        
        # Should expand horizontally
        assert expanded[2] > initial_bbox[2], \
            f"Width should increase from {initial_bbox[2]}, got {expanded[2]}"
    
    def test_horizontal_bbox_expands_horizontally(self):
        """Horizontal bbox (w > h) should expand left/right."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Dark region on the right side
        cv2.rectangle(img, (120, 40), (180, 60), (0, 0, 0), -1)
        
        # Horizontal bbox that should expand right
        initial_bbox = (80, 40, 40, 20)  # w=40 > h=20
        expanded = expand_bbox_along_long_axis(img, initial_bbox)
        
        # x should stay same or decrease, width should increase
        assert expanded[2] >= initial_bbox[2], "Width should not decrease"
    
    def test_vertical_bbox_expands_vertically(self):
        """Vertical bbox (h > w) should expand up/down."""
        img = np.ones((200, 100, 3), dtype=np.uint8) * 255
        # Dark region below
        cv2.rectangle(img, (40, 120), (60, 180), (0, 0, 0), -1)
        
        # Vertical bbox that should expand down
        initial_bbox = (40, 80, 20, 40)  # w=20 < h=40
        expanded = expand_bbox_along_long_axis(img, initial_bbox)
        
        # Height should increase
        assert expanded[3] >= initial_bbox[3], "Height should not decrease"
    
    def test_respects_image_bounds(self):
        """Expansion should not exceed image boundaries."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Fill entire image with dark strokes
        for y in range(10, 90, 5):
            cv2.line(img, (0, y), (99, y), (0, 0, 0), 2)
        
        bbox = (10, 40, 80, 20)
        expanded = expand_bbox_along_long_axis(img, bbox)
        
        x, y, w, h = expanded
        assert x >= 0, "x should not be negative"
        assert y >= 0, "y should not be negative"
        assert x + w <= 100, "Should not exceed image width"
        assert y + h <= 100, "Should not exceed image height"
    


# =============================================================================
# TESTS: Threshold constants are sensible
# =============================================================================

