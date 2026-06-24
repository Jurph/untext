"""Tests for bbox utility functions in utils module.

High diagnostic value tests for critical bbox manipulation functions:
- dilate_bbox with various dilation amounts
- pad_bbox_to_multiple for neural network compatibility  
- clamp_bbox_to_image for boundary conditions
- Edge cases: zero-size, negative, out-of-bounds

These functions are critical for neural network processing (LaMa requires
dimensions divisible by 4/8) and for ensuring masks don't exceed image bounds.
"""

import pytest
import numpy as np

from untextre.utils import (
    dilate_bbox,
    pad_bbox_to_multiple,
    clamp_bbox_to_image,
    calculate_bbox_superset,
    color_distance,
    dilate_by_pixels,
    get_image_files,
)


class TestDilateBbox:
    """Test bbox dilation with diagnostic scenarios."""
    
    def test_basic_dilation(self):
        """Test basic symmetric dilation."""
        bbox = (100, 100, 50, 50)
        dilated = dilate_bbox(bbox, dilation=10, image_shape=None)
        
        # Should expand by 10 pixels on all sides
        assert dilated == (90, 90, 70, 70), f"Expected (90,90,70,70), got {dilated}"
    
    def test_dilation_with_image_bounds(self):
        """Test that dilation respects image boundaries."""
        bbox = (10, 10, 50, 50)
        image_shape = (200, 200)
        
        dilated = dilate_bbox(bbox, dilation=20, image_shape=image_shape)
        
        # Should not go below 0
        assert dilated[0] >= 0, "X should not be negative"
        assert dilated[1] >= 0, "Y should not be negative"
        
        # Should not exceed image dimensions
        assert dilated[0] + dilated[2] <= image_shape[1], "Should not exceed image width"
        assert dilated[1] + dilated[3] <= image_shape[0], "Should not exceed image height"
    
    def test_dilation_at_image_edge(self):
        """Test dilation when bbox is at image edge."""
        bbox = (0, 0, 50, 50)  # Top-left corner
        image_shape = (200, 200)
        
        dilated = dilate_bbox(bbox, dilation=10, image_shape=image_shape)
        
        # Should clamp to 0
        assert dilated[0] == 0, "X should clamp to 0 at left edge"
        assert dilated[1] == 0, "Y should clamp to 0 at top edge"
        
        # Should still expand in positive directions
        assert dilated[2] > 50, "Width should expand"
        assert dilated[3] > 50, "Height should expand"
    
    def test_dilation_near_bottom_right(self):
        """Test dilation when bbox is near bottom-right corner."""
        bbox = (180, 180, 20, 20)  # Near bottom-right of 200x200 image
        image_shape = (200, 200)
        
        dilated = dilate_bbox(bbox, dilation=30, image_shape=image_shape)
        
        # Should not exceed image bounds
        assert dilated[0] + dilated[2] <= 200, "Should not exceed width"
        assert dilated[1] + dilated[3] <= 200, "Should not exceed height"
    
    def test_zero_dilation(self):
        """Test that zero dilation returns identical bbox."""
        bbox = (100, 100, 50, 50)
        dilated = dilate_bbox(bbox, dilation=0, image_shape=None)
        
        assert dilated == bbox, "Zero dilation should return identical bbox"
    
    def test_large_dilation_clamping(self):
        """Test that very large dilation is properly clamped."""
        bbox = (50, 50, 10, 10)
        image_shape = (100, 100)
        
        # Dilate by 1000 pixels (much larger than image)
        dilated = dilate_bbox(bbox, dilation=1000, image_shape=image_shape)
        
        # Should clamp to full image bounds
        assert dilated[0] == 0
        assert dilated[1] == 0
        assert dilated[2] == 100
        assert dilated[3] == 100


class TestPadBboxToMultiple:
    """Test bbox padding for neural network compatibility."""
    
    def test_already_divisible(self):
        """Test bbox already divisible by multiple returns unchanged."""
        bbox = (10, 10, 100, 100)  # 100x100, divisible by 4
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=None)
        
        assert padded == bbox, "Already divisible bbox should return unchanged"
    
    def test_pad_to_mod_4(self):
        """Test padding to mod-4 (critical for neural networks)."""
        bbox = (10, 10, 97, 98)  # Not divisible by 4
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=None)
        
        # Width should be padded to 100 (nearest multiple of 4)
        # Height should be padded to 100
        assert padded[2] % 4 == 0, f"Width {padded[2]} should be divisible by 4"
        assert padded[3] % 4 == 0, f"Height {padded[3]} should be divisible by 4"
        
        # Should be close to original size
        assert abs(padded[2] - 97) <= 3, "Width should not change by more than 3 pixels"
        assert abs(padded[3] - 98) <= 3, "Height should not change by more than 3 pixels"
    
    def test_pad_to_mod_8(self):
        """Test padding to mod-8 for more restrictive networks."""
        bbox = (10, 10, 95, 97)
        padded = pad_bbox_to_multiple(bbox, multiple=8, image_shape=None)
        
        assert padded[2] % 8 == 0, "Width should be divisible by 8"
        assert padded[3] % 8 == 0, "Height should be divisible by 8"
        
        # Should pad to 96 and 104 respectively
        assert padded[2] == 96, "Width should pad to 96"
        assert padded[3] == 104, "Height should pad to 104"
    
    def test_symmetric_padding(self):
        """Test that padding is applied symmetrically when possible."""
        bbox = (50, 50, 97, 97)  # Needs 3 pixels of padding for mod-4
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=(200, 200))
        
        # With 3 pixels needed, should add 1 left, 2 right (or 2 left, 1 right)
        # Resulting width should be 100
        assert padded[2] == 100, "Width should be padded to 100"
        assert padded[3] == 100, "Height should be padded to 100"
        
        # Position should shift by at most half the padding
        assert abs(padded[0] - bbox[0]) <= 2, "X shift should be minimal"
        assert abs(padded[1] - bbox[1]) <= 2, "Y shift should be minimal"
    
    def test_padding_at_image_boundary(self):
        """Test padding when bbox is at image boundary."""
        bbox = (0, 0, 97, 97)  # Top-left corner, needs padding
        image_shape = (200, 200)
        
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=image_shape)
        
        # Should not go below 0
        assert padded[0] == 0, "Should not pad left beyond boundary"
        assert padded[1] == 0, "Should not pad top beyond boundary"
        
        # Should still be divisible
        assert padded[2] % 4 == 0, "Width should be divisible by 4"
        assert padded[3] % 4 == 0, "Height should be divisible by 4"
    
    def test_padding_at_bottom_right(self):
        """Test padding when bbox is at bottom-right boundary."""
        bbox = (150, 150, 47, 47)  # Would exceed 200x200 after padding
        image_shape = (200, 200)
        
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=image_shape)
        
        # Should not exceed image bounds
        assert padded[0] + padded[2] <= 200, "Should not exceed image width"
        assert padded[1] + padded[3] <= 200, "Should not exceed image height"
        
        # Should still be divisible
        assert padded[2] % 4 == 0, "Width should be divisible by 4"
        assert padded[3] % 4 == 0, "Height should be divisible by 4"
    
    def test_minimum_dimensions(self):
        """Test that result maintains minimum dimensions."""
        bbox = (10, 10, 1, 1)  # Tiny bbox
        padded = pad_bbox_to_multiple(bbox, multiple=4, image_shape=(200, 200))
        
        # Should pad to at least 4x4
        assert padded[2] >= 4, "Width should be at least multiple"
        assert padded[3] >= 4, "Height should be at least multiple"
        
        # Should be divisible
        assert padded[2] % 4 == 0
        assert padded[3] % 4 == 0
    
    def test_invalid_multiple(self):
        """Test that invalid multiple raises error."""
        bbox = (10, 10, 50, 50)
        
        with pytest.raises(ValueError, match="Multiple must be positive"):
            pad_bbox_to_multiple(bbox, multiple=0)
        
        with pytest.raises(ValueError, match="Multiple must be positive"):
            pad_bbox_to_multiple(bbox, multiple=-4)


class TestClampBboxToImage:
    """Test bbox clamping to image boundaries."""
    
    def test_bbox_within_bounds(self):
        """Test that bbox already within bounds is unchanged."""
        bbox = (50, 50, 100, 100)
        image_shape = (300, 300)
        
        clamped = clamp_bbox_to_image(bbox, image_shape)
        assert clamped == bbox, "Bbox within bounds should be unchanged"
    
    def test_bbox_exceeds_right_edge(self):
        """Test clamping when bbox exceeds right edge."""
        bbox = (150, 50, 100, 50)  # Extends to x=250 in 200-wide image
        image_shape = (200, 200)
        
        clamped = clamp_bbox_to_image(bbox, image_shape)
        
        # Should clamp width
        assert clamped[0] + clamped[2] <= 200, "Should not exceed image width"
        assert clamped[2] < bbox[2], "Width should be reduced"
    
    def test_bbox_exceeds_bottom_edge(self):
        """Test clamping when bbox exceeds bottom edge."""
        bbox = (50, 150, 50, 100)  # Extends to y=250 in 200-tall image
        image_shape = (200, 200)
        
        clamped = clamp_bbox_to_image(bbox, image_shape)
        
        # Should clamp height
        assert clamped[1] + clamped[3] <= 200, "Should not exceed image height"
        assert clamped[3] < bbox[3], "Height should be reduced"
    
    def test_negative_coordinates(self):
        """Test clamping when bbox has negative coordinates."""
        bbox = (-10, -20, 100, 100)
        image_shape = (200, 200)
        
        clamped = clamp_bbox_to_image(bbox, image_shape)
        
        # Should clamp to 0
        assert clamped[0] == 0, "Negative x should clamp to 0"
        assert clamped[1] == 0, "Negative y should clamp to 0"
        
        # Dimensions should be adjusted
        assert clamped[2] <= 100, "Width should be adjusted for clamped position"
        assert clamped[3] <= 100, "Height should be adjusted for clamped position"
    
    def test_completely_outside_image(self):
        """Test bbox completely outside image bounds."""
        bbox = (300, 300, 50, 50)  # Completely outside 200x200 image
        image_shape = (200, 200)
        
        clamped = clamp_bbox_to_image(bbox, image_shape)
        
        # Should clamp x,y to valid range
        assert 0 <= clamped[0] < 200
        assert 0 <= clamped[1] < 200
        
        # Width and height should be minimal
        assert clamped[2] >= 0
        assert clamped[3] >= 0


class TestCalculateBboxSuperset:
    """Test superset bounding box calculation for multi-region selection."""
    
    def test_single_bbox(self):
        """Test superset of single bbox returns that bbox."""
        bboxes = [(100, 100, 50, 50)]
        superset = calculate_bbox_superset(bboxes)
        
        assert superset == (100, 100, 50, 50), "Single bbox superset should equal original"
    
    def test_two_overlapping_bboxes(self):
        """Test superset of two overlapping bboxes."""
        bboxes = [
            (100, 100, 50, 50),  # x:100-150, y:100-150
            (120, 120, 50, 50),  # x:120-170, y:120-170
        ]
        superset = calculate_bbox_superset(bboxes)
        
        # Should be x:100-170, y:100-170 = (100, 100, 70, 70)
        assert superset == (100, 100, 70, 70), f"Expected (100,100,70,70), got {superset}"
    
    def test_two_disjoint_bboxes(self):
        """Test superset of two non-overlapping bboxes."""
        bboxes = [
            (10, 10, 20, 20),    # x:10-30, y:10-30
            (100, 100, 30, 30),  # x:100-130, y:100-130
        ]
        superset = calculate_bbox_superset(bboxes)
        
        # Should be x:10-130, y:10-130 = (10, 10, 120, 120)
        assert superset == (10, 10, 120, 120), f"Expected (10,10,120,120), got {superset}"
    
    def test_three_bboxes_scattered(self):
        """Test superset of three scattered bboxes."""
        bboxes = [
            (0, 50, 10, 10),     # leftmost
            (100, 0, 10, 10),    # topmost
            (50, 100, 10, 10),   # bottommost
        ]
        superset = calculate_bbox_superset(bboxes)
        
        # min_x=0, min_y=0, max_x=110, max_y=110
        assert superset == (0, 0, 110, 110), f"Expected (0,0,110,110), got {superset}"
    
    def test_empty_list(self):
        """Test superset of empty list returns None."""
        superset = calculate_bbox_superset([])
        assert superset is None, "Empty list should return None"
    
    def test_superset_with_image_clamping(self):
        """Test superset is clamped to image bounds."""
        bboxes = [
            (0, 0, 50, 50),
            (150, 150, 100, 100),  # Would extend to 250x250
        ]
        image_shape = (200, 200)
        superset = calculate_bbox_superset(bboxes, image_shape=image_shape)
        
        # Should be clamped to image bounds
        assert superset[0] + superset[2] <= 200, "Should not exceed image width"
        assert superset[1] + superset[3] <= 200, "Should not exceed image height"
    
    def test_horizontal_row_of_bboxes(self):
        """Test superset of horizontally arranged bboxes (like words in a line)."""
        bboxes = [
            (10, 50, 30, 20),   # "Hello"
            (50, 50, 30, 20),   # "World"  
            (90, 50, 40, 20),   # "Test!"
        ]
        superset = calculate_bbox_superset(bboxes)
        
        # x:10-130, y:50-70 = (10, 50, 120, 20)
        assert superset == (10, 50, 120, 20), f"Expected (10,50,120,20), got {superset}"
    
    def test_vertical_column_of_bboxes(self):
        """Test superset of vertically arranged bboxes."""
        bboxes = [
            (50, 10, 30, 20),
            (50, 40, 30, 20),
            (50, 70, 30, 20),
        ]
        superset = calculate_bbox_superset(bboxes)
        
        # x:50-80, y:10-90 = (50, 10, 30, 80)
        assert superset == (50, 10, 30, 80), f"Expected (50,10,30,80), got {superset}"


class TestCanvasCoordinateScaling:
    """Test coordinate scaling between canvas display and original image coordinates.
    
    These tests verify the math used in streamlit_app.py for converting between
    canvas coordinates (display pixels) and original image coordinates.
    """
    
    def test_scale_factors_calculation(self):
        """Test that scale factors are correctly calculated."""
        orig_width, orig_height = 1920, 1080  # Original HD image
        display_width, display_height = 800, 450  # Scaled canvas
        
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        
        assert scale_x == 2.4, f"Expected scale_x=2.4, got {scale_x}"
        assert scale_y == 2.4, f"Expected scale_y=2.4, got {scale_y}"
    
    def test_canvas_to_image_conversion(self):
        """Test converting canvas coordinates to image coordinates."""
        # Setup: 1920x1080 image displayed at 800x450
        orig_width, orig_height = 1920, 1080
        display_width, display_height = 800, 450
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        
        # User draws at canvas position (100, 100)
        canvas_x, canvas_y = 100, 100
        
        # Convert to image coordinates
        image_x = int(round(canvas_x * scale_x))
        image_y = int(round(canvas_y * scale_y))
        
        assert image_x == 240, f"Expected image_x=240, got {image_x}"
        assert image_y == 240, f"Expected image_y=240, got {image_y}"
    
    def test_image_to_canvas_conversion(self):
        """Test converting image coordinates to canvas coordinates."""
        # Setup: 1920x1080 image displayed at 800x450
        orig_width, orig_height = 1920, 1080
        display_width, display_height = 800, 450
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        
        # BBox in image coordinates
        image_x, image_y, image_w, image_h = 480, 240, 960, 540
        
        # Convert to canvas coordinates
        canvas_x = image_x / scale_x
        canvas_y = image_y / scale_y
        canvas_w = image_w / scale_x
        canvas_h = image_h / scale_y
        
        assert canvas_x == 200, f"Expected canvas_x=200, got {canvas_x}"
        assert canvas_y == 100, f"Expected canvas_y=100, got {canvas_y}"
        assert canvas_w == 400, f"Expected canvas_w=400, got {canvas_w}"
        assert canvas_h == 225, f"Expected canvas_h=225, got {canvas_h}"
    
    def test_roundtrip_conversion(self):
        """Test that canvas->image->canvas roundtrip preserves coordinates."""
        orig_width, orig_height = 1920, 1080
        display_width, display_height = 800, 450
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        
        # Original canvas coordinates
        orig_canvas_x, orig_canvas_y = 350, 200
        
        # Convert to image coordinates
        image_x = int(round(orig_canvas_x * scale_x))
        image_y = int(round(orig_canvas_y * scale_y))
        
        # Convert back to canvas coordinates
        back_canvas_x = image_x / scale_x
        back_canvas_y = image_y / scale_y
        
        # Should be very close (within rounding error)
        assert abs(back_canvas_x - orig_canvas_x) < 1, "X roundtrip should preserve value"
        assert abs(back_canvas_y - orig_canvas_y) < 1, "Y roundtrip should preserve value"
    
    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved in display dimensions."""
        orig_width, orig_height = 1920, 1080
        orig_aspect = orig_width / orig_height
        
        # Simulating: container_width = 800, calculate height from aspect
        container_width = 800
        display_width = container_width
        display_height = int(round(display_width / orig_aspect))
        
        display_aspect = display_width / display_height
        
        # Aspect ratios should be very close
        assert abs(display_aspect - orig_aspect) < 0.01, \
            f"Aspect ratio not preserved: orig={orig_aspect}, display={display_aspect}"
    
    def test_coordinate_clamping(self):
        """Test that coordinates are properly clamped to image bounds."""
        orig_width, orig_height = 1920, 1080
        
        # User draws outside canvas bounds (negative or beyond image)
        test_cases = [
            (-50, 100, 0, 100),      # negative x -> clamp to 0
            (100, -50, 100, 0),      # negative y -> clamp to 0
            (2000, 100, 1919, 100),  # beyond width -> clamp to max
            (100, 1200, 100, 1079),  # beyond height -> clamp to max
        ]
        
        for raw_x, raw_y, expected_x, expected_y in test_cases:
            clamped_x = max(0, min(raw_x, orig_width - 1))
            clamped_y = max(0, min(raw_y, orig_height - 1))
            
            assert clamped_x == expected_x, f"X clamping failed: {raw_x} -> {clamped_x}, expected {expected_x}"
            assert clamped_y == expected_y, f"Y clamping failed: {raw_y} -> {clamped_y}, expected {expected_y}"
    
    def test_fabricjs_scale_handling(self):
        """Test handling Fabric.js scaleX/scaleY on resized rectangles."""
        # When user resizes a rectangle in Fabric.js, it changes scaleX/scaleY
        # not width/height. We must multiply to get actual dimensions.
        
        base_width, base_height = 100, 100  # Initial rectangle
        scale_x_val, scale_y_val = 1.5, 2.0  # User stretched it
        
        # Actual dimensions after scaling
        actual_width = base_width * scale_x_val
        actual_height = base_height * scale_y_val
        
        assert actual_width == 150, f"Expected width=150, got {actual_width}"
        assert actual_height == 200, f"Expected height=200, got {actual_height}"


# =========================================================================
# Tests for utility functions not previously covered
# =========================================================================


class TestColorDistance:
    """Tests for Euclidean BGR color distance."""

    def test_identical_colors_zero(self):
        assert color_distance((128, 128, 128), (128, 128, 128)) == 0.0

    def test_black_to_white(self):
        dist = color_distance((0, 0, 0), (255, 255, 255))
        expected = (255**2 * 3) ** 0.5  # sqrt(3 * 255^2) ≈ 441.67
        assert dist == pytest.approx(expected, abs=0.01)

    def test_single_channel_difference(self):
        dist = color_distance((100, 0, 0), (0, 0, 0))
        assert dist == pytest.approx(100.0, abs=0.01)

    def test_symmetric(self):
        d1 = color_distance((10, 20, 30), (40, 50, 60))
        d2 = color_distance((40, 50, 60), (10, 20, 30))
        assert d1 == pytest.approx(d2)

    def test_returns_float(self):
        assert isinstance(color_distance((0, 0, 0), (1, 1, 1)), float)


class TestDilateByPixels:
    """Tests for dilate_by_pixels (thin wrapper around dilate_bbox)."""

    def test_basic_dilation(self):
        image = np.ones((200, 200, 3), dtype=np.uint8)
        bbox = (50, 50, 40, 40)
        dilated = dilate_by_pixels(image, bbox, 10)
        # Should expand by 10 px on each side
        assert dilated == (40, 40, 60, 60)

    def test_zero_pixels_unchanged(self):
        image = np.ones((200, 200, 3), dtype=np.uint8)
        bbox = (50, 50, 40, 40)
        dilated = dilate_by_pixels(image, bbox, 0)
        assert dilated == bbox

    def test_clamped_to_image_bounds(self):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        bbox = (5, 5, 30, 30)
        dilated = dilate_by_pixels(image, bbox, 50)
        x, y, w, h = dilated
        assert x >= 0 and y >= 0
        assert x + w <= 100 and y + h <= 100


class TestGetImageFiles:
    """Tests for get_image_files (file/directory enumeration)."""

    def test_single_png_file(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG")  # Minimal bytes, just needs to exist
        result = get_image_files(img)
        assert result == [img]

    def test_single_jpg_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8")
        result = get_image_files(img)
        assert result == [img]

    def test_non_image_file_returns_empty(self, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        result = get_image_files(txt)
        assert result == []

    def test_directory_with_mixed_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"\x89PNG")
        (tmp_path / "b.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "c.txt").write_text("not an image")
        (tmp_path / "d.bmp").write_bytes(b"BM")

        result = get_image_files(tmp_path)
        names = {p.name for p in result}
        assert "a.png" in names
        assert "b.jpg" in names
        assert "d.bmp" in names
        assert "c.txt" not in names

    def test_empty_directory(self, tmp_path):
        result = get_image_files(tmp_path)
        assert result == []

    def test_case_insensitive_extensions(self, tmp_path):
        (tmp_path / "upper.PNG").write_bytes(b"\x89PNG")
        (tmp_path / "mixed.JpEg").write_bytes(b"\xff\xd8")
        result = get_image_files(tmp_path)
        names = {p.name for p in result}
        assert "upper.PNG" in names
        assert "mixed.JpEg" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
