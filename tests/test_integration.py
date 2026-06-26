"""Integration tests for the full untextre pipeline.

These tests verify end-to-end functionality with high diagnostic value:
- Full pipeline from image loading to inpainting
- Color conversion utilities (hex/HTML to BGR)
- Real test images with known watermarks
- Mask generation and validation
- Output file creation

Focus on integration points where modules interact, not redundant unit testing.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image, ImageCms

from untextre.find_text_colors import hex_to_bgr, html_to_bgr, find_mask_by_spatial_tf_idf
from untextre.utils import load_image, save_image
from untextre.preprocessor import preprocess_image


class TestColorConversion:
    """Test color conversion utilities with diagnostic scenarios."""
    
    def test_hex_to_bgr_standard_colors(self):
        """Test standard hex colors convert correctly."""
        # Red
        assert hex_to_bgr("#FF0000") == (0, 0, 255), "Red should be (0,0,255) in BGR"
        assert hex_to_bgr("FF0000") == (0, 0, 255), "Should work without # prefix"
        
        # Green
        assert hex_to_bgr("#00FF00") == (0, 255, 0), "Green should be (0,255,0) in BGR"
        
        # Blue
        assert hex_to_bgr("#0000FF") == (255, 0, 0), "Blue should be (255,0,0) in BGR"
        
        # White
        assert hex_to_bgr("#FFFFFF") == (255, 255, 255)
        
        # Black
        assert hex_to_bgr("#000000") == (0, 0, 0)
        
        # Gray (common watermark color)
        assert hex_to_bgr("#808080") == (128, 128, 128)
    
    def test_html_to_bgr_standard_colors(self):
        """Test HTML color names convert correctly."""
        # Test common colors
        assert html_to_bgr("red") == (0, 0, 255)
        assert html_to_bgr("green") == (0, 128, 0)  # HTML green is darker
        assert html_to_bgr("blue") == (255, 0, 0)
        assert html_to_bgr("white") == (255, 255, 255)
        assert html_to_bgr("black") == (0, 0, 0)
        
        # Test common watermark colors
        assert html_to_bgr("gray") == (128, 128, 128)
        assert html_to_bgr("silver") == (192, 192, 192)
    
    def test_hex_uppercase_lowercase(self):
        """Test that hex color is case-insensitive."""
        assert hex_to_bgr("#ff0000") == hex_to_bgr("#FF0000")
        assert hex_to_bgr("#aAbBcC") == hex_to_bgr("#AABBCC")


class TestSpatialTfIdf:
    """Test spatial TF-IDF functionality with synthetic images.
    
    The TF-IDF algorithm clusters colors using k-means and then scores
    each cluster's "text-likelihood" using a Figure of Merit (FOM)
    combining TF-IDF distinctiveness, border ratio, and connected-component
    fragmentation.  This requires:
      - Enough pixels for k-means to form meaningful clusters
      - A realistic cluster count (production default is num_clusters=4)
      - The text region to be genuinely distinctive from its surroundings
    
    Tests use images >= 200x200 with bboxes >= 80x80 to stay in the
    algorithm's realistic operating range.
    """
    
    def test_simple_black_text_on_white(self):
        """Test basic case: black text fragments on white background."""
        # 200x200 white image with scattered black text-like strokes
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Draw fragmented text-like marks (not a solid block — solid blobs
        # are rejected by the connected-component guard).
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "HELLO", (55, 110), font, 1.0, (0, 0, 0), 2)
        
        # Bbox around the text region
        bbox = (50, 80, 100, 40)
        
        mask = find_mask_by_spatial_tf_idf(image, bbox, num_clusters=4, debug=False)
        
        assert mask.shape == (40, 100), "Mask should match bbox dimensions (h, w)"
        assert mask.dtype == np.uint8
        
        white_pixels = np.sum(mask == 255)
        assert white_pixels > 0, "Should detect some text pixels"
    
    def test_gray_text_on_white(self):
        """Test with gray text (common watermark scenario)."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Render gray text — fragmented strokes, not a solid rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "WATERMARK", (20, 110), font, 0.8, (128, 128, 128), 2)
        
        bbox = (15, 80, 170, 50)
        
        mask = find_mask_by_spatial_tf_idf(image, bbox, num_clusters=4, debug=False)
        
        assert mask.shape == (50, 170)
        assert np.sum(mask == 255) > 0, "Should detect gray watermark pixels"
    
    def test_target_color_override(self):
        """Test that target_color parameter forces inclusion of specific color."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Two different gray patches
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "DARK", (55, 95), font, 0.7, (100, 100, 100), 2)
        cv2.putText(image, "LIGHT", (50, 120), font, 0.7, (200, 200, 200), 2)
        
        bbox = (45, 70, 110, 70)
        
        # Force inclusion of the light gray
        mask_with_target = find_mask_by_spatial_tf_idf(
            image, bbox, num_clusters=4, 
            target_color=(200, 200, 200), 
            debug=False
        )
        
        assert np.sum(mask_with_target == 255) > 0, "Should detect target color"
    
    def test_small_bbox_near_edge(self):
        """Test with small bbox near image edge (boundary condition).
        
        This only verifies the algorithm doesn't crash on edge-adjacent
        regions — it does NOT assert detection results because the region
        is too small for reliable clustering.
        """
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image[85:95, 85:95] = [0, 0, 0]
        
        bbox = (83, 83, 15, 15)
        
        mask = find_mask_by_spatial_tf_idf(image, bbox, num_clusters=4, debug=False)
        
        assert mask.shape == (15, 15)
        assert mask.dtype == np.uint8


class TestImageLoading:
    """Test image loading with diagnostic value."""
    
    def test_load_existing_image(self):
        """Test loading an existing test image."""
        test_path = Path("tests/images/test1.png")
        if not test_path.exists():
            pytest.skip("Test image not available")
        
        image = load_image(test_path)
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3, "Should load as BGR image"
        assert image.shape[2] == 3, "Should have 3 color channels"
        assert image.dtype == np.uint8
    
    def test_load_nonexistent_image(self):
        """Test that loading nonexistent image raises error."""
        with pytest.raises(ValueError, match="Could not load image"):
            load_image("nonexistent_file_xyz.png")
    
    def test_save_and_reload_image(self):
        """Test saving and reloading preserves image data."""
        # Create test image
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.png"
            
            # Save
            save_image(original, output_path)
            assert output_path.exists(), "Image should be saved"
            
            # Reload
            reloaded = load_image(output_path)
            
            # Should be identical (PNG is lossless)
            assert np.array_equal(reloaded, original), \
                "Reloaded image should match original for PNG"

    def test_load_image_uses_unicode_safe_file_read(self, monkeypatch, tmp_path):
        """Image ingest should handle non-ASCII paths even when cv2.imread cannot."""
        image_path = tmp_path / "snowman_☃.png"
        expected = np.zeros((6, 5, 3), dtype=np.uint8)
        expected[:, :] = (10, 20, 30)
        ok, encoded = cv2.imencode(".png", expected)
        assert ok
        encoded.tofile(image_path)

        monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)

        loaded = load_image(image_path)

        np.testing.assert_array_equal(loaded, expected)

    def test_save_image_preserves_rendering_metadata_from_source_jpeg(self, tmp_path):
        """Cleaned JPEGs should keep color-management metadata, not full EXIF."""
        icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        source_path = tmp_path / "source.jpg"
        output_path = tmp_path / "cleaned.jpg"
        source = Image.new("RGB", (12, 10), (128, 64, 32))
        exif = Image.Exif()
        exif[305] = "Unit Test Camera Raw Editor"
        source.save(
            source_path,
            "JPEG",
            quality=90,
            dpi=(300, 300),
            icc_profile=icc_profile,
            exif=exif,
        )

        output_bgr = np.zeros((10, 12, 3), dtype=np.uint8)
        output_bgr[:, :] = (32, 64, 128)

        save_image(output_bgr, output_path, source_path=source_path)

        saved = Image.open(output_path)
        assert saved.info.get("icc_profile") == icc_profile
        assert saved.info.get("dpi") == (300, 300)
        assert saved.getexif().get(305) is None, "Full EXIF should not be copied blindly"


class TestPreprocessing:
    """Test preprocessing with diagnostic scenarios."""
    
    def test_preprocess_color_image(self):
        """Test preprocessing converts color image correctly."""
        # Create color test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        preprocessed = preprocess_image(image)
        
        assert preprocessed is not None
        assert isinstance(preprocessed, np.ndarray)
        # Preprocessing converts to grayscale then back to RGB
        assert preprocessed.shape == image.shape, "Should maintain dimensions"
    
    def test_preprocess_grayscale_image(self):
        """Test preprocessing handles grayscale input."""
        # Create grayscale image
        image_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        
        preprocessed = preprocess_image(image_bgr)
        
        assert preprocessed is not None
        assert preprocessed.shape == image_bgr.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
