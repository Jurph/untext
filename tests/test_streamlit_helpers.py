"""Tests for pure helper functions extracted from streamlit_app.py.

These helpers have no Streamlit dependency and can be tested in isolation:
    - ``bbox_to_fabric_rect()``     – image bbox → Fabric.js initial_drawing
    - ``fabric_rect_to_bbox()``     – Fabric.js rect → image bbox (with clamping)
    - ``encode_result_for_download()`` – PIL image → (bytes, filename, MIME)
"""

import io

import pytest
from PIL import Image

from streamlit_app import (
    bbox_to_fabric_rect,
    fabric_rect_to_bbox,
    encode_result_for_download,
    make_image_state_id,
    resolve_active_image,
)


class FakeUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def test_make_image_state_id_includes_content_hash():
    same_name_a = make_image_state_id("photo.png", b"first")
    same_name_b = make_image_state_id("photo.png", b"second")

    assert same_name_a.startswith("photo_")
    assert same_name_b.startswith("photo_")
    assert same_name_a != same_name_b


def test_make_image_state_id_handles_missing_name():
    assert make_image_state_id(None, b"bytes").startswith("image_")


def test_make_image_state_id_none_without_bytes():
    assert make_image_state_id("photo.png", None) is None


def test_resolve_active_image_prefers_ingested_result():
    uploaded = FakeUploadedFile("upload.png", b"uploaded")

    image_bytes, image_name = resolve_active_image(b"ingested", "pass_1.png", uploaded)

    assert image_bytes == b"ingested"
    assert image_name == "pass_1.png"


def test_resolve_active_image_falls_back_to_upload():
    uploaded = FakeUploadedFile("upload.png", b"uploaded")

    image_bytes, image_name = resolve_active_image(None, None, uploaded)

    assert image_bytes == b"uploaded"
    assert image_name == "upload.png"


# =========================================================================
# bbox_to_fabric_rect
# =========================================================================


class TestBboxToFabricRect:
    """Verify image-coord bbox → Fabric.js initial_drawing conversion."""

    def test_returns_dict_with_objects(self):
        result = bbox_to_fabric_rect((100, 200, 50, 30), scale_x=2.0, scale_y=2.0)
        assert isinstance(result, dict)
        assert "objects" in result
        assert len(result["objects"]) == 1

    def test_rect_type(self):
        result = bbox_to_fabric_rect((0, 0, 10, 10), scale_x=1.0, scale_y=1.0)
        rect = result["objects"][0]
        assert rect["type"] == "rect"

    def test_identity_scale(self):
        """When scale is 1:1, canvas coords should equal image coords."""
        result = bbox_to_fabric_rect((50, 60, 100, 80), scale_x=1.0, scale_y=1.0)
        rect = result["objects"][0]
        assert rect["left"] == pytest.approx(50.0)
        assert rect["top"] == pytest.approx(60.0)
        assert rect["width"] == pytest.approx(100.0)
        assert rect["height"] == pytest.approx(80.0)

    def test_downscale(self):
        """Image is 2x canvas → canvas coords are halved."""
        result = bbox_to_fabric_rect((200, 100, 60, 40), scale_x=2.0, scale_y=2.0)
        rect = result["objects"][0]
        assert rect["left"] == pytest.approx(100.0)
        assert rect["top"] == pytest.approx(50.0)
        assert rect["width"] == pytest.approx(30.0)
        assert rect["height"] == pytest.approx(20.0)

    def test_asymmetric_scale(self):
        """Different X and Y scales."""
        result = bbox_to_fabric_rect((300, 200, 90, 60), scale_x=3.0, scale_y=2.0)
        rect = result["objects"][0]
        assert rect["left"] == pytest.approx(100.0)
        assert rect["top"] == pytest.approx(100.0)
        assert rect["width"] == pytest.approx(30.0)
        assert rect["height"] == pytest.approx(30.0)


# =========================================================================
# fabric_rect_to_bbox
# =========================================================================


class TestFabricRectToBbox:
    """Verify Fabric.js rect → image-coord bbox conversion with clamping."""

    def _make_rect(self, left=0, top=0, width=100, height=50,
                   scaleX=1.0, scaleY=1.0):
        """Build a minimal Fabric.js rect dict."""
        return {
            "type": "rect",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "scaleX": scaleX,
            "scaleY": scaleY,
        }

    def test_identity_scale(self):
        rect = self._make_rect(left=10, top=20, width=30, height=40)
        result = fabric_rect_to_bbox(rect, 1.0, 1.0, 200, 200)
        assert result == (10, 20, 30, 40)

    def test_upscale_to_image(self):
        """Canvas is half the image → coords double."""
        rect = self._make_rect(left=10, top=10, width=50, height=50)
        result = fabric_rect_to_bbox(rect, 2.0, 2.0, 400, 400)
        assert result == (20, 20, 100, 100)

    def test_fabric_scale_applied(self):
        """Fabric.js scaleX/scaleY stretches the rect before conversion."""
        rect = self._make_rect(left=0, top=0, width=50, height=50,
                               scaleX=2.0, scaleY=3.0)
        result = fabric_rect_to_bbox(rect, 1.0, 1.0, 500, 500)
        assert result == (0, 0, 100, 150)

    def test_clamped_to_image_bounds(self):
        """Rect extending past image edges gets clamped."""
        rect = self._make_rect(left=180, top=180, width=100, height=100)
        result = fabric_rect_to_bbox(rect, 1.0, 1.0, 200, 200)
        x, y, w, h = result
        assert x >= 0 and y >= 0
        assert x + w <= 200 and y + h <= 200

    def test_minimum_size_one(self):
        """Even a zero-area rect should produce w=1, h=1."""
        rect = self._make_rect(left=50, top=50, width=0, height=0)
        result = fabric_rect_to_bbox(rect, 1.0, 1.0, 200, 200)
        _, _, w, h = result
        assert w >= 1 and h >= 1

    def test_none_for_non_rect(self):
        result = fabric_rect_to_bbox({"type": "circle"}, 1.0, 1.0, 200, 200)
        assert result is None

    def test_none_for_empty_dict(self):
        assert fabric_rect_to_bbox({}, 1.0, 1.0, 200, 200) is None

    def test_none_for_none(self):
        assert fabric_rect_to_bbox(None, 1.0, 1.0, 200, 200) is None

    def test_roundtrip_with_bbox_to_fabric(self):
        """bbox → fabric → bbox should be idempotent (at 1:1 scale)."""
        original = (50, 60, 100, 80)
        fabric = bbox_to_fabric_rect(original, 1.0, 1.0)
        recovered = fabric_rect_to_bbox(
            fabric["objects"][0], 1.0, 1.0, 500, 500
        )
        assert recovered == original

    def test_roundtrip_with_scaling(self):
        """bbox → fabric → bbox should roundtrip through a 2x scale."""
        original = (100, 200, 60, 40)
        fabric = bbox_to_fabric_rect(original, 2.0, 2.0)
        recovered = fabric_rect_to_bbox(
            fabric["objects"][0], 2.0, 2.0, 500, 500
        )
        assert recovered == original


# =========================================================================
# encode_result_for_download
# =========================================================================


class TestEncodeResultForDownload:
    """Verify format selection, naming, and MIME types."""

    @pytest.fixture
    def small_image(self):
        """A 10×10 red image — tiny, encodes fast."""
        return Image.new("RGB", (10, 10), color=(255, 0, 0))

    def test_png_default(self, small_image):
        buf_bytes, name, mime = encode_result_for_download(small_image, "photo.png")
        assert name == "photo_clean.png"
        assert mime == "image/png"
        assert len(buf_bytes) > 0

    def test_jpeg_format(self, small_image):
        buf_bytes, name, mime = encode_result_for_download(small_image, "photo.jpg")
        assert name == "photo_clean.jpg"
        assert mime == "image/jpeg"

    def test_jpeg_extension(self, small_image):
        _, name, mime = encode_result_for_download(small_image, "photo.jpeg")
        assert name == "photo_clean.jpg"
        assert mime == "image/jpeg"

    def test_bmp_converted_to_png(self, small_image):
        _, name, mime = encode_result_for_download(small_image, "photo.bmp")
        assert name == "photo_clean.png"
        assert mime == "image/png"

    def test_tiff_format(self, small_image):
        _, name, mime = encode_result_for_download(small_image, "photo.tiff")
        assert name == "photo_clean.tiff"
        assert mime == "image/tiff"

    def test_tif_format(self, small_image):
        _, name, mime = encode_result_for_download(small_image, "photo.tif")
        assert name == "photo_clean.tiff"
        assert mime == "image/tiff"

    def test_webp_becomes_png(self, small_image):
        """WEBP input is intentionally saved as PNG."""
        _, name, mime = encode_result_for_download(small_image, "photo.webp")
        assert name == "photo_clean.png"
        assert mime == "image/png"

    def test_unknown_extension_becomes_png(self, small_image):
        _, name, mime = encode_result_for_download(small_image, "photo.xyz")
        assert name == "photo_clean.png"
        assert mime == "image/png"

    def test_output_is_valid_image(self, small_image):
        """The encoded bytes should be a valid image we can re-open."""
        buf_bytes, _, _ = encode_result_for_download(small_image, "photo.png")
        reopened = Image.open(io.BytesIO(buf_bytes))
        assert reopened.size == (10, 10)

    def test_preserves_stem_with_dots(self, small_image):
        """Filenames like 'my.photo.v2.png' should keep the full stem."""
        _, name, _ = encode_result_for_download(small_image, "my.photo.v2.png")
        assert name == "my.photo.v2_clean.png"
