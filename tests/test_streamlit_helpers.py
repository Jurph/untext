"""Tests for pure helper functions extracted from streamlit_app.py.

These helpers have no Streamlit dependency and can be tested in isolation:
    - ``bbox_to_fabric_rect()``     – image bbox → Fabric.js initial_drawing
    - ``fabric_rect_to_bbox()``     – Fabric.js rect → image bbox (with clamping)
    - ``encode_result_for_download()`` – PIL image → (bytes, filename, MIME)
"""

import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

import streamlit_app
from streamlit_app import (
    MODE_DRAW_MANUALLY,
    MODE_LOCAL_COLOR,
    MODE_LOCAL_SHAPE,
    MODE_REGIONAL,
    _watermarks_dir_signature,
    bbox_to_fabric_rect,
    encode_result_array_for_download,
    encode_result_for_download,
    fabric_rect_to_bbox,
    load_original_image_from_bytes,
    load_watermark_templates_cached,
    make_detection_signature,
    make_result_placeholder,
    original_file_basename,
    resolve_placeholder_fill,
    make_image_state_id,
    resolve_mask_mode_options,
    resolve_active_image,
    run_watermark_cascade_cached,
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


def test_resolve_mask_mode_options_match_sidebar_choices():
    assert resolve_mask_mode_options(MODE_REGIONAL) == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": True,
    }
    assert resolve_mask_mode_options(MODE_LOCAL_SHAPE) == {
        "expand_bboxes": False,
        "use_grabcut": True,
        "use_grabcut_expand": False,
    }
    assert resolve_mask_mode_options(MODE_LOCAL_COLOR) == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": False,
    }


def test_load_original_image_from_bytes_converts_canvas_unsafe_modes():
    palette_image = Image.new("P", (3, 2))
    buf = io.BytesIO()
    palette_image.save(buf, format="PNG")

    result = load_original_image_from_bytes(buf.getvalue())

    assert result.mode == "RGB"
    assert result.size == (3, 2)


def test_encode_result_array_for_download_uses_existing_format_rules():
    result_array = np.zeros((2, 2, 3), dtype=np.uint8)
    result_array[:, :] = (255, 0, 0)

    buf_bytes, name, mime = encode_result_array_for_download(result_array, "source.webp")

    assert name == "source_clean.png"
    assert mime == "image/png"
    assert Image.open(io.BytesIO(buf_bytes)).size == (2, 2)


def test_original_file_basename_handles_browser_and_platform_paths():
    assert original_file_basename(r"C:\Users\Jurph\Pictures\source.jpg") == "source.jpg"
    assert original_file_basename("/home/jurph/source.png") == "source.png"
    assert original_file_basename(None) == "image.png"


def test_make_result_placeholder_preserves_aspect_without_full_resolution():
    placeholder = make_result_placeholder(6400, 4800)

    height, width, channels = placeholder.shape
    assert channels == 3
    assert width <= 320, "placeholder must not allocate at full source resolution"
    assert abs(height / width - 4800 / 6400) < 0.01
    assert placeholder.dtype == np.uint8


def test_make_result_placeholder_keeps_small_sources_exact():
    assert make_result_placeholder(8, 4).shape == (4, 8, 3)


def test_make_result_placeholder_paints_requested_solid_fill():
    placeholder = make_result_placeholder(8, 4, (38, 39, 48))

    assert np.all(placeholder.reshape(-1, 3) == (38, 39, 48))


def test_resolve_placeholder_fill_matches_theme_and_config():
    # Configured secondaryBackgroundColor wins over the theme default
    assert resolve_placeholder_fill("dark", "#1A2B3C") == (26, 43, 60)
    # Dark theme falls back to Streamlit's default dark secondary background
    assert resolve_placeholder_fill("dark", None) == (38, 39, 48)
    # Light theme (or unknown) falls back to the light secondary background
    assert resolve_placeholder_fill("light", None) == (240, 242, 246)
    assert resolve_placeholder_fill(None, "not-a-color") == (240, 242, 246)


def test_make_detection_signature_is_stable_for_identical_lists():
    detections = [
        {"bbox": (10, 20, 30, 40), "detectors": ["east", "yolo"], "confidence": 0.87},
        {"bbox": (50, 60, 70, 80), "detectors": ["easyocr"], "confidence": 0.42},
    ]

    assert make_detection_signature(detections) == make_detection_signature(
        [dict(det) for det in detections]
    )


def test_make_detection_signature_changes_when_detections_change():
    base = [{"bbox": (10, 20, 30, 40), "detectors": ["east", "yolo"], "confidence": 0.87}]
    moved = [{"bbox": (11, 20, 30, 40), "detectors": ["east", "yolo"], "confidence": 0.87}]
    reordered_pair = [
        {"bbox": (10, 20, 30, 40), "detectors": ["east"], "confidence": 0.5},
        {"bbox": (50, 60, 70, 80), "detectors": ["yolo"], "confidence": 0.5},
    ]

    assert make_detection_signature(base) != make_detection_signature(moved)
    # Index identity matters: checkboxes are keyed by list position, so a
    # reordered list must count as a different detection set.
    assert make_detection_signature(reordered_pair) != make_detection_signature(
        list(reversed(reordered_pair))
    )



# =========================================================================
# Watermark template caching — every Streamlit rerun must skip the SIFT
# feature-extraction cost when the templates directory and image are
# unchanged (see load_watermark_templates_cached / run_watermark_cascade_cached).
# =========================================================================


def _write_rgba_template(path):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = 200
    cv2.imwrite(str(path), rgba)


def test_watermarks_dir_signature_empty_for_missing_or_empty_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert _watermarks_dir_signature(missing) == ()
    assert _watermarks_dir_signature(tmp_path) == ()


def test_watermarks_dir_signature_changes_when_a_template_is_added(tmp_path):
    before = _watermarks_dir_signature(tmp_path)
    _write_rgba_template(tmp_path / "logo.png")
    after = _watermarks_dir_signature(tmp_path)

    assert before == ()
    assert len(after) == 1
    assert after[0][0] == "logo.png"


def test_load_watermark_templates_cached_skips_reload_for_unchanged_signature(tmp_path, monkeypatch):
    _write_rgba_template(tmp_path / "logo.png")
    signature = _watermarks_dir_signature(tmp_path)

    calls = {"n": 0}
    def counting_loader(path):
        calls["n"] += 1
        return [SimpleNamespace(name="logo.png")]

    monkeypatch.setattr(streamlit_app, "load_watermark_templates", counting_loader)

    first = load_watermark_templates_cached(str(tmp_path), signature)
    second = load_watermark_templates_cached(str(tmp_path), signature)

    assert calls["n"] == 1, "second call with an unchanged directory must hit the cache"
    assert len(first) == 1
    assert first is second, "cache_resource returns the same object, not a fresh reload"


def test_run_watermark_cascade_cached_skips_rematch_for_same_image_and_selection(tmp_path, monkeypatch):
    _write_rgba_template(tmp_path / "logo.png")
    signature = _watermarks_dir_signature(tmp_path)

    target = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", target)
    assert ok
    image_bytes = encoded.tobytes()

    calls = {"n": 0}
    monkeypatch.setattr(
        streamlit_app,
        "load_watermark_templates",
        lambda _path: [SimpleNamespace(name="logo.png")],
    )

    def counting_cascade(image, templates, **kwargs):
        calls["n"] += 1
        return ("matched", [template.name for template in templates])

    monkeypatch.setattr(streamlit_app, "try_watermark_cascade", counting_cascade)

    first = run_watermark_cascade_cached(image_bytes, ("logo.png",), str(tmp_path), signature)
    second = run_watermark_cascade_cached(image_bytes, ("logo.png",), str(tmp_path), signature)

    assert calls["n"] == 1, "second call with the same image + selection must hit the cache"
    assert first == second


def test_run_watermark_cascade_cached_skips_matching_when_no_templates_selected(tmp_path, monkeypatch):
    _write_rgba_template(tmp_path / "logo.png")
    signature = _watermarks_dir_signature(tmp_path)

    target = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", target)
    assert ok
    image_bytes = encoded.tobytes()

    calls = {"n": 0}
    monkeypatch.setattr(
        streamlit_app,
        "try_watermark_cascade",
        lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1),
    )

    result = run_watermark_cascade_cached(image_bytes, (), str(tmp_path), signature)

    assert result is None
    assert calls["n"] == 0, "an empty template selection must not run SIFT matching at all"



def test_manual_mode_uses_local_shape_mask_without_expansion():
    assert resolve_mask_mode_options(MODE_DRAW_MANUALLY) == {
        "expand_bboxes": False,
        "use_grabcut": True,
        "use_grabcut_expand": False,
    }


# =========================================================================
# bbox_to_fabric_rect
# =========================================================================


class TestBboxToFabricRect:
    """Verify image-coord bbox → Fabric.js initial_drawing conversion."""



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
