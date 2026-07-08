"""Tests for untextre.pipeline helpers."""

import cv2
import numpy as np
import pytest

import untextre.consensus as consensus_mod
import untextre.metrics as metrics_mod
import untextre.pipeline as pipeline_mod
import untextre.preprocessor as preprocessor_mod
from untextre.pipeline import (
    _apply_color_enhancement,
    _generate_masks_and_inpaint,
    _translate_rotated_bbox_to_original,
    mask_mode_options,
    process_image_array,
    process_single_image,
)


def test_mask_mode_options_match_user_facing_modes():
    assert mask_mode_options("regional") == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": True,
    }
    assert mask_mode_options("local-shape") == {
        "expand_bboxes": False,
        "use_grabcut": True,
        "use_grabcut_expand": False,
    }
    assert mask_mode_options("local-color") == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": False,
    }
    assert mask_mode_options("budgeted-regional") == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": False,
        "use_budgeted_expand": True,
    }


def test_mask_mode_options_reject_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mask mode"):
        mask_mode_options("legacy")


def test_translate_rotated_bbox_to_original_uses_hand_calculated_inverse():
    """A bbox from a 90-degree-clockwise image maps back to original coordinates."""
    # Original image is 80w x 40h. Rotated bbox covers original x=[7,13), y=[25,35).
    assert _translate_rotated_bbox_to_original(
        rotated_bbox=(5, 7, 10, 6),
        original_shape=(40, 80),
    ) == (7, 25, 6, 10)


class TestApplyColorEnhancement:
    """CLI color enhancement does what it should and rejects bad input."""

    def test_invalid_hex_no_hash_raises(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid hex color format"):
            _apply_color_enhancement(image, "FFFFFF")

    def test_invalid_hex_wrong_length_raises(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid hex color format"):
            _apply_color_enhancement(image, "#FFF")

    def test_invalid_hex_non_hex_raises(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid hex color format"):
            _apply_color_enhancement(image, "#GGGGGG")

    def test_valid_hex_same_shape_returned(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        result = _apply_color_enhancement(image, "#000000", sensitivity=3)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_valid_hex_target_pixels_set_to_black(self):
        """Pixels in the target color range are zeroed; others unchanged."""
        image = np.ones((10, 10, 3), dtype=np.uint8) * 255  # All white
        result = _apply_color_enhancement(image, "#FFFFFF", sensitivity=5)
        # All pixels were in range [250,250,250]-[255,255,255], so all become black
        assert np.all(result == 0)

    def test_does_not_modify_original(self):
        image = np.ones((10, 10, 3), dtype=np.uint8) * 128
        original_sum = image.sum()
        _apply_color_enhancement(image, "#808080", sensitivity=3)
        assert image.sum() == original_sum


class TestProcessSingleImageSmoke:
    """Lightweight smoke test using forced bbox + TELEA (no model loading).

    We don't test the full detection failover chain here — that's
    integration testing.  We exercise forced_bbox + telea to verify
    the function runs end-to-end and produces the expected output files.
    """

    @pytest.fixture
    def synthetic_image(self, tmp_path):
        """Create a 200×200 white image with black text, saved to disk."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "HELLO", (30, 120), font, 1.5, (0, 0, 0), 3)
        img_path = tmp_path / "synthetic.png"
        cv2.imwrite(str(img_path), image)
        return img_path

    def test_forced_bbox_telea_produces_output(self, synthetic_image, tmp_path):
        """process_single_image with forced_bbox + telea should create output file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        timings = process_single_image(
            image_path=synthetic_image,
            output_dir=output_dir,
            method="telea",
            forced_bbox=(25, 80, 150, 60),
            expand_bboxes=False,
            auto_retry=False,
        )

        assert timings is not None
        assert timings["total_time"] > 0
        assert timings["consensus_boxes_count"] == 1

        # Output image should exist
        expected_output = output_dir / "synthetic_clean.png"
        assert expected_output.exists(), f"Expected output at {expected_output}"

    def test_forced_bbox_with_keep_masks(self, synthetic_image, tmp_path):
        """keep_masks=True should save a mask PNG alongside the result."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        process_single_image(
            image_path=synthetic_image,
            output_dir=output_dir,
            method="telea",
            forced_bbox=(25, 80, 150, 60),
            keep_masks=True,
            expand_bboxes=False,
            auto_retry=False,
        )

        mask_output = output_dir / "synthetic_mask.png"
        assert mask_output.exists(), f"Expected mask at {mask_output}"


class TestProcessImageArray:
    def test_forced_bbox_telea_returns_result_without_file_io(self, monkeypatch):
        image = np.ones((40, 80, 3), dtype=np.uint8) * 200
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[10:20, 15:35] = 255
        cleaned = image.copy()
        cleaned[mask > 0] = 128

        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)

        def fake_generate(img, boxes, _g, _method, _target, **_kw):
            assert boxes == [(12, 8, 24, 16)]
            return mask.copy(), cleaned.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)

        result = process_image_array(
            image,
            image_name="memory.png",
            method="telea",
            forced_bbox=(12, 8, 24, 16),
            expand_bboxes=False,
            auto_retry=False,
        )

        assert result.timings["image_name"] == "memory.png"
        assert result.timings["consensus_boxes_count"] == 1
        np.testing.assert_array_equal(result.mask, mask)
        np.testing.assert_array_equal(result.image, cleaned)
        np.testing.assert_array_equal(image, np.ones((40, 80, 3), dtype=np.uint8) * 200)

    def test_budgeted_expand_option_reaches_mask_generation(self, monkeypatch):
        image = np.ones((40, 80, 3), dtype=np.uint8) * 200
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)

        captured = {}

        def fake_generate(img, boxes, _g, _method, _target, **kwargs):
            captured["use_budgeted_expand"] = kwargs["use_budgeted_expand"]
            return np.zeros(img.shape[:2], dtype=np.uint8), img.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)

        process_image_array(
            image,
            method="telea",
            forced_bbox=(12, 8, 24, 16),
            auto_retry=False,
            use_budgeted_expand=True,
        )

        assert captured["use_budgeted_expand"] is True


class TestProcessSingleImageFailovers:
    def test_target_color_success_short_circuits_detection(self, monkeypatch, tmp_path):
        image = np.ones((40, 80, 3), dtype=np.uint8) * 255
        image_path = tmp_path / "img.png"
        cv2.imwrite(str(image_path), image)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("Consensus detection should not run when target-color succeeds")

        monkeypatch.setattr(consensus_mod, "run_consensus_detection", fail_if_called)
        monkeypatch.setattr(
            pipeline_mod,
            "_try_color_enhanced_detection",
            lambda *_args, **_kwargs: [(5, 8, 12, 10)],
        )
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)
        monkeypatch.setattr(
            pipeline_mod,
            "_generate_masks_and_inpaint",
            lambda img, _boxes, _g, _method, _target, **_kw: (
                np.zeros(img.shape[:2], dtype=np.uint8),
                img.copy(),
            ),
        )

        timings = process_single_image(
            image_path=image_path,
            output_dir=out_dir,
            target_color=(128, 128, 128),
            method="telea",
            expand_bboxes=False,
            auto_retry=False,
        )

        assert timings is not None
        assert timings["failover_type"] == "target_color"
        assert timings["consensus_boxes_count"] == 1
        assert timings["total_bbox_area"] == 120

    def test_rotation_failover_translates_bboxes_back(self, monkeypatch, tmp_path):
        image = np.ones((40, 80, 3), dtype=np.uint8) * 255
        image_path = tmp_path / "img.png"
        cv2.imwrite(str(image_path), image)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())

        calls = {"n": 0}

        def fake_consensus(_img, _threshold):
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            return [(5, 7, 10, 6)]  # (x_rot, y_rot, w_rot, h_rot)

        monkeypatch.setattr(consensus_mod, "run_consensus_detection", fake_consensus)
        monkeypatch.setattr(
            pipeline_mod,
            "_try_color_enhanced_detection",
            lambda *_args, **_kwargs: [],
        )

        captured = {}

        def fake_generate(img, boxes, _g, _method, _target, **_kw):
            captured["boxes"] = boxes
            return np.zeros(img.shape[:2], dtype=np.uint8), img.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)

        timings = process_single_image(
            image_path=image_path,
            output_dir=out_dir,
            method="telea",
            expand_bboxes=False,
            auto_retry=False,
        )

        assert timings is not None
        assert calls["n"] == 2
        assert timings["failover_type"] == "rotation"
        assert captured["boxes"] == [(7, 25, 6, 10)]
        assert timings["total_bbox_area"] == 60

    def test_all_failovers_exhausted_returns_skipped(self, monkeypatch, tmp_path):
        image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        image_path = tmp_path / "img.png"
        cv2.imwrite(str(image_path), image)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *_args, **_kwargs: [])

        color_attempts = []

        def fake_try_color(_image, _threshold, target_hex, sensitivity=3):
            color_attempts.append((target_hex, sensitivity))
            return []

        monkeypatch.setattr(pipeline_mod, "_try_color_enhanced_detection", fake_try_color)

        timings = process_single_image(
            image_path=image_path,
            output_dir=out_dir,
            method="telea",
            expand_bboxes=False,
            auto_retry=False,
        )

        assert timings is not None
        assert timings["skipped"] is True
        assert timings["mask_found"] is False
        assert timings["consensus_boxes_count"] == 0
        assert color_attempts == [("#808080", 3), ("#FFFFFF", 3)]

    def test_generic_color_failovers_use_configured_sensitivity(self, monkeypatch):
        image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *_args, **_kwargs: [])

        color_attempts = []

        def fake_try_color(_image, _threshold, target_hex, sensitivity=3):
            color_attempts.append((target_hex, sensitivity))
            return []

        monkeypatch.setattr(pipeline_mod, "_try_color_enhanced_detection", fake_try_color)

        result = process_image_array(
            image,
            image_name="memory.png",
            method="telea",
            expand_bboxes=False,
            auto_retry=False,
            color_sensitivity=8,
        )

        assert result.timings["skipped"] is True
        assert color_attempts == [("#808080", 8), ("#FFFFFF", 8)]


class TestGenerateMasksAndInpaint:
    def test_coverage_above_default_limit_skips_inpaint(self, monkeypatch):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:7, :] = 255

        import untextre.find_text_colors as colors_mod
        import untextre.inpaint as inpaint_mod

        monkeypatch.setattr(
            colors_mod,
            "find_mask_by_spatial_tf_idf",
            lambda *_args, **_kwargs: mask,
        )

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("Coverage guard should skip inpainting")

        monkeypatch.setattr(inpaint_mod, "inpaint_image", fail_if_called)

        combined_mask, result = _generate_masks_and_inpaint(
            image,
            [(0, 0, 100, 100)],
            g_value=4,
            method="telea",
            target_color=None,
        )

        np.testing.assert_array_equal(combined_mask, mask)
        np.testing.assert_array_equal(result, image)

    def test_budgeted_expand_requests_cluster_data_and_uses_budgeted_path(self, monkeypatch):
        image = np.ones((20, 30, 3), dtype=np.uint8) * 180
        region_mask = np.zeros((6, 10), dtype=np.uint8)
        region_mask[2, 4:6] = 255
        cluster_data = {
            "centers": np.array([[0.0, 0.0, 0.0], [180.0, 180.0, 180.0]], dtype=np.float32),
            "top_id": 0,
            "bot_id": 1,
            "color_radius": 1.0,
            "bg_radius": 1.0,
        }

        import untextre.find_text_colors as colors_mod
        import untextre.inpaint as inpaint_mod

        calls = {}

        def fake_find(*_args, **kwargs):
            calls["return_cluster_data"] = kwargs["return_cluster_data"]
            calls["use_grabcut"] = kwargs["use_grabcut"]
            return region_mask, cluster_data

        def fake_budgeted(img, bbox, confirmed, centers, top_id, bot_id, color_radius, bg_radius, **_kwargs):
            calls["budgeted"] = {
                "bbox": bbox,
                "confirmed_pixels": int(np.sum(confirmed == 255)),
                "top_id": top_id,
                "bot_id": bot_id,
            }
            expanded = confirmed.copy()
            expanded[12, 20] = 255
            return expanded

        def fail_color_guided(*_args, **_kwargs):
            raise AssertionError("budgeted-regional should not call color_guided_expand")

        monkeypatch.setattr(colors_mod, "find_mask_by_spatial_tf_idf", fake_find)
        monkeypatch.setattr(colors_mod, "geometry_budgeted_expand", fake_budgeted)
        monkeypatch.setattr(colors_mod, "color_guided_expand", fail_color_guided)
        monkeypatch.setattr(inpaint_mod, "inpaint_image", lambda img, _mask, bbox=None, method="telea": img.copy())

        combined_mask, _result = _generate_masks_and_inpaint(
            image,
            [(10, 10, 10, 6)],
            g_value=4,
            method="telea",
            target_color=None,
            use_budgeted_expand=True,
        )

        assert calls["return_cluster_data"] is True
        assert calls["use_grabcut"] is False
        assert calls["budgeted"] == {
            "bbox": (10, 10, 10, 6),
            "confirmed_pixels": 2,
            "top_id": 0,
            "bot_id": 1,
        }
        assert combined_mask[12, 20] == 255


class TestProcessSingleImageEdgeCases:
    """Edge cases for process_single_image()."""

    def _make_image(self, tmp_path, name="test.png", size=(100, 100)):
        """Create a synthetic image and return its path."""
        img = np.ones((*size, 3), dtype=np.uint8) * 200
        path = tmp_path / name
        cv2.imwrite(str(path), img)
        return path

    def test_preprocessing_failure_raises(self, monkeypatch, tmp_path):
        """preprocess returning None → ValueError."""
        img_path = self._make_image(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: np.zeros((50, 50, 3), dtype=np.uint8))
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: None)
        with pytest.raises(ValueError, match="preprocessing failed"):
            process_single_image(
                image_path=img_path, output_dir=out_dir, method="telea",
                expand_bboxes=False, auto_retry=False,
            )

    def test_forced_bbox_clipped_to_bounds(self, monkeypatch, tmp_path):
        """Oversized bbox gets clipped to image dimensions."""
        img = np.ones((50, 80, 3), dtype=np.uint8) * 200
        img_path = self._make_image(tmp_path, size=(50, 80))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)

        captured = {}

        def fake_generate(image, boxes, _g, _method, _target, **_kw):
            captured["boxes"] = boxes
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)

        # Forced bbox extends beyond 80×50 image
        timings = process_single_image(
            image_path=img_path, output_dir=out_dir, method="telea",
            forced_bbox=(70, 40, 30, 20),  # extends to (100, 60) vs image (80, 50)
            expand_bboxes=False, auto_retry=False,
        )
        assert timings is not None
        # The clipped bbox dimensions should be <= image dimensions
        box = captured["boxes"][0]
        x, y, w, h = box
        assert x + w <= 80, f"Clipped bbox exceeds image width: {box}"
        assert y + h <= 50, f"Clipped bbox exceeds image height: {box}"

    def test_maskfile_not_found_raises(self, monkeypatch, tmp_path):
        """Bad mask path → ValueError."""
        img_path = self._make_image(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        with pytest.raises(ValueError, match="Mask file not found"):
            process_single_image(
                image_path=img_path, output_dir=out_dir, method="telea",
                forced_bbox=(0, 0, 10, 10),
                maskfile=str(tmp_path / "nonexistent_mask.png"),
                expand_bboxes=False, auto_retry=False,
            )

    def test_maskfile_loads_and_converts(self, monkeypatch, tmp_path):
        """Valid 3-channel mask is loaded, converted to grayscale, and used for inpainting."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        img_path = self._make_image(tmp_path, size=(50, 50))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Create a 3-channel mask file
        mask_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        mask_path = tmp_path / "mask.png"
        cv2.imwrite(str(mask_path), mask_img)

        load_calls = {}

        def tracking_load(p):
            path_str = str(p)
            if "mask" in path_str:
                load_calls["mask_loaded"] = True
                return mask_img.copy()
            return img.copy()

        monkeypatch.setattr(pipeline_mod, "load_image", tracking_load)
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())

        import untextre.inpaint as inpaint_mod
        inpaint_calls = {}

        def fake_inpaint(image, mask, bbox=None, method="lama"):
            inpaint_calls["mask_shape"] = mask.shape
            inpaint_calls["method"] = method
            return image.copy()

        monkeypatch.setattr(inpaint_mod, "inpaint_image", fake_inpaint)

        timings = process_single_image(
            image_path=img_path, output_dir=out_dir, method="telea",
            forced_bbox=(0, 0, 10, 10),
            maskfile=str(mask_path),
            expand_bboxes=False, auto_retry=False,
        )
        assert timings is not None
        assert load_calls.get("mask_loaded") is True
        # Mask should have been converted from 3-channel to single-channel
        assert len(inpaint_calls["mask_shape"]) == 2
        assert inpaint_calls["method"] == "telea"

    def test_auto_retry_triggers_g8(self, monkeypatch, tmp_path):
        """Remnant detection triggers g=8 retry."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        img_path = self._make_image(tmp_path, size=(50, 50))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *a, **kw: [(5, 5, 10, 10)])
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: True)

        generate_calls = []

        def fake_generate(image, boxes, g, method, target, **_kw):
            generate_calls.append(g)
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)

        timings = process_single_image(
            image_path=img_path, output_dir=out_dir, method="telea",
            expand_bboxes=True, auto_retry=True,
        )
        assert timings is not None
        assert timings["retried_with_g8"] is True
        assert generate_calls == [4, 8]

    def test_no_retry_flag_skips_g8(self, monkeypatch, tmp_path):
        """auto_retry=False prevents g=8 retry even when remnants detected."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        img_path = self._make_image(tmp_path, size=(50, 50))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(pipeline_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *a, **kw: [(5, 5, 10, 10)])
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        # needs_retry would return True, but shouldn't be checked
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: True)

        generate_calls = []

        def fake_generate(image, boxes, g, method, target, **_kw):
            generate_calls.append(g)
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(pipeline_mod, "_generate_masks_and_inpaint", fake_generate)

        timings = process_single_image(
            image_path=img_path, output_dir=out_dir, method="telea",
            expand_bboxes=True, auto_retry=False,
        )
        assert timings is not None
        assert timings["retried_with_g8"] is False
        assert generate_calls == [4]

