"""Tests for untextre.cli module.

Covers:
    - ``parse_args()``  – argument parsing defaults & validation
    - ``_save_clean_timing_report()``  – timing report formatting (including
      template-match and mixed-key entries so we don't regress KeyError)
    - ``_apply_color_enhancement()``  – valid/invalid hex, does/doesn't mutate
    - ``load_watermark_templates()``  – path filtering, RGBA-only loading
    - ``try_watermark_cascade()``  – empty/multi-template cascade behaviour
    - ``process_single_image()``  – smoke test (telea, forced bbox) and
      failover-chain orchestration (target-color, rotation, exhaustion)
    - ``process_with_known_mask()``  – match/no-match branching via stubs
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import untextre
import untextre.cli as cli_mod
import untextre.consensus as consensus_mod
import untextre.metrics as metrics_mod
import untextre.preprocessor as preprocessor_mod
from untextre.cli import (
    _apply_color_enhancement,
    _save_clean_timing_report,
    find_known_mask_in_image,
    load_watermark_templates,
    main,
    parse_args,
    process_with_known_mask,
    process_single_image,
    try_watermark_cascade,
)


# =========================================================================
# parse_args
# =========================================================================

class TestParseArgs:
    """Verify argument defaults and basic validation."""

    def test_required_args_present(self, monkeypatch):
        """Minimum viable invocation: -i and -o."""
        monkeypatch.setattr(sys, "argv", ["prog", "-i", "input.png", "-o", "out/"])
        args = parse_args()
        assert args.input == "input.png"
        assert args.output == "out/"

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-i", "x.png", "-o", "o/"])
        args = parse_args()
        assert args.paint == "lama"
        assert args.device == "cuda"
        assert args.granularity is None
        assert args.no_expand is False
        assert args.no_retry is False
        assert args.keep_masks is False
        assert args.timing is False
        assert args.verbose is False
        assert args.color is None
        assert args.maskfile is None
        assert args.force_bbox is None
        assert args.known_mask is None

    def test_confidence_threshold_default(self, monkeypatch):
        from untextre.utils import CLI_DEFAULT_CONFIDENCE

        monkeypatch.setattr(sys, "argv", ["prog", "-i", "x.png", "-o", "o/"])
        args = parse_args()
        assert args.confidence_threshold == CLI_DEFAULT_CONFIDENCE

    def test_paint_choices(self, monkeypatch):
        for method in ("lama", "telea"):
            monkeypatch.setattr(
                sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "-p", method]
            )
            args = parse_args()
            assert args.paint == method

    def test_invalid_paint_choice_rejected(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "-p", "magic"]
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_granularity_parsed_as_int(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "-g", "8"]
        )
        args = parse_args()
        assert args.granularity == 8

    def test_force_bbox_is_raw_string(self, monkeypatch):
        """force-bbox is parsed as a raw string by argparse; main() splits it."""
        monkeypatch.setattr(
            sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "-f", "10,20,30,40"]
        )
        args = parse_args()
        assert args.force_bbox == "10,20,30,40"

    def test_boolean_flags(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "-i", "x.png", "-o", "o/", "--no-expand", "--no-retry", "-k", "-t", "-v"],
        )
        args = parse_args()
        assert args.no_expand is True
        assert args.no_retry is True
        assert args.keep_masks is True
        assert args.timing is True
        assert args.verbose is True

    def test_force_output_default_false(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "-i", "x.png", "-o", "o/"])
        args = parse_args()
        assert args.force_output is False

    def test_force_output_flag(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "--force-output"]
        )
        args = parse_args()
        assert args.force_output is True

    def test_known_mask_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "-i", "x.png", "-o", "o/", "-K", "watermarks/logo.png"]
        )
        args = parse_args()
        assert args.known_mask == "watermarks/logo.png"


# =========================================================================
# _apply_color_enhancement
# =========================================================================

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


# =========================================================================
# load_watermark_templates
# =========================================================================

class TestLoadWatermarkTemplates:
    """Template loading filters correctly and only accepts RGBA."""

    def test_nonexistent_path_returns_empty(self, tmp_path):
        p = tmp_path / "does_not_exist"
        assert not p.exists()
        assert load_watermark_templates(p) == []

    def test_empty_directory_returns_empty(self, tmp_path):
        assert load_watermark_templates(tmp_path) == []

    def test_loads_single_rgba_png(self, tmp_path):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255
        rgba[:, :, 3] = 200
        path = tmp_path / "logo.png"
        cv2.imwrite(str(path), rgba)
        templates = load_watermark_templates(path)
        assert len(templates) == 1
        assert templates[0][0] == "logo.png"
        assert templates[0][1].shape == (8, 8, 4)

    def test_directory_with_rgba_png_only_includes_png(self, tmp_path):
        rgba = np.zeros((6, 6, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        (tmp_path / "a.png").write_text("not an image")
        cv2.imwrite(str(tmp_path / "b.png"), rgba)
        (tmp_path / "c.txt").write_text("ignore")
        templates = load_watermark_templates(tmp_path)
        assert len(templates) == 1
        assert templates[0][0] == "b.png"

    def test_rgb_only_png_skipped(self, tmp_path):
        """PNG with 3 channels (no alpha) is rejected by our RGBA check."""
        rgb = np.ones((6, 6, 3), dtype=np.uint8) * 128
        path = tmp_path / "rgb_only.png"
        cv2.imwrite(str(path), rgb)
        templates = load_watermark_templates(path)
        assert len(templates) == 0


# =========================================================================
# try_watermark_cascade
# =========================================================================

class TestTryWatermarkCascade:
    """Cascade does what it should and doesn't when it shouldn't."""

    def test_empty_templates_returns_none(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert try_watermark_cascade(image, []) is None

    def test_empty_templates_does_not_touch_image(self):
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        orig = image.copy()
        try_watermark_cascade(image, [])
        np.testing.assert_array_equal(image, orig)

    def test_first_successful_template_wins(self, monkeypatch):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        templates = [
            ("a.png", np.zeros((8, 8, 4), dtype=np.uint8)),
            ("b.png", np.zeros((8, 8, 4), dtype=np.uint8)),
            ("c.png", np.zeros((8, 8, 4), dtype=np.uint8)),
        ]
        matched_mask = np.ones((60, 60), dtype=np.uint8) * 255
        matched_bbox = (10, 20, 15, 12)
        calls = []

        def fake_find_known_mask_in_image(_image, _tmpl, min_matches=6, dilation_pixels=7):
            calls.append((min_matches, dilation_pixels))
            if len(calls) == 1:
                return None
            if len(calls) == 2:
                return matched_mask, matched_bbox
            raise AssertionError("Cascade should stop after first successful template")

        monkeypatch.setattr(
            cli_mod,
            "find_known_mask_in_image",
            fake_find_known_mask_in_image,
        )

        result = try_watermark_cascade(image, templates, min_matches=9, dilation_pixels=5)

        assert result is not None
        mask, bbox, template_name = result
        assert template_name == "b.png"
        np.testing.assert_array_equal(mask, matched_mask)
        assert bbox == matched_bbox
        assert calls == [(9, 5), (9, 5)]


# =========================================================================
# _save_clean_timing_report
# =========================================================================

def _make_timing(
    name="test.png",
    mp=1.0,
    det=0.5,
    color=1.0,
    mask=0.2,
    inpaint=0.8,
    total=2.5,
    boxes=1,
    failover="none",
    retried=False,
    expanded=0,
):
    """Helper to build a timing dict with sane defaults."""
    return {
        "image_name": name,
        "image_mp": mp,
        "detection_time": det,
        "color_time": color,
        "mask_time": mask,
        "inpaint_time": inpaint,
        "total_time": total,
        "consensus_boxes_count": boxes,
        "failover_type": failover,
        "retried_with_g8": retried,
        "bboxes_expanded": expanded,
        "total_bbox_area": 5000,
    }


class TestSaveCleanTimingReport:
    """Verify the timing report renders without crashing and contains key data."""

    def test_single_image_report(self, tmp_path):
        timing = _make_timing()
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=2.5, avg_time=2.5,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()

        assert "test.png" in text
        assert "lama" in text
        assert "Images processed: 1" in text

    def test_multiple_image_report_has_statistics(self, tmp_path):
        timings = [
            _make_timing(name="a.png", total=2.0),
            _make_timing(name="b.png", total=4.0),
        ]
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            timings, total_time=6.0, avg_time=3.0,
            timing_file=out_file, method="telea",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()

        assert "MEDIAN" in text
        assert "AVERAGE" in text
        assert "Images processed: 2" in text

    def test_failover_markers_rendered(self, tmp_path):
        timings = [
            _make_timing(name="rot.png", failover="rotation"),
            _make_timing(name="gray.png", failover="gray_enhancement"),
            _make_timing(name="white.png", failover="white_enhancement"),
            _make_timing(name="target.png", failover="target_color"),
            _make_timing(name="base.png", failover="watermark"),
        ]
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            timings, total_time=10.0, avg_time=2.0,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()

        # Failover summary section
        assert "Rotation failover: 1" in text
        assert "Gray enhancement: 1" in text
        assert "White enhancement: 1" in text
        assert "Target color enhancement: 1" in text
        assert "Baseline watermark regions: 1" in text

    def test_none_timing_values_handled(self, tmp_path):
        """Images that failed mid-pipeline may have None for some timing fields."""
        timing = _make_timing()
        timing["color_time"] = None
        timing["mask_time"] = None
        timing["inpaint_time"] = None

        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=1.0, avg_time=1.0,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()
        assert "N/A" in text

    def test_target_color_and_forced_bbox_noted(self, tmp_path):
        timing = _make_timing()
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=1.0, avg_time=1.0,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3,
            target_color=(128, 128, 128),
            forced_bbox=(10, 20, 100, 50),
        )
        text = out_file.read_text()
        assert "(128, 128, 128)" in text
        assert "(10, 20, 100, 50)" in text

    def test_retry_count_shown(self, tmp_path):
        timings = [
            _make_timing(name="a.png", retried=True),
            _make_timing(name="b.png", retried=False),
        ]
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            timings, total_time=5.0, avg_time=2.5,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()
        assert "retried with g=8: 1" in text

    def test_template_match_entry_does_not_crash(self, tmp_path):
        """Entries from watermark template path use 'image' not 'image_name'; report must not KeyError."""
        template_only = {
            "image": "watermarked_photo.jpg",
            "matched_template": "sg_logo.png",
            "mask_found": True,
            "total_time": 0,
        }
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [template_only], total_time=1.0, avg_time=1.0,
            timing_file=out_file, method="lama",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()
        assert "watermarked_photo.jpg" in text or "watermarked_photo" in text
        assert "Template match: 1" in text

    def test_mixed_consensus_and_template_entries(self, tmp_path):
        """Mix of full consensus timings and template-only timings writes without KeyError."""
        consensus = _make_timing(name="consensus.png", total=2.0)
        template_only = {
            "image": "tpl.png",
            "matched_template": "logo.png",
            "mask_found": True,
            "total_time": 0.5,
        }
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [consensus, template_only], total_time=2.5, avg_time=1.25,
            timing_file=out_file, method="telea",
            confidence_threshold=0.3, target_color=None, forced_bbox=None,
        )
        text = out_file.read_text()
        assert "consensus.png" in text
        assert "tpl.png" in text or "tpl" in text
        assert "Template match: 1" in text
        assert "Images processed: 2" in text


# =========================================================================
# process_single_image — smoke test
# =========================================================================

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


class TestProcessSingleImageFailovers:
    def test_target_color_success_short_circuits_detection(self, monkeypatch, tmp_path):
        image = np.ones((40, 80, 3), dtype=np.uint8) * 255
        image_path = tmp_path / "img.png"
        cv2.imwrite(str(image_path), image)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("Consensus detection should not run when target-color succeeds")

        monkeypatch.setattr(consensus_mod, "run_consensus_detection", fail_if_called)
        monkeypatch.setattr(
            cli_mod,
            "_try_color_enhanced_detection",
            lambda *_args, **_kwargs: [(5, 8, 12, 10)],
        )
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)
        monkeypatch.setattr(
            cli_mod,
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

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())

        calls = {"n": 0}

        def fake_consensus(_img, _threshold):
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            return [(5, 7, 10, 6)]  # (x_rot, y_rot, w_rot, h_rot)

        monkeypatch.setattr(consensus_mod, "run_consensus_detection", fake_consensus)
        monkeypatch.setattr(
            cli_mod,
            "_try_color_enhanced_detection",
            lambda *_args, **_kwargs: [],
        )

        captured = {}

        def fake_generate(img, boxes, _g, _method, _target, **_kw):
            captured["boxes"] = boxes
            return np.zeros(img.shape[:2], dtype=np.uint8), img.copy()

        monkeypatch.setattr(cli_mod, "_generate_masks_and_inpaint", fake_generate)
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

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: image.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *_args, **_kwargs: [])

        color_attempts = []

        def fake_try_color(_image, _threshold, target_hex, sensitivity=3):
            color_attempts.append((target_hex, sensitivity))
            return []

        monkeypatch.setattr(cli_mod, "_try_color_enhanced_detection", fake_try_color)

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


class TestProcessWithKnownMask:
    def test_no_match_saves_original_and_skips_inpaint(self, monkeypatch, tmp_path):
        image = np.ones((30, 30, 3), dtype=np.uint8) * 127
        image_path = tmp_path / "input.png"
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        known_mask = np.zeros((10, 10, 4), dtype=np.uint8)

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(cli_mod, "find_known_mask_in_image", lambda *_args, **_kwargs: None)
        saved = {}

        def fake_save(arr, path):
            saved[str(path)] = arr.copy()

        monkeypatch.setattr(cli_mod, "save_image", fake_save)

        timings = process_with_known_mask(
            image_path=image_path,
            output_dir=output_dir,
            known_mask_rgba=known_mask,
            keep_masks=True,
            method="telea",
        )

        assert timings is not None
        assert timings["mask_found"] is False
        assert any(name.endswith("input_clean.png") for name in saved)
        saved_image = next(arr for name, arr in saved.items() if name.endswith("input_clean.png"))
        np.testing.assert_array_equal(saved_image, image)
        assert not any(name.endswith("input_mask.png") for name in saved)

    def test_match_path_saves_mask_and_inpainted_output(self, monkeypatch, tmp_path):
        image = np.ones((40, 40, 3), dtype=np.uint8) * 180
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:20, 12:26] = 255
        bbox = (12, 8, 14, 12)
        known_mask = np.zeros((10, 10, 4), dtype=np.uint8)
        image_path = tmp_path / "photo.png"
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(
            cli_mod,
            "find_known_mask_in_image",
            lambda *_args, **_kwargs: (mask, bbox),
        )

        import untextre.inpaint as inpaint_mod

        inpaint_calls = {}

        def fake_inpaint(img, inpaint_mask, bbox=None, method="lama"):
            inpaint_calls["bbox"] = bbox
            inpaint_calls["method"] = method
            np.testing.assert_array_equal(inpaint_mask, mask)
            return np.zeros_like(img)

        monkeypatch.setattr(inpaint_mod, "inpaint_image", fake_inpaint)

        saved_paths = []
        monkeypatch.setattr(cli_mod, "save_image", lambda _arr, p: saved_paths.append(p.name))

        timings = process_with_known_mask(
            image_path=image_path,
            output_dir=output_dir,
            known_mask_rgba=known_mask,
            keep_masks=True,
            method="telea",
        )

        assert timings is not None
        assert timings["mask_found"] is True
        assert inpaint_calls["bbox"] == bbox
        assert inpaint_calls["method"] == "telea"
        assert "photo_mask.png" in saved_paths
        assert "photo_clean.png" in saved_paths


# =========================================================================
# find_known_mask_in_image — validation guards
# =========================================================================

class TestFindKnownMaskValidation:
    """Test guard clauses in find_known_mask_in_image() (lines 266-370)."""

    def test_non_rgba_raises(self):
        """A 3-channel image as the known mask must raise ValueError."""
        target = np.zeros((100, 100, 3), dtype=np.uint8)
        known_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="4 channels"):
            find_known_mask_in_image(target, known_rgb)

    def test_no_descriptors_returns_none(self):
        """A solid-color image produces no ORB descriptors → None."""
        target = np.ones((100, 100, 3), dtype=np.uint8) * 128
        known = np.ones((50, 50, 4), dtype=np.uint8) * 128
        known[:, :, 3] = 255
        result = find_known_mask_in_image(target, known)
        assert result is None

    def test_insufficient_keypoints_returns_none(self):
        """An image with fewer keypoints than min_matches → None."""
        # Tiny image with minimal features
        target = np.zeros((20, 20, 3), dtype=np.uint8)
        target[5, 5] = [255, 255, 255]
        known = np.zeros((10, 10, 4), dtype=np.uint8)
        known[3, 3, :3] = 255
        known[:, :, 3] = 255
        # Use a very high min_matches to guarantee insufficient keypoints
        result = find_known_mask_in_image(target, known, min_matches=500)
        assert result is None

    def test_insufficient_good_matches_returns_none(self):
        """Images with features but poor match quality → None."""
        # Two very different textured images — ORB features won't match well
        rng = np.random.RandomState(42)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = rng.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        known[:, :, 3] = 255
        # High min_matches ensures insufficient good matches
        result = find_known_mask_in_image(target, known, min_matches=200)
        assert result is None

    def test_affine_transform_failure_returns_none(self, monkeypatch):
        """cv2.estimateAffine2D returning None → None."""
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (None, None),
        )
        # Need images with enough features for ORB to find matches
        rng = np.random.RandomState(99)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255
        result = find_known_mask_in_image(target, known_rgba, min_matches=3)
        assert result is None

    def test_scale_too_small_returns_none(self, monkeypatch):
        """A transform with near-zero scale → None."""
        # Matrix with scale ~0.01 in both axes
        tiny_scale = np.array([[0.01, 0.0, 10.0],
                               [0.0, 0.01, 10.0]], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (tiny_scale, inlier_mask),
        )
        rng = np.random.RandomState(10)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255
        result = find_known_mask_in_image(target, known_rgba, min_matches=3)
        assert result is None

    def test_scale_too_large_returns_none(self, monkeypatch):
        """A transform with huge scale → None."""
        huge_scale = np.array([[25.0, 0.0, 0.0],
                               [0.0, 25.0, 0.0]], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (huge_scale, inlier_mask),
        )
        rng = np.random.RandomState(11)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255
        result = find_known_mask_in_image(target, known_rgba, min_matches=3)
        assert result is None

    def test_reflection_rejected(self, monkeypatch):
        """A transform with negative determinant (reflection) → None."""
        # det([[−1, 0], [0, 1]]) = −1 → reflection
        reflect = np.array([[-1.0, 0.0, 50.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (reflect, inlier_mask),
        )
        rng = np.random.RandomState(12)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255
        result = find_known_mask_in_image(target, known_rgba, min_matches=3)
        assert result is None

    def test_excessive_stretch_rejected(self, monkeypatch):
        """A transform with non-uniform scaling beyond MAX_STRETCH → None."""
        # scale_major/scale_minor = 5.0/1.0 = 5.0 > MAX_STRETCH (1.25)
        stretch = np.array([[5.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (stretch, inlier_mask),
        )
        rng = np.random.RandomState(13)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255
        result = find_known_mask_in_image(target, known_rgba, min_matches=3)
        assert result is None


# =========================================================================
# main() — force-bbox parsing
# =========================================================================

class TestMainForceBbox:
    """Test force-bbox validation inside main() (lines 584-600)."""

    def _setup_main_mocks(self, monkeypatch, tmp_path, extra_argv=None):
        """Set up argv and mock everything after bbox parsing to avoid model loading."""
        img_path = tmp_path / "img.png"
        cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        argv = ["prog", "-i", str(img_path), "-o", str(out_dir), "-p", "telea"]
        if extra_argv:
            argv.extend(extra_argv)
        monkeypatch.setattr(sys, "argv", argv)
        return img_path, out_dir

    def test_force_bbox_valid_parse(self, monkeypatch, tmp_path):
        """Happy path: '10,20,30,40' parses without error."""
        self._setup_main_mocks(monkeypatch, tmp_path, ["-f", "10,20,30,40"])
        # Mock model init and processing to prevent actual execution
        monkeypatch.setattr(cli_mod, "initialize_consensus_models", lambda *a, **kw: None)
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": False},
        )
        # Should not raise SystemExit
        main()

    def test_force_bbox_wrong_count_exits(self, monkeypatch, tmp_path):
        """'10,20,30' (3 values) → sys.exit(1)."""
        self._setup_main_mocks(monkeypatch, tmp_path, ["-f", "10,20,30"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_force_bbox_negative_exits(self, monkeypatch, tmp_path):
        """Negative coordinate → sys.exit(1)."""
        # Use --force-bbox=VALUE to avoid argparse treating leading '-' as a flag
        self._setup_main_mocks(monkeypatch, tmp_path, ["--force-bbox=-1,0,10,10"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_force_bbox_zero_dimension_exits(self, monkeypatch, tmp_path):
        """'10,20,0,40' (zero width) → sys.exit(1)."""
        self._setup_main_mocks(monkeypatch, tmp_path, ["-f", "10,20,0,40"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# =========================================================================
# main() — integration paths
# =========================================================================

class TestMainIntegrationPaths:
    """Test main() orchestration logic (lines 602-800), heavily mocked."""

    def _run_main(self, monkeypatch, tmp_path, extra_argv=None, create_image=True):
        """Helper: set up a minimal main() invocation and return (img_path, out_dir)."""
        img_path = tmp_path / "test_img.png"
        if create_image:
            cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        argv = ["prog", "-i", str(img_path), "-o", str(out_dir), "-p", "telea"]
        if extra_argv:
            argv.extend(extra_argv)
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(cli_mod, "initialize_consensus_models", lambda *a, **kw: None)
        return img_path, out_dir

    def test_verbose_enables_debug_logging(self, monkeypatch, tmp_path):
        """--verbose should set root logger to DEBUG."""
        import logging
        self._run_main(monkeypatch, tmp_path, ["--verbose"])
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": False},
        )
        main()
        assert logging.getLogger().level == logging.DEBUG

    def test_logfile_creates_file_handler(self, monkeypatch, tmp_path):
        """--logfile should create a log file on disk."""
        log_path = tmp_path / "logs" / "test.log"
        self._run_main(monkeypatch, tmp_path, ["--logfile", str(log_path)])
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": False},
        )
        main()
        assert log_path.exists()

    def test_nonexistent_input_exits(self, monkeypatch, tmp_path):
        """Bad input path → sys.exit(1)."""
        self._run_main(monkeypatch, tmp_path, create_image=False)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_force_output_copies_original(self, monkeypatch, tmp_path):
        """--force-output with skipped image copies original to output."""
        self._run_main(monkeypatch, tmp_path, ["--force-output"])
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": True},
        )
        saved = {}
        original_save = cli_mod.save_image

        def tracking_save(arr, path):
            saved[str(path)] = arr

        monkeypatch.setattr(cli_mod, "save_image", tracking_save)
        main()
        # Should have saved a "_clean" file (the original copied as-is)
        assert any("_clean" in str(p) for p in saved.keys())

    def test_timing_flag_saves_report(self, monkeypatch, tmp_path):
        """--timing produces timing_report.txt."""
        _, out_dir = self._run_main(monkeypatch, tmp_path, ["-t"])
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: _make_timing(name="test_img.png"),
        )
        main()
        assert (out_dir / "timing_report.txt").exists()

    def test_timing_with_logfile_saves_both(self, monkeypatch, tmp_path):
        """--timing + --logfile saves two timing reports."""
        log_path = tmp_path / "run.log"
        _, out_dir = self._run_main(monkeypatch, tmp_path, ["-t", "--logfile", str(log_path)])
        monkeypatch.setattr(
            cli_mod, "process_single_image",
            lambda **kw: _make_timing(name="test_img.png"),
        )
        main()
        assert (out_dir / "timing_report.txt").exists()
        assert log_path.with_suffix(".timing.txt").exists()

    def test_image_error_continues_batch(self, monkeypatch, tmp_path):
        """Exception on one image doesn't stop batch processing."""
        # Create two images
        for name in ("a.png", "b.png"):
            cv2.imwrite(
                str(tmp_path / name),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "-i", str(tmp_path), "-o", str(out_dir), "-p", "telea"],
        )
        monkeypatch.setattr(cli_mod, "initialize_consensus_models", lambda *a, **kw: None)

        calls = {"n": 0}

        def process_or_raise(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Simulated failure on first image")
            return {"total_time": 0.1, "skipped": False}

        monkeypatch.setattr(cli_mod, "process_single_image", process_or_raise)
        # Also mock cleanup_vram (imported lazily in finally block)
        import untextre.detector as detector_mod
        monkeypatch.setattr(detector_mod, "cleanup_vram", lambda: None)
        main()
        assert calls["n"] == 2

    def test_explicit_known_mask_no_fallback(self, monkeypatch, tmp_path):
        """-K with no match warns but doesn't fall back to consensus detection."""
        # Create a valid RGBA template
        template_path = tmp_path / "template.png"
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        cv2.imwrite(str(template_path), rgba)

        img_path = tmp_path / "photo.png"
        cv2.imwrite(str(img_path), np.ones((50, 50, 3), dtype=np.uint8) * 128)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(
            sys, "argv",
            ["prog", "-i", str(img_path), "-o", str(out_dir),
             "-p", "telea", "-K", str(template_path)],
        )

        # Mock LaMa init to no-op
        import untextre.inpaint as inpaint_mod
        monkeypatch.setattr(inpaint_mod, "initialize_lama_model", lambda **kw: True)

        # Make cascade return None (no match)
        monkeypatch.setattr(cli_mod, "try_watermark_cascade", lambda *a, **kw: None)

        # process_single_image should NOT be called (no fallback)
        def fail_if_called(**kw):
            raise AssertionError("Should not fall back to consensus detection with explicit -K")

        monkeypatch.setattr(cli_mod, "process_single_image", fail_if_called)

        import untextre.detector as detector_mod
        monkeypatch.setattr(detector_mod, "cleanup_vram", lambda: None)

        main()  # Should complete without calling process_single_image


# =========================================================================
# process_single_image — edge cases
# =========================================================================

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
        monkeypatch.setattr(cli_mod, "load_image", lambda _p: np.zeros((50, 50, 3), dtype=np.uint8))
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

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: False)

        captured = {}

        def fake_generate(image, boxes, _g, _method, _target, **_kw):
            captured["boxes"] = boxes
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(cli_mod, "_generate_masks_and_inpaint", fake_generate)

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
        monkeypatch.setattr(cli_mod, "load_image", lambda _p: img.copy())
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

        monkeypatch.setattr(cli_mod, "load_image", tracking_load)
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

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *a, **kw: [(5, 5, 10, 10)])
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: True)

        generate_calls = []

        def fake_generate(image, boxes, g, method, target, **_kw):
            generate_calls.append(g)
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(cli_mod, "_generate_masks_and_inpaint", fake_generate)

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

        monkeypatch.setattr(cli_mod, "load_image", lambda _p: img.copy())
        monkeypatch.setattr(preprocessor_mod, "preprocess_image", lambda _img: img.copy())
        monkeypatch.setattr(consensus_mod, "run_consensus_detection", lambda *a, **kw: [(5, 5, 10, 10)])
        monkeypatch.setattr(metrics_mod, "expand_bbox_along_long_axis", lambda _img, bbox: bbox)
        # needs_retry would return True, but shouldn't be checked
        monkeypatch.setattr(metrics_mod, "needs_retry", lambda _region: True)

        generate_calls = []

        def fake_generate(image, boxes, g, method, target, **_kw):
            generate_calls.append(g)
            return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

        monkeypatch.setattr(cli_mod, "_generate_masks_and_inpaint", fake_generate)

        timings = process_single_image(
            image_path=img_path, output_dir=out_dir, method="telea",
            expand_bboxes=True, auto_retry=False,
        )
        assert timings is not None
        assert timings["retried_with_g8"] is False
        assert generate_calls == [4]


# =========================================================================
# -U / --unknown-watermark flag
# =========================================================================

def test_unknown_watermark_flag_exists():
    """Smoke test: -U flag is registered and mutually exclusive with -K."""
    from untextre.cli import create_parser
    parser = create_parser()
    args = parser.parse_args(["-U", "-i", "some/dir", "-o", "out/dir"])
    assert args.unknown_watermark is True


def test_unknown_watermark_and_known_mask_are_mutually_exclusive():
    from untextre.cli import create_parser
    import pytest
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-U", "-K", "template.png", "-i", "some/dir", "-o", "out/dir"])


# =========================================================================
# Error paths and lazy-loader coverage
# =========================================================================

class TestCliErrorPaths:
    """Cover error/guard paths in cli.py — no mocks, pure logic paths."""

    def test_process_with_known_mask_bad_image(self, tmp_path):
        """Zero-byte image file → process_with_known_mask returns None."""
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"")  # zero-byte file
        template = np.zeros((50, 50, 4), dtype=np.uint8)
        result = process_with_known_mask(
            image_path=bad_path,
            output_dir=tmp_path,
            known_mask_rgba=template,
        )
        assert result is None

    def test_main_same_dir_guard(self, tmp_path, monkeypatch):
        """main() with -U and same input/output dir exits with code 1."""
        monkeypatch.setattr(
            sys, "argv", ["untextre", "-i", str(tmp_path), "-o", str(tmp_path), "-U"]
        )
        # Return a non-empty file list so the empty-list guard (line 654) doesn't
        # fire before we reach the same-dir guard (line 679).
        monkeypatch.setattr(
            "untextre.cli.get_image_files",
            lambda _path: [tmp_path / "fake.jpg"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_load_watermark_templates_missing_path(self):
        """Non-existent path → load_watermark_templates returns []."""
        result = load_watermark_templates(Path("definitely_does_not_exist_xyz"))
        assert result == []

    def test_save_timing_report(self, tmp_path):
        """_save_clean_timing_report writes a file containing expected headers."""
        timing_file = tmp_path / "timing.txt"
        detailed = [{"file": "a.jpg", "time": 1.2}]
        _save_clean_timing_report(
            detailed_timings=detailed,
            total_time=1.2,
            avg_time=1.2,
            timing_file=timing_file,
            method="known_mask",
            confidence_threshold=0.5,
            target_color=None,
            forced_bbox=None,
        )
        assert timing_file.exists(), "Timing report file was not created"
        content = timing_file.read_text()
        assert "Inpainting method:" in content


def test_package_lazy_imports():
    """__getattr__ lazy-loader in __init__.py is triggered by attribute access."""
    # Access one name from each lazy branch to maximise coverage:
    # utils branch, find_text_colors branch, inpaint branch, cli branch
    assert callable(untextre.load_image), "load_image not callable"
    assert callable(untextre.find_mask_by_spatial_tf_idf), "find_mask_by_spatial_tf_idf not callable"
    assert callable(untextre.inpaint_image), "inpaint_image not callable"
    assert callable(untextre.main), "main not callable"
