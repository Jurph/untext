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

import untextre.cli as cli_mod
import untextre.consensus as consensus_mod
import untextre.metrics as metrics_mod
import untextre.preprocessor as preprocessor_mod
from untextre.cli import (
    _apply_color_enhancement,
    _save_clean_timing_report,
    load_watermark_templates,
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
            lambda img, _boxes, _g, _method, _target: (
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

        def fake_generate(img, boxes, _g, _method, _target):
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
