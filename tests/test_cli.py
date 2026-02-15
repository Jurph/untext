"""Tests for untextre.cli module.

Covers:
    - ``parse_args()``  – argument parsing defaults & validation
    - ``_save_clean_timing_report()``  – timing report formatting
    - ``process_single_image()``  – lightweight smoke test (telea, forced bbox)

Heavy integration tests (find_known_mask_in_image, full detection failover
chain) are intentionally omitted — they canonize detector behaviour rather
than code logic, and would be fragile under model updates.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from untextre.cli import (
    _save_clean_timing_report,
    parse_args,
    process_single_image,
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
