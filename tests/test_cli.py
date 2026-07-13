"""Tests for untextre.cli argument parsing and main() orchestration."""

import sys

import cv2
import numpy as np
import pytest

import untextre.cli as cli_mod
import untextre.orb_matcher as orb_matcher_mod
import untextre.pipeline as pipeline_mod
import untextre.reports as reports_mod
from untextre.cli import main




# =========================================================================
# parse_args
# =========================================================================



# =========================================================================
# _apply_color_enhancement
# =========================================================================




# =========================================================================
# load_watermark_templates
# =========================================================================




# =========================================================================
# try_watermark_cascade
# =========================================================================






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





# =========================================================================
# process_single_image — smoke test
# =========================================================================








# =========================================================================
# find_known_mask_in_image — validation guards
# =========================================================================







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
        monkeypatch.setattr(pipeline_mod, "initialize_consensus_models", lambda *a, **kw: None)
        monkeypatch.setattr(
            pipeline_mod, "process_single_image",
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
        monkeypatch.setattr(pipeline_mod, "initialize_consensus_models", lambda *a, **kw: None)
        return img_path, out_dir



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
            pipeline_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": True},
        )
        saved = {}

        def tracking_save(arr, path, **kwargs):
            saved[str(path)] = (arr, kwargs.get("source_path"))

        monkeypatch.setattr(cli_mod, "save_image", tracking_save)
        main()
        # Should have saved a "_clean" file (the original copied as-is)
        assert any("_clean" in str(p) for p in saved.keys())
        assert all(source_path is not None for _arr, source_path in saved.values())

    def test_force_output_reuses_already_loaded_image(self, monkeypatch, tmp_path):
        """force_output must not reload the image when the template cascade already loaded it (#10)."""
        img_path, out_dir = self._run_main(monkeypatch, tmp_path, ["--force-output"])

        fake_template = orb_matcher_mod.WatermarkTemplate(
            "fake.png", np.zeros((4, 4, 4), dtype=np.uint8), ()
        )
        monkeypatch.setattr(
            orb_matcher_mod, "load_watermark_templates",
            lambda *_a, **_kw: [fake_template],
        )
        monkeypatch.setattr(orb_matcher_mod, "try_watermark_cascade", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            pipeline_mod, "process_single_image",
            lambda **kw: {"total_time": 0.1, "skipped": True},
        )
        monkeypatch.setattr(cli_mod, "save_image", lambda *_a, **_kw: None)

        load_calls = []
        real_load_image = cli_mod.load_image

        def tracking_load_image(path):
            load_calls.append(path)
            return real_load_image(path)

        monkeypatch.setattr(cli_mod, "load_image", tracking_load_image)

        main()

        # The template cascade already loaded the image (line ~187); force_output
        # must reuse that array instead of calling load_image a second time.
        assert load_calls == [img_path]


    def test_timing_flag_saves_report(self, monkeypatch, tmp_path):
        """--timing produces timing_report.txt."""
        _, out_dir = self._run_main(monkeypatch, tmp_path, ["-t"])
        monkeypatch.setattr(
            pipeline_mod, "process_single_image",
            lambda **kw: _make_timing(name="test_img.png"),
        )
        main()
        assert (out_dir / "timing_report.txt").exists()

    def test_timing_with_logfile_saves_both(self, monkeypatch, tmp_path):
        """--timing + --logfile saves two timing reports."""
        log_path = tmp_path / "run.log"
        _, out_dir = self._run_main(monkeypatch, tmp_path, ["-t", "--logfile", str(log_path)])
        monkeypatch.setattr(
            pipeline_mod, "process_single_image",
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
        monkeypatch.setattr(pipeline_mod, "initialize_consensus_models", lambda *a, **kw: None)

        calls = {"n": 0}

        def process_or_raise(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Simulated failure on first image")
            return {"total_time": 0.1, "skipped": False}

        monkeypatch.setattr(pipeline_mod, "process_single_image", process_or_raise)
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
        monkeypatch.setattr(orb_matcher_mod, "try_watermark_cascade", lambda *a, **kw: None)

        # process_single_image should NOT be called (no fallback)
        def fail_if_called(**kw):
            raise AssertionError("Should not fall back to consensus detection with explicit -K")

        monkeypatch.setattr(pipeline_mod, "process_single_image", fail_if_called)

        import untextre.detector as detector_mod
        monkeypatch.setattr(detector_mod, "cleanup_vram", lambda: None)

        main()  # Should complete without calling process_single_image

    def test_template_match_timing_reports_elapsed_time(self, monkeypatch, tmp_path):
        template_path = tmp_path / "template.png"
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        cv2.imwrite(str(template_path), rgba)

        img_path = tmp_path / "photo.png"
        cv2.imwrite(str(img_path), np.ones((50, 50, 3), dtype=np.uint8) * 128)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "-i",
                str(img_path),
                "-o",
                str(out_dir),
                "-p",
                "telea",
                "-K",
                str(template_path),
                "--timing",
            ],
        )

        import untextre.inpaint as inpaint_mod
        monkeypatch.setattr(inpaint_mod, "initialize_lama_model", lambda **kw: True)
        monkeypatch.setattr(
            orb_matcher_mod,
            "try_watermark_cascade",
            lambda *_a, **_kw: (
                np.zeros((50, 50), dtype=np.uint8),
                (5, 5, 20, 20),
                "template.png",
                42,
            ),
        )
        monkeypatch.setattr(inpaint_mod, "inpaint_image", lambda image, *_a, **_kw: image.copy())
        monkeypatch.setattr(cli_mod, "save_image", lambda *_a, **_kw: None)

        captured = {}

        def fake_save_timing_report(detailed_timings, *_args, **_kwargs):
            captured["timings"] = detailed_timings

        monkeypatch.setattr(reports_mod, "_save_clean_timing_report", fake_save_timing_report)

        import untextre.detector as detector_mod
        monkeypatch.setattr(detector_mod, "cleanup_vram", lambda: None)

        main()

        assert captured["timings"][0]["matched_template"] == "template.png"
        assert captured["timings"][0]["orb_inliers"] == 42
        assert captured["timings"][0]["total_time"] > 0


# =========================================================================
# process_single_image — edge cases
# =========================================================================




# =========================================================================
# -U / --unknown-watermark flag
# =========================================================================



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

