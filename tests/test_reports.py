"""Tests for untextre.reports helpers."""

import cv2
import numpy as np

from untextre.reports import (
    TimingReportConfig,
    _save_clean_timing_report,
    _save_discovered_watermark_candidates,
)

class TestSaveDiscoveredWatermarkCandidates:
    def test_exports_orb_prepped_candidates_only(self, tmp_path):
        raw = np.zeros((11, 11, 4), dtype=np.uint8)
        raw[:, :, :3] = 177
        raw[3:8, 3:8, :3] = 220
        raw[3:8, 3:8, 3] = 255
        raw[5, 5, 3] = 0
        raw[0, 0, 3] = 255

        templates = _save_discovered_watermark_candidates(tmp_path, [raw])

        assert len(templates) == 1
        bgra = templates[0].rgba
        assert templates[0].name == "watermark_candidate.png"
        assert bgra.shape == (15, 15, 4)
        assert np.all(bgra[:2, :, 3] == 0)
        assert np.all(bgra[:, :2, 3] == 0)
        assert bgra[7, 7, 3] == 255

        saved = cv2.imread(str(tmp_path / "watermark_candidate.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved, bgra)


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
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
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
            timing_file=out_file,
            config=TimingReportConfig(method="telea", confidence_threshold=0.3),
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
        ]
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            timings, total_time=10.0, avg_time=2.0,
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
        )
        text = out_file.read_text()

        # Failover summary section
        assert "Rotation failover: 1" in text
        assert "Gray enhancement: 1" in text
        assert "White enhancement: 1" in text
        assert "Target color enhancement: 1" in text

    def test_none_timing_values_handled(self, tmp_path):
        """Images that failed mid-pipeline may have None for some timing fields."""
        timing = _make_timing()
        timing["color_time"] = None
        timing["mask_time"] = None
        timing["inpaint_time"] = None

        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=1.0, avg_time=1.0,
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
        )
        text = out_file.read_text()
        assert "N/A" in text

    def test_target_color_and_forced_bbox_noted(self, tmp_path):
        timing = _make_timing()
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=1.0, avg_time=1.0,
            timing_file=out_file,
            config=TimingReportConfig(
                method="lama",
                confidence_threshold=0.3,
                target_color=(128, 128, 128),
                forced_bbox=(10, 20, 100, 50),
            ),
        )
        text = out_file.read_text()
        assert "Target color: (128, 128, 128)" in text
        assert "(10, 20, 100, 50)" in text

    def test_retry_count_shown(self, tmp_path):
        timings = [
            _make_timing(name="a.png", retried=True),
            _make_timing(name="b.png", retried=False),
        ]
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            timings, total_time=5.0, avg_time=2.5,
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
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
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
        )
        text = out_file.read_text()
        assert "watermarked_photo.jpg" in text or "watermarked_photo" in text
        assert "Template match: 1" in text

    def test_template_match_inliers_rendered_in_report(self, tmp_path):
        """Regression: orb_inliers on a template-match timing dict must show
        up in the rendered report, so a marginal (low-inlier) template match
        is distinguishable from a strong one."""
        template_only = {
            "image": "watermarked_photo.jpg",
            "matched_template": "sg_logo.png",
            "orb_inliers": 87,
            "mask_found": True,
            "total_time": 0,
        }
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [template_only], total_time=1.0, avg_time=1.0,
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
        )
        text = out_file.read_text()
        assert "87" in text

    def test_missing_orb_inliers_renders_as_na(self, tmp_path):
        """Consensus-path entries never set orb_inliers; report must not KeyError
        and must show N/A rather than a bogus 0."""
        timing = _make_timing()
        out_file = tmp_path / "report.txt"
        _save_clean_timing_report(
            [timing], total_time=1.0, avg_time=1.0,
            timing_file=out_file,
            config=TimingReportConfig(method="lama", confidence_threshold=0.3),
        )
        text = out_file.read_text()
        assert "N/A" in text

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
            timing_file=out_file,
            config=TimingReportConfig(method="telea", confidence_threshold=0.3),
        )
        text = out_file.read_text()
        assert "consensus.png" in text
        assert "tpl.png" in text or "tpl" in text
        assert "Template match: 1" in text
        assert "Images processed: 2" in text
