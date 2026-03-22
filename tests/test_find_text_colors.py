"""Direct unit tests for untextre.find_text_colors.compute_cluster_fom().

This function scores how "text-like" a color cluster is using a weighted
combination of three signals.  The weights were empirically determined
from 18,000+ synthetic watermark samples, so we pin the exact formula
here to prevent accidental drift.

    FOM = 0.07 * tf_score + 0.63 * border_score + 0.30 * cc_score

where:
    tf_score     = min(tf_idf_normalized / 255, 1.0)
    border_score = max(0.0, 1.0 - border_ratio)
    cc_score     = max(0.0, 1.0 - largest_cc_fraction)
"""

import cv2
import numpy as np
import pytest
from untextre.find_text_colors import compute_cluster_fom, find_mask_by_spatial_tf_idf


class TestComputeClusterFom:
    """Pin the FOM scoring function's behaviour."""

    # ------------------------------------------------------------------
    # Range / type checks
    # ------------------------------------------------------------------

    def test_returns_float(self):
        result = compute_cluster_fom(128.0, 0.5, 0.5)
        assert isinstance(result, float)

    def test_output_in_zero_one(self):
        """FOM should always be in [0, 1]."""
        # Extreme inputs
        for tf in (0.0, 128.0, 255.0, 500.0):
            for br in (0.0, 0.5, 1.0, 2.0):
                for cc in (0.0, 0.5, 1.0, 2.0):
                    fom = compute_cluster_fom(tf, br, cc)
                    assert 0.0 <= fom <= 1.0, (
                        f"FOM={fom} out of [0,1] for tf={tf}, br={br}, cc={cc}"
                    )

    # ------------------------------------------------------------------
    # Known-value tests (pin the weights)
    # ------------------------------------------------------------------

    def test_all_zeros_gives_maximum(self):
        """Zero border_ratio and zero cc_fraction → maximum scores on those axes."""
        fom = compute_cluster_fom(0.0, 0.0, 0.0)
        # tf_score=0, border_score=1, cc_score=1
        expected = 0.07 * 0.0 + 0.63 * 1.0 + 0.30 * 1.0  # 0.93
        assert fom == pytest.approx(expected, abs=1e-9)

    def test_perfect_text_cluster(self):
        """High TF-IDF, zero border, zero CC → near-maximum FOM."""
        fom = compute_cluster_fom(255.0, 0.0, 0.0)
        expected = 0.07 * 1.0 + 0.63 * 1.0 + 0.30 * 1.0  # 1.0
        assert fom == pytest.approx(expected, abs=1e-9)

    def test_background_cluster(self):
        """Low TF-IDF, high border, high CC → near-zero FOM."""
        fom = compute_cluster_fom(0.0, 1.0, 1.0)
        # tf_score=0, border_score=0, cc_score=0
        expected = 0.0
        assert fom == pytest.approx(expected, abs=1e-9)

    def test_solid_blob_rejected(self):
        """High TF-IDF but cc_fraction=1.0 → low FOM (solid blob, not text)."""
        fom = compute_cluster_fom(255.0, 0.0, 1.0)
        # tf=1, border=1, cc=0 → 0.07 + 0.63 + 0.0 = 0.70
        expected = 0.07 * 1.0 + 0.63 * 1.0 + 0.30 * 0.0
        assert fom == pytest.approx(expected, abs=1e-9)

    def test_border_heavy_cluster(self):
        """High border ratio → low FOM regardless of other signals."""
        fom = compute_cluster_fom(255.0, 1.0, 0.0)
        # tf=1, border=0, cc=1 → 0.07 + 0.0 + 0.30 = 0.37
        expected = 0.07 * 1.0 + 0.63 * 0.0 + 0.30 * 1.0
        assert fom == pytest.approx(expected, abs=1e-9)

    # ------------------------------------------------------------------
    # Weight dominance
    # ------------------------------------------------------------------

    def test_border_ratio_dominates_tf_idf(self):
        """Changing border ratio should move FOM more than changing tf_idf."""
        fom_good_border = compute_cluster_fom(0.0, 0.0, 0.5)   # border=0
        fom_bad_border = compute_cluster_fom(255.0, 1.0, 0.5)   # border=1

        # Good border + zero TF should beat bad border + max TF
        assert fom_good_border > fom_bad_border, (
            f"Border weight (0.63) should dominate TF-IDF weight (0.07): "
            f"{fom_good_border} vs {fom_bad_border}"
        )

    def test_cc_fraction_matters_more_than_tf_idf(self):
        """Changing cc_fraction should move FOM more than changing tf_idf."""
        fom_fragmented = compute_cluster_fom(0.0, 0.5, 0.0)   # cc=0
        fom_solid = compute_cluster_fom(255.0, 0.5, 1.0)       # cc=1

        assert fom_fragmented > fom_solid

    # ------------------------------------------------------------------
    # Clamping / saturation
    # ------------------------------------------------------------------

    def test_tf_idf_above_255_clamped(self):
        """tf_idf_normalized > 255 should be clamped to 1.0 internally."""
        fom_255 = compute_cluster_fom(255.0, 0.5, 0.5)
        fom_500 = compute_cluster_fom(500.0, 0.5, 0.5)
        assert fom_255 == fom_500, "TF-IDF above 255 should clamp to same score"

    def test_border_ratio_above_one_clamped(self):
        """border_ratio > 1 should produce border_score=0 (clamped by max(0,...))."""
        fom = compute_cluster_fom(128.0, 2.0, 0.5)
        # border_score = max(0, 1 - 2.0) = 0
        expected = 0.07 * (128 / 255) + 0.63 * 0.0 + 0.30 * 0.5
        assert fom == pytest.approx(expected, abs=1e-6)

    def test_cc_fraction_above_one_clamped(self):
        """cc_fraction > 1 should produce cc_score=0."""
        fom = compute_cluster_fom(128.0, 0.5, 2.0)
        expected = 0.07 * (128 / 255) + 0.63 * 0.5 + 0.30 * 0.0
        assert fom == pytest.approx(expected, abs=1e-6)


class TestFindMaskDebugAndClusterData:
    """Cover the debug-logging branches and return_cluster_data path
    of find_mask_by_spatial_tf_idf — reachable without mocks."""

    @pytest.fixture
    def text_image(self):
        """100×100 BGR image: vibrant red text on grey background."""
        img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
        cv2.putText(img, "TEST TEXT", (5, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 220), 2)
        return img

    @pytest.fixture
    def text_bbox(self):
        """(x, y, width, height) covering the text region."""
        return (5, 35, 90, 40)

    def test_find_mask_debug_paths(self, text_image, text_bbox):
        """debug=True exercises all if-debug logger.info branches."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, debug=True
        )
        assert isinstance(result, np.ndarray), "Expected ndarray mask"
        assert result.dtype == np.uint8

    def test_find_mask_returns_cluster_data(self, text_image, text_bbox):
        """return_cluster_data=True exercises the dict-construction path."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, return_cluster_data=True
        )
        assert isinstance(result, tuple), "Expected (mask, cluster_data) tuple"
        mask, cluster_data = result
        assert isinstance(mask, np.ndarray)
        assert isinstance(cluster_data, dict)
        for key in ("centers", "top_id", "bot_id", "color_radius", "bg_radius"):
            assert key in cluster_data, f"Missing key: {key}"
        assert cluster_data["centers"].shape == (4, 3)

    def test_find_mask_flat_image_normalization(self):
        """Solid-color image exercises the max_score==min_score guard (line 492)."""
        flat = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        result = find_mask_by_spatial_tf_idf(flat, bbox, num_clusters=4)
        assert isinstance(result, np.ndarray), "Should return mask without raising"
