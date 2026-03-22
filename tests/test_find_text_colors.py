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
from untextre.find_text_colors import (
    color_guided_expand,
    compute_cluster_fom,
    find_mask_by_spatial_tf_idf,
    grabcut_refine,
)


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
        """(x, y, width, height) covering the text region — touches left+top edges
        so the border-exclusion debug branch (lines 528-537) fires."""
        return (0, 0, 100, 75)

    def test_find_mask_debug_paths(self, text_image, text_bbox):
        """debug=True exercises all if-debug logger.info branches."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, debug=True
        )
        assert isinstance(result, np.ndarray), "Expected ndarray mask"
        assert result.dtype == np.uint8

    def test_find_mask_debug_with_target_color(self, text_image, text_bbox):
        """debug=True + target_color exercises the target-color debug branches."""
        result = find_mask_by_spatial_tf_idf(
            text_image, text_bbox, num_clusters=4, debug=True,
            target_color=(0, 0, 220),
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


class TestGrabcutRefine:
    """Cover grabcut_refine branches — pure cv2/numpy, no ML."""

    @pytest.fixture
    def roi(self):
        """50×50 BGR ROI: red block on grey background."""
        img = np.full((50, 50, 3), (128, 128, 128), dtype=np.uint8)
        img[10:40, 10:40] = (0, 0, 220)
        return img

    @pytest.fixture
    def fom_mask(self):
        """Binary mask matching roi: foreground where the red block is."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        return mask

    def test_normal_path(self, roi, fom_mask):
        """Standard refinement returns uint8 ndarray of same shape."""
        result = grabcut_refine(roi, fom_mask)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == fom_mask.shape

    def test_too_small_returns_unchanged(self):
        """2×2 ROI is below the 3×3 minimum — returns fom_mask unchanged."""
        tiny_roi = np.zeros((2, 2, 3), dtype=np.uint8)
        tiny_mask = np.zeros((2, 2), dtype=np.uint8)
        result = grabcut_refine(tiny_roi, tiny_mask)
        np.testing.assert_array_equal(result, tiny_mask)

    def test_no_dilation(self, roi, fom_mask):
        """dilation_px=0 skips the post-GrabCut dilation step."""
        result = grabcut_refine(roi, fom_mask, dilation_px=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == fom_mask.shape

    def test_debug_paths(self, roi, fom_mask):
        """debug=True exercises all GrabCut debug logging branches."""
        result = grabcut_refine(roi, fom_mask, debug=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == fom_mask.shape


class TestColorGuidedExpand:
    """Cover color_guided_expand branches — pure cv2/numpy, no ML."""

    @pytest.fixture
    def expand_setup(self):
        """Full 100×100 image + confirmed_mask + real K-means cluster data."""
        img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
        cv2.putText(img, "TEST TEXT", (5, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 220), 2)
        bbox = (0, 0, 100, 75)
        region_mask, cluster_data = find_mask_by_spatial_tf_idf(
            img, bbox, num_clusters=4, return_cluster_data=True
        )
        full_mask = np.zeros((100, 100), dtype=np.uint8)
        full_mask[0:region_mask.shape[0], 0:region_mask.shape[1]] = region_mask
        return img, bbox, full_mask, cluster_data

    def test_normal_path(self, expand_setup):
        """Returns a mask that is a superset of confirmed_mask."""
        img, bbox, full_mask, cd = expand_setup
        result = color_guided_expand(
            img, bbox, full_mask,
            cd["centers"], cd["top_id"], cd["bot_id"],
            cd["color_radius"], cd["bg_radius"],
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert np.all(result[full_mask == 255] == 255), "Result must include all confirmed pixels"

    def test_tiny_bbox_returns_unchanged(self, expand_setup):
        """1×1 bbox expands to 2×2 which is below the 3px minimum — returns confirmed_mask."""
        img, _, full_mask, cd = expand_setup
        tiny_bbox = (50, 50, 1, 1)
        result = color_guided_expand(
            img, tiny_bbox, full_mask,
            cd["centers"], cd["top_id"], cd["bot_id"],
            cd["color_radius"], cd["bg_radius"],
        )
        np.testing.assert_array_equal(result, full_mask)

    def test_no_fgd_pixels_returns_unchanged(self, expand_setup):
        """Impossible centroid → no pixels within color_radius → returns confirmed_mask."""
        img, bbox, full_mask, cd = expand_setup
        # Centroid far outside the 0-255 gamut so no pixel is within color_radius
        impossible_centers = np.array(
            [[300.0, 300.0, 300.0], [128.0, 128.0, 128.0]], dtype=np.float32
        )
        result = color_guided_expand(
            img, bbox, full_mask,
            impossible_centers, 0, 1,
            color_radius=10.0,
            bg_radius=30.0,
        )
        np.testing.assert_array_equal(result, full_mask)

    def test_debug_paths(self, expand_setup):
        """debug=True exercises all color-expand debug logging branches."""
        img, bbox, full_mask, cd = expand_setup
        result = color_guided_expand(
            img, bbox, full_mask,
            cd["centers"], cd["top_id"], cd["bot_id"],
            cd["color_radius"], cd["bg_radius"],
            debug=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
