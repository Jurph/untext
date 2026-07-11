"""Tests for untextre.orb_matcher helpers."""

import cv2
import numpy as np
import pytest

import untextre.orb_matcher as orb_matcher_mod
from untextre.orb_matcher import (
    WatermarkTemplate,
    find_known_mask_in_image,
    load_watermark_templates,
    try_watermark_cascade,
)

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
        assert templates[0].name == "logo.png"
        assert templates[0].rgba.shape == (8, 8, 4)
        assert len(templates[0].orb_variants) == 3
        assert templates[0].orb_variants[0].keypoint_count >= templates[0].orb_variants[-1].keypoint_count

    def test_directory_with_rgba_png_only_includes_png(self, tmp_path):
        rgba = np.zeros((6, 6, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        (tmp_path / "a.png").write_text("not an image")
        cv2.imwrite(str(tmp_path / "b.png"), rgba)
        (tmp_path / "c.txt").write_text("ignore")
        templates = load_watermark_templates(tmp_path)
        assert len(templates) == 1
        assert templates[0].name == "b.png"

    def test_rgb_only_png_skipped(self, tmp_path):
        """PNG with 3 channels (no alpha) is rejected by our RGBA check."""
        rgb = np.ones((6, 6, 3), dtype=np.uint8) * 128
        path = tmp_path / "rgb_only.png"
        cv2.imwrite(str(path), rgb)
        templates = load_watermark_templates(path)
        assert len(templates) == 0


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

    def test_all_templates_tried_best_wins(self, monkeypatch):
        """All templates are evaluated; the one with the most inliers is returned."""
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        templates = [
            WatermarkTemplate("a.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
            WatermarkTemplate("b.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
            WatermarkTemplate("c.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
        ]
        mask_b = np.ones((60, 60), dtype=np.uint8) * 128
        bbox_b = (10, 20, 15, 12)
        mask_c = np.ones((60, 60), dtype=np.uint8) * 255
        bbox_c = (5, 5, 20, 20)
        calls = []

        def fake_find_known_mask_in_image(
            _image,
            _tmpl,
            min_matches=6,
            dilation_pixels=7,
            prepared_variants=None,
            prepared_target=None,
        ):
            calls.append((min_matches, dilation_pixels))
            if len(calls) == 1:
                return None               # a: no match
            if len(calls) == 2:
                return mask_b, bbox_b, 5  # b: weak match
            if len(calls) == 3:
                return mask_c, bbox_c, 10 # c: stronger match — should win
        monkeypatch.setattr(
            orb_matcher_mod,
            "prepare_target_orb_features",
            lambda _image: (tuple(), np.ones((1, 32), dtype=np.uint8)),
        )
        monkeypatch.setattr(
            orb_matcher_mod,
            "find_known_mask_in_image",
            fake_find_known_mask_in_image,
        )
        result = try_watermark_cascade(image, templates, min_matches=9, dilation_pixels=5)

        assert result is not None
        mask, bbox, template_name = result
        assert template_name == "c.png", "Template with more inliers should win"
        np.testing.assert_array_equal(mask, mask_c)
        assert bbox == bbox_c
        assert len(calls) == 3, "All three templates must be tried"
        assert calls == [(9, 5), (9, 5), (9, 5)]
 
    def test_target_orb_extracted_once_for_multi_template_input(self, monkeypatch):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        templates = [
            WatermarkTemplate("a.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
            WatermarkTemplate("b.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
            WatermarkTemplate("c.png", np.zeros((8, 8, 4), dtype=np.uint8), ()),
        ]
        prepared_target = (tuple(), np.ones((1, 32), dtype=np.uint8))
        prepare_calls = []
        find_calls = []
 
        def fake_prepare_target_orb_features(_image):
            prepare_calls.append("called")
            return prepared_target
 
        def fake_find_known_mask_in_image(
            _image,
            _tmpl,
            min_matches=6,
            dilation_pixels=15,
            prepared_variants=None,
            prepared_target=None,
        ):
            find_calls.append(prepared_target)
            return None
 
        monkeypatch.setattr(
            orb_matcher_mod,
            "prepare_target_orb_features",
            fake_prepare_target_orb_features,
        )
        monkeypatch.setattr(
            orb_matcher_mod,
            "find_known_mask_in_image",
            fake_find_known_mask_in_image,
        )
 
        assert try_watermark_cascade(image, templates) is None
        assert prepare_calls == ["called"]
        assert len(find_calls) == 3
        assert all(call is prepared_target for call in find_calls)


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

    def test_stretch_above_1_20_rejected(self, monkeypatch):
        """A 1.21x axis stretch should be rejected as a spurious template match."""
        stretch = np.array([[1.21, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (stretch, inlier_mask),
        )
        rng = np.random.RandomState(14)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255

        result = find_known_mask_in_image(target, known_rgba, min_matches=3)

        assert result is None

    def test_rotation_above_0_4_degrees_rejected(self, monkeypatch):
        """A 0.5 degree template rotation should be rejected as spurious."""
        angle = np.deg2rad(0.5)
        rotated = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
        ], dtype=np.float64)
        inlier_mask = np.ones((50, 1), dtype=np.uint8)
        monkeypatch.setattr(
            cv2, "estimateAffine2D",
            lambda *args, **kwargs: (rotated, inlier_mask),
        )
        rng = np.random.RandomState(15)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        known = target[:100, :100].copy()
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = known
        known_rgba[:, :, 3] = 255

        result = find_known_mask_in_image(target, known_rgba, min_matches=3)

        assert result is None

    def test_majority_image_mask_rejected(self, monkeypatch):
        """A known-mask match covering most of the image is not a watermark."""
        target = np.zeros((100, 100, 3), dtype=np.uint8)
        known_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        known_rgba[:, :, :3] = 200
        known_rgba[:80, :80, 3] = 255

        alpha = known_rgba[:, :, 3].copy()
        variant = type("Variant", (), {})()
        variant.name = "outside_0"
        variant.outside_value = 0
        variant.alpha = alpha
        variant.gray = np.zeros((100, 100), dtype=np.uint8)
        variant.keypoints = tuple(cv2.KeyPoint(float(i), float(i), 1) for i in range(8))
        variant.descriptors = np.ones((8, 32), dtype=np.uint8)
        variant.keypoint_count = 8
        target_keypoints = tuple(cv2.KeyPoint(float(i), float(i), 1) for i in range(8))
        target_descriptors = np.ones((8, 32), dtype=np.uint8)

        class FakeORB:
            def detectAndCompute(self, _image, _mask):
                return list(target_keypoints), target_descriptors

        class FakeMatcher:
            def knnMatch(self, _descriptors, _target_descriptors, k=2):
                assert k == 2
                return [
                    [
                        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=1),
                        cv2.DMatch(_queryIdx=i, _trainIdx=(i + 1) % 8, _distance=100),
                    ]
                    for i in range(8)
                ]

        monkeypatch.setattr(orb_matcher_mod, "build_candidate_orb_variants", lambda _bgra: [variant], raising=False)
        monkeypatch.setattr(orb_matcher_mod, "create_orb_detector", lambda: FakeORB(), raising=False)
        monkeypatch.setattr(cv2, "BFMatcher", lambda *args, **kwargs: FakeMatcher())
        monkeypatch.setattr(
            cv2,
            "estimateAffine2D",
            lambda *args, **kwargs: (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.ones((8, 1), dtype=np.uint8),
            ),
        )

        result = find_known_mask_in_image(target, known_rgba, min_matches=6, dilation_pixels=0)

        assert result is None

    def test_known_mask_builds_orb_variants_before_matching(self, monkeypatch):
        target = np.zeros((80, 100, 3), dtype=np.uint8)
        known_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        known_rgba[4:16, 4:16, :3] = 200
        known_rgba[4:16, 4:16, 3] = 255

        prepared_mask = np.zeros((20, 20), dtype=np.uint8)
        prepared_mask[6:14, 6:14] = 255
        builder_called = {"value": False}
        orb_masks = []

        def fake_build_candidate_orb_variants(_bgra):
            builder_called["value"] = True
            variant = type("Variant", (), {})()
            variant.name = "outside_0"
            variant.outside_value = 0
            variant.alpha = prepared_mask
            variant.gray = np.zeros((20, 20), dtype=np.uint8)
            variant.keypoints = tuple()
            variant.descriptors = None
            variant.keypoint_count = 0
            return [variant]

        class FakeORB:
            def detectAndCompute(self, _image, mask):
                orb_masks.append(None if mask is None else mask.copy())
                return [], None

        monkeypatch.setattr(orb_matcher_mod, "build_candidate_orb_variants", fake_build_candidate_orb_variants)
        monkeypatch.setattr(cv2.ORB, "create", lambda *args, **kwargs: FakeORB())

        result = find_known_mask_in_image(target, known_rgba)

        assert result is None
        assert builder_called["value"] is True
        assert len(orb_masks) >= 1
        # find_known_mask_in_image passes an explicit all-255 "no restriction"
        # mask instead of None (the cv2 stub rejects None; an all-255 mask is
        # the behaviorally-equivalent, stub-satisfying replacement).
        assert orb_masks[0] is not None
        assert np.all(orb_masks[0] == 255)

    def test_known_mask_falls_back_to_later_prepared_variant(self, monkeypatch):
        target = np.zeros((80, 100, 3), dtype=np.uint8)
        known_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        known_rgba[4:16, 4:16, :3] = 200
        known_rgba[4:16, 4:16, 3] = 255

        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[4:16, 4:16] = 255
        weak = type("Variant", (), {})()
        weak.name = "outside_0"
        weak.outside_value = 0
        weak.alpha = alpha
        weak.gray = np.zeros((20, 20), dtype=np.uint8)
        weak.keypoints = tuple(cv2.KeyPoint(float(i), float(i), 1) for i in range(4))
        weak.descriptors = np.ones((4, 32), dtype=np.uint8)
        weak.keypoint_count = 4

        strong = type("Variant", (), {})()
        strong.name = "outside_255"
        strong.outside_value = 255
        strong.alpha = alpha
        strong.gray = np.zeros((20, 20), dtype=np.uint8)
        strong.keypoints = tuple(cv2.KeyPoint(float(i + 5), float(i + 5), 1) for i in range(8))
        strong.descriptors = np.full((8, 32), 2, dtype=np.uint8)
        strong.keypoint_count = 8

        target_keypoints = tuple(cv2.KeyPoint(float(i + 20), float(i + 20), 1) for i in range(8))
        target_descriptors = np.full((8, 32), 2, dtype=np.uint8)

        class FakeORB:
            def detectAndCompute(self, _image, _mask):
                return list(target_keypoints), target_descriptors

        class FakeMatcher:
            def knnMatch(self, descriptors, _target_descriptors, k=2):
                assert k == 2
                if int(descriptors[0, 0]) != 2:
                    return []
                return [
                    [
                        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=1),
                        cv2.DMatch(_queryIdx=i, _trainIdx=(i + 1) % 8, _distance=100),
                    ]
                    for i in range(8)
                ]

        calls = []

        def fake_build_variants(_bgra):
            calls.append("built")
            return [weak, strong]

        monkeypatch.setattr(orb_matcher_mod, "build_candidate_orb_variants", fake_build_variants, raising=False)
        monkeypatch.setattr(orb_matcher_mod, "create_orb_detector", lambda: FakeORB(), raising=False)
        monkeypatch.setattr(cv2, "BFMatcher", lambda *args, **kwargs: FakeMatcher())
        monkeypatch.setattr(
            cv2,
            "estimateAffine2D",
            lambda *args, **kwargs: (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.ones((8, 1), dtype=np.uint8)),
        )

        result = find_known_mask_in_image(target, known_rgba, min_matches=6, dilation_pixels=0)

        assert calls == ["built"]
        assert result is not None
        mask, bbox, inliers = result
        assert inliers == 8
        assert bbox == (4, 4, 12, 12)
        assert np.count_nonzero(mask) > 0

