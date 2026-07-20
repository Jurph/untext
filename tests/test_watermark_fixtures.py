"""Pipeline tests using real fixture images.

Known-mask (SIFT) pipeline:
    Superimposes tests/images/test-watermark.png on test3-without-text.png
    at known positions/scales, runs SIFT detection + inpainting, and asserts
    SSIM of the result against the original (clean) image.

    Variants:
    - Lower-right corner, mild out-of-bounds (watermark partly clipped)
    - Dead center, 5% horizontal stretch
    - Near SIFT's minimum viable scale for this fixture

Consensus detection pipeline:
    Runs process_single_image on test3-with-text.png (the real production
    path: consensus detection -> spatial TF-IDF -> inpainting) and compares
    the output to test3-without-text.png via SSIM.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from untextre.sift_matcher import find_known_mask_in_image
from untextre.pipeline import process_single_image
from untextre.inpaint import inpaint_image
from untextre.utils import load_image


def _ssim_score(original: np.ndarray, cleaned: np.ndarray) -> float:
    """SSIM(original, cleaned); supports current and older skimage."""
    assert cleaned.shape == original.shape
    try:
        return float(ssim(original, cleaned, data_range=255, channel_axis=2))
    except TypeError:
        return float(ssim(original, cleaned, data_range=255, multichannel=True))


def _tests_images_dir() -> Path:
    return Path(__file__).resolve().parent / "images"


def _composite_watermark(
    base_bgr: np.ndarray,
    watermark_rgba: np.ndarray,
    x: int,
    y: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """Alpha-blend watermark onto base at (x, y) with optional scale.

    Handles out-of-bounds: only the overlapping region is composited;
    watermark is cropped and roi is limited to image bounds.

    Returns:
        (composited_bgr, placement_bbox, ground_truth_mask) where
        placement_bbox is (x, y, w, h) in base coordinates and
        ground_truth_mask is H×W uint8, 255 where watermark was drawn.
    """
    if watermark_rgba.shape[2] != 4:
        raise ValueError("Watermark must be RGBA")
    h_base, w_base = base_bgr.shape[:2]
    h_wm, w_wm = watermark_rgba.shape[:2]

    if scale_x != 1.0 or scale_y != 1.0:
        new_w = max(1, int(round(w_wm * scale_x)))
        new_h = max(1, int(round(h_wm * scale_y)))
        wm_resized = cv2.resize(
            watermark_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        wm_resized = watermark_rgba
    h_wm, w_wm = wm_resized.shape[:2]

    # Overlap region in base image coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_base, x + w_wm)
    y2 = min(h_base, y + h_wm)
    if x1 >= x2 or y1 >= y2:
        return base_bgr.copy(), (x1, y1, 0, 0), np.zeros((h_base, w_base), dtype=np.uint8)

    # Corresponding region in watermark
    wx1 = x1 - x
    wy1 = y1 - y
    wx2 = wx1 + (x2 - x1)
    wy2 = wy1 + (y2 - y1)

    out = base_bgr.copy()
    roi = out[y1:y2, x1:x2]
    wm_bgr = wm_resized[:, :, :3]
    wm_alpha = wm_resized[wy1:wy2, wx1:wx2, 3].astype(np.float32) / 255.0
    if wm_alpha.ndim == 2:
        wm_alpha = wm_alpha[:, :, np.newaxis]
    blended = (
        roi.astype(np.float32) * (1 - wm_alpha)
        + wm_bgr[wy1:wy2, wx1:wx2].astype(np.float32) * wm_alpha
    )
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Ground-truth mask: 255 where we drew (alpha > 127)
    gt_mask = np.zeros((h_base, w_base), dtype=np.uint8)
    wm_alpha_1ch = wm_resized[wy1:wy2, wx1:wx2, 3]
    gt_mask[y1:y2, x1:x2] = (wm_alpha_1ch > 127).astype(np.uint8) * 255

    placement_bbox = (x1, y1, x2 - x1, y2 - y1)
    return out, placement_bbox, gt_mask


@pytest.fixture(scope="module")
def watermark_fixture_images():
    """Load base image and watermark; skip if either is missing."""
    img_dir = _tests_images_dir()
    base_path = img_dir / "test3-without-text.png"
    wm_path = img_dir / "test-watermark.png"
    if not base_path.exists() or not wm_path.exists():
        pytest.skip(
            "Requires tests/images/test3-without-text.png and test-watermark.png"
        )
    base = load_image(base_path)
    if base is None:
        pytest.skip("Could not load test3-without-text.png")
    wm = cv2.imread(str(wm_path), cv2.IMREAD_UNCHANGED)
    if wm is None or wm.ndim != 3 or wm.shape[2] != 4:
        pytest.skip("test-watermark.png must be RGBA")
    return {"base": base, "watermark_rgba": wm, "base_path": base_path}


@pytest.mark.slow
class TestWatermarkCompositedPipeline:
    """Place watermark at known variants, detect + inpaint, compare to original."""

    def _run_detect_inpaint_ssim(
        self,
        composited: np.ndarray,
        original: np.ndarray,
        watermark_rgba: np.ndarray,
        min_ssim: float = 0.92,
        method: str = "telea",
    ) -> float:
        """Run SIFT detection, inpaint, return SSIM(cleaned, original)."""
        result = find_known_mask_in_image(
            composited, watermark_rgba, dilation_pixels=7
        )
        assert result is not None, "Known-mask detection should find the watermark"
        mask, bbox, _inliers = result
        cleaned = inpaint_image(
            composited, mask, bbox=bbox, method=method, auto_retry=False
        )
        score = _ssim_score(original, cleaned)
        assert score >= min_ssim, (
            f"SSIM {score:.4f} below {min_ssim} — inpainting did not restore original"
        )
        return score

    def test_lower_right_mild_out_of_bounds(self, watermark_fixture_images):
        """Watermark hard against lower-right corner, mild clip."""
        base = watermark_fixture_images["base"]
        wm = watermark_fixture_images["watermark_rgba"]
        h_base, w_base = base.shape[:2]
        h_wm, w_wm = wm.shape[:2]
        # Place so ~50px spill off the right and bottom edges
        x = w_base - w_wm + 50
        y = h_base - h_wm + 50
        composited, placement_bbox, _ = _composite_watermark(base, wm, x, y)
        assert placement_bbox[2] > 0 and placement_bbox[3] > 0
        score = self._run_detect_inpaint_ssim(composited, base, wm)
        assert score > 0.90

    def test_center_5pct_horizontal_scale(self, watermark_fixture_images):
        """Watermark dead center, scaled 5% horizontally."""
        base = watermark_fixture_images["base"]
        wm = watermark_fixture_images["watermark_rgba"]
        h_base, w_base = base.shape[:2]
        w_scaled = int(round(wm.shape[1] * 1.05))
        h_scaled = wm.shape[0]
        x = (w_base - w_scaled) // 2
        y = (h_base - h_scaled) // 2
        composited, placement_bbox, _ = _composite_watermark(
            base, wm, x, y, scale_x=1.05, scale_y=1.0
        )
        assert placement_bbox[2] > 0 and placement_bbox[3] > 0
        score = self._run_detect_inpaint_ssim(composited, base, wm)
        assert score > 0.90

    def test_near_sift_minimum_viable_scale(self, watermark_fixture_images):
        """Watermark at 0.35x.

        This test sits near the small-template boundary to catch regressions in
        SIFT matching or the affine transform pipeline.
        """
        base = watermark_fixture_images["base"]
        wm = watermark_fixture_images["watermark_rgba"]
        h_base, w_base = base.shape[:2]
        scale = 0.35
        w_scaled = int(round(wm.shape[1] * scale))
        h_scaled = int(round(wm.shape[0] * scale))
        x = (w_base - w_scaled) // 2
        y = (h_base - h_scaled) // 2
        composited, placement_bbox, _ = _composite_watermark(
            base, wm, x, y, scale_x=scale, scale_y=scale
        )
        assert placement_bbox[2] > 0 and placement_bbox[3] > 0
        score = self._run_detect_inpaint_ssim(composited, base, wm)
        assert score > 0.95  # Tiny region; inpainting should nearly perfectly restore


# =========================================================================
# Consensus detection end-to-end: the production code path
# =========================================================================


@pytest.mark.slow
class TestConsensusDetectionE2E:
    """Full pipeline test: consensus detection -> TF-IDF -> inpainting.

    Uses the real paired test images:
        - test3-with-text.png (has text watermarks)
        - test3-without-text.png (clean reference)

    Runs process_single_image through the *actual* detection path (EAST,
    YOLO11x, EasyOCR consensus) with TELEA inpainting.  Compares the cleaned
    output to the clean reference via SSIM.

    Calibrated 2026-02-16: SSIM was 0.9956 with telea, 4 consensus boxes,
    no failover.  Floor set to 0.95 to allow for model-version variance.
    """

    def test_consensus_pipeline_restores_clean_image(self):
        img_dir = _tests_images_dir()
        dirty_path = img_dir / "test3-with-text.png"
        clean_path = img_dir / "test3-without-text.png"
        if not dirty_path.exists() or not clean_path.exists():
            pytest.skip("Requires test3-with-text.png and test3-without-text.png")

        from untextre.pipeline import initialize_consensus_models

        initialize_consensus_models(device="cuda")

        clean = load_image(clean_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            timings = process_single_image(
                image_path=dirty_path,
                output_dir=out_dir,
                method="telea",
                expand_bboxes=True,
                auto_retry=True,
            )

            assert timings is not None, "process_single_image returned None"
            assert not timings.get("skipped", False), "Pipeline skipped the image"
            assert timings["consensus_boxes_count"] > 0, "No consensus boxes detected"

            result_path = out_dir / "test3-with-text_clean.png"
            assert result_path.exists(), f"Expected output at {result_path}"

            result = load_image(result_path)
            score = _ssim_score(clean, result)
            assert score >= 0.95, (
                f"SSIM {score:.4f} — consensus pipeline did not restore the clean image"
            )
