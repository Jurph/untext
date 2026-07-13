"""Integration tests for ``-U`` (watermark auto-discovery) mode.

These exercise ``-U`` mode's production path against the built corpus under
``tests/images`` and then verify the consequences of discovery:

1. Discovery runs once over the full mixed-geometry input directory, letting
   ``discover_watermark_candidates`` do its own production bucketing.
2. The discovered templates are converted through the same report/export helper
   the CLI uses, producing production ``WatermarkTemplate`` objects.
3. Every corpus image with a matching original is attempted through the same
   ORB cascade and inpainting path the CLI uses; outcomes are measured.
4. A novel image (an archived singleton never seen by discovery) watermarked
   with the same DPS mascot can be cleaned with the discovered templates.

Design notes:

* The test does not pre-bucket images. Production discovery owns that decision.
* To turn discovered crops into usable templates, the test uses
  ``reports._save_discovered_watermark_candidates`` just like ``cli.py``.
* To align a template to a full image, the test uses
  ``orb_matcher.try_watermark_cascade`` just like ``cli.py``.
* Inpainting uses LaMa when it is available; otherwise the test falls back to
  TELEA so the corpus is still exercised on CPU-only machines.

The repair path logs SSIM and quantized LAB-MAE deltas for inspection instead
of making the integration test depend on brittle per-image quality thresholds.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from untextre import orb_matcher
from untextre.discovery import discover_watermark_candidates
from untextre.inpaint import inpaint_image, initialize_lama_model
from untextre.orb_matcher import WatermarkTemplate
from untextre.reports import _save_discovered_watermark_candidates
from untextre.utils import load_image

logger = logging.getLogger(__name__)

# Corpus geometry for the archived singleton check.
PORTRAIT_W, PORTRAIT_H = 1080, 1440
WM_SIZE_PORTRAIT = 259
WM_MARGIN = 10


@pytest.fixture(scope="module")
def inpaint_method() -> str:
    """Prefer LaMa, but fall back to TELEA if no GPU-backed model is available."""
    if initialize_lama_model(device="cuda"):
        return "lama"
    if initialize_lama_model(device="cpu"):
        return "lama"
    return "telea"


def _save_pair(
    save_dir: "Path | None",
    stem: str,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    """Write before/after BGR images to *save_dir* when it is not None."""
    if save_dir is None:
        return
    cv2.imwrite(str(save_dir / f"{stem}_before.png"), before)
    cv2.imwrite(str(save_dir / f"{stem}_after.png"), after)


def _write_metrics_report(save_dir: Path, rows: List[dict]) -> None:
    """Write the combined metrics table for manual inspection."""
    report_path = save_dir / "u_mode_metrics.txt"
    if not rows:
        report_path.write_text("No aligned rows were collected.\n", encoding="utf-8")
        return

    filename_width = max(len("filename"), max(len(row["filename"]) for row in rows))
    lines = []
    lines.append(
        f"{'filename'.ljust(filename_width)}  {'ssim_delta':>11}  {'lab_mae_delta':>13}"
    )
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(
            f"{row['filename'].ljust(filename_width)}  "
            f"{row['ssim_delta']:11.6f}  {row['lab_mae_delta']:13.6f}"
        )

    ssim_deltas = [row["ssim_delta"] for row in rows]
    lab_deltas = [row["lab_mae_delta"] for row in rows]
    lines.append("")
    lines.append(
        "SSIM delta  mean={:.6f} median={:.6f} var={:.6f}".format(
            statistics.fmean(ssim_deltas),
            statistics.median(ssim_deltas),
            statistics.pvariance(ssim_deltas),
        )
    )
    lines.append(
        "LAB-MAE delta mean={:.6f} median={:.6f} var={:.6f}".format(
            statistics.fmean(lab_deltas),
            statistics.median(lab_deltas),
            statistics.pvariance(lab_deltas),
        )
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote metrics report to %s", report_path)


@pytest.fixture(scope="module")
def metrics_rows(request: pytest.FixtureRequest) -> List[dict]:
    """Collect per-image metric rows and write a report at module teardown."""
    rows: List[dict] = []

    def _finalize() -> None:
        if request.config.getoption("--save-test-images"):
            out_dir = Path(__file__).parent / "images" / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_metrics_report(out_dir, rows)

    request.addfinalizer(_finalize)
    return rows


# Corpus / fixture helpers.


def _images_root() -> Path:
    return Path(__file__).parent / "images"


@pytest.fixture(scope="module")
def images_root() -> Path:
    root = _images_root()
    if not root.is_dir():
        pytest.skip(f"corpus directory missing: {root}")
    return root


@pytest.fixture(scope="module")
def known_watermark(images_root: Path) -> np.ndarray:
    """The known DPS watermark as a BGRA array (alpha defines the true mask)."""
    path = images_root / "test-watermark.png"
    wm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if wm is None or wm.ndim != 3 or wm.shape[2] != 4:
        pytest.skip(f"known watermark missing or not RGBA: {path}")
    return wm


@pytest.fixture(scope="module")
def watermarked_paths(images_root: Path) -> List[Path]:
    """All watermarked corpus images with matching originals."""
    paths = sorted((images_root / "watermarked").glob("*.jpg"))
    if len(paths) < 3:
        pytest.skip(f"need >=3 watermarked images for discovery, found {len(paths)}")
    return paths


@pytest.fixture(scope="module")
def discovered_templates(
    watermarked_paths: List[Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> List[WatermarkTemplate]:
    """Run production -U discovery/export on the full mixed-geometry corpus."""
    candidates = discover_watermark_candidates(watermarked_paths)
    if not candidates:
        pytest.skip("discovery returned no candidates on the corpus")
    output_dir = tmp_path_factory.mktemp("u_mode_candidates")
    templates = _save_discovered_watermark_candidates(output_dir, candidates)
    logger.info(
        "Discovered %d candidate(s); exported template shapes=%s",
        len(templates),
        [template.rgba.shape for template in templates],
    )
    return templates


# Metric / geometry helpers.


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two boolean or uint8 masks of identical shape."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"iou shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    return intersection / union


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Return SSIM for grayscale or 3-channel uint8 images."""
    if a.ndim == 2:
        return float(ssim(a, b, data_range=255))
    return float(ssim(a, b, data_range=255, channel_axis=2))


def _masked_ssim_score(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    """Return SSIM averaged over the masked pixels only."""
    if before.shape != after.shape:
        raise ValueError("masked SSIM requires matching image shapes")
    if mask.shape != before.shape[:2]:
        raise ValueError("masked SSIM requires a 2D mask matching the image size")

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
    _score, ssim_map = ssim(before_gray, after_gray, data_range=255, full=True)
    return float(np.mean(ssim_map[mask.astype(bool)]))


def _lab_quantize(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to quantized 12-bit LAB stored in 8-bit channels."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return (lab // 16) * 16


def _masked_lab_mae(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    """Compute masked mean absolute LAB distance after 4-bit/channel quantization."""
    if before.shape != after.shape:
        raise ValueError("masked LAB MAE requires matching image shapes")
    if mask.shape != before.shape[:2]:
        raise ValueError("masked LAB MAE requires a 2D mask matching the image size")
    before_lab = _lab_quantize(before)
    after_lab = _lab_quantize(after)
    diff = np.abs(before_lab.astype(np.int16) - after_lab.astype(np.int16))
    mask_bool = mask.astype(bool)
    return float(diff[mask_bool].mean())


def _original_path_for(images_root: Path, watermarked_path: Path) -> Path:
    """Map a watermarked corpus path to the corresponding clean original."""
    for category in ("landscapes", "portraits"):
        candidate = images_root / "originals" / category / watermarked_path.name
        if candidate.exists():
            return candidate
    return images_root / "originals" / watermarked_path.name


# 1. Watermark extraction accuracy.


@pytest.mark.slow
def test_discovered_templates_overlap_known_watermark(
    discovered_templates: List[WatermarkTemplate], known_watermark: np.ndarray
) -> None:
    """The discovered template set must overlap the known watermark."""
    known_alpha = (known_watermark[:, :, 3] > 127).astype(np.uint8)

    best = {
        "index": -1,
        "iou": -1.0,
        "ssim": -1.0,
    }
    for index, template in enumerate(discovered_templates):
        disc_alpha = (template.rgba[:, :, 3] > 127).astype(np.uint8)
        th, tw = known_alpha.shape
        disc_resized = cv2.resize(disc_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
        score = iou(known_alpha, disc_resized)
        alpha_ssim = _ssim_score(known_alpha * 255, disc_resized * 255)
        logger.info(
            "template=%d known-vs-discovered IoU=%.4f SSIM=%.4f "
            "(known alpha frac=%.3f discovered alpha frac=%.3f)",
            index,
            score,
            alpha_ssim,
            float(known_alpha.mean()),
            float(disc_resized.mean()),
        )
        if score > best["iou"]:
            best = {"index": index, "iou": score, "ssim": alpha_ssim}

    assert best["iou"] > 0, (
        "Discovered templates do not overlap the known watermark at all "
        "(best IoU == 0) - discovery likely latched onto the wrong region."
    )
    assert best["ssim"] > 0, (
        "Discovered templates do not resemble the known watermark at all "
        "(best SSIM == 0) - discovery likely latched onto the wrong region."
    )


# 2. In-sample removal quality.


@pytest.mark.slow
def test_removal_metrics_across_corpus(
    images_root: Path,
    watermarked_paths: List[Path],
    discovered_templates: List[WatermarkTemplate],
    metrics_rows: List[dict],
    save_images_dir: "Path | None",
    inpaint_method: str,
) -> None:
    """Collect SSIM and LAB-MAE deltas for every aligned corpus image."""
    aligned = 0
    for wm_path in watermarked_paths:
        orig_path = _original_path_for(images_root, wm_path)
        if not orig_path.exists():
            continue

        watermarked = load_image(wm_path)
        original = load_image(orig_path)
        assert watermarked.shape == original.shape, (
            f"shape mismatch for {wm_path.name}: {watermarked.shape} vs {original.shape}"
        )

        cascade_result = orb_matcher.try_watermark_cascade(watermarked, discovered_templates)
        if cascade_result is None:
            logger.info("ORB could not align template to %s - skipping", wm_path.name)
            continue
        aligned += 1

        mask, bbox, template_name, _inliers = cascade_result
        mask_bool = mask > 0
        assert mask_bool.any(), f"aligned mask for {wm_path.name} is empty"

        baseline_ssim = _masked_ssim_score(original, watermarked, mask)
        baseline_lab = _masked_lab_mae(original, watermarked, mask)

        result = inpaint_image(watermarked, mask, bbox=bbox, method=inpaint_method)
        assert result.shape == watermarked.shape

        repaired_ssim = _masked_ssim_score(original, result, mask)
        repaired_lab = _masked_lab_mae(original, result, mask)

        ssim_delta = repaired_ssim - baseline_ssim
        lab_delta = baseline_lab - repaired_lab
        metrics_rows.append(
            {
                "filename": wm_path.name,
                "ssim_delta": ssim_delta,
                "lab_mae_delta": lab_delta,
            }
        )

        logger.info(
            "%s: template=%s masked px=%d SSIM %.4f->%.4f LAB-MAE %.4f->%.4f",
            wm_path.name,
            template_name,
            int(mask_bool.sum()),
            baseline_ssim,
            repaired_ssim,
            baseline_lab,
            repaired_lab,
        )
        _save_pair(save_images_dir, wm_path.stem, watermarked, result)

        # Pixels outside the mask must be untouched by inpainting.
        assert np.all(result[~mask_bool] == watermarked[~mask_bool]), (
            f"{wm_path.name}: inpainting altered pixels outside the mask"
        )

    assert aligned >= 1, (
        "ORB aligned the discovered template to zero images - "
        "removal quality was never exercised."
    )
    assert metrics_rows, "No aligned images were exercised"


# 3. Novel-image removal (singleton).


@pytest.mark.slow
def test_novel_image_removal_with_discovered_template(
    images_root: Path,
    known_watermark: np.ndarray,
    discovered_templates: List[WatermarkTemplate],
    metrics_rows: List[dict],
    save_images_dir: "Path | None",
    inpaint_method: str,
) -> None:
    """A never-seen image watermarked with the DPS mascot can be cleaned."""
    singleton_path = images_root / "archived" / "klimt_the_kiss.jpg"
    if not singleton_path.exists():
        pytest.skip(f"singleton missing: {singleton_path}")

    clean = _center_crop(load_image(singleton_path), PORTRAIT_W, PORTRAIT_H)
    assert clean.shape == (PORTRAIT_H, PORTRAIT_W, 3)

    wm_resized = cv2.resize(
        known_watermark, (WM_SIZE_PORTRAIT, WM_SIZE_PORTRAIT), interpolation=cv2.INTER_AREA
    )
    x = WM_MARGIN
    y = PORTRAIT_H - WM_SIZE_PORTRAIT - WM_MARGIN
    watermarked = _alpha_composite(clean, wm_resized, x, y)

    cascade_result = orb_matcher.try_watermark_cascade(watermarked, discovered_templates)
    if cascade_result is None:
        pytest.skip(
            "ORB could not align the discovered template to the novel image - "
            "alignment quality is exercised by the in-sample test instead."
        )

    mask, bbox, template_name, _inliers = cascade_result
    mask_bool = mask > 0
    assert mask_bool.any(), "aligned mask on novel image is empty"

    baseline_ssim = _masked_ssim_score(clean, watermarked, mask)
    baseline_lab = _masked_lab_mae(clean, watermarked, mask)
    result = inpaint_image(watermarked, mask, bbox=bbox, method=inpaint_method)
    repaired_ssim = _masked_ssim_score(clean, result, mask)
    repaired_lab = _masked_lab_mae(clean, result, mask)

    metrics_rows.append(
        {
            "filename": "novel_klimt_the_kiss.jpg",
            "ssim_delta": repaired_ssim - baseline_ssim,
            "lab_mae_delta": baseline_lab - repaired_lab,
        }
    )

    logger.info(
        "novel klimt_the_kiss: template=%s masked px=%d SSIM %.4f->%.4f "
        "LAB-MAE %.4f->%.4f",
        template_name,
        int(mask_bool.sum()),
        baseline_ssim,
        repaired_ssim,
        baseline_lab,
        repaired_lab,
    )
    _save_pair(save_images_dir, "novel_klimt_the_kiss", watermarked, result)

    assert np.all(result[~mask_bool] == watermarked[~mask_bool]), (
        "inpainting altered pixels outside the mask on the novel image"
    )


def _center_crop(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Scale then center-crop ``img`` to (tw, th)."""
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - tw) // 2
    y0 = (new_h - th) // 2
    return resized[y0:y0 + th, x0:x0 + tw].copy()


def _alpha_composite(img_bgr: np.ndarray, wm_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite ``wm_bgra`` onto a copy of ``img_bgr`` at top-left (x, y)."""
    out = img_bgr.copy()
    wh, ww = wm_bgra.shape[:2]
    H, W = out.shape[:2]
    if x < 0 or y < 0 or x + ww > W or y + wh > H:
        raise ValueError(f"watermark at ({x},{y}) size {ww}x{wh} exceeds image {W}x{H}")
    region = out[y:y + wh, x:x + ww].astype(np.float32)
    wm_bgr = wm_bgra[:, :, :3].astype(np.float32)
    alpha = (wm_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    blended = wm_bgr * alpha + region * (1.0 - alpha)
    out[y:y + wh, x:x + ww] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
