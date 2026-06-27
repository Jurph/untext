"""Integration tests for ``-U`` (watermark auto-discovery) mode.

These exercise ``discover_watermark_candidates`` end-to-end against the built
corpus under ``tests/images`` and then verify the *consequences* of discovery:

  1. The discovered watermark template overlaps the known watermark
     (``test-watermark.png``) by a non-zero IoU.
  2. Aligning the discovered template back onto each watermarked image (via the
     same ORB pipeline the CLI uses) and inpainting through that mask reduces
     the error against the clean original inside the watermark region, while
     leaving pixels outside the mask byte-for-byte identical.
  3. A novel image (an archived singleton never seen by discovery) watermarked
     with the same DPS mascot can be cleaned with the discovered portrait
     template, again reducing in-region error.

Design notes (verified against source, not assumed):

  * ``discover_watermark_candidates(image_paths, debug_dir=None)`` returns a
    ``List[np.ndarray]`` of tight BGRA crops (largest family first), NOT
    full-frame masks.  See ``untextre/discovery.py``.
  * To turn a tight crop into a mask aligned to a full image we run ORB
    matching exactly as the CLI does, via ``orb_matcher.try_watermark_cascade``
    / ``find_known_mask_in_image``, which return a full-frame H×W binary mask.
  * Inpainting uses TELEA (pure OpenCV, deterministic, no GPU/model needed) so
    the suite runs under ``uv run pytest`` without LaMa initialisation.

Per CLAUDE.md: no invented statistical thresholds.  Accuracy is asserted as
``iou > 0`` (discovery found *something* overlapping the truth) and quality as
strict per-image improvement (``MAE_after < MAE_before``).  Actual IoU and MAE
values are logged so a human can calibrate a tighter bound later if desired.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest

from untextre import orb_matcher
from untextre.discovery import discover_watermark_candidates
from untextre.inpaint import inpaint_image
from untextre.orb_prep import prepare_candidate_bgra_for_orb
from untextre.utils import load_image

logger = logging.getLogger(__name__)

# ── Corpus geometry (measured from the built corpus, see images.md) ──────────
PORTRAIT_W, PORTRAIT_H = 1080, 1440
# The DPS mascot was composited at 259 px in the lower-left of every portrait.
# Measured placement: watermark left edge ~x=11, bottom margin ~10 px.
WM_SIZE_PORTRAIT = 259
WM_MARGIN = 10


# ─────────────────────────────────────────────────────────────────────────────
# Paths / fixtures
# ─────────────────────────────────────────────────────────────────────────────
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
def portrait_paths(images_root: Path) -> List[Path]:
    paths = sorted((images_root / "watermarked").glob("*.jpg"))
    # Restrict to the portrait bucket (1080x1440) so discovery sees a single,
    # homogeneous batch of 10 images, matching the spec's primary case.
    portraits = [p for p in paths if load_image(p).shape[:2] == (PORTRAIT_H, PORTRAIT_W)]
    if len(portraits) < 3:
        pytest.skip(f"need >=3 portrait images for discovery, found {len(portraits)}")
    return portraits


@pytest.fixture(scope="module")
def discovered_portrait_template(portrait_paths: List[Path]) -> np.ndarray:
    """Run -U discovery on the portrait batch; return the orb-prepped template.

    Returns the same on-disk form a real ``-U`` run would produce (so ORB
    matching in the tests sees exactly what the CLI would feed it).
    """
    candidates = discover_watermark_candidates(portrait_paths)
    if not candidates:
        pytest.skip("discovery returned no candidates on the portrait batch")
    template = prepare_candidate_bgra_for_orb(candidates[0])
    logger.info(
        "Discovered portrait template: %d candidate(s), best crop shape=%s, "
        "prepped shape=%s",
        len(candidates),
        candidates[0].shape,
        template.shape,
    )
    return template


# ─────────────────────────────────────────────────────────────────────────────
# Metric / geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
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


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error over all elements (computed in float to avoid wrap)."""
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def center_crop(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Scale then center-crop ``img`` to (tw, th) — same logic used to build the corpus.

    Scales so the image fully covers the target box (cover, not contain), then
    crops the centred (tw x th) window.
    """
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - tw) // 2
    y0 = (new_h - th) // 2
    return resized[y0:y0 + th, x0:x0 + tw].copy()


def alpha_composite(img_bgr: np.ndarray, wm_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
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


def _align_template_mask(
    image: np.ndarray, template_bgra: np.ndarray
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """ORB-align a BGRA template onto ``image``; return (full-frame mask, bbox).

    Uses ``find_known_mask_in_image`` with zero dilation so the mask reflects
    the discovered footprint itself, not an inflated region.  Returns None when
    no confident match is found.
    """
    result = orb_matcher.find_known_mask_in_image(
        image, template_bgra, dilation_pixels=0
    )
    if result is None:
        return None
    mask, bbox, _inliers = result
    return mask, bbox


# ─────────────────────────────────────────────────────────────────────────────
# 1. Watermark extraction accuracy
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_discovered_template_overlaps_known_watermark(
    discovered_portrait_template: np.ndarray, known_watermark: np.ndarray
) -> None:
    """The discovered template's footprint must overlap the known watermark.

    Both alpha masks are normalised to the known watermark's size before IoU so
    the comparison is shape-consistent.  We assert ``iou > 0`` (discovery
    located the real watermark, not noise elsewhere) and log the value.
    """
    known_alpha = (known_watermark[:, :, 3] > 127).astype(np.uint8)
    disc_alpha = (discovered_portrait_template[:, :, 3] > 127).astype(np.uint8)

    th, tw = known_alpha.shape
    disc_resized = cv2.resize(disc_alpha, (tw, th), interpolation=cv2.INTER_NEAREST)

    score = iou(known_alpha, disc_resized)
    logger.info(
        "Discovered-vs-known watermark IoU = %.4f "
        "(known alpha frac=%.3f, discovered alpha frac=%.3f)",
        score,
        float(known_alpha.mean()),
        float(disc_resized.mean()),
    )
    # EMPIRICAL - threshold not yet calibrated; > 0 proves discovery found the
    # real watermark region rather than spurious structure elsewhere.
    assert score > 0, (
        "Discovered template does not overlap the known watermark at all "
        "(IoU == 0) - discovery likely latched onto the wrong region."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. In-sample removal quality
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_in_sample_removal_improves_each_portrait(
    images_root: Path,
    portrait_paths: List[Path],
    discovered_portrait_template: np.ndarray,
) -> None:
    """Removal with the discovered mask must reduce in-region error per image.

    For every watermarked portrait that ORB can align the discovered template
    to, inpainting through that aligned mask must move masked pixels closer to
    the clean original (MAE_after < MAE_before), and must not touch any pixel
    outside the mask.
    """
    originals_dir = images_root / "originals" / "portraits"

    aligned = 0
    for wm_path in portrait_paths:
        orig_path = originals_dir / wm_path.name
        if not orig_path.exists():
            continue

        watermarked = load_image(wm_path)
        original = load_image(orig_path)
        assert watermarked.shape == original.shape, (
            f"shape mismatch for {wm_path.name}: "
            f"{watermarked.shape} vs {original.shape}"
        )

        aligned_mask = _align_template_mask(watermarked, discovered_portrait_template)
        if aligned_mask is None:
            logger.info("ORB could not align template to %s - skipping", wm_path.name)
            continue
        aligned += 1

        mask, bbox = aligned_mask
        mask_bool = mask > 0
        assert mask_bool.any(), f"aligned mask for {wm_path.name} is empty"

        mae_before = mae(watermarked[mask_bool], original[mask_bool])

        result = inpaint_image(watermarked, mask, bbox=bbox, method="telea")
        assert result.shape == watermarked.shape

        mae_after = mae(result[mask_bool], original[mask_bool])

        logger.info(
            "%s: masked px=%d, MAE_before=%.3f, MAE_after=%.3f",
            wm_path.name, int(mask_bool.sum()), mae_before, mae_after,
        )

        assert mae_after < mae_before, (
            f"{wm_path.name}: removal did not improve in-region error "
            f"(before={mae_before:.3f}, after={mae_after:.3f})"
        )

        # Pixels outside the mask must be untouched by inpainting.
        assert np.all(result[~mask_bool] == watermarked[~mask_bool]), (
            f"{wm_path.name}: inpainting altered pixels outside the mask"
        )

    # Discovery + ORB must succeed on at least one in-sample image, otherwise
    # this test would silently pass without exercising removal at all.
    assert aligned >= 1, (
        "ORB aligned the discovered template to zero portraits - "
        "removal quality was never exercised."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Novel-image removal (singleton)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_novel_image_removal_with_discovered_template(
    images_root: Path,
    known_watermark: np.ndarray,
    discovered_portrait_template: np.ndarray,
) -> None:
    """A never-seen image watermarked with the DPS mascot can be cleaned.

    The archived singleton is resized to the portrait geometry, the known
    watermark is composited at the same lower-left 259 px position used to build
    the corpus, then the discovered portrait template is aligned via ORB and the
    region is inpainted.  In-region error against the clean (pre-composite)
    image must drop.
    """
    singleton_path = images_root / "archived" / "klimt_the_kiss.jpg"
    if not singleton_path.exists():
        pytest.skip(f"singleton missing: {singleton_path}")

    clean = center_crop(load_image(singleton_path), PORTRAIT_W, PORTRAIT_H)
    assert clean.shape == (PORTRAIT_H, PORTRAIT_W, 3)

    wm_resized = cv2.resize(
        known_watermark, (WM_SIZE_PORTRAIT, WM_SIZE_PORTRAIT), interpolation=cv2.INTER_AREA
    )
    x = WM_MARGIN
    y = PORTRAIT_H - WM_SIZE_PORTRAIT - WM_MARGIN
    watermarked = alpha_composite(clean, wm_resized, x, y)

    aligned_mask = _align_template_mask(watermarked, discovered_portrait_template)
    if aligned_mask is None:
        pytest.skip(
            "ORB could not align the discovered template to the novel image - "
            "alignment quality is exercised by the in-sample test instead."
        )

    mask, bbox = aligned_mask
    mask_bool = mask > 0
    assert mask_bool.any(), "aligned mask on novel image is empty"

    mae_before = mae(watermarked[mask_bool], clean[mask_bool])
    result = inpaint_image(watermarked, mask, bbox=bbox, method="telea")
    mae_after = mae(result[mask_bool], clean[mask_bool])

    logger.info(
        "novel klimt_the_kiss: masked px=%d, MAE_before=%.3f, MAE_after=%.3f",
        int(mask_bool.sum()), mae_before, mae_after,
    )

    assert mae_after < mae_before, (
        f"novel-image removal did not improve in-region error "
        f"(before={mae_before:.3f}, after={mae_after:.3f})"
    )
    assert np.all(result[~mask_bool] == watermarked[~mask_bool]), (
        "inpainting altered pixels outside the mask on the novel image"
    )
