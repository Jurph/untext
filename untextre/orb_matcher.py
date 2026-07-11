"""ORB template matching helpers for known watermark masks."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .orb_prep import (
    CandidateOrbVariant,
    build_candidate_orb_variants,
    create_orb_detector,
)
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass(frozen=True)
class WatermarkTemplate:
    name: str
    rgba: np.ndarray
    orb_variants: tuple[CandidateOrbVariant, ...]


def _make_watermark_template(name: str, rgba: np.ndarray) -> WatermarkTemplate:
    return WatermarkTemplate(name, rgba, tuple(build_candidate_orb_variants(rgba)))

def load_watermark_templates(
    path: Path,
) -> List[WatermarkTemplate]:
    """Load RGBA watermark templates from a file or directory.

    Each template is an RGBA PNG where the alpha channel defines the mask.

    Args:
        path: Path to a single RGBA PNG or a directory containing them.

    Returns:
        Watermark templates sorted by filename.
        Returns an empty list if the path doesn't exist, is empty,
        or contains no valid RGBA PNGs.
    """
    templates: List[WatermarkTemplate] = []

    if not path.exists():
        return templates

    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in (".png", ".tif", ".tiff")
        )
    else:
        return templates

    for candidate in candidates:
        rgba = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning(f"Could not read {candidate.name}, skipping")
            continue
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            logger.warning(f"{candidate.name} is not RGBA ({rgba.ndim}D, {rgba.shape[-1] if rgba.ndim == 3 else '?'}ch), skipping")
            continue
        templates.append(_make_watermark_template(candidate.name, rgba))
        logger.debug(f"Loaded template: {candidate.name} ({rgba.shape[1]}x{rgba.shape[0]})")

    if templates:
        logger.info(f"Loaded {len(templates)} watermark template(s) from {path}")

    return templates

def find_known_mask_in_image(
    target_image: np.ndarray,
    known_mask_rgba: np.ndarray,
    min_matches: int = 6,
    dilation_pixels: int = 15,
    prepared_variants: Optional[tuple[CandidateOrbVariant, ...]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
    """Find a known watermark/logo by trying precomputed ORB reference variants."""
    if known_mask_rgba.shape[2] != 4:
        raise ValueError("Known mask must be RGBA image (4 channels)")

    variants = list(prepared_variants or build_candidate_orb_variants(known_mask_rgba))
    if not variants:
        logger.warning("No ORB variants available for known mask")
        return None

    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    orb = create_orb_detector()
    target_keypoints, target_descriptors = orb.detectAndCompute(
        target_gray, np.full(target_gray.shape, 255, dtype=np.uint8)
    )

    if target_descriptors is None or target_keypoints is None:
        logger.warning("Could not compute target ORB descriptors")
        return None

    if len(target_keypoints) < min_matches:
        logger.warning(
            f"Not enough keypoints: known={variants[0].keypoint_count}, "
            f"target={len(target_keypoints)}"
        )
        return None

    for variant in variants:
        known_keypoints = list(variant.keypoints)
        known_descriptors = variant.descriptors

        if known_descriptors is None:
            logger.warning(f"Could not compute ORB descriptors for {variant.name}")
            continue

        if len(known_keypoints) < min_matches:
            logger.warning(
                f"Not enough keypoints for {variant.name}: "
                f"known={len(known_keypoints)}, target={len(target_keypoints)}"
            )
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(known_descriptors, target_descriptors, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Lowe (2004) ratio test
                    good_matches.append(m)

        logger.info(
            f"ORB matching ({variant.name}): "
            f"{len(good_matches)} good matches (need {min_matches})"
        )
        if len(good_matches) < min_matches:
            logger.warning(f"Not enough good matches: {len(good_matches)} < {min_matches}")
            continue

        src_pts = np.array([known_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([target_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        M, inlier_mask = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if M is None:
            logger.warning("Could not compute affine transform")
            continue

        inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        linear = M[:2, :2]
        U, S, Vt = np.linalg.svd(linear)
        scale_major, scale_minor = float(S[0]), float(S[1])
        det = float(np.linalg.det(linear))
        angle = np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))

        logger.info(
            f"Affine transform ({variant.name}): {inliers} inliers, "
            f"scale=({scale_major:.3f}, {scale_minor:.3f}), "
            f"det={det:.3f}, rotation={angle:.1f} degrees"
        )

        if inliers < min_matches:
            logger.warning(f"Not enough inliers: {inliers} < {min_matches}")
            continue

        min_scale, max_scale = 0.05, 20.0
        # EMPIRICAL: pristine branding turns "knock-off" well before 6:5 stretch.
        # 1.20 keeps margin while rejecting transforms that no longer match the watermark.
        max_stretch = 1.20
        # EMPIRICAL: visual review of the has-text-2 template matches found the last
        # true positive at 0.4 degrees; larger rotations were false positives.
        max_rotation_degrees = 0.4
        if scale_major < min_scale or scale_minor < min_scale:
            logger.warning(f"Scale too small ({scale_major:.3f}, {scale_minor:.3f})")
            continue
        if scale_major > max_scale or scale_minor > max_scale:
            logger.warning(f"Scale too large ({scale_major:.3f}, {scale_minor:.3f})")
            continue
        if det < 0:
            logger.warning(f"Reflection detected (det={det:.3f})")
            continue
        if abs(angle) > max_rotation_degrees:
            logger.warning(
                f"Rotation too large ({angle:.1f} degrees) - likely spurious template match"
            )
            continue
        if scale_major / scale_minor > max_stretch:
            logger.warning(
                f"Excessive stretch ({scale_major / scale_minor:.1f}x) "
                "between axes - likely spurious"
            )
            continue

        h_target, w_target = target_image.shape[:2]
        warped_mask = cv2.warpAffine(
            variant.alpha,
            M,
            (w_target, h_target),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        _, binary_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

        if dilation_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1),
            )
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            logger.info(f"Applied {dilation_pixels}px dilation to mask")

        coords = cv2.findNonZero(binary_mask)
        if coords is None:
            logger.warning("Warped mask is empty")
            continue

        x, y, w, h = cv2.boundingRect(coords)
        bbox = (x, y, w, h)
        h_known, w_known = variant.alpha.shape[:2]
        scale_x = w / w_known if w_known > 0 else 1.0
        scale_y = h / h_known if h_known > 0 else 1.0
        logger.info(
            f"Known mask found at ({x}, {y}) size {w}x{h} "
            f"(scale: {scale_x:.2f}x, {scale_y:.2f}x, variant={variant.name})"
        )

        at_edge = []
        if x == 0:
            at_edge.append("left")
        if y == 0:
            at_edge.append("top")
        if x + w >= w_target:
            at_edge.append("right")
        if y + h >= h_target:
            at_edge.append("bottom")
        if at_edge:
            logger.info(f"Mask touches image edge ({', '.join(at_edge)}) - spillover clipped")

        mask_pixels = np.sum(binary_mask > 0)
        image_pixels = h_target * w_target
        # A legitimate watermark rarely covers half the image; beyond 50% the
        # match is almost certainly a false positive (background texture correlation).
        if mask_pixels >= image_pixels * 0.5:
            logger.warning(
                f"Mask covers {100 * mask_pixels / image_pixels:.1f}% of image "
                "- likely spurious template match"
            )
            continue

        logger.info(f"Mask covers {mask_pixels:,} pixels")
        return binary_mask, bbox, int(inliers)

    logger.info(f"No ORB variant matched (tried {len(variants)})")
    return None

def try_watermark_cascade(
    image: np.ndarray,
    templates: List[WatermarkTemplate],
    min_matches: int = 6,
    dilation_pixels: int = 15,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], str]]:
    """Try every watermark template against a loaded image; return the best match.

    All templates are evaluated.  The one with the most RANSAC inliers wins,
    ensuring that a stronger match later in the list beats a weaker earlier hit.

    This is the matching-only step with no file I/O.  The caller decides
    what to do with the result (inpaint, save, fall back to detection, etc.).

    Args:
        image: Target image (H×W×3 BGR), already loaded.
        templates: Watermark templates from load_watermark_templates().
        min_matches: Minimum ORB matches for a valid detection.
        dilation_pixels: Pixels to dilate the matched mask.

    Returns:
        Tuple of (mask, bbox, template_name) for the best-matching template, or None.
    """
    best: Optional[Tuple[np.ndarray, Tuple[int, int, int, int], str]] = None
    best_inliers = -1

    for template in templates:
        tmpl_name = template.name
        logger.info(f"Trying template: {tmpl_name}")
        result = find_known_mask_in_image(
            image, template.rgba,
            min_matches=min_matches,
            dilation_pixels=dilation_pixels,
            prepared_variants=template.orb_variants,
        )
        if result is not None:
            mask, bbox, inliers = result
            logger.info(f"Template {tmpl_name} matched with {inliers} inliers")
            if inliers > best_inliers:
                best = (mask, bbox, tmpl_name)
                best_inliers = inliers

    if best is not None:
        logger.info(f"Best template: {best[2]} ({best_inliers} inliers)")
        return best

    logger.info(f"No template matched (tried {len(templates)})")
    return None
