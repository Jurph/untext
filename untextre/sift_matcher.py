"""SIFT template matching helpers for known watermark masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import setup_logger

logger = setup_logger(__name__)

SIFT_CONTRAST_THRESHOLD = 0.02
SIFT_EDGE_THRESHOLD = 8
SIFT_RATIO = 0.75
SIFT_MIN_MATCHES = 5
SIFT_RANSAC_REPROJ_THRESHOLD = 5.0
SIFT_MAX_KEYPOINTS = 10_000

# Corner "push hard" cascade for scoped -K matching: search cheap corners first,
# then the full frame, with early exit. Window = multiple x the template's mark;
# marks whose window would not fit a corner are tried full-frame only. Only worth
# it for a few user-specified templates (cost is O(templates x regions)), never
# the whole watermarks/ library.
CORNER_SEARCH_ORDER: Tuple[str, ...] = ("LR", "LL", "full", "UR", "UL")
CORNER_WINDOW_MULTIPLE = 2
CORNER_ELIGIBLE_FRACTION = 0.6


@dataclass(frozen=True)
class TemplateSiftFeatures:
    gray: np.ndarray
    alpha: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoints)



@dataclass(frozen=True)
class SiftMaskMatch:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    inliers: int
    good_matches: int
    template_keypoints: int

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / max(1, self.good_matches)

    @property
    def template_coverage(self) -> float:
        return self.inliers / max(1, self.template_keypoints)

    def __iter__(self):
        yield self.mask
        yield self.bbox
        yield self.inliers

@dataclass(frozen=True)
class WatermarkTemplate:
    name: str
    rgba: np.ndarray
    sift_features: TemplateSiftFeatures


@dataclass(frozen=True)
class SiftMatchCandidate:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    template_name: str
    inliers: int
    inlier_ratio: float
    template_coverage: float

    @property
    def rank_key(self) -> tuple[int, float, float]:
        return (self.inliers, self.inlier_ratio, self.template_coverage)


PreparedTargetSift = Tuple[tuple[cv2.KeyPoint, ...], np.ndarray]


def create_sift_detector() -> cv2.SIFT:
    return cv2.SIFT_create(  # type: ignore[attr-defined]
        contrastThreshold=SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=SIFT_EDGE_THRESHOLD,
    )


def _cap_keypoints(
    keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint],
    descriptors: np.ndarray | None,
    max_keypoints: int = SIFT_MAX_KEYPOINTS,
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    if descriptors is None or len(keypoints) <= max_keypoints:
        return tuple(keypoints), descriptors
    order = sorted(range(len(keypoints)), key=lambda index: keypoints[index].response, reverse=True)
    keep = np.array(order[:max_keypoints], dtype=np.int32)
    return tuple(keypoints[int(index)] for index in keep), descriptors[keep]


def _rgba_to_gray_and_alpha(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Known mask must be RGBA/BGRA image (4 channels)")
    alpha = rgba[:, :, 3]
    bgr = rgba[:, :, :3].copy()
    bgr[alpha == 0] = 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, alpha


def build_template_sift_features(rgba: np.ndarray) -> TemplateSiftFeatures:
    gray, alpha = _rgba_to_gray_and_alpha(rgba)
    sift = create_sift_detector()
    keypoints, descriptors = sift.detectAndCompute(gray, alpha)
    capped_keypoints, capped_descriptors = _cap_keypoints(tuple(keypoints or ()), descriptors)
    return TemplateSiftFeatures(gray, alpha, capped_keypoints, capped_descriptors)


def _make_watermark_template(name: str, rgba: np.ndarray) -> WatermarkTemplate:
    return WatermarkTemplate(name, rgba, build_template_sift_features(rgba))


def load_watermark_templates(path: Path) -> List[WatermarkTemplate]:
    """Load RGBA watermark templates from a file or directory."""
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
            logger.warning(
                f"{candidate.name} is not RGBA ({rgba.ndim}D, "
                f"{rgba.shape[-1] if rgba.ndim == 3 else '?'}ch), skipping"
            )
            continue
        template = _make_watermark_template(candidate.name, rgba)
        if template.sift_features.descriptors is None:
            logger.warning(f"Could not compute SIFT descriptors for {candidate.name}, skipping")
            continue
        templates.append(template)
        logger.debug(
            f"Loaded template: {candidate.name} "
            f"({rgba.shape[1]}x{rgba.shape[0]}, sift={template.sift_features.keypoint_count})"
        )

    if templates:
        logger.info(f"Loaded {len(templates)} watermark template(s) from {path}")

    return templates


def prepare_target_sift_features(target_image: np.ndarray) -> Optional[PreparedTargetSift]:
    """Extract capped SIFT features for one target image once, for reuse across templates."""
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    sift = create_sift_detector()
    target_keypoints, target_descriptors = sift.detectAndCompute(
        target_gray,
        np.full(target_gray.shape, 255, dtype=np.uint8),
    )
    if target_descriptors is None or target_keypoints is None:
        logger.warning("Could not compute target SIFT descriptors")
        return None
    capped_keypoints, capped_descriptors = _cap_keypoints(tuple(target_keypoints), target_descriptors)
    if capped_descriptors is None:
        return None
    return capped_keypoints, capped_descriptors


def _good_sift_matches(
    template_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = matcher.knnMatch(template_descriptors, target_descriptors, k=2)
    good_matches: list[cv2.DMatch] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)
    return good_matches


def _affine_is_plausible(matrix: np.ndarray) -> bool:
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    scale_x = float(np.hypot(a, c))
    scale_y = float(np.hypot(b, d))
    if scale_x <= 1e-6 or scale_y <= 1e-6:
        return False
    min_scale, max_scale = 0.05, 20.0
    max_stretch = 2.5
    stretch = max(scale_x / scale_y, scale_y / scale_x)
    det = float(np.linalg.det(matrix[:2, :2]))
    if det < 0:
        logger.warning(f"Reflection detected (det={det:.3f})")
        return False
    if scale_x < min_scale or scale_y < min_scale:
        logger.warning(f"Scale too small ({scale_x:.3f}, {scale_y:.3f})")
        return False
    if scale_x > max_scale or scale_y > max_scale:
        logger.warning(f"Scale too large ({scale_x:.3f}, {scale_y:.3f})")
        return False
    if stretch > max_stretch:
        logger.warning(f"Excessive stretch ({stretch:.1f}x) between axes - likely spurious")
        return False
    return True


def find_known_mask_in_image(
    target_image: np.ndarray,
    known_mask_rgba: np.ndarray,
    min_matches: int = SIFT_MIN_MATCHES,
    dilation_pixels: int = 15,
    prepared_features: Optional[TemplateSiftFeatures] = None,
    prepared_target: Optional[PreparedTargetSift] = None,
    ratio: float = SIFT_RATIO,
    ransac_reproj_threshold: float = SIFT_RANSAC_REPROJ_THRESHOLD,
) -> Optional[SiftMaskMatch]:
    """Find a known watermark/logo with SIFT feature matching."""
    if known_mask_rgba.shape[2] != 4:
        raise ValueError("Known mask must be RGBA image (4 channels)")

    features = prepared_features or build_template_sift_features(known_mask_rgba)
    if features.descriptors is None or features.keypoint_count < min_matches:
        logger.warning(
            f"Not enough template SIFT keypoints: "
            f"known={features.keypoint_count}, need={min_matches}"
        )
        return None

    prepared_target = prepared_target or prepare_target_sift_features(target_image)
    if prepared_target is None:
        return None

    target_keypoints, target_descriptors = prepared_target
    if len(target_keypoints) < min_matches:
        logger.warning(f"Not enough target SIFT keypoints: target={len(target_keypoints)}")
        return None

    good_matches = _good_sift_matches(features.descriptors, target_descriptors, ratio)
    logger.info(f"SIFT matching: {len(good_matches)} good matches (need {min_matches})")
    if len(good_matches) < min_matches:
        logger.warning(f"Not enough good matches: {len(good_matches)} < {min_matches}")
        return None

    src_pts = np.array(
        [features.keypoints[m.queryIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    dst_pts = np.array(
        [target_keypoints[m.trainIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None or inlier_mask is None:
        logger.warning("Could not compute affine transform")
        return None

    inliers = int(inlier_mask.ravel().sum())
    if inliers < min_matches:
        logger.warning(f"Not enough inliers: {inliers} < {min_matches}")
        return None
    if not _affine_is_plausible(matrix):
        return None

    h_target, w_target = target_image.shape[:2]
    warped_mask = cv2.warpAffine(
        features.alpha,
        matrix,
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
        return None

    x, y, w, h = cv2.boundingRect(coords)
    h_known, w_known = features.alpha.shape[:2]
    template_area = max(1, int(np.sum(features.alpha > 0)))
    inlier_ratio = inliers / max(1, len(good_matches))
    template_coverage = inliers / max(1, features.keypoint_count)
    logger.info(
        f"Known mask found at ({x}, {y}) size {w}x{h} "
        f"(scale: {w / max(1, w_known):.2f}x, {h / max(1, h_known):.2f}x, "
        f"inliers={inliers}, ratio={inlier_ratio:.2f}, coverage={template_coverage:.2f})"
    )

    mask_pixels = int(np.sum(binary_mask > 0))
    image_pixels = h_target * w_target
    if mask_pixels >= image_pixels * 0.5:
        logger.warning(
            f"Mask covers {100 * mask_pixels / image_pixels:.1f}% of image "
            "- likely spurious template match"
        )
        return None
    if mask_pixels > template_area * 10_000:
        logger.warning("Mask exploded relative to template area - likely spurious")
        return None

    return SiftMaskMatch(
        binary_mask,
        (int(x), int(y), int(w), int(h)),
        inliers,
        len(good_matches),
        features.keypoint_count,
    )


def _candidate_for_template(
    image: np.ndarray,
    template: WatermarkTemplate,
    min_matches: int,
    dilation_pixels: int,
    prepared_target: PreparedTargetSift,
) -> Optional[SiftMatchCandidate]:
    result = find_known_mask_in_image(
        image,
        template.rgba,
        min_matches=min_matches,
        dilation_pixels=dilation_pixels,
        prepared_features=template.sift_features,
        prepared_target=prepared_target,
    )
    if result is None:
        return None
    return SiftMatchCandidate(
        result.mask,
        result.bbox,
        template.name,
        result.inliers,
        result.inlier_ratio,
        result.template_coverage,
    )


def _best_candidate(
    image: np.ndarray,
    templates: List[WatermarkTemplate],
    min_matches: int,
    dilation_pixels: int,
    prepared_target: PreparedTargetSift,
) -> Optional[SiftMatchCandidate]:
    """Best-ranked template match against an already-prepared target."""
    best: Optional[SiftMatchCandidate] = None
    for template in templates:
        logger.info(f"Trying template: {template.name}")
        candidate = _candidate_for_template(
            image,
            template,
            min_matches=min_matches,
            dilation_pixels=dilation_pixels,
            prepared_target=prepared_target,
        )
        if candidate is None:
            continue
        logger.info(
            f"Template {candidate.template_name} matched with {candidate.inliers} inliers "
            f"(ratio={candidate.inlier_ratio:.2f}, coverage={candidate.template_coverage:.2f})"
        )
        if best is None or candidate.rank_key > best.rank_key:
            best = candidate
    return best


def try_watermark_cascade(
    image: np.ndarray,
    templates: List[WatermarkTemplate],
    min_matches: int = SIFT_MIN_MATCHES,
    dilation_pixels: int = 15,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], str, int]]:
    """Try every watermark template full-frame; return the best single match."""
    if not templates:
        logger.info("No template matched (tried 0)")
        return None

    prepared_target = prepare_target_sift_features(image)
    if prepared_target is None:
        return None

    best = _best_candidate(image, templates, min_matches, dilation_pixels, prepared_target)
    if best is not None:
        logger.info(f"Best template: {best.template_name} ({best.inliers} inliers)")
        return best.mask, best.bbox, best.template_name, best.inliers

    logger.info(f"No template matched (tried {len(templates)})")
    return None


def _template_mark_size(template: WatermarkTemplate) -> int:
    """Longest side (px) of the template's alpha footprint — the visible mark."""
    ys, xs = np.nonzero(template.sift_features.alpha > 0)
    if len(ys) == 0:
        return 0
    return max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))


def _corner_crop(image: np.ndarray, region: str, win: int) -> Tuple[np.ndarray, int, int]:
    """Return (crop, row_offset, col_offset) for a named region ('full' or a corner)."""
    h, w = image.shape[:2]
    if region == "full":
        return image, 0, 0
    side = min(win, h, w)
    r0 = 0 if region in ("UR", "UL") else h - side
    c0 = 0 if region in ("LL", "UL") else w - side
    return image[r0:r0 + side, c0:c0 + side], r0, c0


def cascade_corners(
    image: np.ndarray,
    templates: List[WatermarkTemplate],
    min_matches: int = SIFT_MIN_MATCHES,
    dilation_pixels: int = 15,
    order: Tuple[str, ...] = CORNER_SEARCH_ORDER,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], str, int]]:
    """Scoped "push hard" cascade for user-specified watermark(s).

    Searches cheap corners before the full frame (early exit on first hit) so a
    small/faint corner logo is hunted in a tight per-template window instead of
    being diluted across the whole image. Cost is O(templates x regions), so this
    is for a few scoped -K templates, never the whole watermarks/ library. Marks
    whose window would not fit a corner are tried full-frame only. Crop matches
    are remapped to full-image coordinates.
    """
    if not templates:
        logger.info("No template matched (tried 0)")
        return None

    h, w = image.shape[:2]
    max_corner = CORNER_ELIGIBLE_FRACTION * min(h, w)
    eligible: List[Tuple[WatermarkTemplate, int]] = []
    for template in templates:
        win = CORNER_WINDOW_MULTIPLE * _template_mark_size(template)
        if 0 < win <= max_corner:
            eligible.append((template, win))

    def search_corner(region: str) -> Optional[Tuple[SiftMatchCandidate, int, int]]:
        best: Optional[Tuple[SiftMatchCandidate, int, int]] = None
        for template, win in eligible:
            crop, r0, c0 = _corner_crop(image, region, win)
            prepared = prepare_target_sift_features(crop)
            if prepared is None:
                continue
            candidate = _candidate_for_template(crop, template, min_matches, dilation_pixels, prepared)
            if candidate is None:
                continue
            if best is None or candidate.rank_key > best[0].rank_key:
                best = (candidate, r0, c0)
        return best

    for region in order:
        if region == "full":
            prepared = prepare_target_sift_features(image)
            if prepared is None:
                continue
            candidate = _best_candidate(image, templates, min_matches, dilation_pixels, prepared)
            hit = (candidate, 0, 0) if candidate is not None else None
        else:
            hit = search_corner(region) if eligible else None
        if hit is None:
            continue
        candidate, r0, c0 = hit
        full_mask = np.zeros((h, w), dtype=np.uint8)
        ch, cw = candidate.mask.shape[:2]
        full_mask[r0:r0 + ch, c0:c0 + cw] = candidate.mask
        bx, by, bw, bh = candidate.bbox
        logger.info(
            f"Corner cascade: {candidate.template_name} matched in {region} "
            f"({candidate.inliers} inliers)"
        )
        return full_mask, (bx + c0, by + r0, bw, bh), candidate.template_name, candidate.inliers

    logger.info(f"No template matched in corner cascade (tried {len(templates)})")
    return None
