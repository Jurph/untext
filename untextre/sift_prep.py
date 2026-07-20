"""SIFT-preparation helpers for discovered watermark candidates."""

from __future__ import annotations

import cv2
import numpy as np

SIFT_EXPORT_PADDING = 2


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not np.any(mask):
        return mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.copy()

    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def prepare_candidate_for_sift(
    bgra: np.ndarray,
    min_component_area: int = 2,
    fill_neighbor_threshold: int = 6,
    padding: int = SIFT_EXPORT_PADDING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a BGRA watermark candidate for alpha-masked SIFT extraction.

    The cleanup is intentionally conservative:
    - remove one-pixel salt components
    - fill isolated single-pixel holes
    - zero BGR values outside the cleaned alpha mask
    - add transparent export padding so edge pixels survive file round-trips
    """
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("prepare_candidate_for_sift expects BGRA input")

    mask = (bgra[:, :, 3] > 0).astype(np.uint8)
    if not np.any(mask):
        empty_bgr = np.zeros_like(bgra[:, :, :3])
        empty_mask = np.zeros_like(bgra[:, :, 3], dtype=np.uint8)
        empty_gray = np.zeros_like(bgra[:, :, 3], dtype=np.uint8)
        return empty_bgr, empty_mask, empty_gray

    cleaned = _remove_small_components(mask, min_area=min_component_area)
    if np.any(cleaned):
        neighbor_count = cv2.filter2D(cleaned, cv2.CV_16U, np.ones((3, 3), dtype=np.uint8))
        healed = cleaned.copy()
        healed[(cleaned == 0) & (neighbor_count >= fill_neighbor_threshold)] = 1
        cleaned = _remove_small_components(healed, min_area=min_component_area)

    clean_mask = (cleaned * 255).astype(np.uint8)
    prepared_bgr = bgra[:, :, :3].copy()
    prepared_bgr[clean_mask == 0] = 0

    if padding > 0:
        prepared_bgr = cv2.copyMakeBorder(
            prepared_bgr,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        clean_mask = cv2.copyMakeBorder(
            clean_mask,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(0,),
        )

    prepared_gray = cv2.cvtColor(prepared_bgr, cv2.COLOR_BGR2GRAY)
    return prepared_bgr, clean_mask, prepared_gray


def prepare_candidate_bgra_for_sift(
    bgra: np.ndarray,
    min_component_area: int = 2,
    fill_neighbor_threshold: int = 6,
    padding: int = SIFT_EXPORT_PADDING,
) -> np.ndarray:
    """Return a SIFT-prepped BGRA candidate suitable for disk export and -K reuse."""
    prepared_bgr, clean_mask, _ = prepare_candidate_for_sift(
        bgra,
        min_component_area=min_component_area,
        fill_neighbor_threshold=fill_neighbor_threshold,
        padding=padding,
    )
    return np.dstack([prepared_bgr, clean_mask])


def count_candidate_sift_keypoints(bgra: np.ndarray) -> int:
    from .sift_matcher import build_template_sift_features

    prepared_bgra = prepare_candidate_bgra_for_sift(bgra)
    return build_template_sift_features(prepared_bgra).keypoint_count
