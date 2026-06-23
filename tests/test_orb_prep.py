import cv2
import numpy as np

from untextre.orb_prep import (
    prepare_candidate_for_orb,
    prepare_candidate_bgra_for_orb,
    count_candidate_orb_keypoints,
    build_candidate_orb_variants,
)


def _make_orb_ready_candidate(size: int = 128) -> np.ndarray:
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    corner_size = max(6, size // 10)
    cv2.circle(bgr, center, size // 3, (220, 220, 220), 6, cv2.LINE_AA)
    cv2.circle(bgr, center, size // 7, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.line(
        bgr,
        (size // 5, size // 2),
        (size - size // 5, size // 2),
        (220, 220, 220),
        4,
        cv2.LINE_AA,
    )
    cv2.circle(alpha, center, size // 3, 255, 6, cv2.LINE_AA)
    cv2.circle(alpha, center, size // 7, 255, -1, cv2.LINE_AA)
    cv2.line(
        alpha,
        (size // 5, size // 2),
        (size - size // 5, size // 2),
        255,
        4,
        cv2.LINE_AA,
    )
    for x, y in (
        (size // 4, size // 4),
        (size - size // 4 - corner_size, size // 4),
        (size // 4, size - size // 4 - corner_size),
        (size - size // 4 - corner_size, size - size // 4 - corner_size),
    ):
        cv2.rectangle(bgr, (x, y), (x + corner_size, y + corner_size), (220, 220, 220), -1)
        cv2.rectangle(alpha, (x, y), (x + corner_size, y + corner_size), 255, -1)
    return np.dstack([bgr, alpha])


def _make_orb_dead_bar(width: int = 160, height: int = 40) -> np.ndarray:
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    bgra[10:30, 20:140, :3] = 220
    bgra[10:30, 20:140, 3] = 255
    return bgra


def test_prepare_candidate_for_orb_zeroes_transparent_bgr_and_heals_single_pixels():
    bgra = np.zeros((11, 11, 4), dtype=np.uint8)
    bgra[:, :, :3] = 177
    bgra[3:8, 3:8, :3] = 220
    bgra[3:8, 3:8, 3] = 255
    bgra[5, 5, 3] = 0
    bgra[0, 0, 3] = 255

    prepped_bgr, prepped_mask, prepped_gray = prepare_candidate_for_orb(bgra)

    assert prepped_mask.shape == (75, 75)
    assert prepped_mask[0, 0] == 0
    assert np.all(prepped_mask[:32, :] == 0)
    assert np.all(prepped_mask[:, :32] == 0)
    assert prepped_mask[37, 37] == 255
    assert np.all(prepped_bgr[prepped_mask == 0] == 0)
    assert prepped_gray.shape == prepped_mask.shape


def test_count_candidate_orb_keypoints_recovers_border_heavy_candidates():
    ready = _make_orb_ready_candidate()
    border_heavy = _make_orb_dead_bar()

    ready_keypoints = count_candidate_orb_keypoints(ready)
    border_heavy_keypoints = count_candidate_orb_keypoints(border_heavy)

    assert ready_keypoints >= 6
    assert border_heavy_keypoints >= 6


def test_prepare_candidate_bgra_for_orb_returns_clean_alpha_with_padding():
    bgra = np.zeros((11, 11, 4), dtype=np.uint8)
    bgra[:, :, :3] = 177
    bgra[3:8, 3:8, :3] = 220
    bgra[3:8, 3:8, 3] = 255
    bgra[5, 5, 3] = 0
    bgra[0, 0, 3] = 255

    prepared = prepare_candidate_bgra_for_orb(bgra)

    assert prepared.shape == (15, 15, 4)
    assert np.all(prepared[:2, :, 3] == 0)
    assert np.all(prepared[:, :2, 3] == 0)
    assert prepared[7, 7, 3] == 255
    assert np.all(prepared[prepared[:, :, 3] == 0, :3] == 0)


def test_build_candidate_orb_variants_orders_outside_fills_by_keypoint_count():
    bgra = np.zeros((96, 96, 4), dtype=np.uint8)
    alpha = np.zeros((96, 96), dtype=np.uint8)
    cv2.putText(alpha, "B", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 8, cv2.LINE_AA)
    bgra[:, :, 3] = alpha

    variants = build_candidate_orb_variants(bgra)

    assert [variant.outside_value for variant in variants] == sorted(
        [variant.outside_value for variant in variants],
        key=lambda outside: next(v.keypoint_count for v in variants if v.outside_value == outside),
        reverse=True,
    )
    assert {variant.outside_value for variant in variants} == {0, 127, 255}
    assert variants[0].keypoint_count > variants[-1].keypoint_count
