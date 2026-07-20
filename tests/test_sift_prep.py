import cv2
import numpy as np

from untextre.sift_prep import (
    count_candidate_sift_keypoints,
    prepare_candidate_bgra_for_sift,
    prepare_candidate_for_sift,
)


def _make_sift_ready_candidate(size: int = 128) -> np.ndarray:
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    corner_size = max(6, size // 10)
    cv2.circle(bgr, center, size // 3, (220, 220, 220), 6, cv2.LINE_AA)
    cv2.circle(bgr, center, size // 7, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.line(bgr, (size // 5, size // 2), (size - size // 5, size // 2), (220, 220, 220), 4, cv2.LINE_AA)
    cv2.circle(alpha, center, size // 3, 255, 6, cv2.LINE_AA)
    cv2.circle(alpha, center, size // 7, 255, -1, cv2.LINE_AA)
    cv2.line(alpha, (size // 5, size // 2), (size - size // 5, size // 2), 255, 4, cv2.LINE_AA)
    for x, y in (
        (size // 4, size // 4),
        (size - size // 4 - corner_size, size // 4),
        (size // 4, size - size // 4 - corner_size),
        (size - size // 4 - corner_size, size - size // 4 - corner_size),
    ):
        cv2.rectangle(bgr, (x, y), (x + corner_size, y + corner_size), (220, 220, 220), -1)
        cv2.rectangle(alpha, (x, y), (x + corner_size, y + corner_size), 255, -1)
    return np.dstack([bgr, alpha])


def test_prepare_candidate_for_sift_zeroes_transparent_bgr_and_heals_single_pixels():
    bgra = np.zeros((11, 11, 4), dtype=np.uint8)
    bgra[:, :, :3] = 177
    bgra[3:8, 3:8, :3] = 220
    bgra[3:8, 3:8, 3] = 255
    bgra[5, 5, 3] = 0
    bgra[0, 0, 3] = 255

    prepped_bgr, prepped_mask, prepped_gray = prepare_candidate_for_sift(bgra, padding=0)

    assert prepped_mask[5, 5] == 255
    assert prepped_mask[0, 0] == 0
    assert np.all(prepped_bgr[prepped_mask == 0] == 0)
    assert prepped_gray.shape == prepped_mask.shape


def test_prepare_candidate_bgra_for_sift_returns_clean_alpha_with_padding():
    bgra = np.zeros((11, 11, 4), dtype=np.uint8)
    bgra[:, :, :3] = 177
    bgra[3:8, 3:8, :3] = 220
    bgra[3:8, 3:8, 3] = 255

    prepared = prepare_candidate_bgra_for_sift(bgra)

    assert prepared.shape == (15, 15, 4)
    assert np.all(prepared[:2, :, 3] == 0)
    assert np.all(prepared[:, :2, 3] == 0)
    assert prepared[7, 7, 3] == 255
    assert np.all(prepared[prepared[:, :, 3] == 0, :3] == 0)


def test_count_candidate_sift_keypoints_accepts_structured_candidates():
    ready = _make_sift_ready_candidate()
    assert count_candidate_sift_keypoints(ready) >= 5
