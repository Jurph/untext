import json

import cv2
import numpy as np

from untextre.mask_experiments import (
    MaskExperimentConfig,
    compute_mask_metrics,
    iter_preset_configs,
    rank_mask_summaries,
    truth_target_mask,
    weighted_precision,
)
from untextre.pipeline import MASK_MODE_CHOICES


def test_truth_target_mask_uses_deterministic_elliptical_dilation():
    truth = np.zeros((7, 7), dtype=np.uint8)
    truth[3, 3] = 255

    target = truth_target_mask(truth, dilation_px=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expected = cv2.dilate(truth, kernel)
    np.testing.assert_array_equal(target, expected)


def test_weighted_precision_penalizes_far_false_positive_more_than_near():
    truth = np.zeros((15, 15), dtype=np.uint8)
    truth[7, 7] = 255
    near = truth.copy()
    near[7, 9] = 255
    far = truth.copy()
    far[0, 0] = 255

    assert weighted_precision(near, truth) > weighted_precision(far, truth)


def test_weighted_precision_matches_normal_precision_for_zero_distance_fp():
    target = np.zeros((5, 5), dtype=np.uint8)
    target[2, 2] = 255
    prediction = target.copy()

    assert weighted_precision(prediction, target) == 1.0




def test_compute_mask_metrics_includes_bbox_fp_split_and_score():
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[4:6, 4:6] = 255
    predicted = truth_target_mask(truth, dilation_px=2)
    predicted[0, 0] = 255

    metrics = compute_mask_metrics(predicted, truth, bbox=(3, 3, 4, 4))

    assert metrics["target_recall"] == 1.0
    assert metrics["fp_outside_bbox"] == 1
    assert metrics["fp_inside_bbox"] == 0
    assert metrics["coverage"] == 29 / 100
    assert metrics["score"] > 0


def test_rank_mask_summaries_applies_hard_filters_deterministically():
    rows = [
        {
            "config_id": "bad-recall",
            "mean_target_recall": 0.97,
            "min_target_recall": 0.95,
            "max_coverage": 0.01,
            "mean_score": 10,
        },
        {
            "config_id": "winner",
            "mean_target_recall": 0.99,
            "min_target_recall": 0.96,
            "max_coverage": 0.02,
            "mean_score": 5,
        },
        {
            "config_id": "loser",
            "mean_target_recall": 0.99,
            "min_target_recall": 0.96,
            "max_coverage": 0.02,
            "mean_score": 4,
        },
    ]

    ranked = rank_mask_summaries(rows)

    assert [row["config_id"] for row in ranked] == ["winner", "loser"]




