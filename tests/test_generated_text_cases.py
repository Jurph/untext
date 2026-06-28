"""Fast tests for generated text benchmark helpers."""

import random

import numpy as np

from untextre.synthetic_text_benchmark import (
    EXCLUDED_BENCHMARK_ORIGINALS,
    generate_synthetic_text_case,
    iter_original_images,
    parse_int_csv,
)


def test_parse_int_csv_returns_positive_ints():
    assert parse_int_csv("4, 8,16") == [4, 8, 16]


def test_generate_synthetic_text_case_fits_truth_mask_width():
    rng = random.Random(1234)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220

    case = generate_synthetic_text_case(image, rng)

    width_fraction = case.truth_bbox[2] / image.shape[1]
    assert 0.25 <= width_fraction <= 0.50
    assert case.watermarked.shape == image.shape
    assert case.clean.shape == image.shape
    assert case.truth_bbox[2] > 0
    assert case.truth_bbox[3] > 0
    assert np.any(case.watermarked[case.truth_mask > 0] != case.clean[case.truth_mask > 0])


def test_generated_text_cases_average_near_one_third_image_width():
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220
    width_fractions = []

    for seed in range(20):
        case = generate_synthetic_text_case(image, random.Random(seed))
        width_fractions.append(case.truth_bbox[2] / image.shape[1])

    assert 0.30 <= sum(width_fractions) / len(width_fractions) <= 0.37


def test_iter_original_images_excludes_distracting_backdrop_samples(test_images_dir):
    paths = iter_original_images(test_images_dir)
    names = {path.name for path in paths}
    assert names.isdisjoint(EXCLUDED_BENCHMARK_ORIGINALS)
