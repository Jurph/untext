"""Fast tests for generated text benchmark helpers."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import untextre.synthetic_text_benchmark as stb
from untextre.synthetic_text_benchmark import (
    EXCLUDED_BENCHMARK_ORIGINALS,
    generate_synthetic_text_case,
    iter_original_images,
    parse_int_csv,
    replay_synthetic_text_case,
)
from untextre.pipeline import mask_mode_options, process_image_array


def test_parse_int_csv_returns_positive_ints():
    assert parse_int_csv("4, 8,16") == [4, 8, 16]


def test_generate_synthetic_text_case_fits_truth_mask_width():
    rng = random.Random(1234)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220

    case = generate_synthetic_text_case(image, rng)

    width_fraction = case.metadata["text_bbox"][2] / image.shape[1]
    assert 0.25 <= width_fraction <= 0.50
    assert case.watermarked.shape == image.shape
    assert case.clean.shape == image.shape
    assert case.truth_bbox[2] > 0
    assert case.truth_bbox[3] > 0
    assert np.any(case.watermarked[case.truth_mask > 0] != case.clean[case.truth_mask > 0])






def test_generate_synthetic_text_case_avoids_same_class_corner_matches():
    white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    black_image = np.zeros((480, 640, 3), dtype=np.uint8)

    white_case = generate_synthetic_text_case(white_image, random.Random(99))
    black_case = generate_synthetic_text_case(black_image, random.Random(99))

    assert white_case.metadata["corner_luminance_class"] == "white"
    assert black_case.metadata["corner_luminance_class"] == "black"
    assert white_case.metadata["color_class"] != "white"
    assert black_case.metadata["color_class"] != "black"


def test_contrast_guard_rejects_light_gray_on_white_corner():
    assert stb._is_valid_fill_corner_pair("light_gray", "white") is False


def test_generated_text_cases_average_near_one_third_image_width():
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220
    width_fractions = []

    for seed in range(20):
        case = generate_synthetic_text_case(image, random.Random(seed))
        width_fractions.append(case.metadata["text_bbox"][2] / image.shape[1])

    assert 0.30 <= sum(width_fractions) / len(width_fractions) <= 0.37


def test_load_text_sources_reads_flat_files(tmp_path: Path):
    text_source_dir = tmp_path / "text_sources"
    text_source_dir.mkdir()
    (text_source_dir / "url_prefixes.txt").write_text("Image\nPicture\n", encoding="utf-8")
    (text_source_dir / "url_nouns.txt").write_text("Fans\nPosts\n", encoding="utf-8")
    (text_source_dir / "url_tlds.txt").write_text(".com\n.net\n", encoding="utf-8")
    (text_source_dir / "copyright_first_names.txt").write_text("Ada\nXenia\n", encoding="utf-8")
    (text_source_dir / "copyright_last_names.txt").write_text("Lovelace\nXoroboros\n", encoding="utf-8")

    sources = stb._load_text_sources(text_source_dir)

    assert sources.url_prefixes == ("Image", "Picture")
    assert sources.url_nouns == ("Fans", "Posts")
    assert sources.url_tlds == (".com", ".net")
    assert sources.copyright_first_names == ("Ada", "Xenia")
    assert sources.copyright_last_names == ("Lovelace", "Xoroboros")






def test_generate_synthetic_text_case_visibility_floor_passes_on_first_attempt():
    rng = random.Random(1234)
    image = np.full((480, 640, 3), 220, dtype=np.uint8)

    case = generate_synthetic_text_case(image, rng)

    assert case.metadata["measured_visibility_delta_e"] >= stb.MIN_VISIBILITY_DELTA_E
    assert case.metadata["visibility_attempts"] == 1
    assert case.metadata["visibility_fallback"] is False


def test_generate_synthetic_text_case_visibility_floor_falls_back_when_unreachable(monkeypatch):
    monkeypatch.setattr(stb, "MIN_VISIBILITY_DELTA_E", 1000.0)
    rng = random.Random(1234)
    image = np.full((480, 640, 3), 220, dtype=np.uint8)

    case = generate_synthetic_text_case(image, rng)

    assert case.metadata["visibility_fallback"] is True
    assert case.metadata["visibility_attempts"] == stb.MAX_VISIBILITY_ATTEMPTS
    assert case.metadata["measured_visibility_delta_e"] > 0.0
    assert np.isfinite(case.metadata["measured_visibility_delta_e"])


def test_generate_synthetic_text_case_visibility_retries_are_deterministic(monkeypatch):
    monkeypatch.setattr(stb, "MIN_VISIBILITY_DELTA_E", 1000.0)
    image = np.full((480, 640, 3), 220, dtype=np.uint8)

    case_a = generate_synthetic_text_case(image, random.Random(777))
    case_b = generate_synthetic_text_case(image, random.Random(777))

    assert np.array_equal(case_a.watermarked, case_b.watermarked)
    assert case_a.metadata == case_b.metadata


def test_generate_synthetic_text_case_white_on_black_has_large_delta_e():
    rng = random.Random(1234)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    case = generate_synthetic_text_case(image, rng)

    assert case.metadata["measured_visibility_delta_e"] > 40.0


def test_replay_synthetic_text_sample_images_are_deterministic(test_images_dir: Path, tmp_path: Path):
    samples_dir = test_images_dir / "samples"
    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("sample corpus is not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_indices = [1, 7, 13, 27, 49, 92, 97]

    for index in sample_indices:
        case = manifest["cases"][index]
        source_path = Path(case["source_relpath"])
        clean = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        assert clean is not None, case["source_relpath"]

        replayed = replay_synthetic_text_case(clean, case)
        assert replayed.truth_bbox == tuple(case["truth_bbox"])
        assert replayed.metadata["text"] == case["text"]
        assert replayed.metadata["font_path"] == case["font_path"]

        out_path = tmp_path / case["image_path"]
        Image.fromarray(cv2.cvtColor(replayed.watermarked, cv2.COLOR_BGR2RGB)).save(out_path, quality=95)

        expected = cv2.imread(str(samples_dir / case["image_path"]), cv2.IMREAD_COLOR)
        actual = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        assert expected is not None, case["image_path"]
        assert actual is not None, case["image_path"]
        np.testing.assert_array_equal(actual, expected)


def test_iter_original_images_excludes_distracting_backdrop_samples(test_images_dir):
    paths = iter_original_images(test_images_dir)
    names = {path.name for path in paths}
    assert names.isdisjoint(EXCLUDED_BENCHMARK_ORIGINALS)


@pytest.mark.slow
def test_budgeted_regional_generated_text_fixture_metrics(
    test_images_dir: Path,
    save_images_dir: "Path | None",
):
    fixture_dir = test_images_dir / "generated_text_watermarks"
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("generated text watermark fixtures are not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ious = []
    precisions = []
    recalls = []
    coverages = []

    for case in manifest["cases"]:
        image = cv2.imread(str(fixture_dir / case["image"]))
        truth_mask = cv2.imread(str(fixture_dir / case["mask"]), cv2.IMREAD_GRAYSCALE)
        assert image is not None, case["image"]
        assert truth_mask is not None, case["mask"]

        pipeline_result = process_image_array(
            image,
            image_name=case["slug"],
            method="telea",
            forced_bbox=tuple(case["bbox_xywh"]),
            auto_retry=False,
            coverage_limit=0.06,
            **mask_mode_options("budgeted-regional"),
        )

        predicted = pipeline_result.mask > 0
        truth = truth_mask > 0
        intersection = int(np.logical_and(predicted, truth).sum())
        union = int(np.logical_or(predicted, truth).sum())
        predicted_count = int(predicted.sum())
        truth_count = int(truth.sum())

        ious.append(intersection / union if union else 0.0)
        precisions.append(intersection / predicted_count if predicted_count else 0.0)
        recalls.append(intersection / truth_count if truth_count else 0.0)
        coverages.append(float(predicted.mean()))

        if save_images_dir is not None:
            stem = f"budgeted_regional_{case['slug']}"
            cv2.imwrite(str(save_images_dir / f"{stem}_truth_mask.png"), truth_mask)
            cv2.imwrite(str(save_images_dir / f"{stem}_pipeline_mask.png"), pipeline_result.mask)

    assert len(ious) == manifest["case_count"] == 17
    assert float(np.mean(ious)) >= 0.28
    assert float(np.mean(precisions)) >= 0.28
    assert float(np.mean(recalls)) >= 0.90
    assert max(coverages) <= 0.06
