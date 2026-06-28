"""Very-slow Monte Carlo benchmark for generated text removal."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from untextre.inpaint import initialize_lama_model
from untextre.pipeline import MASK_MODE_CHOICES, initialize_consensus_models, mask_mode_options, process_image_array
from untextre.synthetic_text_benchmark import (
    generate_synthetic_text_case,
    iter_original_images,
    parse_int_csv,
)
from untextre.utils import load_image

pytestmark = [pytest.mark.slow, pytest.mark.very_slow]

_PIPELINE_MODES = MASK_MODE_CHOICES


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _best_bbox_iou(
    truth_bbox: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
) -> float:
    if not boxes:
        return 0.0
    return max(_bbox_iou(truth_bbox, box) for box in boxes)


def _masked_ssim_score(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    _score, ssim_map = ssim(before_gray, after_gray, data_range=255, full=True)
    return float(np.mean(ssim_map[mask.astype(bool)]))


def _masked_lab_mae(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> float:
    before_lab = (cv2.cvtColor(before, cv2.COLOR_BGR2LAB) // 16) * 16
    after_lab = (cv2.cvtColor(after, cv2.COLOR_BGR2LAB) // 16) * 16
    diff = np.abs(before_lab.astype(np.int16) - after_lab.astype(np.int16))
    return float(diff[mask.astype(bool)].mean())


def _classify_outcome(
    *,
    consensus_boxes: list[tuple[int, int, int, int]],
    best_detection_iou: float,
    mask: np.ndarray,
    input_image: np.ndarray,
    output_image: np.ndarray,
    coverage_limit: float,
) -> str:
    if not consensus_boxes:
        return "miss"
    if best_detection_iou <= 0.0:
        return "false_positive"
    if not np.any(mask > 0):
        return "detected_no_mask"
    if float(np.mean(mask > 0)) > coverage_limit and np.array_equal(input_image, output_image):
        return "coverage_rejected"
    return "repaired"


def _row_summary(rows: list[dict]) -> str:
    by_outcome: dict[str, int] = {}
    for row in rows:
        by_outcome[row["outcome"]] = by_outcome.get(row["outcome"], 0) + 1
    repaired = [row for row in rows if row["outcome"] == "repaired"]
    parts = [f"outcomes={by_outcome}"]
    if repaired:
        parts.append(
            "ssim_delta mean={:.6f} median={:.6f}".format(
                statistics.fmean(row["ssim_delta"] for row in repaired),
                statistics.median(row["ssim_delta"] for row in repaired),
            )
        )
        parts.append(
            "lab_mae_delta mean={:.6f} median={:.6f}".format(
                statistics.fmean(row["lab_mae_delta"] for row in repaired),
                statistics.median(row["lab_mae_delta"] for row in repaired),
            )
        )
    return "; ".join(parts)


def test_generated_text_modes_match_pipeline_semantics() -> None:
    assert mask_mode_options("regional") == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": True,
    }
    assert mask_mode_options("local-shape") == {
        "expand_bboxes": False,
        "use_grabcut": True,
        "use_grabcut_expand": False,
    }
    assert mask_mode_options("local-color") == {
        "expand_bboxes": False,
        "use_grabcut": False,
        "use_grabcut_expand": False,
    }


@pytest.fixture(scope="module")
def inpaint_method() -> str:
    if initialize_lama_model(device="cuda"):
        return "lama"
    if initialize_lama_model(device="cpu"):
        return "lama"
    return "telea"


def test_generated_text_monte_carlo_benchmark(
    request: pytest.FixtureRequest,
    test_images_dir: Path,
    inpaint_method: str,
    save_images_dir: "Path | None",
) -> None:
    case_count = request.config.getoption("--generated-text-cases")
    if case_count <= 0:
        pytest.skip("--generated-text-cases must be positive")

    seed = request.config.getoption("--generated-text-seed")
    color_sensitivities = parse_int_csv(
        request.config.getoption("--generated-text-color-sensitivities")
    )
    report_path_value = request.config.getoption("--generated-text-report")
    report_path = Path(report_path_value) if report_path_value else None

    image_paths = iter_original_images(test_images_dir)
    if not image_paths:
        pytest.skip("No clean originals found for generated text benchmark")

    initialize_consensus_models(device="cuda")
    rng = random.Random(seed)
    rows: list[dict] = []
    coverage_limit = 0.06

    for case_index in range(case_count):
        image_path = rng.choice(image_paths)
        clean = load_image(image_path)
        synthetic = generate_synthetic_text_case(clean, rng)
        mode = rng.choice(_PIPELINE_MODES)
        sensitivity = rng.choice(color_sensitivities)

        pipeline_result = process_image_array(
            synthetic.watermarked,
            image_name=f"generated-{case_index:04d}-{image_path.name}",
            method=inpaint_method,
            color_sensitivity=sensitivity,
            coverage_limit=coverage_limit,
            **mask_mode_options(mode),
        )

        best_iou = _best_bbox_iou(synthetic.truth_bbox, pipeline_result.consensus_boxes)
        mask_coverage = float(np.mean(pipeline_result.mask > 0))
        outcome = _classify_outcome(
            consensus_boxes=pipeline_result.consensus_boxes,
            best_detection_iou=best_iou,
            mask=pipeline_result.mask,
            input_image=synthetic.watermarked,
            output_image=pipeline_result.image,
            coverage_limit=coverage_limit,
        )

        row = {
            "case_index": case_index,
            "seed": seed,
            "source_image": str(image_path.relative_to(test_images_dir)),
            "mode": mode,
            "color_sensitivity": sensitivity,
            "inpaint_method": inpaint_method,
            "outcome": outcome,
            "consensus_box_count": len(pipeline_result.consensus_boxes),
            "best_detection_iou": best_iou,
            "mask_coverage": mask_coverage,
            **synthetic.metadata,
        }
        if outcome == "repaired":
            baseline_ssim = _masked_ssim_score(
                synthetic.clean, synthetic.watermarked, synthetic.truth_mask
            )
            repaired_ssim = _masked_ssim_score(
                synthetic.clean, pipeline_result.image, synthetic.truth_mask
            )
            baseline_lab = _masked_lab_mae(
                synthetic.clean, synthetic.watermarked, synthetic.truth_mask
            )
            repaired_lab = _masked_lab_mae(
                synthetic.clean, pipeline_result.image, synthetic.truth_mask
            )
            row["ssim_delta"] = repaired_ssim - baseline_ssim
            row["lab_mae_delta"] = baseline_lab - repaired_lab
        rows.append(row)

        if save_images_dir is not None:
            stem = f"generated_text_{case_index:04d}_{outcome}"
            cv2.imwrite(str(save_images_dir / f"{stem}_clean.png"), synthetic.clean)
            cv2.imwrite(
                str(save_images_dir / f"{stem}_watermarked.png"),
                synthetic.watermarked,
            )
            cv2.imwrite(str(save_images_dir / f"{stem}_truth_mask.png"), synthetic.truth_mask)
            cv2.imwrite(str(save_images_dir / f"{stem}_pipeline_mask.png"), pipeline_result.mask)
            cv2.imwrite(str(save_images_dir / f"{stem}_result.png"), pipeline_result.image)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("generated text benchmark:", _row_summary(rows))
    assert len(rows) == case_count
