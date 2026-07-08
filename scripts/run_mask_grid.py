from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from untextre.mask_experiments import (
    DEFAULT_MANIFEST,
    MaskExperimentConfig,
    compute_mask_metrics,
    iter_preset_configs,
    load_manifest_cases,
)
from untextre.pipeline import process_image_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic mask grid experiments.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-start", type=int, default=0)
    parser.add_argument("--config-step", type=int, default=1)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--save-images", type=Path, default=None)
    args = parser.parse_args()

    configs = list(iter_preset_configs(args.preset))
    if args.config_start < 0:
        raise ValueError("--config-start must be non-negative")
    if args.config_step < 1:
        raise ValueError("--config-step must be at least 1")
    configs = configs[args.config_start :: args.config_step]
    if args.limit_configs is not None:
        if args.limit_configs < 0:
            raise ValueError("--limit-configs must be non-negative")
        configs = configs[: args.limit_configs]

    cases = load_manifest_cases(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        for cfg in configs:
            for case in cases:
                row = run_case_config(case, cfg)
                mask = row.pop("_mask")
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                if args.save_images:
                    mask_path = args.save_images / f"{case['id']}__{cfg.config_id}.png"
                    cv2.imwrite(str(mask_path), mask)


def run_case_config(case: dict, cfg: MaskExperimentConfig) -> dict:
    image = cv2.imread(str(case["image_path"]), cv2.IMREAD_COLOR)
    truth = cv2.imread(str(case["truth_mask_path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {case['image_path']}")
    if truth is None:
        raise ValueError(f"Could not read truth mask: {case['truth_mask_path']}")

    result = process_image_array(
        image,
        image_name=str(case["id"]),
        method="telea",
        forced_bbox=tuple(case["truth_bbox"]),
        expand_bboxes=False,
        auto_retry=False,
        use_grabcut_expand=False,
        use_budgeted_expand=True,
        coverage_limit=0,
        mask_config=cfg.to_dict(),
    )
    predicted = result.mask
    if predicted.shape != truth.shape:
        predicted = cv2.resize(predicted, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST)
    metrics = compute_mask_metrics(predicted, truth, bbox=tuple(case["truth_bbox"]))
    return {
        "case_id": case["id"],
        "config_id": cfg.config_id,
        "preset": cfg.preset,
        "bbox": list(case["truth_bbox"]),
        "config": cfg.to_dict(),
        **metrics,
        "_mask": predicted,
    }


if __name__ == "__main__":
    main()
