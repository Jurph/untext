from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from untextre.color_metrics import delta_e
from untextre.inpaint import InpaintMethod, inpaint_image
from untextre.mask_experiments import DEFAULT_MANIFEST, load_manifest_cases, truth_target_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inpainting for top mask configs.")
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--method", choices=("telea", "lama"), default="telea")
    args = parser.parse_args()

    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    cases = load_manifest_cases(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as handle:
        for cfg in configs:
            config_id = cfg.get("config_id", cfg.get("config", {}).get("config_id", "unknown"))
            for case in cases:
                row = evaluate_case(
                    case, config_id, cast(InpaintMethod, args.method)
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_case(case: dict, config_id: str, method: InpaintMethod) -> dict:
    watermarked = cv2.imread(str(case["image_path"]), cv2.IMREAD_COLOR)
    clean_path = case.get("clean_path")
    clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR) if clean_path else None
    truth = cv2.imread(str(case["truth_mask_path"]), cv2.IMREAD_GRAYSCALE)
    if watermarked is None or truth is None:
        raise ValueError(f"Could not read inpaint eval case assets: {case['id']}")
    if clean is None:
        clean = watermarked.copy()

    mask = truth_target_mask(truth, dilation_px=2)
    repaired = inpaint_image(watermarked, mask, bbox=_window_bbox(case["truth_bbox"], watermarked.shape[:2]), method=method)
    x, y, w, h = _window_bbox(case["truth_bbox"], watermarked.shape[:2], pad=16)
    before = watermarked[y:y + h, x:x + w]
    after = repaired[y:y + h, x:x + w]
    target = clean[y:y + h, x:x + w]

    before_ssim = _ssim(before, target)
    after_ssim = _ssim(after, target)
    before_de = delta_e(before, target)
    after_de  = delta_e(after, target)
    return {
        "case_id": case["id"],
        "config_id": config_id,
        "method": method,
        "window_bbox": [x, y, w, h],
        "ssim_before": before_ssim,
        "ssim_after": after_ssim,
        "ssim_gain_ratio": _gain_ratio(after_ssim - before_ssim, 1.0 - before_ssim),
        "delta_e_before": before_de,
        "delta_e_after":  after_de,
        "delta_e_gain_ratio": _gain_ratio(before_de - after_de, before_de),
        "oracle_mask": "truth+2px",
    }


def _window_bbox(bbox, shape, pad=0):
    img_h, img_w = shape
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    min_side = min(a.shape[:2])
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(
        cast(
            float,
            structural_similarity(a, b, channel_axis=2, win_size=win_size),
        )
    )


def _gain_ratio(gain: float, possible: float) -> float:
    if possible <= 1e-9:
        return 0.0
    return float(gain / possible)


if __name__ == "__main__":
    main()
