"""Shared CIE color-distance helpers for offline quality evaluation.

Not used by the production pipeline -- these back the benchmark/eval scripts
and tests that score repair quality against a known-clean ground truth
(`scripts/run_inpaint_eval.py`, `tests/test_generated_text_benchmark.py`).
Centralized here so a metric change (e.g. LAB-MAE -> CIEDE2000 delta-E)
propagates to every consumer instead of drifting between copies.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000


def bgr_to_lab_cie(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to CIE L*a*b* in standard ranges (L*∈[0,100], a*/b*∈[-128,127])."""
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    raw[..., 0] *= 100.0 / 255.0  # L: OpenCV [0,255] -> CIE [0,100]
    raw[..., 1] -= 128.0  # a: OpenCV [0,255] -> CIE [-128,127]
    raw[..., 2] -= 128.0  # b: OpenCV [0,255] -> CIE [-128,127]
    return raw


def delta_e_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIE ΔE₀₀ between two uint8 BGR images, same shape as the input."""
    return deltaE_ciede2000(bgr_to_lab_cie(a), bgr_to_lab_cie(b))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel CIE ΔE₀₀ between two uint8 BGR images."""
    return float(np.mean(delta_e_map(a, b)))


def masked_delta_e(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-pixel CIE ΔE₀₀ between two uint8 BGR images, restricted to mask."""
    return float(delta_e_map(a, b)[mask.astype(bool)].mean())
