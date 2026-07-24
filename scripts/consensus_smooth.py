"""Contour-smoothed consensus template.

Supersample the vote-weighted mean 4x, Gaussian blur, re-threshold, and
downsample: kills stair-step jaggies while preserving observed geometry.
Composites the dot (slice 5) like consensus_finalize, then SIFT-preps and
saves as a SEPARATE candidate: watermark_consensus_smooth.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from untextre.sift_prep import (  # noqa: E402
    count_candidate_sift_keypoints,
    prepare_candidate_bgra_for_sift,
)

CONS = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\cleaned-2\consensus")
SS = 4  # supersample factor


def smooth_mask(gray: np.ndarray, thresh: float) -> np.ndarray:
    big = cv2.resize(gray, None, fx=SS, fy=SS, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=SS * 1.2)
    mask_big = (big >= thresh * 255).astype(np.uint8) * 255
    # light morphological close at high res to heal nicks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SS + 1, SS + 1))
    mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, k)
    return cv2.resize(mask_big, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)


def main() -> None:
    mean = cv2.imread(str(CONS / "consensus_mean.png"), 0)
    if mean is None:
        sys.exit("run consensus_watermark.py first")

    # Main strokes: >=2-of-4 agreement in the mean (0.5 * 255).
    body = smooth_mask(mean, 0.45)

    # Dot: present only in slice 5 -> grab it from that aligned slice, smooth it.
    s5 = cv2.imread(str(CONS / "aligned_watermark_candidate_5.png"), 0)
    dot_region = np.zeros_like(s5)
    dot_region[55:75, 500:527] = s5[55:75, 500:527]  # dot bbox (507,60) 13x9 + margin
    dot = smooth_mask(dot_region, 0.45)

    combined = np.maximum(body, dot)
    bin_ = (combined > 127).astype(np.uint8)

    # Despeckle: drop anything smaller than the dot (115 px), except keep the dot.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            bin_[labels == i] = 0
            removed += 1
    print(f"despeckle: removed {removed} sub-60px components")

    mask255 = bin_ * 255
    bgra = np.dstack([mask255] * 4)
    prepared = prepare_candidate_bgra_for_sift(bgra)
    # Prep may heal alpha pixels whose BGR stayed black; the watermark is pure
    # white, so whiten BGR under all opaque alpha.
    # (Alpha untouched: A/Q/O counters remain transparent.)
    prepared[prepared[:, :, 3] > 0, :3] = 255
    dark = int(((prepared[:, :, 3] > 0)
                & (cv2.cvtColor(prepared[:, :, :3], cv2.COLOR_BGR2GRAY) < 128)).sum())
    print(f"dark RGB px under alpha after whitening: {dark}")
    out = CONS / "watermark_consensus_smooth.png"
    cv2.imwrite(str(out), prepared)
    print(f"wrote {out} ({prepared.shape[1]}x{prepared.shape[0]}, "
          f"{int((prepared[:, :, 3] > 0).sum())} opaque px)")

    # SIFT matchability check vs the pixel-derived template
    for name in ("watermark_consensus.png", "watermark_consensus_smooth.png"):
        t = cv2.imread(str(CONS / name), cv2.IMREAD_UNCHANGED)
        print(f"{name}: {count_candidate_sift_keypoints(t)} SIFT keypoints")


if __name__ == "__main__":
    main()
