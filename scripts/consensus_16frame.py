"""Candidate-16 texture with consensus-clean alpha in candidate-16's frame.

The unwarped candidate_16 texture is the least resampling-damaged donor. Keep
its RGB bytes untouched where possible and move the clean consensus mask into
its coordinate frame so SIFT sees native gradients inside a corrected alpha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from untextre.sift_matcher import _make_watermark_template, try_watermark_cascade  # noqa: E402

BASE = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\cleaned-2")
CONS = BASE / "consensus"
SRC = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\has-SQ-logo")


def register_to_ref(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """ECC affine mapping ref coords -> prescaled-src coords (as in consensus_watermark)."""
    rh, rw = ref.shape
    sh, sw = src.shape
    scale = rw / sw
    src_s = cv2.resize(src, (rw, max(1, round(sh * scale))), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    src_f = cv2.GaussianBlur(src_s, (9, 9), 0).astype(np.float32) / 255.0
    ref_f = cv2.GaussianBlur(ref, (9, 9), 0).astype(np.float32) / 255.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    cc, warp = cv2.findTransformECC(  # pyright: ignore[reportCallIssue]
        ref_f,
        src_f,
        warp,
        cv2.MOTION_AFFINE,
        criteria,
        None,  # pyright: ignore[reportArgumentType] -- valid OpenCV noArray sentinel
        5,
    )
    print(f"ECC correlation {cc:.4f}")
    return warp, scale


def main() -> None:
    c16 = cv2.imread(str(BASE / "watermark_candidate_16.png"), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(BASE / "watermark_candidate.png"), cv2.IMREAD_GRAYSCALE)
    smooth = cv2.imread(str(CONS / "watermark_consensus_smooth.png"), cv2.IMREAD_UNCHANGED)
    c16_gray = cv2.cvtColor(c16[:, :, :3], cv2.COLOR_BGR2GRAY)

    # M maps ref coords -> prescaled-c16 coords; then undo the prescale.
    M, scale = register_to_ref(c16_gray, ref)
    S = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float64)
    M3 = np.vstack([M.astype(np.float64), [0, 0, 1]])
    comp = (np.vstack([S, [0, 0, 1]]) @ M3)[:2]  # ref -> c16 original frame

    # Clean mask lives in smooth template with +2px prep padding; shift back to ref frame.
    alpha = smooth[:, :, 3]
    T = np.array([[1, 0, -2], [0, 1, -2], [0, 0, 1]], dtype=np.float64)
    comp_from_padded = (np.vstack([comp, [0, 0, 1]]) @ T)[:2]

    sh, sw = c16_gray.shape
    warped_mask = cv2.warpAffine(
        alpha,
        comp_from_padded.astype(np.float32),
        (sw, sh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

    # Alpha = clean consensus UNION c16's own mask: preserves the native
    # candidate_16 texture while adding consensus-only strokes and the dot.
    union_mask = np.maximum(warped_mask, ((c16[:, :, 3] > 0) * 255).astype(np.uint8))
    rgb = c16[:, :, :3].copy()
    # Alpha-only additions (the composited dot, consensus-only strokes) have
    # black RGB in c16's frame -> paint them white or the remnant returns.
    alpha_only = (union_mask > 0) & (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) < 128)
    rgb[alpha_only] = 255
    print(f"whitened {int(alpha_only.sum())} alpha-only px in RGB")
    bgra = np.dstack([rgb, union_mask])
    out = CONS / "watermark_consensus_candidate16.png"
    cv2.imwrite(str(out), bgra)
    print(f"wrote {out} ({bgra.shape[1]}x{bgra.shape[0]}, "
          f"{int((union_mask > 0).sum())} mask px)")

    tmpl = _make_watermark_template(out.name, bgra)
    for iname in ("6mcg58.jpeg", "6r2q85.jpeg", "6tgglx.jpeg"):
        img = cv2.imread(str(SRC / iname))
        r = try_watermark_cascade(img, [tmpl])
        if r is None:
            print(f"{iname}: NO MATCH")
        else:
            _, bbox, _, inliers = r
            print(f"{iname}: bbox={bbox} inliers={inliers}")


if __name__ == "__main__":
    main()
