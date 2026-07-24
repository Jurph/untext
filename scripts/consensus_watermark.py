"""Register several partial watermark masks and fuse them into a consensus.

Reference = watermark_candidate.png. Others are registered to it with ORB
feature matching -> affine estimate, refined (or replaced on failure) by ECC
on Gaussian-blurred copies. Outputs mean, union(max), vote-count, and
thresholded consensus images.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\cleaned-2")
REF_NAME = "watermark_candidate.png"
OTHERS = [
    "watermark_candidate_5.png",
    "watermark_candidate_11.png",
    "watermark_candidate_16.png",
]
OUT_DIR = BASE / "consensus"


def load_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        sys.exit(f"cannot read {path}")
    return im


def orb_affine(src: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    """Estimate 2x3 affine mapping src -> ref via ORB. None on failure."""
    orb = cv2.ORB_create(  # pyright: ignore[reportAttributeAccessIssue] -- runtime OpenCV API missing from stubs
        nfeatures=4000, scaleFactor=1.15, nlevels=12
    )
    # Blur slightly so ORB sees stroke shapes, not binarization speckle.
    src_b = cv2.GaussianBlur(src, (5, 5), 0)
    ref_b = cv2.GaussianBlur(ref, (5, 5), 0)
    kp1, des1 = orb.detectAndCompute(src_b, None)
    kp2, des2 = orb.detectAndCompute(ref_b, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    pts1 = np.asarray([kp1[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray([kp2[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 2)
    M, inliers = cv2.estimateAffinePartial2D(  # pyright: ignore[reportCallIssue]
        pts1,  # pyright: ignore[reportArgumentType] -- ndarray overload missing from stubs
        pts2,  # pyright: ignore[reportArgumentType] -- ndarray overload missing from stubs
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if M is None or inliers is None or inliers.sum() < 6:
        return None
    print(f"    ORB: {len(good)} good matches, {int(inliers.sum())} inliers")
    return M


def ecc_refine(
    src: np.ndarray, ref: np.ndarray, init: np.ndarray | None
) -> np.ndarray | None:
    """Refine/estimate affine with ECC on blurred float images."""
    warp = np.eye(2, 3, dtype=np.float32) if init is None else init.astype(np.float32)
    src_f = cv2.GaussianBlur(src, (9, 9), 0).astype(np.float32) / 255.0
    ref_f = cv2.GaussianBlur(ref, (9, 9), 0).astype(np.float32) / 255.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    try:
        cc, warp = cv2.findTransformECC(  # pyright: ignore[reportCallIssue]
            ref_f,
            src_f,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,  # pyright: ignore[reportArgumentType] -- valid OpenCV noArray sentinel
            5,
        )
        print(f"    ECC: correlation {cc:.4f}")
        return warp
    except cv2.error as e:
        print(f"    ECC failed: {e}")
        return None


def main() -> None:
    ref = load_gray(BASE / REF_NAME)
    rh, rw = ref.shape
    print(f"reference {REF_NAME}: {rw}x{rh}")

    aligned = [ref.astype(np.float32) / 255.0]
    for name in OTHERS:
        src = load_gray(BASE / name)
        sh, sw = src.shape
        # Pre-scale to reference width so ORB/ECC start near identity.
        scale = rw / sw
        src_s = cv2.resize(
            src, (rw, max(1, round(sh * scale))), interpolation=cv2.INTER_AREA
        )
        print(f"{name}: {sw}x{sh} -> prescaled {src_s.shape[1]}x{src_s.shape[0]}")

        M = orb_affine(src_s, ref)
        M = ecc_refine(src_s, ref, M)  # ECC refines ORB result or runs from identity
        if M is None:
            print("    WARNING: registration failed, padding/cropping prescale")
            warped = np.zeros((rh, rw), dtype=src_s.dtype)
            h = min(rh, src_s.shape[0])
            warped[:h, :] = src_s[:h, :rw]
        else:
            # ECC's warp maps ref coords -> src coords; use WARP_INVERSE_MAP
            # to pull src into the reference frame.
            warped = cv2.warpAffine(
                src_s,
                M,
                (rw, rh),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )
        aligned.append(warped.astype(np.float32) / 255.0)

    stack = np.stack(aligned)  # (4, H, W) in [0,1]
    OUT_DIR.mkdir(exist_ok=True)

    mean = stack.mean(axis=0)
    union = stack.max(axis=0)
    votes = (stack > 0.5).sum(axis=0).astype(np.float32)  # 0..4
    at_least_2 = (votes >= 2).astype(np.uint8) * 255
    majority = (votes >= 3).astype(np.uint8) * 255

    outputs = {
        "consensus_mean.png": (mean * 255).astype(np.uint8),
        "consensus_union.png": (union * 255).astype(np.uint8),
        "consensus_votes.png": (votes / 4 * 255).astype(np.uint8),
        "consensus_atleast2.png": at_least_2,
        "consensus_majority3.png": majority,
    }
    for fname, im in outputs.items():
        cv2.imwrite(str(OUT_DIR / fname), im)
        print(f"wrote {OUT_DIR / fname}")

    # Per-input aligned copies for eyeball QA.
    for name, im in zip([REF_NAME, *OTHERS], aligned):
        out = OUT_DIR / f"aligned_{name}"
        cv2.imwrite(str(out), (im * 255).astype(np.uint8))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
