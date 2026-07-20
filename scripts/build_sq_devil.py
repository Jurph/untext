"""Build and test SQ-devil known-mask templates from logo.svg + fuzzy -U candidates.

Ground truth shape comes from logo.svg's masked icon (`mask0` / embedded image0),
not the STASYQ text path. Candidate masks 2/6/12 give observed scale/alignment and
matching texture. Export clean templates into consensus/sq-devil-k-templates.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from untextre.sift_matcher import load_watermark_templates, try_watermark_cascade  # noqa: E402
from untextre.sift_prep import prepare_candidate_bgra_for_sift  # noqa: E402

BASE = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\cleaned-2")
SVG = BASE / "logo.svg"
CONS = BASE / "consensus"
OUT = CONS / "sq-devil-k-templates"
TARGET = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\has-SQ-logo\old logo")
CANDIDATES = ["watermark_candidate_2.png", "watermark_candidate_6.png", "watermark_candidate_12.png"]
PURPLE_BGR = np.array([0xC1, 0x51, 0x77], dtype=np.uint8)  # #7751C1 in BGR


def extract_svg_mask() -> np.ndarray:
    text = SVG.read_text(encoding="utf-8")
    m = re.search(r"base64,([^\"]+)", text)
    if not m:
        raise RuntimeError("no embedded image found in SVG")
    raw = base64.b64decode(m.group(1))
    tmp = CONS / "_sq_svg_embedded.png"
    tmp.write_bytes(raw)
    rgba = np.array(Image.open(tmp).convert("RGBA"))
    # SVG masks use luminance multiplied by alpha. The embedded PNG is an
    # opaque black-background image; using alpha alone would mask the whole box.
    rgb = rgba[:, :, :3].astype(np.float32)
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2])
    alpha = (lum * (rgba[:, :, 3].astype(np.float32) / 255.0)).astype(np.uint8)
    alpha[alpha < 8] = 0
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if coords is None:
        raise RuntimeError("empty SVG mask")
    x, y, w, h = cv2.boundingRect(coords)
    mask = alpha[y:y + h, x:x + w]
    cv2.imwrite(str(CONS / "_sq_svg_mask.png"), mask)
    print(f"SVG embedded {rgba.shape[1]}x{rgba.shape[0]}, crop {w}x{h} at ({x},{y})")
    return mask


def crop_bgra(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        raise RuntimeError(f"bad candidate {path}")
    alpha = im[:, :, 3]
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if coords is None:
        raise RuntimeError(f"empty candidate {path}")
    x, y, w, h = cv2.boundingRect(coords)
    return im[y:y + h, x:x + w].copy()


def build_template(svg_mask: np.ndarray, cand_bgra: np.ndarray, name: str, use_candidate_rgb: bool = True) -> Path:
    cand_alpha = cand_bgra[:, :, 3]
    h, w = cand_alpha.shape
    # Render SVG shape at candidate observed scale.
    shape = cv2.resize(svg_mask, (w, h), interpolation=cv2.INTER_CUBIC)
    _, shape = cv2.threshold(shape, 32, 255, cv2.THRESH_BINARY)
    # Align shape to candidate alpha by ECC translation/affine on blurred masks.
    shape_f = cv2.GaussianBlur(shape, (7, 7), 0).astype(np.float32) / 255
    cand_f = cv2.GaussianBlur(cand_alpha, (7, 7), 0).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(cand_f, shape_f, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7), None, 5)
        shape = cv2.warpAffine(shape, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, shape = cv2.threshold(shape, 96, 255, cv2.THRESH_BINARY)
        print(f"{name}: ECC {cc:.4f}")
    except cv2.error as e:
        print(f"{name}: ECC failed: {e}")

    # Clean alpha from SVG shape, unioned with candidate fuzz so the visible
    # template bytes still provide SIFT texture/gradients. RGB comes from
    # candidate pixels: the production mark is black/dark S plus gray devil-Q,
    # not the purple brand color in the SVG.
    out_alpha = np.maximum(shape, (cand_alpha > 0).astype(np.uint8) * 255)

    if use_candidate_rgb:
        bgr = cand_bgra[:, :, :3].copy()
        # Alpha-only SVG additions need visible descriptor/color bytes.
        opaque_rgb = bgr[cand_alpha > 0]
        fill = np.median(opaque_rgb, axis=0).astype(np.uint8) if len(opaque_rgb) else np.array([60, 60, 60], dtype=np.uint8)
        bgr[(out_alpha > 0) & (cand_alpha == 0)] = fill
    else:
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[out_alpha > 0] = PURPLE_BGR

    bgra = np.dstack([bgr, out_alpha])
    prepared = prepare_candidate_bgra_for_sift(bgra)
    # Fix healed alpha-only pixels without flattening legitimate dark S pixels:
    # only touch places added by prep where all BGR are still zero.
    black_under_alpha = (prepared[:, :, 3] > 0) & (prepared[:, :, :3].max(axis=2) == 0)
    prepared[black_under_alpha, :3] = np.median(bgr[out_alpha > 0], axis=0).astype(np.uint8)
    out = OUT / name
    cv2.imwrite(str(out), prepared)
    print(f"wrote {out.name}: {prepared.shape[1]}x{prepared.shape[0]}, alpha_px={int((prepared[:,:,3]>0).sum())}")
    return out


def sweep(template_dir: Path) -> tuple[int, list[str]]:
    templates = load_watermark_templates(template_dir)
    images = sorted(p for p in TARGET.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    hits = 0
    misses: list[str] = []
    for p in images:
        img = cv2.imread(str(p))
        r = try_watermark_cascade(img, templates)
        if r is None:
            misses.append(p.name)
        else:
            hits += 1
            _, bbox, name, inliers = r
            print(f"{p.name}: {name} inliers={inliers} bbox={bbox}")
    print(f"SWEEP {hits}/{len(images)} localized; misses={misses}")
    return hits, misses


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"):
        p.unlink()
    svg_mask = extract_svg_mask()
    for cname in CANDIDATES:
        cand = crop_bgra(BASE / cname)
        build_template(svg_mask, cand, f"SQ-devil_{Path(cname).stem}.png", True)
    # Pure SVG at median observed candidate size, kept as a visual/control asset.
    cands = [crop_bgra(BASE / c) for c in CANDIDATES]
    med_h = round(float(np.median([c.shape[0] for c in cands])))
    med_w = round(float(np.median([c.shape[1] for c in cands])))
    pure_alpha = cv2.resize(svg_mask, (med_w, med_h), interpolation=cv2.INTER_CUBIC)
    _, pure_alpha = cv2.threshold(pure_alpha, 32, 255, cv2.THRESH_BINARY)
    pure_bgra = np.dstack([np.full((med_h, med_w), PURPLE_BGR[0], dtype=np.uint8),
                           np.full((med_h, med_w), PURPLE_BGR[1], dtype=np.uint8),
                           np.full((med_h, med_w), PURPLE_BGR[2], dtype=np.uint8),
                           pure_alpha])
    build_template(svg_mask, pure_bgra, "SQ-devil_svg-purple.png", False)
    sweep(OUT)


if __name__ == "__main__":
    main()
