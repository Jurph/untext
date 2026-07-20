"""Composite the missing dot into the consensus mask, despeckle, and export
a SIFT-prepped BGRA template ready for untextre -K mode.

Base mask = consensus_atleast2.png (>=2-of-4 vote, already de-speckled by
voting). The '.' in 'StasyQ.com' appears only in the reference slice, so we
recover it by adding reference components that (a) do not touch the base mask
and (b) are large enough to be a real glyph piece, not speckle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from untextre.sift_prep import prepare_candidate_bgra_for_sift  # noqa: E402

CONS = Path(r"G:\Documents and Settings\Jurph\Old\My Pictures\Zero\cleaned-2\consensus")
MIN_GLYPH_AREA = 20   # px: reference components smaller than this are speckle
MIN_KEEP_AREA = 12    # px: final despeckle floor for the composite


def components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n, labels, stats


def main() -> None:
    base = cv2.imread(str(CONS / "consensus_atleast2.png"), cv2.IMREAD_GRAYSCALE)
    if base is None:
        sys.exit("missing consensus inputs; run consensus_watermark.py first")
    base_bin = (base > 127).astype(np.uint8)

    # ── Recover glyphs missed by voting (the '.' lives only in slice 5) ──
    # A donor component qualifies if it barely overlaps the consensus, is big
    # enough to be a glyph piece, and is compact (dot-like, not halo fuzz).
    donors = [
        "aligned_watermark_candidate.png",
        "aligned_watermark_candidate_5.png",
        "aligned_watermark_candidate_11.png",
        "aligned_watermark_candidate_16.png",
    ]
    added = 0
    for donor in donors:
        im = cv2.imread(str(CONS / donor), cv2.IMREAD_GRAYSCALE)
        if im is None:
            sys.exit(f"missing {donor}; run consensus_watermark.py first")
        donor_bin = (im > 127).astype(np.uint8)
        n, labels, stats = components(donor_bin)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_GLYPH_AREA or w > 30 or h > 30:
                continue  # too small (speckle) or too big (letter/halo chunk)
            if area / (w * h) < 0.5:
                continue  # not compact
            comp = labels == i
            overlap = int(base_bin[comp].sum()) / area
            if overlap < 0.3:
                base_bin[comp] = 1
                print(f"composited from {donor}: {w}x{h} at ({x},{y}), area {area}")
                added += 1
    if added == 0:
        print("WARNING: found no missing glyph components to composite")

    # ── Despeckle: drop tiny components, fill pinholes ───────────────────
    n, labels, stats = components(base_bin)
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_KEEP_AREA:
            base_bin[labels == i] = 0
            removed += 1
    print(f"despeckle: removed {removed} components < {MIN_KEEP_AREA}px")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    base_bin = cv2.morphologyEx(base_bin, cv2.MORPH_CLOSE, kernel)

    # ── Black -> alpha, then SIFT-prep exactly like -U's saved templates ──
    mask255 = base_bin * 255
    bgra = np.dstack([mask255, mask255, mask255, mask255])  # white RGB, alpha=mask
    prepared = prepare_candidate_bgra_for_sift(bgra)
    # Prep can add alpha px with black RGB; whiten them.
    prepared[prepared[:, :, 3] > 0, :3] = 255

    out = CONS / "watermark_consensus.png"
    cv2.imwrite(str(out), prepared)
    h, w = prepared.shape[:2]
    on = int((prepared[:, :, 3] > 0).sum())
    print(f"wrote {out} ({w}x{h} BGRA, {on} opaque px)")


if __name__ == "__main__":
    main()
