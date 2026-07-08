from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from untextre.synthetic_text_benchmark import replay_synthetic_text_case
from untextre.utils import load_image


def _save_jpeg(image: np.ndarray, path: Path, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, quality=quality)


def _save_mask(mask: np.ndarray, path: Path, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path, quality=quality)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved synthetic text samples from a manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("tests/images/samples/manifest.json"))
    parser.add_argument("--out-root", type=Path, default=Path("tests/images/samples_replayed"))
    parser.add_argument("--compare-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--write-masks", action="store_true")
    parser.add_argument("--verify-masks", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.quality < 0 or args.quality > 100:
        raise ValueError("--quality must be between 0 and 100")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = list(manifest["cases"])
    if args.limit:
        cases = cases[: args.limit]

    compare_root = args.compare_root or args.manifest.parent
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    matched_images = 0
    matched_masks = 0
    for case in cases:
        source_path = Path(case["source_relpath"])
        clean = load_image(source_path)
        replayed = replay_synthetic_text_case(clean, case)

        image_out = out_root / case["image_path"]
        mask_out = out_root / case["mask_path"]
        _save_jpeg(replayed.watermarked, image_out, quality=args.quality)
        if args.write_masks or args.verify_masks:
            _save_mask(replayed.truth_mask, mask_out, quality=args.quality)

        if compare_root is not None:
            expected_image = cv2.imread(str(compare_root / case["image_path"]), cv2.IMREAD_COLOR)
            actual_image = cv2.imread(str(image_out), cv2.IMREAD_COLOR)
            if expected_image is not None and actual_image is not None and np.array_equal(expected_image, actual_image):
                matched_images += 1
            elif args.strict:
                raise SystemExit(f"image mismatch: {case['image_path']}")

            if args.verify_masks:
                expected_mask = cv2.imread(str(compare_root / case["mask_path"]), cv2.IMREAD_GRAYSCALE)
                actual_mask = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
                if expected_mask is not None and actual_mask is not None and np.array_equal(expected_mask, actual_mask):
                    matched_masks += 1
                elif args.strict:
                    raise SystemExit(f"mask mismatch: {case['mask_path']}")

    manifest_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "out_root": str(out_root),
        "compare_root": str(compare_root) if compare_root is not None else None,
        "case_count": len(cases),
        "image_matches": matched_images if compare_root is not None else None,
        "mask_matches": matched_masks if args.verify_masks and compare_root is not None else None,
        "quality": args.quality,
        "write_masks": args.write_masks,
        "verify_masks": args.verify_masks,
        "cases": cases,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
