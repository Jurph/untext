from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
from PIL import Image

from untextre.detector_pair_harvest import append_jsonl, build_pair_row, pair_id_for_path
from untextre.synthetic_text_benchmark import generate_synthetic_text_case, iter_base_images
from untextre.utils import load_image


def save_bgr_jpeg(image_bgr, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, quality=quality)


def save_mask_png(mask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def completed_pair_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {
        json.loads(line)["pair_id"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired clean/synthetic detector harvest corpus.")
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("tests/images/detector_pair_harvest"))
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root
    pairs_dir = out_root / "pairs"
    manifest_path = pairs_dir / "pair_manifest.jsonl"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    images = iter_base_images(args.clean_dir)
    if args.limit is not None:
        images = images[: args.limit]

    started = time.time()
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()

    completed = completed_pair_ids(manifest_path) if args.resume else set()
    rows_written = 0
    for index, image_path in enumerate(images):
        clean_relative = image_path.relative_to(args.clean_dir)
        pair_id = pair_id_for_path(clean_relative)
        if pair_id in completed:
            continue

        clean = load_image(image_path)
        rng = random.Random(f"{args.seed}:{index}:{image_path.name}")
        case = generate_synthetic_text_case(clean, rng)
        twin_rel = Path("pairs") / "synthetic_twins" / f"{pair_id}.jpg"
        mask_rel = Path("pairs") / "truth_masks" / f"{pair_id}.png"

        save_bgr_jpeg(case.watermarked, out_root / twin_rel, quality=args.quality)
        save_mask_png(case.truth_mask, out_root / mask_rel)

        case_metadata = {**case.metadata, "truth_mask_relative_path": mask_rel.as_posix()}
        row = build_pair_row(
            pair_id,
            clean_relative.as_posix(),
            twin_rel.as_posix(),
            case_metadata,
            case.truth_bbox,
            clean.shape[1],
            clean.shape[0],
        )
        append_jsonl(manifest_path, row)
        rows_written += 1
        print(f"[{index + 1}/{len(images)}] wrote {pair_id}", flush=True)

    top_manifest = {
        "clean_dir": str(args.clean_dir),
        "out_root": str(out_root),
        "seed": args.seed,
        "image_count": len(images),
        "rows_written_this_run": rows_written,
        "elapsed_seconds": time.time() - started,
        "schema": "detector_pair_harvest.v1",
    }
    (out_root / "manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
