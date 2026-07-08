from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import random
from pathlib import Path

import cv2
from PIL import Image

from untextre.synthetic_text_benchmark import (
    build_even_visit_plan,
    generate_synthetic_text_case,
    iter_base_images,
)
from untextre.utils import load_image


def _render_case(spec_payload: dict, out_root: Path, quality: int) -> dict:
    base_path = Path(spec_payload["base_path"])
    clean = load_image(base_path)
    case = generate_synthetic_text_case(clean, random.Random(spec_payload["sample_seed"]))

    image_path = out_root / f"{spec_payload['sample_index']:03d}.jpg"
    mask_path = out_root / f"mask-{spec_payload['sample_index']:03d}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(case.watermarked, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(image_path, quality=quality)
    Image.fromarray(case.truth_mask).save(mask_path, quality=quality)

    row = {
        "index": spec_payload["sample_index"],
        "sample_seed": spec_payload["sample_seed"],
        "base_path": str(base_path),
        "base_relpath": str(base_path),
        "image_path": image_path.name,
        "mask_path": mask_path.name,
        "base_width": int(clean.shape[1]),
        "base_height": int(clean.shape[0]),
        **case.metadata,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic synthetic text sample corpus.")
    parser.add_argument("--base-dir", type=Path, default=Path("tests/images/zero"))
    parser.add_argument("--out-root", type=Path, default=Path("tests/images/samples_100"))
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19930)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 0 <= args.quality <= 100:
        raise ValueError("--quality must be between 0 and 100")

    args.out_root.mkdir(parents=True, exist_ok=True)
    base_paths = iter_base_images(args.base_dir)
    plan = build_even_visit_plan(base_paths, args.sample_count, args.seed)
    specs = [asdict(spec) for spec in plan]

    rows: list[dict] = []
    mode_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(_render_case, spec, args.out_root, args.quality): spec
            for spec in specs
        }
        for future in as_completed(future_map):
            row = future.result()
            rows.append(row)
            mode_counts[row["text_mode"]] += 1

    rows.sort(key=lambda row: row["index"])
    manifest = {
        "description": "Deterministic 100-sample procedural text-watermark corpus using editable text pools.",
        "base_dir": str(args.base_dir),
        "samples_dir": str(args.out_root),
        "benchmark_seed": args.seed,
        "case_count": len(rows),
        "text_mode_counts": dict(sorted(mode_counts.items())),
        "cases": rows,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_root / "README.md").write_text(
        "# Samples 100\n\n"
        "Deterministic 100-sample procedural text-watermark corpus.\n\n"
        f"- seed: {args.seed}\n"
        f"- base dir: {args.base_dir}\n"
        "- text pools: tests/images/text_sources\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_root": str(args.out_root), "case_count": len(rows), "text_mode_counts": dict(sorted(mode_counts.items()))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
