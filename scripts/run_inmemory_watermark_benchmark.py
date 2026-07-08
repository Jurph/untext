from __future__ import annotations

import argparse
from pathlib import Path

from untextre.mask_experiments import MaskExperimentConfig
from untextre.synthetic_text_benchmark import (
    run_in_memory_watermark_benchmark,
    write_in_memory_watermark_benchmark_jsonl,
    write_in_memory_watermark_benchmark_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an in-memory synthetic watermark benchmark.")
    parser.add_argument("--base-dir", type=Path, default=Path("tests/images/zero"))
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--method", choices=("telea", "lama"), default="telea")
    parser.add_argument("--coverage-limit", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--progress-csv", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.sample_count < 0:
        raise ValueError("--sample-count must be non-negative")
    if args.top_n < 0:
        raise ValueError("--top-n must be non-negative")
    if args.coverage_limit < 0:
        raise ValueError("--coverage-limit must be non-negative")

    rows = run_in_memory_watermark_benchmark(
        args.base_dir,
        args.sample_count,
        args.seed,
        method=args.method,
        mask_config=MaskExperimentConfig(),
        coverage_limit=args.coverage_limit,
        sample_start=args.sample_start,
        preload_models=args.preload_models,
        progress_csv=args.progress_csv,
        recursive=args.recursive,
    )
    write_in_memory_watermark_benchmark_jsonl(rows, args.out)

    summary_path = args.summary_json
    if summary_path is None:
        summary_path = args.out.with_name(f"{args.out.stem}.summary.json")
    write_in_memory_watermark_benchmark_summary(rows, summary_path, top_n=args.top_n)


if __name__ == "__main__":
    main()
