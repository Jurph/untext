from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from untextre.mask_experiments import MaskExperimentConfig
from untextre.pipeline import initialize_consensus_models
from untextre.synthetic_text_benchmark import (
    run_in_memory_watermark_benchmark,
    summarize_in_memory_watermark_benchmark,
    write_in_memory_watermark_benchmark_jsonl,
    write_in_memory_watermark_benchmark_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an overnight in-memory watermark benchmark.")
    parser.add_argument("--base-dir", type=Path, default=Path("tests/images/zero"))
    parser.add_argument("--out-root", type=Path, default=Path("tests/images/zero_synthetic_watermark_benchmark"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--time-budget-hours", type=float, default=5.5)
    parser.add_argument("--method", choices=("telea", "lama"), default="telea")
    parser.add_argument("--coverage-limit", type=float, default=0.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--preload-models", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    if args.time_budget_hours <= 0:
        raise ValueError("--time-budget-hours must be positive")
    if args.coverage_limit < 0:
        raise ValueError("--coverage-limit must be non-negative")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("overnight-%Y%m%d-%H%M%S")
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_csv = run_dir / "progress.csv"

    if args.preload_models:
        initialize_consensus_models(device="cuda")

    batches: list[dict] = []
    combined_rows: list[dict] = []
    started = datetime.now(timezone.utc)
    start_ts = started.timestamp()

    for batch_index in range(args.max_batches):
        elapsed_hours = (datetime.now(timezone.utc).timestamp() - start_ts) / 3600.0
        if elapsed_hours >= args.time_budget_hours:
            break

        sample_start = batch_index * args.batch_size
        batch_name = f"batch-{batch_index:03d}"
        batch_out = run_dir / f"{batch_name}.jsonl"
        batch_summary = run_dir / f"{batch_name}.summary.json"
        batch_started = datetime.now(timezone.utc)

        rows = run_in_memory_watermark_benchmark(
            args.base_dir,
            args.batch_size,
            args.seed,
            method=args.method,
            mask_config=MaskExperimentConfig(),
            coverage_limit=args.coverage_limit,
            sample_start=sample_start,
            preload_models=False,
            progress_csv=progress_csv,
            recursive=args.recursive,
        )
        write_in_memory_watermark_benchmark_jsonl(rows, batch_out)
        write_in_memory_watermark_benchmark_summary(rows, batch_summary, top_n=25)
        combined_rows.extend(rows)

        batch_finished = datetime.now(timezone.utc)
        batches.append(
            {
                "batch_index": batch_index,
                "batch_name": batch_name,
                "sample_start": sample_start,
                "sample_count": len(rows),
                "out": str(batch_out),
                "summary": str(batch_summary),
                "started_at": batch_started.isoformat(),
                "finished_at": batch_finished.isoformat(),
                "elapsed_seconds": (batch_finished - batch_started).total_seconds(),
            }
        )

        combined_summary = summarize_in_memory_watermark_benchmark(combined_rows, top_n=50)
        (run_dir / "combined.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in combined_rows) + ("\n" if combined_rows else ""),
            encoding="utf-8",
        )
        (run_dir / "combined.summary.json").write_text(
            json.dumps(combined_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    finished = datetime.now(timezone.utc)
    manifest = {
        "run_name": run_name,
        "base_dir": str(args.base_dir),
        "out_root": str(args.out_root),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "time_budget_hours": args.time_budget_hours,
        "method": args.method,
        "coverage_limit": args.coverage_limit,
        "preload_models": args.preload_models,
        "progress_csv": str(progress_csv),
        "batches": batches,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
