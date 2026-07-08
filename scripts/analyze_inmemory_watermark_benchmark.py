from __future__ import annotations

import argparse
from pathlib import Path

from untextre.inmemory_watermark_analysis import (
    build_batch_overview,
    build_case_rankings,
    build_factor_splits,
    discover_batch_artifacts,
    load_jsonl_rows,
    write_analysis_json,
    write_batch_overview_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze in-memory watermark benchmark batches.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    artifacts = discover_batch_artifacts(args.run_dir)
    if not artifacts:
        raise ValueError(f"No batch artifacts found under {args.run_dir}")

    overview = build_batch_overview(artifacts)
    all_rows = []
    for artifact in artifacts:
        all_rows.extend(load_jsonl_rows(artifact.jsonl_path))
    rankings = build_case_rankings(all_rows, top_n=args.top_n)
    factor_splits = build_factor_splits(all_rows)

    out_json = args.out_json or Path(args.run_dir) / "analysis.json"
    out_csv = args.out_csv or Path(args.run_dir) / "batch-overview.csv"
    write_analysis_json(overview, rankings, factor_splits, out_json)
    write_batch_overview_csv(overview, out_csv)
    print(out_json)
    print(out_csv)


if __name__ == "__main__":
    main()
