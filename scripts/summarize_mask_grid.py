from __future__ import annotations

import argparse
import json
from pathlib import Path

from untextre.mask_experiments import rank_mask_summaries, summarize_grid_rows, write_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and rank mask grid JSONL results.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-json", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    ranked = rank_mask_summaries(summarize_grid_rows(rows))
    write_summary_csv(ranked, args.out)
    args.top_json.parent.mkdir(parents=True, exist_ok=True)
    args.top_json.write_text(json.dumps(ranked[: args.top_n], indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
