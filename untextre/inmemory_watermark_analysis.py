"""Analysis helpers for in-memory synthetic watermark benchmark runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchArtifact:
    batch_name: str
    jsonl_path: Path
    summary_path: Path


def discover_batch_artifacts(run_dir: Path) -> list[BatchArtifact]:
    run_dir = Path(run_dir)
    artifacts: list[BatchArtifact] = []
    for jsonl_path in sorted(run_dir.glob("batch-*.jsonl")):
        summary_path = jsonl_path.with_suffix(".summary.json")
        artifacts.append(
            BatchArtifact(
                batch_name=jsonl_path.stem,
                jsonl_path=jsonl_path,
                summary_path=summary_path,
            )
        )
    return artifacts


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_batch_overview(artifacts: list[BatchArtifact]) -> list[dict]:
    overview = []
    for artifact in artifacts:
        summary = load_summary(artifact.summary_path)
        overview.append(
            {
                "batch_name": artifact.batch_name,
                "sample_count": summary["sample_count"],
                "base_count": summary["base_count"],
                "mean_target_recall": summary["mean_target_recall"],
                "min_target_recall": summary["min_target_recall"],
                "mean_weighted_precision": summary["mean_weighted_precision"],
                "mean_overmask_ratio": summary["mean_overmask_ratio"],
                "max_coverage": summary["max_coverage"],
                "best_score": summary["top_cases"][0]["score"] if summary["top_cases"] else None,
                "worst_score": summary["review_cases"][0]["score"] if summary["review_cases"] else None,
                "jsonl_path": str(artifact.jsonl_path),
                "summary_path": str(artifact.summary_path),
            }
        )
    return overview


def build_case_rankings(rows: list[dict], top_n: int = 25) -> dict:
    rows = list(rows)
    ranked_by_score = sorted(rows, key=lambda row: (row.get("score", 0.0), row.get("sample_index", 0)))
    ranked_by_score_desc = list(reversed(ranked_by_score))
    ranked_by_recall = sorted(
        rows,
        key=lambda row: (row.get("target_recall", 0.0), row.get("sample_index", 0)),
    )
    ranked_by_outside_fp = sorted(
        rows,
        key=lambda row: (row.get("fp_outside_fraction", 0.0), row.get("sample_index", 0)),
        reverse=True,
    )
    return {
        "worst_score": ranked_by_score[:top_n],
        "best_score": ranked_by_score_desc[:top_n],
        "worst_recall": ranked_by_recall[:top_n],
        "most_outside_fp": ranked_by_outside_fp[:top_n],
    }


def build_factor_splits(rows: list[dict]) -> dict:
    rows = list(rows)
    opacity_order = {
        "0.50-0.60": 0,
        "0.60-0.70": 1,
        "0.70-0.80": 2,
        "0.80-0.90": 3,
        "0.90-1.00": 4,
    }
    image_size_order = {
        "<1MP": 0,
        "1-2MP": 1,
        "2-4MP": 2,
        "4-8MP": 3,
        "8MP+": 4,
    }
    truth_px_order = {
        "<2k": 0,
        "2k-5k": 1,
        "5k-10k": 2,
        "10k-25k": 3,
        "25k-50k": 4,
        "50k-100k": 5,
        "100k+": 6,
    }
    coverage_order = {
        "0": 0,
        "<0.25%": 1,
        "0.25-0.50%": 2,
        "0.50-1.00%": 3,
        "1.00-2.00%": 4,
        "2.00%+": 5,
    }
    consensus_order = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3-4": 3,
        "5+": 4,
    }
    fraction_order = {
        "0": 0,
        "<10%": 1,
        "10-25%": 2,
        "25-50%": 3,
        "50-75%": 4,
        "75-100%": 5,
    }
    thickness_order = {
        "0": 0,
        "2px": 1,
        "3px": 2,
        "4px": 3,
        "5px": 4,
        "6px": 5,
    }
    return {
        "method": _bucketed_metrics(rows, lambda row: str(row.get("method", "unknown"))),
        "opacity": _bucketed_metrics(
            rows,
            lambda row: _opacity_bucket(float(row.get("opacity", 0.0))),
            order=opacity_order,
        ),
        "font_family": _bucketed_metrics(rows, lambda row: str(row.get("font_family", "unknown"))),
        "image_pixels": _bucketed_metrics(
            rows,
            lambda row: _image_size_bucket(int(row.get("base_width", 0)), int(row.get("base_height", 0))),
            order=image_size_order,
        ),
        "truth_pixels": _bucketed_metrics(
            rows,
            lambda row: _truth_px_bucket(int(row.get("target_px", 0))),
            order=truth_px_order,
        ),
        "truth_bbox_coverage": _bucketed_metrics(
            rows,
            lambda row: _coverage_bucket(float(row.get("truth_bbox_coverage", 0.0))),
            order=coverage_order,
        ),
        "pipeline_consensus_box_count": _bucketed_metrics(
            rows,
            lambda row: _consensus_bucket(int(row.get("pipeline_consensus_box_count", 0))),
            order=consensus_order,
        ),
        "pipeline_mask_coverage": _bucketed_metrics(
            rows,
            lambda row: _coverage_bucket(float(row.get("pipeline_mask_coverage", 0.0))),
            order=coverage_order,
        ),
        "fp_outside_fraction": _bucketed_metrics(
            rows,
            lambda row: _fraction_bucket(float(row.get("fp_outside_fraction", 0.0))),
            order=fraction_order,
        ),
        "outline_present": _bucketed_metrics(
            rows,
            lambda row: "yes" if bool(row.get("outline_present", False)) else "no",
        ),
        "outline_thickness_px": _bucketed_metrics(
            rows,
            lambda row: _outline_thickness_bucket(int(row.get("outline_thickness_px", 0))),
            order=thickness_order,
        ),
        "outline_color_kind": _bucketed_metrics(
            rows,
            lambda row: _outline_color_kind(row.get("outline_color_hex")),
        ),
        "corner_luminance_class": _bucketed_metrics(
            rows,
            lambda row: str(row.get("corner_luminance_class", "unknown")),
        ),
        "corner": _bucketed_metrics(rows, lambda row: str(row.get("corner", "unknown"))),
    }


def write_batch_overview_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_name",
        "sample_count",
        "base_count",
        "mean_target_recall",
        "min_target_recall",
        "mean_weighted_precision",
        "mean_overmask_ratio",
        "max_coverage",
        "best_score",
        "worst_score",
        "jsonl_path",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_analysis_json(
    overview: list[dict],
    rankings: dict,
    factor_splits: dict,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_overview": overview,
        "rankings": rankings,
        "factor_splits": factor_splits,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bucketed_metrics(rows: list[dict], bucket_fn, order: dict[str, int] | None = None) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(bucket_fn(row), []).append(row)
    output = []
    if order is None:
        iterator = sorted(grouped.items(), key=lambda item: item[0])
    else:
        iterator = sorted(grouped.items(), key=lambda item: (order.get(item[0], 999), item[0]))
    for bucket, items in iterator:
        output.append(
            {
                "bucket": bucket,
                "count": len(items),
                "mean_target_recall": _mean(item["target_recall"] for item in items),
                "mean_weighted_precision": _mean(item["weighted_precision"] for item in items),
                "mean_overmask_ratio": _mean(item.get("overmask_ratio", 0.0) for item in items),
                "mean_score": _mean(item.get("score", 0.0) for item in items),
                "mean_truth_px": _mean(item.get("target_px", 0.0) for item in items),
                "mean_pipeline_px": _mean(item.get("predicted_px", 0.0) for item in items),
                "mean_opacity": _mean(item.get("opacity", 0.0) for item in items),
            }
        )
    return output


def _opacity_bucket(opacity: float) -> str:
    if opacity < 0.6:
        return "0.50-0.60"
    if opacity < 0.7:
        return "0.60-0.70"
    if opacity < 0.8:
        return "0.70-0.80"
    if opacity < 0.9:
        return "0.80-0.90"
    return "0.90-1.00"


def _image_size_bucket(width: int, height: int) -> str:
    mp = (width * height) / 1_000_000.0
    if mp < 1.0:
        return "<1MP"
    if mp < 2.0:
        return "1-2MP"
    if mp < 4.0:
        return "2-4MP"
    if mp < 8.0:
        return "4-8MP"
    return "8MP+"


def _truth_px_bucket(truth_px: int) -> str:
    if truth_px < 2_000:
        return "<2k"
    if truth_px < 5_000:
        return "2k-5k"
    if truth_px < 10_000:
        return "5k-10k"
    if truth_px < 25_000:
        return "10k-25k"
    if truth_px < 50_000:
        return "25k-50k"
    if truth_px < 100_000:
        return "50k-100k"
    return "100k+"


def _coverage_bucket(coverage: float) -> str:
    if coverage == 0.0:
        return "0"
    if coverage < 0.0025:
        return "<0.25%"
    if coverage < 0.005:
        return "0.25-0.50%"
    if coverage < 0.01:
        return "0.50-1.00%"
    if coverage < 0.02:
        return "1.00-2.00%"
    return "2.00%+"


def _consensus_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def _fraction_bucket(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 0.10:
        return "<10%"
    if value < 0.25:
        return "10-25%"
    if value < 0.50:
        return "25-50%"
    if value < 0.75:
        return "50-75%"
    return "75-100%"


def _outline_thickness_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    return f"{value}px"


def _outline_color_kind(hex_value: object) -> str:
    if not hex_value:
        return "none"
    value = str(hex_value).lower()
    if value == "#000000":
        return "black"
    if value == "#ffffff":
        return "white"
    return "vivid"


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
