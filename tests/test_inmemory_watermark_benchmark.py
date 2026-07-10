from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from untextre.synthetic_text_benchmark import (
    SyntheticTextCase,
    build_even_visit_plan,
    iter_base_images,
    run_in_memory_watermark_benchmark,
    summarize_in_memory_watermark_benchmark,
)
from untextre.mask_experiments import truth_target_mask
from untextre.inmemory_watermark_analysis import build_factor_splits


def test_build_even_visit_plan_is_deterministic_and_balanced():
    base_paths = [Path(f"image_{idx}.png") for idx in range(4)]

    plan_one = build_even_visit_plan(base_paths, sample_count=10, seed=7)
    plan_two = build_even_visit_plan(base_paths, sample_count=10, seed=7)

    assert plan_one == plan_two

    counts = Counter(spec.base_path for spec in plan_one)
    assert sorted(counts.values()) == [2, 2, 3, 3]
    assert len({spec.base_path for spec in plan_one[:4]}) == 4


def test_build_even_visit_plan_can_continue_from_an_offset():
    base_paths = [Path(f"image_{idx}.png") for idx in range(3)]

    first = build_even_visit_plan(base_paths, sample_count=3, seed=11)
    second = build_even_visit_plan(base_paths, sample_count=3, seed=11, sample_start=3)

    assert [spec.sample_index for spec in first] == [0, 1, 2]
    assert [spec.sample_index for spec in second] == [3, 4, 5]
    assert len({spec.sample_seed for spec in first} & {spec.sample_seed for spec in second}) == 0


def test_iter_base_images_ignores_non_image_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"ok")
    (tmp_path / "b.png").write_bytes(b"ok")
    (tmp_path / "notes.csv").write_text("ignore me", encoding="utf-8")

    paths = iter_base_images(tmp_path)

    assert [path.name for path in paths] == ["a.jpg", "b.png"]


def test_run_in_memory_watermark_benchmark_keeps_generation_in_memory(tmp_path):
    base_dir = tmp_path / "zero"
    base_dir.mkdir()
    for name in ("one.png", "two.png"):
        (base_dir / name).write_bytes(b"placeholder")

    clean = np.ones((12, 16, 3), dtype=np.uint8) * 220
    truth_mask = np.zeros((12, 16), dtype=np.uint8)
    truth_mask[4:7, 5:11] = 255
    synthetic = SyntheticTextCase(
        clean=clean,
        watermarked=clean.copy(),
        truth_mask=truth_mask,
        truth_bbox=(5, 4, 6, 3),
        metadata={"text": "ImageFans.net", "font_family": "sans"},
    )

    def fake_loader(path: Path) -> np.ndarray:
        assert path.name in {"one.png", "two.png"}
        return clean

    def fake_case_builder(clean_bgr: np.ndarray, rng, *, font_dirs=None):
        assert clean_bgr.shape == clean.shape
        return synthetic

    def fake_process(image, **kwargs):
        assert image.shape == clean.shape
        assert kwargs["forced_bbox"] == synthetic.truth_bbox
        return SimpleNamespace(
            mask=truth_target_mask(truth_mask, dilation_px=2),
            timings={"total_time": 0.123},
            consensus_boxes=[synthetic.truth_bbox],
        )

    rows = run_in_memory_watermark_benchmark(
        base_dir,
        sample_count=3,
        seed=11,
        image_loader=fake_loader,
        case_builder=fake_case_builder,
        process_image_fn=fake_process,
    )

    summary = summarize_in_memory_watermark_benchmark(rows)

    assert len(rows) == 3
    assert summary["sample_count"] == 3
    assert summary["base_count"] == 2
    assert summary["mean_target_recall"] == 1.0
    assert all("sample_seed" in row for row in rows)
    assert all("base_relpath" in row for row in rows)
    assert all("truth_bbox" in row for row in rows)
    assert all("pipeline_consensus_box_count" in row for row in rows)


def test_run_in_memory_watermark_benchmark_appends_progress_rows(tmp_path):
    base_dir = tmp_path / "zero"
    base_dir.mkdir()
    for name in ("one.png", "two.png"):
        (base_dir / name).write_bytes(b"placeholder")

    clean = np.ones((12, 16, 3), dtype=np.uint8) * 220
    truth_mask = np.zeros((12, 16), dtype=np.uint8)
    truth_mask[4:7, 5:11] = 255
    synthetic = SyntheticTextCase(
        clean=clean,
        watermarked=clean.copy(),
        truth_mask=truth_mask,
        truth_bbox=(5, 4, 6, 3),
        metadata={"text": "ImageFans.net", "font_family": "sans"},
    )
    progress_csv = tmp_path / "progress.csv"

    def fake_loader(path: Path) -> np.ndarray:
        return clean

    def fake_case_builder(clean_bgr: np.ndarray, rng, *, font_dirs=None):
        return synthetic

    def fake_process(image, **kwargs):
        return SimpleNamespace(
            mask=truth_target_mask(truth_mask, dilation_px=2),
            timings={"total_time": 0.123},
            consensus_boxes=[synthetic.truth_bbox],
        )

    run_in_memory_watermark_benchmark(
        base_dir,
        sample_count=2,
        seed=11,
        image_loader=fake_loader,
        case_builder=fake_case_builder,
        process_image_fn=fake_process,
        progress_csv=progress_csv,
    )

    lines = progress_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("sample_index,")
    assert "finished_at" in lines[0]
    assert "0," in lines[1]
    assert "1," in lines[2]




def test_analyze_inmemory_watermark_benchmark_writes_batch_overview_and_rankings(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    batch0 = run_dir / "batch-000.jsonl"
    batch0.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_index": 0,
                        "base_relpath": "a.png",
                        "score": 0.2,
                        "target_recall": 0.9,
                        "weighted_precision": 0.8,
                        "fp_outside_fraction": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "sample_index": 1,
                        "base_relpath": "b.png",
                        "score": 0.4,
                        "target_recall": 0.8,
                        "weighted_precision": 0.7,
                        "fp_outside_fraction": 0.2,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    batch0.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "sample_count": 2,
                "base_count": 2,
                "mean_target_recall": 0.85,
                "min_target_recall": 0.8,
                "mean_weighted_precision": 0.75,
                "mean_overmask_ratio": 1.1,
                "max_coverage": 0.02,
                "review_cases": [{"score": 0.2}],
                "top_cases": [{"score": 0.2}],
            }
        ),
        encoding="utf-8",
    )

    from scripts import analyze_inmemory_watermark_benchmark as script

    out_json = tmp_path / "analysis.json"
    out_csv = tmp_path / "overview.csv"
    import sys as _sys

    argv = _sys.argv
    try:
        _sys.argv = [
            "analyze_inmemory_watermark_benchmark.py",
            str(run_dir),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--top-n",
            "1",
        ]
        script.main()
    finally:
        _sys.argv = argv

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["batch_overview"][0]["batch_name"] == "batch-000"
    assert payload["rankings"]["best_score"][0]["base_relpath"] == "b.png"
    assert "batch_name" in out_csv.read_text(encoding="utf-8")


def test_build_factor_splits_groups_by_opacity_font_and_image_size():
    rows = [
        {
            "opacity": 0.55,
            "font_family": "sans",
            "method": "budgeted-regional",
            "base_width": 1000,
            "base_height": 800,
            "target_px": 1500,
            "truth_bbox_coverage": 0.003,
            "pipeline_consensus_box_count": 1,
            "pipeline_mask_coverage": 0.001,
            "fp_outside_fraction": 0.05,
            "outline_present": False,
            "outline_thickness_px": 0,
            "outline_color_hex": None,
            "corner_luminance_class": "white",
            "corner": "upper_left",
            "target_recall": 1.0,
            "weighted_precision": 0.8,
            "overmask_ratio": 1.1,
            "score": 0.2,
            "predicted_px": 1600,
        },
        {
            "opacity": 0.95,
            "font_family": "serif",
            "method": "budgeted-regional",
            "base_width": 5000,
            "base_height": 4000,
            "target_px": 120000,
            "truth_bbox_coverage": 0.02,
            "pipeline_consensus_box_count": 5,
            "pipeline_mask_coverage": 0.03,
            "fp_outside_fraction": 0.8,
            "outline_present": True,
            "outline_thickness_px": 6,
            "outline_color_hex": "#7ef0ff",
            "corner_luminance_class": "black",
            "corner": "lower_right",
            "target_recall": 0.8,
            "weighted_precision": 0.7,
            "overmask_ratio": 1.3,
            "score": 0.1,
            "predicted_px": 130000,
        },
    ]

    splits = build_factor_splits(rows)

    assert [row["bucket"] for row in splits["opacity"]] == ["0.50-0.60", "0.90-1.00"]
    assert [row["bucket"] for row in splits["method"]] == ["budgeted-regional"]
    assert [row["bucket"] for row in splits["font_family"]] == ["sans", "serif"]
    assert [row["bucket"] for row in splits["image_pixels"]] == ["<1MP", "8MP+"]
    assert [row["bucket"] for row in splits["truth_pixels"]] == ["<2k", "100k+"]
    assert [row["bucket"] for row in splits["pipeline_consensus_box_count"]] == ["1", "5+"]
    assert [row["bucket"] for row in splits["pipeline_mask_coverage"]] == ["<0.25%", "2.00%+"]
    assert [row["bucket"] for row in splits["fp_outside_fraction"]] == ["<10%", "75-100%"]
    assert [row["bucket"] for row in splits["outline_present"]] == ["no", "yes"]
    assert [row["bucket"] for row in splits["outline_thickness_px"]] == ["0", "6px"]
    assert [row["bucket"] for row in splits["outline_color_kind"]] == ["none", "vivid"]
    assert [row["bucket"] for row in splits["corner_luminance_class"]] == ["black", "white"]
    assert [row["bucket"] for row in splits["corner"]] == ["lower_right", "upper_left"]
