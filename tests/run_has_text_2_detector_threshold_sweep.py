from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import untextre.consensus as consensus_mod
from harvest_has_text_2_pipeline_bboxes import empty_review, run_one, safe_id, write_summary
from untextre.utils import IMAGE_EXTENSIONS


DEFAULT_THRESHOLDS = [0.30, 0.15, 0.10, 0.05, 0.025]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic has-text-2 full-pipeline detector threshold sweeps."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("tests/images/has_text_2_detector_threshold_sweep"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Detector confidence thresholds to run, as fractions.",
    )
    parser.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--color-sensitivity", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Run each threshold in fresh worker chunks to bound native/GPU memory growth.",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-summary", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.threshold is None:
            raise SystemExit("--worker requires --threshold")
        run_worker(args)
        return

    if args.chunk_size:
        run_chunked_supervisor(args)
    else:
        run_supervisor(args)


def get_image_paths(input_dir: Path, limit: int | None = None) -> list[Path]:
    image_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def run_supervisor(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": str(args.input_dir),
        "thresholds": args.thresholds,
        "color_sensitivity": args.color_sensitivity,
        "limit": args.limit,
        "started_at_epoch": time.time(),
        "notes": [
            "Each threshold is a full production-style bbox-of-record pass.",
            "Detector wrapper floor is lowered inside worker processes only.",
            "Use records for later human/multimodal precision-recall review.",
        ],
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    processes: list[tuple[float, subprocess.Popen]] = []
    pending = list(args.thresholds)
    while pending or processes:
        while pending and len(processes) < max(args.max_parallel, 1):
            threshold = pending.pop(0)
            out_dir = args.out_root / threshold_id(threshold)
            log_dir = args.out_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout = (log_dir / f"{threshold_id(threshold)}.stdout.log").open("a", encoding="utf-8")
            stderr = (log_dir / f"{threshold_id(threshold)}.stderr.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                __file__,
                str(args.input_dir),
                "--out-root",
                str(args.out_root),
                "--threshold",
                str(threshold),
                "--color-sensitivity",
                str(args.color_sensitivity),
                "--worker",
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.no_summary:
                command.append("--no-summary")
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=stdout, stderr=stderr)
            processes.append((threshold, process))
            print(f"started {threshold_id(threshold)} pid={process.pid} out={out_dir}", flush=True)

        time.sleep(30)
        still_running = []
        for threshold, process in processes:
            code = process.poll()
            if code is None:
                still_running.append((threshold, process))
                continue
            print(f"finished {threshold_id(threshold)} exit={code}", flush=True)
        processes = still_running

    summarize_root(args.out_root)


def run_chunked_supervisor(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    image_paths = get_image_paths(args.input_dir, args.limit)
    manifest = {
        "input_dir": str(args.input_dir),
        "thresholds": args.thresholds,
        "color_sensitivity": args.color_sensitivity,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "max_retries": args.max_retries,
        "started_at_epoch": time.time(),
        "notes": [
            "Chunked mode launches fresh worker processes to bound native/GPU memory retention.",
            "Nonzero chunk exits are retried, then isolated to single images with crash records.",
            "Detector wrapper floor is lowered inside worker processes only.",
        ],
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for threshold in args.thresholds:
        out_dir = args.out_root / threshold_id(threshold)
        (out_dir / "records").mkdir(parents=True, exist_ok=True)
        for start in range(0, len(image_paths), args.chunk_size):
            count = min(args.chunk_size, len(image_paths) - start)
            if chunk_complete(args.input_dir, image_paths[start:start + count], out_dir):
                print(
                    f"skip complete {threshold_id(threshold)} start={start} count={count}",
                    flush=True,
                )
                continue

            code = run_chunk(args, threshold, start, count)
            attempts = 0
            while code != 0 and attempts < args.max_retries:
                attempts += 1
                print(
                    f"retry {threshold_id(threshold)} start={start} count={count} "
                    f"attempt={attempts} previous_exit={code}",
                    flush=True,
                )
                code = run_chunk(args, threshold, start, count)

            if code != 0:
                print(
                    f"isolate failed chunk {threshold_id(threshold)} start={start} "
                    f"count={count} exit={code}",
                    flush=True,
                )
                isolate_failed_chunk(args, threshold, image_paths, start, count)

            write_threshold_summary(out_dir)
            summarize_root(args.out_root)

    summarize_root(args.out_root)


def run_chunk(args: argparse.Namespace, threshold: float, start: int, count: int) -> int:
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = f"{threshold_id(threshold)}_{start:04d}_{start + count - 1:04d}"
    stdout_path = log_dir / f"{chunk_name}.stdout.log"
    stderr_path = log_dir / f"{chunk_name}.stderr.log"
    command = [
        sys.executable,
        __file__,
        str(args.input_dir),
        "--out-root",
        str(args.out_root),
        "--threshold",
        str(threshold),
        "--color-sensitivity",
        str(args.color_sensitivity),
        "--start-index",
        str(start),
        "--count",
        str(count),
        "--worker",
        "--no-summary",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")

    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=stdout,
            stderr=stderr,
        )
        print(
            f"started {chunk_name} pid={process.pid} count={count}",
            flush=True,
        )
        code = process.wait()
        print(f"finished {chunk_name} exit={code}", flush=True)
        return code


def isolate_failed_chunk(
    args: argparse.Namespace,
    threshold: float,
    image_paths: list[Path],
    start: int,
    count: int,
) -> None:
    out_dir = args.out_root / threshold_id(threshold)
    records_dir = out_dir / "records"
    for index in range(start, start + count):
        image_path = image_paths[index]
        record_path = records_dir / f"{safe_id(image_path, args.input_dir)}.json"
        if record_path.exists():
            continue
        code = run_chunk(args, threshold, index, 1)
        if code != 0 and not record_path.exists():
            row = crash_record(image_path, args.input_dir, threshold, code)
            record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            print(
                f"marked crash {threshold_id(threshold)} index={index} "
                f"image={ascii(image_path.name)} exit={code}",
                flush=True,
            )


def chunk_complete(input_dir: Path, image_paths: list[Path], out_dir: Path) -> bool:
    records_dir = out_dir / "records"
    return all((records_dir / f"{safe_id(path, input_dir)}.json").exists() for path in image_paths)


def run_worker(args: argparse.Namespace) -> None:
    # Overlays are built post-hoc via .codex-tmp/make_overlays.py; there is no inline
    # overlay step here.
    threshold = float(args.threshold)
    out_dir = args.out_root / threshold_id(threshold)
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    # This is experiment-only lowering. The imported production defaults are not edited.
    consensus_mod.MODEL_CONFIDENCE_FLOOR = min(consensus_mod.MODEL_CONFIDENCE_FLOOR, threshold)
    
    image_paths = get_image_paths(args.input_dir, args.limit)
    if args.start_index:
        image_paths = image_paths[args.start_index:]
    if args.count is not None:
        image_paths = image_paths[: args.count]

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        image_id = safe_id(image_path, args.input_dir)
        record_path = records_dir / f"{image_id}.json"
        if args.resume and record_path.exists():
            row = json.loads(record_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] resume {ascii(image_path.name)}", flush=True)
            continue

        start = time.time()
        print(f"[{index}/{len(image_paths)}] threshold={threshold:.3f} detect {ascii(image_path.name)}", flush=True)
        row = run_one(image_path, args.input_dir, threshold, args.color_sensitivity)
        row["sweep_threshold"] = threshold
        row["detector_floor"] = consensus_mod.MODEL_CONFIDENCE_FLOOR
        row["elapsed_sec"] = time.time() - start
        record_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)

    if not args.no_summary:
        write_summary(out_dir, rows)
        summarize_root(args.out_root)


def write_threshold_summary(out_dir: Path) -> None:
    records_dir = out_dir / "records"
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(records_dir.glob("*.json"))
    ]
    write_summary(out_dir, rows)


def crash_record(image_path: Path, input_dir: Path, threshold: float, exit_code: int) -> dict:
    return {
        "image_id": safe_id(image_path, input_dir),
        "path": str(image_path),
        "relative_path": str(image_path.relative_to(input_dir)),
        "confidence": threshold,
        "sweep_threshold": threshold,
        "failover_type": "process_crash",
        "skipped": True,
        "bbox_count": 0,
        "bbox_records": [],
        "stages": [],
        "total_area_fraction": 0.0,
        "ambiguous": False,
        "error": f"worker exited {exit_code}",
        "review": empty_review(),
    }


def summarize_root(out_root: Path) -> None:
    rows = []
    for summary_path in sorted(out_root.glob("threshold_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = parse_threshold_id(summary_path.parent.name)
        pipeline_jsonl = summary_path.parent / "pipeline_bboxes.jsonl"
        detail_rows = []
        if pipeline_jsonl.exists():
            detail_rows = [
                json.loads(line)
                for line in pipeline_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rows.append(summary_record(threshold, summary, detail_rows))

    if not rows:
        return

    csv_path = out_root / "threshold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_root / "threshold_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def summary_record(threshold: float, summary: dict, rows: list[dict]) -> dict:
    bbox_rows = [row for row in rows if row.get("bbox_count", 0) > 0]
    unresolved_rows = [row for row in rows if row.get("failover_type") == "unresolved"]
    multi_bbox_rows = [row for row in rows if row.get("bbox_count", 0) >= 2]
    near_full_width_rows = [
        row
        for row in rows
        if any(box.get("width_fraction", 0.0) >= 0.5 for box in row.get("bbox_records", []))
    ]
    cross_third_rows = [
        row
        for row in rows
        if any(box.get("width_fraction", 0.0) >= 0.33 for box in row.get("bbox_records", []))
    ]
    big_growth_rows = [
        row
        for row in rows
        if any(box.get("growth_ratio", 0.0) >= 4.0 for box in row.get("bbox_records", []))
    ]
    high_coverage_rows = [row for row in rows if row.get("total_area_fraction", 0.0) >= 0.06]
    detector_counts = {"east": 0, "yolo11x": 0, "easyocr": 0}
    consensus_stage_hits = {"normal": 0, "rotation": 0, "gray_enhancement": 0, "white_enhancement": 0}
    for row in rows:
        for stage in row.get("stages", []):
            if stage.get("consensus_count", 0) > 0:
                name = stage.get("stage")
                if name in consensus_stage_hits:
                    consensus_stage_hits[name] += 1
            for name, boxes in stage.get("detectors", {}).items():
                if boxes:
                    detector_counts[name] = detector_counts.get(name, 0) + 1

    image_count = max(summary.get("image_count", len(rows)), 1)
    return {
        "threshold": threshold,
        "image_count": summary.get("image_count", len(rows)),
        "bbox_image_count": summary.get("bbox_image_count", len(bbox_rows)),
        "bbox_rate": len(bbox_rows) / image_count,
        "unresolved_count": len(unresolved_rows),
        "unresolved_rate": len(unresolved_rows) / image_count,
        "ambiguous_count": summary.get("ambiguous_count", 0),
        "ambiguous_rate": summary.get("ambiguous_count", 0) / image_count,
        "multi_bbox_count": len(multi_bbox_rows),
        "width_ge_033_count": len(cross_third_rows),
        "width_ge_050_count": len(near_full_width_rows),
        "growth_ge_4_count": len(big_growth_rows),
        "coverage_ge_006_count": len(high_coverage_rows),
        "max_total_area_fraction": summary.get("max_total_area_fraction", 0.0),
        "normal_consensus_hits": consensus_stage_hits["normal"],
        "rotation_consensus_hits": consensus_stage_hits["rotation"],
        "gray_consensus_hits": consensus_stage_hits["gray_enhancement"],
        "white_consensus_hits": consensus_stage_hits["white_enhancement"],
        "east_stage_hits": detector_counts.get("east", 0),
        "yolo11x_stage_hits": detector_counts.get("yolo11x", 0),
        "easyocr_stage_hits": detector_counts.get("easyocr", 0),
    }


def threshold_id(threshold: float) -> str:
    return f"threshold_{int(round(threshold * 1000)):03d}"


def parse_threshold_id(value: str) -> float:
    return int(value.rsplit("_", 1)[1]) / 1000.0


if __name__ == "__main__":
    main()
