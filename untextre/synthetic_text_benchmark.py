"""Synthetic text watermark generation for benchmark tests."""

from __future__ import annotations

import hashlib
import csv
import json
import colorsys
import random
from collections import Counter
from functools import lru_cache
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mask_experiments import MaskExperimentConfig, compute_mask_metrics
from .pipeline import PipelineResult, process_image_array
from .utils import load_image


MIN_TEXT_WIDTH_FRACTION = 0.25
MAX_TEXT_WIDTH_FRACTION = 0.50
TARGET_TEXT_WIDTH_FRACTION = 1.0 / 3.0
MIN_VISIBILITY_DELTA_E = 10.0  # EMPIRICAL: corpus p10 ~28, median ~77; trims only the bottom ~1-2%
MAX_VISIBILITY_ATTEMPTS = 8  # EMPIRICAL: median case passes on attempt 1; 8 retries covers pathological corners
EXCLUDED_BENCHMARK_ORIGINALS = {
    "kenna_james_avn_2016.jpg",
    "taylor_swift_2019.jpg",
}

_FONT_CANDIDATES = {
    "cursive": [
        "AlexBrush-Regular.ttf",
        "AlexBrush.ttf",
        "Allicia.ttf",
        "BRUSHSCI.TTF",
        "segoesc.ttf",
    ],
    "sans": [
        "corbel.ttf",
        "corbeli.ttf",
        "arial.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
    ],
    "serif": [
        "javatext.ttf",
        "Renda.ttf",
        "times.ttf",
        "georgia.ttf",
        "DejaVuSerif.ttf",
    ],
}

_COLOR_CLASSES = ("black", "white", "mid_gray", "light_gray", "random")
_CORNERS = ("upper_left", "upper_right", "lower_left", "lower_right")
_OUTLINE_TEXT_CLASSES = {"black", "white", "random"}
_TEXT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "tests" / "images" / "text_sources"
_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass(frozen=True)
class SyntheticTextCase:
    clean: np.ndarray
    watermarked: np.ndarray
    truth_mask: np.ndarray
    truth_bbox: tuple[int, int, int, int]
    metadata: dict


@dataclass(frozen=True)
class BenchmarkSampleSpec:
    sample_index: int
    base_index: int
    visit_round: int
    benchmark_seed: int
    sample_seed: int
    base_path: Path
    base_relpath: str


@dataclass(frozen=True)
class TextSourcePools:
    url_prefixes: tuple[str, ...]
    url_nouns: tuple[str, ...]
    url_tlds: tuple[str, ...]
    copyright_first_names: tuple[str, ...]
    copyright_last_names: tuple[str, ...]


def parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated positive-integer list."""
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected comma-separated positive integers, got: {value!r}")
    return values


def iter_original_images(root: Path) -> list[Path]:
    """Return clean original fixture images used as benchmark backgrounds."""
    originals = root / "originals"
    return sorted(
        path for path in originals.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.name not in EXCLUDED_BENCHMARK_ORIGINALS
    )


def iter_base_images(root: Path, *, recursive: bool = False) -> list[Path]:
    """Return image files from a clean benchmark corpus root."""
    root = Path(root)
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path for path in candidates
        if path.is_file()
        and path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
    )


def build_even_visit_plan(
    base_paths: Sequence[Path],
    sample_count: int,
    seed: int,
    *,
    sample_start: int = 0,
) -> list[BenchmarkSampleSpec]:
    """Build a deterministic corpus schedule that visits each base evenly."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    ordered = sorted(Path(path) for path in base_paths)
    if not ordered:
        return []

    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)

    plan: list[BenchmarkSampleSpec] = []
    for sample_index in range(sample_start, sample_start + sample_count):
        base_index = sample_index % len(shuffled)
        visit_round = sample_index // len(shuffled)
        base_path = shuffled[base_index]
        base_relpath = str(base_path)
        sample_seed = _derive_sample_seed(seed, sample_index, visit_round, base_path)
        plan.append(
            BenchmarkSampleSpec(
                sample_index=sample_index,
                base_index=base_index,
                visit_round=visit_round,
                benchmark_seed=seed,
                sample_seed=sample_seed,
                base_path=base_path,
                base_relpath=base_relpath,
            )
        )
    return plan


def run_in_memory_watermark_benchmark(
    base_dir: Path,
    sample_count: int,
    seed: int,
    *,
    method: str = "telea",
    mask_config: MaskExperimentConfig | dict | None = None,
    coverage_limit: float = 0.0,
    target_dilation_px: int = 2,
    recursive: bool = False,
    sample_start: int = 0,
    preload_models: bool = False,
    progress_csv: Path | None = None,
    font_dirs: Sequence[Path] | None = None,
    image_loader: Callable[[Path], np.ndarray] = load_image,
    case_builder: Callable[..., SyntheticTextCase] | None = None,
    process_image_fn: Callable[..., PipelineResult] = process_image_array,
) -> list[dict]:
    """Run a reproducible synthetic watermark benchmark fully in memory."""
    base_dir = Path(base_dir)
    base_paths = iter_base_images(base_dir, recursive=recursive)
    plan = build_even_visit_plan(base_paths, sample_count, seed, sample_start=sample_start)
    config_payload = _normalize_mask_config(mask_config)
    case_builder = case_builder or generate_synthetic_text_case
    progress_csv = Path(progress_csv) if progress_csv is not None else None

    if preload_models:
        from .pipeline import initialize_consensus_models

        initialize_consensus_models(device="cuda")

    rows: list[dict] = []
    for spec in plan:
        sample_started = datetime.now(timezone.utc)
        clean = image_loader(spec.base_path)
        rng = random.Random(spec.sample_seed)
        synthetic = case_builder(clean, rng, font_dirs=font_dirs)
        result = process_image_fn(
            synthetic.watermarked,
            image_name=f"{spec.base_path.stem}__{spec.sample_index:06d}",
            method=method,
            forced_bbox=synthetic.truth_bbox,
            auto_retry=False,
            expand_bboxes=False,
            use_grabcut=False,
            use_grabcut_expand=False,
            use_budgeted_expand=True,
            coverage_limit=coverage_limit,
            mask_config=config_payload,
        )

        predicted_mask = result.mask
        if predicted_mask.shape != synthetic.truth_mask.shape:
            predicted_mask = cv2.resize(
                predicted_mask,
                (synthetic.truth_mask.shape[1], synthetic.truth_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        metrics = compute_mask_metrics(
            predicted_mask,
            synthetic.truth_mask,
            bbox=synthetic.truth_bbox,
            target_dilation_px=target_dilation_px,
        )
        timings = getattr(result, "timings", {})
        consensus_boxes = getattr(result, "consensus_boxes", [])
        rows.append(
            {
                "benchmark_seed": seed,
                "sample_index": spec.sample_index,
                "visit_round": spec.visit_round,
                "base_index": spec.base_index,
                "base_path": str(spec.base_path),
                "base_relpath": _display_relpath(base_dir, spec.base_path),
                "base_name": spec.base_path.name,
                "base_width": int(clean.shape[1]),
                "base_height": int(clean.shape[0]),
                "sample_seed": spec.sample_seed,
                "method": method,
                "mask_config": config_payload,
                "target_dilation_px": target_dilation_px,
                "truth_bbox": list(synthetic.truth_bbox),
                "truth_bbox_coverage": synthetic.metadata.get("truth_bbox_coverage"),
                "truth_alpha_coverage": synthetic.metadata.get("truth_alpha_coverage"),
                "pipeline_consensus_box_count": len(consensus_boxes),
                "pipeline_mask_px": int(np.sum(predicted_mask > 0)),
                "pipeline_mask_coverage": float(np.mean(predicted_mask > 0)),
                "pipeline_total_time": timings.get("total_time") if isinstance(timings, dict) else None,
                **synthetic.metadata,
                **metrics,
            }
        )
        sample_finished = datetime.now(timezone.utc)
        row = rows[-1]
        row["started_at"] = sample_started.isoformat()
        row["finished_at"] = sample_finished.isoformat()
        row["elapsed_seconds"] = (sample_finished - sample_started).total_seconds()
        if progress_csv is not None:
            _append_progress_row(progress_csv, row)
    return rows


def summarize_in_memory_watermark_benchmark(rows: list[dict], top_n: int = 20) -> dict:
    """Summarize benchmark rows for later review."""
    rows = list(rows)
    if not rows:
        return {
            "sample_count": 0,
            "base_count": 0,
            "mean_target_recall": 0.0,
            "min_target_recall": 0.0,
            "mean_weighted_precision": 0.0,
            "mean_overmask_ratio": 0.0,
            "max_coverage": 0.0,
            "base_visit_counts": {},
            "top_cases": [],
            "review_cases": [],
        }

    base_visit_counts = Counter(row["base_relpath"] for row in rows)
    ranked_desc = _rank_benchmark_rows(rows, descending=True, top_n=top_n)
    ranked_asc = _rank_benchmark_rows(rows, descending=False, top_n=top_n)
    return {
        "sample_count": len(rows),
        "base_count": len(base_visit_counts),
        "benchmark_seed": rows[0].get("benchmark_seed"),
        "mean_target_recall": _mean(row["target_recall"] for row in rows),
        "min_target_recall": min(row["target_recall"] for row in rows),
        "mean_weighted_precision": _mean(row["weighted_precision"] for row in rows),
        "mean_overmask_ratio": _mean(row["overmask_ratio"] for row in rows),
        "max_coverage": max(row["coverage"] for row in rows),
        "base_visit_counts": dict(sorted(base_visit_counts.items())),
        "top_cases": ranked_desc,
        "review_cases": ranked_asc,
    }


def write_in_memory_watermark_benchmark_jsonl(rows: list[dict], path: Path) -> None:
    """Write benchmark rows as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_in_memory_watermark_benchmark_summary(
    rows: list[dict],
    path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Write a JSON summary with ranked review slices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summarize_in_memory_watermark_benchmark(rows, top_n=top_n), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _measure_visibility_delta_e(
    clean_bgr: np.ndarray,
    watermarked_bgr: np.ndarray,
    truth_mask: np.ndarray,
    truth_bbox: tuple[int, int, int, int],
) -> float:
    """Median Lab dE of watermarked vs. clean pixels inside the truth mask."""
    x, y, bbox_w, bbox_h = truth_bbox
    clean_crop = clean_bgr[y:y + bbox_h, x:x + bbox_w]
    watermarked_crop = watermarked_bgr[y:y + bbox_h, x:x + bbox_w]
    mask_crop = truth_mask[y:y + bbox_h, x:x + bbox_w]

    if not np.any(mask_crop > 0):
        return 0.0

    clean_lab = cv2.cvtColor(clean_crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    watermarked_lab = cv2.cvtColor(watermarked_crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    diff = clean_lab - watermarked_lab
    delta_e = np.linalg.norm(diff, axis=2)
    return float(np.median(delta_e[mask_crop > 0]))


def generate_synthetic_text_case(
    clean_bgr: np.ndarray,
    rng: random.Random,
    *,
    font_dirs: Sequence[Path] | None = None,
) -> SyntheticTextCase:
    """Overlay randomized text, re-rolling until the overlay is measurably visible.

    Attempt 0 consumes ``rng`` directly so the random stream stays bit-identical
    to the pre-veto generator (seed-pinned tests and saved corpora depend on
    this). Later attempts derive a fresh RNG from ``rng`` only after a failed
    attempt, preserving determinism for a given seed.
    """
    best_case: SyntheticTextCase | None = None
    best_de = -1.0
    attempt_rng = rng
    for attempt in range(1, MAX_VISIBILITY_ATTEMPTS + 1):
        case = _generate_synthetic_text_case_attempt(clean_bgr, attempt_rng, font_dirs=font_dirs)
        de = _measure_visibility_delta_e(case.clean, case.watermarked, case.truth_mask, case.truth_bbox)
        case.metadata["measured_visibility_delta_e"] = de
        case.metadata["visibility_attempts"] = attempt
        case.metadata["visibility_fallback"] = False

        if de > best_de:
            best_de = de
            best_case = case

        if de >= MIN_VISIBILITY_DELTA_E:
            return case

        attempt_rng = random.Random(rng.getrandbits(64))

    assert best_case is not None
    best_case.metadata["visibility_fallback"] = True
    best_case.metadata["visibility_attempts"] = MAX_VISIBILITY_ATTEMPTS
    return best_case


def _generate_synthetic_text_case_attempt(
    clean_bgr: np.ndarray,
    rng: random.Random,
    *,
    font_dirs: Sequence[Path] | None = None,
) -> SyntheticTextCase:
    """Overlay randomized text sized like a corner watermark."""
    clean = clean_bgr.copy()
    h, w = clean.shape[:2]
    font_family, font_path = _choose_font(rng, font_dirs)
    text_mode, text = _choose_text(rng)
    color_class, rgb, corner, corner_stats, choice_reason = _choose_fill_and_corner(clean, rng)
    opacity = rng.uniform(0.50, 1.00)
    outline_present, outline_thickness_px, outline_rgb = _choose_outline(color_class, rng)
    outline_hex = _rgb_to_hex(outline_rgb) if outline_rgb is not None else None

    fitted = _fit_text_mask(
        (w, h),
        text,
        font_path,
        rng,
        stroke_width=outline_thickness_px if outline_present else 0,
        min_width_fraction=MIN_TEXT_WIDTH_FRACTION,
        max_width_fraction=MAX_TEXT_WIDTH_FRACTION,
        target_width_fraction=TARGET_TEXT_WIDTH_FRACTION,
    )
    font_size, text_mask, local_bbox, raw_bbox = fitted
    x, y = _place_bbox((w, h), (local_bbox[2], local_bbox[3]), corner, rng)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y:y + local_bbox[3], x:x + local_bbox[2]] = text_mask[
        local_bbox[1]:local_bbox[1] + local_bbox[3],
        local_bbox[0]:local_bbox[0] + local_bbox[2],
    ]

    watermarked = _draw_text_overlay(
        clean,
        text,
        font_path,
        font_size,
        x=x - raw_bbox[0],
        y=y - raw_bbox[1],
        fill_rgb=rgb,
        opacity=opacity,
        stroke_width=outline_thickness_px if outline_present else 0,
        stroke_rgb=outline_rgb,
    )
    text_only_mask, _ = _render_text_mask(
        text,
        _load_font(str(font_path), font_size),
        stroke_width=0,
    )
    text_only_coords = cv2.findNonZero((text_only_mask > 0).astype(np.uint8))
    if text_only_coords is None:
        text_only_bbox = (x, y, local_bbox[2], local_bbox[3])
    else:
        text_only_x, text_only_y, text_only_w, text_only_h = cv2.boundingRect(text_only_coords)
        text_only_bbox = (x, y, text_only_w, text_only_h)

    truth_bbox = (x, y, local_bbox[2], local_bbox[3])
    return SyntheticTextCase(
        clean=clean,
        watermarked=watermarked,
        truth_mask=full_mask,
        truth_bbox=truth_bbox,
        metadata={
            "text": text,
            "text_mode": text_mode,
            "font_family": font_family,
            "font_path": str(font_path),
            "font_size": font_size,
            "text_bbox": list(text_only_bbox),
            "color_class": color_class,
            "color_rgb": rgb,
            "opacity": opacity,
            "corner": corner,
            "corner_luminance_class": corner_stats["class"],
            "corner_luminance": corner_stats["luminance"],
            "corner_saturation": corner_stats["saturation"],
            "corner_mean_rgb": corner_stats["mean_rgb"],
            "corner_choice_reason": choice_reason,
            "rejected_pairs_count": corner_stats["rejected_pairs_count"],
            "outline_present": outline_present,
            "outline_thickness_px": outline_thickness_px,
            "outline_color_hex": outline_hex,
            "truth_alpha_coverage": float(np.mean(full_mask > 0)),
            "truth_bbox_coverage": (truth_bbox[2] * truth_bbox[3]) / float(w * h),
        },
    )


def replay_synthetic_text_case(
    clean_bgr: np.ndarray,
    case: dict,
) -> SyntheticTextCase:
    """Rebuild a saved synthetic-text case from manifest fields.

    This does not re-run the RNG-driven generator. It replays the explicit case
    metadata that was persisted with a saved corpus, which makes post-hoc
    troubleshooting independent of future generator changes.
    """
    clean = clean_bgr.copy()
    h, w = clean.shape[:2]
    text = str(case["text"])
    font_path = Path(case["font_path"])
    font_size = int(case["font_size"])
    fill_r, fill_g, fill_b = (int(channel) for channel in case["color_rgb"])
    fill_rgb: tuple[int, int, int] = (fill_r, fill_g, fill_b)
    opacity = float(case["opacity"])
    bbox_x, bbox_y, bbox_w, bbox_h = (int(value) for value in case["truth_bbox"])
    truth_bbox: tuple[int, int, int, int] = (bbox_x, bbox_y, bbox_w, bbox_h)

    outline_present = bool(case.get("outline_present", False))
    outline_thickness_px = int(case.get("outline_thickness_px", 0)) if outline_present else 0
    outline_rgb = _hex_to_rgb(case["outline_color_hex"]) if case.get("outline_color_hex") else None

    font = _load_font(str(font_path), font_size)
    _, raw_bbox = _render_text_mask(
        text,
        font,
        stroke_width=outline_thickness_px,
    )

    watermarked = _draw_text_overlay(
        clean,
        text,
        font_path,
        font_size,
        x=truth_bbox[0] - raw_bbox[0],
        y=truth_bbox[1] - raw_bbox[1],
        fill_rgb=fill_rgb,
        opacity=opacity,
        stroke_width=outline_thickness_px,
        stroke_rgb=outline_rgb,
    )

    full_mask, _ = _render_text_mask(
        text,
        font,
        stroke_width=outline_thickness_px,
    )
    coords = cv2.findNonZero((full_mask > 0).astype(np.uint8))
    truth_mask = np.zeros((h, w), dtype=np.uint8)
    if coords is not None:
        local_x, local_y, local_w, local_h = cv2.boundingRect(coords)
        truth_mask[
            truth_bbox[1]:truth_bbox[1] + local_h,
            truth_bbox[0]:truth_bbox[0] + local_w,
        ] = full_mask[
            local_y:local_y + local_h,
            local_x:local_x + local_w,
        ]

    metadata = dict(case)
    metadata["color_rgb"] = list(fill_rgb)
    metadata["truth_bbox"] = list(truth_bbox)
    metadata.setdefault("text_bbox", list(truth_bbox))
    metadata["measured_visibility_delta_e"] = _measure_visibility_delta_e(
        clean, watermarked, truth_mask, truth_bbox
    )

    return SyntheticTextCase(
        clean=clean,
        watermarked=watermarked,
        truth_mask=truth_mask,
        truth_bbox=truth_bbox,
        metadata=metadata,
    )


def _choose_text(rng: random.Random) -> tuple[str, str]:
    sources = _load_text_sources()
    if rng.random() < 0.25:
        first_name = rng.choice(sources.copyright_first_names)
        last_name = rng.choice(sources.copyright_last_names)
        return "copyright_name", f"© {first_name} {last_name}"

    prefix = rng.choice(sources.url_prefixes)
    noun = rng.choice(sources.url_nouns)
    tld = rng.choice(sources.url_tlds)
    return "url", f"{prefix}{noun}{tld}"


def _choose_font(
    rng: random.Random,
    font_dirs: Sequence[Path] | None,
) -> tuple[str, Path]:
    candidates: list[tuple[str, Path]] = []
    dirs = list(font_dirs or _default_font_dirs())
    for family, names in _FONT_CANDIDATES.items():
        for directory in dirs:
            for name in names:
                path = directory / name
                if path.exists():
                    candidates.append((family, path))
    if not candidates:
        raise RuntimeError("No scalable fonts found for generated text benchmark")
    return rng.choice(candidates)


def _default_font_dirs() -> list[Path]:
    dirs = [Path("C:/Windows/Fonts")]
    local_fonts = Path(__file__).resolve().parent / "fonts"
    if local_fonts.exists():
        dirs.append(local_fonts)
    return dirs


def _make_text(rng: random.Random) -> str:
    return _choose_text(rng)[1]


@lru_cache(maxsize=None)
def _load_text_sources(text_source_dir: Path | None = None) -> TextSourcePools:
    source_dir = Path(text_source_dir) if text_source_dir is not None else _TEXT_SOURCE_DIR
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing text source directory: {source_dir}")

    return TextSourcePools(
        url_prefixes=_load_token_file(source_dir / "url_prefixes.txt"),
        url_nouns=_load_token_file(source_dir / "url_nouns.txt"),
        url_tlds=_load_token_file(source_dir / "url_tlds.txt"),
        copyright_first_names=_load_token_file(source_dir / "copyright_first_names.txt"),
        copyright_last_names=_load_token_file(source_dir / "copyright_last_names.txt"),
    )


def _load_token_file(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Missing text source file: {path}")

    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        tokens.append(token)
    if not tokens:
        raise ValueError(f"Text source file is empty: {path}")
    return tuple(tokens)


def _choose_color(rng: random.Random) -> tuple[str, tuple[int, int, int]]:
    color_class = rng.choice(_COLOR_CLASSES)
    if color_class == "black":
        return color_class, (0, 0, 0)
    if color_class == "white":
        return color_class, (255, 255, 255)
    if color_class == "mid_gray":
        return color_class, (128, 128, 128)
    if color_class == "light_gray":
        return color_class, (210, 210, 210)
    return color_class, _sample_vivid_fill_rgb(rng)


def _choose_fill_and_corner(
    clean_bgr: np.ndarray,
    rng: random.Random,
) -> tuple[str, tuple[int, int, int], str, dict, str]:
    """Choose a fill color and corner jointly from valid contrast pairs."""
    corner_stats = _summarize_corners(clean_bgr)
    fill_candidates = [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("mid_gray", (128, 128, 128)),
        ("light_gray", (210, 210, 210)),
        ("random", _sample_vivid_fill_rgb(rng)),
    ]
    valid_pairs: list[tuple[str, tuple[int, int, int], str]] = []
    rejected_pairs = 0
    for fill_class, rgb in fill_candidates:
        for corner_name in _CORNERS:
            if _is_valid_fill_corner_pair(fill_class, corner_stats[corner_name]["class"]):
                valid_pairs.append((fill_class, rgb, corner_name))
            else:
                rejected_pairs += 1

    if not valid_pairs:
        raise RuntimeError("No valid fill/corner pairs available for synthetic watermark generation")

    fill_class, rgb, corner = rng.choice(valid_pairs)
    choice_reason = f"{fill_class}->{corner}"
    corner_stats[corner]["rejected_pairs_count"] = rejected_pairs
    return fill_class, rgb, corner, corner_stats[corner], choice_reason


def _sample_vivid_fill_rgb(rng: random.Random) -> tuple[int, int, int]:
    hue = rng.uniform(0.0, 1.0)
    saturation = rng.uniform(0.80, 1.00)
    value = rng.uniform(0.45, 0.95)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        max(0, min(255, int(round(red * 255.0)))),
        max(0, min(255, int(round(green * 255.0)))),
        max(0, min(255, int(round(blue * 255.0)))),
    )


def _is_valid_fill_corner_pair(fill_class: str, corner_class: str) -> bool:
    if fill_class == "black":
        return corner_class != "black"
    if fill_class == "white":
        return corner_class != "white"
    if fill_class in {"mid_gray", "light_gray"}:
        invalid_corner_classes = {"gray"}
        if fill_class == "light_gray":
            invalid_corner_classes.add("white")
        return corner_class not in invalid_corner_classes
    return corner_class != "vivid"


def _choose_outline(
    color_class: str,
    rng: random.Random,
) -> tuple[bool, int, tuple[int, int, int] | None]:
    if color_class not in _OUTLINE_TEXT_CLASSES:
        return False, 0, None
    if rng.random() >= 0.20:
        return False, 0, None

    thickness_px = rng.randint(2, 6)
    if color_class == "black":
        return True, thickness_px, (255, 255, 255) if rng.random() < 0.5 else _sample_vivid_outline_rgb(rng)
    if color_class == "white":
        return True, thickness_px, (0, 0, 0) if rng.random() < 0.5 else _sample_vivid_outline_rgb(rng)
    return True, thickness_px, (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)


def _sample_vivid_outline_rgb(rng: random.Random) -> tuple[int, int, int]:
    hue = rng.uniform(0.0, 1.0)
    saturation = rng.uniform(0.20, 1.00)
    value = rng.uniform(0.45, 0.95)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        max(0, min(255, int(round(red * 255.0)))),
        max(0, min(255, int(round(green * 255.0)))),
        max(0, min(255, int(round(blue * 255.0)))),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return (r, g, b)


def _summarize_corners(clean_bgr: np.ndarray) -> dict[str, dict]:
    """Summarize each corner of the image from a tiny downsampled version."""
    h, w = clean_bgr.shape[:2]
    short_side = min(h, w)
    if short_side <= 0:
        raise ValueError("clean image must have positive dimensions")

    scale = 7.0 / float(short_side)
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    downsampled = cv2.resize(clean_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    patch_h = max(1, min(2, scaled_h))
    patch_w = max(1, min(2, scaled_w))
    corners = {
        "upper_left": downsampled[:patch_h, :patch_w],
        "upper_right": downsampled[:patch_h, scaled_w - patch_w:scaled_w],
        "lower_left": downsampled[scaled_h - patch_h:scaled_h, :patch_w],
        "lower_right": downsampled[scaled_h - patch_h:scaled_h, scaled_w - patch_w:scaled_w],
    }
    summaries: dict[str, dict] = {}
    for corner_name, patch in corners.items():
        summaries[corner_name] = _corner_summary(patch)
    return summaries


def _corner_summary(patch_bgr: np.ndarray) -> dict:
    mean_bgr = patch_bgr.reshape(-1, 3).mean(axis=0)
    mean_rgb = tuple(int(round(channel)) for channel in mean_bgr[::-1])
    luminance = _relative_luminance_bgr(mean_bgr)
    saturation = _mean_hsv_saturation_bgr(mean_bgr)
    return {
        "class": _classify_corner(mean_bgr),
        "luminance": luminance,
        "saturation": saturation,
        "mean_rgb": mean_rgb,
    }


def _classify_corner(mean_bgr: np.ndarray) -> str:
    """Classify a corner coarsely for contrast filtering.

    EMPIRICAL: the luminance/saturation banding below is intentionally coarse.
    It only needs to separate obviously dark, light, gray, and vivid corners.
    """
    luminance = _relative_luminance_bgr(mean_bgr)
    saturation = _mean_hsv_saturation_bgr(mean_bgr)
    if saturation >= 0.20:
        return "vivid"
    if luminance < 0.33:
        return "black"
    if luminance > 0.67:
        return "white"
    return "gray"


def _relative_luminance_bgr(mean_bgr: np.ndarray) -> float:
    b, g, r = [float(channel) / 255.0 for channel in mean_bgr]

    def _linearize(component: float) -> float:
        return component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4

    r_lin = _linearize(r)
    g_lin = _linearize(g)
    b_lin = _linearize(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _mean_hsv_saturation_bgr(mean_bgr: np.ndarray) -> float:
    pixel = np.array([[mean_bgr]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[1]) / 255.0


@lru_cache(maxsize=256)
def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Cache FreeType handles; font loads dominate fitting at corpus scale."""
    return ImageFont.truetype(font_path, font_size)


def _metric_text_width(font_path: str, font_size: int, text: str, stroke_width: int) -> int:
    """Layout width from font metrics only -- no rasterization."""
    bbox = _load_font(font_path, font_size).getbbox(text, mode="L", stroke_width=stroke_width)
    return bbox[2] - bbox[0]


def _fit_text_mask(
    image_size: tuple[int, int],
    text: str,
    font_path: Path,
    rng: random.Random,
    *,
    stroke_width: int,
    min_width_fraction: float,
    max_width_fraction: float,
    target_width_fraction: float,
) -> tuple[int, np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Pick the font size whose rendered ink width lands nearest the target fraction.

    Selection matches the previous exhaustive scan (argmin of
    |ink_width_fraction - target| over the step-2 size grid, ties to the
    smaller size), but the crossing size is located by bisecting cheap font
    metrics and only a handful of candidate sizes are rasterized. Ink width
    grows by many pixels per 2 pt step, so it is monotone at grid resolution.
    """
    image_w, image_h = image_size
    min_size = max(12, min(image_w, image_h) // 30)
    max_size = max(min_size + 1, min(image_w, image_h) // 2)
    sizes = range(min_size, max_size + 1, 2)
    last = len(sizes) - 1
    target_width = rng.uniform(target_width_fraction * 0.95, target_width_fraction * 1.05)
    target_px = target_width * image_w
    font_key = str(font_path)

    rendered: dict[int, tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int], int] | None] = {}

    def ink(font_size: int):
        """Rasterize once per size: (mask, ink bbox, layout bbox, ink width)."""
        if font_size not in rendered:
            mask, raw_bbox = _render_text_mask(text, _load_font(font_key, font_size), stroke_width=stroke_width)
            coords = cv2.findNonZero((mask > 0).astype(np.uint8))
            if coords is None:
                rendered[font_size] = None
            else:
                x, y, box_w, box_h = cv2.boundingRect(coords)
                rendered[font_size] = (mask, (x, y, box_w, box_h), raw_bbox, box_w)
        return rendered[font_size]

    def metric(index: int) -> int:
        return _metric_text_width(font_key, sizes[index], text, stroke_width)

    # Bisect font metrics for the first grid size at or above the target width.
    if metric(0) >= target_px:
        index = 0
    elif metric(last) < target_px:
        index = last
    else:
        lo, hi = 0, last
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if metric(mid) >= target_px:
                hi = mid
            else:
                lo = mid
        index = hi

    # Metric width overshoots ink width by the trailing side bearing (<~2%),
    # so the ink crossing sits at or slightly above the metric crossing.
    while index < last:
        info = ink(sizes[index])
        if info is not None and info[3] >= target_px:
            break
        index += 1

    candidates: list[tuple[float, int]] = []
    for grid_index in (index - 1, index):
        if grid_index < 0:
            continue
        info = ink(sizes[grid_index])
        if info is None:
            continue
        width_fraction = info[3] / float(image_w)
        if min_width_fraction <= width_fraction <= max_width_fraction:
            candidates.append((abs(width_fraction - target_width), sizes[grid_index]))

    if not candidates:
        raise RuntimeError(
            f"Could not fit text {text!r} into width fraction range "
            f"{min_width_fraction:.3f}-{max_width_fraction:.3f}"
        )

    candidates.sort()
    font_size = candidates[0][1]
    # font_size came from `candidates`, built only from sizes where ink()
    # returned non-None (line 896); ink() memoizes by size, so this call
    # returns that same cached, already-validated result.
    info = ink(font_size)
    assert info is not None
    mask, local_bbox, raw_bbox, _ink_width = info
    return font_size, mask, local_bbox, raw_bbox


def _render_text_mask(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    stroke_width: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Render text ink onto a text-sized canvas; returns (mask, layout bbox).

    The canvas is sized from font metrics (plus slack for hinting spill)
    instead of the full image; callers crop by the measured ink bbox, so
    geometry is unchanged relative to the previous full-image canvas.
    """
    bbox = font.getbbox(text, mode="L", stroke_width=stroke_width)
    canvas_w = max(1, bbox[2] - bbox[0] + 2)
    canvas_h = max(1, bbox[3] - bbox[1] + 2)
    mask_image = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.text(
        (-bbox[0], -bbox[1]),
        text,
        font=font,
        fill=255,
        stroke_width=stroke_width,
        stroke_fill=255,
    )
    return np.array(mask_image, dtype=np.uint8), bbox


def _place_bbox(
    image_size: tuple[int, int],
    bbox_size: tuple[int, int],
    corner: str,
    rng: random.Random,
) -> tuple[int, int]:
    image_w, image_h = image_size
    bbox_w, bbox_h = bbox_size
    margin_x = max(4, image_w // 100)
    margin_y = max(4, image_h // 100)
    jitter_x = rng.randint(0, max(1, image_w // 50))
    jitter_y = rng.randint(0, max(1, image_h // 50))

    if "left" in corner:
        x = margin_x + jitter_x
    else:
        x = image_w - bbox_w - margin_x - jitter_x
    if "upper" in corner:
        y = margin_y + jitter_y
    else:
        y = image_h - bbox_h - margin_y - jitter_y
    return (
        max(0, min(x, image_w - bbox_w)),
        max(0, min(y, image_h - bbox_h)),
    )


def _composite_text(
    clean_bgr: np.ndarray,
    mask: np.ndarray,
    rgb: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    overlay_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
    alpha = (mask.astype(np.float32) / 255.0) * float(opacity)
    out = clean_bgr.astype(np.float32)
    out = out * (1.0 - alpha[:, :, None]) + overlay_bgr * alpha[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_text_overlay(
    clean_bgr: np.ndarray,
    text: str,
    font_path: Path,
    font_size: int,
    *,
    x: int,
    y: int,
    fill_rgb: tuple[int, int, int],
    opacity: float,
    stroke_width: int = 0,
    stroke_rgb: tuple[int, int, int] | None = None,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(str(font_path), font_size)
    if stroke_width > 0 and stroke_rgb is not None:
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill_rgb,
            stroke_width=stroke_width,
            stroke_fill=stroke_rgb,
        )
    else:
        draw.text((x, y), text, font=font, fill=fill_rgb)
    rendered = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if opacity >= 1.0:
        return rendered
    alpha = float(np.clip(opacity, 0.0, 1.0))
    return np.clip(
        clean_bgr.astype(np.float32) * (1.0 - alpha) + rendered.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _derive_sample_seed(
    benchmark_seed: int,
    sample_index: int,
    visit_round: int,
    base_path: Path,
) -> int:
    payload = f"{benchmark_seed}:{sample_index}:{visit_round}:{base_path.as_posix()}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _normalize_mask_config(mask_config: MaskExperimentConfig | dict | None) -> dict:
    if mask_config is None:
        return MaskExperimentConfig().to_dict()
    if isinstance(mask_config, MaskExperimentConfig):
        return mask_config.to_dict()
    return dict(mask_config)


def _display_relpath(base_dir: Path, base_path: Path) -> str:
    try:
        return str(base_path.relative_to(base_dir))
    except ValueError:
        return str(base_path)


def _rank_benchmark_rows(rows: list[dict], *, descending: bool, top_n: int) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["score"] if descending else row["score"],
            row.get("sample_index", 0),
            row.get("base_relpath", ""),
        ),
    )
    return ranked[:top_n]


def _append_progress_row(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "base_relpath",
        "sample_seed",
        "finished_at",
        "elapsed_seconds",
        "target_recall",
        "weighted_precision",
        "coverage",
        "overmask_ratio",
        "score",
        "pipeline_consensus_box_count",
        "pipeline_total_time",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "sample_index": row.get("sample_index"),
                "base_relpath": row.get("base_relpath"),
                "sample_seed": row.get("sample_seed"),
                "finished_at": row.get("finished_at"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "target_recall": row.get("target_recall"),
                "weighted_precision": row.get("weighted_precision"),
                "coverage": row.get("coverage"),
                "overmask_ratio": row.get("overmask_ratio"),
                "score": row.get("score"),
                "pipeline_consensus_box_count": row.get("pipeline_consensus_box_count"),
                "pipeline_total_time": row.get("pipeline_total_time"),
            }
        )


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
