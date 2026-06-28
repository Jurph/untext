"""Synthetic text watermark generation for benchmark tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MIN_TEXT_WIDTH_FRACTION = 0.25
MAX_TEXT_WIDTH_FRACTION = 0.50
TARGET_TEXT_WIDTH_FRACTION = 1.0 / 3.0
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
    ],
    "serif": [
        "javatext.ttf",
        "Renda.ttf",
        "times.ttf",
        "georgia.ttf",
    ],
}

_PREFIXES = ("Image", "Picture", "Ogle", "Look")
_NOUNS = ("Fans", "Posts", "Blog", "Site")
_TLDS = ("", ".com", ".net", ".org", ".site", ".blog")
_COLOR_CLASSES = ("black", "white", "mid_gray", "light_gray", "random")
_CORNERS = ("upper_left", "upper_right", "lower_left", "lower_right")


@dataclass(frozen=True)
class SyntheticTextCase:
    clean: np.ndarray
    watermarked: np.ndarray
    truth_mask: np.ndarray
    truth_bbox: tuple[int, int, int, int]
    metadata: dict


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


def generate_synthetic_text_case(
    clean_bgr: np.ndarray,
    rng: random.Random,
    *,
    font_dirs: Sequence[Path] | None = None,
) -> SyntheticTextCase:
    """Overlay randomized text sized like a corner watermark."""
    clean = clean_bgr.copy()
    h, w = clean.shape[:2]
    font_family, font_path = _choose_font(rng, font_dirs)
    text = _make_text(rng)
    color_class, rgb = _choose_color(rng)
    opacity = rng.uniform(0.50, 1.00)
    corner = rng.choice(_CORNERS)

    fitted = _fit_text_mask(
        (w, h),
        text,
        font_path,
        rng,
        min_width_fraction=MIN_TEXT_WIDTH_FRACTION,
        max_width_fraction=MAX_TEXT_WIDTH_FRACTION,
        target_width_fraction=TARGET_TEXT_WIDTH_FRACTION,
    )
    font_size, text_mask, local_bbox = fitted
    x, y = _place_bbox((w, h), (local_bbox[2], local_bbox[3]), corner, rng)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y:y + local_bbox[3], x:x + local_bbox[2]] = text_mask[
        local_bbox[1]:local_bbox[1] + local_bbox[3],
        local_bbox[0]:local_bbox[0] + local_bbox[2],
    ]

    watermarked = _composite_text(clean, full_mask, rgb, opacity)
    truth_bbox = (x, y, local_bbox[2], local_bbox[3])
    return SyntheticTextCase(
        clean=clean,
        watermarked=watermarked,
        truth_mask=full_mask,
        truth_bbox=truth_bbox,
        metadata={
            "text": text,
            "font_family": font_family,
            "font_path": str(font_path),
            "font_size": font_size,
            "color_class": color_class,
            "color_rgb": rgb,
            "opacity": opacity,
            "corner": corner,
            "truth_alpha_coverage": float(np.mean(full_mask > 0)),
            "truth_bbox_coverage": (truth_bbox[2] * truth_bbox[3]) / float(w * h),
        },
    )


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
    return f"{rng.choice(_PREFIXES)}{rng.choice(_NOUNS)}{rng.choice(_TLDS)}"


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
    return color_class, tuple(rng.randint(0, 255) for _ in range(3))


def _fit_text_mask(
    image_size: tuple[int, int],
    text: str,
    font_path: Path,
    rng: random.Random,
    *,
    min_width_fraction: float,
    max_width_fraction: float,
    target_width_fraction: float,
) -> tuple[int, np.ndarray, tuple[int, int, int, int]]:
    image_w, image_h = image_size
    min_size = max(12, min(image_w, image_h) // 30)
    max_size = max(min_size + 1, min(image_w, image_h) // 2)
    viable: list[tuple[float, int, np.ndarray, tuple[int, int, int, int]]] = []

    for font_size in range(min_size, max_size + 1, 2):
        font = ImageFont.truetype(str(font_path), font_size)
        mask_image = Image.new("L", (image_w, image_h), 0)
        draw = ImageDraw.Draw(mask_image)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        if text_w >= image_w or text_h >= image_h:
            continue
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=255)
        mask = np.array(mask_image, dtype=np.uint8)
        coords = cv2.findNonZero((mask > 0).astype(np.uint8))
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        width_fraction = w / float(image_w)
        if min_width_fraction <= width_fraction <= max_width_fraction:
            viable.append((width_fraction, font_size, mask, (x, y, w, h)))

    if viable:
        target_width = rng.uniform(target_width_fraction * 0.95, target_width_fraction * 1.05)
        _width_fraction, font_size, mask, bbox = min(
            viable,
            key=lambda item: abs(item[0] - target_width),
        )
        return font_size, mask, bbox

    raise RuntimeError(
        f"Could not fit text {text!r} into width fraction range "
        f"{min_width_fraction:.3f}-{max_width_fraction:.3f}"
    )


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
