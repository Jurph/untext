"""Generate deterministic text-watermark fixture images.

This script creates hand-placed, high-contrast text overlays for integration
tests. It writes watermarked images, binary truth masks, and a manifest under
``tests/images/generated_text_watermarks``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = ROOT / "tests" / "images" / "originals"
OUTPUT_DIR = ROOT / "tests" / "images" / "generated_text_watermarks"

FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {
    "serif": FONT_DIR / "georgiab.ttf",
    "sans": FONT_DIR / "corbelb.ttf",
    "cursive": FONT_DIR / "AlexBrush-Regular.ttf",
}


@dataclass(frozen=True)
class FixtureCase:
    slug: str
    source: str
    text: str
    family: str
    color: tuple[int, int, int]
    anchor: str
    x_frac: float
    y_frac: float
    width_frac: float
    opacity: int = 225


CASES = [
    FixtureCase(
        "elena_floor_black_sans",
        "portraits/elena_koshka_let_my_body.jpg",
        "ElenaFans.co",
        "sans",
        (8, 8, 8),
        "lb",
        0.08,
        0.92,
        0.38,
    ),
    FixtureCase(
        "evelyn_flowers_white_serif",
        "portraits/evelyn_cherry_blossoms.jpg",
        "EvelynBloom.net",
        "serif",
        (245, 245, 245),
        "rt",
        0.94,
        0.08,
        0.40,
    ),
    FixtureCase(
        "evelyn_diner_white_sans",
        "portraits/evelyn_coffee.jpg",
        "DinerEvelyn.com",
        "sans",
        (248, 248, 248),
        "rb",
        0.92,
        0.82,
        0.36,
    ),
    FixtureCase(
        "evelyn_corset_gray_cursive",
        "portraits/evelyn_rennfaire.jpg",
        "FaireEvelyn.co",
        "cursive",
        (132, 132, 132),
        "mm",
        0.50,
        0.63,
        0.40,
    ),
    FixtureCase(
        "kenna_avn_black_serif",
        "portraits/kenna_james_avn_2016.jpg",
        "KennaStage.org",
        "serif",
        (10, 10, 10),
        "lt",
        0.08,
        0.12,
        0.38,
    ),
    FixtureCase(
        "kenna_march_black_sans",
        "portraits/kenna_james_march2023.jpg",
        "KennaLookbook.net",
        "sans",
        (8, 8, 8),
        "rt",
        0.90,
        0.12,
        0.42,
    ),
    FixtureCase(
        "lamar_crowd_white_serif",
        "portraits/lamar_jackson_2021.jpg",
        "RavensZone2026",
        "serif",
        (250, 250, 250),
        "lt",
        0.07,
        0.12,
        0.42,
    ),
    FixtureCase(
        "lamar_turf_white_sans",
        "portraits/lamar_jackson_passing_2020.jpg",
        "LamarFilmRoom.tv",
        "sans",
        (248, 248, 248),
        "lb",
        0.08,
        0.90,
        0.42,
    ),
    FixtureCase(
        "tay_iheart_gray_cursive",
        "portraits/taylor_swift_2019.jpg",
        "SwiftieZone (2026)",
        "cursive",
        (176, 176, 176),
        "rt",
        0.94,
        0.12,
        0.46,
    ),
    FixtureCase(
        "tay_globes_gray_serif",
        "portraits/taylor_swift_golden_globes_2024.jpg",
        "TayStyle.blog",
        "serif",
        (168, 168, 168),
        "rb",
        0.92,
        0.88,
        0.36,
    ),
    FixtureCase(
        "cleveland_bottom_white_sans",
        "landscapes/cleveland_skyline.jpg",
        "ClevelandSkyline.fans",
        "sans",
        (245, 245, 245),
        "mb",
        0.50,
        0.92,
        0.38,
    ),
    FixtureCase(
        "copenhagen_bottom_gray_serif",
        "landscapes/copenhagen_lakes.jpg",
        "cityscapes.nl/fans",
        "serif",
        (220, 220, 220),
        "mb",
        0.50,
        0.91,
        0.34,
    ),
    FixtureCase(
        "frankfurt_bottom_white_sans",
        "landscapes/frankfurt_skyline.jpg",
        "FrankfurtView.de",
        "sans",
        (248, 248, 248),
        "mb",
        0.50,
        0.91,
        0.34,
    ),
    FixtureCase(
        "taipei_bottom_gray_cursive",
        "landscapes/taipei_night.jpg",
        "TaipeiAfterDark.tw",
        "cursive",
        (214, 214, 214),
        "mb",
        0.50,
        0.91,
        0.36,
    ),
    FixtureCase(
        "hongkong_high_black_serif",
        "landscapes/hong_kong_harbour.jpg",
        "HarbourWatch.hk",
        "serif",
        (10, 10, 10),
        "mt",
        0.50,
        0.08,
        0.36,
    ),
    FixtureCase(
        "harbaugh_parka_white_sans",
        "landscapes/harbaugh_iphone.jpg",
        "SidelineReport.net",
        "sans",
        (250, 250, 250),
        "rb",
        0.93,
        0.82,
        0.38,
    ),
    FixtureCase(
        "evelyn_interview_lightgray_serif",
        "landscapes/evelyn_claire_hru_2023.jpg",
        "EvelynInterview.tv",
        "serif",
        (214, 214, 214),
        "rb",
        0.94,
        0.86,
        0.38,
    ),
]


def _load_font(case: FixtureCase, image_width: int) -> ImageFont.FreeTypeFont:
    font_path = FONTS[case.family]
    if not font_path.exists():
        raise FileNotFoundError(f"Missing fixture font: {font_path}")

    target_width = int(image_width * case.width_frac)
    low, high = 12, 220
    best = low
    probe = Image.new("L", (10, 10))
    draw = ImageDraw.Draw(probe)
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(str(font_path), mid)
        bbox = draw.textbbox((0, 0), case.text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= target_width:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return ImageFont.truetype(str(font_path), best)


def _text_position(
    case: FixtureCase,
    image_size: tuple[int, int],
    text_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    width, height = image_size
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = int(width * case.x_frac)
    y = int(height * case.y_frac)

    horizontal = case.anchor[0]
    vertical = case.anchor[1]
    if horizontal == "m":
        x -= text_w // 2
    elif horizontal == "r":
        x -= text_w
    if vertical == "m":
        y -= text_h // 2
    elif vertical == "b":
        y -= text_h

    margin = max(8, min(width, height) // 80)
    x = max(margin, min(x, width - text_w - margin))
    y = max(margin, min(y, height - text_h - margin))
    return x, y


def _draw_case(case: FixtureCase) -> dict:
    source_path = SOURCE_DIR / case.source
    image = Image.open(source_path).convert("RGB")
    font = _load_font(case, image.width)

    probe = Image.new("L", image.size, 0)
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), case.text, font=font)
    x, y = _text_position(case, image.size, bbox)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask = Image.new("L", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask)
    fill = (*case.color, case.opacity)

    offsets = [(0, 0)]
    if case.family == "cursive":
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for dx, dy in offsets:
        overlay_draw.text((x + dx, y + dy), case.text, font=font, fill=fill)
        mask_draw.text((x + dx, y + dy), case.text, font=font, fill=255)

    watermarked = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    image_dir = OUTPUT_DIR / "images"
    mask_dir = OUTPUT_DIR / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_out = image_dir / f"{case.slug}{source_path.suffix.lower()}"
    mask_out = mask_dir / f"{case.slug}_mask.png"
    watermarked.save(image_out, quality=95)
    mask.save(mask_out)

    mask_bbox = mask.getbbox()
    if mask_bbox is None:
        raise RuntimeError(f"Generated empty mask for {case.slug}")

    return {
        **asdict(case),
        "source": case.source.replace("\\", "/"),
        "image": str(image_out.relative_to(OUTPUT_DIR)).replace("\\", "/"),
        "mask": str(mask_out.relative_to(OUTPUT_DIR)).replace("\\", "/"),
        "font": str(FONTS[case.family]),
        "font_size": font.size,
        "bbox_xyxy": list(mask_bbox),
        "bbox_xywh": [
            mask_bbox[0],
            mask_bbox[1],
            mask_bbox[2] - mask_bbox[0],
            mask_bbox[3] - mask_bbox[1],
        ],
        "mask_coverage": sum(1 for value in mask.getdata() if value) / (image.width * image.height),
    }


def main() -> None:
    if len(CASES) != 17:
        raise RuntimeError(f"Expected 17 fixture cases, got {len(CASES)}")
    family_counts = {family: sum(1 for case in CASES if case.family == family) for family in FONTS}
    if family_counts != {"serif": 7, "sans": 7, "cursive": 3}:
        raise RuntimeError(f"Unexpected font-family distribution: {family_counts}")

    rows = [_draw_case(case) for case in CASES]
    manifest = {
        "description": "Deterministic high-contrast generated text-watermark fixtures.",
        "case_count": len(rows),
        "font_family_counts": family_counts,
        "cases": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} fixtures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
