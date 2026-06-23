#!/usr/bin/env python3
"""Sort images in a folder into subfolders by aspect ratio.

Images with the same rounded ratio land in the same subfolder:
    4000×6000  →  0.667  →  startfolder/0_667/
    2000×3000  →  0.667  →  startfolder/0_667/   (same)
    2001×3001  →  0.667  →  startfolder/0_667/   (same, 3 decimal places)
    6000×4000  →  1.500  →  startfolder/1_500/   (different)

Usage:
    python sort_by_ratio.py /path/to/images
    python sort_by_ratio.py /path/to/images --dry-run
    python sort_by_ratio.py /path/to/images --copy
    python sort_by_ratio.py /path/to/images --precision 4

Pillow is used automatically when installed (adds WebP/TIFF/HEIC/BMP support).
Without Pillow, JPEG and PNG are handled natively.
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

# Use Pillow for reading dimensions when available — broader format support.
try:
    from PIL import Image as _PilImage
    _HAS_PILLOW = True
except ImportError:
    _HAS_PILLOW = False


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".bmp"}


# ---------------------------------------------------------------------------
# Image dimension readers
# ---------------------------------------------------------------------------

def _dims_pillow(path: Path):
    try:
        with _PilImage.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def _dims_png(path: Path):
    """Read width × height from PNG IHDR chunk (native, no deps)."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            f.read(8)  # chunk length + "IHDR"
            w, h = struct.unpack(">II", f.read(8))
            return w, h
    except Exception:
        return None


def _dims_jpeg(path: Path):
    """Scan JPEG stream for an SOF marker and read height × width (native, no deps)."""
    # SOF markers that carry image dimensions
    SOF_MARKERS = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = f.read(1)
                if not byte:
                    return None
                if byte != b"\xff":
                    continue
                marker_byte = f.read(1)
                if not marker_byte:
                    return None
                marker = marker_byte[0]
                if marker in (0x00, 0xFF):
                    continue  # padding or fill
                if marker == 0xD9:
                    return None  # EOI
                if marker in SOF_MARKERS:
                    f.read(3)  # segment length (2) + sample precision (1)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                # Skip this segment
                seg_len_bytes = f.read(2)
                if len(seg_len_bytes) < 2:
                    return None
                seg_len = struct.unpack(">H", seg_len_bytes)[0]
                f.seek(max(0, seg_len - 2), 1)
    except Exception:
        return None


def get_dimensions(path: Path):
    """Return (width, height) or None if the file can't be read."""
    if _HAS_PILLOW:
        return _dims_pillow(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _dims_png(path)
    if suffix in (".jpg", ".jpeg"):
        return _dims_jpeg(path)
    return None  # unsupported without Pillow


# ---------------------------------------------------------------------------
# Ratio bucketing
# ---------------------------------------------------------------------------

def ratio_key(width: int, height: int, precision: int) -> str:
    """Round width/height to `precision` decimal places; return as folder name."""
    ratio = round(width / height, precision)
    return f"{ratio:.{precision}f}".replace(".", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sort images into subfolders by aspect ratio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", help="Folder containing images to sort")
    parser.add_argument(
        "--precision", type=int, default=3, metavar="N",
        help="Decimal places for ratio rounding (default: 3). "
             "Lower = more images grouped together.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without moving anything.",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving them.",
    )
    args = parser.parse_args()

    src = Path(args.folder).resolve()
    if not src.is_dir():
        sys.exit(f"Error: '{src}' is not a directory.")

    candidates = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        sys.exit(f"No image files found in '{src}'.")

    if not _HAS_PILLOW:
        print("Note: Pillow not found — only JPEG and PNG are supported natively.\n"
              "      Install Pillow (pip install pillow) for WebP/TIFF/HEIC/BMP.\n")

    skipped = []
    plan = []  # list of (src_path, dest_dir, key, w, h)

    for img_path in candidates:
        dims = get_dimensions(img_path)
        if dims is None:
            skipped.append(img_path.name)
            continue
        w, h = dims
        key = ratio_key(w, h, args.precision)
        dest_dir = src / key
        plan.append((img_path, dest_dir, key, w, h))

    if not plan:
        if skipped:
            sys.exit(f"Could not read dimensions for any file: {', '.join(skipped)}")
        sys.exit("Nothing to do.")

    # Group summary
    buckets: dict[str, list] = {}
    for img_path, dest_dir, key, w, h in plan:
        buckets.setdefault(key, []).append((img_path.name, w, h))

    print(f"Found {len(plan)} image(s) → {len(buckets)} aspect ratio bucket(s):\n")
    for key in sorted(buckets):
        print(f"  {key}/")
        for name, w, h in buckets[key]:
            print(f"    {name}  ({w}×{h})")

    if skipped:
        print(f"\nSkipped ({len(skipped)} unreadable): {', '.join(skipped)}")

    if args.dry_run:
        print("\nDry run — nothing moved.")
        return

    verb = "Copy" if args.copy else "Move"
    print(f"\n{verb} {len(plan)} file(s) into {len(buckets)} subfolder(s)? [y/N] ",
          end="", flush=True)
    if input().strip().lower() != "y":
        print("Aborted.")
        return

    op = shutil.copy2 if args.copy else shutil.move
    moved = 0
    errors = 0
    for img_path, dest_dir, key, w, h in plan:
        try:
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / img_path.name
            if dest.exists():
                print(f"  Warning: '{dest.name}' already exists in '{key}/' — skipping.")
                errors += 1
                continue
            op(str(img_path), str(dest))
            moved += 1
        except Exception as e:
            print(f"  Error moving '{img_path.name}': {e}")
            errors += 1

    verb_past = "copied" if args.copy else "moved"
    print(f"\nDone — {moved} file(s) {verb_past}." +
          (f" {errors} skipped." if errors else ""))


if __name__ == "__main__":
    main()
