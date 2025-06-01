"""Command-line interface for untext."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import torch

from untext.image_patcher import ImagePatcher
from untext.word_mask_generator import WordMaskGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove text-based watermarks from images using Deep Image Prior"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--mask",
        type=str,
        choices=["box", "letters"],
        default="box",
        help="Mask geometry: 'box' (default) or 'letters' for glyph-level masks",
    )
    parser.add_argument(
        "--mask-file",
        type=str,
        help="Path to an existing binary mask image. If given, generation step is skipped.",
    )
    parser.add_argument(
        "--delete-mask",
        action="store_true",
        help="Delete the generated mask file after in-painting completes.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dip", "lama", "edge_fill"],
        default="lama",
        help="Inpainting backend to use (default: lama)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of optimization iterations for DIP (ignored for LaMa)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector and patcher
    print("Initializing models...")
    patcher = ImagePatcher(device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                           num_iterations=args.iterations)
    mask_gen = WordMaskGenerator(mode=args.mask)

    # Process each image in the directory
    if input_path.is_dir():
        for image_file in input_path.glob('*.*'):
            process_image(image_file, output_path, mask_gen, patcher, args)
    else:
        process_image(input_path, output_path, mask_gen, patcher, args)

    print("Done!")


def process_image(image_path, output_dir, mask_gen, patcher, args):
    """Process a single image."""
    print(f"Processing {image_path}...")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image '{image_path}'", file=sys.stderr)
        return
    
    # Generate or load mask
    mask_map = mask_gen.generate_masks([str(image_path)])
    if Path(image_path) not in mask_map:
        print("No mask generated; text detector found nothing.")
        return
    mask_path = mask_map[Path(image_path)]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Inpaint
    tmp_mask = NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp_mask.name, mask)
    result = patcher.patch_image(str(image_path), tmp_mask.name, method=args.method, output_path=None, blend=True)

    # Save result
    output_file = output_dir / f"{image_path.stem}_inpainted.jpg"
    cv2.imwrite(str(output_file), result)
    print(f"Saved inpainted image to '{output_file}'")

    # Clean up
    tmp_mask.close()
    os.unlink(tmp_mask.name)
    if args.delete_mask:
        mask_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main() 