"""Command-line interface for untextre.

This module provides the main CLI entry point that orchestrates the complete
text watermark removal pipeline as defined in the TODO.md workflow.
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional

from .utils import (
    ImageArray, setup_logger, load_image, save_image, 
    get_image_files, IMAGE_EXTENSIONS
)
from .preprocessor import preprocess_image
from .detector import get_largest_text_region
from .find_text_colors import find_text_colors, hex_to_bgr, html_to_bgr
from .mask_generator import generate_mask
from .inpaint import inpaint_image

logger = setup_logger(__name__)

def main() -> None:
    """Main CLI entry point."""
    args = parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Get list of images to process
    image_files = get_image_files(input_path)
    if not image_files:
        logger.error(f"No valid image files found in '{args.input}'")
        sys.exit(1)
    
    # Setup output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse target color if provided
    target_color = None
    if args.color:
        if args.color.startswith('#'):
            target_color = hex_to_bgr(args.color)
        else:
            target_color = html_to_bgr(args.color)
        logger.info(f"Using target color: {target_color}")
    
    logger.info(f"Found {len(image_files)} image(s) to process")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        try:
            process_single_image(image_path, output_path, target_color, args.keep_masks, args.method)
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            if args.keep_masks:
                # Save error log if requested
                error_file = output_path / f"{image_path.stem}.txt"
                error_file.write_text(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    # Calculate and log timing information
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files)
    logger.info(f"\nProcessing complete:")
    logger.info(f"Total elapsed time: {total_time:.1f} seconds")
    logger.info(f"Average time per image: {avg_time:.1f} seconds")
    logger.info(f"Images processed: {len(image_files)}")

def process_single_image(
    image_path: Path, 
    output_dir: Path, 
    target_color: Optional[tuple] = None,
    keep_masks: bool = False,
    method: str = "lama"
) -> None:
    """Process a single image through the complete pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        target_color: Optional target color as BGR tuple
        keep_masks: Whether to save debug masks
        method: Inpainting method to use ("lama" or "telea")
    """
    logger.info(f"Loading image: {image_path.name}")
    
    # 1. Load and preprocess image
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    if preprocessed is None:
        raise ValueError("Image preprocessing failed")
    
    # 2. Detect text regions 
    logger.info("Detecting text regions...")
    try:
        bbox = get_largest_text_region(preprocessed)
        logger.info(f"Found text region: {bbox}")
    except ValueError:
        # TODO: Implement fallback to corner-based region selection
        logger.warning("No text detected - using fallback region")
        h, w = image.shape[:2]
        # Use bottom-right corner as fallback (1/4 width, 1/16 height)
        bbox = (3*w//4, 15*h//16, w//4, h//16)
        logger.info(f"Using fallback region: {bbox}")
    
    # 3. Find text colors
    logger.info("Analyzing colors in text region...")
    representative_color, text_colors = find_text_colors(image, bbox, target_color)
    logger.info(f"Selected representative color: {representative_color}")
    logger.info(f"Found {len(text_colors)} text colors to mask")
    
    # 4. Generate mask
    logger.info("Generating binary mask...")
    mask = generate_mask(image, text_colors, bbox)
    
    # 5. Inpaint image
    logger.info("Inpainting masked regions...")
    result = inpaint_image(image, mask, bbox, method=method)
    
    # Save results
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path)
    logger.info(f"Saved result to: {output_path.name}")
    
    # Optionally save mask for debugging
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to: {mask_path.name}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove text watermarks from images using color-based detection and inpainting."
    )
    
    parser.add_argument(
        "input",
        help="Path to input image file or directory of images"
    )
    
    parser.add_argument(
        "output", 
        help="Path to output directory"
    )
    
    parser.add_argument(
        "-c", "--color",
        help="Target color in hex format (#808080) or HTML color name (gray)"
    )
    
    parser.add_argument(
        "-k", "--keep-masks",
        action="store_true",
        help="Save debug masks alongside output images"
    )
    
    parser.add_argument(
        "-m", "--method",
        choices=["lama", "telea"],
        default="lama",
        help="Inpainting method to use (default: lama)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 