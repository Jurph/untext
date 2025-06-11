"""Command-line interface for untextre.

This module provides the main CLI entry point that orchestrates the complete
text watermark removal pipeline as defined in the TODO.md workflow.
"""

import logging
import argparse
import cv2
import time
import sys
import statistics
from pathlib import Path
from typing import Optional

from .utils import (
    get_image_files, load_image, save_image, setup_logger
)
from .preprocessor import preprocess_image
from .detector import get_largest_text_region, initialize_models
from .find_text_colors import find_text_colors, hex_to_bgr, html_to_bgr
from .mask_generator import generate_mask
from .inpaint import inpaint_image

logger = setup_logger(__name__)

def main() -> None:
    """Main entry point for the text watermark removal tool."""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Setup file logging if requested
    if args.logfile:
        log_path = Path(args.logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")
    
    # Start timing
    start_time = time.time()
    detailed_timings = [] if args.timing else None
    
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
    
    # Initialize models once for persistent loading
    logger.info("Initializing detection models...")
    model_init_start = time.time()
    initialize_models([args.detector])
    model_init_time = time.time() - model_init_start
    logger.info(f"Model initialization complete in {model_init_time:.1f} seconds")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            timing_data = process_single_image(image_path, output_path, target_color, args.keep_masks, args.paint, args.maskfile, args.detector)
            
            if args.timing and timing_data:
                detailed_timings.append(timing_data)
                # Simple progress log - detailed report will be saved to file
                logger.info(f"Image processed in {timing_data['total_time']:.1f}s")
                
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            if args.keep_masks:
                # Save error log if requested
                error_file = output_path / f"{image_path.stem}.txt"
                error_file.write_text(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    # Calculate and log timing information
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files) if image_files else 0
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"Total elapsed time: {total_time:.1f} seconds")
    logger.info(f"Average time per image: {avg_time:.1f} seconds")
    logger.info(f"Images processed: {len(image_files)}")
    
    # Detailed timing report if requested
    if args.timing and detailed_timings:
        # Always save timing report to a clean file
        timing_file = output_path / "timing_report.txt"
        _save_clean_timing_report(detailed_timings, total_time, avg_time, timing_file)
        logger.info(f"Timing report saved to: {timing_file}")
        
        # Also save to logfile location if specified
        if args.logfile:
            log_timing_file = Path(args.logfile).with_suffix('.timing.txt')
            _save_clean_timing_report(detailed_timings, total_time, avg_time, log_timing_file)
            logger.info(f"Timing report also saved to: {log_timing_file}")

def process_single_image(
    image_path: Path, 
    output_dir: Path, 
    target_color: Optional[tuple] = None,
    keep_masks: bool = False,
    method: str = "lama",
    maskfile: Optional[str] = None,
    detector: str = "doctr"
) -> Optional[dict]:
    """Process a single image through the complete pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        target_color: Optional target color as BGR tuple
        keep_masks: Whether to save debug masks
        method: Inpainting method to use ("lama" or "telea")
        maskfile: Optional path to mask file to use instead of auto-generation
        detector: Text detection method to use ("doctr" or "easyocr")
        
    Returns:
        Dictionary with timing details, or None if processing failed
    """
    logger.info(f"Loading image: {image_path.name}")
    
    # Initialize timing dictionary
    timings = {
        'image_name': image_path.name,
        'load_time': 0,
        'detection_time': 0,
        'color_time': 0, 
        'mask_time': 0,
        'inpaint_time': 0,
        'total_time': 0,
        'image_mp': 0,
        'text_colors_count': 0,
        'bbox_area': 0
    }
    
    start_time = time.time()
    
    # 1. Load and preprocess image
    load_start = time.time()
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    if preprocessed is None:
        raise ValueError("Image preprocessing failed")
    
    timings['load_time'] = time.time() - load_start
    timings['image_mp'] = (image.shape[0] * image.shape[1]) / 1_000_000
    
    # 2. Detect text regions 
    logger.info(f"Detecting text regions using {detector.upper()}...")
    detection_start = time.time()
    try:
        bbox = get_largest_text_region(preprocessed, method=detector)
        logger.info(f"Found text region: {bbox}")
    except ValueError:
        # TODO: Implement fallback to corner-based region selection
        logger.warning("No text detected - using fallback region")
        h, w = image.shape[:2]
        # Use bottom-right corner as fallback (1/4 width, 1/8 height)
        bbox = (3*w//4, 7*h//8, w//4, h//8)
        logger.info(f"Using fallback region: {bbox}")
    
    timings['detection_time'] = time.time() - detection_start
    timings['bbox_area'] = bbox[2] * bbox[3]
    
    # 3. Generate or load mask
    if maskfile:
        mask_start = time.time()
        logger.info(f"Loading mask from file: {maskfile}")
        mask_path = Path(maskfile)
        if not mask_path.exists():
            raise ValueError(f"Mask file not found: {maskfile}")
        mask = load_image(mask_path)
        # Ensure mask is single channel
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        timings['mask_time'] = time.time() - mask_start
        timings['text_colors_count'] = 0  # External mask
    else:
        # Find text colors and generate mask
        color_start = time.time()
        logger.info("Analyzing colors in text region...")
        representative_colors, text_colors = find_text_colors(image, bbox, target_color, detector)
        logger.info(f"Selected {len(representative_colors)} representative colors: {representative_colors}")
        logger.info(f"Found {len(text_colors)} text colors to mask")
        timings['color_time'] = time.time() - color_start
        timings['text_colors_count'] = len(text_colors)
        
        mask_start = time.time()
        logger.info("Generating binary mask...")
        mask = generate_mask(image, text_colors, bbox)
        timings['mask_time'] = time.time() - mask_start
    
    # 4. Inpaint image
    inpaint_start = time.time()
    logger.info("Inpainting masked regions...")
    result = inpaint_image(image, mask, bbox, method=method)
    timings['inpaint_time'] = time.time() - inpaint_start
    
    # Save results
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path)
    logger.info(f"Saved result to: {output_path.name}")
    
    # Optionally save mask for debugging
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to: {mask_path.name}")
    
    # Calculate total time and return timings
    timings['total_time'] = time.time() - start_time
    return timings

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove text watermarks from images using color-based detection and inpainting."
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image file or directory of images"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Path to output directory"
    )
    
    parser.add_argument(
        "-c", "--color",
        help="Target color in hex format (#808080) or HTML color name (gray)"
    )
    
    parser.add_argument(
        "-d", "--detector",
        choices=["doctr", "easyocr", "east"],
        default="doctr",
        help="Text detection method to use (default: doctr). EAST: efficient OpenCV-based detector, EasyOCR: OCR-based detector, Doctr: OCR-based detector with DocTR model"
    )
    
    parser.add_argument(
        "-m", "--maskfile",
        help="Path to mask file (PNG) to use instead of auto-generated mask"
    )
    
    parser.add_argument(
        "-p", "--paint",
        choices=["lama", "telea"],
        default="lama",
        help="Inpainting method to use (default: lama)"
    )
    
    parser.add_argument(
        "-k", "--keep-masks",
        action="store_true",
        help="Save debug masks alongside output images"
    )
    
    parser.add_argument(
        "-t", "--timing",
        action="store_true",
        help="Create detailed timing report"
    )
    
    parser.add_argument(
        "-l", "--logfile",
        help="Path to log file for detailed logging"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def _save_clean_timing_report(detailed_timings: list, total_time: float, avg_time: float, timing_file: Path) -> None:
    """Save a clean timing report to file without duplicate logging."""
    with open(timing_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DETAILED TIMING REPORT\n")
        f.write("=" * 60 + "\n")
        
        # Header with wider format
        f.write(f"{'Image Name':<25} {'MP':>4} {'Det':>4} {'Col':>4} {'Msk':>4} {'Inp':>5} {'Tot':>5}\n")
        f.write("-" * 60 + "\n")
        
        # Individual rows with longer names
        for timing in detailed_timings:
            name = timing['image_name'][:25]  # Allow longer names
            row = (f"{name:<25} "
                   f"{timing['image_mp']:>4.1f} "
                   f"{timing['detection_time']:>4.1f} "
                   f"{timing['color_time']:>4.1f} "
                   f"{timing['mask_time']:>4.1f} "
                   f"{timing['inpaint_time']:>5.1f} "
                   f"{timing['total_time']:>5.1f}\n")
            f.write(row)
        
        if len(detailed_timings) > 1:
            f.write("-" * 60 + "\n")
            
            # Statistics
            det_times = [t['detection_time'] for t in detailed_timings]
            col_times = [t['color_time'] for t in detailed_timings] 
            msk_times = [t['mask_time'] for t in detailed_timings]
            inp_times = [t['inpaint_time'] for t in detailed_timings]
            tot_times = [t['total_time'] for t in detailed_timings]
            
            f.write(f"{'MEDIAN':<25} {'':>4} {statistics.median(det_times):>4.1f} "
                   f"{statistics.median(col_times):>4.1f} {statistics.median(msk_times):>4.1f} "
                   f"{statistics.median(inp_times):>5.1f} {statistics.median(tot_times):>5.1f}\n")
            
            f.write(f"{'MAX':<25} {'':>4} {max(det_times):>4.1f} "
                   f"{max(col_times):>4.1f} {max(msk_times):>4.1f} "
                   f"{max(inp_times):>5.1f} {max(tot_times):>5.1f}\n")
        
        f.write(f"{'TOTAL':<25} {'':>4} {'':>4} {'':>4} {'':>4} {'':>5} {total_time:>5.1f}\n")
        f.write("=" * 60 + "\n")
        
        # Add insights section
        f.write("\nKEY INSIGHTS:\n")
        f.write(f"- {len(detailed_timings)} images processed in {total_time:.0f} seconds ({total_time/60:.1f} minutes)\n")
        
        if len(detailed_timings) > 1:
            det_times = [t['detection_time'] for t in detailed_timings]
            col_times = [t['color_time'] for t in detailed_timings]
            msk_times = [t['mask_time'] for t in detailed_timings]
            inp_times = [t['inpaint_time'] for t in detailed_timings]
            
            f.write(f"- Detection time is consistent ({min(det_times):.1f}-{max(det_times):.1f}s, median {statistics.median(det_times):.1f}s)\n")
            f.write(f"- Color analysis is very fast ({min(col_times):.1f}-{max(col_times):.1f}s, median {statistics.median(col_times):.1f}s)\n")
            f.write(f"- Mask generation varies widely ({min(msk_times):.1f}-{max(msk_times):.1f}s, median {statistics.median(msk_times):.1f}s)\n")
            f.write(f"- Inpainting is the biggest variable ({min(inp_times):.1f}-{max(inp_times):.1f}s, median {statistics.median(inp_times):.1f}s)\n")
            
            # Identify bottlenecks
            f.write("\nBOTTLENECKS IDENTIFIED:\n")
            
            # Find worst performers
            worst_mask = max(detailed_timings, key=lambda x: x['mask_time'])
            worst_inpaint = max(detailed_timings, key=lambda x: x['inpaint_time'])
            
            if worst_mask['mask_time'] > 20:
                f.write(f"1. Large images with complex masks ({worst_mask['image_name']}: {worst_mask['mask_time']:.1f}s mask)\n")
            if worst_inpaint['inpaint_time'] > 50:
                f.write(f"2. Very slow inpainting cases ({worst_inpaint['image_name']}: {worst_inpaint['inpaint_time']:.1f}s inpaint)\n")
            
            # Count empty masks
            empty_masks = sum(1 for t in detailed_timings if t['inpaint_time'] == 0.0)
            if empty_masks > 0:
                f.write(f"3. {empty_masks} images had empty masks (correctly detected and skipped)\n")

def _print_detailed_timing_report(detailed_timings: list, total_time: float, avg_time: float) -> None:
    """Print a detailed timing report in 40-column format."""
    
    logger.info("\n" + "=" * 40)
    logger.info("DETAILED TIMING REPORT")
    logger.info("=" * 40)

if __name__ == "__main__":
    main() 