"""Command-line interface for untextre.

This module provides the main CLI entry point that orchestrates the complete
text watermark removal pipeline using consensus detection from multiple detectors.
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List

# Lightweight imports only — heavy ML modules (.detector, .consensus,
# .inpaint, .find_text_colors, .preprocessor, .metrics) are imported
# lazily inside the functions that need them so that `--help` and
# watermark-only runs don't pay the TF/PyTorch startup cost.
from .utils import (
    get_image_files, load_image, save_image, setup_logger,
    CLI_DEFAULT_CONFIDENCE, configure_logging,
)
logger = setup_logger(__name__)

# Compatibility re-exports for callers that imported helpers from untextre.cli.
from .known_mask import process_with_known_mask
from .pipeline import (
    _apply_color_enhancement,
    _generate_masks_and_inpaint,
    _translate_rotated_bbox_to_original,
    _try_color_enhanced_detection,
    initialize_consensus_models,
    process_single_image,
)
from .orb_matcher import (
    WatermarkTemplate,
    find_known_mask_in_image,
    load_watermark_templates,
    try_watermark_cascade,
)
from .reports import (
    _save_clean_timing_report,
    _save_discovered_watermark_candidates,
)








def main() -> None:
    """Main entry point for the consensus-based text watermark removal tool."""
    args = create_parser().parse_args()
    configure_logging(verbose=args.verbose, logfile=args.logfile)
    if args.verbose:
        logger.debug("Debug logging enabled")
    if args.logfile:
        logger.info(f"Logging to file: {args.logfile}")
    
    # Parse forced bounding box if provided
    forced_bbox = None
    if args.force_bbox:
        try:
            parts = args.force_bbox.split(',')
            if len(parts) != 4:
                raise ValueError("Bounding box must have exactly 4 values: x,y,width,height")
            forced_bbox = tuple(int(x.strip()) for x in parts)
            if any(x < 0 for x in forced_bbox):
                raise ValueError("Bounding box values must be non-negative")
            if forced_bbox[2] <= 0 or forced_bbox[3] <= 0:
                raise ValueError("Width and height must be positive")
            logger.info(f"Using forced bounding box: {forced_bbox}")
        except ValueError as e:
            print(f"Error: Invalid bounding box format: {e}")
            print("Use x,y,width,height where x,y is the top-left corner.")
            print("Example: --force-bbox 593,1013,105,39")
            sys.exit(1)
    
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
        from .find_text_colors import hex_to_bgr, html_to_bgr

        if args.color.startswith('#'):
            target_color = hex_to_bgr(args.color)
        else:
            target_color = html_to_bgr(args.color)
        # Convert back to hex for display
        b, g, r = target_color
        target_hex = f"#{r:02X}{g:02X}{b:02X}"
        logger.info(f"Using target color for immediate enhancement: {target_hex} (BGR: {target_color})")
    
    logger.info(f"Found {len(image_files)} image(s) to process")

    # ── -U: same-directory guard ──────────────────────────────────────────
    if args.unknown_watermark:
        if input_path.resolve() == output_path.resolve() and not args.force:
            logger.error(
                "Input and output directories are the same. "
                "This would overwrite originals. Use --force to proceed."
            )
            sys.exit(1)

    # ── -U: auto-discover watermark templates ─────────────────────────────
    if args.unknown_watermark:
        if not input_path.is_dir():
            logger.error("-U requires a directory input, not a single file")
            sys.exit(1)
        from .discovery import discover_watermark_candidates

        logger.info("Running watermark discovery (-U mode)...")
        debug_dir = output_path if args.debug_discovery else None
        candidates = discover_watermark_candidates(image_files, debug_dir=debug_dir)

        if not candidates:
            logger.error(
                "No watermark candidates discovered. "
                "Try -K with a manually-identified template."
            )
            sys.exit(1)

        # Export candidates in orb-prepped form so on-disk templates match -K inputs.
        watermark_templates = _save_discovered_watermark_candidates(output_path, candidates)

    # ── Load watermark templates ─────────────────────────────────────
    # Priority: -U (already loaded above) > -K flag > auto-check watermarks/ dir
    if not args.unknown_watermark:
        watermark_templates: List[WatermarkTemplate] = []
        if args.known_mask:
            known_mask_path = Path(args.known_mask)
            watermark_templates = load_watermark_templates(known_mask_path)
            if not watermark_templates:
                logger.error(f"No valid RGBA templates found at: {args.known_mask}")
                sys.exit(1)
        else:
            # Auto-check the watermarks/ directory next to the package root
            default_watermarks_dir = Path(__file__).resolve().parent.parent / "watermarks"
            watermark_templates = load_watermark_templates(default_watermarks_dir)

    if watermark_templates:
        names = ", ".join(name for name, _ in watermark_templates)
        logger.info(f"Watermark templates loaded: {names}")
    else:
        logger.info(f"Using consensus detection with confidence threshold: {args.confidence_threshold}")
        logger.info(f"Using spatial TF-IDF with g=4 (auto-retry with g=8 if needed: {not args.no_retry})")
        bbox_expansion_on = not args.no_expand and not args.grabcut_expand
        logger.info(f"Bbox expansion enabled: {bbox_expansion_on}"
                    + (" (suppressed by --grabcut-expand)" if args.grabcut_expand else ""))
    
    # Initialize models once for persistent loading.
    # If -K was given (explicit override), we ONLY try templates — no detection fallback.
    # If templates came from auto-check of watermarks/, we try them first but fall
    # back to consensus detection, so we need detection models loaded too.
    explicit_known_mask = bool(args.known_mask) or bool(args.unknown_watermark)
    model_init_start = time.time()

    if explicit_known_mask:
        # Explicit -K: only need inpainting model
        from .inpaint import initialize_lama_model
        if initialize_lama_model(device=args.device):
            logger.info("[OK] LaMa model loaded")
        else:
            logger.warning("LaMa model failed to initialize")
    else:
        # Auto-detected templates (or none): load everything so detection
        # fallback works if no template matches
        initialize_consensus_models(device=args.device)

    model_init_time = time.time() - model_init_start
    logger.info(f"Models ready in {model_init_time:.1f} seconds")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        image_start = time.perf_counter()
        image = None
        result = None
        mask = None
        cascade_result = None
        timing_data = None
        
        try:
            # ── Try watermark templates (first match wins) ────────────
            if watermark_templates:
                image = load_image(image_path)
                if image is not None:
                    cascade_result = try_watermark_cascade(
                        image, watermark_templates,
                    )
                    if cascade_result is not None:
                        mask, bbox, tmpl_name = cascade_result
                        # Inpaint and save
                        from .inpaint import inpaint_image

                        result = inpaint_image(image, mask, bbox=bbox, method=args.paint)
                        output_file = output_path / f"{image_path.stem}_clean{image_path.suffix}"
                        save_image(result, output_file, source_path=image_path)
                        logger.info(f"Saved cleaned image to {output_file}")
                        if args.keep_masks:
                            mask_file = output_path / f"{image_path.stem}_mask.png"
                            save_image(mask, mask_file)
                        timing_data = {
                            "image": image_path.name,
                            "matched_template": tmpl_name,
                            "mask_found": True,
                            "total_time": time.perf_counter() - image_start,
                        }
                    elif explicit_known_mask:
                        logger.warning(f"No template matched {image_path.name}")
                    else:
                        logger.info("No template matched, falling back to consensus detection")

            # ── Fall back to consensus detection ──────────────────────
            if timing_data is None and not explicit_known_mask:
                timing_data = process_single_image(
                    image_path=image_path,
                    output_dir=output_path,
                    target_color=target_color,
                    keep_masks=args.keep_masks,
                    method=args.paint,
                    maskfile=args.maskfile,
                    confidence_threshold=args.confidence_threshold,
                    granularity=args.granularity,
                    forced_bbox=forced_bbox,
                    expand_bboxes=not (args.no_expand or args.grabcut_expand),
                    auto_retry=not args.no_retry,
                    use_grabcut=args.grabcut,
                    use_grabcut_expand=args.grabcut_expand,
                    coverage_limit=args.coverage_limit,
                )
            
            # ── Handle skipped images ──────────────────────────────────
            if timing_data and timing_data.get('skipped'):
                if args.force_output:
                    # Copy the original unchanged so every input has output
                    output_file = output_path / f"{image_path.stem}_clean{image_path.suffix}"
                    save_image(load_image(image_path), output_file, source_path=image_path)
                    logger.info(f"No text detected - copied original to {output_file}")
                else:
                    logger.info(f"Skipped {image_path.name} (no text detected)")

            if args.timing and timing_data:
                detailed_timings.append(timing_data)
                if not timing_data.get('skipped'):
                    logger.info(f"Image processed in {timing_data['total_time']:.1f}s")
                
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            if args.keep_masks:
                # Save error log if requested
                error_file = output_path / f"{image_path.stem}.txt"
                error_file.write_text(f"Error processing {image_path.name}: {str(e)}")
            continue
        finally:
            # Release large per-image arrays before forcing Python/CUDA cleanup.
            image = None
            result = None
            mask = None
            cascade_result = None
            # Clean up VRAM between images to prevent accumulation
            from .detector import cleanup_vram

            cleanup_vram()
    
    # Calculate and log timing information
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files) if image_files else 0
    
    logger.info("\nProcessing complete:")
    logger.info(f"Total elapsed time: {total_time:.1f} seconds")
    logger.info(f"Average time per image: {avg_time:.1f} seconds")
    logger.info(f"Images processed: {len(image_files)}")
    
    # Detailed timing report if requested
    if args.timing and detailed_timings:
        # Always save timing report to a clean file
        timing_file = output_path / "timing_report.txt"
        _save_clean_timing_report(detailed_timings, total_time, avg_time, timing_file, args.paint, args.confidence_threshold, target_color, forced_bbox)
        logger.info(f"Timing report saved to: {timing_file}")
        
        # Also save to logfile location if specified
        if args.logfile:
            log_timing_file = Path(args.logfile).with_suffix('.timing.txt')
            _save_clean_timing_report(detailed_timings, total_time, avg_time, log_timing_file, args.paint, args.confidence_threshold, target_color, forced_bbox)
            logger.info(f"Timing report also saved to: {log_timing_file}")


def create_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (without parsing sys.argv)."""
    parser = argparse.ArgumentParser(
        description="Remove text watermarks from images using consensus detection and color-based inpainting."
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
        help="Target color for color enhancement failover (hex format like #808080 or HTML name like 'gray'). "
             "NOTE: Text colors are automatically detected via spatial TF-IDF. "
             "This flag only triggers immediate color enhancement if consensus detection finds no regions. "
             "Use for subtle watermarks that standard detection misses."
    )
    
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=CLI_DEFAULT_CONFIDENCE,
        help=f"Confidence threshold for consensus detection (default: {CLI_DEFAULT_CONFIDENCE})"
    )
    
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable automatic bbox expansion along long axis"
    )
    
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retry with g=8 if text remnants detected"
    )
    
    parser.add_argument(
        "-g", "--granularity",
        type=int,
        default=None,
        metavar="K",
        help="Override TF-IDF cluster count (e.g. 4, 8). If set, uses this K only (no g=8 retry). Default: auto g=4 with retry at g=8."
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
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    
    parser.add_argument(
        "-f", "--force-bbox",
        help="Force specific bounding box as x,y,width,height where x,y is the TOP-LEFT corner "
             "(e.g., 593,1013,105,39 selects a 105x39 region starting at top-left (593,1013))"
    )
    
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "-K", "--known-mask",
        help="Path to RGBA image (PNG with transparency) of a known watermark/logo, "
             "or a directory of such images. Uses ORB feature matching to find and mask "
             "the watermark at any scale/position (first match wins). The alpha channel "
             "defines the mask. Skips consensus detection when used."
    )
    mask_group.add_argument(
        "-U", "--unknown-watermark",
        action="store_true",
        default=False,
        help="Auto-discover watermark from input directory via low-variance stacking, "
             "save candidate BGRA template(s) to output dir, then process with ORB matching. "
             "Requires directory input. Mutually exclusive with -K."
    )

    parser.add_argument(
        "--debug-discovery",
        action="store_true",
        default=False,
        help="Save per-bucket debug images (log-variance map and mean image) to the output "
             "directory during -U discovery. Useful for diagnosing threshold issues."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow output directory to be the same as input directory. "
             "WARNING: cleaned images will overwrite originals."
    )

    parser.add_argument(
        "--grabcut",
        action="store_true",
        help="Refine FOM masks with GrabCut for spatially coherent edges. "
             "Adds ~50-200ms per region but may produce cleaner mask boundaries."
    )

    parser.add_argument(
        "--grabcut-expand",
        action="store_true",
        help="Extend masks beyond detected bboxes using color-guided GrabCut. "
             "Within an expanded ROI, seeds GrabCut with the highest-FOM color cluster "
             "as foreground and the lowest-FOM cluster as background. Automatically "
             "disables long-axis bbox expansion (--no-expand) since color_guided_expand "
             "handles the outward search itself. Useful for partially-detected watermarks."
    )

    parser.add_argument(
        "--coverage-limit",
        type=float,
        default=0.06,
        metavar="FRACTION",
        help="Skip inpainting when the mask covers more than this fraction of the total "
             "image area (default: 0.06 = 6%%). A has-text-2 calibration run found "
             "the largest true positive at 5.76%%. Use 0 to disable. "
             "The mask PNG is still written so the result can be inspected."
    )

    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Always produce output, even if no text is detected "
             "(copies original to output directory unchanged)"
    )
    
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (thin wrapper around create_parser)."""
    return create_parser().parse_args()



if __name__ == "__main__":
    main() 
