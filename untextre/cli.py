"""Command-line interface for untextre.

This module provides the main CLI entry point that orchestrates the complete
text watermark removal pipeline using consensus detection from multiple detectors.
"""

import argparse
import cv2
import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Lightweight imports only — heavy ML modules (.detector, .consensus,
# .inpaint, .find_text_colors, .preprocessor, .metrics) are imported
# lazily inside the functions that need them so that `--help` and
# watermark-only runs don't pay the TF/PyTorch startup cost.
from .utils import (
    get_image_files, load_image, save_image, setup_logger, pad_bbox_to_multiple,
    CLI_DEFAULT_CONFIDENCE, calculate_bbox_superset, configure_logging,
)
logger = setup_logger(__name__)

# Compatibility re-exports; helpers live in focused modules.
from .known_mask import process_with_known_mask
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



def _translate_rotated_bbox_to_original(
    rotated_bbox: Tuple[int, int, int, int],
    original_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Map bbox from cv2.ROTATE_90_CLOCKWISE image back to original image coordinates."""
    x_rot, y_rot, w_rot, h_rot = rotated_bbox
    original_h, _original_w = original_shape
    # Clockwise rotation maps original (x, y) -> rotated (H - 1 - y, x).
    return (y_rot, original_h - x_rot - w_rot, h_rot, w_rot)



def _apply_color_enhancement(image: np.ndarray, target_hex: str, sensitivity: int = 3) -> np.ndarray:
    """Apply color-based enhancement to make subtle watermarks more visible.
    
    Args:
        image: Original image (H×W×3 BGR uint8)
        target_hex: Target color in hex format (e.g., "#808080", "#FFFFFF")
        sensitivity: Plus-or-minus range around target color (default: 3)
        
    Returns:
        Enhanced image with specified color range converted to black
    """
    # Work with a copy to avoid modifying original
    enhanced = image.copy()
    
    # Convert hex color to BGR values
    if not target_hex.startswith('#') or len(target_hex) != 7:
        raise ValueError(f"Invalid hex color format: {target_hex}. Use format #RRGGBB")
    
    try:
        # Parse hex color (#RRGGBB -> RGB -> BGR)
        hex_value = target_hex[1:]  # Remove '#'
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        target_bgr = np.array([b, g, r], dtype=np.uint8)  # Convert RGB to BGR
        
    except ValueError:
        raise ValueError(f"Invalid hex color format: {target_hex}. Use format #RRGGBB")
    
    # Calculate bounds with sensitivity.
    # Arithmetic is done in int16 to avoid uint8 overflow (e.g. 255 + 3 = 258 → 2).
    target_i16 = target_bgr.astype(np.int16)
    lower_bound = np.maximum(target_i16 - sensitivity, 0).astype(np.uint8)
    upper_bound = np.minimum(target_i16 + sensitivity, 255).astype(np.uint8)
    
    # Convert back to hex for logging
    lower_hex = f"#{lower_bound[2]:02X}{lower_bound[1]:02X}{lower_bound[0]:02X}"
    upper_hex = f"#{upper_bound[2]:02X}{upper_bound[1]:02X}{upper_bound[0]:02X}"
    
    logger.info(f"Applying color enhancement: converting {lower_hex}-{upper_hex} to black (target: {target_hex}, sensitivity: +/-{sensitivity})")
    
    # Create mask for pixels in the target color range
    mask = cv2.inRange(enhanced, lower_bound, upper_bound)
    
    # Count affected pixels
    affected_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (affected_pixels / total_pixels) * 100
    
    logger.info(f"Color enhancement affected {affected_pixels:,} pixels ({percentage:.2f}% of image)")
    
    # Set masked pixels to black
    enhanced[mask > 0] = [0, 0, 0]  # BGR black
    
    return enhanced

def _try_color_enhanced_detection(original_image: np.ndarray, confidence_threshold: float, target_hex: str, sensitivity: int = 3) -> List[Tuple[int, int, int, int]]:
    """Try consensus detection with color enhancement.
    
    Args:
        original_image: Original unprocessed image
        confidence_threshold: Confidence threshold for detection
        target_hex: Target color in hex format (e.g., "#808080", "#FFFFFF")
        sensitivity: Plus-or-minus range around target color (default: 3)
        
    Returns:
        List of consensus bounding boxes, or empty list if none found
    """
    from .preprocessor import preprocess_image
    from .consensus import run_consensus_detection

    logger.info(f"Trying color enhancement for {target_hex} (+/-{sensitivity})...")
    
    # Apply color enhancement to original image
    enhanced_image = _apply_color_enhancement(original_image, target_hex, sensitivity)
    
    # Re-preprocess the enhanced image
    enhanced_preprocessed = preprocess_image(enhanced_image)
    if enhanced_preprocessed is None:
        logger.warning(f"Failed to preprocess color-enhanced image (target: {target_hex})")
        return []
    
    # Run consensus detection on enhanced image
    consensus_boxes = run_consensus_detection(enhanced_preprocessed, confidence_threshold)
    
    if consensus_boxes:
        logger.info(f"Color enhancement ({target_hex}) found {len(consensus_boxes)} consensus regions")
    else:
        logger.info(f"Color enhancement ({target_hex}) found no consensus regions")
    
    return consensus_boxes


def _generate_masks_and_inpaint(
    image: np.ndarray,
    consensus_boxes: List[Tuple[int, int, int, int]],
    g_value: int,
    method: str,
    target_color: Optional[tuple] = None,
    use_grabcut: bool = False,
    use_grabcut_expand: bool = False,
    coverage_limit: float = 0.06,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run spatial TF-IDF masking and inpainting for each consensus region.

    This is the core mask-generation + inpainting step, factored out of
    ``process_single_image`` so it can be called at different granularities
    (e.g. g=4, then retry at g=8) and tested independently.

    Args:
        image: Original image (H×W×3 BGR uint8).
        consensus_boxes: List of ``(x, y, w, h)`` bounding boxes to process.
        g_value: Number of K-means colour clusters for spatial TF-IDF.
        method: Inpainting method (``"lama"`` or ``"telea"``).
        target_color: Optional BGR tuple for forced colour-cluster inclusion.
        use_grabcut: If True, refine FOM masks with GrabCut before morphology.
        use_grabcut_expand: If True, extend each region mask using global color
                            matching and GrabCut seeded by the highest- and
                            lowest-FOM clusters found in the bbox analysis.
        coverage_limit: Fraction of total image pixels (0-1) above which inpainting
                        is skipped as implausible. A has-text-2 batch run found the
                        largest true positive at 5.76%, so 0.06 rejects larger
                        false positives while keeping observed true positives.

    Returns:
        ``(combined_mask, inpainted_image)`` — both same shape as *image*.
        If the coverage guardrail fires, ``inpainted_image`` is the original
        image unchanged and the mask is still returned for inspection.
    """
    from .find_text_colors import find_mask_by_spatial_tf_idf, color_guided_expand
    from .inpaint import inpaint_image

    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    regions_processed = 0

    for i, bbox in enumerate(consensus_boxes, 1):
        logger.info(f"Processing region {i}/{len(consensus_boxes)} with g={g_value}: {bbox}")

        try:
            mask_result = find_mask_by_spatial_tf_idf(
                image, bbox, num_clusters=g_value, debug=True,
                target_color=target_color, use_grabcut=use_grabcut,
                return_cluster_data=use_grabcut_expand,
            )
            if use_grabcut_expand:
                region_mask, cluster_data = mask_result
            else:
                region_mask = mask_result

            if np.sum(region_mask == 255) > 0:
                full_mask = np.zeros((h, w), dtype=np.uint8)
                x, y, box_w, box_h = bbox
                actual_h, actual_w = region_mask.shape[:2]
                full_mask[y:y + actual_h, x:x + actual_w] = region_mask

                if use_grabcut_expand:
                    full_mask = color_guided_expand(
                        image, bbox, full_mask,
                        cluster_data["centers"],
                        cluster_data["top_id"],
                        cluster_data["bot_id"],
                        cluster_data["color_radius"],
                        cluster_data["bg_radius"],
                        debug=True,
                    )

                combined_mask = cv2.bitwise_or(combined_mask, full_mask)
                regions_processed += 1

                mask_pixels = np.sum(full_mask == 255)
                logger.info(f"Region {i}: Generated {mask_pixels} mask pixels")
                full_mask = None
            else:
                logger.warning(f"Region {i}: Generated empty mask")

        except Exception as e:
            logger.error(f"Error processing region {i}: {e}")
            continue

    logger.info(f"Processed {regions_processed}/{len(consensus_boxes)} regions with g={g_value}")

    inpaint_region = calculate_bbox_superset(consensus_boxes, image.shape[:2])

    total_image_pixels = h * w
    mask_pixel_count = int(np.sum(combined_mask > 0))
    image_coverage_fraction = mask_pixel_count / total_image_pixels if total_image_pixels > 0 else 0.0
    if coverage_limit > 0 and image_coverage_fraction > coverage_limit:
        logger.warning(
            f"Coverage guardrail: mask covers {image_coverage_fraction * 100:.1f}% of image "
            f"(limit {coverage_limit * 100:.0f}%). Skipping inpaint - mask saved for inspection."
        )
        return combined_mask, image.copy()

    inpainted = inpaint_image(image, combined_mask, bbox=inpaint_region, method=method)
    return combined_mask, inpainted






def initialize_consensus_models(device: str = "cuda", **_deprecated_kwargs) -> None:
    """Initialize all models (detection and inpainting) to avoid per-image startup costs.
    """
    from .consensus import initialize_consensus_models as init_consensus_models_base

    logger.info("Pre-loading all detection and inpainting models...")
    
    # Initialize consensus detection models (EAST, DocTR, EasyOCR)
    init_consensus_models_base()
    
    # Initialize LaMa inpainting model
    try:
        from .inpaint import initialize_lama_model
        if initialize_lama_model(device=device):
            logger.info("[OK] LaMa model loaded")
        else:
            logger.warning("[FAIL] LaMa model failed to initialize (auto-retry will be used if needed)")
    except Exception as e:
        logger.error(f"Failed to load LaMa: {e}")
    
    logger.info("Model initialization complete - all models cached for reuse")

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
        watermark_templates: List[Tuple[str, np.ndarray]] = []
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

def process_single_image(
    image_path: Path, 
    output_dir: Path, 
    target_color: Optional[tuple] = None,
    keep_masks: bool = False,
    method: str = "lama",
    maskfile: Optional[str] = None,
    confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
    granularity: Optional[int] = None,
    forced_bbox: Optional[tuple] = None,
    color_sensitivity: int = 3,
    expand_bboxes: bool = True,
    auto_retry: bool = True,
    use_grabcut: bool = False,
    use_grabcut_expand: bool = False,
    coverage_limit: float = 0.06,
) -> Optional[dict]:
    """Process a single image through the consensus-based spatial TF-IDF pipeline.

    For CLI (automated detection): Uses g=4 by default with auto-retry at g=8.
    For Web UI (user-controlled): User specifies granularity, no auto-retry.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        target_color: Optional target color as BGR tuple - will be used for color enhancement failover
        keep_masks: Whether to save debug masks
        method: Inpainting method to use ("lama" or "telea")
        maskfile: Optional path to mask file to use instead of auto-generation
        confidence_threshold: Confidence threshold for consensus detection
            (see CLI_DEFAULT_CONFIDENCE / WEB_DEFAULT_CONFIDENCE in utils.py)
        granularity: Number of color clusters for spatial TF-IDF. If None, uses g=4 with
                    auto-retry at g=8. If specified, uses that value without retry.
        forced_bbox: Optional forced bounding box as (x, y, width, height) tuple.
                    When set, disables bbox expansion (user's selection is authoritative).
        color_sensitivity: Plus-or-minus range around target color (default: 3)
        expand_bboxes: Whether to expand detected bboxes along long axis (default: True).
                      Automatically disabled when forced_bbox is set.
        auto_retry: Whether to automatically retry with g=8 if g=4 fails (default: True).
                   Automatically disabled when granularity is explicitly specified.
        use_grabcut: Whether to refine FOM masks with GrabCut (default: False).
        use_grabcut_expand: Whether to extend masks using global color matching
                           and GrabCut seeded by the highest- and lowest-FOM
                           clusters from the bbox analysis (default: False).
        coverage_limit: Passed through to _generate_masks_and_inpaint. Skips
                       inpainting when mask exceeds this fraction of image area.

    Returns:
        Dictionary with timing details, or None if processing failed
    """
    from .preprocessor import preprocess_image
    from .consensus import run_consensus_detection
    from .metrics import expand_bbox_along_long_axis, needs_retry

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
        'consensus_boxes_count': 0,
        'total_bbox_area': 0,
        'failover_type': 'none',  # Track type of failover used
        'retried_with_g8': False,  # Track if g=8 retry was needed
        'bboxes_expanded': 0,  # Track how many bboxes were expanded
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
    
    # 2. Detect consensus regions or use forced bbox
    detection_start = time.time()
    if forced_bbox:
        h, w = image.shape[:2]
        logger.info(f"Using forced bounding box: {forced_bbox}")
        
        # Validate bbox is within image bounds
        if forced_bbox[0] + forced_bbox[2] > w or forced_bbox[1] + forced_bbox[3] > h:
            logger.warning(f"Forced bbox extends beyond image bounds ({w}x{h}), clipping...")
            clipped_bbox = (
                min(forced_bbox[0], w-1),
                min(forced_bbox[1], h-1), 
                min(forced_bbox[2], w - forced_bbox[0]),
                min(forced_bbox[3], h - forced_bbox[1])
            )
            processed_bbox = clipped_bbox
            logger.info(f"Clipped bbox: {clipped_bbox}")
        else:
            processed_bbox = forced_bbox
        
        # Ensure dimensions are divisible by 4 for neural network compatibility
        mod4_bbox = pad_bbox_to_multiple(processed_bbox, multiple=4, image_shape=(h, w))
        consensus_boxes = [mod4_bbox]
        logger.info(f"Final forced bbox (mod4): {mod4_bbox}")
    else:
        # If user specified a target color, try color enhancement FIRST
        if target_color is not None:
            # Convert BGR tuple to hex
            b, g, r = target_color
            target_hex = f"#{r:02X}{g:02X}{b:02X}"
            logger.info(f"User specified target color {target_hex} - trying color enhancement first...")
            
            consensus_boxes = _try_color_enhanced_detection(image, confidence_threshold, target_hex, sensitivity=color_sensitivity)
            
            if consensus_boxes:
                timings['failover_type'] = 'target_color'
                logger.info(f"Target color enhancement succeeded with {len(consensus_boxes)} consensus regions")
        else:
            consensus_boxes = []
        
        # If no target color specified OR target color enhancement failed, try normal consensus detection
        if not consensus_boxes:
            logger.info(f"Running consensus detection with confidence threshold {confidence_threshold}...")
            consensus_boxes = run_consensus_detection(preprocessed, confidence_threshold)
            
            if consensus_boxes:
                logger.info(f"Normal consensus detection found {len(consensus_boxes)} regions")
        
        # Continue with failover sequence if still no consensus
        if not consensus_boxes:
            logger.warning("No consensus regions detected, trying rotation failover...")
            
            # Rotate image 90 degrees clockwise and try detection again
            h, w = preprocessed.shape[:2]
            rotated_image = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)
            logger.info("Rotated image 90 degrees clockwise, running consensus detection again...")
            
            rotated_consensus_boxes = run_consensus_detection(rotated_image, confidence_threshold)
            
            if rotated_consensus_boxes:
                timings['failover_type'] = 'rotation'
                logger.info(f"Found {len(rotated_consensus_boxes)} consensus regions in rotated image")
                
                # Translate consensus boxes back to original coordinates.
                # cv2.ROTATE_90_CLOCKWISE maps original (x, y) -> rotated (H - 1 - y, x).
                consensus_boxes = []
                for bbox in rotated_consensus_boxes:
                    translated_bbox = _translate_rotated_bbox_to_original(bbox, (h, w))
                    consensus_boxes.append(translated_bbox)
                    logger.info(f"Translated rotated bbox {bbox} -> {translated_bbox}")
                
                logger.info(f"Successfully translated {len(consensus_boxes)} consensus regions back to original orientation")
            else:
                logger.warning("No consensus regions detected after rotation failover, trying generic color enhancements...")
                
                # Try gray enhancement (#808080 with ±3 sensitivity gives #7D7D7D-#838383)
                consensus_boxes = _try_color_enhanced_detection(image, confidence_threshold, "#808080", sensitivity=3)
                
                if consensus_boxes:
                    timings['failover_type'] = 'gray_enhancement'
                else:
                    # Try white enhancement (#FFFFFF with ±3 sensitivity gives #FCFCFC-#FFFFFF)
                    consensus_boxes = _try_color_enhanced_detection(image, confidence_threshold, "#FFFFFF", sensitivity=3)
                    
                    if consensus_boxes:
                        timings['failover_type'] = 'white_enhancement'
                    else:
                        logger.warning(
                            f"No text detected in {image_path.name} after all failovers - skipping"
                        )
                        timings['mask_found'] = False
                        timings['detection_time'] = time.time() - detection_start
                        timings['total_time'] = time.time() - start_time
                        timings['skipped'] = True
                        return timings
    
    timings['detection_time'] = time.time() - detection_start
    
    # 2b. Expand bboxes along long axis (CLI only, not for forced_bbox)
    if expand_bboxes and consensus_boxes and not forced_bbox:
        expanded_boxes = []
        for bbox in consensus_boxes:
            expanded = expand_bbox_along_long_axis(image, bbox)
            if expanded != bbox:
                timings['bboxes_expanded'] += 1
            expanded_boxes.append(expanded)
        consensus_boxes = expanded_boxes
    
    timings['consensus_boxes_count'] = len(consensus_boxes)
    timings['total_bbox_area'] = sum(bbox[2] * bbox[3] for bbox in consensus_boxes)
    
    # 3. Generate or load mask
    if maskfile:
        from .inpaint import inpaint_image

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

        # Inpaint using the loaded mask
        inpaint_start = time.time()
        inpaint_region = calculate_bbox_superset(consensus_boxes, image.shape[:2])
        result = inpaint_image(image, mask, bbox=inpaint_region, method=method)
        timings['inpaint_time'] = time.time() - inpaint_start
    else:
        # Process each consensus box with spatial TF-IDF and combine masks
        color_start = time.time()
        h, w = image.shape[:2]
        
        # Determine granularity strategy:
        # - If user specified granularity: use that value, no retry
        # - If granularity is None (CLI default): use g=4 with auto-retry at g=8
        user_specified_granularity = granularity is not None
        initial_g = granularity if user_specified_granularity else 4
        
        # First pass with initial granularity
        mask, result = _generate_masks_and_inpaint(
            image, consensus_boxes, initial_g, method, target_color,
            use_grabcut=use_grabcut,
            use_grabcut_expand=use_grabcut_expand,
            coverage_limit=coverage_limit,
        )

        timings['color_time'] = time.time() - color_start
        timings['mask_time'] = 0  # Included in color_time for this flow

        # Check if retry needed (CLI auto-retry feature)
        # Only retry if: auto_retry enabled AND granularity not user-specified AND not forced_bbox
        should_check_retry = auto_retry and not user_specified_granularity and not forced_bbox
        if should_check_retry:
            # Check inpainted regions for text remnants
            retry_needed = False
            for bbox in consensus_boxes:
                x, y, bw, bh = bbox
                # Clip to image bounds
                x2, y2 = min(x + bw, w), min(y + bh, h)
                region = result[y:y2, x:x2]
                if region.size > 0 and needs_retry(region):
                    retry_needed = True
                    break

            if retry_needed:
                logger.info("Text remnants detected, retrying with granularity=8...")
                retry_start = time.time()
                mask, result = _generate_masks_and_inpaint(
                    image, consensus_boxes, 8, method, target_color,
                    use_grabcut=use_grabcut,
                    use_grabcut_expand=use_grabcut_expand,
                    coverage_limit=coverage_limit,
                )
                timings['retried_with_g8'] = True
                timings['color_time'] += time.time() - retry_start
                logger.info("Retry with g=8 complete")
    
    # 4. Record inpaint time (already done in generate_masks_and_inpaint)
    inpaint_start = time.time()
    timings['inpaint_time'] = time.time() - inpaint_start  # Near-zero, actual time in color_time
    
    # Save results
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path, source_path=image_path)
    logger.info(f"Saved result to: {output_path.name}")
    
    # Optionally save mask for debugging
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to: {mask_path.name}")
    
    # Calculate total time and return timings
    timings['total_time'] = time.time() - start_time
    return timings

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
