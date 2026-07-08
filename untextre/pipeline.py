"""Consensus detection, masking, and inpainting pipeline helpers."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import (
    CLI_DEFAULT_CONFIDENCE,
    calculate_bbox_superset,
    load_image,
    pad_bbox_to_multiple,
    save_image,
    setup_logger,
)

logger = setup_logger(__name__)

MASK_MODE_REGIONAL = "regional"
MASK_MODE_LOCAL_SHAPE = "local-shape"
MASK_MODE_LOCAL_COLOR = "local-color"
MASK_MODE_BUDGETED_REGIONAL = "budgeted-regional"
MASK_MODE_CHOICES = (
    MASK_MODE_REGIONAL,
    MASK_MODE_LOCAL_SHAPE,
    MASK_MODE_LOCAL_COLOR,
)
INTERNAL_MASK_MODE_CHOICES = MASK_MODE_CHOICES + (MASK_MODE_BUDGETED_REGIONAL,)
DEFAULT_MASK_MODE = MASK_MODE_REGIONAL


def mask_mode_options(mask_mode: str) -> dict:
    """Translate a user-facing mask mode to pipeline keyword arguments."""
    if mask_mode not in INTERNAL_MASK_MODE_CHOICES:
        raise ValueError(f"Unknown mask mode: {mask_mode}")
    options = {
        "expand_bboxes": False,
        "use_grabcut": mask_mode == MASK_MODE_LOCAL_SHAPE,
        "use_grabcut_expand": mask_mode == MASK_MODE_REGIONAL,
    }
    if mask_mode == MASK_MODE_BUDGETED_REGIONAL:
        options["use_budgeted_expand"] = True
    return options


@dataclass
class PipelineResult:
    """In-memory result from the consensus text-removal pipeline."""

    image: np.ndarray
    mask: np.ndarray
    timings: dict
    consensus_boxes: List[Tuple[int, int, int, int]]


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
    use_budgeted_expand: bool = False,
    coverage_limit: float = 0.06,
    mask_config: Optional[dict] = None,
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
        use_budgeted_expand: If True, extend masks with color/GrabCut proposals
                             constrained by geometry cost from the detector bbox.
        coverage_limit: Fraction of total image pixels (0-1) above which inpainting
                        is skipped as implausible. A has-text-2 batch run found the
                        largest true positive at 5.76%, so 0.06 rejects larger
                        false positives while keeping observed true positives.

    Returns:
        ``(combined_mask, inpainted_image)`` — both same shape as *image*.
        If the coverage guardrail fires, ``inpainted_image`` is the original
        image unchanged and the mask is still returned for inspection.
    """
    from .find_text_colors import (
        color_guided_expand,
        find_mask_by_spatial_tf_idf,
        geometry_budgeted_expand,
    )
    from .inpaint import inpaint_image

    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    regions_processed = 0
    mask_config = mask_config or {}

    for i, bbox in enumerate(consensus_boxes, 1):
        logger.info(f"Processing region {i}/{len(consensus_boxes)} with g={g_value}: {bbox}")

        try:
            mask_result = find_mask_by_spatial_tf_idf(
                image, bbox, num_clusters=g_value, debug=True,
                target_color=target_color, use_grabcut=use_grabcut,
                return_cluster_data=use_grabcut_expand or use_budgeted_expand,
                fom_threshold=mask_config.get("fom_threshold", 0.30),
                cc_guard=mask_config.get("cc_guard", 0.85),
                cleanup_close_px=mask_config.get("cleanup_close_px", 11),
                cleanup_dilate_px=mask_config.get("cleanup_dilate_px", 13),
            )
            if use_grabcut_expand or use_budgeted_expand:
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
                        expand_factor=mask_config.get("expand_factor", 2.0),
                        min_cc_px=mask_config.get("min_cc_px", 10),
                        color_radius_multiplier=mask_config.get("color_radius_multiplier", 1.0),
                        bg_radius_multiplier=mask_config.get("bg_radius_multiplier", 1.0),
                        zero_bbox_cluster_ids=cluster_data.get("zero_bbox_cluster_ids", ()),
                        debug=True,
                    )
                elif use_budgeted_expand:
                    full_mask = geometry_budgeted_expand(
                        image, bbox, full_mask,
                        cluster_data["centers"],
                        cluster_data["top_id"],
                        cluster_data["bot_id"],
                        cluster_data["color_radius"],
                        cluster_data["bg_radius"],
                        expand_factor=mask_config.get("expand_factor", 2.0),
                        inside_rejection_credit=mask_config.get("inside_rejection_credit", 1.0),
                        max_budget_fraction=mask_config.get("max_budget_fraction", 0.50),
                        long_weight=mask_config.get("long_weight", 1.0),
                        short_weight=mask_config.get("short_weight", 4.0),
                        short_axis_power=mask_config.get("short_axis_power", 1.5),
                        connectivity_dilation_px=mask_config.get("connectivity_dilation_px", 3),
                        min_cc_px=mask_config.get("min_cc_px", 10),
                        color_radius_multiplier=mask_config.get("color_radius_multiplier", 1.0),
                        bg_radius_multiplier=mask_config.get("bg_radius_multiplier", 1.0),
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

def initialize_consensus_models(device: str = "cuda") -> None:
    """Preload detection models and try to initialize LaMa for reuse."""
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
    
    logger.info("Model preload complete")

def process_image_array(
    image: np.ndarray,
    *,
    image_name: str = "<memory>",
    target_color: Optional[tuple] = None,
    method: str = "lama",
    mask: Optional[np.ndarray] = None,
    confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
    granularity: Optional[int] = None,
    forced_bbox: Optional[tuple] = None,
    color_sensitivity: int = 3,
    expand_bboxes: bool = True,
    auto_retry: bool = True,
    use_grabcut: bool = False,
    use_grabcut_expand: bool = False,
    use_budgeted_expand: bool = False,
    coverage_limit: float = 0.06,
    mask_config: Optional[dict] = None,
) -> PipelineResult:
    """Process an in-memory image through the consensus-based text-removal pipeline.

    For CLI (automated detection): Uses g=4 by default with auto-retry at g=8.
    For Web UI (user-controlled): User specifies granularity, no auto-retry.

    Args:
        image: Input image as HxWx3 BGR uint8.
        image_name: Name used for logs and timing reports.
        target_color: Optional target color as BGR tuple - will be used for color enhancement failover
        method: Inpainting method to use ("lama" or "telea")
        mask: Optional in-memory mask to use instead of auto-generation
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
        use_budgeted_expand: Whether to extend masks using the experimental
                           geometry-budgeted regional expansion (default: False).
        coverage_limit: Passed through to _generate_masks_and_inpaint. Skips
                       inpainting when mask exceeds this fraction of image area.

    Returns:
        PipelineResult containing cleaned image, mask, timings, and consensus boxes.
    """
    from .preprocessor import preprocess_image
    from .consensus import run_consensus_detection
    from .metrics import expand_bbox_along_long_axis, needs_retry

    logger.info(f"Processing image array: {image_name}")
    
    # Initialize timing dictionary
    timings = {
        'image_name': image_name,
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
    
    # 1. Preprocess image
    load_start = time.time()
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
                
                # Try gray enhancement with the configured sensitivity.
                consensus_boxes = _try_color_enhanced_detection(
                    image,
                    confidence_threshold,
                    "#808080",
                    sensitivity=color_sensitivity,
                )
                
                if consensus_boxes:
                    timings['failover_type'] = 'gray_enhancement'
                else:
                    # Try white enhancement with the configured sensitivity.
                    consensus_boxes = _try_color_enhanced_detection(
                        image,
                        confidence_threshold,
                        "#FFFFFF",
                        sensitivity=color_sensitivity,
                    )
                    
                    if consensus_boxes:
                        timings['failover_type'] = 'white_enhancement'
                    else:
                        logger.warning(f"No text detected in {image_name} after all failovers - skipping")
                        timings['mask_found'] = False
                        timings['detection_time'] = time.time() - detection_start
                        timings['total_time'] = time.time() - start_time
                        timings['skipped'] = True
                        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        return PipelineResult(image.copy(), empty_mask, timings, [])
    
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
    if mask is not None:
        from .inpaint import inpaint_image

        mask_start = time.time()
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
            use_budgeted_expand=use_budgeted_expand,
            coverage_limit=coverage_limit,
            mask_config=mask_config,
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
                    use_budgeted_expand=use_budgeted_expand,
                    coverage_limit=coverage_limit,
                    mask_config=mask_config,
                )
                timings['retried_with_g8'] = True
                timings['color_time'] += time.time() - retry_start
                logger.info("Retry with g=8 complete")

    timings['total_time'] = time.time() - start_time
    return PipelineResult(result, mask, timings, consensus_boxes)


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
    use_budgeted_expand: bool = False,
    coverage_limit: float = 0.06,
    mask_config: Optional[dict] = None,
) -> Optional[dict]:
    """Process a single image file and save the cleaned output."""
    logger.info(f"Loading image: {image_path.name}")

    load_start = time.time()
    image = load_image(image_path)
    extra_load_time = time.time() - load_start

    loaded_mask = None
    if maskfile:
        logger.info(f"Loading mask from file: {maskfile}")
        mask_path = Path(maskfile)
        if not mask_path.exists():
            raise ValueError(f"Mask file not found: {maskfile}")
        loaded_mask = load_image(mask_path)

    pipeline_result = process_image_array(
        image,
        image_name=image_path.name,
        target_color=target_color,
        method=method,
        mask=loaded_mask,
        confidence_threshold=confidence_threshold,
        granularity=granularity,
        forced_bbox=forced_bbox,
        color_sensitivity=color_sensitivity,
        expand_bboxes=expand_bboxes,
        auto_retry=auto_retry,
        use_grabcut=use_grabcut,
        use_grabcut_expand=use_grabcut_expand,
        use_budgeted_expand=use_budgeted_expand,
        coverage_limit=coverage_limit,
        mask_config=mask_config,
    )
    result = pipeline_result.image
    mask = pipeline_result.mask
    timings = pipeline_result.timings
    timings['load_time'] += extra_load_time

    if timings.get('skipped'):
        return timings

    # Save results
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path, source_path=image_path)
    logger.info(f"Saved result to: {output_path.name}")
    
    # Optionally save mask for debugging
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to: {mask_path.name}")
    
    return timings
