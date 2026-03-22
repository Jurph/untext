"""Command-line interface for untextre.

This module provides the main CLI entry point that orchestrates the complete
text watermark removal pipeline using consensus detection from multiple detectors.
"""

import logging
import argparse
import cv2
import time
import sys
import statistics
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Lightweight imports only — heavy ML modules (.detector, .consensus,
# .inpaint, .find_text_colors, .preprocessor, .metrics) are imported
# lazily inside the functions that need them so that `--help` and
# watermark-only runs don't pay the TF/PyTorch startup cost.
from .utils import (
    get_image_files, load_image, save_image, setup_logger, pad_bbox_to_multiple,
    MODEL_CONFIDENCE_FLOOR, CLI_DEFAULT_CONFIDENCE,
)

logger = setup_logger(__name__)

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
    
    logger.info(f"Applying color enhancement: converting {lower_hex}-{upper_hex} to black (target: {target_hex}, sensitivity: ±{sensitivity})")
    
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

    logger.info(f"Trying color enhancement for {target_hex} (±{sensitivity})...")
    
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

    Returns:
        ``(combined_mask, inpainted_image)`` — both same shape as *image*.
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
            else:
                logger.warning(f"Region {i}: Generated empty mask")

        except Exception as e:
            logger.error(f"Error processing region {i}: {e}")
            continue

    logger.info(f"Processed {regions_processed}/{len(consensus_boxes)} regions with g={g_value}")

    # Determine inpaint region (single box or bounding superset)
    if len(consensus_boxes) == 1:
        inpaint_region = consensus_boxes[0]
    else:
        min_x = min(bbox[0] for bbox in consensus_boxes)
        min_y = min(bbox[1] for bbox in consensus_boxes)
        max_x = max(bbox[0] + bbox[2] for bbox in consensus_boxes)
        max_y = max(bbox[1] + bbox[3] for bbox in consensus_boxes)
        inpaint_region = (min_x, min_y, max_x - min_x, max_y - min_y)

    inpainted = inpaint_image(image, combined_mask, bbox=inpaint_region, method=method)
    return combined_mask, inpainted


def load_watermark_templates(
    path: Path,
) -> List[Tuple[str, np.ndarray]]:
    """Load RGBA watermark templates from a file or directory.

    Each template is an RGBA PNG where the alpha channel defines the mask.

    Args:
        path: Path to a single RGBA PNG or a directory containing them.

    Returns:
        List of (filename, rgba_array) tuples, sorted by filename.
        Returns an empty list if the path doesn't exist, is empty,
        or contains no valid RGBA PNGs.
    """
    templates: List[Tuple[str, np.ndarray]] = []

    if not path.exists():
        return templates

    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in (".png", ".tif", ".tiff")
        )
    else:
        return templates

    for candidate in candidates:
        rgba = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning(f"Could not read {candidate.name}, skipping")
            continue
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            logger.warning(f"{candidate.name} is not RGBA ({rgba.ndim}D, {rgba.shape[-1] if rgba.ndim == 3 else '?'}ch), skipping")
            continue
        templates.append((candidate.name, rgba))
        logger.debug(f"Loaded template: {candidate.name} ({rgba.shape[1]}×{rgba.shape[0]})")

    if templates:
        logger.info(f"Loaded {len(templates)} watermark template(s) from {path}")

    return templates


def find_known_mask_in_image(
    target_image: np.ndarray,
    known_mask_rgba: np.ndarray,
    min_matches: int = 6,
    dilation_pixels: int = 7,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Find a known watermark/logo in target image using ORB feature matching.
    
    Uses ORB (Oriented FAST and Rotated BRIEF) to detect keypoints and match
    the known watermark template to the target image. Fits a partial affine
    transform (scale + rotation + translation) for precise alignment.
    
    Out-of-bounds handling: If the matched location extends beyond image edges,
    the mask is automatically clipped to the image bounds (spillover ignored).
    
    Args:
        target_image: Target image to search (H×W×3 BGR)
        known_mask_rgba: Known watermark as RGBA image (H×W×4, alpha channel is mask)
        min_matches: Minimum number of good matches required (default: 6)
        dilation_pixels: Pixels to dilate the mask for safety margin (default: 7)
        
    Returns:
        Tuple of (warped_mask, bbox) where:
            - warped_mask: Binary mask (H×W) positioned on target image
            - bbox: Bounding box (x, y, w, h) of the detected region
        Returns None if no match found.
    """
    if known_mask_rgba.shape[2] != 4:
        raise ValueError("Known mask must be RGBA image (4 channels)")
    
    # Extract BGR and alpha from known mask
    # (cv2.imread with IMREAD_UNCHANGED reads PNGs as BGRA, not RGBA)
    known_bgr = known_mask_rgba[:, :, :3]
    known_alpha = known_mask_rgba[:, :, 3]
    
    # Convert to grayscale for ORB — both are BGR from cv2
    known_gray = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(known_gray, None)
    kp2, des2 = orb.detectAndCompute(target_gray, None)
    
    if des1 is None or des2 is None:
        logger.warning("Could not compute ORB descriptors")
        return None
    
    if len(kp1) < min_matches or len(kp2) < min_matches:
        logger.warning(f"Not enough keypoints: known={len(kp1)}, target={len(kp2)}")
        return None
    
    # Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    logger.info(f"ORB matching: {len(good_matches)} good matches (need {min_matches})")
    
    if len(good_matches) < min_matches:
        logger.warning(f"Not enough good matches: {len(good_matches)} < {min_matches}")
        return None
    
    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Full affine transform via RANSAC (6 DOF: non-uniform scale, rotation,
    # shear, translation).  Handles mild stretching / skewing that the
    # partial-affine (4 DOF) model cannot.
    M, inlier_mask = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    
    if M is None:
        logger.warning("Could not compute affine transform")
        return None
    
    # Count inliers
    inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
    
    # Decompose the 2x2 linear submatrix via SVD to get independent
    # scale factors and detect reflection / excessive shear.
    #   M = [[a, b, tx],
    #        [c, d, ty]]
    # SVD of [[a,b],[c,d]] gives singular values = scale along each axis.
    linear = M[:2, :2]
    U, S, Vt = np.linalg.svd(linear)
    scale_major, scale_minor = float(S[0]), float(S[1])
    det = float(np.linalg.det(linear))
    angle = np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
    
    logger.info(
        f"Affine transform: {inliers} inliers, "
        f"scale=({scale_major:.3f}, {scale_minor:.3f}), "
        f"det={det:.3f}, rotation={angle:.1f}\u00b0"
    )
    
    if inliers < min_matches:
        logger.warning(f"Not enough inliers: {inliers} < {min_matches}")
        return None
    
    # ── Reject degenerate or implausible transforms ───────────────
    MIN_SCALE, MAX_SCALE = 0.05, 20.0
    MAX_STRETCH = 1.25  # largest allowed ratio between the two scale axes

    if scale_major < MIN_SCALE or scale_minor < MIN_SCALE:
        logger.warning(f"Scale too small ({scale_major:.3f}, {scale_minor:.3f})")
        return None
    if scale_major > MAX_SCALE or scale_minor > MAX_SCALE:
        logger.warning(f"Scale too large ({scale_major:.3f}, {scale_minor:.3f})")
        return None
    if det < 0:
        logger.warning(f"Reflection detected (det={det:.3f})")
        return None
    if scale_major / scale_minor > MAX_STRETCH:
        logger.warning(
            f"Excessive stretch ({scale_major / scale_minor:.1f}x) "
            f"between axes — likely spurious"
        )
        return None
    
    # Warp the alpha channel to target image coordinates using affine transform
    h_target, w_target = target_image.shape[:2]
    warped_mask = cv2.warpAffine(
        known_alpha,
        M,
        (w_target, h_target),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Threshold to binary mask (alpha > 127 means opaque)
    _, binary_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply dilation for safety margin
    if dilation_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1)
        )
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        logger.info(f"Applied {dilation_pixels}px dilation to mask")
    
    # Calculate bounding box of the mask
    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        logger.warning("Warped mask is empty")
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    bbox = (x, y, w, h)
    
    # Log the detection
    h_known, w_known = known_mask_rgba.shape[:2]
    scale_x = w / w_known if w_known > 0 else 1.0
    scale_y = h / h_known if h_known > 0 else 1.0
    logger.info(f"Known mask found at ({x}, {y}) size {w}×{h} (scale: {scale_x:.2f}x, {scale_y:.2f}x)")
    
    # Check for edge clipping
    at_edge = []
    if x == 0:
        at_edge.append("left")
    if y == 0:
        at_edge.append("top")
    if x + w >= w_target:
        at_edge.append("right")
    if y + h >= h_target:
        at_edge.append("bottom")
    if at_edge:
        logger.info(f"Mask touches image edge ({', '.join(at_edge)}) - spillover clipped")
    
    mask_pixels = np.sum(binary_mask > 0)
    logger.info(f"Mask covers {mask_pixels:,} pixels")
    
    return binary_mask, bbox


def try_watermark_cascade(
    image: np.ndarray,
    templates: List[Tuple[str, np.ndarray]],
    min_matches: int = 6,
    dilation_pixels: int = 7,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], str]]:
    """Try each watermark template against a loaded image. First match wins.

    This is the matching-only step with no file I/O.  The caller decides
    what to do with the result (inpaint, save, fall back to detection, etc.).

    Args:
        image: Target image (H×W×3 BGR), already loaded.
        templates: List of (filename, rgba_array) from load_watermark_templates().
        min_matches: Minimum ORB matches for a valid detection.
        dilation_pixels: Pixels to dilate the matched mask.

    Returns:
        Tuple of (mask, bbox, template_name) on first match, or None.
    """
    for tmpl_name, tmpl_rgba in templates:
        logger.info(f"Trying template: {tmpl_name}")
        result = find_known_mask_in_image(
            image, tmpl_rgba,
            min_matches=min_matches,
            dilation_pixels=dilation_pixels,
        )
        if result is not None:
            mask, bbox = result
            logger.info(f"Matched template: {tmpl_name}")
            return mask, bbox, tmpl_name

    logger.info(f"No template matched (tried {len(templates)})")
    return None


def process_with_known_mask(
    image_path: Path,
    output_dir: Path,
    known_mask_rgba: np.ndarray,
    keep_masks: bool = False,
    method: str = "lama",
    min_matches: int = 6,
    dilation_pixels: int = 7,
) -> Optional[dict]:
    """Process a single image using known-mask ORB feature matching.
    
    This bypasses consensus detection entirely and uses ORB to find a known
    watermark/logo template in the image, then inpaints it.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        known_mask_rgba: Known watermark as RGBA image (alpha channel is mask)
        keep_masks: Whether to save debug masks
        method: Inpainting method ("lama" or "telea")
        min_matches: Minimum ORB matches required
        dilation_pixels: Pixels to dilate mask for safety margin
        
    Returns:
        Dictionary with timing details, or None if processing failed
    """
    start_time = time.time()
    timings = {
        'image': image_path.name,
        'load_time': 0,
        'orb_time': 0,
        'inpaint_time': 0,
        'total_time': 0,
        'mask_found': False,
    }
    
    # Load image
    load_start = time.time()
    image = load_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    timings['load_time'] = time.time() - load_start
    logger.info(f"Loaded image: {image_path.name} ({image.shape[1]}×{image.shape[0]})")
    
    # Find known mask using ORB
    orb_start = time.time()
    match_result = find_known_mask_in_image(
        image,
        known_mask_rgba,
        min_matches=min_matches,
        dilation_pixels=dilation_pixels,
    )
    timings['orb_time'] = time.time() - orb_start
    
    if match_result is None:
        logger.warning(f"Known mask not found in {image_path.name} - saving original")
        # Save original image unchanged
        output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
        save_image(image, output_path)
        timings['total_time'] = time.time() - start_time
        return timings
    
    mask, bbox = match_result
    timings['mask_found'] = True
    
    # Save mask if requested
    if keep_masks:
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        save_image(mask, mask_path)
        logger.info(f"Saved mask to {mask_path}")
    
    # Inpaint
    from .inpaint import inpaint_image

    inpaint_start = time.time()
    result = inpaint_image(image, mask, bbox=bbox, method=method)
    timings['inpaint_time'] = time.time() - inpaint_start
    
    # Save result
    output_path = output_dir / f"{image_path.stem}_clean{image_path.suffix}"
    save_image(result, output_path)
    logger.info(f"Saved cleaned image to {output_path}")
    
    timings['total_time'] = time.time() - start_time
    return timings


def initialize_consensus_models(confidence_threshold: float = MODEL_CONFIDENCE_FLOOR, device: str = "cuda") -> None:
    """Initialize all models (detection and inpainting) to avoid per-image startup costs.
    
    Note: confidence_threshold is deprecated and ignored by the underlying
    consensus initializer.  Default is MODEL_CONFIDENCE_FLOOR to match the
    value actually used internally.  See utils.py for the full explanation.
    """
    from .consensus import initialize_consensus_models as init_consensus_models_base

    logger.info("Pre-loading all detection and inpainting models...")
    
    # Initialize consensus detection models (EAST, DocTR, EasyOCR)
    init_consensus_models_base(confidence_threshold)
    
    # Initialize LaMa inpainting model
    try:
        from .inpaint import initialize_lama_model
        if initialize_lama_model(device=device):
            logger.info("[OK] LaMa model loaded")
        else:
            logger.warning("✗ LaMa model failed to initialize (auto-retry will be used if needed)")
    except Exception as e:
        logger.error(f"Failed to load LaMa: {e}")
    
    logger.info("Model initialization complete - all models cached for reuse")

def main() -> None:
    """Main entry point for the consensus-based text watermark removal tool."""
    args = create_parser().parse_args()
    
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
        candidates = discover_watermark_candidates(image_files)

        if not candidates:
            logger.error(
                "No watermark candidates discovered. "
                "Try -K with a manually-identified template."
            )
            sys.exit(1)

        # Write candidates to output dir before processing
        output_path.mkdir(parents=True, exist_ok=True)
        watermark_templates = []
        for i, bgra in enumerate(candidates):
            suffix = "" if i == 0 else f"_{i + 1}"
            candidate_path = output_path / f"watermark_candidate{suffix}.png"
            if candidate_path.exists():
                logger.warning(f"Overwriting existing candidate: {candidate_path.name}")
            # Array is BGRA — cv2.imwrite saves channel 3 as alpha automatically
            cv2.imwrite(str(candidate_path), bgra)
            logger.info(f"Saved watermark candidate: {candidate_path.name}")
            watermark_templates.append((candidate_path.name, bgra))

        # Warn about stale candidates from prior runs
        written_names = {
            f"watermark_candidate{'' if j == 0 else f'_{j+1}'}.png"
            for j in range(len(candidates))
        }
        stale = [p for p in sorted(output_path.glob("watermark_candidate*.png"))
                 if p.name not in written_names]
        if stale:
            logger.warning(
                "Stale candidate file(s) from prior run still present: "
                + ", ".join(p.name for p in stale)
            )

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
        initialize_consensus_models(args.confidence_threshold, args.device)

    model_init_time = time.time() - model_init_start
    logger.info(f"Models ready in {model_init_time:.1f} seconds")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            timing_data = None

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
                        save_image(result, output_file)
                        logger.info(f"Saved cleaned image to {output_file}")
                        if args.keep_masks:
                            mask_file = output_path / f"{image_path.stem}_mask.png"
                            save_image(mask, mask_file)
                        timing_data = {
                            "image": image_path.name,
                            "matched_template": tmpl_name,
                            "mask_found": True,
                            "total_time": 0,  # not timed in this path yet
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
                )
            
            # ── Handle skipped images ──────────────────────────────────
            if timing_data and timing_data.get('skipped'):
                if args.force_output:
                    # Copy the original unchanged so every input has output
                    output_file = output_path / f"{image_path.stem}_clean{image_path.suffix}"
                    save_image(load_image(image_path), output_file)
                    logger.info(f"No text detected — copied original to {output_file}")
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
            logger.warning(f"Forced bbox extends beyond image bounds ({w}×{h}), clipping...")
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
            logger.info("Rotated image 90° clockwise, running consensus detection again...")
            
            rotated_consensus_boxes = run_consensus_detection(rotated_image, confidence_threshold)
            
            if rotated_consensus_boxes:
                timings['failover_type'] = 'rotation'
                logger.info(f"Found {len(rotated_consensus_boxes)} consensus regions in rotated image")
                
                # Translate consensus boxes back to original coordinate system
                # For 90° clockwise rotation then back: need to reverse the transformation
                # Forward: (x, y) -> (y, W - x - 1) where W is original width
                # Reverse: (x_rot, y_rot) -> (H - y_rot - 1, x_rot) where H is original height
                consensus_boxes = []
                for bbox in rotated_consensus_boxes:
                    x_rot, y_rot, w_rot, h_rot = bbox
                    # Transform coordinates back to original orientation
                    # For 90° clockwise rotation: point (x,y) -> (y, W-x-1)
                    # Reverse: point (x_rot, y_rot) -> (H-y_rot-1, x_rot) where H is original height
                    # Wait, let me think about this differently...
                    # If we rotate 90° clockwise then back, we need the inverse transformation
                    x_orig = y_rot
                    y_orig = h - x_rot - w_rot  # h is original height, w_rot is width of detected box
                    w_orig = h_rot  # dimensions swap back
                    h_orig = w_rot
                    
                    translated_bbox = (x_orig, y_orig, w_orig, h_orig)
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
                            f"No text detected in {image_path.name} after all failovers — skipping"
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
        if len(consensus_boxes) == 1:
            inpaint_region = consensus_boxes[0]
        else:
            min_x = min(bbox[0] for bbox in consensus_boxes)
            min_y = min(bbox[1] for bbox in consensus_boxes)
            max_x = max(bbox[0] + bbox[2] for bbox in consensus_boxes)
            max_y = max(bbox[1] + bbox[3] for bbox in consensus_boxes)
            inpaint_region = (min_x, min_y, max_x - min_x, max_y - min_y)
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
                )
                timings['retried_with_g8'] = True
                timings['color_time'] += time.time() - retry_start
                logger.info("Retry with g=8 complete")
    
    # 4. Record inpaint time (already done in generate_masks_and_inpaint)
    inpaint_start = time.time()
    timings['inpaint_time'] = time.time() - inpaint_start  # Near-zero, actual time in color_time
    
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
             "(e.g., 593,1013,105,39 selects a 105×39 region starting at top-left (593,1013))"
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
        "--force-output",
        action="store_true",
        help="Always produce output, even if no text is detected "
             "(copies original to output directory unchanged)"
    )
    
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (thin wrapper around create_parser)."""
    return create_parser().parse_args()


def _save_clean_timing_report(detailed_timings: list, total_time: float, avg_time: float, timing_file: Path, method: str, confidence_threshold: float, target_color: Optional[tuple], forced_bbox: Optional[tuple]) -> None:
    """Save a clean timing report to file without duplicate logging."""
    with open(timing_file, 'w') as f:
        f.write("=" * 74 + "\n")
        f.write("CONSENSUS DETECTION + SPATIAL TF-IDF TIMING REPORT\n")
        f.write("=" * 74 + "\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write("TF-IDF granularity: g=4 (auto-retry with g=8 if needed)\n")
        f.write(f"Inpainting method: {method}\n")
        if target_color:
            f.write(f"Target color (deprecated): {target_color}\n")
        if forced_bbox:
            f.write(f"Forced bbox: {forced_bbox}\n")
        
        # Count retries and expansions
        retried_count = sum(1 for t in detailed_timings if t.get('retried_with_g8', False))
        expanded_count = sum(t.get('bboxes_expanded', 0) for t in detailed_timings)
        if retried_count > 0:
            f.write(f"Images retried with g=8: {retried_count}\n")
        if expanded_count > 0:
            f.write(f"Total bboxes expanded: {expanded_count}\n")
        
        f.write("\nColumns: MP=Megapixels, Det=Detection, Msk=Mask, Inp=Inpaint, Tot=Total, Failover=R/T/G/W/B (Rotation/Target/Gray/White/Baseline)\n")
        
        # Header with wider format
        f.write(f"{'Image Name':<25} {'MP':>4} {'Det':>4} {'TF-IDF':>6} {'Msk':>4} {'Inp':>5} {'Tot':>5} {'Boxes':>5} {'Fail':>4}\n")
        f.write("-" * 74 + "\n")
        
        # Individual rows (support both consensus path keys and template-match path keys)
        for timing in detailed_timings:
            name = (timing.get('image_name') or timing.get('image') or '?')[:25]

            # Map failover type to marker
            failover_type = timing.get('failover_type', 'none')
            if timing.get('matched_template'):
                failover_marker = "Tpl"  # Template match
            elif failover_type == 'rotation':
                failover_marker = "R"
            elif failover_type == 'target_color':
                failover_marker = "T"
            elif failover_type == 'gray_enhancement':
                failover_marker = "G"
            elif failover_type == 'white_enhancement':
                failover_marker = "W"
            elif failover_type == 'watermark':
                failover_marker = "B"  # Baseline watermark regions
            else:
                failover_marker = ""

            # Handle None/missing values (template-match and skipped entries lack some keys)
            color_time = timing.get('color_time')
            mask_time = timing.get('mask_time')
            inpaint_time = timing.get('inpaint_time')
            color_time_str = "N/A" if color_time is None else f"{color_time:>6.1f}"
            mask_time_str = "N/A" if mask_time is None else f"{mask_time:>4.1f}"
            inpaint_time_str = "N/A" if inpaint_time is None else f"{inpaint_time:>5.1f}"

            image_mp = timing.get('image_mp', 0)
            detection_time = timing.get('detection_time', 0)
            total_time = timing.get('total_time', 0)
            consensus_boxes_count = timing.get('consensus_boxes_count', 0)

            row = (f"{name:<25} "
                   f"{image_mp:>4.1f} "
                   f"{detection_time:>4.1f} "
                   f"{color_time_str:>6} "
                   f"{mask_time_str:>4} "
                   f"{inpaint_time_str:>5} "
                   f"{total_time:>5.1f} "
                   f"{consensus_boxes_count:>5d} "
                   f"{failover_marker:>4}\n")
            f.write(row)
        
        if len(detailed_timings) > 1:
            f.write("-" * 74 + "\n")

            # Statistics (use .get() so template-match/skipped entries are included)
            det_times = [t.get('detection_time', 0) for t in detailed_timings]
            col_times = [t['color_time'] for t in detailed_timings if t.get('color_time') is not None]
            msk_times = [t['mask_time'] for t in detailed_timings if t.get('mask_time') is not None]
            inp_times = [t['inpaint_time'] for t in detailed_timings if t.get('inpaint_time') is not None]
            tot_times = [t.get('total_time', 0) for t in detailed_timings]
            box_counts = [t.get('consensus_boxes_count', 0) for t in detailed_timings]
            
            # Handle cases where all values might be None
            col_median = statistics.median(col_times) if col_times else 0.0
            col_mean = statistics.mean(col_times) if col_times else 0.0
            msk_median = statistics.median(msk_times) if msk_times else 0.0
            msk_mean = statistics.mean(msk_times) if msk_times else 0.0
            inp_median = statistics.median(inp_times) if inp_times else 0.0
            inp_mean = statistics.mean(inp_times) if inp_times else 0.0
            
            f.write(f"{'MEDIAN':<25} {'':>4} {statistics.median(det_times):>4.1f} "
                   f"{col_median:>6.1f} {msk_median:>4.1f} "
                   f"{inp_median:>5.1f} {statistics.median(tot_times):>5.1f} "
                   f"{statistics.median(box_counts):>5.1f}\n")
            
            f.write(f"{'AVERAGE':<25} {'':>4} {statistics.mean(det_times):>4.1f} "
                   f"{col_mean:>6.1f} {msk_mean:>4.1f} "
                   f"{inp_mean:>5.1f} {statistics.mean(tot_times):>5.1f} "
                   f"{statistics.mean(box_counts):>5.1f}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"Total processing time: {total_time:.1f} seconds\n")
        f.write(f"Average time per image: {avg_time:.1f} seconds\n")
        f.write(f"Images processed: {len(detailed_timings)}\n")
        
        # Consensus statistics
        total_boxes = sum(t.get('consensus_boxes_count', 0) for t in detailed_timings)
        images_with_consensus = sum(1 for t in detailed_timings if t.get('consensus_boxes_count', 0) > 0)
        f.write(f"Total consensus boxes: {total_boxes}\n")
        f.write(f"Images with consensus: {images_with_consensus}/{len(detailed_timings)} ({100*images_with_consensus/len(detailed_timings):.1f}%)\n")
        
        # Failover statistics
        failover_counts = {}
        template_match_count = 0
        for timing in detailed_timings:
            if timing.get('matched_template'):
                template_match_count += 1
            else:
                failover_type = timing.get('failover_type', 'none')
                failover_counts[failover_type] = failover_counts.get(failover_type, 0) + 1

        f.write("\nFailover usage:\n")
        if template_match_count > 0:
            f.write(f"  Template match: {template_match_count}\n")
        f.write(f"  Normal consensus: {failover_counts.get('none', 0)}\n")
        f.write(f"  Rotation failover: {failover_counts.get('rotation', 0)}\n")
        f.write(f"  Target color enhancement: {failover_counts.get('target_color', 0)}\n")
        f.write(f"  Gray enhancement: {failover_counts.get('gray_enhancement', 0)}\n")
        f.write(f"  White enhancement: {failover_counts.get('white_enhancement', 0)}\n")
        f.write(f"  Baseline watermark regions: {failover_counts.get('watermark', 0)}\n")

if __name__ == "__main__":
    main() 