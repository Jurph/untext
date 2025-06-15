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
from typing import Optional, List, Tuple, Dict, Any

from .utils import (
    get_image_files, load_image, save_image, setup_logger
)
from .preprocessor import preprocess_image
from .detector import initialize_models, TextDetector
from .find_text_colors import find_text_colors, hex_to_bgr, html_to_bgr, find_mask_by_spatial_tf_idf
from .mask_generator import generate_mask
from .inpaint import inpaint_image

logger = setup_logger(__name__)

# Global model instances - initialized once and reused
_global_doctr_detector = None
_global_easyocr_reader = None
_global_east_model = None

def _detect_with_doctr_configurable(image, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run DocTR detection with configurable confidence threshold."""
    global _global_doctr_detector
    try:
        # Create detector only if not already created or if confidence threshold changed
        if _global_doctr_detector is None or _global_doctr_detector.confidence_threshold != confidence_threshold:
            _global_doctr_detector = TextDetector(confidence_threshold=confidence_threshold, min_text_size=3)
            logger.debug(f"Created DocTR detector with confidence threshold {confidence_threshold}")
        
        detections = _global_doctr_detector.detect(image)
        
        results = []
        for detection in detections:
            geometry = detection['geometry']
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            confidence = detection.get('confidence', 0.5) * 100
            
            results.append((x, y, w, h, confidence))
            
        return results
        
    except Exception as e:
        logger.warning(f"DocTR detection failed: {e}")
        return []

def _detect_with_easyocr_configurable(image, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run EasyOCR detection with configurable confidence threshold."""
    global _global_easyocr_reader
    try:
        # Create reader only once - confidence filtering happens later
        if _global_easyocr_reader is None:
            import easyocr
            _global_easyocr_reader = easyocr.Reader(['en'], verbose=False)
            logger.debug("Created EasyOCR reader")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_raw = _global_easyocr_reader.readtext(rgb_image)
        
        results = []
        for bbox_points, text, confidence in results_raw:
            if confidence < confidence_threshold:
                continue
                
            bbox_array = np.array(bbox_points)
            x_coords = bbox_array[:, 0]
            y_coords = bbox_array[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            confidence_pct = confidence * 100
            
            results.append((x, y, w, h, confidence_pct))
            
        return results
        
    except Exception as e:
        logger.warning(f"EasyOCR detection failed: {e}")
        return []

def _detect_with_east_configurable(image, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run EAST detection with configurable confidence threshold."""
    global _global_east_model
    try:
        from .detector import _load_east_model, _detect_with_east
        
        # Cache EAST model to avoid reloading
        if _global_east_model is None:
            _global_east_model = _load_east_model()
            logger.debug("Loaded EAST model")
        
        detections = _detect_with_east(image, _global_east_model, min_confidence=confidence_threshold)
        
        results = []
        for detection in detections:
            geometry = detection['geometry']
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            confidence = detection.get('confidence', 0.5) * 100
            
            results.append((x, y, w, h, confidence))
            
        return results
        
    except Exception as e:
        logger.warning(f"EAST detection failed: {e}")
        return []

def calculate_bbox_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate the overlap area between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left >= right or top >= bottom:
        return 0.0
    
    return (right - left) * (bottom - top)

def calculate_bbox_union(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Calculate the bounding box that encompasses both input boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    
    return (left, top, right - left, bottom - top)

def calculate_hybrid_confidence(confidences: List[float]) -> float:
    """Calculate hybrid confidence using: 1 - (1-conf1) × (1-conf2) × ... × (1-confN)"""
    if not confidences:
        return 0.0
    
    # Convert to 0-1 range if needed
    normalized_confs = [c / 100.0 if c > 1.0 else c for c in confidences]
    
    # Calculate product of (1 - confidence) values
    unconfidence_product = 1.0
    for conf in normalized_confs:
        unconfidence_product *= (1.0 - conf)
    
    return 1.0 - unconfidence_product

def find_consensus_boxes(detections: Dict[str, List[Tuple[int, int, int, int, float]]], 
                        overlap_threshold: float = 0.1) -> List[Dict]:
    """Find consensus boxes where multiple detectors agree."""
    consensus_boxes = []
    all_detections = []
    
    # Flatten all detections with detector info
    for detector_name, detection_list in detections.items():
        for detection in detection_list:
            x, y, w, h, conf = detection
            all_detections.append({
                'detector': detector_name,
                'bbox': (x, y, w, h),
                'confidence': conf / 100.0 if conf > 1.0 else conf,
                'used': False
            })
    
    # Find overlapping groups
    for i, det1 in enumerate(all_detections):
        if det1['used']:
            continue
            
        overlapping_group = [det1]
        det1['used'] = True
        
        # Check for overlaps with remaining detections
        for j in range(i + 1, len(all_detections)):
            det2 = all_detections[j]
            if det2['used'] or det1['detector'] == det2['detector']:
                continue
                
            # Calculate overlap
            overlap_area = calculate_bbox_overlap(det1['bbox'], det2['bbox'])
            bbox1_area = det1['bbox'][2] * det1['bbox'][3]
            bbox2_area = det2['bbox'][2] * det2['bbox'][3]
            
            # Check if overlap is significant relative to smaller box
            min_area = min(bbox1_area, bbox2_area)
            overlap_ratio = overlap_area / min_area if min_area > 0 else 0
            
            if overlap_ratio >= overlap_threshold:
                overlapping_group.append(det2)
                det2['used'] = True
                
                # Check if this detection also overlaps with others in the group
                for k in range(j + 1, len(all_detections)):
                    det3 = all_detections[k]
                    if det3['used'] or det3['detector'] in [d['detector'] for d in overlapping_group]:
                        continue
                    
                    # Check overlap with any member of current group
                    for group_det in overlapping_group:
                        overlap_area = calculate_bbox_overlap(group_det['bbox'], det3['bbox'])
                        det3_area = det3['bbox'][2] * det3['bbox'][3]
                        group_det_area = group_det['bbox'][2] * group_det['bbox'][3]
                        min_area = min(det3_area, group_det_area)
                        overlap_ratio = overlap_area / min_area if min_area > 0 else 0
                        
                        if overlap_ratio >= overlap_threshold:
                            overlapping_group.append(det3)
                            det3['used'] = True
                            break
        
        # Create consensus box if we have multiple detectors
        if len(overlapping_group) >= 2:
            # Calculate union bounding box
            union_bbox = overlapping_group[0]['bbox']
            for det in overlapping_group[1:]:
                union_bbox = calculate_bbox_union(union_bbox, det['bbox'])
            
            # Calculate hybrid confidence
            confidences = [det['confidence'] for det in overlapping_group]
            hybrid_conf = calculate_hybrid_confidence(confidences)
            
            # Get detector names
            detector_names = [det['detector'] for det in overlapping_group]
            
            consensus_boxes.append({
                'bbox': union_bbox,
                'confidence': hybrid_conf,
                'detectors': detector_names,
                'detector_count': len(detector_names),
                'original_confidences': confidences
            })
    
    return consensus_boxes

def run_consensus_detection(image: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Run consensus detection and return padded bounding boxes."""
    # Convert grayscale to BGR for detectors that expect color input
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image
    
    # Run all detectors
    detections = {}
    
    try:
        east_detections = _detect_with_east_configurable(image_bgr, confidence_threshold)
        detections['east'] = east_detections
        logger.debug(f"EAST found {len(east_detections)} detections")
    except Exception as e:
        logger.error(f"EAST detection failed: {e}")
        detections['east'] = []
    
    try:
        doctr_detections = _detect_with_doctr_configurable(image_bgr, confidence_threshold)
        detections['doctr'] = doctr_detections
        logger.debug(f"DocTR found {len(doctr_detections)} detections")
    except Exception as e:
        logger.error(f"DocTR detection failed: {e}")
        detections['doctr'] = []
    
    try:
        easyocr_detections = _detect_with_easyocr_configurable(image_bgr, confidence_threshold)
        detections['easyocr'] = easyocr_detections
        logger.debug(f"EasyOCR found {len(easyocr_detections)} detections")
    except Exception as e:
        logger.error(f"EasyOCR detection failed: {e}")
        detections['easyocr'] = []
    
    # Find consensus boxes
    consensus_boxes = find_consensus_boxes(detections, overlap_threshold=0.1)
    logger.info(f"Found {len(consensus_boxes)} consensus regions")
    
    if not consensus_boxes:
        return []
    
    # Pad consensus boxes by 20% and ensure they stay within image bounds
    h, w = image.shape[:2]
    padded_boxes = []
    
    for consensus in consensus_boxes:
        x, y, box_w, box_h = consensus['bbox']
        detector_names = "+".join(sorted(consensus['detectors']))
        
        # Calculate padding (20% = 10% on each side)
        pad_w = int(box_w * 0.1)
        pad_h = int(box_h * 0.1)
        
        # Apply padding
        padded_x = max(0, x - pad_w)
        padded_y = max(0, y - pad_h)
        padded_w = min(w - padded_x, box_w + 2 * pad_w)
        padded_h = min(h - padded_y, box_h + 2 * pad_h)
        
        padded_box = (padded_x, padded_y, padded_w, padded_h)
        padded_boxes.append(padded_box)
        
        logger.info(f"Consensus box from {detector_names}: {consensus['bbox']} -> padded: {padded_box}")
    
    return padded_boxes

def initialize_consensus_models(confidence_threshold: float = 0.3) -> None:
    """Initialize all models for consensus detection to avoid per-image startup costs."""
    global _global_doctr_detector, _global_easyocr_reader, _global_east_model
    
    logger.info("Pre-loading all detection models for consensus...")
    
    # Initialize DocTR
    try:
        _global_doctr_detector = TextDetector(confidence_threshold=confidence_threshold, min_text_size=3)
        logger.info("✓ DocTR model loaded")
    except Exception as e:
        logger.error(f"Failed to load DocTR: {e}")
        _global_doctr_detector = None
    
    # Initialize EasyOCR  
    try:
        import easyocr
        _global_easyocr_reader = easyocr.Reader(['en'], verbose=False)
        logger.info("✓ EasyOCR model loaded")
    except Exception as e:
        logger.error(f"Failed to load EasyOCR: {e}")
        _global_easyocr_reader = None
    
    # Initialize EAST
    try:
        from .detector import _load_east_model
        _global_east_model = _load_east_model()
        logger.info("✓ EAST model loaded")
    except Exception as e:
        logger.error(f"Failed to load EAST: {e}")
        _global_east_model = None
    
    logger.info("Model initialization complete - all models cached for reuse")

def main() -> None:
    """Main entry point for the consensus-based text watermark removal tool."""
    args = parse_args()
    
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
        if args.color.startswith('#'):
            target_color = hex_to_bgr(args.color)
        else:
            target_color = html_to_bgr(args.color)
        logger.info(f"Using target color: {target_color}")
    
    logger.info(f"Found {len(image_files)} image(s) to process")
    logger.info(f"Using consensus detection with confidence threshold: {args.confidence_threshold}")
    logger.info(f"Using spatial TF-IDF with {args.granularity} color clusters per region")
    
    # Initialize models once for persistent loading
    model_init_start = time.time()
    initialize_consensus_models(args.confidence_threshold)
    model_init_time = time.time() - model_init_start
    logger.info(f"All models cached and ready in {model_init_time:.1f} seconds")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        
        try:
            timing_data = process_single_image(image_path, output_path, target_color, args.keep_masks, args.paint, args.maskfile, args.confidence_threshold, args.granularity, forced_bbox)
            
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
        _save_clean_timing_report(detailed_timings, total_time, avg_time, timing_file, args.paint, args.confidence_threshold, args.granularity, target_color, forced_bbox)
        logger.info(f"Timing report saved to: {timing_file}")
        
        # Also save to logfile location if specified
        if args.logfile:
            log_timing_file = Path(args.logfile).with_suffix('.timing.txt')
            _save_clean_timing_report(detailed_timings, total_time, avg_time, log_timing_file, args.paint, args.confidence_threshold, args.granularity, target_color, forced_bbox)
            logger.info(f"Timing report also saved to: {log_timing_file}")

def process_single_image(
    image_path: Path, 
    output_dir: Path, 
    target_color: Optional[tuple] = None,
    keep_masks: bool = False,
    method: str = "lama",
    maskfile: Optional[str] = None,
    confidence_threshold: float = 0.3,
    granularity: int = 24,
    forced_bbox: Optional[tuple] = None
) -> Optional[dict]:
    """Process a single image through the consensus-based spatial TF-IDF pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        target_color: Optional target color as BGR tuple (deprecated, TF-IDF finds colors automatically)
        keep_masks: Whether to save debug masks
        method: Inpainting method to use ("lama" or "telea")
        maskfile: Optional path to mask file to use instead of auto-generation
        confidence_threshold: Confidence threshold for consensus detection
        granularity: Number of color clusters for spatial TF-IDF analysis
        forced_bbox: Optional forced bounding box as (x, y, width, height) tuple
        
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
        'consensus_boxes_count': 0,
        'total_bbox_area': 0
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
        consensus_boxes = [forced_bbox]
        logger.info(f"Using forced bounding box: {forced_bbox}")
        # Validate bbox is within image bounds
        h, w = image.shape[:2]
        if forced_bbox[0] + forced_bbox[2] > w or forced_bbox[1] + forced_bbox[3] > h:
            logger.warning(f"Forced bbox extends beyond image bounds ({w}×{h}), clipping...")
            clipped_bbox = (
                min(forced_bbox[0], w-1),
                min(forced_bbox[1], h-1), 
                min(forced_bbox[2], w - forced_bbox[0]),
                min(forced_bbox[3], h - forced_bbox[1])
            )
            consensus_boxes = [clipped_bbox]
            logger.info(f"Clipped bbox: {clipped_bbox}")
    else:
        logger.info(f"Running consensus detection with confidence threshold {confidence_threshold}...")
        consensus_boxes = run_consensus_detection(preprocessed, confidence_threshold)
        
        if not consensus_boxes:
            logger.warning("No consensus regions detected, trying rotation failover...")
            
            # Rotate image 90 degrees clockwise and try detection again
            h, w = preprocessed.shape[:2]
            rotated_image = cv2.rotate(preprocessed, cv2.ROTATE_90_CLOCKWISE)
            logger.info("Rotated image 90° clockwise, running consensus detection again...")
            
            rotated_consensus_boxes = run_consensus_detection(rotated_image, confidence_threshold)
            used_rotation_failover = True
            
            if rotated_consensus_boxes:
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
                logger.error("No consensus regions detected even after rotation failover")
                # Return partial timing data for failed images
                timings['detection_time'] = time.time() - detection_start
                timings['consensus_boxes_count'] = 0
                timings['total_bbox_area'] = 0
                timings['used_rotation_failover'] = used_rotation_failover
                timings['color_time'] = None  # N/A
                timings['mask_time'] = None   # N/A
                timings['inpaint_time'] = None # N/A
                timings['total_time'] = time.time() - start_time
                timings['failed'] = True
                return timings
        else:
            used_rotation_failover = False
    
    timings['detection_time'] = time.time() - detection_start
    timings['consensus_boxes_count'] = len(consensus_boxes)
    timings['total_bbox_area'] = sum(bbox[2] * bbox[3] for bbox in consensus_boxes)
    timings['used_rotation_failover'] = used_rotation_failover
    
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
    else:
        # Process each consensus box with spatial TF-IDF and combine masks
        color_start = time.time()
        logger.info(f"Applying spatial TF-IDF analysis to {len(consensus_boxes)} consensus regions...")
        
        # Create combined mask
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        regions_processed = 0
        
        for i, bbox in enumerate(consensus_boxes, 1):
            logger.info(f"Processing consensus region {i}/{len(consensus_boxes)} with spatial TF-IDF: {bbox}")
            
            try:
                # Generate spatial TF-IDF mask for this region
                region_mask = find_mask_by_spatial_tf_idf(image, bbox, num_clusters=granularity, debug=False)
                
                if np.sum(region_mask == 255) > 0:
                    # Create full-sized mask positioned at the bbox location
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    x, y, box_w, box_h = bbox
                    # Use actual region dimensions in case of boundary clipping
                    actual_h, actual_w = region_mask.shape[:2]
                    full_mask[y:y+actual_h, x:x+actual_w] = region_mask
                    
                    # Add to combined mask
                    combined_mask = cv2.bitwise_or(combined_mask, full_mask)
                    regions_processed += 1
                    
                    mask_pixels = np.sum(region_mask == 255)
                    logger.info(f"Region {i}: Generated {mask_pixels} mask pixels")
                else:
                    logger.warning(f"Region {i}: Generated empty mask")
                    
            except Exception as e:
                logger.error(f"Error processing region {i}: {e}")
                continue
        
        logger.info(f"Spatial TF-IDF processed {regions_processed}/{len(consensus_boxes)} regions successfully")
        timings['color_time'] = time.time() - color_start
        
        mask_start = time.time()
        mask = combined_mask
        timings['mask_time'] = time.time() - mask_start
    
    # 4. Inpaint image
    inpaint_start = time.time()
    logger.info("Inpainting masked regions...")
    
    # For inpainting, use the union of all consensus boxes as the target region
    if len(consensus_boxes) == 1:
        inpaint_region = consensus_boxes[0]
    else:
        # Calculate union of all consensus boxes
        min_x = min(bbox[0] for bbox in consensus_boxes)
        min_y = min(bbox[1] for bbox in consensus_boxes)
        max_x = max(bbox[0] + bbox[2] for bbox in consensus_boxes)
        max_y = max(bbox[1] + bbox[3] for bbox in consensus_boxes)
        inpaint_region = (min_x, min_y, max_x - min_x, max_y - min_y)
        logger.info(f"Using union region for inpainting: {inpaint_region}")
    
    result = inpaint_image(image, mask, bbox=inpaint_region, method=method)
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
        help="Target color in hex format (#808080) or HTML color name (gray)"
    )
    
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.3,
        help="Confidence threshold for consensus detection (default: 0.3)"
    )
    
    parser.add_argument(
        "--granularity",
        type=int,
        default=24,
        help="Number of color clusters for spatial TF-IDF analysis (default: 24)"
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
    
    return parser.parse_args()

def _save_clean_timing_report(detailed_timings: list, total_time: float, avg_time: float, timing_file: Path, method: str, confidence_threshold: float, granularity: int, target_color: Optional[tuple], forced_bbox: Optional[tuple]) -> None:
    """Save a clean timing report to file without duplicate logging."""
    with open(timing_file, 'w') as f:
        f.write("=" * 74 + "\n")
        f.write("CONSENSUS DETECTION + SPATIAL TF-IDF TIMING REPORT\n")
        f.write("=" * 74 + "\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"TF-IDF granularity: {granularity} color clusters\n")
        f.write(f"Inpainting method: {method}\n")
        if target_color:
            f.write(f"Target color (deprecated): {target_color}\n")
        if forced_bbox:
            f.write(f"Forced bbox: {forced_bbox}\n")
        f.write("\nColumns: MP=Megapixels, Det=Detection, Msk=Mask, Inp=Inpaint, Tot=Total, Rot=Rotation Failover (R=used)\n")
        
        # Header with wider format
        f.write(f"{'Image Name':<25} {'MP':>4} {'Det':>4} {'TF-IDF':>6} {'Msk':>4} {'Inp':>5} {'Tot':>5} {'Boxes':>5} {'Rot':>3}\n")
        f.write("-" * 74 + "\n")
        
        # Individual rows
        for timing in detailed_timings:
            name = timing['image_name'][:25]  # Allow longer names
            rotation_marker = "R" if timing.get('used_rotation_failover', False) else ""
            row = (f"{name:<25} "
                   f"{timing['image_mp']:>4.1f} "
                   f"{timing['detection_time']:>4.1f} "
                   f"{timing['color_time']:>6.1f} "
                   f"{timing['mask_time']:>4.1f} "
                   f"{timing['inpaint_time']:>5.1f} "
                   f"{timing['total_time']:>5.1f} "
                   f"{timing['consensus_boxes_count']:>5d} "
                   f"{rotation_marker:>3}\n")
            f.write(row)
        
        if len(detailed_timings) > 1:
            f.write("-" * 74 + "\n")
            
            # Statistics
            det_times = [t['detection_time'] for t in detailed_timings]
            col_times = [t['color_time'] for t in detailed_timings] 
            msk_times = [t['mask_time'] for t in detailed_timings]
            inp_times = [t['inpaint_time'] for t in detailed_timings]
            tot_times = [t['total_time'] for t in detailed_timings]
            box_counts = [t['consensus_boxes_count'] for t in detailed_timings]
            
            f.write(f"{'MEDIAN':<25} {'':>4} {statistics.median(det_times):>4.1f} "
                   f"{statistics.median(col_times):>6.1f} {statistics.median(msk_times):>4.1f} "
                   f"{statistics.median(inp_times):>5.1f} {statistics.median(tot_times):>5.1f} "
                   f"{statistics.median(box_counts):>5.1f}\n")
            
            f.write(f"{'AVERAGE':<25} {'':>4} {statistics.mean(det_times):>4.1f} "
                   f"{statistics.mean(col_times):>6.1f} {statistics.mean(msk_times):>4.1f} "
                   f"{statistics.mean(inp_times):>5.1f} {statistics.mean(tot_times):>5.1f} "
                   f"{statistics.mean(box_counts):>5.1f}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"Total processing time: {total_time:.1f} seconds\n")
        f.write(f"Average time per image: {avg_time:.1f} seconds\n")
        f.write(f"Images processed: {len(detailed_timings)}\n")
        
        # Consensus statistics
        total_boxes = sum(t['consensus_boxes_count'] for t in detailed_timings)
        images_with_consensus = sum(1 for t in detailed_timings if t['consensus_boxes_count'] > 0)
        f.write(f"Total consensus boxes: {total_boxes}\n")
        f.write(f"Images with consensus: {images_with_consensus}/{len(detailed_timings)} ({100*images_with_consensus/len(detailed_timings):.1f}%)\n")

if __name__ == "__main__":
    main() 