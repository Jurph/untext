"""Consensus detection utilities combining multiple text detectors.

This module provides functions to run EAST, DocTR, and EasyOCR detectors
and combine their results using consensus logic to find high-confidence
text regions where multiple detectors agree.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

from .utils import setup_logger, pad_bbox_to_multiple, MODEL_CONFIDENCE_FLOOR, CLI_DEFAULT_CONFIDENCE
from .detector import TextDetector, cleanup_vram

logger = setup_logger(__name__)

# Global model instances for persistent loading
_global_doctr_detector = None
_global_easyocr_reader = None
_global_east_model = None


def detect_with_doctr(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int, float]]:
    """Run DocTR detection with configurable confidence threshold.
    
    Args:
        image: Input image as H×W×3 BGR numpy array
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        
    Returns:
        List of (x, y, width, height, confidence_pct) tuples
    """
    global _global_doctr_detector
    try:
        # Create detector only once - confidence threshold is applied as post-filter
        # (Recreating the detector loads the model again, wasting VRAM)
        if _global_doctr_detector is None:
            # Use MODEL_CONFIDENCE_FLOOR so the model captures everything;
            # the caller's confidence_threshold is applied as a post-filter.
            _global_doctr_detector = TextDetector(confidence_threshold=MODEL_CONFIDENCE_FLOOR, min_text_size=3)
            logger.debug("Created DocTR detector (one-time initialization)")
        
        detections = _global_doctr_detector.detect(image)
        # Apply confidence threshold as post-filter
        detections = [d for d in detections if d.get('confidence', 0) >= confidence_threshold]
        
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


def detect_with_easyocr(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int, float]]:
    """Run EasyOCR detection with configurable confidence threshold.
    
    Args:
        image: Input image as H×W×3 BGR numpy array
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        
    Returns:
        List of (x, y, width, height, confidence_pct) tuples
    """
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


def _run_east_at_resolution(
    image: np.ndarray,
    net,
    resolution: int,
    confidence_threshold: float,
) -> List[Tuple[int, int, int, int, float]]:
    """Run EAST at a given resolution and return (x, y, w, h, confidence_pct) list."""
    from .detector import _detect_with_east

    detections = _detect_with_east(
        image, net,
        min_confidence=confidence_threshold,
        width=resolution,
        height=resolution,
    )
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


def detect_with_east(image: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run EAST detection: 640px first; if zero detections, re-run at 1280px.

    Args:
        image: Input image as H×W×3 BGR numpy array
        confidence_threshold: Minimum confidence for detections (0.0-1.0)

    Returns:
        List of (x, y, width, height, confidence_pct) tuples
    """
    global _global_east_model
    try:
        from .detector import _load_east_model

        if _global_east_model is None:
            _global_east_model = _load_east_model()
            logger.debug("Loaded EAST model")

        boxes = _run_east_at_resolution(
            image, _global_east_model, resolution=640, confidence_threshold=confidence_threshold
        )
        if not boxes:
            logger.debug("EAST at 640px found nothing; re-running at 1280px")
            boxes = _run_east_at_resolution(
                image, _global_east_model, resolution=1280, confidence_threshold=confidence_threshold
            )
        logger.debug(f"EAST found {len(boxes)} detections")
        return boxes

    except Exception as e:
        logger.warning(f"EAST detection failed: {e}")
        return []


def calculate_bbox_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate the overlap area between two bounding boxes.
    
    Args:
        bbox1: First bounding box as (x, y, width, height)
        bbox2: Second bounding box as (x, y, width, height)
        
    Returns:
        Overlap area in pixels
    """
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
    """Calculate the bounding box that encompasses both input boxes.
    
    Args:
        bbox1: First bounding box as (x, y, width, height)
        bbox2: Second bounding box as (x, y, width, height)
        
    Returns:
        Union bounding box as (x, y, width, height)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    
    return (left, top, right - left, bottom - top)


def calculate_hybrid_confidence(confidences: List[float]) -> float:
    """Calculate hybrid confidence using: 1 - (1-conf1) × (1-conf2) × ... × (1-confN).
    
    Args:
        confidences: List of confidence values (0.0-1.0 or 0-100)
        
    Returns:
        Hybrid confidence (0.0-1.0)
    """
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
    """Find consensus boxes where multiple detectors agree.
    
    Args:
        detections: Dictionary mapping detector name to list of (x, y, w, h, conf) tuples
        overlap_threshold: Minimum overlap ratio for consensus (default: 0.1)
        
    Returns:
        List of consensus box dictionaries with keys:
            - 'bbox': (x, y, width, height)
            - 'confidence': hybrid confidence score
            - 'detectors': list of detector names that agreed
            - 'detector_count': number of detectors that agreed
            - 'original_confidences': list of original confidence values
    """
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
            if det2['used'] or det2['detector'] in {d['detector'] for d in overlapping_group}:
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
            
            # Get unique detector names (each detector counts once)
            detector_names = sorted({det['detector'] for det in overlapping_group})
            
            consensus_boxes.append({
                'bbox': union_bbox,
                'confidence': hybrid_conf,
                'detectors': detector_names,
                'detector_count': len(detector_names),
                'original_confidences': confidences
            })
    
    return consensus_boxes


def run_consensus_detection(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int]]:
    """Run consensus detection and return padded bounding boxes.
    
    This is the main entry point for consensus detection. It runs all three
    detectors (EAST, DocTR, EasyOCR), finds regions where 2+ detectors agree,
    and returns padded bounding boxes ready for processing.
    
    Args:
        image: Input image as H×W×3 BGR or H×W grayscale numpy array
        confidence_threshold: Minimum confidence for individual detections (0.0-1.0)
        
    Returns:
        List of consensus bounding boxes as (x, y, width, height) tuples,
        padded by 20% and aligned to mod-4 boundaries
    """
    # Convert grayscale to BGR for detectors that expect color input
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image
    
    # Run all detectors
    detections = {}
    
    try:
        east_detections = detect_with_east(image_bgr, confidence_threshold)
        detections['east'] = east_detections
        logger.debug(f"EAST found {len(east_detections)} detections")
    except Exception as e:
        logger.error(f"EAST detection failed: {e}")
        detections['east'] = []
    
    try:
        doctr_detections = detect_with_doctr(image_bgr, confidence_threshold)
        detections['doctr'] = doctr_detections
        logger.debug(f"DocTR found {len(doctr_detections)} detections")
    except Exception as e:
        logger.error(f"DocTR detection failed: {e}")
        detections['doctr'] = []
    
    try:
        easyocr_detections = detect_with_easyocr(image_bgr, confidence_threshold)
        detections['easyocr'] = easyocr_detections
        logger.debug(f"EasyOCR found {len(easyocr_detections)} detections")
    except Exception as e:
        logger.error(f"EasyOCR detection failed: {e}")
        detections['easyocr'] = []
    
    # Free VRAM after all GPU-based detection is complete
    cleanup_vram()
    
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
        
        # Ensure dimensions are divisible by 4 for neural network compatibility
        mod4_box = pad_bbox_to_multiple(padded_box, multiple=4, image_shape=(h, w))
        padded_boxes.append(mod4_box)
        
        logger.info(f"Consensus box from {detector_names}: {consensus['bbox']} -> padded: {padded_box} -> mod4: {mod4_box}")
    
    return padded_boxes


def initialize_consensus_models(confidence_threshold: float = MODEL_CONFIDENCE_FLOOR) -> None:
    """Initialize all detection models to avoid per-image startup costs.
    
    The confidence_threshold parameter is **deprecated and ignored**.
    Models are always initialized with MODEL_CONFIDENCE_FLOOR internally;
    the real user-facing threshold is applied as a post-filter at detection
    time so users can adjust it without re-initializing (which wastes VRAM).
    
    The default is set to MODEL_CONFIDENCE_FLOOR so that call sites match
    reality and don't mislead future readers.
    
    Args:
        confidence_threshold: Deprecated -- ignored; kept for API compatibility.
    """
    global _global_doctr_detector, _global_easyocr_reader, _global_east_model
    
    logger.info("Pre-loading all detection models...")
    
    # Initialize DocTR with MODEL_CONFIDENCE_FLOOR -- actual threshold applied as post-filter
    if _global_doctr_detector is None:
        try:
            _global_doctr_detector = TextDetector(confidence_threshold=MODEL_CONFIDENCE_FLOOR, min_text_size=3)
            logger.info("[OK] DocTR model loaded")
        except Exception as e:
            logger.error(f"Failed to load DocTR: {e}")
            _global_doctr_detector = None
    else:
        logger.info("[OK] DocTR model already loaded")
    
    # Initialize EasyOCR
    if _global_easyocr_reader is None:
        try:
            import easyocr
            _global_easyocr_reader = easyocr.Reader(['en'], verbose=False)
            logger.info("[OK] EasyOCR model loaded")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            _global_easyocr_reader = None
    else:
        logger.info("[OK] EasyOCR model already loaded")
    
    # Initialize EAST
    if _global_east_model is None:
        try:
            from .detector import _load_east_model
            _global_east_model = _load_east_model()
            logger.info("[OK] EAST model loaded")
        except Exception as e:
            logger.error(f"Failed to load EAST: {e}")
            _global_east_model = None
    else:
        logger.info("[OK] EAST model already loaded")
    
    logger.info("Detection model initialization complete")
