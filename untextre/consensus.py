"""Consensus detection utilities combining multiple text detectors.

This module provides functions to run EAST, EasyOCR, and YOLO11x detectors
and combine their results using consensus logic to find high-confidence
text regions where multiple detectors agree.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

from .utils import setup_logger, pad_bbox_to_multiple, MODEL_CONFIDENCE_FLOOR, CLI_DEFAULT_CONFIDENCE
from . import detector as detector_mod
from .detector import cleanup_vram

logger = setup_logger(__name__)



def detect_with_easyocr(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int, float]]:
    """Run EasyOCR detection with configurable confidence threshold.
    
    Args:
        image: Input image as H×W×3 BGR numpy array
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        
    Returns:
        List of (x, y, width, height, confidence_pct) tuples
    """
    try:
        reader = detector_mod.get_easyocr_reader()
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_raw = reader.readtext(rgb_image)
        
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


def detect_with_yolo11x(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int, float]]:
    """Run YOLO11x watermark detection with configurable confidence threshold.

    Args:
        image: Input image as H×W×3 BGR numpy array
        confidence_threshold: Minimum confidence for detections (0.0-1.0)

    Returns:
        List of (x, y, width, height, confidence_pct) tuples
    """
    try:
        model = detector_mod.get_yolo11x_model()
        # Run at MODEL_CONFIDENCE_FLOOR so the model captures everything;
        # the caller's confidence_threshold is applied as a post-filter.
        results = model.predict(image, conf=MODEL_CONFIDENCE_FLOOR, verbose=False)

        detections = []
        for result in results:
            boxes = getattr(result, "boxes", [])
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue

                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                detections.append((x, y, w, h, confidence * 100))

        return detections

    except Exception as e:
        logger.warning(f"YOLO11x detection failed: {e}")
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
    try:
        east_model = detector_mod.get_east_net()

        boxes = _run_east_at_resolution(
            image, east_model, resolution=640, confidence_threshold=confidence_threshold
        )
        if not boxes:
            logger.debug("EAST at 640px found nothing; re-running at 1280px")
            boxes = _run_east_at_resolution(
                image, east_model, resolution=1280, confidence_threshold=confidence_threshold
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


def calculate_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate intersection over union for two bounding boxes."""
    overlap_area = calculate_bbox_overlap(bbox1, bbox2)
    if overlap_area <= 0:
        return 0.0

    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - overlap_area
    return overlap_area / union_area if union_area > 0 else 0.0


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

    Uses graph connected components rather than a seed-based nested loop.
    Nodes are individual detections; an edge connects two detections from
    DISTINCT detectors when their IoU >= overlap_threshold.  Every connected
    component with 2+ distinct detectors becomes one consensus box whose bbox
    is the union of all member bboxes.

    This is order-independent: the old nested-loop approach could drop a
    detection B (yolo) when the production flattening order put B before a
    later detection C (easyocr) that overlapped both B and the seed A (east).
    B was skipped against A (no overlap), C was added and marked used, and the
    inner-inner scan started after C so B was never reconsidered against C.
    Graph-CC adds A-C and B-C edges independently, placing all three in one
    component regardless of iteration order.

    Args:
        detections: Dictionary mapping detector name to list of (x, y, w, h, conf) tuples
        overlap_threshold: Minimum bbox IoU for consensus (default: 0.1)

    Returns:
        List of consensus box dictionaries with keys:
            - 'bbox': (x, y, width, height)
            - 'confidence': hybrid confidence score
            - 'detectors': list of detector names that agreed
            - 'detector_count': number of detectors that agreed
            - 'original_confidences': list of original confidence values
    """
    # Flatten all detections into a node list.
    nodes: List[Dict] = []
    for detector_name, detection_list in detections.items():
        for detection in detection_list:
            x, y, w, h, conf = detection
            nodes.append({
                'detector': detector_name,
                'bbox': (x, y, w, h),
                'confidence': conf / 100.0 if conf > 1.0 else conf,
            })

    n = len(nodes)
    if n == 0:
        return []

    # Union-Find with path compression.
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        parent[_find(a)] = _find(b)

    # Add edges: only between distinct-detector pairs with IoU >= threshold.
    # EMPIRICAL — IoU > 0.1 validated on ~400 has-text-2 samples.
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i]['detector'] == nodes[j]['detector']:
                continue
            if calculate_bbox_iou(nodes[i]['bbox'], nodes[j]['bbox']) >= overlap_threshold:
                _union(i, j)

    # Group indices by component root.
    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(_find(i), []).append(i)

    # Emit one consensus box per component that spans >= 2 distinct detectors.
    consensus_boxes = []
    for indices in components.values():
        members = [nodes[i] for i in indices]
        detector_names = sorted({m['detector'] for m in members})
        if len(detector_names) < 2:
            continue

        union_bbox = members[0]['bbox']
        for m in members[1:]:
            union_bbox = calculate_bbox_union(union_bbox, m['bbox'])

        confidences = [m['confidence'] for m in members]
        consensus_boxes.append({
            'bbox': union_bbox,
            'confidence': calculate_hybrid_confidence(confidences),
            'detectors': detector_names,
            'detector_count': len(detector_names),
            'original_confidences': confidences,
        })

    return consensus_boxes


def run_consensus_detection(image: np.ndarray, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Tuple[int, int, int, int]]:
    """Run consensus detection and return padded bounding boxes.
    
    This is the main entry point for consensus detection. It runs all three
    production detectors (EAST, EasyOCR, YOLO11x), finds regions where 2+ detectors agree,
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
        yolo11x_detections = detect_with_yolo11x(image_bgr, confidence_threshold)
        detections['yolo11x'] = yolo11x_detections
        logger.debug(f"YOLO11x found {len(yolo11x_detections)} detections")
    except Exception as e:
        logger.error(f"YOLO11x detection failed: {e}")
        detections['yolo11x'] = []
    
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
        
        # EMPIRICAL — 10% per side chosen as a soft expansion; not formally validated.
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


def initialize_consensus_models() -> None:
    """Initialize all detection models to avoid per-image startup costs.

    Models are always initialized with MODEL_CONFIDENCE_FLOOR internally;
    the real user-facing threshold is applied as a post-filter at detection
    time so users can adjust it without re-initializing (which wastes VRAM).
    """
    logger.info("Pre-loading all detection models...")
    
    # Initialize YOLO11x
    try:
        detector_mod.get_yolo11x_model()
        logger.info("[OK] YOLO11x model loaded")
    except Exception as e:
        logger.error(f"Failed to load YOLO11x: {e}")
    
    # Initialize EasyOCR
    try:
        detector_mod.get_easyocr_reader()
        logger.info("[OK] EasyOCR model loaded")
    except Exception as e:
        logger.error(f"Failed to load EasyOCR: {e}")
    
    # Initialize EAST
    try:
        detector_mod.get_east_net()
        logger.info("[OK] EAST model loaded")
    except Exception as e:
        logger.error(f"Failed to load EAST: {e}")
    
    logger.info("Detection model initialization complete")
