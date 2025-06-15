#!/usr/bin/env python3
"""Comprehensive debug script for text detectors with all features.

This unified script provides:
1. Configurable confidence thresholds for all detectors
2. Visual debug images with colored bounding boxes  
3. Consensus box detection and highlighting
4. Fixed-width summary table generation
5. Detailed per-image statistics

Features:
- RED: EAST detections
- BLUE: DocTR detections  
- GREEN: EasyOCR detections
- MAGENTA: 2-detector consensus
- CYAN: 3-detector consensus

Usage:
python debug_detectors.py path/to/images --confidence-threshold 0.25
python debug_detectors.py path/to/images --summary-only --output-file results.txt
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Import our production modules
from untextre.utils import load_image, setup_logger
from untextre.preprocessor import preprocess_image
from untextre.detector import initialize_models, TextDetector

logger = setup_logger(__name__)

# Detection colors (BGR format for OpenCV)
COLORS = {
    'east': (0, 0, 255),      # RED
    'doctr': (255, 0, 0),     # BLUE  
    'easyocr': (0, 255, 0),   # GREEN
    'consensus_2': (255, 0, 255),  # MAGENTA (2 detectors)
    'consensus_3': (255, 255, 0)   # CYAN (3 detectors)
}

# Label positions to avoid overlap
LABEL_POSITIONS = {
    'east': 'top',          # Above box
    'doctr': 'top_right',   # Upper right
    'easyocr': 'top_left'   # Upper left
}

def _detect_with_doctr_configurable(image, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run DocTR detection with configurable confidence threshold."""
    try:
        detector = TextDetector(confidence_threshold=confidence_threshold, min_text_size=3)
        detections = detector.detect(image)
        
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
    try:
        import easyocr
        reader = easyocr.Reader(['en'], verbose=False)
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

def _detect_with_east_configurable(image, confidence_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """Run EAST detection with configurable confidence threshold."""
    try:
        from untextre.detector import _load_east_model, _detect_with_east
        
        net = _load_east_model()
        detections = _detect_with_east(image, net, min_confidence=confidence_threshold)
        
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

def get_label_position(x: int, y: int, w: int, h: int, position_type: str, 
                      label_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate label position based on position type."""
    label_w, label_h = label_size
    
    if position_type == 'top':
        label_x = x
        label_y = y - 10 if y > 30 else y + h + 20
    elif position_type == 'top_left':
        label_x = x
        label_y = y - 10 if y > 30 else y + label_h + 5
    elif position_type == 'top_right':
        label_x = x + w - label_w - 5
        label_y = y - 10 if y > 30 else y + label_h + 5
    else:
        # Default to top
        label_x = x
        label_y = y - 10 if y > 30 else y + h + 20
    
    return label_x, label_y

def run_all_detectors(image: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
    """Run all three detectors with configurable confidence threshold."""
    results = {}
    
    # Convert grayscale to BGR for detectors that expect color input
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image
    
    if not logger.isEnabledFor(logging.INFO):
        # Only log if verbose mode
        pass
    else:
        logger.info("Running EAST detector...")
    
    try:
        east_detections = _detect_with_east_configurable(image_bgr, confidence_threshold)
        results['east'] = east_detections
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"EAST found {len(east_detections)} detections")
    except Exception as e:
        logger.error(f"EAST detection failed: {e}")
        results['east'] = []
    
    if logger.isEnabledFor(logging.INFO):
        logger.info("Running DocTR detector...")
    
    try:
        doctr_detections = _detect_with_doctr_configurable(image_bgr, confidence_threshold)
        results['doctr'] = doctr_detections
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"DocTR found {len(doctr_detections)} detections")
    except Exception as e:
        logger.error(f"DocTR detection failed: {e}")
        results['doctr'] = []
    
    if logger.isEnabledFor(logging.INFO):
        logger.info("Running EasyOCR detector...")
    
    try:
        easyocr_detections = _detect_with_easyocr_configurable(image_bgr, confidence_threshold)
        results['easyocr'] = easyocr_detections
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"EasyOCR found {len(easyocr_detections)} detections")
    except Exception as e:
        logger.error(f"EasyOCR detection failed: {e}")
        results['easyocr'] = []
    
    return results

def draw_detections(image: np.ndarray, detections: Dict[str, List[Tuple[int, int, int, int, float]]], 
                   consensus_boxes: List[Dict] = None) -> np.ndarray:
    """Draw colored bounding boxes for all detections and consensus boxes."""
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw individual detector results
    for detector_name, detection_list in detections.items():
        color = COLORS.get(detector_name, (128, 128, 128))
        position_type = LABEL_POSITIONS.get(detector_name, 'top')
        
        for detection in detection_list:
            x, y, w, h, confidence = detection
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score with improved positioning
            label = f"{detector_name.upper()}: {confidence:.1f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            label_x, label_y = get_label_position(x, y, w, h, position_type, label_size)
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (label_x, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (label_x + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw consensus boxes
    if consensus_boxes:
        for consensus in consensus_boxes:
            x, y, w, h = consensus['bbox']
            detector_count = consensus['detector_count']
            hybrid_conf = consensus['confidence']
            
            # Choose color based on detector count
            if detector_count == 2:
                color = COLORS['consensus_2']  # Magenta
                label_prefix = "2-WAY"
            elif detector_count == 3:
                color = COLORS['consensus_3']  # Cyan
                label_prefix = "3-WAY"
            else:
                color = (128, 128, 128)  # Gray fallback
                label_prefix = f"{detector_count}-WAY"
            
            # Draw thicker consensus box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            
            # Create consensus label
            label = f"{label_prefix}: {hybrid_conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Position consensus label at bottom of box
            label_x = x
            label_y = y + h + label_size[1] + 10
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (label_x, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (label_x + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image

def create_summary_info(detections: Dict[str, List[Tuple[int, int, int, int, float]]], 
                       consensus_boxes: List[Dict] = None) -> str:
    """Create a summary of detection results including consensus boxes."""
    summary_lines = ["DETECTION SUMMARY:"]
    
    # Individual detector summaries
    for detector_name in ['east', 'doctr', 'easyocr']:  # Specific order
        detection_list = detections.get(detector_name, [])
        if detection_list:
            confidences = [det[4] for det in detection_list]
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            min_conf = np.min(confidences)
            
            summary_lines.append(f"- {detector_name.upper()} detected {len(detection_list)} regions")
            summary_lines.append(f"  Confidence: avg={avg_conf:.1f}, max={max_conf:.1f}, min={min_conf:.1f}")
        else:
            summary_lines.append(f"- {detector_name.upper()} detected 0 regions")
    
    # Consensus box summary
    if consensus_boxes:
        summary_lines.append(f"\n- Consensus boxes: {len(consensus_boxes)}")
        for consensus in consensus_boxes:
            x, y, w, h = consensus['bbox']
            detector_count = consensus['detector_count']
            hybrid_conf = consensus['confidence']
            detector_names = "+".join(sorted(consensus['detectors']))
            
            summary_lines.append(f"  -- ({x}, {y}, {w}, {h}) - {detector_count} detectors ({detector_names}), {hybrid_conf:.2f} confidence")
    else:
        summary_lines.append(f"\n- Consensus boxes: 0")
    
    return "\n".join(summary_lines)

def process_single_image(image_path: Path, confidence_threshold: float = 0.3, 
                        overlap_threshold: float = 0.1, output_dir: Path = None, 
                        save_images: bool = True) -> Dict[str, Any]:
    """Process a single image and return detection statistics."""
    try:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Processing {image_path.name}")
        
        # Load original image
        original_image = load_image(image_path)
        if original_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Preprocess image (same as production)
        preprocessed_image = preprocess_image(original_image)
        
        # Run all detectors
        detections = run_all_detectors(preprocessed_image, confidence_threshold)
        
        # Find consensus boxes
        consensus_boxes = find_consensus_boxes(detections, overlap_threshold)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Found {len(consensus_boxes)} consensus regions")
        
        # Create visualization if saving images
        if save_images and output_dir:
            vis_image = draw_detections(preprocessed_image, detections, consensus_boxes)
            
            # Create summary
            summary = create_summary_info(detections, consensus_boxes)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"\n{summary}")
            
            # Save debug image
            debug_filename = f"debug_{image_path.stem}.png"
            debug_path = output_dir / debug_filename
            cv2.imwrite(str(debug_path), vis_image)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Saved debug image: {debug_path}")
            
            # Save summary to text file
            summary_filename = f"debug_{image_path.stem}_summary.txt"
            summary_path = output_dir / summary_filename
            with open(summary_path, 'w') as f:
                f.write(f"Image: {image_path.name}\n")
                f.write(f"Original size: {original_image.shape[1]}×{original_image.shape[0]}\n")
                f.write(f"Preprocessed size: {preprocessed_image.shape[1]}×{preprocessed_image.shape[0]}\n\n")
                f.write(summary)
                f.write(f"\n\nConfidence threshold: {confidence_threshold}")
                f.write(f"\nOverlap threshold: {overlap_threshold}")
                f.write("\n\nDetection Details:\n")
                
                for detector_name, detection_list in detections.items():
                    f.write(f"\n{detector_name.upper()} Detections:\n")
                    if detection_list:
                        for i, (x, y, w, h, conf) in enumerate(detection_list, 1):
                            f.write(f"  {i}: bbox=({x}, {y}, {w}, {h}) confidence={conf:.1f}\n")
                    else:
                        f.write("  No detections\n")
                
                # Add consensus box details
                if consensus_boxes:
                    f.write(f"\nConsensus Boxes:\n")
                    for i, consensus in enumerate(consensus_boxes, 1):
                        x, y, w, h = consensus['bbox']
                        detector_names = "+".join(sorted(consensus['detectors']))
                        original_confs = [f"{c:.2f}" for c in consensus['original_confidences']]
                        f.write(f"  {i}: bbox=({x}, {y}, {w}, {h}) detectors=({detector_names}) ")
                        f.write(f"hybrid_conf={consensus['confidence']:.2f} original_confs=[{', '.join(original_confs)}]\n")
                else:
                    f.write(f"\nConsensus Boxes: None\n")
            
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Saved summary: {summary_path}")
        
        # Count consensus by detector count
        consensus_2 = sum(1 for cb in consensus_boxes if cb['detector_count'] == 2)
        consensus_3 = sum(1 for cb in consensus_boxes if cb['detector_count'] == 3)
        
        return {
            'image_name': image_path.name,
            'east_count': len(detections.get('east', [])),
            'doctr_count': len(detections.get('doctr', [])),
            'easyocr_count': len(detections.get('easyocr', [])),
            'consensus_2_count': consensus_2,
            'consensus_3_count': consensus_3,
            'total_consensus': consensus_2 + consensus_3
        }
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return {
            'image_name': image_path.name,
            'east_count': 0,
            'doctr_count': 0,
            'easyocr_count': 0,
            'consensus_2_count': 0,
            'consensus_3_count': 0,
            'total_consensus': 0
        }

def generate_summary_table(results: List[Dict[str, Any]], confidence_threshold: float) -> str:
    """Generate a fixed-width summary table."""
    if not results:
        return "No results to display."
    
    # Table header
    lines = []
    lines.append(f"DETECTION SUMMARY (confidence threshold: {confidence_threshold})")
    lines.append("=" * 95)
    lines.append("| {:<35} | {:>4} | {:>5} | {:>7} | {:>4} | {:>4} | {:>5} |".format(
        "Image Name", "EAST", "DocTR", "EasyOCR", "2-C", "3-C", "Total"
    ))
    lines.append("|" + "-" * 37 + "|" + "-" * 6 + "|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 6 + "|" + "-" * 6 + "|" + "-" * 7 + "|")
    
    # Data rows
    total_east = 0
    total_doctr = 0
    total_easyocr = 0
    total_consensus_2 = 0
    total_consensus_3 = 0
    images_with_consensus = 0
    
    for result in results:
        if result is None:
            continue
            
        # Truncate long filenames
        display_name = result['image_name']
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."
        
        lines.append("| {:<35} | {:>4} | {:>5} | {:>7} | {:>4} | {:>4} | {:>5} |".format(
            display_name,
            result['east_count'],
            result['doctr_count'], 
            result['easyocr_count'],
            result['consensus_2_count'],
            result['consensus_3_count'],
            result['total_consensus']
        ))
        
        # Track totals
        total_east += result['east_count']
        total_doctr += result['doctr_count']
        total_easyocr += result['easyocr_count']
        total_consensus_2 += result['consensus_2_count']
        total_consensus_3 += result['consensus_3_count']
        
        if result['total_consensus'] > 0:
            images_with_consensus += 1
    
    # Summary footer
    lines.append("|" + "-" * 37 + "|" + "-" * 6 + "|" + "-" * 7 + "|" + "-" * 9 + "|" + "-" * 6 + "|" + "-" * 6 + "|" + "-" * 7 + "|")
    lines.append("| {:<35} | {:>4} | {:>5} | {:>7} | {:>4} | {:>4} | {:>5} |".format(
        "TOTALS",
        total_east,
        total_doctr,
        total_easyocr,
        total_consensus_2,
        total_consensus_3,
        total_consensus_2 + total_consensus_3
    ))
    lines.append("=" * 95)
    
    # Statistics
    total_images = len([r for r in results if r is not None])
    if total_images > 0:
        consensus_rate = (images_with_consensus / total_images) * 100
        lines.append(f"Images with consensus: {images_with_consensus}/{total_images} ({consensus_rate:.1f}%)")
        lines.append(f"Average detections per image: EAST={total_east/total_images:.1f}, DocTR={total_doctr/total_images:.1f}, EasyOCR={total_easyocr/total_images:.1f}")
    
    return "\n".join(lines)

def main():
    """Main entry point for comprehensive detector debugging."""
    parser = argparse.ArgumentParser(description="Comprehensive debug for text detectors with consensus visualization and summary tables")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for all detectors (default: 0.3)")
    parser.add_argument("--overlap-threshold", type=float, default=0.1,
                       help="Minimum overlap ratio for consensus (default: 0.1)")
    parser.add_argument("--output-dir", default="debug/images", 
                       help="Output directory for debug images (default: debug/images)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Generate summary table only, skip saving debug images")
    parser.add_argument("--output-file", help="Output file for summary table (default: print to console)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)  # Suppress INFO messages for cleaner output
    
    # Create output directory if needed
    output_dir = None
    if not args.summary_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            logger.info(f"Debug output directory: {output_dir}")
    
    # Initialize models (same as production)
    print("Initializing detection models...")
    initialize_models(['doctr', 'easyocr'])  # EAST is loaded on-demand
    
    # Process input
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        # Single image
        result = process_single_image(input_path, args.confidence_threshold, args.overlap_threshold, 
                                     output_dir, not args.summary_only)
        if result:
            results.append(result)
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = sorted([f for f in input_path.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        if not image_files:
            logger.error(f"No image files found in {input_path}")
            return
        
        print(f"Processing {len(image_files)} images with confidence threshold {args.confidence_threshold}...")
        
        for i, image_file in enumerate(image_files, 1):
            if not args.verbose:
                print(f"Processing {i}/{len(image_files)}: {image_file.name}", end="\r")
            else:
                print(f"\n--- Processing image {i}/{len(image_files)} ---")
            
            try:
                result = process_single_image(image_file, args.confidence_threshold, args.overlap_threshold,
                                            output_dir, not args.summary_only)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                continue
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Generate summary table
    summary_table = generate_summary_table(results, args.confidence_threshold)
    
    if args.output_file:
        # Save to file
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            f.write(summary_table)
        print(f"\nSummary table saved to: {output_path}")
    else:
        # Print to console
        print("\n" + summary_table)
    
    if not args.summary_only and output_dir:
        print(f"\nDebug images and detailed summaries saved to: {output_dir}")

if __name__ == "__main__":
    main() 