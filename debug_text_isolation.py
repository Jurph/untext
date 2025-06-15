#!/usr/bin/env python3
"""
Text isolation proof of concept using TF-IDF color analysis with score histogram generation.

This script demonstrates a new approach to text watermark removal:
1. Uses all three detectors to find consensus text regions
2. Applies TF-IDF analysis to find distinctive text colors in each region
3. Creates region-specific masks using only the colors found for that region
4. Combines all regional masks for final inpainting
5. Collects TF-IDF scores and generates histograms for threshold analysis

Usage:
python debug_text_isolation.py path/to/image.jpg --output results/ --collect-scores
python debug_text_isolation.py path/to/images/ --spatial --collect-scores --histogram
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Import our production modules
from untextre.utils import load_image, save_image, setup_logger, dilate_by_pixels, dilate_by_percent
from untextre.preprocessor import preprocess_image
from untextre.detector import initialize_models, TextDetector
from untextre.find_text_colors import find_colors_by_tf_idf, hex_to_bgr, find_mask_by_spatial_tf_idf

logger = setup_logger(__name__)

# Global score collection
GLOBAL_TF_IDF_SCORES = []
GLOBAL_OTSU_THRESHOLDS = []

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

def run_all_detectors(image: np.ndarray, confidence_threshold: float = 0.3) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
    """Run all three detectors with configurable confidence threshold."""
    results = {}
    
    # Convert grayscale to BGR for detectors that expect color input
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image
    
    logger.info("Running EAST detector...")
    try:
        east_detections = _detect_with_east_configurable(image_bgr, confidence_threshold)
        results['east'] = east_detections
        logger.info(f"EAST found {len(east_detections)} detections")
    except Exception as e:
        logger.error(f"EAST detection failed: {e}")
        results['east'] = []
    
    logger.info("Running DocTR detector...")
    try:
        doctr_detections = _detect_with_doctr_configurable(image_bgr, confidence_threshold)
        results['doctr'] = doctr_detections
        logger.info(f"DocTR found {len(doctr_detections)} detections")
    except Exception as e:
        logger.error(f"DocTR detection failed: {e}")
        results['doctr'] = []
    
    logger.info("Running EasyOCR detector...")
    try:
        easyocr_detections = _detect_with_easyocr_configurable(image_bgr, confidence_threshold)
        results['easyocr'] = easyocr_detections
        logger.info(f"EasyOCR found {len(easyocr_detections)} detections")
    except Exception as e:
        logger.error(f"EasyOCR detection failed: {e}")
        results['easyocr'] = []
    
    return results

def create_color_mask(image: np.ndarray, colors: List[str], tolerance: int = 10) -> np.ndarray:
    """Create a binary mask for specified colors."""
    if not colors:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert hex colors to BGR
    bgr_colors = [hex_to_bgr(color) for color in colors]
    
    # Create combined mask
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for bgr_color in bgr_colors:
        # Create bounds with tolerance
        lower = np.array([max(0, c - tolerance) for c in bgr_color])
        upper = np.array([min(255, c + tolerance) for c in bgr_color])
        
        # Create mask for this color
        color_mask = cv2.inRange(image, lower, upper)
        
        # Add to combined mask
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)
    
    return combined_mask

def process_image_with_tf_idf(image_path: Path, confidence_threshold: float = 0.3,
                             overlap_threshold: float = 0.1, dilation_pixels: int = 10,
                             output_dir: Path = None, debug: bool = False) -> Dict[str, Any]:
    """Process a single image using TF-IDF based text isolation."""
    logger.info(f"Processing {image_path.name}")
    
    # Load and preprocess image
    original_image = load_image(image_path)
    preprocessed_image = preprocess_image(original_image)
    
    # Run all detectors
    detections = run_all_detectors(preprocessed_image, confidence_threshold)
    
    # Find consensus boxes
    consensus_boxes = find_consensus_boxes(detections, overlap_threshold)
    logger.info(f"Found {len(consensus_boxes)} consensus regions")
    
    if not consensus_boxes:
        logger.warning("No consensus boxes found - nothing to process")
        return {
            'success': False,
            'reason': 'No consensus boxes found',
            'consensus_count': 0,
            'regions_processed': 0
        }
    
    # Process each consensus region
    regional_masks = []
    regions_processed = 0
    
    for i, consensus in enumerate(consensus_boxes):
        bbox = consensus['bbox']
        detector_names = "+".join(sorted(consensus['detectors']))
        
        logger.info(f"Processing region {i+1}/{len(consensus_boxes)}: {detector_names}")
        
        # Dilate the bbox
        dilated_bbox = dilate_by_pixels(preprocessed_image, bbox, dilation_pixels)
        
        # Find TF-IDF colors for this region using ORIGINAL COLOR IMAGE
        try:
            tf_idf_colors = find_colors_by_tf_idf(original_image, bbox, debug=debug)
            logger.info(f"Region {i+1} found {len(tf_idf_colors)} distinctive colors")
            
            if tf_idf_colors:
                # Extract the dilated region from ORIGINAL COLOR IMAGE
                x, y, w, h = dilated_bbox
                region_roi = original_image[y:y+h, x:x+w]
                
                # Create mask for this region using only its TF-IDF colors
                region_mask = create_color_mask(region_roi, tf_idf_colors)
                
                # Create full-sized mask
                full_mask = np.zeros(preprocessed_image.shape[:2], dtype=np.uint8)
                full_mask[y:y+h, x:x+w] = region_mask
                
                regional_masks.append(full_mask)
                regions_processed += 1
                
                if debug:
                    logger.info(f"Region {i+1} colors: {tf_idf_colors[:3]}")  # Show first 3 colors
            else:
                logger.warning(f"Region {i+1} found no distinctive colors")
                
        except Exception as e:
            logger.error(f"Error processing region {i+1}: {e}")
            continue
    
    if not regional_masks:
        logger.warning("No regional masks generated")
        return {
            'success': False,
            'reason': 'No regional masks generated',
            'consensus_count': len(consensus_boxes),
            'regions_processed': 0
        }
    
    # Combine all regional masks
    logger.info(f"Combining {len(regional_masks)} regional masks")
    combined_mask = regional_masks[0].copy()
    for mask in regional_masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply inpainting
    logger.info("Applying inpainting...")
    try:
        inpainted = cv2.inpaint(original_image, combined_mask, 3, cv2.INPAINT_TELEA)
        
        # Save results if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save inpainted result as high-quality JPEG
            result_path = output_dir / f"{image_path.stem}_tf_idf_result.jpg"
            cv2.imwrite(str(result_path), inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Saved result: {result_path}")
            
            # Save combined mask
            mask_path = output_dir / f"{image_path.stem}_tf_idf_mask.png"
            save_image(combined_mask, mask_path)
            logger.info(f"Saved mask: {mask_path}")
            
            # Save debug visualization if requested
            if debug:
                # Create visualization with bboxes
                vis_image = original_image.copy()
                for i, consensus in enumerate(consensus_boxes):
                    x, y, w, h = consensus['bbox']
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis_image, f"C{i+1}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                debug_path = output_dir / f"{image_path.stem}_tf_idf_debug.jpg"
                cv2.imwrite(str(debug_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Saved debug visualization: {debug_path}")
        
        return {
            'success': True,
            'consensus_count': len(consensus_boxes),
            'regions_processed': regions_processed,
            'mask_pixels': np.sum(combined_mask > 0)
        }
        
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")
        return {
            'success': False,
            'reason': f'Inpainting failed: {e}',
            'consensus_count': len(consensus_boxes),
            'regions_processed': regions_processed
        }

def process_image_with_spatial_tf_idf(image_path: Path, confidence_threshold: float = 0.3,
                                     overlap_threshold: float = 0.1, dilation_pixels: int = 10,
                                     output_dir: Path = None, debug: bool = False, 
                                     collect_scores: bool = False) -> Dict[str, Any]:
    """Process a single image using spatial TF-IDF masking with Otsu thresholding."""
    logger.info(f"Processing {image_path.name} with spatial TF-IDF")
    
    # Load and preprocess image
    original_image = load_image(image_path)
    preprocessed_image = preprocess_image(original_image)
    
    # Run all detectors
    detections = run_all_detectors(preprocessed_image, confidence_threshold)
    
    # Find consensus boxes
    consensus_boxes = find_consensus_boxes(detections, overlap_threshold)
    logger.info(f"Found {len(consensus_boxes)} consensus regions")
    
    if not consensus_boxes:
        logger.warning("No consensus boxes found - nothing to process")
        return {
            'success': False,
            'reason': 'No consensus boxes found',
            'consensus_count': 0,
            'regions_processed': 0
        }
    
    # Process each consensus region with spatial TF-IDF
    regional_masks = []
    regions_processed = 0
    
    for i, consensus in enumerate(consensus_boxes):
        bbox = consensus['bbox']
        detector_names = "+".join(sorted(consensus['detectors']))
        
        logger.info(f"Processing region {i+1}/{len(consensus_boxes)}: {detector_names}")
        
        # Dilate the bbox
        dilated_bbox = dilate_by_pixels(preprocessed_image, bbox, dilation_pixels)
        
        # Generate spatial TF-IDF mask for this region
        try:
            if collect_scores:
                # Extract scores for analysis
                tf_idf_map, region_mask, fixed_threshold = extract_spatial_tf_idf_scores(original_image, bbox)
                
                # Collect scores for global analysis
                GLOBAL_TF_IDF_SCORES.extend(tf_idf_map.flatten().tolist())
                GLOBAL_OTSU_THRESHOLDS.append(fixed_threshold)  # Keep same variable name for compatibility
                
                logger.info(f"Region {i+1} spatial mask: {np.sum(region_mask == 255)} pixels, Fixed threshold: {fixed_threshold}")
            else:
                # Use the spatial TF-IDF approach - returns binary mask directly
                region_mask = find_mask_by_spatial_tf_idf(original_image, bbox, debug=debug)
                logger.info(f"Region {i+1} spatial mask: {np.sum(region_mask == 255)} pixels")
            
            if np.sum(region_mask == 255) > 0:  # Only proceed if mask has content
                # Create full-sized mask positioned at the original bbox location
                full_mask = np.zeros(preprocessed_image.shape[:2], dtype=np.uint8)
                x, y, w, h = bbox
                full_mask[y:y+h, x:x+w] = region_mask
                
                regional_masks.append(full_mask)
                regions_processed += 1
                
                if debug:
                    # Save individual region mask for debugging
                    if output_dir:
                        region_mask_path = output_dir / f"{image_path.stem}_region_{i+1}_spatial_mask.png"
                        save_image(region_mask, region_mask_path)
                        logger.info(f"Saved region {i+1} mask: {region_mask_path}")
            else:
                logger.warning(f"Region {i+1} generated empty mask")
                
        except Exception as e:
            logger.error(f"Error processing region {i+1}: {e}")
            continue
    
    if not regional_masks:
        logger.warning("No regional masks generated")
        return {
            'success': False,
            'reason': 'No regional masks generated',
            'consensus_count': len(consensus_boxes),
            'regions_processed': 0
        }
    
    # Combine all regional masks
    logger.info(f"Combining {len(regional_masks)} spatial masks")
    combined_mask = regional_masks[0].copy()
    for mask in regional_masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply inpainting
    logger.info("Applying inpainting...")
    try:
        inpainted = cv2.inpaint(original_image, combined_mask, 3, cv2.INPAINT_TELEA)
        
        # Save results if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save inpainted result as high-quality JPEG
            result_path = output_dir / f"{image_path.stem}_spatial_tf_idf_result.jpg"
            cv2.imwrite(str(result_path), inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Saved result: {result_path}")
            
            # Save combined mask
            mask_path = output_dir / f"{image_path.stem}_spatial_tf_idf_mask.png"
            save_image(combined_mask, mask_path)
            logger.info(f"Saved mask: {mask_path}")
            
            # Save debug visualization if requested
            if debug:
                # Create visualization with bboxes
                vis_image = original_image.copy()
                for i, consensus in enumerate(consensus_boxes):
                    x, y, w, h = consensus['bbox']
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis_image, f"S{i+1}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                debug_path = output_dir / f"{image_path.stem}_spatial_tf_idf_debug.jpg"
                cv2.imwrite(str(debug_path), vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"Saved debug visualization: {debug_path}")
        
        return {
            'success': True,
            'consensus_count': len(consensus_boxes),
            'regions_processed': regions_processed,
            'mask_pixels': np.sum(combined_mask > 0)
        }
        
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")
        return {
            'success': False,
            'reason': f'Inpainting failed: {e}',
            'consensus_count': len(consensus_boxes),
            'regions_processed': regions_processed
        }

def extract_spatial_tf_idf_scores(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                 num_clusters: int = 24) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract raw TF-IDF scores from spatial analysis without applying morphological operations.
    
    This is a modified version of find_mask_by_spatial_tf_idf that returns the raw scores
    and Otsu threshold for analysis.
    
    Returns:
        Tuple of (tf_idf_map, binary_mask, otsu_threshold)
    """
    x, y, w, h = bbox
    image_h, image_w = image.shape[:2]
    
    # Calculate expanded region (1.414x to double the area)
    scale_factor = 1.414  # sqrt(2)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Center the expanded region on the original bbox center
    bbox_center_x = x + w // 2
    bbox_center_y = y + h // 2
    
    # Calculate ideal expanded bbox position
    ideal_new_x = bbox_center_x - new_w // 2
    ideal_new_y = bbox_center_y - new_h // 2
    
    # Adjust for image boundaries
    new_x = max(0, min(ideal_new_x, image_w - new_w))
    new_y = max(0, min(ideal_new_y, image_h - new_h))
    
    # Ensure we don't exceed image bounds
    actual_new_w = min(new_w, image_w - new_x)
    actual_new_h = min(new_h, image_h - new_y)
    
    # Extract regions
    bbox_region = image[y:y+h, x:x+w]  # The "document"
    expanded_region = image[new_y:new_y+actual_new_h, new_x:new_x+actual_new_w]
    
    # Create mask for the original bbox within the expanded region
    bbox_mask = np.zeros((actual_new_h, actual_new_w), dtype=bool)
    
    # Calculate bbox position within expanded region
    bbox_in_expanded_x = x - new_x
    bbox_in_expanded_y = y - new_y
    
    # Ensure bbox fits within expanded region
    bbox_end_x = min(bbox_in_expanded_x + w, actual_new_w)
    bbox_end_y = min(bbox_in_expanded_y + h, actual_new_h)
    bbox_start_x = max(0, bbox_in_expanded_x)
    bbox_start_y = max(0, bbox_in_expanded_y)
    
    bbox_mask[bbox_start_y:bbox_end_y, bbox_start_x:bbox_end_x] = True
    
    # Extract surrounding region (corpus) by excluding bbox area
    surrounding_region = expanded_region[~bbox_mask]
    
    # Combine both regions for k-means clustering
    bbox_pixels = bbox_region.reshape(-1, 3)
    surrounding_pixels = surrounding_region.reshape(-1, 3)
    all_pixels = np.vstack([bbox_pixels, surrounding_pixels])
    
    # Convert to RGB for k-means
    all_pixels_rgb = all_pixels[:, [2, 1, 0]]  # BGR to RGB
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        all_pixels_rgb.astype(np.float32), 
        num_clusters, 
        None, 
        criteria, 
        10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Split labels back into bbox and surrounding regions
    bbox_labels = labels[:len(bbox_pixels)]
    surrounding_labels = labels[len(bbox_pixels):]
    
    # Calculate TF (Term Frequency) for bbox region
    bbox_cluster_counts = np.bincount(bbox_labels.flatten(), minlength=num_clusters)
    bbox_total = len(bbox_pixels)
    tf_scores = bbox_cluster_counts / bbox_total if bbox_total > 0 else np.zeros(num_clusters)
    
    # Calculate IDF (Inverse Document Frequency) for surrounding region
    surrounding_cluster_counts = np.bincount(surrounding_labels.flatten(), minlength=num_clusters)
    surrounding_total = len(surrounding_pixels)
    
    # IDF calculation: log(total_pixels / (cluster_count + 1))
    # Adding 1 to avoid division by zero
    idf_scores = np.log(surrounding_total / (surrounding_cluster_counts + 1))
    
    # Calculate TF-IDF scores
    tf_idf_scores = tf_scores * idf_scores
    
    # Normalize TF-IDF scores to 0-255 range
    min_score = np.min(tf_idf_scores)
    max_score = np.max(tf_idf_scores)
    
    if max_score > min_score:
        normalized_scores = ((tf_idf_scores - min_score) / (max_score - min_score) * 255).astype(np.uint8)
    else:
        # If all scores are the same, set to middle gray
        normalized_scores = np.full(num_clusters, 128, dtype=np.uint8)
    
    # Create spatial TF-IDF map for the bbox region only
    actual_h, actual_w = bbox_region.shape[:2]
    tf_idf_map = np.zeros((actual_h, actual_w), dtype=np.uint8)
    bbox_labels_2d = bbox_labels.reshape(actual_h, actual_w)
    
    for cluster_id in range(num_clusters):
        cluster_mask = (bbox_labels_2d == cluster_id)
        tf_idf_map[cluster_mask] = normalized_scores[cluster_id]
    
    # Apply fixed threshold (188) to create binary mask
    # Scores are already normalized to 0-255, so 188 = slightly more permissive than top 25%
    fixed_threshold = 188
    binary_mask = (tf_idf_map >= fixed_threshold).astype(np.uint8) * 255
    
    return tf_idf_map, binary_mask, fixed_threshold

def generate_ascii_histogram(scores: List[float], title: str = "TF-IDF Score Distribution", 
                           bins: int = 20, width: int = 60) -> str:
    """Generate an ASCII histogram of the scores."""
    if not scores:
        return f"{title}: No data available"
    
    # Calculate histogram
    hist, bin_edges = np.histogram(scores, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    
    # Generate ASCII representation
    lines = [f"\n{title}"]
    lines.append("=" * len(title))
    lines.append(f"Total samples: {len(scores):,}")
    lines.append(f"Range: {min(scores):.3f} to {max(scores):.3f}")
    lines.append(f"Mean: {np.mean(scores):.3f}, Median: {np.median(scores):.3f}")
    lines.append(f"Std Dev: {np.std(scores):.3f}")
    lines.append("")
    
    # Histogram bars
    for i in range(bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = hist[i]
        
        # Calculate bar length
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        bar = "#" * bar_length
        
        # Format the line
        lines.append(f"{bin_start:6.2f}-{bin_end:6.2f} |{bar:<{width}} {count:>6}")
    
    lines.append("")
    
    # Percentile analysis
    percentiles = [5, 10, 15, 20, 25, 30, 33.33, 35, 40, 45, 50, 55, 60, 65, 66.67, 70, 75, 80, 85, 90, 95]
    lines.append("Percentiles:")
    for p in percentiles:
        value = np.percentile(scores, p)
        lines.append(f"  {p:2d}th: {value:.3f}")
    
    return "\n".join(lines)

def generate_threshold_analysis(scores: List[float], fixed_thresholds: List[float]) -> str:
    """Generate analysis comparing different threshold strategies."""
    if not scores or not fixed_thresholds:
        return "Threshold Analysis: No data available"
    
    lines = ["\nThreshold Analysis"]
    lines.append("=" * 18)
    
    # Convert normalized scores back to 0-255 range for comparison
    score_array = np.array(scores)
    
    # Current fixed threshold statistics
    lines.append(f"\nCurrent Fixed Threshold (n={len(fixed_thresholds)} regions):")
    lines.append(f"  Value: {fixed_thresholds[0]} (consistent across all regions)")
    current_threshold = fixed_thresholds[0]
    pct_above_current = np.mean(score_array >= current_threshold) * 100
    lines.append(f"  Pixels above threshold: {pct_above_current:.1f}%")
    
    # Test different fixed thresholds
    test_thresholds = [64, 96, 128, 160, 192, 224]
    lines.append(f"\nAlternative Threshold Analysis:")
    lines.append(f"{'Threshold':<10} {'% Above':<8} {'Description':<20}")
    lines.append("-" * 40)
    
    for threshold in test_thresholds:
        pct_above = np.mean(score_array >= threshold) * 100
        if threshold == 64:
            desc = "Very permissive"
        elif threshold == 96:
            desc = "Permissive"
        elif threshold == 128:
            desc = "Moderate"
        elif threshold == 160:
            desc = "Conservative"
        elif threshold == 192:
            desc = "Very conservative"
        elif threshold == 224:
            desc = "Extremely conservative"
        else:
            desc = ""
            
        marker = " ← CURRENT" if threshold == current_threshold else ""
        lines.append(f"{threshold:<10} {pct_above:<8.1f} {desc:<20}{marker}")
    
    return "\n".join(lines)

def main():
    """Main entry point for TF-IDF text isolation."""
    parser = argparse.ArgumentParser(description="Text isolation using TF-IDF color analysis")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--output", default="results", help="Output directory (default: results)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for detectors (default: 0.3)")
    parser.add_argument("--overlap-threshold", type=float, default=0.1,
                       help="Overlap threshold for consensus (default: 0.1)")
    parser.add_argument("--dilation-pixels", type=int, default=10,
                       help="Pixels to dilate consensus regions (default: 10)")
    parser.add_argument("--spatial", action="store_true", 
                       help="Use spatial TF-IDF with Otsu thresholding (experimental)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--collect-scores", action="store_true", help="Collect TF-IDF scores")
    parser.add_argument("--histogram", action="store_true", help="Generate score histograms")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize models
    print("Initializing detection models...")
    initialize_models(['doctr', 'easyocr'])  # EAST is loaded on-demand
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # Single image
        if args.spatial:
            result = process_image_with_spatial_tf_idf(input_path, args.confidence_threshold, 
                                                      args.overlap_threshold, args.dilation_pixels,
                                                      output_dir, args.debug, args.collect_scores)
        else:
            result = process_image_with_tf_idf(input_path, args.confidence_threshold, 
                                              args.overlap_threshold, args.dilation_pixels,
                                              output_dir, args.debug)
        
        if result['success']:
            print(f"✓ Successfully processed {input_path.name}")
            print(f"  - Consensus regions: {result['consensus_count']}")
            print(f"  - Regions processed: {result['regions_processed']}")
            print(f"  - Mask pixels: {result['mask_pixels']}")
        else:
            print(f"✗ Failed to process {input_path.name}: {result['reason']}")
    
    elif input_path.is_dir():
        # Directory of images
        from untextre.utils import get_image_files
        image_files = get_image_files(input_path)
        
        if not image_files:
            logger.error(f"No image files found in {input_path}")
            return
        
        print(f"Processing {len(image_files)} images...")
        successful = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                if args.spatial:
                    result = process_image_with_spatial_tf_idf(image_file, args.confidence_threshold,
                                                              args.overlap_threshold, args.dilation_pixels, 
                                                              output_dir, args.debug, args.collect_scores)
                else:
                    result = process_image_with_tf_idf(image_file, args.confidence_threshold,
                                                      args.overlap_threshold, args.dilation_pixels, 
                                                      output_dir, args.debug)
                
                if result['success']:
                    print(f"  ✓ Success - regions: {result['consensus_count']}, "
                          f"processed: {result['regions_processed']}, "
                          f"mask pixels: {result['mask_pixels']}")
                    successful += 1
                else:
                    print(f"  ✗ Failed: {result['reason']}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                print(f"  ✗ Error: {e}")
        
        print(f"\nCompleted: {successful}/{len(image_files)} images processed successfully")
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
    
    # Generate histograms if requested and scores were collected
    if args.collect_scores and (args.histogram or args.spatial):
        if GLOBAL_TF_IDF_SCORES:
            print("\n" + "="*80)
            print("SPATIAL TF-IDF SCORE ANALYSIS")
            print("="*80)
            
            # Generate and display histogram
            histogram = generate_ascii_histogram(GLOBAL_TF_IDF_SCORES, 
                                               "Spatial TF-IDF Score Distribution (0-255)", 
                                               bins=25, width=50)
            print(histogram)
            
            # Generate threshold analysis
            if GLOBAL_OTSU_THRESHOLDS:
                threshold_analysis = generate_threshold_analysis(GLOBAL_TF_IDF_SCORES, GLOBAL_OTSU_THRESHOLDS)
                print(threshold_analysis)
            
            # Save histogram to file if output directory exists
            if output_dir:
                histogram_file = output_dir / "tf_idf_score_analysis.txt"
                with open(histogram_file, 'w', encoding='utf-8') as f:
                    f.write("SPATIAL TF-IDF SCORE ANALYSIS\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Generated from {len(GLOBAL_OTSU_THRESHOLDS)} regions across processed images\n\n")
                    f.write(histogram)
                    if GLOBAL_OTSU_THRESHOLDS:
                        f.write("\n" + threshold_analysis)
                print(f"\nScore analysis saved to: {histogram_file}")
        else:
            print("\nNo TF-IDF scores collected. Use --collect-scores with --spatial to gather data.")

if __name__ == "__main__":
    main() 