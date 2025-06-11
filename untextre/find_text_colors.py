"""Color analysis module for finding text colors in images.

This module analyzes colors within detected text regions to identify the colors
that represent text watermarks. It uses k-means quantization to reduce the
color space and then selects the most appropriate colors based on either
target matching or "grayness" criteria.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import easyocr

from .utils import ImageArray, BBox, Color, setup_logger, color_distance
from .detector import TextDetector, detect_text_regions

logger = setup_logger(__name__)

def find_text_colors(
    image: ImageArray, 
    bbox: BBox, 
    target_color: Optional[Color] = None,
    detector_method: str = "doctr"
) -> Tuple[List[Color], List[Color]]:
    """Find text colors within a bounding box using color quantization.
    
    This is the main entry point for color analysis. It quantizes colors
    within the bounding box and selects the 2 most appropriate colors based
    on either target matching or "grayness" criteria.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height) to analyze
        target_color: Optional target BGR color to match against
        detector_method: Detection method to use (kept for API compatibility, not used)
        
    Returns:
        Tuple of:
        - List of 2 representative BGR color tuples (the quantized colors)
        - List of original BGR colors that were assigned to these quantized colors
    """
    # Quantize to 12 colors and select the 2 best matches
    return _get_most_common_color(image, bbox, num_colors=12, target_color=target_color)

def hex_to_bgr(hex_color: str) -> Color:
    """Convert hex color string to BGR tuple.
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
        
    Returns:
        BGR color tuple
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Parse hex values
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to BGR
    return (rgb[2], rgb[1], rgb[0])

def html_to_bgr(html_color: str) -> Color:
    """Convert HTML color name to BGR tuple.
    
    Args:
        html_color: HTML color name (e.g., "red", "ghostwhite")
        
    Returns:
        BGR color tuple
    """
    # Convert HTML name to hex using PIL
    from PIL import ImageColor
    rgb = ImageColor.getrgb(html_color)
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def _create_color_mask_for_ocr(image: ImageArray, bbox: BBox, target_color: Color, tolerance: int = 2) -> ImageArray:
    """Create a black-on-white mask for OCR analysis of a specific color.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height) to analyze
        target_color: BGR color to create mask for
        tolerance: Color matching tolerance
        
    Returns:
        Black-on-white mask as H×W×3 BGR uint8 numpy array for DocTR detection
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    # Create color bounds with tolerance
    lower = np.array([max(0, c - tolerance) for c in target_color])
    upper = np.array([min(255, c + tolerance) for c in target_color])
    
    # Create binary mask for this color
    color_mask = cv2.inRange(roi, lower, upper)
    
    # Create black-on-white image: text pixels are black (0), background is white (255)
    # Invert the mask so text appears black
    ocr_mask = 255 - color_mask
    
    # Convert to 3-channel BGR for DocTR (it expects BGR input)
    ocr_image = cv2.cvtColor(ocr_mask, cv2.COLOR_GRAY2BGR)
    
    # Resize back to original bbox size to create full-sized image for detection
    full_mask = np.full(image.shape, 255, dtype=np.uint8)  # White background
    full_mask[y:y+h, x:x+w] = ocr_image
    
    return full_mask

def _detect_with_doctr_for_ocr(image: ImageArray) -> List[Tuple[int, int, int, int, float]]:
    """Run DocTR detection and return results in a consistent format for OCR analysis.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        List of (x, y, width, height, confidence) tuples
    """
    try:
        detector = TextDetector(confidence_threshold=0.1, min_text_size=3)
        detections = detector.detect(image)
        
        results = []
        for detection in detections:
            # Extract geometry and convert to bbox
            geometry = detection['geometry']
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            
            # Get confidence score
            confidence = detection.get('confidence', 0.5) * 100  # Convert to percentage
            
            results.append((x, y, w, h, confidence))
            
        return results
        
    except Exception as e:
        logger.warning(f"DocTR detection failed: {e}")
        return []

def _detect_with_easyocr_for_ocr(image: ImageArray) -> List[Tuple[int, int, int, int, float]]:
    """Run EasyOCR detection and return results in a consistent format for OCR analysis.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        List of (x, y, width, height, confidence) tuples
    """
    try:
        # Initialize EasyOCR reader (could be cached globally for efficiency)
        reader = easyocr.Reader(['en'], verbose=False)
        
        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results_raw = reader.readtext(rgb_image)
        
        results = []
        for bbox_points, text, confidence in results_raw:
            # EasyOCR returns bbox as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Convert to bounding box
            bbox_array = np.array(bbox_points)
            x_coords = bbox_array[:, 0]
            y_coords = bbox_array[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            
            # EasyOCR confidence is already 0-1, convert to percentage
            confidence_pct = confidence * 100
            
            results.append((x, y, w, h, confidence_pct))
            
        return results
        
    except Exception as e:
        logger.warning(f"EasyOCR detection failed: {e}")
        return []

def _detect_with_east_for_ocr(image: ImageArray) -> List[Tuple[int, int, int, int, float]]:
    """Run EAST detection and return results in a consistent format for OCR analysis.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        List of (x, y, width, height, confidence) tuples
    """
    try:
        # Import EAST detector components from main detector module
        from .detector import _load_east_model, _detect_with_east
        
        # Load EAST model (this will be cached globally)
        net = _load_east_model()
        
        # Run EAST detection with standard parameters
        detections = _detect_with_east(image, net, min_confidence=0.3)
        
        results = []
        for detection in detections:
            # Extract geometry and convert to bbox
            geometry = detection['geometry']
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            
            # Get confidence score (already in 0-1 range)
            confidence = detection.get('confidence', 0.5) * 100  # Convert to percentage
            
            results.append((x, y, w, h, confidence))
            
        return results
        
    except Exception as e:
        logger.warning(f"EAST detection failed: {e}")
        return []

def _analyze_color_with_ocr(image: ImageArray, bbox: BBox, color: Color, detector_method: str = "doctr") -> Tuple[float, float]:
    """Analyze a quantized color with text detection to get coverage and confidence metrics.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height) to analyze
        color: BGR color to analyze
        detector_method: Detection method to use ("doctr", "easyocr", or "east")
        
    Returns:
        Tuple of (ocr_coverage_percent, ocr_confidence_percent)
    """
    try:
        # Create black-on-white mask for this color
        ocr_image = _create_color_mask_for_ocr(image, bbox, color)
        
        # Run detection using the appropriate method
        if detector_method.lower() == "doctr":
            detections = _detect_with_doctr_for_ocr(ocr_image)
        elif detector_method.lower() == "easyocr":
            detections = _detect_with_easyocr_for_ocr(ocr_image)
        elif detector_method.lower() == "east":
            detections = _detect_with_east_for_ocr(ocr_image)
        else:
            logger.warning(f"Unknown detector method: {detector_method}")
            return 0.0, 0.0
        
        if not detections:
            # No text detected in this color mask
            return 0.0, 0.0
        
        # Calculate coverage and confidence from detections
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        bbox_area = bbox_w * bbox_h
        
        total_detected_area = 0
        confidences = []
        
        # Process each detection: (x, y, width, height, confidence)
        for det_x, det_y, det_w, det_h, det_confidence in detections:
            detected_area = det_w * det_h
            
            # Only count detections that overlap with our original bbox
            if (det_x < bbox_x + bbox_w and det_x + det_w > bbox_x and 
                det_y < bbox_y + bbox_h and det_y + det_h > bbox_y):
                total_detected_area += detected_area
                confidences.append(det_confidence)
        
        # Calculate coverage as percentage
        coverage_percent = min(100.0, (total_detected_area / bbox_area) * 100)
        
        # Calculate average confidence
        if confidences:
            avg_confidence = np.mean(confidences)
        else:
            avg_confidence = 0.0
            
        return coverage_percent, avg_confidence
        
    except Exception as e:
        logger.warning(f"{detector_method.upper()} analysis failed for color {color}: {e}")
        return 0.0, 0.0

def _get_most_common_color(
    image: ImageArray, 
    bbox: BBox, 
    num_colors: int = 12, 
    target_color: Optional[Color] = None
) -> Tuple[List[Color], List[Color]]:
    """Get the most common colors in the specified region, using color quantization.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height)
        num_colors: Number of colors to quantize to (default: 12)
        target_color: Optional target BGR color to match against
        
    Returns:
        Tuple of:
        - List of 2 representative BGR color tuples (the quantized colors)
        - List of original BGR colors that were assigned to these quantized colors
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    # Convert BGR to LAB color space for perceptually uniform quantization
    # LAB space is designed to match human vision - equal distances in LAB 
    # correspond to equal perceived color differences. This is crucial for text
    # detection because:
    # 1. The L channel isolates brightness, perfect for text/background contrast
    # 2. Anti-aliased text edges cluster more naturally
    # 3. Gray watermarks (common case) are better discriminated
    # 4. Background color variations are handled more robustly
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # Reshape ROI to 2D array of LAB pixels
    lab_pixels = lab_roi.reshape(-1, 3)
    
    # Convert to float32 for k-means and normalize channels for equal weighting
    # LAB channels have different ranges: L(0-100), A(-127 to +127), B(-127 to +127)
    # We normalize to give equal weight to all channels in the clustering
    lab_pixels_float = lab_pixels.astype(np.float32)
    lab_pixels_float[:, 0] /= 100.0  # Normalize L channel from 0-100 to 0-1
    lab_pixels_float[:, 1] = (lab_pixels_float[:, 1] + 127) / 254.0  # Normalize A channel to 0-1
    lab_pixels_float[:, 2] = (lab_pixels_float[:, 2] + 127) / 254.0  # Normalize B channel to 0-1
    
    # Define criteria and apply k-means clustering in LAB space
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, lab_centers = cv2.kmeans(lab_pixels_float, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Denormalize the LAB cluster centers back to original LAB ranges
    lab_centers[:, 0] *= 100.0  # Denormalize L channel back to 0-100
    lab_centers[:, 1] = lab_centers[:, 1] * 254.0 - 127  # Denormalize A channel back to -127 to +127
    lab_centers[:, 2] = lab_centers[:, 2] * 254.0 - 127  # Denormalize B channel back to -127 to +127
    
    # Convert LAB cluster centers back to BGR color space for compatibility with rest of pipeline
    lab_centers_uint8 = lab_centers.astype(np.uint8)
    lab_centers_reshaped = lab_centers_uint8.reshape(1, -1, 3)  # Shape needed for cv2.cvtColor
    bgr_centers_reshaped = cv2.cvtColor(lab_centers_reshaped, cv2.COLOR_LAB2BGR)
    centers = bgr_centers_reshaped.reshape(-1, 3)  # Back to (num_colors, 3) shape
    
    # Count occurrences of each color cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Get top colors by frequency (up to num_colors)
    top_indices = np.argsort(counts)[-num_colors:]
    top_colors = centers[unique_labels[top_indices]]
    top_counts = counts[top_indices]
    
    # Calculate variance between R,G,B channels for each color (in BGR space for interpretability)
    variances = np.var(top_colors, axis=1)
    
    # Log the top colors and their properties
    logger.info(f"\nTop {len(top_colors)} colors by frequency (quantized in LAB space):")
    logger.info("Rank  BGR Color    Variance    Count    % Pixels")
    logger.info("------------------------------------------------")
    for i, (color, var, count) in enumerate(zip(top_colors, variances, top_counts)):
        percent = (count / len(lab_pixels)) * 100
        logger.info(f"{i+1:2d}    {tuple(color)}    {var:8.2f}    {count:6d}    {percent:6.1f}%")
    
    if target_color is not None:
        # Find the 2 closest colors to the target (in BGR space for simplicity)
        target_float = np.float32([target_color])
        distances = np.linalg.norm(top_colors - target_float, axis=1)
        closest_indices = np.argsort(distances)[:2]  # Get 2 closest
        selected_colors = [tuple(top_colors[idx]) for idx in closest_indices]
        selected_label_indices = closest_indices
        logger.info(f"\nTarget color: {target_color}")
        logger.info(f"2 closest matches:")
        for i, (color, idx) in enumerate(zip(selected_colors, closest_indices)):
            logger.info(f"  {i+1}. {color} (distance: {distances[idx]:.2f})")
    else:
        # Find the 2 "grayest" colors among top colors (lowest RGB variance = most neutral)
        grayest_indices = np.argsort(variances)[:2]  # Get 2 grayest
        selected_colors = [tuple(top_colors[idx]) for idx in grayest_indices]
        selected_label_indices = grayest_indices
        logger.info(f"\n2 grayest colors selected:")
        for i, (color, idx) in enumerate(zip(selected_colors, grayest_indices)):
            logger.info(f"  {i+1}. {color} with variance {variances[idx]:.2f}")
    
    # Get all original BGR colors that were assigned to these quantized color clusters
    all_original_colors = []
    for selected_label_idx in selected_label_indices:
        selected_label = unique_labels[top_indices[selected_label_idx]]
        original_bgr_colors = roi.reshape(-1, 3)[labels.flatten() == selected_label]
        all_original_colors.extend(original_bgr_colors)
    
    # Convert back to array and get unique colors
    original_bgr_colors = np.array(all_original_colors)
    
    # Count occurrences of each original BGR color
    unique_colors, color_counts = np.unique(original_bgr_colors, axis=0, return_counts=True)
    
    # Filter out colors that appear in fewer than MIN_PIXELS pixels
    # TODO: Make minimum pixel threshold configurable
    MIN_PIXELS = 4
    mask = color_counts >= MIN_PIXELS
    filtered_colors = unique_colors[mask]
    filtered_counts = color_counts[mask]
    
    # Log color filtering results
    logger.info(f"Found {len(unique_colors)} original colors in selected cluster")
    logger.info(f"After filtering colors with < {MIN_PIXELS} pixels: {len(filtered_colors)} colors remain")
    
    # Log the distribution of remaining colors
    logger.info("\nColor frequency distribution after filtering:")
    logger.info("Count Range    Number of Colors")
    logger.info("-------------------------------")
    ranges = [(10, 50), (51, 100), (101, 500), (501, 1000), (1001, None)]
    for start, end in ranges:
        if end is None:
            count = np.sum(filtered_counts >= start)
            logger.info(f"{start:4d}+          {count:4d}")
        else:
            count = np.sum((filtered_counts >= start) & (filtered_counts < end))
            logger.info(f"{start:4d}-{end:4d}     {count:4d}")
    
    return selected_colors, [tuple(color) for color in filtered_colors] 