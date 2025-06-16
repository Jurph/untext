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

def _simple_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Simple linear regression using numpy - no sklearn needed!
    
    Args:
        x: X coordinates as 1D array
        y: Y coordinates as 1D array
        
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    # Calculate slope and intercept using least squares formulas
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Slope: m = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        slope = 0.0
        intercept = y_mean
        r_squared = 0.0
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R² = 1 - (SS_res / SS_tot)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared

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
        detector = TextDetector(confidence_threshold=0.3, min_text_size=3)
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
            # Skip detections below confidence threshold (0.3)
            if confidence < 0.3:
                continue
                
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
    num_colors: int = 8, 
    target_color: Optional[Color] = None
) -> Tuple[List[Color], List[Color]]:
    """Get the most common colors in the specified region, using color quantization.
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height)
        num_colors: Number of colors to quantize to (default: 8)
        target_color: Optional target BGR color to match against
        
    Returns:
        Tuple of:
        - List of 1 representative BGR color tuple (the best quantized color)
        - List of original BGR colors that were assigned to this quantized color
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
    _, kmeans_labels, lab_centers = cv2.kmeans(lab_pixels_float, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
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
    unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
    
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
        # Find the closest color to the target (in BGR space for simplicity)
        target_float = np.float32([target_color])
        distances = np.linalg.norm(top_colors - target_float, axis=1)
        closest_index = np.argsort(distances)[0]  # Get 1 closest
        selected_colors = [tuple(top_colors[closest_index])]
        selected_label_indices = [closest_index]
        logger.info(f"\nTarget color: {target_color}")
        logger.info(f"Closest match: {selected_colors[0]} (distance: {distances[closest_index]:.2f})")
    else:
        # Find the "grayest" color among top colors (lowest RGB variance = most neutral)
        grayest_index = np.argsort(variances)[0]  # Get 1 grayest
        selected_colors = [tuple(top_colors[grayest_index])]
        selected_label_indices = [grayest_index]
        logger.info(f"\nGrayest color selected: {selected_colors[0]} with variance {variances[grayest_index]:.2f}")
    
    # Get all original BGR colors that were assigned to these quantized color clusters
    all_original_colors = []
    for selected_label_idx in selected_label_indices:
        selected_label = unique_labels[top_indices[selected_label_idx]]
        original_bgr_colors = roi.reshape(-1, 3)[kmeans_labels.flatten() == selected_label]
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

def exhaustive_color_search(image: ImageArray, bbox: BBox) -> Tuple[List[Color], List[Color]]:
    """Exhaustively search for the best text colors using systematic quantization and detection.
    
    This function implements a thorough approach to finding text colors by:
    1. Systematically trying quantization levels from 4 to 16 colors
    2. For each quantized color, creating a black-on-pale-gray test image
    3. Running EAST detection to measure text detection confidence
    4. Stopping when confidence > 0.5 is achieved, or trying all quantization levels
    5. Returning the color set that achieved the highest detection confidence
    
    Args:
        image: Input image in BGR format
        bbox: Bounding box (x, y, width, height) to analyze
        
    Returns:
        Tuple of:
        - List of representative BGR color tuples (the best quantized colors)
        - List of original BGR colors that match the best quantized colors
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    logger.info(f"Starting exhaustive color search on {w}×{h} region...")
    
    best_confidence = 0.0
    best_colors = []
    best_original_colors = []
    best_quantization = 0
    best_color_index = 0
    best_component_count = 0  # Start with 0 so first result is always better
    best_test_image = None
    best_analysis_debug = {"overall_score": 0.0}  # Initialize with default  # Store the best test image for debugging
    
    # Try quantization levels from 2 to 16 colors
    for num_colors in range(2, 17):
        logger.debug(f"Testing quantization with {num_colors} colors...")
        
        # Quantize colors in LAB space (same as our main pipeline)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lab_pixels = lab_roi.reshape(-1, 3).astype(np.float32)
        
        # Normalize LAB channels for k-means
        lab_pixels[:, 0] /= 100.0  # L: 0-100 → 0-1
        lab_pixels[:, 1] = (lab_pixels[:, 1] + 127) / 254.0  # A: -127,+127 → 0-1
        lab_pixels[:, 2] = (lab_pixels[:, 2] + 127) / 254.0  # B: -127,+127 → 0-1
        
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, kmeans_labels, lab_centers = cv2.kmeans(lab_pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Denormalize LAB centers back to original ranges
        lab_centers[:, 0] *= 100.0
        lab_centers[:, 1] = lab_centers[:, 1] * 254.0 - 127
        lab_centers[:, 2] = lab_centers[:, 2] * 254.0 - 127
        
        # Convert LAB centers to BGR for testing
        lab_centers_uint8 = lab_centers.astype(np.uint8)
        lab_centers_bgr = []
        for center in lab_centers_uint8:
            # Convert single LAB pixel to BGR
            lab_pixel = center.reshape(1, 1, 3)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            lab_centers_bgr.append(tuple(bgr_pixel[0, 0]))
        
        # Test each quantized color
        for color_idx, test_color in enumerate(lab_centers_bgr):
            logger.debug(f"  Testing color {color_idx+1}/{num_colors}: {test_color}")
            
            # Create black-on-pale-gray test image
            test_image = _create_exhaustive_test_image(roi, test_color, kmeans_labels, color_idx)
            
            # Run EAST detection on the test image
            confidence = _test_color_with_east(test_image)
            
            logger.debug(f"    Confidence: {confidence:.3f}")
            
            # Track the best result (prefer higher confidence, then more connected components for ties)
            # More components = more likely to be separate letters rather than contiguous background
            pixels_in_cluster = np.sum(kmeans_labels.flatten() == color_idx)
            
            # Count connected components in this test image to measure fragmentation
            test_binary = (test_image[:, :, 0] == 0).astype(np.uint8)  # Black pixels = 1
            
            # Aggressively clean up speckles and noise before counting components
            test_binary_clean = test_binary.copy()
            
            # Repeat cleanup process 3 times to eliminate stubborn noise
            for cleanup_round in range(3):
                # 1. Opening with larger kernel to remove small noise
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                test_binary_clean = cv2.morphologyEx(test_binary_clean, cv2.MORPH_OPEN, kernel_open)
                
                # 2. Additional erosion to separate touching components
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                test_binary_clean = cv2.erode(test_binary_clean, kernel_erode)
                
                # 3. Small dilation to restore size (but keep separation)
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                test_binary_clean = cv2.dilate(test_binary_clean, kernel_dilate)
            
            num_components, component_labels = cv2.connectedComponents(test_binary_clean)
            
            # Filter components based on text-like geometric properties
            text_like_components = 0
            component_areas = []
            component_centroids = []
            component_masks = []
            
            for label in range(1, num_components):  # Skip background (label 0)
                # Get component mask
                component_mask = (component_labels == label)
                area = np.sum(component_mask)
                
                # Filter by size - text components should be reasonably sized
                if area < 10 or area > 5000:  # Too small (noise) or too large (not individual letters)
                    continue
                
                # Calculate centroid
                y_coords, x_coords = np.where(component_mask)
                if len(y_coords) > 0:
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    
                    # Calculate bounding box for aspect ratio check
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    width = max_x - min_x + 1
                    height = max_y - min_y + 1
                    
                    # Text components should have reasonable aspect ratios (not too thin/wide)
                    if height > 0 and width > 0:
                        aspect_ratio = width / height
                        if 0.1 <= aspect_ratio <= 10:  # Reasonable aspect ratio for letters
                            component_areas.append(area)
                            component_centroids.append((centroid_x, centroid_y))
                            text_like_components += 1
                            component_masks.append(component_mask)
            
            # Use sophisticated text-like properties analysis
            if len(component_centroids) >= 2:
                text_likelihood_score, analysis_debug = _analyze_text_like_properties(
                    component_centroids, component_areas, component_masks, bbox
                )
                
                # Convert score to component count for compatibility with existing logic
                # High-scoring components get boosted, low-scoring get penalized
                if text_likelihood_score >= 0.7:
                    component_count = min(text_like_components * 2, text_like_components + 10)
                elif text_likelihood_score >= 0.5:
                    component_count = text_like_components
                else:
                    component_count = max(1, text_like_components // 2)
                
                logger.debug(f"    Text analysis: score={text_likelihood_score:.3f}, "
                           f"components={text_like_components}→{component_count}")
                logger.debug(f"    Analysis details: {analysis_debug}")
                
                # Optionally save visualization for debugging best candidates
                if text_likelihood_score >= 0.6:  # Only visualize promising candidates
                    debug_image_path = f"text_analysis_debug_{num_colors}c_{color_idx}.png"
                    _visualize_text_analysis(test_image, component_labels, 
                                           component_centroids, analysis_debug, debug_image_path)
            else:
                # Fallback to old simple logic for insufficient components
                component_count = text_like_components
                logger.debug(f"    Insufficient components for analysis, using simple count: {component_count}")
            
            # Improved scoring that combines EAST confidence with text analysis
            combined_score = confidence
            current_analysis_debug = {"overall_score": 0.0}
            if len(component_centroids) >= 2:
                # Boost score based on text-like properties
                combined_score = confidence * (1.0 + text_likelihood_score * 0.5)
                current_analysis_debug = analysis_debug
                
            best_combined_score = best_confidence * (1.0 + best_analysis_debug.get("overall_score", 0) * 0.5)
            is_better = (combined_score > best_combined_score or
                        (abs(combined_score - best_combined_score) < 0.01 and 
                         component_count > best_component_count))
            
            if is_better:
                best_confidence = confidence
                best_quantization = num_colors
                best_color_index = color_idx
                best_component_count = component_count
                best_analysis_debug = current_analysis_debug
                
                # Get original BGR colors that map to this quantized color cluster
                # Break down the dense line for debugging
                
                # Step 1: Reshape ROI from (H,W,3) to (H*W, 3) - each row is one pixel's BGR values
                roi_pixels = roi.reshape(-1, 3)
                logger.debug(f"    ROI shape: {roi.shape} -> reshaped to {roi_pixels.shape}")
                
                # Step 2: Flatten labels to 1D array (should already be 1D from k-means)
                labels_1d = kmeans_labels.flatten()
                logger.debug(f"    Labels shape: {kmeans_labels.shape} -> flattened to {labels_1d.shape}")
                
                # Step 3: Create boolean mask for pixels belonging to this cluster
                cluster_mask = (labels_1d == color_idx)
                pixels_in_cluster = np.sum(cluster_mask)
                logger.debug(f"    Cluster {color_idx}: {pixels_in_cluster} pixels belong to this cluster")
                
                # Step 4: Extract the original BGR colors for pixels in this cluster
                original_bgr_colors = roi_pixels[cluster_mask]
                logger.debug(f"    Extracted {len(original_bgr_colors)} BGR color values")
                
                # Step 5: Get unique original colors only (remove duplicates)
                unique_original_colors = np.unique(original_bgr_colors, axis=0)
                logger.debug(f"    Unique colors: {len(unique_original_colors)} (after removing duplicates)")
                
                # Step 6: Show sample colors for verification
                if len(unique_original_colors) > 0:
                    sample_colors = unique_original_colors[:5]  # Show first 5
                    logger.debug(f"    Sample colors: {[tuple(c) for c in sample_colors]}")
                
                best_original_colors = [tuple(color) for color in unique_original_colors]
                
                # Use the representative test color for display, but actual originals for masking
                best_colors = [test_color]
                best_test_image = test_image.copy()  # Save the best test image
                
                logger.info(f"New best result: {confidence:.3f} confidence with color {test_color} "
                           f"(quantization: {num_colors}, color {color_idx+1}/{num_colors}, "
                           f"{len(original_bgr_colors)} pixels, {component_count} components)")
        
        # Early exit if we found high confidence (but only after testing all colors in this quantization level)
        if best_confidence > 0.975:
            logger.info(f"High confidence achieved ({best_confidence:.3f}), stopping search after completing {num_colors}-color quantization")
            break
    
    # Save the best test image for debugging
    if best_test_image is not None:
        cv2.imwrite("best_color_mask.png", best_test_image)
        logger.debug(f"Saved best test image to best_color_mask.png")
    
    # Final result summary
    logger.info(f"Exhaustive search complete:")
    logger.info(f"  Best confidence: {best_confidence:.3f}")
    logger.info(f"  Best quantization: {best_quantization} colors")
    logger.info(f"  Best color: {best_colors[0] if best_colors else 'None'}")
    logger.info(f"  Original colors found: {len(best_original_colors)}")
    
    if not best_colors:
        logger.warning("No colors found with measurable confidence, falling back to darkest color")
        # Fallback: return the darkest color in the region
        gray_values = np.mean(roi.reshape(-1, 3), axis=1)
        darkest_idx = np.argmin(gray_values)
        darkest_color = tuple(roi.reshape(-1, 3)[darkest_idx])
        return [darkest_color], [darkest_color]
    
    return best_colors, best_original_colors

def _create_exhaustive_test_image(roi: ImageArray, target_color: Color, labels: np.ndarray, target_label: int) -> ImageArray:
    """Create a black-on-pale-gray test image for a specific quantized color.
    
    Args:
        roi: Region of interest as H×W×3 BGR array
        target_color: The color to test (BGR tuple)
        labels: K-means cluster labels for each pixel
        target_label: The cluster label to make black
        
    Returns:
        Test image where target color is black, everything else is pale gray
    """
    h, w, c = roi.shape
    
    # Create pale gray background (192, 192, 192)
    test_image = np.full((h, w, 3), 192, dtype=np.uint8)
    
    # Make pixels belonging to target cluster black
    labels_2d = labels.reshape(h, w)
    mask = (labels_2d == target_label)
    test_image[mask] = [0, 0, 0]  # Black
    
    return test_image

def _test_color_with_east(test_image: ImageArray) -> float:
    """Test a color pattern using EAST detection and return maximum confidence.
    
    Args:
        test_image: Black-on-pale-gray test image
        
    Returns:
        Maximum confidence score from EAST detection (0.0 if no detections)
    """
    try:
        # Import EAST detector components
        from .detector import _load_east_model, _detect_with_east
        
        # Load EAST model (cached globally)
        net = _load_east_model()
        
        # Run EAST detection with lower confidence threshold for testing
        detections = _detect_with_east(test_image, net, min_confidence=0.1)
        
        if not detections:
            return 0.0
        
        # Return the maximum confidence found
        max_confidence = max(det.get('confidence', 0.0) for det in detections)
        return max_confidence
        
    except Exception as e:
        logger.warning(f"EAST testing failed: {e}")
        return 0.0

def _analyze_text_like_properties(component_centroids: List[Tuple[float, float]], 
                                 component_areas: List[float],
                                 component_masks: List[np.ndarray],
                                 bbox: Optional[BBox] = None) -> Tuple[float, dict]:
    """Analyze a set of connected components for text-like properties.
    
    Args:
        component_centroids: List of (x, y) centroid coordinates
        component_areas: List of component areas
        component_masks: List of component binary masks for convexity analysis
        bbox: Optional bounding box (x, y, width, height) to determine text orientation
        
    Returns:
        Tuple of (text_likelihood_score, debug_info_dict)
        Score is 0-1 where 1 is most text-like
    """
    if len(component_centroids) < 2:
        return 0.0, {"reason": "insufficient_components", "count": len(component_centroids)}
    
    # Determine text orientation based on bounding box aspect ratio
    is_vertical_text = False
    if bbox is not None:
        _, _, width, height = bbox
        if height > width:  # Taller than wide = vertical text
            is_vertical_text = True
    
    debug_info = {
        "component_count": len(component_centroids),
        "centroids": component_centroids[:10],  # First 10 for logging
        "areas": component_areas[:10],
        "is_vertical_text": is_vertical_text
    }
    
    scores = []
    
    # 1. COLINEARITY ANALYSIS
    # Check if centroids form a line (text baseline)
    centroids_array = np.array(component_centroids)
    x_coords = centroids_array[:, 0]
    y_coords = centroids_array[:, 1]
    
    # Fit a line to the centroids
    if len(centroids_array) >= 2:
        # Use our simple numpy-based linear regression
        slope, intercept, r_squared = _simple_linear_regression(x_coords, y_coords)
        
        # Calculate predicted y values and residuals
        y_pred = slope * x_coords + intercept
        residuals = np.abs(y_coords - y_pred)
        mean_residual = np.mean(residuals)
        
        # Good text should have high R² (points near a line) and low residuals
        colinearity_score = r_squared * (1.0 / (1.0 + mean_residual))
        scores.append(colinearity_score)
        
        debug_info.update({
            "colinearity_r_squared": r_squared,
            "mean_residual": mean_residual,
            "colinearity_score": colinearity_score,
            "line_slope": slope
        })
    else:
        scores.append(0.5)  # Neutral score for insufficient data
    
    # 2. SIZE UNIFORMITY ANALYSIS  
    # Text letters should have similar sizes with low variance
    if len(component_areas) >= 2:
        mean_area = np.mean(component_areas)
        std_area = np.std(component_areas)
        cv_area = std_area / mean_area if mean_area > 0 else float('inf')
        
        # Lower coefficient of variation = more uniform = more text-like
        # Good text typically has CV < 0.5, excellent text has CV < 0.3
        size_uniformity_score = max(0.0, min(1.0, (0.6 - cv_area) / 0.6))
        scores.append(size_uniformity_score)
        
        debug_info.update({
            "mean_area": mean_area,
            "area_cv": cv_area,
            "size_uniformity_score": size_uniformity_score
        })
    else:
        scores.append(0.5)
    
    # 3. SPACING UNIFORMITY ANALYSIS
    # Letter spacing should be relatively consistent
    if len(component_centroids) >= 3:
        if is_vertical_text:
            # For vertical text, sort by y-coordinate (top to bottom) and measure y-spacing
            centroids_sorted = sorted(component_centroids, key=lambda c: c[1])
            spacings = []
            for i in range(len(centroids_sorted) - 1):
                spacing = centroids_sorted[i+1][1] - centroids_sorted[i][1]
                spacings.append(spacing)
            spacing_type = "vertical"
        else:
            # For horizontal text, sort by x-coordinate (left to right) and measure x-spacing
            centroids_sorted = sorted(component_centroids, key=lambda c: c[0])
            spacings = []
            for i in range(len(centroids_sorted) - 1):
                spacing = centroids_sorted[i+1][0] - centroids_sorted[i][0]
                spacings.append(spacing)
            spacing_type = "horizontal"
        
        if len(spacings) >= 2:
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            cv_spacing = std_spacing / mean_spacing if mean_spacing > 0 else float('inf')
            
            # Good text spacing typically has CV < 0.4
            spacing_uniformity_score = max(0.0, min(1.0, (0.5 - cv_spacing) / 0.5))
            scores.append(spacing_uniformity_score)
            
            debug_info.update({
                "mean_spacing": mean_spacing,
                "spacing_cv": cv_spacing,
                "spacing_uniformity_score": spacing_uniformity_score,
                "spacings": spacings[:10],  # First 10 for logging
                "spacing_type": spacing_type
            })
        else:
            scores.append(0.5)
    else:
        scores.append(0.5)
    
    # 4. CONVEXITY ANALYSIS (optional, can be computationally expensive)
    # Text letters tend to have similar convexity (how "solid" vs "hollow" they are)
    convexity_scores = []
    for mask in component_masks[:10]:  # Limit to first 10 for performance
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour (should be the component itself)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate convexity ratio: area / convex_hull_area
            area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                convexity = area / hull_area
                convexity_scores.append(convexity)
    
    if len(convexity_scores) >= 2:
        mean_convexity = np.mean(convexity_scores)
        std_convexity = np.std(convexity_scores)
        cv_convexity = std_convexity / mean_convexity if mean_convexity > 0 else float('inf')
        
        # Good text has uniform convexity (CV < 0.3) and reasonable convexity values (0.6-0.95)
        convexity_uniformity = max(0.0, min(1.0, (0.4 - cv_convexity) / 0.4))
        convexity_quality = 1.0 if 0.6 <= mean_convexity <= 0.95 else 0.5
        convexity_overall_score = (convexity_uniformity + convexity_quality) / 2
        
        scores.append(convexity_overall_score)
        
        debug_info.update({
            "mean_convexity": mean_convexity,
            "convexity_cv": cv_convexity,
            "convexity_uniformity": convexity_uniformity,
            "convexity_quality": convexity_quality,
            "convexity_overall_score": convexity_overall_score
        })
    else:
        scores.append(0.5)  # Neutral if we can't calculate
    
    # 5. COMPONENT COUNT BONUS/PENALTY
    # More components can indicate more letters, but too many might indicate noise
    count_score = 1.0
    if len(component_centroids) >= 3:
        # Bonus for having multiple components (likely multiple letters)
        count_score = min(1.0, 0.5 + 0.1 * len(component_centroids))
    elif len(component_centroids) < 2:
        # Penalty for too few components
        count_score = 0.3
    
    scores.append(count_score)
    debug_info["count_score"] = count_score
    
    # Calculate overall score as weighted average
    weights = [0.3, 0.25, 0.25, 0.15, 0.05]  # colinearity, size, spacing, convexity, count
    overall_score = np.average(scores, weights=weights)
    
    debug_info["individual_scores"] = {
        "colinearity": scores[0] if len(scores) > 0 else 0,
        "size_uniformity": scores[1] if len(scores) > 1 else 0,
        "spacing_uniformity": scores[2] if len(scores) > 2 else 0,
        "convexity": scores[3] if len(scores) > 3 else 0,
        "count": scores[4] if len(scores) > 4 else 0
    }
    debug_info["overall_score"] = overall_score
    
    return overall_score, debug_info

def _visualize_text_analysis(test_image: ImageArray, 
                           component_labels: np.ndarray,
                           component_centroids: List[Tuple[float, float]], 
                           analysis_debug: dict,
                           output_path: str = "text_analysis_debug.png") -> None:
    """Create a visualization of the text analysis for debugging.
    
    Args:
        test_image: The black-on-gray test image
        component_labels: Connected component labels
        component_centroids: List of component centroids
        analysis_debug: Debug information from text analysis
        output_path: Where to save the visualization
    """
    if len(component_centroids) < 2:
        return
    
    # Create a colorized version for visualization
    vis_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    h, w = vis_image.shape[:2]
    
    # Create overlay for drawing
    overlay = vis_image.copy()
    
    # Draw component centroids
    for i, (cx, cy) in enumerate(component_centroids):
        cv2.circle(overlay, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        cv2.putText(overlay, str(i+1), (int(cx)+5, int(cy)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Draw regression line if we have colinearity analysis
    if "colinearity_r_squared" in analysis_debug and "line_slope" in analysis_debug:
        centroids_array = np.array(component_centroids)
        x_coords = centroids_array[:, 0]
        y_coords = centroids_array[:, 1]
        
        # Recreate the regression line using our simple function
        slope, intercept, _ = _simple_linear_regression(x_coords, y_coords)
        
        # Draw line across the width of components
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_start = slope * x_min + intercept
        y_end = slope * x_max + intercept
        cv2.line(overlay, (x_min, int(y_start)), (x_max, int(y_end)), (0, 255, 0), 2)
    
    # Draw component bounding boxes with different colors
    unique_labels = np.unique(component_labels)[1:]  # Skip background
    colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0)]
    
    for i, label in enumerate(unique_labels[:len(component_centroids)]):
        color = colors[i % len(colors)]
        component_mask = (component_labels == label)
        
        # Find bounding box
        y_coords, x_coords = np.where(component_mask)
        if len(y_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 1)
    
    # Add text with analysis results
    text_lines = [
        f"Overall Score: {analysis_debug.get('overall_score', 0):.3f}",
        f"Components: {analysis_debug.get('component_count', 0)}",
        f"Colinearity: {analysis_debug.get('colinearity_r_squared', 0):.3f}",
        f"Size CV: {analysis_debug.get('area_cv', 0):.3f}",
        f"Spacing CV: {analysis_debug.get('spacing_cv', 0):.3f}",
        f"Convexity: {analysis_debug.get('mean_convexity', 0):.3f}"
    ]
    
    for i, line in enumerate(text_lines):
        cv2.putText(overlay, line, (10, 20 + i*15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.debug(f"Saved text analysis visualization to {output_path}")

def find_colors_by_edginess(image: ImageArray, bbox: BBox, num_clusters: int = 4, debug: bool = False) -> List[str]:
    """
    Find text colors by correlating color clusters with edge density.
    
    This approach uses the assumption that text colors should correlate with areas
    of high edge density. It divides the region into tiles, measures edge density
    in each tile, and uses linear regression to find which colors predict edges.
    
    Args:
        image: Input image in BGR format  
        bbox: Bounding box (x, y, width, height) to analyze
        num_clusters: Number of color clusters for K-means (default: 4)
        debug: If True, prints diagnostic information
        
    Returns:
        List of hex color strings ranked by likelihood of being text colors
    """
    x, y, w, h = bbox
    
    # Extract ROI but don't pad yet - we'll dilate the bbox instead
    roi = image[y:y+h, x:x+w]
    
    # Calculate optimal tiling: 6 rows of square tiles
    # Block size is determined by dividing height into 6 parts
    block_size = int(np.ceil(h / 6))
    
    # Calculate new dimensions that work with square tiling
    new_h = block_size * 6  # Exactly 6 rows
    new_w = block_size * int(np.ceil(w / block_size))  # Integer number of columns
    
    # Calculate how much to dilate up and left
    dilate_up = new_h - h
    dilate_left = new_w - w
    
    # Make sure we don't go outside image bounds
    image_h, image_w = image.shape[:2]
    actual_dilate_up = min(dilate_up, y)  # Can't go above top edge
    actual_dilate_left = min(dilate_left, x)  # Can't go left of left edge
    
    # Calculate the actual expanded bbox
    new_x = x - actual_dilate_left
    new_y = y - actual_dilate_up
    actual_new_w = min(new_w, image_w - new_x)  # Don't exceed right edge
    actual_new_h = min(new_h, image_h - new_y)  # Don't exceed bottom edge
    
    # Extract the expanded ROI
    expanded_roi = image[new_y:new_y+actual_new_h, new_x:new_x+actual_new_w]
    
    if debug:
        logger.info(f"Original bbox: ({x}, {y}, {w}, {h})")
        logger.info(f"Block size: {block_size}")
        logger.info(f"Target size: {new_w} x {new_h}")
        logger.info(f"Actual expanded ROI: ({new_x}, {new_y}, {actual_new_w}, {actual_new_h})")
    
    # Convert to RGB for k-means (sklearn expects RGB)
    roi_rgb = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2RGB)
    
    # K-means clustering on colors
    flat_pixels = roi_rgb.reshape(-1, 3)
    
    # Use cv2.kmeans instead of sklearn to avoid dependency
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        flat_pixels.astype(np.float32), 
        num_clusters, 
        None, 
        criteria, 
        10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Reshape labels back to image shape
    cluster_labels = labels.reshape(expanded_roi.shape[:2])
    
    # Edge detection on the expanded ROI
    gray = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Tile analysis - use actual dimensions for tiling
    tiles_y = actual_new_h // block_size
    tiles_x = actual_new_w // block_size
    
    if tiles_y == 0 or tiles_x == 0:
        logger.warning("ROI too small for tiling analysis")
        return []
    
    tile_scores = []
    
    for i in range(tiles_y):
        for j in range(tiles_x):
            y0 = i * block_size
            y1 = min(y0 + block_size, actual_new_h)  # Don't exceed bounds
            x0 = j * block_size  
            x1 = min(x0 + block_size, actual_new_w)  # Don't exceed bounds
            
            # Skip tiles that are too small
            if (y1 - y0) < block_size // 2 or (x1 - x0) < block_size // 2:
                continue
            
            tile_edges = edges[y0:y1, x0:x1]
            tile_labels = cluster_labels[y0:y1, x0:x1]
            
            # Calculate edge density
            edge_density = np.mean(tile_edges > 0)
            
            # Calculate cluster distribution in this tile
            cluster_distribution = []
            for k in range(num_clusters):
                cluster_distribution.append((tile_labels == k).mean())
            
            tile_scores.append((edge_density, cluster_distribution))
    
    if len(tile_scores) < 2:
        logger.warning("Not enough valid tiles for regression analysis")
        return []
    
    # Linear regression: edge density ~ cluster distributions
    X = np.array([distribution for _, distribution in tile_scores])
    y = np.array([edge_density for edge_density, _ in tile_scores])
    
    # Simple linear regression using our helper function
    # We need to do multiple regression, so we'll do it manually
    n_samples, n_features = X.shape
    
    if n_samples < n_features:
        logger.warning("Not enough samples for regression")
        return []
    
    # Use numpy's least squares solver for multiple regression
    try:
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        coefs = coefficients[1:]  # Skip intercept
    except np.linalg.LinAlgError:
        logger.warning("Regression failed due to singular matrix")
        return []
    
    if debug:
        logger.info(f"Edge correlation coefficients: {coefs}")
        logger.info(f"Analyzed {len(tile_scores)} tiles ({tiles_y}x{tiles_x})")
    
    # Rank clusters by their positive correlation with edge density
    ranked_indices = np.argsort(coefs)[::-1]  # Sort descending
    
    # Convert cluster centers back to BGR and then to hex
    # Only include clusters with positive correlation
    text_colors = []
    for i in ranked_indices:
        if coefs[i] > 0:
            # Convert from RGB back to BGR
            bgr_color = centers[i][[2, 1, 0]]  # RGB to BGR
            hex_color = '#%02x%02x%02x' % tuple(map(int, bgr_color))
            text_colors.append(hex_color)
            
            if debug:
                logger.info(f"Cluster {i}: coefficient={coefs[i]:.3f}, color={hex_color}")
    
    return text_colors

def find_colors_by_tf_idf(image: ImageArray, bbox: BBox, num_clusters: int = 16, debug: bool = False) -> List[str]:
    """
    Find text colors using TF-IDF analysis on color clusters.
    
    This approach treats the bbox as a "document" and surrounding area as "corpus".
    It identifies colors that appear frequently in the text region but rarely in 
    the background, similar to how TF-IDF identifies distinctive terms in documents.
    
    Args:
        image: Input image in BGR format  
        bbox: Bounding box (x, y, width, height) to analyze
        num_clusters: Number of color clusters for K-means (default: 16)
        debug: If True, prints diagnostic information
        
    Returns:
        List containing the top 2 most distinctive hex color strings for the bbox
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
    
    if debug:
        logger.info(f"Original bbox: ({x}, {y}, {w}, {h}) - {w*h} pixels")
        logger.info(f"Expanded region: ({new_x}, {new_y}, {actual_new_w}, {actual_new_h}) - {actual_new_w*actual_new_h} pixels")
        logger.info(f"Surrounding area: {actual_new_w*actual_new_h - w*h} pixels")
    
    # Extract regions
    bbox_region = image[y:y+h, x:x+w]  # The "document"
    expanded_region = image[new_y:new_y+actual_new_h, new_x:new_x+actual_new_w]  # Total area
    
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
    
    if debug:
        logger.info(f"Actual bbox pixels: {np.sum(bbox_mask)}")
        logger.info(f"Surrounding pixels: {len(surrounding_region)}")
    
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
    
    if debug:
        logger.info(f"K-means found {num_clusters} color clusters")
        logger.info(f"TF-IDF scores range: {np.min(tf_idf_scores):.4f} to {np.max(tf_idf_scores):.4f}")
    
    # Rank clusters by TF-IDF score (descending)
    ranked_indices = np.argsort(tf_idf_scores)[::-1]
    
    # Convert top 2 cluster centers to hex colors
    text_colors = []
    colors_added = 0
    max_colors = 2  # Only return the top 2 most distinctive colors
    
    for i in ranked_indices:
        if colors_added >= max_colors:
            break
            
        if tf_idf_scores[i] > 0:  # Only include clusters with positive TF-IDF
            # Convert from RGB back to BGR for hex conversion
            rgb_color = centers[i]
            hex_color = '#%02x%02x%02x' % tuple(map(int, rgb_color))
            text_colors.append(hex_color)
            colors_added += 1
            
            if debug:
                logger.info(f"Selected cluster {i}: TF={tf_scores[i]:.4f}, IDF={idf_scores[i]:.4f}, "
                           f"TF-IDF={tf_idf_scores[i]:.4f}, color={hex_color}")
        elif debug:
            logger.info(f"Skipped cluster {i}: TF-IDF={tf_idf_scores[i]:.4f} (not positive)")
    
    if debug:
        logger.info(f"Returning {len(text_colors)} distinctive colors from {num_clusters} clusters")
    
    return text_colors 

def find_mask_by_spatial_tf_idf(image: ImageArray, bbox: BBox, num_clusters: int = 24, debug: bool = False, target_color: Optional[Color] = None) -> np.ndarray:
    """
    Create a binary mask using spatial TF-IDF analysis with adaptive thresholding and morphological cleanup.
    
    This approach:
    1. K-means clusters colors into many groups (24 by default)
    2. Calculates TF-IDF scores for each cluster
    3. Creates a grayscale "text-likelihood" map where pixel intensity = TF-IDF score
    4. Uses Otsu thresholding to automatically find optimal binary threshold
    5. Applies morphological operations to clean up the mask
    6. If target_color is specified, forces selection of the color group containing that exact color
    
    Args:
        image: Input image in BGR format  
        bbox: Bounding box (x, y, width, height) to analyze
        num_clusters: Number of color clusters for K-means (default: 24)
        debug: If True, prints diagnostic information
        target_color: Optional BGR color tuple - if provided, forces selection of exact color matches
        
    Returns:
        Binary mask (uint8) where 255 = likely text, 0 = likely background
    """
    from .mask_generator import morph_clean_mask
    
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
    
    if debug:
        logger.info(f"Spatial TF-IDF: bbox=({x}, {y}, {w}, {h}), expanded=({new_x}, {new_y}, {actual_new_w}, {actual_new_h})")
    
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
    # Need to apply mask to each color channel separately to preserve shape
    surrounding_mask_3d = np.stack([~bbox_mask, ~bbox_mask, ~bbox_mask], axis=2)
    surrounding_region = expanded_region[surrounding_mask_3d].reshape(-1, 3)
    
    # Combine both regions for k-means clustering
    bbox_pixels = bbox_region.reshape(-1, 3)
    surrounding_pixels = surrounding_region
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
    
    if debug:
        logger.info(f"TF-IDF scores range: {np.min(tf_idf_scores):.4f} to {np.max(tf_idf_scores):.4f}")
        positive_scores = np.sum(tf_idf_scores > 0)
        logger.info(f"Clusters with positive TF-IDF: {positive_scores}/{num_clusters}")
    
    # Normalize TF-IDF scores to 0-255 range
    min_score = np.min(tf_idf_scores)
    max_score = np.max(tf_idf_scores)
    
    if max_score > min_score:
        normalized_scores = ((tf_idf_scores - min_score) / (max_score - min_score) * 255).astype(np.uint8)
    else:
        # If all scores are the same, set to middle gray
        normalized_scores = np.full(num_clusters, 128, dtype=np.uint8)
    
    if debug:
        logger.info(f"Normalized scores range: {np.min(normalized_scores)} to {np.max(normalized_scores)}")
    
    # Create spatial TF-IDF map for the bbox region only
    # Use actual bbox_region dimensions, not original h,w (in case of boundary clipping)
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
    
    # Target color override: if target_color is specified, force inclusion of the color cluster containing it
    if target_color is not None:
        if debug:
            logger.info(f"Target color override: forcing inclusion of color cluster containing {target_color}")
        
        # Find which cluster the target color belongs to
        target_color_rgb = np.array([target_color[2], target_color[1], target_color[0]])  # Convert BGR to RGB for comparison with centers
        
        # Calculate distances from target color to all cluster centers
        distances = np.linalg.norm(centers - target_color_rgb, axis=1)
        target_cluster_id = np.argmin(distances)
        
        if debug:
            closest_center = centers[target_cluster_id]
            distance = distances[target_cluster_id]
            logger.info(f"Target color {target_color} (RGB: {target_color_rgb}) closest to cluster {target_cluster_id}")
            logger.info(f"Cluster center: {closest_center}, distance: {distance:.2f}")
        
        # Create mask for all pixels in the target cluster
        target_cluster_mask = (bbox_labels_2d == target_cluster_id)
        target_pixel_count = np.sum(target_cluster_mask)
        
        if target_pixel_count > 0:
            if debug:
                logger.info(f"Forcing inclusion of {target_pixel_count} pixels in target color cluster {target_cluster_id}")
                original_tf_idf = tf_idf_scores[target_cluster_id]
                logger.info(f"Original TF-IDF score for target cluster: {original_tf_idf:.4f}")
            
            # Add target cluster pixels to the existing TF-IDF mask (combine both)
            target_mask = target_cluster_mask.astype(np.uint8) * 255
            binary_mask = cv2.bitwise_or(binary_mask, target_mask)
            
            if debug:
                combined_pixels = np.sum(binary_mask == 255)
                logger.info(f"Combined mask (TF-IDF + target color): {combined_pixels} pixels")
        else:
            if debug:
                logger.info("No pixels found in target color cluster, using TF-IDF only")
    
    if debug:
        if target_color is not None:
            logger.info(f"Using TF-IDF threshold {fixed_threshold} + target color override")
        else:
            logger.info(f"Using TF-IDF threshold {fixed_threshold} only")
        mask_pixels_before_morph = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        logger.info(f"Mask coverage before morphology: {mask_pixels_before_morph}/{total_pixels} pixels ({100*mask_pixels_before_morph/total_pixels:.1f}%)")
    
    # Apply morphological operations to clean up the mask
    cleaned_mask = morph_clean_mask(binary_mask, bbox)
    
    if debug:
        mask_pixels_after_morph = np.sum(cleaned_mask == 255)
        logger.info(f"Mask coverage after morphology: {mask_pixels_after_morph}/{total_pixels} pixels ({100*mask_pixels_after_morph/total_pixels:.1f}%)")
        pixel_change = mask_pixels_after_morph - mask_pixels_before_morph
        logger.info(f"Morphological operations changed mask by {pixel_change:+d} pixels ({100*pixel_change/total_pixels:+.1f}%)")
    
    return cleaned_mask 