"""Spatial TF-IDF color analysis for text watermark detection.

This module provides spatial TF-IDF analysis to identify text-like colors
in detected text regions. The main function creates binary masks by:
1. Clustering colors in the text region and surrounding background
2. Computing a Figure of Merit (FOM) for each cluster based on:
   - TF-IDF score (color distinctiveness)
   - Border ratio (text is underrepresented on bbox edges)
   - Largest connected component fraction (text is fragmented, not solid)
3. Selecting clusters with high FOM and rejecting solid blobs
4. Applying morphological cleanup for final mask

This approach automatically identifies text colors without needing to specify
target colors, handling complex scenarios like multi-colored text and varying
backgrounds.
"""

import cv2
import numpy as np
from typing import Optional

from .utils import ImageArray, BBox, Color, setup_logger

logger = setup_logger(__name__)


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


def compute_cluster_fom(
    tf_idf_normalized: float,
    border_ratio: float,
    largest_cc_fraction: float
) -> float:
    """Compute Figure of Merit for a cluster's text-likelihood.
    
    Weights empirically determined from experiments on 18,000+ synthetic
    watermark samples across K=4,5,6,7,9.
    
    Args:
        tf_idf_normalized: TF-IDF score normalized to [0, 255]
        border_ratio: Ratio of cluster's border presence to its bbox presence
                     (< 1 means underrepresented on border = text-like)
        largest_cc_fraction: Fraction of cluster pixels in largest connected component
                            (< 1 means fragmented = text-like)
    
    Returns:
        FOM in [0, 1] where higher = more likely text
    """
    tf_score = min(tf_idf_normalized / 255.0, 1.0)
    border_score = max(0.0, 1.0 - border_ratio)
    cc_score = max(0.0, 1.0 - largest_cc_fraction)
    
    # Weights: border_ratio is the strongest predictor (63%)
    return 0.07 * tf_score + 0.63 * border_score + 0.30 * cc_score


def find_mask_by_spatial_tf_idf(
    image: ImageArray, 
    bbox: BBox, 
    num_clusters: int = 4, 
    debug: bool = False, 
    target_color: Optional[Color] = None,
    fom_threshold: float = 0.30,
    cc_guard: float = 0.85,
) -> np.ndarray:
    """Create a binary mask using Figure of Merit analysis.
    
    This approach identifies text colors by computing a Figure of Merit (FOM)
    for each color cluster, combining multiple signals:
    - TF-IDF: colors distinctive to text region vs background
    - Border ratio: text is underrepresented at bbox edges
    - Connected components: text is fragmented, not one solid blob
    
    Process:
    1. Extract the text bbox region (the "document")
    2. Extract a surrounding region with same pixel count (the "corpus")
    3. K-means cluster all colors together
    4. For each cluster, compute:
       - TF-IDF score (color distinctiveness)
       - Border ratio (edge underrepresentation)
       - Largest CC fraction (fragmentation)
    5. Compute FOM from weighted combination of metrics
    6. Select clusters with FOM >= threshold AND largest_cc <= guard
    7. Apply morphological operations for cleanup
    
    Args:
        image: Input image in BGR format  
        bbox: Bounding box (x, y, width, height) to analyze
        num_clusters: Number of color clusters for K-means (default: 4)
        debug: If True, prints diagnostic information
        target_color: Optional BGR color tuple - if provided, forces selection 
                     of the color cluster containing that exact color
        fom_threshold: Minimum Figure of Merit to accept cluster (default: 0.30)
        cc_guard: Maximum largest_cc_fraction to accept (default: 0.85)
                 Rejects solid blobs that would over-mask for inpainting
        
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
    
    # Use actual bbox_region dimensions, not original h,w (in case of boundary clipping)
    actual_h, actual_w = bbox_region.shape[:2]
    bbox_labels_2d = bbox_labels.reshape(actual_h, actual_w)
    total_bbox_pixels = actual_h * actual_w
    
    # Create border mask (2px ring around the edge of the bbox)
    # IMPORTANT: Exclude bbox edges that coincide with image edges
    # Background colors naturally extend to image edges, so those shouldn't count
    border_width = 2
    border_mask = np.zeros((actual_h, actual_w), dtype=bool)
    
    # Check which bbox edges touch the image boundary
    touches_top = (y == 0)
    touches_bottom = (y + h >= image_h)
    touches_left = (x == 0)
    touches_right = (x + w >= image_w)
    
    # Only include internal borders (not touching image edge)
    if not touches_top:
        border_mask[:border_width, :] = True
    if not touches_bottom:
        border_mask[-border_width:, :] = True
    if not touches_left:
        border_mask[:, :border_width] = True
    if not touches_right:
        border_mask[:, -border_width:] = True
    
    total_border_pixels = np.sum(border_mask)
    
    if debug and (touches_top or touches_bottom or touches_left or touches_right):
        excluded = []
        if touches_top:
            excluded.append("top")
        if touches_bottom:
            excluded.append("bottom")
        if touches_left:
            excluded.append("left")
        if touches_right:
            excluded.append("right")
        logger.info(f"Border mask excludes image edges: {', '.join(excluded)}")
    
    # Compute per-cluster metrics and Figure of Merit
    cluster_foms = np.zeros(num_clusters)
    cluster_accepted = np.zeros(num_clusters, dtype=bool)
    
    for cluster_id in range(num_clusters):
        cluster_mask = (bbox_labels_2d == cluster_id)
        cluster_pixel_count = np.sum(cluster_mask)
        
        if cluster_pixel_count == 0:
            continue
        
        # Border ratio: (cluster's % of border) / (cluster's % of bbox)
        # < 1 means underrepresented on border = text-like
        cluster_border_pixels = np.sum(cluster_mask & border_mask)
        cluster_bbox_share = cluster_pixel_count / total_bbox_pixels
        cluster_border_share = cluster_border_pixels / total_border_pixels if total_border_pixels > 0 else 0.0
        border_ratio = cluster_border_share / cluster_bbox_share if cluster_bbox_share > 0 else 1.0
        
        # Largest connected component fraction
        # Text is fragmented (many small CCs), background is unified (one big CC)
        cluster_mask_uint8 = cluster_mask.astype(np.uint8)
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cluster_mask_uint8, connectivity=8)
        if num_labels > 1:
            # stats[0] is background, stats[1:] are components
            component_sizes = stats[1:, cv2.CC_STAT_AREA]
            largest_cc_size = np.max(component_sizes) if len(component_sizes) > 0 else 0
        else:
            largest_cc_size = 0
        largest_cc_fraction = largest_cc_size / cluster_pixel_count if cluster_pixel_count > 0 else 0.0
        
        # Compute Figure of Merit
        fom = compute_cluster_fom(normalized_scores[cluster_id], border_ratio, largest_cc_fraction)
        cluster_foms[cluster_id] = fom
        
        # Accept if FOM meets threshold AND not a solid blob
        is_text_like = fom >= fom_threshold
        is_not_solid_blob = largest_cc_fraction <= cc_guard
        cluster_accepted[cluster_id] = is_text_like and is_not_solid_blob
        
        if debug:
            logger.info(f"Cluster {cluster_id}: TF-IDF={normalized_scores[cluster_id]}, "
                       f"border_ratio={border_ratio:.3f}, largest_cc={largest_cc_fraction:.3f}, "
                       f"FOM={fom:.3f}, accepted={cluster_accepted[cluster_id]}")
    
    # Create binary mask from accepted clusters
    binary_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
    for cluster_id in range(num_clusters):
        if cluster_accepted[cluster_id]:
            cluster_mask = (bbox_labels_2d == cluster_id)
            binary_mask[cluster_mask] = 255
    
    if debug:
        accepted_count = np.sum(cluster_accepted)
        logger.info(f"Accepted {accepted_count}/{num_clusters} clusters (FOM>={fom_threshold}, CC<={cc_guard})")
    
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
            logger.info(f"Using FOM>={fom_threshold}, CC<={cc_guard} + target color override")
        else:
            logger.info(f"Using FOM>={fom_threshold}, CC<={cc_guard}")
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
