"""Image metrics for bbox expansion and retry decisions.

These metrics measure structural properties of image regions that correlate
with text-like content (stroke structures, edge patterns, etc.).

Developed through empirical experimentation to:
1. Identify text-like regions for bbox expansion (before metrics)
2. Detect when inpainting failed and retry is needed (after metrics)

Thresholds are based on experimental data from experiments/granularity_experiment.py.
"""

import cv2
import numpy as np
from typing import Tuple

from .utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# THRESHOLDS (empirically determined from experimental data)
# =============================================================================

# For bbox expansion: minimum values that indicate text-like regions (P25 of detected text)
EXPANSION_BLACKHAT_THRESHOLD = 2.5  # P25 of detected text regions
EXPANSION_EDGE_ROW_THRESHOLD = 0.15  # P25 of detected text regions

# For retry decision: values above these suggest text remnants remain
RETRY_BLACKHAT_THRESHOLD = 3.0  # Balances recall (85%) vs unnecessary retries (45%)
RETRY_EDGE_ROW_THRESHOLD = 0.08  # Complements blackhat for horizontal watermarks


# =============================================================================
# METRIC MEASUREMENT FUNCTIONS
# =============================================================================

def measure_blackhat_energy(region_bgr: np.ndarray, kernel_size: int = 5) -> float:
    """Measure black-hat morphological response - excellent for stroke structures.
    
    Black-hat extracts dark objects on light backgrounds (typical text).
    Correlation with remnancy: r=0.844 (EXCELLENT)
    
    Args:
        region_bgr: BGR image region
        kernel_size: Size of morphological kernel (default 5x5)
        
    Returns:
        Mean black-hat response (higher = more stroke-like structures)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return float(np.mean(blackhat))


def measure_edge_row_energy(region_bgr: np.ndarray) -> float:
    """Measure edge row energy - band detection for horizontal watermarks.
    
    Computes row-wise edge concentration using Canny.
    Provides independent signal from blackhat (r=0.47 correlation).
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Max row edge fraction (0.0-1.0)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Canny edges with automatic thresholds
    median = np.median(gray)
    lower = int(max(0, 0.67 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    
    # Row-wise edge sum
    row_edge_sums = np.sum(edges > 0, axis=1)
    width = edges.shape[1]
    
    if width == 0:
        return 0.0
    return float(np.max(row_edge_sums) / width)


# =============================================================================
# BBOX EXPANSION
# =============================================================================

def expand_bbox_along_long_axis(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    max_gap: int = 2,
    blackhat_threshold: float = EXPANSION_BLACKHAT_THRESHOLD,
    edge_row_threshold: float = EXPANSION_EDGE_ROW_THRESHOLD,
) -> Tuple[int, int, int, int]:
    """Expand a detection bbox along its long axis to capture more text.
    
    Scans square regions adjacent to the bbox and includes them if they
    have text-like characteristics (high blackhat or edge_row energy).
    Stops when max_gap consecutive low-energy squares are found.
    
    Args:
        image: Full image (BGR)
        bbox: Original detection bbox (x, y, width, height)
        max_gap: Stop after this many consecutive low-energy squares
        blackhat_threshold: Min blackhat energy to consider text-like
        edge_row_threshold: Min edge row energy to consider text-like
        
    Returns:
        Expanded bbox (x, y, width, height)
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    # Determine expansion direction based on aspect ratio
    if w >= h:
        # Horizontal text - expand left and right
        step = h  # Use height as step size
        expand_horizontal = True
    else:
        # Vertical text - expand up and down
        step = w  # Use width as step size
        expand_horizontal = False
    
    # Ensure minimum step size
    step = max(step, 16)
    
    # Track expansion bounds
    new_x, new_y, new_w, new_h = x, y, w, h
    
    if expand_horizontal:
        # Expand left
        consecutive_low = 0
        check_x = x - step
        while check_x >= 0 and consecutive_low < max_gap:
            # Sample square region
            sample_x = max(0, check_x)
            sample_w = min(step, x - sample_x)
            if sample_w <= 0:
                break
                
            sample = image[y:y+h, sample_x:sample_x+sample_w]
            if sample.size == 0:
                break
                
            blackhat = measure_blackhat_energy(sample)
            edge_row = measure_edge_row_energy(sample)
            
            if blackhat >= blackhat_threshold or edge_row >= edge_row_threshold:
                # Text-like, expand
                new_x = sample_x
                new_w = (x + w) - new_x
                consecutive_low = 0
                logger.debug(f"Expanding left: blackhat={blackhat:.2f}, edge_row={edge_row:.2f}")
            else:
                consecutive_low += 1
                
            check_x -= step
        
        # Expand right
        consecutive_low = 0
        check_x = x + w
        while check_x < img_w and consecutive_low < max_gap:
            sample_x = check_x
            sample_w = min(step, img_w - sample_x)
            if sample_w <= 0:
                break
                
            sample = image[y:y+h, sample_x:sample_x+sample_w]
            if sample.size == 0:
                break
                
            blackhat = measure_blackhat_energy(sample)
            edge_row = measure_edge_row_energy(sample)
            
            if blackhat >= blackhat_threshold or edge_row >= edge_row_threshold:
                new_w = (sample_x + sample_w) - new_x
                consecutive_low = 0
                logger.debug(f"Expanding right: blackhat={blackhat:.2f}, edge_row={edge_row:.2f}")
            else:
                consecutive_low += 1
                
            check_x += step
    else:
        # Expand up
        consecutive_low = 0
        check_y = y - step
        while check_y >= 0 and consecutive_low < max_gap:
            sample_y = max(0, check_y)
            sample_h = min(step, y - sample_y)
            if sample_h <= 0:
                break
                
            sample = image[sample_y:sample_y+sample_h, x:x+w]
            if sample.size == 0:
                break
                
            blackhat = measure_blackhat_energy(sample)
            edge_row = measure_edge_row_energy(sample)
            
            if blackhat >= blackhat_threshold or edge_row >= edge_row_threshold:
                new_y = sample_y
                new_h = (y + h) - new_y
                consecutive_low = 0
                logger.debug(f"Expanding up: blackhat={blackhat:.2f}, edge_row={edge_row:.2f}")
            else:
                consecutive_low += 1
                
            check_y -= step
        
        # Expand down
        consecutive_low = 0
        check_y = y + h
        while check_y < img_h and consecutive_low < max_gap:
            sample_y = check_y
            sample_h = min(step, img_h - sample_y)
            if sample_h <= 0:
                break
                
            sample = image[sample_y:sample_y+sample_h, x:x+w]
            if sample.size == 0:
                break
                
            blackhat = measure_blackhat_energy(sample)
            edge_row = measure_edge_row_energy(sample)
            
            if blackhat >= blackhat_threshold or edge_row >= edge_row_threshold:
                new_h = (sample_y + sample_h) - new_y
                consecutive_low = 0
                logger.debug(f"Expanding down: blackhat={blackhat:.2f}, edge_row={edge_row:.2f}")
            else:
                consecutive_low += 1
                
            check_y += step
    
    expanded_bbox = (new_x, new_y, new_w, new_h)
    
    if expanded_bbox != bbox:
        original_area = w * h
        new_area = new_w * new_h
        logger.info(f"Expanded bbox {bbox} -> {expanded_bbox} ({new_area/original_area:.1f}x area)")
    
    return expanded_bbox


# =============================================================================
# RETRY DECISION
# =============================================================================

def needs_retry(
    inpainted_region: np.ndarray,
    blackhat_threshold: float = RETRY_BLACKHAT_THRESHOLD,
    edge_row_threshold: float = RETRY_EDGE_ROW_THRESHOLD,
) -> bool:
    """Check if inpainting likely failed and retry with higher granularity is needed.
    
    Uses blackhat_energy and edge_row_energy as proxies for remnancy.
    Combined rule catches 85% of failures with 45% false positive rate.
    
    Args:
        inpainted_region: The inpainted image region (BGR)
        blackhat_threshold: Threshold for blackhat energy (default 3.0)
        edge_row_threshold: Threshold for edge row energy (default 0.08)
        
    Returns:
        True if retry is recommended, False otherwise
    """
    blackhat = measure_blackhat_energy(inpainted_region)
    edge_row = measure_edge_row_energy(inpainted_region)
    
    should_retry = blackhat > blackhat_threshold or edge_row > edge_row_threshold
    
    if should_retry:
        logger.debug(f"Retry recommended: blackhat={blackhat:.2f} (>{blackhat_threshold}), "
                    f"edge_row={edge_row:.3f} (>{edge_row_threshold})")
    
    return should_retry
