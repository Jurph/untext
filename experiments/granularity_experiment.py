"""Experimental rig to find optimal granularity based on region statistics.

This experiment:
1. Runs detection ONCE per image to find consensus regions
2. Crops experimental region (detection + padding) ONCE
3. Measures BEFORE statistics (region stats + edge_density + cc_count)
4. Tests granularity values [4, 8, 16, 24] for TF-IDF masking
5. Measures AFTER metrics (remnancy, edge_density, cc_count)
6. Writes results LINE BY LINE in APPEND mode (crash-safe)

Usage:
    python -m experiments.granularity_experiment -i <input_dir> -o <output_csv> [options]
"""

import argparse
import csv
import cv2
import numpy as np
import sys
import time
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from untextre.utils import load_image, setup_logger
from untextre.consensus import run_consensus_detection, initialize_consensus_models
from untextre.find_text_colors import find_mask_by_spatial_tf_idf
from untextre.inpaint import inpaint_image
from untextre.metrics import measure_blackhat_energy, measure_edge_row_energy

logger = setup_logger(__name__)

# Granularity values to test
DEFAULT_GRANULARITIES = [4, 8, 16, 24]

# CSV field names - defined once for consistency
FIELDNAMES = [
    'image', 'detection_idx', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_area',
    # Before metrics (region characteristics)
    'dynamic_range_before', 'std_dev_before', 'otsu_threshold_before', 'entropy_before',
    'num_peaks', 'peak_prominence', 'bimodality', 'coef_variation', 'color_std',
    'edge_density_before', 'cc_count_before',
    # New before metrics (Tier 0-3)
    'laplacian_var_before', 'canny_density_before', 'row_peakiness_before',
    'blackhat_energy_before', 'tophat_energy_before', 'otsu_separability_before',
    'gradient_energy_before', 'edge_row_energy_before',
    # Experiment parameters
    'granularity',
    # After metrics (lower is better for most)
    'remnancy', 'detection_count',
    'otsu_threshold_after', 'entropy_after',
    'edge_density_after', 'cc_count_after',
    # New after metrics
    'laplacian_var_after', 'canny_density_after', 'row_peakiness_after',
    'blackhat_energy_after', 'tophat_energy_after', 'otsu_separability_after',
    'gradient_energy_after', 'edge_row_energy_after',
    # Deltas (more negative = more change, usually better)
    'otsu_threshold_delta', 'entropy_delta',
    'edge_density_delta', 'cc_count_delta',
    # Ratios for new metrics (after/before, <1.0 = reduction)
    'laplacian_var_ratio', 'canny_density_ratio', 'row_peakiness_ratio',
    'blackhat_energy_ratio', 'tophat_energy_ratio', 'otsu_separability_ratio',
    'gradient_energy_ratio', 'edge_row_energy_ratio',
    # Metadata
    'processing_time'
]


def measure_edge_density(region_bgr: np.ndarray) -> float:
    """Measure edge density using Sobel operator.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Edge density (edge pixels / total pixels), range 0.0-1.0
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Threshold to get edge pixels (using Otsu on magnitude)
    mag_uint8 = np.clip(magnitude, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Edge density = edge pixels / total pixels
    edge_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    
    return float(edge_pixels / total_pixels)


def measure_cc_count(region_bgr: np.ndarray) -> int:
    """Count connected components using Otsu threshold.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Number of connected components (excluding background)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold using Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Count connected components
    num_labels, _ = cv2.connectedComponents(binary)
    
    # Subtract 1 for background label
    return int(num_labels - 1)


def safe_ratio(after: float, before: float) -> float:
    """Compute after/before ratio safely, handling zero denominators.
    
    Returns:
        after/before if before != 0
        1.0 if both are 0 (no change)
        float('inf') if before=0 but after>0
        0.0 if before>0 but after=0
    """
    if before == 0:
        return 1.0 if after == 0 else float('inf')
    return after / before


# =============================================================================
# NEW METRICS - Tier 0-3 from the "text-likeness" menu
# =============================================================================

def measure_laplacian_variance(region_bgr: np.ndarray) -> float:
    """Measure Laplacian variance - sensitive to thin stroke remnants.
    
    Tier 1 metric: very cheap, good for detecting fine detail/edges.
    Higher values indicate more high-frequency content (text-like).
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Variance of Laplacian response
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def measure_canny_density(region_bgr: np.ndarray) -> float:
    """Measure Canny edge density - robust for text strokes.
    
    Tier 2 metric: cheap, different signal than Sobel thresholding.
    Uses automatic threshold calculation based on median intensity.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Edge density (edge pixels / total pixels), range 0.0-1.0
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Automatic threshold based on median (Canny recommends sigma=0.33)
    median = np.median(gray)
    lower = int(max(0, 0.67 * median))
    upper = int(min(255, 1.33 * median))
    
    edges = cv2.Canny(gray, lower, upper)
    return float(np.sum(edges > 0) / edges.size)


def measure_row_projection_peakiness(region_bgr: np.ndarray) -> float:
    """Measure row projection peakiness - good for horizontal text bands.
    
    Tier 0-2 metric: very cheap, specifically informative for URL watermarks.
    Computes row-wise mean intensity and measures how "peaky" the profile is.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Peakiness score (std of row means / mean of row means)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Row-wise mean intensity
    row_means = np.mean(gray, axis=1)
    
    # Peakiness = coefficient of variation of row means
    mean_val = np.mean(row_means)
    if mean_val == 0:
        return 0.0
    return float(np.std(row_means) / mean_val)


# measure_blackhat_energy imported from untextre.metrics


def measure_tophat_energy(region_bgr: np.ndarray, kernel_size: int = 5) -> float:
    """Measure top-hat morphological response - for light text on dark.
    
    Tier 3 metric: complement to black-hat for reverse polarity text.
    Top-hat extracts light objects on dark backgrounds.
    
    Args:
        region_bgr: BGR image region
        kernel_size: Size of morphological kernel (default 5x5)
        
    Returns:
        Mean top-hat response (higher = more light stroke structures)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Top-hat: original - opening (finds light regions smaller than kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    return float(np.mean(tophat))


def measure_otsu_separability(region_bgr: np.ndarray) -> float:
    """Measure Otsu separability score - indicates bimodality.
    
    Tier 2 metric: we already compute Otsu, this adds the quality score.
    Higher separability means clearer foreground/background distinction.
    Watermarks often increase bimodality; good removal reduces it.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Between-class variance ratio (0.0-1.0, higher = more bimodal)
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    
    # Find Otsu threshold
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = int(otsu_thresh)
    
    # Compute class probabilities and means
    w0 = hist_norm[:otsu_thresh].sum()  # Background weight
    w1 = hist_norm[otsu_thresh:].sum()  # Foreground weight
    
    if w0 == 0 or w1 == 0:
        return 0.0
    
    # Class means
    bins = np.arange(256)
    mu0 = (bins[:otsu_thresh] * hist_norm[:otsu_thresh]).sum() / w0
    mu1 = (bins[otsu_thresh:] * hist_norm[otsu_thresh:]).sum() / w1
    
    # Total mean
    mu_total = (bins * hist_norm).sum()
    
    # Between-class variance
    sigma_b_sq = w0 * w1 * (mu0 - mu1) ** 2
    
    # Total variance
    sigma_total_sq = ((bins - mu_total) ** 2 * hist_norm).sum()
    
    if sigma_total_sq == 0:
        return 0.0
    
    # Separability = between-class variance / total variance
    return float(sigma_b_sq / sigma_total_sq)


def measure_gradient_energy(region_bgr: np.ndarray) -> float:
    """Measure gradient energy using Scharr operator.
    
    Tier 1 metric: very cheap, measures overall "edginess".
    Different from edge_density: this is mean magnitude, not thresholded count.
    
    Args:
        region_bgr: BGR image region
        
    Returns:
        Mean gradient magnitude
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    # Scharr is slightly more accurate than Sobel for small kernels
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    
    magnitude = np.sqrt(scharrx**2 + scharry**2)
    return float(np.mean(magnitude))


# measure_edge_row_energy imported from untextre.metrics


def measure_region_statistics(region_bgr: np.ndarray, suffix: str = '') -> Dict[str, float]:
    """Measure various statistics about a region's color distribution.
    
    Args:
        region_bgr: BGR image region
        suffix: Optional suffix for keys (e.g., '_before', '_after')
        
    Returns:
        Dictionary of statistics
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    
    stats = {}
    
    # Core metrics (tracked before/after)
    stats[f'dynamic_range{suffix}'] = float(gray.max() - gray.min())
    stats[f'std_dev{suffix}'] = float(np.std(gray))
    
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stats[f'otsu_threshold{suffix}'] = float(otsu_thresh)
    
    # Histogram analysis
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    
    # Entropy (information content)
    nonzero = hist_norm[hist_norm > 0]
    stats[f'entropy{suffix}'] = float(-np.sum(nonzero * np.log2(nonzero)))
    
    # Only compute these for 'before' (they're predictors, not outcomes)
    if suffix == '_before' or suffix == '':
        # Number of significant peaks
        peaks, _ = find_peaks(hist, height=hist.max() * 0.05, distance=15)
        stats['num_peaks'] = len(peaks)
        
        # Peak prominence
        if len(peaks) >= 2:
            peak_heights = hist[peaks]
            stats['peak_prominence'] = float(peak_heights.max() - peak_heights.min())
        else:
            stats['peak_prominence'] = 0.0
        
        # Bimodality coefficient
        flat_gray = gray.flatten()
        s = skew(flat_gray)
        k = kurtosis(flat_gray)
        stats['bimodality'] = float((s**2 + 1) / (k + 3)) if (k + 3) != 0 else 0.0
        
        # Coefficient of variation
        mean_val = np.mean(gray)
        std_val = stats.get(f'std_dev{suffix}', np.std(gray))
        stats['coef_variation'] = float(std_val / mean_val) if mean_val > 0 else 0.0
        
        # Color channel variance
        b, g, r = cv2.split(region_bgr)
        color_std = np.mean([np.std(b), np.std(g), np.std(r)])
        stats['color_std'] = float(color_std)
    
    return stats


def measure_remnancy(region_bgr: np.ndarray, confidence_threshold: float = 0.1) -> Tuple[float, int]:
    """Measure how much text remains in a region after processing.
    
    Returns:
        Tuple of (sum of confidences, detection count) - lower is better
    """
    from untextre.consensus import detect_with_doctr, detect_with_easyocr, detect_with_east
    from untextre.preprocessor import preprocess_image
    
    detections = []
    
    try:
        preprocessed = preprocess_image(region_bgr)
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}")
        return 0.0, 0
    
    # Run all detectors
    try:
        detections.extend(detect_with_doctr(preprocessed, confidence_threshold))
    except Exception:
        pass
    
    try:
        detections.extend(detect_with_easyocr(region_bgr, confidence_threshold))
    except Exception:
        pass
    
    try:
        detections.extend(detect_with_east(preprocessed, confidence_threshold))
    except Exception:
        pass
    
    # Sum confidences - each detection is (x, y, w, h, confidence)
    total_confidence = sum(d[4] if len(d) > 4 else 0.5 for d in detections)
    return total_confidence, len(detections)


class ExperimentalRegion:
    """Cached experimental region for efficient granularity sweep."""
    
    def __init__(self, image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 64):
        """Crop and cache the experimental region once."""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # Calculate padded bounds
        self.pad_x1 = max(0, x - padding)
        self.pad_y1 = max(0, y - padding)
        self.pad_x2 = min(img_w, x + w + padding)
        self.pad_y2 = min(img_h, y + h + padding)
        
        # Relative bbox within experimental region
        self.rel_x = x - self.pad_x1
        self.rel_y = y - self.pad_y1
        self.rel_bbox = (self.rel_x, self.rel_y, w, h)
        self.bbox = bbox
        
        # Cache the cropped region
        self.region = image[self.pad_y1:self.pad_y2, self.pad_x1:self.pad_x2].copy()
        self.region_h, self.region_w = self.region.shape[:2]
        
        # Pre-compute BEFORE statistics (only once per region)
        self.stats_before = measure_region_statistics(self.region, suffix='_before')
        self.edge_density_before = measure_edge_density(self.region)
        self.cc_count_before = measure_cc_count(self.region)
        
        # New Tier 0-3 metrics (before)
        self.laplacian_var_before = measure_laplacian_variance(self.region)
        self.canny_density_before = measure_canny_density(self.region)
        self.row_peakiness_before = measure_row_projection_peakiness(self.region)
        self.blackhat_energy_before = measure_blackhat_energy(self.region)
        self.tophat_energy_before = measure_tophat_energy(self.region)
        self.otsu_separability_before = measure_otsu_separability(self.region)
        self.gradient_energy_before = measure_gradient_energy(self.region)
        self.edge_row_energy_before = measure_edge_row_energy(self.region)
    
    def test_granularity(self, granularity: int, confidence_threshold: float = 0.1) -> Dict:
        """Test a specific granularity value on this region."""
        start_time = time.time()
        
        try:
            # Generate mask using TF-IDF
            x, y, w, h = self.rel_bbox
            mask = find_mask_by_spatial_tf_idf(
                self.region, 
                self.rel_bbox, 
                num_clusters=granularity,
                debug=False
            )
            
            # Create full-size mask for the experimental region
            full_mask = np.zeros((self.region_h, self.region_w), dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = mask
            
            # Inpaint
            inpainted = inpaint_image(self.region, full_mask, method='telea')
            
            # Measure AFTER metrics
            remnancy, det_count = measure_remnancy(inpainted, confidence_threshold)
            stats_after = measure_region_statistics(inpainted, suffix='_after')
            edge_density_after = measure_edge_density(inpainted)
            cc_count_after = measure_cc_count(inpainted)
            
            # New Tier 0-3 metrics (after)
            laplacian_var_after = measure_laplacian_variance(inpainted)
            canny_density_after = measure_canny_density(inpainted)
            row_peakiness_after = measure_row_projection_peakiness(inpainted)
            blackhat_energy_after = measure_blackhat_energy(inpainted)
            tophat_energy_after = measure_tophat_energy(inpainted)
            otsu_separability_after = measure_otsu_separability(inpainted)
            gradient_energy_after = measure_gradient_energy(inpainted)
            edge_row_energy_after = measure_edge_row_energy(inpainted)
            
            elapsed = time.time() - start_time
            
            return {
                'granularity': granularity,
                'remnancy': remnancy,
                'detection_count': det_count,
                # After stats
                'otsu_threshold_after': stats_after['otsu_threshold_after'],
                'entropy_after': stats_after['entropy_after'],
                'edge_density_after': edge_density_after,
                'cc_count_after': cc_count_after,
                # New after metrics
                'laplacian_var_after': laplacian_var_after,
                'canny_density_after': canny_density_after,
                'row_peakiness_after': row_peakiness_after,
                'blackhat_energy_after': blackhat_energy_after,
                'tophat_energy_after': tophat_energy_after,
                'otsu_separability_after': otsu_separability_after,
                'gradient_energy_after': gradient_energy_after,
                'edge_row_energy_after': edge_row_energy_after,
                # Deltas (after - before) for metrics with meaningful absolute scales
                'otsu_threshold_delta': stats_after['otsu_threshold_after'] - self.stats_before['otsu_threshold_before'],
                'entropy_delta': stats_after['entropy_after'] - self.stats_before['entropy_before'],
                'edge_density_delta': edge_density_after - self.edge_density_before,
                'cc_count_delta': cc_count_after - self.cc_count_before,
                # Ratios (after/before) for metrics with varying scales (<1.0 = reduction)
                'laplacian_var_ratio': safe_ratio(laplacian_var_after, self.laplacian_var_before),
                'canny_density_ratio': safe_ratio(canny_density_after, self.canny_density_before),
                'row_peakiness_ratio': safe_ratio(row_peakiness_after, self.row_peakiness_before),
                'blackhat_energy_ratio': safe_ratio(blackhat_energy_after, self.blackhat_energy_before),
                'tophat_energy_ratio': safe_ratio(tophat_energy_after, self.tophat_energy_before),
                'otsu_separability_ratio': safe_ratio(otsu_separability_after, self.otsu_separability_before),
                'gradient_energy_ratio': safe_ratio(gradient_energy_after, self.gradient_energy_before),
                'edge_row_energy_ratio': safe_ratio(edge_row_energy_after, self.edge_row_energy_before),
                'processing_time': elapsed,
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"Granularity {granularity} failed: {e}")
            return {
                'granularity': granularity,
                'remnancy': float('inf'),
                'detection_count': -1,
                'otsu_threshold_after': float('inf'),
                'entropy_after': float('inf'),
                'edge_density_after': float('inf'),
                'cc_count_after': -1,
                # New after metrics (error values)
                'laplacian_var_after': float('inf'),
                'canny_density_after': float('inf'),
                'row_peakiness_after': float('inf'),
                'blackhat_energy_after': float('inf'),
                'tophat_energy_after': float('inf'),
                'otsu_separability_after': float('inf'),
                'gradient_energy_after': float('inf'),
                'edge_row_energy_after': float('inf'),
                # Deltas (error values)
                'otsu_threshold_delta': float('inf'),
                'entropy_delta': float('inf'),
                'edge_density_delta': float('inf'),
                'cc_count_delta': 0,
                # Ratios (error value = inf, meaning "could not compute")
                'laplacian_var_ratio': float('inf'),
                'canny_density_ratio': float('inf'),
                'row_peakiness_ratio': float('inf'),
                'blackhat_energy_ratio': float('inf'),
                'tophat_energy_ratio': float('inf'),
                'otsu_separability_ratio': float('inf'),
                'gradient_energy_ratio': float('inf'),
                'edge_row_energy_ratio': float('inf'),
                'processing_time': time.time() - start_time,
                'success': False
            }


def write_csv_row(filepath: Path, row: Dict, write_header: bool):
    """Write a single row to CSV in append mode.
    
    Args:
        filepath: Path to CSV file
        row: Dictionary of field values
        write_header: Whether to write header row first
    """
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()  # Force write to disk


def load_results(csv_path: Path) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    if '.' in str(row[key]):
                        row[key] = float(row[key])
                    else:
                        row[key] = int(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def run_analysis(csv_path: Path):
    """Run analysis on experiment results and print summary."""
    from collections import defaultdict
    
    results = load_results(csv_path)
    if not results:
        print("No results to analyze.")
        return
    
    # Find optimal granularity for each detection (lowest remnancy wins)
    best_by_detection = {}
    for r in results:
        key = (r['image'], r['detection_idx'])
        if key not in best_by_detection or r['remnancy'] < best_by_detection[key]['remnancy']:
            best_by_detection[key] = r
    
    # Group by optimal granularity
    by_optimal_g = defaultdict(list)
    for r in best_by_detection.values():
        by_optimal_g[r['granularity']].append(r)
    
    # Distribution
    print("\n" + "="*70)
    print("OPTIMAL GRANULARITY DISTRIBUTION (by lowest remnancy)")
    print("="*70)
    for g in sorted(by_optimal_g.keys()):
        count = len(by_optimal_g[g])
        pct = 100 * count / len(best_by_detection)
        bar = "#" * int(pct / 2)
        print(f"  g={g:>2}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Before statistics to correlate with optimal granularity
    stat_fields = [
        'dynamic_range_before', 'std_dev_before', 'otsu_threshold_before', 'entropy_before',
        'num_peaks', 'peak_prominence', 'bimodality', 'coef_variation',
        'color_std', 'bbox_area', 'edge_density_before', 'cc_count_before'
    ]
    
    # Calculate correlations
    print("\n" + "="*70)
    print("MEAN STATISTICS BY OPTIMAL GRANULARITY")
    print("="*70)
    
    header = f"{'Statistic':<22}"
    for g in sorted(by_optimal_g.keys()):
        header += f"  g={g:<8}"
    print(header)
    print("-"*70)
    
    correlations = {}
    for stat in stat_fields:
        row = f"{stat:<22}"
        means = []
        granularities = []
        
        for g in sorted(by_optimal_g.keys()):
            values = [r.get(stat, 0) for r in by_optimal_g[g] if stat in r]
            if values:
                mean_val = np.mean(values)
                row += f"  {mean_val:>9.2f}"
                means.append(mean_val)
                granularities.append(g)
            else:
                row += f"  {'N/A':>9}"
        
        print(row)
        
        if len(means) >= 3:
            corr = np.corrcoef(granularities, means)[0, 1]
            if not np.isnan(corr):
                correlations[stat] = corr
    
    # Print correlations
    print("\n" + "="*70)
    print("CORRELATION COEFFICIENTS (stat vs optimal granularity)")
    print("="*70)
    print("+ correlation: higher stat -> higher optimal granularity")
    print("- correlation: higher stat -> lower optimal granularity")
    print("-"*70)
    
    for stat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        strength = "STRONG" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {stat:<22}: {corr:>+.3f} ({strength})")
    
    # After-metric analysis: which granularity produces best results?
    print("\n" + "="*70)
    print("AFTER-METRICS BY GRANULARITY (lower is better)")
    print("="*70)
    
    # Group all results by granularity (not just optimal)
    by_g = defaultdict(list)
    for r in results:
        by_g[r['granularity']].append(r)
    
    header = f"{'Metric':<22}"
    for g in sorted(by_g.keys()):
        header += f"  g={g:<8}"
    print(header)
    print("-"*70)
    
    for metric in ['remnancy', 'otsu_threshold_after', 'entropy_after', 'edge_density_after', 'cc_count_after']:
        row = f"{metric:<22}"
        for g in sorted(by_g.keys()):
            values = [r.get(metric, 0) for r in by_g[g] 
                      if metric in r and r.get(metric) != float('inf')]
            if values:
                mean_val = np.mean(values)
                row += f"  {mean_val:>9.2f}"
            else:
                row += f"  {'N/A':>9}"
        print(row)
    
    # Correlation between after-metrics and remnancy (to find cheap proxies)
    print("\n" + "="*70)
    print("PROXY ANALYSIS: Which after-metrics correlate with remnancy?")
    print("="*70)
    print("(Higher |correlation| = better proxy for skipping detector pass)")
    print("-"*70)
    
    # Get valid results with remnancy
    valid_results = [r for r in results 
                     if 'remnancy' in r and r.get('remnancy') != float('inf')
                     and r.get('remnancy') is not None]
    
    if len(valid_results) > 10:
        remnancy_vals = np.array([r['remnancy'] for r in valid_results])
        
        proxy_metrics = [
            'edge_density_after', 'cc_count_after', 
            'otsu_threshold_after', 'entropy_after',
            'edge_density_delta', 'cc_count_delta',
            'otsu_threshold_delta', 'entropy_delta'
        ]
        
        proxy_correlations = {}
        for metric in proxy_metrics:
            metric_vals = []
            remnancy_subset = []
            for i, r in enumerate(valid_results):
                val = r.get(metric)
                if val is not None and val != float('inf') and not np.isnan(val) if isinstance(val, float) else True:
                    metric_vals.append(val)
                    remnancy_subset.append(remnancy_vals[i])
            
            if len(metric_vals) > 10:
                corr = np.corrcoef(metric_vals, remnancy_subset)[0, 1]
                if not np.isnan(corr):
                    proxy_correlations[metric] = corr
        
        for metric, corr in sorted(proxy_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            strength = "EXCELLENT" if abs(corr) > 0.7 else "GOOD" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
            print(f"  {metric:<22}: {corr:>+.3f} ({strength})")
        
        # Recommendation
        best_proxy = max(proxy_correlations.items(), key=lambda x: abs(x[1])) if proxy_correlations else None
        if best_proxy and abs(best_proxy[1]) > 0.5:
            print(f"\n  RECOMMENDATION: Use '{best_proxy[0]}' as remnancy proxy (r={best_proxy[1]:+.3f})")
        else:
            print(f"\n  No strong proxy found. Detector pass may be necessary.")
    else:
        print("  Not enough valid results for proxy analysis.")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment to find optimal granularity based on region statistics"
    )
    parser.add_argument('-i', '--input', required=True, help='Input directory of images')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file (append mode)')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--padding', type=int, default=64, help='Padding around detected regions')
    default_g_str = ','.join(str(g) for g in DEFAULT_GRANULARITIES)
    parser.add_argument('--granularities', type=str, default=default_g_str,
                       help=f'Comma-separated granularity values (default: {default_g_str})')
    parser.add_argument('--device', default='cuda', help='Device for models (cuda/cpu)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse granularity values
    granularity_values = [int(x.strip()) for x in args.granularities.split(',')]
    logger.info(f"Testing granularity values: {granularity_values}")
    
    # Find images
    input_dir = Path(args.input)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([f for f in input_dir.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize models (with normal logging)
    logger.info("Initializing detection models...")
    initialize_consensus_models(args.confidence)
    logger.info("Using Telea inpainting (fast, classical method)")
    
    # Suppress verbose logging from submodules during experiment
    import logging
    for module in ['untextre.consensus', 'untextre.find_text_colors', 'untextre.inpaint',
                   'untextre.detector', 'untextre.preprocessor', 'untextre.lama_inpainter',
                   'untextre.telea_inpainter', 'easyocr', 'doctr']:
        logging.getLogger(module).setLevel(logging.WARNING)
    logger.info("Submodule logging set to WARNING for cleaner progress display")
    
    # Setup output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to write header (file doesn't exist or is empty)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    
    # Resume feature: skip already-processed images
    already_done = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'image' in row:
                    already_done.add(row['image'])
        logger.info(f"Found {len(already_done)} already-processed images in {output_path}")
    
    # Filter out already-done images
    original_count = len(image_files)
    image_files = [f for f in image_files if f.name not in already_done]
    
    if original_count != len(image_files):
        logger.info(f"Skipping {original_count - len(image_files)} already-processed images")
        logger.info(f"Remaining: {len(image_files)} images to process")
    
    if write_header:
        logger.info(f"Creating new output file: {output_path}")
    else:
        logger.info(f"Appending to existing file: {output_path}")
    
    # Process images
    total_start = time.time()
    total_experiments = 0
    total_detections = 0
    header_written = False
    
    # Outer progress bar for images
    image_pbar = tqdm(image_files, desc="Images", unit="img", position=0)
    for image_path in image_pbar:
        image_pbar.set_postfix_str(image_path.name[:20])
        
        try:
            # Load image
            image = load_image(str(image_path))
            if image is None:
                continue
            
            # Run detection ONCE
            detections = run_consensus_detection(image, args.confidence)
            
            if not detections:
                continue
            
            total_detections += len(detections)
            
            # Process each detection
            for det_idx, bbox in enumerate(detections):
                x, y, w, h = bbox
                
                # Create cached experimental region
                exp_region = ExperimentalRegion(image, bbox, args.padding)
                
                # Inner progress bar for granularity sweep
                g_pbar = tqdm(
                    granularity_values, 
                    desc=f"  det {det_idx}", 
                    unit="g",
                    position=1,
                    leave=False
                )
                for granularity in g_pbar:
                    g_pbar.set_postfix_str(f"g={granularity}")
                    test_result = exp_region.test_granularity(granularity, confidence_threshold=0.1)
                    
                    # Build result row
                    row = {
                        'image': image_path.name,
                        'detection_idx': det_idx,
                        'bbox_x': x,
                        'bbox_y': y,
                        'bbox_w': w,
                        'bbox_h': h,
                        'bbox_area': w * h,
                        # Before statistics (includes _before suffix for tracked metrics)
                        **exp_region.stats_before,
                        'edge_density_before': exp_region.edge_density_before,
                        'cc_count_before': exp_region.cc_count_before,
                        # New before metrics (Tier 0-3)
                        'laplacian_var_before': exp_region.laplacian_var_before,
                        'canny_density_before': exp_region.canny_density_before,
                        'row_peakiness_before': exp_region.row_peakiness_before,
                        'blackhat_energy_before': exp_region.blackhat_energy_before,
                        'tophat_energy_before': exp_region.tophat_energy_before,
                        'otsu_separability_before': exp_region.otsu_separability_before,
                        'gradient_energy_before': exp_region.gradient_energy_before,
                        'edge_row_energy_before': exp_region.edge_row_energy_before,
                        # Experiment
                        'granularity': test_result['granularity'],
                        # After metrics
                        'remnancy': test_result['remnancy'],
                        'detection_count': test_result['detection_count'],
                        'otsu_threshold_after': test_result['otsu_threshold_after'],
                        'entropy_after': test_result['entropy_after'],
                        'edge_density_after': test_result['edge_density_after'],
                        'cc_count_after': test_result['cc_count_after'],
                        # New after metrics
                        'laplacian_var_after': test_result['laplacian_var_after'],
                        'canny_density_after': test_result['canny_density_after'],
                        'row_peakiness_after': test_result['row_peakiness_after'],
                        'blackhat_energy_after': test_result['blackhat_energy_after'],
                        'tophat_energy_after': test_result['tophat_energy_after'],
                        'otsu_separability_after': test_result['otsu_separability_after'],
                        'gradient_energy_after': test_result['gradient_energy_after'],
                        'edge_row_energy_after': test_result['edge_row_energy_after'],
                        # Deltas (for metrics with meaningful absolute scales)
                        'otsu_threshold_delta': test_result['otsu_threshold_delta'],
                        'entropy_delta': test_result['entropy_delta'],
                        'edge_density_delta': test_result['edge_density_delta'],
                        'cc_count_delta': test_result['cc_count_delta'],
                        # Ratios (for metrics with varying scales, <1.0 = reduction)
                        'laplacian_var_ratio': test_result['laplacian_var_ratio'],
                        'canny_density_ratio': test_result['canny_density_ratio'],
                        'row_peakiness_ratio': test_result['row_peakiness_ratio'],
                        'blackhat_energy_ratio': test_result['blackhat_energy_ratio'],
                        'tophat_energy_ratio': test_result['tophat_energy_ratio'],
                        'otsu_separability_ratio': test_result['otsu_separability_ratio'],
                        'gradient_energy_ratio': test_result['gradient_energy_ratio'],
                        'edge_row_energy_ratio': test_result['edge_row_energy_ratio'],
                        'processing_time': test_result['processing_time']
                    }
                    
                    # Write row immediately (append mode)
                    should_write_header = write_header and not header_written
                    write_csv_row(output_path, row, should_write_header)
                    header_written = True
                    
                    total_experiments += 1
                    
        except Exception as e:
            tqdm.write(f"ERROR: {image_path.name}: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Images processed: {len(image_files)}")
    print(f"Detections found: {total_detections}")
    print(f"Total experiments: {total_experiments}")
    print(f"Granularity values tested: {granularity_values}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if total_experiments > 0:
        print(f"Average per experiment: {total_time/total_experiments:.2f}s")
    print(f"Results file: {output_path}")
    
    # Run analysis on the results
    if total_experiments > 0:
        run_analysis(output_path)


if __name__ == '__main__':
    main()
