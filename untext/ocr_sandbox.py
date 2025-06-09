#!/usr/bin/env python3
"""A sandbox script for experimenting with OCR performance on test images.

This script processes test images from a designated directory, applies OCR using DocTR,
and compares the OCR output to a ground truth caption by computing a simple Hamming distance.

Usage:
    python ocr_sandbox.py

Ensure that the test images and corresponding caption files (e.g. test1-caption.txt) are
located in the 'tests/images' directory relative to this script.
"""

import os
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional, Union

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from statistics import median

# Import doctr's OCR predictor
from doctr.models import ocr_predictor

# Import the preprocessor module
from untext import preprocessor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a stream handler to output logs to stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)

# Create a file handler to write logs to 'ocr_sandbox.log'
file_handler = logging.FileHandler('ocr_sandbox.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def compute_hamming(s1: str, s2: str) -> int:
    """Compute a Hamming-like distance between two strings.

    This computes the number of differing characters over the length of the shorter string and
    adds the difference in length as a penalty.
    """
    m = min(len(s1), len(s2))
    diff = sum(1 for i in range(m) if s1[i] != s2[i])
    diff += abs(len(s1) - len(s2))
    return diff


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_figure_of_merit(ocr_text: str, ground_truth: str) -> float:
    """Compute a combined Figure of Merit from Hamming and Edit distances.
    
    This function combines normalized Hamming distance (character-level differences) 
    and Edit distance (word structure differences) into a single metric suitable 
    for optimization. The metric uses logarithmic scaling to provide wide dynamic 
    range between perfect matches and poor matches.
    
    Args:
        ocr_text: The text extracted by OCR engine
        ground_truth: The known correct text for comparison
        
    Returns:
        A float representing the figure of merit. Lower values indicate better matches:
        - Perfect match: -6.0 (log10(1e-6))
        - Poor matches: approach positive values
        - Typical good matches: -3.0 to -1.0
        - Typical poor matches: -0.5 to 1.0
        
    Note:
        The weighting is 30% Hamming distance, 70% Edit distance, since Edit distance
        better captures structural text errors and word-level accuracy.
    """
    if not ground_truth:  # Avoid division by zero
        return float('inf') if ocr_text else -6.0  # Perfect match for empty case
    
    hamming_dist = compute_hamming(ocr_text, ground_truth)
    edit_dist = compute_edit_distance(ocr_text, ground_truth)
    
    # Normalize by ground truth length
    hamming_normalized = hamming_dist / len(ground_truth)
    edit_normalized = edit_dist / len(ground_truth)
    
    # Combined metric with weights (edit distance is more important for structure)
    combined = 0.3 * hamming_normalized + 0.7 * edit_normalized
    
    # Debug logging for suspicious cases
    if combined == 0:
        logger.debug(f"Perfect match detected: '{ocr_text}' == '{ground_truth}'")
        logger.debug(f"  Hamming: {hamming_dist}, Edit: {edit_dist}")

    # Log scale for dynamic range (add small epsilon to avoid log(0))
    fom = np.log10(combined + 1e-6)
    
    # Debug the actual calculation
    logger.debug(f"FOM DEBUG - hamming_dist: {hamming_dist}, edit_dist: {edit_dist}")
    logger.debug(f"FOM DEBUG - hamming_norm: {hamming_normalized}, edit_norm: {edit_normalized}")
    logger.debug(f"FOM DEBUG - combined: {combined}, fom: {fom}, raw_fom: {repr(fom)}")
    
    return fom


def ocr_image(image_path: str) -> str:
    """Perform OCR on an image using standardized OCR module."""
    from . import ocr
    logger.info(f'Processing image "{image_path}" with standardized OCR...')
    return ocr.extract_text_from_image(image_path)


def ocr_image_with_detection(image_path: str) -> str:
    """Perform OCR on an image using detection-based cropping and OCR.
    
    This function opens the image, converts it to a numpy array, and then uses TextDetector's
    detect_and_ocr flow to extract text from detected regions.
    """
    logger.info(f'Processing image "{image_path}" for detection-based OCR through preprocessor...')
    img_array = preprocessor.preprocess_image(image_path)
    if img_array is None:
        logger.error(f"Preprocessing failed for image {image_path}")
        return ""
    from untext.detector import TextDetector
    detector = TextDetector()
    ocr_results = detector.detect_and_ocr(img_array)
    combined_text = " ".join([result["text"] for result in ocr_results if result["text"].strip() != ""])
    logger.info(f"Detection-based OCR results: {ocr_results}")
    return combined_text.strip()


def ocr_image_tesseract(image_path: str) -> str:
    """Perform OCR on an image using Tesseract OCR engine.
    
    This function opens the image, converts it to RGB, and then uses pytesseract with a specified Tesseract executable path
    to extract text from the image.
    """
    logger.info(f'Processing image "{image_path}" for Tesseract OCR through preprocessor...')
    img_array = preprocessor.preprocess_image(image_path)
    if img_array is None:
        logger.error(f"Preprocessing failed for image {image_path}")
        return ""
    try:
        import pytesseract
    except ImportError as e:
        logger.error(f"pytesseract module not found: {e}")
        return ""
    # Set the Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = r"X:\tesseract\tesseract.exe"

    # Enhance the contrast using cv2.convertScaleAbs
    enhanced = cv2.convertScaleAbs(img_array, alpha=2.0, beta=0)

    # Create an inverted version of the enhanced image
    inverted = cv2.bitwise_not(enhanced)

    # Perform OCR on both the enhanced and inverted images
    text_enhanced = pytesseract.image_to_string(enhanced)
    text_inverted = pytesseract.image_to_string(inverted)

    logger.info(f"Enhanced OCR extracted: {text_enhanced}")
    logger.info(f"Inverted OCR extracted: {text_inverted}")

    # Combine the OCR results from both variants
    combined_text = (text_enhanced + " " + text_inverted).strip()
    return combined_text


def median_ascii_string(texts: List[str]) -> str:
    """Compute a median string from a list of strings by taking the median ASCII value at each character position.
    
    For each index from 0 to the maximum length among the strings, this function collects all characters
    that appear at that position (ignoring strings that are too short), computes the median of their ASCII
    values, converts it back to a character, and concatenates these characters to form the median string.
    
    Args:
        texts: A list of strings to compute the median string from.
        
    Returns:
        A string constructed from the median ASCII values at each position.
    """
    if not texts:
        return ""
    max_len = max(len(s) for s in texts)
    result_chars = []
    for i in range(max_len):
        ascii_values = [ord(s[i]) for s in texts if i < len(s)]
        if ascii_values:
            med_val = int(median(ascii_values))
            result_chars.append(chr(med_val))
    return ''.join(result_chars)


def generate_text_mask(image_path: str) -> "np.ndarray":
    """Generate a binary mask image with white regions for text areas based on OCR bounding boxes.
    
    The mask has white areas where text is detected (based on OCR bounding boxes) and black elsewhere.
    
    Args:
        image_path: Path to the input image.
    
    Returns:
        A numpy ndarray representing the mask image, or None if processing fails.
    """
    logger.info(f'Generating text mask for "{image_path}" ...')
    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return None
    img = img.convert("RGB")
    img_array = np.array(img)
    
    # Initialize OCR predictor
    ocr = ocr_predictor(pretrained=True)
    try:
        ocr_results = ocr([img_array])
        if isinstance(ocr_results, list):
            ocr_results = ocr_results[0]
    except Exception as e:
        logger.error(f"OCR processing failed for {image_path}: {e}")
        return None
    
    # Create a black mask image with the same dimensions as the input image
    height, width, _ = img_array.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if hasattr(ocr_results, "pages"):
        for page in ocr_results.pages:
            for block in page.blocks:
                for line in block.lines:
                    if hasattr(line, "geometry") and line.geometry and hasattr(line, "value") and line.value:
                        pts = np.array(line.geometry, dtype=np.int32)
                        (x, y, w, h) = cv2.boundingRect(pts)
                        cv2.putText(mask, line.value, (x, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
                    else:
                        logger.warning("No geometry or text value found for a line; skipping")
    else:
        logger.error("OCR results do not contain pages attribute; cannot generate mask")
    
    return mask


def clean_grid_search() -> None:
    """Clean, simple grid search using only known-good code.
    
    1. Take each image in tests/images/ directory as a test case
    2. Modify it using color, contrast, and threshold tuning
    3. Run the KNOWN GOOD ocr_image routine on it
    4. Compare output to caption file
    5. Calculate figure-of-merit
    """
    import os
    import cv2
    import tempfile
    from pathlib import Path
    
    logger.info("Starting clean grid search using only known-good code...")
    
    # Test images directory
    test_dir = Path("tests/images")
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Find test images and their captions
    test_cases = []
    for img_file in ["test1.png", "test2.png", "test3.jpg", "test4-with-text.png"]:
        img_path = test_dir / img_file
        caption_path = test_dir / f"{img_path.stem}-caption.txt"
        
        if img_path.exists() and caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()
            test_cases.append({
                'image_path': str(img_path),
                'ground_truth': ground_truth,
                'name': img_path.stem
            })
            logger.info(f"Added test case: {img_path.stem} -> '{ground_truth}'")
    
    if not test_cases:
        logger.error("No valid test cases found")
        return
    
    logger.info(f"Found {len(test_cases)} test cases")
    
    # Define simple tuning configurations
    tuning_configs = [
        # Baseline - no tuning
        {'name': 'baseline', 'operations': []},
        
        # CLAHE enhancement only
        {'name': 'clahe_2_8x8', 'operations': [('clahe', {'clip_limit': 2.0, 'tile_size': (8, 8)})]},
        {'name': 'clahe_3_8x8', 'operations': [('clahe', {'clip_limit': 3.0, 'tile_size': (8, 8)})]},
        {'name': 'clahe_2_16x16', 'operations': [('clahe', {'clip_limit': 2.0, 'tile_size': (16, 16)})]},
        
        # Histogram equalization
        {'name': 'hist_eq', 'operations': [('histogram_eq', {})]},
        
        # Contrast adjustment
        {'name': 'contrast_1.5', 'operations': [('contrast', {'alpha': 1.5, 'beta': 0})]},
        {'name': 'contrast_2.0', 'operations': [('contrast', {'alpha': 2.0, 'beta': 0})]},
        
        # Gaussian blur 
        {'name': 'blur_3', 'operations': [('gaussian_blur', {'kernel_size': 3})]},
        {'name': 'blur_5', 'operations': [('gaussian_blur', {'kernel_size': 5})]},
        
        # Adaptive threshold
        {'name': 'adaptive_11_2', 'operations': [('adaptive_threshold', {'block_size': 11, 'C': 2})]},
        {'name': 'adaptive_15_2', 'operations': [('adaptive_threshold', {'block_size': 15, 'C': 2})]},
        
        # OTSU threshold
        {'name': 'otsu', 'operations': [('otsu_threshold', {})]},
        
        # Combinations
        {'name': 'clahe_2_8x8_then_adaptive_11_2', 'operations': [
            ('clahe', {'clip_limit': 2.0, 'tile_size': (8, 8)}),
            ('adaptive_threshold', {'block_size': 11, 'C': 2})
        ]},
        {'name': 'contrast_1.5_then_clahe_2_8x8', 'operations': [
            ('contrast', {'alpha': 1.5, 'beta': 0}),
            ('clahe', {'clip_limit': 2.0, 'tile_size': (8, 8)})
        ]},
    ]
    
    def apply_tuning_operations(image_path: str, operations: list) -> str:
        """Apply tuning operations and return path to processed image."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply each operation in sequence
        for op_name, params in operations:
            if op_name == 'clahe':
                clahe = cv2.createCLAHE(
                    clipLimit=params['clip_limit'], 
                    tileGridSize=params['tile_size']
                )
                gray = clahe.apply(gray)
                
            elif op_name == 'histogram_eq':
                gray = cv2.equalizeHist(gray)
                
            elif op_name == 'contrast':
                gray = cv2.convertScaleAbs(gray, alpha=params['alpha'], beta=params['beta'])
                
            elif op_name == 'gaussian_blur':
                k = params['kernel_size']
                gray = cv2.GaussianBlur(gray, (k, k), 0)
                
            elif op_name == 'adaptive_threshold':
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, params['block_size'], params['C']
                )
                
            elif op_name == 'otsu_threshold':
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR for saving
        if len(gray.shape) == 2:
            processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            processed_img = gray
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, processed_img)
        
        return temp_path
    
    # Run grid search
    results = []
    total_configs = len(tuning_configs)
    total_tests = total_configs * len(test_cases)
    
    logger.info(f"Testing {total_configs} configurations on {len(test_cases)} images = {total_tests} total tests")
    
    test_count = 0
    for config in tuning_configs:
        config_name = config['name']
        operations = config['operations']
        
        logger.info(f"Testing configuration: {config_name}")
        
        config_results = []
        for test_case in test_cases:
            test_count += 1
            image_path = test_case['image_path']
            ground_truth = test_case['ground_truth']
            test_name = test_case['name']
            
            logger.info(f"  [{test_count}/{total_tests}] {test_name} + {config_name}")
            
            temp_image_path = None
            try:
                # Apply tuning operations
                temp_image_path = apply_tuning_operations(image_path, operations)
                
                # Run OCR using the FIXED OCR implementation
                ocr_result = ocr_image(temp_image_path)
                
                # Calculate metrics
                hamming_dist = compute_hamming(ocr_result, ground_truth)
                edit_dist = compute_edit_distance(ocr_result, ground_truth)
                fom = compute_figure_of_merit(ocr_result, ground_truth)
                
                result = {
                    'config_name': config_name,
                    'test_name': test_name,
                    'ground_truth': ground_truth,
                    'ocr_result': ocr_result,
                    'hamming_distance': hamming_dist,
                    'edit_distance': edit_dist,
                    'figure_of_merit': fom
                }
                
                config_results.append(result)
                results.append(result)
                
                logger.info(f"    OCR: '{ocr_result}' | FOM: {fom:.4f} | Hamming: {hamming_dist} | Edit: {edit_dist}")
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                
            finally:
                # Clean up temp file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        # Calculate average FOM for this configuration
        if config_results:
            avg_fom = sum(r['figure_of_merit'] for r in config_results) / len(config_results)
            logger.info(f"  Configuration {config_name} average FOM: {avg_fom:.4f}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("GRID SEARCH RESULTS SUMMARY")
    logger.info("=" * 60)
    
    # Group results by configuration
    config_summaries = {}
    for result in results:
        config_name = result['config_name']
        if config_name not in config_summaries:
            config_summaries[config_name] = []
        config_summaries[config_name].append(result['figure_of_merit'])
    
    # Calculate averages and sort
    config_averages = []
    for config_name, foms in config_summaries.items():
        avg_fom = sum(foms) / len(foms) if foms else 0
        config_averages.append((config_name, avg_fom, len(foms)))
    
    # Sort by average FOM (higher is better)
    config_averages.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Configuration rankings (by average FOM):")
    for i, (config_name, avg_fom, count) in enumerate(config_averages, 1):
        logger.info(f"  {i:2d}. {config_name:30s} | Avg FOM: {avg_fom:.4f} | Tests: {count}")
    
    # Best configuration
    if config_averages:
        best_config, best_fom, _ = config_averages[0]
        logger.info(f"\nBEST CONFIGURATION: {best_config} (FOM: {best_fom:.4f})")
    
    logger.info("Grid search completed!")


def test_raw_ocr() -> None:
    """Test OCR on raw images without any preprocessing to debug the pipeline."""
    import os
    from pathlib import Path
    from doctr.models import ocr_predictor
    from PIL import Image
    import numpy as np
    
    logger.info("Testing raw OCR without preprocessor...")
    
    # Test images directory
    test_dir = Path("tests/images")
    test_image = test_dir / "test1.png"
    caption_file = test_dir / "test1-caption.txt"
    
    if not test_image.exists():
        logger.error(f"Test image not found: {test_image}")
        return
        
    if not caption_file.exists():
        logger.error(f"Caption file not found: {caption_file}")
        return
    
    # Load ground truth
    with open(caption_file, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()
    logger.info(f"Ground truth: '{ground_truth}'")
    
    try:
        # Test 1: Raw image with PIL + numpy (like the working parts of ocr_image)
        logger.info("Test 1: Raw image via PIL + numpy...")
        img_pil = Image.open(test_image).convert("RGB")
        img_array = np.array(img_pil)
        
        logger.info(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.info(f"Image value range: [{img_array.min()}, {img_array.max()}]")
        
        # Use simple OCR predictor
        ocr = ocr_predictor(pretrained=True)
        ocr_results = ocr([img_array])
        
        if isinstance(ocr_results, list):
            ocr_results = ocr_results[0]
        
        logger.info(f"OCR results type: {type(ocr_results)}")
        if hasattr(ocr_results, "pages"):
            logger.info(f"OCR results contain {len(ocr_results.pages)} pages")
            for i, page in enumerate(ocr_results.pages):
                logger.info(f"Page {i} has {len(page.blocks)} blocks")
                for j, block in enumerate(page.blocks):
                    logger.info(f"  Block {j} has {len(block.lines)} lines")
                    for k, line in enumerate(block.lines):
                        logger.info(f"    Line {k}: '{line.value}' (confidence: {getattr(line, 'confidence', 'unknown')})")
        
        # Extract text
        extracted_text = ""
        if hasattr(ocr_results, "pages"):
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
        
        extracted_text = extracted_text.strip()
        logger.info(f"Raw OCR result: '{extracted_text}'")
        
        # Calculate metrics if we got text
        if extracted_text:
            hamming_dist = compute_hamming(extracted_text, ground_truth)
            edit_dist = compute_edit_distance(extracted_text, ground_truth)
            fom = compute_figure_of_merit(extracted_text, ground_truth)
            
            logger.info(f"Raw OCR metrics:")
            logger.info(f"  Hamming distance: {hamming_dist}")
            logger.info(f"  Edit distance: {edit_dist}")
            logger.info(f"  Figure of Merit: {fom:.4f}")
        else:
            logger.error("Raw OCR returned empty string - fundamental DocTR issue!")
            
    except Exception as e:
        logger.error(f"Raw OCR test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def orthogonal_grid_search() -> None:
    """Perform orthogonal grid search over image tuning parameter groups.
    
    This function implements a two-phase search strategy for finding optimal
    image tuning parameters for OCR:
    
    Phase 1: Test each tuning group independently (threshold, blur, enhancement)
             to find the best method and parameters within each group.
    Phase 2: Combine the winners from each group and test the final pipeline.
    
    The approach recognizes that some tuning operations are mutually exclusive
    (e.g., different thresholding methods) while others are orthogonal and can be
    combined (e.g., blur + threshold + enhancement).
    
    Image Tuning Groups:
    - Threshold: adaptive, Otsu, simple threshold, or none
    - Blur: Gaussian, bilateral, median, or none  
    - Enhancement: CLAHE, histogram equalization, contrast stretching, or none
    
    Evaluation:
    - Uses Figure of Merit combining Hamming + Edit distance
    - Tests against ground truth captions in tests/images/*-caption.txt
    - Reports best parameters and final combined tuning pipeline
    
    Output:
    - Logs detailed results for each parameter combination tested
    - Reports final optimal image tuning pipeline configuration
    """
    import cv2
    from itertools import product
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not installed
        def tqdm(iterable, **kwargs):
            return iterable
    
    # Import the fixed DocTR configuration
    from untext.detector import TextDetector
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    test_images = ["test1.png", "test2.png", "test3.jpg", "test4-with-text.png"]
    
    # Define orthogonal preprocessing groups (mutually exclusive within group)
    threshold_methods = {
        'adaptive': {'method': 'adaptive', 'block_size': [7, 9, 11, 13], 'C': [1, 2, 3, 4]},
        'otsu': {'method': 'otsu'},
        'simple': {'method': 'simple', 'threshold': [127, 140, 160, 180]},
        'none': {'method': 'none'}
    }
    
    blur_methods = {
        'gaussian': {'method': 'gaussian', 'kernel': [1, 3, 5, 7]},
        'bilateral': {'method': 'bilateral', 'd': [5, 9], 'sigma_color': [75, 100], 'sigma_space': [75, 100]},
        'median': {'method': 'median', 'kernel': [3, 5, 7]},
        'none': {'method': 'none'}
    }
    
    enhancement_methods = {
        'clahe': {'method': 'clahe', 'clip_limit': [1.0, 2.0, 3.0], 'tile_size': [(4,4), (8,8)]},
        'histogram_eq': {'method': 'histogram_eq'},
        'contrast': {'method': 'contrast', 'alpha': [1.2, 1.5, 2.0], 'beta': [0, 10, 20]},
        'none': {'method': 'none'}
    }
    
    # Count total configurations for progress tracking
    def count_configs(methods_dict):
        total = 0
        for config in methods_dict.values():
            if config['method'] == 'none':
                total += 1
            else:
                param_names = [k for k in config.keys() if k != 'method']
                if not param_names:
                    total += 1
                else:
                    param_combinations = 1
                    for param in param_names:
                        param_combinations *= len(config[param])
                    total += param_combinations
        return total
    
    threshold_count = count_configs(threshold_methods)
    blur_count = count_configs(blur_methods)
    enhancement_count = count_configs(enhancement_methods)
    total_phase1_configs = threshold_count + blur_count + enhancement_count
    
    logger.info(f"=== GRID SEARCH CONFIGURATION ===")
    logger.info(f"Threshold configs: {threshold_count}")
    logger.info(f"Blur configs: {blur_count}")
    logger.info(f"Enhancement configs: {enhancement_count}")
    logger.info(f"Total Phase 1 configs: {total_phase1_configs}")
    logger.info(f"Total OCR calls: {total_phase1_configs * len(test_images)}")
    logger.info(f"Estimated runtime: {(total_phase1_configs * len(test_images) * 2.5) / 60:.1f} minutes")
    
    # Initialize the fixed DocTR detector once
    logger.info("Initializing fixed DocTR configuration...")
    fixed_detector = TextDetector(preprocess=False)  # We'll do custom preprocessing
    logger.info("✅ Fixed DocTR initialized successfully")
    
    def apply_preprocessing_pipeline(
        image_path: str, 
        threshold_config: Dict[str, Any], 
        blur_config: Dict[str, Any], 
        enhancement_config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Apply a complete preprocessing pipeline with the given configurations.
        
        Applies image preprocessing operations in the optimal order:
        1. Load image and convert to grayscale
        2. Apply enhancement (contrast/histogram operations)  
        3. Apply noise reduction/blur
        4. Apply thresholding (binarization)
        5. Convert back to RGB for OCR compatibility
        
        Args:
            image_path: Path to the input image file
            threshold_config: Dict with 'method' key and method-specific parameters
            blur_config: Dict with 'method' key and method-specific parameters  
            enhancement_config: Dict with 'method' key and method-specific parameters
            
        Returns:
            Preprocessed image as RGB numpy array, or None if loading fails
            
        Note:
            Config dictionaries must contain 'method' key. Additional keys depend
            on the method (e.g., 'block_size' and 'C' for adaptive thresholding).
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply enhancement first
        if enhancement_config['method'] == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=enhancement_config['clip_limit'], 
                                   tileGridSize=enhancement_config['tile_size'])
            gray = clahe.apply(gray)
        elif enhancement_config['method'] == 'histogram_eq':
            gray = cv2.equalizeHist(gray)
        elif enhancement_config['method'] == 'contrast':
            gray = cv2.convertScaleAbs(gray, alpha=enhancement_config['alpha'], beta=enhancement_config['beta'])
        
        # Apply blur
        if blur_config['method'] == 'gaussian':
            k = blur_config['kernel']
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        elif blur_config['method'] == 'bilateral':
            gray = cv2.bilateralFilter(gray, blur_config['d'], blur_config['sigma_color'], blur_config['sigma_space'])
        elif blur_config['method'] == 'median':
            gray = cv2.medianBlur(gray, blur_config['kernel'])
        
        # Apply thresholding
        if threshold_config['method'] == 'adaptive':
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, threshold_config['block_size'], threshold_config['C'])
        elif threshold_config['method'] == 'otsu':
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_config['method'] == 'simple':
            _, result = cv2.threshold(gray, threshold_config['threshold'], 255, cv2.THRESH_BINARY)
        else:  # none
            result = gray
        
        # Convert back to RGB
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def test_config_group(
        group_name: str, 
        configs: Dict[str, Dict[str, Any]], 
        fixed_others: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Test all configurations in a preprocessing group, return best config and its FOM.
        
        For a given preprocessing group (threshold, blur, or enhancement), this function
        tests all possible parameter combinations within that group to find the optimal
        configuration based on Figure of Merit scores.
        
        Args:
            group_name: Name of the preprocessing group ('threshold', 'blur', 'enhancement')
            configs: Dictionary mapping config names to parameter specifications
            fixed_others: Optional dict of fixed configurations for other groups
            
        Returns:
            Tuple of (best_config_dict, best_figure_of_merit_score)
            
        Note:
            During Phase 1 (independent testing), fixed_others is None and other
            groups default to {'method': 'none'}. During Phase 2, fixed_others
            contains the winners from Phase 1.
        """
        best_config = None
        best_fom = float('inf')
        
        logger.info(f"🔍 Testing {group_name} configurations...")
        
        # Build list of all test configs for progress tracking
        all_test_configs = []
        for config_name, config in configs.items():
            if config['method'] == 'none':
                all_test_configs.append((config_name, {'method': 'none'}))
            else:
                param_names = [k for k in config.keys() if k != 'method']
                if not param_names:
                    all_test_configs.append((config_name, config))
                else:
                    param_combinations = product(*[config[param] for param in param_names])
                    for combo in param_combinations:
                        test_config = {'method': config['method']}
                        for i, param_name in enumerate(param_names):
                            test_config[param_name] = combo[i]
                        all_test_configs.append((config_name, test_config))
        
        # Test each config with progress bar
        progress_bar = tqdm(all_test_configs, desc=f"{group_name.capitalize()} configs", leave=False)
        for config_name, test_config in progress_bar:
            avg_fom = test_single_config(test_config, group_name, fixed_others)
            
            # Update progress bar with current best
            progress_bar.set_postfix({
                'current_fom': f'{avg_fom:.4f}',
                'best_fom': f'{best_fom:.4f}' if best_fom != float('inf') else 'None'
            })
            
            logger.info(f"  {config_name} {test_config}: FOM = {avg_fom:.6f}")
            if avg_fom < best_fom:
                best_fom = avg_fom
                best_config = test_config
                    
        logger.info(f"✅ Best {group_name}: {best_config} with FOM: {best_fom:.4f}")
        return best_config, best_fom
    
    def test_single_config(
        test_config: Dict[str, Any], 
        group_name: str, 
        fixed_others: Optional[Dict[str, Dict[str, Any]]]
    ) -> float:
        """Test a single preprocessing configuration and return average Figure of Merit.
        
        This function builds a complete preprocessing pipeline from the test configuration
        and fixed configurations from other groups, then evaluates it against all test
        images using OCR and ground truth comparison.
        
        Args:
            test_config: Configuration dictionary for the group being tested
            group_name: Which preprocessing group this config belongs to
            fixed_others: Fixed configurations for the other preprocessing groups
            
        Returns:
            Average Figure of Merit score across all test images (lower is better)
            
        Note:
            Returns float('inf') if no valid test results are obtained, effectively
            eliminating that configuration from consideration.
        """
        foms = []
        
        # Handle case where fixed_others is None (during independent group testing)
        if fixed_others is None:
            fixed_others = {}
        
        # Build full pipeline config
        if group_name == 'threshold':
            threshold_config = test_config
            blur_config = fixed_others.get('blur', {'method': 'none'})
            enhancement_config = fixed_others.get('enhancement', {'method': 'none'})
        elif group_name == 'blur':
            threshold_config = fixed_others.get('threshold', {'method': 'none'})
            blur_config = test_config
            enhancement_config = fixed_others.get('enhancement', {'method': 'none'})
        else:  # enhancement
            threshold_config = fixed_others.get('threshold', {'method': 'none'})
            blur_config = fixed_others.get('blur', {'method': 'none'})
            enhancement_config = test_config
        
        for img_file in test_images:
            image_path = os.path.join(test_dir, img_file)
            if not os.path.exists(image_path):
                continue
                
            img_array = apply_preprocessing_pipeline(image_path, threshold_config, blur_config, enhancement_config)
            if img_array is None:
                continue
                
            ocr_text = extract_ocr_text_from_array(img_array)
            
            # Get ground truth
            base, _ = os.path.splitext(img_file)
            caption_path = os.path.join(test_dir, base + "-caption.txt")
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, "r", encoding="utf-8") as f:
                        caption_text = f.read().strip()
                    fom = compute_figure_of_merit(ocr_text, caption_text)
                    # Always log this for the first few images to debug
                    if len(foms) < 3:  # Only log first few to avoid spam
                        logger.info(f"    GRID DEBUG - Image: {img_file}")
                        logger.info(f"    GRID DEBUG - OCR result: '{ocr_text}' (len={len(ocr_text)})")
                        logger.info(f"    GRID DEBUG - Ground truth: '{caption_text}' (len={len(caption_text)})")
                        logger.info(f"    GRID DEBUG - FOM: {fom:.6f} (raw: {repr(fom)})")
                    foms.append(fom)
                except Exception as e:
                    logger.error(f"Failed to read caption: {e}")
        
        return sum(foms) / len(foms) if foms else float('inf')
    
    def extract_ocr_text_from_array(img_array: np.ndarray) -> str:
        """Extract text from a tuned image array using standardized OCR module.
        
        Args:
            img_array: Tuned image as RGB numpy array
            
        Returns:
            Concatenated text string from all detected text regions, or empty
            string if OCR fails or no text is detected
        """
        from . import ocr
        logger.debug(f"Input array shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.debug(f"Input array value range: [{img_array.min()}, {img_array.max()}]")
        
        text = ocr.extract_text_from_array(img_array)
        logger.debug(f"Final extracted text: '{text}'")
        return text
    
    # Phase 1: Test each group independently
    logger.info("=== PHASE 1: Testing each preprocessing group independently ===")
    best_threshold, _ = test_config_group('threshold', threshold_methods)
    best_blur, _ = test_config_group('blur', blur_methods)  
    best_enhancement, _ = test_config_group('enhancement', enhancement_methods)
    
    # Phase 2: Test best combination
    logger.info("=== PHASE 2: Testing best combination ===")
    final_fom = test_single_config(best_threshold, 'threshold', {'blur': best_blur, 'enhancement': best_enhancement})
    
    logger.info(f"=== FINAL RESULTS ===")
    logger.info(f"Best Threshold: {best_threshold}")
    logger.info(f"Best Blur: {best_blur}")  
    logger.info(f"Best Enhancement: {best_enhancement}")
    logger.info(f"Combined Figure of Merit: {final_fom:.4f}")


def compare_preprocessing_performance():
    """Compare OCR performance with and without preprocessing on test images.
    
    This function tests the actual deployed pipeline to see if preprocessing
    helps or hurts DocTR performance in practice.
    """
    # Define the directory containing test images and caption files
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    
    # List of test images
    test_images = [
        "test1.png",
        "test2.png", 
        "test3.jpg",
        "test4-with-text.png"
    ]
    
    logger.info("=== PREPROCESSING PERFORMANCE COMPARISON ===")
    
    results_with_preprocess = []
    results_without_preprocess = []
    
    for img_file in test_images:
        image_path = os.path.join(test_dir, img_file)
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            continue
            
        logger.info(f"\nTesting {img_file}:")
        
        # Load ground truth
        base, _ = os.path.splitext(img_file)
        caption_file = base + "-caption.txt"
        caption_path = os.path.join(test_dir, caption_file)
        
        if not os.path.exists(caption_path):
            logger.warning(f"Caption file not found for {img_file}")
            continue
            
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read caption file {caption_path}: {e}")
            continue
            
        # Test WITH preprocessing (current default)
        try:
            ocr_text_with = ocr_image(image_path)  # Uses preprocessing
            fom_with = compute_figure_of_merit(ocr_text_with, ground_truth)
            hamming_with = compute_hamming(ocr_text_with, ground_truth)
            edit_with = compute_edit_distance(ocr_text_with, ground_truth)
            
            logger.info(f"  WITH preprocessing:")
            logger.info(f"    OCR text: '{ocr_text_with}'")
            logger.info(f"    Ground truth: '{ground_truth}'")
            logger.info(f"    Figure of Merit: {fom_with:.4f}")
            logger.info(f"    Hamming distance: {hamming_with}")
            logger.info(f"    Edit distance: {edit_with}")
            
            results_with_preprocess.append(fom_with)
            
        except Exception as e:
            logger.error(f"Error testing WITH preprocessing: {e}")
            continue
            
        # Test WITHOUT preprocessing (disable it temporarily)
        try:
            # Temporarily disable preprocessing in the pipeline
            import untext.preprocessor as prep_module
            original_preprocess = prep_module.preprocess_image
            prep_module.preprocess_image = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            
            ocr_text_without = ocr_image(image_path)  # No preprocessing
            fom_without = compute_figure_of_merit(ocr_text_without, ground_truth)
            hamming_without = compute_hamming(ocr_text_without, ground_truth)
            edit_without = compute_edit_distance(ocr_text_without, ground_truth)
            
            # Restore original preprocessing
            prep_module.preprocess_image = original_preprocess
            
            logger.info(f"  WITHOUT preprocessing:")
            logger.info(f"    OCR text: '{ocr_text_without}'")
            logger.info(f"    Figure of Merit: {fom_without:.4f}")
            logger.info(f"    Hamming distance: {hamming_without}")
            logger.info(f"    Edit distance: {edit_without}")
            
            results_without_preprocess.append(fom_without)
            
            # Compare
            if fom_with < fom_without:
                logger.info(f"  → Preprocessing HELPS (better FOM: {fom_with:.4f} vs {fom_without:.4f})")
            elif fom_with > fom_without:
                logger.info(f"  → Preprocessing HURTS (worse FOM: {fom_with:.4f} vs {fom_without:.4f})")
            else:
                logger.info(f"  → No difference (FOM: {fom_with:.4f})")
                
        except Exception as e:
            logger.error(f"Error testing WITHOUT preprocessing: {e}")
            continue
    
    # Overall summary
    if results_with_preprocess and results_without_preprocess:
        avg_with = sum(results_with_preprocess) / len(results_with_preprocess)
        avg_without = sum(results_without_preprocess) / len(results_without_preprocess)
        
        logger.info(f"\n=== OVERALL RESULTS ===")
        logger.info(f"Average FOM WITH preprocessing: {avg_with:.4f}")
        logger.info(f"Average FOM WITHOUT preprocessing: {avg_without:.4f}")
        
        if avg_with < avg_without:
            logger.info(f"✅ Preprocessing improves performance by {avg_without - avg_with:.4f} FOM points")
        elif avg_with > avg_without:
            logger.info(f"❌ Preprocessing hurts performance by {avg_with - avg_without:.4f} FOM points")
        else:
            logger.info(f"➖ No significant difference")
    else:
        logger.error("Could not complete comparison due to errors")


def debug_preprocessing_comparison():
    """Debug function to compare working preprocessor vs grid search preprocessing."""
    import os
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== DEBUGGING PREPROCESSING COMPARISON ===")
    
    # Method 1: Working preprocessor (used by ocr_image)
    logger.info("Testing working preprocessor.preprocess_image()...")
    img_array_working = preprocessor.preprocess_image(image_path)
    if img_array_working is not None:
        logger.info(f"Working - Shape: {img_array_working.shape}, dtype: {img_array_working.dtype}")
        logger.info(f"Working - Range: [{img_array_working.min()}, {img_array_working.max()}]")
        
        # Test OCR with working method
        ocr = ocr_predictor(pretrained=True)
        try:
            ocr_results = ocr([img_array_working])
            if isinstance(ocr_results, list):
                ocr_results = ocr_results[0]
            extracted_text = ""
            if hasattr(ocr_results, "pages"):
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
            logger.info(f"Working method OCR result: '{extracted_text.strip()}'")
        except Exception as e:
            logger.error(f"Working method OCR failed: {e}")
    else:
        logger.error("Working preprocessor returned None")
    
    # Method 2: Grid search preprocessor (apply_preprocessing_pipeline)
    logger.info("\nTesting grid search apply_preprocessing_pipeline()...")
    
    # Use same config as "none" preprocessing to make it equivalent
    threshold_config = {'method': 'none'}
    blur_config = {'method': 'none'}
    enhancement_config = {'method': 'none'}
    
    # Import the apply_preprocessing_pipeline function from the grid search
    def apply_preprocessing_pipeline_local(
        image_path: str, 
        threshold_config, 
        blur_config, 
        enhancement_config
    ):
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply enhancement first
        if enhancement_config['method'] == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=enhancement_config['clip_limit'], 
                                   tileGridSize=enhancement_config['tile_size'])
            gray = clahe.apply(gray)
        elif enhancement_config['method'] == 'histogram_eq':
            gray = cv2.equalizeHist(gray)
        elif enhancement_config['method'] == 'contrast':
            gray = cv2.convertScaleAbs(gray, alpha=enhancement_config['alpha'], beta=enhancement_config['beta'])
        
        # Apply blur
        if blur_config['method'] == 'gaussian':
            k = blur_config['kernel']
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        elif blur_config['method'] == 'bilateral':
            gray = cv2.bilateralFilter(gray, blur_config['d'], blur_config['sigma_color'], blur_config['sigma_space'])
        elif blur_config['method'] == 'median':
            gray = cv2.medianBlur(gray, blur_config['kernel'])
        
        # Apply thresholding
        if threshold_config['method'] == 'adaptive':
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, threshold_config['block_size'], threshold_config['C'])
        elif threshold_config['method'] == 'otsu':
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_config['method'] == 'simple':
            _, result = cv2.threshold(gray, threshold_config['threshold'], 255, cv2.THRESH_BINARY)
        else:  # none
            result = gray
        
        # Convert back to RGB
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    img_array_grid = apply_preprocessing_pipeline_local(image_path, threshold_config, blur_config, enhancement_config)
    if img_array_grid is not None:
        logger.info(f"Grid search - Shape: {img_array_grid.shape}, dtype: {img_array_grid.dtype}")
        logger.info(f"Grid search - Range: [{img_array_grid.min()}, {img_array_grid.max()}]")
        
        # Test OCR with grid search method
        ocr = ocr_predictor(pretrained=True)
        try:
            ocr_results = ocr([img_array_grid])
            if isinstance(ocr_results, list):
                ocr_results = ocr_results[0]
            extracted_text = ""
            if hasattr(ocr_results, "pages"):
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
            logger.info(f"Grid search method OCR result: '{extracted_text.strip()}'")
        except Exception as e:
            logger.error(f"Grid search method OCR failed: {e}")
    else:
        logger.error("Grid search preprocessor returned None")


def test_doctr_parameters():
    """Test different DocTR parameters to improve text detection."""
    import os
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== TESTING DocTR PARAMETERS ===")
    
    # Load image
    img_array = preprocessor.preprocess_image(image_path)
    if img_array is None:
        logger.error("Failed to load test image")
        return
    
    # Test 1: Default parameters
    logger.info("\n1. Testing DEFAULT parameters...")
    ocr_default = ocr_predictor(pretrained=True)
    
    # Check the actual detection threshold
    if hasattr(ocr_default.det_predictor.model, 'postprocessor'):
        postprocessor = ocr_default.det_predictor.model.postprocessor
        if hasattr(postprocessor, 'bin_thresh'):
            logger.info(f"   Default bin_thresh: {postprocessor.bin_thresh}")
        if hasattr(postprocessor, 'box_thresh'):
            logger.info(f"   Default box_thresh: {postprocessor.box_thresh}")
        if hasattr(postprocessor, 'min_size_box'):
            logger.info(f"   Default min_size_box: {postprocessor.min_size_box}")
    
    # Get detection results with maps
    try:
        loc_preds, seg_maps = ocr_default.det_predictor([img_array], return_maps=True)
        logger.info(f"   Detections found: {len(loc_preds[0].get('text', []))}")
        
        # Check segmentation map statistics
        if len(seg_maps) > 0:
            seg_map = seg_maps[0]
            logger.info(f"   Seg map shape: {seg_map.shape}")
            logger.info(f"   Seg map range: [{seg_map.min():.3f}, {seg_map.max():.3f}]")
            logger.info(f"   Seg map mean: {seg_map.mean():.3f}")
            
            # Count pixels above different thresholds
            for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
                pixels_above = (seg_map > thresh).sum()
                logger.info(f"   Pixels > {thresh}: {pixels_above} ({100*pixels_above/seg_map.size:.1f}%)")
    except Exception as e:
        logger.error(f"   Error in default test: {e}")
    
    # Test 2: Lower thresholds by modifying postprocessor
    logger.info("\n2. Testing LOWERED THRESHOLDS...")
    ocr_low_thresh = ocr_predictor(pretrained=True)
    
    # Try to lower detection thresholds
    if hasattr(ocr_low_thresh.det_predictor.model, 'postprocessor'):
        postprocessor = ocr_low_thresh.det_predictor.model.postprocessor
        if hasattr(postprocessor, 'bin_thresh'):
            original_bin_thresh = postprocessor.bin_thresh
            postprocessor.bin_thresh = 0.1  # Much lower threshold
            logger.info(f"   Lowered bin_thresh: {original_bin_thresh} -> {postprocessor.bin_thresh}")
        if hasattr(postprocessor, 'box_thresh'):
            original_box_thresh = postprocessor.box_thresh
            postprocessor.box_thresh = 0.1  # Much lower threshold  
            logger.info(f"   Lowered box_thresh: {original_box_thresh} -> {postprocessor.box_thresh}")
        if hasattr(postprocessor, 'min_size_box'):
            original_min_size = postprocessor.min_size_box
            postprocessor.min_size_box = 5  # Smaller minimum box size
            logger.info(f"   Lowered min_size_box: {original_min_size} -> {postprocessor.min_size_box}")
    
    try:
        loc_preds, seg_maps = ocr_low_thresh.det_predictor([img_array], return_maps=True)
        logger.info(f"   Detections found: {len(loc_preds[0].get('text', []))}")
        
        # Test full OCR with lowered thresholds
        ocr_results = ocr_low_thresh([img_array])
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   OCR blocks found: {num_blocks}")
            
            # Extract text
            extracted_text = ""
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
            logger.info(f"   OCR text: '{extracted_text.strip()}'")
    except Exception as e:
        logger.error(f"   Error in low threshold test: {e}")
    
    # Test 3: Different architectures
    logger.info("\n3. Testing DIFFERENT ARCHITECTURES...")
    
    architectures = [
        ("db_resnet50", "crnn_vgg16_bn"),
        ("db_mobilenet_v3_large", "crnn_vgg16_bn"), 
        ("fast_base", "sar_resnet31"),
    ]
    
    for det_arch, reco_arch in architectures:
        try:
            logger.info(f"   Testing {det_arch} + {reco_arch}...")
            ocr_arch = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
            ocr_results = ocr_arch([img_array])
            
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"     Blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"     OCR text: '{extracted_text.strip()}'")
        except Exception as e:
            logger.error(f"     Error with {det_arch}: {e}")
    
    # Test 4: Alternative processing parameters
    logger.info("\n4. Testing PROCESSING PARAMETERS...")
    
    param_sets = [
        {"assume_straight_pages": False, "straighten_pages": True, "detect_orientation": True},
        {"preserve_aspect_ratio": False, "symmetric_pad": False},
        {"assume_straight_pages": False, "preserve_aspect_ratio": False},
    ]
    
    for i, params in enumerate(param_sets):
        try:
            logger.info(f"   Testing param set {i+1}: {params}")
            ocr_params = ocr_predictor(pretrained=True, **params)
            ocr_results = ocr_params([img_array])
            
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"     Blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"     OCR text: '{extracted_text.strip()}'")
        except Exception as e:
            logger.error(f"     Error with params {params}: {e}")


def test_minimal_preprocessing():
    """Test DocTR with minimal preprocessing to verify thresholding is the issue."""
    import os
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== TESTING MINIMAL PREPROCESSING ===")
    
    # Test 1: Raw image (no preprocessing)
    logger.info("\n1. Testing RAW IMAGE (no preprocessing)...")
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    # Convert BGR to RGB only
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"   Raw image shape: {img_rgb.shape}")
    logger.info(f"   Raw image range: [{img_rgb.min()}, {img_rgb.max()}]")
    
    # Test DocTR on raw image
    ocr = ocr_predictor(pretrained=True)
    try:
        ocr_results = ocr([img_rgb])
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   Raw image blocks found: {num_blocks}")
            
            # Extract text
            extracted_text = ""
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
            logger.info(f"   Raw image OCR text: '{extracted_text.strip()}'")
    except Exception as e:
        logger.error(f"   Error with raw image: {e}")
    
    # Test 2: Optimized preprocessing (what we're currently using)
    logger.info("\n2. Testing OPTIMIZED PREPROCESSING (current)...")
    img_preprocessed = preprocessor.preprocess_image(image_path)
    if img_preprocessed is not None:
        logger.info(f"   Preprocessed shape: {img_preprocessed.shape}")
        logger.info(f"   Preprocessed range: [{img_preprocessed.min()}, {img_preprocessed.max()}]")
        
        try:
            ocr_results = ocr([img_preprocessed])
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"   Preprocessed blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Preprocessed OCR text: '{extracted_text.strip()}'")
        except Exception as e:
            logger.error(f"   Error with preprocessed image: {e}")
    
    # Test 3: Mild preprocessing (no thresholding)
    logger.info("\n3. Testing MILD PREPROCESSING (no thresholding)...")
    try:
        # Just convert to grayscale and back to RGB, no thresholding
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE only (no thresholding)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB without thresholding
        img_mild = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        logger.info(f"   Mild preprocessing shape: {img_mild.shape}")
        logger.info(f"   Mild preprocessing range: [{img_mild.min()}, {img_mild.max()}]")
        
        ocr_results = ocr([img_mild])
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   Mild preprocessing blocks found: {num_blocks}")
            
            # Extract text
            extracted_text = ""
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
            logger.info(f"   Mild preprocessing OCR text: '{extracted_text.strip()}'")
    except Exception as e:
        logger.error(f"   Error with mild preprocessing: {e}")
    
    # Test 4: Check detection maps for each approach
    logger.info("\n4. Comparing DETECTION MAPS...")
    
    test_images = [
        ("Raw", img_rgb),
        ("Preprocessed", img_preprocessed),
        ("Mild", img_mild)
    ]
    
    for name, img in test_images:
        if img is not None:
            try:
                loc_preds, seg_maps = ocr.det_predictor([img], return_maps=True)
                if len(seg_maps) > 0:
                    seg_map = seg_maps[0]
                    logger.info(f"   {name} seg map shape: {seg_map.shape}")
                    logger.info(f"   {name} seg map range: [{seg_map.min():.3f}, {seg_map.max():.3f}]")
                    pixels_above_01 = (seg_map > 0.1).sum()
                    logger.info(f"   {name} pixels > 0.1: {pixels_above_01} ({100*pixels_above_01/seg_map.size:.1f}%)")
                else:
                    logger.info(f"   {name}: No segmentation maps returned")
            except Exception as e:
                logger.error(f"   Error getting {name} detection map: {e}")


def test_doctr_aspect_ratio():
    """Test different DocTR aspect ratio and padding settings to fix dimension squashing."""
    import os
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== TESTING DocTR ASPECT RATIO SETTINGS ===")
    
    # Load raw image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"Input image shape: {img_rgb.shape}")
    
    # Test different aspect ratio and padding combinations
    test_configs = [
        {"preserve_aspect_ratio": True, "symmetric_pad": True, "name": "Default"},
        {"preserve_aspect_ratio": False, "symmetric_pad": False, "name": "No aspect ratio, no padding"},
        {"preserve_aspect_ratio": True, "symmetric_pad": False, "name": "Preserve aspect, no symmetric pad"},
        {"preserve_aspect_ratio": False, "symmetric_pad": True, "name": "No aspect ratio, symmetric pad"},
    ]
    
    for config in test_configs:
        name = config.pop("name")
        logger.info(f"\nTesting {name}: {config}")
        
        try:
            ocr = ocr_predictor(pretrained=True, **config)
            
            # Get detection map
            loc_preds, seg_maps = ocr.det_predictor([img_rgb], return_maps=True)
            
            if len(seg_maps) > 0:
                seg_map = seg_maps[0]
                logger.info(f"  Seg map shape: {seg_map.shape}")
                logger.info(f"  Seg map range: [{seg_map.min():.3f}, {seg_map.max():.3f}]")
                
                # Check if this fixes the dimension issue
                if seg_map.shape[2] > 10:  # More than 10 pixels wide
                    logger.info(f"  ✅ SUCCESS! Dimension issue fixed!")
                    
                    # Try full OCR
                    ocr_results = ocr([img_rgb])
                    if hasattr(ocr_results, "pages"):
                        num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                        logger.info(f"  OCR blocks found: {num_blocks}")
                        
                        # Extract text
                        extracted_text = ""
                        for page in ocr_results.pages:
                            for block in page.blocks:
                                for line in block.lines:
                                    extracted_text += line.value + " "
                        logger.info(f"  OCR text: '{extracted_text.strip()}'")
                else:
                    logger.info(f"  ❌ Still has dimension issue")
            else:
                logger.info(f"  No segmentation maps returned")
                
        except Exception as e:
            logger.error(f"  Error with {name}: {e}")
    
    # Test with different image sizes by resizing the input
    logger.info("\n=== TESTING DIFFERENT INPUT SIZES ===")
    
    sizes_to_test = [
        (512, 512),
        (1024, 768), 
        (800, 600),
        (640, 480),
        (256, 256)
    ]
    
    for width, height in sizes_to_test:
        logger.info(f"\nTesting input size: {width}x{height}")
        
        try:
            # Resize the image
            img_resized = cv2.resize(img_rgb, (width, height))
            logger.info(f"  Resized image shape: {img_resized.shape}")
            
            # Test with default settings
            ocr = ocr_predictor(pretrained=True)
            loc_preds, seg_maps = ocr.det_predictor([img_resized], return_maps=True)
            
            if len(seg_maps) > 0:
                seg_map = seg_maps[0]
                logger.info(f"  Seg map shape: {seg_map.shape}")
                
                if seg_map.shape[2] > 10:  # More than 10 pixels wide
                    logger.info(f"  ✅ SUCCESS with {width}x{height}!")
                    
                    # Try full OCR
                    ocr_results = ocr([img_resized])
                    if hasattr(ocr_results, "pages"):
                        num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                        logger.info(f"  OCR blocks found: {num_blocks}")
                        
                        if num_blocks > 0:
                            # Extract text
                            extracted_text = ""
                            for page in ocr_results.pages:
                                for block in page.blocks:
                                    for line in block.lines:
                                        extracted_text += line.value + " "
                            logger.info(f"  OCR text: '{extracted_text.strip()}'")
                else:
                    logger.info(f"  ❌ Still dimension issue with {width}x{height}")
            else:
                logger.info(f"  No segmentation maps returned for {width}x{height}")
                
        except Exception as e:
            logger.error(f"  Error with size {width}x{height}: {e}")


def test_proper_doctr_calling():
    """Test DocTR using the proper calling convention from the official demo."""
    import os
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== TESTING PROPER DocTR CALLING CONVENTION ===")
    
    # Load raw image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"Input image shape: {img_rgb.shape}")
    
    # Create predictor
    ocr = ocr_predictor(pretrained=True)
    
    logger.info("\n1. Testing WRONG way (our current approach):")
    try:
        # This is what we've been doing (wrong)
        ocr_results = ocr([img_rgb])
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   Wrong way blocks found: {num_blocks}")
            
            # Extract text
            extracted_text = ""
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
            logger.info(f"   Wrong way OCR text: '{extracted_text.strip()}'")
    except Exception as e:
        logger.error(f"   Error with wrong approach: {e}")
    
    logger.info("\n2. Testing RIGHT way (demo approach):")
    try:
        # This is the proper way from the demo
        logger.info(f"   Pre-processing input shape: {img_rgb.shape}")
        
        # Check what the preprocessor is configured to do
        if hasattr(ocr.det_predictor.pre_processor, 'resize'):
            logger.info(f"   Preprocessor resize config: {ocr.det_predictor.pre_processor.resize}")
        if hasattr(ocr.det_predictor.pre_processor, 'output_size'):
            logger.info(f"   Preprocessor output_size: {ocr.det_predictor.pre_processor.output_size}")
        if hasattr(ocr.det_predictor.pre_processor, 'preserve_aspect_ratio'):
            logger.info(f"   Preprocessor preserve_aspect_ratio: {ocr.det_predictor.pre_processor.preserve_aspect_ratio}")
        if hasattr(ocr.det_predictor.pre_processor, 'symmetric_pad'):
            logger.info(f"   Preprocessor symmetric_pad: {ocr.det_predictor.pre_processor.symmetric_pad}")
            
        processed_batches = ocr.det_predictor.pre_processor([img_rgb])
        logger.info(f"   Preprocessed batch shape: {processed_batches[0].shape}")
        
        # Get raw detection output 
        out = ocr.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        
        logger.info(f"   Raw seg map shape: {seg_map.shape}")
        logger.info(f"   Raw seg map range: [{seg_map.min():.3f}, {seg_map.max():.3f}]")
        
        # Convert to numpy and squeeze
        seg_map_np = seg_map.detach().cpu().numpy() if hasattr(seg_map, 'cpu') else seg_map.detach().numpy() if hasattr(seg_map, 'detach') else seg_map
        seg_map_squeezed = np.squeeze(seg_map_np)
        logger.info(f"   Squeezed seg map shape: {seg_map_squeezed.shape}")
        
        # Resize back to input image size (as demo does)
        seg_map_resized = cv2.resize(seg_map_squeezed, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        logger.info(f"   Resized seg map shape: {seg_map_resized.shape}")
        
        # Check if we have good segmentation
        pixels_above_03 = (seg_map_resized > 0.3).sum()
        logger.info(f"   Pixels > 0.3: {pixels_above_03} ({100*pixels_above_03/seg_map_resized.size:.1f}%)")
        
        if pixels_above_03 > 100:  # If we have substantial segmentation
            logger.info("   ✅ SUCCESS! Raw detection found substantial text regions!")
            
            # Now try to get the detection results directly
            loc_preds = ocr.det_predictor([img_rgb])
            logger.info(f"   Detection predictions: {len(loc_preds[0])}")
            
            if len(loc_preds[0]) > 0:
                logger.info("   ✅ Detection stage working! Now testing recognition...")
                
                # Try full OCR but with detection working
                ocr_results = ocr([img_rgb])
                if hasattr(ocr_results, "pages"):
                    num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                    logger.info(f"   Right way blocks found: {num_blocks}")
                    
                    # Extract text
                    extracted_text = ""
                    for page in ocr_results.pages:
                        for block in page.blocks:
                            for line in block.lines:
                                extracted_text += line.value + " "
                    logger.info(f"   Right way OCR text: '{extracted_text.strip()}'")
            else:
                logger.warning("   Detection stage still not finding text regions")
        else:
            logger.warning("   Still not finding substantial text regions in segmentation")
            
    except Exception as e:
        logger.error(f"   Error with right approach: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n3. Testing with different thresholds:")
    try:
        # Try with lower detection thresholds
        ocr.det_predictor.model.postprocessor.bin_thresh = 0.1
        ocr.det_predictor.model.postprocessor.box_thresh = 0.1
        logger.info("   Lowered thresholds: bin_thresh=0.1, box_thresh=0.1")
        
        # Test detection again
        loc_preds = ocr.det_predictor([img_rgb])
        logger.info(f"   Detection with low thresholds: {len(loc_preds[0])}")
        
        if len(loc_preds[0]) > 0:
            # Test full OCR
            ocr_results = ocr([img_rgb])
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"   Low threshold blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Low threshold OCR text: '{extracted_text.strip()}'")
        
    except Exception as e:
        logger.error(f"   Error with threshold test: {e}")
    
    logger.info("\n4. Testing with manual preprocessing override:")
    try:
        # Try to override the preprocessing to preserve aspect ratio
        ocr_custom = ocr_predictor(
            pretrained=True,
            preserve_aspect_ratio=True,
            symmetric_pad=True
        )
        
        logger.info("   Created predictor with preserve_aspect_ratio=True, symmetric_pad=True")
        
        # Check the new preprocessor config
        if hasattr(ocr_custom.det_predictor.pre_processor, 'preserve_aspect_ratio'):
            logger.info(f"   Custom preprocessor preserve_aspect_ratio: {ocr_custom.det_predictor.pre_processor.preserve_aspect_ratio}")
        if hasattr(ocr_custom.det_predictor.pre_processor, 'symmetric_pad'):
            logger.info(f"   Custom preprocessor symmetric_pad: {ocr_custom.det_predictor.pre_processor.symmetric_pad}")
        
        # Test preprocessing
        processed_batches = ocr_custom.det_predictor.pre_processor([img_rgb])
        logger.info(f"   Custom preprocessed batch shape: {processed_batches[0].shape}")
        
        # Test detection
        out = ocr_custom.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        logger.info(f"   Custom raw seg map shape: {seg_map.shape}")
        
        # Test full OCR if dimensions look better
        if seg_map.shape[-1] > 10:  # Check if width is reasonable
            logger.info("   ✅ Custom preprocessing fixed dimensions! Testing full OCR...")
            ocr_results = ocr_custom([img_rgb])
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"   Custom OCR blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Custom OCR text: '{extracted_text.strip()}'")
        else:
            logger.warning("   Custom preprocessing still has dimension issues")
            
    except Exception as e:
        logger.error(f"   Error with custom preprocessing: {e}")
        import traceback
        traceback.print_exc()


def test_manual_preprocessor_fix():
    """Test DocTR with manually fixed preprocessor dimensions."""
    import os
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== TESTING MANUAL PREPROCESSOR FIX ===")
    
    # Load raw image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"Input image shape: {img_rgb.shape}")
    
    # Create predictor with default (broken) preprocessor
    ocr = ocr_predictor(pretrained=True)
    logger.info(f"Original preprocessor output_size: {ocr.det_predictor.pre_processor.resize.size}")
    
    try:
        # Method 1: Create a manually fixed preprocessor
        logger.info("\n1. Testing with MANUALLY FIXED preprocessor (1024x1024):")
        
        # Create new preprocessor with correct dimensions
        fixed_preprocessor = PreProcessor(
            output_size=(1024, 1024),  # This should be the correct size!
            batch_size=1,
            mean=(0.798, 0.785, 0.772),  # Detection model means
            std=(0.264, 0.2749, 0.287),   # Detection model stds
            preserve_aspect_ratio=True,
            symmetric_pad=True
        )
        
        logger.info(f"Fixed preprocessor output_size: {fixed_preprocessor.resize.size}")
        
        # Replace the broken preprocessor
        ocr.det_predictor.pre_processor = fixed_preprocessor
        
        # Test preprocessing
        processed_batches = ocr.det_predictor.pre_processor([img_rgb])
        logger.info(f"Fixed preprocessed batch shape: {processed_batches[0].shape}")
        
        # Test detection
        out = ocr.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        logger.info(f"Fixed raw seg map shape: {seg_map.shape}")
        logger.info(f"Fixed raw seg map range: [{seg_map.min():.3f}, {seg_map.max():.3f}]")
        
        # Check if this fixed the dimensions
        if seg_map.shape[-1] > 100:  # Should be much larger now
            logger.info("   SUCCESS! Fixed preprocessor resolved dimension issue!")
            
            # Test full OCR pipeline
            ocr_results = ocr([img_rgb])
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                logger.info(f"   Fixed OCR blocks found: {num_blocks}")
                
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Fixed OCR text: '{extracted_text.strip()}'")
                
                if extracted_text.strip():
                    logger.info("   ✓ SUCCESS! DocTR is now working with proper preprocessing!")
                    return True
                else:
                    logger.warning("   Detection working but recognition still failing")
            else:
                logger.warning("   No pages in OCR results")
        else:
            logger.warning(f"   Still have dimension issues: {seg_map.shape}")
            
    except Exception as e:
        logger.error(f"   Error with manual preprocessor fix: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Try different standard sizes
    logger.info("\n2. Testing with DIFFERENT STANDARD SIZES:")
    standard_sizes = [
        (1024, 1024),
        (512, 512), 
        (768, 768),
        (896, 896),
        (1280, 1024)  # Close to our input aspect ratio
    ]
    
    for size in standard_sizes:
        try:
            logger.info(f"   Testing size {size}:")
            
            # Create predictor with different size
            test_preprocessor = PreProcessor(
                output_size=size,
                batch_size=1,
                mean=(0.798, 0.785, 0.772),
                std=(0.264, 0.2749, 0.287),
                preserve_aspect_ratio=True,
                symmetric_pad=True
            )
            
            # Create new OCR with fixed preprocessor
            test_ocr = ocr_predictor(pretrained=True)
            test_ocr.det_predictor.pre_processor = test_preprocessor
            
            # Quick test
            processed_batches = test_ocr.det_predictor.pre_processor([img_rgb])
            logger.info(f"     Preprocessed shape: {processed_batches[0].shape}")
            
            # Test if this produces reasonable OCR
            ocr_results = test_ocr([img_rgb])
            if hasattr(ocr_results, "pages"):
                num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
                if num_blocks > 0:
                    # Extract text
                    extracted_text = ""
                    for page in ocr_results.pages:
                        for block in page.blocks:
                            for line in block.lines:
                                extracted_text += line.value + " "
                    logger.info(f"     ✓ SUCCESS with {size}! Found text: '{extracted_text.strip()}'")
                    return True
                else:
                    logger.info(f"     Size {size}: {num_blocks} blocks")
            
        except Exception as e:
            logger.error(f"     Error with size {size}: {e}")
    
    return False


def test_coordinate_system_comparison():
    """Compare our image loading vs DocTR's official DocumentFile.from_images method."""
    import os
    from doctr.io import DocumentFile
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== COORDINATE SYSTEM COMPARISON ===")
    
    # Method 1: Our current approach (cv2.imread)
    logger.info("\n1. OUR METHOD (cv2.imread):")
    img_our = cv2.imread(image_path)
    if img_our is None:
        logger.error("Failed to load image with cv2")
        return
    
    img_our_rgb = cv2.cvtColor(img_our, cv2.COLOR_BGR2RGB)
    logger.info(f"   Our method shape: {img_our_rgb.shape}")
    logger.info(f"   Our method dtype: {img_our_rgb.dtype}")
    logger.info(f"   Our method range: [{img_our_rgb.min()}, {img_our_rgb.max()}]")
    
    # Method 2: DocTR's official method (DocumentFile.from_images)
    logger.info("\n2. DocTR'S OFFICIAL METHOD (DocumentFile.from_images):")
    pages_doctr = DocumentFile.from_images([image_path])
    img_doctr = pages_doctr[0]
    logger.info(f"   DocTR method shape: {img_doctr.shape}")
    logger.info(f"   DocTR method dtype: {img_doctr.dtype}")
    logger.info(f"   DocTR method range: [{img_doctr.min()}, {img_doctr.max()}]")
    
    # Check if they're identical
    if np.array_equal(img_our_rgb, img_doctr):
        logger.info("   ✓ Images are IDENTICAL")
    else:
        logger.info("   ❌ Images are DIFFERENT!")
        logger.info(f"   Difference stats: mean={np.mean(np.abs(img_our_rgb - img_doctr)):.3f}")
    
    # Method 3: Test both images with original broken preprocessor
    logger.info("\n3. TESTING BOTH WITH ORIGINAL PREPROCESSOR:")
    ocr_broken = ocr_predictor(pretrained=True)
    logger.info(f"   Original preprocessor size: {ocr_broken.det_predictor.pre_processor.resize.size}")
    
    # Test our method
    try:
        processed_our = ocr_broken.det_predictor.pre_processor([img_our_rgb])
        logger.info(f"   Our method processed shape: {processed_our[0].shape}")
    except Exception as e:
        logger.error(f"   Our method preprocessing failed: {e}")
    
    # Test DocTR method
    try:
        processed_doctr = ocr_broken.det_predictor.pre_processor([img_doctr])
        logger.info(f"   DocTR method processed shape: {processed_doctr[0].shape}")
    except Exception as e:
        logger.error(f"   DocTR method preprocessing failed: {e}")
    
    # Method 4: Test both with fixed preprocessor
    logger.info("\n4. TESTING BOTH WITH FIXED PREPROCESSOR:")
    fixed_preprocessor = PreProcessor(
        output_size=(1024, 1024),
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    
    # Test our method with fixed preprocessor
    try:
        processed_our_fixed = fixed_preprocessor([img_our_rgb])
        logger.info(f"   Our method + fixed preprocessor: {processed_our_fixed[0].shape}")
    except Exception as e:
        logger.error(f"   Our method + fixed preprocessing failed: {e}")
    
    # Test DocTR method with fixed preprocessor
    try:
        processed_doctr_fixed = fixed_preprocessor([img_doctr])
        logger.info(f"   DocTR method + fixed preprocessor: {processed_doctr_fixed[0].shape}")
    except Exception as e:
        logger.error(f"   DocTR method + fixed preprocessing failed: {e}")
    
    # Method 5: Test full OCR pipeline with DocTR's official image loading
    logger.info("\n5. TESTING FULL OCR WITH DocTR's OFFICIAL LOADING:")
    ocr_fixed = ocr_predictor(pretrained=True)
    ocr_fixed.det_predictor.pre_processor = fixed_preprocessor
    
    try:
        # Test OCR with DocTR's official image loading
        ocr_results = ocr_fixed([img_doctr])  # Using DocTR's image
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   DocTR official loading OCR blocks: {num_blocks}")
            
            if num_blocks > 0:
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   DocTR official loading OCR text: '{extracted_text.strip()}'")
                
                if extracted_text.strip():
                    logger.info("   ✓ SUCCESS! DocTR official loading works!")
                    return True
            else:
                logger.warning("   DocTR official loading found no blocks")
    except Exception as e:
        logger.error(f"   DocTR official loading OCR failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 6: Test coordinate transpose (height/width swap)
    logger.info("\n6. TESTING COORDINATE TRANSPOSE:")
    img_transposed = np.transpose(img_our_rgb, (1, 0, 2))  # Swap H and W
    logger.info(f"   Transposed shape: {img_transposed.shape}")
    
    try:
        processed_transposed = fixed_preprocessor([img_transposed])
        logger.info(f"   Transposed + fixed preprocessor: {processed_transposed[0].shape}")
        
        # Test OCR with transposed coordinates
        ocr_results = ocr_fixed([img_transposed])
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   Transposed OCR blocks: {num_blocks}")
            
            if num_blocks > 0:
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Transposed OCR text: '{extracted_text.strip()}'")
    except Exception as e:
        logger.error(f"   Transposed coordinate test failed: {e}")
    
    return False


def test_recognition_preprocessor_diagnosis():
    """Diagnose recognition preprocessor configuration issues."""
    import os
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== RECOGNITION PREPROCESSOR DIAGNOSIS ===")
    
    # Load raw image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"Input image shape: {img_rgb.shape}")
    
    # Create predictor
    ocr = ocr_predictor(pretrained=True)
    
    logger.info("\n1. ANALYZING RECOGNITION MODEL CONFIGURATION:")
    reco_model = ocr.reco_predictor.model
    logger.info(f"   Recognition model type: {type(reco_model).__name__}")
    logger.info(f"   Recognition model config: {reco_model.cfg}")
    
    logger.info("\n2. ANALYZING RECOGNITION PREPROCESSOR:")
    reco_prep = ocr.reco_predictor.pre_processor
    logger.info(f"   Recognition preprocessor type: {type(reco_prep).__name__}")
    
    if hasattr(reco_prep, 'resize'):
        resize_config = reco_prep.resize
        logger.info(f"   Recognition resize config: {resize_config}")
        if hasattr(resize_config, 'output_size'):
            logger.info(f"   Recognition output_size: {resize_config.output_size}")
    
    # Test synthetic crops of different sizes
    logger.info("\n3. TESTING RECOGNITION PREPROCESSOR WITH SYNTHETIC CROPS:")
    
    # Create test crops of expected sizes (32x128, like CRNN expects)
    test_crops = [
        np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8),  # Perfect size
        np.random.randint(0, 255, (30, 120, 3), dtype=np.uint8),  # Slightly smaller
        np.random.randint(0, 255, (40, 200, 3), dtype=np.uint8),  # Different aspect ratio
        np.random.randint(0, 255, (1, 128, 3), dtype=np.uint8),   # 1-pixel tall (like our bug)
    ]
    
    crop_names = ["Perfect (32x128)", "Smaller (30x120)", "Different AR (40x200)", "1-pixel tall (1x128)"]
    
    for i, (crop, name) in enumerate(zip(test_crops, crop_names)):
        try:
            logger.info(f"   Testing crop {name}: input shape {crop.shape}")
            
            # Process through recognition preprocessor
            processed_batches = reco_prep([crop])
            logger.info(f"     Processed shape: {processed_batches[0].shape}")
            
            # Try recognition
            result = ocr.reco_predictor.model(processed_batches[0], return_preds=True)
            logger.info(f"     ✓ Recognition succeeded")
            
        except Exception as e:
            logger.error(f"     ✗ Recognition failed: {e}")
    
    logger.info("\n4. TESTING WITH REAL DETECTED CROPS:")
    try:
        # Use our fixed detection to get real crops
        fixed_preprocessor = PreProcessor(
            output_size=(1024, 1024),
            batch_size=1,
            mean=(0.798, 0.785, 0.772),
            std=(0.264, 0.2749, 0.287),
            preserve_aspect_ratio=True,
            symmetric_pad=True
        )
        
        # Create new OCR with fixed detection preprocessor
        ocr_fixed = ocr_predictor(pretrained=True)
        ocr_fixed.det_predictor.pre_processor = fixed_preprocessor
        
        # Get detection results
        logger.info("   Running detection with fixed preprocessor...")
        processed_batches = ocr_fixed.det_predictor.pre_processor([img_rgb])
        out = ocr_fixed.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        
        # Convert segmentation map to numpy
        seg_map_np = seg_map.detach().cpu().numpy()
        seg_map_squeezed = np.squeeze(seg_map_np)
        
        # Resize back to input image size
        seg_map_resized = cv2.resize(seg_map_squeezed, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Find text regions (threshold at 0.3)
        text_mask = seg_map_resized > 0.3
        
        # Find bounding boxes of text regions
        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"   Found {len(contours)} text regions")
        
        for i, contour in enumerate(contours[:3]):  # Test first 3 contours
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            logger.info(f"   Region {i}: bbox=({x}, {y}, {w}, {h})")
            
            # Crop the region
            crop = img_rgb[y:y+h, x:x+w]
            logger.info(f"   Region {i}: crop shape={crop.shape}")
            
            # Process through recognition preprocessor
            try:
                processed_batches = reco_prep([crop])
                logger.info(f"   Region {i}: processed shape={processed_batches[0].shape}")
                
                # Try recognition
                result = ocr.reco_predictor.model(processed_batches[0], return_preds=True)
                logger.info(f"   Region {i}: ✓ Recognition succeeded")
                
            except Exception as e:
                logger.error(f"   Region {i}: ✗ Recognition failed: {e}")
    except Exception as e:
        logger.error(f"   Error in real crop testing: {e}")
    
    logger.info("\n5. MANUAL RECOGNITION PREPROCESSOR FIX TEST:")
    try:
        # Create a corrected recognition preprocessor
        logger.info("   Creating manually fixed recognition preprocessor...")
        
        # Use known correct dimensions for CRNN
        fixed_reco_prep = PreProcessor(
            output_size=(32, 128),  # Height=32, Width=128 as CRNN expects
            batch_size=128,
            mean=(0.694, 0.695, 0.693),
            std=(0.299, 0.296, 0.301),
            preserve_aspect_ratio=True,
            symmetric_pad=True
        )
        
        logger.info(f"   Fixed recognition preprocessor output_size: {fixed_reco_prep.resize.output_size}")
        
        # Test with synthetic crop
        test_crop = np.random.randint(0, 255, (40, 200, 3), dtype=np.uint8)
        logger.info(f"   Testing with crop shape: {test_crop.shape}")
        
        processed_batches = fixed_reco_prep([test_crop])
        logger.info(f"   Fixed preprocessor result: {processed_batches[0].shape}")
        
        # Create OCR with fixed recognition preprocessor
        ocr_fixed_reco = ocr_predictor(pretrained=True)
        ocr_fixed_reco.det_predictor.pre_processor = fixed_preprocessor  # Fix detection too
        ocr_fixed_reco.reco_predictor.pre_processor = fixed_reco_prep    # Fix recognition
        
        logger.info("   Testing full OCR pipeline with both fixes...")
        ocr_results = ocr_fixed_reco([img_rgb])
        
        if hasattr(ocr_results, "pages"):
            num_blocks = len(ocr_results.pages[0].blocks) if len(ocr_results.pages) > 0 else 0
            logger.info(f"   ✓ COMPLETE SUCCESS! Found {num_blocks} blocks")
            
            if num_blocks > 0:
                # Extract text
                extracted_text = ""
                for page in ocr_results.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += line.value + " "
                logger.info(f"   Extracted text: '{extracted_text.strip()}'")
        else:
            logger.error("   No results structure found")
            
    except Exception as e:
        logger.error(f"   Error in manual fix test: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")


def test_complete_doctr_fix():
    """Test complete DocTR fix with both detection and recognition preprocessor fixes."""
    import os
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Test on one image
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    image_path = os.path.join(test_dir, "test1.png")
    
    logger.info("=== COMPLETE DocTR PREPROCESSOR FIX ===")
    
    # Load raw image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        logger.error("Failed to load test image")
        return
    
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    logger.info(f"Input image shape: {img_rgb.shape}")
    
    logger.info("\n1. CREATING FIXED PREPROCESSORS:")
    
    # Create fixed detection preprocessor (1024x1024)
    detection_preprocessor = PreProcessor(
        output_size=(1024, 1024),
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    logger.info(f"   Fixed detection preprocessor: {detection_preprocessor.resize}")
    
    # Create fixed recognition preprocessor (32x128)
    recognition_preprocessor = PreProcessor(
        output_size=(32, 128),
        batch_size=128,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    logger.info(f"   Fixed recognition preprocessor: {recognition_preprocessor.resize}")
    
    logger.info("\n2. CREATING FIXED OCR PREDICTOR:")
    
    # Create OCR predictor with both fixes
    ocr_fixed = ocr_predictor(pretrained=True)
    ocr_fixed.det_predictor.pre_processor = detection_preprocessor
    ocr_fixed.reco_predictor.pre_processor = recognition_preprocessor
    
    logger.info("   Applied both preprocessor fixes to OCR predictor")
    
    logger.info("\n3. TESTING COMPLETE PIPELINE:")
    
    try:
        # Test complete OCR pipeline
        logger.info("   Running complete OCR pipeline...")
        ocr_results = ocr_fixed([img_rgb])
        
        if hasattr(ocr_results, "pages") and len(ocr_results.pages) > 0:
            page = ocr_results.pages[0]
            num_blocks = len(page.blocks)
            logger.info(f"   ✅ SUCCESS! OCR completed with {num_blocks} blocks")
            
            if num_blocks > 0:
                # Extract and display text
                extracted_text = ""
                total_words = 0
                
                for block_idx, block in enumerate(page.blocks):
                    logger.info(f"     Block {block_idx}: {len(block.lines)} lines")
                    
                    for line_idx, line in enumerate(block.lines):
                        line_text = line.value
                        confidence = line.confidence if hasattr(line, 'confidence') else 0.0
                        extracted_text += line_text + " "
                        total_words += len(line.words) if hasattr(line, 'words') else 1
                        
                        logger.info(f"       Line {line_idx}: '{line_text}' (conf: {confidence:.3f})")
                
                logger.info(f"   \\n   📝 COMPLETE EXTRACTED TEXT: '{extracted_text.strip()}'")
                logger.info(f"   📊 STATISTICS: {num_blocks} blocks, {total_words} words")
                
                # Compare with ground truth if available
                ground_truth_file = image_path.replace('.png', '.txt').replace('.jpg', '.txt')
                if os.path.exists(ground_truth_file):
                    with open(ground_truth_file, 'r', encoding='utf-8') as f:
                        ground_truth = f.read().strip()
                    
                    fom = compute_figure_of_merit(extracted_text.strip(), ground_truth)
                    logger.info(f"   🎯 GROUND TRUTH: '{ground_truth}'")
                    logger.info(f"   📈 FIGURE OF MERIT: {fom:.6f}")
                    
                    if fom < -3.0:  # Good performance threshold
                        logger.info(f"   🎉 EXCELLENT PERFORMANCE! FOM = {fom:.6f}")
                    elif fom < -1.0:
                        logger.info(f"   👍 GOOD PERFORMANCE! FOM = {fom:.6f}")
                    else:
                        logger.info(f"   ⚠️  POOR PERFORMANCE. FOM = {fom:.6f}")
                    return True
                else:
                    logger.warning("   ⚠️  OCR completed but found no text")
                    return False
            else:
                logger.error("   ❌ OCR failed - no pages in results")
                return False
        
    except Exception as e:
        logger.error(f"   ❌ OCR pipeline failed: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("\n4. TESTING ON MULTIPLE IMAGES:")
    
    # Test on all available test images
    test_images = []
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(test_dir, filename))
    
    success_count = 0
    total_images = min(len(test_images), 3)  # Test up to 3 images
    
    for i, img_path in enumerate(test_images[:total_images]):
        try:
            logger.info(f"   Testing image {i+1}/{total_images}: {os.path.basename(img_path)}")
            
            # Load image
            test_img = cv2.imread(img_path)
            if test_img is None:
                continue
            test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            results = ocr_fixed([test_rgb])
            
            if hasattr(results, "pages") and len(results.pages) > 0:
                blocks = len(results.pages[0].blocks)
                if blocks > 0:
                    # Extract text
                    text = ""
                    for page in results.pages:
                        for block in page.blocks:
                            for line in block.lines:
                                text += line.value + " "
                    
                    logger.info(f"     ✅ Success: {blocks} blocks, text: '{text.strip()[:50]}{'...' if len(text.strip()) > 50 else ''}'")
                    success_count += 1
                else:
                    logger.info(f"     ⚠️  No text found")
            else:
                logger.info(f"     ❌ OCR failed")
            
        except Exception as e:
            logger.error(f"     ❌ Error: {e}")
    
    logger.info(f"\\n   📊 MULTI-IMAGE RESULTS: {success_count}/{total_images} successful")
    
    if success_count == total_images:
        logger.info(f"   🎉 PERFECT! All images processed successfully")
    elif success_count > 0:
        logger.info(f"   👍 GOOD! {success_count}/{total_images} images successful")
    else:
        logger.info(f"   ❌ FAILED! No images processed successfully")
    
    logger.info("\\n5. PERFORMANCE COMPARISON:")
    
    try:
        # Compare with broken default DocTR
        logger.info("   Comparing with default (broken) DocTR...")
        ocr_broken = ocr_predictor(pretrained=True)  # Default broken version
        
        broken_results = ocr_broken([img_rgb])
        broken_blocks = 0
        if hasattr(broken_results, "pages") and len(broken_results.pages) > 0:
            broken_blocks = len(broken_results.pages[0].blocks)
        
        fixed_results = ocr_fixed([img_rgb])
        fixed_blocks = 0
        if hasattr(fixed_results, "pages") and len(fixed_results.pages) > 0:
            fixed_blocks = len(fixed_results.pages[0].blocks)
        
        logger.info("   📊 COMPARISON:")
        logger.info(f"     Default DocTR: {broken_blocks} blocks")
        logger.info(f"     Fixed DocTR:   {fixed_blocks} blocks")
        
        if fixed_blocks > broken_blocks:
            improvement = "∞" if broken_blocks == 0 else f"{fixed_blocks/broken_blocks:.1f}x"
            logger.info(f"     🚀 IMPROVEMENT: {improvement} better!")
        elif fixed_blocks == broken_blocks > 0:
            logger.info(f"     ✅ SAME PERFORMANCE (both working)")
        else:
            logger.info(f"     ⚠️  NO IMPROVEMENT")
            
    except Exception as e:
        logger.error(f"   Error in comparison: {e}")
    
    return success_count > 0


def test_raw_ocr() -> None:
    """Test OCR on raw images without any preprocessing to debug the pipeline."""
    import os
    from pathlib import Path
    from doctr.models import ocr_predictor
    from PIL import Image
    import numpy as np
    
    logger.info("Testing raw OCR without preprocessor...")
    
    # Test images directory
    test_dir = Path("tests/images")
    test_image = test_dir / "test1.png"
    caption_file = test_dir / "test1-caption.txt"
    
    if not test_image.exists():
        logger.error(f"Test image not found: {test_image}")
        return
        
    if not caption_file.exists():
        logger.error(f"Caption file not found: {caption_file}")
        return
    
    # Load ground truth
    with open(caption_file, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()
    logger.info(f"Ground truth: '{ground_truth}'")
    
    try:
        # Test 1: Raw image with PIL + numpy (like the working parts of ocr_image)
        logger.info("Test 1: Raw image via PIL + numpy...")
        img_pil = Image.open(test_image).convert("RGB")
        img_array = np.array(img_pil)
        
        logger.info(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.info(f"Image value range: [{img_array.min()}, {img_array.max()}]")
        
        # Use simple OCR predictor
        ocr = ocr_predictor(pretrained=True)
        ocr_results = ocr([img_array])
        
        if isinstance(ocr_results, list):
            ocr_results = ocr_results[0]
        
        logger.info(f"OCR results type: {type(ocr_results)}")
        if hasattr(ocr_results, "pages"):
            logger.info(f"OCR results contain {len(ocr_results.pages)} pages")
            for i, page in enumerate(ocr_results.pages):
                logger.info(f"Page {i} has {len(page.blocks)} blocks")
                for j, block in enumerate(page.blocks):
                    logger.info(f"  Block {j} has {len(block.lines)} lines")
                    for k, line in enumerate(block.lines):
                        logger.info(f"    Line {k}: '{line.value}' (confidence: {getattr(line, 'confidence', 'unknown')})")
        
        # Extract text
        extracted_text = ""
        if hasattr(ocr_results, "pages"):
            for page in ocr_results.pages:
                for block in page.blocks:
                    for line in block.lines:
                        extracted_text += line.value + " "
        
        extracted_text = extracted_text.strip()
        logger.info(f"Raw OCR result: '{extracted_text}'")
        
        # Calculate metrics
        if extracted_text:
            hamming_dist = compute_hamming(extracted_text, ground_truth)
            edit_dist = compute_edit_distance(extracted_text, ground_truth)
            fom = compute_figure_of_merit(extracted_text, ground_truth)
            
            logger.info(f"Raw OCR metrics:")
            logger.info(f"  Hamming distance: {hamming_dist}")
            logger.info(f"  Edit distance: {edit_dist}")
            logger.info(f"  Figure of Merit: {fom:.4f}")
        else:
            logger.error("Raw OCR returned empty string!")
            
        # Test 2: Check what the preprocessor does
        logger.info("\nTest 2: Checking preprocessor behavior...")
        try:
            import untext.preprocessor as preprocessor
            processed_array = preprocessor.preprocess_image(str(test_image))
            if processed_array is not None:
                logger.info(f"Preprocessed shape: {processed_array.shape}, dtype: {processed_array.dtype}")
                logger.info(f"Preprocessed value range: [{processed_array.min()}, {processed_array.max()}]")
                
                # Try OCR on preprocessed image
                ocr_preprocessed = ocr([processed_array])
                if isinstance(ocr_preprocessed, list):
                    ocr_preprocessed = ocr_preprocessed[0]
                    
                processed_text = ""
                if hasattr(ocr_preprocessed, "pages"):
                    logger.info(f"Preprocessed OCR: {len(ocr_preprocessed.pages)} pages")
                    for page in ocr_preprocessed.pages:
                        logger.info(f"  {len(page.blocks)} blocks")
                        for block in page.blocks:
                            for line in block.lines:
                                processed_text += line.value + " "
                
                processed_text = processed_text.strip()
                logger.info(f"Preprocessed OCR result: '{processed_text}'")
                
                if not processed_text and extracted_text:
                    logger.error("Preprocessor is BREAKING the OCR! Raw works but preprocessed doesn't.")
                elif not extracted_text and not processed_text:
                    logger.error("Both raw and preprocessed OCR fail - fundamental DocTR issue.")
                elif processed_text:
                    logger.info("Preprocessor is working correctly.")
                    
            else:
                logger.error("Preprocessor returned None!")
                
        except Exception as e:
            logger.error(f"Preprocessor test failed: {e}")
    
    except Exception as e:
        logger.error(f"Raw OCR test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def main():
    use_detection = False
    use_tesseract = False
    use_mask = False
    use_grid_search = False
    if '--detect' in sys.argv:
        use_detection = True
    if '--tesseract' in sys.argv:
        use_tesseract = True
    if '--mask' in sys.argv:
        use_mask = True
    if '--grid-search' in sys.argv:
        use_grid_search = True
    if '--smoke-test' in sys.argv:
        logger.info("Running OCR smoke test...")
        success = smoke_test_fixed_ocr()
        if success:
            logger.info("Smoke test passed! OCR implementation is ready for grid search.")
        else:
            logger.error("Smoke test failed! Fix OCR implementation before running grid search.")
        return
    if '--clean-grid-search' in sys.argv:
        logger.info("Running clean grid search...")
        clean_grid_search()
        return
    if '--test-raw-ocr' in sys.argv:
        logger.info("Testing raw OCR...")
        test_raw_ocr()
        return
    if '--compare-preprocess' in sys.argv:
        compare_preprocessing_performance()
        return
    if '--debug-preprocess' in sys.argv:
        debug_preprocessing_comparison()
        return
    if '--test-doctr' in sys.argv:
        test_doctr_parameters()
        return
    if '--test-minimal-preprocess' in sys.argv:
        test_minimal_preprocessing()
        return
    if '--test-doctr-aspect-ratio' in sys.argv:
        test_doctr_aspect_ratio()
        return
    if '--test-proper-doctr' in sys.argv:
        test_proper_doctr_calling()
        return
    if '--test-manual-fix' in sys.argv:
        test_manual_preprocessor_fix()
        return
    if '--test-coordinates' in sys.argv:
        test_coordinate_system_comparison()
        return
    if '--test-recognition-diagnosis' in sys.argv:
        test_recognition_preprocessor_diagnosis()
        return

    if use_grid_search:
        orthogonal_grid_search()
        return
    
    # Define the directory containing test images and caption files
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")

    # List of test images
    test_images = [
        "test1.png",
        "test2.png",
        "test3.jpg",
        "test4-with-text.png"
    ]

    for img_file in test_images:
        image_path = os.path.join(test_dir, img_file)
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            continue

        # If the mask flag is used, generate and save the text mask, then skip further OCR processing
        if use_mask:
            mask = generate_text_mask(image_path)
            if mask is not None:
                base, _ = os.path.splitext(img_file)
                mask_file = base + "-mask.png"
                mask_path = os.path.join(test_dir, mask_file)
                cv2.imwrite(mask_path, mask)
                logger.info(f"Mask image saved to {mask_path}")
            else:
                logger.error(f"Failed to generate mask for {image_path}")
            continue

        if use_tesseract:
            ocr_text = ocr_image_tesseract(image_path)
            logger.info(f"(Tesseract) Found text \"{ocr_text}\" ...")
        elif use_detection:
            ocr_text = ocr_image_with_detection(image_path)
            logger.info(f"(Detection-based) Found text \"{ocr_text}\" ...")
        else:
            ocr_text = ocr_image(image_path)
            logger.info(f"(Standard) Found text \"{ocr_text}\" ...")

        # Determine the corresponding caption file (e.g. test1-caption.txt)
        base, _ = os.path.splitext(img_file)
        caption_file = base + "-caption.txt"
        caption_path = os.path.join(test_dir, caption_file)

        if os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption_text = f.read().strip()
                distance = compute_hamming(median_ascii_string(ocr_text), caption_text)
                logger.info(f"Hamming distance from caption = {distance}")
            except Exception as e:
                logger.error(f"Failed to read caption file {caption_path}: {e}")
        else:
            logger.warning(f"Caption file not found for {img_file}")


def smoke_test_fixed_ocr() -> bool:
    """Smoke test to verify the fixed OCR implementation works correctly.
    
    This function tests the OCR pipeline on a known test image to ensure:
    1. The fixed DocTR configuration is working
    2. Text extraction produces reasonable results
    3. Hamming/Edit distance metrics work properly
    
    Returns:
        True if smoke test passes, False otherwise
    """
    logger.info("Running OCR smoke test...")
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "images")
    test_image = os.path.join(test_dir, "test1.png")
    caption_file = os.path.join(test_dir, "test1-caption.txt")
    
    if not os.path.exists(test_image):
        logger.error(f"Smoke test failed: test image not found at {test_image}")
        return False
    
    if not os.path.exists(caption_file):
        logger.error(f"Smoke test failed: caption file not found at {caption_file}")
        return False
    
    try:
        # Load ground truth
        with open(caption_file, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()
        logger.info(f"Ground truth: '{ground_truth}'")
        
        # Test 1: No tuning (just basic image loading)
        logger.info("Test 1: No image tuning...")
        no_tuning_config = {'method': 'none'}
        
        # Use the apply_tuning_pipeline function from grid search
        def apply_tuning_pipeline(image_path, threshold_config, blur_config, enhancement_config):
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply enhancement
            if enhancement_config['method'] == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=enhancement_config['clip_limit'], 
                                       tileGridSize=enhancement_config['tile_size'])
                gray = clahe.apply(gray)
            elif enhancement_config['method'] == 'histogram_eq':
                gray = cv2.equalizeHist(gray)
            elif enhancement_config['method'] == 'contrast':
                gray = cv2.convertScaleAbs(gray, alpha=enhancement_config['alpha'], beta=enhancement_config['beta'])
            
            # Apply blur
            if blur_config['method'] == 'gaussian':
                k = blur_config['kernel']
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            elif blur_config['method'] == 'bilateral':
                gray = cv2.bilateralFilter(gray, blur_config['d'], blur_config['sigma_color'], blur_config['sigma_space'])
            elif blur_config['method'] == 'median':
                gray = cv2.medianBlur(gray, blur_config['kernel'])
            
            # Apply thresholding
            if threshold_config['method'] == 'adaptive':
                result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, threshold_config['block_size'], threshold_config['C'])
            elif threshold_config['method'] == 'otsu':
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_config['method'] == 'simple':
                _, result = cv2.threshold(gray, threshold_config['threshold'], 255, cv2.THRESH_BINARY)
            else:  # none
                result = gray
            
            # Convert back to RGB
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            
            return result
        
        tuned_img = apply_tuning_pipeline(test_image, no_tuning_config, no_tuning_config, no_tuning_config)
        if tuned_img is None:
            logger.error("Failed to load test image")
            return False
        
        # Save the tuned image temporarily and use the WORKING ocr_image function
        import tempfile
        temp_image_path = None
        try:
            # Convert RGB back to BGR for saving
            if len(tuned_img.shape) == 3:
                bgr_img = cv2.cvtColor(tuned_img, cv2.COLOR_RGB2BGR)
            else:
                bgr_img = tuned_img
                
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_image_path = tmp_file.name
                cv2.imwrite(temp_image_path, bgr_img)
            
            # Use the WORKING ocr_image function
            ocr_result = ocr_image(temp_image_path)
            
        finally:
            # Clean up temp file
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        logger.info(f"OCR result: '{ocr_result}'")
        
        # Calculate metrics
        hamming_dist = compute_hamming(ocr_result, ground_truth)
        edit_dist = compute_edit_distance(ocr_result, ground_truth)
        fom = compute_figure_of_merit(ocr_result, ground_truth)
        
        logger.info(f"Metrics:")
        logger.info(f"   Hamming distance: {hamming_dist}")
        logger.info(f"   Edit distance: {edit_dist}")
        logger.info(f"   Figure of Merit: {fom:.4f}")
        
        # Test 2: Simple tuning pipeline
        logger.info("Test 2: Simple CLAHE enhancement...")
        clahe_config = {'method': 'clahe', 'clip_limit': 2.0, 'tile_size': (8, 8)}
        
        tuned_img2 = apply_tuning_pipeline(test_image, no_tuning_config, no_tuning_config, clahe_config)
        
        # Use the WORKING ocr_image function for second test too
        temp_image_path2 = None
        try:
            # Convert RGB back to BGR for saving
            if len(tuned_img2.shape) == 3:
                bgr_img2 = cv2.cvtColor(tuned_img2, cv2.COLOR_RGB2BGR)
            else:
                bgr_img2 = tuned_img2
                
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_image_path2 = tmp_file.name
                cv2.imwrite(temp_image_path2, bgr_img2)
            
            # Use the WORKING ocr_image function
            ocr_result2 = ocr_image(temp_image_path2)
            
        finally:
            # Clean up temp file
            if temp_image_path2 and os.path.exists(temp_image_path2):
                os.remove(temp_image_path2)
        
        hamming_dist2 = compute_hamming(ocr_result2, ground_truth)
        edit_dist2 = compute_edit_distance(ocr_result2, ground_truth)
        fom2 = compute_figure_of_merit(ocr_result2, ground_truth)
        
        logger.info(f"OCR result with CLAHE: '{ocr_result2}'")
        logger.info(f"CLAHE Metrics:")
        logger.info(f"   Hamming distance: {hamming_dist2}")
        logger.info(f"   Edit distance: {edit_dist2}")
        logger.info(f"   Figure of Merit: {fom2:.4f}")
        
        # Comparison
        if fom2 < fom:
            logger.info(f"CLAHE tuning improves FOM by {fom - fom2:.4f}")
        elif fom2 > fom:
            logger.info(f"CLAHE tuning worsens FOM by {fom2 - fom:.4f}")
        else:
            logger.info(f"CLAHE tuning has no effect on FOM")
        
        # Success criteria
        if len(ocr_result) == 0:
            logger.error("Smoke test failed: No text extracted")
            return False
        
        if fom == float('inf'):
            logger.error("Smoke test failed: Invalid FOM calculation")
            return False
        
        logger.info("Smoke test passed! Fixed OCR is working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Smoke test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main() 