#!/usr/bin/env python3
"""
Working OCR implementation using the documented DocTR fixes.
"""

import cv2
import numpy as np
from pathlib import Path

def create_fixed_ocr_predictor():
    """Create OCR predictor with fixed preprocessors as documented."""
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Create OCR predictor (will have broken preprocessors)
    ocr = ocr_predictor(pretrained=True)
    
    # Create FIXED detection preprocessor
    detection_preprocessor = PreProcessor(
        output_size=(1024, 1024),  # FIX: was (3, 1024)
        batch_size=1,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    
    # Create FIXED recognition preprocessor  
    recognition_preprocessor = PreProcessor(
        output_size=(32, 128),  # FIX: was (3, 32)
        batch_size=128,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301),
        preserve_aspect_ratio=True,
        symmetric_pad=True
    )
    
    # Replace broken preprocessors with fixed ones
    ocr.det_predictor.pre_processor = detection_preprocessor
    ocr.reco_predictor.pre_processor = recognition_preprocessor
    
    return ocr

def extract_text_with_fixed_ocr(image_path):
    """Extract text using the fixed OCR predictor."""
    # Load and convert image (BGR -> RGB as documented)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create fixed OCR predictor
    ocr = create_fixed_ocr_predictor()
    
    # Run OCR with fixed preprocessors
    results = ocr([img_rgb])
    
    # Extract text following the documented hierarchy
    extracted_text = ""
    for page in results.pages:
        print(f"Page has {len(page.blocks)} blocks")
        for block in page.blocks:
            print(f"  Block has {len(block.lines)} lines")
            for line in block.lines:
                print(f"    Line has {len(line.words)} words")
                for word in line.words:
                    text = word.value
                    confidence = word.confidence
                    print(f"      Word: '{text}' (conf: {confidence:.3f})")
                    extracted_text += text + " "
    
    return extracted_text.strip()

def test_fixed_ocr():
    """Test the fixed OCR on test1.png."""
    print("Testing fixed OCR implementation...")
    
    # Load ground truth
    caption_path = "tests/images/test1-caption.txt"
    with open(caption_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()
    print(f"Ground truth: '{ground_truth}'")
    
    # Test fixed OCR
    try:
        img_path = "tests/images/test1.png"
        result = extract_text_with_fixed_ocr(img_path)
        print(f"Fixed OCR result: '{result}'")
        
        if result:
            print("SUCCESS! Fixed OCR is working!")
            
            # Calculate metrics
            from untext.ocr_sandbox import compute_hamming, compute_edit_distance, compute_figure_of_merit
            
            hamming_dist = compute_hamming(result, ground_truth)
            edit_dist = compute_edit_distance(result, ground_truth)
            fom = compute_figure_of_merit(result, ground_truth)
            
            print(f"Metrics:")
            print(f"  Hamming distance: {hamming_dist}")
            print(f"  Edit distance: {edit_dist}")
            print(f"  Figure of Merit: {fom:.4f}")
            
            return True
        else:
            print("FAILED: Fixed OCR returned empty string")
            return False
            
    except Exception as e:
        print(f"FAILED: Fixed OCR crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_ocr() 