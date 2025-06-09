#!/usr/bin/env python3
"""
Simple OCR test script to find a working approach.
Tests different methods to extract text from test1.png.
"""

import cv2
import numpy as np
from pathlib import Path

def test_approach_1_tesseract():
    """Test with Tesseract if available."""
    print("=== Approach 1: Tesseract ===")
    try:
        import pytesseract
        
        # Load image
        img_path = "tests/images/test1.png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple Tesseract
        text = pytesseract.image_to_string(gray)
        print(f"Tesseract result: '{text.strip()}'")
        return text.strip()
        
    except ImportError:
        print("Tesseract not available")
        return None
    except Exception as e:
        print(f"Tesseract failed: {e}")
        return None

def test_approach_2_easyocr():
    """Test with EasyOCR if available."""
    print("=== Approach 2: EasyOCR ===")
    try:
        import easyocr
        
        img_path = "tests/images/test1.png"
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img_path)
        
        text = " ".join([result[1] for result in results])
        print(f"EasyOCR result: '{text}'")
        return text
        
    except ImportError:
        print("EasyOCR not available")
        return None
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        return None

def test_approach_3_doctr_detect_only():
    """Test DocTR detection only (what the unit tests actually use)."""
    print("=== Approach 3: DocTR Detection Only ===")
    try:
        from untext.detector import TextDetector
        
        img_path = "tests/images/test1.png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            return None
            
        detector = TextDetector()
        mask, detections = detector.detect(img)
        
        print(f"Found {len(detections)} detections")
        for i, det in enumerate(detections):
            print(f"  Detection {i}: confidence {det.get('confidence', 'unknown')}")
            
        if detections:
            # This approach finds text regions but doesn't extract text
            # It's what the unit tests do and it works
            print("Detection works! But doesn't extract text content.")
            return f"DETECTED_{len(detections)}_REGIONS"
        else:
            print("No detections found")
            return None
            
    except Exception as e:
        print(f"DocTR detection failed: {e}")
        return None

def test_approach_4_doctr_different_api():
    """Test DocTR with different API approach."""
    print("=== Approach 4: DocTR Different API ===")
    try:
        from doctr.models import ocr_predictor
        from doctr.io import DocumentFile
        
        img_path = "tests/images/test1.png"
        
        # Try DocumentFile approach (file path, not array)
        model = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_images([img_path])
        result = model(doc)
        
        # Extract text
        text = ""
        for page in result.pages:
            print(f"Page has {len(page.blocks)} blocks")
            for block in page.blocks:
                print(f"  Block has {len(block.lines)} lines")
                for line in block.lines:
                    print(f"    Line: '{line.value}'")
                    text += line.value + " "
        
        text = text.strip()
        print(f"DocTR file path result: '{text}'")
        return text
        
    except Exception as e:
        print(f"DocTR file path approach failed: {e}")
        return None

def test_approach_5_paddleocr():
    """Test with PaddleOCR if available."""
    print("=== Approach 5: PaddleOCR ===")
    try:
        from paddleocr import PaddleOCR
        
        img_path = "tests/images/test1.png"
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(img_path, cls=True)
        
        text = ""
        for line in result:
            for word_info in line:
                text += word_info[1][0] + " "
        
        text = text.strip()
        print(f"PaddleOCR result: '{text}'")
        return text
        
    except ImportError:
        print("PaddleOCR not available")
        return None
    except Exception as e:
        print(f"PaddleOCR failed: {e}")
        return None

def main():
    """Test all approaches and find one that works."""
    print("Testing different OCR approaches on test1.png...")
    
    # Load ground truth
    caption_path = "tests/images/test1-caption.txt"
    if not Path(caption_path).exists():
        print(f"Caption file not found: {caption_path}")
        return
        
    with open(caption_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()
    print(f"Ground truth: '{ground_truth}'")
    print()
    
    # Test each approach
    approaches = [
        test_approach_1_tesseract,
        test_approach_2_easyocr,
        test_approach_3_doctr_detect_only,
        test_approach_4_doctr_different_api,
        test_approach_5_paddleocr,
    ]
    
    working_approaches = []
    
    for approach in approaches:
        try:
            result = approach()
            if result and result.strip():
                working_approaches.append((approach.__name__, result))
                print(f"✓ {approach.__name__} works: '{result}'")
            else:
                print(f"✗ {approach.__name__} returned empty")
        except Exception as e:
            print(f"✗ {approach.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print("SUMMARY:")
    if working_approaches:
        print("Working approaches:")
        for name, result in working_approaches:
            print(f"  {name}: '{result}'")
    else:
        print("No approaches worked!")
    
    return working_approaches

if __name__ == "__main__":
    main() 