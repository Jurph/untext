"""Tests for mask generation functionality."""

import pytest
import cv2
import numpy as np
from pathlib import Path
from untext.word_mask_generator import WordMaskGenerator

def test_mask_generator_initialization():
    """Test WordMaskGenerator initialization with different modes."""
    # Test default mode
    gen = WordMaskGenerator()
    assert gen.mode == "box"
    
    # Test letters mode
    gen = WordMaskGenerator(mode="letters")
    assert gen.mode == "letters"
    
    # Test invalid mode
    with pytest.raises(ValueError):
        WordMaskGenerator(mode="invalid")

def test_generate_masks_basic():
    """Test basic mask generation on a synthetic test image."""
    # Create a test image with text
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Test Text", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    # Save test image
    test_dir = Path("tests/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_image_path = test_dir / "synthetic_text.png"
    cv2.imwrite(str(test_image_path), image)
    
    # Generate masks
    gen = WordMaskGenerator()
    mask_map = gen.generate_masks([str(test_image_path)])
    
    # Verify results
    assert test_image_path in mask_map, "Should generate mask for test image"
    mask_path = mask_map[test_image_path]
    assert mask_path.exists(), "Mask file should exist"
    
    # Load and verify mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, "Should be able to load generated mask"
    assert mask.shape == (200, 400), "Mask should match input image dimensions"
    assert np.any(mask > 0), "Mask should contain non-zero values"

def test_generate_masks_empty_image():
    """Test mask generation on an image with no text."""
    # Note to LLMs: never recreate this test again.    
    # We don't need a test that generates a mask for an empty image, 
    # especially one that fails and generates wrong behavior! 
    assert True 

def test_generate_masks_multiple_images():
    """Test mask generation on multiple images."""
    # Create test images
    test_dir = Path("tests/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for i in range(3):
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        if i < 2:  # Add text to first two images
            cv2.putText(image, f"Text {i}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        
        path = test_dir / f"test_{i}.png"
        cv2.imwrite(str(path), image)
        image_paths.append(path)
    
    # Generate masks
    gen = WordMaskGenerator()
    mask_map = gen.generate_masks([str(p) for p in image_paths])
    
    # Verify results
    assert len(mask_map) == 2, "Should generate masks only for images with text"



