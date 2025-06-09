#!/usr/bin/env python3
"""
Character-Level Position Detection Demo

This script demonstrates the character-level position detection and masking
capabilities of the untext library using real test images.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

from untext.word_mask_generator import WordMaskGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_test_image_path():
    """Get path to a test image with text."""
    test_dir = Path(__file__).parent.parent / "tests" / "images"
    
    # Try different test images in order of preference
    candidates = ["test1.png", "test2.png", "test3.jpg", "test4-with-text.png"]
    
    for candidate in candidates:
        test_path = test_dir / candidate
        if test_path.exists():
            return test_path
    
    raise FileNotFoundError(f"No test images found in {test_dir}")


def load_test_image():
    """Load a test image with ground truth text."""
    test_image_path = get_test_image_path()
    
    # Load the image
    image = cv2.imread(str(test_image_path))
    if image is None:
        raise ValueError(f"Failed to load test image from {test_image_path}")
    
    # Try to load ground truth caption
    caption_path = test_image_path.with_name(f"{test_image_path.stem}-caption.txt")
    ground_truth = ""
    if caption_path.exists():
        ground_truth = caption_path.read_text().strip()
    
    logger.info(f"Loaded test image: {test_image_path.name}")
    if ground_truth:
        logger.info(f"Ground truth text: '{ground_truth}'")
    
    return image, ground_truth, test_image_path.name


def demo_character_positioning():
    """Demonstrate character-level position detection."""
    print("=== Character-Level Position Detection Demo ===\n")
    
    # Load test image
    image, ground_truth, image_name = load_test_image()
    
    print(f"Using test image: {image_name}")
    print(f"Image size: {image.shape[1]}×{image.shape[0]} pixels")
    if ground_truth:
        print(f"Expected text: '{ground_truth}'")
    print()
    
    # Initialize mask generator in letters mode for precise character detection
    mask_gen = WordMaskGenerator(mode="letters", preprocess=True)
    
    print("1. Box Mode (Fast Rectangular Masks):")
    print("   - Uses detection only")
    print("   - Creates rectangular bounding boxes")
    print("   - Faster processing")
    
    # Generate box mask
    mask_gen.set_mode("box")
    box_mask = mask_gen.generate_mask_from_array(image)
    
    print(f"   - Box mask shape: {box_mask.shape}")
    print(f"   - Text pixels detected: {np.sum(box_mask > 0)}")
    
    print("\n2. Letters Mode (Precise Character-Level Masks):")
    print("   - Uses full OCR pipeline with fixed preprocessors")
    print("   - Creates precise polygon masks")
    print("   - Character-level accuracy")
    
    # Generate letter mask
    mask_gen.set_mode("letters")
    letter_mask = mask_gen.generate_mask_from_array(image)
    
    print(f"   - Letter mask shape: {letter_mask.shape}")
    print(f"   - Text pixels detected: {np.sum(letter_mask > 0)}")
    
    print("\n3. Character Position Information:")
    
    # Get detailed character positions
    positions = mask_gen.get_character_positions(image)
    
    for i, pos in enumerate(positions):
        geometry = pos['geometry']
        confidence = pos['confidence']
        
        # Calculate bounding box from polygon
        x_coords = geometry[:, 0]
        y_coords = geometry[:, 1]
        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
        width, height = x2 - x1, y2 - y1
        
        print(f"   Detection {i+1}:")
        print(f"     - Position: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"     - Size: {width}×{height} pixels")
        print(f"     - Confidence: {confidence:.3f}")
        print(f"     - Polygon points: {len(geometry)} points")
    
    print("\n4. Mask Comparison:")
    box_coverage = np.sum(box_mask > 0)
    letter_coverage = np.sum(letter_mask > 0)
    difference = abs(letter_coverage - box_coverage)
    
    print(f"   - Box mode coverage: {box_coverage} pixels")
    print(f"   - Letters mode coverage: {letter_coverage} pixels")
    print(f"   - Difference: {difference} pixels")
    
    if letter_coverage < box_coverage:
        print("   → Letters mode is more precise (less over-masking)")
    elif letter_coverage > box_coverage:
        print("   → Letters mode captures more detail")
    else:
        print("   → Similar coverage between modes")
    
    # Save demo results
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save original image
    cv2.imwrite(str(output_dir / f"original_{image_name}"), image)
    
    # Save box mask
    cv2.imwrite(str(output_dir / f"box_mask_{image_name}"), box_mask)
    
    # Save letter mask
    cv2.imwrite(str(output_dir / f"letter_mask_{image_name}"), letter_mask)
    
    # Create visualization with detection overlays
    vis_image = image.copy()
    for i, pos in enumerate(positions):
        geometry = pos['geometry'].astype(np.int32)
        
        # Draw polygon outline
        cv2.polylines(vis_image, [geometry], True, (0, 255, 0), 2)
        
        # Draw detection number
        x, y = geometry[0]
        cv2.putText(vis_image, str(i+1), (x-20, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(str(output_dir / f"detections_overlay_{image_name}"), vis_image)
    
    print(f"\n5. Output Files:")
    print(f"   - Original image: {output_dir}/original_{image_name}")
    print(f"   - Box mask: {output_dir}/box_mask_{image_name}") 
    print(f"   - Letter mask: {output_dir}/letter_mask_{image_name}")
    print(f"   - Detection overlay: {output_dir}/detections_overlay_{image_name}")
    
    print("\n=== Demo Complete ===")
    print("\nTo use character-level masking in your workflow:")
    print("  python -m untext.cli -i image.jpg -o output.jpg --mask letters")


if __name__ == "__main__":
    demo_character_positioning() 