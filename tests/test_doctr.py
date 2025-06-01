"""Test script for image masking and patching functionality."""

import cv2
import numpy as np
from untext.mask_image import process_images

def test_image_processing():
    """Test the image masking and patching functionality."""
    # Create a test image with text
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.putText(image, "TEST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite("images/test_image.jpg", image)

    # Process test images
    image_paths = ["images/test1.png", "images/test2.png", "images/test3.jpg"]
    results = process_images(image_paths, device='cpu')
    
    # Verify results
    assert len(results['masks']) > 0, "No masks were generated"
    assert len(results['patched']) > 0, "No images were patched"
    
    # Print results for inspection
    for image_path in image_paths:
        print(f"\nProcessing results for {image_path}:")
        print(f"  Mask: {results['masks'].get(image_path, 'Not generated')}")
        print(f"  Patched: {results['patched'].get(image_path, 'Not generated')}")

if __name__ == "__main__":
    test_image_processing() 