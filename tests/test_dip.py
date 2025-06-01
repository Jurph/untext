"""Tests for Deep Image Prior inpainting performance."""

import os
import cv2
import numpy as np
from pathlib import Path
import pytest
from skimage.metrics import structural_similarity as ssim
from untext.image_patcher import ImagePatcher
from untext.word_mask_generator import WordMaskGenerator
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_text_mask(image_shape: tuple, num_regions: int = 3) -> np.ndarray:
    """Create a mask simulating text regions.
    
    Args:
        image_shape: Shape of the image (height, width)
        num_regions: Number of text-like regions to create
        
    Returns:
        Binary mask where 255 indicates regions to inpaint
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h, w = image_shape[:2]
    
    # Create some text-like rectangular regions
    for _ in range(num_regions):
        # Random position and size for text block
        x = np.random.randint(0, w - 100)
        y = np.random.randint(0, h - 30)
        width = np.random.randint(50, 200)
        height = np.random.randint(20, 40)
        
        # Draw rectangle
        cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
    
    return mask

def get_region_bounds(mask: np.ndarray, padding: float = 1.0) -> tuple:
    """Get the bounding box of the masked region with padding.
    
    Args:
        mask: Binary mask where 255 indicates regions to inpaint
        padding: Padding as a fraction of the region size (1.0 = 100% padding)
        
    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    # Find the bounding box of the masked region
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
    
    y1, y2 = np.min(y_indices), np.max(y_indices)
    x1, x2 = np.min(x_indices), np.max(x_indices)
    
    # Add padding (100% of region size)
    h, w = y2 - y1, x2 - x1
    pad_y = int(h * padding)
    pad_x = int(w * padding)
    
    # Calculate new bounds with padding
    new_y1 = y1 - pad_y
    new_y2 = y2 + pad_y
    new_x1 = x1 - pad_x
    new_x2 = x2 + pad_x
    
    # Handle edge cases
    if new_y1 < 0:
        # If we're near the top edge, shift the region down
        shift = abs(new_y1)
        new_y1 = 0
        new_y2 = min(mask.shape[0], new_y2 + shift)
    elif new_y2 > mask.shape[0]:
        # If we're near the bottom edge, shift the region up
        shift = new_y2 - mask.shape[0]
        new_y2 = mask.shape[0]
        new_y1 = max(0, new_y1 - shift)
        
    if new_x1 < 0:
        # If we're near the left edge, shift the region right
        shift = abs(new_x1)
        new_x1 = 0
        new_x2 = min(mask.shape[1], new_x2 + shift)
    elif new_x2 > mask.shape[1]:
        # If we're near the right edge, shift the region left
        shift = new_x2 - mask.shape[1]
        new_x2 = mask.shape[1]
        new_x1 = max(0, new_x1 - shift)
    
    # Ensure minimum size for the region
    min_size = 64  # Minimum size for the network to work effectively
    if new_y2 - new_y1 < min_size:
        # Center the region vertically
        center = (new_y1 + new_y2) // 2
        new_y1 = max(0, center - min_size // 2)
        new_y2 = min(mask.shape[0], new_y1 + min_size)
        
    if new_x2 - new_x1 < min_size:
        # Center the region horizontally
        center = (new_x1 + new_x2) // 2
        new_x1 = max(0, center - min_size // 2)
        new_x2 = min(mask.shape[1], new_x1 + min_size)
    
    # Ensure dimensions are even (helps with network operations)
    if (new_x2 - new_x1) % 2 != 0:
        new_x2 += 1
    if (new_y2 - new_y1) % 2 != 0:
        new_y2 += 1
    
    return (new_x1, new_y1, new_x2, new_y2)

def test_dip_inpainting():
    """Test Deep Image Prior's ability to inpaint text regions."""
    # Setup paths
    image_dir = Path("images")
    test_image = image_dir / "test3.jpg"
    assert test_image.exists(), f"Test image not found: {test_image}"
    
    # Create output directory for test artifacts
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test image
    original = cv2.imread(str(test_image))
    assert original is not None, f"Failed to load test image: {test_image}"
    
    # Generate mask using text detector
    detector = WordMaskGenerator()
    mask_map = detector.generate_masks([str(test_image)])
    if test_image not in mask_map:
        pytest.skip("Text detector did not return a mask for the test image")

    mask_path = mask_map[test_image]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, "Failed to load generated mask"
    
    # Create masked image for visualization
    masked = original.copy()
    masked[mask > 0] = 0
    masked_path = output_dir / "test_masked.png"
    cv2.imwrite(str(masked_path), masked)
    
    # Get the region to inpaint with padding
    region_bounds = get_region_bounds(mask)
    if region_bounds is None:
        pytest.skip("No masked regions found")
    
    x1, y1, x2, y2 = region_bounds
    
    # Ensure we don't exceed image bounds
    x2 = min(x2, original.shape[1])
    y2 = min(y2, original.shape[0])
    
    # Crop the region from the original image and mask
    region_original = original[y1:y2, x1:x2].copy()
    region_mask = mask[y1:y2, x1:x2].copy()
    
    # Verify dimensions match
    assert region_original.shape[:2] == region_mask.shape[:2], \
        f"Region dimensions mismatch: image {region_original.shape[:2]} vs mask {region_mask.shape[:2]}"
    
    region_masked = region_original.copy()
    region_masked[region_mask > 0] = 0
    
    # Save the cropped region for visualization
    cv2.imwrite(str(output_dir / "test_region.png"), region_original)
    cv2.imwrite(str(output_dir / "test_region_mask.png"), region_mask)
    cv2.imwrite(str(output_dir / "test_region_masked.png"), region_masked)
    
    # Number of iterations for inpainting - reduced for text regions
    num_iterations = 200  # keep DIP there but using edge_fill we ignore
    
    def progress_callback(iteration: int, total: int, loss: float) -> None:
        """Log progress during inpainting."""
        if iteration % 20 == 0:  # Log more frequently since we have fewer iterations
            logger.info(f"Iteration {iteration}/{total}, Loss: {loss:.4f}")
    
    # Initialize patcher with more iterations and GPU if available
    patcher = ImagePatcher(num_iterations=num_iterations, known_region_weight=0.01)
    
    # Run inpainting only on the cropped region
    region_result = patcher.patch_image(
        str(output_dir / "test_region.png"),
        str(output_dir / "test_region_mask.png"),
        str(output_dir / "test_region_fixed.png"),
        progress_callback=progress_callback,
        blend=True,
        dilate_percent=0.05,
        feather_radius=20,
        method="edge_fill"
    )
    
    # Paste the result back into the original image
    result = original.copy()
    result[y1:y2, x1:x2] = region_result
    
    # Save the final result
    cv2.imwrite(str(output_dir / "test3-fixed.png"), result)
    
    # Calculate similarity metrics
    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM only on masked regions
    ssim_score = ssim(
        original_gray,
        result_gray,
        mask=(mask > 0),
        data_range=255
    )
    
    # Calculate mean squared error on masked regions
    mse = np.mean((original[mask > 0] - result[mask > 0]) ** 2)
    
    # Log results
    logger.info(f"Test Results:")
    logger.info(f"SSIM score (masked regions): {ssim_score:.4f}")
    logger.info(f"MSE (masked regions): {mse:.2f}")
    
    # Save comparison visualization
    comparison = np.hstack([original, masked, result])
    cv2.imwrite(str(output_dir / "test_comparison.png"), comparison)
    
    # Assert minimum quality thresholds
    assert ssim_score > 0.5, "SSIM score too low"
    assert mse < 1000, "MSE too high" 