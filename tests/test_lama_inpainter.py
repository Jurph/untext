"""Tests for the LaMa inpainting functionality."""

import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple, Optional
from untext.lama_inpainter import LamaInpainter

def create_test_image(size: Tuple[int, int] = (100, 100), text: str = "Test") -> Tuple[np.ndarray, np.ndarray]:
    """Create a test image with text.
    
    Args:
        size: Image size (height, width)
        text: Text to write on image
        
    Returns:
        Tuple of (image array, mask array)
    """
    # Create white background
    image = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 0)  # Black text
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position to center text
    x = (size[1] - text_width) // 2
    y = (size[0] + text_height) // 2
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    # Create mask (same size as image)
    mask = np.zeros(size, dtype=np.uint8)
    
    # Fill text region in mask
    cv2.putText(mask, text, (x, y), font, font_scale, 255, thickness)
    
    return image, mask

@pytest.fixture
def test_image_dir(tmp_path: Path) -> Tuple[Path, list[Path]]:
    """Create a temporary directory with test images."""
    # Create test images
    image1, mask1 = create_test_image((200, 200), "Test 1")
    image2, mask2 = create_test_image((200, 200), "Test 2")
    image3, mask3 = create_test_image((200, 200), "Test 3")
    
    # Save images
    image_paths = []
    for i, (img, msk) in enumerate([(image1, mask1), (image2, mask2), (image3, mask3)]):
        img_path = tmp_path / f"test{i+1}.jpg"
        msk_path = tmp_path / f"test{i+1}_mask.png"
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(msk_path), msk)
        image_paths.append(img_path)
    
    return tmp_path, image_paths

def test_inpainter_initialization() -> None:
    """Test LamaInpainter initialization."""
    try:
        inpainter = LamaInpainter()
        assert inpainter is not None
    except RuntimeError as e:
        if "LaMa dependencies not installed" in str(e):
            pytest.skip("LaMa not installed - skipping test")
        else:
            raise

def test_inpaint_single_image() -> None:
    """Test inpainting a single image."""
    try:
        # Create test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Initialize inpainter
        inpainter = LamaInpainter()
        
        # Inpaint image
        result = inpainter.inpaint(image, mask)
        
        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        
        # Check that masked region changed
        assert not np.array_equal(result[mask > 0], image[mask > 0])
        
    except RuntimeError as e:
        if "LaMa dependencies not installed" in str(e):
            pytest.skip("LaMa not installed - skipping test")
        else:
            raise

def test_inpaint_with_subregion() -> None:
    """Test inpainting with a subregion."""
    try:
        # Create test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Initialize inpainter
        inpainter = LamaInpainter()
        
        # Define subregion
        subregion = (50, 50, 150, 150)
        
        # Inpaint image
        result = inpainter.inpaint(image, mask, subregion=subregion)
        
        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        
        # Check that masked region in subregion changed
        subregion_mask = np.zeros_like(mask)
        subregion_mask[subregion[1]:subregion[3], subregion[0]:subregion[2]] = 1
        masked_subregion = (mask > 0) & (subregion_mask > 0)
        assert not np.array_equal(result[masked_subregion], image[masked_subregion])
        
    except RuntimeError as e:
        if "LaMa dependencies not installed" in str(e):
            pytest.skip("LaMa not installed - skipping test")
        else:
            raise

def test_inpaint_with_invalid_input() -> None:
    """Test inpainting with invalid inputs."""
    try:
        inpainter = LamaInpainter()
        
        # Create valid test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Test with None
        with pytest.raises(ValueError):
            inpainter.inpaint(None, mask)
        with pytest.raises(ValueError):
            inpainter.inpaint(image, None)
        
        # Test with wrong dimensions
        with pytest.raises(ValueError):
            inpainter.inpaint(np.ones((100, 100), dtype=np.uint8), mask)  # 2D image
        with pytest.raises(ValueError):
            inpainter.inpaint(image, np.ones((100, 100, 3), dtype=np.uint8))  # 3D mask
        
        # Test with wrong number of channels
        with pytest.raises(ValueError):
            inpainter.inpaint(np.ones((100, 100, 4), dtype=np.uint8), mask)  # 4 channels
        
        # Test with mismatched sizes
        with pytest.raises(ValueError):
            inpainter.inpaint(image, np.zeros((100, 100), dtype=np.uint8))  # Different size
        
    except RuntimeError as e:
        if "LaMa dependencies not installed" in str(e):
            pytest.skip("LaMa not installed - skipping test")
        else:
            raise

def test_inpaint_with_invalid_subregion() -> None:
    """Test inpainting with invalid subregion."""
    try:
        # Create test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Initialize inpainter
        inpainter = LamaInpainter()
        
        # Test with invalid subregion
        with pytest.raises(ValueError):
            inpainter.inpaint(image, mask, subregion=(0, 0, 0, 0))  # Zero size
        with pytest.raises(ValueError):
            inpainter.inpaint(image, mask, subregion=(200, 200, 0, 0))  # Negative size
        with pytest.raises(ValueError):
            inpainter.inpaint(image, mask, subregion=(-1, -1, 100, 100))  # Negative coordinates
        with pytest.raises(ValueError):
            inpainter.inpaint(image, mask, subregion=(0, 0, 300, 300))  # Out of bounds
        
    except RuntimeError as e:
        if "LaMa dependencies not installed" in str(e):
            pytest.skip("LaMa not installed - skipping test")
        else:
            raise 