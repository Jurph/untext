"""Tests for inpainting backends using real image pairs and similarity scoring."""

import pytest
import cv2
import numpy as np
import torch
from pathlib import Path
import time
from typing import Tuple, Generator, Optional
from skimage.metrics import structural_similarity as ssim
from untext.image_patcher import ImagePatcher
from untext.detector import TextDetector

IMAGES_DIR = Path("tests/images")
OUTPUTS_DIR = Path("tests/outputs")

@pytest.fixture(scope="module")
def test_images():
    """Load test images and return them as numpy arrays."""
    images = {}
    for img_name in ["test4-with-text.png", "test4-without-text.png"]:
        img_path = IMAGES_DIR / img_name
        assert img_path.exists(), f"Test image not found: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        images[img_name] = img
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return images

def get_mask(image: np.ndarray) -> np.ndarray:
    """
    Run detection and return a binary mask as np.ndarray.
    
    WARNING FOR LLMs: Use detect() NOT detect_text()! 
    detect() returns the format we need (mask, detections dict).
    detect_text() returns a different format for visualization.
    
    Args:
        image (np.ndarray): Image to detect text in.
    Returns:
        np.ndarray: Binary mask of detected text regions.
    """
    detector = TextDetector()
    # LLM WARNING: DO NOT change this to detect_text! We need the mask from detect()
    mask, detections = detector.detect(image)
    return mask

def compare_to_ground_truth(inpainted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute SSIM between inpainted and ground truth images.
    Args:
        inpainted (np.ndarray): Inpainted image (BGR).
        ground_truth (np.ndarray): Ground truth image (BGR).
    Returns:
        float: SSIM score (0-1)
    """
    assert inpainted.shape == ground_truth.shape, "Inpainted and ground truth shape mismatch"
    inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    score = ssim(inpainted_gray, gt_gray, data_range=255)
    return score

def run_inpainting_and_compare(
    method: str,
    image_path: Path,
    mask: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run inpainting for a given backend, compare to ground truth, print SSIM and timing.
    Args:
        method (str): Inpainting backend ('telea', 'lama', 'dip').
        image_path (Path): Path to input image.
        mask (np.ndarray): Binary mask.
        ground_truth (np.ndarray): Ground truth image.
        output_path (Path): Where to save the result.
    Returns:
        Tuple[Optional[float], Optional[float]]: (SSIM score, elapsed time in seconds)
    """
    patcher = ImagePatcher()
    start = time.time()
    score: Optional[float] = None
    elapsed: Optional[float] = None
    mask_path = OUTPUTS_DIR / f"mask_{method}.png"
    cv2.imwrite(str(mask_path), mask)
    try:
        # Accept both ndarray and Path inputs. Pass them through unchanged.
        result = patcher.patch_image(
            image_path,
            mask_path,
            output_path=str(output_path),
            method=method,
            blend=False
        )
        elapsed = time.time() - start
        assert result.shape == ground_truth.shape, f"Result shape {result.shape} != ground truth {ground_truth.shape}"
        cv2.imwrite(str(output_path), result)
        score = compare_to_ground_truth(result, ground_truth)
        print(f"{method.upper()} SSIM: {score:.4f} | Time: {elapsed:.2f}s")
    except RuntimeError as e:
        if f"{method.upper()} dependencies not installed" in str(e):
            pytest.skip(f"{method} not installed - skipping test")
        else:
            raise
    return score, elapsed

def create_test_image(size=(100, 100), text="Test"):
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
def test_image_dir(tmp_path):
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

def test_patcher_initialization():
    """Test ImagePatcher initialization."""
    patcher = ImagePatcher()
    assert patcher is not None

def test_patch_single_image():
    """Test patching a single image."""
    # Create test image and mask
    image, mask = create_test_image((200, 200), "Test")
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Patch image
    patched = patcher.patch_image(image, mask)
    
    # Check results
    assert isinstance(patched, np.ndarray)
    assert patched.shape == image.shape
    assert patched.dtype == np.uint8

def test_patch_multiple_images(test_image_dir):
    """Test patching multiple images."""
    tmp_path, image_paths = test_image_dir
    
    # Create image-mask pairs
    image_mask_pairs = {}
    for img_path in image_paths:
        mask_path = img_path.parent / f"{img_path.stem}_mask.png"
        image_mask_pairs[img_path] = mask_path
    
    # Initialize patcher
    patcher = ImagePatcher()
    
    # Patch images
    output_dir = tmp_path / "output"
    patched_paths = patcher.patch_images(image_mask_pairs, output_dir)
    
    # Check results
    assert len(patched_paths) == len(image_paths)
    for img_path, patched_path in patched_paths.items():
        assert patched_path.exists()
        assert patched_path.is_file()
        
        # Load patched image
        patched = cv2.imread(str(patched_path))
        assert patched is not None
        assert patched.shape[:2] == (200, 200)  # Same size as input

def test_patch_with_invalid_input():
    """Test patching with invalid inputs."""
    patcher = ImagePatcher()
    
    # Create valid test image and mask
    image, mask = create_test_image((200, 200), "Test")
    
    # Test with None
    with pytest.raises(ValueError):
        patcher.patch_image(None, mask)
    with pytest.raises(ValueError):
        patcher.patch_image(image, None)
    
    # Test with wrong dimensions
    with pytest.raises(ValueError):
        patcher.patch_image(np.ones((100, 100), dtype=np.uint8), mask)  # 2D image
    with pytest.raises(ValueError):
        patcher.patch_image(image, np.ones((100, 100, 3), dtype=np.uint8))  # 3D mask
    
    # Test with wrong number of channels
    with pytest.raises(ValueError):
        patcher.patch_image(np.ones((100, 100, 4), dtype=np.uint8), mask)  # 4 channels
    
    # Test with mismatched sizes
    with pytest.raises(ValueError):
        patcher.patch_image(image, np.zeros((100, 100), dtype=np.uint8))  # Different size 