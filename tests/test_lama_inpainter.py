"""Tests for the LaMa inpainting functionality.

Covers:
    - ``TeleaInpainter`` – basic API, subregion, validation
    - ``select_device()`` – CPU, CUDA fallback paths
    - ``LamaInpainter`` – constructor guards, input validation,
      subregion edge-padding, SimpleLama output cropping, error handling
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

from untextre.telea_inpainter import TeleaInpainter
import untextre.lama_inpainter as lama_mod
from untextre.lama_inpainter import select_device, LamaInpainter

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
        inpainter = TeleaInpainter()
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
        inpainter = TeleaInpainter()
        
        # Inpaint image
        result = inpainter.inpaint(image, mask)
        
        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        
        # Check that masked region changed
        assert not np.array_equal(result[mask > 0], image[mask > 0])
        
    except RuntimeError as e:
        if "dependencies not installed" in str(e):
            pytest.skip("Telea not installed - skipping test")
        else:
            raise

def test_inpaint_with_subregion() -> None:
    """Test inpainting with a subregion."""
    try:
        # Create test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Initialize inpainter
        inpainter = TeleaInpainter()
        
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
        if "dependencies not installed" in str(e):
            pytest.skip("Telea not installed - skipping test")
        else:
            raise

def test_inpaint_with_invalid_input() -> None:
    """Test inpainting with invalid inputs."""
    try:

        inpainter = TeleaInpainter()
        
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
        if "dependencies not installed" in str(e):
            pytest.skip("Telea not installed - skipping test")
        else:
            raise

def test_inpaint_with_invalid_subregion() -> None:
    """Test inpainting with invalid subregion."""
    try:
        # Create test image and mask
        image, mask = create_test_image((200, 200), "Test")
        
        # Initialize inpainter
        inpainter = TeleaInpainter()
        
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
        if "dependencies not installed" in str(e):
            pytest.skip("Telea not installed - skipping test")
        else:
            raise


# =========================================================================
# select_device
# =========================================================================

class TestSelectDevice:
    """Test device selection and fallback logic."""

    def test_cpu_returns_cpu(self):
        assert select_device("cpu") == "cpu"

    def test_cuda_available_returns_cuda(self, monkeypatch):
        """When CUDA is available, returns 'cuda'."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "FakeGPU"
        monkeypatch.setitem(__builtins__ if isinstance(__builtins__, dict) else vars(__builtins__), "__import__", lambda *a, **kw: mock_torch)
        # Simpler: just monkeypatch the torch module reference
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "FakeGPU")
        result = select_device("cuda")
        assert result == "cuda"

    def test_cuda_not_available_falls_back(self, monkeypatch):
        """When CUDA is not available, falls back to CPU."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = select_device("cuda")
        assert result == "cpu"

    def test_cuda_exception_falls_back(self, monkeypatch):
        """When CUDA init raises, falls back to CPU."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(RuntimeError("GPU broken")))
        result = select_device("cuda")
        assert result == "cpu"


# =========================================================================
# LamaInpainter — constructor guards
# =========================================================================

class TestLamaInpainterConstructor:
    """Test LamaInpainter __init__ error paths."""

    def test_no_torch_raises(self, monkeypatch):
        """torch=None → RuntimeError with install hint."""
        monkeypatch.setattr(lama_mod, "torch", None)
        monkeypatch.setattr(lama_mod, "_IMPORT_ERROR", ImportError("no torch"))
        with pytest.raises(RuntimeError, match="PyTorch is required"):
            LamaInpainter()

    def test_no_backends_raises(self, monkeypatch):
        """Neither SimpleLama nor load_checkpoint → RuntimeError."""
        monkeypatch.setattr(lama_mod, "SimpleLama", None)
        monkeypatch.setattr(lama_mod, "load_checkpoint", None)
        with pytest.raises(RuntimeError, match="Neither simple-lama-inpainting"):
            LamaInpainter()


# =========================================================================
# LamaInpainter — inpaint validation and processing
# =========================================================================

class _FakeSimpleLama:
    """Mock SimpleLama that returns a PIL-like image from numpy input."""

    def __call__(self, img_rgb, mask):
        """Return a fake PIL Image (numpy array with a .convert method)."""
        out = img_rgb.copy()

        class FakePIL:
            """Mimics PIL Image just enough for the isinstance/convert check."""
            def __init__(self, arr):
                self._arr = arr
                self.shape = arr.shape
                self.dtype = arr.dtype

            def convert(self, mode):
                return self

            def __array__(self):
                return self._arr

        return FakePIL(out)


@pytest.fixture
def lama_inpainter(monkeypatch):
    """Construct a LamaInpainter with a mocked SimpleLama backend."""
    fake = _FakeSimpleLama()
    monkeypatch.setattr(lama_mod, "SimpleLama", _FakeSimpleLama)
    monkeypatch.setattr(lama_mod, "select_device", lambda d: "cpu")

    inpainter = LamaInpainter(device="cpu")
    inpainter.model = fake
    return inpainter


class TestLamaInpainterValidation:
    """Test input validation in LamaInpainter.inpaint()."""

    def test_none_image_raises(self, lama_inpainter):
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="image must be a numpy ndarray"):
            lama_inpainter.inpaint(None, mask)

    def test_none_mask_raises(self, lama_inpainter):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="mask must be a numpy ndarray"):
            lama_inpainter.inpaint(image, None)

    def test_wrong_image_shape_raises(self, lama_inpainter):
        image = np.zeros((50, 50), dtype=np.uint8)  # 2D
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxWx3"):
            lama_inpainter.inpaint(image, mask)

    def test_4channel_image_raises(self, lama_inpainter):
        image = np.zeros((50, 50, 4), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxWx3"):
            lama_inpainter.inpaint(image, mask)

    def test_3d_mask_auto_converts(self, lama_inpainter):
        """3-channel mask should be auto-converted to single channel."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask_3d = np.zeros((50, 50, 3), dtype=np.uint8)
        mask_3d[20:30, 20:30] = 255
        # Should not raise
        result = lama_inpainter.inpaint(image, mask_3d)
        assert result.shape == image.shape

    def test_mismatched_sizes_raises(self, lama_inpainter):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical height/width"):
            lama_inpainter.inpaint(image, mask)

    def test_subregion_zero_width_raises(self, lama_inpainter):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive width/height"):
            lama_inpainter.inpaint(image, mask, subregion=(10, 10, 10, 20))

    def test_subregion_out_of_bounds_raises(self, lama_inpainter):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="out of image bounds"):
            lama_inpainter.inpaint(image, mask, subregion=(0, 0, 100, 100))


class TestLamaInpainterProcessing:
    """Test inpaint processing paths: subregion, edge padding, output cropping."""

    def test_basic_inpaint_returns_correct_shape(self, lama_inpainter):
        """Full-image inpaint returns same shape as input."""
        image = np.ones((64, 64, 3), dtype=np.uint8) * 200
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        result = lama_inpainter.inpaint(image, mask)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_subregion_inpaint_returns_full_size(self, lama_inpainter):
        """Subregion inpaint should return the full image size."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 30:50] = 255
        result = lama_inpainter.inpaint(image, mask, subregion=(20, 20, 60, 60))
        assert result.shape == image.shape

    def test_subregion_touching_top_left_edge_pads(self, lama_inpainter):
        """Subregion at (0,0) should trigger edge padding."""
        image = np.ones((80, 80, 3), dtype=np.uint8) * 150
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        result = lama_inpainter.inpaint(image, mask, subregion=(0, 0, 20, 20))
        assert result.shape == image.shape

    def test_subregion_touching_bottom_right_edge_pads(self, lama_inpainter):
        """Subregion touching bottom-right boundary should trigger edge padding."""
        image = np.ones((80, 80, 3), dtype=np.uint8) * 150
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[70:80, 70:80] = 255
        result = lama_inpainter.inpaint(image, mask, subregion=(60, 60, 80, 80))
        assert result.shape == image.shape

    def test_subregion_touching_all_edges(self, lama_inpainter):
        """Subregion spanning the entire image should pad all sides and restore."""
        image = np.ones((64, 64, 3), dtype=np.uint8) * 100
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        result = lama_inpainter.inpaint(image, mask, subregion=(0, 0, 64, 64))
        assert result.shape == image.shape

    def test_simplelama_output_cropped_when_padded(self, monkeypatch):
        """If SimpleLama output is larger than input (mod-8 padding), it gets cropped."""
        # Create a model that returns output 8px larger in each dimension
        class PaddingSimpleLama:
            def __call__(self, img_rgb, mask):
                h, w = img_rgb.shape[:2]
                # Simulate mod-8 padding: add extra pixels
                padded = np.zeros((h + 5, w + 3, 3), dtype=np.uint8)
                padded[:h, :w] = img_rgb

                class FakePIL:
                    def __init__(self, arr):
                        self._arr = arr
                        self.shape = arr.shape
                        self.dtype = arr.dtype
                    def convert(self, mode):
                        return self
                    def __array__(self):
                        return self._arr

                return FakePIL(padded)

        monkeypatch.setattr(lama_mod, "SimpleLama", PaddingSimpleLama)
        monkeypatch.setattr(lama_mod, "select_device", lambda d: "cpu")

        inpainter = LamaInpainter(device="cpu")
        inpainter.model = PaddingSimpleLama()

        image = np.ones((50, 70, 3), dtype=np.uint8) * 200
        mask = np.zeros((50, 70), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        result = inpainter.inpaint(image, mask)
        # Output should be cropped back to original dimensions
        assert result.shape == image.shape

    def test_inpaint_error_raises_and_cleans_up(self, monkeypatch):
        """Exception during model inference is re-raised."""
        class ExplodingModel:
            def __call__(self, img_rgb, mask):
                raise RuntimeError("GPU exploded")

        monkeypatch.setattr(lama_mod, "SimpleLama", _FakeSimpleLama)
        monkeypatch.setattr(lama_mod, "select_device", lambda d: "cpu")

        inpainter = LamaInpainter(device="cpu")
        inpainter.model = ExplodingModel()

        image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        with pytest.raises(RuntimeError, match="GPU exploded"):
            inpainter.inpaint(image, mask)

    def test_subregion_output_size_mismatch_resized(self, monkeypatch):
        """If LaMa output doesn't match subregion size, it should be resized."""
        class WrongSizeSimpleLama(_FakeSimpleLama):
            """Subclass of _FakeSimpleLama so isinstance check passes."""
            def __call__(self, img_rgb, mask):
                h, w = img_rgb.shape[:2]
                # Return wrong size (half height)
                wrong = np.ones((h // 2, w, 3), dtype=np.uint8) * 99

                class FakePIL:
                    def __init__(self, arr):
                        self._arr = arr
                        self.shape = arr.shape
                        self.dtype = arr.dtype
                    def convert(self, mode):
                        return self
                    def __array__(self):
                        return self._arr

                return FakePIL(wrong)

        monkeypatch.setattr(lama_mod, "SimpleLama", _FakeSimpleLama)
        monkeypatch.setattr(lama_mod, "select_device", lambda d: "cpu")

        inpainter = LamaInpainter(device="cpu")
        inpainter.model = WrongSizeSimpleLama()

        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 30:50] = 255
        result = inpainter.inpaint(image, mask, subregion=(20, 20, 60, 60))
        # Despite wrong model output, final result should match original image size
        assert result.shape == image.shape
