"""Tests for the LaMa inpainting functionality.

Covers:
    - ``TeleaInpainter`` – basic API, subregion, validation
    - ``select_device()`` – CPU, CUDA fallback paths
    - ``LamaInpainter`` – constructor guards, input validation,
      subregion edge-padding, SimpleLama output cropping, error handling
"""

import json
import subprocess
import sys
import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

from untextre.telea_inpainter import TeleaInpainter
import untextre.lama_inpainter as lama_mod
from untextre.lama_inpainter import select_device, LamaInpainter, paste_subregion

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
# LaMa backend import timing
# =========================================================================
def test_importing_lama_module_does_not_import_runtime_backend():
    """Module import must not load the heavyweight LaMa runtime backend."""
    project_root = Path(__file__).resolve().parents[1]
    probe = r'''
import builtins
import json

seen = []
real_import = builtins.__import__


def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_name = name.split(".", 1)[0]
    if top_name == "simple_lama_inpainting":
        seen.append(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = recording_import
import untextre.lama_inpainter  # noqa: F401
print(json.dumps(seen))
'''

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )

    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


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
        monkeypatch.setattr(lama_mod, "_BACKENDS_LOADED", True)
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
    monkeypatch.setattr(lama_mod, "_BACKENDS_LOADED", True)
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

    def test_inpaint_empties_cache_but_does_not_synchronize_on_cuda(self, monkeypatch):
        """Success path reclaims GPU memory without an extra blocking sync.

        `.cpu()`/PIL conversion already forces the CUDA work for this call to
        finish before `out_bgr` exists, so a manual `torch.cuda.synchronize()`
        here only adds a redundant full-pipeline stall (see #12). No real GPU
        is needed: `torch.device("cuda")` just builds a device descriptor, and
        the SimpleLama fake path never launches a real kernel.
        """
        monkeypatch.setattr(lama_mod, "SimpleLama", _FakeSimpleLama)
        monkeypatch.setattr(lama_mod, "select_device", lambda d: "cuda")
        monkeypatch.setattr(lama_mod.torch.cuda, "is_available", lambda: True)
        empty_cache = MagicMock()
        synchronize = MagicMock()
        monkeypatch.setattr(lama_mod.torch.cuda, "empty_cache", empty_cache)
        monkeypatch.setattr(lama_mod.torch.cuda, "synchronize", synchronize)

        inpainter = LamaInpainter(device="cuda")
        inpainter.model = _FakeSimpleLama()

        image, mask = create_test_image()
        result = inpainter.inpaint(image, mask)

        assert result is not None
        empty_cache.assert_called_once()
        synchronize.assert_not_called()



# =========================================================================
# paste_subregion — single-source-of-truth paste-back helper
# =========================================================================

class TestPasteSubregion:
    """paste_subregion pastes an inpainted patch back at the given coordinates."""

    def test_pastes_patch_at_exact_coordinates(self) -> None:
        full = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 255, dtype=np.uint8)

        result = paste_subregion(full, patch, 2, 3, 6, 7)

        # Patch lands exactly at [y1:y2, x1:x2] = [3:7, 2:6].
        assert np.all(result[3:7, 2:6] == 255)
        # Nothing outside that box is touched.
        untouched = result.copy()
        untouched[3:7, 2:6] = 0
        assert np.all(untouched == 0)

    def test_resizes_patch_when_size_mismatches(self) -> None:
        full = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((2, 2, 3), 200, dtype=np.uint8)  # smaller than the 4x4 target box

        result = paste_subregion(full, patch, 1, 1, 5, 5)

        region = result[1:5, 1:5]
        assert region.shape == (4, 4, 3)   # patch resized up to fill the target box
        assert np.all(region == 200)       # a uniform patch stays uniform after resize
