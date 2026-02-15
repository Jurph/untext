"""Tests for untextre.inpaint orchestration layer.

Replaces the disabled test_inpainting.py (which targeted the removed
``untext.image_patcher.ImagePatcher`` class) and test_subregion_cropping.py
(which targeted the removed ``ImagePatcher.calculate_subregion`` method).

These tests exercise the *current* public API:
    - ``inpaint_image()``   – main entry point (method dispatch, validation)
    - ``_inpaint_with_telea()`` – TELEA path
    - ``_has_pixels_to_inpaint()`` – helper
    - ``_calculate_inpainting_subregion()`` – subregion calculation
    - LaMa status helpers (``is_lama_available``, ``is_lama_initialized``, etc.)
    - Comparative inpainting quality (TELEA vs LaMa on same synthetic image)
"""

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from untextre.inpaint import (
    inpaint_image,
    _has_pixels_to_inpaint,
    _calculate_inpainting_subregion,
    _inpaint_with_telea,
    is_lama_available,
    is_lama_initialized,
    get_lama_status,
    initialize_lama_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_200():
    """200×200 white BGR image."""
    return np.ones((200, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def text_image_and_mask():
    """200×200 white image with black text and a matching binary mask."""
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mask = np.zeros((200, 200), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "TEST", (40, 120), font, 1.5, (0, 0, 0), 3)
    cv2.putText(mask, "TEST", (40, 120), font, 1.5, 255, 3)

    return image, mask


@pytest.fixture
def empty_mask():
    """200×200 all-zero mask (nothing to inpaint)."""
    return np.zeros((200, 200), dtype=np.uint8)


@pytest.fixture
def small_centered_mask():
    """200×200 mask with a small white rectangle in the center."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[80:120, 60:140] = 255
    return mask


# =========================================================================
# _has_pixels_to_inpaint
# =========================================================================

class TestHasPixelsToInpaint:
    """Tests for the quick empty-mask check."""

    def test_empty_mask_returns_false(self, empty_mask):
        assert not _has_pixels_to_inpaint(empty_mask)

    def test_non_empty_mask_returns_true(self, small_centered_mask):
        assert _has_pixels_to_inpaint(small_centered_mask)

    def test_single_white_pixel(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        assert _has_pixels_to_inpaint(mask)


# =========================================================================
# _calculate_inpainting_subregion
# =========================================================================

class TestCalculateInpaintingSubregion:
    """Tests for subregion calculation from a binary mask.

    The function returns (x1, y1, x2, y2) enclosing the mask's white
    pixels, dilated by 64 px, and padded to mod-8 dimensions.
    """

    def test_empty_mask_returns_none(self, empty_mask):
        result = _calculate_inpainting_subregion(empty_mask, image_shape=(200, 200))
        assert result is None

    def test_centered_mask_returns_valid_subregion(self, small_centered_mask):
        subregion = _calculate_inpainting_subregion(
            small_centered_mask, image_shape=(200, 200)
        )
        assert subregion is not None
        x1, y1, x2, y2 = subregion

        # Basic sanity: the subregion contains the mask region
        assert x1 <= 60, f"x1={x1} should be <= mask left edge 60"
        assert y1 <= 80, f"y1={y1} should be <= mask top edge 80"
        assert x2 >= 140, f"x2={x2} should be >= mask right edge 140"
        assert y2 >= 120, f"y2={y2} should be >= mask bottom edge 120"

    def test_subregion_within_image_bounds(self, small_centered_mask):
        subregion = _calculate_inpainting_subregion(
            small_centered_mask, image_shape=(200, 200)
        )
        x1, y1, x2, y2 = subregion
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 200
        assert y2 <= 200

    def test_subregion_dimensions_are_mod8(self, small_centered_mask):
        """Subregion width and height should be multiples of 8 (NN compat)."""
        subregion = _calculate_inpainting_subregion(
            small_centered_mask, image_shape=(200, 200)
        )
        x1, y1, x2, y2 = subregion
        width = x2 - x1
        height = y2 - y1
        assert width % 8 == 0, f"Width {width} not divisible by 8"
        assert height % 8 == 0, f"Height {height} not divisible by 8"

    def test_mask_near_edge_clamps_to_bounds(self):
        """Mask near the image edge should not produce out-of-bounds subregion."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[180:200, 180:200] = 255  # Bottom-right corner

        subregion = _calculate_inpainting_subregion(mask, image_shape=(200, 200))
        assert subregion is not None
        x1, y1, x2, y2 = subregion
        assert x2 <= 200
        assert y2 <= 200

    def test_mask_near_origin_clamps_to_zero(self):
        """Mask near the top-left corner should not produce negative coords."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[0:10, 0:10] = 255

        subregion = _calculate_inpainting_subregion(mask, image_shape=(200, 200))
        assert subregion is not None
        x1, y1, x2, y2 = subregion
        assert x1 >= 0
        assert y1 >= 0

    def test_full_image_mask(self):
        """Mask covering every pixel should produce a subregion covering the image."""
        mask = np.ones((200, 200), dtype=np.uint8) * 255
        subregion = _calculate_inpainting_subregion(mask, image_shape=(200, 200))
        assert subregion is not None
        x1, y1, x2, y2 = subregion
        # Should span the entire image (after dilation clamps to bounds)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 200
        assert y2 == 200


# =========================================================================
# inpaint_image  – method dispatch and validation
# =========================================================================

class TestInpaintImage:
    """Tests for the main inpaint_image entry point."""

    def test_invalid_method_raises(self, white_200, small_centered_mask):
        with pytest.raises(ValueError, match="Invalid inpainting method"):
            inpaint_image(white_200, small_centered_mask, method="magic")

    def test_empty_mask_returns_copy(self, white_200, empty_mask):
        """Empty mask should return an unchanged copy of the original."""
        result = inpaint_image(white_200, empty_mask, method="telea")
        assert np.array_equal(result, white_200)
        # Verify it's a *copy*, not the same object
        assert result is not white_200

    def test_telea_inpainting_modifies_masked_region(self, text_image_and_mask):
        """TELEA method should modify pixels under the mask."""
        image, mask = text_image_and_mask
        result = inpaint_image(image, mask, method="telea")

        assert result.shape == image.shape
        assert result.dtype == np.uint8

        # The masked region should differ from the original (text was removed)
        masked_pixels_original = image[mask > 0]
        masked_pixels_result = result[mask > 0]
        assert not np.array_equal(masked_pixels_result, masked_pixels_original), (
            "Inpainting should change pixels under the mask"
        )

    def test_telea_preserves_unmasked_region(self, text_image_and_mask):
        """Pixels outside the mask should be unchanged after TELEA inpainting."""
        image, mask = text_image_and_mask
        result = inpaint_image(image, mask, method="telea")

        unmasked = mask == 0
        assert np.array_equal(result[unmasked], image[unmasked]), (
            "TELEA should not modify pixels outside the mask"
        )

    def test_telea_output_shape_matches_input(self, text_image_and_mask):
        image, mask = text_image_and_mask
        result = inpaint_image(image, mask, method="telea")
        assert result.shape == image.shape


# =========================================================================
# _inpaint_with_telea – direct internal tests
# =========================================================================

class TestInpaintWithTelea:
    """Tests for the TELEA inpainting path specifically."""

    def test_basic_inpainting(self, text_image_and_mask):
        image, mask = text_image_and_mask
        result = _inpaint_with_telea(image, mask)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_3channel_mask_handled(self, white_200):
        """TELEA path should gracefully handle a 3-channel mask."""
        mask_3ch = np.zeros((200, 200, 3), dtype=np.uint8)
        mask_3ch[80:120, 60:140] = 255
        # Should not raise — the function converts to single channel
        result = _inpaint_with_telea(white_200, mask_3ch)
        assert result.shape == white_200.shape


# =========================================================================
# LaMa status helpers
# =========================================================================

@pytest.mark.slow
class TestLamaStatusHelpers:
    """Tests for LaMa availability / initialization queries.

    We don't require LaMa to be installed, so these just verify the
    functions return consistent types without crashing.
    """

    def test_is_lama_available_returns_bool(self):
        assert isinstance(is_lama_available(), bool)

    def test_is_lama_initialized_returns_bool(self):
        assert isinstance(is_lama_initialized(), bool)

    def test_get_lama_status_returns_dict(self):
        status = get_lama_status()
        assert isinstance(status, dict)
        # Required keys
        for key in ("available", "initialized", "healthy", "device", "init_failed"):
            assert key in status, f"Missing key '{key}' in LaMa status"

    def test_status_keys_consistent_with_helpers(self):
        """Status dict values should agree with the individual helper functions."""
        status = get_lama_status()
        assert status["available"] == is_lama_available()
        assert status["initialized"] == is_lama_initialized()


# =========================================================================
# Comparative inpainting quality: TELEA vs LaMa on same synthetic image
# =========================================================================

def _make_shrub_image(size: int = 480, seed: int = 42) -> np.ndarray:
    """Generate a synthetic 'shrub-like' textured BGR image.

    Uses layered Gaussian-blurred noise in green/brown tones to create
    a natural-looking texture that both inpainters must reconstruct.
    Deterministic via *seed* so the test is reproducible.
    """
    rng = np.random.RandomState(seed)

    # Base green channel with variation
    g = rng.randint(60, 160, (size, size), dtype=np.uint8)
    r = rng.randint(30, 100, (size, size), dtype=np.uint8)
    b = rng.randint(20, 80, (size, size), dtype=np.uint8)
    image = np.stack([b, g, r], axis=-1)  # BGR

    # Blur to create organic-looking patches
    image = cv2.GaussianBlur(image, (15, 15), 5)

    # Add fine grain noise for realism
    noise = rng.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def _stamp_letter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stamp a white capital 'A' onto *image*, filling corresponding *mask* pixels."""
    stamped = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Place a thick "A" in the center-ish area
    cv2.putText(stamped, "A", (185, 310), font, 4.0, (255, 255, 255), 8)
    cv2.putText(mask, "A", (185, 310), font, 4.0, 255, 8)
    return stamped


@pytest.fixture(scope="module")
def shrub_test_data():
    """Module-scoped fixture: clean shrub, stamped shrub, and mask.

    Returns (clean_image, stamped_image, mask) — all 480×480.
    """
    clean = _make_shrub_image(480)
    mask = np.zeros((480, 480), dtype=np.uint8)
    stamped = _stamp_letter(clean, mask)
    return clean, stamped, mask


@pytest.fixture(scope="module")
def lama_ready():
    """Ensure LaMa is initialized for the comparative tests."""
    if not is_lama_available():
        pytest.skip("LaMa inpainter not installed")
    if not is_lama_initialized():
        if not initialize_lama_model(device="cuda"):
            pytest.skip("LaMa model failed to initialize")
    return True


@pytest.mark.slow
class TestComparativeInpainting:
    """Compare TELEA and LaMa inpainting on the same synthetic image.

    Test image: a 480×480 'shrub' texture with a white capital 'A'
    stamped on it.  Both inpainters receive the stamped image and
    the mask marking the 'A'.  We compare each result to the original
    clean shrub using SSIM.

    These tests verify:
        1. Both methods produce output of correct shape/dtype
        2. Both methods modify only the masked region
        3. Both methods achieve a minimum SSIM against the clean original
        4. (Informational) which method scores higher
    """

    def test_telea_restores_shrub(self, shrub_test_data):
        clean, stamped, mask = shrub_test_data
        result = inpaint_image(stamped, mask, method="telea")

        assert result.shape == clean.shape
        assert result.dtype == np.uint8

        # SSIM against the clean original (grayscale comparison)
        clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        score = ssim(clean_gray, result_gray, data_range=255)

        # TELEA on a small region with natural texture should do reasonably well
        assert score > 0.80, f"TELEA SSIM {score:.4f} too low (expected > 0.80)"

    def test_telea_preserves_unmasked(self, shrub_test_data):
        _, stamped, mask = shrub_test_data
        result = inpaint_image(stamped, mask, method="telea")
        unmasked = mask == 0
        assert np.array_equal(result[unmasked], stamped[unmasked])

    def test_lama_restores_shrub(self, shrub_test_data, lama_ready):
        clean, stamped, mask = shrub_test_data
        result = inpaint_image(stamped, mask, method="lama")

        assert result.shape == clean.shape
        assert result.dtype == np.uint8

        clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        score = ssim(clean_gray, result_gray, data_range=255)

        assert score > 0.80, f"LaMa SSIM {score:.4f} too low (expected > 0.80)"

    def test_lama_preserves_unmasked(self, shrub_test_data, lama_ready):
        _, stamped, mask = shrub_test_data
        result = inpaint_image(stamped, mask, method="lama")
        unmasked = mask == 0
        assert np.array_equal(result[unmasked], stamped[unmasked])

    def test_both_methods_beat_stamped_baseline(self, shrub_test_data, lama_ready):
        """Both inpainted results should be closer to the clean original
        than the stamped (damaged) image is."""
        clean, stamped, mask = shrub_test_data
        clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        stamped_gray = cv2.cvtColor(stamped, cv2.COLOR_BGR2GRAY)

        baseline_ssim = ssim(clean_gray, stamped_gray, data_range=255)

        telea_result = inpaint_image(stamped, mask, method="telea")
        telea_gray = cv2.cvtColor(telea_result, cv2.COLOR_BGR2GRAY)
        telea_ssim = ssim(clean_gray, telea_gray, data_range=255)

        lama_result = inpaint_image(stamped, mask, method="lama")
        lama_gray = cv2.cvtColor(lama_result, cv2.COLOR_BGR2GRAY)
        lama_ssim = ssim(clean_gray, lama_gray, data_range=255)

        assert telea_ssim > baseline_ssim, (
            f"TELEA ({telea_ssim:.4f}) should beat stamped baseline ({baseline_ssim:.4f})"
        )
        assert lama_ssim > baseline_ssim, (
            f"LaMa ({lama_ssim:.4f}) should beat stamped baseline ({baseline_ssim:.4f})"
        )
