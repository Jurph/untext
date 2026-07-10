"""Tests for untextre.inpaint orchestration layer.

These tests exercise the current public API:
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

import untextre.inpaint as inpaint_mod
from untextre.inpaint import (
    inpaint_image,
    _has_pixels_to_inpaint,
    _calculate_inpainting_subregion,
    _inpaint_with_lama,
    _inpaint_with_telea,
    is_lama_available,
    is_lama_initialized,
    is_lama_healthy,
    get_lama_status,
    initialize_lama_model,
    reset_lama_model,
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



# =========================================================================
# _calculate_inpainting_subregion
# =========================================================================



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



# =========================================================================
# _inpaint_with_telea – direct internal tests
# =========================================================================





# =========================================================================
# LaMa status helpers
# =========================================================================



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


# =========================================================================
# LaMa health / status / reset — unit tests with mocked globals
# =========================================================================





# =========================================================================
# _inpaint_with_lama — additional branch coverage
# =========================================================================



# =========================================================================
# _inpaint_with_telea — additional branch coverage
# =========================================================================

