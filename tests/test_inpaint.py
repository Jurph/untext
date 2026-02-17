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


class TestInpaintWithLamaRecovery:
    """Branch tests for LaMa orchestration/retry behavior."""

    def test_uninitialized_and_no_retry_raises(self):
        image = np.ones((32, 32, 3), dtype=np.uint8) * 255
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:12, 10:12] = 255

        original_lama_cls = inpaint_mod.LamaInpainter
        inpaint_mod.LamaInpainter = object

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(inpaint_mod, "get_lama_inpainter", lambda: None)
                with pytest.raises(RuntimeError, match="not initialized"):
                    _inpaint_with_lama(image, mask, auto_retry=False)
        finally:
            inpaint_mod.LamaInpainter = original_lama_cls

    def test_retry_reinitialize_then_succeeds(self):
        image = np.ones((40, 40, 3), dtype=np.uint8) * 200
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:24, 14:28] = 255

        class FailingInpainter:
            def inpaint(self, *_args, **_kwargs):
                raise RuntimeError("boom")

        class RecoveredInpainter:
            def inpaint(self, img, _mask, subregion=None):
                assert subregion is not None
                out = img.copy()
                out[mask > 0] = 77
                return out

        state = {"reinitialized": False, "force_reinit": None}

        def fake_get_lama_inpainter():
            if state["reinitialized"]:
                return RecoveredInpainter()
            return FailingInpainter()

        def fake_initialize(device="cuda", force_reinit=False):
            state["reinitialized"] = True
            state["force_reinit"] = force_reinit
            return True

        original_lama_cls = inpaint_mod.LamaInpainter
        inpaint_mod.LamaInpainter = object

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(inpaint_mod, "get_lama_inpainter", fake_get_lama_inpainter)
                mp.setattr(inpaint_mod, "initialize_lama_model", fake_initialize)
                result = _inpaint_with_lama(image, mask, auto_retry=True)

            assert state["force_reinit"] is True
            assert np.all(result[mask > 0] == 77)
        finally:
            inpaint_mod.LamaInpainter = original_lama_cls

    def test_retry_reinitialize_failure_raises(self):
        image = np.ones((32, 32, 3), dtype=np.uint8) * 255
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:14, 8:14] = 255

        class FailingInpainter:
            def inpaint(self, *_args, **_kwargs):
                raise RuntimeError("primary failure")

        original_lama_cls = inpaint_mod.LamaInpainter
        inpaint_mod.LamaInpainter = object

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(inpaint_mod, "get_lama_inpainter", lambda: FailingInpainter())
                mp.setattr(inpaint_mod, "initialize_lama_model", lambda **_kwargs: False)
                with pytest.raises(RuntimeError, match="LaMa inpainting failed"):
                    _inpaint_with_lama(image, mask, auto_retry=True)
        finally:
            inpaint_mod.LamaInpainter = original_lama_cls


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


# =========================================================================
# LaMa health / status / reset — unit tests with mocked globals
# =========================================================================

class TestLamaHealthAndReset:
    """Test is_lama_healthy, reset_lama_model, and edge cases in initialize."""

    def test_healthy_returns_false_when_not_initialized(self, monkeypatch):
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        assert is_lama_healthy() is False

    def test_healthy_returns_false_on_bad_result_shape(self, monkeypatch):
        """Health check returns False when model returns wrong shape."""
        class BadInpainter:
            def inpaint(self, image, mask):
                return np.zeros((10, 10, 3), dtype=np.uint8)  # Wrong size

        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", BadInpainter())
        assert is_lama_healthy() is False

    def test_healthy_returns_false_on_exception(self, monkeypatch):
        """Health check returns False when model raises."""
        class BrokenInpainter:
            def inpaint(self, *_args, **_kwargs):
                raise RuntimeError("model broken")

        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", BrokenInpainter())
        assert is_lama_healthy() is False

    def test_healthy_returns_true_on_good_result(self, monkeypatch):
        """Health check returns True when model returns correct shape."""
        class GoodInpainter:
            def inpaint(self, image, mask):
                return np.zeros_like(image)

        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", GoodInpainter())
        assert is_lama_healthy() is True

    def test_reset_clears_state(self, monkeypatch):
        """reset_lama_model sets inpainter to None and clears failure flag."""
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", "fake_model")
        monkeypatch.setattr(inpaint_mod, "_lama_init_failed", True)
        reset_lama_model()
        assert inpaint_mod._lama_inpainter is None
        assert inpaint_mod._lama_init_failed is False

    def test_reset_handles_cuda_cleanup_error(self, monkeypatch):
        """reset_lama_model handles exceptions during GPU cleanup gracefully."""
        class CudaInpainter:
            class device:
                type = "cuda"

        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", CudaInpainter())
        monkeypatch.setattr(inpaint_mod, "_lama_init_failed", False)
        # Even if torch.cuda.empty_cache fails, reset should complete
        import torch
        original_empty_cache = torch.cuda.empty_cache
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("cleanup boom")))
        reset_lama_model()
        assert inpaint_mod._lama_inpainter is None


class TestInitializeLamaModel:
    """Test initialize_lama_model edge cases."""

    def test_unavailable_returns_false(self, monkeypatch):
        """LamaInpainter is None → returns False."""
        monkeypatch.setattr(inpaint_mod, "LamaInpainter", None)
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        monkeypatch.setattr(inpaint_mod, "_lama_init_failed", False)
        assert initialize_lama_model() is False
        assert inpaint_mod._lama_init_failed is True

    def test_already_initialized_returns_true(self, monkeypatch):
        """Already-initialized model returns True without reloading."""
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", "already_loaded")
        assert initialize_lama_model() is True

    def test_force_reinit_calls_reset(self, monkeypatch):
        """force_reinit=True resets and reinitializes."""
        reset_called = {"n": 0}
        original_reset = reset_lama_model

        def tracking_reset():
            reset_called["n"] += 1
            # Actually clear the global so init proceeds
            inpaint_mod._lama_inpainter = None
            inpaint_mod._lama_init_failed = False

        monkeypatch.setattr(inpaint_mod, "reset_lama_model", tracking_reset)
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", "stale_model")

        class FakeInpainter:
            def inpaint(self, image, mask):
                return np.zeros_like(image)

        monkeypatch.setattr(inpaint_mod, "LamaInpainter", lambda **kw: FakeInpainter())

        result = initialize_lama_model(force_reinit=True)
        assert reset_called["n"] >= 1
        assert result is True

    def test_health_check_failure_resets(self, monkeypatch):
        """Failed health check after construction → reset and return False."""
        class UnhealthyInpainter:
            def inpaint(self, image, mask):
                raise RuntimeError("always broken")

        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        monkeypatch.setattr(inpaint_mod, "_lama_init_failed", False)
        monkeypatch.setattr(inpaint_mod, "LamaInpainter", lambda **kw: UnhealthyInpainter())

        result = initialize_lama_model()
        assert result is False

    def test_constructor_exception_returns_false(self, monkeypatch):
        """Exception during LamaInpainter() → returns False.

        Note: the exception handler calls reset_lama_model() which clears
        _lama_init_failed, so we only verify the return value here.
        """
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        monkeypatch.setattr(inpaint_mod, "_lama_init_failed", False)

        def kaboom(**kw):
            raise RuntimeError("model download failed")

        monkeypatch.setattr(inpaint_mod, "LamaInpainter", kaboom)

        result = initialize_lama_model()
        assert result is False


# =========================================================================
# _inpaint_with_lama — additional branch coverage
# =========================================================================

class TestInpaintWithLamaBranches:
    """Cover remaining branches in _inpaint_with_lama."""

    def test_lama_not_available_raises(self, monkeypatch):
        """LamaInpainter is None → RuntimeError."""
        monkeypatch.setattr(inpaint_mod, "LamaInpainter", None)
        image = np.ones((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        with pytest.raises(RuntimeError, match="not available"):
            _inpaint_with_lama(image, mask)

    def test_auto_init_when_not_initialized(self, monkeypatch):
        """When inpainter is None and auto_retry=True, auto-initializes."""
        class MockInpainter:
            def inpaint(self, image, mask, subregion=None):
                return image.copy()

        init_called = {"n": 0}

        def fake_init(**kw):
            init_called["n"] += 1
            inpaint_mod._lama_inpainter = MockInpainter()
            return True

        monkeypatch.setattr(inpaint_mod, "LamaInpainter", object)
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        monkeypatch.setattr(inpaint_mod, "initialize_lama_model", fake_init)
        monkeypatch.setattr(inpaint_mod, "get_lama_inpainter",
                            lambda: inpaint_mod._lama_inpainter)

        image = np.ones((40, 40, 3), dtype=np.uint8) * 128
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        result = _inpaint_with_lama(image, mask, auto_retry=True)
        assert init_called["n"] == 1
        assert result.shape == image.shape

    def test_auto_init_failure_raises(self, monkeypatch):
        """Auto-init fails → RuntimeError."""
        monkeypatch.setattr(inpaint_mod, "LamaInpainter", object)
        monkeypatch.setattr(inpaint_mod, "_lama_inpainter", None)
        monkeypatch.setattr(inpaint_mod, "get_lama_inpainter", lambda: None)
        monkeypatch.setattr(inpaint_mod, "initialize_lama_model", lambda **kw: False)

        image = np.ones((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        with pytest.raises(RuntimeError, match="Failed to auto-initialize"):
            _inpaint_with_lama(image, mask, auto_retry=True)

    def test_no_subregion_returns_copy(self, monkeypatch):
        """Empty mask after subregion calc → returns original copy."""
        monkeypatch.setattr(inpaint_mod, "LamaInpainter", object)

        class MockInpainter:
            pass

        monkeypatch.setattr(inpaint_mod, "get_lama_inpainter", lambda: MockInpainter())
        monkeypatch.setattr(inpaint_mod, "_calculate_inpainting_subregion", lambda *a, **kw: None)

        image = np.ones((32, 32, 3), dtype=np.uint8) * 200
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        result = _inpaint_with_lama(image, mask)
        np.testing.assert_array_equal(result, image)
        assert result is not image  # It's a copy

    def test_retry_exception_raises_original(self, monkeypatch):
        """When both primary and retry fail, raises RuntimeError."""
        class AlwaysFailInpainter:
            def inpaint(self, *_args, **_kwargs):
                raise RuntimeError("always fails")

        monkeypatch.setattr(inpaint_mod, "LamaInpainter", object)
        monkeypatch.setattr(inpaint_mod, "get_lama_inpainter", lambda: AlwaysFailInpainter())
        monkeypatch.setattr(inpaint_mod, "initialize_lama_model", lambda **kw: True)

        image = np.ones((40, 40, 3), dtype=np.uint8) * 128
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        with pytest.raises(RuntimeError, match="LaMa inpainting failed"):
            _inpaint_with_lama(image, mask, auto_retry=True)


# =========================================================================
# _inpaint_with_telea — additional branch coverage
# =========================================================================

class TestInpaintWithTeleaBranches:
    """Cover remaining branches in _inpaint_with_telea."""

    def test_empty_processed_mask_returns_copy(self):
        """Mask with all values <= 0 after processing returns original."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 200
        mask = np.zeros((50, 50), dtype=np.uint8)  # All zero
        # _has_pixels_to_inpaint catches this at the dispatch level,
        # but _inpaint_with_telea has its own check after mask processing.
        result = _inpaint_with_telea(image, mask)
        np.testing.assert_array_equal(result, image)
        assert result is not image
