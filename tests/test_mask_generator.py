"""Tests for untextre.mask_generator module.

These tests exercise the morphological cleanup function that refines
binary masks for inpainting.  All inputs are synthetic numpy arrays,
so the tests are fast and deterministic.
"""

import numpy as np
import pytest

from untextre.mask_generator import morph_clean_mask


# ---------------------------------------------------------------------------
# Fixtures local to this test module
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_mask():
    """100x100 binary mask with a solid 20x20 white block at (40,40)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    return mask


# ---------------------------------------------------------------------------
# morph_clean_mask
# ---------------------------------------------------------------------------

class TestMorphCleanMask:
    """Tests for morphological cleanup of binary masks."""

    def test_output_is_binary(self, simple_mask):
        """Cleaned mask should contain only 0 and 255 values."""
        cleaned = morph_clean_mask(simple_mask)

        unique_values = set(np.unique(cleaned))
        assert unique_values <= {0, 255}, f"Non-binary values found: {unique_values}"

    def test_preserves_shape(self, simple_mask):
        """Cleaned mask should have the same dimensions as input."""
        cleaned = morph_clean_mask(simple_mask)
        assert cleaned.shape == simple_mask.shape

    def test_does_not_erase_solid_block(self, simple_mask):
        """A solid block should survive cleanup (possibly grow, but not vanish)."""
        cleaned = morph_clean_mask(simple_mask)

        # The original 20x20 block had 400 white pixels.
        # After closing + dilation it should have at least as many.
        assert np.sum(cleaned == 255) >= 400, "Cleanup should not erase a solid block"

    def test_empty_mask_stays_empty(self):
        """An all-black mask should remain all-black after cleanup."""
        empty = np.zeros((100, 100), dtype=np.uint8)
        cleaned = morph_clean_mask(empty)
        assert np.sum(cleaned == 255) == 0
