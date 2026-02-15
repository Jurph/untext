from __future__ import annotations

"""Light-weight wrapper around OpenCV's fast Telea in-painting algorithm.

This exists mainly to give a uniform interface (`inpaint(image, mask, …)`) so
`ImagePatcher` can treat Telea the same way it treats LaMa or DIP.
It has **no external dependencies** beyond OpenCV.
"""

from typing import Optional, Tuple
from pathlib import Path
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Subregion = Tuple[int, int, int, int]


class TeleaInpainter:  # pylint: disable=too-few-public-methods
    """Thin wrapper around `cv2.inpaint(..., cv2.INPAINT_TELEA)`."""

    def __init__(self, radius: int = 3) -> None:
        """Create a Telea inpainter.

        Args:
            radius: Pixel neighbourhood radius passed to `cv2.inpaint`.
        """
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = radius

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        subregion: Optional[Subregion] = None,
    ) -> np.ndarray:
        """Inpaint ``mask`` region of ``image`` using the Telea method.

        Args:
            image: H×W×3 BGR ``uint8`` array.
            mask: H×W ``uint8`` array – non-zero pixels mark the regions to fill.
            subregion: Optional (x1, y1, x2, y2) tuple – if given, only that crop
                        is processed and then pasted back so output size matches
                        the input.

        Returns:
            Inpainted image of the same shape as *image*.
        """
        if image is None or mask is None:
            raise ValueError("image and mask must not be None")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be H×W×3 BGR")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != image.shape[:2]:
            raise ValueError("mask and image must have identical height/width")

        # Ensure binary mask values are 0 / 255
        mask_bin = (mask > 127).astype(np.uint8) * 255

        full_image = None
        if subregion is not None:
            x1, y1, x2, y2 = subregion
            if x2 <= x1 or y2 <= y1:
                raise ValueError("subregion must have positive width/height")
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                raise ValueError("subregion coordinates out of bounds")
            full_image = image.copy()
            image = image[y1:y2, x1:x2]
            mask_bin = mask_bin[y1:y2, x1:x2]

        logger.debug("Running cv2.inpaint (Telea) on region %sx%s", image.shape[1], image.shape[0])
        result = cv2.inpaint(src=image, inpaintMask=mask_bin, inpaintRadius=self.radius, flags=cv2.INPAINT_TELEA)

        if full_image is not None:
            full_image[y1:y2, x1:x2] = result
            return full_image
        return result 