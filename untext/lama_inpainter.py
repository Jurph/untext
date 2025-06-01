"""Wrapper for SAIC-AI LaMa in-painting model.

Attempts to load a pre-trained LaMa checkpoint ("big-lama") on first use
and exposes a single `inpaint(image, mask)` method compatible with
`ImagePatcher`.

If LaMa or its dependencies are not installed, a RuntimeError is raised
with installation hints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2  # OpenCV for colour space conversions

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
except ImportError as _torch_err:  # pragma: no cover
    torch = None  # type: ignore
    _IMPORT_ERROR = _torch_err
else:
    _IMPORT_ERROR = None

# Prefer the lightweight wheel; if it's available we do not require the heavy original repo
try:
    from simple_lama_inpainting import SimpleLama  # type: ignore
except ImportError:  # pragma: no cover
    SimpleLama = None  # type: ignore

# Only try to import the original LaMa repo if SimpleLama is missing
if SimpleLama is None:
    try:
        from saicinpainting.training.trainers import load_checkpoint  # type: ignore
    except ImportError:
        load_checkpoint = None  # type: ignore
    else:
        load_checkpoint = load_checkpoint  # just to satisfy linters
else:
    load_checkpoint = None  # type: ignore


class LamaInpainter:  # pylint: disable=too-few-public-methods
    """Thin convenience wrapper around the SAIC-AI LaMa model."""

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: str = "cuda",
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for LaMa. Install with `pip install torch torchvision`"
            ) from _IMPORT_ERROR

        if SimpleLama is None and load_checkpoint is None:
            raise RuntimeError(
                "Neither simple-lama-inpainting nor the original LaMa repo is available. "
                "Install one of them: `pip install simple-lama-inpainting` or "
                "`pip install git+https://github.com/advimman/lama.git@main#subdirectory=saicinpainting`"
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            # Default to the official big-lama checkpoint name. The loader will
            # download it to ~/.cache/ if not present.
            checkpoint_path = "big-lama"

        logger.info("Loading LaMa model (%s) on %s", checkpoint_path, self.device)

        if SimpleLama is not None:
            logger.info("Loading simple-lama-inpainting model on %s", self.device)
            # SimpleLama handles device internally via torch default device
            self.model = SimpleLama()
        else:
            logger.info("Falling back to original LaMa repo loader on %s", self.device)
            self.model = load_checkpoint(checkpoint_path, map_location=self.device)
            self.model.freeze()
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()  # type: ignore[misc]
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint `mask` region of `image`.

        Args:
            image: H×W×3 BGR uint8
            mask: H×W uint8, 255 = hole
        Returns:
            Inpainted H×W×3 BGR uint8
        """
        import torch  # local import to satisfy mypy in absence of torch above

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore[name-defined]
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        msk_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        img_t = img_t.to(self.device)
        msk_t = msk_t.to(self.device)

        if SimpleLama is not None and isinstance(self.model, SimpleLama):
            # SimpleLama expects PIL.Image
            from PIL import Image  # type: ignore

            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)
            out_pil = self.model(pil_img, pil_mask)
            out_np = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            return out_np

        # Original repo path
        out = self.model(img_t, msk_t)
        out = out.clamp(0, 1) * 255.0
        out_np = out[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)  # type: ignore[name-defined]
        return out_bgr 