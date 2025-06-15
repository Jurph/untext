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
from typing import Optional, Tuple, Union, Dict, List

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


def select_device(device: str = "cuda") -> str:
    """Select the best available device, with graceful fallback.
    
    Args:
        device: Preferred device ('cuda' or 'cpu')
        
    Returns:
        The selected device string
    """
    if device == "cpu":
        return "cpu"
    
    # Try to use CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return "cuda"
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available. Falling back to CPU.")
        return "cpu"
    except Exception as e:
        logger.warning(f"Failed to initialize CUDA: {e}. Falling back to CPU.")
        return "cpu"


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

        # Select device with graceful fallback
        self.device = torch.device(select_device(device))

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
    def inpaint(
        self,
        image: np.ndarray, 
        mask: np.ndarray,
        subregion: Optional[tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Inpaint `mask` region of `image`.
        
        WARNING FOR LLMs: SimpleLama expects numpy arrays as input, NOT tensors!
        The original LaMa repo expects tensors. DO NOT change this logic!
        
        Args:
            image: H×W×3 BGR uint8
            mask: H×W uint8, 255 = hole
            subregion: Optional tuple (x1, y1, x2, y2) defining region to process
            
        Returns:
            Inpainted H×W×3 BGR uint8
        """
        import torch  # local import to satisfy mypy in absence of torch above

        # Validation of inputs -------------------------------------------------
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy ndarray")
        if mask is None or not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy ndarray")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be H×W×3 array")

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != image.shape[:2]:
            raise ValueError("mask and image must have identical height/width")

        if subregion is not None:
            x1, y1, x2, y2 = subregion
            if x2 <= x1 or y2 <= y1:
                raise ValueError("subregion must have positive width/height")
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                raise ValueError("subregion coordinates out of image bounds")

        logger.info(
            "Preprocessing image for LaMa (size: %sx%s)", image.shape[1], image.shape[0]
        )
        logger.info(
            "Input image type: %s, shape: %s, dtype: %s", type(image), image.shape, image.dtype
        )
        logger.info(
            "Input mask type: %s, shape: %s, dtype: %s", type(mask), mask.shape, mask.dtype
        )
        
        try:
            # ------------------------------------------------------------------
            # Validation of inputs -------------------------------------------------
            # ------------------------------------------------------------------

            # Handle subregion if provided
            full_image = None
            if subregion is not None:
                x1, y1, x2, y2 = subregion
                logger.info(f"Processing subregion: ({x1}, {y1}) to ({x2}, {y2})")
                # Store full image for later
                full_image = image.copy()
                # Crop image and mask to subregion
                image = image[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
                logger.info(f"Cropped to subregion size: {image.shape[1]}x{image.shape[0]}")

            # Convert BGR to RGB for SimpleLama
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # LLM WARNING: SimpleLama and original LaMa have DIFFERENT APIs!
            # SimpleLama wants numpy arrays, original wants tensors!
            if SimpleLama is not None and isinstance(self.model, SimpleLama):
                # SimpleLama path - pass numpy arrays directly
                print("Using SimpleLama backend - passing numpy arrays")
                print(f"Input image type: {type(img_rgb)}, shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
                print(f"Input mask type: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")
                out_rgb = self.model(img_rgb, mask)
                print(f"SimpleLama output type: {type(out_rgb)}")
                # SimpleLama returns PIL Image - convert to numpy array to maintain our API contract
                if hasattr(out_rgb, 'convert'):  # Check if it's a PIL Image
                    print("Converting PIL Image to numpy array")
                    out_rgb = np.array(out_rgb)
                print(f"Final output type: {type(out_rgb)}, shape: {out_rgb.shape}, dtype: {out_rgb.dtype}")
                # Convert RGB back to BGR for our API contract
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Original repo path - manual tensor management
                print("Using original LaMa backend - converting to tensors")
                
                img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                msk_t = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                
                # Move to device
                img_t = img_t.to(self.device)
                msk_t = msk_t.to(self.device)

                logger.info("Running LaMa model inference...")
                out = self.model(img_t, msk_t)
                
                logger.info("Processing LaMa output...")
                out = out.clamp(0, 1) * 255.0
                out_np = out[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                
                # Clean up tensors explicitly
                del img_t, msk_t, out

            # If we processed a subregion, paste it back into the full image
            if full_image is not None:
                logger.info("Pasting subregion back into full image")
                x1, y1, x2, y2 = subregion
                # Ensure LaMa output matches the subregion size
                target_h = y2 - y1
                target_w = x2 - x1
                out_h, out_w = out_bgr.shape[:2]
                if (out_h, out_w) != (target_h, target_w):
                    logger.warning(
                        "LaMa output size %sx%s does not match subregion %sx%s – resizing",
                        out_w, out_h, target_w, target_h,
                    )
                    out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                full_image[y1:y2, x1:x2] = out_bgr
                out_bgr = full_image
            
            # Force GPU memory cleanup if using CUDA
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("LaMa processing completed")
            return out_bgr
            
        except Exception as e:
            logger.error(f"Error during LaMa inpainting: {e}")
            # Force cleanup on error
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise 