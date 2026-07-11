"""Wrapper for SAIC-AI LaMa in-painting model.

Attempts to load a pre-trained LaMa checkpoint ("big-lama") on first use
and exposes a single `inpaint(image, mask)` method with the same signature
as `TeleaInpainter`.

If LaMa or its dependencies are not installed, a RuntimeError is raised
with installation hints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

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


def paste_subregion(
    full_image: np.ndarray,
    patch: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Paste an inpainted ``patch`` back into ``full_image`` at ``[y1:y2, x1:x2]``.

    Single source of truth for subregion paste-back geometry: the caller passes
    the same (already edge-pad-adjusted) coordinates it used to crop the
    subregion, so the crop and the paste cannot drift apart.
    """
    target_h, target_w = y2 - y1, x2 - x1
    patch_h, patch_w = patch.shape[:2]
    if (patch_h, patch_w) != (target_h, target_w):
        logger.warning(
            "Inpainted patch size %sx%s does not match subregion %sx%s - resizing",
            patch_w, patch_h, target_w, target_h,
        )
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
    full_image[y1:y2, x1:x2] = patch
    return full_image


class LamaInpainter:  # pylint: disable=too-few-public-methods
    """Thin convenience wrapper around the SAIC-AI LaMa model."""

    # Genuinely dynamic: either a SimpleLama instance (typed) or an
    # original-LaMa-repo model object from the unstubbed `saicinpainting`
    # package. No shared interface exists to type this more precisely.
    model: Any

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
            if load_checkpoint is None:
                raise RuntimeError(
                    "Neither simple-lama-inpainting nor the original LaMa repo is available."
                )
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
            raise ValueError("image must be HxWx3 array")

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
            edge_pad_size = 0
            edge_pad_top = edge_pad_bottom = edge_pad_left = edge_pad_right = 0
            
            if subregion is not None:
                x1, y1, x2, y2 = subregion
                img_h, img_w = image.shape[:2]
                logger.info(f"Processing subregion: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Check if subregion touches image edges - LaMa needs context beyond edges
                touches_left = (x1 == 0)
                touches_right = (x2 >= img_w)
                touches_top = (y1 == 0)
                touches_bottom = (y2 >= img_h)
                
                # If any edge is touched, pad the image with BORDER_REFLECT to give LaMa context
                if touches_left or touches_right or touches_top or touches_bottom:
                    edge_pad_size = 32  # EMPIRICAL — provides reflected context at edges; not formally validated
                    
                    edge_pad_left = edge_pad_size if touches_left else 0
                    edge_pad_right = edge_pad_size if touches_right else 0
                    edge_pad_top = edge_pad_size if touches_top else 0
                    edge_pad_bottom = edge_pad_size if touches_bottom else 0
                    
                    logger.info(f"Subregion touches edges (L:{touches_left} R:{touches_right} T:{touches_top} B:{touches_bottom})")
                    logger.info(f"Padding image: top={edge_pad_top}, bottom={edge_pad_bottom}, left={edge_pad_left}, right={edge_pad_right}")
                    
                    # Pad both image and mask with BORDER_REFLECT
                    image = cv2.copyMakeBorder(
                        image, edge_pad_top, edge_pad_bottom, edge_pad_left, edge_pad_right,
                        cv2.BORDER_REFLECT
                    )
                    mask = cv2.copyMakeBorder(
                        mask, edge_pad_top, edge_pad_bottom, edge_pad_left, edge_pad_right,
                        cv2.BORDER_REFLECT
                    )
                    
                    # Adjust subregion coordinates to account for padding
                    x1 += edge_pad_left
                    y1 += edge_pad_top
                    x2 += edge_pad_left
                    y2 += edge_pad_top
                    
                    logger.info(f"Adjusted subregion after padding: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Store full (possibly padded) image for later
                full_image = image.copy()
                # Crop image and mask to subregion
                image = image[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]
                logger.info(f"Cropped to subregion size: {image.shape[1]}x{image.shape[0]}")

            # Convert BGR to RGB for SimpleLama
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # SimpleLama and the original LaMa repo expose different APIs:
            # SimpleLama takes numpy arrays, while the original backend takes tensors.
            # VERIFIED BEHAVIOR (from simple_lama_inpainting source):
            # - SimpleLama pads input to mod-8 on RIGHT and BOTTOM edges using symmetric padding
            # - Output is the PADDED size (not cropped back to original)
            # - We must CROP (not resize) the output to restore original dimensions
            #
            if SimpleLama is not None and isinstance(self.model, SimpleLama):
                # SimpleLama path - pass numpy arrays directly
                # Track original dimensions BEFORE SimpleLama's internal padding
                orig_h, orig_w = img_rgb.shape[:2]
                
                logger.debug("Using SimpleLama backend - passing numpy arrays")
                logger.debug(f"Input image type: {type(img_rgb)}, shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
                logger.debug(f"Input mask type: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")
                out_rgb = self.model(img_rgb, mask)
                logger.debug(f"SimpleLama output type: {type(out_rgb)}")
                # SimpleLama always returns a PIL Image (Image.fromarray(...) in its
                # source); np.asarray() converts it to a real ndarray. Using asarray
                # (not a conditional + np.array) also accepts any PIL-like duck type
                # (anything implementing __array__, e.g. test doubles) without a
                # brittle isinstance/hasattr check, and is a no-op if already ndarray.
                out_rgb = np.asarray(out_rgb)
                logger.debug(f"Final output type: {type(out_rgb)}, shape: {out_rgb.shape}, dtype: {out_rgb.dtype}")
                
                # CRITICAL: SimpleLama pads to mod-8, output is padded size
                # We must CROP back to original dimensions (not resize!)
                out_h, out_w = out_rgb.shape[:2]
                if out_h != orig_h or out_w != orig_w:
                    logger.debug(f"SimpleLama output {out_w}x{out_h} differs from input {orig_w}x{orig_h} - cropping (not resizing)")
                    out_rgb = out_rgb[:orig_h, :orig_w]
                
                # Convert RGB back to BGR for our API contract
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Original repo path - manual tensor management
                logger.debug("Using original LaMa backend - converting to tensors")
                
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
                
                # x1, y1, x2, y2 are the edge-pad-adjusted coordinates already used
                # to crop this subregion above — the single source of truth for placement.
                out_bgr = paste_subregion(full_image, out_bgr, x1, y1, x2, y2)
                
                # If we padded the image for edge handling, crop back to original dimensions
                if edge_pad_size > 0:
                    orig_h = out_bgr.shape[0] - edge_pad_top - edge_pad_bottom
                    orig_w = out_bgr.shape[1] - edge_pad_left - edge_pad_right
                    out_bgr = out_bgr[edge_pad_top:edge_pad_top+orig_h, edge_pad_left:edge_pad_left+orig_w]
                    logger.info(f"Cropped padded result back to original size: {out_bgr.shape[1]}x{out_bgr.shape[0]}")
            
            # Reclaim cached GPU memory. No explicit torch.cuda.synchronize()
            # here: by this point `out_bgr` is already CPU-resident (the
            # manual-tensor branch called `.cpu()` on `out`; the SimpleLama
            # branch got a PIL Image back from `self.model(...)`), and both
            # of those transfers implicitly block until this call's CUDA work
            # completes. An extra manual synchronize() adds nothing for
            # correctness here -- it only forces the CPU to wait on *every*
            # queued CUDA stream, which serializes back-to-back images in a
            # batch run instead of letting their GPU work overlap.
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info("LaMa processing completed")
            return out_bgr
            
        except Exception as e:
            logger.error(f"Error during LaMa inpainting: {e}")
            # Force cleanup on error
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise 
