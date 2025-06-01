"""Module for image inpainting using Deep Image Prior.

This module provides functionality to inpaint masked regions in images using the
Deep Image Prior (DIP) library.

Example:
    >>> from untext.image_patcher import ImagePatcher
    >>> patcher = ImagePatcher(device='cuda')
    >>> patched_paths = patcher.patch_images({'image1.jpg': 'mask1.jpg'})
    >>> print(f"Patched images: {patched_paths}")
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Literal
from .dip_model import DeepImagePrior
from typing import TYPE_CHECKING

try:
    from .lama_inpainter import LamaInpainter
except Exception:  # pragma: no cover
    LamaInpainter = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePatcher:
    """Class for patching images to remove watermarks."""
    
    def __init__(
        self,
        device: str = 'cuda',
        learning_rate: float = 1e-2,
        num_iterations: int = 500,
        known_region_weight: float = 0.01
    ) -> None:
        """Initialize the ImagePatcher.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            learning_rate: Learning rate for the optimizer
            num_iterations: Number of optimization iterations
            known_region_weight: Weight for known regions
        """
        self.dip = DeepImagePrior(
            device=device,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            known_region_weight=known_region_weight
        )
        logger.info("Initialized ImagePatcher with Deep Image Prior")
        
        # LaMa will be loaded lazily when first requested
        self._lama: Optional["LamaInpainter"] = None  # type: ignore[misc]
    
    def patch_image(
        self,
        image_path: str,
        mask_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        blend: bool = True,
        dilate_percent: float = 0.05,
        feather_radius: int = 20,
        method: str = "lama"
    ) -> np.ndarray:
        """Patch an image using its mask.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the binary mask (white pixels indicate regions to inpaint)
            output_path: Optional path to save the result
            progress_callback: Optional callback function to report progress
            blend: Whether to apply Poisson blending
            dilate_percent: Percentage to dilate the mask
            feather_radius: Width of edge ramp (pixels) for feathering edges
            method: 'dip' (default) or 'edge_fill' for fast random edge fill then blur
            
        Returns:
            Patched image as numpy array
        """
        # Load image and mask
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from {mask_path}")
        
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Dilate (bloom) mask to cover text fully
        if dilate_percent > 0:
            h_m, w_m = mask.shape
            dia = int(max(h_m, w_m) * dilate_percent)
            if dia < 3:
                dia = 3
            # Make diameter odd
            if dia % 2 == 0:
                dia += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dia, dia))
            mask = cv2.dilate(mask, kernel)
        
        if method == "edge_fill":
            # Simple fast fill: sample colors from border and blur
            result = image.copy()
            # identify boundary: dilate mask then subtract mask
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            boundary = cv2.dilate(mask, kernel3) - mask
            ys, xs = np.where(boundary > 0)
            if len(xs) == 0:
                # fallback to original image if no boundary (shouldn't happen)
                logger.warning("Edge fill fallback: no boundary pixels found")
            else:
                colors = image[ys, xs]
                ys_m, xs_m = np.where(mask > 0)
                # assign random colors
                rand_idx = np.random.randint(0, len(colors), size=len(ys_m))
                result[ys_m, xs_m] = colors[rand_idx]
                # blur inside mask area slightly
                blur = cv2.GaussianBlur(result, (5,5), 0)
                result[ys_m, xs_m] = blur[ys_m, xs_m]
        elif method == "lama":
            if LamaInpainter is None:
                raise RuntimeError("LaMa dependencies not installed. See lama_inpainter.py for instructions.")
            if self._lama is None:
                # Reuse same device assumption as DIP
                lama_device = 'cuda' if str(self.dip.device).startswith('cuda') else 'cpu'
                self._lama = LamaInpainter(device=lama_device)
            result = self._lama.inpaint(image, mask)
        else:
            # Inpaint using Deep Image Prior
            result = self.dip.inpaint(image, mask, progress_callback)
        
        # Optional blending to smooth edges (edge ramp based on distance transform)
        if blend:
            ramp = feather_radius
            if ramp < 1:
                ramp = 1
            mask_bin = (mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
            alpha = np.clip(dist / float(ramp), 0.0, 1.0)
            alpha = alpha[..., None]

            # Handle potential size mismatch (e.g. LaMa output may differ by 1–2 px after padding)
            h = min(result.shape[0], image.shape[0], alpha.shape[0])
            w = min(result.shape[1], image.shape[1], alpha.shape[1])
            result_crop = result[:h, :w].astype(np.float32)
            image_crop = image[:h, :w].astype(np.float32)
            alpha_crop = alpha[:h, :w]
            blended = (result_crop * alpha_crop + image_crop * (1 - alpha_crop)).astype(np.uint8)
            # Replace the blended area back into result to preserve original size if needed
            result[:h, :w] = blended
        
        # Ensure output has the same spatial size as the input image
        if result.shape[:2] != image.shape[:2]:
            h0, w0 = image.shape[:2]
            result = result[:h0, :w0]
        
        # Save result if output path is provided
        if output_path:
            cv2.imwrite(str(output_path), result)
            logger.info("Saved patched image to %s", output_path)
        
        return result

    def patch_images(
        self,
        image_mask_pairs: Dict[Union[str, Path], Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[Path, Path]:
        """Patch multiple images using their corresponding masks.
        
        Args:
            image_mask_pairs: Dictionary mapping image paths to their mask paths.
            output_dir: Directory to save patched images. If None, will save in the
                       same directory as input images.
        
        Returns:
            Dictionary mapping input image paths to their patched image paths.
        
        Raises:
            ValueError: If any input path is invalid.
            FileNotFoundError: If any input file cannot be found.
            RuntimeError: If inpainting fails.
        """
        # Convert output_dir to Path if provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        patched_paths = {}
        for image_path, mask_path in image_mask_pairs.items():
            try:
                if output_dir is not None:
                    output_path = output_dir / f"{Path(image_path).stem}_patched.jpg"
                else:
                    output_path = None
                    
                patched_path = self.patch_image(image_path, mask_path, output_path)
                patched_paths[Path(image_path)] = patched_path
                
            except Exception as e:
                logger.error("Failed to process image pair %s -> %s: %s",
                           image_path, mask_path, str(e))
                continue
        
        if not patched_paths:
            raise RuntimeError("No images were successfully patched")
            
        return patched_paths 