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
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
from .dip_model import DeepImagePrior
from untext.detector import TextDetector

try:
    from .lama_inpainter import LamaInpainter
except Exception:  # pragma: no cover
    LamaInpainter = None  # type: ignore

try:
    from .telea_inpainter import TeleaInpainter
except Exception:  # pragma: no cover
    TeleaInpainter = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ImageArray = np.ndarray  # H×W×3 BGR uint8
MaskArray = np.ndarray   # H×W uint8
ImagePath = Union[str, Path]
MaskPath = Union[str, Path]
Subregion = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

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
        # Select device with graceful fallback
        self.device = select_device(device)
        
        self.dip = DeepImagePrior(
            device=self.device,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            known_region_weight=known_region_weight
        )
        logger.info("Initialized ImagePatcher with Deep Image Prior")
        
        # LaMa will be loaded lazily when first requested
        self._lama: Optional["LamaInpainter"] = None  # type: ignore[misc]
    
    def patch_image(
        self,
        image: ImagePath | ImageArray,
        mask: MaskPath | MaskArray,
        output_path: Optional[ImagePath] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        blend: bool = True,
        dilate_percent: float = 0.05,
        feather_radius: int = 20,
        method: str = "lama",
        subregion: Optional[Subregion] = None
    ) -> ImageArray:
        """Patch an image using its mask.
        
        Args:
            image: Input image path or H×W×3 BGR uint8 numpy array
            mask: Binary mask path or H×W uint8 numpy array (255 = region to inpaint)
            output_path: Optional path to save the result
            progress_callback: Optional callback function to report progress
            blend: Whether to apply Poisson blending
            dilate_percent: Percentage to dilate the mask
            feather_radius: Width of edge ramp (pixels) for feathering edges
            method: 'lama' (default), 'telea', or 'dip' for different inpainting backends
            subregion: Optional tuple (x1, y1, x2, y2) specifying the subregion to inpaint
        
        Returns:
            Patched image as H×W×3 BGR uint8 numpy array
        
        Raises:
            ValueError: If image or mask is invalid
            RuntimeError: If inpainting fails
        """
        # ------------------------------------------------------------------
        # Input sanitisation ------------------------------------------------
        # ------------------------------------------------------------------

        # Load image from path if necessary
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

        # Load mask from path if necessary (always in grayscale)
        if isinstance(mask, (str, Path)):
            mask_path = Path(mask)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask from {mask_path}")

        # Basic validation ----------------------------------------------------------------
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy ndarray, got: " + str(type(image)))
        if mask is None:
            raise ValueError("mask cannot be None")
        if not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy ndarray, got: " + str(type(mask)))

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be H×W×3 BGR array")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != image.shape[:2]:
            raise ValueError("mask and image must have the same height/width")

        logger.info(f"Processing mask (dilate: {dilate_percent}%)")
        
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
        
        # Detect text regions and calculate subregion if not provided
        if subregion is None:
            text_detector = TextDetector()
            _, detections = text_detector.detect(image)
            subregion = self.calculate_subregion(detections, image.shape, mask=mask)
            if subregion:
                logger.info(f"Calculated subregion coordinates: x1={subregion[0]}, y1={subregion[1]}, x2={subregion[2]}, y2={subregion[3]}")
            else:
                logger.info("No text detected - processing entire image")

        logger.info(f"Starting {method} inpainting...")
        
        if method == "lama":
            if LamaInpainter is None:
                raise RuntimeError("LaMa dependencies not installed. See lama_inpainter.py for instructions.")
            if self._lama is None:
                logger.info("Initializing LaMa model (this may take a moment on first run)...")
                # Use the same device selection as the ImagePatcher
                self._lama = LamaInpainter(device=self.device)
                logger.info("LaMa model loaded successfully")
            
            logger.info("Running LaMa inference...")
            logger.info(f"About to call LaMa with image type: {type(image)}, shape: {image.shape}")
            logger.info(f"About to call LaMa with mask type: {type(mask)}, shape: {mask.shape}")
            try:
                result = self._lama.inpaint(image, mask, subregion=subregion)
                logger.info("LaMa inference completed")
            except Exception as e:
                logger.error(f"LaMa inference failed: {e}")
                # Reset LaMa model on error to prevent corruption
                self._lama = None
                raise
        elif method == "telea":
            if TeleaInpainter is None:
                raise RuntimeError("Telea dependencies not installed. See telea_inpainter.py for instructions.")
            if not hasattr(self, '_telea') or self._telea is None:
                logger.info("Initializing Telea model (this may take a moment on first run)...")
                # Use the same device selection as the ImagePatcher
                self._telea = TeleaInpainter()
                logger.info("Telea inpainter ready (OpenCV)")
            
            logger.info("Running Telea inference...")
            try:
                result = self._telea.inpaint(image, mask, subregion=subregion)
                logger.info("Telea inference completed")
            except Exception as e:
                logger.error(f"Telea inference failed: {e}")
                self._telea = None
                raise
        else:
            logger.info("Using Deep Image Prior...")
            # Inpaint using Deep Image Prior
            result = self.dip.inpaint(image, mask, progress_callback, subregion=subregion)
        
        # Optional blending to smooth edges (edge ramp based on distance transform)
        if blend:
            logger.info("Applying edge blending...")
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
            logger.info("Edge blending completed")
        
        # Ensure output has the same spatial size as the input image
        if result.shape[:2] != image.shape[:2]:
            h0, w0 = image.shape[:2]
            result = result[:h0, :w0]
        
        # Save result if output path is provided
        if output_path:
            cv2.imwrite(str(output_path), result)
            logger.info("Saved patched image to %s", output_path)
        
        logger.info("Inpainting process completed successfully")
        return result

    def calculate_subregion(
        self,
        detections: List[Dict],
        image_shape: Tuple[int, int],
        mask: Optional[MaskArray] = None,
        scale_factor: int = 3,
        min_margin: int = 16
    ) -> Optional[Subregion]:
        """Calculate the subregion to process based on detections and mask.
        
        Args:
            detections: List of detection dictionaries from TextDetector
            image_shape: Shape of the image (height, width)
            mask: Optional binary mask array
            scale_factor: Factor to scale the detection region
            min_margin: Minimum margin around detections
            
        Returns:
            Optional tuple (x1, y1, x2, y2) defining the subregion to process.
            All dimensions will be multiples of 4 to ensure compatibility with
            inpainting models.
        """
        def round_to_multiple_of_4(x: int) -> int:
            """Round a number to the nearest multiple of 4."""
            return (x + 2) & ~3  # Round up to nearest multiple of 4
        
        # 1. Try detections (apply both scale_factor and min_margin)
        if detections:
            min_x = min(det['geometry'][:, 0].min() for det in detections)
            min_y = min(det['geometry'][:, 1].min() for det in detections)
            max_x = max(det['geometry'][:, 0].max() for det in detections)
            max_y = max(det['geometry'][:, 1].max() for det in detections)
            if max_x > min_x and max_y > min_y:
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                width = max_x - min_x
                height = max_y - min_y
                
                # Apply both scale_factor and min_margin constraints
                margin_x = max(int(width * (scale_factor - 1) / 2), min_margin)
                margin_y = max(int(height * (scale_factor - 1) / 2), min_margin)
                
                x1 = max(0, int(center_x - width / 2 - margin_x))
                y1 = max(0, int(center_y - height / 2 - margin_y))
                x2 = min(image_shape[1], int(center_x + width / 2 + margin_x))
                y2 = min(image_shape[0], int(center_y + height / 2 + margin_y))
                if x2 > x1 and y2 > y1:
                    # Round dimensions to multiples of 4
                    x1 = round_to_multiple_of_4(x1)
                    y1 = round_to_multiple_of_4(y1)
                    x2 = round_to_multiple_of_4(x2)
                    y2 = round_to_multiple_of_4(y2)
                    # Clamp to image bounds
                    x2 = min(x2, image_shape[1])
                    y2 = min(y2, image_shape[0])
                    return tuple(int(v) for v in (x1, y1, x2, y2))
        
        # 2. Use mask bounding box, and DILATE it
        if mask is not None and np.any(mask > 0):
            ys, xs = np.where(mask > 0)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            width = max_x - min_x
            height = max_y - min_y
            margin_x = max(int(width * (scale_factor - 1) / 2), min_margin)
            margin_y = max(int(height * (scale_factor - 1) / 2), min_margin)
            x1 = max(0, min_x - margin_x)
            y1 = max(0, min_y - margin_y)
            x2 = min(image_shape[1], max_x + margin_x)
            y2 = min(image_shape[0], max_y + margin_y)
            if x2 > x1 and y2 > y1:
                # Round dimensions to multiples of 4
                x1 = round_to_multiple_of_4(x1)
                y1 = round_to_multiple_of_4(y1)
                x2 = round_to_multiple_of_4(x2)
                y2 = round_to_multiple_of_4(y2)
                # Clamp to image bounds
                x2 = min(x2, image_shape[1])
                y2 = min(y2, image_shape[0])
                return tuple(int(v) for v in (x1, y1, x2, y2))
        
        # 3. Fallback: whole image, rounded to multiples of 4
        x1 = 0
        y1 = 0
        x2 = round_to_multiple_of_4(image_shape[1])
        y2 = round_to_multiple_of_4(image_shape[0])
        # Clamp to image bounds
        x2 = min(x2, image_shape[1])
        y2 = min(y2, image_shape[0])
        return tuple(int(v) for v in (x1, y1, x2, y2))

    def patch_images(
        self,
        image_mask_pairs: Dict[ImagePath, MaskPath],
        output_dir: Optional[ImagePath] = None
    ) -> Dict[Path, Path]:
        """Patch multiple images using their corresponding masks.
        
        Args:
            image_mask_pairs: Dictionary mapping image paths to their mask paths
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
                # Load image and mask
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not load mask from {mask_path}")
                
                if output_dir is not None:
                    output_path = output_dir / f"{Path(image_path).stem}_patched.jpg"
                else:
                    output_path = None
                    
                patched_path = self.patch_image(image, mask, output_path)
                # Always return a Path to the saved file when output_path is provided.
                if output_path is not None:
                    patched_paths[Path(image_path)] = Path(output_path)
                else:
                    # Fall back to in-memory result by saving a temporary file next to the image
                    tmp_out = Path(image_path).with_name(f"{Path(image_path).stem}_patched.jpg")
                    cv2.imwrite(str(tmp_out), patched_path)
                    patched_paths[Path(image_path)] = tmp_out
                
            except Exception as e:
                logger.error("Failed to process image pair %s -> %s: %s",
                           image_path, mask_path, str(e))
                continue
        
        if not patched_paths:
            raise RuntimeError("No images were successfully patched")
            
        return patched_paths 