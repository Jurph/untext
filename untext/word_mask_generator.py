"""Module for generating word masks using DocTR text detection.

This module provides functionality to detect text in images and generate binary masks
for the detected text regions using the DocTR library.

Example:
    >>> from untext.word_mask_generator import WordMaskGenerator
    >>> generator = WordMaskGenerator()
    >>> mask_paths = generator.generate_masks(['image1.jpg', 'image2.png'])
    >>> print(f"Generated masks: {mask_paths}")
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor
import argparse


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WordMaskGenerator:
    """Generate binary masks for text regions.

    Parameters
    ----------
    mode: str
        "box"   – fill rectangular bounding boxes (default, cheap)
        "letters" – recognise each word, render the glyphs, and mask pixels that overlap the rendered text.
    """

    def __init__(self, mode: str = "box") -> None:
        if mode not in {"box", "letters"}:
            raise ValueError("mask mode must be 'box' or 'letters'")

        self.mode = mode

        try:
            if mode == "box":
                # detection-only pipeline (fast)
                self.model = detection.db_resnet50(pretrained=True)
                self.pre_processor = PreProcessor(
                    output_size=(1024, 1024),
                    batch_size=1,
                    mean=(0.798, 0.785, 0.772),
                    std=(0.264, 0.2749, 0.287)
                )
                self.predictor = DetectionPredictor(
                    pre_processor=self.pre_processor,
                    model=self.model,
                )
            else:
                from doctr.models import ocr_predictor  # lazy import heavy modules

                self.predictor = ocr_predictor(pretrained=True,
                                               det_arch="db_resnet50",
                                               reco_arch="crnn_vgg16_bn")
            logger.info("WordMaskGenerator initialised in %s mode", mode)
        except Exception as e:
            logger.exception("Could not initialise DocTR OCR stack: %s", e)
            raise RuntimeError("Failed to initialise WordMaskGenerator") from e

    def generate_masks(
        self,
        image_paths: List[Union[str, Path]]
    ) -> Dict[Path, Path]:
        """Generate binary masks for detected text in images.
        
        Args:
            image_paths: List of paths to images to process. Images should be in a
                        format supported by OpenCV (e.g., PNG, JPEG).
        
        Returns:
            Dictionary mapping input image paths to their corresponding mask paths.
        
        Raises:
            ValueError: If any input path is invalid.
            FileNotFoundError: If any input image cannot be found.
            RuntimeError: If mask generation fails.
        """
        # Convert paths to Path objects
        image_paths = [Path(p) for p in image_paths]
        
        # Validate input files
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
        
        mask_paths = {}
        for image_path in image_paths:
            try:
                logger.info("Processing image: %s", image_path)
                
                # Load the image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                # Helper to get words according to mode
                def _extract_words(image_data):
                    if self.mode == "box":
                        raw = self.predictor([image_data])
                        if not raw:
                            return []
                        if 'words' not in raw[0]:
                            return []
                        return raw[0]['words']  # list of boxes
                    # letters mode – use OCR predictor
                    # Directly use image_data as input
                    pages_doc = self.predictor([image_data])
                    page = pages_doc.pages[0]
                    h_img, w_img = image_data.shape[:2]
                    return [w for b in page.blocks for ln in b.lines for w in ln.words]

                words = _extract_words(image)

                # If nothing found, try simple pre-processing variants
                if len(words) == 0:
                    aug_fns = [
                        lambda im: cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)),
                        lambda im: cv2.convertScaleAbs(im, alpha=1.5, beta=0),
                        lambda im: 255 - im,
                    ]

                    for fn in aug_fns:
                        aug = fn(image.copy())
                        if aug.ndim == 2:
                            aug = cv2.cvtColor(aug, cv2.COLOR_GRAY2BGR)
                        words = _extract_words(aug)
                        if len(words) > 0:
                            logger.info("Detection succeeded on augmented variant for %s", image_path)
                            break

                if len(words) == 0:
                    logger.warning("No text detected in %s after all variants", image_path)
                    continue

                # ensure words variable ready for downstream depending on mode
                first_result = {'words': words} if self.mode == "box" else None

                # Create a single mask for all words in the image
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                if self.mode == "box":
                    words = first_result['words']
                    for word in words:
                        if not isinstance(word, (list, np.ndarray)) or len(word) != 5:
                            continue
                        h_img, w_img = image.shape[:2]
                        x1, y1, x2, y2, _ = word
                        cv2.rectangle(
                            mask,
                            (int(x1 * w_img), int(y1 * h_img)),
                            (int(x2 * w_img), int(y2 * h_img)),
                            255,
                            -1,
                        )
                else:  # letters mode
                    # result is a doctr Document
                    page = self.predictor([image]).pages[0]
                    h_img, w_img = image.shape[:2]
                    words_iter = [w for b in page.blocks for ln in b.lines for w in ln.words]
                    for w in words_iter:
                        pts = np.asarray(w.geometry).reshape(-1, 2)
                        x1_i = int(pts[:, 0].min() * w_img)
                        y1_i = int(pts[:, 1].min() * h_img)
                        x2_i = int(pts[:, 0].max() * w_img)
                        y2_i = int(pts[:, 1].max() * h_img)

                        bbox_w = max(2, x2_i - x1_i)
                        bbox_h = max(2, y2_i - y1_i)

                        # Draw the recognised text into a tiny canvas same size as image
                        word_canvas = np.zeros_like(mask)
                        font_scale = bbox_h / 30.0  # heuristic: 30 px font height baseline
                        thickness = max(1, int(font_scale * 2))
                        cv2.putText(
                            word_canvas,
                            w.value,
                            (x1_i, y2_i),  # baseline at bottom-left
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            255,
                            thickness,
                            lineType=cv2.LINE_AA,
                        )
                        mask = cv2.bitwise_or(mask, word_canvas)

                # Save the mask
                mask_path = image_path.with_suffix('').with_name(
                    f"{image_path.stem}_mask.jpg"
                )
                cv2.imwrite(str(mask_path), mask)
                mask_paths[image_path] = mask_path
                logger.info("Generated mask for %s: %s", image_path, mask_path)

            except Exception as e:
                logger.error("Failed to process %s: %s", image_path, str(e))
                continue

        if not mask_paths:
            raise RuntimeError("No masks were generated for any images")
            
        return mask_paths 

def process_images_in_directory(
    input_dir: str,
    output_dir: str,
    generate_masks: bool = False
) -> None:
    """Process all images in a directory, optionally generating masks.

    Args:
        input_dir: Directory containing images to process.
        output_dir: Directory to save processed images.
        generate_masks: Whether to generate mask files (default: False).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in input_path.glob('*.*'):
        try:
            # Process each image
            print(f"Processing {image_file}")
            generator = WordMaskGenerator(mode="letters")
            mask_paths = generator.generate_masks([str(image_file)])

            # If masks were generated and generate_masks is True, save them
            if generate_masks and mask_paths:
                for original_path, mask_path in mask_paths.items():
                    mask_output_path = output_path / mask_path.name
                    os.rename(mask_path, mask_output_path)
                    print(f"Mask saved to {mask_output_path}")

            # Save the inpainted image if text was found
            if mask_paths:
                inpainted_image_path = output_path / f"{image_file.stem}_inpainted.jpg"
                # Assuming inpainting logic is implemented
                # inpaint_image(image_file, mask_paths[image_file], inpainted_image_path)
                print(f"Inpainted image saved to {inpainted_image_path}")

        except Exception as e:
            print(f"Failed to process {image_file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Batch process images in a directory.")
    parser.add_argument("input_dir", type=str, help="Directory containing images to process.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images.")
    parser.add_argument(
        "--generate-masks", action="store_true", help="Generate mask files (default: False)."
    )
    args = parser.parse_args()

    process_images_in_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        generate_masks=args.generate_masks
    )

if __name__ == "__main__":
    main() 