"""Command-line interface for untext."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs

import argparse
import sys
import logging
import signal
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from tempfile import NamedTemporaryFile, TemporaryDirectory
import torch
from typing import Optional

from untext.image_patcher import ImagePatcher
from untext.word_mask_generator import WordMaskGenerator
from untext.detector import TextDetector

# Set up logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from word_mask_generator
logging.getLogger('untext.word_mask_generator').setLevel(logging.WARNING)
logging.getLogger('untext.image_patcher').setLevel(logging.WARNING)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Operation timed out")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove text-based watermarks from images using inpainting"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output image path or directory"
    )
    parser.add_argument(
        "--mask",
        type=str,
        choices=["box", "letters"],
        default="box",
        help="Mask geometry: 'box' (default) or 'letters' for glyph-level masks",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save mask files alongside output images (useful for debugging)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dip", "lama", "telea"],
        default="lama",
        help="Inpainting backend to use (default: lama)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of optimization iterations for DIP (ignored for LaMa)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on: 'cuda' (default, falls back to CPU if unavailable) or 'cpu'"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have output files"
    )
    parser.add_argument(
        "--mask-file",
        type=str,
        default=None,
        help="Path to a mask file to use instead of generating one"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=True,
        help="Apply optimized preprocessing before text detection (enabled by default)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing (for testing or comparison)"
    )
    return parser.parse_args()


def get_image_files(path: Path) -> list[Path]:
    """Get all image files from a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        else:
            logger.warning(f"File {path} doesn't have a supported image extension")
            return []
    elif path.is_dir():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(path.glob(f'*{ext}'))
            files.extend(path.glob(f'*{ext.upper()}'))
        return sorted(set(files))  # Remove duplicates and sort
    else:
        return []


def process_single_image(
    image_path: Path,
    output_path: Path,
    mask_gen: WordMaskGenerator,
    patcher: ImagePatcher,
    save_masks: bool,
    method: str,
    verbose: bool,
    mask_file: Optional[str] = None
) -> bool:
    """Process a single image. Returns True if successful, False otherwise."""
    start_time = time.time()
    
    try:
        logger.info(f"Loading image: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False
        
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            print(f"Image is not a numpy array: {type(image)}")
            return False
        
        logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        logger.info(f"Image type: {type(image)}, dtype: {image.dtype}")
        
        # If a mask file was provided, load it; otherwise generate the mask
        if mask_file is not None:
            mask_file_path = Path(mask_file)
            if not mask_file_path.exists():
                logger.warning(f"Mask file {mask_file} does not exist.")
                return False
            mask = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.error(f"Failed to load mask file: {mask_file}")
                return False
            logger.info(f"Using provided mask file: {mask_file_path.name}")
        else:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy image to temp directory to avoid polluting source
                temp_image = temp_path / image_path.name
                cv2.imwrite(str(temp_image), image)
                
                logger.info(f"Detecting text in {image_path.name}...")
                
                # Generate mask with timeout
                mask_start = time.time()
                mask_map = mask_gen.generate_masks([str(temp_image)])
                mask_time = time.time() - mask_start
                logger.info(f"Mask generation took {mask_time:.2f}s")
                
                if not mask_map or temp_image not in mask_map:
                    logger.info(f"No text detected in {image_path.name}")
                    return False
                
                mask_file_generated = mask_map[temp_image]
                logger.info(f"Text detected, generated mask: {mask_file_generated.name}")
                mask = cv2.imread(str(mask_file_generated), cv2.IMREAD_GRAYSCALE)
        
        logger.info(f"Inpainting with {method}...")
        
        # Calculate subregion for inpainting
        if mask_file is not None:
            # Use the provided mask to calculate the subregion
            subregion = patcher.calculate_subregion([], image.shape, mask=mask)
            if subregion:
                logger.info(f"CLI: Cropping to subregion (from provided mask): {subregion}")
            else:
                logger.info("CLI: No subregion found from provided mask, using whole image")
        else:
            # Use text detection results to calculate the subregion
            text_detector = TextDetector()
            detections = text_detector.detect(image)[1]
            subregion = patcher.calculate_subregion(detections, image.shape, mask=mask)
            if subregion:
                logger.info(f"CLI: Cropping to subregion: {subregion}")
            else:
                logger.info("CLI: No subregion found, using whole image")
        
        # Inpaint with timeout monitoring
        inpaint_start = time.time()
        result = patcher.patch_image(
            image,  # Pass the image array instead of path
            mask,  # Use the loaded mask
            method=method, 
            output_path=None, 
            blend=True,
            subregion=subregion
        )
        inpaint_time = time.time() - inpaint_start
        logger.info(f"Inpainting took {inpaint_time:.2f}s")
        
        # Save result
        cv2.imwrite(str(output_path), result)
        
        total_time = time.time() - start_time
        logger.info(f"✓ Processed {image_path.name} → {output_path.name} (total: {total_time:.2f}s)")
        
        # Optionally save mask
        if save_masks:
            mask_output = output_path.with_name(f"{output_path.stem}_mask.png")
            cv2.imwrite(str(mask_output), mask)
            logger.info(f"  Saved mask to {mask_output.name}")
        
        # Force cleanup
        del image, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Failed to process {image_path} after {total_time:.2f}s: {e}")
        # Force cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level - always show INFO level for batch processing visibility
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('untext.word_mask_generator').setLevel(logging.INFO)
        logging.getLogger('untext.image_patcher').setLevel(logging.INFO)
        logging.getLogger('untext.lama_inpainter').setLevel(logging.INFO)
    else:
        # Even in non-verbose mode, show progress for batch processing
        logger.setLevel(logging.INFO)
        logging.getLogger('untext.lama_inpainter').setLevel(logging.INFO)
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Get list of images to process
    image_files = get_image_files(input_path)
    if not image_files:
        logger.error(f"No valid image files found in '{args.input}'")
        sys.exit(1)
    
    # Determine output setup
    output_path = Path(args.output)
    single_file_mode = len(image_files) == 1 and input_path.is_file()
    
    if single_file_mode:
        # Single file mode: output can be a file
        if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
            output_path = output_path.with_suffix('.jpg')
        output_files = [(image_files[0], output_path)]
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Directory mode: output must be a directory
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = [
            (img, output_path / f"{img.stem}_cleaned{img.suffix}")
            for img in image_files
        ]
    
    logger.info(f"Found {len(image_files)} image(s) to process")
    
    # Initialize models
    logger.info("Initializing models...")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info("Loading ImagePatcher...")
        patcher = ImagePatcher(device=device, num_iterations=args.iterations)
        logger.info("Loading WordMaskGenerator...")
        
        # Handle preprocessing flags
        use_preprocess = args.preprocess and not args.no_preprocess
        mask_gen = WordMaskGenerator(mode=args.mask, preprocess=use_preprocess)
        logger.info(f"Preprocessing {'enabled' if use_preprocess else 'disabled'}")
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        sys.exit(1)
    
    # Process images with progress bar
    success_count = 0
    is_batch_mode = len(output_files) > 1
    
    # Use TQDM only for batch mode and when not verbose
    use_tqdm = is_batch_mode and not args.verbose
    
    if use_tqdm:
        output_files_iter = tqdm(output_files, desc="Processing images")
    else:
        output_files_iter = output_files
        if is_batch_mode:
            logger.info(f"Starting batch processing of {len(output_files)} images...")
    
    for i, (input_file, output_file) in enumerate(output_files_iter, 1):
        if is_batch_mode and not use_tqdm:
            logger.info(f"\n[{i}/{len(output_files)}] Processing {input_file.name}")
        
        # Skip if output already exists
        if args.skip_existing and output_file.exists():
            logger.info(f"Skipping {input_file.name} (output already exists)")
            success_count += 1  # Count as success since output exists
            continue
        
        # Add watchdog for hanging detection
        start_time = time.time()
        last_log_time = start_time
        
        success = process_single_image(
            input_file,
            output_file,
            mask_gen,
            patcher,
            args.save_masks,
            args.method,
            args.verbose,
            args.mask_file
        )
        
        elapsed = time.time() - start_time
        if elapsed > 120:  # Warn if any single image takes over 2 minutes
            logger.warning(f"Image {input_file.name} took {elapsed:.1f}s to process")
        
        if success:
            success_count += 1
        
        # Update progress bar if using TQDM
        if use_tqdm:
            output_files_iter.set_postfix({
                'processed': success_count,
                'skipped': i - success_count,
                'last_time': f"{elapsed:.1f}s"
            })
    
    # Summary
    logger.info(f"\nCompleted: {success_count}/{len(image_files)} images processed successfully")
    if success_count < len(image_files):
        logger.info(f"Skipped {len(image_files) - success_count} images (no text detected)")


if __name__ == "__main__":
    main() 