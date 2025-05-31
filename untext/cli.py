"""Command-line interface for untext."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from untext.detector import TextDetector
from untext.inpainter import DeepImagePriorInpainter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove text-based watermarks from images using Deep Image Prior"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output image path"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for text detection (0-1)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of optimization iterations for inpainting"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print("Loading image...")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Could not load image '{args.input}'", file=sys.stderr)
        sys.exit(1)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize detector and inpainter
    print("Initializing models...")
    detector = TextDetector(confidence_threshold=args.confidence)
    inpainter = DeepImagePriorInpainter(
        num_iterations=args.iterations,
        device=args.device
    )
    
    # Detect text
    print("Detecting text...")
    mask, detections = detector.detect(str(input_path))
    
    if not detections:
        print("No text detected in image")
        sys.exit(0)
    
    print(f"Found {len(detections)} text regions")
    
    # Inpaint
    print("Inpainting...")
    result, losses = inpainter.inpaint(image, mask)
    
    # Convert RGB back to BGR
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Save result
    print(f"Saving result to '{args.output}'...")
    cv2.imwrite(str(output_path), result)
    
    print("Done!")


if __name__ == "__main__":
    main() 