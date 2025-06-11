# untext

A tool for removing text-based watermarks from images using OCR, masking, and inpainting models.

Features:
* Multiple text detection methods:
  * **DocTR** (default): Deep learning-based detection with high accuracy
  * **EasyOCR**: Simple OCR-based detection  
  * **EAST**: Efficient OpenCV-based detector (fast, no PyTorch dependency)
* High-quality inpainting using LaMa (default) or TELEA 
* Automatic subregion detection and processing
* Batch processing 
* Timing measurements 

**untext** looks at an image for text and locates it within the image using the OCR engine's detection rules. We look for clusters of similar coloration within the detection region and decide which one is most likely to be text, then mask that set of colors throughout the neighborhood of the detection region. The white pixels in the binary mask are dilated, eroded, blurred, and more to ensure that noisy one-off pixels aren't inpainted and text-like contiguous shapes are. The masked region(s) are cropped to a manageable pixel size before inpainting so we're only computing over the pixels that contribute context to the inpainted area, about 2x the size of the masked region. Then we inpaint the original image 

## Installation

**Important: Install PyTorch First**

Before installing other dependencies, we strongly recommend installing PyTorch through their official ["Get Started" page](https://pytorch.org/get-started/locally/). PyTorch's installation tool automatically handles the complex deconfliction of CUDA versions, operating systems, and wheel files to ensure you get the right build for your system.

Visit https://pytorch.org/get-started/locally/ and select your configuration:
- Your OS (Linux, Mac, Windows)  
- Package manager (Pip, Conda, etc.)
- Python version
- CUDA version (or CPU-only)

Then run the generated command, for example:
```bash
# Example for Linux/Windows with CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CPU-only
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Then install remaining dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python -m untext.cli -i image.jpg -o output.jpg
```

Advanced options:

```bash
# Use EAST detector (faster, requires only OpenCV)
python -m untextre.cli -i input_dir/ -o output_dir/ --detector east

# Use EasyOCR with TELEA inpainting
python -m untextre.cli -i image.jpg -o output.jpg --detector easyocr --paint telea

# Full pipeline with debug masks
python -m untextre.cli -i input_dir/ -o output_dir/ --keep-masks --timing
```

### Options

* `-i`, `--input`: Input image or directory
* `-o`, `--output`: Output image or directory
* `--mask`: Mask geometry (`"box"` or `"letters"`, default: `"box"`)
* `--save-masks`: Save intermediate mask files (useful for debugging, default = `"False"`)
* `--method`: Inpainting backend to use (`"lama"`, `"talea"`, or `"dip"`)
* `--device`: Device to run on (`"cuda"` or `"cpu"`, default: `"cuda"`)
* `--verbose`: Enable verbose output
* `--skip-existing`: Skip images that already have output files

## API

```python
from untext import ImagePatcher

patcher = ImagePatcher(device='cuda')
result = patcher.patch_image('input.jpg', 'mask.jpg', 'output.jpg')
```

The main `patch_image()` method accepts these parameters:

* `image_path`: Path to input image
* `mask_path`: Path to binary mask (white = area to inpaint)
* `output_path`: Optional path to save result
* `method`: Inpainting backend (`"lama"`, `"talea"`, or `"dip"`)
* `blend`: Whether to apply edge blending (default: `True`)
* `dilate_percent`: How much to expand the mask (default: `0.05`)
* `feather_radius`: Width of edge feathering in pixels (default: `20`)

## License

MIT



