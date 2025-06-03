# untext

A tool for removing text-based watermarks from images using state-of-the-art inpainting models.

Features:
* Fast text detection using DocTR
* High-quality inpainting using LaMa (default)
* Alternative inpainting backends:
  * TALEA for high-quality results on complex textures
  * Deep Image Prior (DIP) for cases where neural approaches struggle
* Automatic subregion detection and processing
* Edge blending for seamless results

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended), also install PyTorch with CUDA support:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage

Basic usage:

```bash
python -m untext.cli -i image.jpg -o output.jpg
```

Advanced options:

```bash
python -m untext.cli -i input_dir/ -o output_dir/ --method lama --save-masks
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



