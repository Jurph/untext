# untext

A tool for removing text-based watermarks from images using OCR, masking, and inpainting models.

Features:
* Fast text detection using DocTR
* High-quality inpainting using LaMa (default)
* Alternative inpainting backends:
  * TELEA for high-quality results on complex textures
  * Deep Image Prior (DIP) for cases where neural approaches struggle
* Automatic subregion detection and processing

**untext** looks at an image for text and locates it within the image using DocTR's detection rules. In `--letters` mode (the default) we use DocTR's OCR engine to recognize individual letters and generate a mask that silghtly dilates the letters' shapes. In `--box` mode we avoid the OCR step and simply mask a large region covered by the text bounding box. The masked region(s) are cropped to a manageable pixel size before inpainting so we're only computing over the pixels that contribute context to the inpainted area, about 2x the size of the masked region. Then inpainting defaults to `--method LaMa` but can also use `--method TELEA` or, if you really need it, Deep Image Prior (but this is slow and you should ensure you have the mask perfect first!). 


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



