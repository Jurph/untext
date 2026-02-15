# untextre

A tool for removing watermarks from images using consensus detection, Figure of Merit color analysis, and state-of-the-art inpainting.

## Key Features

* **Consensus Detection**: Combines three text detection methods (EAST, DocTR, EasyOCR) to find regions where multiple detectors agree, ensuring high-confidence text detection
* **Figure of Merit (FOM) Analysis**: Identifies text-like color clusters using a weighted combination of TF-IDF distinctiveness, border underrepresentation, and connected-component fragmentation
* **High-Quality Inpainting**: LaMa (default) or TELEA inpainting with optimized region processing

## How It Works

**untextre** uses the following approach: 

1. **Consensus Detection**: Runs EAST, DocTR, and EasyOCR detectors simultaneously to find text regions where 2+ detectors agree
2. **Color Clustering**: For each consensus region, clusters all colors (inside and surrounding) using K-means
3. **Figure of Merit Scoring**: Evaluates each cluster with a weighted FOM combining TF-IDF score (color distinctiveness vs. background), border ratio (text underrepresented at bbox edges), and connected-component fraction (text is fragmented, not one solid blob)
4. **Adaptive Masking**: Accepts clusters whose FOM exceeds a threshold and whose largest connected component is below a guard value, then applies morphological cleanup
5. **Regional Processing**: Each consensus region gets its own color analysis, allowing different text colors in different areas
6. **Smart Inpainting**: Combines regional masks and applies LaMa or TELEA inpainting for seamless text removal 

## Installation

### Prerequisites

- **Python 3.10+**
- **GPU (recommended)**: NVIDIA GPU with [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) installed. Verify with `nvidia-smi`. CPU-only works but LaMa inpainting will be slow.

### Step 1: Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate.bat
# Linux / macOS
source venv/bin/activate
```

### Step 2: Install PyTorch (with CUDA support)

Install PyTorch **before** the other dependencies. The `requirements.txt` intentionally excludes torch because `pip install torch` from PyPI pulls a CPU-only build that will silently overwrite a CUDA-enabled installation.

Visit the [PyTorch "Get Started" page](https://pytorch.org/get-started/locally/) and select your OS, package manager, Python version, and CUDA version. Then run the generated command:

```bash
# Example for Windows/Linux with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Example for CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install remaining dependencies

```bash
# Core dependencies (detection, masking, inpainting)
pip install -r requirements.txt

# Optional: web interface (Streamlit)
pip install -r requirements_streamlit.txt

# Optional: development tools (pytest, black, flake8, mypy)
pip install -r requirements_dev.txt
```

## Usage

**untextre** provides two interfaces for removing text watermarks:

### Web Interface (Recommended for Beginners)

The easiest way to use **untextre** is through the web interface - just drag and drop images in your browser!

**Quick Start:**
```bash
# Install web interface dependencies
pip install -r requirements_streamlit.txt

# Launch the web interface
python run_web_interface.py
```

The interface will open automatically at `http://localhost:8501`

**Web Interface Options:**
- **Confidence Threshold** (0.1-0.9): Lower = detect more text, higher = more conservative
- **Color Granularity** (3-20): Number of color clusters for text detection. Default of 4 works well for most images. Increase for low-contrast watermarks where 4 is too aggressive.
- **Inpainting Method**: LaMa (high quality) or TELEA (fast) 
- **Color Sensitivity** (0-32): Tolerance around a user-specified target color. Only relevant when a target color is provided.
- **Show Masks**: Display detected text regions for debugging

### Command Line Interface

For batch processing, automation, or advanced control, use the command line:

**Basic Syntax:**
```bash
python -m untextre.cli -i <input> -o <output> [options]
```

**CLI Quick Start:**

**Single image:**
```bash
python -m untextre.cli -i image.jpg -o cleaned_image.jpg
```

**Batch processing:**
```bash
python -m untextre.cli -i input_folder/ -o output_folder/
```

**With debug info:**
```bash
python -m untextre.cli -i image.jpg -o results/ --keep-masks --verbose --timing
```

**Command-Line Options:**

#### Required Arguments

* `-i`, `--input` - Input image file or directory containing images
* `-o`, `--output` - Output file (for single image) or directory (for batch processing)

#### Detection & Analysis

* `--confidence-threshold FLOAT` - Confidence threshold for consensus detection (default: 0.3)
  - Lower values (0.1-0.2): More sensitive, may include false positives
  - Higher values (0.4-0.6): More conservative, may miss faint text

* `-g`, `--granularity K` - Override TF-IDF cluster count (e.g. 4, 8). If set, uses this K only (no g=8 retry). Default: auto g=4 with retry at g=8.

* `--no-expand` - Disable automatic bbox expansion along long axis
  - By default, detected bboxes are expanded to catch text that detectors may have missed
  - Use this flag if expansion is capturing too much background

* `--no-retry` - Disable automatic retry with granularity=8
  - By default, CLI uses g=4 and auto-retries with g=8 if text remnants are detected
  - Use this flag to skip the retry pass for faster processing

#### Input/Output Control

* `-K`, `--known-mask PATH` - Path to RGBA PNG of a known watermark/logo template
  - Uses ORB feature matching to find the template at any scale/position
  - The alpha channel defines which pixels to mask
  - **Skips consensus detection** - about 10x faster for consistent watermarks
  - Example: `--known-mask logo_template.png`

* `-c`, `--color COLOR` - Target text color as hex (#FF0000) or HTML name (red). Text colors are normally detected automatically via FOM analysis. This flag triggers immediate color enhancement if consensus detection finds no regions. Use for subtle watermarks that standard detection misses.

* `-m`, `--maskfile PATH` - Use existing mask file instead of generating one

* `-f`, `--force-bbox X,Y,W,H` - Force specific bounding box (x,y,width,height) where x,y is the top-left corner
  - Example: `--force-bbox 100,200,300,50` for 300x50 region at position (100,200)

#### Inpainting Options

* `-p`, `--paint METHOD` - Inpainting method (default: lama)
  - `lama`: High-quality deep learning inpainting (recommended)
  - `telea`: Fast OpenCV inpainting method

* `--device cuda|cpu` - Device for LaMa inference (default: cuda)

#### Debug & Monitoring

* `-k`, `--keep-masks` - Save debug masks alongside output images

* `-t`, `--timing` - Generate detailed timing reports

* `-l`, `--logfile PATH` - Save detailed logs to file

* `-v`, `--verbose` - Enable verbose console output

**CLI Advanced Examples:**

**High-precision processing:**
```bash
python -m untextre.cli -i photos/ -o cleaned/ --confidence-threshold 0.4
```

**Fast processing (skip retry pass):**
```bash
python -m untextre.cli -i images/ -o results/ --no-retry --paint telea
```

**Debug and analyze performance:**
```bash
python -m untextre.cli -i test.jpg -o debug/ --keep-masks --timing --verbose --logfile process.log
```

**Force specific region:**
```bash
python -m untextre.cli -i logo.png -o clean.png --force-bbox 50,100,200,30
```

**Remove known watermark using template matching:**
```bash
python -m untextre.cli -i photos/ -o cleaned/ -K watermark_template.png
```

**Override granularity:**
```bash
python -m untextre.cli -i photos/ -o cleaned/ -g 8
```

## Performance Tips

* **First run**: Model loading takes 10-15 seconds, then processing is fast
* **Batch processing**: Models loaded once, subsequent images process in 2-5 seconds. VRAM is automatically cleaned between images.
* **Granularity**: CLI uses g=4 by default with auto-retry at g=8 if remnants detected. Web UI lets you choose (3-20).
* **Confidence**: Start with 0.3, increase to 0.4-0.5 if too many false positives
* **Speed vs Quality**: Use `--no-retry` for faster processing, or `--paint telea` for CPU-only inpainting

## Output Files

When processing images, **untextre** generates several files:

### Cleaned Image  
* `image_clean.jpg` - Main result with text removed (high-quality JPEG at 95% quality)

### Image Mask File (with `--keep-masks`)
* `image_mask.png` - Binary mask showing detected text regions (white = text, black = background)

### Timing Reports (with `--timing`)
* `timing_report.txt` - Detailed performance metrics including:
  - Per-image processing times (detection, TF-IDF, masking, inpainting)
  - Consensus region counts and success rates
  - Average times and statistics for batch processing

## Troubleshooting

### No Consensus Regions Detected
If no text regions are found, try:
* Lower confidence threshold: `--confidence-threshold 0.025` improves the odds that two detectors will guess the same area 
* Use forced bounding box: `--force-bbox x,y,width,height`

### Poor Color Detection
If wrong colors are being masked:
* **CLI**: The auto-retry with g=8 usually handles this. If still problematic, the region may need manual adjustment in the web UI.
* **Web UI**: Adjust the granularity slider (3-20). Lower values (3-6) are more aggressive, higher values (12-20) are more selective. 

### Debugging
* The `-v / --verbose` option will show what's happening at each step 
* The `-k / --keep-masks` option will let you see what the model ended up removing  
* The `-l / --logfile {filename}` option stores logs in a file of your choice 


## Technical Details

### Consensus Detection
The system runs three different text detection algorithms:
- **EAST**: Fast OpenCV-based detection
- **DocTR**: Deep learning document text recognition  
- **EasyOCR**: OCR-based text detection

Regions where 2 or more detectors agree (with configurable overlap threshold) become "consensus regions" - areas of high confidence for containing text.

### Figure of Merit (FOM) Analysis
This is the key innovation of **untextre**. In a region where text is known to exist, we identify the text color by scoring each color cluster on multiple axes:

1. Generate a surrounding region outside the detection bbox, with roughly the same pixel count as the detection region  
2. Cluster all colors in both regions using K-means (CLI: g=4 with auto-retry at g=8; Web UI: user-configurable 3-20)
3. For each cluster, compute three signals:
   - **TF-IDF score**: How distinctive is this color to the text region vs. background?
   - **Border ratio**: Is this color underrepresented at the edges of the bbox? (Text tends to sit in the interior.)
   - **Connected-component fraction**: Is this color fragmented into many small shapes? (Text strokes are fragmented; solid backgrounds are not.)
4. Combine these into a weighted Figure of Merit: `FOM = 0.07 * tf_idf + 0.63 * border + 0.30 * cc`
5. Accept clusters with FOM >= 0.30 and largest connected component < 85% of the cluster area
6. The accepted clusters form the binary mask, cleaned up with morphological operations (closing, dilation, blur)

The weights were determined empirically through systematic experimentation on a variety of watermarked images.

### License

MIT
