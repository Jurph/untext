# untextre

A tool for removing text watermarks from images using consensus detection, spatial TF-IDF color analysis, and state-of-the-art inpainting.

## Key Features

* **Consensus Detection**: Combines three text detection methods (EAST, DocTR, EasyOCR) to find regions where multiple detectors agree, ensuring high-confidence text detection
* **Spatial TF-IDF Analysis**: Finds the most text-like color family in each consensus region and masks it out 
* **High-Quality Inpainting**: LaMa (default) or TELEA inpainting with optimized region processing

## How It Works

**untextre** uses the following approach: 

1. **Consensus Detection**: Runs EAST, DocTR, and EasyOCR detectors simultaneously to find text regions where 2+ detectors agree
2. **Spatial TF-IDF**: For each consensus region, clusters colors and calculates a pseudo-TF-IDF scores to identify colors distinctive to text vs. background  
3. **Adaptive Masking**: Creates grayscale "text-likelihood" maps from TF-IDF scores then thresholds the text from the background 
4. **Regional Processing**: Each consensus region gets its own color analysis, allowing different text colors in different areas
5. **Smart Inpainting**: Combines regional masks and applies LaMa or TELEA inpainting for seamless text removal 

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

**untextre** provides two interfaces for removing text watermarks:

### 🖥️ Web Interface (Recommended for Beginners)

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
- **Color Granularity** (8-48): Number of color clusters for text detection  
- **Inpainting Method**: LaMa (high quality) or TELEA (fast)
- **Show Masks**: Display detected text regions for debugging

### 💻 Command Line Interface

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

* `--granularity INT` - Number of color clusters for spatial TF-IDF analysis (default: 24)
  - Lower values (6-12): Faster processing, broader color groupings
  - Higher values (32-48): More precise color separation, slower processing

#### Input/Output Control

* `-c`, `--color COLOR` - Target text color as hex (#FF0000) or HTML name (red) *(deprecated - TF-IDF finds colors automatically)*

* `-m`, `--maskfile PATH` - Use existing mask file instead of generating one

* `-f`, `--force-bbox X,Y,W,H` - Force specific bounding box (x,y,width,height) instead of detection
  - Example: `--force-bbox 100,200,300,50` for 300×50 region at position (100,200)

#### Inpainting Options

* `-p`, `--paint METHOD` - Inpainting method (default: lama)
  - `lama`: High-quality deep learning inpainting (recommended)
  - `telea`: Fast OpenCV inpainting method

#### Debug & Monitoring

* `-k`, `--keep-masks` - Save debug masks alongside output images

* `-t`, `--timing` - Generate detailed timing reports

* `-l`, `--logfile PATH` - Save detailed logs to file

* `-v`, `--verbose` - Enable verbose console output

**CLI Advanced Examples:**

**High-precision processing:**
```bash
python -m untextre.cli -i photos/ -o cleaned/ --confidence-threshold 0.4 --granularity 32
```

**Fast processing for simple text:**
```bash
python -m untextre.cli -i images/ -o results/ --granularity 12 --paint telea
```

**Debug and analyze performance:**
```bash
python -m untextre.cli -i test.jpg -o debug/ --keep-masks --timing --verbose --logfile process.log
```

**Force specific region:**
```bash
python -m untextre.cli -i logo.png -o clean.png --force-bbox 50,100,200,30
```

## Performance Tips

* **First run**: Model loading takes 10-15 seconds, then processing is fast
* **Batch processing**: Models loaded once, subsequent images process in 2-5 seconds
* **Granularity**: Start with default (24), reduce to 12-16 for speed, increase to 32-48 for accuracy
* **Confidence**: Start with 0.3, increase to 0.4-0.5 if too many false positives

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
If wrong colors are being masked, adjust granularity: Settings as coarse as `--granularity 4` work well on smaller images; values higher than `--granularity 32` almost always slice the color spectrum too thinly and leave important colors out. 

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

### Spatial TF-IDF Analysis
This is the key innovation of **untextre**. In a region where text is known to exist, we identify the text color by finding the family of colors that are most distinctive to the text region. 
1. Generate a bounding or outer region outside the detection region, with the same pixel count as the detection region  
2. Cluster all colors in both regions using K-means (with K = `--granularity`, anywhere from 4 - 256)
3. Calculate scores for each color cluster using an approach like TF-IDF (where the detection region is the "document" and the outer region is the "corpus")
4. Normalize those scores to a 0-255 scale 
5. Empirical investigation revealed that text tends to have a TF-IDF score of >192 

This approach automatically identifies colors that are distinctive to text regions vs. background, handling complex scenarios like multi-colored text, varying backgrounds, and edge anti-aliasing.

### License

MIT



