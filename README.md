# untextre

A tool for removing text watermarks from images using consensus detection, spatial TF-IDF color analysis, and state-of-the-art inpainting.

## Key Features

* **Consensus Detection**: Combines three text detection methods (EAST, DocTR, EasyOCR) to find regions where multiple detectors agree, ensuring high-confidence text detection
* **Spatial TF-IDF Analysis**: Revolutionary color detection that treats text regions as "documents" and surrounding areas as "corpus" to identify colors distinctive to text
* **Adaptive Thresholding**: Uses Otsu thresholding on TF-IDF likelihood maps for automatic binary mask generation
* **High-Quality Inpainting**: LaMa (default) or TELEA inpainting with optimized region processing
* **Performance Optimized**: Models loaded once and reused across all images for fast batch processing
* **Configurable Granularity**: Adjustable color clustering (4-48 clusters) for speed vs. accuracy trade-offs

## How It Works

**untextre** uses the following approach: 

1. **Consensus Detection**: Runs EAST, DocTR, and EasyOCR detectors simultaneously to find text regions where 2+ detectors agree
2. **Spatial TF-IDF**: For each consensus region, clusters colors and calculates a pseudo-TF-IDF scores to identify colors distinctive to text vs. background  
3. **Adaptive Masking**: Creates grayscale "text-likelihood" maps from TF-IDF scores, then uses Otsu thresholding for optimal binary masks
4. **Regional Processing**: Each consensus region gets its own color analysis, allowing different text colors in different areas
5. **Smart Inpainting**: Combines regional masks and applies LaMa inpainting for seamless text removal 

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

### Basic Syntax

```bash
python -m untextre.cli -i <input> -o <output> [options]
```

### Quick Start

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

### Command-Line Options

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

### Advanced Examples

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

### Performance Tips

* **First run**: Model loading takes 10-15 seconds, then processing is fast
* **Batch processing**: Models loaded once, subsequent images process in 2-5 seconds
* **Granularity**: Start with default (24), reduce to 12-16 for speed, increase to 32-48 for accuracy
* **Confidence**: Start with 0.3, increase to 0.4-0.5 if too many false positives

## Output Files

When processing images, **untextre** generates several files:

### Primary Output
* `image_clean.jpg` - Main result with text removed (high-quality JPEG at 95% quality)

### Debug Files (with `--keep-masks`)
* `image_mask.png` - Binary mask showing detected text regions (white = text, black = background)

### Timing Reports (with `--timing`)
* `timing_report.txt` - Detailed performance metrics including:
  - Per-image processing times (detection, TF-IDF, masking, inpainting)
  - Consensus region counts and success rates
  - Average times and statistics for batch processing

### Logs (with `--logfile`)
* Custom log file with detailed processing information and any errors

## Troubleshooting

### No Consensus Regions Detected
If no text regions are found, try:
* Lower confidence threshold: `--confidence-threshold 0.2`
* Use forced bounding box: `--force-bbox x,y,width,height`
* Check that text is clearly visible and not too small/faint

### Poor Color Detection
If wrong colors are being masked:
* Increase granularity: `--granularity 32` or `--granularity 48`
* Use debug mode to analyze: `--keep-masks --verbose`

### Slow Performance
For faster processing:
* Reduce granularity: `--granularity 12` or `--granularity 16`
* Use TELEA inpainting: `--paint telea`

### Memory Issues
For large images or limited memory:
* Reduce granularity: `--granularity 8`
* Process images individually rather than in batches

## Technical Details

### Consensus Detection
The system runs three different text detection algorithms:
- **EAST**: Fast OpenCV-based detection
- **DocTR**: Deep learning document text recognition  
- **EasyOCR**: OCR-based text detection

Regions where 2 or more detectors agree (with configurable overlap threshold) become "consensus regions" - areas of high confidence for containing text.

### Spatial TF-IDF Analysis
This is the key innovation of **untextre**. Traditional approaches try to find text colors globally, but this fails when:
- Text colors appear in the background  
- Different regions have different text colors
- Background colors are similar to text colors

Instead, **untextre** treats each text region as a "document" and its surrounding area as the "corpus", then:
1. Clusters all colors in both regions using K-means
2. Calculates TF-IDF scores for each color cluster
3. Creates a spatial map where pixel intensity = text likelihood
4. Uses Otsu thresholding to automatically find the optimal binary threshold

This approach automatically identifies colors that are distinctive to text regions vs. background, handling complex scenarios like multi-colored text, varying backgrounds, and edge anti-aliasing.

### Performance Optimizations
- **Model Caching**: All three detection models loaded once at startup, reused for all images
- **Regional Processing**: Only analyzes colors within consensus regions, not entire images
- **Configurable Granularity**: Trade speed vs. accuracy with 6-48 color clusters
- **Memory Efficiency**: Processes regions independently to minimize memory usage

## License

MIT



