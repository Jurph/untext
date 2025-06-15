# UnTextre Web Interface

A user-friendly web interface for removing text watermarks from images using drag-and-drop functionality.

## Features

- 🖱️ **Drag & Drop**: Simply drag images into the browser
- 🔄 **Real-time Processing**: See results immediately 
- ⚙️ **Configurable Options**: Adjust detection sensitivity and processing method
- 📊 **Processing Stats**: View detailed timing and detection information
- 💾 **Easy Download**: One-click download of processed images
- 🎭 **Debug Mode**: Visualize detected text regions
- 📱 **Responsive Design**: Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Launch the Web Interface

**Option A: Using the startup script (recommended)**
```bash
python run_web_interface.py
```

**Option B: Direct streamlit command**
```bash
streamlit run streamlit_app.py
```

### 3. Open Your Browser

The interface will automatically open at `http://localhost:8501`

## Usage

1. **Upload Image**: Drag and drop an image or click to browse
2. **Adjust Settings** (optional): Use the sidebar to configure:
   - **Confidence Threshold**: Lower = detect more text (may include false positives)
   - **Color Granularity**: Number of color clusters for text detection
   - **Inpainting Method**: LaMa (high quality) or TELEA (fast)
   - **Show Masks**: Enable to see detected text regions
3. **Process**: Click "Remove Text Watermarks"
4. **Download**: Click "Download Result" to save the processed image

## Configuration Options

### Detection Settings

- **Confidence Threshold** (0.01-0.99, default: 0.025)
  - Lower values detect more text but may include false positives
  - Higher values are more conservative but may miss faint text
  - Surprisingly 2.5% confidence and 2-of-3 consensus is very effective! 
  

- **Color Granularity** (2-256, default: 24)
  - Number of color clusters for spatial TF-IDF analysis
  - 4-8 colors does coarse separation and can overcorrect 
  - Most numbers between 8-24 work very well 
  - Over 32 colors and the granularity is too fine for effective mask creation 

### Inpainting Methods

- **LaMa** (default): High-quality neural inpainting, slower but better results
- **TELEA**: Fast traditional inpainting, good for simple cases

### Debug Options

- **Show Detection Masks**: Displays the detected text regions as a binary mask

## Technical Details

### Model Loading

All AI models are loaded once at startup and cached in memory:
- **DocTR**: Document text recognition
- **EasyOCR**: Optical character recognition  
- **EAST**: Efficient and Accurate Scene Text detector
- **LaMa**: Large Mask Inpainting neural network

This ensures fast processing after the initial startup time.

### Processing Pipeline

1. **Consensus Detection**: Three detectors vote on text regions
2. **Rotation Failover**: If no consensus, try 90° rotation
3. **Watermark Regions**: If still no consensus, check common watermark locations
4. **Spatial TF-IDF**: Analyze color patterns to identify text pixels
5. **Morphological Cleanup**: Refine the detection mask
6. **Neural Inpainting**: Remove text and reconstruct background

### Performance

- **First Run**: 30-60 seconds (model loading)
- **Subsequent Images**: 2-10 seconds depending on image size and complexity
- **Memory Usage**: ~4-6GB GPU memory for optimal performance

## Troubleshooting

### Common Issues

**"Models not loaded" error**
- Ensure all dependencies are installed: `pip install -r requirements_streamlit.txt`
- Check that you have sufficient GPU memory (4GB+ recommended)
- Try CPU mode by editing `streamlit_app.py` and changing `device="cuda"` to `device="cpu"`

**Slow processing**
- First image always takes longer due to model initialization
- Large images (>4K) take more time to process
- Consider using TELEA method for faster processing

**Upload fails**
- Check file size (max 50MB)
- Ensure file format is supported (PNG, JPG, JPEG, BMP, TIFF)
- Try refreshing the page

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for best performance
2. **Batch Processing**: For many images, consider using the CLI tool instead
3. **Image Size**: Resize very large images before processing
4. **Browser**: Use Chrome or Firefox for best compatibility

## Development

### File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── run_web_interface.py      # Startup script
├── requirements_streamlit.txt # Web interface dependencies
└── untextre/                 # Core processing modules
    ├── cli.py               # Command-line interface (reused)
    ├── inpaint.py           # Inpainting functionality
    └── ...                  # Other modules
```

### Customization

The web interface can be customized by editing `streamlit_app.py`:

- **UI Layout**: Modify the Streamlit components
- **Default Settings**: Change default values for sliders/options
- **Styling**: Add custom CSS with `st.markdown()`
- **Additional Features**: Add new processing options or visualizations

### API Integration

The core processing function `process_image_streamlit()` can be adapted for other web frameworks or API endpoints.

## License

Same as the main UnTextre project. 