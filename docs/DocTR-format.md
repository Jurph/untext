# DocTR Format and Usage Guide

This document provides authoritative information about DocTR's input/output formats, common pitfalls, and the correct usage patterns. This is essential reading for any code working with DocTR OCR.

## Critical Bug Warning

⚠️ **IMPORTANT**: DocTR has critical preprocessor configuration bugs that cause dimension squashing:

- **Detection preprocessor**: Default configuration uses `output_size=(3, 1024)` instead of `(1024, 1024)`
- **Recognition preprocessor**: Default configuration uses `output_size=(3, 32)` instead of `(32, 128)`

These bugs cause complete OCR failure with errors like:
```
RuntimeError: Given input size: (128x1x16). Calculated output size: (128x0x8). Output size is too small
```

**Solution**: Always manually create and assign fixed preprocessors (see Fixed Usage section below).

## DocTR Operating Modes

DocTR operates in two distinct modes that should not be confused:

### 1. Detection Mode
- **Purpose**: Find text regions (bounding boxes/polygons) without extracting text content
- **Speed**: Fast
- **Output**: Geometry coordinates only
- **Use case**: When you need to know WHERE text is located
- **Method**: Direct detection model calls with `return_model_output=True`

### 2. OCR Mode (Detection + Recognition)
- **Purpose**: Find text regions AND extract the actual text content
- **Speed**: Slower (runs both detection and recognition)
- **Output**: Geometry coordinates + extracted text + confidence scores
- **Use case**: When you need WHAT the text says
- **Method**: Full `ocr_predictor()` pipeline

## Input Format Requirements

### Images
- **Format**: NumPy arrays with shape `(height, width, 3)`
- **Color space**: RGB (not BGR!)
- **Data type**: `uint8`
- **Value range**: 0-255

⚠️ **Common mistake**: OpenCV loads images as BGR, but DocTR expects RGB:
```python
# WRONG
img = cv2.imread('image.jpg')  # BGR format
results = ocr_predictor([img])  # Will have color issues

# CORRECT
img_bgr = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
results = ocr_predictor([img_rgb])
```

## Output Format Structure

DocTR returns a nested object hierarchy:

```
Document
├── pages: List[Page]
    ├── blocks: List[Block]
        ├── lines: List[Line]
            ├── words: List[Word]
                ├── value: str (the actual text)
                ├── confidence: float (0.0 to 1.0)
                ├── geometry: List[List[float]] (normalized coordinates)
                └── [other attributes...]
```

### Coordinate System

- **Normalization**: All coordinates are normalized to [0, 1] range
- **Format**: List of [x, y] points defining a polygon
- **Origin**: Top-left corner (0, 0)
- **Conversion to pixels**:
  ```python
  pixel_x = normalized_x * image_width
  pixel_y = normalized_y * image_height
  ```

### Example Word Object
```python
word.value = "Hello"
word.confidence = 0.95
word.geometry = [
    [0.1, 0.2],  # Top-left
    [0.3, 0.2],  # Top-right
    [0.3, 0.4],  # Bottom-right
    [0.1, 0.4]   # Bottom-left
]
```

## Fixed Usage Patterns

### Creating Fixed OCR Predictor

```python
from doctr.models import ocr_predictor
from doctr.models.preprocessor.pytorch import PreProcessor

# Create OCR predictor (will have broken preprocessors)
ocr = ocr_predictor(pretrained=True)

# Create FIXED detection preprocessor
detection_preprocessor = PreProcessor(
    output_size=(1024, 1024),  # FIX: was (3, 1024)
    batch_size=1,
    mean=(0.798, 0.785, 0.772),
    std=(0.264, 0.2749, 0.287),
    preserve_aspect_ratio=True,
    symmetric_pad=True
)

# Create FIXED recognition preprocessor  
recognition_preprocessor = PreProcessor(
    output_size=(32, 128),  # FIX: was (3, 32)
    batch_size=128,
    mean=(0.694, 0.695, 0.693),
    std=(0.299, 0.296, 0.301),
    preserve_aspect_ratio=True,
    symmetric_pad=True
)

# Replace broken preprocessors with fixed ones
ocr.det_predictor.pre_processor = detection_preprocessor
ocr.reco_predictor.pre_processor = recognition_preprocessor

# Now OCR will work correctly
```

### Processing Images

```python
import cv2

# Load and convert image
img_bgr = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Run OCR with fixed preprocessors
results = ocr([img_rgb])

# Extract text
for page in results.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                text = word.value
                confidence = word.confidence
                
                # Convert normalized geometry to pixels
                geometry = word.geometry
                height, width = img_rgb.shape[:2]
                pixel_coords = [
                    [int(x * width), int(y * height)]
                    for x, y in geometry
                ]
                
                print(f"Text: '{text}' (conf: {confidence:.2f})")
```

### Detection-Only Mode

For faster processing when you only need text locations:

```python
# Use the detection predictor directly
processed_batches = ocr.det_predictor.pre_processor([img_rgb])
raw_output = ocr.det_predictor.model(
    processed_batches[0], 
    return_model_output=True
)
segmentation_map = raw_output["out_map"]

# Process segmentation map to extract bounding boxes
# (implementation depends on specific needs)
```

## Model Architecture Information

### Detection Models
- **Input shape**: (3, 1024, 1024) for PyTorch, (1024, 1024, 3) for TensorFlow
- **Output**: Segmentation maps and geometric predictions
- **Common architectures**: `db_resnet50`, `db_mobilenet_v3_large`

### Recognition Models  
- **Input shape**: (3, 32, 128) for PyTorch, (32, 128, 3) for TensorFlow
- **Output**: Character predictions
- **Common architectures**: `crnn_vgg16_bn`, `crnn_mobilenet_v3_large`, `vitstr_base`

## Common Pitfalls and Solutions

### 1. Dimension Squashing Errors
**Problem**: `RuntimeError: Given input size: (128x1x16). Calculated output size: (128x0x8). Output size is too small`

**Cause**: Using default broken preprocessor configurations

**Solution**: Apply the fixed preprocessor configurations shown above

### 2. Empty OCR Results
**Problem**: DocTR returns no text even on images with visible text

**Possible causes**:
- Using broken preprocessors (apply fix above)
- Image in wrong color space (convert BGR → RGB)
- Image quality issues (try preprocessing)
- Confidence threshold too high

### 3. Coordinate Conversion Errors
**Problem**: Text regions appear in wrong locations

**Cause**: Forgetting to convert normalized coordinates to pixels

**Solution**: Always multiply by image dimensions:
```python
pixel_x = normalized_x * image_width
pixel_y = normalized_y * image_height
```

### 4. Model Loading Issues
**Problem**: Models fail to download or load

**Solutions**:
- Check internet connection for model downloads
- Verify PyTorch/TensorFlow installation
- Use explicit architecture names: `ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn")`

## Performance Considerations

### Speed vs Accuracy Trade-offs

| Mode | Speed | Accuracy | Use Case |
|------|--------|----------|----------|
| Detection only | Fast | Locations only | Text region masking |
| OCR (fixed) | Medium | Full text + locations | Complete text extraction |
| OCR (with preprocessing) | Slow | Potentially better | Poor quality images |

### Memory Usage
- **Detection**: ~500MB GPU memory for 1024×1024 images
- **Recognition**: Additional ~200MB per text region
- **Batch processing**: Memory usage scales linearly with batch size

## Version Compatibility

This guide is based on DocTR version 0.6+ with PyTorch backend. Behavior may vary with:
- Different DocTR versions
- TensorFlow backend vs PyTorch backend  
- Different model architectures
- Custom model training

## Testing and Validation

Always test DocTR integration with known working examples:

```python
# Minimal working example
import cv2
from doctr.models import ocr_predictor
from doctr.models.preprocessor.pytorch import PreProcessor

# Create simple test image with text
test_img = create_test_image_with_text()  # Your implementation

# Apply fixes and test
ocr = ocr_predictor(pretrained=True)
# ... apply preprocessor fixes as shown above ...

results = ocr([test_img])
assert len(results.pages) > 0, "No pages detected"
assert len(results.pages[0].blocks) > 0, "No text blocks found"

print("DocTR is working correctly!")
```

## References

- [DocTR Official Documentation](https://mindee.github.io/doctr/)
- [DocTR GitHub Repository](https://github.com/mindee/doctr)
- [Model Architecture Details](https://mindee.github.io/doctr/models.html)

---

**Last Updated**: December 2024  
**Validated With**: DocTR 0.6+, PyTorch backend  
**Status**: Fixes verified and working 