"""Test script to verify DocTR API structure."""

import numpy as np
import cv2
import inspect

# First verify basic imports
try:
    import doctr
    print("Successfully imported doctr")
    print("Version:", doctr.__version__)
except ImportError as e:
    print("Error importing doctr:", e)

# Create a test image
image = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.putText(image, "TEST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite("test_image.jpg", image)

# Try loading the image
try:
    from doctr.io import DocumentFile
    doc = DocumentFile.from_images("test_image.jpg")
    print("\nSuccessfully loaded image")
    print("Document type:", type(doc))
    print("Document length:", len(doc))
    print("First page type:", type(doc[0]))
    print("First page shape:", doc[0].shape)
except Exception as e:
    print("Error loading image:", e)

# Try model initialization
try:
    from doctr.models import detection
    model = detection.db_resnet50(pretrained=True)
    print("\nSuccessfully initialized model")
    print("Model type:", type(model))
except Exception as e:
    print("Error initializing model:", e)

# Try prediction
try:
    from doctr.models.detection.predictor import DetectionPredictor
    from doctr.models.preprocessor.pytorch import PreProcessor
    
    # Print the signature of DetectionPredictor to see what arguments it needs
    print("\nDetectionPredictor signature:")
    print(inspect.signature(DetectionPredictor.__init__))
    
    # Print the signature of PreProcessor to see what arguments it needs
    print("\nPreProcessor signature:")
    print(inspect.signature(PreProcessor.__init__))
    
    # Try to find the pre-processor in the detection module
    print("\nAvailable in detection module:")
    for name in dir(detection):
        if not name.startswith('_'):  # Skip private attributes
            print(f"- {name}")
    
    # Create pre-processor with required arguments
    pre_processor = PreProcessor(output_size=(1024, 1024), batch_size=1)
    print("\nPre-processor type:", type(pre_processor))
            
    predictor = DetectionPredictor(pre_processor=pre_processor, model=model)
    result = predictor(doc)
    print("\nSuccessfully ran prediction")
    print("Result type:", type(result))
    print("Result length:", len(result))
    
    # Inspect the first result
    if result:
        first_result = result[0]
        print("\nFirst result type:", type(first_result))
        print("First result attributes:", dir(first_result))
        
        # Try to access the geometry if it exists
        if hasattr(first_result, 'geometry'):
            print("\nGeometry:", first_result.geometry)
        else:
            print("\nNo geometry attribute found")
            
        # Try to access the confidence if it exists
        if hasattr(first_result, 'confidence'):
            print("Confidence:", first_result.confidence)
        else:
            print("No confidence attribute found")
except Exception as e:
    print("Error running prediction:", e)

# Inspect first page results
page = result.pages[0]
print("\nPage type:", type(page))
print("Number of blocks:", len(page.blocks))

# Inspect first block
block = page.blocks[0]
print("\nBlock type:", type(block))
print("Number of lines:", len(block.lines))

# Inspect first line
line = block.lines[0]
print("\nLine type:", type(line))
print("Number of words:", len(line.words))

# Inspect first word
word = line.words[0]
print("\nWord type:", type(word))
print("Word value:", word.value)
print("Word confidence:", word.confidence)
print("Word geometry:", word.geometry)
print("Word geometry type:", type(word.geometry))
print("Word geometry shape:", np.array(word.geometry).shape) 