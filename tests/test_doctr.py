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
cv2.imwrite("images/test_image.jpg", image)

# Try loading the image
try:
    from doctr.io import DocumentFile
    doc = DocumentFile.from_images("images/test_image.jpg")
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
    
    # Create pre-processor with correct parameters from the model's config
    pre_processor = PreProcessor(
        output_size=(1024, 1024),  # From default_cfgs in DBNet
        batch_size=1,
        mean=(0.798, 0.785, 0.772),  # From default_cfgs in DBNet
        std=(0.264, 0.2749, 0.287)   # From default_cfgs in DBNet
    )
    print("\nPre-processor type:", type(pre_processor))
            
    predictor = DetectionPredictor(pre_processor=pre_processor, model=model)
    result = predictor(doc)
    print("\nSuccessfully ran prediction")
    print("Result type:", type(result))
    print("Result length:", len(result))
    
    # Process results based on the actual implementation
    if result:
        first_result = result[0]
        print("\nFirst result type:", type(first_result))
        print("First result keys:", first_result.keys())
        
        # The result should be a dict with 'words' entry
        if 'words' in first_result:
            words = first_result['words']
            print("\nNumber of detected words:", len(words))
            for i, word in enumerate(words):
                print(f"\nWord {i}:")
                print(f"  Type: {type(word)}")
                print(f"  Shape: {word.shape if hasattr(word, 'shape') else 'N/A'}")
                print(f"  Content: {word}")
                
                # Draw the box on the image
                h, w = image.shape[:2]
                # Convert box coordinates to polygon points
                x1, y1, x2, y2, score = word
                points = np.array([
                    [int(x1 * w), int(y1 * h)],
                    [int(x2 * w), int(y1 * h)],
                    [int(x2 * w), int(y2 * h)],
                    [int(x1 * w), int(y2 * h)]
                ], dtype=np.int32)
                cv2.polylines(image, [points], True, (0, 255, 0), 2)
                
        # Save the annotated image
        cv2.imwrite("images/test_image_detected.jpg", image)
except Exception as e:
    print("Error running prediction:", e)
    import traceback
    traceback.print_exc() 