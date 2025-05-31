"""Test script to verify DocTR API structure."""

from doctr.io import DocumentFile
from doctr.models import db_resnet50
from doctr.models.detection.predictor import DetectionPredictor

# Create a test image
import numpy as np
import cv2

# Create a white image with black text
image = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.putText(image, "TEST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite("test_image.jpg", image)

# Load image with DocTR
doc = DocumentFile.from_images("test_image.jpg")
print("Document type:", type(doc))
print("Document length:", len(doc))
print("First page type:", type(doc[0]))
print("First page shape:", doc[0].shape)

# Initialize model
model = db_resnet50(pretrained=True)
predictor = DetectionPredictor(model)

# Run detection
result = predictor(doc)
print("\nDetection result type:", type(result))
print("Number of pages:", len(result.pages))

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