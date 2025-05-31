# Vision 

The goal of this tool is to detect text-based watermarks in images, generate a mask that exactly matches that text, and use Deep Image Prior to inpaint only the masked area. 

To get there, we need the following:  

- Text detection: fast, lightweight, reliable. Probably something that already exists from this list (https://github.com/topics/text-detection), and ideally would generate not just a bounding box, but a set of pixels where text is detected
- Text capture: perhaps edge detection within the text detector's bounding box can catch all of the text?  
- Mask generation: take the set of pixels that contain text, "bloom" them by 2px or so, and store that in memory as a black-and-white bitmask for inpainting 
- Inpainting: use Deep Image Prior and the mask to generate an inpainted copy of the original with no text

# TODO: 

- Choose a text detection approach that emits a pixel-map of pixels containing text 
- Test text detection on a variety of images
- Set up Deep Image Prior 
- Implement a simple CLI pipeline
- Eventually: add a local web service with drag-and-drop
- Eventually: add batch processing for files in a directory 
- Test, test, test!

