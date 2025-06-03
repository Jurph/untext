# Vision 

The goal of this tool is to detect text-based watermarks in images, generate a mask that exactly matches that text, and then inpaint only the masked area. 

To get there, we need the following:  

- Text detection: fast, lightweight, reliable. Probably something that already exists from this list (https://github.com/topics/text-detection), and ideally would generate not just a bounding box, but a set of pixels where text is detected 
- Text capture: perhaps edge detection within the text detector's bounding box can catch all of the text?  
- Mask generation: take the set of pixels that contain text, "bloom" them by 2px or so, and store that in memory as a black-and-white bitmask for inpainting 
- Inpainting: Generate an inpainted copy of the original with no text. Technologies under consideration include TELEA, LaMa, or Deep Image Prior, but anything **fast** and **accurate** will do. 

# TODO: 

- Choose a text detection approach that emits a pixel-map of pixels containing text : [OPEN]
- Test text detection on a variety of images : [PARTIAL]
- Implement a simple CLI pipeline : [DONE]
- Select small useful sub-regions [DONE]
- Add --mask "maskfile.png" so users using --save-mask can edit the mask and retry
- Add --threshold NN (in percent) to tune false positives for bad masks
- Add --letters mode to fill the boxes with letters that approximately match the watermark 
- Run a bake-off between TELEA, LaMa, and DIP 
- Consider a bake-off that adds EAST to our detection suite and tests it against DocTR 
- Eventually: add a local web service with drag-and-drop
- Eventually: add batch processing for files in a directory 
- Test, test, test!

# Flow: 

1. Scan incoming image with DocTR and identify bounding box(es) for all detected text 
2. Define a bounding box around the text that we call the "mask region"  
2b. OCR the text in the mask region 
2c. Use cv2 methods to quickly write white text as the mask region, then dilate that text 
3. Generate a "mask image" that is all black, with the mask region in white 
4. Define a larger bounding box around the mask region called the "image subregion" 
5. Crop the original image and the mask image so that we only operate on the subregion 
6. Perform inpainting (TALEA, LaMa, or in extremis, Deep Image Prior)

