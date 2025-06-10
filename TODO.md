# Vision 

The goal of this tool is fast and accurate watermark removal. We detect text-based watermarks in images, we identify regions of that color, generate a mask that exactly matches those pixelated regions, and then inpaint only the masked area. 


# TODO: 

- Choose a text detection approach that emits a pixel-map of pixels containing text : [SOLVED] in postprocessing 
- Test text detection on a variety of images
- Implement a simple CLI pipeline 
- Select small useful sub-regions for inpainting 
- Add --mask "maskfile.png" so users using --save-mask can edit the mask and retry
- Add --threshold NN (in percent) to tune false positives for bad masks
- Add --letters mode to fill the boxes with letters that approximately match the watermark 
- Run a bake-off between TELEA, LaMa, and DIP 
- Consider a bake-off that adds EAST to our detection suite and tests it against DocTR 
- Eventually: add a local web service with drag-and-drop
- Eventually: add batch processing for files in a directory 
- Test, test, test!

# Our New Flow: 

0. The user can specify an image file or directory of image files; an optional target color in hex or in HTML names; an OCR or text detection approach (EAST, EasyOCR, or DocTR); an inpainting approach (LaMa, TELEA); an optional mask file; and an optional output directory. Users may also specify verbose mode, a logging location, creation of a timing report, or to store mask files alongside outputs.   

1. **preprocessor.py** - Open the target image ("image") for processing; if appropriate run the EAST, EasyOCR, or DocTR preprocessing approach best suited to maximizing detection. If no preprocessor is selected then the "preprocessed image" is just a copy of the image. 

2. **detector.py** - Run EAST, EasyOCR, or DocTR against the preprocessed image to get a set of bounding boxes where text is most likely to be. Watermarks are usually in a lower corner, so choose the bounding box closest to a lower-left or lower-right corner (centroid to corner distance). Look for any other bounding boxes whose vertical span (Y coordinates) include the Y-coordinate of the centroid of the cornermost box. Group these boxes together by drawing a single bbox that surrounds all of them; they are "the bounding box". 

3. **find_text_colors.py** - Make a copy of the original image. Within the bounding box coordinates, quantize the colors to 8 total colors. If there is a target color, find the color closest to that target color. If there is no target color specified, the target color is the color with the least variance between its R/G/B channels (the "grayest" color), since watermarks are often black, white, or gray. Select all the pixels within the bounding box that have the target color, and then re-map those pixel locations back to the original image. The colors of those pixels are the "text colors" list. 

4. **mask_generator.py** - To maximally capture the watermark we allow pixels to occur at any X-coordinates between a span of Y-values equal to the bounding box's original span plus a full bounding box height above and below. This triples the bbox height to account for long-stemmed letters, and extends it all the way left and right to account for long lines of text. Coordinates outside the image coordinates are dropped. All of these pixels are "candidate pixels". Use a series of image processing steps (morphological open 2x2, morph close 3x3, dilate 6x6 ellipse, erode 3x3 ellipse, gaussian blur 3x3, sharpen, dilate 3x3 ellipse, blur 3x3 ellipse) to capture the pixels' full footprints. Generate a "binary mask" where the remaining candidate pixels are white and all other pixels are black.  

5. **inpaint.py** - Draw a bounding box around the white pixels of the binary mask. Dilate this area by up to 64 pixels, staying within the boundaries of the image. Crop the image and mask to this "inpainting subregion". Remove any pixels from the image whose corresponding pixels in the mask are white. Inpaint the missing pixels using the selected or default (LaMa) inpainting approach. Write the image to the specified output directory. Optionally write the mask file as well. 

# Command Line Interface: 

untextre.py -i infile -o outfile ... 

-h / --help (this menu)
-i / --input *filename.ext* or --input *directory* 
-o / --output *directory*
-d / --detector (default = 'east', 'easyocr', 'doctr')
-c / --color ('#808080' or any hex color; 'gray' or any HTML color name)
-m / --maskfile filename.png 
-p / --paint (default = 'lama', 'telea')
-v / --verbose 
-k / --keep-masks 
-t / --timing 
-l / --logfile *filename*
