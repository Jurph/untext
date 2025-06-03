## Images provided for testing 

test1.png - an image with a watermark in the lower-right corner 
test2.png - an image with a watermark in the upper-left corner 
test3.jpg - a real watermarked photo, a sample of what we're going for later 
test4-without-text.png - an image of a flower without text 
test4-with-text.png - the same image, but with text overlaid 

## Testing approach 

We can test all of these for detection. The 4th one presents a unique opportunity to test detection and inpainting and then compare the inpainted version to the original without text, using similarity scoring. 

## IMPORTANT 

Don't leave any artifacts in this directory except the original test images! Outputs should be placed in /tests/outputs and deleted before each run of pytest. 

