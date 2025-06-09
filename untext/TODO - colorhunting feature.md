## New Feature

Let's create a separate script called "text_hunting_sandbox.py" that follows this process: 

1. OCR the starting image using DocTR detect() to get a bounding box around a text region; if we have no text region, use the bottom right quarter (1/4 of the image width, 1/16 of its height)

2. Dilate the bounding box around the largest text region in the image by 4px and use this as the subregion of investigation 

3. Quantize the colors within the bounding box ("subregion") down to only 16 colors 

4. For the top 8 quantized colors, identify the color that is "most gray" - that is, the color with the least variance in its R/G/B channels   

5. Map the grayest color back to a list of colors in the original image - the colors of pixels that quantized down to this gray - and let the set of original pre-quantized colors be called "text colors"    

6. Generate a binary mask of pixels that are "text colors" in the original image; white if they are text color pixels, black if not  

7. Through successive morph_open, dilate, blur, and morph_close operations, reassemble these pixels into closed regions that are slightly larger than the original text. Anchor the connected components closest to the detected text as valid and drop the outliers.      

8. Mask out the image and perform LaMa inpainting on it using the refined binary mask  

## Distance Checking 

This code is intended to help filter out disconnected pixels far away from the center of our original bounding box. 

```

def anchor_connected_components(mask: np.ndarray, bbox: Tuple[int, int, int, int], max_dist: int = 100) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    keep = np.zeros(num_labels, dtype=bool)
    for i in range(1, num_labels):  # skip background
        px, py = centroids[i]
        if (px - cx) ** 2 + (py - cy) ** 2 <= max_dist ** 2:
            keep[i] = True

    return np.where(keep[labels], 255, 0).astype(np.uint8)
    
    ```
