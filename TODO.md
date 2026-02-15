# Vision 

The goal of this tool is fast and accurate watermark removal. We detect text-based watermarks in images, we identify regions of that color, generate a mask that exactly matches those pixelated regions, and then inpaint only the masked area. 


# TODO

## Completed Features

- Text detection approach with pixel-accurate masking (spatial TF-IDF / FOM)
- Extensive testing (246 tests across 14 test modules)
- Full-featured CLI pipeline with all options
- Smart sub-region selection with padding and mod-4/mod-8 alignment
- `--maskfile` option for mask editing and retry
- LaMa (default) vs TELEA inpainting, both implemented
- EAST added to detection suite — consensus of 3 detectors (EAST+DocTR+EasyOCR)
- Local web service with drag-and-drop (Streamlit app)
- Batch processing for directories
- Automatic granularity optimization (g=4 default, auto-retry at g=8)
- CLI bbox expansion along long axis for missed text
- VRAM management for steady-state batch processing
- Known-mask mode (`-K`) for ORB-based template matching of consistent watermarks
- `@pytest.mark.slow` on model-loading tests; fast suite runs in ~18s
- Streamlit `main()` decomposed: pure helpers extracted, section comments, constants
- `mask_generator.py` pruned to single production function (`morph_clean_mask`)
- `cli.py` mask+inpaint closure extracted to module-level `_generate_masks_and_inpaint`

## Future Ideas

- **`--letters` mode**: Fill detected regions with approximate matching letters (font rendering + OCR text extraction). Lower priority — current inpainting works well.
- **End-to-end pipeline tests**: Full image-in / image-out tests with SSIM/PSNR quality gates.
- **Shape-metric fusion**: Combine FOM with per-cluster blackhat_energy and edge_row_energy for even better text/background separation. Experiment infrastructure exists in `experiments/cluster_shape_experiment.py`.


# Architecture

## Pipeline Flow

### 1. `preprocessor.py` — Image Enhancement
- CLAHE contrast enhancement
- Bilateral filtering for noise reduction while preserving edges
- Grayscale → RGB conversion for detector compatibility

### 2. `consensus.py` — Consensus Detection
- Run all 3 detectors simultaneously: EAST, DocTR, EasyOCR
- Find regions where 2+ detectors agree (configurable overlap threshold)
- Calculate hybrid confidence: 1 − (1−c₁)×(1−c₂)×…×(1−cₙ)
- Merge overlapping detections into consensus bounding boxes
- Pad boxes by 20% and align to mod-4
- Failover cascade if no consensus:
  1. Rotation (90° clockwise)
  2. Target color enhancement (if user specified)
  3. Generic gray enhancement (#808080)
  4. White enhancement (#FFFFFF)
  5. Fall back to standard watermark regions (bottom corners)

### 3. `find_text_colors.py` — Figure of Merit Analysis
- For each consensus region:
  - Extract region + surrounding area (1.414× expanded for equal pixel count)
  - K-means cluster colors (default: 4 clusters, auto-retry at 8 if remnants detected)
  - For each cluster, compute TF-IDF, border ratio, and largest CC fraction
  - Compute weighted FOM = 0.07 × tf_idf + 0.63 × border + 0.30 × cc
  - Accept clusters with FOM ≥ 0.30 and largest CC < 85%
  - Optional: force inclusion of target color cluster
- Combine all regional masks into final mask

### 4. `mask_generator.py` — Morphological Cleanup
- Closing to fill gaps and connect fragments (11×11 kernel)
- Dilation for inpainting coverage (13×13 ellipse)
- Gaussian blur for smooth edges (9×9)
- Re-threshold to binary mask

### 5. `inpaint.py` — LaMa/TELEA Inpainting
- Calculate optimal subregion from mask bounds
- Dilate subregion by 64 pixels for context
- Pad to mod-8 for neural network compatibility
- Crop image and mask to subregion
- Run LaMa (GPU-accelerated) or TELEA (fast CPU fallback)
- Paste result back into full image
- Save cleaned image and optional mask 
