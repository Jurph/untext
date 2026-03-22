# Design: `-U` Unknown Watermark Discovery Mode

**Date:** 2026-03-22
**Status:** Approved for implementation
**Contrast with:** `-K` (Known watermark, user-supplied RGBA template)

---

## Problem

Some watermarks include a semitransparent backing rectangle behind the text. After text removal, a faint rectangular artifact remains. More broadly, when processing a large batch of consistently-watermarked images, manually identifying the watermark for `-K` is tedious. `-U` automates that discovery.

**Known limitation:** ORB keypoint matching requires texture. A purely featureless backing rectangle with no overlapping text or logo will produce zero ORB keypoints and fail to match in Phase 4. In practice, most watermarks include text or a logo with sufficient texture; a plain tinted rectangle alone is not a supported input.

**Assumption:** All images in the input directory that carry a watermark carry the same one (or a scaled variant of it). Mixing watermarked and unwatermarked images in one directory degrades discovery quality; the user should ensure the directory is homogeneous. See Failure Modes.

---

## Goal

Given a directory of images that share a common watermark (possibly at different resolutions), automatically:

1. Discover the watermark as a cropped RGBA PNG
2. Save that PNG to the output directory for future `-K` reuse
3. Process every image in the directory using the discovered template via the existing `-K` / ORB pipeline

---

## Design

### Phase 1 — Bucketing

Group all images in the input directory by **exact pixel dimensions** (W×H). Portrait and landscape are separate buckets. Freeze the image file list before writing any output (so `watermark_candidate.png` files written later are not picked up as inputs).

Buckets with fewer than 3 images cannot self-discover but are still processed in Phase 4. All images — regardless of bucket size — are processed uniformly against all discovered templates in Phase 4.

---

### Phase 2 — Per-Bucket Discovery

For each bucket with ≥ 3 images:

**Step 1 — Random pair stacking and variance.**
Repeatedly draw random **pairs** (2 images, **with replacement**) from the bucket. For each pair, convert both images to grayscale (luminance, [0, 1] float). Compute per-pixel population variance using `np.var(stack, axis=0)` where `stack` has shape `(2, H, W)`. Pixels with variance below **0.01** are low-variance candidates. (Population variance of a 2-image pair equals `((a − b) / 2)²`; a threshold of 0.01 corresponds to a per-pixel luminance difference of about 20%.)

**Step 2 — Blob extraction.**
Threshold the variance map to binary (1 = low-variance candidate). Apply morphological closing with an 11×11 rectangular kernel, then dilation with a 13×13 elliptical kernel, using direct `cv2` operations — **do not call `morph_clean_mask`**, which requires a bounding box not yet available. The Gaussian blur step in `morph_clean_mask` is intentionally omitted; smooth edges are not needed for bounding-box extraction. Extract **8-connected** contiguous blobs with area ≥ **0.05% of total image area** (e.g., 300 px² in a 2000×3000 image). Blobs below this area are discarded as noise.

**Step 3 — Positional clustering.**
Divide the image into a **4×4 grid** (16 cells, each 25% of width and height). For each draw, each extracted blob independently contributes to the grid cell containing its centroid. A draw that produces two blobs populates two cells. Convergence is tracked globally across the full set of occupied cells (not per-cell). Zero-blob draws count toward the consecutive-stable-draw streak — they do not change the set, so they count as stable. Continue drawing until either: (a) the set of occupied grid cells has not changed for **10 consecutive draws**, or (b) **50 draws** total have been performed. If 50 draws complete without convergence, accept the current occupied set and continue (log a warning). Each occupied cell at convergence represents one candidate marking strategy (e.g., upper-right, lower-left).

**Step 4 — Crop to RGBA.**
For each candidate cluster (occupied grid cell), compute the **pixel-wise mean of all unique images in the bucket** — the cluster's grid cell determines only the crop bounding box, not which images are averaged. Crop the union of all blobs assigned to that grid cell from this mean image, plus an **8-pixel transparent border** on all sides. The crop's alpha channel is 255 inside the blob mask and 0 in the border. No position metadata is retained beyond this point.

**Step 5 — Select best candidate per bucket.**
If multiple candidates have aspect ratios within **10%** of each other (symmetric relative difference: `2 × |r1 − r2| / (r1 + r2) < 0.10`), they are likely scaled variants of the same mark. Keep the **largest crop by pixel area**; more pixels yields more ORB keypoints and sharper edges.

---

### Phase 3 — Cross-Bucket Validation

If only one qualifying bucket exists (or only one produced candidates), skip cross-validation, proceed with the single candidate, and log that cross-validation was unavailable.

For each pair of buckets, compare their best candidates:

1. Resize the smaller crop (by **pixel area**) to match the larger's dimensions (bicubic).
2. Compute **IoU on the alpha channels** (alpha > 127 = foreground).
3. If IoU ≥ **0.50**, the candidates belong to the same watermark family.

**One output per family**: select the largest crop by pixel area within the family as the canonical output RGBA PNG. When multiple families exist, they are ordered by descending pixel area for Phase 4 processing.

If buckets disagree (IoU < 0.50), each becomes its own family and produces its own output file.

---

### Phase 4 — Processing

1. Write each discovered watermark family to the output directory **before** inpainting (image file list was frozen in Phase 1, so these writes do not contaminate input iteration):
   - First family: `watermark_candidate.png`
   - Additional families: `watermark_candidate_2.png`, `watermark_candidate_3.png`, etc.
   - If a file with that name already exists, overwrite it and log a warning. Any higher-numbered files from a prior run that exceed the current family count are **not** deleted; log them by name so the user can clean up manually.

2. For **all images in the directory** (regardless of source bucket size), run the existing `-K` ORB pipeline with each candidate in order of **descending crop pixel area** (largest first). Use **first-match-wins** — consistent with `try_watermark_cascade`; no changes to matching or inpainting logic.

3. Images where ORB finds no match across all candidates are skipped and **logged**.

---

## Failure Modes

| Situation | Behavior |
|---|---|
| No bucket reaches 3 images | Warn and exit cleanly; suggest `-K` with a manually-identified template |
| A bucket's blob extraction yields no blobs | Warn; skip that bucket; continue with remaining buckets |
| All qualifying buckets yield no blobs | Warn and exit cleanly; suggest lowering `--variance-threshold` or using `-K` |
| Single bucket, no cross-validation possible | Proceed; log that cross-validation was unavailable |
| Buckets yield disagreeing candidates (IoU < 0.50) | Write multiple candidate PNGs; log which template matched each image |
| Image matches no candidate | Skip and log (consistent with `-K` behavior) |
| Discovered crop is larger than a target image | Skip that image; log crop dimensions and image dimensions |
| Discovered crop has no ORB keypoints (featureless rectangle) | No images will match; all skipped and logged; user should inspect `watermark_candidate.png` |
| Corrupt or unreadable image file | Skip and log; do not abort the bucket or batch |
| Output directory not writable | Abort with a clear error before processing begins |
| Input and output directories are the same path | Log a warning; image list was frozen before any writes, so discovery is unaffected, but output images will overwrite originals — require `--force` to proceed |
| All images in a bucket are pixel-identical (duplicates) | Variance map is all zero; no blobs extracted; treated as "no blobs" failure mode above; additionally log a hint that the bucket may contain duplicate files |
| Directory contains mixed watermarked and unwatermarked images | Discovery quality degrades; pairs that draw one of each will have high variance at watermark pixels, reducing consistency. No special handling — user should ensure the directory is homogeneous (stated in Assumptions) |

---

## Output Artifacts

- `{output_dir}/watermark_candidate.png` — tight RGBA crop, 8px transparent border, no size/position encoding in filename
- `{output_dir}/watermark_candidate_N.png` — additional families if multiple distinct marks found
- Cleaned images, as normal

---

## CLI

- Flag: `-U` / `--unknown-watermark`
- Mutually exclusive with `-K`
- Requires directory input; single-image use is not supported (use the web UI instead)
- Optional tuning parameter: `--variance-threshold FLOAT` (default 0.01)

---

## Integration Points

- **Existing `-K` pipeline**: discovery feeds directly into `try_watermark_cascade` / ORB matching — **no changes required** to the matching or inpainting stages
- **Morphological cleanup**: Phase 2 uses direct `cv2` morphological operations rather than `morph_clean_mask` (which requires a bbox not available at that stage)
- **Batch processing**: reuses existing directory iteration infrastructure; image list frozen before Phase 4 writes
