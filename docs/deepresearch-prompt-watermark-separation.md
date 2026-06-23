# DeepResearch Prompt: Separating Watermark from Stable Background in Multi-Image Stacks

**Context and goal**

I am building a Python tool that automatically discovers and extracts a visible watermark template from a directory of N (≥6) JPEG/PNG images. Every image in the batch has been composited with the same semi-transparent watermark (text, logo, or graphic overlay) at a consistent pixel position. The goal is to recover a clean BGRA template — watermark RGB color plus alpha/opacity — that can then be used by a separate removal pipeline. No prior knowledge of watermark shape, color, or position is assumed. The tool must work on watermarks that range from nearly opaque text to semi-transparent polygon overlays (α ≈ 0.15–0.7).

---

## Current pipeline (pseudocode)

```
INPUT: N images, all consistently watermarked, binned into buckets by pixel dimensions

For each bucket with |bucket| ≥ 3:

  STEP 1 — Variance map
    For each pixel (i,j):
      Compute running mean μ(i,j) and population variance σ²(i,j) via Welford's algorithm
      (grayscale variance, BGR mean stored separately)

  STEP 2 — Stable-pixel detection via Tukey fence
    Pool log-precision values ( = −log₁₀(σ²) ) across all buckets
    Apply Tukey extreme-outlier upper fence: threshold τ = 10^(−(Q3 + 3·IQR))
    stable_mask = pixels where σ²(i,j) ≤ τ
    // Rationale: genuine watermark pixels have σ² ≈ 0; this fence flags ~1–7% of pixels

  STEP 3 — Cross-sub-sample validation
    Split bucket images randomly into halves A and B (|A|, |B| ≥ 3)
    Compute σ²_A, σ²_B independently
    stable_mask = stable_mask AND (σ²_A ≤ τ) AND (σ²_B ≤ τ)
    // Keeps only pixels stable across independent sub-samples

  STEP 4 — Composite score
    score(i,j) = normalize(−log σ²(i,j)) × normalize(|∇ log σ²(i,j)|)
    // High score at pixels that are both stable AND at a variance step-edge

  STEP 5 — Zone-based candidate selection
    Find connected components of stable_mask
    Assign each component centroid to one of 6 spatial zones (3×2 grid, long edge divided into thirds)
    Rank zones by total stable-pixel count
    Select top 3 zones as candidates

  STEP 6 — Component-area filtering (deblotching)
    For each candidate zone mask:
      Sort connected components by area
      Apply Otsu threshold on log(area) distribution
      Discard components below the Otsu split
    // Intended to remove small satellite noise while keeping signal components

  STEP 7 — Transparency model inversion
    For each stable pixel: estimate background via inpainting of non-stable neighbors
    α_hat(i,j) = 1 − sqrt( σ²(i,j) / σ²_background_inpainted(i,j) )
    WM_color(i,j) = ( μ(i,j) − (1−α_hat)·μ_background(i,j) ) / α_hat(i,j)

  STEP 8 — Output
    Crop each zone mask + WM_color to a tight BGRA image

OUTPUT: 6–9 BGRA watermark candidates
```

---

## What we observe

The Tukey fence and cross-sub-sample intersection reliably find the watermark pixels when the watermark is the dominant source of reduced variance. The composite score (Step 4) correctly fires at variance step-edges, cleanly locating watermark boundaries. Steps 1–4 are working well.

The pipeline fails in Steps 5–6. The core problem is that **the stable mask contains two indistinguishable populations**:

1. **Genuine watermark pixels**: low variance because the watermark composites the same color at those pixels in every image. Modeled as: `pixel_i = α·WM + (1−α)·BG_i`, so `σ² = (1−α)²·Var[BG]` — reduced but nonzero for semi-transparent marks.

2. **False-positive stable pixels**: low variance because the photograph consistently shows the same scene content at those positions (static background, consistent studio setup, recurring clothing). These have `σ² ≈ 0` just as strongly as the watermark.

Both populations survive the Tukey fence and the cross-sub-sample intersection (if the scene element is present in all images). The zone-density ranking (Step 5) sometimes selects zones dominated by these false-positive patches rather than the true watermark zone. The component-area filter (Step 6) removes *small* satellite blotches but cannot distinguish a *large* false-positive stable patch from a *large* genuine watermark component.

**Results:** typically 2–3 of the 6–9 candidates approximate the true watermark (the text "TheLovelyNora" is visible but surrounded by satellite blotches). The remaining 4–6 candidates are large blotchy crops of consistently-lit scene regions with no watermark structure.

---

## What we have tried and rejected

- Arbitrary area thresholds for component selection (not generalizable to unknown watermarks)
- A downstream BGRA-level consensus step aligning crops from different zones (wrong: the crops are in different coordinate systems and don't share a common spatial reference)
- Hard fill-ratio cutoffs (arbitrary, not robust across watermark types)

---

## The open question

Given a per-pixel variance map and mean image computed from N aligned same-size images that all contain the same composited semi-transparent overlay:

**How do you robustly separate "low variance due to a composited watermark" from "low variance due to consistent background content"?**

We want techniques that derive any thresholds or parameters from the data itself rather than requiring domain constants about watermark size, opacity, or shape.

---

## Specific areas to investigate

1. **Multi-image visible watermark removal** — is there academic work on automatically extracting a watermark template from a batch of watermarked images (particularly Dekel et al. CVPR 2017 "On the Effectiveness of Visible Watermarks," and any follow-on work)? What statistical or optimization approaches do those methods use to isolate the watermark layer from the background?

2. **Layer separation / blind source separation** — are there methods from intrinsic image decomposition, ICA, or NMF that separate a consistent additive/compositing layer from varying background content across an image stack?

3. **Background modeling from image stacks** — video background subtraction literature (e.g., mixture-of-Gaussians, RPCA / robust PCA, ViBe) treats "consistent background" as the thing to model. We have the inverse problem: consistent background is the nuisance we want to exclude, and the composited foreground layer is the signal. Are there methods that exploit the *composite model* (alpha-blending) specifically to distinguish watermark pixels from static background pixels?

4. **Variance-structure segmentation** — are there methods that segment a variance map not just by magnitude (low/high variance) but by the *spatial structure* of low-variance regions — e.g., distinguishing a spatially coherent designed graphic from scattered coincidentally-stable patches?

5. **Existing Python libraries** — are there open-source Python libraries (OpenCV, scikit-image, PIL, or specialized watermarking/inpainting libraries) that implement any of the above in a way that could slot into a pipeline like ours?

Please prioritize methods that are (a) implementable without deep learning, (b) parameter-light or parameter-free, and (c) published in peer-reviewed venues or implemented in maintained open-source libraries.
