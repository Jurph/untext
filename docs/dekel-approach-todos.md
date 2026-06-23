# Dekel et al. CVPR 2017 — Feature Roadmap

Source: [alpla-watermark-removal](https://github.com/sweihub/alpla-watermark-removal)
Paper: "On the Effectiveness of Visible Watermarks" (Dekel, Rubinstein, Liu, Freeman, CVPR 2017)

Each item is a discrete improvement to the `untext -U` discovery pipeline.
Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` rejected/deferred

---

## Lesson 1 — Median gradient as primary watermark signal

**Status:** `[x]`

**What it is:**
Compute the image gradient `∇J_k` for each source image `k`. Take the per-pixel median across the stack:

```
∇W_m(p) = median_k( ∇J_k(p) )
```

The median gradient magnitude `|∇W_m|` is a robust estimator of where the watermark has structure. Scene background gradients are statistically independent across varying images and cancel toward zero under the median; the watermark gradient is constant at fixed position and survives.

**Why it matters:**
The current composite score uses variance and variance step-edges. Both fail to distinguish watermark pixels from consistently-lit static background content. The median-gradient signal is orthogonal: a static textured background produces consistent gradients too, but they are *scene gradients*, not an *overlay gradient*. A watermark with internal edges (text strokes, logo outlines) shows gradient magnitude specifically at those overlay edges — which scene content at the same location does not replicate unless scene and watermark edges happen to align.

**Expected impact:**
Adds a third dimension to the composite score, weighting candidate zones that contain spatially structured gradient evidence. Should preferentially score text and logo watermarks over flat-colored stable patches.

**Implementation notes:**
- Compute Sobel `(dx, dy)` on grayscale for each image in the bucket
- Accumulate into a median buffer (or use `np.median` over a (N, H, W) stack; memory trade-off)
- Normalize median gradient magnitude to [0, 1]
- Combine with existing score: `score = score_variance × score_median_grad` (product keeps both conditions active)
- This does NOT require changing zone selection or output format

**Known risks:**
JPEG compression introduces blocking artifacts at 8×8 block boundaries; these produce spurious gradients. May need a mild Gaussian blur (σ≈0.5) before gradient computation on JPEG inputs.

---

## Lesson 2 — Poisson reconstruction for watermark color recovery

**Status:** `[ ]`

**What it is:**
Instead of recovering watermark color from `mean_bgr` via the transparency inversion formula, integrate the median gradient field `∇W_m` to reconstruct the watermark image directly:

```
W_reconstructed = Poisson_solve( div(∇W_m) )
```

Poisson reconstruction (FFT/DST-based) finds the image whose Laplacian matches the divergence of the median gradient field. It recovers a clean, smooth watermark image from the gradient evidence.

**Why it matters:**
The current `extract_watermark_colors` implementation uses inpainted mean values as a proxy for background color, then inverts the compositing model. This is noisy because (a) the inpainting radius is a free parameter, (b) the mean is polluted by stable scene content at the same location, and (c) alpha estimation via variance ratio degrades near faint edges. Poisson reconstruction avoids all of these: it works in gradient space, which is where the watermark signal is cleanest.

**Expected impact:**
Cleaner watermark color extraction, fewer background-bleed artifacts in the BGRA output. Especially beneficial for semi-transparent marks where `extract_watermark_colors` currently produces washed-out alpha estimates.

**Implementation notes:**
- Requires `scipy.fft` (already available) or the DST approach from the Dekel repo
- Input: median gradient field `(gx, gy)` over the full image
- Boundary condition: Neumann (zero normal derivative at image borders)
- Output: grayscale or per-channel reconstructed watermark values (zero-mean; needs offset calibration)
- Should replace or augment `extract_watermark_colors()` rather than `_deblot_mask_by_area`

**Dependencies:**
Requires Lesson 1 (median gradient computation) to be implemented first.

---

## Lesson 3 — Iterative alternating solver

**Status:** `[ ]` (deferred — implement only after Lessons 1 and 2 are stable)

**What it is:**
The full Dekel algorithm is an iterative optimization that alternates between:
1. **Estimate background** `I_k` given current watermark estimate `W` and `α`
2. **Update watermark** `W` given estimated backgrounds
3. **Update opacity** `α` given `W` and estimated backgrounds

Each iteration refines all three unknowns jointly. The gradient-median step (Lesson 1) is the initialization for this loop; Poisson reconstruction (Lesson 2) is the per-iteration watermark update.

**Why it matters:**
A single-pass pipeline (current approach) makes one estimate of `W` and stops. The iterative solver converges toward a fixed point where `W`, `α`, and `{I_k}` are mutually consistent. This reduces residual error significantly in the published results.

**Expected impact:**
Potential large improvement in watermark fidelity (text strokes sharper, alpha boundary cleaner) but at significant computational cost. Probably 5–15 iterations needed for convergence.

**Implementation notes:**
- At `untext -U` scale (6–30 images, 1–4 MP each), memory for the full image stack is tractable
- The Dekel repo processes one channel at a time to reduce peak memory
- A convergence criterion: stop when `|W_new - W_old|_∞ < ε` (e.g. ε=1/255)
- Do not implement this before the single-pass pipeline is working well; iterating on a bad initialization diverges

**Dependencies:**
Requires Lessons 1 and 2.

---

## Lesson 4 — Gradient-consistency candidate weighting

**Status:** `[x]`

**What it is:**
For each candidate zone/mask, compute the fraction of pixels that have both (a) low variance (stable) and (b) high median gradient magnitude. Call this the *gradient-consistency score* of the candidate. Use it to rank candidates instead of raw stable-pixel count.

**Why it matters:**
The current zone-density ranking (Step 5) picks zones with the most stable pixels. A large flat-colored stable region (e.g. consistent white wall in the background) scores well on stable-pixel count but has near-zero gradient magnitude. A text watermark like "TheLovelyNora" scores lower on count but has high gradient magnitude at every stroke edge. Gradient-consistency scoring inverts this advantage.

**Expected impact:**
Should demote flat background blobs and promote structured watermark candidates. Medium-impact improvement; less fundamental than Lesson 1 but cheaper to implement.

**Implementation notes:**
- Threshold median gradient magnitude at some data-derived level (e.g. 75th percentile of gradient magnitudes within stable pixels)
- Gradient-consistency score = (pixels with high grad AND stable) / (pixels stable)
- Use as a secondary ranking criterion or multiplicative weight with stable-pixel count

**Dependencies:**
Requires Lesson 1.

---

## Bug Fix A — `_deblot_mask_by_area` kept < 2 fallback

**Status:** `[x]`

**What it is:**
When Otsu correctly identifies a large clean signal component as the only keeper, the current guard `if kept < 2: return mask` reverts to the original noisy mask. Should be `if kept < 1`.

**Why it matters:**
The deblotching function silently undoes its own correct output when only one large component survives the Otsu split. This is observed in real runs: candidate masks that should be deblotted to a clean text strip are returned noisy instead.

**Fix:**
```python
# line: if kept < 2:
if kept < 1:
```

One-line fix. Low risk.

---

## Bug Fix B — Otsu unreliable on small component counts

**Status:** `[x]`

**What it is:**
`_deblot_mask_by_area` applies Otsu on the log-area distribution. When a candidate has only 4–6 components (common after cross-sub-sample validation reduces the stable mask), Otsu has no statistical basis to find a bimodal split and produces arbitrary thresholds.

**Why it matters:**
Observed in real runs: candidate_5 had 6 components after cross-sub-sample intersection. Otsu set `min_area=487px` and kept only 3 components — behavior indistinguishable from a hand-tuned threshold.

**Fix options:**
1. Add `if len(areas) < 8: return mask` (skip deblotching when sample is too small)
2. Replace Otsu with a relative-area filter: keep only components with area ≥ (max_area × 0.05)
3. Skip deblotching entirely; rely on gradient-consistency (Lesson 4) to demote blotchy zones upstream

Option 3 is preferred if Lesson 4 is implemented; otherwise Option 2 as interim fix.

---

## Deferred / Rejected

### Cross-sub-sample consensus over BGRA crops
**Status:** `[!]` rejected

Post-hoc alignment and averaging of BGRA crops from different spatial zones does not work because the crops are in different coordinate systems and represent different scene regions. The correct intervention is upstream zone classification, not downstream crop fusion. See `2026-03-31-watermark-consensus-rebuttal.md`.

### Hard fill-ratio cutoffs
**Status:** `[!]` rejected

Thresholds like "keep only zones where >10% of pixels are stable" are not generalizable across watermark sizes, opacities, or image content. Replaced by data-derived Otsu and zone-density ranking.
