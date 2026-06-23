## Plan for an LLM Coding Agent: Replace “low-variance blob export” with “overlay discovery + template recovery”

### Objective

Refactor the watermark auto-discovery path so it no longer assumes that watermark pixels are nearly byte-identical across the stack. The new pipeline should:

1. detect the watermark **zone** using a composite score built from:

   * low variance across the image stack,
   * strong structure in the mean image,
   * optional border/corner prior;
2. recover an approximate **watermark-only BGRA template** from that zone;
3. emit debug artifacts that make each intermediate decision visually inspectable.

The current implementation instead computes grayscale running variance, thresholds for very low variance, labels connected components, and exports BGRA crops whose RGB comes from the mean image and whose alpha comes from the binary blob mask. That is the behavior to replace, not merely retune. 

---

## Ground rules

* **Do not** try to salvage the current fixed threshold as the primary detector.
* **Do not** detect in grayscale only.
* **Do not** export the mean-image crop as if it were the recovered watermark.
* Keep the existing external contract as intact as possible:

  * input: `List[Path]`
  * output: `List[np.ndarray]` of BGRA crops
* Prefer OpenCV / NumPy first. If needed, make scikit-image background estimation optional behind a guarded import. OpenCV’s morphology operators are built for binary-shape cleanup, and `connectedComponentsWithStats` is the right primitive for post-threshold blob analysis. ([OpenCV Documentation][1])

---

## Phase 1: Preserve current behavior behind a feature flag

### Goal

Create a safe branch point so the new algorithm can be developed and compared without breaking the existing CLI path.

### Tasks

* Add a new internal switch, for example:

  * `DISCOVERY_V2_ENABLED = True`
  * or a function-level kwarg like `method: str = "composite_score"`
* Rename the current algorithm to something explicit, e.g.:

  * `_discover_candidates_low_variance_v1(...)`
* Create a new entry point:

  * `_discover_candidates_composite_v2(...)`
* Keep `discover_watermark_candidates(...)` as the public dispatcher.

### Acceptance criteria

* Existing tests still pass unchanged.
* Turning the flag off restores exact current behavior.

---

## Phase 2: Compute richer stack statistics

### Goal

Stop collapsing the stack to grayscale variance only. Build reusable per-pixel statistics in color.

### Tasks

Implement a helper such as:

```python
def compute_stack_statistics(paths: List[Path]) -> Dict[str, np.ndarray]:
    ...
```

It should compute at minimum:

* `mean_bgr`: `H x W x 3`, float32 or uint8
* `var_bgr`: `H x W x 3`, float32
* `var_gray`: `H x W`, float32
* `log_var_gray`: `H x W`, float32
* `grad_mean_gray`: gradient magnitude of grayscale version of `mean_bgr`
* optionally:

  * `grad_mean_bgr`: channelwise gradient energy
  * `edge_map`: Canny or Laplacian response on `mean_bgr`

### Implementation notes

* Keep the existing Welford-style streaming approach if memory matters.
* Upgrade it from grayscale-only variance to per-channel mean/variance.
* Normalize all score-bearing arrays into `[0, 1]` with helper functions:

  * percentile clipping
  * epsilon-safe division
  * NaN guards

### Acceptance criteria

* A debug run produces saved images for:

  * mean BGR
  * gray log-variance
  * per-channel variance summaries
  * gradient magnitude of mean image

---

## Phase 3: Replace absolute-threshold detection with a composite score

### Goal

Detect “persistent overlay structure” rather than “pixels that barely changed.”

### Tasks

Implement a helper like:

```python
def build_watermark_score(stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ...
```

Construct these components:

#### 1. Variance suppression score

Use low variance as a **relative cue**, not a hard identity test.

Candidate forms:

* `low_var = normalize(-log(var_gray + eps))`
* or `low_var = 1 - normalize(var_gray)`

#### 2. Mean-structure score

Use the mean image to capture watermark edges/glyph structure:

* `structure = normalize(gradient_magnitude(mean_gray))`
* consider combining gradient + Laplacian

#### 3. Optional chroma stability score

Useful for multicolor marks:

* low variance in chroma channels
* or consistent channel relationship across stack

#### 4. Optional border prior

Many watermarks live near edges/corners:

* distance transform from border
* convert to a soft prior, not a hard exclusion

#### 5. Final score

Start simple:

```python
score = low_var * structure
```

Then try:

```python
score = low_var * structure * border_prior
```

or a weighted blend if multiplication is too brittle.

### Acceptance criteria

* The score map visually lights up the watermark zone more cleanly than the raw low-variance mask.
* The score map still finds translucent / anti-aliased watermark edges that the old binary threshold drops.

---

## Phase 4: Convert score map into a usable support mask

### Goal

Turn a noisy score image into a clean watermark-support mask.

### Tasks

Implement:

```python
def score_to_mask(score: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    ...
```

Pipeline:

1. smooth the score map slightly

   * Gaussian blur or median blur
2. threshold adaptively

   * percentile threshold
   * Otsu on a normalized 8-bit score
   * or keep top `N` percent of pixels
3. morphological cleanup

   * close to connect glyph fragments
   * open lightly only if speckle is severe
   * fill holes
   * optional small dilation to capture anti-aliased edges

OpenCV’s `morphologyEx` is the right tool for this binary cleanup stage, and `connectedComponentsWithStats` can then filter blobs by area and location. ([OpenCV Documentation][1])

4. connected components filtering

   * reject tiny blobs
   * rank surviving blobs by:

     * area,
     * mean score,
     * border adjacency,
     * shape compactness if useful
5. merge nearby blobs if they plausibly belong to one watermark

   * especially logo + text combinations

### Acceptance criteria

* The final binary support mask covers the watermark body and its soft edges with minimal unrelated background.
* The mask is more contiguous than the current speckled low-variance result.

---

## Phase 5: Replace “masked mean crop” with actual template recovery

### Goal

Export a BGRA image that approximates the watermark itself, not just the mean image under a binary mask.

### Tasks

Implement:

```python
def recover_watermark_template(
    mean_bgr: np.ndarray,
    support_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    ...
```

Inside the bbox:

#### Step A: Estimate background-without-watermark

Use one of these approaches:

**Option 1: OpenCV-only**

* dilate the support mask
* inpaint the masked region in the mean image
* treat that as estimated background

**Option 2: Smooth background model**

* large-radius blur / bilateral / guided smoothing from outside-mask pixels

**Option 3: Optional scikit-image**

* use `rolling_ball` background estimation or `inpaint_biharmonic` if present

scikit-image documents both biharmonic inpainting and rolling-ball background estimation, so either is a reasonable optional dependency for this stage. ([Scikit-image][2])

#### Step B: Compute watermark residual

Within the bbox:

```python
residual = mean_bgr_bbox - estimated_background_bbox
```

Use:

* residual magnitude for soft alpha
* residual color for recovered RGB

#### Step C: Build soft alpha

Instead of binary alpha:

* derive alpha from normalized residual magnitude
* blend with normalized local score map if that improves stability

Example:

```python
alpha = normalize(residual_mag) * normalize(score_bbox)
```

Then rescale to `uint8 [0, 255]`.

#### Step D: Build RGB

Possible first pass:

* `rgb = clip(background + residual, 0, 255)` is redundant with mean
* better: use residual sign/direction but mask heavily by alpha
* simplest practical starting point:

  * keep `mean_bgr_bbox` where alpha is strong
  * set alpha low/zero elsewhere
* better second pass:

  * estimate watermark color from residual directly

### Acceptance criteria

* Exported BGRA has:

  * transparent non-watermark pixels,
  * soft alpha on anti-aliased edges,
  * less scene contamination than the current mean-crop export.

---

## Phase 6: Improve bounding-box selection

### Goal

Produce tight, useful crops for the ORB matcher without clipping soft edges.

### Tasks

* Replace “union of all blobs in a zone” with a scored candidate selection step.
* Rank candidates by:

  * average watermark score,
  * total alpha mass,
  * border proximity,
  * reasonable size fraction of image
* Expand bbox by configurable padding:

  * `CROP_BORDER_PX` can remain, but compute it after final support mask
* Consider merging components only if:

  * bbox gap is small,
  * score continuity exists,
  * or they share a common border-side anchor

### Acceptance criteria

* A watermark near the corner yields one bbox around the whole mark, not several fragments.
* Non-watermark invariant clutter in the same zone is less likely to hitch a ride.

---

## Phase 7: Keep and expand debug artifacts

### Goal

Make the new routine diagnosable without guessing.

### Tasks

Write debug images for every qualifying bucket:

1. `mean.png`
2. `log_variance_gray.png`
3. `low_variance_score.png`
4. `mean_structure_score.png`
5. `border_prior.png` if used
6. `composite_score.png`
7. `threshold_mask_raw.png`
8. `threshold_mask_clean.png`
9. `best_component_overlay.png`
10. `estimated_background.png`
11. `residual_magnitude.png`
12. `final_alpha.png`
13. `final_bgra_preview.png` against checkerboard

Also emit a machine-readable JSON summary:

* bucket size
* number of images loaded
* threshold values chosen
* number of components
* winning component stats
* bbox coordinates
* alpha mass / area / mean score

### Acceptance criteria

* A human can inspect one bucket folder and understand exactly why the algorithm selected that watermark.

---

## Phase 8: Add quantitative scoring hooks

### Goal

Give the agent a way to compare candidate parameter sets instead of eyeballing everything.

### Tasks

Implement simple internal metrics:

* `support_area_fraction`
* `alpha_mass`
* `mean_score_inside / mean_score_outside`
* `edge_energy_inside / outside`
* `residual_energy_inside / outside`
* optional ORB downstream fitness:

  * how often the discovered template re-localizes into held-out images

Define a provisional objective like:

* maximize score concentration and downstream re-detectability
* penalize masks that are too huge or too tiny

### Acceptance criteria

* The agent can run several parameter combinations and rank them automatically.

---

## Phase 9: Add targeted tests

### Goal

Prevent regressions and make the new method iteratable.

### Tasks

Create synthetic tests that generate stacks with known watermark truth:

1. **Opaque text watermark**
2. **Semi-transparent text watermark**
3. **Multicolor logo watermark**
4. **Logo + text composite**
5. **JPEG-compressed stack with mild artifacts**
6. **Hard negative**:

   * no watermark,
   * or invariant corner object that is not an overlay

For each synthetic case, assert bounds such as:

* bbox IoU with ground truth above threshold
* alpha support recall above threshold
* false-positive area below threshold
* exported BGRA not fully opaque rectangle
* recovered alpha is not purely binary for anti-aliased marks

### Acceptance criteria

* The new algorithm passes synthetic tests the old one would fail, especially translucent and multicolor cases.

---

## Phase 10: Parameter tuning pass

### Goal

Tune the first working version without overfitting to one image family.

### Tasks

Expose a small config object:

```python
@dataclass
class DiscoveryV2Config:
    score_blur_ksize: int
    threshold_percentile: float
    min_blob_area_fraction: float
    close_kernel_size: int
    dilate_kernel_size: int
    crop_border_px: int
    border_prior_strength: float
    alpha_gamma: float
```

Run the debug harness on several real buckets and tune only these few knobs first.

### Acceptance criteria

* One config works tolerably across several watermark styles.
* No single magic threshold is doing all the work.

---

## Suggested implementation order

1. feature-flag split
2. richer stack statistics
3. composite score
4. score-to-mask cleanup
5. bbox selection
6. debug artifact dump
7. background estimation + residual recovery
8. soft alpha export
9. tests
10. tuning

---

## Minimal first milestone

If time is tight, land this narrower version first:

* per-channel mean/variance stats
* `score = normalize(-log(var_gray)) * normalize(grad(mean_gray))`
* threshold + morphology + connected components
* export:

  * RGB = mean crop
  * alpha = **soft score-derived alpha**, not binary mask

That alone should outperform the current low-variance-only export, even before proper background subtraction.

---

## Definition of done

The work is complete when all of the following are true:

* the new path detects watermark zones that the old low-variance mask misses;
* the exported BGRA template has transparent background and a soft alpha edge;
* recovered template RGB is visibly less contaminated by scene content than the current mean-crop export;
* the ORB downstream stage can localize the recovered template on held-out images better than before;
* the debug artifacts clearly show why the detector succeeded or failed on each bucket.

---

## Deliverables the agent should produce

* modified `discovery.py`
* any new helper module if needed, e.g. `discovery_v2.py`
* tests for synthetic watermark stacks
* a debug runner script or CLI mode
* a short markdown note documenting:

  * scoring formula,
  * recovery method,
  * known failure modes,
  * parameters worth tuning

If you want, I can also turn this into a stricter “agent task sheet” with explicit file edits, function signatures, and commit-sized milestones.

[1]: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html?utm_source=chatgpt.com "Morphological Transformations"
[2]: https://scikit-image.org/docs/stable/auto_examples/filters/plot_inpaint.html?utm_source=chatgpt.com "Fill in defects with inpainting"
