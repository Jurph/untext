What you want is not really “feature matching” in the ORB sense. It is **partial-overlap registration under unknown scale and translation, with heavy structured corruption**. That is why the obvious text feels matchable to a human while ORB falls over.

The clean way to think about it is:

* each candidate is a noisy, partial observation of one latent watermark
* the true watermark is the only structure that should align consistently across many candidates
* the peripheral junk is image-specific corruption, so it will not align coherently

The practical recipe is:

## 1. Register on shape/support, not on RGB

For this task, your best signal is usually not color. It is one of:

* alpha mask
* binarized support mask from alpha
* edge map of the support
* distance-transform field derived from that support

That makes the method agnostic to whether the watermark is text, logo, full-color, translucent, or mixed.

A good preprocessing stack is:

```python
alpha = rgba[..., 3].astype(np.float32) / 255.0
soft = cv2.GaussianBlur(alpha, (0, 0), 1.0)

# Optional: weak binarization only for geometry, not for final stacking
mask = (soft > np.quantile(soft[soft > 0], 0.25)).astype(np.uint8)

# Edges or a distance field tend to register better than raw translucent blobs
edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
dist  = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
geom  = np.exp(-dist / 4.0)   # soft edge-likelihood field
```

That `geom` image is often much easier to register than the original candidate.

## 2. Use direct registration, not keypoints

Since you only need **scale + translation** and can ignore rotation, the strongest families are:

### A. Multiscale phase correlation

This is the best first thing to try.

For a pair `(A, B)`:

1. search over scale on a geometric ladder, like `2**k` for `k` in a coarse range
2. resize `B` to each trial scale
3. pad both images onto a common canvas
4. estimate translation with phase correlation
5. score the alignment by overlap quality

Why this works:

* phase correlation is very good at translation
* coarse scale search is cheap if done on downsampled masks/edge fields
* it does not need “corners” or distinctive local descriptors

A skeleton looks like this:

```python
def register_pair(ref_geom, mov_geom, scales):
    best = None

    for s in scales:
        mov_s = cv2.resize(mov_geom, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        H = max(ref_geom.shape[0], mov_s.shape[0]) * 2
        W = max(ref_geom.shape[1], mov_s.shape[1]) * 2

        R = np.zeros((H, W), np.float32)
        M = np.zeros((H, W), np.float32)

        ry = (H - ref_geom.shape[0]) // 2
        rx = (W - ref_geom.shape[1]) // 2
        my = (H - mov_s.shape[0]) // 2
        mx = (W - mov_s.shape[1]) // 2

        R[ry:ry+ref_geom.shape[0], rx:rx+ref_geom.shape[1]] = ref_geom
        M[my:my+mov_s.shape[0],   mx:mx+mov_s.shape[1]]   = mov_s

        hann = cv2.createHanningWindow((W, H), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(R * hann, M * hann)

        aligned = warp_translate(M, shift)  # your own helper
        score = overlap_score(R, aligned)   # Dice / IoU / NCC on support

        candidate = (score, response, s, shift)
        if best is None or candidate > best:
            best = candidate

    return best
```

### B. Log-polar FFT magnitude, then phase correlation

This is the elegant version when the scale range is huge.

* translation does not change FFT magnitude
* scaling becomes a radial shift in log-polar coordinates

So:

1. take FFT magnitude of the geometry image
2. convert that magnitude image to log-polar
3. phase-correlate in log-polar space to get scale ratio
4. rescale
5. do ordinary phase correlation for translation

This avoids brute-forcing a wide scale range.

### C. Distance-transform chamfer matching

This is excellent when the watermark is thin or fragmented and overlap is partial.

* build edge maps
* turn one into a distance transform
* warp the other across candidate scale/translation hypotheses
* score by average distance of warped edges into the reference distance field

This is often more stable than raw overlap metrics when you have partial fragments.

In practice, I would try the methods in this order:

1. multiscale phase correlation on alpha-derived geometry
2. log-polar + phase correlation if scale range is too wide
3. chamfer matching if the watermark is sparse/thin

## 3. Reject bad candidates with a similarity graph

Do not try to force all eight into one stack.

Instead, compute pairwise registrations and turn them into a graph:

* node = candidate image
* edge = “these two align well under some scale+translation”
* edge weight = registration confidence

A useful edge score is a product of:

* phase correlation peak strength
* post-warp Dice/IoU on support
* symmetry check: `A->B` and `B->A` agree on scale/shift
* overlap fraction after warp

Then:

* keep the **largest mutually consistent component**
* reject isolated nodes and weakly attached nodes
* optionally require basic cycle consistency, e.g. `T_ac ≈ T_ab ∘ T_bc`

This is how you satisfy your “not all candidates are useful” constraint cleanly. The junk ones usually fail to form a strong consistent cluster.

A robust rule is better than a hand-picked threshold:

* compute all edge weights
* use median and MAD
* keep edges above `median + k*MAD`
* or cluster edge weights and keep the high-confidence mode

That gives you thresholding driven by the data, not thumb-suck constants.

## 4. Build the latent watermark by consensus, not by naïve summing

Once you have a consistent subset and transforms into one anchor frame:

1. pick an anchor image, ideally the graph medoid
2. warp every accepted candidate into anchor coordinates
3. accumulate a few maps

Useful maps:

### Support frequency map

For each pixel, count how many warped candidates say “something is here”.

```python
support += (warped_alpha > 0).astype(np.float32)
```

True watermark structure rises sharply here. Peripheral junk does not.

### Weighted alpha sum

```python
alpha_sum += weight_i * warped_alpha
weight_sum += weight_i
mean_alpha = alpha_sum / np.maximum(weight_sum, 1e-6)
```

### Robust per-pixel estimator

Better than a mean:

* weighted median
* trimmed mean
* Huber mean

These prevent one weird candidate from smearing the consensus.

## 5. Crop out gibberish by persistence across candidates

This is the key step.

After stacking, the true watermark is not just “bright”; it is **persistent**. The junk is low-frequency across the set.

So derive the crop from the support frequency map, not the raw sum.

A strong method is:

1. compute support frequency `F(x)`
2. examine connected components across several levels of `F`
3. keep the component that is stable over the widest range of levels

That avoids a brittle fixed threshold and naturally selects the repeated structure.

A simpler version:

* normalize `F` to `[0,1]`
* threshold at several quantiles, say 50%, 60%, 70%, 80%
* find connected components at each level
* keep the component that persists and overlaps itself across levels
* take its union or its most stable core as the consensus crop

This is much better than saying “keep anything seen in 4/8 images”, because your useful set may only be 3 or 5.

## What I would actually implement

Given your assumptions, I would build exactly this:

### Stage 1: Geometry extraction

From RGBA candidate:

* use alpha
* make a soft support image
* make an edge/distance-field image for registration

### Stage 2: Pairwise alignment

For every pair:

* estimate scale with either:

  * coarse geometric scale ladder, or
  * log-polar FFT magnitude
* estimate translation with phase correlation
* compute final score with Dice/IoU on support and maybe chamfer distance

### Stage 3: Candidate pruning

* build weighted graph
* keep largest high-confidence connected component
* discard the rest

### Stage 4: Consensus registration

* choose medoid anchor
* compose transforms into anchor frame
* optionally refine each candidate against the emerging consensus template

### Stage 5: Robust stack

* accumulate support frequency
* accumulate weighted alpha/color
* derive stable connected component from frequency persistence
* crop to that component’s bounding box

## Why this works better than “register lower-left corners” or “square-normalize”

Because you are not imposing an arbitrary image frame. You are estimating the latent transform from the repeated structure itself.

The true watermark is the only thing that will:

* align under one similarity transform family
* recur across several candidates
* survive graph consistency checks
* survive support-frequency persistence after stacking

The gibberish fails at least one of those.

## One important tactical note

ORB failing here is not really a warning sign about the whole problem. It mostly means:

* the common structure is too sparse / too smooth / too translucent
* there are not enough repeatable keypoints
* local descriptors are the wrong primitive

For watermarks, **dense registration on support/edge fields** is usually the right primitive.

## The shortest practical answer

Use:

* **alpha-derived geometry**
* **multiscale phase correlation** for translation+scale registration
* **pairwise similarity graph** to reject bad candidates
* **support-frequency consensus map** to isolate the repeated structure
* **stable connected-component crop** to remove peripheral junk

That gives you a consensus watermark without assuming text, opacity, logo shape, or fixed placement.

If you want to push this into an existing pipeline with minimal new machinery, the best single starting point is:

**edge/distance-transform preprocessing + coarse scale search + phase correlation + support-frequency consensus.**
