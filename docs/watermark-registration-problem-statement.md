# Watermark Registration Problem Statement

## Context

`untext -U` mode processes a directory of images hypothesized to share a common
watermark. The pipeline produces one or more BGRA candidate crops per image-size
bucket. The final step — currently broken or absent — must take those candidates,
register them to each other, and derive a consensus watermark from their overlap.

## Guaranteed Invariants

1. **Same-bucket consistency.** All images of the same pixel dimensions received
   the same watermark applied in the same way. Overlaying them in pixel space yields
   a consistently watermarked region. The per-bucket statistics (variance, mean,
   median gradient) are therefore valid estimators of the watermark's properties
   within that bucket.

2. **Shape and color invariance across buckets.** The watermark graphic is the same
   object regardless of which bucket it came from. Its shape (topology, internal
   structure) and color palette do not change between image sizes.

3. **Scale and position may differ across buckets.** We make no assumption that a
   watermark at offset (x, y) in a 1200×900 image occupies the same offset or
   fractional position in a 2400×1600 image. Different image sizes may have had the
   watermark placed differently — different corner, different padding, different scale.

## What We Do Not Know A Priori

- Whether the watermark was applied uniformly to different buckets (in fact, we know that it was NOT)
- We cannot assume corner registration or fixed offsets 
- We do not know the watermark's position or offset within any given bucket
- Whether the watermark is connected or disconnected (text is disconnected; a logo
  may be a solid shape)
- The watermark's maximum size (no upper bound other than the image itself)
- Whether the watermark appears in every bucket (some buckets may be watermark-free)
- The watermark's opacity (may be fully opaque, semi-transparent, or anything between)

## Watermark Appearance

The watermark may be any of the following, singly or in combination:

- Matte (solid color, fully opaque)
- Semi-transparent overlay
- Vivid multi-color image or photograph
- Text (any font, size, script)
- Logotype or wordmark
- Script / handwritten style
- Block lettering
- Geometric shape (star, rectangle, circle, etc.)

No assumption of spatial connectedness is valid.

## The Registration Problem

After per-bucket extraction, we have N BGRA candidate crops. Each crop shows a
watermark-shaped alpha mask floating in transparency, with estimated watermark
colors in the RGB channels. These crops:

- Are in different coordinate spaces (different source image sizes)
- May be at different scales relative to the watermark's true size
- Show the same underlying graphic (invariant shape and color)

The goal is to find an **affine transform without rotation or shear** (i.e., a
**similarity transform restricted to translation and uniform scale**) that maps
each candidate crop onto a common canvas such that the watermark features align.
Taking the overlap (logical AND of alpha channels, or pixel-wise minimum) of all
registered crops then yields the consensus watermark.

### Degrees of Freedom

The transform between any two candidates has exactly two free parameters:
- **Scale** (uniform — no aspect ratio change, no shear, no rotation)
- **Translation** (x, y offset on the common canvas)

Rotation is excluded because watermarks are applied upright.
Shear is excluded because watermarks are not distorted by the application process.

### Bounding Constraints on Scale

A candidate watermark from bucket A will never need to be scaled such that its
width exceeds the original width of bucket B's images (and vice versa for height).
A candidate will never need to be scaled below approximately 10 pixels in its
smallest dimension.

These bounds define the search space for scale during registration.

### Registration Method

The appropriate mathematical framework is **partial congruency** / **feature-based
registration** under a restricted affine model (scale + translation only).

Candidate approaches:

1. **Feature-point matching** (ORB, SIFT, or similar) on the alpha-masked regions,
   followed by RANSAC to find the consensus (scale, tx, ty) that maps the most
   feature pairs into agreement. The alpha mask restricts keypoint extraction to
   the watermark region only, avoiding background contamination.

2. **Phase correlation / log-polar spectrum** for scale estimation, followed by
   normalized cross-correlation for translation. Works well when the crops are
   clean enough to have meaningful frequency content.

3. **Iterative closest point on the alpha boundary** — treat the alpha boundary as
   a point set and find the scale + translation that minimizes boundary-point
   distance. Rotation-free ICP is a degenerate but well-posed case.

The preferred approach should be robust to partial captures (one crop may show only
70% of the watermark pixels that another shows) and to mild differences in alpha
coverage caused by varying background content across buckets.

### Consensus After Registration

Once all candidates are registered to a common canvas:

- **Overlap mask**: keep pixels where ≥ 2 registered alpha masks agree (vote ≥ 2).
  This is now valid because registration has placed the same watermark features at
  the same canvas coordinates — the "count ≥ 2" criterion is applied in a coordinate
  space where it actually means something.

- **Color consensus**: average (or take the highest-confidence estimate of) the RGB
  values at each surviving pixel across all registered crops.

## What This Document Is Not

This is not a specification for the full `untext -U` pipeline. It covers only the
final registration-and-consensus step that takes already-extracted BGRA candidate
crops and produces a clean watermark template.

The per-bucket extraction steps (Welford statistics, Tukey fence stable mask,
median gradient scoring, zone selection) are described separately and are considered
correct for their inputs.
