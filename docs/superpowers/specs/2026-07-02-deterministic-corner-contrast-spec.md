# Deterministic Corner Contrast Sampling

## Goal
Reduce synthetic watermark cases that are trivially invisible because the fill color and placement corner land on nearly identical background color. Keep generation deterministic, seed-driven, and fast.

## Problem
The current generator samples:
- fill color
- opacity
- corner

independently. That can produce near-zero-contrast cases such as white text on a white corner patch, or black text on a black corner patch. Those examples are not useful for the current benchmark because they inflate the hardest tier with cases that would not be representative of typical watermark placement.

## Proposed Rule
Replace unconstrained corner choice with constrained deterministic sampling over valid `(fill color class, corner)` pairs.

1. Downsample the source image to a tiny representation whose short side is about 7 px.
2. Sample each corner from that reduced image.
3. Classify each corner into a coarse contrast bucket from its local color statistics.
4. Build the set of valid `(color_class, corner)` pairs for the current sample.
5. Deterministically choose from the remaining valid pairs using the existing per-sample RNG.

No fallback search is needed. The generator assumes at least one valid pairing exists for the available fill classes, which is true by construction for the intended watermark color set.

## Contrast Model
Use simple, fast image statistics rather than a heavy perceptual model.

Recommended signals:
- relative luminance for black/white/gray separation
- Lab-space distance for vivid colors
- coarse corner classification such as `dark`, `mid`, `light`, `colorful`

This keeps the rule cheap enough for bulk generation and stable enough for reproducibility.

## Validity Rules
The generator should reject obvious low-contrast combinations:
- white fill against a light corner
- black fill against a dark corner
- gray fill against a similarly gray corner
- vivid fill against a corner that is too close in perceptual distance

The exact thresholds are empirical and should be documented in code comments when introduced.

## Metadata
Persist the following for every synthetic case:
- `outline_present`
- `outline_thickness_px`
- `outline_color_hex`
- `corner_luminance_class`
- `corner_lab_centroid` or equivalent compact summary
- `corner_choice_reason`
- `rejected_pairs_count`

Keep the existing case metadata intact:
- `text`
- `font_family`
- `font_path`
- `font_size`
- `color_class`
- `color_rgb`
- `opacity`
- `corner`
- `truth_bbox`
- `truth_alpha_coverage`
- `truth_bbox_coverage`

## Determinism
All randomness remains seed-driven.

The sequence should be:
1. derive the sample seed
2. generate the text case
3. inspect corner statistics
4. filter invalid corner/color pairs
5. sample one valid pair from the per-sample RNG

Changing image content should change the valid set, but repeated runs with the same seed and inputs must reproduce the same output.

## Tests
Add tests that verify:
- outline metadata is written
- deterministic repeated generation yields the same results
- gray text never gets an outline
- black and white text sometimes get black/white or vivid outlines per the rule
- the contrast filter rejects obvious low-contrast corner/color pairs
- metadata includes the corner-classification fields needed for later splits

## Non-Goals
- Do not add a fallback optimizer that searches for the best possible corner.
- Do not introduce logo or picture watermark modes here.
- Do not make the generator adaptively maximize contrast.
- Do not change detector behavior or replay logic in this change.

## Outcome
This change should preserve the reproducible benchmark structure while reducing useless zero-contrast synthetic watermarks. It also makes later analysis easier because contrast-relevant metadata is recorded alongside each sample.
