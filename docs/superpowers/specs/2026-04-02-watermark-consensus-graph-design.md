# Watermark Consensus Graph Design

## Goal

Replace rival near-duplicate finalists with a precision-biased consolidation stage that:

- preserves strong same-watermark variants long enough to support each other,
- avoids polluting a real watermark with one-off noise blobs,
- tolerates partial captures and subset/superset relationships,
- uses cross-family corroboration as a confidence booster, not a hard gate,
- emits conservative outputs by default, with optional generous alternates when ambiguity is mostly peripheral.

This stage operates on already-extracted watermark candidates and produces cluster-derived watermark templates.

## Problem Model

The discovery pipeline already makes family-local candidate extractions from image stacks.
Those family-level candidates can be:

- strong: substantially the full watermark, correct color, little or no peripheral junk,
- weak: some or all of the watermark plus unrelated stable blobs,
- noise: no watermark signal, but apparently stable content from background or subject repetition.

The same true watermark may appear as multiple candidates:

- within the same family,
- across different families,
- as a nearly complete crop, a conservative subset, or a noisy superset.

The current pain point is not merely false positives. It is that true variants often compete against each other instead of contributing evidence toward one consolidated output.

## Core Constraints

### Family assumptions

- Within one exact `WxH` family, watermark placement is likely automated and consistent.
- Across different families, position and scale are effectively unconstrained.
- Some runs may contain only one family; single-family outputs must remain valid.

### Watermark assumptions

- The true watermark is consistent across images.
- It may be text or image content, monochrome or full-color, opaque or semi-transparent.
- Rotation and skew are fixed at the dataset level. A “sideways” watermark is still treated as baseline orientation if it is consistently sideways everywhere.

### Safety preference

- False merges are worse than duplicate finalists.
- Noise carried into a final template can break ORB-based downstream matching and can cause erroneous masking/infill.
- Duplicate strong finalists are tolerable because downstream ORB can arbitrate among them.

## Danger-Stripe Rule

Candidate metadata is dangerous.

Metadata may be carried forward for provenance and confidence bookkeeping, but it must never invent a cross-family geometric relationship.

Allowed uses:

- provenance,
- family counting,
- confidence modulation after a content-based relationship already exists,
- debugging and traceability.

Forbidden uses:

- cross-family corner registration,
- cross-family zone-based alignment,
- assuming similar offsets or scales because two families place candidates in matching coarse zones,
- promoting a geometric match based on metadata alone.

Cross-family claims must begin with the candidate pixels themselves.

## Admissible Inferences

### Strong or primary evidence

- support-mask agreement after content-based alignment,
- soft-alpha overlap,
- edge-field or distance-transform agreement,
- corroboration of disputed regions by additional independent candidates,
- independent support from multiple families after a pixel-based relationship is already established.

### Weak or secondary evidence

- cropped aspect-ratio similarity,
- color agreement,
- color disagreement as contamination diagnostics rather than identity denial.

### Non-evidence

- cross-family zone identity,
- cross-family corner identity,
- any metadata-only geometric prior.

### Ambiguous evidence

- containment of one aligned support field inside another:
  - may indicate conservative extraction or noisy halo,
  - must not be interpreted directionally on its own.

## Recommended Architecture

Use a two-stage strategy:

1. build a precision-biased compatibility graph from pairwise content-based scoring,
2. run cluster-local pixel consensus inside each high-confidence candidate community.

This is preferred over direct rank-and-dedupe because the real task is consolidation, not simply choosing one candidate.

## Candidate Representation

Each candidate should be represented as a bundle of intrinsic fields plus dangerous metadata:

- `bgra`: original candidate crop,
- `alpha_soft`: normalized alpha channel,
- `support_mask`: thresholded support mask used for geometry,
- `edge_field`: edge or edge-likelihood image derived from support,
- `distance_field`: distance-transform-derived geometry field for robust matching,
- `bbox_shape`: crop width, height, aspect ratio,
- `provenance`: family id, source dimensions, extraction lineage, candidate index,
- `diagnostics`: support area, alpha mass, color summaries, etc.

The geometry fields are the primary matching substrate. RGB is diagnostic only.

## Pairwise Alignment

For every candidate pair, perform exhaustive content-based registration under:

- uniform scale,
- x translation,
- y translation.

No rotation and no shear.

### Scale search policy

Scale search must be generous.

Do not clip search ranges aggressively for convenience. If the only reason a hypothesis is excluded is “that scale seems unlikely,” it should probably still be tested unless it collapses the candidate to implausible tiny size or near-full-image size.

Recommended strategy:

- coarse geometric or log-scale ladder across a wide admissible range,
- retain the best few coarse scales,
- refine locally around those scales,
- keep the best alignment summary, not just a single scalar score.

Log-polar FFT scale estimation may be used as a seed, but should not be trusted as the only search path.

### Geometry substrate

Alignment should use support-derived geometry, such as:

- soft alpha,
- thresholded support mask,
- edge maps,
- distance-transform fields,
- frequency-domain summaries derived from alpha/support.

ORB/RANSAC should not be part of this stage.

## Pairwise Score

Each pairwise comparison should produce both a scalar compatibility score and a structured decomposition.

Recommended components:

- soft Dice / IoU on aligned support,
- edge-distance or chamfer-style agreement,
- Hausdorff-lite boundary penalty,
- matched-core mass,
- unmatched mass for candidate A,
- unmatched mass for candidate B,
- containment flags,
- aspect-ratio prior as a weak additive or multiplicative factor,
- color contamination diagnostics as a weak post-hoc signal.

The design must preserve the distinction between:

- a strong shared core,
- peripheral mass unique to one candidate,
- ambiguous appendages awaiting corroboration.

## Compatibility Graph

Construct a weighted graph with:

- node = candidate,
- edge = pairwise compatibility strong enough to claim “shared watermark core is plausible,”
- edge metadata = best transform and overlap decomposition.

This graph should be conservative.

### Graph policy

- weak edges are pruned early,
- bridges are treated skeptically,
- graph methods may assist but must not override edge-level evidence,
- multiple clusters are allowed when ambiguity is real.

Useful graph tools:

- thresholded compatibility graph,
- connected components or community detection,
- spectral embedding / normalized cut for exploratory grouping,
- eigenvector centrality or medoid selection for cluster anchors.

These methods are aids, not decision makers. A black-box eigensolver must not merge candidates the pairwise evidence would reject.

## Cluster-Local Consensus

Within one compatible cluster:

1. choose an anchor candidate, preferably the graph medoid or most central high-quality node,
2. warp every cluster member into anchor coordinates using its content-based transform,
3. accumulate pixel-level evidence maps.

Recommended accumulated maps:

- support frequency,
- weighted soft-alpha sum,
- edge-consistency map,
- unmatched-residual maps,
- optional color anomaly maps for contamination diagnostics.

### Consensus rule

Pixels should survive by redundant support, not by naive union.

- strong shared core should remain,
- one-off appendages should decay,
- disputed peripheral regions should stay provisional,
- a peripheral region may be promoted if a third independent candidate corroborates it.

This is the central mechanism for letting true variants reinforce each other while leaving noise unsupported.

## Output Policy

The output object should be cluster-derived templates, not raw candidates.

### Clear case

If one cluster has a stable core and no major unresolved periphery, emit one conservative template.

### Peripheral ambiguity case

If a cluster has a stable core but disputed appendages, emit two variants:

- `stingy`: only strongly corroborated support,
- `generous`: core plus moderately supported peripheral regions.

This is safe because downstream ORB matching can choose between them.

### Hard split case

If the graph truly supports two incompatible clusters, emit both cluster-derived templates separately.

## Ranking

Rank final emitted templates or cluster outputs by:

- internal cluster coherence,
- matched-core support,
- leave-one-out stability,
- number of distinct supporting families,
- unsupported peripheral mass penalty,
- optional contamination diagnostics.

Family count is a trust booster, not a requirement.

## Failure Behavior

- Single-family support remains valid.
- Cross-family corroboration increases trust when it exists.
- If no strong merge is justified, preserve multiple finalists rather than forcing a risky union.
- If a candidate appears compatible only through metadata and not through pixels, reject that relationship.

## Debug Output

The first implementation should be extremely inspectable.

Recommended debug artifacts:

- pairwise scale-search score curves,
- chosen transform per pair,
- aligned overlays for top-scoring pairs,
- matched-core vs unmatched-mass visualizations,
- graph edge list with weights,
- cluster membership summaries,
- support-frequency maps per cluster,
- stingy vs generous output previews.

The goal is to make every merge or non-merge explainable.

## Testing Strategy

Add focused tests for:

- partial-overlap same-watermark pairs,
- subset/superset ambiguity,
- same-watermark candidates with peripheral contamination,
- noise candidates with no compatible partners,
- single-family only runs,
- multi-family corroboration boosts,
- clusters that should yield stingy/generous paired outputs.

Tests should validate both pairwise scoring behavior and final cluster-derived outputs.

## Non-Goals

- no metadata-driven cross-family alignment,
- no ORB/RANSAC-based candidate consolidation,
- no assumption that one family alone defines the global placement model,
- no forced single-output behavior when evidence remains ambiguous.

## Recommended First Implementation Scope

Implement the full path in one shot:

1. candidate representation with dangerous metadata clearly separated,
2. exhaustive pairwise geometry-first scoring,
3. compatibility graph construction,
4. conservative cluster extraction,
5. cluster-local consensus support maps,
6. stingy/generous emission policy,
7. rich debug outputs,
8. focused tests that probe ambiguous and contaminated cases.

This is the smallest design that actually addresses the current failure mode of rival true variants that fail to support one another.
