# Rebuttal: Watermark Consensus Plan (2026-03-31)

## Summary of Objections

**1. Alignment is solving the wrong problem.**
Phase correlation finds translational shift between two images of the *same content*. The BGRA candidates are not the same content — they come from different spatial zones, have different sizes, and many represent qualitatively different things (scattered scene blobs vs. text). Aligning a 1250×1362 blob crop to a 160×1011 text strip is not alignment; it's noise. The only test that validates alignment uses a synthetic rectangular mask shifted 9 px — the only case the approach handles correctly.

**2. Consensus is operating at the wrong level.**
The plan takes final BGRA crops and tries to merge them. But crops from different zones are in different coordinate systems and don't spatially overlap. There is nothing to align. The average of zone (2,1) content and zone (0,0) content on a shared canvas is a blurry unintelligible mess.

**3. IRLS may converge to the noise.**
IRLS correctly down-weights outliers when the majority is signal. In the real output (roughly 6 of 9 candidates are scene-content blobs, 2–3 are text), the initial consensus is dominated by blobs. Errors get assigned backwards, and the algorithm may converge to the wrong attractor. No classification step gates which candidates are eligible for consensus.

**4. The approach is not as threshold-light as claimed.**
`delta = 0.10` in the Huber error kernel and `sigma = 1.0` in the IRLS smoothing step are hardcoded constants with significant effect on convergence. Both require justification or derivation from the data.

**5. Module placement and import paths are wrong.**
`watermark_consensus.py` at the repo root is inconsistent with the rest of the codebase under `untextre/`. The integration test uses `import discovery` which will fail — the module is at `untextre/discovery`.

---

## Reframing the Actual Problem

The plan is designed for a scenario that does not exist here: *N noisy photographs of the same watermark that need to be averaged to recover the clean template.*

The actual scenario is different in a fundamental way:

> **All candidates come from the same source images. Different candidates represent different spatial zones. Some zones contain the real watermark; others contain accidentally-stable scene content. The problem is *classification*, not *averaging*.**

No amount of sophisticated post-hoc consensus over BGRA crops fixes a pipeline that is selecting the wrong zones and outputting the wrong candidates to begin with. The correct intervention is upstream.

The useful signal is the **per-pixel variance field**, computed in image-space coordinates, before any cropping or zone selection happens. A genuine watermark pixel has near-zero variance across the full image set — and critically, across *every independent sub-sample* of that set. An accidentally-stable noise pixel may hold up in the full population but fail in one half.

The cross-correlation that matters is therefore:

- Split the N source images into K independent sub-samples
- Compute the per-pixel stable mask in each sub-sample using the same Tukey-fence threshold
- Take the intersection: retain only pixels that are statistical outliers in *all* sub-samples
- The intersection is the consensus — derived entirely from the source data, in one coordinate system, with no alignment required

This is principled because the threshold ("stable in all sub-samples") is logically justified: a true watermark pixel is identical in every image, so it is stable in every sub-sample. An incidental blotch is not guaranteed to hold up under repeated independent sampling.

If a downstream consensus stage is still wanted after this, its inputs should be restricted to same-zone candidates from the same coordinate system — not arbitrary crops from different zones. At that point the alignment problem becomes trivial (there is no shift), and simple alpha-weighted averaging suffices.

The support-mass crop from the original plan (`support_mass_bbox`) is the one idea worth salvaging; it is correct and useful independent of everything else.
