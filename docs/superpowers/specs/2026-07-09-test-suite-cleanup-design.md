# Test Suite Cleanup Design

**Date:** 2026-07-09
**Scope:** `untext` pytest suite
**Basis:** Pocock-guided unit-test review in `cardinal/reviews/pocock-unit-test-review/`

## Goal

Excise tests that are not serving the repo's actual goals, while preserving or strengthening protection around real behavioral contracts.

## What "not serving our goals" means

A test is a cleanup candidate when it mainly does one of these:

1. **Plumbing / shape smoke**
   - `isinstance(...)`
   - tuple length / dict-key presence
   - handler topology / parser registration / logger wiring
   - broad "does not crash" assertions

2. **Implementation coupling**
   - patched internal collaborators with call-count / call-order assertions
   - underscore-private helper tests where the same contract can be tested at a public seam
   - cache / init / reset choreography tests that pin refactor-sensitive internals

3. **Tautology**
   - expected value recomputed with the same formula or primitive the code uses
   - arithmetic restatement tests that pass by construction

## What we keep

We keep tests that protect real, externally meaningful behavior:

- end-to-end pipeline behavior
- file / output artifact behavior
- corpus / fixture / replay regressions
- image-quality invariants (SSIM, unmasked preservation, etc.)
- exact geometry / transform worked examples
- benchmark determinism and artifact generation
- platform guarantees we care about, such as Unicode-safe image loading

## Rewrite vs delete

Default bias: **aggressive deletion**.

A weak test is rewritten instead of deleted only when deleting it would leave a real contract unprotected.

This means:
- if a stronger public-seam test already covers the contract, delete the weak one
- if the weak test is the only thing protecting a real contract, replace it with a behavior-level test before moving on

## Execution order

Use a risk-ranked cleanup order:

1. **Inpainting**
   - biggest concentration of low-value internal tests
2. **Detection**
   - weak detector internals and smoke-format checks
3. **Discovery `_consensus_vote` block**
   - concentrated internal wiring assertions
4. **CLI / logging**
   - parser-default and logger-topology oversupply
5. **Arithmetic tautology cleanup**
   - color / consensus / mask experiments
6. **Light pass on healthy clusters**
   - color / ORB / preprocess / utils
   - consensus / watermark
   - metrics / masks / benchmarks
   - pipeline / integration

## Cluster policy

### Inpainting

- Keep: `inpaint_image` public behavior, SSIM-based comparisons, paste-back behavior, masked/unmasked invariants
- Delete/rewrite: `_inpaint_with_lama`, `_calculate_inpainting_subregion`, cache-global orchestration, init/reset choreography tests that do not protect external behavior

### Detection

- Keep: evidence-row / harvest behavior, threshold behavior, real region-overlap and false-positive protections
- Delete/rewrite: detector cache/init internals, list/tuple/format smoke tests, private loader choreography

### Discovery

- Keep: strong math/spec tests
- Rewrite/delete: `_consensus_vote` tests that assert record packaging passed to internal graph-builder collaborators rather than returned behavior

### CLI / logging

- Keep: real CLI validation and output-artifact behavior
- Delete/rewrite: parser default registration, logger handler topology, exact internal logging wiring

### Arithmetic tautologies

- Replace formula-restatement tests with independent literals, worked examples, or stronger invariants

## Verification rule

After each cluster:

1. run only the touched test modules
2. confirm deleted tests were either redundant or replaced by stronger behavior tests
3. avoid broad suite runs until the end of the cleanup campaign

## Success criteria

The pass is successful when:

- the suite has fewer internal-wiring and plumbing tests
- public-seam / E2E protection remains intact
- remaining tests read more like behavioral specs than implementation diaries
- touched modules still pass their targeted verification runs

## Non-goals

- not a wholesale rewrite of the entire test suite
- not a broad architecture refactor of production code
- not "Pocock purity" for its own sake
- not replacing every deleted weak test with a new test automatically

## Practical principle

This pass should be:

**prune aggressively, rewrite selectively, preserve strong coverage**.
