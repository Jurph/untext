# Watermark Scale Selector Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a permissive component-driven scale proposal layer that seeds pairwise watermark registration with likely scales and lightweight bbox metadata, while also retaining strong singleton outputs and enforcing a structural discovery floor.

**Architecture:** Extend `untextre.watermark_consensus` with connected-component descriptors for both full-support and filtered-scoring views, produce weighted log-scale proposal peaks from those descriptors, and thread proposal metadata into pairwise scoring without letting it decide identity on its own. Keep final confirmation in the existing 2D scale+translation overlap path, preserve the broad fallback scale ladder, and make `build_final_templates(...)` retain uncovered singleton candidates instead of dropping them when any cluster exists.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest

---

## File Structure

- Modify: `untextre/watermark_consensus.py`
  - Add connected-component descriptors, scale proposal helpers, proposal metadata, pairwise integration, and singleton retention.
- Modify: `untextre/discovery.py`
  - Keep the structural lower-bound gate for discovery candidates and expose the stronger edge/fill floor.
- Modify: `tests/test_watermark_consensus.py`
  - Add unit tests for scale proposals, proposal metadata, and singleton retention.
- Modify: `tests/test_discovery.py`
  - Add/keep discovery-level tests for the structural candidate floor.

## Chunk 1: Proposal Scales And Metadata

### Task 1: Add scale proposal tests first

**Files:**
- Modify: `tests/test_watermark_consensus.py`

- [ ] **Step 1: Write a failing test for component-driven scale proposals**
- [ ] **Step 2: Run the focused pytest command and verify it fails for the expected reason**
- [ ] **Step 3: Write a failing test that the best pairwise match carries proposal bbox metadata**
- [ ] **Step 4: Run the focused pytest command and verify it fails for the expected reason**

### Task 2: Implement component descriptors and scale proposal helpers

**Files:**
- Modify: `untextre/watermark_consensus.py`

- [ ] **Step 1: Add component/scale-proposal dataclasses**
- [ ] **Step 2: Build top-N component descriptors for full-support and filtered-scoring views**
- [ ] **Step 3: Add log-scale proposal voting from component size ratios plus soft shape compatibility**
- [ ] **Step 4: Store lightweight bbox metadata with each proposal peak**
- [ ] **Step 5: Run the focused pytest command and verify the new tests pass**

## Chunk 2: Pairwise Integration And Output Safety

### Task 3: Integrate proposal scales into pairwise scoring without removing the fallback ladder

**Files:**
- Modify: `untextre/watermark_consensus.py`

- [ ] **Step 1: Feed proposal scales into the pairwise search ahead of the broad fallback ladder**
- [ ] **Step 2: Preserve broad fallback coverage and local refinement**
- [ ] **Step 3: Propagate proposal-origin metadata onto the best `PairwiseScore` when applicable**
- [ ] **Step 4: Run the focused pytest command and verify pairwise tests pass**

### Task 4: Keep uncovered singleton candidates in the final output

**Files:**
- Modify: `untextre/watermark_consensus.py`
- Modify: `tests/test_watermark_consensus.py`

- [ ] **Step 1: Use the existing failing singleton-retention test as the red test**
- [ ] **Step 2: Modify `build_final_templates(...)` to append uncovered singleton templates when clusters exist**
- [ ] **Step 3: Sort outputs conservatively so large plausible singletons are not eclipsed by tiny cluster artifacts**
- [ ] **Step 4: Run the focused pytest command and verify singleton retention passes**

## Chunk 3: Discovery Floor Verification

### Task 5: Verify and keep the structural discovery floor

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Use the existing tiny/sparse-spray discovery tests as the red/green checks**
- [ ] **Step 2: Keep the area, bbox, edge-count, and fill-ratio gate intact**
- [ ] **Step 3: Run the focused discovery pytest command and verify it passes**

## Final Verification

- [ ] **Step 1: Run the focused consensus/discovery pytest slice**

Run:

```powershell
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py tests/test_discovery.py -k "proposal or singleton or sparse_spray or tiny_candidates or consensus_vote" -q
```

- [ ] **Step 2: Run the relevant broader suite**

Run:

```powershell
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py tests/test_watermark_consensus_fixtures.py tests/test_discovery.py -q
```

