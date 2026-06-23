# Watermark Anti-Kerning Splitter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split raw watermark candidates into cleaner sub-candidates before graph consensus by recursively merging nearby connected components with anisotropic horizontal/vertical expansion, then re-cropping each final group with percentage-based transparent padding.

**Architecture:** Add a preprocessing helper that works directly on BGRA candidate crops. It will detect connected components in the alpha mask, expand each component anisotropically along its bbox major axis, recursively merge overlapping groups until stable, then emit one or more BGRA sub-crops padded by a percentage of each group’s size. Integrate that helper into `untextre.discovery._consensus_vote(...)` before minimum-size filtering and `build_candidate_record(...)`.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest

---

## File Structure

- Modify: `untextre/watermark_consensus.py`
  - Add the anti-kerning splitter helper and percent-padding crop helper.
- Modify: `untextre/discovery.py`
  - Run the splitter on raw BGRA candidates before graph consensus record creation.
- Modify: `tests/test_watermark_consensus.py`
  - Add unit tests for anisotropic component merging and percent padding.
- Modify: `tests/test_discovery.py`
  - Add an integration test that one raw candidate can fan out into multiple pre-consensus records.

## Task 1: Write failing splitter tests

**Files:**
- Modify: `tests/test_watermark_consensus.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Add a failing unit test for horizontal anti-kerning merges plus remote-blob splitting**
- [ ] **Step 2: Add a failing unit test for percentage padding on emitted sub-crops**
- [ ] **Step 3: Add a failing discovery test showing one raw candidate becomes multiple consensus inputs**
- [ ] **Step 4: Run the focused test slice and verify it fails**

## Task 2: Implement the splitter helper

**Files:**
- Modify: `untextre/watermark_consensus.py`

- [ ] **Step 1: Add component-group helpers for bbox overlap and recursive anisotropic merging**
- [ ] **Step 2: Add group re-cropping with percentage-based transparent padding**
- [ ] **Step 3: Expose a helper that returns one or more BGRA sub-candidates from one raw candidate**
- [ ] **Step 4: Run the focused unit tests and verify they pass**

## Task 3: Integrate splitter into discovery

**Files:**
- Modify: `untextre/discovery.py`

- [ ] **Step 1: Apply the splitter before structural minimum checks and record creation**
- [ ] **Step 2: Preserve metadata lineage well enough for downstream debugging**
- [ ] **Step 3: Run the focused discovery test slice and verify it passes**

## Final Verification

- [ ] **Step 1: Run the focused splitter/discovery slice**

Run:

```powershell
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py tests/test_discovery.py -k "split or padding or consensus_vote" -q
```

- [ ] **Step 2: Run the relevant broader suite**

Run:

```powershell
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py tests/test_watermark_consensus_fixtures.py tests/test_discovery.py -q
```

