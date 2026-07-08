# Deterministic Corner Contrast Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make synthetic watermark generation avoid trivially low-contrast corner placements while staying deterministic and seed-driven.

**Architecture:** Keep the change inside `untextre/synthetic_text_benchmark.py` and the related tests. Add a tiny corner-sampling helper that downscales the source image, classifies each corner coarsely, and filters invalid fill/corner pairs before placement. Persist the chosen corner statistics in metadata so later benchmark splits can inspect how the generator behaved.

**Tech Stack:** Python, Pillow, OpenCV, numpy, pytest

---

### Task 1: Add deterministic corner sampling

**Files:**
- Modify: `untextre/synthetic_text_benchmark.py`
- Test: `tests/test_generated_text_cases.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run the focused test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run the test to verify it passes**
- [ ] **Step 5: Commit**

### Task 2: Persist corner metadata

**Files:**
- Modify: `untextre/synthetic_text_benchmark.py`
- Modify: `untextre/inmemory_watermark_analysis.py`
- Test: `tests/test_inmemory_watermark_benchmark.py`

- [ ] **Step 1: Extend the analysis splits**
- [ ] **Step 2: Verify the split buckets with a unit test**
- [ ] **Step 3: Run the focused tests**
- [ ] **Step 4: Commit**

