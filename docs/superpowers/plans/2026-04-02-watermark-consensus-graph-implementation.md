# Watermark Consensus Graph Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace rival near-duplicate watermark finalists with a precision-biased graph-and-consensus stage that consolidates compatible candidates while keeping ambiguous generous/stingy alternates separate.

**Architecture:** Add a dedicated `untextre.watermark_consensus` module that accepts already-extracted BGRA candidates plus dangerous provenance metadata, performs exhaustive pairwise geometry-first scoring under scale+translation, builds a conservative compatibility graph, and emits cluster-derived consensus templates. Integrate it into `untextre.discovery` as the replacement for the current corner-anchored `_consensus_vote`, while preserving the existing family extraction path and `crop_zone_to_bgra` utilities.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest

---

## File Structure

- Create: `untextre/watermark_consensus.py`
  - Single responsibility: candidate dataclasses, pairwise geometry scoring, compatibility graph construction, cluster-local consensus, and stingy/generous emission.
- Modify: `untextre/discovery.py`
  - Single responsibility change: replace the current `_consensus_vote` implementation with a wrapper around `untextre.watermark_consensus`, while keeping family extraction and candidate cropping intact.
- Create: `tests/test_watermark_consensus.py`
  - Single responsibility: deterministic unit tests for pairwise scoring, graph clustering, and cluster-local consensus.
- Modify: `tests/test_discovery.py`
  - Single responsibility change: update discovery-facing integration tests to assert graph-consensus behavior instead of corner anchoring assumptions.
- Create: `tests/test_watermark_consensus_fixtures.py`
  - Single responsibility: regression tests over `tests/images/fixtures/watermark_candidate*.png` and their derived alpha outputs.

Keep the new graph logic out of `discovery.py`. That file is already large and should remain focused on family extraction. The new module should expose a narrow interface that accepts extracted candidates and returns final BGRA templates plus diagnostics.

## Chunk 1: Candidate Model And Pairwise Geometry Scoring

### Task 1: Add the candidate dataclasses and geometry helpers

**Files:**
- Create: `untextre/watermark_consensus.py`
- Create: `tests/test_watermark_consensus.py`

- [ ] **Step 1: Write the failing dataclass and helper tests**

Add tests that pin down the candidate model and geometry helpers:

```python
from dataclasses import is_dataclass

import numpy as np

from untextre.watermark_consensus import (
    CandidateGeometry,
    CandidateMetadata,
    CandidateRecord,
    alpha_to_soft_mask,
    build_distance_field,
    build_edge_field,
)


def test_candidate_record_separates_pixels_from_metadata():
    assert is_dataclass(CandidateRecord)
    assert is_dataclass(CandidateGeometry)
    assert is_dataclass(CandidateMetadata)


def test_alpha_to_soft_mask_normalizes_uint8_alpha():
    alpha = np.array([[0, 128, 255]], dtype=np.uint8)
    soft = alpha_to_soft_mask(alpha)
    assert soft.dtype == np.float32
    assert np.isclose(soft[0, 0], 0.0)
    assert np.isclose(soft[0, 2], 1.0)


def test_geometry_helpers_return_nonempty_fields_for_simple_blob():
    alpha = np.zeros((40, 60), dtype=np.uint8)
    alpha[10:30, 20:45] = 255
    edge = build_edge_field(alpha)
    dist = build_distance_field(alpha)
    assert edge.shape == alpha.shape
    assert dist.shape == alpha.shape
    assert float(edge.max()) > 0.0
    assert float(dist.max()) > 0.0
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "candidate_record or alpha_to_soft_mask or geometry_helpers" -q
```

Expected: import failure because `untextre.watermark_consensus` does not exist yet.

- [ ] **Step 3: Write the minimal model and helper implementation**

Create `untextre/watermark_consensus.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateMetadata:
    family_key: tuple[int, int]
    zone: tuple[int, int]
    candidate_index: int
    source_kind: str


@dataclass(frozen=True)
class CandidateGeometry:
    alpha_soft: np.ndarray
    support_mask: np.ndarray
    edge_field: np.ndarray
    distance_field: np.ndarray


@dataclass(frozen=True)
class CandidateRecord:
    bgra: np.ndarray
    geometry: CandidateGeometry
    metadata: CandidateMetadata


def alpha_to_soft_mask(alpha_u8: np.ndarray) -> np.ndarray:
    if alpha_u8.ndim != 2:
        raise ValueError("alpha_to_soft_mask expects HxW alpha")
    return (alpha_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def build_edge_field(alpha_u8: np.ndarray) -> np.ndarray:
    soft = alpha_to_soft_mask(alpha_u8)
    blurred = cv2.GaussianBlur(soft, (0, 0), 1.0)
    thresh = float(np.quantile(blurred[blurred > 0], 0.25)) if np.any(blurred > 0) else 0.0
    mask = (blurred > thresh).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 150)
    return edges.astype(np.float32) / 255.0


def build_distance_field(alpha_u8: np.ndarray) -> np.ndarray:
    edges = (build_edge_field(alpha_u8) * 255).astype(np.uint8)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return np.exp(-dist / 4.0).astype(np.float32)
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "candidate_record or alpha_to_soft_mask or geometry_helpers" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add untextre/watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: add watermark consensus candidate model"
```

### Task 2: Add exhaustive pairwise scoring under scale+translation

**Files:**
- Modify: `untextre/watermark_consensus.py`
- Modify: `tests/test_watermark_consensus.py`

- [ ] **Step 1: Write failing pairwise-scoring tests**

Add tests for:

- partial-overlap same-watermark pairs,
- noisy superset vs conservative subset,
- unrelated noise candidates.

```python
def test_pairwise_score_prefers_related_partial_candidates():
    a = make_rect_candidate(width=120, height=60, dx=0, dy=0, extra_blob=None)
    b = make_rect_candidate(width=90, height=45, dx=12, dy=6, extra_blob=None)
    result = score_candidate_pair(a, b)
    assert result is not None
    assert result.compatibility > 0.5
    assert result.shared_core_score > result.unmatched_a_penalty
    assert result.shared_core_score > result.unmatched_b_penalty


def test_pairwise_score_flags_unrelated_noise_as_incompatible():
    a = make_rect_candidate(width=120, height=60, dx=0, dy=0, extra_blob=None)
    b = make_noise_candidate(seed=7)
    result = score_candidate_pair(a, b)
    assert result is not None
    assert result.compatibility < 0.2
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "pairwise_score" -q
```

Expected: failure because `score_candidate_pair` and its result model do not exist.

- [ ] **Step 3: Implement the minimal pairwise scoring path**

Extend `untextre/watermark_consensus.py` with:

- `PairwiseScore` dataclass,
- wide coarse-to-fine scale search,
- translation estimation on support-derived geometry,
- overlap decomposition into shared core / unmatched A / unmatched B,
- a compatibility scalar that is high only when shared support dominates penalties.

Use only:

- soft-alpha overlap,
- support-mask overlap,
- edge/distance-field agreement,
- weak aspect-ratio prior.

Do not use metadata or color in this scoring function.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "pairwise_score" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add untextre/watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: add pairwise watermark compatibility scoring"
```

## Chunk 2: Compatibility Graph And Cluster Consensus

### Task 3: Build the conservative compatibility graph

**Files:**
- Modify: `untextre/watermark_consensus.py`
- Modify: `tests/test_watermark_consensus.py`

- [ ] **Step 1: Write failing graph-construction tests**

Add tests that create:

- one coherent three-node cluster plus one isolated noise node,
- one ambiguous bridge that should be excluded by conservative pruning.

```python
def test_build_candidate_graph_isolates_noise_node():
    records = [make_related_record(i) for i in range(3)] + [make_noise_record(99)]
    graph = build_candidate_graph(records)
    assert graph.weights.shape == (4, 4)
    assert graph.weights[0, 1] > 0.0
    assert graph.weights[0, 3] == 0.0


def test_extract_candidate_clusters_returns_only_high_confidence_group():
    records = [make_related_record(i) for i in range(3)] + [make_noise_record(99)]
    graph = build_candidate_graph(records)
    clusters = extract_candidate_clusters(graph)
    assert len(clusters) == 1
    assert len(clusters[0].member_indices) == 3
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "candidate_graph or extract_candidate_clusters" -q
```

Expected: failure because graph helpers do not exist.

- [ ] **Step 3: Implement the graph layer**

Add:

- `CandidateGraph` dataclass,
- `ClusterRecord` dataclass,
- `build_candidate_graph(records)`,
- `extract_candidate_clusters(graph)`.

Rules:

- compute the full pairwise matrix,
- prune weak edges,
- keep bridges skeptical,
- allow multiple clusters,
- never use metadata to create edges.

If spectral helpers are used for ordering or diagnostics, guard them with explicit edge thresholds so they cannot create memberships unsupported by pairwise scores.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "candidate_graph or extract_candidate_clusters" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add untextre/watermark_consensus.py tests/test_watermark_consensus.py
git commit -m "feat: add conservative watermark compatibility graph"
```

### Task 4: Implement cluster-local consensus with stingy/generous variants

**Files:**
- Modify: `untextre/watermark_consensus.py`
- Modify: `tests/test_watermark_consensus.py`
- Create: `tests/test_watermark_consensus_fixtures.py`

- [ ] **Step 1: Write failing consensus tests**

Add synthetic tests for:

- stable core with one disputed appendage,
- third-candidate corroboration promoting the appendage,
- stingy/generous dual output when the periphery remains unresolved.

```python
def test_consensus_emits_stingy_and_generous_when_periphery_is_disputed():
    records = make_disputed_periphery_records()
    outputs = build_cluster_outputs(records)
    assert len(outputs) == 2
    assert outputs[0].label == "stingy"
    assert outputs[1].label == "generous"
    assert outputs[1].alpha_mass > outputs[0].alpha_mass


def test_consensus_promotes_appendage_when_third_candidate_supports_it():
    records = make_triply_supported_appendage_records()
    outputs = build_cluster_outputs(records)
    assert len(outputs) == 1
    assert outputs[0].label == "consensus"
```

Also add a fixture-backed regression in `tests/test_watermark_consensus_fixtures.py` that loads:

- `tests/images/fixtures/watermark_candidate.png`
- `tests/images/fixtures/watermark_candidate_2.png`
- `tests/images/fixtures/watermark_candidate_3.png`
- `tests/images/fixtures/watermark_candidate_4.png`
- `tests/images/fixtures/watermark_candidate_5.png`
- `tests/images/fixtures/watermark_candidate_6.png`

and asserts that:

- at least one nontrivial compatibility cluster forms,
- the top cluster emits either one consensus or a stingy/generous pair,
- the emitted template count is smaller than the raw candidate count.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "consensus_emits or promotes_appendage" -q
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus_fixtures.py -q
```

Expected: failures because the cluster-consensus functions and fixture regression do not exist yet.

- [ ] **Step 3: Implement cluster-local consensus**

Add:

- anchor selection by cluster medoid or central node,
- warping of every member into anchor coordinates,
- support-frequency map,
- weighted soft-alpha sum,
- optional contamination diagnostics from RGB residuals,
- conservative mask extraction for the shared core,
- generous-mask expansion for moderately supported disputed pixels,
- final BGRA emission with labels `consensus`, `stingy`, or `generous`.

Keep color far down the priority list. Color may only demote suspicious appendages or explain contamination; it must not overturn a strong geometric match.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "consensus_emits or promotes_appendage" -q
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus_fixtures.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add untextre/watermark_consensus.py tests/test_watermark_consensus.py tests/test_watermark_consensus_fixtures.py
git commit -m "feat: add cluster-local watermark consensus outputs"
```

## Chunk 3: Discovery Integration And Regression Coverage

### Task 5: Replace `_consensus_vote` with the graph-consensus wrapper

**Files:**
- Modify: `untextre/discovery.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing discovery integration tests**

Add tests that prove:

- same-family different-zone candidates remain separate inputs to the graph layer,
- cross-family compatible candidates can consolidate into one output,
- ambiguous periphery can yield stingy/generous outputs,
- single-family runs still emit valid outputs.

Example:

```python
def test_discovery_graph_consensus_replaces_corner_anchored_vote(monkeypatch):
    zone_data = make_zone_data_for_two_family_match()

    def fake_build_final_templates(records, debug_dir=None):
        assert len(records) == 2
        return [make_output_bgra("consensus")]

    monkeypatch.setattr("untextre.discovery.build_final_templates", fake_build_final_templates)
    outputs = _consensus_vote(zone_data)
    assert len(outputs) == 1
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_discovery.py -k "graph_consensus or consensus_vote" -q
```

Expected: failure because discovery still owns the corner-anchored vote logic.

- [ ] **Step 3: Implement the wrapper integration**

Modify `untextre/discovery.py` so that `_consensus_vote`:

- converts zone tuples into `CandidateRecord` values,
- preserves dangerous provenance metadata,
- delegates to `untextre.watermark_consensus.build_final_templates(...)`,
- returns one or more BGRA outputs from that module.

Delete the old corner-anchored overlay logic rather than leaving two competing implementations in place.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_discovery.py -k "graph_consensus or consensus_vote" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add untextre/discovery.py tests/test_discovery.py
git commit -m "feat: integrate graph-based watermark consensus into discovery"
```

### Task 6: Add debug artifacts and run the full regression surface

**Files:**
- Modify: `untextre/watermark_consensus.py`
- Modify: `tests/test_watermark_consensus.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing debug-output tests**

Add tests that assert debug mode writes:

- pairwise score summaries,
- aligned-overlap previews,
- cluster membership summaries,
- support-frequency maps,
- stingy/generous previews.

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py -k "debug" -q
```

Expected: failure because the debug writer does not yet exist or is incomplete.

- [ ] **Step 3: Implement the debug writer**

Add a debug-output helper that writes:

- one JSON or text summary per pair,
- one preview image per accepted cluster,
- one support-frequency visualization,
- one stingy preview and one generous preview when both exist.

Keep file naming deterministic so fixture-based debugging is stable across runs.

- [ ] **Step 4: Run the full regression surface**

Run:

```bash
.\.codex-run\Scripts\python.exe -m pytest tests/test_watermark_consensus.py tests/test_watermark_consensus_fixtures.py tests/test_discovery.py -q
.\.codex-run\Scripts\python.exe -m pytest tests -m "not slow"
```

Expected:

- focused consensus tests pass,
- full project test suite passes from the `tests/` path.

- [ ] **Step 5: Commit**

```bash
git add untextre/watermark_consensus.py tests/test_watermark_consensus.py tests/test_watermark_consensus_fixtures.py tests/test_discovery.py
git commit -m "feat: add debugable graph-based watermark consensus"
```

Plan complete and saved to `docs/superpowers/plans/2026-04-02-watermark-consensus-graph-implementation.md`. Ready to execute?
