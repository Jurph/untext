from dataclasses import is_dataclass
import logging

import numpy as np

from untextre.watermark_consensus import (
    ComponentDescriptor,
    CandidateGeometry,
    CandidateGraph,
    ClusterRecord,
    CandidateMetadata,
    CandidateRecord,
    ConsensusTemplate,
    PairwiseScore,
    ScaleProposal,
    alpha_to_soft_mask,
    build_candidate_graph,
    build_candidate_record,
    build_distance_field,
    build_edge_field,
    build_final_templates,
    extract_candidate_clusters,
    propose_pair_scales,
    score_candidate_pair,
    split_candidate_bgra,
)


def make_rect_bgra(
    height: int = 120,
    width: int = 180,
    rect_h: int = 44,
    rect_w: int = 96,
    dx: int = 0,
    dy: int = 0,
    extra_blob: tuple[int, int, int, int] | None = None,
    color: tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    y0 = 28 + dy
    x0 = 42 + dx
    y1 = y0 + rect_h
    x1 = x0 + rect_w
    bgra[y0:y1, x0:x1, :3] = color
    bgra[y0:y1, x0:x1, 3] = 255
    if extra_blob is not None:
        by, bx, bh, bw = extra_blob
        bgra[by:by + bh, bx:bx + bw, :3] = (40, 40, 220)
        bgra[by:by + bh, bx:bx + bw, 3] = 255
    return bgra


def make_noise_bgra(height: int = 120, width: int = 180, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    for _ in range(8):
        y = int(rng.randint(0, height - 10))
        x = int(rng.randint(0, width - 10))
        h = int(rng.randint(4, 12))
        w = int(rng.randint(4, 12))
        bgra[y:y + h, x:x + w, :3] = rng.randint(0, 255, size=(3,), dtype=np.uint8)
        bgra[y:y + h, x:x + w, 3] = 255
    return bgra


def make_record(
    bgra: np.ndarray,
    family_key: tuple[int, int] = (700, 500),
    zone: tuple[int, int] = (2, 0),
    candidate_index: int = 0,
) -> CandidateRecord:
    return build_candidate_record(
        bgra=bgra,
        metadata=CandidateMetadata(
            family_key=family_key,
            zone=zone,
            candidate_index=candidate_index,
            source_kind="test",
        ),
    )


def make_related_record(index: int) -> CandidateRecord:
    variants = [
        dict(rect_h=46, rect_w=98, dx=0, dy=0),
        dict(rect_h=42, rect_w=90, dx=7, dy=5),
        dict(rect_h=40, rect_w=86, dx=12, dy=8),
    ]
    return make_record(
        make_rect_bgra(**variants[index]),
        family_key=(720 + index * 11, 540 + index * 7),
        candidate_index=index,
    )


def make_noise_record(seed: int) -> CandidateRecord:
    return make_record(
        make_noise_bgra(seed=seed),
        family_key=(640 + seed, 480 + seed),
        candidate_index=seed,
    )


def make_tiny_record(
    dx: int = 0,
    dy: int = 0,
    family_key: tuple[int, int] = (700, 500),
    candidate_index: int = 0,
) -> CandidateRecord:
    return make_record(
        make_rect_bgra(height=60, width=60, rect_h=6, rect_w=8, dx=dx, dy=dy),
        family_key=family_key,
        candidate_index=candidate_index,
    )


def make_bridge_record() -> CandidateRecord:
    bridge = make_rect_bgra(rect_h=28, rect_w=56, dx=22, dy=16, extra_blob=(6, 126, 40, 38))
    return make_record(
        bridge,
        family_key=(812, 612),
        candidate_index=9,
    )


def make_multi_component_bgra(
    components: list[tuple[int, int, int, int]],
    canvas_shape: tuple[int, int] = (220, 220),
    color: tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    height, width = canvas_shape
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    for x, y, w, h in components:
        bgra[y:y + h, x:x + w, :3] = color
        bgra[y:y + h, x:x + w, 3] = 255
    return bgra


def _offset_blob(blob: tuple[int, int, int, int], dx: int, dy: int) -> tuple[int, int, int, int]:
    by, bx, bh, bw = blob
    return by + dy, bx + dx, bh, bw


def make_disputed_periphery_records() -> list[CandidateRecord]:
    blob = (86, 136, 14, 18)
    variants = [
        dict(dx=0, dy=0, extra_blob=None),
        dict(dx=8, dy=5, extra_blob=None),
        dict(dx=12, dy=7, extra_blob=_offset_blob(blob, dx=12, dy=7)),
    ]
    return [
        make_record(
            make_rect_bgra(rect_h=44, rect_w=96, **variant),
            family_key=(760 + index * 13, 520 + index * 17),
            candidate_index=index,
        )
        for index, variant in enumerate(variants)
    ]


def make_triply_supported_appendage_records() -> list[CandidateRecord]:
    blob = (86, 136, 14, 18)
    variants = [
        dict(dx=0, dy=0),
        dict(dx=7, dy=5),
        dict(dx=12, dy=8),
    ]
    return [
        make_record(
            make_rect_bgra(
                rect_h=44,
                rect_w=96,
                dx=variant["dx"],
                dy=variant["dy"],
                extra_blob=_offset_blob(blob, dx=variant["dx"], dy=variant["dy"]),
            ),
            family_key=(804 + index * 19, 612 + index * 11),
            candidate_index=20 + index,
        )
        for index, variant in enumerate(variants)
    ]


def test_candidate_record_separates_pixels_from_metadata():
    assert is_dataclass(CandidateRecord)
    assert is_dataclass(CandidateGeometry)
    assert is_dataclass(CandidateMetadata)
    assert is_dataclass(ComponentDescriptor)
    assert is_dataclass(ScaleProposal)
    assert is_dataclass(PairwiseScore)
    assert is_dataclass(CandidateGraph)
    assert is_dataclass(ClusterRecord)
    assert is_dataclass(ConsensusTemplate)


def test_consensus_module_logger_is_configured_for_info_output():
    import untextre.watermark_consensus as consensus_mod

    assert consensus_mod.logger.handlers
    assert consensus_mod.logger.level == logging.INFO


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


def test_build_candidate_record_constructs_geometry():
    record = make_record(make_rect_bgra())
    assert record.bgra.shape[2] == 4
    assert record.geometry.alpha_soft.shape == record.bgra.shape[:2]
    assert record.geometry.support_mask.dtype == np.uint8
    assert float(record.geometry.distance_field.max()) > 0.0


def test_pairwise_score_prefers_related_partial_candidates():
    a = make_record(make_rect_bgra(rect_h=46, rect_w=98, dx=0, dy=0))
    b = make_record(make_rect_bgra(rect_h=38, rect_w=80, dx=10, dy=6))
    result = score_candidate_pair(a, b)
    assert result is not None
    assert result.compatibility > 0.5
    assert result.shared_core_score > result.unmatched_a_penalty
    assert result.shared_core_score > result.unmatched_b_penalty
    assert 0.25 <= result.scale <= 4.0


def test_pairwise_score_tolerates_noisy_superset_after_scoring_filter():
    a = make_record(make_rect_bgra())
    b = make_record(make_rect_bgra(extra_blob=(85, 135, 16, 18)))
    result = score_candidate_pair(a, b)
    assert result is not None
    assert result.compatibility > 0.35
    assert result.shared_core_score > 0.95


def test_pairwise_score_flags_unrelated_noise_as_incompatible():
    a = make_record(make_rect_bgra())
    b = make_record(make_noise_bgra(seed=7))
    result = score_candidate_pair(a, b)
    assert result is not None
    assert result.compatibility < 0.2


def test_propose_pair_scales_finds_component_ratio_peak_with_bbox_metadata():
    large = make_record(
        make_multi_component_bgra([
            (20, 30, 90, 42),
            (125, 24, 52, 24),
            (146, 110, 28, 72),
        ]),
        family_key=(4000, 3000),
        candidate_index=0,
    )
    small = make_record(
        make_multi_component_bgra([
            (12, 18, 45, 21),
            (66, 14, 26, 12),
            (76, 60, 14, 36),
        ], canvas_shape=(140, 140)),
        family_key=(1920, 1280),
        candidate_index=1,
    )

    proposals = propose_pair_scales(large, small)

    assert proposals
    assert any(abs(proposal.scale - 2.0) < 0.2 for proposal in proposals)
    best = proposals[0]
    assert best.ref_bbox is not None
    assert best.mov_bbox is not None
    assert best.view_name in {"full", "filtered"}
    assert best.feature_name in {"long_side", "short_side", "hypotenuse", "sqrt_area"}


def test_pairwise_score_carries_scale_proposal_metadata():
    large = make_record(
        make_multi_component_bgra([
            (20, 30, 90, 42),
            (125, 24, 52, 24),
            (146, 110, 28, 72),
        ]),
        family_key=(4000, 3000),
        candidate_index=0,
    )
    small = make_record(
        make_multi_component_bgra([
            (12, 18, 45, 21),
            (66, 14, 26, 12),
            (76, 60, 14, 36),
        ], canvas_shape=(140, 140)),
        family_key=(1920, 1280),
        candidate_index=1,
    )

    result = score_candidate_pair(large, small)

    assert result.scale_proposals
    assert any(abs(proposal.scale - 2.0) < 0.2 for proposal in result.scale_proposals)
    assert result.scale_origin != "fallback"


def test_split_candidate_bgra_merges_horizontal_chain_and_separates_remote_blob():
    bgra = make_multi_component_bgra(
        [
            (20, 100, 8, 12),
            (30, 100, 8, 12),
            (40, 100, 8, 12),
            (150, 20, 18, 18),
        ],
        canvas_shape=(180, 220),
    )

    subcrops = split_candidate_bgra(bgra)

    assert len(subcrops) == 2
    shapes = sorted([crop.shape[:2] for crop in subcrops], key=lambda shape: shape[0] * shape[1], reverse=True)
    assert shapes[0][1] > shapes[0][0]
    assert shapes[1][0] >= 20
    assert shapes[1][1] >= 20


def test_split_candidate_bgra_applies_percentage_padding():
    bgra = make_multi_component_bgra(
        [(20, 30, 40, 20)],
        canvas_shape=(100, 120),
    )

    subcrops = split_candidate_bgra(bgra, crop_padding_ratio=0.05)

    assert len(subcrops) == 1
    assert subcrops[0].shape[:2] == (22, 44)


def test_build_candidate_graph_isolates_noise_node():
    records = [make_related_record(i) for i in range(3)] + [make_noise_record(99)]
    graph = build_candidate_graph(records)
    assert graph.weights.shape == (4, 4)
    assert graph.weights[0, 1] > 0.0
    assert graph.weights[0, 3] == 0.0


def test_build_candidate_graph_logs_pair_workload_and_progress(caplog):
    records = [make_related_record(i) for i in range(3)]

    with caplog.at_level(logging.INFO):
        build_candidate_graph(records)

    assert any(
        "Consensus graph start:" in message and "3 unordered pairs" in message
        for message in caplog.messages
    )
    assert any(
        "Consensus graph progress:" in message and "3/3 pairs" in message
        for message in caplog.messages
    )


def test_extract_candidate_clusters_returns_only_high_confidence_group():
    records = [make_related_record(i) for i in range(3)] + [make_noise_record(99)]
    graph = build_candidate_graph(records)
    clusters = extract_candidate_clusters(graph)
    assert len(clusters) == 1
    assert tuple(clusters[0].member_indices) == (0, 1, 2)


def test_extract_candidate_clusters_excludes_weak_bridge_node():
    records = [make_related_record(i) for i in range(3)] + [make_bridge_record()]
    graph = build_candidate_graph(records)
    assert graph.weights[1, 3] > 0.0
    clusters = extract_candidate_clusters(graph)
    assert len(clusters) == 1
    assert tuple(clusters[0].member_indices) == (0, 1, 2)


def test_consensus_emits_stingy_and_generous_when_periphery_is_disputed():
    outputs = build_final_templates(make_disputed_periphery_records())
    assert len(outputs) == 5
    assert outputs[0].label == "stingy"
    assert outputs[1].label == "generous"
    assert outputs[1].alpha_mass > outputs[0].alpha_mass
    assert {(0,), (1,), (2,)}.issubset({output.member_indices for output in outputs if output.label == "source"})


def test_consensus_promotes_appendage_when_third_candidate_supports_it():
    outputs = build_final_templates(make_triply_supported_appendage_records())
    assert len(outputs) == 4
    assert outputs[0].label == "consensus"
    assert outputs[0].alpha_mass > 0.0
    assert {(20,), (21,), (22,)}.issubset({output.member_indices for output in outputs if output.label == "source"})


def test_build_final_templates_retains_large_singleton_when_tiny_pair_clusters():
    records = [
        make_record(
            make_rect_bgra(rect_h=44, rect_w=96),
            family_key=(900, 700),
            candidate_index=0,
        ),
        make_tiny_record(dx=0, dy=0, family_key=(1200, 900), candidate_index=1),
        make_tiny_record(dx=0, dy=0, family_key=(1280, 960), candidate_index=2),
    ]

    outputs = build_final_templates(records)

    assert any(
        output.member_indices == (0,) and output.alpha_mass > 1000.0
        for output in outputs
    )
