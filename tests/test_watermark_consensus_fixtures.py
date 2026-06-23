from pathlib import Path

import cv2

from untextre.watermark_consensus import (
    CandidateMetadata,
    build_candidate_graph,
    build_candidate_record,
    build_final_templates,
    extract_candidate_clusters,
    score_candidate_pair,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "images" / "fixtures"
FIXTURE_NAMES = [
    "watermark_candidate.png",
    "watermark_candidate_2.png",
    "watermark_candidate_3.png",
    "watermark_candidate_4.png",
    "watermark_candidate_5.png",
    "watermark_candidate_6.png",
]


def _load_fixture_records():
    images = [
        cv2.imread(str(FIXTURE_DIR / name), cv2.IMREAD_UNCHANGED)
        for name in FIXTURE_NAMES
    ]
    fixture_source_bounds = (3000, 3000)
    records = []
    for index, image in enumerate(images):
        records.append(
            build_candidate_record(
                image,
                CandidateMetadata(
                    # Fixture PNGs are detached candidates, not full source images, so
                    # they use a stable generous source bound for scale-cap testing.
                    family_key=fixture_source_bounds,
                    zone=(index % 3, index // 3),
                    candidate_index=index,
                    source_kind="fixture",
                ),
            )
        )
    return records


def _load_fixture_pair(name_a: str, name_b: str, family_key: tuple[int, int]):
    records = []
    for index, name in enumerate((name_a, name_b)):
        image = cv2.imread(str(FIXTURE_DIR / name), cv2.IMREAD_UNCHANGED)
        records.append(
            build_candidate_record(
                image,
                CandidateMetadata(
                    family_key=family_key,
                    zone=(index, 0),
                    candidate_index=index,
                    source_kind="fixture",
                ),
            )
        )
    return records


def test_fixture_candidates_reduce_to_cluster_outputs():
    records = _load_fixture_records()
    graph = build_candidate_graph(records)
    clusters = extract_candidate_clusters(graph, edge_threshold=0.4)
    outputs = build_final_templates(records)

    assert len(clusters) >= 1
    assert len(clusters[0].member_indices) >= 2
    assert len(outputs) >= 1
    assert any(len(output.member_indices) >= 2 for output in outputs)
    assert any(output.label in {"consensus", "stingy"} and output.alpha_mass > 0.0 for output in outputs)


def test_fixture_pair_score_survives_shared_bound_changes():
    square_records = _load_fixture_pair(
        "watermark_candidate.png",
        "watermark_candidate_5.png",
        family_key=(3000, 3000),
    )
    rectangular_records = _load_fixture_pair(
        "watermark_candidate.png",
        "watermark_candidate_5.png",
        family_key=(4000, 3000),
    )

    square_ab = score_candidate_pair(square_records[0], square_records[1])
    square_ba = score_candidate_pair(square_records[1], square_records[0])
    rect_ab = score_candidate_pair(rectangular_records[0], rectangular_records[1])
    rect_ba = score_candidate_pair(rectangular_records[1], rectangular_records[0])

    assert max(square_ab.compatibility, square_ba.compatibility) > 0.3
    assert max(rect_ab.compatibility, rect_ba.compatibility) > 0.3
