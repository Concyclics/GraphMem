from __future__ import annotations

import math
from dataclasses import replace

from graphmem.build.coarsen import (
    build_parent_gated_relations,
    build_recursive_hierarchy,
    classify_typed_relation,
)
from graphmem.domain import GraphNode, NodeType, RelationType


def _card(index: int, *, topic: int | None = None, negated: bool = False) -> GraphNode:
    topic = index % 4 if topic is None else topic
    summary = (
        f"Alice {'did not continue' if negated else 'continued'} project topic {topic} "
        f"during January detail {index}")
    return GraphNode(
        f"leaf:{index:04d}", "m", NodeType.ROUTING_CARD, 1,
        summary, f"g:{index}",
        attributes={
            "session_id": f"s:{index}",
            "roles": ("route",),
            "owners": ("Alice",),
            "predicates": ("continue project",),
            "values": (f"topic {topic}",),
            "times": (f"2025-01-{index % 28 + 1:02d}",),
        },
    )


def test_real_hnsw_coarsening_is_balanced_and_near_linear() -> None:
    leaves = tuple(_card(index) for index in range(64))

    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=8, summary_words=64,
        max_candidates=12, assignment_method="hnsw")

    assert hierarchy.stats.assignment_method == "hnsw"
    assert hierarchy.stats.vector_dimension == 256
    assert hierarchy.stats.ann_queries > 0
    assert hierarchy.stats.cluster_candidate_comparisons < len(leaves) ** 2
    assert len(hierarchy.parent_cards) < len(leaves) / 2
    assert all(1 < len(children) <= 4 for children in hierarchy.children.values())
    assert all(card.attributes["coarsen_method"] == "hnsw"
               for card in hierarchy.parent_cards)


def test_typed_classifier_materializes_coreference_and_defers_other_types() -> None:
    earlier = replace(_card(1), node_type=NodeType.CANONICAL_FACT)
    later = replace(_card(2, topic=1), node_type=NodeType.CANONICAL_FACT)
    negated = replace(
        _card(3, topic=1, negated=True), node_type=NodeType.CANONICAL_FACT)

    duplicate = replace(
        _card(4, topic=1), node_type=NodeType.CANONICAL_FACT,
        summary=earlier.summary)
    coreference = classify_typed_relation(earlier, duplicate, 0.9)

    assert coreference and coreference[0] == RelationType.COREFERENCE
    assert coreference[1] >= 0.82
    assert classify_typed_relation(earlier, later, 0.8) is None
    assert classify_typed_relation(earlier, negated, 0.8) is None


def test_parent_gate_restores_typed_cross_session_relations_with_hnsw() -> None:
    leaves = tuple(replace(
        _card(index, topic=0), node_type=NodeType.CANONICAL_FACT,
        summary="Alice continued project topic zero")
        for index in range(16))
    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=6, summary_words=64,
        max_candidates=8, assignment_method="hnsw")
    nodes = {row.node_id: row for row in (*leaves, *hierarchy.parent_cards)}

    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, hierarchy.children,
        embedding_k=4, max_candidates_per_node=8,
        low_threshold=0.05, high_threshold=0.75,
        refine_mode="ambiguous_only", candidate_method="hnsw",
        vectors=hierarchy.vectors, cross_session_quota=2,
        typed_restoration=True, typed_min_confidence=0.82)

    assert plan.candidate_method == "hnsw"
    assert plan.typed_pairs
    assert all(left != right and confidence >= 0.82
               for left, right, _relation, confidence, _level, _source
               in plan.typed_pairs)
    assert {relation for _left, _right, relation, _confidence, _level, _source
            in plan.typed_pairs} <= {
                RelationType.SAME_ENTITY_STATE,
                RelationType.TEMPORAL_CONTINUATION,
                RelationType.CONTRADICTION_UPDATE,
                RelationType.CAUSAL,
                RelationType.COLLECTION_CO_MEMBER,
                RelationType.COREFERENCE,
            }


def test_refine_admission_is_priority_ordered_and_linearly_bounded() -> None:
    leaves = tuple(_card(index, topic=0) for index in range(64))
    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=8, summary_words=64,
        max_candidates=12, assignment_method="hnsw")
    nodes = {row.node_id: row for row in (*leaves, *hierarchy.parent_cards)}

    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, hierarchy.children,
        embedding_k=8, max_candidates_per_node=12,
        low_threshold=0.05, high_threshold=0.99,
        refine_mode="ambiguous_only", candidate_method="hnsw",
        vectors=hierarchy.vectors,
        max_refine_candidates_per_node=2,
        max_refine_candidates_per_1000_nodes=480)

    total_cap = max(1, math.ceil(len(nodes) * 480 / 1000))
    assert len(plan.refine_candidates) <= total_cap
    assert plan.refine_candidates_generated >= len(plan.refine_candidates)
    assert plan.refine_candidates_dropped == (
        plan.refine_candidates_generated - len(plan.refine_candidates))
    assert [row.priority for row in plan.refine_candidates] == sorted(
        (row.priority for row in plan.refine_candidates), reverse=True)
    degree: dict[str, int] = {}
    for row in plan.refine_candidates:
        degree[row.left_id] = degree.get(row.left_id, 0) + 1
        degree[row.right_id] = degree.get(row.right_id, 0) + 1
    assert max(degree.values(), default=0) <= 2


def test_parent_gate_frontier_work_is_bounded_by_hierarchy_size() -> None:
    leaves = tuple(_card(index, topic=0) for index in range(256))
    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=8, summary_words=64,
        max_candidates=12, assignment_method="hnsw")
    nodes = {row.node_id: row for row in (*leaves, *hierarchy.parent_cards)}

    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, hierarchy.children,
        embedding_k=4, max_candidates_per_node=12,
        low_threshold=0.05, high_threshold=0.75,
        refine_mode="ambiguous_only", candidate_method="hnsw",
        vectors=hierarchy.vectors)

    # Four-by-four child scoring for at most k gates per node and level leaves
    # generous constant-factor headroom while ruling out quadratic expansion.
    assert plan.score_comparisons < len(nodes) * 4 * 4 * 4 * 8
