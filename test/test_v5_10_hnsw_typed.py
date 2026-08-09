from __future__ import annotations

import math
from dataclasses import replace

from graphmem.build.coarsen import (
    admit_llm_refined_relation,
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


def test_llm_relation_materialization_requires_validated_field_agreement() -> None:
    left = replace(
        _card(1), node_type=NodeType.CANONICAL_FACT,
        attributes={**_card(1).attributes, "predicate": "owns bicycle",
                    "scope": "transport", "value": "red bicycle",
                    "polarity": "positive"})
    aligned = replace(
        _card(2), node_type=NodeType.CANONICAL_FACT,
        attributes={**_card(2).attributes, "predicate": "owns bicycle",
                    "scope": "transport", "value": "blue bicycle",
                    "polarity": "positive"})
    unrelated = replace(
        _card(3), node_type=NodeType.CANONICAL_FACT,
        attributes={**_card(3).attributes, "predicate": "visited museum",
                    "scope": "travel", "value": "Paris"})

    assert not admit_llm_refined_relation(
        RelationType.CONTRADICTION_UPDATE, left, aligned, 0.9,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.CONTRADICTION_UPDATE, left, unrelated, 0.99,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.SAME_ENTITY_STATE, left, aligned, 0.99,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.COARSE_RELATED, left, aligned, 0.99,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.TEMPORAL_CONTINUATION, left, aligned, 0.9,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.TEMPORAL_CONTINUATION, left,
        replace(aligned, attributes={**aligned.attributes,
                                     "polarity": "negative"}),
        0.99, min_confidence=0.82)
    # A model confidence alone cannot establish directional causality.  Causal
    # candidates remain deferred until a second-stage verifier is available.
    assert not admit_llm_refined_relation(
        RelationType.CAUSAL, left, aligned, 0.99,
        min_confidence=0.82)
    same_owner_left = replace(
        left, attributes={**left.attributes, "owner_id": "alice",
                          "value": "touring bicycle"})
    same_owner_right = replace(
        aligned, attributes={**aligned.attributes, "owner_id": "alice",
                             "value": "touring bicycle"})
    assert admit_llm_refined_relation(
        RelationType.COREFERENCE, same_owner_left, same_owner_right, 0.9,
        min_confidence=0.82)
    assert not admit_llm_refined_relation(
        RelationType.COREFERENCE, same_owner_left,
        replace(same_owner_right, attributes={**same_owner_right.attributes,
                                               "owner_id": "bob"}),
        0.99, min_confidence=0.82)


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


def test_refine_candidates_keep_atomic_labels_off_routing_cards() -> None:
    leaves = tuple(replace(
        _card(index, topic=0), node_type=NodeType.CANONICAL_FACT)
        for index in range(16))
    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=6, summary_words=64,
        max_candidates=8, assignment_method="hnsw")
    nodes = {row.node_id: row for row in (*leaves, *hierarchy.parent_cards)}

    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, hierarchy.children,
        embedding_k=4, max_candidates_per_node=8,
        low_threshold=0.05, high_threshold=0.99,
        refine_mode="ambiguous_only", candidate_method="hnsw",
        vectors=hierarchy.vectors,
        max_refine_candidates_per_node=4,
        max_refine_candidates_per_1000_nodes=1000)

    assert plan.refine_candidates
    typed = {
        str(RelationType.SAME_ENTITY_STATE),
        str(RelationType.TEMPORAL_CONTINUATION),
        str(RelationType.CONTRADICTION_UPDATE),
        str(RelationType.CAUSAL),
        str(RelationType.COREFERENCE),
    }
    for candidate in plan.refine_candidates:
        endpoint_types = {
            nodes[candidate.left_id].node_type,
            nodes[candidate.right_id].node_type,
        }
        if NodeType.ROUTING_CARD in endpoint_types:
            assert not (typed & set(candidate.allowed_relations))
        if typed & set(candidate.allowed_relations):
            assert candidate.cross_session


def test_high_similarity_cross_session_atomic_pair_still_reaches_refiner() -> None:
    leaves = tuple(replace(
        _card(index, topic=0), node_type=NodeType.CANONICAL_FACT,
        summary="Alice still leads the same project")
        for index in range(8))
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
        typed_restoration=True, max_refine_candidates_per_node=4,
        max_refine_candidates_per_1000_nodes=1000)

    atomic_candidates = [row for row in plan.refine_candidates
                         if str(RelationType.COREFERENCE)
                         in row.allowed_relations]
    assert atomic_candidates
    assert all(row.cross_session for row in atomic_candidates)
    assert any((row.similarity or 0.0) >= 0.75 for row in atomic_candidates)
    atomic_ids = {row.node_id for row in leaves}
    assert not any(left in atomic_ids and right in atomic_ids
                   for left, right, _score, _level in plan.accepted_pairs)
    assert all(nodes[left].node_type in {
                   NodeType.ROUTING_CARD, NodeType.SCENE}
               and nodes[right].node_type in {
                   NodeType.ROUTING_CARD, NodeType.SCENE}
               for left, right, _score, _level in plan.accepted_pairs)


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
