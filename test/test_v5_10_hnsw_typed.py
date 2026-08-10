from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from graphmem.build.coarsen import (
    admit_llm_refined_relation,
    bounded_relation_view_pairs,
    build_rare_lexical_node_terms,
    build_relation_features,
    build_parent_gated_relations,
    build_recursive_hierarchy,
    classify_typed_relation,
    RelationSignal,
)
from graphmem.build.pipeline import GraphBuildPipeline
from graphmem.domain import (
    GraphEdge, GraphNode, NodeType, RelationType, SourceTurn,
)


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


def test_relation_mask_recovers_state_pair_below_semantic_gate() -> None:
    sessions = (
        replace(_card(0), summary="orchid greenhouse notes",
                attributes={"session_id": "s:0"}),
        replace(_card(1), summary="quantum circuit notebook",
                attributes={"session_id": "s:1"}),
    )
    facts = (
        GraphNode("fact:0", "m", NodeType.CANONICAL_FACT, 0,
                  "Alice owns a red touring bicycle", "g:0",
                  attributes={"session_id": "s:0", "owner_id": "alice",
                              "predicate": "owns bicycle", "scope": "transport",
                              "value": "touring bicycle", "observed_at": {
                                  "start": "2025-01-01T00:00:00+00:00"}}),
        GraphNode("fact:1", "m", NodeType.CANONICAL_FACT, 0,
                  "Alice keeps the same touring bicycle", "g:1",
                  attributes={"session_id": "s:1", "owner_id": "alice",
                              "predicate": "owns bicycle", "scope": "transport",
                              "value": "touring bicycle", "observed_at": {
                                  "start": "2025-02-01T00:00:00+00:00"}}),
    )
    hierarchy = build_recursive_hierarchy(
        "m", sessions, fanout=2, max_levels=4, summary_words=32,
        max_candidates=4, assignment_method="bounded_semantic_partition")
    nodes = {row.node_id: row for row in (
        *sessions, *facts, *hierarchy.parent_cards)}
    children = {key: list(value) for key, value in hierarchy.children.items()}
    children[sessions[0].node_id] = [facts[0].node_id]
    children[sessions[1].node_id] = [facts[1].node_id]

    legacy = build_parent_gated_relations(
        "m", hierarchy, nodes, children, embedding_k=4,
        max_candidates_per_node=8, low_threshold=0.35,
        high_threshold=0.78, refine_mode="ambiguous_only",
        typed_restoration=True)
    masked = build_parent_gated_relations(
        "m", hierarchy, nodes, children, embedding_k=4,
        max_candidates_per_node=8, low_threshold=0.35,
        high_threshold=0.78, refine_mode="ambiguous_only",
        typed_restoration=True, relation_mask_propagation=True,
        atomic_relation_multiview=True,
        relation_view_quotas={"entity": 2, "state": 2, "temporal": 1,
                              "collection": 1, "lexical": 0,
                              "semantic": 0})

    assert not legacy.accepted_pairs
    assert masked.accepted_pairs
    assert any("state_compatible" in signals
               for signals in masked.accepted_pair_signals.values())
    assert masked.relation_mask_counts[str(RelationSignal.STATE_COMPATIBLE)] > 0
    assert any(str(RelationType.COREFERENCE) in row.allowed_relations
               for row in masked.refine_candidates)
    assert masked.atomic_candidate_source_counts["state"] > 0

    semantic_only = build_parent_gated_relations(
        "m", hierarchy, nodes, children, embedding_k=4,
        max_candidates_per_node=8, low_threshold=0.35,
        high_threshold=0.78, refine_mode="ambiguous_only",
        typed_restoration=True, relation_mask_propagation=True,
        atomic_relation_multiview=True, hnsw_dimension=16,
        atomic_vector_channels=({facts[0].node_id: (1.0, *([0.0] * 15)),
                                 facts[1].node_id: (0.99, 0.01,
                                                         *([0.0] * 14))},),
        relation_view_quotas={"entity": 2, "state": 2, "temporal": 1,
                              "collection": 1, "lexical": 2,
                              "semantic": 2},
        enabled_signals=(RelationSignal.SCENE_SIMILAR,))
    assert semantic_only.atomic_candidate_source_counts.get("semantic", 0) > 0
    assert semantic_only.atomic_candidate_source_counts.get("rare_lexical", 0) == 0
    assert set(semantic_only.atomic_candidate_signal_counts) == {"scene_similar"}
    assert all(set(signals) == {"scene_similar"} for signals in
               semantic_only.refine_candidate_signals.values())


def test_multiview_candidates_are_independently_bounded() -> None:
    nodes = tuple(replace(
        _card(index, topic=index), node_type=NodeType.CANONICAL_FACT,
        attributes={**_card(index).attributes,
                    "owner_id": f"entity:{index % 3}",
                    "predicate": f"activity:{index % 5}",
                    "scope": f"scope:{index % 4}",
                    "observed_at": {"start": f"2025-01-{index % 28 + 1:02d}"}})
        for index in range(60))
    mapped = {node.node_id: node for node in nodes}
    features = build_relation_features(mapped, {})
    pairs, comparisons = bounded_relation_view_pairs(
        nodes, features, eligible_entities=frozenset(),
        quotas={"state": 2, "temporal": 1, "collection": 1},
        max_candidates=8, cross_session_only=True)

    per_signal_degree: dict[tuple[str, RelationSignal], int] = {}
    for left, right, _score, signal in pairs:
        per_signal_degree[(left, signal)] = per_signal_degree.get(
            (left, signal), 0) + 1
        per_signal_degree[(right, signal)] = per_signal_degree.get(
            (right, signal), 0) + 1
    # Unioning directed top-k proposals can double the endpoint degree, but it
    # cannot grow with N for a fixed per-view quota.
    assert max(per_signal_degree.values(), default=0) <= 4
    assert comparisons < len(nodes) * 8 * 8


def test_multiview_signal_allow_list_filters_candidate_generation() -> None:
    nodes = tuple(replace(
        _card(index, topic=index), node_type=NodeType.CANONICAL_FACT,
        attributes={**_card(index).attributes,
                    "owner_id": "shared-entity",
                    "predicate": "shared-state",
                    "observed_at": {"start": f"2025-01-{index + 1:02d}"}})
        for index in range(8))
    features = build_relation_features({node.node_id: node for node in nodes}, {})

    pairs, _ = bounded_relation_view_pairs(
        nodes, features, eligible_entities=frozenset({"shared-entity"}),
        quotas={"entity": 4, "state": 4, "temporal": 4},
        max_candidates=8, cross_session_only=True,
        enabled_signals=(RelationSignal.TEMPORAL_NEAR,))

    assert pairs
    assert {signal for _left, _right, _score, signal in pairs} == {
        RelationSignal.TEMPORAL_NEAR}


def test_rare_lexical_relation_is_one_multi_attribute_edge() -> None:
    left = replace(
        _card(1), summary="restored a classic roadster",
        attributes={**_card(1).attributes,
                    "entities": ("ethereal-dreams",)})
    right = replace(
        _card(2), summary="moved a framed painting",
        attributes={**_card(2).attributes,
                    "entities": ("ethereal-dreams",)})
    turns = (
        SourceTurn("t:1", "m", "s:1", 0, "Alice", "Bob", "user", None,
                   "Ethereal Dreams sapphire canvas bedroom", "h:1"),
        SourceTurn("t:2", "m", "s:2", 0, "Alice", "Bob", "user", None,
                   "Ethereal Dreams sapphire canvas hallway", "h:2"),
        SourceTurn("t:3", "m", "s:3", 0, "Alice", "Bob", "user", None,
                   "ordinary conversation unrelated topic", "h:3"),
    )
    lexical = build_rare_lexical_node_terms((left, right), turns)
    assert {"ethereal", "dreams", "sapphire", "canvas"} <= lexical[left.node_id]

    hierarchy = build_recursive_hierarchy(
        "m", (left, right), fanout=2, max_levels=3, summary_words=24,
        max_candidates=4, assignment_method="bounded_semantic_partition")
    left_fact = replace(
        left, node_id="fact:left", node_type=NodeType.CANONICAL_FACT,
        attributes={"session_id": "s:1", "owner_id": "ethereal-dreams",
                    "predicate": "display location"})
    right_fact = replace(
        right, node_id="fact:right", node_type=NodeType.CANONICAL_FACT,
        attributes={"session_id": "s:2", "owner_id": "ethereal-dreams",
                    "predicate": "display location"})
    nodes = {node.node_id: node for node in (
        left, right, left_fact, right_fact, *hierarchy.parent_cards)}
    children = {key: list(value) for key, value in hierarchy.children.items()}
    children[left.node_id] = [left_fact.node_id]
    children[right.node_id] = [right_fact.node_id]
    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, children,
        embedding_k=2, max_candidates_per_node=4,
        low_threshold=0.35, high_threshold=0.78,
        refine_mode="ambiguous_only", relation_mask_propagation=True,
        lexical_rare_terms=lexical, rare_lexical_min_shared=3,
        relation_view_quotas={"entity": 2, "rare_lexical": 2})

    pair = tuple(sorted((left.node_id, right.node_id)))
    assert sum(row[:2] == pair for row in plan.accepted_pairs) == 1
    assert {"lexical_rare", "shared_entity"} <= set(
        plan.accepted_pair_signals[pair])

    no_lexical = build_parent_gated_relations(
        "m", hierarchy, nodes, children,
        embedding_k=2, max_candidates_per_node=4,
        low_threshold=0.35, high_threshold=0.78,
        refine_mode="ambiguous_only", relation_mask_propagation=True,
        lexical_rare_terms=lexical, rare_lexical_min_shared=3,
        relation_view_quotas={"entity": 2, "rare_lexical": 2},
        enabled_signals=(RelationSignal.SCENE_SIMILAR,
                         RelationSignal.SHARED_ENTITY))
    assert all("lexical_rare" not in signals
               for signals in no_lexical.accepted_pair_signals.values())
    assert all("lexical_rare" not in signals
               for signals in no_lexical.refine_candidate_signals.values())


def test_undirected_degree_cap_applies_to_both_endpoints() -> None:
    edges = [GraphEdge(
        f"e:{index}", "m", f"n:{index}", RelationType.COREFERENCE, "hub",
        f"g:{index}", False, 1.0 - index / 100, "test")
        for index in range(5)]
    profile = SimpleNamespace(edges=SimpleNamespace(
        relation_degree_caps={str(RelationType.COREFERENCE): 2},
        max_degree_per_relation=2))

    bounded = GraphBuildPipeline._bounded_edges(edges, profile)

    assert len(bounded) == 2
    assert sum(edge.dst_id == "hub" for edge in bounded) == 2
