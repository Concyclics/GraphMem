"""Tests for high-confidence root graph edge construction and pruning."""
from __future__ import annotations

from graphmem_demo.models import GraphEdge, SummaryNode
from graphmem_demo.root_graph_edges import (
    RootGraphEdgePolicy,
    _entity_edge_allowed,
    build_root_graph,
    prune_noisy_root_edges,
)


def _root(
    node_id: str,
    session_id: str,
    embedding: list[float],
    *,
    anchor_terms: dict[str, list[str]] | None = None,
    retrieval_text: str = "",
) -> SummaryNode:
    return SummaryNode(
        node_id=node_id,
        question_id="q",
        session_id=session_id,
        session_date=None,
        level=1,
        child_ids=[],
        leaf_ids=[],
        summary=session_id,
        retrieval_text=retrieval_text,
        embedding=embedding,
        anchor_terms=anchor_terms,
    )


def test_entity_edge_allows_specific_entity_with_single_shared_term() -> None:
    policy = RootGraphEdgePolicy()
    assert _entity_edge_allowed({"architectural digest"}, policy)


def test_entity_edge_rejects_single_generic_entity() -> None:
    policy = RootGraphEdgePolicy()
    assert not _entity_edge_allowed({"project"}, policy)


def test_entity_edge_allows_two_generic_entities() -> None:
    policy = RootGraphEdgePolicy()
    assert _entity_edge_allowed({"project", "team"}, policy)


def test_build_root_graph_keeps_corpus_keyword_with_typed_edges() -> None:
    first = _root(
        "root-a",
        "a",
        [1.0, 0.0],
        retrieval_text="Memory: subscribed. Search cues: Architectural Digest; subscription; cancelled",
    )
    second = _root(
        "root-b",
        "b",
        [0.95, 0.05],
        retrieval_text="Memory: updated. Search cues: Architectural Digest; subscription; current",
    )
    third = _root(
        "root-c",
        "c",
        [0.0, -1.0],
        retrieval_text="Memory: visited a clinic. Search cues: clinic; 9 AM; arrived",
    )
    first.anchor_terms = {"entities": ["Architectural Digest"], "actions": ["subscribed"]}
    second.anchor_terms = {"entities": ["Architectural Digest"], "actions": ["cancelled"]}
    third.anchor_terms = {"entities": ["Clinic"], "times": ["9 AM"]}

    edges = build_root_graph(
        [first, second, third],
        RootGraphEdgePolicy(
            graph_neighbor_k=1,
            enable_typed_edges=True,
            typed_neighbors_per_relation=1,
            keyword_neighbors_per_root=1,
            semantic_neighbors_per_root=1,
        ),
    )

    assert any(
        edge.relation == "keyword_neighbor" and {edge.src, edge.dst} == {"root-a", "root-b"}
        for edge in edges
    )
    assert any(
        edge.relation == "entity_neighbor" and {edge.src, edge.dst} == {"root-a", "root-b"}
        for edge in edges
    )


def test_prune_drops_low_semantic_typed_bridge() -> None:
    left = _root("root-a", "a", [1.0, 0.0], anchor_terms={"entities": ["Ethereum"]})
    right = _root("root-b", "b", [0.0, 1.0], anchor_terms={"entities": ["Ethereum"]})
    policy = RootGraphEdgePolicy(require_semantic_support=True, semantic_support_min_cosine=0.25)
    edges = [
        GraphEdge("root-a", "root-b", 0.9, "entity_neighbor"),
    ]
    pruned = prune_noisy_root_edges(edges, {"root-a": left, "root-b": right}, policy)
    assert pruned == []


def test_update_edge_requires_multiple_shared_actions() -> None:
    first = _root("root-a", "a", [1.0, 0.0], anchor_terms={"entities": ["project"], "actions": ["led"]})
    second = _root(
        "root-b",
        "b",
        [0.95, 0.05],
        anchor_terms={"entities": ["project"], "actions": ["led", "managed"]},
    )
    third = _root("root-c", "c", [0.0, -1.0], anchor_terms={"entities": ["clinic"]})

    edges = build_root_graph(
        [first, second, third],
        RootGraphEdgePolicy(
            graph_neighbor_k=1,
            enable_typed_edges=True,
            typed_neighbors_per_relation=2,
            keyword_neighbors_per_root=0,
            semantic_neighbors_per_root=0,
            update_min_actions=2,
        ),
    )
    assert not any(edge.relation == "update_neighbor" for edge in edges)
