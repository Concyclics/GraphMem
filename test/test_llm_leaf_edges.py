from __future__ import annotations

from graphmem_demo.llm_leaf_edges import parse_llm_leaf_edges, prune_llm_leaf_edges
from graphmem_demo.models import GraphEdge


def test_parse_llm_leaf_edges_filters_low_confidence() -> None:
    text = (
        '{"edges": ['
        '{"src": "a", "dst": "b", "relation": "entity_neighbor", "confidence": 0.95},'
        '{"src": "a", "dst": "c", "relation": "event_neighbor", "confidence": 0.6}'
        "]}"
    )
    edges = parse_llm_leaf_edges(
        text,
        valid_leaf_ids={"a", "b", "c"},
        min_confidence=0.8,
        max_edges_per_leaf=3,
        max_edges_per_session=16,
    )
    assert len(edges) == 1
    assert edges[0].dst == "b"
    assert edges[0].score == 0.95


def test_prune_llm_leaf_edges_caps_per_leaf_and_session() -> None:
    candidates = [
        GraphEdge(src="a", dst="b", score=0.95, relation="entity_neighbor"),
        GraphEdge(src="a", dst="c", score=0.92, relation="event_neighbor"),
        GraphEdge(src="a", dst="d", score=0.9, relation="update_neighbor"),
        GraphEdge(src="a", dst="e", score=0.88, relation="state_neighbor"),
        GraphEdge(src="f", dst="g", score=0.99, relation="entity_neighbor"),
    ]
    kept = prune_llm_leaf_edges(
        candidates,
        max_edges_per_leaf=2,
        max_edges_per_session=3,
    )
    assert len(kept) == 3
    assert kept[0].score == 0.99
    assert sum(1 for edge in kept if "a" in {edge.src, edge.dst}) == 2
