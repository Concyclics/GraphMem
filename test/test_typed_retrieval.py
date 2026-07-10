from __future__ import annotations

from graphmem_demo.fusion_retrieval import FusionRetrievalConfig, rank_leaves_fusion
from graphmem_demo.models import LeafNode, SummaryNode
from graphmem_demo.pipeline import _body_keyword_cues, _effective_typed_root_edges, DemoConfig
from graphmem_demo.typed_retrieval import (
    anchor_terms_to_sets,
    leaf_typed_score,
    query_anchor_terms,
    rank_roots_hybrid,
    typed_overlap_score,
)
from pathlib import Path


def _leaf(**kwargs) -> LeafNode:
    defaults = {
        "node_id": "q1:s1:leaf:0",
        "question_id": "q1",
        "session_id": "s1",
        "session_date": "2023/05/12",
        "turn_index": 0,
        "raw_text": "User: hello",
        "user_text": "User: hello",
        "message_count": 1,
    }
    defaults.update(kwargs)
    return LeafNode(**defaults)


def test_query_anchor_terms_extracts_entities_and_times() -> None:
    anchors = query_anchor_terms(
        'When did I visit "Blue Bottle" last week and spend $12?'
    )
    assert "Blue Bottle" in anchors.get("entities", [])
    assert anchors.get("times") or anchors.get("quantities")


def test_typed_overlap_prefers_entity_match() -> None:
    query_sets = anchor_terms_to_sets({"entities": ["Blue Bottle"], "keywords": ["coffee"]})
    good = anchor_terms_to_sets(
        {"entities": ["Blue Bottle"], "keywords": ["oat milk"], "state_phrases": ["prefers oat milk"]}
    )
    bad = anchor_terms_to_sets({"keywords": ["weather"]})
    assert typed_overlap_score(good, query_sets) > typed_overlap_score(bad, query_sets)


def test_rank_roots_hybrid_prefers_typed_root() -> None:
    roots = [
        SummaryNode(
            node_id="r1",
            question_id="q1",
            session_id="s1",
            session_date="2023/05/12",
            level=1,
            child_ids=[],
            leaf_ids=[],
            summary="User talked about weather.",
            embedding=[1.0, 0.0],
            anchor_terms={"entities": ["Blue Bottle"], "keywords": ["coffee"]},
        ),
        SummaryNode(
            node_id="r2",
            question_id="q1",
            session_id="s2",
            session_date="2023/05/13",
            level=1,
            child_ids=[],
            leaf_ids=[],
            summary="User talked about hiking.",
            embedding=[0.99, 0.01],
            anchor_terms={"keywords": ["weather"]},
        ),
    ]
    ranked = rank_roots_hybrid(
        roots,
        [0.0, 1.0],
        'What did I say about "Blue Bottle"?',
        embedding_blend=0.35,
    )
    assert ranked[0].node_id == "r1"


def test_leaf_typed_score_uses_written_anchor_terms() -> None:
    leaf = _leaf(
        anchor_terms={"entities": ["Architectural Digest"], "keywords": ["subscription"]},
        compact_facts=["User subscribed to Architectural Digest."],
    )
    score = leaf_typed_score(leaf, "Did I subscribe to Architectural Digest?")
    assert score > 0.0


def test_protected_fusion_keeps_semantic_top_leaves() -> None:
    leaves = [
        _leaf(node_id="l1", embedding=[1.0, 0.0], raw_text="semantic winner"),
        _leaf(node_id="l2", embedding=[0.2, 0.9], raw_text="keyword heavy Blue Bottle coffee"),
        _leaf(node_id="l3", embedding=[0.1, 0.1], raw_text="noise"),
    ]
    leaves[1].anchor_terms = {"entities": ["Blue Bottle"], "keywords": ["coffee"]}
    ranked = rank_leaves_fusion(
        leaves,
        [1.0, 0.0],
        'What about "Blue Bottle" coffee?',
        config=FusionRetrievalConfig(protect_semantic_top_k=1),
    )
    assert ranked[0].node_id == "l1"


def test_effective_typed_root_edges_auto_enables_for_graph_first() -> None:
    config = DemoConfig(
        data_path="data.json",
        output_dir=Path("runs/test"),
        enable_graph_first_retrieval=True,
        enable_typed_root_edges=False,
        enable_typed_retrieval=True,
    )
    assert _effective_typed_root_edges(config) is True


def test_body_keyword_cues_extracts_terms_from_lossless_body() -> None:
    cues = _body_keyword_cues(
        "User: I subscribed to Architectural Digest.\nAssistant: Great choice."
    )
    assert any("architectural" in cue.casefold() for cue in cues)
