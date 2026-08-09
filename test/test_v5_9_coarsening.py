from __future__ import annotations

import hashlib
from dataclasses import replace

from graphmem.build import GraphBuildPipeline
from graphmem.build.coarsen import (
    build_parent_gated_relations,
    build_recursive_hierarchy,
)
from graphmem.config import GraphMemV5Config
from graphmem.domain import (
    Conversation,
    GraphNode,
    NodeType,
    RelationType,
    Session,
    SourceTurn,
    stable_id,
)
from graphmem.storage import SQLiteGraphStore


def _leaf(index: int, topic: int | None = None) -> GraphNode:
    topic = index % 8 if topic is None else topic
    return GraphNode(
        f"leaf:{index:04d}", "m", NodeType.ROUTING_CARD, 1,
        f"shared project topic-{topic} detail-{index}", f"g:{index}",
        attributes={"session_id": f"s{index}", "roles": ("route",),
                    "provenance_scope": "route"},
    )


def test_recursive_coarsening_is_bounded_and_builds_an_arbitrary_depth_tree() -> None:
    leaves = tuple(_leaf(index) for index in range(128))

    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=6, summary_words=64,
        max_candidates=12)

    assert hierarchy.root.level > 3
    assert hierarchy.stats.max_fanout <= 4
    assert hierarchy.stats.cluster_candidate_comparisons <= 128 * 12 * 6
    assert set(hierarchy.children[hierarchy.root.node_id])
    assert all(card.attributes["provenance_compact"]
               for card in hierarchy.parent_cards)
    assert all(len(card.all_evidence_group_ids) == 1
               for card in hierarchy.parent_cards)


def test_parent_edge_gate_generates_child_candidates_without_all_pairs() -> None:
    leaves = tuple(_leaf(index, topic=index % 2) for index in range(32))
    hierarchy = build_recursive_hierarchy(
        "m", leaves, fanout=4, max_levels=5, summary_words=64,
        max_candidates=8)
    nodes = {row.node_id: row for row in (*leaves, *hierarchy.parent_cards)}

    plan = build_parent_gated_relations(
        "m", hierarchy, nodes, hierarchy.children,
        embedding_k=4, max_candidates_per_node=8,
        low_threshold=0.05, high_threshold=0.95,
        refine_mode="ambiguous_only")

    assert plan.coarse_candidate_pairs > 0
    assert plan.gated_child_pairs > 0
    assert plan.refine_candidates
    assert plan.score_comparisons < len(nodes) ** 2
    assert all(row.gate_level >= 1 for row in plan.refine_candidates)


def _store(path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(path)
    memory_id = "m"
    sessions = [Session(f"s{index}", memory_id, index, f"2025-01-0{index + 1}",
                        f"session-{index}") for index in range(4)]
    turns = []
    for session_index, session in enumerate(sessions):
        for turn_index in range(2):
            text = (f"Alice discussed shared project topic {session_index % 2} "
                    f"detail {session_index}-{turn_index}")
            turns.append(SourceTurn(
                stable_id("turn", memory_id, session.session_id, turn_index),
                memory_id, session.session_id, turn_index, "Alice", "Bob", "user",
                session.timestamp, text, hashlib.sha256(text.encode()).hexdigest()))
    store.ingest_conversation(
        Conversation(memory_id, "synthetic", memory_id, "memory-hash"), sessions, turns)
    return store


def test_pipeline_report_arm_emits_recursive_tree_and_cir_diagnostics(tmp_path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    base = GraphMemV5Config(profile="b5")
    config = replace(
        base,
        coarsen=replace(base.coarsen, recursive_hierarchy=True, fanout=2,
                        max_levels=5),
        edges=replace(base.edges, parent_gated_relations=True,
                      low_threshold=0.05, high_threshold=0.8,
                      refine_mode="none"),
    )

    manifest = GraphBuildPipeline(store, dataset_hash="dataset").build("m", config)
    nodes = store.nodes("m")
    edges = store.edges("m")
    cards = [row for row in nodes if row.node_type == NodeType.ROUTING_CARD]
    roots = [row for row in cards if "memory" in row.attributes.get("roles", ())]

    assert len(roots) == 1
    assert {row.level for row in cards} >= {1, 2, 3}
    assert any(row.relation == RelationType.REFINES_TO for row in edges)
    assert manifest.build_diagnostics["method"]["recursive_hierarchy_enabled"]
    assert manifest.build_diagnostics["method"]["parent_gated_relations_enabled"]
    assert manifest.build_token_usage["coarsen_candidate_comparisons"] >= 0
