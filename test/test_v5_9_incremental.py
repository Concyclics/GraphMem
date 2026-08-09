from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from graphmem.build import GraphBuildPipeline
from graphmem.build.incremental import plan_affected_paths, publish_affected_path
from graphmem.build.incremental import (
    IncrementalWriteState,
    IncrementalWriter,
    plan_new_session_insertion,
    publish_new_session_partition,
)
from graphmem.config import GraphMemV5Config
from graphmem.domain import (
    EvidenceGroup, EvidenceMember, GraphEdge, GraphNode, NodeType, RelationType,
    Session, SourceTurn, logical_graph_checksum, stable_id,
)
from graphmem.runtime import GraphReadView, SQLiteSnapshotRuntime

from test_v5_9_coarsening import _store


def _recursive_config() -> GraphMemV5Config:
    base = GraphMemV5Config(profile="b5")
    return replace(
        base,
        coarsen=replace(base.coarsen, recursive_hierarchy=True, fanout=2,
                        max_levels=5),
        edges=replace(base.edges, parent_gated_relations=True,
                      low_threshold=0.05, high_threshold=0.8,
                      refine_mode="none"),
    )


def test_affected_path_recompiles_only_leaf_and_ancestors(tmp_path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    version = store.graph_version("m")
    view = GraphReadView(store.nodes("m"), store.edges("m"))
    plan = plan_affected_paths(
        view, memory_id="m", source_version=version,
        changed_session_ids=("s0",))
    leaf = view.nodes[plan.session_card_ids[0]]
    untouched_id = next(node_id for node_id, node in view.nodes.items()
                        if node.attributes.get("session_id") == "s3")
    replacement = replace(leaf, summary="Alice launched the cobalt project")

    result = publish_affected_path(
        store, view, plan, replacement_nodes=(replacement,), summary_words=64)
    updated = {node.node_id: node for node in store.nodes("m")}

    assert result.graph_version == version + 1
    assert result.upserted_nodes == 1 + len(plan.ancestor_card_ids)
    assert result.touched_rows < len(view.nodes)
    assert updated[leaf.node_id].summary == replacement.summary
    assert updated[untouched_id] == view.nodes[untouched_id]
    assert all(updated[node_id].attributes["incremental_recompiled"]
               for node_id in plan.ancestor_card_ids)
    assert "cobalt" in updated[plan.ancestor_card_ids[-1]].summary.casefold()
    assert store.graph_checksum("m") == logical_graph_checksum(
        store.nodes("m"), store.edges("m"))


def test_delta_publish_rejects_stale_writer_and_rolls_back_invalid_edge(tmp_path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    version = store.graph_version("m")
    view = GraphReadView(store.nodes("m"), store.edges("m"))
    plan = plan_affected_paths(
        view, memory_id="m", source_version=version,
        changed_session_ids=("s0",))
    leaf = view.nodes[plan.session_card_ids[0]]
    publish_affected_path(
        store, view, plan,
        replacement_nodes=(replace(leaf, summary="first update"),))

    with pytest.raises(RuntimeError, match="stale graph delta"):
        publish_affected_path(
            store, view, plan,
            replacement_nodes=(replace(leaf, summary="lost update"),))

    current = store.graph_version("m")
    bad_edge = GraphEdge(
        stable_id("edge", "m", leaf.node_id, "missing"), "m", leaf.node_id,
        RelationType.COARSE_RELATED, "missing-node", leaf.evidence_group_id,
        False, 1.0, "test")
    with pytest.raises(ValueError, match="dangling edge"):
        store.apply_graph_delta(
            "m", upsert_edges=(bad_edge,), expected_version=current)
    assert store.graph_version("m") == current
    assert all(edge.edge_id != bad_edge.edge_id for edge in store.edges("m"))


def _new_session_rows():
    text = "Alice launched the cobalt observability project with Bob"
    session = Session("s4", "m", 4, "2025-01-05", "session-4")
    turn = SourceTurn(
        stable_id("turn", "m", "s4", 0), "m", "s4", 0,
        "Alice", "Bob", "user", session.timestamp, text,
        hashlib.sha256(text.encode()).hexdigest(),
    )
    group = EvidenceGroup(
        stable_id("evidence", "m", turn.turn_id), "m",
        (EvidenceMember(turn.turn_id, 0, len(text), "direct"),),
        turn.content_hash, session.timestamp, session.timestamp,
    )
    card = GraphNode(
        stable_id("node", "m", "routing-card", 1, session.session_id),
        "m", NodeType.ROUTING_CARD, 1, text, group.evidence_group_id,
        attributes={"session_id": session.session_id, "roles": ("route",),
                    "provenance_scope": "route"},
    )
    return session, turn, group, card


def test_incremental_state_machine_is_idempotent_and_graph_transition_atomic(tmp_path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    writer = IncrementalWriter(store)
    session, turn, _group, card = _new_session_rows()

    job = writer.receive(
        job_id="append:s4", session=session, turns=(turn,), source_offset=4,
        payload={"kind": "new_session"})
    replay = writer.receive(
        job_id="append:s4", session=session, turns=(turn,), source_offset=4,
        payload={"kind": "new_session"})

    assert job == replay
    assert job.state == IncrementalWriteState.RAW_DURABLE
    assert [row.turn_id for row in store.turns("m")].count(turn.turn_id) == 1
    assert store.incremental_high_watermark("m") == 4

    version = store.graph_version("m")
    bad_edge = GraphEdge(
        "bad:incremental", "m", card.node_id, RelationType.COARSE_RELATED,
        "missing", card.evidence_group_id, False, 1.0, "fault")
    with pytest.raises(ValueError, match="dangling edge"):
        writer.publish_stage(
            job, expected_state=IncrementalWriteState.RAW_DURABLE,
            next_state=IncrementalWriteState.FACT_INDEXED,
            upsert_edges=(bad_edge,))

    current = store.incremental_job(job.job_id)
    assert current is not None
    assert current.state == IncrementalWriteState.RAW_DURABLE
    assert store.graph_version("m") == version


def test_new_session_partition_becomes_visible_as_one_immutable_route_version(tmp_path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    runtime = SQLiteSnapshotRuntime(store)
    old_view = runtime.view("m")
    writer = IncrementalWriter(store)
    session, turn, group, card = _new_session_rows()
    job = writer.receive(
        job_id="append:s4", session=session, turns=(turn,), source_offset=4)
    writer.publish_stage(
        job, expected_state=IncrementalWriteState.RAW_DURABLE,
        next_state=IncrementalWriteState.FACT_INDEXED)
    job = store.incremental_job(job.job_id)
    assert job is not None
    writer.publish_stage(
        job, expected_state=IncrementalWriteState.FACT_INDEXED,
        next_state=IncrementalWriteState.RELATION_INDEXED)
    job = store.incremental_job(job.job_id)
    assert job is not None

    view = GraphReadView(
        store.nodes("m"), store.edges("m"),
        graph_version=store.graph_version("m"),
        graph_checksum=store.graph_checksum("m"),
    )
    plan = plan_new_session_insertion(
        view, memory_id="m", source_version=view.graph_version,
        session_card=card, target_fanout=2)
    result = publish_new_session_partition(
        store, view, plan, session_card=card,
        upsert_evidence_groups=(group,), incremental_job_id=job.job_id)

    updated_job = store.incremental_job(job.job_id)
    new_view = runtime.view("m")
    assert updated_job is not None
    assert updated_job.state == IncrementalWriteState.ROUTE_PUBLISHED
    assert updated_job.graph_version == result.graph_version == new_view.graph_version
    assert card.node_id not in old_view.nodes
    assert card.node_id in new_view.nodes
    assert card.node_id in new_view.hierarchy_children(plan.parent_card_id)
    assert "cobalt" in new_view.nodes[plan.parent_card_id].summary.casefold()
    assert store.graph_checksum("m") == logical_graph_checksum(
        store.nodes("m"), store.edges("m"))
