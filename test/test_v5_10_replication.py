from __future__ import annotations

from dataclasses import replace

import pytest

from graphmem.build import GraphBuildPipeline
from graphmem.build.incremental import plan_affected_paths, publish_affected_path
from graphmem.runtime import GraphReadView
from graphmem.serving import (
    ReplicaCorruptionError,
    ReplicaStaleError,
    SQLiteSnapshotReplicator,
)

from test_v5_9_coarsening import _store
from test_v5_9_incremental import _recursive_config


def _update_one_route(store, summary: str) -> None:
    version = store.graph_version("m")
    view = GraphReadView(store.nodes("m"), store.edges("m"))
    plan = plan_affected_paths(
        view, memory_id="m", source_version=version,
        changed_session_ids=("s0",))
    leaf = view.nodes[plan.session_card_ids[0]]
    publish_affected_path(
        store, view, plan,
        replacement_nodes=(replace(leaf, summary=summary),))


def test_replica_pointer_survives_commit_boundary_crash_and_rejects_stale_promotion(tmp_path) -> None:
    store = _store(tmp_path / "primary.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    replicator = SQLiteSnapshotReplicator(
        store, tmp_path / "replica", config_hash="test-config")
    first = replicator.replicate("m")
    _update_one_route(store, "second authority version")

    with pytest.raises(RuntimeError, match="injected crash"):
        replicator.replicate("m", fail_after_copy=True)

    assert replicator.latest().graph_version == first.graph_version
    assert replicator.lag_versions("m") == 1
    with pytest.raises(ReplicaStaleError, match="1 graph versions"):
        replicator.promote(tmp_path / "rejected.sqlite", max_lag_versions=0)

    latest = replicator.replicate("m")
    promoted = replicator.promote(tmp_path / "promoted.sqlite")
    try:
        assert promoted.graph_version("m") == latest.graph_version
        assert promoted.graph_checksum("m") == latest.graph_checksum
    finally:
        promoted.close()


def test_replica_detects_corruption_and_capacity_failure_keeps_latest(tmp_path) -> None:
    store = _store(tmp_path / "primary.sqlite")
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", _recursive_config())
    replicator = SQLiteSnapshotReplicator(store, tmp_path / "replica")
    committed = replicator.replicate("m")

    with pytest.raises(OSError, match="quota"):
        replicator.replicate("m", max_snapshot_bytes=1)
    assert replicator.latest().graph_version == committed.graph_version

    snapshot = tmp_path / "replica" / committed.snapshot_file
    with snapshot.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ReplicaCorruptionError, match="wrong size"):
        replicator.latest()
