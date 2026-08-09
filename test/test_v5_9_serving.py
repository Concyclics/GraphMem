from __future__ import annotations

from dataclasses import replace

import pytest

from graphmem.build import GraphBuildPipeline
from graphmem.config import GraphMemV5Config
from graphmem.domain import QueryBudget, stable_id
from graphmem.retrieval import HarnessProfile
from graphmem.retrieval import GraphNavigator
from graphmem.serving import ProcessShardedNavigator
from graphmem.serving import AdmissionRejected, BoundedAdmissionController
from graphmem.storage import SQLiteGraphStore

from test_v5_9_coarsening import _store


def test_process_shards_share_a_versioned_read_plane(tmp_path) -> None:
    path = tmp_path / "graph.sqlite"
    store = _store(path)
    base = GraphMemV5Config(profile="b5")
    config = replace(
        base,
        coarsen=replace(base.coarsen, recursive_hierarchy=True, fanout=2,
                        max_levels=5),
        edges=replace(base.edges, parent_gated_relations=True,
                      refine_mode="none"),
    )
    GraphBuildPipeline(store, dataset_hash="dataset").build("m", config)
    budget = QueryBudget(max_evidence_turns=8, max_evidence_tokens=1000)

    with ProcessShardedNavigator(
        path, workers=2,
        navigator_options={
            "harness_profile": HarnessProfile.H10_AST,
            "hierarchical_routing": True,
        },
    ) as pool:
        snapshots = pool.warm("m", ("What project did Alice discuss?",), budget)
        assert len(snapshots) == 2
        assert len({(row.graph_version, row.graph_checksum)
                    for row in snapshots}) == 1
        results = pool.navigate_many(
            ("m", f"What happened in session {index}?", budget)
            for index in range(8)
        )

    expected_artifact = stable_id(
        "graph-artifact", "m", store.graph_version("m"),
        store.graph_checksum("m"))
    assert len(results) == 8
    assert all(row.graph_artifact_id == expected_artifact for row in results)
    assert all(row.memory_id == "m" for row in results)


def test_admission_controller_bounds_each_tenant_and_releases_capacity() -> None:
    controller = BoundedAdmissionController(2, 1)
    controller.acquire("noisy")
    try:
        controller.acquire("noisy")
    except AdmissionRejected:
        pass
    else:
        raise AssertionError("tenant queue must reject above its quota")
    controller.acquire("quiet")
    assert controller.stats()["outstanding"] == 2
    controller.release("noisy", failed=False)
    controller.release("quiet", failed=True)
    stats = controller.stats()
    assert stats["outstanding"] == 0
    assert stats["rejected"] == 1
    assert stats["completed"] == 1 and stats["failed"] == 1


def test_memory_affinity_is_stable(tmp_path) -> None:
    path = tmp_path / "empty.sqlite"
    store = _store(path)
    store.close()
    with ProcessShardedNavigator(path, workers=3) as pool:
        first = pool.shard_for_memory("tenant-a:memory-7")
        assert first == pool.shard_for_memory("tenant-a:memory-7")
        assert 0 <= first < 3
    with ProcessShardedNavigator(
            path, workers=3, affinity_replicas=2) as pool:
        candidates = pool.affinity_shards("tenant-a:memory-7")
        assert len(candidates) == 2 and len(set(candidates)) == 2
        assert candidates == pool.affinity_shards("tenant-a:memory-7")


def test_worker_cpu_affinity_is_reported_when_requested(tmp_path) -> None:
    available = sorted(__import__("os").sched_getaffinity(0))
    if not available:
        pytest.skip("no schedulable CPU is visible")
    path = tmp_path / "empty.sqlite"
    store = _store(path)
    store.close()
    with ProcessShardedNavigator(
            path, workers=1, worker_cpu_ids=(available[0],)) as pool:
        rows = pool.worker_cache_stats()
    assert tuple(rows[0].stats["process"]["cpu_affinity"]) == (available[0],)


def test_worker_cpu_affinity_configuration_is_validated(tmp_path) -> None:
    path = tmp_path / "empty.sqlite"
    store = _store(path)
    store.close()
    with pytest.raises(ValueError, match="one CPU ID per worker"):
        ProcessShardedNavigator(path, workers=2, worker_cpu_ids=(0,))
    with pytest.raises(ValueError, match="must be unique"):
        ProcessShardedNavigator(path, workers=2, worker_cpu_ids=(0, 0))


def test_killed_worker_is_restarted_and_request_is_retried(tmp_path) -> None:
    path = tmp_path / "graph.sqlite"
    store = _store(path)
    GraphBuildPipeline(store, dataset_hash="dataset").build(
        "m", GraphMemV5Config(profile="b5"))
    budget = QueryBudget(max_evidence_turns=8, max_evidence_tokens=1000)
    with ProcessShardedNavigator(
        path, workers=1, retry_broken_worker=1,
        navigator_options={"harness_profile": HarnessProfile.H10_AST},
    ) as pool:
        pool.warm("m", ("What project did Alice discuss?",), budget)
        with pytest.raises(BaseException):
            pool.inject_worker_crash(0).result(timeout=10)
        result = pool.submit(
            "m", "What project did Alice discuss?", budget,
            deadline_monotonic=__import__("time").monotonic() + 10,
        ).result(timeout=15)
        stats = pool.admission_stats()

    assert result.memory_id == "m"
    assert stats["worker_restarts"] == 1
    assert stats["inflight_retries"] == 1
    assert stats["failed"] == 0


def test_compiled_memory_sidecar_preserves_results_and_skips_cold_build(tmp_path) -> None:
    path = tmp_path / "graph.sqlite"
    compiled = tmp_path / "compiled"
    store = _store(path)
    GraphBuildPipeline(store, dataset_hash="dataset").build(
        "m", GraphMemV5Config(profile="b5"))
    budget = QueryBudget(max_evidence_turns=8, max_evidence_tokens=1000)
    query = "What project did Alice discuss?"
    baseline = GraphNavigator(
        store, harness_profile=HarnessProfile.H10_AST,
        compiled_cache_dir=compiled)
    expected = baseline.navigate("m", query, budget)
    artifact = baseline.precompile_memory("m", force=True)
    assert artifact.total_retained_bytes >= artifact.view_retained_bytes > 0
    assert baseline.compiled_sidecar is not None
    sidecar = baseline.compiled_sidecar.path_for("m")
    assert sidecar.is_file()
    store.close()

    reader_store = SQLiteGraphStore(path, read_only=True)
    try:
        reader = GraphNavigator(
            reader_store, harness_profile=HarnessProfile.H10_AST,
            compiled_cache_dir=compiled)
        actual = reader.navigate("m", query, budget)
        stats = reader.cache_stats()
        assert actual.retrieved_turn_ids == expected.retrieved_turn_ids
        assert actual.graph_artifact_id == expected.graph_artifact_id
        assert stats["runtime"]["builds"] == 0
        assert stats["compiled"]["hits"] == 1
        assert stats["compiled"]["hydrations"] == 1
        assert reader.compiled_sidecar is not None
        assert reader.compiled_sidecar.load(
            "m", artifact.graph_version, "not-the-authority-checksum") is None
        assert reader.compiled_sidecar.stats()["invalid"] == 1
    finally:
        reader_store.close()


def test_affinity_warm_does_not_duplicate_one_memory_to_every_worker(tmp_path) -> None:
    path = tmp_path / "graph.sqlite"
    store = _store(path)
    GraphBuildPipeline(store, dataset_hash="dataset").build(
        "m", GraphMemV5Config(profile="b5"))
    store.close()
    budget = QueryBudget(max_evidence_turns=8, max_evidence_tokens=1000)
    with ProcessShardedNavigator(
            path, workers=2, affinity_replicas=2,
            navigator_options={"harness_profile": HarnessProfile.H10_AST}) as pool:
        snapshots = pool.warm_affinity(
            {"m": ("What project did Alice discuss?",)}, budget, replicas=1)
        cache_rows = pool.worker_cache_stats()
    assert len(snapshots) == 1
    assert snapshots[0].memory_id == "m"
    assert sum(int(row.stats["runtime"]["views"])
               for row in cache_rows) == 1
