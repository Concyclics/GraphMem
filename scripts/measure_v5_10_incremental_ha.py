#!/usr/bin/env python3
"""Measure V5.10 incremental authority latency and local HA recovery gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import threading
import time

from graphmem.build import (
    IncrementalWriteState,
    IncrementalWriter,
    plan_new_session_insertion,
    publish_new_session_partition,
)
from graphmem.domain import (
    EvidenceGroup, EvidenceMember, GraphNode, NodeType, QueryBudget, Session,
    SourceTurn, logical_graph_checksum, stable_id,
)
from graphmem.retrieval import HarnessProfile
from graphmem.runtime import GraphReadView, SQLiteSnapshotRuntime
from graphmem.serving import (
    ProcessShardedNavigator, ReplicaStaleError, SQLiteSnapshotReplicator,
)
from graphmem.storage import SQLiteGraphStore


def percentile(rows: list[float], p: float) -> float:
    if not rows:
        return 0.0
    ordered = sorted(rows)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(rows: list[float]) -> dict[str, float]:
    return {
        "count": len(rows), "mean_ms": statistics.fmean(rows) if rows else 0.0,
        "p50_ms": percentile(rows, 0.50), "p95_ms": percentile(rows, 0.95),
        "p99_ms": percentile(rows, 0.99), "max_ms": max(rows, default=0.0),
    }


def new_rows(store: SQLiteGraphStore, memory_id: str, index: int):
    sessions = store.sessions(memory_id)
    ordinal = max((row.ordinal for row in sessions), default=-1) + 1
    session_id = f"v510-incremental-{index:04d}"
    timestamp = "2026-08-09T12:00:00Z"
    text = (
        f"Alice recorded incremental cobalt project checkpoint {index} "
        "with latency, token, and availability measurements."
    )
    digest = hashlib.sha256(text.encode()).hexdigest()
    session = Session(session_id, memory_id, ordinal, timestamp, digest)
    turn = SourceTurn(
        stable_id("turn", memory_id, session_id, 0), memory_id, session_id, 0,
        "Alice", "Bob", "user", timestamp, text, digest,
    )
    group = EvidenceGroup(
        stable_id("evidence", memory_id, turn.turn_id), memory_id,
        (EvidenceMember(turn.turn_id, 0, len(text), "direct"),),
        digest, timestamp, timestamp,
    )
    card = GraphNode(
        stable_id("node", memory_id, "routing-card", 1, session_id),
        memory_id, NodeType.ROUTING_CARD, 1, text, group.evidence_group_id,
        attributes={"session_id": session_id, "roles": ("route",),
                    "provenance_scope": "route"},
    )
    return session, turn, group, card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-db",
        default="../artifacts/report/v5_10/hnsw_qwen_typed_dev200_graph_bounded_frontier/report_graph.sqlite",
    )
    parser.add_argument("--output", default="../artifacts/report/v5_10/incremental_ha_gate")
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_db).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with tempfile.TemporaryDirectory(prefix="graphmem-v510-ha-", dir=output) as temp_name:
        temp = Path(temp_name)
        primary_path = temp / "primary.sqlite"
        shutil.copy2(source, primary_path)
        store = SQLiteGraphStore(primary_path)
        store.enable_read_pool(8)
        memory_ids = [str(row[0]) for row in store._read(
            "SELECT memory_id FROM graph_versions ORDER BY memory_id LIMIT ?",
            (args.jobs,))]
        if not memory_ids:
            raise RuntimeError("source graph contains no published memories")
        writer = IncrementalWriter(store)
        runtime = SQLiteSnapshotRuntime(store, max_cached_views=max(16, len(memory_ids)))
        stop = threading.Event()
        reader_errors: list[str] = []
        reader_observations = 0
        reader_lock = threading.Lock()

        def reader(worker: int) -> None:
            nonlocal reader_observations
            cursor = worker
            while not stop.is_set():
                memory_id = memory_ids[cursor % len(memory_ids)]
                cursor += 1
                try:
                    view = runtime.view(memory_id)
                    if any(edge.src_id not in view.nodes or edge.dst_id not in view.nodes
                           for edge in tuple(view.edges.values())[:64]):
                        raise AssertionError("reader observed a dangling edge")
                    if not view.graph_version or not view.graph_checksum:
                        raise AssertionError("reader observed an unversioned snapshot")
                    with reader_lock:
                        reader_observations += 1
                    time.sleep(0.002)
                except BaseException as error:
                    with reader_lock:
                        reader_errors.append(repr(error))
                    stop.set()

        readers = [threading.Thread(target=reader, args=(index,), daemon=True)
                   for index in range(4)]
        for thread in readers:
            thread.start()

        raw_ms: list[float] = []
        fact_commit_ms: list[float] = []
        relation_commit_ms: list[float] = []
        route_plan_ms: list[float] = []
        route_commit_ms: list[float] = []
        touched_rows: list[int] = []
        recompute_fraction: list[float] = []
        rebalances = 0
        commit_fault_rolled_back = False
        for index, memory_id in enumerate(memory_ids):
            session, turn, group, card = new_rows(store, memory_id, index)
            job_id = f"v510:{memory_id}:{index}"
            started = time.perf_counter()
            job = writer.receive(
                job_id=job_id, session=session, turns=(turn,),
                source_offset=10_000 + index, payload={"benchmark": True})
            raw_ms.append((time.perf_counter() - started) * 1000.0)

            # These two timings are authority publication only.  LLM/embedding
            # service time is intentionally excluded and reported separately.
            started = time.perf_counter()
            writer.publish_stage(
                job, expected_state=IncrementalWriteState.RAW_DURABLE,
                next_state=IncrementalWriteState.FACT_INDEXED)
            fact_commit_ms.append((time.perf_counter() - started) * 1000.0)
            job = store.incremental_job(job_id)
            assert job is not None
            started = time.perf_counter()
            writer.publish_stage(
                job, expected_state=IncrementalWriteState.FACT_INDEXED,
                next_state=IncrementalWriteState.RELATION_INDEXED)
            relation_commit_ms.append((time.perf_counter() - started) * 1000.0)
            job = store.incremental_job(job_id)
            assert job is not None

            version = store.graph_version(memory_id)
            view = GraphReadView(
                store.nodes(memory_id), store.edges(memory_id),
                graph_version=version, graph_checksum=store.graph_checksum(memory_id))
            started = time.perf_counter()
            plan = plan_new_session_insertion(
                view, memory_id=memory_id, source_version=version,
                session_card=card, target_fanout=8)
            route_plan_ms.append((time.perf_counter() - started) * 1000.0)
            rebalances += int(plan.needs_background_rebalance)
            started = time.perf_counter()
            result = publish_new_session_partition(
                store, view, plan, session_card=card,
                upsert_evidence_groups=(group,), incremental_job_id=job_id)
            route_commit_ms.append((time.perf_counter() - started) * 1000.0)
            touched_rows.append(result.touched_rows)
            recompute_fraction.append(plan.affected.recompute_fraction)
            final_job = store.incremental_job(job_id)
            assert final_job is not None
            assert final_job.state == IncrementalWriteState.ROUTE_PUBLISHED
            if index in {0, len(memory_ids) - 1}:
                assert store.graph_checksum(memory_id) == logical_graph_checksum(
                    store.nodes(memory_id), store.edges(memory_id))

            if index == 0:
                before = store.graph_version(memory_id)
                try:
                    writer.publish_stage(
                        final_job,
                        expected_state=IncrementalWriteState.RELATION_INDEXED,
                        next_state=IncrementalWriteState.ROUTE_PUBLISHED,
                    )
                except RuntimeError:
                    commit_fault_rolled_back = store.graph_version(memory_id) == before
            print(f"incremental job {index + 1}/{len(memory_ids)} complete", flush=True)

        stop.set()
        for thread in readers:
            thread.join(timeout=5)

        ha_memory = memory_ids[0]
        replicator = SQLiteSnapshotReplicator(
            store, temp / "replica", config_hash="v5.10-bounded-frontier")
        first_replica = replicator.replicate(ha_memory)
        pointer_before = replicator.latest().graph_version
        current = store.graph_version(ha_memory)
        store.apply_graph_delta(
            ha_memory, expected_version=current, event_type="fault_boundary_probe")
        pointer_atomic = False
        try:
            replicator.replicate(ha_memory, fail_after_copy=True)
        except RuntimeError:
            pointer_atomic = replicator.latest().graph_version == pointer_before
        stale_rejected = False
        try:
            replicator.promote(temp / "must-not-promote.sqlite")
        except ReplicaStaleError:
            stale_rejected = True
        second_replica = replicator.replicate(ha_memory)
        promoted_path = temp / "promoted.sqlite"
        promotion_started = time.perf_counter()
        promoted = replicator.promote(promoted_path)
        promotion_ms = (time.perf_counter() - promotion_started) * 1000.0
        promoted_valid = (
            promoted.graph_version(ha_memory) == second_replica.graph_version
            and promoted.graph_checksum(ha_memory) == second_replica.graph_checksum)
        promoted.close()

        query_text = store.turns(ha_memory)[0].raw_text
        budget = QueryBudget(max_evidence_turns=16, max_evidence_tokens=2400)
        worker_recovery_ms = None
        worker_recovery_ok = False
        worker_stats: dict[str, object] = {}
        with ProcessShardedNavigator(
            primary_path, workers=1, retry_broken_worker=1,
            navigator_options={
                "harness_profile": HarnessProfile.H11_UNIFIED_IR,
                "hierarchical_routing": True,
                "native_seed_fusion": True,
            },
        ) as pool:
            pool.warm(ha_memory, (query_text,), budget)
            try:
                pool.inject_worker_crash(0).result(timeout=15)
            except BaseException:
                pass
            started = time.perf_counter()
            recovered = pool.submit(
                ha_memory, query_text, budget,
                deadline_monotonic=time.monotonic() + 30,
            ).result(timeout=35)
            worker_recovery_ms = (time.perf_counter() - started) * 1000.0
            worker_recovery_ok = recovered.memory_id == ha_memory
            worker_stats = dict(pool.admission_stats())

        summary = {
            "source_db": str(source),
            "jobs": len(memory_ids),
            "foreground": {
                "raw_durable": latency_summary(raw_ms),
                "fact_authority_commit_only": latency_summary(fact_commit_ms),
                "relation_authority_commit_only": latency_summary(relation_commit_ms),
                "route_plan": latency_summary(route_plan_ms),
                "route_publish": latency_summary(route_commit_ms),
                "touched_rows": latency_summary([float(row) for row in touched_rows]),
                "mean_recompute_fraction": statistics.fmean(recompute_fraction),
                "background_rebalance_requested": rebalances,
                "semantic_service_latency_included": False,
            },
            "concurrent_readers": {
                "threads": len(readers), "observations": reader_observations,
                "errors": reader_errors, "zero_torn_reads": not reader_errors,
            },
            "fault_injection": {
                "invalid_or_stale_transition_rolled_back": commit_fault_rolled_back,
                "crash_before_latest_pointer_kept_old_snapshot": pointer_atomic,
                "stale_promotion_rejected": stale_rejected,
                "promoted_checksum_valid": promoted_valid,
                "worker_sigkill_recovered": worker_recovery_ok,
            },
            "replication": {
                "first": asdict(first_replica), "second": asdict(second_replica),
                "promotion_ms": promotion_ms,
            },
            "worker_recovery": {
                "rto_ms": worker_recovery_ms, "stats": worker_stats,
            },
            "scope_warning": (
                "Fact/relation numbers measure transactional publication only; "
                "they do not include remote extraction or embedding latency."
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        store.close()


if __name__ == "__main__":
    main()
