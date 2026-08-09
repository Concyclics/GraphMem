#!/usr/bin/env python3
"""Zipf multi-tenant retrieval load with bounded admission and affinity."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_runtime_config, runtime_config_hash  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.serving import (  # noqa: E402
    AdmissionRejected, ProcessShardedNavigator, RequestDeadlineExceeded)


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/"
                        "hnsw_qwen_typed_dev200_graph_bounded_frontier/"
                        "report_graph.sqlite")
    parser.add_argument(
        "--runtime-config", type=Path,
        default=ROOT / "configs/v5/runtime_v5_11_balanced.json")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--tenants", type=int, default=8)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--burst", type=int, default=128)
    parser.add_argument("--max-queued", type=int, default=32)
    parser.add_argument("--per-tenant", type=int, default=8)
    parser.add_argument("--affinity-replicas", type=int, default=2)
    parser.add_argument("--cache-memories", type=int, default=16)
    parser.add_argument("--cache-mib", type=int, default=512)
    parser.add_argument("--compiled-cache-dir", type=Path)
    parser.add_argument(
        "--warm-mode", choices=("sample-all", "affinity-all", "none"),
        default="sample-all",
        help="sample-all preserves V5.10; affinity-all warms each tenant only on affinity shards.")
    parser.add_argument(
        "--warm-replicas", type=int, default=1,
        help="Number of affinity replicas to prewarm in affinity-all mode.")
    parser.add_argument(
        "--memory-ids", default="",
        help="Optional comma-separated memory IDs used for a matched-backend sample.")
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/multi_tenant_60s")
    return parser.parse_args()


def percentile(values, q: float) -> float:
    rows = sorted(values)
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))] if rows else 0.0


def rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0.0


def process_memory_mib(pid: int) -> dict[str, float]:
    result = {"rss": rss_mib(pid), "pss": 0.0, "private_dirty": 0.0,
              "shared_clean": 0.0}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Pss", "Private_Dirty", "Shared_Clean"}:
                name = {"Pss": "pss", "Private_Dirty": "private_dirty",
                        "Shared_Clean": "shared_clean"}[key]
                result[name] = int(value.split()[0]) / 1024
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    requested_memories = {
        value.strip() for value in args.memory_ids.split(",") if value.strip()
    }
    if requested_memories:
        questions = [row for row in questions
                     if row.memory_id in requested_memories]
        found_memories = {row.memory_id for row in questions}
        missing = requested_memories - found_memories
        if missing:
            raise SystemExit(
                "Requested memory IDs have no benchmark questions: "
                + ", ".join(sorted(missing)))
    if not questions:
        raise SystemExit("No benchmark questions matched the selected memories")
    # A fixed popularity rank makes the workload reproducible.  Multiple
    # questions for one memory share its Zipf probability.
    by_memory = {}
    for row in questions:
        by_memory.setdefault(row.memory_id, []).append(row)
    memories = sorted(by_memory)
    weights = [1.0 / ((index + 1) ** args.zipf)
               for index in range(len(memories))]
    runtime_config = load_runtime_config(args.runtime_config)
    budget = replace(
        runtime_config.query_budget,
        max_evidence_turns=32, max_evidence_tokens=5000)
    options = {
        **runtime_config.retrieval.navigator_options(
            compiled_cache_dir=(args.compiled_cache_dir or "")),
        "snapshot_cache_memories": args.cache_memories,
        "metadata_cache_memories": args.cache_memories,
        "snapshot_cache_bytes": args.cache_mib * 1024 * 1024,
    }
    latencies = []
    queue_latencies = []
    counts = {"completed": 0, "failed": 0, "rejected": 0,
              "deadline": 0, "wrong_memory": 0}
    lock = threading.Lock()
    with ProcessShardedNavigator(
            args.db, workers=args.workers, navigator_options=options,
            start_method=runtime_config.serving.start_method,
            max_queued=args.max_queued,
            per_tenant_outstanding=args.per_tenant,
            affinity_replicas=args.affinity_replicas,
            retry_broken_worker=runtime_config.serving.retry_broken_worker) as pool:
        sample = by_memory[memories[0]][0]
        warm = ()
        if args.warm_mode == "sample-all":
            warm = pool.warm(sample.memory_id, (sample.query,), budget)
        elif args.warm_mode == "affinity-all":
            warm = pool.warm_affinity({
                # The workload popularity rank is ``memories`` order.  Warm
                # cold-to-hot so a bounded LRU finishes with the Zipf head,
                # rather than evicting it while touching the long tail.
                memory_id: (by_memory[memory_id][0].query,)
                for memory_id in reversed(memories)
            }, budget, replicas=args.warm_replicas)

        # Open-loop burst proves overload is rejected rather than accumulated in
        # an unbounded ProcessPoolExecutor work queue.
        burst_futures = []
        burst_rejected = 0
        for index in range(args.burst):
            question = by_memory[memories[index % len(memories)]][0]
            try:
                submitted = time.monotonic()
                burst_futures.append((pool.submit(
                    question.memory_id, question.query, budget,
                    tenant_id=f"burst-{index % args.tenants}"), submitted))
            except AdmissionRejected:
                burst_rejected += 1
        burst_end_to_end_ms = []
        burst_service_ms = []
        for future in as_completed([row[0] for row in burst_futures]):
            try:
                result = future.result(timeout=max(30.0, args.deadline * 4))
                submitted = next(row[1] for row in burst_futures
                                 if row[0] is future)
                burst_end_to_end_ms.append(
                    (time.monotonic() - submitted) * 1000)
                burst_service_ms.append(float(
                    result.stage_latency_ms.get("total", 0.0)))
            except BaseException:
                pass

        stop_at = time.monotonic() + args.duration

        def client(client_id: int) -> None:
            rng = random.Random(42000 + client_id)
            tenant = f"tenant-{client_id % args.tenants}"
            while time.monotonic() < stop_at:
                memory_id = rng.choices(memories, weights=weights, k=1)[0]
                question = rng.choice(by_memory[memory_id])
                submitted = time.monotonic()
                try:
                    future = pool.submit(
                        memory_id, question.query, budget, tenant_id=tenant,
                        deadline_monotonic=submitted + args.deadline)
                    result = future.result(timeout=args.deadline + 1.0)
                    finished = time.monotonic()
                    with lock:
                        latencies.append((finished - submitted) * 1000)
                        queue_latencies.append(max(
                            0.0, (finished - submitted) * 1000
                            - float(result.stage_latency_ms.get("total", 0.0))))
                        counts["completed"] += 1
                        counts["wrong_memory"] += int(
                            result.memory_id != memory_id)
                except AdmissionRejected:
                    with lock:
                        counts["rejected"] += 1
                except (RequestDeadlineExceeded, TimeoutError):
                    with lock:
                        counts["deadline"] += 1
                except BaseException:
                    with lock:
                        counts["failed"] += 1

        workload_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.clients) as clients:
            futures = [clients.submit(client, index)
                       for index in range(args.clients)]
            for future in as_completed(futures):
                future.result()
        elapsed = time.monotonic() - workload_started
        admission = dict(pool.admission_stats())
        cache_snapshots = pool.worker_cache_stats()
        pids = tuple(row.pid for row in cache_snapshots)
        worker_memory = {
            str(pid): process_memory_mib(pid) for pid in pids}
        worker_rss = {pid: row["rss"] for pid, row in worker_memory.items()}
        shard_load = {str(shard): sum(
            1 for memory_id in memories
            if pool.shard_for_memory(memory_id) == shard)
            for shard in range(args.workers)}

    summary = {
        "schema_version": "graphmem-v5.10-multitenant-v1",
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_hash": runtime_config_hash(runtime_config),
        "workload": {
            "duration_requested_sec": args.duration,
            "duration_measured_sec": elapsed,
            "workers": args.workers, "clients": args.clients,
            "tenants": args.tenants, "zipf_alpha": args.zipf,
            "deadline_sec": args.deadline,
            "max_queued": args.max_queued,
            "per_tenant_outstanding": args.per_tenant,
            "affinity_replicas": args.affinity_replicas,
            "cache_memories_per_worker": args.cache_memories,
            "cache_mib_per_worker": args.cache_mib,
            "warm_mode": args.warm_mode,
            "warm_replicas": args.warm_replicas,
            "compiled_cache_dir": (str(args.compiled_cache_dir)
                                   if args.compiled_cache_dir else None),
            "memory_ids": memories,
        },
        "counts": counts,
        "qps": counts["completed"] / max(elapsed, 1e-9),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, .50),
            "p95": percentile(latencies, .95),
            "p99": percentile(latencies, .99),
        },
        "estimated_queue_ms": {
            "mean": statistics.fmean(queue_latencies) if queue_latencies else 0.0,
            "p95": percentile(queue_latencies, .95),
        },
        "burst": {
            "submitted": len(burst_futures),
            "rejected": burst_rejected,
            "requested": args.burst,
            "completed": len(burst_end_to_end_ms),
            "end_to_end_latency_ms": {
                "mean": (statistics.fmean(burst_end_to_end_ms)
                         if burst_end_to_end_ms else 0.0),
                "p50": percentile(burst_end_to_end_ms, .50),
                "p95": percentile(burst_end_to_end_ms, .95),
            },
            "service_latency_ms": {
                "mean": (statistics.fmean(burst_service_ms)
                         if burst_service_ms else 0.0),
                "p50": percentile(burst_service_ms, .50),
                "p95": percentile(burst_service_ms, .95),
            },
        },
        "admission": admission,
        "worker_rss_mib": worker_rss,
        "total_worker_rss_mib": sum(worker_rss.values()),
        "worker_memory_mib": worker_memory,
        "total_worker_pss_mib": sum(
            row["pss"] for row in worker_memory.values()),
        "worker_cache_stats": {
            str(row.pid): row.stats for row in cache_snapshots},
        "warm_snapshots": [{
            "pid": row.pid,
            "memory_id": row.memory_id,
            "graph_version": row.graph_version,
            "graph_checksum": row.graph_checksum,
            "cached_views": row.cached_views,
            "estimated_bytes": row.estimated_bytes,
            "cache_hits": row.cache_hits,
            "cache_misses": row.cache_misses,
            "cache_evictions": row.cache_evictions,
            "compiled_hits": row.compiled_hits,
            "compiled_misses": row.compiled_misses,
            "compiled_hydrations": row.compiled_hydrations,
            "compiled_retained_bytes": row.compiled_retained_bytes,
        } for row in warm],
        "compiled_replica_load_factor": (
            sum(int(row.stats["compiled"].get("hits", 0))
                for row in cache_snapshots)
            / max(1, len(memories))),
        "memory_count_by_primary_affinity_shard": shard_load,
        "correct_snapshot_memory_contract": counts["wrong_memory"] == 0,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
