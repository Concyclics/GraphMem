#!/usr/bin/env python3
"""Measure GraphMem on the frozen worker/client Pareto matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import random
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphmem.config import load_runtime_config, runtime_config_hash  # noqa: E402
from graphmem.serving import (  # noqa: E402
    AdmissionRejected,
    ProcessShardedNavigator,
    RequestDeadlineExceeded,
)
from v5_11_pareto_common import (  # noqa: E402
    CLIENT_LEVELS,
    default_cpu_ids,
    latency_summary,
    load_workload,
    parse_cpu_list,
    parse_int_list,
    popularity_weights,
    process_memory_mib,
    summarize_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path,
        default=ROOT / "configs/v5/runtime_v5_11_balanced.json")
    parser.add_argument("--compiled-cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--clients", default=",".join(map(str, CLIENT_LEVELS)))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--cache-memories", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--cpu-ids", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clients_levels = parse_int_list(args.clients)
    if args.repetitions <= 0 or args.duration <= 0 or args.warmup < 0:
        raise SystemExit("duration/repetitions must be positive and warmup nonnegative")
    cpu_ids = (parse_cpu_list(args.cpu_ids) if args.cpu_ids
               else default_cpu_ids(args.workers))
    if len(cpu_ids) != args.workers:
        raise SystemExit("--cpu-ids must contain one CPU per worker")
    manifest, by_memory = load_workload(args.workload)
    memories = list(manifest["memory_ids"])
    weights = popularity_weights(memories, args.zipf)
    runtime_config = load_runtime_config(args.runtime_config)
    budget = replace(
        runtime_config.query_budget,
        max_evidence_turns=args.top_k,
        max_evidence_tokens=5000,
    )
    navigator_options = {
        **runtime_config.retrieval.navigator_options(
            compiled_cache_dir=args.compiled_cache_dir),
        "snapshot_cache_memories": args.cache_memories,
        "metadata_cache_memories": args.cache_memories,
        "snapshot_cache_bytes": args.cache_mib * 1024 * 1024,
    }
    affinity_replicas = min(runtime_config.serving.affinity_replicas, args.workers)
    max_clients = max(clients_levels)
    args.output.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []

    with ProcessShardedNavigator(
        args.db,
        workers=args.workers,
        navigator_options=navigator_options,
        start_method=runtime_config.serving.start_method,
        max_queued=max(0, max_clients - args.workers),
        per_tenant_outstanding=1,
        affinity_replicas=affinity_replicas,
        retry_broken_worker=runtime_config.serving.retry_broken_worker,
        worker_cpu_ids=cpu_ids,
    ) as pool:
        readiness_started = time.monotonic()
        readiness_rounds = []
        for _round in range(2):
            rows = pool.warm(
                memories[0],
                (by_memory[memories[0]][0]["query"],),
                budget,
            )
            readiness_rounds.append([{
                "pid": row.pid,
                "memory_id": row.memory_id,
                "cached_views": row.cached_views,
                "compiled_hits": row.compiled_hits,
                "compiled_hydrations": row.compiled_hydrations,
            } for row in rows])
        worker_readiness = {
            "rounds": readiness_rounds,
            "elapsed_ms": (time.monotonic() - readiness_started) * 1000,
        }

        def run_interval(clients: int, duration: float, trial: int) -> dict:
            latencies: list[float] = []
            service_latencies: list[float] = []
            queue_latencies: list[float] = []
            result_counts: list[int] = []
            counts = {"completed": 0, "failed": 0, "timed_out": 0,
                      "rejected": 0, "wrong_memory": 0}
            errors: list[str] = []
            lock = threading.Lock()
            barrier = threading.Barrier(clients + 1)
            stop_at_box = [0.0]

            def client(client_id: int) -> None:
                rng = random.Random(710_000 + trial * 10_000 + client_id)
                barrier.wait()
                while time.monotonic() < stop_at_box[0]:
                    memory_id = rng.choices(memories, weights=weights, k=1)[0]
                    row = rng.choice(by_memory[memory_id])
                    submitted = time.monotonic()
                    try:
                        result = pool.submit(
                            memory_id,
                            row["query"],
                            budget,
                            tenant_id=f"user-{client_id}",
                            deadline_monotonic=submitted + args.request_timeout,
                        ).result(timeout=args.request_timeout + 5)
                        elapsed_ms = (time.monotonic() - submitted) * 1000
                        service_ms = float(result.stage_latency_ms.get("total", 0.0))
                        with lock:
                            latencies.append(elapsed_ms)
                            service_latencies.append(service_ms)
                            queue_latencies.append(max(0.0, elapsed_ms - service_ms))
                            result_counts.append(len(result.retrieved_turn_ids))
                            counts["completed"] += 1
                            counts["wrong_memory"] += int(result.memory_id != memory_id)
                    except AdmissionRejected as error:
                        with lock:
                            counts["rejected"] += 1
                            if len(errors) < 10:
                                errors.append(f"{type(error).__name__}: {error}")
                    except (RequestDeadlineExceeded, TimeoutError) as error:
                        with lock:
                            counts["timed_out"] += 1
                            if len(errors) < 10:
                                errors.append(f"{type(error).__name__}: {error}")
                    except BaseException as error:
                        with lock:
                            counts["failed"] += 1
                            if len(errors) < 10:
                                errors.append(f"{type(error).__name__}: {error}")

            threads = []
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=clients) as executor:
                threads = [executor.submit(client, index) for index in range(clients)]
                stop_at_box[0] = time.monotonic() + duration
                barrier.wait()
                for future in as_completed(threads):
                    future.result()
            elapsed = time.monotonic() - started
            return {
                "trial": trial,
                "duration_requested_sec": duration,
                "duration_measured_sec": elapsed,
                **counts,
                "errors": errors,
                "qps": counts["completed"] / max(elapsed, 1e-9),
                "latency_ms": latency_summary(latencies),
                "service_ms": latency_summary(service_latencies),
                "queue_ms": latency_summary(queue_latencies),
                "results_per_query": latency_summary(
                    [float(value) for value in result_counts]),
            }

        for clients in clients_levels:
            warmup_clients = min(clients, max(1, args.workers * 2))
            warmup = (run_interval(warmup_clients, args.warmup, -clients)
                      if args.warmup else None)
            trials = []
            for trial in range(args.repetitions):
                row = run_interval(clients, args.duration, trial)
                trials.append(row)
                print(json.dumps({
                    "system": "GraphMem",
                    "workers": args.workers,
                    "clients": clients,
                    "trial": trial,
                    "qps": row["qps"],
                    "p95_ms": row["latency_ms"]["p95"],
                    "completed": row["completed"],
                }), flush=True)
            cache_rows = pool.worker_cache_stats()
            pids = [row.pid for row in cache_rows]
            worker_memory = {str(pid): process_memory_mib(pid) for pid in pids}
            cell = {
                "system": "GraphMem V5.11",
                "workers": args.workers,
                "clients": clients,
                "logical_users": clients,
                "warmup_clients": warmup_clients,
                "warmup": warmup,
                "trials": trials,
                "aggregate": summarize_trials(trials),
                "worker_pids": pids,
                "worker_memory_mib": worker_memory,
                "total_worker_rss_mib": sum(row["rss"] for row in worker_memory.values()),
                "total_worker_pss_mib": sum(row["pss"] for row in worker_memory.values()),
                "worker_cache_stats": {str(row.pid): row.stats for row in cache_rows},
                "admission": dict(pool.admission_stats()),
            }
            cells.append(cell)
            (args.output / f"clients_{clients}.json").write_text(
                json.dumps(cell, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    summary = {
        "schema_version": "graphmem-mem0-pareto-result-v1",
        "system": {
            "name": "GraphMem V5.11",
            "mode": "QueryIR + hierarchical graph + compiled immutable views",
            "query_embedding": False,
        },
        "protocol": manifest["protocol"],
        "workload_sha256": manifest["source"]["turn_payload_sha256"],
        "query_count": len(manifest["queries"]),
        "memory_count": manifest["memory_count"],
        "workers": args.workers,
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_hash": runtime_config_hash(runtime_config),
        "worker_cpu_ids": cpu_ids,
        "affinity_replicas": affinity_replicas,
        "cache_memories_per_worker": args.cache_memories,
        "cache_mib_per_worker": args.cache_mib,
        "compiled_cache_dir": str(args.compiled_cache_dir.resolve()),
        "worker_readiness": worker_readiness,
        "duration_sec": args.duration,
        "warmup_sec": args.warmup,
        "repetitions": args.repetitions,
        "cells": cells,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(args.output / "summary.json"),
                      "cells": len(cells)}))


if __name__ == "__main__":
    main()
