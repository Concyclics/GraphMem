#!/usr/bin/env python3
"""Measure Mem0 OSS on the frozen worker/client Pareto matrix."""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v5_11_pareto_common import (  # noqa: E402
    CLIENT_LEVELS,
    affinity_shards,
    default_cpu_ids,
    latency_summary,
    load_workload,
    parse_cpu_list,
    parse_int_list,
    popularity_weights,
    process_memory_mib,
    summarize_trials,
)


_MEMORY = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--clients", default=",".join(map(str, CLIENT_LEVELS)))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dims", type=int, default=1024)
    parser.add_argument("--cpu-ids", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _close_worker() -> None:
    global _MEMORY
    if _MEMORY is not None:
        try:
            _MEMORY.vector_store.client.close()
        except BaseException:
            pass
    _MEMORY = None


def _initialize_worker(config: dict, state_dir: str, cpu_id: int) -> None:
    global _MEMORY
    os.sched_setaffinity(0, {cpu_id})
    os.environ["MEM0_DIR"] = state_dir
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["OPENAI_API_KEY"] = "local-benchmark"
    from mem0 import Memory
    _MEMORY = Memory.from_config(config)
    atexit.register(_close_worker)


def _search_worker(request: tuple[str, str, int]) -> dict[str, Any]:
    if _MEMORY is None:
        raise RuntimeError("Mem0 worker was not initialized")
    memory_id, query, top_k = request
    started = time.monotonic()
    payload = _MEMORY.search(
        query,
        top_k=top_k,
        filters={"user_id": memory_id},
        rerank=False,
    )
    service_ms = (time.monotonic() - started) * 1000
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    wrong_user = 0
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        returned_user = row.get("user_id") if isinstance(row, dict) else None
        returned_user = returned_user or metadata.get("user_id")
        wrong_user += int(returned_user not in (None, memory_id))
    return {
        "service_ms": service_ms,
        "result_count": len(rows),
        "wrong_user": wrong_user,
        "pid": os.getpid(),
    }


def _worker_stats() -> dict[str, Any]:
    return {"pid": os.getpid(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))}


class Mem0ShardedPool:
    def __init__(self, args: argparse.Namespace, cpu_ids: tuple[int, ...]) -> None:
        self.workers = args.workers
        self.replicas = min(2, args.workers)
        self._context = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._outstanding = [0] * args.workers
        self._executors: list[ProcessPoolExecutor] = []
        for shard in range(args.workers):
            qdrant_path = args.replica_root / f"replica_{shard}"
            if not (qdrant_path / "graphmem_pareto_manifest.json").is_file():
                raise RuntimeError(f"missing prepared Mem0 replica: {qdrant_path}")
            state_dir = args.output / "worker_state" / f"worker_{shard}"
            state_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "vector_store": {"provider": "qdrant", "config": {
                    "collection_name": args.collection,
                    "embedding_model_dims": args.embedding_dims,
                    "path": str(qdrant_path),
                    "on_disk": False,
                }},
                "embedder": {"provider": "openai", "config": {
                    "model": args.embedding_model,
                    "openai_base_url": args.embedding_url,
                    "api_key": "local-benchmark",
                }},
                "llm": {"provider": "openai", "config": {
                    "model": "unused-infer-false",
                    "api_key": "local-benchmark",
                    "openai_base_url": "http://127.0.0.1:8002/v1",
                }},
                "history_db_path": str(state_dir / "history.db"),
            }
            self._executors.append(ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._context,
                initializer=_initialize_worker,
                initargs=(config, str(state_dir), cpu_ids[shard]),
            ))

    def submit(self, memory_id: str, query: str, top_k: int) -> Future:
        with self._lock:
            shard = min(
                affinity_shards(memory_id, self.workers, self.replicas),
                key=lambda value: (self._outstanding[value], value),
            )
            self._outstanding[shard] += 1
        inner = self._executors[shard].submit(
            _search_worker, (memory_id, query, top_k))

        def release(_future: Future) -> None:
            with self._lock:
                self._outstanding[shard] -= 1

        inner.add_done_callback(release)
        return inner

    def worker_stats(self) -> list[dict[str, Any]]:
        return [executor.submit(_worker_stats).result()
                for executor in self._executors]

    def warm_all(self, memory_id: str, query: str, top_k: int,
                 *, rounds: int = 2) -> dict[str, Any]:
        started = time.monotonic()
        results = []
        for round_index in range(rounds):
            futures = [executor.submit(
                _search_worker, (memory_id, query, top_k))
                for executor in self._executors]
            rows = [future.result() for future in futures]
            results.append({"round": round_index, "workers": rows})
        return {
            "rounds": results,
            "elapsed_ms": (time.monotonic() - started) * 1000,
        }

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "Mem0ShardedPool":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def main() -> None:
    args = parse_args()
    clients_levels = parse_int_list(args.clients)
    cpu_ids = (parse_cpu_list(args.cpu_ids) if args.cpu_ids
               else default_cpu_ids(args.workers))
    if len(cpu_ids) != args.workers:
        raise SystemExit("--cpu-ids must contain one CPU per worker")
    manifest, by_memory = load_workload(args.workload)
    memories = list(manifest["memory_ids"])
    weights = popularity_weights(memories, args.zipf)
    args.output.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []

    with Mem0ShardedPool(args, cpu_ids) as pool:
        worker_readiness = pool.warm_all(
            memories[0], by_memory[memories[0]][0]["query"], args.top_k)

        def run_interval(clients: int, duration: float, trial: int) -> dict:
            latencies: list[float] = []
            service_latencies: list[float] = []
            queue_latencies: list[float] = []
            result_counts: list[int] = []
            counts = {"completed": 0, "failed": 0, "timed_out": 0,
                      "rejected": 0, "wrong_user": 0}
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
                            memory_id, row["query"], args.top_k,
                        ).result(timeout=args.request_timeout)
                        elapsed_ms = (time.monotonic() - submitted) * 1000
                        service_ms = float(result["service_ms"])
                        with lock:
                            latencies.append(elapsed_ms)
                            service_latencies.append(service_ms)
                            queue_latencies.append(max(0.0, elapsed_ms - service_ms))
                            result_counts.append(int(result["result_count"]))
                            counts["completed"] += 1
                            counts["wrong_user"] += int(result["wrong_user"])
                    except TimeoutError as error:
                        with lock:
                            counts["timed_out"] += 1
                            if len(errors) < 10:
                                errors.append(f"{type(error).__name__}: {error}")
                    except BaseException as error:
                        with lock:
                            counts["failed"] += 1
                            if len(errors) < 10:
                                errors.append(f"{type(error).__name__}: {error}")

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=clients) as executor:
                futures = [executor.submit(client, index) for index in range(clients)]
                stop_at_box[0] = time.monotonic() + duration
                barrier.wait()
                for future in as_completed(futures):
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
                    "system": "Mem0 OSS",
                    "workers": args.workers,
                    "clients": clients,
                    "trial": trial,
                    "qps": row["qps"],
                    "p95_ms": row["latency_ms"]["p95"],
                    "completed": row["completed"],
                }), flush=True)
            stats = pool.worker_stats()
            pids = [int(row["pid"]) for row in stats]
            worker_memory = {str(pid): process_memory_mib(pid) for pid in pids}
            cell = {
                "system": "Mem0 OSS 2.0.17",
                "workers": args.workers,
                "clients": clients,
                "logical_users": clients,
                "warmup_clients": warmup_clients,
                "warmup": warmup,
                "trials": trials,
                "aggregate": summarize_trials(trials),
                "worker_pids": pids,
                "worker_process_stats": stats,
                "worker_memory_mib": worker_memory,
                "total_worker_rss_mib": sum(row["rss"] for row in worker_memory.values()),
                "total_worker_pss_mib": sum(row["pss"] for row in worker_memory.values()),
            }
            cells.append(cell)
            (args.output / f"clients_{clients}.json").write_text(
                json.dumps(cell, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    summary = {
        "schema_version": "graphmem-mem0-pareto-result-v1",
        "system": {
            "name": "Mem0 OSS",
            "version": "2.0.17",
            "mode": "infer=False raw turns + Qwen query embedding + Qdrant dense search",
            "query_embedding": True,
            "reranker": False,
            "bm25": False,
        },
        "protocol": manifest["protocol"],
        "workload_sha256": manifest["source"]["turn_payload_sha256"],
        "query_count": len(manifest["queries"]),
        "memory_count": manifest["memory_count"],
        "workers": args.workers,
        "worker_cpu_ids": cpu_ids,
        "affinity_replicas": min(2, args.workers),
        "duration_sec": args.duration,
        "warmup_sec": args.warmup,
        "repetitions": args.repetitions,
        "embedding_url": args.embedding_url,
        "embedding_model": args.embedding_model,
        "collection": args.collection,
        "worker_readiness": worker_readiness,
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
