#!/usr/bin/env python3
"""Run a reproducible Mem0 OSS raw-turn, multi-user retrieval sample.

This deliberately uses ``infer=False`` so that the sample isolates the memory
data plane (embedding + vector store) from Mem0's LLM extraction policy.  Each
LoCoMo conversation is mapped to one Mem0 ``user_id`` and queries are always
filtered by that ID.  The companion GraphMem benchmark accepts the same
``--memory-ids`` list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_DEV = WORKSPACE / (
    "artifacts/development_sets/"
    "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804/"
    "locomo_hard_cat1_multihop50_cat2_temporal50.json")
DEFAULT_DB = WORKSPACE / (
    "artifacts/v5_10/full_benchmark_20260809/graph/report_graph.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--locomo", type=Path, default=DEFAULT_DEV)
    parser.add_argument(
        "--memory-ids",
        default="locomo:conv-26,locomo:conv-30,locomo:conv-41,locomo:conv-42")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dims", type=int, default=1024)
    parser.add_argument("--qdrant-path", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/mem0_qdrant_raw4")
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/mem0_raw4_c8_20s")
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse an exactly matching completed ingest.")
    parser.add_argument("--max-turns-per-memory", type=int, default=0,
                        help="Smoke-only cap; zero ingests every source turn.")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


def self_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def selected_ids(raw: str) -> list[str]:
    rows = [value.strip() for value in raw.split(",") if value.strip()]
    if not rows:
        raise SystemExit("--memory-ids selected no memories")
    return rows


def load_questions(path: Path, memory_ids: list[str]) -> dict[str, list[str]]:
    selected = set(memory_ids)
    by_memory: dict[str, list[str]] = {memory_id: [] for memory_id in memory_ids}
    for row in json.loads(path.read_text(encoding="utf-8")):
        memory_id = "locomo:" + row["locomo_sample_id"]
        if memory_id in selected:
            by_memory[memory_id].append(str(row["question"]))
    missing = [key for key, rows in by_memory.items() if not rows]
    if missing:
        raise SystemExit("No LoCoMo questions for: " + ", ".join(missing))
    return by_memory


def load_turns(
        db_path: Path, memory_ids: list[str], max_turns: int
) -> dict[str, list[dict[str, str]]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    result: dict[str, list[dict[str, str]]] = {}
    try:
        for memory_id in memory_ids:
            rows = con.execute(
                """SELECT turn_id, session_id, turn_index, role, speaker,
                          timestamp, raw_text
                   FROM source_turns WHERE memory_id=?
                   ORDER BY session_id, turn_index, turn_id""",
                (memory_id,)).fetchall()
            if max_turns:
                rows = rows[:max_turns]
            if not rows:
                raise SystemExit(f"No source turns for {memory_id}")
            result[memory_id] = [{
                "turn_id": str(row[0]), "session_id": str(row[1]),
                "turn_index": str(row[2]), "role": str(row[3]),
                "speaker": str(row[4]), "timestamp": str(row[5] or ""),
                "raw_text": str(row[6]),
            } for row in rows]
    finally:
        con.close()
    return result


def source_digest(turns: dict[str, list[dict[str, str]]]) -> str:
    digest = hashlib.sha256()
    for memory_id in sorted(turns):
        for row in turns[memory_id]:
            digest.update(memory_id.encode())
            digest.update(b"\0")
            digest.update(row["turn_id"].encode())
            digest.update(b"\0")
            digest.update(row["raw_text"].encode())
            digest.update(b"\n")
    return digest.hexdigest()


def search_results(payload) -> list[dict]:
    if isinstance(payload, dict):
        rows = payload.get("results", [])
        return rows if isinstance(rows, list) else []
    return payload if isinstance(payload, list) else []


def main() -> None:
    args = parse_args()
    memory_ids = selected_ids(args.memory_ids)
    by_memory = load_questions(args.locomo, memory_ids)
    turns = load_turns(args.db, memory_ids, args.max_turns_per_memory)
    digest = source_digest(turns)
    args.output.mkdir(parents=True, exist_ok=True)
    args.qdrant_path.parent.mkdir(parents=True, exist_ok=True)

    # These must be set before importing Mem0: it otherwise writes under $HOME.
    os.environ.setdefault("MEM0_DIR", str(args.output / "mem0_state"))
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    os.environ.setdefault("OPENAI_API_KEY", "local-benchmark")
    from mem0 import Memory
    import mem0

    collection = "graphmem_v510_raw4_" + digest[:12]
    config = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": collection,
            "embedding_model_dims": args.embedding_dims,
            "path": str(args.qdrant_path), "on_disk": False,
        }},
        "embedder": {"provider": "openai", "config": {
            "model": args.embedding_model,
            # Do not pass ``dimensions`` to vLLM: Qwen3-Embedding is not a
            # matryoshka model.  Qdrant still needs the actual 1024-vector size.
            "openai_base_url": args.embedding_url,
            "api_key": "local-benchmark",
        }},
        # infer=False means the LLM is not invoked, but Memory requires a valid
        # provider configuration during construction.
        "llm": {"provider": "openai", "config": {
            "model": "unused-infer-false", "api_key": "local-benchmark",
            "openai_base_url": "http://127.0.0.1:8002/v1",
        }},
        "history_db_path": str(args.output / "history.db"),
    }
    memory = Memory.from_config(config)
    manifest_path = args.qdrant_path / "graphmem_sample_manifest.json"
    expected_manifest = {
        "schema_version": "mem0-raw-turn-sample-v1",
        "collection": collection,
        "source_sha256": digest,
        "memory_ids": memory_ids,
        "turn_counts": {key: len(value) for key, value in turns.items()},
        "embedding_model": args.embedding_model,
        "embedding_dims": args.embedding_dims,
        "infer": False,
    }
    reusable = False
    if args.reuse and manifest_path.exists():
        reusable = json.loads(manifest_path.read_text()) == expected_manifest
        if not reusable:
            raise SystemExit("--reuse manifest does not exactly match this run")

    ingest_ms: dict[str, float] = {}
    if not reusable:
        if manifest_path.exists():
            raise SystemExit(
                "Qdrant path has a sample manifest; choose a fresh path or use "
                "--reuse with the exact same source")
        for memory_id in memory_ids:
            messages = []
            for row in turns[memory_id]:
                prefix = (f"[{row['session_id']}][{row['timestamp']}] "
                          f"{row['speaker']}: ")
                messages.append({
                    "role": row["role"] if row["role"] in {
                        "user", "assistant"} else "user",
                    "name": row["speaker"],
                    "content": prefix + row["raw_text"],
                })
            started = time.monotonic()
            memory.add(messages, user_id=memory_id, infer=False,
                       metadata={"source": "graphmem-v5.10-locomo-raw-turn"})
            ingest_ms[memory_id] = (time.monotonic() - started) * 1000
            print(json.dumps({"ingested": memory_id,
                              "turns": len(messages),
                              "latency_ms": ingest_ms[memory_id]}), flush=True)
        manifest_path.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    # Warm exactly one query per user outside the measured interval.
    warm_latencies = {}
    for memory_id in memory_ids:
        started = time.monotonic()
        memory.search(by_memory[memory_id][0], top_k=args.top_k,
                      filters={"user_id": memory_id}, rerank=False)
        warm_latencies[memory_id] = (time.monotonic() - started) * 1000

    weights = [1.0 / ((index + 1) ** args.zipf)
               for index in range(len(memory_ids))]
    latencies: list[float] = []
    result_counts: list[int] = []
    counts = {"completed": 0, "failed": 0, "wrong_user": 0}
    errors: list[str] = []
    lock = threading.Lock()
    stop_at = time.monotonic() + args.duration

    def client(client_id: int) -> None:
        rng = random.Random(52000 + client_id)
        while time.monotonic() < stop_at:
            memory_id = rng.choices(memory_ids, weights=weights, k=1)[0]
            query = rng.choice(by_memory[memory_id])
            started = time.monotonic()
            try:
                payload = memory.search(
                    query, top_k=args.top_k,
                    filters={"user_id": memory_id}, rerank=False)
                elapsed_ms = (time.monotonic() - started) * 1000
                rows = search_results(payload)
                wrong = 0
                for row in rows:
                    metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
                    returned_user = row.get("user_id") if isinstance(row, dict) else None
                    returned_user = returned_user or metadata.get("user_id")
                    wrong += int(returned_user not in (None, memory_id))
                with lock:
                    latencies.append(elapsed_ms)
                    result_counts.append(len(rows))
                    counts["completed"] += 1
                    counts["wrong_user"] += wrong
            except BaseException as exc:
                with lock:
                    counts["failed"] += 1
                    if len(errors) < 10:
                        errors.append(f"{type(exc).__name__}: {exc}")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.clients) as executor:
        futures = [executor.submit(client, index) for index in range(args.clients)]
        for future in as_completed(futures):
            future.result()
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": "mem0-raw-turn-multitenant-v1",
        "system": {
            "name": "Mem0 OSS", "version": getattr(mem0, "__version__", "unknown"),
            "mode": "raw source turns, infer=False, dense semantic search",
            "vector_store": "Qdrant local embedded, vectors in memory",
            "embedding_model": args.embedding_model,
            "collection": collection,
        },
        "workload": {
            "duration_requested_sec": args.duration,
            "duration_measured_sec": elapsed,
            "clients": args.clients, "users": len(memory_ids),
            "memory_ids": memory_ids, "zipf_alpha": args.zipf,
            "top_k": args.top_k,
            "turn_counts": {key: len(value) for key, value in turns.items()},
            "question_counts": {key: len(value) for key, value in by_memory.items()},
        },
        "counts": counts,
        "errors": errors,
        "qps": counts["completed"] / max(elapsed, 1e-9),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, .50),
            "p95": percentile(latencies, .95),
            "p99": percentile(latencies, .99),
        },
        "results_per_query": {
            "mean": statistics.fmean(result_counts) if result_counts else 0.0,
            "min": min(result_counts, default=0),
            "max": max(result_counts, default=0),
        },
        "warmup_latency_ms": warm_latencies,
        "ingest": {"reused": reusable, "latency_ms_by_user": ingest_ms},
        "process_rss_mib": self_rss_mib(),
        "correct_user_filter_contract": counts["wrong_user"] == 0,
        "limitations": [
            "This is a retrieval data-plane sample, not Mem0's full infer=True pipeline.",
            "Mem0 is one Python process with embedded Qdrant; GraphMem worker counts must be reported.",
            "No reranker or sparse/BM25 dependency is enabled; search is dense-only.",
            "QPS/latency are hardware- and local-embedding-service-specific.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
