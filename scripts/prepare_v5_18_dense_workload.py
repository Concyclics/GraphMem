#!/usr/bin/env python3
"""Build the 16-memory QPS workload and an offline warm-query cache.

The cache vectors are sampled from existing normalized turn embeddings.  They
are suitable only for measuring the warm query data plane while the embedding
service is unavailable; accuracy/parity runs must use real query embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config, load_runtime_config  # noqa: E402
from graphmem.embedding import QUERY_INSTRUCTION, QUERY_INSTRUCTION_REVISION  # noqa: E402
from graphmem.query_embedding_cache import QueryEmbeddingCache  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.retrieval.query_ir import compile_query  # noqa: E402
from graphmem.retrieval.seeding import build_views  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--workload-output", type=Path, required=True)
    parser.add_argument("--query-cache-output", type=Path, required=True)
    parser.add_argument("--mem0-query-cache-output", type=Path,
                        help="optional raw query->vector cache for warm Qdrant comparison")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = load_runtime_config(args.runtime_config)
    answers = {
        str(row["question_id"]): str(row["question"])
        for row in read_jsonl(args.answers)
    }
    retrieval = read_jsonl(args.retrieval)
    queries = []
    for row in retrieval:
        question_id = str(row["dev_question_id"])
        queries.append({
            "question_id": question_id,
            "memory_id": str(row["memory_id"]),
            "benchmark": str(row.get("benchmark", "")),
            "stratum": str(row.get("stratum", "")),
            "query": answers[question_id],
        })
    queries.sort(key=lambda row: row["question_id"])

    store = SQLiteGraphStore(args.db, read_only=True)
    navigator = GraphNavigator(
        store, harness_profile=HarnessProfile.H11_UNIFIED_IR,
        compiled_cache_dir=None,
        queryir_soft_fallback=runtime.retrieval.queryir_soft_fallback,
        queryir_soft_fallback_threshold=(
            runtime.retrieval.queryir_soft_fallback_threshold),
    )
    cached_vectors: dict[str, np.ndarray] = {}
    mem0_vectors: dict[str, list[float]] = {}
    payload_hash = hashlib.sha256()
    turn_counts: dict[str, int] = {}
    for memory_id in sorted({row["memory_id"] for row in queries}):
        turns = tuple(store.turns(memory_id))
        turn_counts[memory_id] = len(turns)
        for turn in turns:
            payload_hash.update(memory_id.encode("utf-8"))
            payload_hash.update(b"\0")
            payload_hash.update(turn.turn_id.encode("utf-8"))
            payload_hash.update(b"\0")
            payload_hash.update(turn.content_hash.encode("ascii"))
        embedding_row = store._read_one(
            "SELECT dimension,vector FROM embeddings "
            "WHERE memory_id=? AND model_id=? ORDER BY item_id LIMIT 1",
            (memory_id, config.models.embedding_model),
        )
        if embedding_row is None:
            raise ValueError(f"no turn embedding for {memory_id}")
        seed_vector = np.frombuffer(
            embedding_row["vector"], dtype=np.float32,
            count=int(embedding_row["dimension"])).copy()
        view = navigator.runtime.view(memory_id)
        registry = navigator._principals(memory_id, view)
        for row in (item for item in queries if item["memory_id"] == memory_id):
            mem0_vectors[row["query"]] = seed_vector.astype(float).tolist()
            compiled = compile_query(row["query"], view, registry=registry)
            ir = compiled.promote_ast()
            if (runtime.retrieval.queryir_soft_fallback
                    and compiled.compile_confidence
                    < runtime.retrieval.queryir_soft_fallback_threshold):
                ir = ir.soften_with_legacy(compiled)
            for query_view in build_views(
                    ir, max_per_operand=runtime.query_budget.max_query_views_per_operand):
                if not query_view.dense:
                    continue
                query_text = QUERY_INSTRUCTION + query_view.text
                cache_key = hashlib.sha256((
                    config.models.embedding_model + "\n" + QUERY_INSTRUCTION_REVISION
                    + "\n" + query_text
                ).encode()).hexdigest()
                cached_vectors[cache_key] = seed_vector
    QueryEmbeddingCache(args.query_cache_output).put_many(cached_vectors)
    if args.mem0_query_cache_output is not None:
        args.mem0_query_cache_output.parent.mkdir(parents=True, exist_ok=True)
        args.mem0_query_cache_output.write_text(
            json.dumps(mem0_vectors, ensure_ascii=False) + "\n", encoding="utf-8")
    store.close()

    workload = {
        "schema_version": "graphmem-mem0-pareto-workload-v1",
        "protocol": {
            "plane": "retrieval data plane",
            "query_mode": "closed-loop; one outstanding request per logical user",
            "top_k": runtime.query_budget.max_evidence_turns,
            "zipf_alpha": 1.1,
            "clients": [1, 4, 16, 64, 128, 256],
            "workers": [1, 4, 8],
            "affinity_replicas": "min(2, workers)",
            "query_cache": "offline warm-cache vectors; performance only, not accuracy",
        },
        "source": {
            "db": str(args.db.resolve()),
            "db_size_bytes": args.db.stat().st_size,
            "turn_payload_sha256": payload_hash.hexdigest(),
            "answers": str(args.answers.resolve()),
            "retrieval": str(args.retrieval.resolve()),
        },
        "memory_ids": sorted({row["memory_id"] for row in queries}),
        "memory_count": len({row["memory_id"] for row in queries}),
        "turn_counts": turn_counts,
        "turn_count": sum(turn_counts.values()),
        "queries": queries,
    }
    args.workload_output.parent.mkdir(parents=True, exist_ok=True)
    args.workload_output.write_text(
        json.dumps(workload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "workload": str(args.workload_output),
        "memories": workload["memory_count"],
        "queries": len(queries),
        "cached_query_views": len(cached_vectors),
        "query_cache": str(args.query_cache_output),
        "mem0_query_cache": (str(args.mem0_query_cache_output)
                             if args.mem0_query_cache_output else None),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
