#!/usr/bin/env python3
"""Compare serial SQLite dense retrieval with batched SQLite and FAISS sidecars."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config, load_runtime_config  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.retrieval import GraphNavigator  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True,
                        help="answer JSONL carrying question_id and question")
    parser.add_argument("--retrieval", type=Path, required=True,
                        help="retrieval JSONL carrying question_id to memory_id")
    parser.add_argument("--dense-sidecar-dir", type=Path, required=True)
    parser.add_argument("--query-embedding-cache", type=Path,
                        help="optional prefilled cache; use when 8001 is unavailable")
    parser.add_argument("--warm-repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def main() -> None:
    args = parse_args()
    if args.warm_repetitions <= 0:
        raise SystemExit("--warm-repetitions must be positive")
    answers = {
        str(row["question_id"]): str(row["question"])
        for row in read_jsonl(args.answers)
    }
    retrieval = read_jsonl(args.retrieval)
    questions = []
    for row in retrieval:
        question_id = str(row["dev_question_id"])
        if question_id not in answers:
            raise ValueError(f"answer text missing for {question_id}")
        questions.append({
            "question_id": question_id,
            "memory_id": str(row["memory_id"]),
            "query": answers[question_id],
            "benchmark": str(row.get("benchmark", "")),
            "stratum": str(row.get("stratum", "")),
        })
    build_config = load_config(args.config)
    runtime = load_runtime_config(args.runtime_config)
    navigator_options = runtime.retrieval.navigator_options(compiled_cache_dir="")

    variants: dict[str, tuple[SQLiteGraphStore, QwenEmbeddingIndex, GraphNavigator]] = {}
    for name, batched, sidecar in (
        ("serial_sqlite", False, None),
        ("batched_sqlite", True, None),
        ("batched_faiss", True, args.dense_sidecar_dir),
    ):
        store = SQLiteGraphStore(args.db, read_only=True)
        embedding = QwenEmbeddingIndex(
            store,
            build_config,
            record_usage=False,
            dense_sidecar_dir=sidecar,
            dense_backend="faiss_flat" if sidecar else "auto",
            query_cache_path=args.query_embedding_cache,
            memory_cache_memories=32,
            dense_cache_memories=32,
        )
        navigator = GraphNavigator(
            store,
            dense_search=embedding.search,
            dense_search_many=embedding.search_many if batched else None,
            **navigator_options,
        )
        variants[name] = (store, embedding, navigator)

    details: list[dict[str, Any]] = []
    results: dict[tuple[str, str, int, str], Any] = {}
    names = tuple(variants)
    started = time.perf_counter()
    try:
        for phase, repetitions in (("cold", 1), ("warm", args.warm_repetitions)):
            for repetition in range(repetitions):
                for question_index, question in enumerate(questions):
                    # Rotate the execution order so embedding-service drift and
                    # OS page warmth do not always favour the last variant.
                    offset = (question_index + repetition) % len(names)
                    order = names[offset:] + names[:offset]
                    for name in order:
                        navigator = variants[name][2]
                        tick = time.perf_counter()
                        result = navigator.navigate(
                            question["memory_id"], question["query"],
                            runtime.query_budget)
                        wall_ms = (time.perf_counter() - tick) * 1000
                        results[(phase, question["question_id"], repetition, name)] = result
                        details.append({
                            **question,
                            "phase": phase,
                            "repetition": repetition,
                            "variant": name,
                            "wall_ms": wall_ms,
                            "stage_latency_ms": dict(result.stage_latency_ms),
                            "retrieved_turn_ids": list(result.retrieved_turn_ids),
                            "candidate_turn_ids": [
                                row.turn_id for row in result.candidate_scores],
                            "evidence_tokens": result.evidence_tokens,
                        })
    finally:
        for store, _embedding, _navigator in variants.values():
            store.close()

    summary: dict[str, Any] = {}
    for phase in ("cold", "warm"):
        summary[phase] = {}
        for name in names:
            rows = [row for row in details
                    if row["phase"] == phase and row["variant"] == name]
            stage_names = sorted({key for row in rows for key in row["stage_latency_ms"]})
            summary[phase][name] = {
                "queries": len(rows),
                "wall_ms": summarize([float(row["wall_ms"]) for row in rows]),
                "stage_mean_ms": {
                    stage: statistics.fmean(
                        float(row["stage_latency_ms"].get(stage, 0.0)) for row in rows)
                    for stage in stage_names
                },
            }

    parity: dict[str, Any] = {}
    for name in names[1:]:
        rows = []
        for question in questions:
            baseline = results[("cold", question["question_id"], 0, "serial_sqlite")]
            candidate = results[("cold", question["question_id"], 0, name)]
            base_set = set(baseline.candidate_scores[index].turn_id
                           for index in range(len(baseline.candidate_scores)))
            candidate_set = set(candidate.candidate_scores[index].turn_id
                                for index in range(len(candidate.candidate_scores)))
            union = base_set | candidate_set
            rows.append({
                "question_id": question["question_id"],
                "retrieved_order_equal": (
                    baseline.retrieved_turn_ids == candidate.retrieved_turn_ids),
                "retrieved_set_equal": (
                    set(baseline.retrieved_turn_ids) == set(candidate.retrieved_turn_ids)),
                "candidate_jaccard": (
                    len(base_set & candidate_set) / len(union) if union else 1.0),
            })
        parity[name] = {
            "retrieved_order_equal": sum(row["retrieved_order_equal"] for row in rows),
            "retrieved_set_equal": sum(row["retrieved_set_equal"] for row in rows),
            "candidate_jaccard_mean": statistics.fmean(
                row["candidate_jaccard"] for row in rows),
            "questions": rows,
        }

    embedding_stats = {
        name: dict(embedding.stats)
        for name, (_store, embedding, _navigator) in variants.items()
    }
    payload = {
        "schema_version": "graphmem-v5.18-dense-index-benchmark-v1",
        "db": str(args.db),
        "config": str(args.config),
        "runtime_config": str(args.runtime_config),
        "dense_sidecar_dir": str(args.dense_sidecar_dir),
        "query_embedding_cache": (str(args.query_embedding_cache)
                                  if args.query_embedding_cache else None),
        "cold_phase_note": ("graph/index cold; query embeddings persistent-warm"
                            if args.query_embedding_cache else
                            "graph/index/query-embedding cold"),
        "questions": len(questions),
        "warm_repetitions": args.warm_repetitions,
        "wall_time_sec": time.perf_counter() - started,
        "summary": summary,
        "parity_vs_serial_sqlite": parity,
        "embedding_stats": embedding_stats,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
