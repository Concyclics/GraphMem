#!/usr/bin/env python3
"""Run a no-answer full-corpus retrieval gate and retain candidate rankings."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.eval.metrics import navigation_metrics  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def _nearest(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _nearest(values, 0.50),
        "p95": _nearest(values, 0.95),
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dense-sidecar-dir", type=Path, required=True)
    parser.add_argument("--query-embedding-cache", type=Path, required=True)
    parser.add_argument("--dialogue-response-closure", action="store_true")
    parser.add_argument("--proof-priority-bonus", type=float)
    parser.add_argument("--proof-priority-flood-threshold", type=int, default=0)
    parser.add_argument("--speaker-owner-bonus", type=float, default=0.0)
    parser.add_argument("--query-witness-bonus", type=float, default=0.0)
    parser.add_argument("--query-witness-seed-count", type=int, default=16)
    parser.add_argument("--query-witness-rare-df", type=int, default=4)
    parser.add_argument("--query-witness-min-shared-terms", type=int, default=2)
    parser.add_argument("--dialogue-response-flood-threshold", type=int, default=0)
    parser.add_argument("--max-evidence-turns", type=int, default=64)
    parser.add_argument("--max-evidence-tokens", type=int, default=12_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    rows_path = args.output_root / "retrieval_candidates.jsonl"
    completed: dict[str, dict] = {}
    if args.resume and rows_path.exists():
        completed = {
            str(row["question_id"]): row
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}

    config = load_config(args.config)
    questions = load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    embedding = QwenEmbeddingIndex(
        store, config, record_usage=False,
        query_cache_path=args.query_embedding_cache,
        dense_sidecar_dir=args.dense_sidecar_dir,
        dense_backend="auto")
    navigator = GraphNavigator(
        store,
        dense_search=embedding.search,
        dense_search_many=embedding.search_many,
        harness_profile=HarnessProfile.H11_UNIFIED_IR,
        hierarchical_routing=True,
        obligation_aware_packing=True,
        precision_aware_packing=False,
        native_seed_fusion=True,
        queryir_soft_fallback=True,
        queryir_soft_fallback_threshold=0.80,
        graph_hop_decay=0.3,
        expansion_beam=2,
        hierarchy_descent_beam=1,
        dialogue_response_closure=args.dialogue_response_closure,
        dialogue_response_flood_threshold=(
            args.dialogue_response_flood_threshold),
        proof_priority_bonus=args.proof_priority_bonus,
        proof_priority_flood_threshold=args.proof_priority_flood_threshold,
        speaker_owner_bonus=args.speaker_owner_bonus,
        query_witness_bonus=args.query_witness_bonus,
        query_witness_seed_count=args.query_witness_seed_count,
        query_witness_rare_df=args.query_witness_rare_df,
        query_witness_min_shared_terms=args.query_witness_min_shared_terms,
    )
    budget = replace(
        config.query_budget,
        max_evidence_turns=args.max_evidence_turns,
        max_evidence_tokens=args.max_evidence_tokens)
    started = time.perf_counter()
    with rows_path.open("a", encoding="utf-8") as handle:
        for index, question in enumerate(questions, 1):
            if question.question_id in completed:
                continue
            result = navigator.navigate(question.memory_id, question.query, budget)
            metric = navigation_metrics(question.question, result, store)
            candidate_rows = [{
                "turn_id": item.turn_id,
                "rank": rank,
                "fused_score": item.fused_score,
                "exact_score": item.exact_score,
                "bm25_score": item.bm25_score,
                "dense_score": item.dense_score,
                "graph_score": item.graph_score,
                "adjacency_score": item.adjacency_score,
                "binding_score": item.binding_score,
                "operand_ids": list(item.operand_ids),
                "mandatory": item.mandatory,
                "packed": item.turn_id in set(result.retrieved_turn_ids),
            } for rank, item in enumerate(result.candidate_scores, 1)]
            row = {
                "question_id": question.question_id,
                "benchmark": question.benchmark,
                "stratum": question.stratum,
                "memory_id": question.memory_id,
                "query": question.query,
                "has_turn_gold": question.has_turn_gold,
                "retrieved_turn_ids": list(result.retrieved_turn_ids),
                "metrics": metric,
                "trace": result.trace,
                "candidate_scores": candidate_rows,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            completed[question.question_id] = row
            if index % 25 == 0:
                elapsed = time.perf_counter() - started
                print(f"retrieved {index}/{len(questions)} elapsed={elapsed:.1f}s",
                      flush=True)

    ordered = [completed[row.question_id] for row in questions]
    rows_path.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in ordered),
        encoding="utf-8")
    summary = {"questions": len(ordered), "dialogue_response_closure": (
        args.dialogue_response_closure),
        "proof_priority_bonus": args.proof_priority_bonus,
        "proof_priority_flood_threshold": args.proof_priority_flood_threshold,
        "speaker_owner_bonus": args.speaker_owner_bonus,
        "query_witness_bonus": args.query_witness_bonus,
        "query_witness_seed_count": args.query_witness_seed_count,
        "query_witness_rare_df": args.query_witness_rare_df,
        "query_witness_min_shared_terms": (
            args.query_witness_min_shared_terms),
        "dialogue_response_flood_threshold": (
            args.dialogue_response_flood_threshold),
        "max_evidence_turns": args.max_evidence_turns,
        "max_evidence_tokens": args.max_evidence_tokens,
        "benchmarks": {}}
    for benchmark in ("longmemeval", "locomo"):
        selected = [row for row in ordered if row["benchmark"] == benchmark
                    and row["has_turn_gold"]]
        summary["benchmarks"][benchmark] = {
            "annotated_questions": len(selected),
            "candidate_recall": _stats([
                float(row["metrics"]["candidate_turn_recall"])
                for row in selected]),
            "packed_recall": _stats([
                float(row["metrics"]["turn_recall"]) for row in selected]),
            "packed_all_hit": sum(bool(row["metrics"]["turn_all_hit"])
                                  for row in selected) / max(1, len(selected)),
            "candidate_average_precision": _stats([
                float(row["metrics"]["candidate_average_precision"])
                for row in selected]),
            "candidate_last_gold_rank": _stats([
                float(row["metrics"]["candidate_last_gold_rank"])
                for row in selected]),
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    store.close()


if __name__ == "__main__":
    main()
