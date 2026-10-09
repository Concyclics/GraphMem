#!/usr/bin/env python3
"""Measure whether a deeper dense lane can rescue packed LoCoMo evidence.

This script is evaluator-only: gold annotations choose no production evidence.
It audits full-memory dense ranks and reports the upper bound of reserving a
small, query-only dense witness lane beside the validated graph/lexical pack.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval.fullset import load_full_questions
from graphmem.storage import SQLiteGraphStore


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def best_of_failures(candidates: Path, judge_root: Path) -> set[str]:
    verdicts: dict[tuple[str, str], bool] = {}
    for path in sorted(judge_root.glob("candidate_*/auto_eval.jsonl")):
        for row in read_jsonl(path):
            key = (str(row["question_id"]), str(row["prediction_sha256"]))
            verdicts[key] = verdicts.get(key, False) or bool(row["correct"])
    failures: set[str] = set()
    for row in read_jsonl(candidates):
        question_id = str(row["question_id"])
        if not any(verdicts.get((question_id, str(candidate.get(
                "prediction_sha256") or "")), False)
                   for candidate in row.get("candidates", ())):
            failures.add(question_id)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dense-sidecar-dir", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--request-model-id", default="bge-m3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    questions = load_full_questions(
        None, args.data, None, expect_lme=0, expect_locomo=1540)
    retrieval = {
        str(row["dev_question_id"]): row for row in read_jsonl(args.retrieval)}
    fixed_failures = best_of_failures(args.candidates, args.judge_root)
    store = SQLiteGraphStore(args.db, read_only=True)
    config = load_config(args.config)
    embedding = QwenEmbeddingIndex(
        store, config, record_usage=False,
        query_cache_path=args.query_cache,
        dense_sidecar_dir=args.dense_sidecar_dir,
        dense_backend="faiss_flat", model_id=args.model_id,
        request_model_id=args.request_model_id, base_url=args.base_url)

    cutoffs = (8, 16, 32, 64, 96, 128, 192, 256)
    reserve_sizes = (1, 2, 4, 8, 16)
    counters: Counter[str] = Counter()
    rank_rows: list[int] = []
    last_rank_rows: list[int] = []
    for item in questions:
        question = item.question
        turns = tuple(store.turns(question.memory_id))
        by_coordinate = {
            (turn.session_id, turn.turn_index): turn.turn_id for turn in turns}
        gold = {
            by_coordinate[(ref.session_id, ref.turn_index)]
            for ref in question.gold_turns
            if (ref.session_id, ref.turn_index) in by_coordinate}
        if not gold:
            continue
        dense = tuple(embedding.search(
            question.memory_id, question.query, len(turns)))
        rank = {turn_id: index for index, (turn_id, _score) in enumerate(
            dense, start=1)}
        ranks = sorted(rank[turn_id] for turn_id in gold if turn_id in rank)
        if ranks:
            rank_rows.append(ranks[0])
            last_rank_rows.append(ranks[-1])
        packed = set(retrieval.get(question.question_id, {}).get(
            "retrieved_turn_ids", ()))
        scope = ("fixed_failure" if question.question_id in fixed_failures
                 else "other")
        counters[f"{scope}:questions"] += 1
        counters[f"{scope}:pack_all"] += gold <= packed
        counters[f"{scope}:pack_any"] += bool(gold & packed)
        for cutoff in cutoffs:
            top = {turn_id for turn_id, _score in dense[:cutoff]}
            counters[f"{scope}:dense_all@{cutoff}"] += gold <= top
            counters[f"{scope}:dense_any@{cutoff}"] += bool(gold & top)
        for reserve in reserve_sizes:
            selected = set(packed)
            selected.update(turn_id for turn_id, _score in dense
                            if turn_id not in selected and len(selected - packed) < reserve)
            counters[f"{scope}:union_all@{reserve}"] += gold <= selected
            counters[f"{scope}:union_any@{reserve}"] += bool(gold & selected)

    def percentile(values: list[int], p: float) -> int:
        ordered = sorted(values)
        return ordered[max(0, int(__import__("math").ceil(
            p * len(ordered))) - 1)] if ordered else 0

    result: dict[str, Any] = {
        "schema_version": "graphmem-v5.78-deep-dense-audit-v1",
        "fixed_best_of_failures": len(fixed_failures),
        "cutoffs": list(cutoffs),
        "reserve_sizes": list(reserve_sizes),
        "first_gold_dense_rank": {
            "mean": sum(rank_rows) / max(1, len(rank_rows)),
            "p50": percentile(rank_rows, 0.50),
            "p90": percentile(rank_rows, 0.90),
            "p95": percentile(rank_rows, 0.95),
            "max": max(rank_rows, default=0),
        },
        "last_gold_dense_rank": {
            "mean": sum(last_rank_rows) / max(1, len(last_rank_rows)),
            "p50": percentile(last_rank_rows, 0.50),
            "p90": percentile(last_rank_rows, 0.90),
            "p95": percentile(last_rank_rows, 0.95),
            "max": max(last_rank_rows, default=0),
        },
        "counts": dict(sorted(counters.items())),
        "embedding_stats": dict(embedding.stats),
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    store.close()


if __name__ == "__main__":
    main()
