#!/usr/bin/env python3
"""Measure accuracy-first evidence budgets without rewarding full retrieval."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/"
                        "hnsw_qwen_typed_dev200_graph_bounded_frontier/"
                        "report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--embedding-db", type=Path,
                        help="read turn vectors from a separate immutable SQLite sidecar")
    parser.add_argument(
        "--obligation-aware-relations", action="store_true",
        help="prioritize typed edges that match QueryIR proof obligations")
    parser.add_argument("--expansion-beam", type=int, default=2)
    parser.add_argument("--candidate-pool-limit", type=int, default=0)
    parser.add_argument("--graph-hop-decay", type=float, default=0.3)
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--turn-budgets", type=int, nargs="+",
                        default=(24, 32, 48, 64))
    parser.add_argument("--max-evidence-tokens", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_12/accuracy_budget_gate_dev200")
    return parser.parse_args()


def ratio(a: int | float, b: int | float) -> float:
    return float(a) / float(b) if b else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metrics(gold: set, predicted: set) -> dict[str, float]:
    hits = len(gold & predicted)
    precision = ratio(hits, len(predicted))
    recall = ratio(hits, len(gold))
    return {
        "all_hit": float(gold <= predicted),
        "recall": recall,
        "precision": precision,
        "f1": f1(precision, recall),
        "gold_hits": float(hits),
        "turns": float(len(predicted)),
    }


def paired_ci(rows: list[dict[str, Any]], left: str, right: str, field: str,
              resamples: int = 4000) -> list[float]:
    rng = random.Random(42)
    values = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        values.append(statistics.fmean(
            row[right][field] - row[left][field] for row in sample))
    values.sort()
    return [values[int(.025 * len(values))], values[int(.975 * len(values))]]


def main() -> None:
    args = parse_args()
    budgets = tuple(dict.fromkeys(sorted(args.turn_budgets)))
    if 32 not in budgets:
        raise ValueError("--turn-budgets must contain the 32-turn baseline")
    if any(value <= 0 for value in budgets):
        raise ValueError("turn budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    if args.limit:
        questions = questions[:args.limit]
    config = load_config(args.config)
    store = SQLiteGraphStore(args.db, read_only=True)
    embedding_store = (SQLiteGraphStore(args.embedding_db, read_only=True)
                       if args.embedding_db else None)
    embedding = (QwenEmbeddingIndex(
        embedding_store, config, record_usage=False) if embedding_store else None)
    navigator = GraphNavigator(
        store, dense_search=embedding.search if embedding else None,
        harness_profile=HarnessProfile.H11_UNIFIED_IR,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=args.graph_hop_decay,
        expansion_beam=args.expansion_beam,
        obligation_aware_relations=args.obligation_aware_relations,
        obligation_aware_packing=True, span_pack_window=96,
        candidate_pool_limit=args.candidate_pool_limit,
        native_seed_fusion=True, read_pool_size=4)
    rows: list[dict[str, Any]] = []
    turn_cache: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        turn_map = turn_cache.setdefault(
            question.memory_id,
            {turn.turn_id: turn for turn in store.turns(question.memory_id)})
        gold = {(item.session_id, item.turn_index) for item in question.gold_turns}
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "memory_id": question.memory_id,
            "stratum": question.stratum,
            "gold_refs": sorted(gold),
        }
        rotated = budgets[index % len(budgets):] + budgets[:index % len(budgets)]
        for turns in rotated:
            key = f"turn{turns}"
            budget = replace(
                config.query_budget, max_evidence_turns=turns,
                max_evidence_tokens=args.max_evidence_tokens)
            tick = time.perf_counter()
            result = navigator.navigate(question.memory_id, question.query, budget)
            latency = (time.perf_counter() - tick) * 1000
            packed = {
                (turn_map[item].session_id, turn_map[item].turn_index)
                for item in result.packed_turn_ids if item in turn_map}
            candidate = {
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map}
            graph_only = {
                (turn_map[item].session_id, turn_map[item].turn_index)
                for item in result.graph_only_candidate_turn_ids
                if item in turn_map}
            values = metrics(gold, packed)
            values.update({
                "evidence_tokens": float(result.evidence_tokens),
                "latency_ms": latency,
                "token_cap": float(bool(result.pack_exhaustion.get("token_cap_reached"))),
                "packed_refs": sorted(packed),
                "candidate_all_hit": float(gold <= candidate),
                "candidate_recall": ratio(len(gold & candidate), len(gold)),
                "candidate_turns": float(len(candidate)),
                "candidate_turns_before_limit": float(
                    result.trace.get("candidate_count_before_limit", len(candidate))),
                "graph_only_turns": float(len(graph_only)),
                "graph_only_gold_hits": float(len(gold & graph_only)),
                "visited_nodes": float(result.visited_nodes),
                "visited_edges": float(result.visited_edges),
                "relation_counts": dict(result.trace.get("relation_counts", {})),
            })
            row[key] = values
        rows.append(row)
        if index % 20 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    arms = tuple(f"turn{value}" for value in budgets)
    fields = ("all_hit", "recall", "precision", "f1", "gold_hits", "turns",
              "evidence_tokens", "latency_ms", "token_cap",
              "candidate_all_hit", "candidate_recall", "candidate_turns",
              "candidate_turns_before_limit",
              "graph_only_turns", "graph_only_gold_hits", "visited_nodes",
              "visited_edges")

    def aggregate(items: list[dict[str, Any]], arm: str) -> dict[str, float]:
        return {field: statistics.fmean(row[arm][field] for row in items)
                for field in fields}

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    baseline = "turn32"
    comparisons = {}
    for arm in arms:
        if arm == baseline:
            continue
        comparisons[f"{baseline}->{arm}"] = {
            field: {
                "mean": statistics.fmean(
                    row[arm][field] - row[baseline][field] for row in rows),
                "ci95": paired_ci(rows, baseline, arm, field),
            } for field in ("all_hit", "recall", "precision", "f1", "turns",
                            "evidence_tokens", "latency_ms")
        }
    payload = {
        "schema_version": "graphmem-v5.12-accuracy-budget-gate-v1",
        "precision_scope": "official_gold_turns_only_lower_bound",
        "questions": len(rows),
        "max_evidence_tokens": args.max_evidence_tokens,
        "embedding_db": str(args.embedding_db) if args.embedding_db else None,
        "obligation_aware_relations": args.obligation_aware_relations,
        "expansion_beam": args.expansion_beam,
        "candidate_pool_limit": args.candidate_pool_limit,
        "graph_hop_decay": args.graph_hop_decay,
        "overall": {arm: aggregate(rows, arm) for arm in arms},
        "relation_walks": {
            arm: dict(sum(
                (Counter(row[arm]["relation_counts"]) for row in rows),
                Counter()))
            for arm in arms
        },
        "per_stratum": {
            name: {arm: aggregate(items, arm) for arm in arms}
            for name, items in sorted(strata.items())},
        "comparisons": comparisons,
        "wall_time_sec": time.perf_counter() - started,
        "latency_note": "budget order rotates per question; OS cache is shared",
    }
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    store.close()
    if embedding_store is not None:
        embedding_store.close()


if __name__ == "__main__":
    main()
