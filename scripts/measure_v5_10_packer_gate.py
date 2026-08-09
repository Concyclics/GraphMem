#!/usr/bin/env python3
"""Gate C: compare frozen rank packing with V5.10 obligation/span packing."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/v5_9/full_benchmark_20260809/graph/report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    hard = WORKSPACE / "artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804"
    parser.add_argument("--lme", type=Path,
                        default=hard / "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path,
                        default=hard / "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path,
                        default=ROOT / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--max-evidence-turns", type=int, default=16)
    parser.add_argument("--max-evidence-tokens", type=int, default=5000)
    parser.add_argument("--span-window", type=int, default=96)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-id", default="",
                        help="Run one exact question id for transition auditing.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/report/v5_10/packer_gate")
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1,
                              int(probability * len(ordered)) - 1))]


def paired_ci(rows: list[dict[str, Any]], key: str, resamples: int = 4000) -> list[float]:
    if not rows:
        return [0.0, 0.0]
    rng = random.Random(42)
    deltas = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        deltas.append(sum(row["obligation"][key] - row["baseline"][key]
                          for row in sample) / len(sample))
    deltas.sort()
    return [deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    gold = load_gold_turns(args.gold)
    questions = load_dev_questions(args.lme, args.locomo, gold)
    if args.question_id:
        questions = [row for row in questions if row.question_id == args.question_id]
        if not questions:
            raise ValueError(f"Unknown question id: {args.question_id}")
    if args.limit:
        questions = questions[:args.limit]
    budget = replace(
        config.query_budget,
        max_evidence_turns=args.max_evidence_turns,
        max_evidence_tokens=args.max_evidence_tokens,
    )
    store = SQLiteGraphStore(args.db, read_only=not args.embedding)
    embedding = QwenEmbeddingIndex(store, config, record_usage=False) if args.embedding else None
    common = dict(
        dense_search=embedding.search if embedding else None,
        harness_profile=HarnessProfile.H10_AST,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        read_pool_size=4,
    )
    navigators = {
        "baseline": GraphNavigator(store, **common),
        "obligation": GraphNavigator(
            store, **common, obligation_aware_packing=True,
            span_pack_window=args.span_window),
    }
    current_memory = None
    turn_map = {}
    rows = []
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        if question.memory_id != current_memory:
            current_memory = question.memory_id
            turn_map = {turn.turn_id: turn for turn in store.turns(current_memory)}
        gold_refs = {(row.session_id, row.turn_index) for row in question.gold_turns}
        result_row: dict[str, Any] = {
            "question_id": question.question_id,
            "stratum": question.stratum,
            "query": question.query,
            "gold_turns": len(gold_refs),
        }
        for name, navigator in navigators.items():
            tick = time.perf_counter()
            result = navigator.navigate(question.memory_id, question.query, budget)
            latency = (time.perf_counter() - tick) * 1000
            packed_refs = {
                (turn_map[turn_id].session_id, turn_map[turn_id].turn_index)
                for turn_id in result.packed_turn_ids if turn_id in turn_map
            }
            candidate_refs = {
                (turn_map[item.turn_id].session_id, turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map
            }
            ranked_candidate_refs = tuple(dict.fromkeys(
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map))
            candidate_ranks = {
                (turn_map[item.turn_id].session_id, turn_map[item.turn_id].turn_index): rank
                for rank, item in enumerate(result.candidate_scores, 1)
                if item.turn_id in turn_map
            }
            recall = len(gold_refs & packed_refs) / len(gold_refs) if gold_refs else 1.0
            packed_hits = len(gold_refs & packed_refs)
            candidate_hits = len(gold_refs & candidate_refs)
            precision = packed_hits / len(packed_refs) if packed_refs else 0.0
            candidate_precision = (
                candidate_hits / len(candidate_refs) if candidate_refs else 0.0)
            candidate_recall = (
                candidate_hits / len(gold_refs) if gold_refs else 1.0)
            top32 = set(ranked_candidate_refs[:32])
            top32_hits = len(gold_refs & top32)
            top32_precision = top32_hits / len(top32) if top32 else 0.0
            top32_recall = top32_hits / len(gold_refs) if gold_refs else 1.0
            result_row[name] = {
                "all_hit": float(gold_refs <= packed_refs),
                "recall": recall,
                "precision": precision,
                "f1": (2 * precision * recall / (precision + recall)
                       if precision + recall else 0.0),
                "candidate_all_hit": float(gold_refs <= candidate_refs),
                "candidate_recall": candidate_recall,
                "candidate_precision": candidate_precision,
                "candidate_f1": (
                    2 * candidate_precision * candidate_recall
                    / (candidate_precision + candidate_recall)
                    if candidate_precision + candidate_recall else 0.0),
                "candidate_count": len(candidate_refs),
                "candidate_selectivity": (
                    len(candidate_refs) / len(turn_map) if turn_map else 0.0),
                "top32_all_hit": float(gold_refs <= top32),
                "top32_recall": top32_recall,
                "top32_precision": top32_precision,
                "top32_f1": (2 * top32_precision * top32_recall
                             / (top32_precision + top32_recall)
                             if top32_precision + top32_recall else 0.0),
                "candidate_to_pack_recall_loss": candidate_recall - recall,
                "candidate_to_pack_precision_gain": precision - candidate_precision,
                "turns": len(result.packed_turn_ids),
                "evidence_tokens": result.evidence_tokens,
                "latency_ms": latency,
                "false_complete": float(bool(
                    result.certificate and result.certificate.false_complete)),
                "post_pack_complete": float(bool(
                    result.certificate and result.certificate.post_pack_complete)),
                "obligation_incomplete": float(bool(
                    result.pack_exhaustion.get("obligation_incomplete", False))),
                "operand_incomplete": float(bool(
                    result.pack_exhaustion.get("operand_incomplete", False))),
                "gold_candidate_ranks": sorted(
                    candidate_ranks.get(ref, -1) for ref in gold_refs),
                "missed_gold_refs": sorted(
                    f"{session_id}:{turn_index}" for session_id, turn_index
                    in gold_refs - packed_refs),
                "packed_turn_ids": list(result.packed_turn_ids),
            }
        rows.append(result_row)
        if index % 25 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    def aggregate(items: list[dict[str, Any]], arm: str) -> dict[str, Any]:
        metrics = [row[arm] for row in items]
        latencies = [row["latency_ms"] for row in metrics]
        tokens = [row["evidence_tokens"] for row in metrics]
        return {
            "n": len(metrics),
            "all_hit": statistics.fmean(row["all_hit"] for row in metrics),
            "recall": statistics.fmean(row["recall"] for row in metrics),
            "precision": statistics.fmean(row["precision"] for row in metrics),
            "f1": statistics.fmean(row["f1"] for row in metrics),
            "candidate_all_hit": statistics.fmean(row["candidate_all_hit"] for row in metrics),
            "candidate_recall": statistics.fmean(
                row["candidate_recall"] for row in metrics),
            "candidate_precision": statistics.fmean(
                row["candidate_precision"] for row in metrics),
            "candidate_f1": statistics.fmean(
                row["candidate_f1"] for row in metrics),
            "candidate_count": statistics.fmean(
                row["candidate_count"] for row in metrics),
            "candidate_selectivity": statistics.fmean(
                row["candidate_selectivity"] for row in metrics),
            "top32_all_hit": statistics.fmean(
                row["top32_all_hit"] for row in metrics),
            "top32_recall": statistics.fmean(
                row["top32_recall"] for row in metrics),
            "top32_precision": statistics.fmean(
                row["top32_precision"] for row in metrics),
            "top32_f1": statistics.fmean(
                row["top32_f1"] for row in metrics),
            "candidate_to_pack_recall_loss": statistics.fmean(
                row["candidate_to_pack_recall_loss"] for row in metrics),
            "candidate_to_pack_precision_gain": statistics.fmean(
                row["candidate_to_pack_precision_gain"] for row in metrics),
            "mean_turns": statistics.fmean(row["turns"] for row in metrics),
            "mean_evidence_tokens": statistics.fmean(tokens),
            "p95_evidence_tokens": percentile(tokens, 0.95),
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": percentile(latencies, 0.95),
            "false_complete": statistics.fmean(row["false_complete"] for row in metrics),
            "post_pack_complete": statistics.fmean(
                row["post_pack_complete"] for row in metrics),
            "obligation_incomplete": statistics.fmean(
                row["obligation_incomplete"] for row in metrics),
        }

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    summary = {
        "config": str(args.config), "db": str(args.db),
        "budget": {
            "max_evidence_turns": budget.max_evidence_turns,
            "max_evidence_tokens": budget.max_evidence_tokens,
            "span_window": args.span_window,
        },
        "overall": {arm: aggregate(rows, arm) for arm in navigators},
        "per_stratum": {
            stratum: {arm: aggregate(items, arm) for arm in navigators}
            for stratum, items in sorted(strata.items())
        },
        "paired_delta": {
            key: {
                "mean": statistics.fmean(
                    row["obligation"][key] - row["baseline"][key] for row in rows),
                "ci95": paired_ci(rows, key),
            } for key in ("all_hit", "recall", "precision", "f1",
                          "candidate_precision", "candidate_selectivity",
                          "top32_recall", "top32_precision",
                          "evidence_tokens", "latency_ms")
        },
        "transitions": dict(Counter(
            f"{int(row['baseline']['all_hit'])}->{int(row['obligation']['all_hit'])}"
            for row in rows)),
        "wall_time_sec": time.perf_counter() - started,
    }
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
