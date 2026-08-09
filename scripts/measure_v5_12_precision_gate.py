#!/usr/bin/env python3
"""Measure bounded retrieval on a Recall--Precision--Token operating curve.

Unlike the older all-hit gates, this experiment reports the full candidate
reservoir, Top-K ranking, final evidence pack and their paired conversion.  The
``candidate128`` arm is intentionally bounded so returning the whole memory can
no longer receive a free candidate-all-hit win.
"""
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
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--arms", nargs="+",
        choices=("baseline", "precision_pack", "precision_soft",
                 "precision_soft_candidate128"),
        default=("baseline", "precision_pack", "precision_soft",
                 "precision_soft_candidate128"),
        help="run a subset; baseline is required for paired deltas")
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_12/precision_gate_dev200")
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def set_metrics(gold: set, predicted: set, prefix: str = "") -> dict[str, float]:
    hits = len(gold & predicted)
    precision = ratio(hits, len(predicted))
    recall = ratio(hits, len(gold))
    return {
        f"{prefix}all_hit": float(gold <= predicted),
        f"{prefix}recall": recall,
        f"{prefix}precision": precision,
        f"{prefix}f1": f1(precision, recall),
    }


def paired_ci(rows: list[dict[str, Any]], left: str, right: str, key: str,
              resamples: int = 4000) -> list[float]:
    rng = random.Random(42)
    values = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        values.append(statistics.fmean(
            row[right][key] - row[left][key] for row in sample))
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    if args.limit:
        questions = questions[:args.limit]
    config = load_config(args.config)
    budget = replace(
        config.query_budget, max_evidence_turns=32, max_evidence_tokens=5000)
    store = SQLiteGraphStore(args.db, read_only=True)
    common = dict(
        harness_profile=HarnessProfile.H11_UNIFIED_IR,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        obligation_aware_packing=True, span_pack_window=96,
        native_seed_fusion=True, read_pool_size=4)
    all_navigators = {
        "baseline": GraphNavigator(store, **common),
        "precision_pack": GraphNavigator(
            store, precision_aware_packing=True, **common),
        "precision_soft": GraphNavigator(
            store, precision_aware_packing=True,
            queryir_soft_fallback=True, **common),
        "precision_soft_candidate128": GraphNavigator(
            store, precision_aware_packing=True,
            queryir_soft_fallback=True, candidate_pool_limit=128, **common),
    }
    if "baseline" not in args.arms:
        raise ValueError("--arms must include baseline")
    navigators = {name: all_navigators[name] for name in args.arms}
    arms = tuple(navigators)
    turn_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
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
            "query": question.query,
            "gold_refs": sorted(gold),
        }
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        for arm in order:
            tick = time.perf_counter()
            result = navigators[arm].navigate(
                question.memory_id, question.query, budget)
            latency = (time.perf_counter() - tick) * 1000
            packed = {
                (turn_map[item].session_id, turn_map[item].turn_index)
                for item in result.packed_turn_ids if item in turn_map}
            ranked = tuple(dict.fromkeys(
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map))
            candidates = set(ranked)
            metrics = {
                **set_metrics(gold, packed),
                **set_metrics(gold, candidates, "candidate_"),
            }
            for cutoff in (8, 16, 32, 64, 128):
                metrics.update(set_metrics(
                    gold, set(ranked[:cutoff]), f"top{cutoff}_"))
            metrics.update({
                "candidate_count": len(candidates),
                "candidate_selectivity": ratio(len(candidates), len(turn_map)),
                "pack_turns": len(packed),
                "pack_selectivity": ratio(len(packed), len(candidates)),
                "candidate_to_pack_recall_loss": (
                    metrics["candidate_recall"] - metrics["recall"]),
                "candidate_to_pack_precision_gain": (
                    metrics["precision"] - metrics["candidate_precision"]),
                "evidence_tokens": result.evidence_tokens,
                "latency_ms": latency,
                "soft_fallback": float(bool(
                    result.trace.get("query_ir_soft_fallback"))),
                "compile_confidence": float(
                    result.trace.get("query_ir_confidence", 1.0)),
                "adaptive_pack_turn_limit": int(
                    result.trace.get("adaptive_pack_turn_limit", 32)),
                "packed_refs": sorted(packed),
                "candidate_refs": list(ranked),
            })
            row[arm] = metrics
        rows.append(row)
        if index % 20 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    scalar_fields = (
        "all_hit", "recall", "precision", "f1",
        "candidate_all_hit", "candidate_recall", "candidate_precision",
        "candidate_f1", "candidate_count", "candidate_selectivity",
        "top8_all_hit", "top8_recall", "top8_precision", "top8_f1",
        "top16_all_hit", "top16_recall", "top16_precision", "top16_f1",
        "top32_all_hit", "top32_recall", "top32_precision", "top32_f1",
        "top64_all_hit", "top64_recall", "top64_precision", "top64_f1",
        "top128_all_hit", "top128_recall", "top128_precision", "top128_f1",
        "pack_turns", "pack_selectivity", "candidate_to_pack_recall_loss",
        "candidate_to_pack_precision_gain", "evidence_tokens", "latency_ms",
        "soft_fallback", "compile_confidence", "adaptive_pack_turn_limit",
    )

    def aggregate(items: list[dict[str, Any]], arm: str) -> dict[str, float]:
        return {field: statistics.fmean(row[arm][field] for row in items)
                for field in scalar_fields}

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    comparisons = {}
    for right in arms[1:]:
        key = f"baseline->{right}"
        comparisons[key] = {
            field: {
                "mean": statistics.fmean(
                    row[right][field] - row["baseline"][field] for row in rows),
                "ci95": paired_ci(rows, "baseline", right, field),
            }
            for field in (
                "all_hit", "recall", "precision", "f1",
                "candidate_recall", "candidate_precision",
                "candidate_selectivity", "top32_recall", "top32_precision",
                "pack_turns", "evidence_tokens", "latency_ms")
        }
        comparisons[key]["all_hit_transitions"] = dict(Counter(
            f"{int(row['baseline']['all_hit'])}->{int(row[right]['all_hit'])}"
            for row in rows))
    summary = {
        "schema_version": "graphmem-v5.12-precision-gate-v1",
        "precision_scope": "official_gold_turns_only",
        "db": str(args.db),
        "questions": len(rows),
        "budget": {"turns": budget.max_evidence_turns,
                   "tokens": budget.max_evidence_tokens},
        "overall": {arm: aggregate(rows, arm) for arm in arms},
        "per_stratum": {
            stratum: {arm: aggregate(items, arm) for arm in arms}
            for stratum, items in sorted(strata.items())},
        "comparisons": comparisons,
        "wall_time_sec": time.perf_counter() - started,
        "latency_note": "arm order rotates per question; OS cache is shared",
    }
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
