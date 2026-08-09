#!/usr/bin/env python3
"""Gate C: frozen split QueryIR vs one promoted AST and directed relations."""
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
from graphmem.domain import RelationType  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


TYPED = frozenset({
    RelationType.COREFERENCE, RelationType.SAME_ENTITY_STATE,
    RelationType.TEMPORAL_CONTINUATION, RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
})


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
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/queryir_gate_dev200")
    return parser.parse_args()


def ci(rows: list[dict[str, Any]], left: str, right: str, key: str,
       samples: int = 4000) -> list[float]:
    rng = random.Random(42)
    values = []
    for _ in range(samples):
        chosen = [rows[rng.randrange(len(rows))] for _row in rows]
        values.append(statistics.fmean(
            row[right][key] - row[left][key] for row in chosen))
    values.sort()
    return [values[int(0.025 * samples)], values[int(0.975 * samples)]]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    if args.limit:
        questions = questions[:args.limit]
    config = load_config(args.config)
    budget = replace(
        config.query_budget, max_evidence_turns=32,
        max_evidence_tokens=5000)
    store = SQLiteGraphStore(args.db, read_only=True)
    common = dict(
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        obligation_aware_packing=True, span_pack_window=96,
        read_pool_size=4)
    navigators = {
        "split_h10": GraphNavigator(
            store, harness_profile=HarnessProfile.H10_AST, **common),
        "unified_h11": GraphNavigator(
            store, harness_profile=HarnessProfile.H11_UNIFIED_IR, **common),
        "unified_directed": GraphNavigator(
            store, harness_profile=HarnessProfile.H11_UNIFIED_IR,
            obligation_aware_relations=True, **common),
    }
    arms = tuple(navigators)
    typed_names = {str(item) for item in TYPED}
    turn_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        turn_map = turn_cache.setdefault(
            question.memory_id,
            {turn.turn_id: turn for turn in store.turns(question.memory_id)})
        gold_refs = {(item.session_id, item.turn_index)
                     for item in question.gold_turns}
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "memory_id": question.memory_id,
            "stratum": question.stratum,
            "query": question.query,
            "gold_refs": sorted(gold_refs),
        }
        # Rotate arm order by question to reduce systematic warm-cache bias.
        arm_order = arms[index % len(arms):] + arms[:index % len(arms)]
        for arm in arm_order:
            tick = time.perf_counter()
            result = navigators[arm].navigate(
                question.memory_id, question.query, budget)
            latency = (time.perf_counter() - tick) * 1000
            packed_refs = {
                (turn_map[turn_id].session_id, turn_map[turn_id].turn_index)
                for turn_id in result.packed_turn_ids if turn_id in turn_map}
            candidate_refs = {
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map}
            ranked_candidate_refs = tuple(dict.fromkeys(
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map))
            packed_hits = len(gold_refs & packed_refs)
            candidate_hits = len(gold_refs & candidate_refs)
            precision = packed_hits / len(packed_refs) if packed_refs else 0.0
            recall = packed_hits / len(gold_refs) if gold_refs else 1.0
            candidate_precision = (
                candidate_hits / len(candidate_refs) if candidate_refs else 0.0)
            candidate_recall = (
                candidate_hits / len(gold_refs) if gold_refs else 1.0)
            top32 = set(ranked_candidate_refs[:32])
            top32_hits = len(gold_refs & top32)
            top32_precision = top32_hits / len(top32) if top32 else 0.0
            top32_recall = top32_hits / len(gold_refs) if gold_refs else 1.0
            trace = result.trace
            row[arm] = {
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
                "tokens": result.evidence_tokens,
                "latency_ms": latency,
                "visited_nodes": result.visited_nodes,
                "visited_edges": result.visited_edges,
                "typed_edges_walked": sum(
                    count for relation, count in
                    trace.get("relation_counts", {}).items()
                    if relation in typed_names),
                "false_complete": float(bool(
                    result.certificate and result.certificate.false_complete)),
                "ast_diverges": float(bool(trace.get("ast_diverges"))),
                "packed_refs": sorted(packed_refs),
                "query_ir_mode": trace.get("query_ir_mode", ""),
            }
        rows.append(row)
        if index % 25 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    fields = (
        "all_hit", "recall", "precision", "f1", "candidate_all_hit",
        "candidate_recall", "candidate_precision", "candidate_f1",
        "candidate_count", "candidate_selectivity", "top32_all_hit",
        "top32_recall", "top32_precision", "top32_f1",
        "candidate_to_pack_recall_loss", "candidate_to_pack_precision_gain",
        "turns", "tokens",
        "latency_ms", "visited_nodes", "visited_edges", "typed_edges_walked",
        "false_complete", "ast_diverges")

    def aggregate(items, arm):
        return {field: statistics.fmean(row[arm][field] for row in items)
                for field in fields}

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    comparisons = {}
    for left, right in (("split_h10", "unified_h11"),
                        ("unified_h11", "unified_directed"),
                        ("split_h10", "unified_directed")):
        comparisons[f"{left}->{right}"] = {
            key: {
                "mean": statistics.fmean(
                    row[right][key] - row[left][key] for row in rows),
                "ci95": ci(rows, left, right, key),
            } for key in ("all_hit", "recall", "precision", "f1",
                          "candidate_recall", "candidate_precision",
                          "candidate_selectivity", "top32_recall",
                          "top32_precision", "tokens", "latency_ms",
                          "typed_edges_walked", "false_complete")
        }
        comparisons[f"{left}->{right}"]["transitions"] = dict(Counter(
            f"{int(row[left]['all_hit'])}->{int(row[right]['all_hit'])}"
            for row in rows))
    summary = {
        "schema_version": "graphmem-v5.10-queryir-gate-v1",
        "db": str(args.db), "questions": len(rows),
        "budget": {"turns": budget.max_evidence_turns,
                   "tokens": budget.max_evidence_tokens},
        "overall": {arm: aggregate(rows, arm) for arm in arms},
        "per_stratum": {
            key: {arm: aggregate(items, arm) for arm in arms}
            for key, items in sorted(strata.items())},
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
