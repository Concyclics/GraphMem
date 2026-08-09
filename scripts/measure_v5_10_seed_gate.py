#!/usr/bin/env python3
"""Gate D: SQLite FTS-per-view versus immutable memory-local seed fusion."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path


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
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/seed_gate_dev200")
    return parser.parse_args()


def percentile(values, q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    return rows[min(len(rows) - 1, max(0, math.ceil(q * len(rows)) - 1))]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    if args.limit:
        questions = questions[:args.limit]
    budget = replace(
        load_config(args.config).query_budget,
        max_evidence_turns=32, max_evidence_tokens=5000)
    store = SQLiteGraphStore(args.db, read_only=True)
    common = dict(
        harness_profile=HarnessProfile.H11_UNIFIED_IR,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        obligation_aware_packing=True, span_pack_window=96,
        read_pool_size=4)
    navigators = {
        "sqlite_fts": GraphNavigator(store, native_seed_fusion=False, **common),
        "native_index": GraphNavigator(store, native_seed_fusion=True, **common),
    }
    arms = tuple(navigators)
    turn_cache = {}
    rows = []
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        turn_map = turn_cache.setdefault(
            question.memory_id,
            {turn.turn_id: turn for turn in store.turns(question.memory_id)})
        gold = {(item.session_id, item.turn_index)
                for item in question.gold_turns}
        row = {"question_id": question.question_id,
               "stratum": question.stratum, "query": question.query}
        order = arms[index % 2:] + arms[:index % 2]
        for arm in order:
            tick = time.perf_counter()
            result = navigators[arm].navigate(
                question.memory_id, question.query, budget)
            measured_total = (time.perf_counter() - tick) * 1000
            packed = {
                (turn_map[turn_id].session_id, turn_map[turn_id].turn_index)
                for turn_id in result.packed_turn_ids if turn_id in turn_map}
            row[arm] = {
                "all_hit": float(gold <= packed),
                "recall": len(gold & packed) / len(gold) if gold else 1.0,
                "tokens": result.evidence_tokens,
                "latency_ms": measured_total,
                "seed_fusion_ms": float(
                    result.stage_latency_ms.get("seed_fusion", 0.0)),
                "hierarchy_ms": float(
                    result.stage_latency_ms.get("hierarchical_route", 0.0)),
                "graph_ms": float(
                    result.stage_latency_ms.get("graph_read_view", 0.0)),
                "pack_ms": float(
                    result.stage_latency_ms.get("evidence_pack", 0.0)),
                "candidate_count": len(result.candidate_scores),
                "backend": result.trace.get("seeding", {}).get(
                    "bm25_backend", ""),
            }
        rows.append(row)
        if index % 25 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    def aggregate(items, arm):
        result = {}
        for field in ("all_hit", "recall", "tokens", "latency_ms",
                      "seed_fusion_ms", "hierarchy_ms", "graph_ms",
                      "pack_ms", "candidate_count"):
            values = [row[arm][field] for row in items]
            result[field] = statistics.fmean(values)
            if field.endswith("_ms"):
                result[field.replace("_ms", "_p50_ms")] = percentile(values, .50)
                result[field.replace("_ms", "_p95_ms")] = percentile(values, .95)
                result[field.replace("_ms", "_p99_ms")] = percentile(values, .99)
        return result

    strata = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    transitions = dict(Counter(
        f"{int(row['sqlite_fts']['all_hit'])}->{int(row['native_index']['all_hit'])}"
        for row in rows))
    summary = {
        "schema_version": "graphmem-v5.10-seed-gate-v1",
        "db": str(args.db), "questions": len(rows),
        "overall": {arm: aggregate(rows, arm) for arm in arms},
        "per_stratum": {
            key: {arm: aggregate(items, arm) for arm in arms}
            for key, items in sorted(strata.items())},
        "delta_native_minus_sqlite": {
            field: statistics.fmean(
                row["native_index"][field] - row["sqlite_fts"][field]
                for row in rows)
            for field in ("all_hit", "recall", "tokens", "latency_ms",
                          "seed_fusion_ms")},
        "transitions": transitions,
        "wall_time_sec": time.perf_counter() - started,
        "latency_note": "arm order rotates per question; cold and warm memories are mixed",
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
