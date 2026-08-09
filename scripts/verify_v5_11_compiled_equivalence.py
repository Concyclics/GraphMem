#!/usr/bin/env python3
"""Verify that compiled sidecars change latency/caching, never retrieval output."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NavigationResult  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/v5_10/full_benchmark_20260809/graph/report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--compiled-cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--cache-memories", type=int, default=8)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_11/compiled_equivalence_dev200")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    rows = sorted(values)
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


SEMANTIC_FIELDS = tuple(
    row.name for row in fields(NavigationResult)
    if row.name not in {"stage_latency_ms", "trace"})


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))[:args.limit]
    budget = replace(
        load_config(args.config).query_budget,
        max_evidence_turns=32, max_evidence_tokens=5000)
    common = {
        "harness_profile": HarnessProfile.H11_UNIFIED_IR,
        "hierarchical_routing": True,
        "hierarchy_operator_aware": True,
        "obligation_aware_packing": True,
        "span_pack_window": 96,
        "native_seed_fusion": True,
        "graph_hop_decay": 0.3,
        "expansion_beam": 2,
        "read_pool_size": 1,
    }
    store = SQLiteGraphStore(args.db, read_only=True)
    try:
        baseline = GraphNavigator(store, **common)
        compiled = GraphNavigator(
            store, **common,
            snapshot_cache_memories=args.cache_memories,
            metadata_cache_memories=args.cache_memories,
            snapshot_cache_bytes=256 * 1024 * 1024,
            compiled_cache_dir=args.compiled_cache_dir)
        latencies = {"baseline": [], "compiled": []}
        mismatches = []
        all_hit = 0
        for index, question in enumerate(questions, 1):
            tick = time.perf_counter()
            expected = baseline.navigate(
                question.memory_id, question.query, budget)
            latencies["baseline"].append((time.perf_counter() - tick) * 1000)
            tick = time.perf_counter()
            actual = compiled.navigate(
                question.memory_id, question.query, budget)
            latencies["compiled"].append((time.perf_counter() - tick) * 1000)
            changed = [name for name in SEMANTIC_FIELDS
                       if getattr(expected, name) != getattr(actual, name)]
            if changed:
                mismatches.append({
                    "question_id": question.question_id,
                    "memory_id": question.memory_id,
                    "fields": changed,
                })
            gold = {(row.session_id, row.turn_index) for row in question.gold_turns}
            turn_map = {row.turn_id: (row.session_id, row.turn_index)
                        for row in store.turns(question.memory_id)}
            packed = {turn_map[row] for row in actual.packed_turn_ids
                      if row in turn_map}
            all_hit += int(gold <= packed)
            if index % 25 == 0:
                print(f"{index}/{len(questions)}", flush=True)
        summary = {
            "schema_version": "graphmem-v5.11-compiled-equivalence-v1",
            "questions": len(questions),
            "semantic_fields_compared": list(SEMANTIC_FIELDS),
            "mismatches": len(mismatches),
            "exact_equivalence": not mismatches,
            "compiled_all_hit": all_hit / max(1, len(questions)),
            "latency_ms": {
                name: {
                    "mean": statistics.fmean(rows),
                    "p95": percentile(rows, .95),
                } for name, rows in latencies.items()
            },
            "compiled_cache": compiled.cache_stats(),
            "mismatch_rows": mismatches,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if mismatches:
            raise SystemExit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
