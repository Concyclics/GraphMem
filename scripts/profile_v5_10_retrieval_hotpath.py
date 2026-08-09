#!/usr/bin/env python3
"""Profile the warm V5.10 retrieval hot path on selected LoCoMo users."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import pstats
import statistics
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    rows = sorted(values)
    return rows[min(len(rows) - 1, max(0, math.ceil(len(rows) * q) - 1))]


def main() -> None:
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
    parser.add_argument("--memory-ids", default=(
        "locomo:conv-26,locomo:conv-30,locomo:conv-41,locomo:conv-42"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    selected = {value.strip() for value in args.memory_ids.split(",")
                if value.strip()}
    questions = [row for row in load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))
        if row.memory_id in selected]
    budget = replace(load_config(args.config).query_budget,
                     max_evidence_turns=32, max_evidence_tokens=5000)
    options = {
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
        navigator = GraphNavigator(store, **options)
        by_memory = {}
        for row in questions:
            by_memory.setdefault(row.memory_id, row)
        for row in by_memory.values():
            navigator.navigate(row.memory_id, row.query, budget)

        results = []
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.repeats):
            for row in questions:
                results.append(navigator.navigate(
                    row.memory_id, row.query, budget))
        profiler.disable()
        stages = sorted({key for row in results for key in row.stage_latency_ms})
        report = {stage: {
            "mean_ms": statistics.fmean(
                float(row.stage_latency_ms.get(stage, 0.0)) for row in results),
            "p95_ms": percentile([
                float(row.stage_latency_ms.get(stage, 0.0)) for row in results
            ], .95),
        } for stage in stages}
        print(json.dumps({
            "queries": len(results), "unique_queries": len(questions),
            "stage_latency": report,
        }, indent=2))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative").print_stats(args.top)
        print(stream.getvalue())
    finally:
        store.close()


if __name__ == "__main__":
    main()
