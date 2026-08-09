#!/usr/bin/env python3
"""Measured scaling Gate for the production HNSW coarsen/relation path."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphmem.build.coarsen import (  # noqa: E402
    build_parent_gated_relations,
    build_recursive_hierarchy,
)
from measure_report_c1_scaling import gold_pairs, make_nodes  # noqa: E402
from measure_v5_9_path_retention import quality  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,2000,5000,10000,20000")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "artifacts/report/v5_10/hnsw_scaling")
    return parser.parse_args()


def run_size(n: int) -> dict:
    started = time.perf_counter()
    nodes = make_nodes(n)
    hierarchy = build_recursive_hierarchy(
        "scale", nodes, fanout=8, max_levels=12,
        summary_words=96, max_candidates=24,
        assignment_method="hnsw", hnsw_dimension=256,
        hnsw_m=16, hnsw_ef_construction=100)
    node_map = {row.node_id: row for row in (*nodes, *hierarchy.parent_cards)}
    plan = build_parent_gated_relations(
        "scale", hierarchy, node_map, hierarchy.children,
        embedding_k=8, max_candidates_per_node=24,
        low_threshold=0.30, high_threshold=0.45,
        refine_mode="ambiguous_only", candidate_method="hnsw",
        vectors=hierarchy.vectors, cross_session_quota=2,
        typed_restoration=False,
        max_refine_candidates_per_node=2,
        max_refine_candidates_per_1000_nodes=480)
    leaf_ids = {row.node_id for row in nodes}
    pairs = {
        tuple(sorted((left, right)))
        for left, right, _score, _level in plan.accepted_pairs
        if left in leaf_ids and right in leaf_ids}
    pairs.update(
        tuple(sorted((row.left_id, row.right_id)))
        for row in plan.refine_candidates
        if row.left_id in leaf_ids and row.right_id in leaf_ids)
    coarsen_work = hierarchy.stats.cluster_candidate_comparisons
    relation_work = plan.score_comparisons
    total_work = coarsen_work + relation_work
    all_pairs = n * (n - 1) // 2
    return {
        "n": n,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "hierarchy_levels": hierarchy.stats.levels,
        "parent_cards": len(hierarchy.parent_cards),
        "ann_queries": hierarchy.stats.ann_queries,
        "coarsen_comparisons": coarsen_work,
        "relation_comparisons": relation_work,
        "total_candidate_work": total_work,
        "all_pairs": all_pairs,
        "work_vs_all_pairs": total_work / all_pairs,
        "accepted_relations": len(plan.accepted_pairs),
        "refine_candidates": len(plan.refine_candidates),
        "refine_candidates_generated": plan.refine_candidates_generated,
        "refine_candidates_dropped": plan.refine_candidates_dropped,
        "relation_decision_token_upper_bound": len(plan.refine_candidates) * 448,
        **quality(pairs, gold_pairs(n)),
    }


def main() -> None:
    args = parse_args()
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    rows = []
    # Fresh worker per N makes ru_maxrss a per-size value instead of a monotone
    # process-lifetime maximum.
    for n in sizes:
        with ProcessPoolExecutor(max_workers=1) as pool:
            row = pool.submit(run_size, n).result()
        rows.append(row)
        print(json.dumps(row), flush=True)
    log_n = np.log(np.asarray([row["n"] for row in rows], dtype=float))
    log_work = np.log(np.asarray(
        [row["total_candidate_work"] for row in rows], dtype=float))
    log_time = np.log(np.asarray(
        [max(row["wall_seconds"], 1e-9) for row in rows], dtype=float))
    work_exponent = float(np.polyfit(log_n, log_work, 1)[0])
    time_exponent = float(np.polyfit(log_n, log_time, 1)[0])
    summary = {
        "schema_version": "graphmem-v5.10-hnsw-scaling-v1",
        "method": "real hnsw balanced coarsening + parent-gated hnsw relations",
        "rows": rows,
        "candidate_work_exponent": work_exponent,
        "wall_time_exponent": time_exponent,
        "gate_candidate_exponent_le_1_15": work_exponent <= 1.15,
        "all_timings_measured": True,
        "notes": (
            "Synthetic controlled topology; quality is structural path retention, "
            "not end-to-end benchmark QA or typed-relation precision."),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
