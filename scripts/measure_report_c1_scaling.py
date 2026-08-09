#!/usr/bin/env python3
"""Reproducible C1 scaling microbenchmark for report figures.

All-pairs counts are exact.  Its wall time above ``--all-pairs-limit`` is an
explicit projection from the largest materialised loop; every row carries a
``timing_kind`` field so projected and measured results cannot be confused.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hnswlib  # noqa: E402
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graphmem.build.coarsen import (  # noqa: E402
    _bounded_sparse_pairs,
    build_parent_gated_relations,
    build_recursive_hierarchy,
)
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import GraphNode, NodeType  # noqa: E402


COLORS = {
    "all_pairs": "#D84A4A",
    "ann_only": "#F2A93B",
    "flat_sparse": "#2378D7",
    "cir": "#18A999",
}
LABELS = {
    "all_pairs": "All-pairs（精确计数）",
    "ann_only": "仅 ANN",
    "flat_sparse": "全局稀疏、无父门控",
    "cir": "完整 CIR",
}
RELATION_DECISION_TOKENS = 96 * 2 + 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/report/v5_9/c1"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v5/v5_9_report.json"))
    parser.add_argument("--sizes", default="1000,2000,5000,10000,20000")
    parser.add_argument("--all-pairs-limit", type=int, default=2000)
    return parser.parse_args()


def make_nodes(n: int) -> tuple[GraphNode, ...]:
    # Keep about 32 nodes in each semantic topic.  Consecutive nodes in a topic
    # form the controlled gold relation/path set below.
    topics = max(8, math.ceil(n / 32))
    rows = []
    for index in range(n):
        topic = index % topics
        group = (index // topics) // 4
        rows.append(GraphNode(
            f"leaf:{index:06d}", "scale", NodeType.ROUTING_CARD, 1,
            f"memory topic_{topic} entity_{topic} group_{group} detail_{index}",
            f"evidence:{index:06d}",
            attributes={"session_id": f"session:{index:06d}",
                        "roles": ("route",), "provenance_scope": "route"},
        ))
    return tuple(rows)


def gold_pairs(n: int) -> set[tuple[str, str]]:
    topics = max(8, math.ceil(n / 32))
    by_local_group: dict[tuple[int, int], list[str]] = {}
    for index in range(n):
        topic = index % topics
        group = (index // topics) // 4
        by_local_group.setdefault((topic, group), []).append(f"leaf:{index:06d}")
    return {
        tuple(sorted((left, right)))
        for rows in by_local_group.values()
        for left_index, left in enumerate(rows)
        for right in rows[left_index + 1:]
    }


def quality(candidate_pairs: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict:
    hits = len(candidate_pairs & gold)
    # Each controlled four-node semantic group is a small multi-hop component;
    # edge retention is the conservative lower bound on its path retention.
    return {
        "gold_edges": len(gold),
        "gold_edge_recall": hits / max(1, len(gold)),
        "multi_hop_path_retention": hits / max(1, len(gold)),
    }


def ann_pairs(n: int, k: int) -> set[tuple[str, str]]:
    topics = max(8, math.ceil(n / 32))
    rng = np.random.default_rng(42)
    bases = rng.normal(size=(topics, 24)).astype(np.float32)
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    vectors = np.empty((n, 32), dtype=np.float32)
    for index in range(n):
        topic = index % topics
        position = index // topics
        vector = np.zeros(32, dtype=np.float32)
        vector[:24] = bases[topic]
        angle = position / 32 * math.pi * 2
        vector[24:28] = (math.sin(angle), math.cos(angle),
                         math.sin(angle / 2), math.cos(angle / 2))
        vector[28:] = rng.normal(scale=0.01, size=4)
        vectors[index] = vector / np.linalg.norm(vector)
    index = hnswlib.Index(space="cosine", dim=32)
    index.init_index(max_elements=n, ef_construction=100, M=16,
                     random_seed=42)
    index.add_items(vectors, np.arange(n), num_threads=1)
    index.set_ef(max(32, k * 2))
    labels, _distances = index.knn_query(vectors, k=min(n, k + 1), num_threads=1)
    pairs = set()
    for source, neighbours in enumerate(labels):
        for target in neighbours:
            target = int(target)
            if source != target:
                pairs.add(tuple(sorted((f"leaf:{source:06d}",
                                        f"leaf:{target:06d}"))))
    return pairs


def run_row(spec: tuple[str, int, int]) -> dict:
    method, n, all_pairs_limit = spec
    tracemalloc.start()
    started = time.perf_counter()
    gold = gold_pairs(n)
    comparisons = 0
    refine_candidates = 0
    timing_kind = "measured"
    hierarchy_levels = 0
    if method == "all_pairs":
        candidates = n * (n - 1) // 2
        probe_n = min(n, all_pairs_limit)
        probe_pairs = probe_n * (probe_n - 1) // 2
        checksum = 0
        tick = time.perf_counter()
        for left in range(probe_n):
            for right in range(left + 1, probe_n):
                checksum ^= (left + right) & 1
        probe_seconds = time.perf_counter() - tick
        seconds = probe_seconds * candidates / max(1, probe_pairs)
        timing_kind = "measured" if n <= all_pairs_limit else "projected_from_loop"
        candidate_pairs = gold
        comparisons = candidates
        refine_candidates = candidates
        _ = checksum
    else:
        nodes = make_nodes(n)
        if method == "ann_only":
            candidate_pairs = ann_pairs(n, 8)
            comparisons = len(candidate_pairs)
            refine_candidates = 0
        elif method == "flat_sparse":
            rows, comparisons = _bounded_sparse_pairs(
                nodes, max_candidates=24, per_node_k=8)
            candidate_pairs = {
                tuple(sorted((left, right))) for left, right, _score in rows
            }
            refine_candidates = len(candidate_pairs)
        elif method == "cir":
            hierarchy = build_recursive_hierarchy(
                "scale", nodes, fanout=8, max_levels=8,
                summary_words=96, max_candidates=24)
            node_map = {row.node_id: row for row in
                        (*nodes, *hierarchy.parent_cards)}
            plan = build_parent_gated_relations(
                "scale", hierarchy, node_map, hierarchy.children,
                embedding_k=8, max_candidates_per_node=24,
                low_threshold=0.30, high_threshold=0.45,
                refine_mode="ambiguous_only")
            leaf_ids = {row.node_id for row in nodes}
            candidate_pairs = {
                tuple(sorted((left, right)))
                for left, right, _score, _level in plan.accepted_pairs
                if left in leaf_ids and right in leaf_ids
            }
            candidate_pairs.update(
                tuple(sorted((row.left_id, row.right_id)))
                for row in plan.refine_candidates
                if row.left_id in leaf_ids and row.right_id in leaf_ids
            )
            comparisons = (hierarchy.stats.cluster_candidate_comparisons
                           + plan.score_comparisons)
            refine_candidates = len(plan.refine_candidates)
            hierarchy_levels = hierarchy.stats.levels
        else:
            raise ValueError(method)
        candidates = len(candidate_pairs)
        seconds = time.perf_counter() - started
    _current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "method": method,
        "n": n,
        "candidate_relations": candidates,
        "score_comparisons": comparisons,
        "refine_candidates": refine_candidates,
        "relation_decision_tokens": refine_candidates * RELATION_DECISION_TOKENS,
        "wall_seconds": seconds,
        "timing_kind": timing_kind,
        "peak_rss_mib": rss_mib,
        "python_peak_mib": python_peak / 1024 / 1024,
        "hierarchy_levels": hierarchy_levels,
        **quality(candidate_pairs, gold),
    }
    return result


def setup_chinese_font() -> None:
    path = Path("/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")
    if path.exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = ["FandolHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def plot(rows: pd.DataFrame, output: Path) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.8))
    for method in LABELS:
        part = rows[rows.method == method].sort_values("n")
        axes[0].plot(part.n, part.candidate_relations, marker="o",
                     color=COLORS[method], label=LABELS[method])
        if method != "ann_only":
            axes[1].plot(part.n, part.relation_decision_tokens, marker="o",
                         color=COLORS[method], label=LABELS[method])
        axes[2].plot(part.n, part.multi_hop_path_retention * 100, marker="o",
                     color=COLORS[method], label=LABELS[method])
    axes[0].set(xscale="log", yscale="log", xlabel="Session Card 数量 $N$",
                ylabel="候选关系数", title="(a) 候选规模")
    axes[1].set(xscale="log", yscale="log", xlabel="Session Card 数量 $N$",
                ylabel="关系判定 Token 上界", title="(b) 关系索引开支")
    axes[1].text(0.04, 0.06, "仅 ANN：0 关系判定 Token\n（不生成类型关系）",
                 transform=axes[1].transAxes, color=COLORS["ann_only"], fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": COLORS["ann_only"],
                       "alpha": 0.85, "boxstyle": "round,pad=0.3"})
    axes[2].set(xscale="log", xlabel="Session Card 数量 $N$",
                ylabel="受控多跳路径保留率（%）", title="(c) 关系表达保留")
    axes[2].set_ylim(-3, 103)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"eval_c1.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sizes = tuple(int(item) for item in args.sizes.split(",") if item)
    specs = [(method, n, args.all_pairs_limit)
             for n in sizes for method in LABELS]
    rows = []
    # One task per process makes peak RSS comparable across methods.
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        for row in pool.map(run_row, specs):
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows).sort_values(["method", "n"])
    frame.to_csv(args.output / "c1_scaling.csv", index=False)
    exponents = {}
    for method in LABELS:
        part = frame[(frame.method == method) & (frame.candidate_relations > 0)]
        exponents[method] = float(np.polyfit(
            np.log(part.n), np.log(part.candidate_relations), 1)[0])
    config = load_config(args.config)
    metadata = {
        "experiment": "report_c1_scaling",
        "config": str(args.config),
        "config_hash": config_hash(config),
        "sizes": sizes,
        "candidate_scaling_exponent": exponents,
        "token_definition": {
            "per_refine_candidate": RELATION_DECISION_TOKENS,
            "formula": "2*96 input + 256 output",
            "scope": "relation-decision envelope; shared fact extraction excluded",
        },
        "all_pairs_timing": (
            f"measured through N={args.all_pairs_limit}; larger rows projected from "
            "the same Python pair-enumeration loop"),
        "quality_workload": "controlled topic chains; candidate/path retention, not QA accuracy",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=False).stdout.strip(),
        "rows": rows,
    }
    (args.output / "c1_scaling.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(frame, args.output)
    print(json.dumps({"exponents": exponents, "output": str(args.output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
