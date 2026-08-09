#!/usr/bin/env python3
"""Re-evaluate C1 with actual graph-path metrics instead of edge recall aliases.

The original C1 microbenchmark labels gold-edge recall as path retention.  This
companion experiment keeps the exact same deterministic workload/candidate
builders but separately measures direct-edge recall, <=2-hop reachability,
unbounded reachability, and complete connectivity of each controlled semantic
component.  It therefore tests whether coarsening preserves usable multi-hop
paths rather than merely retaining individual candidate pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.coarsen import (  # noqa: E402
    _bounded_sparse_pairs,
    build_parent_gated_relations,
    build_recursive_hierarchy,
)

from measure_report_c1_scaling import ann_pairs, gold_pairs, make_nodes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,2000,5000,10000,20000")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/report/v5_9/path_retention",
    )
    return parser.parse_args()


def candidate_pairs(method: str, n: int) -> set[tuple[str, str]]:
    gold = gold_pairs(n)
    if method == "all_pairs":
        # For the quality calculation an all-pairs graph need not be physically
        # materialised; every gold component is necessarily complete.
        return gold
    if method == "ann_only":
        return ann_pairs(n, 8)
    nodes = make_nodes(n)
    if method == "flat_sparse":
        rows, _ = _bounded_sparse_pairs(nodes, max_candidates=24, per_node_k=8)
        return {tuple(sorted((left, right))) for left, right, _score in rows}
    if method == "cir":
        hierarchy = build_recursive_hierarchy(
            "scale", nodes, fanout=8, max_levels=8,
            summary_words=96, max_candidates=24,
        )
        node_map = {row.node_id: row for row in (*nodes, *hierarchy.parent_cards)}
        plan = build_parent_gated_relations(
            "scale", hierarchy, node_map, hierarchy.children,
            embedding_k=8, max_candidates_per_node=24,
            low_threshold=0.30, high_threshold=0.45,
            refine_mode="ambiguous_only",
        )
        leaves = {row.node_id for row in nodes}
        result = {
            tuple(sorted((left, right)))
            for left, right, _score, _level in plan.accepted_pairs
            if left in leaves and right in leaves
        }
        result.update(
            tuple(sorted((row.left_id, row.right_id)))
            for row in plan.refine_candidates
            if row.left_id in leaves and row.right_id in leaves
        )
        return result
    raise ValueError(method)


def gold_components(gold: set[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in gold:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    result = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        result.append(component)
    return result


def reachable(adjacency: dict[str, set[str]], source: str, target: str,
              max_hops: int | None) -> bool:
    if source == target:
        return True
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, depth = queue.popleft()
        if max_hops is not None and depth >= max_hops:
            continue
        for neighbour in adjacency.get(node, ()):
            if neighbour == target:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, depth + 1))
    return False


def quality(pairs: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict:
    adjacency: dict[str, set[str]] = defaultdict(set)
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
        union(left, right)
    direct = sum(pair in pairs for pair in gold)
    two_hop = sum(reachable(adjacency, left, right, 2) for left, right in gold)
    any_hop = sum(find(left) == find(right) for left, right in gold)
    components = gold_components(gold)
    fully_connected = 0
    internally_connected = 0
    for component in components:
        anchor = next(iter(component))
        if all(find(anchor) == find(node) for node in component):
            fully_connected += 1
        internal = {
            node: adjacency.get(node, set()) & component for node in component
        }
        if all(reachable(internal, anchor, node, None) for node in component):
            internally_connected += 1
    degrees = [len(adjacency[node]) for node in adjacency]
    return {
        "candidate_edges": len(pairs),
        "gold_edges": len(gold),
        "gold_edge_recall": direct / max(1, len(gold)),
        "gold_pair_reachable_within_2_hops": two_hop / max(1, len(gold)),
        "gold_pair_reachable_any_hops": any_hop / max(1, len(gold)),
        "gold_components": len(components),
        "component_connected_global_graph_rate": fully_connected / max(1, len(components)),
        "component_connected_internal_edges_rate": internally_connected / max(1, len(components)),
        "mean_nonzero_degree": fmean(degrees) if degrees else 0.0,
    }


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    rows = []
    for n in sizes:
        gold = gold_pairs(n)
        for method in ("all_pairs", "ann_only", "flat_sparse", "cir"):
            print(f"measure {method} N={n}", flush=True)
            rows.append({"method": method, "n": n,
                         **quality(candidate_pairs(method, n), gold)})
    payload = {
        "schema_version": "graphmem-v5.9-path-retention-v1",
        "workload": "identical controlled topic/group workload as report C1",
        "interpretation": (
            "structural path quality only; not end-to-end QA and not typed-relation precision"
        ),
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "path_retention.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv = [
        "method,n,candidate_edges,gold_edges,gold_edge_recall,"
        "gold_pair_reachable_within_2_hops,gold_pair_reachable_any_hops,"
        "component_connected_global_graph_rate,component_connected_internal_edges_rate,"
        "mean_nonzero_degree"
    ]
    for row in rows:
        csv.append(",".join(str(row[key]) for key in (
            "method", "n", "candidate_edges", "gold_edges", "gold_edge_recall",
            "gold_pair_reachable_within_2_hops", "gold_pair_reachable_any_hops",
            "component_connected_global_graph_rate", "component_connected_internal_edges_rate",
            "mean_nonzero_degree",
        )))
    (args.output / "path_retention.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
