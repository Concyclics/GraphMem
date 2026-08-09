#!/usr/bin/env python3
"""Gate B: frozen graph vs Qwen-HNSW coarsening and typed restoration."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import RelationType  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.retrieval.scheduler import DEFAULT_PREFERRED  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


TYPED = frozenset({
    RelationType.SAME_ENTITY_STATE,
    RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
    RelationType.COREFERENCE,
})
PATH_RELATIONS = frozenset({
    RelationType.COARSE_RELATED, RelationType.SHARED_REFERENT,
    RelationType.PORTAL, *TYPED,
})


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / "artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-db", type=Path, default=WORKSPACE /
                        "artifacts/v5_9/diagnostic_ablations/dense_dev200/report_graph_dense.sqlite")
    parser.add_argument("--hnsw-db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/hnsw_qwen_typed_dev200_graph_released/report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/graph_gate_dev200")
    return parser.parse_args()


def _session_path_graph(store: SQLiteGraphStore, memory_id: str, *,
                        include_typed: bool) -> tuple[
                            dict[str, str], dict[str, set[str]]]:
    nodes = {node.node_id: node for node in store.nodes(memory_id)}
    session_nodes = {
        str(node.attributes["session_id"]): node.node_id
        for node in nodes.values()
        if node.node_type.value == "routing_card"
        and node.attributes.get("session_id") is not None
    }
    allowed = PATH_RELATIONS if include_typed else PATH_RELATIONS - TYPED
    allowed = frozenset((*allowed, RelationType.REFINES_TO))
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in store.edges(memory_id):
        if edge.relation not in allowed:
            continue
        adjacency[edge.src_id].add(edge.dst_id)
        adjacency[edge.dst_id].add(edge.src_id)
    return session_nodes, adjacency


def _reachable(adjacency: dict[str, set[str]], source: str, target: str,
               max_hops: int) -> bool:
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in adjacency.get(node, ()):
            if neighbour == target:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, depth + 1))
    return False


def _ci(rows: list[dict[str, Any]], left: str, right: str, key: str,
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
        config.query_budget,
        max_evidence_turns=args.max_evidence_turns,
        max_evidence_tokens=args.max_evidence_tokens)
    baseline_store = SQLiteGraphStore(args.baseline_db, read_only=True)
    hnsw_store = SQLiteGraphStore(args.hnsw_db, read_only=True)
    common = dict(
        harness_profile=HarnessProfile.H10_AST,
        hierarchical_routing=True, hierarchy_operator_aware=True,
        hierarchy_root_beam=2, hierarchy_child_beam=4,
        graph_hop_decay=0.3, expansion_beam=2,
        obligation_aware_packing=True, span_pack_window=96,
        read_pool_size=4)
    navigators = {
        "frozen": GraphNavigator(baseline_store, **common),
        "hnsw": GraphNavigator(
            hnsw_store, **common,
            preferred_relations=tuple(row for row in DEFAULT_PREFERRED
                                      if row not in TYPED)),
        "hnsw_typed": GraphNavigator(hnsw_store, **common),
    }
    stores = {"frozen": baseline_store, "hnsw": hnsw_store,
              "hnsw_typed": hnsw_store}
    path_cache: dict[tuple[str, str], tuple[
        dict[str, str], dict[str, set[str]]]] = {}
    turn_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "stratum": question.stratum,
            "query": question.query,
        }
        gold_refs = {(item.session_id, item.turn_index)
                     for item in question.gold_turns}
        gold_sessions = sorted({item.session_id for item in question.gold_turns})
        gold_pairs = list(combinations(gold_sessions, 2))
        for arm, navigator in navigators.items():
            store = stores[arm]
            cache_key = (arm, question.memory_id)
            if cache_key not in turn_cache:
                turn_cache[cache_key] = {
                    turn.turn_id: turn for turn in store.turns(question.memory_id)}
            turn_map = turn_cache[cache_key]
            tick = time.perf_counter()
            result = navigator.navigate(question.memory_id, question.query, budget)
            latency = (time.perf_counter() - tick) * 1000
            packed_refs = {
                (turn_map[turn_id].session_id, turn_map[turn_id].turn_index)
                for turn_id in result.packed_turn_ids if turn_id in turn_map}
            candidate_refs = {
                (turn_map[item.turn_id].session_id,
                 turn_map[item.turn_id].turn_index)
                for item in result.candidate_scores if item.turn_id in turn_map}
            path_key = (("frozen" if arm == "frozen" else
                         "hnsw_typed" if arm == "hnsw_typed" else "hnsw"),
                        question.memory_id)
            if path_key not in path_cache:
                path_cache[path_key] = _session_path_graph(
                    store, question.memory_id, include_typed=arm == "hnsw_typed")
            session_nodes, adjacency = path_cache[path_key]
            direct = statistics.fmean(
                float(session_nodes.get(right, "") in adjacency.get(
                    session_nodes.get(left, "missing:left"), ()))
                for left, right in gold_pairs
            ) if gold_pairs else 1.0
            two_hop = statistics.fmean(
                float(_reachable(
                    adjacency, session_nodes.get(left, "missing:left"),
                    session_nodes.get(right, "missing:right"), 2))
                for left, right in gold_pairs
            ) if gold_pairs else 1.0
            recall = (len(gold_refs & packed_refs) / len(gold_refs)
                      if gold_refs else 1.0)
            row[arm] = {
                "all_hit": float(gold_refs <= packed_refs),
                "recall": recall,
                "candidate_all_hit": float(gold_refs <= candidate_refs),
                "turns": len(result.packed_turn_ids),
                "tokens": result.evidence_tokens,
                "latency_ms": latency,
                "direct_gold_session_path": direct,
                "two_hop_gold_session_path": two_hop,
                "visited_nodes": result.visited_nodes,
                "visited_edges": result.visited_edges,
                "typed_edges_walked": sum(
                    count for relation, count in result.trace.get(
                        "relation_counts", {}).items()
                    if relation in {str(item) for item in TYPED}),
            }
        rows.append(row)
        if index % 25 == 0:
            print(f"{index}/{len(questions)}", flush=True)

    def aggregate(items, arm):
        fields = (
            "all_hit", "recall", "candidate_all_hit", "turns", "tokens",
            "latency_ms", "direct_gold_session_path",
            "two_hop_gold_session_path", "visited_nodes", "visited_edges",
            "typed_edges_walked")
        return {field: statistics.fmean(row[arm][field] for row in items)
                for field in fields}

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    comparisons = {}
    for left, right in (("frozen", "hnsw"), ("hnsw", "hnsw_typed"),
                        ("frozen", "hnsw_typed")):
        comparisons[f"{left}->{right}"] = {
            key: {
                "mean": statistics.fmean(
                    row[right][key] - row[left][key] for row in rows),
                "ci95": _ci(rows, left, right, key),
            } for key in (
                "all_hit", "recall", "tokens", "latency_ms",
                "direct_gold_session_path", "two_hop_gold_session_path")
        }
        comparisons[f"{left}->{right}"]["transitions"] = dict(Counter(
            f"{int(row[left]['all_hit'])}->{int(row[right]['all_hit'])}"
            for row in rows))
    summary = {
        "baseline_db": str(args.baseline_db), "hnsw_db": str(args.hnsw_db),
        "questions": len(rows),
        "budget": {"turns": budget.max_evidence_turns,
                   "tokens": budget.max_evidence_tokens},
        "overall": {arm: aggregate(rows, arm) for arm in navigators},
        "per_stratum": {
            key: {arm: aggregate(items, arm) for arm in navigators}
            for key, items in sorted(strata.items())},
        "comparisons": comparisons,
        "wall_time_sec": time.perf_counter() - started,
    }
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    baseline_store.close(); hnsw_store.close()


if __name__ == "__main__":
    main()
