#!/usr/bin/env python3
"""Audit source->fact->relation->path losses on a frozen GraphMem graph.

Gold annotations are used only for offline evaluation.  The audit separates a
missing atomic fact from a missing relation path, and reports relation topology
without changing the graph or calling a model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import NodeType, RelationType  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


TYPED_RELATIONS = frozenset({
    RelationType.COREFERENCE,
    RelationType.SAME_ENTITY_STATE,
    RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
})
CONTENT_RELATIONS = frozenset({
    RelationType.COARSE_RELATED,
    RelationType.SHARED_REFERENT,
    RelationType.PORTAL,
    RelationType.COLLECTION_CO_MEMBER,
    RelationType.STATE_NEXT,
    RelationType.TEMPORAL_BEFORE,
    RelationType.SHARED_VALUE,
    RelationType.FACT_VALUE,
    *TYPED_RELATIONS,
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
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--atomic-summary", type=Path, default=ROOT /
                        "artifacts/report/v5_10/atomic_gate_v3/summary.json")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/build_navigation_audit_dev200")
    return parser.parse_args()


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def node_sessions(node) -> frozenset[str]:
    values = node.attributes.get("session_ids", ())
    result = ({str(values)} if isinstance(values, (str, bytes))
              else set(map(str, values or ())))
    if node.attributes.get("session_id"):
        result.add(str(node.attributes["session_id"]))
    return frozenset(item for item in result if item)


def reachable(adjacency: Mapping[str, set[str]], sources: Iterable[str],
              targets: set[str], max_hops: int) -> bool:
    source_set = set(sources)
    if source_set & targets:
        return True
    queue = deque((node_id, 0) for node_id in source_set)
    seen = set(source_set)
    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in adjacency.get(node_id, ()):
            if neighbour in targets:
                return True
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, depth + 1))
    return False


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "gold_turn_fact_recall", "all_gold_turns_have_fact",
        "direct_fact_pair_recall", "two_hop_fact_pair_recall",
        "typed_direct_fact_pair_recall", "all_fact_pairs_two_hop",
        "owner_portal_fact_pair_recall", "all_fact_pairs_owner_portal",
        "content_or_owner_pair_recall", "all_pairs_content_or_owner",
        "direct_session_pair_recall",
        "two_hop_session_pair_recall", "all_session_pairs_two_hop",
        "gold_gold_edge_yield",
    )
    failures = Counter(row["build_failure_stage"] for row in rows)
    return {
        "questions": len(rows),
        **{
            field: statistics.fmean(float(row[field]) for row in rows)
            if rows else 0.0
            for field in fields
        },
        "build_failure_stage": dict(sorted(failures.items())),
    }


def read_optional(path: Path | None) -> Any:
    return (json.loads(path.read_text(encoding="utf-8"))
            if path is not None and path.exists() else None)


def main() -> None:
    args = parse_args()
    if args.max_hops < 1:
        raise ValueError("--max-hops must be positive")
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    store = SQLiteGraphStore(args.db, read_only=True)
    cache: dict[str, dict[str, Any]] = {}
    topology_relation = Counter()
    topology_endpoints = Counter()
    topology_cross_session = Counter()
    atomic_content_degrees: list[int] = []
    atomic_typed_degrees: list[int] = []
    typed_component_sizes: list[int] = []
    typed_largest_component_ratios: list[float] = []

    def memory_view(memory_id: str) -> dict[str, Any]:
        cached = cache.get(memory_id)
        if cached is not None:
            return cached
        turns = tuple(store.turns(memory_id))
        turn_by_ref = {
            (turn.session_id, turn.turn_index): turn.turn_id for turn in turns}
        groups = {
            group.evidence_group_id: tuple(member.turn_id for member in group.members)
            for group in store.evidence_groups(memory_id)}
        nodes = {node.node_id: node for node in store.nodes(memory_id)}
        node_turns = {
            node_id: frozenset(
                turn_id for group_id in node.all_evidence_group_ids
                for turn_id in groups.get(group_id, ()))
            for node_id, node in nodes.items()}
        facts_by_turn: dict[str, set[str]] = defaultdict(set)
        owners_by_fact: dict[str, set[str]] = defaultdict(set)
        route_by_session: dict[str, set[str]] = defaultdict(set)
        for node_id, node in nodes.items():
            if node.node_type == NodeType.CANONICAL_FACT:
                for turn_id in node_turns[node_id]:
                    facts_by_turn[turn_id].add(node_id)
                owner_id = str(node.attributes.get("owner_id", ""))
                if owner_id:
                    owners_by_fact[node_id].add(owner_id)
            if (node.node_type == NodeType.ROUTING_CARD
                    and node.attributes.get("session_id") is not None):
                route_by_session[str(node.attributes["session_id"])].add(node_id)
        content: dict[str, set[str]] = defaultdict(set)
        typed: dict[str, set[str]] = defaultdict(set)
        content_edges = []
        for edge in store.edges(memory_id):
            topology_relation[str(edge.relation)] += 1
            left = nodes.get(edge.src_id); right = nodes.get(edge.dst_id)
            if left is None or right is None:
                continue
            topology_endpoints[
                f"{edge.relation}:{left.node_type}->{right.node_type}"] += 1
            left_sessions = node_sessions(left); right_sessions = node_sessions(right)
            if (left_sessions and right_sessions
                    and left_sessions.isdisjoint(right_sessions)):
                topology_cross_session[str(edge.relation)] += 1
            if edge.relation in CONTENT_RELATIONS:
                content[edge.src_id].add(edge.dst_id)
                content[edge.dst_id].add(edge.src_id)
                content_edges.append(edge)
            if (edge.relation == RelationType.HAS_FACT
                    and left.node_type == NodeType.CANONICAL_ENTITY
                    and right.node_type == NodeType.CANONICAL_FACT):
                owners_by_fact[edge.dst_id].add(edge.src_id)
            if edge.relation in TYPED_RELATIONS:
                typed[edge.src_id].add(edge.dst_id)
                typed[edge.dst_id].add(edge.src_id)
        atomic_ids = {
            node_id for node_id, node in nodes.items()
            if node.node_type in {
                NodeType.CANONICAL_FACT, NodeType.EVENT_FRAME,
                NodeType.EVENT_SKELETON, NodeType.STATE_HEAD,
                NodeType.STATE_VALUE,
            }}
        atomic_content_degrees.extend(
            len(content.get(node_id, ())) for node_id in atomic_ids)
        atomic_typed_degrees.extend(
            len(typed.get(node_id, ())) for node_id in atomic_ids)
        unseen = set(atomic_ids)
        local_components: list[int] = []
        while unseen:
            seed = unseen.pop()
            queue = [seed]
            size = 0
            while queue:
                current = queue.pop()
                size += 1
                neighbours = set(typed.get(current, ())) & unseen
                unseen.difference_update(neighbours)
                queue.extend(neighbours)
            local_components.append(size)
        typed_component_sizes.extend(local_components)
        typed_largest_component_ratios.append(
            ratio(max(local_components, default=0), len(atomic_ids)))
        cached = {
            "turn_by_ref": turn_by_ref,
            "node_turns": node_turns,
            "facts_by_turn": facts_by_turn,
            "owners_by_fact": owners_by_fact,
            "route_by_session": route_by_session,
            "content": content,
            "typed": typed,
            "content_edges": tuple(content_edges),
        }
        cache[memory_id] = cached
        return cached

    rows: list[dict[str, Any]] = []
    for question in questions:
        view = memory_view(question.memory_id)
        gold_refs = tuple(dict.fromkeys(
            (item.session_id, item.turn_index) for item in question.gold_turns))
        gold_turns = tuple(
            view["turn_by_ref"].get(ref) for ref in gold_refs)
        resolved_turns = tuple(item for item in gold_turns if item is not None)
        fact_sets = [set(view["facts_by_turn"].get(turn_id, ()))
                     for turn_id in resolved_turns]
        covered = sum(bool(item) for item in fact_sets)
        fact_pairs = list(combinations(fact_sets, 2))
        direct_fact = [reachable(view["content"], left, right, 1)
                       if left and right else False
                       for left, right in fact_pairs]
        two_hop_fact = [reachable(
            view["content"], left, right, args.max_hops)
                        if left and right else False
                        for left, right in fact_pairs]
        typed_direct = [reachable(view["typed"], left, right, 1)
                        if left and right else False
                        for left, right in fact_pairs]
        owner_portal = [bool(
            set().union(*(view["owners_by_fact"].get(item, ()) for item in left))
            & set().union(*(view["owners_by_fact"].get(item, ()) for item in right)))
            if left and right else False for left, right in fact_pairs]
        content_or_owner = [content_hit or owner_hit for content_hit, owner_hit
                            in zip(two_hop_fact, owner_portal)]

        gold_sessions = tuple(dict.fromkeys(ref[0] for ref in gold_refs))
        session_sets = [set(view["route_by_session"].get(session_id, ()))
                        for session_id in gold_sessions]
        session_pairs = list(combinations(session_sets, 2))
        direct_session = [reachable(view["content"], left, right, 1)
                          if left and right else False
                          for left, right in session_pairs]
        two_hop_session = [reachable(
            view["content"], left, right, args.max_hops)
                           if left and right else False
                           for left, right in session_pairs]

        gold_fact_nodes = set().union(*fact_sets) if fact_sets else set()
        gold_turn_set = set(resolved_turns)
        incident = 0; gold_gold = 0
        for edge in view["content_edges"]:
            touches = edge.src_id in gold_fact_nodes or edge.dst_id in gold_fact_nodes
            if not touches:
                continue
            incident += 1
            left_gold = view["node_turns"].get(edge.src_id, frozenset()) & gold_turn_set
            right_gold = view["node_turns"].get(edge.dst_id, frozenset()) & gold_turn_set
            # A relation between two fact projections of the same source turn
            # is not a navigation bridge.  Count only edges that connect two
            # distinct annotated evidence turns.
            gold_gold += int(any(left != right for left in left_gold
                                 for right in right_gold))
        all_facts = bool(resolved_turns) and covered == len(resolved_turns)
        all_paths = all(two_hop_fact) if fact_pairs else all_facts
        all_owner_paths = all(owner_portal) if fact_pairs else all_facts
        all_combined_paths = all(content_or_owner) if fact_pairs else all_facts
        if not all_facts:
            failure = "source_to_fact_missing"
        elif all_paths:
            failure = "content_path_available"
        elif all_owner_paths:
            failure = "owner_portal_only"
        elif all_combined_paths:
            failure = "mixed_content_owner_path"
        else:
            failure = "relation_path_missing"
        rows.append({
            "question_id": question.question_id,
            "memory_id": question.memory_id,
            "stratum": question.stratum,
            "gold_turns": len(resolved_turns),
            "gold_turns_with_fact": covered,
            "gold_turn_fact_recall": ratio(covered, len(resolved_turns)),
            "all_gold_turns_have_fact": float(all_facts),
            "fact_pairs": len(fact_pairs),
            "direct_fact_pair_recall": ratio(sum(direct_fact), len(direct_fact))
            if direct_fact else float(all_facts),
            "two_hop_fact_pair_recall": ratio(sum(two_hop_fact), len(two_hop_fact))
            if two_hop_fact else float(all_facts),
            "typed_direct_fact_pair_recall": ratio(
                sum(typed_direct), len(typed_direct)) if typed_direct else 0.0,
            "all_fact_pairs_two_hop": float(all_paths),
            "owner_portal_fact_pair_recall": ratio(
                sum(owner_portal), len(owner_portal))
            if owner_portal else float(all_facts),
            "all_fact_pairs_owner_portal": float(all(owner_portal))
            if owner_portal else float(all_facts),
            "content_or_owner_pair_recall": ratio(
                sum(content_or_owner), len(content_or_owner))
            if content_or_owner else float(all_facts),
            "all_pairs_content_or_owner": float(all_combined_paths),
            "session_pairs": len(session_pairs),
            "direct_session_pair_recall": ratio(
                sum(direct_session), len(direct_session))
            if direct_session else 1.0,
            "two_hop_session_pair_recall": ratio(
                sum(two_hop_session), len(two_hop_session))
            if two_hop_session else 1.0,
            "all_session_pairs_two_hop": float(all(two_hop_session)),
            "gold_incident_content_edges": incident,
            "gold_gold_content_edges": gold_gold,
            "gold_gold_edge_yield": ratio(gold_gold, incident),
            "build_failure_stage": failure,
        })

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    manifest = read_optional(args.manifest)
    manifest_aggregate = None
    if manifest is not None:
        manifest_rows = manifest.get("rows", ())
        manifest_aggregate = {
            key: sum(int(row.get(key, 0)) for row in manifest_rows)
            for key in (
                "relation_candidates", "accepted_relations",
                "deferred_refine_candidates", "generated_refine_candidates",
                "typed_relations", "llm_refine_decisions",
                "llm_refine_truncated", "llm_refined_relations")
        }
    payload = {
        "schema_version": "graphmem-v5.13-build-navigation-audit-v1",
        "method": {
            "read_only": True,
            "llm_calls": 0,
            "gold_used_only_for_offline_evaluation": True,
            "content_relations": sorted(map(str, CONTENT_RELATIONS)),
            "max_hops": args.max_hops,
        },
        "inputs": {
            "db": str(args.db),
            "manifest": str(args.manifest) if args.manifest else None,
        },
        "overall": aggregate(rows),
        "per_stratum": {
            name: aggregate(items) for name, items in sorted(strata.items())},
        "topology": {
            "relation_counts": dict(topology_relation.most_common()),
            "cross_session_relation_counts": dict(
                topology_cross_session.most_common()),
            "endpoint_type_counts": dict(topology_endpoints.most_common()),
            "typed_edges": sum(topology_relation[str(item)]
                               for item in TYPED_RELATIONS),
            "atomic_nodes": len(atomic_typed_degrees),
            "atomic_content_degree": {
                "mean": statistics.fmean(atomic_content_degrees)
                if atomic_content_degrees else 0.0,
                "p50": percentile(atomic_content_degrees, 0.50),
                "p95": percentile(atomic_content_degrees, 0.95),
                "max": max(atomic_content_degrees, default=0),
            },
            "atomic_typed_degree": {
                "mean": statistics.fmean(atomic_typed_degrees)
                if atomic_typed_degrees else 0.0,
                "p50": percentile(atomic_typed_degrees, 0.50),
                "p95": percentile(atomic_typed_degrees, 0.95),
                "max": max(atomic_typed_degrees, default=0),
                "isolated_ratio": ratio(sum(
                    degree == 0 for degree in atomic_typed_degrees),
                    len(atomic_typed_degrees)),
                "degree_gt_8_ratio": ratio(sum(
                    degree > 8 for degree in atomic_typed_degrees),
                    len(atomic_typed_degrees)),
            },
            "typed_components": {
                "count": len(typed_component_sizes),
                "p95_size": percentile(typed_component_sizes, 0.95),
                "max_size": max(typed_component_sizes, default=0),
                "mean_largest_component_ratio_per_memory": (
                    statistics.fmean(typed_largest_component_ratios)
                    if typed_largest_component_ratios else 0.0),
            },
        },
        "build_manifest": manifest_aggregate,
        "atomic_extraction_gate": read_optional(args.atomic_summary),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "overall": payload["overall"],
        "typed_edges": payload["topology"]["typed_edges"],
        "manifest": manifest_aggregate,
    }, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
