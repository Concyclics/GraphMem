#!/usr/bin/env python3
"""Rebuild only routing hierarchy/parent gates over a frozen V5.8 fact graph."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.coarsen import (  # noqa: E402
    build_parent_gated_relations,
    build_recursive_hierarchy,
)
from graphmem.domain import GraphEdge, NodeType, RelationType, stable_id  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path,
                        default=Path("../artifacts/v5_8/lme_gold/graphmem.sqlite"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/report/v5_9/c23_graph"))
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Question limit; zero processes the complete selected set.")
    parser.add_argument("--development-set", action="store_true",
                        help="Load arbitrary-size fixed development subsets.")
    parser.add_argument(
        "--all-memories",
        action="store_true",
        help=(
            "recoarsen every benchmark memory, including LongMemEval rows without "
            "turn-level gold; required for the 500+1540 end-to-end benchmark"
        ),
    )
    parser.add_argument("--fanout", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--assignment-method", choices=(
        "bounded_semantic_partition", "hnsw"), default="hnsw")
    parser.add_argument("--relation-candidate-method", choices=(
        "bounded_sparse", "hnsw"), default="hnsw")
    parser.add_argument("--typed-restoration", action="store_true")
    parser.add_argument("--cross-session-quota", type=int, default=2)
    parser.add_argument("--embedding-model",
                        default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def structural_edge(memory_id, left, right):
    groups = tuple(dict.fromkeys((left.evidence_group_id,
                                 right.evidence_group_id)))
    return GraphEdge(
        stable_id("edge", memory_id, left.node_id,
                  RelationType.REFINES_TO, right.node_id),
        memory_id, left.node_id, RelationType.REFINES_TO, right.node_id,
        groups[0], True, 1.0, "report_recursive_recoarsening", groups[1:])


def relation_edge(memory_id, left, right, score):
    groups = tuple(dict.fromkeys((left.evidence_group_id,
                                 right.evidence_group_id)))
    return GraphEdge(
        stable_id("edge", memory_id, left.node_id,
                  RelationType.COARSE_RELATED, right.node_id),
        memory_id, left.node_id, RelationType.COARSE_RELATED, right.node_id,
        groups[0], False, score, "report_cir_high_confidence", groups[1:])


def typed_edge(memory_id, left, right, relation, confidence, source):
    groups = tuple(dict.fromkeys((left.evidence_group_id,
                                 right.evidence_group_id)))
    return GraphEdge(
        stable_id("edge", memory_id, left.node_id, relation, right.node_id),
        memory_id, left.node_id, relation, right.node_id,
        groups[0], True, confidence, source, groups[1:])


def bounded_new_relations(edges, *, generic_cap: int = 16,
                          typed_cap: int = 8):
    degree: dict[tuple[str, RelationType], int] = defaultdict(int)
    result = []
    for edge in sorted(edges, key=lambda row: (
            not row.directed, -row.confidence, str(row.relation),
            row.src_id, row.dst_id, row.edge_id)):
        cap = generic_cap if edge.relation == RelationType.COARSE_RELATED else typed_cap
        left = (edge.src_id, edge.relation)
        right = (edge.dst_id, edge.relation)
        if degree[left] >= cap or degree[right] >= cap:
            continue
        result.append(edge)
        degree[left] += 1
        degree[right] += 1
    return result


def session_card_vectors(memory_id: str, source: SQLiteGraphStore,
                         leaves, model_id: str) -> dict[str, np.ndarray]:
    """Average cached Qwen turn vectors into session-card vectors."""

    by_session: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in source._read(
            "SELECT t.session_id,e.dimension,e.vector FROM embeddings e "
            "JOIN source_turns t ON t.turn_id=e.item_id "
            "WHERE e.memory_id=? AND e.model_id=?",
            (memory_id, model_id)):
        by_session[str(row["session_id"])].append(np.frombuffer(
            row["vector"], dtype=np.float32, count=int(row["dimension"])))
    result = {}
    for leaf in leaves:
        rows = by_session.get(str(leaf.attributes.get("session_id", "")), ())
        if rows:
            vector = np.mean(rows, axis=0).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            result[leaf.node_id] = vector
    return result


def recoarsen(memory_id: str, source: SQLiteGraphStore,
              target: SQLiteGraphStore, fanout: int, max_levels: int, *,
              assignment_method: str, relation_candidate_method: str,
              typed_restoration: bool, cross_session_quota: int,
              embedding_model: str) -> dict:
    nodes = list(source.nodes(memory_id))
    edges = list(source.edges(memory_id))
    groups = source.evidence_groups(memory_id)
    leaves = [node for node in nodes
              if (node.node_type == NodeType.ROUTING_CARD
                  and node.attributes.get("session_id") is not None)]
    if not leaves:
        target.replace_graph(memory_id, nodes, edges, groups)
        return {
            "memory_id": memory_id,
            "sessions": 0,
            "old_nodes": len(nodes), "new_nodes": len(nodes),
            "old_edges": len(edges), "new_edges": len(edges),
            "old_parent_cards": 0, "new_parent_cards": 0,
            "hierarchy_levels": 0, "coarsen_comparisons": 0,
            "relation_comparisons": 0, "relation_candidates": 0,
            "accepted_relations": 0, "deferred_refine_candidates": 0,
            "recoarsened": False, "reason": "no_session_cards_in_source_snapshot",
        }
    old_parent_ids = {
        node.node_id for node in nodes
        if node.node_type == NodeType.ROUTING_CARD and node.node_id not in {
            leaf.node_id for leaf in leaves}
    }
    vectors = session_card_vectors(memory_id, source, leaves, embedding_model)
    hierarchy = build_recursive_hierarchy(
        memory_id, leaves, fanout=fanout, max_levels=max_levels,
        summary_words=320, max_candidates=24,
        assignment_method=assignment_method, vectors=vectors or None)
    kept_nodes = [node for node in nodes if node.node_id not in old_parent_ids]
    kept_ids = {node.node_id for node in kept_nodes}
    kept_edges = [
        edge for edge in edges
        if edge.src_id in kept_ids and edge.dst_id in kept_ids
        and edge.relation != RelationType.COARSE_RELATED
    ]
    node_map = {node.node_id: node for node in
                (*kept_nodes, *hierarchy.parent_cards)}
    for parent_id, child_ids in hierarchy.children.items():
        parent = node_map[parent_id]
        kept_edges.extend(structural_edge(memory_id, parent, node_map[child_id])
                          for child_id in child_ids)

    child_map: dict[str, list[str]] = {
        parent: list(children) for parent, children in hierarchy.children.items()
    }
    for edge in kept_edges:
        if edge.relation in {RelationType.REFINES_TO, RelationType.SCENE_CONTAINS}:
            child_map.setdefault(edge.src_id, []).append(edge.dst_id)
    relation_plan = build_parent_gated_relations(
        memory_id, hierarchy, node_map, child_map,
        embedding_k=8, max_candidates_per_node=24,
        low_threshold=0.35, high_threshold=0.78,
        refine_mode="ambiguous_only",
        candidate_method=relation_candidate_method,
        vectors=hierarchy.vectors,
        cross_session_quota=cross_session_quota,
        typed_restoration=typed_restoration,
        max_refine_candidates_per_node=2,
        max_refine_candidates_per_1000_nodes=480)
    new_relation_edges = []
    for left_id, right_id, score, _level in relation_plan.accepted_pairs:
        new_relation_edges.append(relation_edge(
            memory_id, node_map[left_id], node_map[right_id], score))
    for (left_id, right_id, relation, confidence, _level,
         source_name) in relation_plan.typed_pairs:
        new_relation_edges.append(typed_edge(
            memory_id, node_map[left_id], node_map[right_id], relation,
            confidence, source_name))
    materialized_relations = bounded_new_relations(new_relation_edges)
    kept_edges.extend(materialized_relations)

    dedup_edges = tuple({edge.edge_id: edge for edge in kept_edges}.values())
    target.replace_graph(
        memory_id, tuple(node_map.values()), dedup_edges, groups)
    return {
        "memory_id": memory_id,
        "sessions": len(leaves),
        "old_nodes": len(nodes),
        "new_nodes": len(node_map),
        "old_edges": len(edges),
        "new_edges": len(dedup_edges),
        "old_parent_cards": len(old_parent_ids),
        "new_parent_cards": len(hierarchy.parent_cards),
        "hierarchy_levels": hierarchy.stats.levels,
        "coarsen_comparisons": hierarchy.stats.cluster_candidate_comparisons,
        "relation_comparisons": relation_plan.score_comparisons,
        "relation_candidates": relation_plan.coarse_candidate_pairs,
        "accepted_relations": len(relation_plan.accepted_pairs),
        "materialized_relations": len(materialized_relations),
        "deferred_refine_candidates": len(relation_plan.refine_candidates),
        "generated_refine_candidates": relation_plan.refine_candidates_generated,
        "dropped_refine_candidates": relation_plan.refine_candidates_dropped,
        "typed_relations": len(relation_plan.typed_pairs),
        "assignment_method": hierarchy.stats.assignment_method,
        "relation_candidate_method": relation_plan.candidate_method,
        "ann_queries": hierarchy.stats.ann_queries,
        "vector_dimension": hierarchy.stats.vector_dimension,
        "qwen_session_vectors": len(vectors),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target_path = args.output / "report_graph.sqlite"
    if target_path.exists() and not args.resume:
        raise SystemExit(
            f"{target_path} already exists; use --resume or choose a new --output")
    source = SQLiteGraphStore(args.source_db, read_only=True)
    target = SQLiteGraphStore(target_path)
    present = {str(row["memory_id"]) for row in
               source._read("SELECT memory_id FROM conversations")}
    gold = load_gold_turns(args.gold)
    if args.development_set:
        questions = list(load_dev_questions(args.lme, args.locomo, gold))
    else:
        questions = [row.question for row in load_full_questions(
            args.lme, args.locomo, gold)]
    questions = [row for row in questions
                 if row.memory_id in present
                 and (args.all_memories or row.gold_turns)]
    questions.sort(key=lambda row: (row.memory_id, row.question_id))
    questions = questions[:args.limit] if args.limit else questions
    memory_ids = tuple(dict.fromkeys(row.memory_id for row in questions))
    checkpoint_path = args.output / "recoarsen_rows.jsonl"
    checkpoint_rows = []
    if args.resume and checkpoint_path.exists():
        checkpoint_rows = [
            json.loads(line)
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {str(row["memory_id"]) for row in checkpoint_rows}
    rows = list(checkpoint_rows)
    started = time.perf_counter()
    for index, memory_id in enumerate(memory_ids, 1):
        if memory_id in completed:
            print(f"{index}/{len(memory_ids)} {memory_id}: resumed", flush=True)
            continue
        conversation = source.conversation(memory_id)
        if conversation is None:
            continue
        target.ingest_conversation(
            conversation, source.sessions(memory_id), source.turns(memory_id))
        row = recoarsen(
            memory_id, source, target, args.fanout, args.max_levels,
            assignment_method=args.assignment_method,
            relation_candidate_method=args.relation_candidate_method,
            typed_restoration=args.typed_restoration,
            cross_session_quota=args.cross_session_quota,
            embedding_model=args.embedding_model)
        rows.append(row)
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{index}/{len(memory_ids)} {memory_id}: "
              f"levels={row['hierarchy_levels']} nodes={row['new_nodes']} "
              f"relations={row['accepted_relations']}", flush=True)
    payload = {
        "experiment": "recoarsen_frozen_v5_8_snapshot",
        "source_db": str(args.source_db),
        "target_db": str(target_path),
        "questions": len(questions),
        "memories": len(rows),
        "all_memories": args.all_memories,
        "development_set": args.development_set,
        "fanout": args.fanout,
        "max_levels": args.max_levels,
        "assignment_method": args.assignment_method,
        "relation_candidate_method": args.relation_candidate_method,
        "typed_restoration": args.typed_restoration,
        "cross_session_quota": args.cross_session_quota,
        "embedding_model": args.embedding_model,
        "wall_seconds": time.perf_counter() - started,
        "fact_graph_contract": (
            "all non-routing nodes and non-hierarchy edges are copied unchanged; "
            "no extraction or answer model call"),
        "rows": rows,
    }
    (args.output / "recoarsen_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target.close(); source.close()


if __name__ == "__main__":
    main()
