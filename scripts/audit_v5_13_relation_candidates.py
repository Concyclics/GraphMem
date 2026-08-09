#!/usr/bin/env python3
"""Measure relation-candidate recall before relation classification.

This read-only audit compares the former hashed lexical fallback with cached
Qwen turn embeddings projected through node provenance.  It deliberately stops
at the bounded flat ANN candidate set: no gold label changes candidate
generation and no LLM relation decision is made.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import hnswlib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build.coarsen import _feature_vectors  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/atomic_augmented_dev200/"
                        "report_graph.sqlite")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--embedding-model",
                        default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--node-embedding-db", type=Path,
        help="optional sidecar containing embeddings keyed by atomic node_id")
    parser.add_argument(
        "--skip-provenance", action="store_true",
        help="skip the supporting-turn projection arm for faster hybrid audits")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--ann-pool", type=int, default=24)
    parser.add_argument("--cross-session-quota", type=int, default=2)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/relation_candidate_vector_audit")
    return parser.parse_args()


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def projected_vectors(store: SQLiteGraphStore, memory_id: str, nodes,
                      model_id: str) -> dict[str, np.ndarray]:
    by_group: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in store._read(
            "SELECT em.evidence_group_id,e.item_id,e.dimension,e.vector "
            "FROM evidence_members em JOIN embeddings e ON e.item_id=em.turn_id "
            "WHERE e.memory_id=? AND e.model_id=?", (memory_id, model_id)):
        by_group[str(row["evidence_group_id"])][str(row["item_id"])] = (
            np.frombuffer(row["vector"], dtype=np.float32,
                          count=int(row["dimension"])))
    result = {}
    for node in nodes:
        supporting: dict[str, np.ndarray] = {}
        for group_id in node.all_evidence_group_ids:
            supporting.update(by_group.get(group_id, {}))
        if supporting:
            vector = np.mean(list(supporting.values()), axis=0).astype(np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            result[node.node_id] = vector
    return result


def stored_node_vectors(store: SQLiteGraphStore, memory_id: str, nodes,
                        model_id: str) -> dict[str, np.ndarray]:
    expected = {node.node_id for node in nodes}
    result = {}
    for row in store._read(
            "SELECT item_id,dimension,vector FROM embeddings "
            "WHERE memory_id=? AND model_id=?", (memory_id, model_id)):
        item_id = str(row["item_id"])
        if item_id in expected:
            vector = np.frombuffer(row["vector"], dtype=np.float32,
                                   count=int(row["dimension"])).copy()
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            result[item_id] = vector
    return result


def bounded_pairs_by_k(
    nodes, vectors: Mapping[str, np.ndarray], *, ks: Sequence[int],
    ann_pool: int, cross_session_quota: int,
) -> dict[int, set[tuple[str, str]]]:
    """Build one HNSW index and expose several bounded-degree operating points."""
    ordered = tuple(sorted(nodes, key=lambda row: row.node_id))
    if len(ordered) < 2:
        return {k: set() for k in ks}
    unique_ks = tuple(sorted(set(int(k) for k in ks)))
    if not unique_ks or unique_ks[0] < 1:
        raise ValueError("ks must contain positive integers")
    matrix = np.stack([vectors[node.node_id] for node in ordered]).astype(np.float32)
    index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    index.init_index(max_elements=len(ordered), ef_construction=100, M=16,
                     random_seed=42)
    index.add_items(matrix, np.arange(len(ordered), dtype=np.int32), num_threads=1)
    query_k = min(len(ordered), max(unique_ks[-1] + 1, ann_pool + 1))
    index.set_ef(max(32, query_k * 2))
    labels, distances = index.knn_query(matrix, k=query_k, num_threads=1)
    result: dict[int, set[tuple[str, str]]] = {k: set() for k in unique_ks}
    for source_index, (neighbours, neighbour_distances) in enumerate(
            zip(labels, distances)):
        source = ordered[source_index]
        source_session = str(source.attributes.get("session_id", ""))
        ranked = sorted((
            (1.0 - float(distance), int(target_index))
            for target_index, distance in zip(neighbours, neighbour_distances)
            if int(target_index) != source_index
        ), key=lambda row: (-row[0], ordered[row[1]].node_id))
        cross = [row for row in ranked if source_session and str(
            ordered[row[1]].attributes.get("session_id", ""))
            not in {"", source_session}]
        for k in unique_ks:
            selected = list(cross[:min(k, cross_session_quota)])
            selected_ids = {target_index for _score, target_index in selected}
            selected.extend(row for row in ranked if row[1] not in selected_ids)
            for _score, target_index in selected[:k]:
                result[k].add(tuple(sorted((source.node_id,
                                            ordered[target_index].node_id))))
    return result


def bounded_pairs(nodes, vectors: Mapping[str, np.ndarray], *, k: int,
                  ann_pool: int, cross_session_quota: int) -> set[tuple[str, str]]:
    return bounded_pairs_by_k(
        nodes, vectors, ks=(k,), ann_pool=ann_pool,
        cross_session_quota=cross_session_quota)[k]


def connected(pairs: set[tuple[str, str]], left: set[str],
              right: set[str]) -> bool:
    return any(tuple(sorted((a, b))) in pairs for a in left for b in right
               if a != b)


def aggregate(rows: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "gold_pair_candidate_recall": statistics.fmean(
            row[mode]["gold_pair_candidate_recall"] for row in rows) if rows else 0.0,
        "all_gold_pairs_candidates": statistics.fmean(
            row[mode]["all_gold_pairs_candidates"] for row in rows) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.k < 1 or args.ann_pool < args.k:
        raise ValueError("require 1 <= k <= ann-pool")
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    store = SQLiteGraphStore(args.db, read_only=True)
    node_embedding_store = (SQLiteGraphStore(args.node_embedding_db, read_only=True)
                            if args.node_embedding_db else None)
    views: dict[str, dict[str, Any]] = {}
    coverage = {"memories": 0, "facts": 0, "projected_facts": 0}

    def view(memory_id: str) -> dict[str, Any]:
        if memory_id in views:
            return views[memory_id]
        turns = tuple(store.turns(memory_id))
        turn_by_ref = {(turn.session_id, turn.turn_index): turn.turn_id
                       for turn in turns}
        groups = {
            group.evidence_group_id: tuple(member.turn_id for member in group.members)
            for group in store.evidence_groups(memory_id)}
        facts = tuple(node for node in store.nodes(memory_id)
                      if node.node_type == NodeType.CANONICAL_FACT)
        facts_by_turn: dict[str, set[str]] = defaultdict(set)
        for node in facts:
            for group_id in node.all_evidence_group_ids:
                for turn_id in groups.get(group_id, ()):
                    facts_by_turn[turn_id].add(node.node_id)
        lexical = _feature_vectors(facts, 1024)
        projected = ({} if args.skip_provenance else projected_vectors(
            store, memory_id, facts, args.embedding_model))
        semantic = dict(lexical)
        semantic.update(projected)
        node_summary = (stored_node_vectors(
            node_embedding_store, memory_id, facts, args.embedding_model)
            if node_embedding_store is not None else {})
        summary_vectors = dict(lexical)
        summary_vectors.update(node_summary)
        half_k = max(1, args.k // 2)
        lexical_by_k = bounded_pairs_by_k(
            facts, lexical, ks=(half_k, args.k), ann_pool=args.ann_pool,
            cross_session_quota=args.cross_session_quota)
        lexical_pairs = lexical_by_k[args.k]
        projected_pairs = (bounded_pairs(
            facts, semantic, k=args.k, ann_pool=args.ann_pool,
            cross_session_quota=args.cross_session_quota)
            if not args.skip_provenance else set())
        views[memory_id] = {
            "turn_by_ref": turn_by_ref,
            "facts_by_turn": facts_by_turn,
            "hashed_lexical": lexical_pairs,
        }
        if not args.skip_provenance:
            views[memory_id].update({
                "provenance_projected": projected_pairs,
                f"lexical_or_projected_k_le_{2 * args.k}": (
                    lexical_pairs | projected_pairs),
            })
        if node_embedding_store is not None:
            summary_by_k = bounded_pairs_by_k(
                facts, summary_vectors, ks=(half_k, args.k),
                ann_pool=args.ann_pool,
                cross_session_quota=args.cross_session_quota)
            views[memory_id].update({
                "atomic_summary": summary_by_k[args.k],
                f"lexical_or_atomic_fixed_k_{args.k}": (
                    lexical_by_k[half_k] | summary_by_k[half_k]),
                f"lexical_or_atomic_k_le_{2 * args.k}": (
                    lexical_pairs | summary_by_k[args.k]),
            })
        coverage["memories"] += 1
        coverage["facts"] += len(facts)
        coverage["projected_facts"] += len(projected)
        coverage["node_summary_facts"] = (
            coverage.get("node_summary_facts", 0) + len(node_summary))
        if coverage["memories"] % 10 == 0:
            print(f"{coverage['memories']} memories audited", flush=True)
        return views[memory_id]

    rows = []
    modes = ("hashed_lexical", *(() if args.skip_provenance else (
        "provenance_projected", f"lexical_or_projected_k_le_{2 * args.k}")), *(
        ("atomic_summary", f"lexical_or_atomic_fixed_k_{args.k}",
         f"lexical_or_atomic_k_le_{2 * args.k}")
        if node_embedding_store is not None else ()))
    for question in questions:
        current = view(question.memory_id)
        refs = tuple(dict.fromkeys(
            (turn.session_id, turn.turn_index) for turn in question.gold_turns))
        fact_sets = [set(current["facts_by_turn"].get(
            current["turn_by_ref"].get(ref, ""), ())) for ref in refs]
        fact_pairs = list(combinations(fact_sets, 2))
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "stratum": question.stratum,
            "gold_pairs": len(fact_pairs),
        }
        for mode in modes:
            hits = [connected(current[mode], left, right)
                    if left and right else False for left, right in fact_pairs]
            row[mode] = {
                "gold_pair_candidate_recall": ratio(sum(hits), len(hits))
                if hits else float(bool(fact_sets and all(fact_sets))),
                "all_gold_pairs_candidates": float(all(hits))
                if hits else float(bool(fact_sets and all(fact_sets))),
            }
        rows.append(row)

    strata = sorted({row["stratum"] for row in rows})
    payload = {
        "schema_version": "graphmem-v5.13-relation-candidate-audit-v1",
        "inputs": {"db": str(args.db), "embedding_model": args.embedding_model},
        "method": {"k": args.k, "ann_pool": args.ann_pool,
                   "cross_session_quota": args.cross_session_quota,
                   "gold_used_only_for_offline_evaluation": True,
                   "llm_calls": 0},
        "vector_coverage": {**coverage, "projected_fact_ratio": ratio(
            coverage["projected_facts"], coverage["facts"]),
            "node_summary_fact_ratio": ratio(
                coverage.get("node_summary_facts", 0), coverage["facts"])},
        "overall": {mode: aggregate(rows, mode) for mode in modes},
        "per_stratum": {
            stratum: {mode: aggregate(
                [row for row in rows if row["stratum"] == stratum], mode)
                      for mode in modes}
            for stratum in strata},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_question.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "vector_coverage": payload["vector_coverage"],
                      "overall": payload["overall"]},
                     ensure_ascii=False, indent=2))
    store.close()
    if node_embedding_store is not None:
        node_embedding_store.close()


if __name__ == "__main__":
    main()
