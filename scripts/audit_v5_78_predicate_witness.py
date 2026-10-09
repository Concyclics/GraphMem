#!/usr/bin/env python3
"""Audit raw relation-query probes against build-time predicate embeddings."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config
from graphmem.domain import NodeType, stable_id
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval.fullset import load_full_questions
from graphmem.retrieval.query_ir import compile_query
from graphmem.runtime import GraphReadView
from graphmem.storage import SQLiteGraphStore
from graphmem.text import content_terms


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def failure_ids(candidates: Path, judge_root: Path) -> set[str]:
    verdicts: dict[tuple[str, str], bool] = {}
    for path in sorted(judge_root.glob("candidate_*/auto_eval.jsonl")):
        for row in read_jsonl(path):
            key = (str(row["question_id"]), str(row["prediction_sha256"]))
            verdicts[key] = verdicts.get(key, False) or bool(row["correct"])
    return {
        str(row["question_id"]) for row in read_jsonl(candidates)
        if not any(verdicts.get((str(row["question_id"]), str(candidate.get(
            "prediction_sha256") or "")), False)
                   for candidate in row.get("candidates", ()))
    }


def relation_probe(query: str, ir, turns) -> str:
    """Remove only explicit transcript principals from the relation surface."""

    query_terms = content_terms(query)
    owner_terms = frozenset(
        term for turn in turns
        if (speaker_terms := content_terms(turn.speaker))
        and not speaker_terms <= {"user", "assistant", "system", "human"}
        and speaker_terms <= query_terms
        for term in speaker_terms)
    source = ir.slots.content_terms if ir.slots is not None else tuple(query_terms)
    terms = tuple(term for term in source if term not in owner_terms)
    return " ".join(terms) or query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--query-instruction", action="store_true",
        help="reuse the normal turn-retrieval query vector instead of a raw probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = load_full_questions(
        None, args.data, None, expect_lme=0, expect_locomo=1540)
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(args.retrieval)}
    failures = failure_ids(args.candidates, args.judge_root)
    store = SQLiteGraphStore(args.db, read_only=True)
    store.enable_read_pool(max(1, args.workers))
    embedding = QwenEmbeddingIndex(
        store, load_config(args.config), record_usage=False,
        query_cache_path=args.query_cache, model_id="BAAI/bge-m3",
        request_model_id="bge-m3", base_url=args.base_url)
    index_model = "BAAI/bge-m3:predicate-v1"

    memory_cache: dict[str, dict[str, Any]] = {}
    for item in questions:
        memory_id = item.question.memory_id
        if memory_id in memory_cache:
            continue
        version, checksum = store.graph_identity(memory_id)
        view = GraphReadView(
            store.nodes(memory_id), store.edges(memory_id),
            graph_version=version, graph_checksum=checksum)
        turns = store.turns(memory_id)
        predicate_by_id: dict[str, str] = {}
        available = {
            str(row["item_id"]) for row in store._read(
                "SELECT item_id FROM embeddings WHERE memory_id=? AND model_id=?",
                (memory_id, index_model))}
        for predicate in view.predicate_index:
            item_id = stable_id("predicate", memory_id, predicate)
            if item_id in available:
                predicate_by_id[item_id] = predicate
        fact_turns: dict[str, tuple[str, ...]] = {}
        for node in view.nodes.values():
            if node.node_type != NodeType.CANONICAL_FACT:
                continue
            ids: list[str] = []
            for group_id in view.terminal_groups_for_nodes((node.node_id,)):
                group = store.evidence_group(group_id)
                if group:
                    ids.extend(member.turn_id for member in group.members)
            fact_turns[node.node_id] = tuple(dict.fromkeys(ids))
        coordinates = {
            (turn.session_id, turn.turn_index): turn.turn_id for turn in turns}
        memory_cache[memory_id] = {
            "view": view, "turns": turns,
            "predicate_by_id": predicate_by_id,
            "fact_turns": fact_turns, "coordinates": coordinates,
        }

    cutoffs = (1, 2, 4, 8, 16)
    reserves = (1, 2, 4, 8)

    def inspect(item):
        question = item.question
        cached = memory_cache[question.memory_id]
        view = cached["view"]
        ir = compile_query(question.query, view).promote_ast()
        probe = relation_probe(question.query, ir, cached["turns"])
        predicate_by_id = cached["predicate_by_id"]
        dense_query = question.query if args.query_instruction else probe
        hits = embedding.search_items(
            question.memory_id, dense_query, tuple(predicate_by_id), max(cutoffs),
            index_model_id=index_model,
            query_instruction=args.query_instruction)
        owner_ids = frozenset().union(*(
            set(view.owner_alias_index.get(alias.casefold(), ()))
            for operand in ir.operands for alias in operand.owner_aliases))
        ranked_turns: list[str] = []
        rows: list[dict[str, Any]] = []
        turns_at_cutoff: dict[int, set[str]] = {}
        for rank, (predicate_id, score) in enumerate(hits, 1):
            predicate = predicate_by_id[predicate_id]
            fact_ids = view.lookup_facts(
                owner_ids=tuple(owner_ids), predicates=(predicate,),
                limit=8, rank_terms=content_terms(question.query))
            accepted_facts: list[str] = []
            for fact_id in fact_ids:
                node = view.nodes.get(fact_id)
                if node is None:
                    continue
                fact_owner = str(node.attributes.get("owner_id", ""))
                if owner_ids and fact_owner and fact_owner not in owner_ids:
                    continue
                accepted_facts.append(fact_id)
                for turn_id in cached["fact_turns"].get(fact_id, ()):
                    if turn_id not in ranked_turns:
                        ranked_turns.append(turn_id)
            rows.append({
                "rank": rank, "predicate": predicate,
                "score": float(score), "fact_ids": accepted_facts,
            })
            if rank in cutoffs:
                turns_at_cutoff[rank] = set(ranked_turns)
        gold = {
            cached["coordinates"][(ref.session_id, ref.turn_index)]
            for ref in question.gold_turns
            if (ref.session_id, ref.turn_index) in cached["coordinates"]}
        packed = tuple(retrieval.get(question.question_id, {}).get(
            "retrieved_turn_ids", ()))
        return question, probe, rows, ranked_turns, turns_at_cutoff, gold, packed

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        inspected = list(pool.map(inspect, questions))

    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for (question, probe, rows, ranked_turns, turns_at_cutoff,
         gold, packed) in inspected:
        if not gold:
            continue
        scope = "fixed_failure" if question.question_id in failures else "other"
        counts[f"{scope}:questions"] += 1
        pack_set = set(packed)
        counts[f"{scope}:pack_all"] += gold <= pack_set
        for cutoff in cutoffs:
            evidence = turns_at_cutoff.get(cutoff, set(ranked_turns))
            counts[f"{scope}:predicate_all@{cutoff}"] += gold <= evidence
            counts[f"{scope}:predicate_any@{cutoff}"] += bool(gold & evidence)
        for reserve in reserves:
            additions = [turn_id for turn_id in ranked_turns
                         if turn_id not in pack_set][:reserve]
            union = pack_set | set(additions)
            replacement = set((*additions, *packed)[:len(packed)])
            pack_all = gold <= pack_set
            replacement_all = gold <= replacement
            counts[f"{scope}:union_all@{reserve}"] += gold <= union
            counts[f"{scope}:replace_all@{reserve}"] += replacement_all
            counts[f"{scope}:replace_gain@{reserve}"] += (
                replacement_all and not pack_all)
            counts[f"{scope}:replace_loss@{reserve}"] += (
                pack_all and not replacement_all)
        if (question.question_id in failures and not gold <= pack_set
                and len(examples) < 24):
            examples.append({
                "question_id": question.question_id,
                "question": question.query,
                "probe": probe,
                "gold": sorted(gold),
                "predicate_hits": rows[:8],
                "gold_ranked_turn_positions": [
                    rank for rank, turn_id in enumerate(ranked_turns, 1)
                    if turn_id in gold],
            })

    result = {
        "schema_version": "graphmem-v5.78-predicate-witness-audit-v1",
        "fixed_best_of_failures": len(failures),
        "query_instruction": args.query_instruction,
        "counts": dict(sorted(counts.items())),
        "examples": examples,
        "embedding_stats": dict(embedding.stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"counts": result["counts"],
                      "embedding_stats": result["embedding_stats"]}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
