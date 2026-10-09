#!/usr/bin/env python3
"""Audit a bounded semantic CanonicalFact witness lane against packed evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config
from graphmem.domain import NodeType
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.eval.fullset import load_full_questions
from graphmem.runtime import GraphReadView
from graphmem.storage import SQLiteGraphStore


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--relation-db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = load_full_questions(
        None, args.data, None, expect_lme=0, expect_locomo=1540)
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(args.retrieval)}
    failures = failure_ids(args.candidates, args.judge_root)
    store = SQLiteGraphStore(args.db, read_only=True)
    relation_store = SQLiteGraphStore(args.relation_db, read_only=True)
    embedding = QwenEmbeddingIndex(
        store, load_config(args.config), record_usage=False,
        query_cache_path=args.query_cache, model_id="BAAI/bge-m3",
        request_model_id="bge-m3", base_url=args.base_url)

    cutoffs = (1, 2, 4, 8, 16, 24, 32)
    reserves = (1, 2, 4, 8, 16)
    counts: Counter[str] = Counter()
    cache: dict[str, tuple[GraphReadView, tuple[str, ...], dict[str, tuple[str, ...]],
                           dict[tuple[str, int], str]]] = {}
    examples: list[dict[str, Any]] = []
    for item in questions:
        question = item.question
        if question.memory_id not in cache:
            version, checksum = store.graph_identity(question.memory_id)
            view = GraphReadView(
                store.nodes(question.memory_id), store.edges(question.memory_id),
                graph_version=version, graph_checksum=checksum)
            facts = tuple(node.node_id for node in view.nodes.values()
                          if node.node_type == NodeType.CANONICAL_FACT)
            fact_turns: dict[str, tuple[str, ...]] = {}
            for fact_id in facts:
                ids: list[str] = []
                for group_id in view.terminal_groups_for_nodes((fact_id,)):
                    group = store.evidence_group(group_id)
                    if group:
                        ids.extend(member.turn_id for member in group.members)
                fact_turns[fact_id] = tuple(dict.fromkeys(ids))
            coordinates = {
                (turn.session_id, turn.turn_index): turn.turn_id
                for turn in store.turns(question.memory_id)}
            cache[question.memory_id] = view, facts, fact_turns, coordinates
        view, facts, fact_turns, coordinates = cache[question.memory_id]
        gold = {coordinates[(ref.session_id, ref.turn_index)]
                for ref in question.gold_turns
                if (ref.session_id, ref.turn_index) in coordinates}
        if not gold:
            continue
        rows = tuple(embedding.search_items(
            question.memory_id, question.query, facts, max(cutoffs),
            source_store=relation_store))
        packed = set(retrieval.get(question.question_id, {}).get(
            "retrieved_turn_ids", ()))
        scope = "fixed_failure" if question.question_id in failures else "other"
        counts[f"{scope}:questions"] += 1
        counts[f"{scope}:pack_all"] += gold <= packed
        for cutoff in cutoffs:
            fact_evidence = {turn_id for fact_id, _score in rows[:cutoff]
                             for turn_id in fact_turns.get(fact_id, ())}
            counts[f"{scope}:fact_all@{cutoff}"] += gold <= fact_evidence
            counts[f"{scope}:fact_any@{cutoff}"] += bool(gold & fact_evidence)
        for reserve in reserves:
            selected = set(packed)
            additions = 0
            for fact_id, _score in rows:
                for turn_id in fact_turns.get(fact_id, ()):
                    if turn_id in selected:
                        continue
                    selected.add(turn_id)
                    additions += 1
                    if additions >= reserve:
                        break
                if additions >= reserve:
                    break
            counts[f"{scope}:union_all@{reserve}"] += gold <= selected
            counts[f"{scope}:union_any@{reserve}"] += bool(gold & selected)

            # Model a true fixed-budget reserve as well: semantic-fact source
            # turns are placed first and the original pack then fills the
            # remaining seats in its existing order.  Unlike the union upper
            # bound above, this exposes losses caused by evicting the tail.
            promoted: list[str] = []
            for fact_id, _score in rows:
                for turn_id in fact_turns.get(fact_id, ()):
                    if turn_id not in promoted and turn_id not in packed:
                        promoted.append(turn_id)
                        if len(promoted) >= reserve:
                            break
                if len(promoted) >= reserve:
                    break
            replacement = set((*promoted, *packed)[:len(packed)])
            replacement_all = gold <= replacement
            pack_all = gold <= packed
            counts[f"{scope}:replace_all@{reserve}"] += replacement_all
            counts[f"{scope}:replace_gain@{reserve}"] += (
                replacement_all and not pack_all)
            counts[f"{scope}:replace_loss@{reserve}"] += (
                pack_all and not replacement_all)
        if (question.question_id in failures and not gold <= packed
                and len(examples) < 20):
            examples.append({
                "question_id": question.question_id,
                "question": question.query,
                "gold": sorted(gold),
                "fact_hits": [{
                    "rank": rank,
                    "summary": view.nodes[fact_id].summary,
                    "score": score,
                    "gold_turn_hit": bool(gold & set(fact_turns.get(fact_id, ()))),
                } for rank, (fact_id, score) in enumerate(rows[:12], 1)],
            })

    result = {
        "schema_version": "graphmem-v5.78-semantic-fact-witness-audit-v1",
        "fixed_best_of_failures": len(failures),
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
    relation_store.close()
    store.close()


if __name__ == "__main__":
    main()
