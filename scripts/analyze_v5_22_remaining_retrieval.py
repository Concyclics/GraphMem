#!/usr/bin/env python3
"""Materialize remaining full-corpus misses with gold candidate diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def _index(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path, required=True)
    parser.add_argument("--judge-locomo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    judges = _index(args.judge_lme, "question_id")
    judges.update(_index(args.judge_locomo, "question_id"))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    output = []
    cached_memory = ""; turns = {}; by_position = {}
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); question_id = str(row["question_id"])
            if judges[question_id]["correct"] or row["metrics"]["turn_all_hit"]:
                continue
            question = questions[question_id]
            if row["memory_id"] != cached_memory:
                cached_memory = row["memory_id"]
                memory_turns = store.turns(cached_memory)
                turns = {turn.turn_id: turn for turn in memory_turns}
                by_position = {(turn.session_id, turn.turn_index): turn.turn_id
                               for turn in memory_turns}
            candidate_by_id = {item["turn_id"]: item
                               for item in row["candidate_scores"]}
            packed = set(row["retrieved_turn_ids"])
            gold_rows = []
            for ref in question.gold_turns:
                turn_id = by_position.get((ref.session_id, ref.turn_index))
                if turn_id is None:
                    continue
                turn = turns[turn_id]; candidate = candidate_by_id.get(turn_id, {})
                previous_id = by_position.get((turn.session_id, turn.turn_index - 1))
                following_id = by_position.get((turn.session_id, turn.turn_index + 1))
                gold_rows.append({
                    "turn_id": turn_id, "packed": turn_id in packed,
                    "rank": candidate.get("rank"),
                    "mandatory": candidate.get("mandatory"),
                    "fused_score": candidate.get("fused_score"),
                    "binding_score": candidate.get("binding_score"),
                    "channels": {key: candidate.get(key) for key in (
                        "exact_score", "bm25_score", "dense_score",
                        "graph_score", "adjacency_score")},
                    "text": turn.raw_text,
                    "previous": turns[previous_id].raw_text if previous_id else "",
                    "following": turns[following_id].raw_text if following_id else "",
                })
            output.append({
                "question_id": question_id, "stratum": row["stratum"],
                "query": question.query,
                "candidate_count": len(row["candidate_scores"]),
                "gold": gold_rows,
                "operator": row["trace"].get("query_operator"),
                "ast": row["trace"].get("operator_ast"),
                "obligations": row["trace"].get("ast_obligations"),
                "fact_reservoir": row["trace"].get("fact_reservoir"),
                "binding_reasons": row["trace"].get("binding_reasons"),
                "traversal_relations": row["trace"].get("relation_counts"),
            })
    store.close()
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
