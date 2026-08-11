#!/usr/bin/env python3
"""Measure how often a missing gold answer is named by its preceding turn."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.text import content_terms  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path, required=True)
    parser.add_argument("--judge-locomo", type=Path, required=True)
    args = parser.parse_args()

    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    answers = {str(row["question_id"]): row for row in _rows(args.answers)}
    prepared = {str(row["question_id"]): row for row in _rows(args.prepared)}
    retrieval = {str(row["dev_question_id"]): row for row in _rows(args.retrieval)}
    judges = {str(row["question_id"]): row for path in (
        args.judge_lme, args.judge_locomo) for row in _rows(path)}
    store = SQLiteGraphStore(args.source_db, read_only=True)

    counts: Counter[str] = Counter()
    question_sets: dict[str, set[str]] = {
        "packing_loss": set(),
        "predecessor_more_relevant": set(),
        "predecessor_overlap_ge_1": set(),
        "cross_speaker_question_overlap_ge_1": set(),
    }
    examples = []
    for question_id, question in questions.items():
        if question.benchmark != "locomo" or judges[question_id]["correct"]:
            continue
        row = retrieval[question_id]
        if not row.get("has_turn_gold") or float(row.get("turn_recall") or 0.0) >= 1.0:
            continue
        question_sets["packing_loss"].add(question_id)
        turns = store.turns(question.memory_id)
        by_position = {(turn.session_id, turn.turn_index): turn for turn in turns}
        query_terms = content_terms(question.query)
        for ref in question.gold_turns:
            gold_turn = by_position.get((ref.session_id, ref.turn_index))
            if gold_turn is None:
                continue
            # Only count gold witnesses absent from the final pack.
            if gold_turn.turn_id in set(prepared[question_id].get("evidence_turn_ids", ())):
                continue
            previous = by_position.get((ref.session_id, ref.turn_index - 1))
            gold_overlap = len(query_terms & content_terms(gold_turn.raw_text))
            previous_overlap = (
                len(query_terms & content_terms(previous.raw_text)) if previous else 0)
            counts["missing_gold_turns"] += 1
            counts[f"gold_overlap_ge_{min(gold_overlap, 3)}"] += 1
            if previous:
                counts["has_predecessor"] += 1
            if previous_overlap > gold_overlap:
                counts["predecessor_more_relevant"] += 1
                question_sets["predecessor_more_relevant"].add(question_id)
            if previous_overlap >= 1:
                counts["predecessor_overlap_ge_1"] += 1
                question_sets["predecessor_overlap_ge_1"].add(question_id)
            if previous_overlap >= 2:
                counts["predecessor_overlap_ge_2"] += 1
            if (previous and "?" in previous.raw_text
                    and previous.speaker != gold_turn.speaker):
                counts["cross_speaker_question_predecessor"] += 1
                if previous_overlap >= 1:
                    counts["cross_speaker_question_overlap_ge_1"] += 1
                    question_sets["cross_speaker_question_overlap_ge_1"].add(
                        question_id)
            if previous_overlap > gold_overlap and len(examples) < 40:
                examples.append({
                    "question_id": question_id,
                    "question": question.query,
                    "gold_answer": answers[question_id].get("gold_answer"),
                    "gold_turn": gold_turn.raw_text,
                    "gold_overlap": gold_overlap,
                    "previous_turn": previous.raw_text if previous else "",
                    "previous_overlap": previous_overlap,
                    "candidate_last_gold_rank": row.get("candidate_last_gold_rank"),
                })
    store.close()
    print(json.dumps({
        "counts": dict(counts),
        "question_counts": {key: len(value) for key, value in question_sets.items()},
        "examples": examples},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
