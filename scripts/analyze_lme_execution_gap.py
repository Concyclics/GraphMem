#!/usr/bin/env python3
"""List retained LongMemEval execution failures with ledger/gold coverage."""
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


def _index(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    args = parser.parse_args()
    answers = _index(args.run_root / "answers.jsonl", "question_id")
    retrieval = _index(args.run_root / "retrieval.jsonl", "dev_question_id")
    prepared = _index(args.run_root / "prepared_answers.jsonl", "question_id")
    judges = _index(args.judge, "question_id")
    questions = {row.question_id: row for row in load_full_questions(
        args.lme, None, load_gold_turns(args.gold), expect_locomo=None)}
    store = SQLiteGraphStore(args.source_db, read_only=True)
    rows = []; counts = Counter()
    for question_id, question in questions.items():
        if judges[question_id]["correct"]:
            continue
        nav = retrieval[question_id]
        ledger = nav.get("aggregation_ledger") or {}
        turns = {(turn.session_id, turn.turn_index): turn
                 for turn in store.turns(question.memory_id)}
        gold_turns = [turns[(ref.session_id, ref.turn_index)]
                      for ref in question.gold_turns
                      if (ref.session_id, ref.turn_index) in turns]
        gold_ids = {turn.turn_id for turn in gold_turns}
        prompt_ids = set(prepared[question_id]["evidence_turn_ids"])
        ledger_ids = set(ledger.get("candidate_turn_ids", ()))
        stage = ("unannotated" if not question.has_turn_gold else
                 "prompt_complete" if gold_ids <= prompt_ids else
                 "navigation_complete_prompt_loss" if nav["turn_all_hit"] else
                 "packing_loss")
        counts[stage] += 1
        counts[f"operation:{ledger.get('operation') or 'none'}"] += 1
        rows.append({
            "question_id": question_id,
            "question_type": answers[question_id]["question_type"],
            "stage": stage,
            "operation": ledger.get("operation"),
            "question": question.query,
            "gold_answer": answers[question_id]["gold_answer"],
            "prediction": answers[question_id]["prediction"],
            "gold_in_ledger": bool(gold_ids and gold_ids <= ledger_ids),
            "gold_turns": [{"turn_id": turn.turn_id, "text": turn.raw_text}
                           for turn in gold_turns],
        })
    store.close()
    print(json.dumps({"counts": dict(counts), "rows": rows},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
