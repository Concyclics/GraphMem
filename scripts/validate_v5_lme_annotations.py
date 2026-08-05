#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    by_id = {str(row["question_id"]): row for row in cases}
    gold = load_gold_turns(args.annotations)
    gold.validate_question_coverage(by_id)
    roles: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    second: Counter[str] = Counter()
    for annotation in gold.annotations:
        case = by_id[annotation.question_id]
        sessions = dict(zip(case["haystack_session_ids"], case["haystack_sessions"]))
        if annotation.session_id not in set(map(str, case["answer_session_ids"])):
            raise ValueError(f"annotation outside official gold session: {annotation}")
        if annotation.session_id not in sessions:
            raise ValueError(f"unknown session: {annotation}")
        turns = sessions[annotation.session_id]
        if annotation.turn_index >= len(turns):
            raise ValueError(f"unknown turn: {annotation}")
        text = str(turns[annotation.turn_index].get("content") or "")
        if annotation.span_end > len(text):
            raise ValueError(f"span exceeds source turn: {annotation}")
        roles[annotation.support_role] += 1
        confidence[annotation.confidence] += 1
        second[annotation.second_review] += 1
    print(json.dumps({
        "questions": len(gold.question_ids), "annotations": len(gold.annotations),
        "support_roles": roles, "confidence": confidence, "second_review": second,
    }, indent=2))


if __name__ == "__main__":
    main()
