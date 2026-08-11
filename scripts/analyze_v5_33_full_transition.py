#!/usr/bin/env python3
"""Paired full-corpus retrieval/prompt transition with retained correctness."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def index(path: Path, key: str) -> dict[str, dict]:
    return {
        str(row[key]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line))
    }


def rate(counter: Counter, numerator: str, denominator: str = "questions") -> float:
    return counter[numerator] / max(1, counter[denominator])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-retrieval", type=Path, required=True)
    parser.add_argument("--baseline-prepared", type=Path, required=True)
    parser.add_argument("--candidate-retrieval", type=Path, required=True)
    parser.add_argument("--candidate-prepared", type=Path, required=True)
    parser.add_argument("--judge", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = index(args.baseline_retrieval, "dev_question_id")
    after = index(args.candidate_retrieval, "dev_question_id")
    before_prompts = index(args.baseline_prepared, "question_id")
    after_prompts = index(args.candidate_prepared, "question_id")
    judges = {}
    for path in args.judge:
        judges.update(index(path, "question_id"))
    expected = set(before)
    if not (set(after) == set(before_prompts) == set(after_prompts) == expected):
        raise ValueError("paired question-id sets differ")

    counters: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    for question_id in sorted(expected):
        old = before[question_id]
        new = after[question_id]
        old_prompt = before_prompts[question_id]
        new_prompt = after_prompts[question_id]
        benchmark = str(new["benchmark"])
        stratum = str(new["stratum"])
        buckets = (counters["all"], counters[benchmark], counters[stratum])
        has_gold = bool(new.get("has_turn_gold", True))
        old_hit = bool(old.get("turn_all_hit"))
        new_hit = bool(new.get("turn_all_hit"))
        old_hits = int(old.get("turn_hits", 0))
        new_hits = int(new.get("turn_hits", 0))
        old_ids = tuple(old_prompt.get("evidence_turn_ids", ()))
        new_ids = tuple(new_prompt.get("evidence_turn_ids", ()))
        evidence_changed = old_ids != new_ids
        prompt_changed = (old_prompt.get("prompt_payload_hash")
                          != new_prompt.get("prompt_payload_hash"))
        old_correct = (bool(judges[question_id]["correct"])
                       if question_id in judges else None)
        state_traversals = int((new.get("traversed_relation_signals") or {}).get(
            "state_compatible", 0))
        for counter in buckets:
            counter["questions"] += 1
            counter["has_turn_gold"] += has_gold
            counter["evidence_changed"] += evidence_changed
            counter["prompt_changed"] += prompt_changed
            counter["state_traversed_questions"] += state_traversals > 0
            counter["state_traversals"] += state_traversals
            counter["packed_turn_delta"] += (
                int(new.get("packed_turns", 0)) - int(old.get("packed_turns", 0)))
            if has_gold:
                counter["old_all_hit"] += old_hit
                counter["new_all_hit"] += new_hit
                counter["all_hit_gain"] += new_hit and not old_hit
                counter["all_hit_loss"] += old_hit and not new_hit
                counter["gold_hit_delta"] += new_hits - old_hits
                counter["old_wrong_all_hit_gain"] += bool(
                    old_correct is False and new_hit and not old_hit)
                counter["old_correct_all_hit_loss"] += bool(
                    old_correct is True and old_hit and not new_hit)
        transition = (
            "all_hit_gain" if has_gold and new_hit and not old_hit else
            "all_hit_loss" if has_gold and old_hit and not new_hit else
            "gold_hit_gain" if has_gold and new_hits > old_hits else
            "gold_hit_loss" if has_gold and new_hits < old_hits else "")
        if transition and len(examples[transition]) < 200:
            examples[transition].append({
                "question_id": question_id,
                "benchmark": benchmark,
                "stratum": stratum,
                "old_correct": old_correct,
                "old_turn_hits": old_hits,
                "new_turn_hits": new_hits,
                "old_turn_all_hit": old_hit,
                "new_turn_all_hit": new_hit,
                "state_traversals": state_traversals,
                "evidence_changed": evidence_changed,
                "prompt_changed": prompt_changed,
            })

    rendered = {}
    for bucket, counter in sorted(counters.items()):
        row = dict(counter)
        row["old_all_hit_rate"] = rate(counter, "old_all_hit", "has_turn_gold")
        row["new_all_hit_rate"] = rate(counter, "new_all_hit", "has_turn_gold")
        row["evidence_change_rate"] = rate(counter, "evidence_changed")
        row["prompt_change_rate"] = rate(counter, "prompt_changed")
        row["mean_state_traversals"] = rate(counter, "state_traversals")
        rendered[bucket] = row
    payload = {
        "schema_version": "graphmem-v5.33-full-transition-v1",
        "questions": len(expected),
        "judge_questions": len(judges),
        "buckets": rendered,
        "examples": dict(examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(expected), "buckets": rendered}, indent=2))


if __name__ == "__main__":
    main()
