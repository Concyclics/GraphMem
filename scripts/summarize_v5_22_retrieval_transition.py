#!/usr/bin/env python3
"""Join the full V5.22 retrieval gate to retained answer correctness."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _index(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-retrieval", type=Path, required=True)
    parser.add_argument("--new-candidates", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path, required=True)
    parser.add_argument("--judge-locomo", type=Path, required=True)
    args = parser.parse_args()
    old = _index(args.old_retrieval, "dev_question_id")
    judges = {}
    judges.update(_index(args.judge_lme, "question_id"))
    judges.update(_index(args.judge_locomo, "question_id"))
    totals: dict[str, Counter] = defaultdict(Counter)
    strata: dict[str, Counter] = defaultdict(Counter)
    with args.new_candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); question_id = str(row["question_id"])
            if not row["has_turn_gold"]:
                continue
            before = old[question_id]; after = row["metrics"]
            benchmark = str(row["benchmark"]); stratum = str(row["stratum"])
            correct = bool(judges[question_id]["correct"])
            old_hit = bool(before["turn_all_hit"]); new_hit = bool(after["turn_all_hit"])
            for counter in (totals[benchmark], strata[stratum]):
                counter["questions"] += 1
                counter["old_all_hit"] += old_hit
                counter["new_all_hit"] += new_hit
                counter["all_hit_gain"] += new_hit and not old_hit
                counter["all_hit_loss"] += old_hit and not new_hit
                counter["old_wrong"] += not correct
                counter["old_wrong_new_all_hit"] += (not correct and new_hit)
                counter["old_wrong_all_hit_gain"] += (
                    not correct and new_hit and not old_hit)
                counter["old_correct_all_hit_loss"] += (
                    correct and old_hit and not new_hit)
    print(json.dumps({
        "benchmarks": {key: dict(value) for key, value in totals.items()},
        "strata": {key: dict(value) for key, value in strata.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
