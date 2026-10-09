#!/usr/bin/env python3
"""Audit V5.70 selection after materialization; never influence retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


VERSION = "graphmem-v5.70-verified-fusion-audit-v1"


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    data = list(rows(path))
    result = {str(row["question_id"]): row for row in data}
    if len(result) != len(data):
        raise ValueError(f"duplicate question ids in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--baseline-prepared", type=Path, required=True)
    parser.add_argument("--candidate-prepared", type=Path, required=True)
    parser.add_argument("--baseline-judge-lme", type=Path, required=True)
    parser.add_argument("--baseline-judge-locomo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-token-increase", type=int, default=500)
    args = parser.parse_args()

    questions = {
        item.question.question_id: item.question
        for item in load_full_questions(
            args.lme, args.locomo, load_gold_turns(args.gold))}
    baseline = keyed(args.baseline_prepared)
    candidate = keyed(args.candidate_prepared)
    verdicts = {
        **keyed(args.baseline_judge_lme),
        **keyed(args.baseline_judge_locomo),
    }
    expected = set(questions)
    for name, values in (("baseline", baseline), ("candidate", candidate),
                         ("verdicts", verdicts)):
        if set(values) != expected:
            raise ValueError(
                f"{name} ids differ: missing={len(expected - set(values))} "
                f"extra={len(set(values) - expected)}")

    store = SQLiteGraphStore(args.source_db, read_only=True)
    positions: dict[str, dict[tuple[str, int], str]] = {}
    valid_turns: dict[str, frozenset[str]] = {}
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    invariants = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for question_id, question in questions.items():
        if question.memory_id not in positions:
            memory_turns = tuple(store.turns(question.memory_id))
            positions[question.memory_id] = {
                (turn.session_id, turn.turn_index): turn.turn_id
                for turn in memory_turns}
            valid_turns[question.memory_id] = frozenset(
                turn.turn_id for turn in memory_turns)
        old = baseline[question_id]
        new = candidate[question_id]
        old_ids = tuple(map(str, old.get("evidence_turn_ids", ())))
        new_ids = tuple(map(str, new.get("evidence_turn_ids", ())))
        old_set, new_set = frozenset(old_ids), frozenset(new_ids)
        valid = valid_turns[question.memory_id]
        invariants["questions"] += 1
        invariants["duplicate_evidence_ids"] += len(new_ids) != len(new_set)
        invariants["unknown_evidence_ids"] += bool(new_set - valid)
        invariants["removed_baseline_turns"] += len(old_set - new_set)
        trace = new.get("trace", {})
        gate = trace.get("v5_70_verification_gate", {})
        plan = trace.get("v5_70_flat_plan", {})
        invariants["gold_or_judge_selection_flag"] += bool(
            gate.get("selection_uses_gold_or_judge"))
        if gate.get("eligible"):
            delta = int(trace.get("v5_70_prompt_token_delta", 0))
            invariants["verified_questions"] += 1
            invariants["prompt_token_cap_violations"] += (
                delta > args.max_token_increase)
            invariants["added_flat_turns"] += int(plan.get("added_flat", 0))
        else:
            invariants["frozen_questions"] += 1

        gold = {
            positions[question.memory_id][(ref.session_id, ref.turn_index)]
            for ref in question.gold_turns
            if (ref.session_id, ref.turn_index) in positions[question.memory_id]
        }
        if not gold:
            continue
        status = "correct" if bool(verdicts[question_id]["correct"]) else "wrong"
        key = f"{question.benchmark}:{status}"
        bucket = coverage[key]
        bucket["questions"] += 1
        bucket["gold_turns"] += len(gold)
        bucket["baseline_hits"] += len(gold & old_set)
        bucket["candidate_hits"] += len(gold & new_set)
        old_all, new_all = gold <= old_set, gold <= new_set
        bucket["baseline_all_hit"] += old_all
        bucket["candidate_all_hit"] += new_all
        bucket["all_hit_recoveries"] += new_all and not old_all
        bucket["all_hit_losses"] += old_all and not new_all
        if new_all and not old_all and len(examples[key]) < 20:
            examples[key].append({
                "question_id": question_id,
                "question": question.query,
                "new_gold_turns": sorted(gold - old_set),
            })

    coverage_rows = {}
    for key, bucket in coverage.items():
        values = dict(bucket)
        values.update({
            "baseline_recall": (
                bucket["baseline_hits"] / max(1, bucket["gold_turns"])),
            "candidate_recall": (
                bucket["candidate_hits"] / max(1, bucket["gold_turns"])),
            "baseline_all_hit_rate": (
                bucket["baseline_all_hit"] / max(1, bucket["questions"])),
            "candidate_all_hit_rate": (
                bucket["candidate_all_hit"] / max(1, bucket["questions"])),
        })
        coverage_rows[key] = values

    passed = not any((
        invariants["duplicate_evidence_ids"],
        invariants["unknown_evidence_ids"],
        invariants["removed_baseline_turns"],
        invariants["gold_or_judge_selection_flag"],
        invariants["prompt_token_cap_violations"],
        sum(row["all_hit_losses"] for row in coverage_rows.values()),
    ))
    result = {
        "schema_version": VERSION,
        "passed": passed,
        "invariants": dict(invariants),
        "gold_after_selection_only": coverage_rows,
        "all_hit_recovery_examples": dict(examples),
        "selection_uses_gold_per_question": False,
        "safety_mode_chosen_after_aggregate_gold_audit": True,
        "selection_was_rerun_after_gold_audit": True,
        "paths": {
            "baseline_prepared": str(args.baseline_prepared),
            "candidate_prepared": str(args.candidate_prepared),
            "source_db": str(args.source_db),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
