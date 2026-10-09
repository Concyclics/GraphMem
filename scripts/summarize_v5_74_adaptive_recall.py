#!/usr/bin/env python3
"""Compose unchanged V5.73 results with V5.74 changed-prompt verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


def load(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row.get(key) or row.get("question_id") or "")
        if not question_id or question_id in rows:
            raise RuntimeError(f"invalid question id in {path}: {question_id!r}")
        rows[question_id] = row
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def mcnemar_exact(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(rescued, regressed) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path, required=True)
    parser.add_argument("--adaptive-answers", type=Path, required=True)
    parser.add_argument("--adaptive-judge", type=Path, required=True)
    parser.add_argument(
        "--baseline-rejudge", type=Path,
        help=("optional same-period verdicts for the triggered baseline "
              "answers; identical predictions then share one verdict"))
    parser.add_argument("--baseline-retrieval", type=Path, required=True)
    parser.add_argument("--adaptive-retrieval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    baseline_answers = load(args.baseline_answers)
    baseline_judge = load(args.baseline_judge)
    adaptive_answers = load(args.adaptive_answers)
    adaptive_judge = load(args.adaptive_judge)
    baseline_rejudge = (
        load(args.baseline_rejudge)
        if args.baseline_rejudge is not None else None)
    baseline_retrieval = load(args.baseline_retrieval, "dev_question_id")
    adaptive_retrieval = load(args.adaptive_retrieval, "dev_question_id")
    full_ids = set(baseline_answers)
    if any(set(rows) != full_ids for rows in (
            baseline_judge, baseline_retrieval, adaptive_retrieval)):
        raise RuntimeError("full-run question sets differ")
    if set(adaptive_answers) != set(adaptive_judge):
        raise RuntimeError("adaptive answer/judge question sets differ")
    changed = {item for item in full_ids
               if adaptive_retrieval[item].get("adaptive_recall_triggered")}
    if set(adaptive_answers) != changed:
        raise RuntimeError("adaptive answers must cover exactly triggered prompts")
    if baseline_rejudge is not None and set(baseline_rejudge) != changed:
        raise RuntimeError(
            "baseline rejudge must cover exactly triggered prompts")
    if args.expected and len(full_ids) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(full_ids)}")

    combined_answers = []
    combined_judge = []
    for item in sorted(full_ids):
        use_adaptive = item in changed
        answer = dict(adaptive_answers[item] if use_adaptive
                      else baseline_answers[item])
        verdict = dict(adaptive_judge[item] if use_adaptive
                       else baseline_judge[item])
        answer["adaptive_recall_used"] = use_adaptive
        verdict["adaptive_recall_used"] = use_adaptive
        combined_answers.append(answer)
        combined_judge.append(verdict)

    before_correct = sum(bool(row["correct"])
                         for row in baseline_judge.values())
    after_correct = sum(bool(row["correct"]) for row in combined_judge)
    rescued = sum(not bool(baseline_judge[item]["correct"])
                  and bool(adaptive_judge[item]["correct"])
                  for item in changed)
    regressed = sum(bool(baseline_judge[item]["correct"])
                    and not bool(adaptive_judge[item]["correct"])
                    for item in changed)
    prediction_equal = {
        item for item in changed
        if str(baseline_answers[item].get("prediction") or "").strip()
        == str(adaptive_answers[item].get("prediction") or "").strip()}
    prediction_changed = changed - prediction_equal

    controlled = None
    if baseline_rejudge is not None:
        controlled_rescued = sum(
            not bool(baseline_rejudge[item]["correct"])
            and bool(adaptive_judge[item]["correct"])
            for item in prediction_changed)
        controlled_regressed = sum(
            bool(baseline_rejudge[item]["correct"])
            and not bool(adaptive_judge[item]["correct"])
            for item in prediction_changed)
        old_subset_correct = sum(
            bool(baseline_judge[item]["correct"]) for item in changed)
        current_subset_correct = sum(
            bool(baseline_rejudge[item]["correct"]) for item in changed)
        controlled_baseline_correct = (
            before_correct - old_subset_correct + current_subset_correct)
        controlled_adaptive_correct = before_correct - old_subset_correct + sum(
            bool(baseline_rejudge[item]["correct"])
            if item in prediction_equal
            else bool(adaptive_judge[item]["correct"])
            for item in changed)
        controlled = {
            "prediction_equal": len(prediction_equal),
            "prediction_changed": len(prediction_changed),
            "identical_prediction_judge_disagreements": sum(
                bool(baseline_rejudge[item]["correct"])
                != bool(adaptive_judge[item]["correct"])
                for item in prediction_equal),
            "baseline_correct": controlled_baseline_correct,
            "adaptive_correct": controlled_adaptive_correct,
            "baseline_accuracy": controlled_baseline_correct / len(full_ids),
            "adaptive_accuracy": controlled_adaptive_correct / len(full_ids),
            "accuracy_delta_pp": 100.0 * (
                controlled_adaptive_correct - controlled_baseline_correct)
                / len(full_ids),
            "rescued": controlled_rescued,
            "regressed": controlled_regressed,
            "net_transitions": controlled_rescued - controlled_regressed,
            "mcnemar_exact_p": mcnemar_exact(
                controlled_rescued, controlled_regressed),
            "verdict_policy": (
                "same-period baseline/adaptive judge; identical prediction "
                "reuses baseline verdict"),
        }
    annotated = [item for item in full_ids
                 if baseline_retrieval[item].get("has_turn_gold")]

    by_category: dict[str, dict[str, Any]] = {}
    categories = sorted({str(adaptive_retrieval[item]["stratum"])
                         for item in full_ids})
    for category in categories:
        ids = [item for item in full_ids
               if str(adaptive_retrieval[item]["stratum"]) == category]
        before = sum(bool(baseline_judge[item]["correct"]) for item in ids)
        after = sum(bool((adaptive_judge[item] if item in changed
                          else baseline_judge[item])["correct"])
                    for item in ids)
        by_category[category] = {
            "questions": len(ids), "baseline_correct": before,
            "adaptive_correct": after,
            "baseline_accuracy": before / len(ids),
            "adaptive_accuracy": after / len(ids),
            "delta_pp": 100.0 * (after - before) / len(ids),
        }

    prompt_before = [int(baseline_retrieval[item]["prompt_tokens"])
                     for item in full_ids]
    prompt_after = [int(adaptive_retrieval[item]["prompt_tokens"])
                    for item in full_ids]
    input_paths = [
        ("baseline_answers", args.baseline_answers),
        ("baseline_judge", args.baseline_judge),
        ("adaptive_answers", args.adaptive_answers),
        ("adaptive_judge", args.adaptive_judge),
        ("baseline_retrieval", args.baseline_retrieval),
        ("adaptive_retrieval", args.adaptive_retrieval),
    ]
    if args.baseline_rejudge is not None:
        input_paths.append(("baseline_rejudge", args.baseline_rejudge))
    result = {
        "schema_version": "graphmem-v5.74-adaptive-result-v1",
        "questions": len(full_ids), "changed_questions": len(changed),
        "baseline_correct": before_correct,
        "adaptive_correct": after_correct,
        "baseline_accuracy": before_correct / len(full_ids),
        "adaptive_accuracy": after_correct / len(full_ids),
        "accuracy_delta_pp": 100.0 * (
            after_correct - before_correct) / len(full_ids),
        "rescued": rescued, "regressed": regressed,
        "net_transitions": rescued - regressed,
        "raw_cross_batch_mcnemar_exact_p": mcnemar_exact(
            rescued, regressed),
        "controlled_same_period": controlled,
        "by_category": by_category,
        "retrieval": {
            field: {
                "baseline": fmean(float(baseline_retrieval[item][field])
                                  for item in annotated),
                "adaptive": fmean(float(adaptive_retrieval[item][field])
                                  for item in annotated),
            }
            for field in ("turn_any_hit", "turn_all_hit", "turn_recall",
                          "turn_precision", "turn_f1")
        },
        "prompt_tokens": {
            "baseline_mean": fmean(prompt_before),
            "adaptive_mean": fmean(prompt_after),
            "mean_delta": fmean(prompt_after) - fmean(prompt_before),
            "baseline_p95": nearest(prompt_before, 0.95),
            "adaptive_p95": nearest(prompt_after, 0.95),
            "baseline_max": max(prompt_before),
            "adaptive_max": max(prompt_after),
        },
        "selection_modes": dict(sorted(Counter(
            row.get("selection_mode", "") for row in combined_answers).items())),
        "inputs": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in input_paths
        },
    }
    for values in result["retrieval"].values():
        values["delta"] = values["adaptive"] - values["baseline"]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("answers.jsonl", combined_answers),
                           ("judge.jsonl", combined_judge)):
        (args.output_root / filename).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
    (args.output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
