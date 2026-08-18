#!/usr/bin/env python3
"""Summarize paired V5.55 prompt ablations against frozen LME baselines."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    lower = min(gains, losses)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def compare(baseline: dict[str, bool], candidate: dict[str, bool],
            routes: dict[str, str]) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("baseline/candidate question IDs differ")
    by_route: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "baseline_correct": 0,
                 "candidate_correct": 0, "gains": 0, "losses": 0})
    gains = losses = 0
    for question_id, before in baseline.items():
        after = candidate[question_id]
        bucket = by_route[routes.get(question_id, "unknown")]
        bucket["questions"] += 1
        bucket["baseline_correct"] += int(before)
        bucket["candidate_correct"] += int(after)
        bucket["gains"] += int(after and not before)
        bucket["losses"] += int(before and not after)
        gains += int(after and not before)
        losses += int(before and not after)
    questions = len(baseline)
    baseline_correct = sum(baseline.values())
    candidate_correct = sum(candidate.values())
    return {
        "questions": questions,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_correct / questions,
        "candidate_accuracy": candidate_correct / questions,
        "accuracy_delta": (candidate_correct - baseline_correct) / questions,
        "gains": gains,
        "losses": losses,
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
        "by_route": dict(sorted(by_route.items())),
    }


def verdicts(path: Path, selected: set[str] | None = None) -> dict[str, bool]:
    rows = {str(row["question_id"]): bool(row["correct"])
            for row in read_jsonl(path)}
    if selected is not None:
        rows = {key: value for key, value in rows.items() if key in selected}
    return rows


def records(path: Path) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in read_jsonl(path)}


def paired_verdicts(*, baseline: dict[str, bool], candidate: dict[str, bool],
                    baseline_answers: dict[str, dict],
                    candidate_answers: dict[str, dict]) -> tuple[dict[str, bool], int]:
    """Carry the old verdict when the actual answer text is byte-identical.

    Re-judging an unchanged prediction can flip a verdict despite temperature
    zero and seed zero.  A paired ablation must not attribute that judge noise
    to a prompt intervention.
    """
    merged: dict[str, bool] = {}
    carried = 0
    for question_id, fresh in candidate.items():
        before = baseline_answers[question_id].get("prediction", "")
        after = candidate_answers[question_id].get("prediction", "")
        if before == after:
            merged[question_id] = baseline[question_id]
            carried += 1
        else:
            merged[question_id] = fresh
    return merged, carried


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qwen-baseline-judge", type=Path, required=True)
    parser.add_argument("--gpt54-baseline-judge", type=Path, required=True)
    parser.add_argument("--qwen-baseline-answers", type=Path, required=True)
    parser.add_argument("--gpt54-baseline-answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnostic = args.root / "allhit90"
    full = args.root / "lme500"
    prepared = read_jsonl(diagnostic / "selective_v1" / "prepared_answers.jsonl")
    selected = {str(row["question_id"]) for row in prepared}
    routes = {str(row["question_id"]): str(row["trace"]["answer_route"])
              for row in prepared}
    baselines = {
        "qwen_none": verdicts(args.qwen_baseline_judge, selected),
        "gpt54_none": verdicts(args.gpt54_baseline_judge, selected),
    }
    baseline_answers = {
        "qwen_none": records(args.qwen_baseline_answers),
        "gpt54_none": records(args.gpt54_baseline_answers),
    }
    summary: dict = {
        "schema_version": "graphmem-v5.55-prompt-ablation-summary-v1",
        "diagnostic_contract": {
            "retrieval": "turn-level gold available and packed all-hit",
            "model_prompt_questions": len(selected),
            "prompt_transform_reads_gold": False,
            "judge": "gpt-5.6-luna reasoning none",
        },
        "diagnostic": {},
        "full_lme500": {},
    }
    for model, baseline in baselines.items():
        model_rows = {}
        for prepared_path in sorted(diagnostic.glob("*/prepared_answers.jsonl")):
            arm = prepared_path.parent.name
            candidates = (["gpt54_none_v2", "gpt54_none"]
                          if model == "gpt54_none" else [model])
            judge = next((diagnostic / arm / directory / "judge" / "auto_eval.jsonl"
                          for directory in candidates
                          if (diagnostic / arm / directory / "judge" /
                              "auto_eval.jsonl").exists()), None)
            if judge is not None and len(read_jsonl(judge)) == len(selected):
                fresh = verdicts(judge)
                candidate_answers_path = judge.parents[1] / "answers_longmemeval.jsonl"
                paired, carried = paired_verdicts(
                    baseline=baseline, candidate=fresh,
                    baseline_answers=baseline_answers[model],
                    candidate_answers=records(candidate_answers_path))
                model_rows[arm] = compare(baseline, paired, routes)
                model_rows[arm]["carried_identical_predictions"] = carried
        summary["diagnostic"][model] = model_rows

    full_routes = {
        str(row["question_id"]): str(row.get("trace", {}).get(
            "answer_route", "unknown"))
        for row in read_jsonl(full / "selective_v1" / "prepared_answers.jsonl")}
    for model, baseline_path in (
        ("qwen_none", args.qwen_baseline_judge),
        ("gpt54_none", args.gpt54_baseline_judge),
    ):
        candidate_path = full / "selective_v1" / model / "judge" / "auto_eval.jsonl"
        if not candidate_path.exists() or len(read_jsonl(candidate_path)) != 500:
            continue
        candidate = verdicts(candidate_path)
        baseline = verdicts(baseline_path, set(candidate))
        candidate_answers = records(
            full / "selective_v1" / model / "answers_longmemeval.jsonl")
        paired, carried = paired_verdicts(
            baseline=baseline, candidate=candidate,
            baseline_answers=baseline_answers[model],
            candidate_answers=candidate_answers)
        summary["full_lme500"][model] = compare(
            baseline, paired, full_routes)
        summary["full_lme500"][model][
            "carried_identical_predictions"] = carried

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
