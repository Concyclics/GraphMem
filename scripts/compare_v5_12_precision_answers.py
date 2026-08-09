#!/usr/bin/env python3
"""Paired answer-accuracy gate for two retrieval/packing operating points."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_12/answer_dev200/baseline")
    parser.add_argument("--treatment", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_12/answer_dev200/precision_soft")
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_12/answer_dev200/paired_summary")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_run(root: Path) -> dict[str, dict[str, Any]]:
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(root / "retrieval.jsonl")}
    answers = {str(row["question_id"]): row
               for row in read_jsonl(root / "answers.jsonl")}
    correct = {}
    for benchmark, directory in (("longmemeval", "judge_lme"),
                                 ("locomo", "judge_locomo")):
        for row in read_jsonl(root / directory / "auto_eval.jsonl"):
            correct[str(row["question_id"])] = (benchmark, bool(row["correct"]))
    return {question_id: {
        "benchmark": correct[question_id][0],
        "correct": correct[question_id][1],
        "retrieval": row,
        "answer": answers.get(question_id, {}),
    } for question_id, row in retrieval.items() if question_id in correct}


def mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return fmean(values) if values else 0.0


def paired_ci(rows: list[dict[str, Any]], resamples: int = 10000) -> list[float]:
    rng = random.Random(42)
    values = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(len(rows))] for _item in rows]
        values.append(fmean(
            int(row["treatment"]["correct"])
            - int(row["baseline"]["correct"]) for row in sample))
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]


def exact_mcnemar(better: int, worse: int) -> float:
    discordant = better + worse
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(0, min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(
        f"{int(row['baseline']['correct'])}->{int(row['treatment']['correct'])}"
        for row in rows)
    fields = (
        "turn_all_hit", "turn_recall", "turn_precision", "turn_f1",
        "candidate_turn_recall", "candidate_turn_precision", "candidate_turn_f1",
        "candidate_selectivity", "candidate_top8_turn_recall",
        "candidate_top8_turn_precision", "candidate_top16_turn_recall",
        "candidate_top16_turn_precision", "candidate_top32_turn_recall",
        "candidate_top32_turn_precision", "packed_turns", "evidence_tokens",
        "prompt_tokens")
    arms = {}
    for arm in ("baseline", "treatment"):
        arms[arm] = {
            "accuracy": mean(rows, (arm, "correct")),
            **{field: mean(rows, (arm, "retrieval", field)) for field in fields},
        }
    better = transitions["0->1"]
    worse = transitions["1->0"]
    return {
        "questions": len(rows),
        "arms": arms,
        "accuracy_delta": arms["treatment"]["accuracy"] - arms["baseline"]["accuracy"],
        "accuracy_delta_ci95": paired_ci(rows),
        "transitions": dict(transitions),
        "mcnemar_exact_p": exact_mcnemar(better, worse),
    }


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline)
    treatment = load_run(args.treatment)
    common = sorted(set(baseline) & set(treatment))
    rows = [{
        "question_id": question_id,
        "benchmark": baseline[question_id]["benchmark"],
        "stratum": baseline[question_id]["retrieval"]["stratum"],
        "baseline": baseline[question_id],
        "treatment": treatment[question_id],
    } for question_id in common]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"benchmark:{row['benchmark']}"].append(row)
        groups[f"stratum:{row['stratum']}"].append(row)
    payload = {
        "schema_version": "graphmem-v5.12-precision-answer-paired-v1",
        "precision_scope": "official_gold_turns_only_lower_bound",
        "baseline": str(args.baseline),
        "treatment": str(args.treatment),
        "overall": summarize(rows),
        "groups": {name: summarize(items) for name, items in sorted(groups.items())},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "transitions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            baseline_row = row["baseline"]
            treatment_row = row["treatment"]
            left = baseline_row["retrieval"]
            right = treatment_row["retrieval"]
            handle.write(json.dumps({
                "question_id": row["question_id"],
                "benchmark": row["benchmark"],
                "stratum": row["stratum"],
                "transition": (
                    f"{int(baseline_row['correct'])}->{int(treatment_row['correct'])}"),
                "question": baseline_row["answer"].get("question", ""),
                "gold_answer": baseline_row["answer"].get("gold_answer", ""),
                "baseline_prediction": baseline_row["answer"].get("prediction", ""),
                "treatment_prediction": treatment_row["answer"].get("prediction", ""),
                "baseline": {field: left.get(field) for field in (
                    "turn_all_hit", "turn_recall", "turn_precision", "turn_f1",
                    "packed_turns", "evidence_tokens", "prompt_tokens")},
                "treatment": {field: right.get(field) for field in (
                    "turn_all_hit", "turn_recall", "turn_precision", "turn_f1",
                    "packed_turns", "evidence_tokens", "prompt_tokens")},
            }, ensure_ascii=False) + "\n")
    lines = [
        "# Precision-aware packing -> Answer Accuracy",
        "",
        "> Precision 仅相对官方 gold turn 计算，是 annotation-scoped lower bound。",
        "",
        "| Scope | N | Baseline Acc. | Treatment Acc. | Delta | 95% paired CI | 0->1 | 1->0 | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("overall", payload["overall"]), *payload["groups"].items()):
        lines.append(
            f"| {name} | {row['questions']} | {row['arms']['baseline']['accuracy']:.2%} | "
            f"{row['arms']['treatment']['accuracy']:.2%} | {row['accuracy_delta']:+.2%} | "
            f"[{row['accuracy_delta_ci95'][0]:+.2%}, {row['accuracy_delta_ci95'][1]:+.2%}] | "
            f"{row['transitions'].get('0->1', 0)} | {row['transitions'].get('1->0', 0)} | "
            f"{row['mcnemar_exact_p']:.4f} |")
    lines.extend([
        "", "## Overall retrieval and cost", "",
        "| Arm | Recall | Precision | F1 | All-hit | Pack turns | Evidence tokens | Prompt tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm, row in payload["overall"]["arms"].items():
        lines.append(
            f"| {arm} | {row['turn_recall']:.2%} | {row['turn_precision']:.2%} | "
            f"{row['turn_f1']:.2%} | {row['turn_all_hit']:.2%} | {row['packed_turns']:.1f} | "
            f"{row['evidence_tokens']:.1f} | {row['prompt_tokens']:.1f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
