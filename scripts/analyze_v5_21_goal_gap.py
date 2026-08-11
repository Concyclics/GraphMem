#!/usr/bin/env python3
"""Attribute the retained V5.21 full-corpus errors to the pipeline stage.

Only questions with explicit turn-level gold receive a retrieval-stage label.
This avoids treating LongMemEval's unannotated rows as vacuous all-hit cases.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _index(path: Path, key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in _rows(path)}


def _nearest(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _nearest(values, 0.50),
        "p95": _nearest(values, 0.95),
        "max": max(values) if values else None,
    }


def _stage(row: dict) -> str:
    if not row.get("has_turn_gold"):
        return "unannotated"
    candidate = float(row.get("candidate_turn_recall") or 0.0)
    packed = float(row.get("turn_recall") or 0.0)
    if candidate < 1.0:
        return "candidate_missing_all" if candidate == 0.0 else "candidate_missing_partial"
    if packed < 1.0:
        return "packing_zero" if packed == 0.0 else "packing_partial"
    return "complete_evidence_answer_error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judge-lme", type=Path)
    parser.add_argument("--judge-locomo", type=Path)
    args = parser.parse_args()

    answers = _index(args.run_root / "answers.jsonl", "question_id")
    retrieval = _index(args.run_root / "retrieval.jsonl", "dev_question_id")
    judges: dict[str, dict] = {}
    judge_paths = {
        "lme": (args.judge_lme or args.run_root
                / "paired_vs_m4_judge_lme" / "auto_eval.jsonl"),
        "locomo": (args.judge_locomo or args.run_root
                   / "paired_vs_m4_judge_locomo" / "auto_eval.jsonl"),
    }
    for path in judge_paths.values():
        judges.update(_index(path, "question_id"))

    report: dict = {"run_root": str(args.run_root), "benchmarks": {}}
    for benchmark in ("longmemeval", "locomo"):
        ids = [question_id for question_id, answer in answers.items()
               if answer["benchmark"] == benchmark]
        wrong_ids = [question_id for question_id in ids
                     if not judges[question_id]["correct"]]
        by_type: dict[str, list[str]] = defaultdict(list)
        by_stage: dict[str, list[str]] = defaultdict(list)
        for question_id in wrong_ids:
            answer = answers[question_id]
            row = retrieval[question_id]
            by_type[str(answer.get("question_type") or answer.get("stratum"))].append(
                question_id)
            by_stage[_stage(row)].append(question_id)

        annotated_wrong = [question_id for question_id in wrong_ids
                           if retrieval[question_id].get("has_turn_gold")]
        report["benchmarks"][benchmark] = {
            "questions": len(ids),
            "correct": len(ids) - len(wrong_ids),
            "accuracy": (len(ids) - len(wrong_ids)) / len(ids),
            "needed_for_target": max(
                0,
                math.ceil((0.75 if benchmark == "longmemeval" else 0.85) * len(ids))
                - (len(ids) - len(wrong_ids))),
            "wrong": len(wrong_ids),
            "annotated_wrong": len(annotated_wrong),
            "wrong_by_type": {
                key: len(value) for key, value in sorted(by_type.items())},
            "wrong_by_stage": {
                key: len(value) for key, value in sorted(by_stage.items())},
            "annotated_wrong_metrics": {
                "candidate_recall": _summary([
                    float(retrieval[qid].get("candidate_turn_recall") or 0.0)
                    for qid in annotated_wrong]),
                "packed_recall": _summary([
                    float(retrieval[qid].get("turn_recall") or 0.0)
                    for qid in annotated_wrong]),
                "candidate_average_precision": _summary([
                    float(retrieval[qid].get("candidate_average_precision") or 0.0)
                    for qid in annotated_wrong]),
                "candidate_last_gold_rank": _summary([
                    float(retrieval[qid]["candidate_last_gold_rank"])
                    for qid in annotated_wrong
                    if retrieval[qid].get("candidate_last_gold_rank") is not None]),
            },
            "stage_by_type": {
                key: dict(Counter(_stage(retrieval[qid]) for qid in value))
                for key, value in sorted(by_type.items())},
            "examples": {
                stage: [{
                    "question_id": qid,
                    "question_type": answers[qid].get("question_type"),
                    "question": answers[qid].get("question"),
                    "gold_answer": answers[qid].get("gold_answer"),
                    "prediction": answers[qid].get("prediction"),
                    "candidate_recall": retrieval[qid].get("candidate_turn_recall"),
                    "packed_recall": retrieval[qid].get("turn_recall"),
                    "last_gold_rank": retrieval[qid].get("candidate_last_gold_rank"),
                    "candidate_ap": retrieval[qid].get("candidate_average_precision"),
                    "judge_reasoning": judges[qid].get("reasoning"),
                } for qid in stage_ids[:20]]
                for stage, stage_ids in sorted(by_stage.items())
            },
        }

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
