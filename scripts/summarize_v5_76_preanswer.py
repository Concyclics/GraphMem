#!/usr/bin/env python3
"""Summarize the label-free pre-answer adaptive-budget operating point."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").split("\n") if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    values = rows(path)
    result = {str(row[key]): row for row in values}
    if len(result) != len(values):
        raise RuntimeError(f"duplicate {key} in {path}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(values)

    def nearest(fraction: float) -> int | float:
        if not ordered:
            return 0
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / max(1, len(ordered)),
        "p50": nearest(.50),
        "p95": nearest(.95),
        "p99": nearest(.99),
        "max": max(ordered, default=0),
        "sum": sum(ordered),
        "percentile_method": "nearest_rank",
    }


def token_bundle(
    question_ids: Iterable[str], usage: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = list(question_ids)
    return {
        name: stat(int(usage[question_id][name + "_tokens"])
                   for question_id in selected)
        for name in ("prompt", "completion", "total")
    }


def savings(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: 100.0 * (
            1.0 - float(candidate[name]["mean"]) / float(baseline[name]["mean"])
        )
        for name in ("prompt", "completion", "total")
    }


def prefix_sets(root: Path, count: int) -> list[set[str]]:
    correct: set[str] = set()
    result = []
    for index in range(1, count + 1):
        path = root / f"candidate_{index}" / "auto_eval.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing prefix verdicts: {path}")
        correct.update(
            str(row["question_id"]) for row in rows(path)
            if bool(row.get("correct")))
        result.append(set(correct))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--fixed64-usage", type=Path, required=True)
    parser.add_argument("--fixed64-judge-root", type=Path, required=True)
    parser.add_argument("--base32-usage", type=Path)
    parser.add_argument("--base32-judge-root", type=Path)
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    retrieval = keyed(args.retrieval, "dev_question_id")
    candidates = keyed(args.candidates)
    usage = keyed(args.usage)
    fixed_usage = keyed(args.fixed64_usage)
    if bool(args.base32_usage) != bool(args.base32_judge_root):
        raise ValueError(
            "--base32-usage and --base32-judge-root must be supplied together")
    base_usage = keyed(args.base32_usage) if args.base32_usage else None
    ids = set(retrieval)
    if args.expected and len(ids) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(ids)}")
    aligned = [candidates, usage, fixed_usage]
    if base_usage is not None:
        aligned.append(base_usage)
    if any(set(value) != ids for value in aligned):
        raise RuntimeError("retrieval, candidates, and usage question sets differ")

    adaptive_prefix = prefix_sets(args.judge_root, args.prefix)
    fixed_prefix = prefix_sets(args.fixed64_judge_root, args.prefix)
    base_prefix = (
        prefix_sets(args.base32_judge_root, args.prefix)
        if args.base32_judge_root else None)
    judge_sets = adaptive_prefix + fixed_prefix
    if base_prefix is not None:
        judge_sets += base_prefix
    if any(not value <= ids for value in judge_sets):
        raise RuntimeError("judge result contains an unexpected question ID")

    simple = {
        question_id for question_id, row in retrieval.items()
        if not bool(row.get("adaptive_recall_active"))
    }
    expanded = ids - simple
    annotated = {
        question_id for question_id, row in retrieval.items()
        if bool(row.get("has_turn_gold")) and not bool(row.get("is_abstention"))
    }

    answer_tokens = token_bundle(ids, usage)
    fixed_tokens = token_bundle(ids, fixed_usage)
    base_tokens = token_bundle(ids, base_usage) if base_usage is not None else None
    simple_tokens = token_bundle(simple, usage)
    simple_fixed_tokens = token_bundle(simple, fixed_usage)

    prefix_curve = []
    for index, (adaptive, fixed) in enumerate(
            zip(adaptive_prefix, fixed_prefix, strict=True), start=1):
        point = {
            "prefix": index,
            "adaptive_correct": len(adaptive),
            "adaptive_accuracy_percent": 100.0 * len(adaptive) / len(ids),
            "fixed64_correct": len(fixed),
            "fixed64_accuracy_percent": 100.0 * len(fixed) / len(ids),
            "delta_pp": 100.0 * (len(adaptive) - len(fixed)) / len(ids),
        }
        if base_prefix is not None:
            base = base_prefix[index - 1]
            point.update({
                "base32_correct": len(base),
                "base32_accuracy_percent": 100.0 * len(base) / len(ids),
                "adaptive_vs_base32_pp": (
                    100.0 * (len(adaptive) - len(base)) / len(ids)),
            })
        prefix_curve.append(point)

    first = adaptive_prefix[0]
    oracle = adaptive_prefix[-1]
    fixed_first = fixed_prefix[0]
    base_first = base_prefix[0] if base_prefix is not None else None
    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in candidates.values()}):
        category_ids = {
            question_id for question_id, row in candidates.items()
            if str(row["category"]) == category
        }
        by_category[category] = {
            "questions": len(category_ids),
            "candidate_1_accuracy_percent": (
                100.0 * len(category_ids & first) / len(category_ids)),
            "candidate_oracle_accuracy_percent": (
                100.0 * len(category_ids & oracle) / len(category_ids)),
        }

    input_paths = [
        ("retrieval", args.retrieval),
        ("candidates", args.candidates),
        ("usage", args.usage),
        ("fixed64_usage", args.fixed64_usage),
    ]
    if args.base32_usage:
        input_paths.append(("base32_usage", args.base32_usage))

    payload = {
        "schema_version": "graphmem-v5.76-preanswer-summary-v1",
        "interpretation": (
            "QueryIR/evidence-certificate routing is label-free and runs before "
            "the only answer request. Candidate oracle is evaluator-only."),
        "questions": len(ids),
        "routing": {
            "base_only": len(simple),
            "expanded": len(expanded),
            "target_turns": dict(sorted(Counter(
                str(row.get("adaptive_recall_target_turns"))
                for row in retrieval.values()).items())),
            "target_tokens": dict(sorted(Counter(
                str(row.get("adaptive_recall_target_tokens"))
                for row in retrieval.values()).items())),
            "routes": dict(sorted(Counter(
                str(row.get("adaptive_recall_route") or "unknown")
                for row in retrieval.values()).items())),
            "reasons": dict(sorted(Counter(
                reason for row in retrieval.values()
                for reason in row.get("adaptive_recall_reasons", [])).items())),
            "single_answer_call_for_every_question": True,
        },
        "retrieval": {
            "packed_turns": stat(int(row["packed_turns"])
                                 for row in retrieval.values()),
            "evidence_tokens": stat(int(row["evidence_tokens"])
                                    for row in retrieval.values()),
            "packing_prompt_tokens": stat(int(row["prompt_tokens"])
                                          for row in retrieval.values()),
            "annotated_questions": len(annotated),
            "turn_any_hit_percent": 100.0 * sum(bool(
                retrieval[item].get("turn_any_hit")) for item in annotated)
                / max(1, len(annotated)),
            "turn_all_hit_percent": 100.0 * sum(bool(
                retrieval[item].get("turn_all_hit")) for item in annotated)
                / max(1, len(annotated)),
            "turn_recall_percent": 100.0 * sum(float(
                retrieval[item].get("turn_recall", 0.0)) for item in annotated)
                / max(1, len(annotated)),
            "turn_precision_percent": 100.0 * sum(float(
                retrieval[item].get("turn_precision", 0.0)) for item in annotated)
                / max(1, len(annotated)),
        },
        "accuracy": {
            "candidate_1": {
                "correct": len(first),
                "accuracy_percent": 100.0 * len(first) / len(ids),
                "fixed64_correct": len(fixed_first),
                "fixed64_accuracy_percent": 100.0 * len(fixed_first) / len(ids),
                **({
                    "base32_correct": len(base_first),
                    "base32_accuracy_percent": (
                        100.0 * len(base_first) / len(ids)),
                } if base_first is not None else {}),
            },
            "candidate_oracle": {
                "warning": "Evaluator-only availability ceiling; not a selector.",
                "prefix": args.prefix,
                "correct": len(oracle),
                "accuracy_percent": 100.0 * len(oracle) / len(ids),
                "fixed64_correct": len(fixed_prefix[-1]),
                "fixed64_accuracy_percent": (
                    100.0 * len(fixed_prefix[-1]) / len(ids)),
                **({
                    "base32_correct": len(base_prefix[-1]),
                    "base32_accuracy_percent": (
                        100.0 * len(base_prefix[-1]) / len(ids)),
                } if base_prefix is not None else {}),
            },
            "prefix_curve": prefix_curve,
            "by_category": by_category,
            "candidate_1_transitions_vs_fixed64": {
                "both_correct": len(first & fixed_first),
                "adaptive_only": len(first - fixed_first),
                "fixed64_only": len(fixed_first - first),
                "both_wrong": len(ids - (first | fixed_first)),
            },
        },
        "answer_tokens": {
            "adaptive": answer_tokens,
            "fixed64": fixed_tokens,
            "saving_percent": savings(answer_tokens, fixed_tokens),
            **({
                "base32": base_tokens,
                "adaptive_vs_base32_increase_percent": {
                    name: 100.0 * (
                        float(answer_tokens[name]["mean"])
                        / float(base_tokens[name]["mean"]) - 1.0)
                    for name in ("prompt", "completion", "total")
                },
            } if base_tokens is not None else {}),
        },
        "simple_question_tokens": {
            "questions": len(simple),
            "adaptive": simple_tokens,
            "fixed64": simple_fixed_tokens,
            "saving_percent": savings(simple_tokens, simple_fixed_tokens),
        },
        "retry_count": sum(int(row.get("retry_count", 0))
                           for row in usage.values()),
        "inputs": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in input_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
