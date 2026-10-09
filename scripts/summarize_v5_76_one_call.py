#!/usr/bin/env python3
"""Audit the pre-answer, one-call adaptive budget operating point."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    values = rows(path)
    result = {str(row[key]): row for row in values}
    if len(result) != len(values):
        raise RuntimeError(f"duplicate {key} in {path}")
    return result


def correct_prefix(root: Path, count: int) -> set[str]:
    result: set[str] = set()
    for index in range(1, count + 1):
        path = root / f"candidate_{index}" / "auto_eval.jsonl"
        for row in rows(path):
            if bool(row.get("correct")):
                result.add(str(row["question_id"]))
    return result


def stat(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(values)

    def nearest(fraction: float) -> int | float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] \
            if ordered else 0

    return {
        "count": len(ordered), "mean": sum(ordered) / max(1, len(ordered)),
        "p50": nearest(.50), "p95": nearest(.95), "p99": nearest(.99),
        "max": max(ordered, default=0), "sum": sum(ordered),
        "percentile_method": "nearest_rank",
    }


def token_bundle(question_ids: Iterable[str], usage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = list(question_ids)
    return {name: stat(int(usage[item][name + "_tokens"])
                       for item in selected)
            for name in ("prompt", "completion", "total")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-retrieval", type=Path, required=True)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--base-usage", type=Path, required=True)
    parser.add_argument("--expanded-usage", type=Path, required=True)
    parser.add_argument("--fixed64-usage", type=Path, required=True)
    parser.add_argument("--base-judge-root", type=Path, required=True)
    parser.add_argument("--expanded-judge-root", type=Path, required=True)
    parser.add_argument("--base-prefix", type=int, default=4)
    parser.add_argument("--expanded-prefix", type=int, default=8)
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = keyed(args.decisions)
    retrieval = keyed(args.base_retrieval, "dev_question_id")
    candidates = keyed(args.base_candidates)
    base_usage = keyed(args.base_usage)
    expanded_usage = keyed(args.expanded_usage)
    fixed_usage = keyed(args.fixed64_usage)
    ids = set(decisions)
    if args.expected and len(ids) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(ids)}")
    if any(set(values) != ids for values in (
            retrieval, candidates, base_usage, fixed_usage)):
        raise RuntimeError("full input question sets differ")
    post_gate = {
        question_id for question_id, decision in decisions.items()
        if bool(decision.get("expand"))
        and not bool(retrieval[question_id].get(
            "adaptive_recall_triggered", False))}
    if set(expanded_usage) != post_gate:
        raise RuntimeError("expanded usage must exactly cover post-gate IDs")
    simple = {question_id for question_id, decision in decisions.items()
              if not bool(decision.get("expand"))}
    intrinsic = ids - simple - post_gate

    base_first = correct_prefix(args.base_judge_root, 1)
    base_oracle = correct_prefix(args.base_judge_root, args.base_prefix)
    expanded_first = correct_prefix(args.expanded_judge_root, 1)
    expanded_oracle = correct_prefix(
        args.expanded_judge_root, args.expanded_prefix)
    first_correct = (base_first - post_gate) | (expanded_first & post_gate)
    oracle_correct = (base_oracle - post_gate) | (expanded_oracle & post_gate)

    adaptive_usage: dict[str, dict[str, Any]] = {}
    for question_id in ids:
        adaptive_usage[question_id] = (
            expanded_usage[question_id]
            if question_id in post_gate else base_usage[question_id])
    adaptive_tokens = token_bundle(ids, adaptive_usage)
    fixed_tokens = token_bundle(ids, fixed_usage)
    simple_tokens = token_bundle(simple, adaptive_usage)
    simple_fixed = token_bundle(simple, fixed_usage)

    def savings(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
        return {name: 100.0 * (1.0 - float(candidate[name]["mean"])
                               / float(baseline[name]["mean"]))
                for name in ("prompt", "completion", "total")}

    by_category: dict[str, Any] = {}
    for category in sorted({str(row.get("category"))
                            for row in candidates.values()}):
        category_ids = {question_id for question_id, row in candidates.items()
                        if str(row.get("category")) == category}
        by_category[category] = {
            "questions": len(category_ids),
            "candidate_1_correct": len(category_ids & first_correct),
            "candidate_1_accuracy": len(category_ids & first_correct)
            / max(1, len(category_ids)),
            "candidate_oracle_correct": len(category_ids & oracle_correct),
            "candidate_oracle_accuracy": len(category_ids & oracle_correct)
            / max(1, len(category_ids)),
        }
    payload = {
        "schema_version": "graphmem-v5.76-one-call-summary-v1",
        "interpretation": (
            "The label-free retrieval gate chooses the final prompt before "
            "answer generation. Oracle fields are evaluator-only."),
        "questions": len(ids),
        "routing": {
            "base_only": len(simple),
            "intrinsic_retrieval_expansion": len(intrinsic),
            "post_gate_expansion": len(post_gate),
            "single_answer_call_for_every_question": True,
        },
        "candidate_1": {
            "correct": len(first_correct),
            "accuracy": len(first_correct) / max(1, len(ids)),
            "accuracy_percent": 100.0 * len(first_correct) / max(1, len(ids)),
        },
        "candidate_oracle": {
            "warning": "Evaluator-only availability ceiling; not a selector.",
            "base_prefix": args.base_prefix,
            "expanded_prefix": args.expanded_prefix,
            "correct": len(oracle_correct),
            "accuracy": len(oracle_correct) / max(1, len(ids)),
            "accuracy_percent": 100.0 * len(oracle_correct) / max(1, len(ids)),
        },
        "by_category": by_category,
        "answer_tokens": {
            "adaptive": adaptive_tokens, "fixed64": fixed_tokens,
            "saving_percent": savings(adaptive_tokens, fixed_tokens),
        },
        "simple_question_tokens": {
            "adaptive": simple_tokens, "fixed64": simple_fixed,
            "saving_percent": savings(simple_tokens, simple_fixed),
        },
        "routes": dict(sorted(Counter(
            str(retrieval[item].get("adaptive_recall_route") or "unknown")
            for item in post_gate).items())),
        "inputs": {name: {"path": str(path), "sha256": hashlib.sha256(
            path.read_bytes()).hexdigest()} for name, path in (
                ("decisions", args.decisions),
                ("base_retrieval", args.base_retrieval),
                ("base_candidates", args.base_candidates),
                ("base_usage", args.base_usage),
                ("expanded_usage", args.expanded_usage),
                ("fixed64_usage", args.fixed64_usage))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
