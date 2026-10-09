#!/usr/bin/env python3
"""Summarize measured cost and optional accuracy of an adaptive answer run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    values = rows(path)
    result = {str(row["question_id"]): row for row in values}
    if len(result) != len(values):
        raise RuntimeError(f"duplicate question_id in {path}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)

    def nearest(fraction: float) -> int | float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] \
            if ordered else 0

    return {
        "count": len(ordered),
        "mean": sum(ordered) / max(1, len(ordered)),
        "p50": nearest(0.50), "p95": nearest(0.95),
        "p99": nearest(0.99), "max": max(ordered, default=0),
        "sum": sum(ordered), "percentile_method": "nearest_rank",
    }


def accuracy(path: Path | None, expected: set[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    verdicts = keyed(path)
    if set(verdicts) != expected:
        raise RuntimeError(
            f"judge coverage mismatch: expected={len(expected)}, "
            f"actual={len(verdicts)}")
    correct = sum(bool(row.get("correct")) for row in verdicts.values())
    return {
        "questions": len(verdicts), "correct": correct,
        "accuracy": correct / max(1, len(verdicts)),
        "accuracy_percent": 100.0 * correct / max(1, len(verdicts)),
        "path": str(path), "sha256": digest(path),
    }


def token_rows(question_ids: Iterable[str], base: Mapping[str, Mapping[str, Any]],
               incremental: Mapping[str, Mapping[str, Any]],
               verifier: Mapping[str, Mapping[str, Any]] | None = None,
               *, include_incremental: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("prompt", "completion", "total"):
        key = metric + "_tokens"
        values = []
        for question_id in question_ids:
            value = int(base[question_id][key])
            if include_incremental and question_id in incremental:
                value += int(incremental[question_id][key])
            if verifier is not None:
                value += int(verifier[question_id][key])
            values.append(value)
        result[metric] = stats(values)
    return result


def comparison(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("prompt", "completion", "total"):
        before = float(baseline[metric]["mean"])
        after = float(candidate[metric]["mean"])
        output[metric] = {
            "candidate_mean": after, "baseline_mean": before,
            "saving_fraction": (1.0 - after / before) if before else 0.0,
            "saving_percent": (100.0 * (1.0 - after / before)
                               if before else 0.0),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-usage", type=Path, required=True)
    parser.add_argument("--incremental-usage", type=Path, required=True)
    parser.add_argument("--fixed64-usage", type=Path, required=True)
    parser.add_argument("--verifier-usage", type=Path)
    parser.add_argument("--selection-judge", type=Path)
    parser.add_argument("--candidate-oracle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    decisions = keyed(args.decisions)
    base = keyed(args.base_usage)
    incremental = keyed(args.incremental_usage)
    fixed = keyed(args.fixed64_usage)
    verifier = keyed(args.verifier_usage) if args.verifier_usage else None
    ids = set(decisions)
    expanded = {question_id for question_id, row in decisions.items()
                if bool(row.get("expand"))}
    simple = ids - expanded
    if args.expected and len(ids) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(ids)}")
    if set(base) != ids or set(fixed) != ids:
        raise RuntimeError("base/fixed usage must cover all decision IDs")
    if set(incremental) != expanded:
        raise RuntimeError(
            "incremental usage must exactly cover expanded decision IDs")
    if verifier is not None and set(verifier) != ids:
        raise RuntimeError("verifier usage must cover all decision IDs")

    adaptive_generation = token_rows(ids, base, incremental)
    adaptive_with_verifier = (
        token_rows(ids, base, incremental, verifier)
        if verifier is not None else None)
    fixed_all = token_rows(ids, fixed, {}, include_incremental=False)
    simple_adaptive = token_rows(simple, base, {}, include_incremental=False)
    simple_fixed = token_rows(simple, fixed, {}, include_incremental=False)
    expanded_adaptive = token_rows(expanded, base, incremental)
    expanded_fixed = token_rows(expanded, fixed, {}, include_incremental=False)
    payload = {
        "schema_version": "graphmem-v5.76-adaptive-budget-summary-v1",
        "questions": len(ids), "expanded_questions": len(expanded),
        "base_only_questions": len(simple),
        "expansion_fraction": len(expanded) / max(1, len(ids)),
        "answer_generation_tokens": {
            "adaptive": adaptive_generation,
            "fixed64": fixed_all,
            "adaptive_vs_fixed64": comparison(adaptive_generation, fixed_all),
        },
        "simple_question_tokens": {
            "adaptive_base": simple_adaptive,
            "fixed64": simple_fixed,
            "adaptive_vs_fixed64": comparison(simple_adaptive, simple_fixed),
        },
        "expanded_question_tokens": {
            "adaptive_base_plus_delta": expanded_adaptive,
            "fixed64": expanded_fixed,
            "adaptive_vs_fixed64": comparison(expanded_adaptive, expanded_fixed),
        },
        "answer_generation_plus_verifier_tokens": adaptive_with_verifier,
        "selection_accuracy": accuracy(args.selection_judge, ids),
        "candidate_oracle": (
            json.loads(args.candidate_oracle.read_text(encoding="utf-8"))
            if args.candidate_oracle else None),
        "retry_count": {
            "base": sum(int(row.get("retry_count", 0)) for row in base.values()),
            "incremental": sum(int(row.get("retry_count", 0))
                               for row in incremental.values()),
        },
        "inputs": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in (
                ("decisions", args.decisions), ("base_usage", args.base_usage),
                ("incremental_usage", args.incremental_usage),
                ("fixed64_usage", args.fixed64_usage),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
