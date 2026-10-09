#!/usr/bin/env python3
"""Evaluator-only ceiling for a label-free V5.76 budget manifest.

This audit is deliberately separate from the online controller.  It reads
judge verdicts only to measure whether any already-emitted candidate was
correct; it never produces a deployable selected answer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in (
        json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines() if line.strip())}


def prefix_correct(root: Path, count: int) -> set[str]:
    correct: set[str] = set()
    for index in range(1, count + 1):
        rows = keyed(root / f"candidate_{index}" / "auto_eval.jsonl")
        correct.update(question_id for question_id, row in rows.items()
                       if bool(row["correct"]))
    return correct


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--base-judge-root", type=Path, required=True)
    parser.add_argument("--expanded-direct-judge-root", type=Path, required=True)
    parser.add_argument("--expanded-structured-judge-root", type=Path)
    parser.add_argument("--overlay-candidates", type=Path)
    parser.add_argument("--overlay-structured-judge-root", type=Path)
    parser.add_argument("--base-prefix", type=int, default=4)
    parser.add_argument("--expanded-direct-prefix", type=int, default=4)
    parser.add_argument("--expanded-structured-prefix", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    pools = keyed(args.candidate_pool)
    decisions = keyed(args.decisions)
    overlay_ids = (set(keyed(args.overlay_candidates))
                   if args.overlay_candidates is not None else set())
    if set(pools) != set(decisions):
        raise RuntimeError("candidate pool and decision ids differ")
    if args.expected and len(pools) != args.expected:
        raise RuntimeError(f"expected {args.expected}, got {len(pools)}")

    base = prefix_correct(args.base_judge_root, args.base_prefix)
    expanded_direct = prefix_correct(
        args.expanded_direct_judge_root, args.expanded_direct_prefix)
    if args.expanded_structured_prefix and (
            args.expanded_structured_judge_root is None):
        raise ValueError(
            "expanded structured root required when its prefix is positive")
    if overlay_ids and args.overlay_structured_judge_root is None:
        raise ValueError("overlay structured judge root is required")
    expanded_structured = (
        prefix_correct(args.expanded_structured_judge_root,
                       args.expanded_structured_prefix)
        if args.expanded_structured_prefix else set())
    overlay_structured = (
        prefix_correct(args.overlay_structured_judge_root,
                       args.expanded_structured_prefix)
        if overlay_ids and args.expanded_structured_prefix else set())
    expanded = ((expanded_direct | expanded_structured) - overlay_ids) | (
        (expanded_direct & overlay_ids) | overlay_structured)

    oracle: set[str] = set()
    for question_id, decision in decisions.items():
        if question_id in base:
            oracle.add(question_id)
        if bool(decision["expand"]) and question_id in expanded:
            oracle.add(question_id)
    expanded_ids = {question_id for question_id, row in decisions.items()
                    if bool(row["expand"])}
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row.get("category")) for row in pools.values()}):
        ids = {question_id for question_id, row in pools.items()
               if str(row.get("category")) == category}
        by_category[category] = {
            "questions": len(ids), "correct": len(ids & oracle),
            "accuracy": len(ids & oracle) / max(1, len(ids)),
        }
    payload = {
        "schema_version": "graphmem-v5.76-budget-oracle-audit-v1",
        "warning": (
            "Evaluator-only candidate-availability ceiling. Judge verdicts "
            "are not visible to the runtime budget controller or selector."),
        "questions": len(pools),
        "correct": len(oracle),
        "accuracy": len(oracle) / len(pools),
        "accuracy_percent": 100.0 * len(oracle) / len(pools),
        "base_prefix_correct": len(base),
        "expanded_questions": len(expanded_ids),
        "expanded_candidate_correct": len(expanded & expanded_ids),
        "rescued_by_expansion": len((expanded - base) & expanded_ids),
        "by_category": by_category,
        "expansion_reasons": dict(sorted(Counter(
            reason for row in decisions.values()
            for reason in row.get("reasons", ())).items())),
        "inputs": {
            "candidate_pool": {
                "path": str(args.candidate_pool),
                "sha256": digest(args.candidate_pool),
            },
            "decisions": {
                "path": str(args.decisions),
                "sha256": digest(args.decisions),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
