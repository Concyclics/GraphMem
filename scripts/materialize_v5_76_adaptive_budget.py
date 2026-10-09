#!/usr/bin/env python3
"""Materialize a label-free two-stage budget policy over frozen artifacts.

The script consumes retrieval telemetry and candidate text only.  Gold answers
and judge verdicts are intentionally absent from the interface; evaluator-only
oracle analysis must be performed separately on the emitted candidate pool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.budget_controller import (  # noqa: E402
    BUDGET_CONTROLLER_SCHEMA_VERSION, BUDGET_POLICIES,
    decide_answer_budget,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_rows(path):
        question_id = str(row.get(key) or row.get("question_id") or "")
        if not question_id or question_id in result:
            raise RuntimeError(f"invalid question id in {path}: {question_id!r}")
        result[question_id] = row
    return result


def overlay(
    base: dict[str, dict[str, Any]], paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    result = dict(base)
    for path in paths:
        result.update(keyed(path))
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest(values: Iterable[int | float], fraction: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def stats(values: Iterable[int | float]) -> dict[str, int | float]:
    rows = list(values)
    return {
        "count": len(rows),
        "mean": sum(rows) / max(1, len(rows)),
        "p50": nearest(rows, 0.50) if rows else 0,
        "p95": nearest(rows, 0.95) if rows else 0,
        "p99": nearest(rows, 0.99) if rows else 0,
        "max": max(rows, default=0),
        "sum": sum(rows),
        "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument(
        "--expanded-candidate-overlay", type=Path, action="append", default=[])
    parser.add_argument("--base-retrieval", type=Path, required=True)
    parser.add_argument("--expanded-retrieval", type=Path, required=True)
    parser.add_argument("--base-prepared", type=Path, required=True)
    parser.add_argument("--expanded-prepared", type=Path, required=True)
    parser.add_argument(
        "--expanded-prepared-overlay", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=BUDGET_POLICIES, default="balanced")
    parser.add_argument("--base-prefix", type=int, default=4)
    parser.add_argument("--disagreement-prefix", type=int, default=4)
    parser.add_argument("--expanded-prefix", type=int, default=8)
    parser.add_argument("--base-turns", type=int, default=32)
    parser.add_argument("--medium-turns", type=int, default=64)
    parser.add_argument("--maximum-turns", type=int, default=80)
    parser.add_argument("--base-tokens", type=int, default=2200)
    parser.add_argument("--medium-tokens", type=int, default=3400)
    parser.add_argument("--maximum-tokens", type=int, default=4500)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()
    if min(args.base_prefix, args.disagreement_prefix,
           args.expanded_prefix) <= 0:
        raise ValueError("candidate prefixes must be positive")

    base_candidates = keyed(args.base_candidates)
    expanded_candidates = overlay(
        keyed(args.expanded_candidates), args.expanded_candidate_overlay)
    base_retrieval = keyed(args.base_retrieval, "dev_question_id")
    expanded_retrieval = keyed(args.expanded_retrieval, "dev_question_id")
    base_prepared = keyed(args.base_prepared)
    expanded_prepared = overlay(
        keyed(args.expanded_prepared), args.expanded_prepared_overlay)
    ids = set(base_candidates)
    for label, records in (
            ("expanded candidates", expanded_candidates),
            ("base retrieval", base_retrieval),
            ("expanded retrieval", expanded_retrieval),
            ("base prepared", base_prepared),
            ("expanded prepared", expanded_prepared)):
        if set(records) != ids:
            raise RuntimeError(
                f"{label} differs from base question set: "
                f"{len(set(records) ^ ids)} mismatched ids")
    if args.expected and len(ids) != args.expected:
        raise RuntimeError(f"expected {args.expected} questions, got {len(ids)}")

    order = [str(row["question_id"])
             for row in read_rows(args.base_candidates)]
    decisions: list[dict[str, Any]] = []
    pools: list[dict[str, Any]] = []
    selection_prepared: list[dict[str, Any]] = []
    prompt_costs: list[int] = []
    for question_id in order:
        base_choices = list(base_candidates[question_id].get("candidates", ()))
        expanded_choices = list(
            expanded_candidates[question_id].get("candidates", ()))
        if (len(base_choices) < max(args.base_prefix, args.disagreement_prefix)
                or len(expanded_choices) < args.expanded_prefix):
            raise RuntimeError(f"candidate prefix unavailable for {question_id}")
        decision = decide_answer_budget(
            base_retrieval[question_id], base_choices,
            policy=args.policy,
            disagreement_prefix=args.disagreement_prefix,
            base_turns=args.base_turns,
            medium_turns=args.medium_turns,
            maximum_turns=args.maximum_turns,
            base_tokens=args.base_tokens,
            medium_tokens=args.medium_tokens,
            maximum_tokens=args.maximum_tokens,
            base_candidates=args.base_prefix,
            expanded_candidates=(args.base_prefix + args.expanded_prefix))
        # Online, the expanded pack is first attempted under the base Token
        # tier and only then escalated if it hits that cap.  The frozen replay
        # already exposes its realized evidence size, so record the same
        # post-pack decision without pretending the base trace knew it early.
        effective_target_tokens = decision.target_tokens
        if (decision.expand
                and int(expanded_retrieval[question_id].get(
                    "evidence_tokens", 0)) > effective_target_tokens):
            effective_target_tokens = (
                args.maximum_tokens
                if decision.retrieval_severity >= 6
                else args.medium_tokens)
        selected_prepared = (
            expanded_prepared[question_id]
            if decision.expand else base_prepared[question_id])
        choices: list[dict[str, Any]] = []
        for stage, source in (
                ("base32", base_choices[:args.base_prefix]),
                ("expanded64", expanded_choices[:args.expanded_prefix]
                 if decision.expand else ())):
            for source_row in source:
                row = dict(source_row)
                row["budget_stage"] = stage
                row["family"] = f"{stage}:{row.get('family', 'direct')}"
                row["rank"] = len(choices) + 1
                choices.append(row)
        decisions.append({
            "question_id": question_id,
            "policy": decision.policy,
            "expand": decision.expand,
            "target_turns": decision.target_turns,
            "target_tokens": effective_target_tokens,
            "target_candidates": len(choices),
            "retain_first_pass": decision.retain_first_pass,
            "reasons": list(decision.reasons),
            "retrieval_severity": decision.retrieval_severity,
            "candidate_unique": decision.candidate_unique,
        })
        source = base_candidates[question_id]
        pools.append({
            "question_id": question_id,
            "conversation_id": source.get("conversation_id"),
            "category": source.get("category"),
            "memory_id": source.get("memory_id"),
            "budget_policy": decision.policy,
            "budget_expanded": decision.expand,
            "candidates": choices,
        })
        prepared = dict(selected_prepared)
        trace = dict(prepared.get("trace", {}))
        trace["adaptive_answer_budget"] = decisions[-1]
        prepared["trace"] = trace
        selection_prepared.append(prepared)
        prompt_costs.append(
            int(base_retrieval[question_id].get("prompt_tokens", 0))
            + (int(expanded_retrieval[question_id].get("prompt_tokens", 0))
               if decision.expand else 0))

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
            ("decisions.jsonl", decisions),
            ("candidates.jsonl", pools),
            ("prepared_selection.jsonl", selection_prepared)):
        (args.output_root / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8")
    reasons = Counter(reason for row in decisions for reason in row["reasons"])
    manifest = {
        "schema_version": BUDGET_CONTROLLER_SCHEMA_VERSION,
        "policy": args.policy,
        "questions": len(order),
        "selection_uses_gold_or_judge": False,
        "base_prefix": args.base_prefix,
        "disagreement_prefix": args.disagreement_prefix,
        "expanded_prefix": args.expanded_prefix,
        "expanded_questions": sum(row["expand"] for row in decisions),
        "base_only_questions": sum(not row["expand"] for row in decisions),
        "expansion_reasons": dict(sorted(reasons.items())),
        "target_turns": dict(sorted(Counter(
            str(row["target_turns"]) for row in decisions).items())),
        "target_tokens": dict(sorted(Counter(
            str(row["target_tokens"]) for row in decisions).items())),
        "candidate_pool_size": stats(
            row["target_candidates"] for row in decisions),
        "answer_prompt_tokens_two_stage_upper_bound": stats(prompt_costs),
        "inputs": {
            "base_candidates": {
                "path": str(args.base_candidates),
                "sha256": digest(args.base_candidates),
            },
            "expanded_candidates": {
                "path": str(args.expanded_candidates),
                "sha256": digest(args.expanded_candidates),
            },
            "base_retrieval": {
                "path": str(args.base_retrieval),
                "sha256": digest(args.base_retrieval),
            },
            "expanded_retrieval": {
                "path": str(args.expanded_retrieval),
                "sha256": digest(args.expanded_retrieval),
            },
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
