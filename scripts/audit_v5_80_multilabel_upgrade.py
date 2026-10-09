#!/usr/bin/env python3
"""Audit a multi-label QueryIR prompt upgrade against a frozen baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False)


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--max-token-delta", type=int, default=500)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    candidate_rows = read_jsonl(args.candidate)
    baseline = {str(row["question_id"]): row for row in baseline_rows}
    candidate = {str(row["question_id"]): row for row in candidate_rows}
    if len(baseline) != len(baseline_rows):
        raise RuntimeError("baseline contains duplicate question IDs")
    if len(candidate) != len(candidate_rows):
        raise RuntimeError("candidate contains duplicate question IDs")
    if len(candidate) != args.expected or set(candidate) != set(baseline):
        raise RuntimeError(
            f"alignment failure: baseline={len(baseline)}, "
            f"candidate={len(candidate)}, expected={args.expected}")

    removed: list[dict[str, Any]] = []
    changed_payloads: list[str] = []
    changed_internal_stage_hashes: list[str] = []
    evidence_added_questions = 0
    evidence_added_turns = 0
    token_deltas: list[int] = []
    hash_failures: list[str] = []
    runtime_policy_failures: list[str] = []
    for question_id in baseline:
        old = baseline[question_id]
        new = candidate[question_id]
        old_evidence = set(map(str, old.get("evidence_turn_ids", ())))
        new_evidence = set(map(str, new.get("evidence_turn_ids", ())))
        missing = sorted(old_evidence - new_evidence)
        if missing:
            removed.append({"question_id": question_id,
                            "removed_turn_ids": missing})
        additions = new_evidence - old_evidence
        evidence_added_questions += bool(additions)
        evidence_added_turns += len(additions)
        if old.get("prompt_payload_hash") != new.get("prompt_payload_hash"):
            changed_payloads.append(question_id)
        if old.get("prompt_hash") != new.get("prompt_hash"):
            changed_internal_stage_hashes.append(question_id)
        payload_hash = hashlib.sha256(
            canonical_json(new["messages"]).encode()).hexdigest()
        if payload_hash != str(new.get("prompt_payload_hash") or ""):
            hash_failures.append(question_id)
        trace = dict(new.get("trace", {})).get("unified_source_readout", {})
        if any(bool(trace.get(key)) for key in (
                "uses_gold_or_judge", "uses_answer_or_label",
                "uses_previous_answer_as_evidence")):
            runtime_policy_failures.append(question_id)
        token_deltas.append(
            int(new["packing_prompt_tokens"])
            - int(old["packing_prompt_tokens"]))

    over_budget = [
        question_id for question_id in baseline
        if (int(candidate[question_id]["packing_prompt_tokens"])
            - int(baseline[question_id]["packing_prompt_tokens"]))
        > args.max_token_delta]
    materialization_audit_path = args.candidate.parent / "audit.jsonl"
    obligation_stats: dict[str, Any] | None = None
    if materialization_audit_path.exists():
        audit_rows = read_jsonl(materialization_audit_path)
        if ({str(row["question_id"]) for row in audit_rows}
                != set(candidate)):
            raise RuntimeError("materialization audit question IDs do not match")
        routes: Counter[str] = Counter()
        compiled: Counter[str] = Counter()
        rendered: Counter[str] = Counter()
        for row in audit_rows:
            routes[str(row["query_route"])] += 1
            compiled.update(map(str, row.get("query_obligations", ())))
            rendered.update(map(
                str, row.get("rendered_query_obligations", ())))
        obligation_stats = {
            "query_routes": dict(sorted(routes.items())),
            "compiled_obligations": dict(sorted(compiled.items())),
            "rendered_obligations": dict(sorted(rendered.items())),
        }
    summary = {
        "schema_version": "graphmem-v5.80-multilabel-upgrade-audit-v1",
        "questions": len(candidate),
        "aligned": True,
        "final_prompt_payloads_changed": len(changed_payloads),
        "internal_stage_hashes_changed": len(changed_internal_stage_hashes),
        "cache_reusable_questions": len(candidate) - len(changed_payloads),
        "evidence_added_questions": evidence_added_questions,
        "evidence_added_turns": evidence_added_turns,
        "evidence_removed_questions": len(removed),
        "prompt_payload_hash_failures": len(hash_failures),
        "runtime_policy_failures": len(runtime_policy_failures),
        "obligation_stats": obligation_stats,
        "token_delta": {
            "mean": statistics.mean(token_deltas),
            "p50": nearest_rank(token_deltas, 0.50),
            "p95": nearest_rank(token_deltas, 0.95),
            "p99": nearest_rank(token_deltas, 0.99),
            "max": max(token_deltas),
            "over_limit": len(over_budget),
            "limit": args.max_token_delta,
            "unit": "packing-token/question",
            "percentile_method": "nearest-rank",
        },
        "pass": not (
            removed or hash_failures or runtime_policy_failures or over_budget),
        "failure_details": {
            "removed_evidence": removed,
            "payload_hash_question_ids": hash_failures,
            "runtime_policy_question_ids": runtime_policy_failures,
            "over_budget_question_ids": over_budget,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    printable = dict(summary)
    if not summary["pass"]:
        printable["failure_details"] = {
            key: {"count": len(values), "examples": values[:10]}
            for key, values in summary["failure_details"].items()}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
