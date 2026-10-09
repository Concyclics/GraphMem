#!/usr/bin/env python3
"""Summarize the selected V5.73 LoCoMo answer run and its readout diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def stats(values: list[int | float]) -> dict[str, Any]:
    rows = sorted(values)
    nearest = lambda p: rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0
    return {"n": len(rows), "mean": sum(rows) / max(1, len(rows)),
            "p50": nearest(.5), "p95": nearest(.95), "p99": nearest(.99),
            "max": max(rows, default=0), "method": "nearest_rank"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument(
        "--oracle-judge-root", type=Path,
        help="optional candidate_1..candidate_8 prefix-judge directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judge = {str(row["question_id"]): row for row in read_jsonl(args.judge)}
    candidates = {str(row["question_id"]): row
                  for row in read_jsonl(args.candidates)}
    selections = {str(row["question_id"]): row
                  for row in read_jsonl(args.selections)}
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(args.retrieval)}
    ids = set(judge) & set(candidates) & set(selections) & set(retrieval)
    if len(ids) != 1540:
        raise RuntimeError(f"expected 1540 complete rows, got {len(ids)}")

    by_category: dict[int, list[bool]] = defaultdict(list)
    correct = 0
    for question_id in ids:
        label = str(judge[question_id].get("label") or "").upper()
        is_correct = label == "CORRECT"
        correct += is_correct
        by_category[int(candidates[question_id]["category"])].append(is_correct)
    witness_counts = [int(retrieval[item].get("obligation_witness_packed", 0))
                      for item in ids]
    card_tokens = [int(retrieval[item].get("typed_evidence_card_tokens", 0))
                   for item in ids]
    unique = [len({row["prediction_sha256"]
                   for row in candidates[item]["candidates"]}) for item in ids]
    payload = {
        "schema_version": "graphmem-v5.73-locomo-summary-v1",
        "questions": len(ids), "correct": correct,
        "accuracy": correct / len(ids),
        "accuracy_percent": 100.0 * correct / len(ids),
        "by_category": {
            str(category): {
                "questions": len(values), "correct": sum(values),
                "accuracy": sum(values) / len(values),
            } for category, values in sorted(by_category.items())},
        "selection_modes": dict(sorted(Counter(
            selections[item]["selection_mode"] for item in ids).items())),
        "unique_candidates": stats(unique),
        "witness_reserve_packed": stats(witness_counts),
        "evidence_card_tokens": stats(card_tokens),
        "evidence_card_routes": dict(sorted(Counter(
            str(retrieval[item].get("typed_evidence_card_route") or "none")
            for item in ids).items())),
        "event_table_questions": sum(
            str(((retrieval[item].get("aggregation_ledger") or {})
                 .get("worksheet_route") or "")) == "event_table"
            for item in ids),
    }
    if args.oracle_judge_root is not None:
        verdicts: dict[tuple[str, str], bool] = {}
        judge_requests = 0
        for index in range(1, 9):
            path = args.oracle_judge_root / f"candidate_{index}" / "auto_eval.jsonl"
            if not path.exists():
                raise RuntimeError(f"missing candidate-prefix verdicts: {path}")
            for row in read_jsonl(path):
                judge_requests += 1
                key = (str(row["question_id"]), str(row["prediction_sha256"]))
                flag = bool(row["correct"])
                if key in verdicts and verdicts[key] != flag:
                    raise RuntimeError(f"conflicting candidate verdict for {key}")
                verdicts[key] = flag

        first_correct: dict[str, int | None] = {}
        missing: list[str] = []
        for question_id in ids:
            first: int | None = None
            seen: set[str] = set()
            for index, candidate in enumerate(
                    candidates[question_id]["candidates"], 1):
                digest = str(candidate["prediction_sha256"])
                if digest in seen:
                    continue
                seen.add(digest)
                flag = verdicts.get((question_id, digest))
                if flag is None:
                    missing.append(f"{question_id}:{index}")
                    break
                if flag:
                    first = index
                    break
            first_correct[question_id] = first
        if missing:
            raise RuntimeError(
                f"missing {len(missing)} required candidate verdicts; "
                f"first={missing[0]}")

        curve = []
        for k in range(1, 9):
            value = sum(index is not None and index <= k
                        for index in first_correct.values())
            curve.append({
                "k": k, "correct": value, "questions": len(ids),
                "accuracy": value / len(ids),
            })
        oracle_correct = {
            question_id for question_id, index in first_correct.items()
            if index is not None}
        selected_correct = {
            question_id for question_id in ids
            if str(judge[question_id].get("label") or "").upper() == "CORRECT"}
        payload["candidate_oracle"] = {
            "warning": (
                "Evaluator-only upper bound: Luna labels choose whether any "
                "candidate is correct; it is not a deployable selector."),
            "prefix_curve": curve,
            "judge_requests": judge_requests,
            "candidate_1_to_selected_uplift_pp": 100.0 * (
                correct - curve[0]["correct"]) / len(ids),
            "selected_to_oracle_gap_pp": 100.0 * (
                curve[-1]["correct"] - correct) / len(ids),
            "oracle_correct_selected_wrong": len(
                oracle_correct - selected_correct),
            "selected_correct_oracle_unresolved": len(
                selected_correct - oracle_correct),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
