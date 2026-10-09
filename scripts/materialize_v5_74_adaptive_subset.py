#!/usr/bin/env python3
"""Freeze only prompts changed by the V5.74 adaptive-recall gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, key: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = {str(row.get(key) or row.get("question_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError(f"duplicate or missing question id in {path}")
    return rows, by_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-retrieval", type=Path, required=True)
    parser.add_argument("--adaptive-retrieval", type=Path, required=True)
    parser.add_argument("--baseline-prepared", type=Path, required=True)
    parser.add_argument("--adaptive-prepared", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--direct-candidates", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    _base_rows, baseline = read_jsonl(
        args.baseline_retrieval, "dev_question_id")
    adaptive_rows, adaptive = read_jsonl(
        args.adaptive_retrieval, "dev_question_id")
    _bp_rows, baseline_prepared = read_jsonl(
        args.baseline_prepared, "question_id")
    prepared_rows, adaptive_prepared = read_jsonl(
        args.adaptive_prepared, "question_id")
    direct_rows, direct = read_jsonl(args.direct_candidates, "question_id")
    baseline_answers = None
    if args.baseline_answers is not None:
        _answer_rows, baseline_answers = read_jsonl(
            args.baseline_answers, "question_id")
    expected_ids = set(baseline)
    for name, values in (
        ("adaptive retrieval", adaptive),
        ("baseline prepared", baseline_prepared),
        ("adaptive prepared", adaptive_prepared),
        ("direct candidates", direct),
    ):
        if set(values) != expected_ids:
            raise RuntimeError(f"{name} question set differs from baseline")
    if baseline_answers is not None and set(baseline_answers) != expected_ids:
        raise RuntimeError("baseline answer question set differs from baseline")
    if args.expected and len(expected_ids) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} questions, got {len(expected_ids)}")

    changed = {
        question_id for question_id, row in adaptive.items()
        if bool(row.get("adaptive_recall_triggered"))}
    non_trigger_hash_mismatches = [
        question_id for question_id in sorted(expected_ids - changed)
        if baseline_prepared[question_id].get("prompt_payload_hash")
        != adaptive_prepared[question_id].get("prompt_payload_hash")]
    trigger_hash_unchanged = [
        question_id for question_id in sorted(changed)
        if baseline_prepared[question_id].get("prompt_payload_hash")
        == adaptive_prepared[question_id].get("prompt_payload_hash")]
    if non_trigger_hash_mismatches or trigger_hash_unchanged:
        raise RuntimeError(
            "adaptive prompt identity contract failed: "
            f"{len(non_trigger_hash_mismatches)} non-trigger mismatches, "
            f"{len(trigger_hash_unchanged)} unchanged triggers")

    order = [str(row["question_id"]) for row in prepared_rows
             if str(row["question_id"]) in changed]
    locomo = json.loads(args.locomo_data.read_text(encoding="utf-8"))
    locomo_subset = [row for row in locomo
                     if str(row["question_id"]) in changed]
    if {str(row["question_id"]) for row in locomo_subset} != changed:
        raise RuntimeError("LoCoMo source does not cover every changed prompt")

    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_root / "prepared_answers.jsonl"
    direct_path = args.output_root / "direct_candidates.jsonl"
    baseline_answer_path = args.output_root / "baseline_answers.jsonl"
    locomo_path = args.output_root / "locomo.json"
    write_jsonl(prepared_path, [adaptive_prepared[item] for item in order])
    write_jsonl(direct_path, [direct[item] for item in order])
    if baseline_answers is not None:
        write_jsonl(
            baseline_answer_path, [baseline_answers[item] for item in order])
    locomo_path.write_text(
        json.dumps(locomo_subset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    manifest = {
        "schema_version": "graphmem-v5.74-adaptive-subset-v1",
        "full_questions": len(expected_ids),
        "changed_questions": len(changed),
        "targets": dict(sorted(Counter(
            str(adaptive[item].get("adaptive_recall_target_turns"))
            for item in changed).items())),
        "routes": dict(sorted(Counter(
            str(adaptive[item].get("adaptive_recall_route"))
            for item in changed).items())),
        "prepared": str(prepared_path),
        "prepared_sha256": digest(prepared_path),
        "direct_candidates": str(direct_path),
        "direct_candidates_sha256": digest(direct_path),
        "baseline_answers": (
            str(baseline_answer_path) if baseline_answers is not None else None),
        "baseline_answers_sha256": (
            digest(baseline_answer_path)
            if baseline_answers is not None else None),
        "locomo": str(locomo_path),
        "locomo_sha256": digest(locomo_path),
        "baseline_retrieval_sha256": digest(args.baseline_retrieval),
        "adaptive_retrieval_sha256": digest(args.adaptive_retrieval),
        "non_trigger_prompt_hash_mismatches": 0,
        "trigger_prompt_hash_unchanged": 0,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
