#!/usr/bin/env python3
"""Prepare a source-free LoCoMo candidate-1 judge payload with cache reuse.

The emitted judge data contains only question metadata and the reference
answer.  The emitted answer rows contain only the candidate answer and its
hash.  Conversation turns, prepared prompts, evidence IDs and graph metadata
are deliberately excluded from the external-judge input boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(
            encoding="utf-8").split("\n") if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--select-wrong-judge", type=Path,
        help=("optional prior auto_eval JSONL; retain only question IDs whose "
              "verdict is wrong, for source-free later-candidate audits"))
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    source_cases = [
        row for row in json.loads(args.data.read_text(encoding="utf-8"))
        if int(row["locomo_category"]) in {1, 2, 3, 4}
    ]
    selected_question_ids: set[str] | None = None
    if args.select_wrong_judge is not None:
        selection_rows = read_jsonl(args.select_wrong_judge)
        selected_question_ids = {
            str(row["question_id"]) for row in selection_rows
            if not bool(row.get("correct"))}
        if len({str(row["question_id"]) for row in selection_rows}) != len(
                selection_rows):
            raise RuntimeError("selection judge contains duplicate question IDs")
        source_cases = [
            row for row in source_cases
            if str(row["question_id"]) in selected_question_ids]
    candidates = read_jsonl(args.candidates)
    if args.expected and len(source_cases) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} cases, got {len(source_cases)}")
    candidate_by_id = {str(row["question_id"]): row for row in candidates}
    case_ids = [str(row["question_id"]) for row in source_cases]
    if len(candidate_by_id) != len(candidates):
        raise RuntimeError("candidate file contains duplicate question IDs")
    if not set(case_ids).issubset(candidate_by_id):
        raise RuntimeError("candidate file does not cover selected benchmark IDs")
    if selected_question_ids is None and set(candidate_by_id) != set(case_ids):
        raise RuntimeError("candidate and benchmark question IDs do not match")

    known: dict[tuple[str, str], dict[str, Any]] = {}
    for root in args.cache_root:
        for path in sorted(root.glob("candidate_*/auto_eval.jsonl")):
            for row in read_jsonl(path):
                key = (str(row["question_id"]), str(row["prediction_sha256"]))
                previous = known.setdefault(key, row)
                if bool(previous["correct"]) != bool(row["correct"]):
                    raise RuntimeError(f"conflicting cached verdict for {key}")
        direct = root / "auto_eval.jsonl"
        for row in read_jsonl(direct):
            key = (str(row["question_id"]), str(row["prediction_sha256"]))
            previous = known.setdefault(key, row)
            if bool(previous["correct"]) != bool(row["correct"]):
                raise RuntimeError(f"conflicting cached verdict for {key}")

    sanitized_cases: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    cached: list[dict[str, Any]] = []
    for case in source_cases:
        question_id = str(case["question_id"])
        sanitized_cases.append({
            "question_id": question_id,
            "question": str(case["question"]),
            "answer": str(case.get("answer") or ""),
            "locomo_category": int(case["locomo_category"]),
            "locomo_sample_id": str(case["locomo_sample_id"]),
        })
        row = candidate_by_id[question_id]
        choices = list(row.get("candidates", ()))
        if not 1 <= args.candidate_index <= len(choices):
            raise RuntimeError(
                f"{question_id}: candidate index {args.candidate_index} absent")
        choice = choices[args.candidate_index - 1]
        prediction = str(choice.get("prediction") or "")
        digest = hashlib.sha256(prediction.encode("utf-8")).hexdigest()
        if digest != str(choice.get("prediction_sha256") or ""):
            raise RuntimeError(f"{question_id}: prediction hash mismatch")
        answers.append({
            "question_id": question_id,
            "prediction": prediction,
            "prediction_sha256": digest,
            "benchmark": "locomo",
            "stratum": f"locomo_cat{int(case['locomo_category'])}",
            "candidate_index": args.candidate_index,
        })
        cached_row = known.get((question_id, digest))
        if cached_row is not None:
            cached.append(cached_row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    data_path = args.output_root / "judge_data_source_free.json"
    data_path.write_text(
        json.dumps(sanitized_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    answers_path = args.output_root / "answers_candidate_1.jsonl"
    write_jsonl(answers_path, answers)
    judge_root = args.output_root / "judge_candidate_1"
    write_jsonl(judge_root / "auto_eval.jsonl", cached)
    (judge_root / "judge_calls.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.79-source-free-paired-judge-v1",
        "questions": len(answers),
        "selection": ({
            "kind": "prior_judge_wrong_only",
            "path": str(args.select_wrong_judge),
            "sha256": hashlib.sha256(
                args.select_wrong_judge.read_bytes()).hexdigest(),
        } if args.select_wrong_judge is not None else None),
        "candidate_index": args.candidate_index,
        "cached_verdicts": len(cached),
        "external_requests_remaining": len(answers) - len(cached),
        "external_payload_fields": [
            "question", "reference_answer", "candidate_answer"],
        "excluded_fields": [
            "conversation_turns", "source_memories", "prepared_prompt",
            "evidence_turn_ids", "graph", "database"],
        "source_candidates": str(args.candidates),
        "source_candidates_sha256": hashlib.sha256(
            args.candidates.read_bytes()).hexdigest(),
        "cache_roots": [str(path) for path in args.cache_root],
        "outputs": {
            "data": str(data_path),
            "answers": str(answers_path),
            "judge": str(judge_root),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
