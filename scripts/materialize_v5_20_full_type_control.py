#!/usr/bin/env python3
"""Materialize the 869-question topology control from a completed full run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TARGET_STRATA = frozenset({
    "lme_multi_session", "lme_temporal_reasoning",
    "locomo_cat1", "locomo_cat2",
})


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n"
                            for row in values), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values) -> dict:
    ordered = sorted(int(value or 0) for value in values)
    def rank(p: float) -> int:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else 0,
        "p50": rank(0.50), "p95": rank(0.95), "p99": rank(0.99),
        "max": max(ordered, default=0),
        "unit": "tokens_per_question", "percentile_method": "nearest_rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="completed 2,040-question answer directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="derived topology_layout/answer directory")
    args = parser.parse_args()

    retrieval = rows(args.source / "retrieval.jsonl")
    selected_retrieval = [row for row in retrieval
                          if str(row.get("stratum")) in TARGET_STRATA]
    selected_ids = {str(row["dev_question_id"]) for row in selected_retrieval}
    if len(selected_ids) != 869:
        raise RuntimeError(f"expected 869 selected questions, found {len(selected_ids)}")

    specifications = (
        ("answers.jsonl", "question_id"),
        ("answers_longmemeval.jsonl", "question_id"),
        ("answers_locomo.jsonl", "question_id"),
        ("prepared_answers.jsonl", "question_id"),
    )
    selected: dict[str, list[dict]] = {"retrieval.jsonl": selected_retrieval}
    for name, key in specifications:
        selected[name] = [row for row in rows(args.source / name)
                          if str(row[key]) in selected_ids]
    if len(selected["answers.jsonl"]) != 869 or len(selected["prepared_answers.jsonl"]) != 869:
        raise RuntimeError("source answer/prepared rows do not cover the selected questions")

    for name, values in selected.items():
        write_rows(args.output / name, values)
    for judge in ("judge_lme", "judge_locomo"):
        values = [row for row in rows(args.source / judge / "auto_eval.jsonl")
                  if str(row["question_id"]) in selected_ids]
        write_rows(args.output / judge / "auto_eval.jsonl", values)
    verdicts = sum(len(rows(args.output / judge / "auto_eval.jsonl"))
                   for judge in ("judge_lme", "judge_locomo"))
    if verdicts != 869:
        raise RuntimeError(f"expected 869 derived verdicts, found {verdicts}")

    manifest = json.loads((args.source / "run_manifest.json").read_text(encoding="utf-8"))
    retrieval_rows = selected_retrieval
    manifest.update({
        "questions": 869,
        "question_filters": {
            "lme_types": ["multi-session", "temporal-reasoning"],
            "locomo_categories": [1, 2],
        },
        "derived_subset": {
            "source": str(args.source),
            "source_manifest_sha256": sha256(args.source / "run_manifest.json"),
            "question_id_sha256": hashlib.sha256(
                "\n".join(sorted(selected_ids)).encode()).hexdigest(),
            "no_answer_or_judge_calls": True,
        },
        "output_truncated": sum(
            row.get("answer_finish_reason") == "length" for row in retrieval_rows),
        "answer_api_tokens": {
            key: stats(row.get(field, 0) for row in retrieval_rows)
            for key, field in (
                ("prompt", "api_prompt_tokens"),
                ("completion", "completion_tokens"),
                ("total", "answer_total_tokens"),
            )
        },
        "answer_api_usage_sums": {
            key: sum(int(row.get(field) or 0) for row in retrieval_rows)
            for key, field in (
                ("prompt", "api_prompt_tokens"),
                ("completion", "completion_tokens"),
                ("total", "answer_total_tokens"),
            )
        },
        "prepared_answers": str(args.output / "prepared_answers.jsonl"),
        "prepared_prompt_hashes": len({
            str(row.get("prompt_payload_hash")) for row in retrieval_rows}),
    })
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "questions": 869,
        "lme": len(selected["answers_longmemeval.jsonl"]),
        "locomo": len(selected["answers_locomo.jsonl"]),
        "verdicts": verdicts,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
