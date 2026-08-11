#!/usr/bin/env python3
"""Merge benchmark-split PreparedAnswer replays into one audited run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def stats(values) -> dict:
    rows = sorted(int(value) for value in values)
    def nearest(probability: float) -> int:
        return rows[max(0, math.ceil(probability * len(rows)) - 1)]
    return {
        "count": len(rows), "mean": statistics.fmean(rows),
        "p50": nearest(.50), "p95": nearest(.95),
        "p99": nearest(.99), "max": max(rows),
        "unit": "tokens_per_question",
        "percentile_method": "nearest-rank",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--answer-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()

    prepared = read_jsonl(args.prepared)
    retrieval = read_jsonl(args.retrieval)
    answers = [row for root in args.answer_dir
               for row in read_jsonl(root / "answers.jsonl")]
    usage = [row for root in args.answer_dir
             for row in read_jsonl(root / "answer_usage.jsonl")]
    prepared_ids = [str(row["question_id"]) for row in prepared]
    prepared_by_id = {str(row["question_id"]): row for row in prepared}
    retrieval_by_id = {str(row["dev_question_id"]): row for row in retrieval}
    answer_by_id = {str(row["question_id"]): row for row in answers}
    usage_by_id = {str(row["question_id"]): row for row in usage}
    for name, rows, index in (
            ("prepared", prepared, prepared_by_id),
            ("retrieval", retrieval, retrieval_by_id),
            ("answers", answers, answer_by_id), ("usage", usage, usage_by_id)):
        if len(rows) != len(index):
            raise ValueError(f"duplicate question id in {name}")
    expected = set(prepared_ids)
    if len(expected) != args.expected:
        raise ValueError(f"expected {args.expected} prepared rows, got {len(expected)}")
    if any(set(index) != expected for index in (
            retrieval_by_id, answer_by_id, usage_by_id)):
        raise ValueError("prepared/retrieval/answer/usage question sets differ")
    mismatches = [
        question_id for question_id in prepared_ids
        if str(prepared_by_id[question_id]["prompt_payload_hash"])
        != str(answer_by_id[question_id].get("prompt_payload_hash") or "")
        or str(prepared_by_id[question_id]["prompt_payload_hash"])
        != str(usage_by_id[question_id].get("prompt_payload_hash") or "")]
    if mismatches:
        raise ValueError(f"{len(mismatches)} answer prompt hash mismatches")
    if not all(
            int(row.get("api_prompt_tokens") or 0)
            + int(row.get("completion_tokens") or 0)
            == int(row.get("total_tokens") or 0) for row in usage):
        raise ValueError("answer API usage is not additive")

    ordered_answers = [answer_by_id[item] for item in prepared_ids]
    ordered_usage = [usage_by_id[item] for item in prepared_ids]
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "prepared_answers.jsonl", prepared)
    write_jsonl(args.output / "retrieval.jsonl", [
        retrieval_by_id[item] for item in prepared_ids])
    write_jsonl(args.output / "answers.jsonl", ordered_answers)
    write_jsonl(args.output / "answer_usage.jsonl", ordered_usage)
    for benchmark in ("longmemeval", "locomo"):
        write_jsonl(args.output / f"answers_{benchmark}.jsonl", [
            row for row in ordered_answers if row.get("benchmark") == benchmark])

    baseline = (json.loads(args.baseline_manifest.read_text())
                if args.baseline_manifest else {})
    manifest = {
        "schema_version": "graphmem-merged-prepared-answer-replay-v1",
        "questions": len(prepared_ids),
        "answer_model": next(iter({str(row["answer_model"]) for row in answers})),
        "max_output_tokens": 2000,
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "prompt_identity_audit": {
            "question_ids_match": True,
            "evidence_and_order_frozen_in_prepared_artifact": True,
            "prompt_hash_mismatches": 0,
        },
        "api_tokens": {
            name: stats(row[field] for row in ordered_usage)
            for name, field in (("prompt", "api_prompt_tokens"),
                                ("completion", "completion_tokens"),
                                ("total", "total_tokens"))
        },
        "api_tokens_by_benchmark": {
            benchmark: {
                name: stats(row[field] for row in ordered_usage
                            if row.get("benchmark") == benchmark)
                for name, field in (("prompt", "api_prompt_tokens"),
                                    ("completion", "completion_tokens"),
                                    ("total", "total_tokens"))
            } for benchmark in ("longmemeval", "locomo")
        },
        "baseline_total_token_mean": (
            (baseline.get("answer_api_tokens") or {}).get("total", {}).get("mean")),
        "output_truncated": sum(
            row.get("finish_reason") == "length" for row in ordered_usage),
        "api_usage_sums": {
            field: sum(int(row[key]) for row in ordered_usage)
            for field, key in (("prompt", "api_prompt_tokens"),
                               ("completion", "completion_tokens"),
                               ("total", "total_tokens"))
        },
        "api_usage_additivity_ok": True,
    }
    if manifest["baseline_total_token_mean"] is not None:
        manifest["total_token_mean_delta_vs_baseline"] = (
            manifest["api_tokens"]["total"]["mean"]
            - float(manifest["baseline_total_token_mean"]))
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
