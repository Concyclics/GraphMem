#!/usr/bin/env python3
"""Replace a validated subset of full PreparedAnswer/retrieval rows.

This is intentionally stricter than concatenating JSONL files: the base order is
preserved, prepared/retrieval IDs must match, and every packed evidence ID must
remain present in the corresponding retrieval record.
"""
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


def index_rows(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed = {str(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate {key} in {label}")
    return indexed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def nearest_rank(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()

    base_prepared = read_jsonl(args.base / "prepared_answers.jsonl")
    base_retrieval = read_jsonl(args.base / "retrieval.jsonl")
    patch_prepared = read_jsonl(args.patch / "prepared_answers.jsonl")
    patch_retrieval = read_jsonl(args.patch / "retrieval.jsonl")

    if len(base_prepared) != args.expected or len(base_retrieval) != args.expected:
        raise ValueError(
            f"expected {args.expected} base rows, got "
            f"{len(base_prepared)} prepared/{len(base_retrieval)} retrieval")

    base_prepared_by_id = index_rows(
        base_prepared, "question_id", "base prepared")
    base_retrieval_by_id = index_rows(
        base_retrieval, "dev_question_id", "base retrieval")
    patch_prepared_by_id = index_rows(
        patch_prepared, "question_id", "patch prepared")
    patch_retrieval_by_id = index_rows(
        patch_retrieval, "dev_question_id", "patch retrieval")

    if set(base_prepared_by_id) != set(base_retrieval_by_id):
        raise ValueError("base prepared/retrieval ID sets differ")
    if set(patch_prepared_by_id) != set(patch_retrieval_by_id):
        raise ValueError("patch prepared/retrieval ID sets differ")
    unknown = set(patch_prepared_by_id) - set(base_prepared_by_id)
    if unknown:
        raise ValueError(f"patch contains unknown IDs: {sorted(unknown)}")
    if not patch_prepared_by_id:
        raise ValueError("patch is empty")

    prepared: list[dict] = []
    retrieval: list[dict] = []
    for base_row in base_prepared:
        question_id = str(base_row["question_id"])
        prepared_row = patch_prepared_by_id.get(question_id, base_row)
        retrieval_row = patch_retrieval_by_id.get(
            question_id, base_retrieval_by_id[question_id])
        if question_id != str(retrieval_row["dev_question_id"]):
            raise AssertionError(f"row ID mismatch for {question_id}")
        evidence_ids = set(prepared_row.get("evidence_turn_ids") or ())
        retrieved_ids = set(retrieval_row.get("retrieved_turn_ids") or ())
        if not evidence_ids <= retrieved_ids:
            raise ValueError(
                f"prepared evidence not contained in retrieval for {question_id}")
        prepared.append(prepared_row)
        retrieval.append(retrieval_row)

    args.output.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output / "prepared_answers.jsonl"
    retrieval_path = args.output / "retrieval.jsonl"
    write_jsonl(prepared_path, prepared)
    write_jsonl(retrieval_path, retrieval)

    base_manifest = json.loads(
        (args.base / "prepare_manifest.json").read_text(encoding="utf-8"))
    prompt_tokens = [int(row["prompt_tokens"]) for row in retrieval]
    manifest = {
        **base_manifest,
        "schema_version": "graphmem-v5.63-patched-full-prepare-v1",
        "questions": len(prepared),
        "replacement_questions": sorted(patch_prepared_by_id),
        "replacement_count": len(patch_prepared_by_id),
        "base_prepare": str(args.base.resolve()),
        "patch_prepare": str(args.patch.resolve()),
        "prepared_answers": str(prepared_path.resolve()),
        "retrieval": str(retrieval_path.resolve()),
        "prompt_tokens": {
            "count": len(prompt_tokens),
            "mean": statistics.fmean(prompt_tokens),
            "p50": nearest_rank(prompt_tokens, .50),
            "p95": nearest_rank(prompt_tokens, .95),
            "p99": nearest_rank(prompt_tokens, .99),
            "max": max(prompt_tokens),
            "percentile_method": "nearest-rank",
        },
        "sha256": {
            "prepared_answers": sha256(prepared_path),
            "retrieval": sha256(retrieval_path),
        },
    }
    manifest_path = args.output / "prepare_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
