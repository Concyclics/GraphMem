#!/usr/bin/env python3
"""Validate and merge memory-sharded ``--prepare-only`` answer artifacts."""
from __future__ import annotations

import argparse
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


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2040)
    parser.add_argument(
        "--order-from", type=Path,
        help="PreparedAnswer JSONL whose question order is canonical")
    args = parser.parse_args()

    shard_dirs = []
    for shard_root in sorted(args.shard_root.glob("shard_*")):
        if (shard_root / "prepare_manifest.json").is_file():
            shard_dirs.append(shard_root)
            continue
        # Without --run-root, the runner creates one timestamped directory
        # below each memory shard. Accept exactly one completed child while
        # rejecting ambiguous/retried outputs.
        completed = [
            path.parent for path in shard_root.glob(
                "v5_6_answer_*/prepare_manifest.json")]
        if len(completed) > 1:
            raise RuntimeError(
                f"multiple completed prepare runs below {shard_root}")
        if completed:
            shard_dirs.append(completed[0])
    if not shard_dirs:
        raise RuntimeError("no completed prepare shards")
    prepared = [row for shard in shard_dirs
                for row in read_jsonl(shard / "prepared_answers.jsonl")]
    retrieval = [row for shard in shard_dirs
                 for row in read_jsonl(shard / "retrieval.jsonl")]
    prepared_by_id = {str(row["question_id"]): row for row in prepared}
    retrieval_by_id = {str(row["dev_question_id"]): row for row in retrieval}
    if len(prepared_by_id) != len(prepared):
        raise ValueError("duplicate prepared question id across shards")
    if len(retrieval_by_id) != len(retrieval):
        raise ValueError("duplicate retrieval question id across shards")
    if set(prepared_by_id) != set(retrieval_by_id):
        raise ValueError("prepared/retrieval question id sets differ")
    if len(prepared_by_id) != args.expected:
        raise ValueError(
            f"expected {args.expected} questions, got {len(prepared_by_id)}")

    if args.order_from:
        ordered_ids = [str(row["question_id"])
                       for row in read_jsonl(args.order_from)]
        if set(ordered_ids) != set(prepared_by_id):
            raise ValueError("canonical order question ids differ from shards")
    else:
        ordered_ids = sorted(prepared_by_id)
    prepared = [prepared_by_id[item] for item in ordered_ids]
    retrieval = [retrieval_by_id[item] for item in ordered_ids]

    manifests = [json.loads((shard / "prepare_manifest.json").read_text())
                 for shard in shard_dirs]
    for key in (
            "schema_version", "profile", "full", "evidence_order",
            "max_evidence_turns", "max_evidence_tokens", "max_answer_tokens",
            "aggregation_source_reserve", "speaker_owner_bonus",
            "query_witness_bonus"):
        values = {json.dumps(row.get(key), sort_keys=True) for row in manifests}
        if len(values) != 1:
            raise ValueError(f"shard manifest mismatch for {key}: {values}")
    for prepared_row, retrieval_row in zip(prepared, retrieval):
        if (str(prepared_row["question_id"])
                != str(retrieval_row["dev_question_id"])):
            raise AssertionError("canonical merge order mismatch")
        evidence_ids = tuple(prepared_row.get("evidence_turn_ids") or ())
        retrieved_ids = tuple(retrieval_row.get("retrieved_turn_ids") or ())
        if not set(evidence_ids) <= set(retrieved_ids):
            raise ValueError(
                f"prepared evidence not contained in retrieval for "
                f"{prepared_row['question_id']}")

    tokens = [int(row["prompt_tokens"]) for row in retrieval]
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "prepared_answers.jsonl", prepared)
    write_jsonl(args.output / "retrieval.jsonl", retrieval)
    merged = {
        **{key: manifests[0].get(key) for key in manifests[0]},
        "schema_version": "graphmem-v5.40-merged-full-prepare-v1",
        "shards": len(shard_dirs),
        "questions": len(prepared),
        "answer_calls": 0,
        "answer_generation_tokens": 0,
        "prepared_answers": str(args.output / "prepared_answers.jsonl"),
        "retrieval": str(args.output / "retrieval.jsonl"),
        "prompt_tokens": {
            "count": len(tokens), "mean": statistics.fmean(tokens),
            "p50": percentile(tokens, .50),
            "p95": percentile(tokens, .95),
            "p99": percentile(tokens, .99), "max": max(tokens),
            "percentile_method": "nearest-rank",
        },
    }
    (args.output / "prepare_manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
