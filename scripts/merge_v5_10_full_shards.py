#!/usr/bin/env python3
"""Validate and merge checkpointed V5.10 answer shards."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def percentile(rows: list[int | float], probability: float) -> float:
    ordered = sorted(rows)
    return ordered[max(0, min(len(ordered) - 1,
                              math.ceil(len(ordered) * probability) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=2040)
    args = parser.parse_args()
    shard_dirs = sorted(
        row for row in args.shard_root.glob("shard_*")
        if (row / "run_manifest.json").is_file())
    if not shard_dirs:
        raise RuntimeError("no completed shard manifests found")
    answers = [row for shard in shard_dirs for row in read_jsonl(shard / "answers.jsonl")]
    retrieval = [row for shard in shard_dirs for row in read_jsonl(shard / "retrieval.jsonl")]
    answer_by_id = {str(row["question_id"]): row for row in answers}
    retrieval_by_id = {str(row["dev_question_id"]): row for row in retrieval}
    if len(answer_by_id) != len(answers):
        raise ValueError("duplicate answer question id across shards")
    if len(retrieval_by_id) != len(retrieval):
        raise ValueError("duplicate retrieval question id across shards")
    if set(answer_by_id) != set(retrieval_by_id):
        raise ValueError("answer/retrieval checkpoint id sets differ")
    if len(answer_by_id) != args.expected:
        raise ValueError(f"expected {args.expected} questions, got {len(answer_by_id)}")
    ordered_ids = sorted(answer_by_id)
    answers = [answer_by_id[item] for item in ordered_ids]
    retrieval = [retrieval_by_id[item] for item in ordered_ids]
    manifests = [json.loads((shard / "run_manifest.json").read_text())
                 for shard in shard_dirs]
    for key in ("profile", "config_hash", "answer_prompt_hash",
                "obligation_aware_packing", "native_seed_fusion",
                "graph_hop_decay", "expansion_beam",
                "rare_lexical_relations"):
        if len({json.dumps(row.get(key), sort_keys=True) for row in manifests}) != 1:
            raise ValueError(f"shard manifest mismatch for {key}")
    tokens = [int(row["prompt_tokens"]) for row in retrieval]
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "answers.jsonl", answers)
    write_jsonl(args.output / "retrieval.jsonl", retrieval)
    write_jsonl(args.output / "answers_longmemeval.jsonl", [
        row for row in answers if row["benchmark"] == "longmemeval"])
    write_jsonl(args.output / "answers_locomo.jsonl", [
        row for row in answers if row["benchmark"] == "locomo"])
    merged = {
        "schema_version": "graphmem-v5.10-full-answers-v1",
        "shards": len(shard_dirs), "questions": len(answers),
        "longmemeval_questions": sum(row["benchmark"] == "longmemeval"
                                     for row in answers),
        "locomo_questions": sum(row["benchmark"] == "locomo" for row in answers),
        **{key: manifests[0].get(key) for key in (
            "profile", "label", "source_db", "config_hash", "answer_prompt_hash",
            "obligation_aware_packing", "obligation_aware_relations",
            "native_seed_fusion", "graph_hop_decay", "expansion_beam",
            "rare_lexical_relations",
            "span_pack_window", "closed_form_enabled", "budget", "token_counter")},
        "prompt_tokens": {
            "mean": statistics.fmean(tokens), "p50": percentile(tokens, .50),
            "p95": percentile(tokens, .95), "max": max(tokens),
            "over_soft_budget": sum(value > 10_000 for value in tokens),
        },
        "closed_form_rate": statistics.fmean(
            bool(row["closed_form"]) for row in retrieval),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
