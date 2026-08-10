#!/usr/bin/env python3
"""Summarize paired 32/64-turn GraphMem and archived Mem0 top-50/top-200."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def nearest(values: list[int], percentile: float) -> int | None:
    values = sorted(values)
    return values[max(0, math.ceil(percentile * len(values)) - 1)] if values else None


def token_stats(values) -> dict:
    data = [int(value) for value in values]
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p95": nearest(data, 0.95), "p99": nearest(data, 0.99),
        "max": max(data) if data else None,
        "unit": "tokens_per_question", "percentile_method": "nearest_rank",
    }


def summarize_arm(root: Path, turns: int) -> list[dict]:
    answer_root = root / f"turn{turns}" / "answer"
    answers = rows(answer_root / "answers.jsonl")
    retrieval = rows(answer_root / "retrieval.jsonl")
    by_id = {str(row["question_id"]): row for row in answers}
    metrics = {str(row["dev_question_id"]): row for row in retrieval}
    verdicts = (rows(answer_root / "judge_lme" / "auto_eval.jsonl")
                + rows(answer_root / "judge_locomo" / "auto_eval.jsonl"))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for verdict in verdicts:
        question_id = str(verdict["question_id"])
        answer = by_id.get(question_id)
        metric = metrics.get(question_id)
        if answer is None or metric is None:
            continue
        buckets[str(answer["benchmark"])].append({
            "correct": bool(verdict["correct"]), **metric})
    manifest_path = answer_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result = []
    for benchmark in ("longmemeval", "locomo"):
        data = buckets.get(benchmark, [])
        result.append({
            "method": "GraphMem", "answer_model": manifest.get("answer_model"),
            "benchmark": benchmark, "retrieval_setting": f"{turns}-turn",
            "turn_budget": turns, "questions": len(data),
            "correct": sum(row["correct"] for row in data),
            "accuracy": (sum(row["correct"] for row in data) / len(data)
                         if data else None),
            "answer_tokens": token_stats(
                row.get("answer_total_tokens", 0) for row in data),
            "prompt_tokens": token_stats(
                row.get("api_prompt_tokens", 0) for row in data),
            "completion_tokens": token_stats(
                row.get("completion_tokens", 0) for row in data),
            "mean_packed_turns": (sum(float(row.get("packed_turns", 0))
                                      for row in data) / len(data) if data else None),
            "turn_recall": (sum(float(row.get("turn_recall", 0)) for row in data)
                            / len(data) if data else None),
            "turn_precision": (sum(float(row.get("turn_precision", 0)) for row in data)
                               / len(data) if data else None),
            "turn_all_hit": (sum(bool(row.get("turn_all_hit")) for row in data)
                             / len(data) if data else None),
            "answer_manifest": str(manifest_path),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mem0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graphmem = summarize_arm(args.root, 32) + summarize_arm(args.root, 64)
    mem0 = json.loads(args.mem0.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "graphmem-v5.20-budget-benchmark-v1",
        "judge_model": "gpt-5.6-luna", "graphmem": graphmem,
        "mem0": [row for row in mem0.get("rows", ())
                 if int(row.get("cutoff", 0)) in {50, 200}],
        "comparison_note": (
            "Pair rows by benchmark and compare measured answer Token, not "
            "nominal turn/top-k labels."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"graphmem": graphmem}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
