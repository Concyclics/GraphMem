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


def token_stats(values, *, unit: str = "tokens_per_question") -> dict:
    data = [int(value) for value in values]
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p95": nearest(data, 0.95), "p99": nearest(data, 0.99),
        "max": max(data) if data else None,
        "unit": unit, "percentile_method": "nearest_rank",
    }


def accuracy_by_type(data: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in data:
        label = str(row.get("question_type") or "unknown")
        buckets[label].append(bool(row.get("correct")))
    return {
        label: {
            "questions": len(values), "correct": sum(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
        for label, values in sorted(buckets.items())
    }


def build_stats(path: Path) -> dict[str, dict]:
    """Split the frozen 510-owner build ledger by benchmark.

    LongMemEval owns one graph per Memory (500 rows); LoCoMo owns one graph per
    conversation (10 rows).  Reporting the combined 510-row mean next to both
    baselines would use different statistical units, so preserve the benchmark
    owner unit before calculating nearest-rank percentiles.
    """

    report = json.loads(path.read_text(encoding="utf-8"))
    buckets = {
        "longmemeval": [row for row in report.get("rows", ())
                        if not str(row.get("memory_id", "")).startswith("locomo:")],
        "locomo": [row for row in report.get("rows", ())
                   if str(row.get("memory_id", "")).startswith("locomo:")],
    }
    expected = {"longmemeval": 500, "locomo": 10}
    output = {}
    for benchmark, data in buckets.items():
        if len(data) != expected[benchmark]:
            raise RuntimeError(
                f"build ledger has {len(data)} {benchmark} owners; "
                f"expected {expected[benchmark]}")
        unit = ("tokens_per_memory" if benchmark == "longmemeval"
                else "tokens_per_conversation")
        output[benchmark] = {
            "total": token_stats((row.get("tokens", 0) for row in data), unit=unit),
            "input": token_stats((row.get("input_tokens", 0) for row in data), unit=unit),
            "output": token_stats((row.get("output_tokens", 0) for row in data), unit=unit),
            "cached_input": token_stats(
                (row.get("cached_input_tokens", 0) for row in data), unit=unit),
            "owners": len(data),
        }
    return output


def summarize_arm(root: Path, turns: int,
                  build: dict[str, dict]) -> list[dict]:
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
            "correct": bool(verdict["correct"]),
            "question_type": answer.get("question_type"), **metric})
    manifest_path = answer_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result = []
    for benchmark in ("longmemeval", "locomo"):
        data = buckets.get(benchmark, [])
        build_row = build.get(benchmark, {})
        result.append({
            "method": "GraphMem", "answer_model": manifest.get("answer_model"),
            "benchmark": benchmark, "retrieval_setting": f"{turns}-turn",
            "turn_budget": turns, "questions": len(data),
            "correct": sum(row["correct"] for row in data),
            "accuracy": (sum(row["correct"] for row in data) / len(data)
                         if data else None),
            "accuracy_by_type": accuracy_by_type(data),
            "answer_tokens": token_stats(
                row.get("answer_total_tokens", 0) for row in data),
            "build_tokens": build_row.get("total"),
            "build_api_tokens": {
                key: build_row.get(key)
                for key in ("input", "cached_input", "output", "total")
            },
            "build_owner_count": build_row.get("owners"),
            "build_tokens_shared_across_turn_budgets": True,
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


def enrich_mem0(rows_: list[dict]) -> list[dict]:
    enriched = []
    for source in rows_:
        row = dict(source)
        artifact = row.get("verdict_artifact")
        if (row.get("status") != "complete" or not artifact
                or not Path(str(artifact)).exists()):
            row["accuracy_by_type"] = None
            enriched.append(row)
            continue
        verdicts = rows(Path(str(artifact)))
        if row.get("benchmark") == "locomo":
            verdicts = [item for item in verdicts
                        if int(item.get("category", 0)) in {1, 2, 3, 4}]
            typed = [{"correct": bool(item.get("correct")),
                      "question_type": f"category_{int(item['category'])}"}
                     for item in verdicts]
        else:
            typed = [{"correct": bool(item.get("correct")),
                      "question_type": str(item.get("question_type") or "unknown")}
                     for item in verdicts]
        expected = int(row.get("questions") or 0)
        if len(typed) != expected:
            raise RuntimeError(
                f"{row.get('benchmark')} {row.get('retrieval_setting')} type ledger "
                f"has {len(typed)} rows; expected {expected}")
        row["accuracy_by_type"] = accuracy_by_type(typed)
        enriched.append(row)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mem0", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = build_stats(args.build_report)
    graphmem = (summarize_arm(args.root, 32, build)
                + summarize_arm(args.root, 64, build))
    mem0 = json.loads(args.mem0.read_text(encoding="utf-8"))
    mem0_rows = enrich_mem0([
        row for row in mem0.get("rows", ())
        if int(row.get("cutoff", 0)) in {50, 200}
    ])
    payload = {
        "schema_version": "graphmem-v5.20-budget-benchmark-v2",
        "judge_model": "gpt-5.6-luna", "graphmem": graphmem,
        "build_contract": {
            "model": "Qwen3-30B", "embedding_tokens_excluded": True,
            "judge_tokens_excluded": True,
            "shared_across_graphmem_turn_budgets": True,
            "source": str(args.build_report),
            "owner_units": {
                "longmemeval": "memory", "locomo": "conversation"},
        },
        "mem0": mem0_rows,
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
