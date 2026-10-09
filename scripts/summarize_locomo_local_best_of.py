#!/usr/bin/env python3
"""Summarize local LoCoMo build/answer tokens and prefix Best-of-N accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def stats(values: Iterable[int], unit: str) -> dict[str, Any]:
    rows = sorted(int(value) for value in values)

    def nearest(p: float) -> int:
        return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0

    return {
        "count": len(rows), "mean": sum(rows) / max(1, len(rows)),
        "p50": nearest(0.50), "p95": nearest(0.95),
        "p99": nearest(0.99), "max": max(rows, default=0),
        "sum": sum(rows), "unit": unit, "percentile_method": "nearest_rank",
    }


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def embedding_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"calls": 0, "items": 0, "input_tokens": 0}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT count(*),coalesce(sum(item_count),0),"
            "coalesce(sum(input_tokens),0) FROM embedding_calls").fetchone()
    return {"calls": int(row[0]), "items": int(row[1]),
            "input_tokens": int(row[2])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--answer-manifest", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--relation-db", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = read_jsonl(args.candidates)
    if len(candidate_rows) != 1540:
        raise RuntimeError(f"expected 1540 candidates, got {len(candidate_rows)}")
    answer_manifest = json.loads(args.answer_manifest.read_text(encoding="utf-8"))
    build_report = json.loads(args.build_report.read_text(encoding="utf-8"))
    build_rows = list(build_report.get("rows", ()))
    if len(build_rows) != 10 or any(
            not str(row["memory_id"]).startswith("locomo:") for row in build_rows):
        raise RuntimeError("build report must contain exactly ten LoCoMo owners")

    verdicts: dict[tuple[str, str], bool] = {}
    judge_requests = 0
    for index in range(1, int(answer_manifest["n"]) + 1):
        path = args.judge_root / f"candidate_{index}" / "auto_eval.jsonl"
        rows = read_jsonl(path)
        judge_requests += len(rows)
        for row in rows:
            key = (str(row["question_id"]), str(row["prediction_sha256"]))
            flag = bool(row["correct"])
            if key in verdicts and verdicts[key] != flag:
                raise RuntimeError(f"judge conflict for {key}")
            verdicts[key] = flag

    first_correct: dict[str, int | None] = {}
    categories: dict[str, int] = {}
    missing: list[str] = []
    for row in candidate_rows:
        question_id = str(row["question_id"])
        categories[question_id] = int(row["category"])
        first: int | None = None
        seen: set[str] = set()
        for index, candidate in enumerate(row["candidates"], start=1):
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
            f"missing {len(missing)} required verdicts; first={missing[0]}")

    curve = []
    n = int(answer_manifest["n"])
    for k in range(1, n + 1):
        correct = sum(value is not None and value <= k
                      for value in first_correct.values())
        by_category: dict[str, dict[str, Any]] = {}
        for category in sorted(set(categories.values())):
            ids = [question_id for question_id, value in categories.items()
                   if value == category]
            cat_correct = sum(
                first_correct[question_id] is not None
                and int(first_correct[question_id]) <= k for question_id in ids)
            by_category[str(category)] = {
                "correct": cat_correct, "questions": len(ids),
                "accuracy": cat_correct / len(ids),
            }
        curve.append({
            "k": k, "correct": correct, "questions": len(first_correct),
            "accuracy": correct / len(first_correct),
            "wilson95": wilson(correct, len(first_correct)),
            "uplift_vs_best_of_1_pp": (
                0.0 if k == 1 else
                100.0 * (correct - curve[0]["correct"]) / len(first_correct)),
            "reaches_90_percent": correct / len(first_correct) >= 0.90,
            "by_category": by_category,
        })

    graph_embedding = embedding_stats(args.graph_db)
    relation_embedding = embedding_stats(args.relation_db)
    summary = {
        "schema_version": "graphmem-local-locomo-best-of-summary-v1",
        "benchmark": "locomo_category_1_4",
        "questions": len(first_correct),
        "build_model": build_report["summary"].get("llm_model", "local Qwen3-30B"),
        "build_config": build_report["summary"].get("config"),
        "build_config_hash": build_report["summary"].get("config_hash"),
        "embedding_request_model": build_report["summary"].get(
            "embedding_request_model"),
        "answer_model": answer_manifest["model"],
        "judge_model": "gpt-5.6-luna",
        "judge_reasoning_effort": "medium",
        "oracle_warning": (
            "Best-of-k uses judge labels to select whether any of the first k "
            "answers is correct; it is an evaluator-only upper bound, not a "
            "deployable selector."),
        "build_tokens": {
            "input": stats((row.get("input_tokens", 0) for row in build_rows),
                           "tokens_per_conversation"),
            "output": stats((row.get("output_tokens", 0) for row in build_rows),
                            "tokens_per_conversation"),
            "total": stats((row.get("tokens", 0) for row in build_rows),
                           "tokens_per_conversation"),
            "retry_count": int(build_report["summary"].get("retry_count", 0)),
        },
        "embedding_input": {
            "excluded_from_generative_token_total": True,
            "graph_db": graph_embedding,
            "relation_db": relation_embedding,
            "combined": {
                key: graph_embedding[key] + relation_embedding[key]
                for key in ("calls", "items", "input_tokens")},
        },
        "answer_n8_actual_api_tokens": answer_manifest[
            "api_tokens_per_n8_request"],
        "answer_api_usage_sums": answer_manifest["api_usage_sums"],
        "best_of_curve": curve,
        "first_correct_histogram": dict(sorted(Counter(
            str(value) if value is not None else "never"
            for value in first_correct.values()).items())),
        "judge_requests_executed": judge_requests,
        "judge_requests_avoided_by_early_stop_and_dedup": (
            len(first_correct) * n - judge_requests),
        "artifacts": {
            "build_report": str(args.build_report),
            "answer_manifest": str(args.answer_manifest),
            "candidates": str(args.candidates),
            "candidates_sha256": hashlib.sha256(
                args.candidates.read_bytes()).hexdigest(),
            "judge_root": str(args.judge_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "best_of_curve": [{"k": row["k"], "accuracy": row["accuracy"],
                           "correct": row["correct"]} for row in curve],
        "build_tokens": summary["build_tokens"],
        "answer_tokens": summary["answer_n8_actual_api_tokens"],
        "judge_requests": judge_requests,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
