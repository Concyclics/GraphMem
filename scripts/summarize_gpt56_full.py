#!/usr/bin/env python3
"""Audit and summarize a frozen-prompt GPT-5.6 full benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return [max(0.0, center - margin), min(1.0, center + margin)]


def nearest(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": nearest(values, 0.50),
        "p95": nearest(values, 0.95),
        "p99": nearest(values, 0.99),
        "max": max(values),
    }


def prediction_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row.get("prediction") or "").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.93)
    parser.add_argument(
        "--judges", nargs="+", choices=("sol_medium", "luna_medium"),
        default=("sol_medium", "luna_medium"),
        help="Judge arms that must be present and included in the summary.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    answers = keyed(args.root / "answer/answers.jsonl")
    usage = keyed(args.root / "answer/answer_usage.jsonl")
    answer_manifest = json.loads(
        (args.root / "answer/run_manifest.json").read_text(encoding="utf-8"))
    prepared = keyed(args.prepared)
    if set(answers) != set(usage) or set(answers) != set(prepared):
        raise ValueError("answer, usage and PreparedAnswer IDs differ")
    prompt_mismatches = [
        question_id for question_id in answers
        if str(answers[question_id].get("prompt_payload_hash"))
        != str(prepared[question_id].get("prompt_payload_hash"))]
    if prompt_mismatches:
        raise ValueError(f"{len(prompt_mismatches)} prompt hash mismatches")

    judges: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    output: dict[str, Any] = {
        "schema_version": "graphmem-gpt56-frozen-prompt-full-v1",
        "answer_model": answer_manifest["answer_model"],
        "answer_reasoning_effort": answer_manifest.get(
            "answer_reasoning_effort", "none"),
        "answer_max_output_tokens": answer_manifest.get("max_output_tokens"),
        "target_accuracy": args.target,
        "prepared": str(args.prepared),
        "prepared_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "questions": len(answers),
        "prompt_hash_mismatches": len(prompt_mismatches),
        "output_truncated": sum(
            str(row.get("finish_reason")) == "length" for row in usage.values()),
        "benchmarks": {},
        "judge_agreement": {},
    }
    for judge in args.judges:
        judges[judge] = {}
        for benchmark, suffix in (("longmemeval", "lme"),
                                  ("locomo", "locomo")):
            verdict_path = args.root / f"judge/{judge}/{suffix}/auto_eval.jsonl"
            verdicts = keyed(verdict_path)
            expected = {question_id for question_id, row in answers.items()
                        if str(row.get("benchmark")) == benchmark}
            if set(verdicts) != expected:
                raise ValueError(f"{judge}/{benchmark} verdict coverage mismatch")
            bad_hashes = [
                question_id for question_id, row in verdicts.items()
                if str(row.get("prediction_sha256") or "")
                and str(row["prediction_sha256"])
                != prediction_sha256(answers[question_id])]
            if bad_hashes:
                raise ValueError(
                    f"{judge}/{benchmark} has {len(bad_hashes)} prediction mismatches")
            judges[judge][benchmark] = verdicts
            token_stats = json.loads((
                args.root / f"judge/{judge}/{suffix}/judge_token_stats.json"
            ).read_text())
            correct = sum(bool(row["correct"]) for row in verdicts.values())
            total = len(verdicts)
            target_correct = math.ceil(args.target * total)
            benchmark_output = output["benchmarks"].setdefault(
                benchmark, {"questions": total, "judges": {}})
            benchmark_output["judges"][judge] = {
                "model": token_stats["model"],
                "reasoning_effort": token_stats["reasoning_effort"],
                "correct": correct,
                "accuracy": correct / total,
                "wilson95": wilson(correct, total),
                "target_correct": target_correct,
                "correct_gap_to_target": correct - target_correct,
                "reaches_target": correct >= target_correct,
                "by_type": (token_stats.get("by_question_type")
                            or token_stats.get("by_category")),
                "request_retry_count": token_stats["request_retry_count"],
                "semantic_retry_count": token_stats.get(
                    "semantic_retry_count", 0),
                "failure_count": token_stats["failure_count"],
                "judge_reasoning_tokens": token_stats["reasoning_tokens"],
                "judge_total_tokens": token_stats["total_tokens"],
                "verdicts": str(verdict_path),
            }

    for benchmark in ("longmemeval", "locomo"):
        if {"sol_medium", "luna_medium"}.issubset(judges):
            sol = judges["sol_medium"][benchmark]
            luna = judges["luna_medium"][benchmark]
            ids = set(sol)
            both_correct = sum(bool(sol[item]["correct"]) and bool(
                luna[item]["correct"]) for item in ids)
            both_wrong = sum(not bool(sol[item]["correct"]) and not bool(
                luna[item]["correct"]) for item in ids)
            sol_only = sum(bool(sol[item]["correct"]) and not bool(
                luna[item]["correct"]) for item in ids)
            luna_only = sum(not bool(sol[item]["correct"]) and bool(
                luna[item]["correct"]) for item in ids)
            output["judge_agreement"][benchmark] = {
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "sol_only_correct": sol_only,
                "luna_only_correct": luna_only,
                "agreement": (both_correct + both_wrong) / len(ids),
            }
        else:
            ids = set(judges[args.judges[0]][benchmark])
        selected_usage = [usage[item] for item in ids]
        output["benchmarks"][benchmark]["answer_usage"] = {
            "prompt_tokens": stats([
                float(row.get("api_prompt_tokens", 0)) for row in selected_usage]),
            "completion_tokens": stats([
                float(row.get("completion_tokens", 0)) for row in selected_usage]),
            "reasoning_tokens": stats([
                float(row.get("reasoning_tokens", 0)) for row in selected_usage]),
            "total_tokens": stats([
                float(row.get("total_tokens", 0)) for row in selected_usage]),
            "latency_ms": stats([
                float(row.get("latency_ms", 0)) for row in selected_usage]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        benchmark: {
            judge: {
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "wilson95": result["wilson95"],
                "gap": result["correct_gap_to_target"],
            } for judge, result in values["judges"].items()
        } for benchmark, values in output["benchmarks"].items()
    }, ensure_ascii=False, indent=2))
    print(json.dumps(output["judge_agreement"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
