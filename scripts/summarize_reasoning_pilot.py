#!/usr/bin/env python3
"""Summarize the frozen-prompt GPT-5.6 reasoning pilot and its 93% ceiling."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("qwen_baseline", "luna_low", "luna_medium", "sol_low", "sol_medium")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def nearest(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": nearest(values, 0.50) if values else 0.0,
        "p95": nearest(values, 0.95) if values else 0.0,
        "p99": nearest(values, 0.99) if values else 0.0,
        "max": max(values, default=0.0),
    }


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not total:
        return 0.0, 1.0
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = min(gains, losses)
    mass = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * mass / (2 ** discordant))


def stratum(benchmark: str, row: dict[str, Any]) -> str:
    return str(row.get("question_type") if benchmark == "longmemeval"
               else row.get("category") or row.get("locomo_category"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--judge", default="sol_medium")
    parser.add_argument("--target", type=float, default=0.93)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(
        (args.root / "selection/selection_manifest.json").read_text())
    metadata = keyed(args.root / "selection/metadata_answers.jsonl")
    roles = keyed(args.root / "selection/baseline_verdicts.jsonl")
    judges = {
        arm: {
            benchmark: keyed(
                args.root / f"judge/{args.judge}/{arm}/{suffix}/auto_eval.jsonl")
            for benchmark, suffix in (("longmemeval", "lme"),
                                      ("locomo", "locomo"))
        } for arm in ARMS
    }
    answer_usage = {
        arm: keyed(args.root / f"answer/{arm}/answer_usage.jsonl")
        for arm in ARMS if arm != "qwen_baseline"
    }

    output: dict[str, Any] = {
        "schema_version": "graphmem-gpt56-reasoning-pilot-summary-v1",
        "primary_judge": args.judge,
        "target_accuracy": args.target,
        "selection_manifest": str(
            args.root / "selection/selection_manifest.json"),
        "benchmarks": {},
    }
    for benchmark in ("longmemeval", "locomo"):
        selected_ids = [
            question_id for question_id, row in metadata.items()
            if str(row.get("benchmark")) == benchmark]
        base = judges["qwen_baseline"][benchmark]
        if set(base) != set(selected_ids):
            raise ValueError(f"baseline judge coverage mismatch for {benchmark}")
        benchmark_summary: dict[str, Any] = {"arms": {}}
        full = selection["benchmarks"][benchmark]
        full_questions = int(full["full_questions"])
        target_correct = math.ceil(args.target * full_questions)
        for arm in ARMS:
            verdicts = judges[arm][benchmark]
            if set(verdicts) != set(selected_ids):
                raise ValueError(f"{arm}/{benchmark} judge coverage mismatch")
            gains = sum(
                not bool(base[question_id]["correct"])
                and bool(verdicts[question_id]["correct"])
                for question_id in selected_ids)
            losses = sum(
                bool(base[question_id]["correct"])
                and not bool(verdicts[question_id]["correct"])
                for question_id in selected_ids)
            projected_correct = 0.0
            projected_low = 0.0
            projected_high = 0.0
            by_stratum: dict[str, Any] = {}
            for name, counts in full["full_by_stratum"].items():
                ids = [question_id for question_id in selected_ids
                       if stratum(
                           benchmark,
                           {**roles[question_id], **metadata[question_id]})
                       == name]
                wrong_ids = [question_id for question_id in ids
                             if roles[question_id]["pilot_role"] == "baseline_wrong"]
                sentinel_ids = [
                    question_id for question_id in ids
                    if roles[question_id]["pilot_role"]
                    == "baseline_correct_sentinel"]
                rescued = sum(bool(verdicts[item]["correct"]) for item in wrong_ids)
                retained = sum(bool(verdicts[item]["correct"])
                               for item in sentinel_ids)
                rescue_rate = rescued / len(wrong_ids) if wrong_ids else 0.0
                retention_rate = (retained / len(sentinel_ids)
                                  if sentinel_ids else 0.0)
                retention_low, retention_high = wilson(
                    retained, len(sentinel_ids))
                full_wrong = int(counts["baseline_wrong"])
                full_correct = int(counts["baseline_correct"])
                projected_correct += (
                    full_wrong * rescue_rate + full_correct * retention_rate)
                projected_low += (
                    full_wrong * rescue_rate + full_correct * retention_low)
                projected_high += (
                    full_wrong * rescue_rate + full_correct * retention_high)
                by_stratum[name] = {
                    "full_baseline_wrong": full_wrong,
                    "full_baseline_correct": full_correct,
                    "wrong_rescued": rescued,
                    "wrong_evaluated": len(wrong_ids),
                    "rescue_rate": rescue_rate,
                    "correct_sentinels_retained": retained,
                    "correct_sentinels_evaluated": len(sentinel_ids),
                    "retention_rate": retention_rate,
                    "retention_wilson95": [retention_low, retention_high],
                }
            pilot_correct = sum(bool(verdicts[item]["correct"])
                                for item in selected_ids)
            payload: dict[str, Any] = {
                "pilot_questions": len(selected_ids),
                "pilot_correct": pilot_correct,
                "pilot_accuracy": pilot_correct / len(selected_ids),
                "paired_vs_qwen": {
                    "gains": gains,
                    "losses": losses,
                    "net": gains - losses,
                    "mcnemar_exact_p": exact_mcnemar(gains, losses),
                },
                "full_projection": {
                    "questions": full_questions,
                    "projected_correct": projected_correct,
                    "projected_accuracy": projected_correct / full_questions,
                    "sentinel_sampling_wilson95_accuracy": [
                        projected_low / full_questions,
                        projected_high / full_questions],
                    "target_correct": target_correct,
                    "target_accuracy": args.target,
                    "projected_correct_gap_to_target": (
                        projected_correct - target_correct),
                    "projected_reaches_target": projected_correct >= target_correct,
                },
                "by_stratum": by_stratum,
            }
            if arm in answer_usage:
                usage = [answer_usage[arm][item] for item in selected_ids]
                payload["answer_cost"] = {
                    "total_tokens": stats([
                        float(row.get("total_tokens", 0)) for row in usage]),
                    "reasoning_tokens": stats([
                        float(row.get("reasoning_tokens", 0)) for row in usage]),
                    "latency_ms": stats([
                        float(row.get("latency_ms", 0)) for row in usage]),
                }
            benchmark_summary["arms"][arm] = payload
        output["benchmarks"][benchmark] = benchmark_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    concise = {
        benchmark: {
            arm: {
                "pilot_accuracy": values["pilot_accuracy"],
                "projected_accuracy": values["full_projection"][
                    "projected_accuracy"],
                "projected_95": values["full_projection"][
                    "sentinel_sampling_wilson95_accuracy"],
                "gains": values["paired_vs_qwen"]["gains"],
                "losses": values["paired_vs_qwen"]["losses"],
            } for arm, values in result["arms"].items()
        } for benchmark, result in output["benchmarks"].items()
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
