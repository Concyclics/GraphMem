#!/usr/bin/env python3
"""Summarize a sharded GraphMem full run without changing its artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["longmemeval", "locomo"], required=True)
    parser.add_argument("--judge", type=Path)
    parser.add_argument("--variant", default="hierarchical_hybrid_graph_v4_1_query")
    parser.add_argument("--expected", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(json.loads(line))
    return output


def unique_by_question(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item["question_id"]): item
        for item in items
        if item.get("question_id") is not None
    }


def percentile(values: list[int | float], quantile: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def distribution(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {"sum": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    return {
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "calls": len(calls),
        "cached_input_tokens": sum(int(row.get("prompt_cache_hit_tokens", 0)) for row in calls),
        "uncached_input_tokens": sum(int(row.get("prompt_cache_miss_tokens", 0)) for row in calls),
        "output_tokens": sum(int(row.get("completion_tokens", 0)) for row in calls),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens", 0)) for row in calls),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
        "inferred_breakdown_calls": sum(bool(row.get("breakdown_inferred")) for row in calls),
    }
    totals["accounting_valid"] = all(
        int(row.get("prompt_cache_hit_tokens", 0))
        + int(row.get("prompt_cache_miss_tokens", 0))
        == int(row.get("prompt_tokens", 0))
        and int(row.get("prompt_tokens", 0))
        + int(row.get("completion_tokens", 0))
        == int(row.get("total_tokens", 0))
        for row in calls
    )
    return totals


def judged_correct(row: dict[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    return str(row.get("verdict") or row.get("label") or "").casefold() in {
        "yes", "correct", "true",
    }


def question_group(
    benchmark: str, question_id: str, answer: dict[str, Any], judge: dict[str, Any],
) -> str:
    if benchmark == "locomo":
        raw = judge.get("category", answer.get("locomo_category", answer.get("question_type", "unknown")))
        text = str(raw)
        return text if text.startswith("category_") else f"category_{text}"
    return str(judge.get("question_type") or answer.get("question_type") or "unknown")


def main() -> None:
    args = parse_args()
    variant_dirs = sorted(
        path for path in args.run_root.glob(f"shards/shard_*/{args.variant}")
        if path.is_dir()
    )
    answer_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for directory in variant_dirs:
        answer_rows.extend(rows(directory / "answers.jsonl"))
        stat_rows.extend(rows(directory / "question_stats.jsonl"))
        calls.extend(rows(directory / "llm_calls.jsonl"))
    answers = unique_by_question(answer_rows)
    stats = unique_by_question(stat_rows)
    judge_path = args.judge
    if judge_path and judge_path.is_dir():
        judge_path = judge_path / "auto_eval.jsonl"
    judges = unique_by_question(rows(judge_path)) if judge_path else {}

    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    models = Counter()
    for call in calls:
        if call.get("excluded_from_budget"):
            continue
        stage = str(call.get("stage") or "unknown")
        by_stage[stage].append(call)
        by_question[str(call.get("question_id"))].append(call)
        models[str(call.get("model") or "unknown")] += 1

    query_totals = []
    query_rows = {}
    for question_id in answers:
        question_calls = [
            call for call in by_question.get(question_id, [])
            if str(call.get("stage", "")).startswith("answer_")
        ]
        row_usage = usage(question_calls)
        query_rows[question_id] = row_usage
        query_totals.append(row_usage["total_tokens"])

    build_calls = [
        call for call in calls
        if str(call.get("stage", "")).startswith("build_")
        and not call.get("excluded_from_budget")
    ]
    build_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in build_calls:
        build_by_owner[str(call.get("question_id"))].append(call)
    build_totals = [usage(group)["total_tokens"] for group in build_by_owner.values()]

    judged_ids = sorted(set(answers).intersection(judges))
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for question_id in judged_ids:
        group = question_group(
            args.benchmark, question_id, answers[question_id], judges[question_id],
        )
        group_counts[group]["n"] += 1
        group_counts[group]["correct"] += judged_correct(judges[question_id])
    accuracy_by_group = {
        group: {
            "n": counts["n"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["n"] if counts["n"] else 0.0,
        }
        for group, counts in sorted(group_counts.items())
    }
    correct = sum(judged_correct(judges[question_id]) for question_id in judged_ids)
    retrieval_rows = [stats[question_id] for question_id in answers if question_id in stats]

    summary = {
        "benchmark": args.benchmark,
        "variant": args.variant,
        "models": dict(models),
        "expected_questions": args.expected,
        "answered_questions": len(answers),
        "judged_questions": len(judged_ids),
        "correct_questions": correct,
        "accuracy": correct / len(judged_ids) if judged_ids else None,
        "complete_answer_set": args.expected is None or len(answers) == args.expected,
        "complete_judge_set": args.expected is None or len(judged_ids) == args.expected,
        "accuracy_by_type": accuracy_by_group,
        "token_usage_by_stage": {
            stage: usage(stage_calls) for stage, stage_calls in sorted(by_stage.items())
        },
        "build": {
            "owner_count": len(build_by_owner),
            "tokens_per_owner": distribution(build_totals),
            "usage": usage(build_calls),
        },
        "query": {
            "question_count": len(query_rows),
            "tokens_per_question": distribution(query_totals),
            "over_10000": sum(value > 10_000 for value in query_totals),
            "over_12000": sum(value > 12_000 for value in query_totals),
            "over_13000": sum(value > 13_000 for value in query_totals),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in query_rows.values()),
        },
        "retrieval": {
            "question_count": len(retrieval_rows),
            "answer_session_any_hit_rate": (
                sum(bool(row.get("retrieved_answer_session_hit")) for row in retrieval_rows)
                / len(retrieval_rows) if retrieval_rows else None
            ),
            "answer_session_all_hit_rate": (
                sum(bool(row.get("retrieved_answer_session_all_hit")) for row in retrieval_rows)
                / len(retrieval_rows) if retrieval_rows else None
            ),
            "mean_answer_session_recall": (
                sum(float(row.get("retrieved_answer_session_recall", 0.0)) for row in retrieval_rows)
                / len(retrieval_rows) if retrieval_rows else None
            ),
            "latency_seconds": distribution([
                float(row.get("retrieval_latency_sec", 0.0)) for row in retrieval_rows
            ]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    accuracy = "pending" if summary["accuracy"] is None else f"{100 * summary['accuracy']:.2f}%"
    markdown = [
        f"# {args.benchmark} unified benchmark summary",
        "",
        f"- Variant: `{args.variant}`",
        f"- Answered: {len(answers)}/{args.expected or '?'}",
        f"- Judged: {len(judged_ids)}/{args.expected or '?'}",
        f"- Accuracy: {accuracy}",
        f"- Query mean/P95/max tokens: {summary['query']['tokens_per_question']['mean']:.1f} / "
        f"{summary['query']['tokens_per_question']['p95']} / {summary['query']['tokens_per_question']['max']}",
        f"- Query reasoning tokens: {summary['query']['reasoning_tokens']}",
        "",
        "## Accuracy by type",
        "",
        "| Type | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {group} | {value['correct']} | {value['n']} | {100 * value['accuracy']:.2f}% |"
        for group, value in accuracy_by_group.items()
    )
    markdown.extend(["", "## Token use by stage", "", "| Stage | Calls | Cached input | Uncached input | Output | Total | Reasoning |", "|---|---:|---:|---:|---:|---:|---:|"])
    markdown.extend(
        f"| {stage} | {value['calls']} | {value['cached_input_tokens']} | "
        f"{value['uncached_input_tokens']} | {value['output_tokens']} | "
        f"{value['total_tokens']} | {value['reasoning_tokens']} |"
        for stage, value in summary["token_usage_by_stage"].items()
    )
    (args.output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
