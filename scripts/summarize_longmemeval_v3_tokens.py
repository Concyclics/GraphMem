#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"p50": 0, "p95": 0, "max": 0, "mean": 0.0, "sum": 0}
    ordered = sorted(values)

    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "sum": sum(ordered),
    }


def _call_usage_valid(call: dict[str, Any]) -> bool:
    hit = int(call.get("prompt_cache_hit_tokens", 0))
    miss = int(call.get("prompt_cache_miss_tokens", 0))
    prompt = int(call.get("prompt_tokens", 0))
    output = int(call.get("completion_tokens", 0))
    total = int(call.get("total_tokens", 0))
    return hit + miss == prompt and prompt + output == total


def _usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cache_miss_input_tokens": sum(
            int(call.get("prompt_cache_miss_tokens", 0)) for call in calls
        ),
        "cache_hit_input_tokens": sum(
            int(call.get("prompt_cache_hit_tokens", 0)) for call in calls
        ),
        "output_tokens": sum(int(call.get("completion_tokens", 0)) for call in calls),
        "total_tokens": sum(int(call.get("total_tokens", 0)) for call in calls),
        "reasoning_tokens": sum(int(call.get("reasoning_tokens", 0)) for call in calls),
        "breakdown_inferred_calls": sum(
            bool(call.get("breakdown_inferred", False)) for call in calls
        ),
    }


def _matches_stats(
    stats: dict[str, Any], prefix: str, usage: dict[str, int]
) -> bool:
    return all(
        int(stats.get(f"{prefix}_{suffix}", -1)) == usage[key]
        for suffix, key in (
            ("cache_miss_input_tokens", "cache_miss_input_tokens"),
            ("cache_hit_input_tokens", "cache_hit_input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit LongMemEval per-question build and answer tokens against raw calls."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-budget", type=int, default=300_000)
    parser.add_argument("--answer-budget", type=int, default=10_000)
    args = parser.parse_args()

    cases = json.loads(args.data.read_text(encoding="utf-8"))
    expected_ids = [str(case["question_id"]) for case in cases]
    stats_by_id = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.run_dir / "question_stats.jsonl")
    }
    calls_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in _read_jsonl(args.run_dir / "llm_calls.jsonl"):
        calls_by_id[str(call.get("question_id"))].append(call)

    rows: list[dict[str, Any]] = []
    for question_id in expected_ids:
        stats = stats_by_id.get(question_id)
        if stats is None:
            continue
        budget_calls = [
            call
            for call in calls_by_id.get(question_id, [])
            if not call.get("excluded_from_budget", False)
        ]
        build_calls = [
            call for call in budget_calls if str(call.get("stage", "")).startswith("build_")
        ]
        answer_calls = [
            call for call in budget_calls if str(call.get("stage", "")).startswith("answer_")
        ]
        unknown_calls = [
            call for call in budget_calls if call not in build_calls and call not in answer_calls
        ]
        build = _usage(build_calls)
        answer = _usage(answer_calls)
        build_valid = (
            bool(build_calls)
            and all(_call_usage_valid(call) for call in build_calls)
            and _matches_stats(stats, "build", build)
            and build["cache_miss_input_tokens"]
            + build["cache_hit_input_tokens"]
            + build["output_tokens"]
            == build["total_tokens"]
        )
        answer_valid = (
            len(answer_calls) == 1
            and all(_call_usage_valid(call) for call in answer_calls)
            and _matches_stats(stats, "answer", answer)
            and answer["cache_miss_input_tokens"]
            + answer["cache_hit_input_tokens"]
            + answer["output_tokens"]
            == answer["total_tokens"]
        )
        rows.append(
            {
                "question_id": question_id,
                "build_call_count": len(build_calls),
                "build_cache_miss_input_tokens": build["cache_miss_input_tokens"],
                "build_cache_hit_input_tokens": build["cache_hit_input_tokens"],
                "build_output_tokens": build["output_tokens"],
                "build_total_tokens": build["total_tokens"],
                "build_reasoning_tokens": build["reasoning_tokens"],
                "build_breakdown_inferred_calls": build["breakdown_inferred_calls"],
                "build_budget_pass": build["total_tokens"] <= args.build_budget,
                "build_accounting_valid": build_valid,
                "answer_call_count": len(answer_calls),
                "answer_cache_miss_input_tokens": answer["cache_miss_input_tokens"],
                "answer_cache_hit_input_tokens": answer["cache_hit_input_tokens"],
                "answer_output_tokens": answer["output_tokens"],
                "answer_total_tokens": answer["total_tokens"],
                "answer_reasoning_tokens": answer["reasoning_tokens"],
                "answer_breakdown_inferred_calls": answer["breakdown_inferred_calls"],
                "answer_budget_pass": answer["total_tokens"] <= args.answer_budget,
                "answer_accounting_valid": answer_valid,
                "unknown_budget_call_count": len(unknown_calls),
            }
        )

    build_values = [int(row["build_total_tokens"]) for row in rows]
    answer_values = [int(row["answer_total_tokens"]) for row in rows]
    summary = {
        "question_count_expected": len(expected_ids),
        "question_count_reported": len(rows),
        "build_tokens": _percentiles(build_values),
        "answer_tokens": _percentiles(answer_values),
        "build_token_components": {
            "cache_miss_input_tokens": sum(
                int(row["build_cache_miss_input_tokens"]) for row in rows
            ),
            "cache_hit_input_tokens": sum(
                int(row["build_cache_hit_input_tokens"]) for row in rows
            ),
            "output_tokens": sum(int(row["build_output_tokens"]) for row in rows),
        },
        "answer_token_components": {
            "cache_miss_input_tokens": sum(
                int(row["answer_cache_miss_input_tokens"]) for row in rows
            ),
            "cache_hit_input_tokens": sum(
                int(row["answer_cache_hit_input_tokens"]) for row in rows
            ),
            "output_tokens": sum(int(row["answer_output_tokens"]) for row in rows),
        },
        "build_budget_limit": args.build_budget,
        "answer_budget_limit": args.answer_budget,
        "build_budget_pass": len(rows) == len(expected_ids)
        and all(bool(row["build_budget_pass"]) for row in rows),
        "answer_budget_pass": len(rows) == len(expected_ids)
        and all(bool(row["answer_budget_pass"]) for row in rows),
        "over_build_budget_question_ids": [
            row["question_id"] for row in rows if not row["build_budget_pass"]
        ],
        "over_answer_budget_question_ids": [
            row["question_id"] for row in rows if not row["answer_budget_pass"]
        ],
        "build_reasoning_tokens": sum(
            int(row["build_reasoning_tokens"]) for row in rows
        ),
        "answer_reasoning_tokens": sum(
            int(row["answer_reasoning_tokens"]) for row in rows
        ),
        "build_breakdown_inferred_calls": sum(
            int(row["build_breakdown_inferred_calls"]) for row in rows
        ),
        "answer_breakdown_inferred_calls": sum(
            int(row["answer_breakdown_inferred_calls"]) for row in rows
        ),
        "token_accounting_valid": len(rows) == len(expected_ids)
        and all(
            bool(row["build_accounting_valid"])
            and bool(row["answer_accounting_valid"])
            and int(row["unknown_budget_call_count"]) == 0
            for row in rows
        ),
        "judge_excluded_from_budget": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "question_tokens.jsonl", rows)
    (args.output_dir / "longmemeval_token_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# LongMemEval GraphMem V3 token report",
        "",
        f"- Questions: {len(rows)}/{len(expected_ids)}",
        f"- Build max / P95 / mean: {summary['build_tokens']['max']} / "
        f"{summary['build_tokens']['p95']} / {summary['build_tokens']['mean']:.1f}",
        f"- Answer max / P95 / mean: {summary['answer_tokens']['max']} / "
        f"{summary['answer_tokens']['p95']} / {summary['answer_tokens']['mean']:.1f}",
        f"- Build budget pass: {summary['build_budget_pass']}",
        f"- Answer budget pass: {summary['answer_budget_pass']}",
        f"- Token accounting valid: {summary['token_accounting_valid']}",
        f"- Reasoning tokens (build/answer): "
        f"{summary['build_reasoning_tokens']}/{summary['answer_reasoning_tokens']}",
        "",
    ]
    (args.output_dir / "longmemeval_token_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
