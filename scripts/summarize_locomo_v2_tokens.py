#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"p50": 0, "p95": 0, "max": 0, "mean": 0.0, "sum": 0}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
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
        description="Write per-conversation build and per-question answer token reports."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.data.read_text(encoding="utf-8"))
    case_by_id = {row["question_id"]: row for row in cases}
    stats_by_id = {
        row["question_id"]: row
        for row in _read_jsonl(args.run_dir / "question_stats.jsonl")
    }
    calls = _read_jsonl(args.run_dir / "llm_calls.jsonl")
    calls_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        calls_by_question[str(call.get("question_id"))].append(call)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case["locomo_sample_id"])].append(case)

    conversation_rows = []
    for conversation_id, group in groups.items():
        present = [stats_by_id[row["question_id"]] for row in group if row["question_id"] in stats_by_id]
        owners = [row for row in present if int(row.get("build_total_tokens", 0)) > 0]
        build = owners[0] if owners else {}
        owner_ids = [row["question_id"] for row in owners]
        build_calls = [
            call
            for owner_id in owner_ids
            for call in calls_by_question.get(owner_id, [])
            if str(call.get("stage", "")).startswith("build_")
            and not call.get("excluded_from_budget", False)
        ]
        build_usage = _usage(build_calls)
        build_accounting_valid = (
            len(owners) == 1
            and bool(build_calls)
            and all(_call_usage_valid(call) for call in build_calls)
            and _matches_stats(build, "build", build_usage)
            and build_usage["cache_miss_input_tokens"]
            + build_usage["cache_hit_input_tokens"]
            + build_usage["output_tokens"]
            == build_usage["total_tokens"]
        )
        conversation_rows.append(
            {
                "conversation_id": conversation_id,
                "sample_index": int(group[0]["locomo_sample_index"]),
                "session_count": len(group[0]["haystack_sessions"]),
                "question_count": len(group),
                "completed_question_count": len(present),
                "build_owner_question_ids": owner_ids,
                "build_call_count": len(build_calls),
                "build_cache_miss_input_tokens": int(build.get("build_cache_miss_input_tokens", 0)),
                "build_cache_hit_input_tokens": int(build.get("build_cache_hit_input_tokens", 0)),
                "build_output_tokens": int(build.get("build_output_tokens", 0)),
                "build_total_tokens": int(build.get("build_total_tokens", 0)),
                "build_reasoning_tokens": build_usage["reasoning_tokens"],
                "build_breakdown_inferred_calls": build_usage[
                    "breakdown_inferred_calls"
                ],
                "build_budget_pass": bool(build.get("build_budget_pass", False))
                and build_usage["total_tokens"] <= 300_000,
                "token_accounting_valid": build_accounting_valid,
            }
        )
    conversation_rows.sort(key=lambda row: row["sample_index"])

    question_rows = []
    for case in cases:
        question_id = case["question_id"]
        stats = stats_by_id.get(question_id)
        if stats is None:
            continue
        answer_calls = [
            call
            for call in calls_by_question.get(question_id, [])
            if str(call.get("stage", "")).startswith("answer_")
            and not call.get("excluded_from_budget", False)
        ]
        answer_usage = _usage(answer_calls)
        answer_accounting_valid = (
            len(answer_calls) == 1
            and all(_call_usage_valid(call) for call in answer_calls)
            and _matches_stats(stats, "answer", answer_usage)
            and answer_usage["cache_miss_input_tokens"]
            + answer_usage["cache_hit_input_tokens"]
            + answer_usage["output_tokens"]
            == answer_usage["total_tokens"]
        )
        question_rows.append(
            {
                "question_id": question_id,
                "conversation_id": case["locomo_sample_id"],
                "sample_index": int(case["locomo_sample_index"]),
                "category": int(case["locomo_category"]),
                "answer_call_count": len(answer_calls),
                "answer_cache_miss_input_tokens": int(stats.get("answer_cache_miss_input_tokens", 0)),
                "answer_cache_hit_input_tokens": int(stats.get("answer_cache_hit_input_tokens", 0)),
                "answer_output_tokens": int(stats.get("answer_output_tokens", 0)),
                "answer_total_tokens": int(stats.get("answer_total_tokens", 0)),
                "answer_reasoning_tokens": answer_usage["reasoning_tokens"],
                "answer_breakdown_inferred_calls": answer_usage[
                    "breakdown_inferred_calls"
                ],
                "answer_budget_pass": bool(stats.get("answer_budget_pass", False))
                and answer_usage["total_tokens"] <= 10_000,
                "token_accounting_valid": answer_accounting_valid,
                "retrieved_answer_session_hit": bool(
                    stats.get("retrieved_answer_session_hit", False)
                ),
                "retrieved_answer_session_all_hit": bool(
                    stats.get("retrieved_answer_session_all_hit", False)
                ),
                "retrieved_answer_session_recall": float(
                    stats.get("retrieved_answer_session_recall", 0.0)
                ),
                "retrieval_latency_sec": float(stats.get("retrieval_latency_sec", 0.0)),
                "answer_latency_sec": float(stats.get("answer_latency_sec", 0.0)),
            }
        )

    build_values = [row["build_total_tokens"] for row in conversation_rows]
    answer_values = [row["answer_total_tokens"] for row in question_rows]
    summary = {
        "conversation_count_expected": len(groups),
        "conversation_count_reported": len(conversation_rows),
        "question_count_expected": len(cases),
        "question_count_reported": len(question_rows),
        "build_owner_count": sum(len(row["build_owner_question_ids"]) for row in conversation_rows),
        "build_owner_valid": all(
            len(row["build_owner_question_ids"]) == 1 for row in conversation_rows
        ),
        "build_tokens": _percentiles(build_values),
        "answer_tokens": _percentiles(answer_values),
        "build_token_components": {
            "cache_miss_input_tokens": sum(
                int(row["build_cache_miss_input_tokens"]) for row in conversation_rows
            ),
            "cache_hit_input_tokens": sum(
                int(row["build_cache_hit_input_tokens"]) for row in conversation_rows
            ),
            "output_tokens": sum(
                int(row["build_output_tokens"]) for row in conversation_rows
            ),
        },
        "answer_token_components": {
            "cache_miss_input_tokens": sum(
                int(row["answer_cache_miss_input_tokens"]) for row in question_rows
            ),
            "cache_hit_input_tokens": sum(
                int(row["answer_cache_hit_input_tokens"]) for row in question_rows
            ),
            "output_tokens": sum(
                int(row["answer_output_tokens"]) for row in question_rows
            ),
        },
        "build_budget_limit": 300000,
        "answer_budget_limit": 10000,
        "build_budget_pass": all(row["build_budget_pass"] for row in conversation_rows),
        "answer_budget_pass": all(row["answer_budget_pass"] for row in question_rows)
        and len(question_rows) == len(cases),
        "over_build_budget_conversation_ids": [
            row["conversation_id"] for row in conversation_rows if not row["build_budget_pass"]
        ],
        "over_answer_budget_question_ids": [
            row["question_id"] for row in question_rows if not row["answer_budget_pass"]
        ],
        "build_reasoning_tokens": sum(row["build_reasoning_tokens"] for row in conversation_rows),
        "answer_reasoning_tokens": sum(row["answer_reasoning_tokens"] for row in question_rows),
        "build_breakdown_inferred_calls": sum(
            row["build_breakdown_inferred_calls"] for row in conversation_rows
        ),
        "answer_breakdown_inferred_calls": sum(
            row["answer_breakdown_inferred_calls"] for row in question_rows
        ),
        "token_accounting_valid": all(row["token_accounting_valid"] for row in conversation_rows)
        and all(row["token_accounting_valid"] for row in question_rows),
        "judge_excluded_from_budget": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "conversation_build_tokens.jsonl", conversation_rows)
    _write_jsonl(args.output_dir / "question_answer_tokens.jsonl", question_rows)
    (args.output_dir / "locomo_token_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# LoCoMo GraphMem V3 token report",
        "",
        f"- Conversations: {len(conversation_rows)}/{len(groups)}",
        f"- Questions: {len(question_rows)}/{len(cases)}",
        f"- Build max / P95 / mean: {summary['build_tokens']['max']} / "
        f"{summary['build_tokens']['p95']} / {summary['build_tokens']['mean']:.1f}",
        f"- Answer max / P95 / mean: {summary['answer_tokens']['max']} / "
        f"{summary['answer_tokens']['p95']} / {summary['answer_tokens']['mean']:.1f}",
        f"- Build budget pass: {summary['build_budget_pass']}",
        f"- Answer budget pass: {summary['answer_budget_pass']}",
        f"- Reasoning tokens (build/answer): "
        f"{summary['build_reasoning_tokens']}/{summary['answer_reasoning_tokens']}",
        "",
        "| Conversation | Sessions | Questions | Cache miss input | Cache hit input | Output | Total | ≤300K |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in conversation_rows:
        markdown.append(
            f"| {row['conversation_id']} | {row['session_count']} | {row['question_count']} | "
            f"{row['build_cache_miss_input_tokens']} | {row['build_cache_hit_input_tokens']} | "
            f"{row['build_output_tokens']} | {row['build_total_tokens']} | "
            f"{'yes' if row['build_budget_pass'] else 'no'} |"
        )
    (args.output_dir / "locomo_token_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
