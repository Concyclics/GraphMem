#!/usr/bin/env python3
"""Summarize the frozen V5.9 LongMemEval + LoCoMo end-to-end run."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT.parent / "artifacts/v5_9/full_benchmark_20260809/answers/merged",
    )
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        default=ROOT.parent
        / "artifacts/v5_9/full_benchmark_20260809/graph/recoarsen_manifest.json",
    )
    parser.add_argument(
        "--previous-root",
        type=Path,
        default=ROOT.parent / "artifacts/v5_8/answers_rank/merged",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/report/v5_9/full_benchmark",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def rate(rows: list[dict[str, Any]], predicate) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = [row for row in rows if row.get("has_turn_gold")]
    prompt = [float(row["prompt_tokens"]) for row in rows]
    evidence = [float(row["evidence_tokens"]) for row in rows]
    latency = [float(row["latency_total_ms"]) for row in rows]
    return {
        "questions": len(rows),
        "annotated_questions": len(annotated),
        "prompt_tokens": {
            "mean": mean(prompt),
            "p50": quantile(prompt, 0.50),
            "p95": quantile(prompt, 0.95),
            "max": max(prompt),
            "over_soft_budget": sum(value > 10_000 for value in prompt),
        },
        "evidence_tokens": {
            "mean": mean(evidence),
            "p50": quantile(evidence, 0.50),
            "p95": quantile(evidence, 0.95),
            "max": max(evidence),
        },
        "retrieval_latency_ms": {
            "mean": mean(latency),
            "p50": quantile(latency, 0.50),
            "p95": quantile(latency, 0.95),
            "p99": quantile(latency, 0.99),
        },
        "turn_all_hit": rate(annotated, lambda row: bool(row.get("turn_all_hit"))),
        "turn_recall": mean(float(row["turn_recall"]) for row in annotated),
        "session_all_hit": rate(annotated, lambda row: bool(row.get("session_all_hit"))),
        "closed_form_rate": rate(rows, lambda row: bool(row.get("closed_form"))),
        "certificate_complete_rate": rate(
            rows, lambda row: bool(row.get("certificate_complete"))
        ),
        "budget_relaxed_rate": rate(rows, lambda row: bool(row.get("budget_relaxed"))),
        "visited_nodes_mean": mean(float(row["visited_nodes"]) for row in rows),
        "visited_edges_mean": mean(float(row["visited_edges"]) for row in rows),
    }


def paired_stats(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    old = {str(row["question_id"]): bool(row["correct"]) for row in previous}
    new = {str(row["question_id"]): bool(row["correct"]) for row in current}
    question_ids = sorted(set(old) & set(new))
    new_only = sum(new[key] and not old[key] for key in question_ids)
    old_only = sum(old[key] and not new[key] for key in question_ids)
    discordant = new_only + old_only
    exact_p = 1.0
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(new_only, old_only) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2 * tail)
    sample_count = len(question_ids)
    delta = (new_only - old_only) / sample_count if sample_count else 0.0
    sample_variance = 0.0
    if sample_count > 1:
        sample_variance = (
            (discordant / sample_count) - delta**2
        ) * sample_count / (sample_count - 1)
    standard_error = math.sqrt(sample_variance / sample_count) if sample_count else 0.0
    return {
        "questions": sample_count,
        "previous_correct": sum(old[key] for key in question_ids),
        "current_correct": sum(new[key] for key in question_ids),
        "new_only": new_only,
        "previous_only": old_only,
        "delta": delta,
        "mcnemar_exact_p": exact_p,
        "normal_95ci": [delta - 1.96 * standard_error, delta + 1.96 * standard_error],
    }


def closed_form_audit(
    retrieval: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> dict[str, Any]:
    closed = {
        str(row["dev_question_id"])
        for row in retrieval
        if bool(row.get("closed_form"))
    }
    old = {str(row["question_id"]): bool(row["correct"]) for row in previous}
    new = {str(row["question_id"]): bool(row["correct"]) for row in current}
    question_ids = sorted(closed & set(old) & set(new))
    return {
        "questions": len(question_ids),
        "previous_correct": sum(old[key] for key in question_ids),
        "current_correct": sum(new[key] for key in question_ids),
        "new_only": sum(new[key] and not old[key] for key in question_ids),
        "previous_only": sum(old[key] and not new[key] for key in question_ids),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V5.9 full benchmark audit",
        "",
        "## End-to-end headline",
        "",
        "| Benchmark | Questions | Accuracy | Annotated all-hit | Prompt Token mean / p95 | Retrieval p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("longmemeval", "LongMemEval"), ("locomo", "LoCoMo Cat1--4")):
        row = payload["benchmarks"][key]
        retrieval = row["retrieval"]
        lines.append(
            f"| {label} | {row['accuracy']['question_count']} | "
            f"{row['accuracy']['accuracy']:.2%} | {retrieval['turn_all_hit']:.2%} "
            f"(n={retrieval['annotated_questions']}) | "
            f"{retrieval['prompt_tokens']['mean']:.0f} / "
            f"{retrieval['prompt_tokens']['p95']:.0f} | "
            f"{retrieval['retrieval_latency_ms']['p95']:.1f} ms |"
        )
    lines.extend(["", "## Paired comparison with the frozen V5.8 rank-mandatory run", ""])
    lines.extend([
        "| Benchmark | Previous | Current | Delta | New-only / old-only | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("longmemeval", "LongMemEval"), ("locomo", "LoCoMo")):
        paired = payload["benchmarks"][key]["paired_vs_v5_8"]
        lines.append(
            f"| {label} | {paired['previous_correct']}/{paired['questions']} | "
            f"{paired['current_correct']}/{paired['questions']} | {paired['delta']:+.2%} | "
            f"{paired['new_only']} / {paired['previous_only']} | "
            f"{paired['mcnemar_exact_p']:.4f} |"
        )
    lines.extend([
        "",
        "The accuracy deltas are directional but not statistically significant at 0.05. "
        "The run uses the same local Qwen-30B answer and judge backbone, temperature 0, "
        "and the pinned Mem0/memory-benchmarks judge prompts.",
        "",
        "## Scope and caveats",
        "",
        "- The frozen V5.8 fact graph was recoarsened for all 510 memories; no extractor call was rerun.",
        "- Dense retrieval was disabled, matching the report mechanism-isolation path.",
        "- LoCoMo category 5 is excluded by the pinned memory-benchmarks protocol.",
        "- LoCoMo token-F1 remains format-sensitive and is reported separately from judge accuracy.",
        "- Closed-form questions are harder than the remaining population; their raw accuracy is not a causal bypass ablation.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    retrieval = read_jsonl(args.run_root / "retrieval.jsonl")
    retrieval_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval:
        retrieval_by_benchmark[str(row["benchmark"])].append(row)

    payload: dict[str, Any] = {
        "schema_version": "graphmem-v5.9-full-benchmark-v1",
        "run_root": str(args.run_root),
        "graph": read_json(args.graph_manifest),
        "run_manifest": read_json(args.run_root / "run_manifest.json"),
        "benchmarks": {},
    }
    for key, judge_dir, previous_dir in (
        ("longmemeval", "judge_lme", "judge_lme"),
        ("locomo", "judge_locomo", "judge_locomo"),
    ):
        current_eval = read_jsonl(args.run_root / judge_dir / "auto_eval.jsonl")
        previous_eval = read_jsonl(args.previous_root / previous_dir / "auto_eval.jsonl")
        payload["benchmarks"][key] = {
            "accuracy": read_json(args.run_root / judge_dir / "judge_token_stats.json"),
            "retrieval": retrieval_summary(retrieval_by_benchmark[key]),
            "paired_vs_v5_8": paired_stats(previous_eval, current_eval),
            "closed_form_audit": closed_form_audit(
                retrieval_by_benchmark[key], previous_eval, current_eval
            ),
        }
    payload["benchmarks"]["locomo"]["official_token_f1"] = read_json(
        args.run_root / "locomo_official_f1/official_eval.json"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "longmemeval_accuracy": payload["benchmarks"]["longmemeval"]["accuracy"]["accuracy"],
        "locomo_accuracy": payload["benchmarks"]["locomo"]["accuracy"]["accuracy"],
    }))


if __name__ == "__main__":
    main()
