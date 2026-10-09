#!/usr/bin/env python3
"""Summarize a fixed-answerer/fixed-judge Mem0 versus GraphMem comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem0-answer-manifest", type=Path, required=True)
    parser.add_argument("--mem0-lme-answers", type=Path, required=True)
    parser.add_argument("--mem0-locomo-answers", type=Path, required=True)
    parser.add_argument("--mem0-lme-verdicts", type=Path, required=True)
    parser.add_argument("--mem0-lme-stats", type=Path, required=True)
    parser.add_argument("--mem0-locomo-verdicts", type=Path, required=True)
    parser.add_argument("--mem0-locomo-stats", type=Path, required=True)
    parser.add_argument("--graphmem-answer-manifest", type=Path, required=True)
    parser.add_argument("--graphmem-answers", type=Path, required=True)
    parser.add_argument("--graphmem-lme-verdicts", type=Path, required=True)
    parser.add_argument("--graphmem-lme-stats", type=Path, required=True)
    parser.add_argument("--graphmem-locomo-verdicts", type=Path, required=True)
    parser.add_argument("--graphmem-locomo-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-improvement-target", type=float, default=0.20)
    parser.add_argument("--mem0-label", default="Mem0 top-200")
    parser.add_argument("--graphmem-label", default="GraphMem 64-turn")
    parser.add_argument(
        "--graphmem-retrieval-label", default="V5.63 selective 64-turn"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - margin, center + margin]


def exact_mcnemar(graphmem_wins: int, mem0_wins: int) -> float:
    discordant = graphmem_wins + mem0_wins
    if discordant == 0:
        return 1.0
    tail = min(graphmem_wins, mem0_wins)
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def validate_contract(
    mem0_manifest: dict[str, Any],
    graphmem_manifest: dict[str, Any],
    mem0_stats: dict[str, Any],
    graphmem_stats: dict[str, Any],
    benchmark: str,
) -> None:
    for name, manifest in (("Mem0", mem0_manifest), ("GraphMem", graphmem_manifest)):
        if str(manifest.get("answer_model")) != "gpt-5.6-luna":
            raise RuntimeError(f"{name} {benchmark} answer model is not gpt-5.6-luna")
        if str(manifest.get("answer_reasoning_effort")) != "max":
            raise RuntimeError(f"{name} {benchmark} answer effort is not max")
    for name, stats in (("Mem0", mem0_stats), ("GraphMem", graphmem_stats)):
        if str(stats.get("model")) != "gpt-5.6-luna":
            raise RuntimeError(f"{name} {benchmark} judge model is not gpt-5.6-luna")
        if str(stats.get("reasoning_effort")) != "medium":
            raise RuntimeError(f"{name} {benchmark} judge effort is not medium")
        if float(stats.get("temperature")) != 0.0 or int(stats.get("seed")) != 0:
            raise RuntimeError(f"{name} {benchmark} judge is not temperature=0, seed=0")


def summarize_benchmark(
    benchmark: str,
    mem0_verdict_rows: list[dict[str, Any]],
    graphmem_verdict_rows: list[dict[str, Any]],
    mem0_answer_rows: dict[str, dict[str, Any]],
    graphmem_answer_rows: dict[str, dict[str, Any]],
    relative_target: float,
) -> dict[str, Any]:
    mem0 = {str(row["question_id"]): bool(row["correct"]) for row in mem0_verdict_rows}
    graphmem = {
        str(row["question_id"]): bool(row["correct"])
        for row in graphmem_verdict_rows
    }
    if set(mem0) != set(graphmem):
        raise RuntimeError(
            f"{benchmark} verdict IDs differ: Mem0={len(mem0)} GraphMem={len(graphmem)}"
        )
    ids = sorted(mem0)
    if not set(ids).issubset(mem0_answer_rows) or not set(ids).issubset(graphmem_answer_rows):
        raise RuntimeError(f"{benchmark} answer metadata does not cover all verdicts")
    mem0_correct = sum(mem0.values())
    graphmem_correct = sum(graphmem.values())
    total = len(ids)
    mem0_accuracy = mem0_correct / total
    graphmem_accuracy = graphmem_correct / total
    delta = graphmem_accuracy - mem0_accuracy
    relative = delta / mem0_accuracy if mem0_accuracy else math.inf
    graphmem_wins = sum(graphmem[qid] and not mem0[qid] for qid in ids)
    mem0_wins = sum(mem0[qid] and not graphmem[qid] for qid in ids)
    both_correct = sum(mem0[qid] and graphmem[qid] for qid in ids)
    both_wrong = total - graphmem_wins - mem0_wins - both_correct

    strata: dict[str, dict[str, int]] = {}
    for qid in ids:
        row = mem0_answer_rows[qid]
        stratum = str(row.get("stratum") or row.get("question_type") or row.get("category") or "unknown")
        bucket = strata.setdefault(
            stratum,
            {"questions": 0, "mem0_correct": 0, "graphmem_correct": 0},
        )
        bucket["questions"] += 1
        bucket["mem0_correct"] += int(mem0[qid])
        bucket["graphmem_correct"] += int(graphmem[qid])
    by_stratum: dict[str, Any] = {}
    for stratum, bucket in sorted(strata.items()):
        count = bucket["questions"]
        m_acc = bucket["mem0_correct"] / count
        g_acc = bucket["graphmem_correct"] / count
        by_stratum[stratum] = {
            **bucket,
            "mem0_accuracy": m_acc,
            "graphmem_accuracy": g_acc,
            "delta_percentage_points": 100 * (g_acc - m_acc),
            "relative_improvement": ((g_acc - m_acc) / m_acc if m_acc else None),
        }
    return {
        "questions": total,
        "mem0": {
            "correct": mem0_correct,
            "accuracy": mem0_accuracy,
            "wilson95": wilson(mem0_correct, total),
        },
        "graphmem": {
            "correct": graphmem_correct,
            "accuracy": graphmem_accuracy,
            "wilson95": wilson(graphmem_correct, total),
        },
        "uplift": {
            "delta_percentage_points": 100 * delta,
            "relative_improvement": relative,
            "relative_improvement_percent": 100 * relative,
            "target_relative_improvement": relative_target,
            "retains_target_relative_improvement": relative >= relative_target,
            "error_reduction": (
                delta / (1 - mem0_accuracy) if mem0_accuracy < 1 else None
            ),
        },
        "paired": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "graphmem_only_correct": graphmem_wins,
            "mem0_only_correct": mem0_wins,
            "mcnemar_exact_two_sided_p": exact_mcnemar(graphmem_wins, mem0_wins),
        },
        "by_stratum": by_stratum,
    }


def main() -> None:
    args = parse_args()
    mem0_manifest = read_json(args.mem0_answer_manifest)
    graphmem_manifest = read_json(args.graphmem_answer_manifest)
    mem0_answer_sources = (args.mem0_lme_answers, args.mem0_locomo_answers)
    mem0_answers = {
        str(row["question_id"]): row
        for path in mem0_answer_sources
        for row in read_jsonl(path)
    }
    graphmem_answers = {
        str(row["question_id"]): row for row in read_jsonl(args.graphmem_answers)
    }
    paths = {
        "longmemeval": {
            "mem0_verdicts": args.mem0_lme_verdicts,
            "mem0_stats": args.mem0_lme_stats,
            "graphmem_verdicts": args.graphmem_lme_verdicts,
            "graphmem_stats": args.graphmem_lme_stats,
        },
        "locomo": {
            "mem0_verdicts": args.mem0_locomo_verdicts,
            "mem0_stats": args.mem0_locomo_stats,
            "graphmem_verdicts": args.graphmem_locomo_verdicts,
            "graphmem_stats": args.graphmem_locomo_stats,
        },
    }
    benchmarks: dict[str, Any] = {}
    for benchmark, source in paths.items():
        mem0_stats = read_json(source["mem0_stats"])
        graphmem_stats = read_json(source["graphmem_stats"])
        validate_contract(
            mem0_manifest, graphmem_manifest, mem0_stats, graphmem_stats, benchmark
        )
        benchmarks[benchmark] = summarize_benchmark(
            benchmark,
            read_jsonl(source["mem0_verdicts"]),
            read_jsonl(source["graphmem_verdicts"]),
            mem0_answers,
            graphmem_answers,
            args.relative_improvement_target,
        )

    mem0_micro_correct = sum(row["mem0"]["correct"] for row in benchmarks.values())
    graphmem_micro_correct = sum(
        row["graphmem"]["correct"] for row in benchmarks.values()
    )
    micro_questions = sum(row["questions"] for row in benchmarks.values())
    mem0_micro_accuracy = mem0_micro_correct / micro_questions
    graphmem_micro_accuracy = graphmem_micro_correct / micro_questions
    micro_delta = graphmem_micro_accuracy - mem0_micro_accuracy
    mem0_macro_accuracy = sum(
        row["mem0"]["accuracy"] for row in benchmarks.values()
    ) / len(benchmarks)
    graphmem_macro_accuracy = sum(
        row["graphmem"]["accuracy"] for row in benchmarks.values()
    ) / len(benchmarks)
    macro_delta = graphmem_macro_accuracy - mem0_macro_accuracy
    aggregate = {
        "micro": {
            "questions": micro_questions,
            "mem0_correct": mem0_micro_correct,
            "graphmem_correct": graphmem_micro_correct,
            "mem0_accuracy": mem0_micro_accuracy,
            "graphmem_accuracy": graphmem_micro_accuracy,
            "delta_percentage_points": 100 * micro_delta,
            "relative_improvement": micro_delta / mem0_micro_accuracy,
            "error_reduction": micro_delta / (1 - mem0_micro_accuracy),
        },
        "macro_benchmark_average": {
            "mem0_accuracy": mem0_macro_accuracy,
            "graphmem_accuracy": graphmem_macro_accuracy,
            "delta_percentage_points": 100 * macro_delta,
            "relative_improvement": macro_delta / mem0_macro_accuracy,
        },
        "benchmarks_retaining_relative_target": sum(
            row["uplift"]["retains_target_relative_improvement"]
            for row in benchmarks.values()
        ),
        "benchmark_count": len(benchmarks),
        "all_benchmarks_retain_relative_target": all(
            row["uplift"]["retains_target_relative_improvement"]
            for row in benchmarks.values()
        ),
    }

    manifest = {
        "schema_version": "mem0-graphmem-luna-upper-bound-v1",
        "protocol": {
            "memory_build_model": "Qwen3-30B",
            "memory_and_retrieval_frozen": True,
            "mem0_retrieval": mem0_manifest.get("retrieval_setting"),
            "graphmem_retrieval": args.graphmem_retrieval_label,
            "answer_model": "gpt-5.6-luna",
            "answer_reasoning_effort": "max",
            "judge_model": "gpt-5.6-luna",
            "judge_reasoning_effort": "medium",
            "judge_temperature": 0.0,
            "judge_seed": 0,
        },
        "benchmarks": benchmarks,
        "aggregate": aggregate,
        "sources": {
            "mem0_answer_manifest": {
                "path": str(args.mem0_answer_manifest),
                "sha256": sha256(args.mem0_answer_manifest),
            },
            "graphmem_answer_manifest": {
                "path": str(args.graphmem_answer_manifest),
                "sha256": sha256(args.graphmem_answer_manifest),
            },
            "mem0_lme_answers": {
                "path": str(args.mem0_lme_answers),
                "sha256": sha256(args.mem0_lme_answers),
            },
            "mem0_locomo_answers": {
                "path": str(args.mem0_locomo_answers),
                "sha256": sha256(args.mem0_locomo_answers),
            },
            **{
                f"{benchmark}_{name}": {"path": str(path), "sha256": sha256(path)}
                for benchmark, source in paths.items()
                for name, path in source.items()
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"| Benchmark | {args.mem0_label} | {args.graphmem_label} | Δ (pp) | Relative uplift | ≥ target | Paired wins/losses |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for benchmark, result in benchmarks.items():
        lines.append(
            "| {name} | {m:.2f}% | {g:.2f}% | {delta:+.2f} | {relative:+.2f}% | {target} | {wins}/{losses} |".format(
                name="LongMemEval" if benchmark == "longmemeval" else "LoCoMo",
                m=100 * result["mem0"]["accuracy"],
                g=100 * result["graphmem"]["accuracy"],
                delta=result["uplift"]["delta_percentage_points"],
                relative=result["uplift"]["relative_improvement_percent"],
                target="yes" if result["uplift"]["retains_target_relative_improvement"] else "no",
                wins=result["paired"]["graphmem_only_correct"],
                losses=result["paired"]["mem0_only_correct"],
            )
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(benchmarks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
