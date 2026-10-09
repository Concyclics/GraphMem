#!/usr/bin/env python3
"""Summarize the paired hard64 semantic-build experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
ARMS = ("qwen_control", "luna_none", "luna_low")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else 0.0


def wilson(correct: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row.get("correct")) for row in rows)
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "wilson95": wilson(correct, total),
    }


def exact_mcnemar(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * (0.5 ** discordant)
        for value in range(0, min(gains, regressions) + 1))
    return min(1.0, 2 * tail)


def paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired verdict question IDs differ")
    gains = sum(not left[q] and right[q] for q in left)
    regressions = sum(left[q] and not right[q] for q in left)
    stable_correct = sum(left[q] and right[q] for q in left)
    stable_wrong = sum(not left[q] and not right[q] for q in left)
    return {
        "gains": gains,
        "regressions": regressions,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "net_correct": gains - regressions,
        "accuracy_delta_pp": 100 * (gains - regressions) / len(left),
        "mcnemar_exact_p": exact_mcnemar(gains, regressions),
    }


def nearest(values: Iterable[float]) -> dict[str, Any]:
    rows = sorted(float(value) for value in values)
    def at(p: float) -> float:
        return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0.0
    return {
        "count": len(rows), "mean": mean(rows), "p50": at(0.50),
        "p95": at(0.95), "max": max(rows, default=0.0),
        "percentile_method": "nearest_rank",
    }


def load_verdicts(root: Path, arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = root / arm / "judge_luna_medium"
    lme = read_jsonl(base / "lme/auto_eval.jsonl")
    locomo = read_jsonl(base / "locomo/auto_eval.jsonl")
    if len(lme) != 32 or len(locomo) != 32:
        raise ValueError(f"{arm}: expected 32+32 verdicts, got {len(lme)}+{len(locomo)}")
    rows = [dict(row, benchmark="longmemeval") for row in lme]
    rows += [dict(row, benchmark="locomo") for row in locomo]
    by_stratum: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("question_type")) if row["benchmark"] == "longmemeval"
               else f"category_{row.get('category')}")
        grouped[f"{row['benchmark']}:{key}"].append(row)
    for key, group in sorted(grouped.items()):
        by_stratum[key] = accuracy(group)
    return rows, {
        "overall": accuracy(rows),
        "longmemeval": accuracy(lme),
        "locomo": accuracy(locomo),
        "by_stratum": by_stratum,
    }


def build_summary(report_path: Path, memory_ids: set[str]) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"]
            if str(row["memory_id"]) in memory_ids]
    if len(rows) != len(memory_ids):
        raise ValueError(
            f"{report_path}: build rows {len(rows)} != memories {len(memory_ids)}")
    ledger_by_memory = {
        str(row["memory_id"]): row for row in payload.get("token_ledger", ())
        if str(row.get("memory_id")) in memory_ids}
    ledger = ([ledger_by_memory[memory_id] for memory_id in sorted(memory_ids)]
              if len(ledger_by_memory) == len(memory_ids) else rows)
    quality = [dict(row.get("build_quality") or {}) for row in rows]
    warm_report = report_path.with_name("build_report_semantic_warm.json")
    warm_wall_minutes = None
    if warm_report.exists() and warm_report != report_path:
        warm_wall_minutes = json.loads(warm_report.read_text(
            encoding="utf-8")).get("summary", {}).get("wall_minutes")
    return {
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "memories": len(rows),
        "input_tokens": nearest(row.get("input_tokens", 0) for row in ledger),
        "output_tokens": nearest(row.get("output_tokens", 0) for row in ledger),
        "reasoning_tokens": nearest(row.get("reasoning_tokens", 0) for row in ledger),
        "total_tokens": nearest(
            row.get("total_tokens", row.get("tokens", 0)) for row in ledger),
        "sum_total_tokens": sum(int(
            row.get("total_tokens", row.get("tokens", 0))) for row in ledger),
        "nodes": nearest(row.get("nodes", 0) for row in rows),
        "edges": nearest(row.get("edges", 0) for row in rows),
        "per_memory_seconds": nearest(row.get("seconds", 0) for row in rows),
        "extraction": {
            "scenes": sum(int(row.get("extraction_scenes", 0)) for row in quality),
            "success_scenes": sum(int(row.get("extraction_success_scenes", 0)) for row in quality),
            "fallback_scenes": sum(int(row.get("extraction_fallback_scenes", 0)) for row in quality),
            "retry_calls": sum(int(row.get("extraction_retry_calls", 0)) for row in quality),
            "budget_degraded_memories": sum(bool(row.get("budget_degraded")) for row in quality),
            "budget_skipped_scenes": sum(int(row.get("budget_skipped_scenes", 0)) for row in quality),
        },
        "declared_reasoning_effort": payload.get("summary", {}).get(
            "llm_reasoning_effort", "none"),
        "wall_minutes": payload.get("summary", {}).get("wall_minutes"),
        "semantic_warm_wall_minutes": warm_wall_minutes,
        "token_gate_violations": payload.get("summary", {}).get(
            "token_gate_violations", []),
    }


def retrieval_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 64:
        raise ValueError(f"{path}: expected 64 retrieval rows, got {len(rows)}")
    result: dict[str, Any] = {}
    for benchmark in ("all", "longmemeval", "locomo"):
        selected = (rows if benchmark == "all" else
                    [row for row in rows if row.get("benchmark") == benchmark])
        gold = [row for row in selected if bool(row.get("has_turn_gold"))]
        signal_counts: dict[str, int] = defaultdict(int)
        for row in selected:
            for signal, count in (row.get("traversed_relation_signals") or {}).items():
                signal_counts[str(signal)] += int(count)
        result[benchmark] = {
            "questions": len(selected),
            "packed_turns_mean": mean(row.get("packed_turns", 0) for row in selected),
            "evidence_tokens_mean": mean(row.get("evidence_tokens", 0) for row in selected),
            "visited_nodes_mean": mean(row.get("visited_nodes", 0) for row in selected),
            "visited_edges_mean": mean(row.get("visited_edges", 0) for row in selected),
            "retrieval_latency_ms_mean": mean(row.get("latency_total_ms", 0) for row in selected),
            "gold_labeled_questions": len(gold),
            "turn_recall_mean": mean(row.get("turn_recall", 0) for row in gold),
            "turn_precision_mean": mean(row.get("turn_precision", 0) for row in gold),
            "turn_f1_mean": mean(row.get("turn_f1", 0) for row in gold),
            "turn_all_hit_rate": mean(bool(row.get("turn_all_hit")) for row in gold),
            "candidate_turn_recall_mean": mean(row.get("candidate_turn_recall", 0) for row in gold),
            "candidate_turn_precision_mean": mean(row.get("candidate_turn_precision", 0) for row in gold),
            "traversed_relation_signals": dict(sorted(signal_counts.items())),
        }
    return result


def prepared_overlap(root: Path, left: str, right: str) -> dict[str, Any]:
    def load(arm: str) -> dict[str, dict[str, Any]]:
        return {str(row["question_id"]): row for row in read_jsonl(
            root / arm / "prepare/prepared_answers.jsonl")}
    first, second = load(left), load(right)
    if set(first) != set(second):
        raise ValueError(f"prepared IDs differ: {left} vs {right}")
    jaccards = []
    prompt_equal = 0
    for question_id in first:
        a = set(first[question_id].get("evidence_turn_ids") or ())
        b = set(second[question_id].get("evidence_turn_ids") or ())
        jaccards.append(len(a & b) / len(a | b) if a | b else 1.0)
        prompt_equal += (first[question_id].get("prompt_payload_hash")
                         == second[question_id].get("prompt_payload_hash"))
    return {
        "questions": len(first),
        "evidence_jaccard_mean": mean(jaccards),
        "identical_prompt_payloads": prompt_equal,
    }


def graph_checksums(db_path: Path, memory_ids: set[str]) -> dict[str, str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT memory_id,graph_checksum FROM graph_versions").fetchall()
    result = {str(memory_id): str(checksum) for memory_id, checksum in rows
              if str(memory_id) in memory_ids}
    if len(result) != len(memory_ids):
        raise ValueError(f"{db_path}: missing graph checksums")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    memory_ids = set((args.root / "selection/memory_ids.txt").read_text(
        encoding="utf-8").split())
    question_ids = set((args.root / "selection/question_ids.txt").read_text(
        encoding="utf-8").split())
    if len(memory_ids) != 42 or len(question_ids) != 64:
        raise ValueError("selection must contain 42 memories and 64 questions")

    verdict_rows: dict[str, list[dict[str, Any]]] = {}
    accuracy_rows: dict[str, Any] = {}
    for arm in ARMS:
        verdict_rows[arm], accuracy_rows[arm] = load_verdicts(args.root, arm)
    verdict_maps = {
        arm: {str(row["question_id"]): bool(row.get("correct"))
              for row in rows}
        for arm, rows in verdict_rows.items()
    }
    pairwise: dict[str, Any] = {}
    for left, right in (("qwen_control", "luna_none"),
                        ("qwen_control", "luna_low"),
                        ("luna_none", "luna_low")):
        pairwise[f"{left}_to_{right}"] = paired(
            verdict_maps[left], verdict_maps[right])

    control_report = WORKSPACE / "artifacts/report/v5_57/full/build_report.json"
    build = {
        "qwen_control": build_summary(control_report, memory_ids),
        "luna_none": build_summary(args.root / "luna_none/build_report.json", memory_ids),
        "luna_low": build_summary(args.root / "luna_low/build_report.json", memory_ids),
    }
    graph_paths = {
        "qwen_control": WORKSPACE / "artifacts/report/v5_57/full/graph/graphmem.sqlite",
        "luna_none": args.root / "luna_none/graph/graphmem.sqlite",
        "luna_low": args.root / "luna_low/graph/graphmem.sqlite",
    }
    checksums = {
        arm: graph_checksums(path, memory_ids) for arm, path in graph_paths.items()}
    checksum_comparison = {}
    for left, right in (("qwen_control", "luna_none"),
                        ("qwen_control", "luna_low"),
                        ("luna_none", "luna_low")):
        equal = sum(checksums[left][memory_id] == checksums[right][memory_id]
                    for memory_id in memory_ids)
        checksum_comparison[f"{left}_vs_{right}"] = {
            "equal": equal, "different": len(memory_ids) - equal}

    retrieval = {
        arm: retrieval_summary(args.root / arm / "prepare/retrieval.jsonl")
        for arm in ARMS}
    prompt_overlap = {
        f"{left}_vs_{right}": prepared_overlap(args.root, left, right)
        for left, right in (("qwen_control", "luna_none"),
                            ("qwen_control", "luna_low"),
                            ("luna_none", "luna_low"))}
    answer_usage = {
        arm: json.loads((args.root / arm / "answer_luna_high/run_manifest.json").read_text(
            encoding="utf-8"))["api_tokens"] for arm in ARMS}

    summary = {
        "schema_version": "graphmem-luna-build-hard64-summary-v1",
        "experiment": {
            "questions": 64, "memories": 42,
            "answer_model": "gpt-5.6-luna",
            "answer_reasoning_effort": "high",
            "judge_model": "gpt-5.6-luna",
            "judge_reasoning_effort": "medium",
            "controlled_query_and_prompt_policy": "V5.63 64-turn",
            "selection_manifest": str(args.root / "selection/selection_manifest.json"),
            "selection_manifest_sha256": sha256(
                args.root / "selection/selection_manifest.json"),
        },
        "accuracy": accuracy_rows,
        "paired_transitions": pairwise,
        "build": build,
        "graph_checksum_comparison": checksum_comparison,
        "retrieval": retrieval,
        "prompt_and_evidence_overlap": prompt_overlap,
        "answer_usage": answer_usage,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        "# Luna 构建质量 hard64 对照", "",
        "回答统一为 Luna-high，Judge 统一为 Luna-medium；仅替换语义构建模型/推理档位。",
        "",
        "| 构建臂 | LME | LoCoMo | 总体 | 构建 Token/Memory | Reasoning Token/Memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        acc = accuracy_rows[arm]
        costs = build[arm]
        lines.append(
            f"| {arm} | {acc['longmemeval']['accuracy']:.1%} | "
            f"{acc['locomo']['accuracy']:.1%} | {acc['overall']['accuracy']:.1%} | "
            f"{costs['total_tokens']['mean']:.0f} | "
            f"{costs['reasoning_tokens']['mean']:.0f} |")
    lines += ["", "## 相对当前 Qwen 构建", ""]
    for arm in ("luna_none", "luna_low"):
        row = pairwise[f"qwen_control_to_{arm}"]
        lines.append(
            f"- {arm}: +{row['gains']} / -{row['regressions']}，净变化 "
            f"{row['accuracy_delta_pp']:+.2f} pp，McNemar p={row['mcnemar_exact_p']:.4f}。")
    lines += ["", f"完整机器可读结果：`{args.output}`", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accuracy": {arm: accuracy_rows[arm]["overall"] for arm in ARMS},
        "paired_transitions": pairwise,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
