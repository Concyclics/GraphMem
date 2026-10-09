#!/usr/bin/env python3
"""Summarize the paired full Qwen-build vs Luna-none-build experiment."""
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


BENCHMARK_SIZES = {"longmemeval": 500, "locomo": 1540}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else 0.0


def nearest(values: Iterable[float]) -> dict[str, float | int | str]:
    rows = sorted(float(value) for value in values)

    def at(p: float) -> float:
        return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0.0

    return {
        "count": len(rows), "mean": mean(rows), "p50": at(0.50),
        "p95": at(0.95), "p99": at(0.99),
        "max": max(rows, default=0.0), "percentile_method": "nearest_rank",
    }


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


def accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    correct = sum(bool(row.get("correct")) for row in selected)
    return {
        "correct": correct,
        "total": len(selected),
        "accuracy": correct / len(selected) if selected else 0.0,
        "wilson95": wilson(correct, len(selected)),
    }


def exact_mcnemar(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) * 0.5 ** discordant
               for value in range(min(gains, regressions) + 1))
    return min(1.0, 2 * tail)


def paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    if not left or set(left) != set(right):
        raise ValueError("paired verdict question IDs differ or are empty")
    gains = sum(not left[item] and right[item] for item in left)
    regressions = sum(left[item] and not right[item] for item in left)
    return {
        "questions": len(left),
        "gains": gains,
        "regressions": regressions,
        "stable_correct": sum(left[item] and right[item] for item in left),
        "stable_wrong": sum(not left[item] and not right[item] for item in left),
        "net_correct": gains - regressions,
        "accuracy_delta_pp": 100 * (gains - regressions) / len(left),
        "mcnemar_exact_p": exact_mcnemar(gains, regressions),
    }


def load_arm_verdicts(root: Path, arm: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    all_rows: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for benchmark, suffix in (("longmemeval", "lme"), ("locomo", "locomo")):
        path = root / arm / f"judge/luna_medium/{suffix}/auto_eval.jsonl"
        rows = read_jsonl(path)
        if len(rows) != BENCHMARK_SIZES[benchmark]:
            raise ValueError(
                f"{arm}/{benchmark}: expected {BENCHMARK_SIZES[benchmark]}, got {len(rows)}")
        summary[benchmark] = accuracy(rows)
        for row in rows:
            question_id = str(row["question_id"])
            if question_id in all_rows:
                raise ValueError(f"duplicate verdict question ID: {question_id}")
            enriched = dict(row, benchmark=benchmark)
            all_rows[question_id] = enriched
            stratum = (str(row.get("question_type")) if benchmark == "longmemeval"
                       else f"category_{row.get('category')}")
            by_stratum[f"{benchmark}:{stratum}"].append(enriched)
    summary["overall"] = accuracy(all_rows.values())
    summary["by_stratum"] = {
        key: accuracy(rows) for key, rows in sorted(by_stratum.items())}
    return all_rows, summary


def paired_breakdown(
        control: dict[str, dict[str, Any]],
        candidate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("control and candidate verdict coverage differs")
    groups: dict[str, list[str]] = defaultdict(list)
    for question_id, row in control.items():
        benchmark = str(row["benchmark"])
        groups["overall"].append(question_id)
        groups[benchmark].append(question_id)
        stratum = (str(row.get("question_type")) if benchmark == "longmemeval"
                   else f"category_{row.get('category')}")
        groups[f"{benchmark}:{stratum}"].append(question_id)
    output: dict[str, Any] = {}
    for key, question_ids in sorted(groups.items()):
        left = {item: bool(control[item]["correct"]) for item in question_ids}
        right = {item: bool(candidate[item]["correct"]) for item in question_ids}
        output[key] = paired(left, right)
    return output


def build_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    ledger = payload.get("token_ledger") or rows
    if len(rows) != 510 or len(ledger) != 510:
        raise ValueError(f"{path}: expected 510 build rows and ledger rows")
    quality = [row.get("build_quality") or {} for row in rows]
    return {
        "report": str(path),
        "report_sha256": sha256(path),
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
        "fallback_scenes": sum(int(q.get("extraction_fallback_scenes", 0)) for q in quality),
        "semantic_retry_calls": sum(int(q.get("extraction_retry_calls", 0)) for q in quality),
        "budget_degraded_memories": sum(bool(q.get("budget_degraded")) for q in quality),
        "token_gate_violations": payload.get("summary", {}).get(
            "token_gate_violations", []),
        "wall_minutes": payload.get("summary", {}).get("wall_minutes"),
        "declared_model": payload.get("summary", {}).get("llm_model"),
        "declared_reasoning_effort": payload.get("summary", {}).get(
            "llm_reasoning_effort", "none"),
    }


def add_recovery_spend(summary: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Add discarded unpublished attempts to the successful build ledger."""
    rows = read_jsonl(audit_path) if audit_path.exists() else []
    discarded_tokens = sum(int(row.get("total_api_tokens", 0)) for row in rows)
    discarded_calls = sum(int(row.get("api_calls", 0)) for row in rows)
    ledger_tokens = int(summary["sum_total_tokens"])
    actual_tokens = ledger_tokens + discarded_tokens
    enriched = dict(summary)
    enriched["recovery_overhead"] = {
        "audit": str(audit_path) if audit_path.exists() else None,
        "audit_sha256": sha256(audit_path) if audit_path.exists() else None,
        "unpublished_attempts_reset": len(rows),
        "discarded_api_calls": discarded_calls,
        "discarded_api_tokens": discarded_tokens,
        "discarded_fraction_of_actual_spend": (
            discarded_tokens / actual_tokens if actual_tokens else 0.0),
    }
    enriched["actual_total_tokens_including_recovery"] = actual_tokens
    enriched["actual_mean_tokens_per_memory_including_recovery"] = (
        actual_tokens / int(summary["memories"]))
    return enriched


def graph_checksum_comparison(control_db: Path, candidate_db: Path) -> dict[str, Any]:
    def load(path: Path) -> dict[str, str]:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT memory_id, graph_checksum FROM graph_versions").fetchall()
        result = {str(memory_id): str(checksum) for memory_id, checksum in rows}
        if len(result) != 510:
            raise ValueError(f"{path}: expected 510 graph checksums, got {len(result)}")
        return result

    control, candidate = load(control_db), load(candidate_db)
    if set(control) != set(candidate):
        raise ValueError("control and candidate graph memory IDs differ")
    equal = sum(control[item] == candidate[item] for item in control)
    return {"memories": 510, "equal": equal, "different": 510 - equal}


def prompt_overlap(
        control_path: Path, candidate_path: Path,
        metadata_path: Path) -> dict[str, Any]:
    control, candidate, metadata = keyed(control_path), keyed(candidate_path), keyed(metadata_path)
    if set(control) != set(candidate) or set(control) != set(metadata):
        raise ValueError("PreparedAnswer or metadata question IDs differ")
    groups: dict[str, list[str]] = defaultdict(list)
    for item, row in metadata.items():
        groups["overall"].append(item)
        groups[str(row["benchmark"])].append(item)
    output: dict[str, Any] = {}
    for group, ids in sorted(groups.items()):
        jaccards: list[float] = []
        turn_deltas: list[float] = []
        token_deltas: list[float] = []
        exact_order = exact_prompt = 0
        for item in ids:
            left, right = control[item], candidate[item]
            left_ids = list(left.get("evidence_turn_ids") or ())
            right_ids = list(right.get("evidence_turn_ids") or ())
            left_set, right_set = set(left_ids), set(right_ids)
            jaccards.append(
                len(left_set & right_set) / len(left_set | right_set)
                if left_set | right_set else 1.0)
            turn_deltas.append(len(right_ids) - len(left_ids))
            token_deltas.append(float(right.get("evidence_tokens", 0)) -
                                float(left.get("evidence_tokens", 0)))
            exact_order += left_ids == right_ids
            exact_prompt += (left.get("prompt_payload_hash") ==
                             right.get("prompt_payload_hash"))
        output[group] = {
            "questions": len(ids),
            "evidence_jaccard_mean": mean(jaccards),
            "exact_evidence_id_and_order": exact_order,
            "identical_prompt_payloads": exact_prompt,
            "packed_turn_delta": nearest(turn_deltas),
            "evidence_token_delta": nearest(token_deltas),
        }
    return output


def answer_usage(root: Path, arm: str) -> dict[str, Any]:
    rows = read_jsonl(root / arm / "answer/answer_usage.jsonl")
    if len(rows) != sum(BENCHMARK_SIZES.values()):
        raise ValueError(f"{arm}: expected 2,040 usage rows, got {len(rows)}")
    output: dict[str, Any] = {}
    for benchmark in ("overall", *BENCHMARK_SIZES):
        selected = (rows if benchmark == "overall" else
                    [row for row in rows if row.get("benchmark") == benchmark])
        output[benchmark] = {
            "prompt_tokens": nearest(row.get("api_prompt_tokens", 0) for row in selected),
            "completion_tokens": nearest(row.get("completion_tokens", 0) for row in selected),
            "reasoning_tokens": nearest(row.get("reasoning_tokens", 0) for row in selected),
            "total_tokens": nearest(row.get("total_tokens", 0) for row in selected),
            "latency_ms": nearest(row.get("latency_ms", 0) for row in selected),
            "finish_reason_length": sum(
                str(row.get("finish_reason")) == "length" for row in selected),
        }
    return output


def candidate_retrieval(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != sum(BENCHMARK_SIZES.values()):
        raise ValueError(f"expected 2,040 retrieval rows, got {len(rows)}")
    output: dict[str, Any] = {}
    for benchmark in ("overall", *BENCHMARK_SIZES):
        selected = (rows if benchmark == "overall" else
                    [row for row in rows if row.get("benchmark") == benchmark])
        gold = [row for row in selected if row.get("has_turn_gold")]
        output[benchmark] = {
            "questions": len(selected),
            "packed_turns": nearest(row.get("packed_turns", 0) for row in selected),
            "evidence_tokens": nearest(row.get("evidence_tokens", 0) for row in selected),
            "visited_nodes": nearest(row.get("visited_nodes", 0) for row in selected),
            "visited_edges": nearest(row.get("visited_edges", 0) for row in selected),
            "latency_total_ms": nearest(row.get("latency_total_ms", 0) for row in selected),
            "gold_labeled_questions": len(gold),
            "turn_recall_mean": mean(row.get("turn_recall", 0) for row in gold),
            "turn_precision_mean": mean(row.get("turn_precision", 0) for row in gold),
            "turn_all_hit_rate": mean(bool(row.get("turn_all_hit")) for row in gold),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control-prepared", type=Path, required=True)
    parser.add_argument("--control-metadata", type=Path, required=True)
    parser.add_argument("--control-build-report", type=Path, required=True)
    parser.add_argument("--control-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verdicts: dict[str, dict[str, dict[str, Any]]] = {}
    accuracies: dict[str, Any] = {}
    for arm in ("control", "candidate"):
        verdicts[arm], accuracies[arm] = load_arm_verdicts(args.root, arm)

    candidate_prepared = args.root / "candidate/prepare/prepared_answers.jsonl"
    control_build = add_recovery_spend(
        build_summary(args.control_build_report),
        args.control_build_report.with_name("build_report_recovery.jsonl"))
    candidate_build = add_recovery_spend(
        build_summary(args.root / "build_report.json"),
        args.root / "build_report_recovery.jsonl")
    candidate_build["actual_token_saving_vs_control"] = {
        "tokens": (control_build["actual_total_tokens_including_recovery"] -
                   candidate_build["actual_total_tokens_including_recovery"]),
        "fraction": 1.0 - (
            candidate_build["actual_total_tokens_including_recovery"] /
            control_build["actual_total_tokens_including_recovery"]),
    }

    summary = {
        "schema_version": "graphmem-v5.66-luna-none-full-paired-v1",
        "protocol": {
            "questions": 2040,
            "memories": 510,
            "query_policy": "V5.63 64-turn",
            "answer_model": "gpt-5.6-luna",
            "answer_reasoning_effort": "high",
            "judge_model": "gpt-5.6-luna",
            "judge_reasoning_effort": "medium",
            "control_build_model": "Qwen3-30B",
            "candidate_build_model": "gpt-5.6-luna",
            "candidate_build_reasoning_effort": "none",
        },
        "accuracy": accuracies,
        "paired_transitions": paired_breakdown(
            verdicts["control"], verdicts["candidate"]),
        "build": {
            "control": control_build,
            "candidate": candidate_build,
        },
        "graph_checksum_comparison": graph_checksum_comparison(
            args.control_db, args.root / "graph/graphmem.sqlite"),
        "prompt_and_evidence_overlap": prompt_overlap(
            args.control_prepared, candidate_prepared, args.control_metadata),
        "answer_usage": {
            arm: answer_usage(args.root, arm) for arm in ("control", "candidate")},
        "candidate_retrieval": candidate_retrieval(
            args.root / "candidate/prepare/retrieval.jsonl"),
        "artifacts": {
            "control_prepared": str(args.control_prepared),
            "control_prepared_sha256": sha256(args.control_prepared),
            "candidate_prepared": str(candidate_prepared),
            "candidate_prepared_sha256": sha256(candidate_prepared),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Luna-none 全量重建配对结果", "",
        "查询统一为 V5.63 64-turn，回答统一为 Luna-high，Judge 统一为 Luna-medium。", "",
        "| 构建图 | LongMemEval | LoCoMo |", "|---|---:|---:|",
    ]
    for arm in ("control", "candidate"):
        lines.append(
            f"| {arm} | {accuracies[arm]['longmemeval']['accuracy']:.1%} | "
            f"{accuracies[arm]['locomo']['accuracy']:.1%} |")
    lines += ["", "## 逐题配对变化", ""]
    for benchmark in ("longmemeval", "locomo", "overall"):
        row = summary["paired_transitions"][benchmark]
        lines.append(
            f"- {benchmark}: +{row['gains']} / -{row['regressions']}，"
            f"净变化 {row['accuracy_delta_pp']:+.2f} pp，"
            f"McNemar p={row['mcnemar_exact_p']:.4g}。")
    lines += ["", f"机器可读结果：`{args.output}`", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accuracy": accuracies,
        "paired": {key: summary["paired_transitions"][key]
                   for key in ("longmemeval", "locomo", "overall")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
