#!/usr/bin/env python3
"""Build report accuracy rows from the audited V5.54 Full-arm verdicts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TYPE_KEYS = {
    "lme_single_session_user": "single-session-user",
    "lme_single_session_assistant": "single-session-assistant",
    "lme_single_session_preference": "single-session-preference",
    "lme_multi_session": "multi-session",
    "lme_temporal_reasoning": "temporal-reasoning",
    "lme_knowledge_update": "knowledge-update",
    "locomo_cat1": "category_1",
    "locomo_cat2": "category_2",
    "locomo_cat3": "category_3",
    "locomo_cat4": "category_4",
}
EXPECTED = {"longmemeval": 500, "locomo": 1540}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def by_id(data: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row[key]): row for row in data}
    if len(result) != len(data):
        raise ValueError(f"duplicate {key} in {len(data)} rows")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty Token series")
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def token_stats(values: Sequence[int]) -> dict[str, Any]:
    data = [int(value) for value in values]
    return {
        "count": len(data), "mean": sum(data) / len(data),
        "p95": nearest_rank(data, 0.95), "p99": nearest_rank(data, 0.99),
        "max": max(data), "unit": "tokens_per_question",
        "percentile_method": "nearest_rank",
    }


def mean(rows_: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows_ if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def accuracy_by_type(
        question_ids: Sequence[str],
        retrieval: Mapping[str, Mapping[str, Any]],
        verdicts: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for question_id in question_ids:
        stratum = str(retrieval[question_id].get("stratum") or "")
        if stratum not in TYPE_KEYS:
            raise ValueError(f"unknown V5.54 stratum: {stratum}")
        grouped[TYPE_KEYS[stratum]].append(bool(verdicts[question_id]))
    return {
        key: {"questions": len(values), "correct": sum(values),
              "accuracy": sum(values) / len(values)}
        for key, values in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_path = args.ablation_root / "final_audit.json"
    audit = read(audit_path)
    if audit.get("passed") is not True or len(audit.get("arms", {})) != 8:
        raise RuntimeError("V5.54 final audit must pass for all eight arms")
    baseline = read(args.baseline_summary)
    baseline_graph = {
        (str(row.get("benchmark")), int(row.get("turn_budget"))): row
        for row in baseline.get("graphmem", [])
        if row.get("benchmark") and row.get("turn_budget") is not None
    }
    output: dict[str, Any] = {
        "schema_version": "graphmem-v5.54-report-accuracy-v1",
        "judge_model": "gpt-5.6-luna",
        "annotation_version": "v5.54-full-paired-verdicts-final-audit",
        "graphmem": [],
        "build_contract": copy.deepcopy(baseline.get("build_contract")),
        "mem0": copy.deepcopy(baseline.get("mem0", [])),
        "comparison_note": (
            "GraphMem accuracy and answer Token use audited V5.54 Full-arm "
            "verdicts; Mem0 rows and shared build ledgers retain the frozen baseline."),
        "sources": {
            "ablation_root": str(args.ablation_root),
            "final_audit": str(audit_path),
            "final_audit_sha256": sha256(audit_path),
            "baseline_summary": str(args.baseline_summary),
            "baseline_summary_sha256": sha256(args.baseline_summary),
        },
    }

    for budget in (32, 64):
        arm = args.ablation_root / f"turn{budget}" / "full"
        prepare = arm / "prepare"
        answer = arm / "answer"
        retrieval = by_id(rows(prepare / "retrieval.jsonl"), "dev_question_id")
        usage = by_id(rows(answer / "answer_usage.jsonl"), "question_id")
        verdict_rows = []
        verdict_paths = {}
        for benchmark in EXPECTED:
            path = answer / f"judge_{benchmark}" / "paired_verdicts.jsonl"
            verdict_paths[benchmark] = path
            verdict_rows.extend(rows(path))
        verdict_data = by_id(verdict_rows, "question_id")
        verdicts = {key: bool(row.get("correct"))
                    for key, row in verdict_data.items()}
        expected_ids = set(retrieval)
        if set(usage) != expected_ids or set(verdicts) != expected_ids:
            raise ValueError(f"turn{budget}: retrieval/usage/verdict IDs differ")

        for benchmark, expected_count in EXPECTED.items():
            ids = sorted(question_id for question_id, row in retrieval.items()
                         if row.get("benchmark") == benchmark)
            if len(ids) != expected_count:
                raise ValueError(
                    f"turn{budget}/{benchmark}: {len(ids)} != {expected_count}")
            baseline_row = copy.deepcopy(baseline_graph[(benchmark, budget)])
            selected_usage = [usage[question_id] for question_id in ids]
            selected_retrieval = [retrieval[question_id] for question_id in ids]
            annotated = [row for row in selected_retrieval if row.get("has_turn_gold")]
            correct = sum(verdicts[question_id] for question_id in ids)
            baseline_row.update({
                "method": "GraphMem",
                "answer_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                "benchmark": benchmark,
                "retrieval_setting": f"{budget}-turn",
                "turn_budget": budget,
                "questions": len(ids), "correct": correct,
                "accuracy": correct / len(ids),
                "accuracy_by_type": accuracy_by_type(
                    ids, retrieval, verdicts),
                "answer_tokens": token_stats([
                    int(row.get("total_tokens") or 0) for row in selected_usage]),
                "prompt_tokens": token_stats([
                    int(row.get("api_prompt_tokens") or 0)
                    for row in selected_usage]),
                "completion_tokens": token_stats([
                    int(row.get("completion_tokens") or 0)
                    for row in selected_usage]),
                "mean_packed_turns": mean(selected_retrieval, "packed_turns"),
                "turn_recall": mean(annotated, "turn_recall"),
                "turn_precision": mean(annotated, "turn_precision"),
                "turn_all_hit": mean(annotated, "turn_all_hit"),
                "answer_manifest": str(answer / "run_manifest.json"),
                "annotation_contract": {
                    "version": output["annotation_version"],
                    "judge_model": output["judge_model"],
                    "verdicts": str(verdict_paths[benchmark]),
                    "verdicts_sha256": sha256(verdict_paths[benchmark]),
                    "final_audit": str(audit_path),
                },
            })
            output["graphmem"].append(baseline_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "annotation_version": output["annotation_version"],
        "graphmem": [{"benchmark": row["benchmark"],
                      "setting": row["retrieval_setting"],
                      "accuracy": row["accuracy"]}
                     for row in output["graphmem"]],
    }, indent=2))


if __name__ == "__main__":
    main()
