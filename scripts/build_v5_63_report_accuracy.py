#!/usr/bin/env python3
"""Build the audited report manifest with V5.63 32/64-turn results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    WORKSPACE_ROOT / "artifacts/report/v5_54/latest_accuracy/summary.json"
)
DEFAULT_V563_ROOT = WORKSPACE_ROOT / "artifacts/report/v5_63/selective64_v3"
DEFAULT_SAFE64_ROOT = WORKSPACE_ROOT / "artifacts/report/v5_59/safe_dual64/result"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "artifacts/report/v5_63/latest_accuracy/summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty token series")
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def token_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [int(row[field]) for row in rows]
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "max": max(values),
        "sum": sum(values),
        "unit": "tokens_per_question",
        "percentile_method": "nearest_rank",
    }


def accuracy_summary(
    rows: list[dict[str, Any]], *, type_field: str, locomo: bool = False
) -> tuple[int, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        raw_type = row[type_field]
        type_name = f"category_{raw_type}" if locomo else str(raw_type)
        grouped[type_name][1] += 1
        grouped[type_name][0] += int(bool(row["correct"]))
    by_type = {
        type_name: {
            "questions": questions,
            "correct": correct,
            "accuracy": correct / questions,
        }
        for type_name, (correct, questions) in sorted(grouped.items())
    }
    return sum(int(bool(row["correct"])) for row in rows), by_type


def unique_by_question(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise ValueError(f"duplicate {label} question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def replace_64_turn_rows(
    baseline: dict[str, Any], baseline_path: Path, v563_root: Path,
    safe64_root: Path
) -> dict[str, Any]:
    usage_path = v563_root / "answer/answer_usage.jsonl"
    run_manifest_path = v563_root / "answer/run_manifest.json"
    selective_manifest_path = v563_root / "manifest.json"
    lme_verdict_path = v563_root / "answer/judge_longmemeval/paired_verdicts.jsonl"
    locomo_verdict_path = safe64_root / "paired_judge_locomo.jsonl"
    current_locomo_answers = v563_root / "answer/answers_locomo.jsonl"
    safe_locomo_answers = safe64_root / "answers_locomo.jsonl"

    if sha256(current_locomo_answers) != sha256(safe_locomo_answers):
        raise ValueError("V5.63 LoCoMo answers are not byte-identical to audited Safe64")

    usage = read_jsonl(usage_path)
    usage_by_id = unique_by_question(usage, "usage")
    if len(usage_by_id) != 2040:
        raise ValueError(f"expected 2,040 usage rows, got {len(usage_by_id)}")

    verdict_specs = {
        "longmemeval": (read_jsonl(lme_verdict_path), "question_type", False),
        "locomo": (read_jsonl(locomo_verdict_path), "category", True),
    }
    expected_questions = {"longmemeval": 500, "locomo": 1540}
    new_rows: dict[str, dict[str, Any]] = {}
    old_rows = {
        (row["benchmark"], row["retrieval_setting"]): row
        for row in baseline["graphmem"]
    }

    for benchmark, (verdicts, type_field, is_locomo) in verdict_specs.items():
        verdict_by_id = unique_by_question(verdicts, f"{benchmark} verdict")
        benchmark_usage = [row for row in usage if row["benchmark"] == benchmark]
        benchmark_usage_ids = {str(row["question_id"]) for row in benchmark_usage}
        if len(verdicts) != expected_questions[benchmark]:
            raise ValueError(
                f"expected {expected_questions[benchmark]} {benchmark} verdicts, "
                f"got {len(verdicts)}"
            )
        if benchmark_usage_ids != set(verdict_by_id):
            raise ValueError(f"{benchmark} usage/verdict question IDs differ")

        correct, by_type = accuracy_summary(
            verdicts, type_field=type_field, locomo=is_locomo
        )
        row = copy.deepcopy(old_rows[(benchmark, "64-turn")])
        row.update(
            {
                "answer_model": "Qwen3-30B",
                "questions": len(verdicts),
                "correct": correct,
                "accuracy": correct / len(verdicts),
                "accuracy_by_type": by_type,
                "answer_tokens": token_stats(benchmark_usage, "total_tokens"),
                "prompt_tokens": token_stats(benchmark_usage, "api_prompt_tokens"),
                "completion_tokens": token_stats(benchmark_usage, "completion_tokens"),
                "answer_manifest": str(run_manifest_path.resolve()),
            }
        )
        for stale_field in ("turn_recall", "turn_precision", "turn_all_hit"):
            row.pop(stale_field, None)
        verdict_path = lme_verdict_path if benchmark == "longmemeval" else locomo_verdict_path
        row["annotation_contract"] = {
            "version": "v5.63-selective64-paired-verdicts",
            "judge_model": "gpt-5.6-luna",
            "verdicts": str(verdict_path.resolve()),
            "verdicts_sha256": sha256(verdict_path),
            "selective_manifest": str(selective_manifest_path.resolve()),
            "selective_manifest_sha256": sha256(selective_manifest_path),
        }
        new_rows[benchmark] = row

    output = copy.deepcopy(baseline)
    output["schema_version"] = "graphmem-v5.63-report-accuracy-v1"
    output["annotation_version"] = "v5.63-selective64-with-frozen-32turn"
    output["graphmem"] = [
        new_rows[row["benchmark"]]
        if row["retrieval_setting"] == "64-turn"
        else row
        for row in output["graphmem"]
    ]
    output["comparison_note"] = (
        "GraphMem 32-turn retains the frozen full-benchmark low-budget point; "
        "64-turn accuracy and answer Token use the audited V5.63 selective policy. "
        "Mem0 rows and the shared GraphMem build ledger remain frozen."
    )
    output["sources"] = {
        "frozen_32turn_and_mem0": str(baseline_path.resolve()),
        "frozen_32turn_and_mem0_sha256": sha256(baseline_path),
        "v5_63_selective_manifest": str(selective_manifest_path.resolve()),
        "v5_63_selective_manifest_sha256": sha256(selective_manifest_path),
        "v5_63_answer_manifest": str(run_manifest_path.resolve()),
        "v5_63_answer_manifest_sha256": sha256(run_manifest_path),
        "safe64_locomo_answers_sha256": sha256(safe_locomo_answers),
    }
    return output


def replace_32_turn_rows(output: dict[str, Any], run_root: Path) -> dict[str, Any]:
    answer_root = run_root / "answer"
    usage_path = answer_root / "answer_usage.jsonl"
    retrieval_path = answer_root / "retrieval.jsonl"
    run_manifest_path = answer_root / "run_manifest.json"
    verdict_specs = {
        "longmemeval": (
            answer_root / "judge_longmemeval/paired_verdicts.jsonl",
            "question_type",
            False,
        ),
        "locomo": (
            answer_root / "judge_locomo/paired_verdicts.jsonl",
            "category",
            True,
        ),
    }
    required = [run_manifest_path] + [
        spec[0] for spec in verdict_specs.values()
    ]
    if not usage_path.exists() and not retrieval_path.exists():
        required.append(usage_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("V5.63 32-turn artifacts incomplete: " + ", ".join(missing))

    if usage_path.exists():
        usage = read_jsonl(usage_path)
    else:
        usage = [
            {
                "question_id": row["dev_question_id"],
                "benchmark": row["benchmark"],
                "api_prompt_tokens": row["api_prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["answer_total_tokens"],
                "packed_turns": row.get("packed_turns"),
            }
            for row in read_jsonl(retrieval_path)
        ]
    unique_by_question(usage, "32-turn usage")
    if len(usage) != 2040:
        raise ValueError(f"expected 2,040 32-turn usage rows, got {len(usage)}")
    expected_questions = {"longmemeval": 500, "locomo": 1540}
    replacements: dict[str, dict[str, Any]] = {}
    old_rows = {
        (row["benchmark"], row["retrieval_setting"]): row
        for row in output["graphmem"]
    }
    for benchmark, (verdict_path, type_field, is_locomo) in verdict_specs.items():
        verdicts = read_jsonl(verdict_path)
        unique_by_question(verdicts, f"32-turn {benchmark} verdict")
        benchmark_usage = [row for row in usage if row["benchmark"] == benchmark]
        if len(verdicts) != expected_questions[benchmark]:
            raise ValueError(
                f"expected {expected_questions[benchmark]} 32-turn {benchmark} verdicts, "
                f"got {len(verdicts)}"
            )
        if {str(row["question_id"]) for row in verdicts} != {
            str(row["question_id"]) for row in benchmark_usage
        }:
            raise ValueError(f"32-turn {benchmark} usage/verdict question IDs differ")
        correct, by_type = accuracy_summary(
            verdicts, type_field=type_field, locomo=is_locomo
        )
        row = copy.deepcopy(old_rows[(benchmark, "32-turn")])
        row.update(
            {
                "answer_model": "Qwen3-30B",
                "questions": len(verdicts),
                "correct": correct,
                "accuracy": correct / len(verdicts),
                "accuracy_by_type": by_type,
                "answer_tokens": token_stats(benchmark_usage, "total_tokens"),
                "prompt_tokens": token_stats(benchmark_usage, "api_prompt_tokens"),
                "completion_tokens": token_stats(benchmark_usage, "completion_tokens"),
                "answer_manifest": str(run_manifest_path.resolve()),
                "annotation_contract": {
                    "version": "v5.63-selective32-paired-verdicts",
                    "judge_model": "gpt-5.6-luna",
                    "verdicts": str(verdict_path.resolve()),
                    "verdicts_sha256": sha256(verdict_path),
                    "run_manifest": str(run_manifest_path.resolve()),
                    "run_manifest_sha256": sha256(run_manifest_path),
                },
            }
        )
        packed_turns = [
            int(item["packed_turns"])
            for item in benchmark_usage
            if item.get("packed_turns") is not None
        ]
        if packed_turns:
            row["mean_packed_turns"] = sum(packed_turns) / len(packed_turns)
        for stale_field in ("turn_recall", "turn_precision", "turn_all_hit"):
            row.pop(stale_field, None)
        replacements[benchmark] = row

    output["graphmem"] = [
        replacements[row["benchmark"]]
        if row["retrieval_setting"] == "32-turn"
        else row
        for row in output["graphmem"]
    ]
    output["annotation_version"] = "v5.63-selective32-and-selective64"
    output["comparison_note"] = (
        "GraphMem 32-turn and 64-turn accuracy and answer Token use audited "
        "V5.63 paired verdicts. Mem0 rows and the shared GraphMem build ledger remain frozen."
    )
    output["sources"].update(
        {
            "v5_63_32_run_manifest": str(run_manifest_path.resolve()),
            "v5_63_32_run_manifest_sha256": sha256(run_manifest_path),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--v563-root", type=Path, default=DEFAULT_V563_ROOT)
    parser.add_argument("--safe64-root", type=Path, default=DEFAULT_SAFE64_ROOT)
    parser.add_argument("--v563-32-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    baseline = read_json(args.baseline)
    output = replace_64_turn_rows(
        baseline, args.baseline, args.v563_root, args.safe64_root
    )
    if args.v563_32_root is not None:
        output = replace_32_turn_rows(output, args.v563_32_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "graphmem": [
                    {
                        "benchmark": row["benchmark"],
                        "setting": row["retrieval_setting"],
                        "accuracy": row["accuracy"],
                        "answer_token_mean": row["answer_tokens"]["mean"],
                    }
                    for row in output["graphmem"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
