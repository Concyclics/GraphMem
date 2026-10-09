#!/usr/bin/env python3
"""Paired comparison of two answer models on frozen GraphMem prompts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate question IDs in {path}")
    return result


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sol-root", type=Path, required=True)
    parser.add_argument("--luna-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = {"sol_answer": args.sol_root, "luna_answer": args.luna_root}
    answers = {name: keyed(root / "answer/answers.jsonl")
               for name, root in roots.items()}
    usage = {name: keyed(root / "answer/answer_usage.jsonl")
             for name, root in roots.items()}
    ids = set(answers["sol_answer"])
    if ids != set(answers["luna_answer"]):
        raise ValueError("answer question IDs differ")
    prompt_mismatches = [
        item for item in ids
        if answers["sol_answer"][item].get("prompt_payload_hash")
        != answers["luna_answer"][item].get("prompt_payload_hash")]
    if prompt_mismatches:
        raise ValueError(f"{len(prompt_mismatches)} prompt hash mismatches")

    output: dict[str, Any] = {
        "schema_version": "graphmem-gpt56-answer-model-paired-v1",
        "roots": {name: str(root) for name, root in roots.items()},
        "questions": len(ids),
        "prompt_hash_mismatches": 0,
        "identical_predictions": sum(
            answers["sol_answer"][item].get("prediction")
            == answers["luna_answer"][item].get("prediction") for item in ids),
        "benchmarks": {},
    }
    for benchmark, suffix in (("longmemeval", "lme"), ("locomo", "locomo")):
        benchmark_ids = {
            item for item in ids
            if answers["sol_answer"][item].get("benchmark") == benchmark}
        section: dict[str, Any] = {"questions": len(benchmark_ids), "judges": {}}
        for judge in ("sol_medium", "luna_medium"):
            verdicts = {
                name: keyed(root / f"judge/{judge}/{suffix}/auto_eval.jsonl")
                for name, root in roots.items()}
            if any(set(rows) != benchmark_ids for rows in verdicts.values()):
                raise ValueError(f"{judge}/{benchmark} coverage mismatch")
            sol_correct = sum(bool(verdicts["sol_answer"][item]["correct"])
                              for item in benchmark_ids)
            luna_correct = sum(bool(verdicts["luna_answer"][item]["correct"])
                               for item in benchmark_ids)
            gains = sum(
                not bool(verdicts["sol_answer"][item]["correct"])
                and bool(verdicts["luna_answer"][item]["correct"])
                for item in benchmark_ids)
            losses = sum(
                bool(verdicts["sol_answer"][item]["correct"])
                and not bool(verdicts["luna_answer"][item]["correct"])
                for item in benchmark_ids)
            by_question_type: dict[str, Any] = {}
            question_types = sorted({
                str(answers["sol_answer"][item].get("question_type"))
                for item in benchmark_ids})
            for question_type in question_types:
                type_ids = {
                    item for item in benchmark_ids
                    if str(answers["sol_answer"][item].get("question_type"))
                    == question_type}
                type_sol = sum(bool(verdicts["sol_answer"][item]["correct"])
                               for item in type_ids)
                type_luna = sum(bool(verdicts["luna_answer"][item]["correct"])
                                for item in type_ids)
                by_question_type[question_type] = {
                    "questions": len(type_ids),
                    "sol_answer_correct": type_sol,
                    "luna_answer_correct": type_luna,
                    "luna_net_correct": type_luna - type_sol,
                    "luna_delta_percentage_points": 100.0 * (
                        type_luna - type_sol) / len(type_ids),
                }
            section["judges"][judge] = {
                "sol_answer_correct": sol_correct,
                "sol_answer_accuracy": sol_correct / len(benchmark_ids),
                "luna_answer_correct": luna_correct,
                "luna_answer_accuracy": luna_correct / len(benchmark_ids),
                "luna_gains": gains,
                "luna_losses": losses,
                "luna_net_correct": gains - losses,
                "luna_delta_percentage_points": 100.0 * (
                    luna_correct - sol_correct) / len(benchmark_ids),
                "mcnemar_exact_two_sided_p": exact_mcnemar(gains, losses),
                "by_question_type": by_question_type,
            }
        section["answer_usage_mean"] = {
            name: {
                key: mean([usage[name][item] for item in benchmark_ids], key)
                for key in ("api_prompt_tokens", "completion_tokens",
                            "reasoning_tokens", "total_tokens")
            } for name in roots
        }
        output["benchmarks"][benchmark] = section

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
