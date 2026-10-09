#!/usr/bin/env python3
"""Compare multiple answer-model runs under one fixed judge."""
from __future__ import annotations

import argparse
import itertools
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


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=ROOT")
    label, root = value.split("=", 1)
    if not label or not root:
        raise argparse.ArgumentTypeError("run must be LABEL=ROOT")
    return label, Path(root)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--judge", default="luna_medium")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = dict(args.run)
    if len(roots) != len(args.run):
        raise ValueError("duplicate run labels")
    if args.reference not in roots:
        raise ValueError("reference label is not a supplied run")

    answers = {label: keyed(root / "answer/answers.jsonl")
               for label, root in roots.items()}
    usage = {label: keyed(root / "answer/answer_usage.jsonl")
             for label, root in roots.items()}
    manifests = {label: json.loads(
        (root / "answer/run_manifest.json").read_text(encoding="utf-8"))
        for label, root in roots.items()}
    ids = set(answers[args.reference])
    for label, rows in answers.items():
        if set(rows) != ids:
            raise ValueError(f"{label} answer coverage mismatch")
        mismatches = [item for item in ids if rows[item]["prompt_payload_hash"]
                      != answers[args.reference][item]["prompt_payload_hash"]]
        if mismatches:
            raise ValueError(f"{label} has {len(mismatches)} prompt mismatches")

    output: dict[str, Any] = {
        "schema_version": "graphmem-fixed-judge-answer-models-v1",
        "judge": args.judge,
        "reference": args.reference,
        "questions": len(ids),
        "prompt_hash_mismatches": 0,
        "runs": {
            label: {
                "root": str(roots[label]),
                "answer_model": manifest["answer_model"],
                "reasoning_effort": manifest.get("answer_reasoning_effort"),
                "max_output_tokens": manifest.get("max_output_tokens"),
                "output_truncated": manifest.get("output_truncated"),
            } for label, manifest in manifests.items()
        },
        "benchmarks": {},
    }
    for benchmark, suffix in (("longmemeval", "lme"), ("locomo", "locomo")):
        benchmark_ids = {item for item in ids
                         if answers[args.reference][item]["benchmark"] == benchmark}
        verdicts = {
            label: keyed(root / f"judge/{args.judge}/{suffix}/auto_eval.jsonl")
            for label, root in roots.items()}
        if any(set(rows) != benchmark_ids for rows in verdicts.values()):
            raise ValueError(f"{benchmark} verdict coverage mismatch")
        section: dict[str, Any] = {
            "questions": len(benchmark_ids), "runs": {},
            "paired_vs_reference": {}, "pairwise": {}}
        for label in roots:
            correct = sum(bool(verdicts[label][item]["correct"])
                          for item in benchmark_ids)
            selected_usage = [usage[label][item] for item in benchmark_ids]
            section["runs"][label] = {
                "correct": correct,
                "accuracy": correct / len(benchmark_ids),
                "answer_usage_mean": {
                    key: mean(selected_usage, key) for key in (
                        "api_prompt_tokens", "completion_tokens",
                        "reasoning_tokens", "total_tokens")},
            }
            if label == args.reference:
                continue
            gains = sum(
                not bool(verdicts[args.reference][item]["correct"])
                and bool(verdicts[label][item]["correct"])
                for item in benchmark_ids)
            losses = sum(
                bool(verdicts[args.reference][item]["correct"])
                and not bool(verdicts[label][item]["correct"])
                for item in benchmark_ids)
            section["paired_vs_reference"][label] = {
                "gains": gains,
                "losses": losses,
                "net_correct": gains - losses,
                "delta_percentage_points": 100.0 * (gains - losses)
                / len(benchmark_ids),
                "mcnemar_exact_two_sided_p": exact_mcnemar(gains, losses),
            }
        for first, second in itertools.combinations(roots, 2):
            second_gains = sum(
                not bool(verdicts[first][item]["correct"])
                and bool(verdicts[second][item]["correct"])
                for item in benchmark_ids)
            second_losses = sum(
                bool(verdicts[first][item]["correct"])
                and not bool(verdicts[second][item]["correct"])
                for item in benchmark_ids)
            section["pairwise"][f"{second}_vs_{first}"] = {
                "second_gains": second_gains,
                "second_losses": second_losses,
                "second_net_correct": second_gains - second_losses,
                "second_delta_percentage_points": 100.0 * (
                    second_gains - second_losses) / len(benchmark_ids),
                "mcnemar_exact_two_sided_p": exact_mcnemar(
                    second_gains, second_losses),
            }
        output["benchmarks"][benchmark] = section

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
