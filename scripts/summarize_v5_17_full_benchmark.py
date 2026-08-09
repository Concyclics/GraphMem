#!/usr/bin/env python3
"""Summarize the fresh-build V5.17 lexical OFF/ON full benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_v5_9_full_benchmark import (
    paired_stats,
    read_json,
    read_jsonl,
    retrieval_summary,
)


def paired_retrieval(left: list[dict[str, Any]],
                     right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {str(row["dev_question_id"]): row for row in left}
    right_by_id = {str(row["dev_question_id"]): row for row in right}
    question_ids = sorted(set(left_by_id) & set(right_by_id))
    numeric = (
        "turn_recall", "session_recall", "visited_nodes", "visited_edges",
        "latency_total_ms", "prompt_tokens", "evidence_tokens", "packed_turns",
    )
    boolean = ("turn_all_hit", "session_all_hit", "certificate_complete")
    payload: dict[str, Any] = {"questions": len(question_ids), "delta": {}}
    for key in numeric:
        values = [
            float(right_by_id[item].get(key, 0.0))
            - float(left_by_id[item].get(key, 0.0))
            for item in question_ids
        ]
        payload["delta"][key] = statistics.fmean(values) if values else 0.0
    payload["transitions"] = {}
    for key in boolean:
        transitions: dict[str, int] = defaultdict(int)
        for item in question_ids:
            old = int(bool(left_by_id[item].get(key)))
            new = int(bool(right_by_id[item].get(key)))
            transitions[f"{old}->{new}"] += 1
        payload["transitions"][key] = dict(sorted(transitions.items()))
        payload["delta"][key] = (
            sum((bool(right_by_id[item].get(key))
                 - bool(left_by_id[item].get(key))) for item in question_ids)
            / len(question_ids) if question_ids else 0.0)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--lexical-off-root", type=Path, required=True)
    parser.add_argument("--lexical-on-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = {
        "lexical_off": args.lexical_off_root,
        "lexical_on": args.lexical_on_root,
    }
    retrieval: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm, root in roots.items():
        by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(root / "retrieval.jsonl"):
            by_benchmark[str(row["benchmark"])].append(row)
        retrieval[arm] = dict(by_benchmark)

    payload: dict[str, Any] = {
        "schema_version": "graphmem-v5.17-fresh-full-benchmark-v1",
        "build": read_json(args.build_report),
        "arms": {
            arm: {
                "root": str(root),
                "manifest": read_json(root / "run_manifest.json"),
            } for arm, root in roots.items()
        },
        "benchmarks": {},
    }
    for benchmark, judge_dir in (
        ("longmemeval", "judge_lme"),
        ("locomo", "judge_locomo"),
    ):
        old_eval = read_jsonl(args.previous_root / judge_dir / "auto_eval.jsonl")
        evaluations = {
            arm: read_jsonl(root / judge_dir / "auto_eval.jsonl")
            for arm, root in roots.items()
        }
        row: dict[str, Any] = {
            "arms": {},
            "paired_off_to_on": paired_stats(
                evaluations["lexical_off"], evaluations["lexical_on"]),
            "retrieval_paired_off_to_on": paired_retrieval(
                retrieval["lexical_off"][benchmark],
                retrieval["lexical_on"][benchmark]),
        }
        for arm, root in roots.items():
            row["arms"][arm] = {
                "accuracy": read_json(root / judge_dir / "judge_token_stats.json"),
                "retrieval": retrieval_summary(retrieval[arm][benchmark]),
                "paired_vs_v5_10": paired_stats(old_eval, evaluations[arm]),
            }
            if benchmark == "locomo":
                row["arms"][arm]["official_token_f1"] = read_json(
                    root / "locomo_official_f1" / "official_eval.json")
        payload["benchmarks"][benchmark] = row

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    lines = [
        "# V5.17 fresh-build full benchmark", "",
        "| Benchmark | Lexical OFF | Lexical ON | ON-OFF | New-only / OFF-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark, label in (("longmemeval", "LongMemEval 500"),
                             ("locomo", "LoCoMo Cat1--4 1540")):
        row = payload["benchmarks"][benchmark]
        off = row["arms"]["lexical_off"]["accuracy"]["accuracy"]
        on = row["arms"]["lexical_on"]["accuracy"]["accuracy"]
        paired = row["paired_off_to_on"]
        lines.append(
            f"| {label} | {off:.2%} | {on:.2%} | "
            f"{paired['delta']:+.2%} | "
            f"{paired['new_only']} / {paired['previous_only']} |")
    (args.output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        benchmark: {
            arm: payload["benchmarks"][benchmark]["arms"][arm]
            ["accuracy"]["accuracy"] for arm in roots
        } for benchmark in ("longmemeval", "locomo")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
