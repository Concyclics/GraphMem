#!/usr/bin/env python3
"""Prepare and merge judge deltas for byte-identical paired answer runs.

Temperature-zero remote judges can still flip a verdict for identical answer
bytes.  A paired systems experiment should not attribute that noise to the
candidate.  ``prepare`` emits only predictions that changed; ``merge`` carries
the baseline verdict for identical predictions and uses fresh verdicts for the
changed set, while producing a full canonical-order verdict artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def answer_rows(path: Path, benchmark: str | None = None) -> list[dict[str, Any]]:
    rows = read_rows(path)
    if benchmark is None:
        return rows
    return [row for row in rows if str(row.get("benchmark") or "") == benchmark]


def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate question_id")
    return result


def changed_ids(baseline: Sequence[Mapping[str, Any]],
                candidate: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    base = by_id(baseline); cand = by_id(candidate)
    if set(base) != set(cand):
        raise ValueError("baseline and candidate question IDs differ")
    return tuple(
        str(row["question_id"]) for row in candidate
        if str(row.get("prediction", "")) != str(
            base[str(row["question_id"])].get("prediction", "")))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=True) + "\n"
                            for row in rows), encoding="utf-8")


def prepare(baseline_path: Path, candidate_path: Path,
            output_path: Path, manifest_path: Path,
            benchmark: str | None = None) -> dict[str, Any]:
    baseline = answer_rows(baseline_path, benchmark)
    candidate = answer_rows(candidate_path, benchmark)
    changed = set(changed_ids(baseline, candidate))
    delta = [row for row in candidate if str(row["question_id"]) in changed]
    write_rows(output_path, delta)
    payload = {
        "schema_version": "graphmem-v5.21-paired-judge-delta-v1",
        "baseline_answers": str(baseline_path),
        "candidate_answers": str(candidate_path),
        "benchmark_filter": benchmark,
        "questions": len(candidate), "changed": len(delta),
        "carried_identical": len(candidate) - len(delta),
        "delta_answers": str(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def merge(baseline_answers_path: Path, candidate_answers_path: Path,
          baseline_judge_path: Path, delta_judge_path: Path,
          output_path: Path, manifest_path: Path,
          benchmark: str | None = None) -> dict[str, Any]:
    baseline_answers = answer_rows(baseline_answers_path, benchmark)
    candidate_answers = answer_rows(candidate_answers_path, benchmark)
    changed = set(changed_ids(baseline_answers, candidate_answers))
    baseline_judge = by_id(read_rows(baseline_judge_path))
    delta_judge = by_id(read_rows(delta_judge_path)) if changed else {}
    expected = {str(row["question_id"]) for row in candidate_answers}
    if set(baseline_judge) != expected:
        raise ValueError("baseline judge does not cover the full paired question set")
    if not changed.issubset(delta_judge):
        raise ValueError(
            f"delta judge misses changed IDs: expected={len(changed)} "
            f"available={len(delta_judge)}")
    merged = []
    for answer in candidate_answers:
        question_id = str(answer["question_id"])
        source = delta_judge if question_id in changed else baseline_judge
        row = dict(source[question_id])
        row["paired_verdict_source"] = (
            "fresh_changed_prediction" if question_id in changed
            else "carried_identical_prediction")
        merged.append(row)
    write_rows(output_path, merged)
    correct = sum(bool(row.get("correct")) for row in merged)
    payload = {
        "schema_version": "graphmem-v5.21-paired-judge-merge-v1",
        "benchmark_filter": benchmark,
        "questions": len(merged), "correct": correct,
        "accuracy": correct / max(1, len(merged)),
        "changed_fresh": len(changed),
        "delta_verdicts_available": len(delta_judge),
        "carried_identical": len(merged) - len(changed),
        "output": str(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--baseline-answers", type=Path, required=True)
    prep.add_argument("--candidate-answers", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--benchmark")
    join = sub.add_parser("merge")
    join.add_argument("--baseline-answers", type=Path, required=True)
    join.add_argument("--candidate-answers", type=Path, required=True)
    join.add_argument("--baseline-judge", type=Path, required=True)
    join.add_argument("--delta-judge", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--manifest", type=Path, required=True)
    join.add_argument("--benchmark")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare(args.baseline_answers, args.candidate_answers,
                          args.output, args.manifest, args.benchmark)
    else:
        payload = merge(
            args.baseline_answers, args.candidate_answers,
            args.baseline_judge, args.delta_judge,
            args.output, args.manifest, args.benchmark)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
