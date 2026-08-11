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
import hashlib
import json
import math
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
                candidate: Sequence[Mapping[str, Any]], *,
                identity_field: str = "prediction") -> tuple[str, ...]:
    base = by_id(baseline); cand = by_id(candidate)
    if set(base) != set(cand):
        raise ValueError("baseline and candidate question IDs differ")
    # A remote verdict is a judgment of the *prediction*, not of the prompt.
    # Local temperature-zero generation is not guaranteed byte deterministic
    # across requests or serving restarts.  Consequently an identical prompt
    # hash is useful for an answer-call replay decision, but it can never by
    # itself authorize carrying a prior verdict.  Keep the historical CLI
    # option for artifact compatibility while making it a compound identity:
    # both the prompt payload and prediction bytes must match.
    fields = ((identity_field, "prediction")
              if identity_field == "prompt_payload_hash"
              else (identity_field,))
    return tuple(
        str(row["question_id"]) for row in candidate
        if any(
            str(row.get(field, "")) != str(
                base[str(row["question_id"])].get(field, ""))
            for field in fields))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=True) + "\n"
                            for row in rows), encoding="utf-8")


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = min(gains, losses)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** discordant))


def prediction_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        str(row.get("prediction", row.get("response", ""))).encode("utf-8")
    ).hexdigest()


def validate_judge_alignment(
        answers: Sequence[Mapping[str, Any]],
        judges: Mapping[str, Mapping[str, Any]], *,
        selected_ids: set[str] | None = None) -> None:
    """Reject a verdict artifact that declares a different answer payload.

    Historical judge rows did not record the digest, so they remain readable.
    New artifacts carry it and turn an otherwise silent cross-run mix-up into a
    hard contract failure.
    """

    for answer in answers:
        question_id = str(answer["question_id"])
        if selected_ids is not None and question_id not in selected_ids:
            continue
        declared = str(judges[question_id].get("prediction_sha256") or "")
        if declared and declared != prediction_sha256(answer):
            raise ValueError(
                f"judge prediction hash mismatch for {question_id}: "
                f"{declared} != {prediction_sha256(answer)}")


def prepare(baseline_path: Path, candidate_path: Path,
            output_path: Path, manifest_path: Path,
            benchmark: str | None = None,
            identity_field: str = "prediction") -> dict[str, Any]:
    baseline = answer_rows(baseline_path, benchmark)
    candidate = answer_rows(candidate_path, benchmark)
    changed = set(changed_ids(
        baseline, candidate, identity_field=identity_field))
    delta = [row for row in candidate if str(row["question_id"]) in changed]
    write_rows(output_path, delta)
    payload = {
        "schema_version": "graphmem-v5.21-paired-judge-delta-v1",
        "baseline_answers": str(baseline_path),
        "candidate_answers": str(candidate_path),
        "benchmark_filter": benchmark,
        "identity_field": identity_field,
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
          benchmark: str | None = None,
          identity_field: str = "prediction") -> dict[str, Any]:
    baseline_answers = answer_rows(baseline_answers_path, benchmark)
    candidate_answers = answer_rows(candidate_answers_path, benchmark)
    changed = set(changed_ids(
        baseline_answers, candidate_answers,
        identity_field=identity_field))
    baseline_judge = by_id(read_rows(baseline_judge_path))
    delta_judge = by_id(read_rows(delta_judge_path)) if changed else {}
    expected = {str(row["question_id"]) for row in candidate_answers}
    if set(baseline_judge) != expected:
        raise ValueError("baseline judge does not cover the full paired question set")
    if not changed.issubset(delta_judge):
        raise ValueError(
            f"delta judge misses changed IDs: expected={len(changed)} "
            f"available={len(delta_judge)}")
    validate_judge_alignment(baseline_answers, baseline_judge)
    validate_judge_alignment(candidate_answers, delta_judge,
                             selected_ids=changed)
    merged = []
    for answer in candidate_answers:
        question_id = str(answer["question_id"])
        source = delta_judge if question_id in changed else baseline_judge
        row = dict(source[question_id])
        row["paired_verdict_source"] = (
            f"fresh_changed_{identity_field}" if question_id in changed
            else f"carried_identical_{identity_field}")
        merged.append(row)
    write_rows(output_path, merged)
    correct = sum(bool(row.get("correct")) for row in merged)
    baseline_correct = sum(bool(baseline_judge[item].get("correct"))
                           for item in expected)
    gains = sum(
        not bool(baseline_judge[item].get("correct"))
        and bool((delta_judge if item in changed else baseline_judge)[item].get("correct"))
        for item in expected)
    losses = sum(
        bool(baseline_judge[item].get("correct"))
        and not bool((delta_judge if item in changed else baseline_judge)[item].get("correct"))
        for item in expected)
    payload = {
        "schema_version": "graphmem-v5.21-paired-judge-merge-v1",
        "benchmark_filter": benchmark,
        "identity_field": identity_field,
        "questions": len(merged), "correct": correct,
        "accuracy": correct / max(1, len(merged)),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / max(1, len(merged)),
        "accuracy_delta": (correct - baseline_correct) / max(1, len(merged)),
        "gains": gains, "losses": losses,
        "stable_correct": baseline_correct - losses,
        "stable_wrong": len(merged) - baseline_correct - gains,
        "mcnemar_exact_p": exact_mcnemar(gains, losses),
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
    prep.add_argument(
        "--identity-field", default="prediction",
        choices=("prediction", "prompt_payload_hash"))
    join = sub.add_parser("merge")
    join.add_argument("--baseline-answers", type=Path, required=True)
    join.add_argument("--candidate-answers", type=Path, required=True)
    join.add_argument("--baseline-judge", type=Path, required=True)
    join.add_argument("--delta-judge", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--manifest", type=Path, required=True)
    join.add_argument("--benchmark")
    join.add_argument(
        "--identity-field", default="prediction",
        choices=("prediction", "prompt_payload_hash"))
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare(args.baseline_answers, args.candidate_answers,
                          args.output, args.manifest, args.benchmark,
                          args.identity_field)
    else:
        payload = merge(
            args.baseline_answers, args.candidate_answers,
            args.baseline_judge, args.delta_judge,
            args.output, args.manifest, args.benchmark,
            args.identity_field)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
