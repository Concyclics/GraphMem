#!/usr/bin/env python3
"""Freeze a deterministic answer-model pilot around the current error set.

The pilot includes every baseline error plus a stratified sample of baseline
correct questions.  The latter is a regression sentinel: an answer model that
rescues errors but breaks already-correct questions cannot claim the naive
error-only ceiling as its expected full-benchmark gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} contains duplicate question IDs")
    return result


def stable_key(question_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def sentinel_stratum(benchmark: str, metadata: dict[str, Any],
                     verdict: dict[str, Any]) -> str:
    if benchmark == "longmemeval":
        return str(metadata.get("question_type")
                   or verdict.get("question_type") or "unknown")
    return str(metadata.get("category") or metadata.get("locomo_category")
               or verdict.get("category") or "unknown")


def stratified_sentinels(
    benchmark: str,
    correct_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    count: int,
    seed: str,
) -> list[str]:
    buckets: dict[str, deque[str]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for question_id in correct_ids:
        grouped[sentinel_stratum(
            benchmark, metadata[question_id], verdicts[question_id])].append(
                question_id)
    for stratum, question_ids in grouped.items():
        buckets[stratum] = deque(sorted(
            question_ids, key=lambda item: stable_key(item, seed)))
    selected: list[str] = []
    strata = sorted(buckets)
    while len(selected) < min(count, len(correct_ids)):
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata-answers", type=Path, required=True)
    parser.add_argument("--lme-verdicts", type=Path, required=True)
    parser.add_argument("--locomo-verdicts", type=Path, required=True)
    parser.add_argument(
        "--locomo-data", type=Path,
        help="optional full LoCoMo case JSON; writes a selected judge input")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--correct-sentinels", type=int, default=100)
    parser.add_argument("--seed", default="graphmem-gpt56-reasoning-pilot-v1")
    args = parser.parse_args()

    prepared_rows = read_jsonl(args.prepared)
    metadata_rows = read_jsonl(args.metadata_answers)
    prepared = keyed(prepared_rows, "prepared")
    metadata = keyed(metadata_rows, "metadata")
    if set(prepared) != set(metadata):
        raise ValueError("prepared and metadata question IDs differ")

    verdict_sources = {
        "longmemeval": keyed(read_jsonl(args.lme_verdicts), "LME verdicts"),
        "locomo": keyed(read_jsonl(args.locomo_verdicts), "LoCoMo verdicts"),
    }
    selected_ids: set[str] = set()
    manifest_benchmarks: dict[str, Any] = {}
    selected_verdicts: list[dict[str, Any]] = []
    for benchmark, verdicts in verdict_sources.items():
        expected = {
            question_id for question_id, row in metadata.items()
            if str(row.get("benchmark")) == benchmark}
        if set(verdicts) != expected:
            raise ValueError(
                f"{benchmark} verdict coverage differs: "
                f"verdicts={len(verdicts)} expected={len(expected)}")
        wrong = sorted(
            (question_id for question_id in expected
             if not bool(verdicts[question_id].get("correct"))),
            key=lambda item: stable_key(item, args.seed))
        correct = [question_id for question_id in expected
                   if bool(verdicts[question_id].get("correct"))]
        sentinels = stratified_sentinels(
            benchmark, correct, metadata, verdicts,
            args.correct_sentinels, args.seed)
        chosen = set(wrong) | set(sentinels)
        selected_ids |= chosen
        for question_id in chosen:
            selected_verdicts.append({
                **verdicts[question_id],
                "benchmark": benchmark,
                "pilot_role": ("baseline_wrong" if question_id in set(wrong)
                               else "baseline_correct_sentinel"),
            })
        strata: dict[str, dict[str, int]] = defaultdict(
            lambda: {"baseline_wrong": 0, "baseline_correct_sentinel": 0})
        for question_id in chosen:
            stratum = sentinel_stratum(
                benchmark, metadata[question_id], verdicts[question_id])
            role = ("baseline_wrong" if question_id in set(wrong)
                    else "baseline_correct_sentinel")
            strata[stratum][role] += 1
        full_strata: dict[str, dict[str, int]] = defaultdict(
            lambda: {"baseline_wrong": 0, "baseline_correct": 0})
        for question_id in expected:
            stratum = sentinel_stratum(
                benchmark, metadata[question_id], verdicts[question_id])
            role = ("baseline_correct" if bool(
                verdicts[question_id].get("correct")) else "baseline_wrong")
            full_strata[stratum][role] += 1
        manifest_benchmarks[benchmark] = {
            "full_questions": len(expected),
            "baseline_correct": len(correct),
            "baseline_wrong": len(wrong),
            "correct_sentinels": len(sentinels),
            "pilot_questions": len(chosen),
            "pilot_by_stratum": dict(sorted(strata.items())),
            "full_by_stratum": dict(sorted(full_strata.items())),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_prepared = [row for row in prepared_rows
                         if str(row["question_id"]) in selected_ids]
    selected_metadata = [row for row in metadata_rows
                         if str(row["question_id"]) in selected_ids]
    verdict_order = {str(row["question_id"]): row for row in selected_verdicts}
    selected_verdicts = [verdict_order[str(row["question_id"])]
                         for row in selected_metadata]
    write_jsonl(args.output_root / "prepared_answers.jsonl", selected_prepared)
    write_jsonl(args.output_root / "metadata_answers.jsonl", selected_metadata)
    write_jsonl(args.output_root / "baseline_verdicts.jsonl", selected_verdicts)
    locomo_pilot_path: Path | None = None
    if args.locomo_data:
        locomo_cases = json.loads(args.locomo_data.read_text(encoding="utf-8"))
        locomo_ids = {
            question_id for question_id in selected_ids
            if str(metadata[question_id].get("benchmark")) == "locomo"}
        locomo_pilot = [row for row in locomo_cases
                        if str(row.get("question_id")) in locomo_ids]
        if len(locomo_pilot) != len(locomo_ids):
            raise ValueError(
                "selected LoCoMo data does not cover all pilot questions")
        locomo_pilot_path = args.output_root / "locomo_pilot.json"
        locomo_pilot_path.write_text(
            json.dumps(locomo_pilot, ensure_ascii=False) + "\n",
            encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-gpt56-reasoning-pilot-v1",
        "selection_seed": args.seed,
        "prepared_source": str(args.prepared),
        "prepared_source_sha256": hashlib.sha256(
            args.prepared.read_bytes()).hexdigest(),
        "metadata_source": str(args.metadata_answers),
        "lme_verdict_source": str(args.lme_verdicts),
        "locomo_verdict_source": str(args.locomo_verdicts),
        "locomo_pilot_data": (str(locomo_pilot_path)
                              if locomo_pilot_path else None),
        "questions": len(selected_ids),
        "correct_sentinels_per_benchmark": args.correct_sentinels,
        "benchmarks": manifest_benchmarks,
        "selection_uses_predictions": False,
        "selection_uses_only_baseline_verdict": True,
    }
    (args.output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
