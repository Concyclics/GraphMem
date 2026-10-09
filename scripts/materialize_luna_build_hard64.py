#!/usr/bin/env python3
"""Freeze the 64-question subset used to isolate semantic-build quality.

The subset contains 32 LongMemEval and 32 LoCoMo questions.  Candidates must
be wrong under both Luna-medium and Luna-max answers on the current Qwen-built
graph (all judged by Luna-medium).  Selection is then deterministic and
round-robin stratified by LongMemEval question type or LoCoMo category.  No
candidate prediction text, gold answer, or future experimental-arm result is
used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


VERSION = "graphmem-luna-build-hard64-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def stratum(benchmark: str, verdict: dict[str, Any]) -> str:
    return str(verdict[
        "question_type" if benchmark == "longmemeval" else "category"])


def stratified_take(
    benchmark: str,
    candidates: list[str],
    verdicts: dict[str, dict[str, Any]],
    count: int,
    seed: str,
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for question_id in candidates:
        grouped[stratum(benchmark, verdicts[question_id])].append(question_id)
    buckets = {
        name: deque(sorted(rows, key=lambda item: stable_key(seed, item)))
        for name, rows in grouped.items()
    }
    selected: list[str] = []
    while len(selected) < count:
        progressed = False
        for name in sorted(buckets):
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(
            f"{benchmark}: only {len(selected)} jointly-hard questions; need {count}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata-answers", type=Path, required=True)
    parser.add_argument("--luna-medium-lme", type=Path, required=True)
    parser.add_argument("--luna-max-lme", type=Path, required=True)
    parser.add_argument("--luna-medium-locomo", type=Path, required=True)
    parser.add_argument("--luna-max-locomo", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=32)
    parser.add_argument("--seed", default=VERSION)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    prepared_rows = read_jsonl(args.prepared)
    metadata_rows = read_jsonl(args.metadata_answers)
    prepared = keyed(prepared_rows, "prepared")
    metadata = keyed(metadata_rows, "metadata")
    if set(prepared) != set(metadata):
        raise ValueError("prepared and metadata question IDs differ")

    paths = {
        "longmemeval": (args.luna_medium_lme, args.luna_max_lme),
        "locomo": (args.luna_medium_locomo, args.luna_max_locomo),
    }
    selected_order: list[str] = []
    benchmark_manifest: dict[str, Any] = {}
    for benchmark, (medium_path, max_path) in paths.items():
        medium = keyed(read_jsonl(medium_path), f"{benchmark} medium verdicts")
        maximum = keyed(read_jsonl(max_path), f"{benchmark} max verdicts")
        expected = {
            question_id for question_id, row in metadata.items()
            if str(row.get("benchmark")) == benchmark}
        if set(medium) != expected or set(maximum) != expected:
            raise ValueError(f"{benchmark}: verdict coverage differs from metadata")
        jointly_wrong = [
            question_id for question_id in expected
            if not bool(medium[question_id].get("correct"))
            and not bool(maximum[question_id].get("correct"))]
        selected = stratified_take(
            benchmark, jointly_wrong, medium, args.per_benchmark, args.seed)
        selected_order.extend(selected)
        available_by_stratum: dict[str, int] = defaultdict(int)
        selected_by_stratum: dict[str, int] = defaultdict(int)
        for question_id in jointly_wrong:
            available_by_stratum[stratum(benchmark, medium[question_id])] += 1
        for question_id in selected:
            selected_by_stratum[stratum(benchmark, medium[question_id])] += 1
        benchmark_manifest[benchmark] = {
            "jointly_wrong_candidates": len(jointly_wrong),
            "selected": len(selected),
            "available_by_stratum": dict(sorted(available_by_stratum.items())),
            "selected_by_stratum": dict(sorted(selected_by_stratum.items())),
        }

    selected_ids = set(selected_order)
    selected_prepared = [prepared[question_id] for question_id in selected_order]
    selected_metadata = [metadata[question_id] for question_id in selected_order]
    memory_ids = sorted({str(row["memory_id"]) for row in selected_prepared})

    locomo_cases = json.loads(args.locomo_data.read_text(encoding="utf-8"))
    selected_locomo_ids = {
        question_id for question_id in selected_ids
        if str(metadata[question_id].get("benchmark")) == "locomo"}
    selected_locomo = [
        row for row in locomo_cases
        if str(row.get("question_id")) in selected_locomo_ids]
    if len(selected_locomo) != len(selected_locomo_ids):
        raise ValueError("LoCoMo source does not cover every selected question")

    args.output_root.mkdir(parents=True)
    write_jsonl(args.output_root / "prepared_control_source.jsonl", selected_prepared)
    write_jsonl(args.output_root / "metadata_answers.jsonl", selected_metadata)
    (args.output_root / "question_ids.txt").write_text(
        "\n".join(selected_order) + "\n", encoding="utf-8")
    (args.output_root / "memory_ids.txt").write_text(
        "\n".join(memory_ids) + "\n", encoding="utf-8")
    (args.output_root / "locomo_hard64.json").write_text(
        json.dumps(selected_locomo, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": VERSION,
        "seed": args.seed,
        "questions": len(selected_order),
        "memories": len(memory_ids),
        "per_benchmark": args.per_benchmark,
        "selection": (
            "wrong under both Luna-medium and Luna-max answers on the current "
            "Qwen-built graph, then deterministic round-robin stratification"),
        "uses_prediction_text": False,
        "uses_gold_answer": False,
        "uses_experimental_arm_results": False,
        "benchmarks": benchmark_manifest,
        "sources": {
            "prepared": {"path": str(args.prepared), "sha256": sha256(args.prepared)},
            "metadata": {"path": str(args.metadata_answers), "sha256": sha256(args.metadata_answers)},
            "luna_medium_lme": {"path": str(args.luna_medium_lme), "sha256": sha256(args.luna_medium_lme)},
            "luna_max_lme": {"path": str(args.luna_max_lme), "sha256": sha256(args.luna_max_lme)},
            "luna_medium_locomo": {"path": str(args.luna_medium_locomo), "sha256": sha256(args.luna_medium_locomo)},
            "luna_max_locomo": {"path": str(args.luna_max_locomo), "sha256": sha256(args.luna_max_locomo)},
        },
    }
    (args.output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
