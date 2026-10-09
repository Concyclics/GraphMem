#!/usr/bin/env python3
"""Summarize a fair, evaluator-only Best-of-3 answer oracle.

Each method must provide three answer replicas produced from byte-identical
per-question prompts and three independently generated judge ledgers.  The
summary reports both the raw oracle and a prediction-byte-stable oracle that
does not let repeated judging of an identical answer create a false gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


EXPECTED = {"longmemeval": 500, "locomo": 1540}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_id(benchmark: str, question_id: str) -> str:
    if benchmark != "locomo":
        return question_id
    match = re.fullmatch(r"conv(\d+)_q(\d+)", question_id)
    if match:
        return f"locomo{int(match.group(1)):02d}_{int(match.group(2)):04d}"
    return question_id


def normalize_prediction(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - radius, center + radius]


def mcnemar_exact(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_answers(root: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    path = root / f"answers_{benchmark}.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_id = canonical_id(benchmark, str(row["question_id"]))
        if question_id in rows:
            raise RuntimeError(f"duplicate answer ID in {path}: {question_id}")
        rows[question_id] = row
    if len(rows) != EXPECTED[benchmark]:
        raise RuntimeError(f"{path}: expected {EXPECTED[benchmark]}, found {len(rows)}")
    return rows


def load_verdicts(path: Path, benchmark: str) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    for row in read_jsonl(path):
        question_id = canonical_id(benchmark, str(row["question_id"]))
        if question_id in rows:
            raise RuntimeError(f"duplicate verdict ID in {path}: {question_id}")
        rows[question_id] = bool(row["correct"])
    if len(rows) != EXPECTED[benchmark]:
        raise RuntimeError(f"{path}: expected {EXPECTED[benchmark]}, found {len(rows)}")
    return rows


def parse_replica(values: list[str]) -> tuple[Path, dict[str, Path]]:
    if len(values) != 3:
        raise ValueError("replica requires ANSWER_ROOT LME_VERDICTS LOCOMO_VERDICTS")
    return Path(values[0]), {"longmemeval": Path(values[1]), "locomo": Path(values[2])}


def summarize_method(
    label: str,
    specs: list[tuple[Path, dict[str, Path]]],
) -> tuple[dict[str, Any], dict[str, dict[str, bool]]]:
    if len(specs) != 3:
        raise RuntimeError(f"{label}: exactly three replicas are required")
    result: dict[str, Any] = {"replicates": [], "benchmarks": {}}
    stable_oracles: dict[str, dict[str, bool]] = {}

    answer_sets: dict[str, list[dict[str, dict[str, Any]]]] = {
        benchmark: [] for benchmark in EXPECTED
    }
    verdict_sets: dict[str, list[dict[str, bool]]] = {
        benchmark: [] for benchmark in EXPECTED
    }
    for index, (answer_root, verdict_paths) in enumerate(specs, start=1):
        manifest_path = answer_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replica_record = {
            "replicate": index,
            "answer_root": str(answer_root),
            "answer_manifest": str(manifest_path),
            "answer_manifest_sha256": sha256(manifest_path),
            "answer_model": manifest.get("answer_model"),
            "answer_reasoning_effort": manifest.get("answer_reasoning_effort"),
            "max_output_tokens": manifest.get("max_output_tokens"),
            "temperature": manifest.get("temperature", 0.0),
            "seed": manifest.get("seed", 0),
            "questions": manifest.get("questions", manifest.get("prepared_questions")),
            "prompt_source_sha256": manifest.get("prepared_sha256"),
            "verdicts": {},
        }
        for benchmark, verdict_path in verdict_paths.items():
            answers = load_answers(answer_root, benchmark)
            verdicts = load_verdicts(verdict_path, benchmark)
            if set(answers) != set(verdicts):
                raise RuntimeError(f"{label} replica {index} {benchmark}: answer/verdict IDs differ")
            answer_sets[benchmark].append(answers)
            verdict_sets[benchmark].append(verdicts)
            replica_record["verdicts"][benchmark] = {
                "path": str(verdict_path),
                "sha256": sha256(verdict_path),
                "correct": sum(verdicts.values()),
                "questions": len(verdicts),
                "accuracy": sum(verdicts.values()) / len(verdicts),
            }
        result["replicates"].append(replica_record)

    for benchmark in EXPECTED:
        answers = answer_sets[benchmark]
        verdicts = verdict_sets[benchmark]
        ids = set(answers[0])
        if any(set(rows) != ids for rows in answers[1:]) or any(set(rows) != ids for rows in verdicts):
            raise RuntimeError(f"{label} {benchmark}: question sets differ across replicas")

        prompt_mismatches = 0
        raw_oracle: dict[str, bool] = {}
        stable_oracle: dict[str, bool] = {}
        judge_flip_questions = 0
        unique_prediction_histogram: Counter[int] = Counter()
        correctness_patterns: Counter[str] = Counter()
        all_predictions_identical = 0
        for question_id in sorted(ids):
            prompt_hashes = [str(rows[question_id].get("prompt_payload_hash") or "") for rows in answers]
            if len(set(prompt_hashes)) != 1:
                prompt_mismatches += 1
            predictions = [normalize_prediction(rows[question_id].get("prediction")) for rows in answers]
            unique_prediction_histogram[len(set(predictions))] += 1
            if len(set(predictions)) == 1:
                all_predictions_identical += 1
            flags = [rows[question_id] for rows in verdicts]
            correctness_patterns["".join("1" if flag else "0" for flag in flags)] += 1
            raw_oracle[question_id] = any(flags)

            # The first verdict for each prediction byte string is authoritative.
            # This makes repeated identical answers incapable of gaining merely
            # because a nondeterministic judge flips its label later.
            authoritative: dict[str, bool] = {}
            conflict = False
            stable_flags: list[bool] = []
            for prediction, flag in zip(predictions, flags):
                if prediction in authoritative and authoritative[prediction] != flag:
                    conflict = True
                authoritative.setdefault(prediction, flag)
                stable_flags.append(authoritative[prediction])
            judge_flip_questions += int(conflict)
            stable_oracle[question_id] = any(stable_flags)

        per_replica_correct = [sum(rows.values()) for rows in verdicts]
        raw_correct = sum(raw_oracle.values())
        stable_correct = sum(stable_oracle.values())
        best_single = max(per_replica_correct)
        result["benchmarks"][benchmark] = {
            "questions": len(ids),
            "prompt_hash_mismatches_across_replicates": prompt_mismatches,
            "replicate_correct": per_replica_correct,
            "replicate_accuracy": [value / len(ids) for value in per_replica_correct],
            "single_run_accuracy_mean": mean(value / len(ids) for value in per_replica_correct),
            "single_run_accuracy_population_sd": pstdev(value / len(ids) for value in per_replica_correct),
            "best_single_correct": best_single,
            "best_single_accuracy": best_single / len(ids),
            "raw_oracle": {
                "correct": raw_correct,
                "accuracy": raw_correct / len(ids),
                "uplift_vs_best_single_pp": 100.0 * (raw_correct - best_single) / len(ids),
                "wilson95": wilson(raw_correct, len(ids)),
            },
            "prediction_byte_stable_oracle": {
                "correct": stable_correct,
                "accuracy": stable_correct / len(ids),
                "uplift_vs_best_single_pp": 100.0 * (stable_correct - best_single) / len(ids),
                "wilson95": wilson(stable_correct, len(ids)),
            },
            "all_three_predictions_identical": all_predictions_identical,
            "all_three_predictions_identical_rate": all_predictions_identical / len(ids),
            "unique_prediction_count_histogram": dict(sorted(unique_prediction_histogram.items())),
            "correctness_pattern_histogram": dict(sorted(correctness_patterns.items())),
            "identical_prediction_judge_flip_questions": judge_flip_questions,
        }
        stable_oracles[benchmark] = stable_oracle
    return result, stable_oracles


def compare_oracles(
    graphmem: dict[str, dict[str, bool]], mem0: dict[str, dict[str, bool]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for benchmark in EXPECTED:
        if set(graphmem[benchmark]) != set(mem0[benchmark]):
            raise RuntimeError(f"{benchmark}: method question sets differ")
        ids = sorted(graphmem[benchmark])
        both = sum(graphmem[benchmark][qid] and mem0[benchmark][qid] for qid in ids)
        graph_only = sum(graphmem[benchmark][qid] and not mem0[benchmark][qid] for qid in ids)
        mem0_only = sum(mem0[benchmark][qid] and not graphmem[benchmark][qid] for qid in ids)
        neither = len(ids) - both - graph_only - mem0_only
        graph_correct = both + graph_only
        mem0_correct = both + mem0_only
        comparisons[benchmark] = {
            "questions": len(ids),
            "graphmem_correct": graph_correct,
            "graphmem_accuracy": graph_correct / len(ids),
            "mem0_correct": mem0_correct,
            "mem0_accuracy": mem0_correct / len(ids),
            "delta_percentage_points": 100.0 * (graph_correct - mem0_correct) / len(ids),
            "relative_improvement": (
                graph_correct / mem0_correct - 1.0 if mem0_correct else None
            ),
            "both_correct": both,
            "graphmem_only_correct": graph_only,
            "mem0_only_correct": mem0_only,
            "both_wrong": neither,
            "mcnemar_exact_two_sided_p": mcnemar_exact(graph_only, mem0_only),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphmem-replicate", nargs=3, action="append", required=True,
        metavar=("ANSWER_ROOT", "LME_VERDICTS", "LOCOMO_VERDICTS"),
    )
    parser.add_argument(
        "--mem0-replicate", nargs=3, action="append", required=True,
        metavar=("ANSWER_ROOT", "LME_VERDICTS", "LOCOMO_VERDICTS"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graph_result, graph_oracles = summarize_method(
        "GraphMem", [parse_replica(value) for value in args.graphmem_replicate]
    )
    mem0_result, mem0_oracles = summarize_method(
        "Mem0", [parse_replica(value) for value in args.mem0_replicate]
    )
    summary = {
        "schema_version": "graphmem-mem0-luna-max-best-of-3-oracle-v1",
        "interpretation": (
            "Evaluator-only upper bound. Per-question selection uses judge labels and is not deployable."
        ),
        "methods": {"graphmem": graph_result, "mem0": mem0_result},
        "prediction_byte_stable_oracle_comparison": compare_oracles(
            graph_oracles, mem0_oracles
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
