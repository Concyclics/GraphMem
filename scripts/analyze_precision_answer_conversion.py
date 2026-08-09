#!/usr/bin/env python3
"""Join retrieval precision/recall with judged answer correctness.

Recall and all-hit alone reward a reservoir that returns the whole memory.  This
audit keeps the official-gold recall, adds annotation-scoped precision/F1 and
selectivity, and measures whether each retrieval-quality region actually
converts into a correct answer.  It is read-only with respect to the benchmark
run and makes no model calls.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=WORKSPACE / "artifacts/v5_10/full_benchmark_20260809/answers/merged",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "artifacts/report/v5_12/precision_answer_conversion",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def harmonic(precision: float, recall: float) -> float:
    return (2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def mean(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return fmean(values) if values else 0.0


def optional_mean(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return fmean(values) if values else None


def precision_bucket(value: float) -> str:
    if value <= 0:
        return "p=0"
    if value <= 0.05:
        return "0<p<=0.05"
    if value <= 0.10:
        return "0.05<p<=0.10"
    if value <= 0.25:
        return "0.10<p<=0.25"
    return "p>0.25"


def recall_bucket(value: float) -> str:
    if value <= 0:
        return "r=0"
    if value < 1:
        return "0<r<1"
    return "r=1"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    precision_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matrix: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        precision_groups[precision_bucket(float(row["turn_precision"]))].append(row)
        matrix[(recall_bucket(float(row["turn_recall"])),
                precision_bucket(float(row["turn_precision"])))].append(row)

    def gate(name: str, predicate) -> tuple[str, dict[str, Any]]:
        selected = [row for row in rows if predicate(row)]
        return name, {
            "questions": len(selected),
            "coverage": ratio(len(selected), len(rows)),
            "accuracy": mean(selected, "correct"),
            "mean_precision": mean(selected, "turn_precision"),
            "mean_recall": mean(selected, "turn_recall"),
            "mean_f1": mean(selected, "turn_f1"),
            "mean_evidence_tokens": mean(selected, "evidence_tokens"),
        }

    gates = dict((gate("all", lambda _row: True),
                  gate("all_hit", lambda row: bool(row["turn_all_hit"])),
                  gate("all_hit_p>=0.05", lambda row: bool(row["turn_all_hit"])
                       and float(row["turn_precision"]) >= 0.05),
                  gate("all_hit_p>=0.10", lambda row: bool(row["turn_all_hit"])
                       and float(row["turn_precision"]) >= 0.10),
                  gate("f1>=0.10", lambda row: float(row["turn_f1"]) >= 0.10),
                  gate("f1>=0.20", lambda row: float(row["turn_f1"]) >= 0.20)))
    return {
        "questions": len(rows),
        "accuracy": mean(rows, "correct"),
        "retrieval": {
            "turn_all_hit": mean(rows, "turn_all_hit"),
            "turn_recall": mean(rows, "turn_recall"),
            "turn_precision": mean(rows, "turn_precision"),
            "turn_f1": mean(rows, "turn_f1"),
            "candidate_turn_recall": optional_mean(rows, "candidate_turn_recall"),
            "candidate_turn_precision": optional_mean(rows, "candidate_turn_precision"),
            "candidate_turn_f1": optional_mean(rows, "candidate_turn_f1"),
            "candidate_selectivity": optional_mean(rows, "candidate_selectivity"),
            "candidate_top32_turn_recall": optional_mean(
                rows, "candidate_top32_turn_recall"),
            "candidate_top32_turn_precision": optional_mean(
                rows, "candidate_top32_turn_precision"),
            "candidate_to_pack_recall_loss": optional_mean(
                rows, "candidate_to_pack_recall_loss"),
            "candidate_to_pack_precision_gain": optional_mean(
                rows, "candidate_to_pack_precision_gain"),
        },
        "answer_conversion_gates": gates,
        "by_precision_bucket": {
            bucket: {
                "questions": len(items),
                "accuracy": mean(items, "correct"),
                "mean_recall": mean(items, "turn_recall"),
                "mean_precision": mean(items, "turn_precision"),
                "mean_f1": mean(items, "turn_f1"),
            }
            for bucket, items in sorted(precision_groups.items())
        },
        "recall_precision_answer_matrix": {
            f"{recall}|{precision}": {
                "questions": len(items), "accuracy": mean(items, "correct")}
            for (recall, precision), items in sorted(matrix.items())
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = [
        "# Retrieval Precision -> Answer Conversion",
        "",
        "Precision is scored only against official gold turns. Because the gold evidence "
        "need not be exhaustive, it is a paired lower-bound signal rather than an absolute "
        "relevance label.",
        "",
        "| Benchmark | N | Accuracy | All-hit | Recall | Precision | F1 | Candidate precision | Candidate selectivity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, row in payload["benchmarks"].items():
        retrieval = row["retrieval"]
        lines.append(
            f"| {benchmark} | {row['questions']} | {row['accuracy']:.2%} | "
            f"{retrieval['turn_all_hit']:.2%} | {retrieval['turn_recall']:.2%} | "
            f"{retrieval['turn_precision']:.2%} | {retrieval['turn_f1']:.2%} | "
            f"{pct(retrieval['candidate_turn_precision'])} | "
            f"{pct(retrieval['candidate_selectivity'])} |")
    lines.extend(["", "## Answer conversion gates", ""])
    for benchmark, row in payload["benchmarks"].items():
        lines.extend([f"### {benchmark}", "",
                      "| Gate | Coverage | Accuracy | Precision | Recall | F1 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for name, gate in row["answer_conversion_gates"].items():
            lines.append(
                f"| {name} | {gate['coverage']:.2%} | {gate['accuracy']:.2%} | "
                f"{gate['mean_precision']:.2%} | {gate['mean_recall']:.2%} | "
                f"{gate['mean_f1']:.2%} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    retrieval = read_jsonl(args.run_root / "retrieval.jsonl")
    judges = {
        "longmemeval": {
            str(row["question_id"]): bool(row["correct"])
            for row in read_jsonl(args.run_root / "judge_lme/auto_eval.jsonl")},
        "locomo": {
            str(row["question_id"]): bool(row["correct"])
            for row in read_jsonl(args.run_root / "judge_locomo/auto_eval.jsonl")},
    }
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in retrieval:
        if not source.get("has_turn_gold"):
            continue
        benchmark = str(source["benchmark"])
        qid = str(source["dev_question_id"])
        if qid not in judges.get(benchmark, {}):
            continue
        row = dict(source)
        row["correct"] = judges[benchmark][qid]
        precision = float(row.get("turn_precision", 0.0))
        recall = float(row.get("turn_recall", 0.0))
        row["turn_f1"] = float(row.get("turn_f1", harmonic(precision, recall)))
        # Candidate precision requires candidate identities/counts and is only
        # available in runs produced after the precision telemetry upgrade.
        row.setdefault("candidate_turn_precision", None)
        row.setdefault("candidate_turn_f1", None)
        row.setdefault("candidate_selectivity", None)
        row.setdefault("candidate_top32_turn_recall", None)
        row.setdefault("candidate_top32_turn_precision", None)
        row.setdefault("candidate_to_pack_recall_loss", None)
        row.setdefault("candidate_to_pack_precision_gain", None)
        rows[benchmark].append(row)
    payload = {
        "schema_version": "graphmem-precision-answer-conversion-v1",
        "run_root": str(args.run_root),
        "precision_scope": "official_gold_turns_only",
        "benchmarks": {
            benchmark: summarize(items) for benchmark, items in sorted(rows.items())},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
