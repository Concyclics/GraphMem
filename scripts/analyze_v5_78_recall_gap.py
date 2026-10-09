#!/usr/bin/env python3
"""Split LoCoMo failures into retrieval, presentation, and sampling gaps.

The audit uses annotations only after retrieval and answer generation have
finished.  It never feeds a gold evidence ID, reference answer, or judge label
back into the production path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


EVIDENCE_RE = re.compile(r"D(\d+):(\d+)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1,
                              int((fraction * len(ordered) + 0.999999)) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, required=True)
    parser.add_argument(
        "--first-judge", type=Path,
        help=("optional full candidate-1 auto_eval JSONL when judge-root "
              "contains only later-candidate wrong-subset audits"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1540)
    args = parser.parse_args()

    cases = {
        str(row["question_id"]): row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
        if int(row.get("locomo_category") or 0) in {1, 2, 3, 4}
    }
    prepared_rows = read_jsonl(args.prepared)
    prepared = {str(row["question_id"]): row for row in prepared_rows}
    if args.expected and (len(cases) != args.expected
                          or len(prepared) != args.expected):
        raise RuntimeError(
            f"expected {args.expected}, got cases={len(cases)}, "
            f"prepared={len(prepared)}")
    if set(cases) != set(prepared):
        raise RuntimeError("data/prepared question IDs do not match")

    with sqlite3.connect(args.source_db) as connection:
        turn_by_coordinate = {
            (str(memory_id), str(session_id), int(turn_index)): str(turn_id)
            for turn_id, memory_id, session_id, turn_index in connection.execute(
                "SELECT turn_id,memory_id,session_id,turn_index "
                "FROM source_turns")
        }

    correct_rank: dict[str, int] = {}
    first_correct: dict[str, bool] = {}
    judge_by_rank: dict[int, Path] = {}
    if args.first_judge is not None:
        judge_by_rank[1] = args.first_judge
    for path in args.judge_root.glob("candidate_*/auto_eval.jsonl"):
        judge_by_rank[int(path.parent.name.split("_")[-1])] = path
    for path in args.judge_root.glob(
            "candidate_*/judge_candidate_1/auto_eval.jsonl"):
        judge_by_rank[int(path.parents[1].name.split("_")[-1])] = path
    judge_files = sorted(judge_by_rank.items())
    for rank, path in judge_files:
        for row in read_jsonl(path):
            question_id = str(row["question_id"])
            correct = bool(row["correct"])
            if rank == 1:
                first_correct[question_id] = correct
            if correct:
                correct_rank.setdefault(question_id, rank)

    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    state_first_correct: Counter[str] = Counter()
    state_best_correct: Counter[str] = Counter()
    category_state: Counter[tuple[int, str]] = Counter()
    category_first_correct: Counter[int] = Counter()
    category_best_correct: Counter[int] = Counter()
    route_questions: Counter[str] = Counter()
    route_first_correct: Counter[str] = Counter()
    route_best_correct: Counter[str] = Counter()
    evidence_ranks: list[int] = []
    for question_id, case in cases.items():
        memory_id = f"locomo:{case['locomo_sample_id']}"
        gold_turn_ids: list[str] = []
        unmapped: list[str] = []
        for value in case.get("locomo_evidence", ()):
            for session, turn in EVIDENCE_RE.findall(str(value)):
                coordinate = (memory_id, f"session_{session}", int(turn) - 1)
                turn_id = turn_by_coordinate.get(coordinate)
                if turn_id is None:
                    unmapped.append(f"D{session}:{turn}")
                else:
                    gold_turn_ids.append(turn_id)
        gold_turn_ids = list(dict.fromkeys(gold_turn_ids))
        evidence = list(map(str, prepared[question_id].get(
            "evidence_turn_ids", ())))
        positions = {turn_id: index + 1
                     for index, turn_id in enumerate(evidence)}
        hits = [turn_id for turn_id in gold_turn_ids if turn_id in positions]
        evidence_ranks.extend(positions[turn_id] for turn_id in hits)
        if not gold_turn_ids:
            state = "unannotated"
        elif len(hits) == len(gold_turn_ids):
            state = "all"
        elif hits:
            state = "partial"
        else:
            state = "missing"
        category = int(case["locomo_category"])
        route = str(dict(prepared[question_id].get("trace", {})).get(
            "unified_source_readout", {}).get("route") or "unknown")
        first = bool(first_correct.get(question_id, False))
        rank = correct_rank.get(question_id)
        best = rank is not None
        if first:
            failure_class = "first_answer_correct"
        elif best:
            failure_class = "sampling_recoverable"
        elif state == "all":
            failure_class = "presentation_or_model_gap"
        else:
            failure_class = "retrieval_gap"
        state_counts[state] += 1
        state_first_correct[state] += int(first)
        state_best_correct[state] += int(best)
        category_state[(category, state)] += 1
        category_first_correct[category] += int(first)
        category_best_correct[category] += int(best)
        route_questions[route] += 1
        route_first_correct[route] += int(first)
        route_best_correct[route] += int(best)
        rows.append({
            "question_id": question_id,
            "category": category,
            "query_route": route,
            "question": case["question"],
            "reference_answer": case["answer"],
            "gold_evidence_turns": len(gold_turn_ids),
            "gold_evidence_hits": len(hits),
            "gold_evidence_state": state,
            "gold_evidence_positions": [positions[row] for row in hits],
            "unmapped_annotation_ids": unmapped,
            "first_answer_correct": first,
            "first_correct_candidate_rank": rank,
            "best_of_available_correct": best,
            "failure_class": failure_class,
        })

    failure_classes = Counter(row["failure_class"] for row in rows)
    cumulative_best = {
        str(rank): {
            "correct": sum(
                correct_rank.get(question_id, len(judge_files) + 1) <= rank
                for question_id in cases),
            "questions": len(rows),
        }
        for rank, _path in judge_files
    }
    previous = 0
    for values in cumulative_best.values():
        values["accuracy"] = values["correct"] / max(1, len(rows))
        values["incremental_recoveries"] = values["correct"] - previous
        previous = values["correct"]
    summary = {
        "schema_version": "graphmem-v5.78-recall-gap-audit-v1",
        "questions": len(rows),
        "judge_candidates_available": len(judge_files),
        "first_answer_correct": sum(first_correct.values()),
        "best_of_available_correct": len(correct_rank),
        "best_of_k": cumulative_best,
        "failure_classes": dict(failure_classes),
        "gold_evidence_states": {
            state: {
                "questions": count,
                "first_correct": state_first_correct[state],
                "best_correct": state_best_correct[state],
            }
            for state, count in sorted(state_counts.items())
        },
        "category_best_of_available": {
            str(category): {
                "first_correct": category_first_correct[category],
                "correct": category_best_correct[category],
                "questions": sum(
                    count for (value, _state), count in category_state.items()
                    if value == category),
            }
            for category in range(1, 5)
        },
        "query_route_first_and_best": {
            route: {
                "first_correct": route_first_correct[route],
                "best_correct": route_best_correct[route],
                "questions": count,
            }
            for route, count in sorted(route_questions.items())
        },
        "retrieved_gold_position": {
            "samples": len(evidence_ranks),
            "p50": percentile(evidence_ranks, 0.50),
            "p95": percentile(evidence_ranks, 0.95),
            "p99": percentile(evidence_ranks, 0.99),
            "max": max(evidence_ranks) if evidence_ranks else None,
        },
        "interpretation": {
            "retrieval_gap": (
                "No sampled answer was correct and annotated evidence was "
                "partial or absent."),
            "presentation_or_model_gap": (
                "No sampled answer was correct even though all annotated "
                "evidence turns were supplied."),
            "sampling_recoverable": (
                "The first answer was wrong but a later answer from the exact "
                "same immutable prompt was correct."),
        },
        "uses_gold_or_judge_in_runtime": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "per_question.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
