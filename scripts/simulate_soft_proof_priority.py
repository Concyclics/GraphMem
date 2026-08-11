#!/usr/bin/env python3
"""Evaluate finite proof-unit priority on the full LoCoMo candidate trace."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()

    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    variants: tuple[tuple[float | None, int | None, str], ...] = (
        (None, None, "generic"), (0.0, 0, "generic"),
        (0.5, 0, "generic"), (1.0, 0, "generic"),
        (0.5, 8, "generic"), (0.5, 16, "generic"),
        (0.5, 24, "generic"), (0.5, 32, "generic"),
        (0.5, 16, "dialogue"))
    totals = {benchmark: {variant: defaultdict(int) for variant in variants}
              for benchmark in ("longmemeval", "locomo")}
    store = SQLiteGraphStore(args.source_db, read_only=True)
    cached_memory = ""; by_position = {}; turns = {}
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["has_turn_gold"]:
                continue
            question = questions[row["question_id"]]
            if row["memory_id"] != cached_memory:
                cached_memory = row["memory_id"]
                memory_turns = store.turns(cached_memory)
                turns = {turn.turn_id: turn for turn in memory_turns}
                by_position = {(turn.session_id, turn.turn_index): turn.turn_id
                               for turn in memory_turns}
            gold = {by_position[(ref.session_id, ref.turn_index)]
                    for ref in question.gold_turns
                    if (ref.session_id, ref.turn_index) in by_position}
            candidates = row["candidate_scores"]
            channel_base = {item["turn_id"]:
                            1.2 * float(item["exact_score"])
                            + float(item["bm25_score"])
                            + float(item["dense_score"])
                            for item in candidates}
            generic = defaultdict(float)
            for turn_id, value in channel_base.items():
                turn = turns[turn_id]
                for distance in (1, 2):
                    for index in (turn.turn_index - distance,
                                  turn.turn_index + distance):
                        neighbour = by_position.get((turn.session_id, index))
                        if neighbour:
                            generic[neighbour] = max(
                                generic[neighbour], value * (0.35 / distance))
            generic_scored = [(item, float(item["fused_score"])
                       - float(item["adjacency_score"])
                       + generic[item["turn_id"]]) for item in candidates]
            dialogue_scored = [(item, float(item["fused_score"]))
                               for item in candidates]
            mandatory_count = sum(bool(item["mandatory"])
                                  for item, _ in generic_scored)
            gold_mandatory = sum(bool(item["mandatory"])
                                 for item, _ in generic_scored
                                 if item["turn_id"] in gold)
            for bonus, flood_threshold, adjacency_mode in variants:
                scored = (dialogue_scored if adjacency_mode == "dialogue"
                          else generic_scored)
                hard_priority = (bonus is None or (
                    flood_threshold is not None
                    and mandatory_count <= flood_threshold))
                if hard_priority:
                    ordered = sorted(scored, key=lambda row: (
                        -bool(row[0]["mandatory"]), -row[1], row[0]["turn_id"]))
                else:
                    ordered = sorted(scored, key=lambda row: (
                        -(row[1] + bonus * bool(row[0]["mandatory"])),
                        row[0]["turn_id"]))
                packed = {item["turn_id"] for item, _score in ordered[:64]}
                counter = totals[row["benchmark"]][(
                    bonus, flood_threshold, adjacency_mode)]
                counter["questions"] += 1
                counter["all_hit"] += gold <= packed
                counter["any_hit"] += bool(gold & packed)
                counter["gold_hits"] += len(gold & packed)
                counter["gold_turns"] += len(gold)
                counter["mandatory_candidates"] += mandatory_count
                counter["gold_mandatory"] += gold_mandatory
    store.close()
    output = {}
    for benchmark, benchmark_totals in totals.items():
        rows = []
        hard = benchmark_totals[(None, None, "generic")]["all_hit"]
        for (bonus, flood_threshold, adjacency_mode), counter in benchmark_totals.items():
            rows.append({
                "mandatory_priority": "hard" if bonus is None else bonus,
                "soften_only_above": flood_threshold,
                "adjacency_mode": adjacency_mode,
                **dict(counter),
                "all_hit_rate": counter["all_hit"] / counter["questions"],
                "all_hit_delta_vs_hard": counter["all_hit"] - hard,
                "gold_recall": counter["gold_hits"] / counter["gold_turns"],
                "mean_mandatory_candidates": (
                    counter["mandatory_candidates"] / counter["questions"]),
                "gold_mandatory_rate": (
                    counter["gold_mandatory"] / counter["gold_turns"]),
            })
        output[benchmark] = rows
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
