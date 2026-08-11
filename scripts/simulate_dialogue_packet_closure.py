#!/usr/bin/env python3
"""Replay bounded dialogue-packet quotas over a full candidate trace."""
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
from graphmem.text import content_terms  # noqa: E402


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
    variants = [
        (anchor, quota, overlap, direction)
        for anchor in (8, 16, 32)
        for quota in (2, 4, 8)
        for overlap in (1, 2)
        for direction in ("next", "both")]
    totals = {variant: defaultdict(int) for variant in variants}
    store = SQLiteGraphStore(args.source_db, read_only=True)
    cached_memory = ""
    turns = {}
    by_position = {}

    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["benchmark"] != "locomo" or not row["has_turn_gold"]:
                continue
            question = questions[row["question_id"]]
            if row["memory_id"] != cached_memory:
                cached_memory = row["memory_id"]
                memory_turns = store.turns(cached_memory)
                turns = {turn.turn_id: turn for turn in memory_turns}
                by_position = {(turn.session_id, turn.turn_index): turn.turn_id
                               for turn in memory_turns}
            gold = {
                by_position[(ref.session_id, ref.turn_index)]
                for ref in question.gold_turns
                if (ref.session_id, ref.turn_index) in by_position}
            candidates = row["candidate_scores"]
            channel_base = {
                item["turn_id"]: 1.2 * float(item["exact_score"])
                + float(item["bm25_score"]) + float(item["dense_score"])
                for item in candidates}
            generic_adjacency: dict[str, float] = defaultdict(float)
            for turn_id, value in channel_base.items():
                turn = turns[turn_id]
                for distance in (1, 2):
                    for index in (turn.turn_index - distance,
                                  turn.turn_index + distance):
                        neighbour = by_position.get((turn.session_id, index))
                        if neighbour:
                            generic_adjacency[neighbour] = max(
                                generic_adjacency[neighbour],
                                value * (0.35 / distance))
            base_rows = sorted(candidates, key=lambda item: (
                -bool(item["mandatory"]),
                -(float(item["fused_score"])
                  - float(item["adjacency_score"])
                  + generic_adjacency[item["turn_id"]]),
                item["turn_id"]))
            baseline = {item["turn_id"] for item in base_rows[:64]}
            query_terms = content_terms(question.query)
            base_all_hit = gold <= baseline
            for variant in variants:
                anchor_limit, quota, overlap_floor, direction = variant
                proposals = []
                for anchor_rank, item in enumerate(base_rows[:anchor_limit], 1):
                    anchor_turn = turns[item["turn_id"]]
                    overlap = len(query_terms & content_terms(anchor_turn.raw_text))
                    if overlap < overlap_floor:
                        continue
                    offsets = (1,) if direction == "next" else (-1, 1)
                    for offset in offsets:
                        witness_id = by_position.get((
                            anchor_turn.session_id,
                            anchor_turn.turn_index + offset))
                        witness = turns.get(witness_id) if witness_id else None
                        if (witness is None or witness.speaker == anchor_turn.speaker
                                or witness_id in baseline):
                            continue
                        # Query-conditioned packet: at least one side must be a
                        # dialogue question, otherwise this is generic proximity.
                        if "?" not in anchor_turn.raw_text and "?" not in witness.raw_text:
                            continue
                        proposals.append((anchor_rank, -overlap,
                                          0 if offset == 1 else 1, witness_id))
                witnesses = []
                for _rank, _overlap, _direction, witness_id in sorted(proposals):
                    if witness_id not in witnesses:
                        witnesses.append(witness_id)
                    if len(witnesses) >= quota:
                        break
                retained = [item["turn_id"] for item in base_rows
                            if item["turn_id"] not in witnesses]
                packed = set((*witnesses, *retained[:64 - len(witnesses)]))
                all_hit = gold <= packed
                counter = totals[variant]
                counter["questions"] += 1
                counter["baseline_all_hit"] += base_all_hit
                counter["variant_all_hit"] += all_hit
                counter["gains"] += all_hit and not base_all_hit
                counter["losses"] += base_all_hit and not all_hit
                counter["gold_hits"] += len(gold & packed)
                counter["gold_turns"] += len(gold)
                counter["witnesses"] += len(witnesses)
    store.close()
    output = []
    for variant, counter in totals.items():
        anchor, quota, overlap, direction = variant
        output.append({
            "anchor_limit": anchor,
            "quota": quota,
            "overlap_floor": overlap,
            "direction": direction,
            **dict(counter),
            "delta_all_hit": counter["variant_all_hit"]
            - counter["baseline_all_hit"],
            "variant_all_hit_rate": counter["variant_all_hit"]
            / max(1, counter["questions"]),
            "gold_recall": counter["gold_hits"] / max(1, counter["gold_turns"]),
            "mean_witnesses": counter["witnesses"] / max(1, counter["questions"]),
        })
    print(json.dumps(sorted(
        output, key=lambda item: (-item["delta_all_hit"], item["losses"],
                                  item["mean_witnesses"])), indent=2))


if __name__ == "__main__":
    main()
