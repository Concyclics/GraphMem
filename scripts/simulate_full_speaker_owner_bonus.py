#!/usr/bin/env python3
"""Simulate a zero-Token speaker/owner rerank on the complete benchmark.

This is deliberately an offline, full-corpus screen.  It does not select a
version on a development subset and it does not mutate the authority graph.
The simulation reconstructs the flood-aware proof score used by V5.22, then
adds a bounded bonus when a source turn's explicit speaker name occurs in the
question.  The result is useful only as a gate before implementing the score;
the final decision still requires a fresh 2,040-question answer and judge run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _speaker_is_explicit(query: str, speaker: str) -> bool:
    speaker_words = _normalise(speaker).split()
    if not speaker_words:
        return False
    query_words = set(_normalise(query).split())
    # Source speaker labels are normally a single proper name.  Requiring every
    # word also handles labels such as "John Doe" without substring accidents.
    return all(word in query_words for word in speaker_words)


def _effective_score(item: dict, *, proof_bonus: float) -> float:
    score = float(item.get("fused_score") or 0.0)
    if item.get("mandatory"):
        score += proof_bonus
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path)
    parser.add_argument("--judge-locomo", type=Path)
    parser.add_argument("--bonuses", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--turn-limit", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bonuses = tuple(float(value) for value in args.bonuses.split(","))
    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    judges: dict[str, dict] = {}
    for path in (args.judge_lme, args.judge_locomo):
        if path is None:
            continue
        with path.open(encoding="utf-8") as handle:
            judges.update({
                str(row["question_id"]): row for line in handle
                if line.strip() and (row := json.loads(line))
            })
    store = SQLiteGraphStore(args.source_db, read_only=True)
    totals: dict[str, Counter] = defaultdict(Counter)
    strata: dict[str, Counter] = defaultdict(Counter)
    cached_memory = ""
    turns = {}
    by_position = {}

    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = questions[str(row["question_id"])]
            if not row.get("has_turn_gold"):
                continue
            memory_id = str(row["memory_id"])
            if memory_id != cached_memory:
                cached_memory = memory_id
                memory_turns = store.turns(memory_id)
                turns = {turn.turn_id: turn for turn in memory_turns}
                by_position = {(turn.session_id, turn.turn_index): turn.turn_id
                               for turn in memory_turns}
            gold_ids = {
                turn_id for ref in question.gold_turns
                if (turn_id := by_position.get((ref.session_id, ref.turn_index)))
            }
            if not gold_ids:
                continue
            candidates = list(row["candidate_scores"])
            observed = set(row["retrieved_turn_ids"])
            softened = bool(row["trace"].get("proof_priority_softened"))
            proof_bonus = float(row["trace"].get("proof_priority_bonus") or 0.5)

            def ordered(extra_bonus: float) -> list[str]:
                def key(item: dict) -> tuple:
                    turn = turns.get(str(item["turn_id"]))
                    owner_bonus = (extra_bonus if turn is not None and
                                   _speaker_is_explicit(question.query, turn.speaker)
                                   else 0.0)
                    if softened:
                        score = _effective_score(item, proof_bonus=proof_bonus)
                        return (-score - owner_bonus, int(item.get("rank") or math.inf))
                    # Preserve the historical hard proof partition outside the
                    # structural-flood gate; the owner signal reranks only within
                    # each partition there.
                    return (-int(bool(item.get("mandatory"))),
                            -float(item.get("fused_score") or 0.0) - owner_bonus,
                            int(item.get("rank") or math.inf))
                return [str(item["turn_id"]) for item in sorted(candidates, key=key)
                        [:args.turn_limit]]

            reconstructed = set(ordered(0.0))
            observed_hit = gold_ids <= observed
            reconstructed_hit = gold_ids <= reconstructed
            benchmark = str(row["benchmark"])
            judged_correct = bool(judges.get(question.question_id, {}).get("correct"))
            counters = (totals[benchmark], strata[str(row["stratum"])])
            for counter in counters:
                counter["questions"] += 1
                counter["observed_all_hit"] += observed_hit
                counter["reconstructed_all_hit"] += reconstructed_hit
                counter["reconstruction_gain"] += reconstructed_hit and not observed_hit
                counter["reconstruction_loss"] += observed_hit and not reconstructed_hit
                if question.question_id in judges:
                    counter["judged_correct"] += judged_correct
                    counter["judged_wrong"] += not judged_correct
            for bonus in bonuses:
                selected = set(ordered(bonus))
                hit = gold_ids <= selected
                label = f"bonus_{bonus:g}"
                for counter in counters:
                    counter[f"{label}_all_hit"] += hit
                    counter[f"{label}_gain"] += hit and not reconstructed_hit
                    counter[f"{label}_loss"] += reconstructed_hit and not hit
                    if question.question_id in judges:
                        counter[f"{label}_wrong_new_all_hit"] += (
                            not judged_correct and hit)
                        counter[f"{label}_wrong_all_hit_gain"] += (
                            not judged_correct and hit and not reconstructed_hit)
                        counter[f"{label}_correct_all_hit_loss"] += (
                            judged_correct and reconstructed_hit and not hit)

    store.close()
    payload = {
        "scope": "full_longmemeval500_locomo1540",
        "turn_limit": args.turn_limit,
        "bonuses": list(bonuses),
        "benchmarks": {key: dict(value) for key, value in totals.items()},
        "strata": {key: dict(value) for key, value in strata.items()},
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
