#!/usr/bin/env python3
"""Full-corpus screen for bounded dialogue-packet score propagation.

The candidate reservoir already has perfect recall on annotated V5.22 errors.
This simulation asks whether relevance on a query-matching source turn should be
propagated to adjacent turns in the same session before the fixed 64-turn cut.
It uses no gold signal in scoring; gold and Luna verdicts are read only after
ranking to report the full-corpus gate.
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


def _index(path: Path | None, key: str) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return {str(row[key]): row for line in handle
                if line.strip() and (row := json.loads(line))}


def _normalise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _explicit_speaker(query: str, speaker: str) -> bool:
    words = _normalise(speaker)
    return bool(words) and words <= _normalise(query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path)
    parser.add_argument("--judge-locomo", type=Path)
    parser.add_argument("--packet-gains", default="0.25,0.5,1,2")
    parser.add_argument("--speaker-bonuses", default="0,0.5,1")
    parser.add_argument("--windows", default="1,2")
    parser.add_argument("--turn-limit", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    packet_gains = tuple(float(value) for value in args.packet_gains.split(","))
    speaker_bonuses = tuple(
        float(value) for value in args.speaker_bonuses.split(","))
    windows = tuple(int(value) for value in args.windows.split(","))
    variants = tuple((window, packet_gain, speaker_bonus)
                     for window in windows for packet_gain in packet_gains
                     for speaker_bonus in speaker_bonuses)
    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    judges = _index(args.judge_lme, "question_id")
    judges.update(_index(args.judge_locomo, "question_id"))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    totals: dict[str, Counter] = defaultdict(Counter)
    strata: dict[str, Counter] = defaultdict(Counter)
    cached_memory = ""
    turns = {}
    by_position = {}

    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("has_turn_gold"):
                continue
            question_id = str(row["question_id"])
            question = questions[question_id]
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
            by_id = {str(item["turn_id"]): item for item in candidates}
            softened = bool(row["trace"].get("proof_priority_softened"))
            proof_bonus = float(row["trace"].get("proof_priority_bonus") or 0.5)

            def direct_seed(turn_id: str) -> float:
                item = by_id.get(turn_id)
                if item is None:
                    return 0.0
                return max(float(item.get(channel) or 0.0) for channel in (
                    "exact_score", "bm25_score", "dense_score"))

            def packet_signal(turn_id: str, window: int) -> float:
                turn = turns.get(turn_id)
                if turn is None:
                    return 0.0
                signal = 0.0
                for distance in range(1, window + 1):
                    decay = 1.0 / distance
                    for offset in (-distance, distance):
                        neighbor_id = by_position.get(
                            (turn.session_id, turn.turn_index + offset))
                        if not neighbor_id:
                            continue
                        neighbor = turns[neighbor_id]
                        # Dialogue packets are most useful when relevance crosses
                        # a speaker boundary (question->answer or image->comment).
                        cross_speaker = neighbor.speaker != turn.speaker
                        multiplier = 1.0 if cross_speaker else 0.5
                        signal = max(
                            signal,
                            direct_seed(neighbor_id) * decay * multiplier)
                return signal

            judged_correct = bool(judges.get(question_id, {}).get("correct"))
            benchmark = str(row["benchmark"])
            counters = (totals[benchmark], strata[str(row["stratum"])])
            baseline_hit = gold_ids <= set(row["retrieved_turn_ids"])
            for counter in counters:
                counter["questions"] += 1
                counter["baseline_all_hit"] += baseline_hit
                counter["judged_correct"] += judged_correct
                counter["judged_wrong"] += not judged_correct

            packet_cache = {
                window: {str(item["turn_id"]): packet_signal(
                    str(item["turn_id"]), window) for item in candidates}
                for window in windows}
            for window, packet_gain, speaker_bonus in variants:
                label = f"w{window}_p{packet_gain:g}_s{speaker_bonus:g}"

                def key(item: dict) -> tuple:
                    turn_id = str(item["turn_id"])
                    turn = turns.get(turn_id)
                    score = float(item.get("fused_score") or 0.0)
                    if softened and item.get("mandatory"):
                        score += proof_bonus
                    if softened:
                        score += packet_gain * packet_cache[window][turn_id]
                        if (turn is not None and
                                _explicit_speaker(question.query, turn.speaker)):
                            score += speaker_bonus
                        return (-score, int(item.get("rank") or math.inf))
                    # Preserve historical ordering outside structural floods.
                    return (-int(bool(item.get("mandatory"))), -score,
                            int(item.get("rank") or math.inf))

                selected = {str(item["turn_id"]) for item in
                            sorted(candidates, key=key)[:args.turn_limit]}
                hit = gold_ids <= selected
                for counter in counters:
                    counter[f"{label}_all_hit"] += hit
                    counter[f"{label}_gain"] += hit and not baseline_hit
                    counter[f"{label}_loss"] += baseline_hit and not hit
                    counter[f"{label}_wrong_all_hit_gain"] += (
                        not judged_correct and hit and not baseline_hit)
                    counter[f"{label}_correct_all_hit_loss"] += (
                        judged_correct and baseline_hit and not hit)

    store.close()
    payload = {
        "scope": "full_longmemeval500_locomo1540",
        "turn_limit": args.turn_limit,
        "variants": [
            {"label": f"w{window}_p{packet_gain:g}_s{speaker_bonus:g}",
             "window": window, "packet_gain": packet_gain,
             "speaker_bonus": speaker_bonus}
            for window, packet_gain, speaker_bonus in variants],
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
