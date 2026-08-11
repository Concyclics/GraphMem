#!/usr/bin/env python3
"""Screen a bounded query-induced lexical witness closure on the full corpus.

The authority graph deliberately rejects broad lexical-only coarse edges: a
frequent topic word connects too many regions and makes graph navigation less
selective.  That does not rule out a *query-local* witness edge.  This offline
screen starts from a few high-ranked source turns and promotes another turn
only when the pair shares a memory-rare term.  Explicit speaker ownership is
respected when the question names a participant.  No graph, extraction cache,
answer request, or judge request is mutated by this script.

This is a full 500 + 1,540 gate.  Gold labels are used only for evaluation and
never enter the closure score.
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
from graphmem.text import content_terms  # noqa: E402


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _explicit_speakers(query: str, speakers: set[str]) -> frozenset[str]:
    query_words = set(_normalise(query).split())
    matches = set()
    for speaker in speakers:
        words = _normalise(speaker).split()
        if (words and set(words) <= query_words
                and not set(words) <= {"user", "assistant", "system"}):
            matches.add(speaker)
    return frozenset(matches)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path)
    parser.add_argument("--judge-locomo", type=Path)
    parser.add_argument("--seed-counts", default="8,16,24")
    parser.add_argument("--bonuses", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--rare-df", type=int, default=4)
    parser.add_argument("--min-shared-terms", type=int, default=1)
    parser.add_argument("--require-explicit-speaker", action="store_true")
    parser.add_argument("--turn-limit", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seed_counts = _parse_ints(args.seed_counts)
    bonuses = _parse_floats(args.bonuses)
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
    terms_by_turn: dict[str, frozenset[str]] = {}
    document_frequency: Counter[str] = Counter()
    speakers: set[str] = set()

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
                terms_by_turn = {
                    turn.turn_id: content_terms(turn.raw_text)
                    for turn in memory_turns}
                document_frequency = Counter(
                    term for terms in terms_by_turn.values() for term in terms)
                speakers = {turn.speaker for turn in memory_turns}
            gold_ids = {
                turn_id for ref in question.gold_turns
                if (turn_id := by_position.get((ref.session_id, ref.turn_index)))
            }
            if not gold_ids:
                continue

            candidates = list(row["candidate_scores"])
            observed = set(row["retrieved_turn_ids"])
            explicit = _explicit_speakers(question.query, speakers)
            benchmark = str(row["benchmark"])
            judged = judges.get(question.question_id)
            judged_correct = bool(judged and judged.get("correct"))
            counters = (totals[benchmark], strata[str(row["stratum"])])
            observed_hit = gold_ids <= observed
            for counter in counters:
                counter["questions"] += 1
                counter["observed_all_hit"] += observed_hit
                if judged is not None:
                    counter["judged_correct"] += judged_correct
                    counter["judged_wrong"] += not judged_correct

            for seed_count in seed_counts:
                seed_rows = candidates[:seed_count]
                # A named owner is a hard scope for the local witness closure,
                # not for the base rank.  It stops a bridge term mentioned by
                # the other participant from propagating into the named user's
                # evidence packet.
                eligible_seed_ids = [
                    str(item["turn_id"]) for item in seed_rows
                    if (not explicit or turns[str(item["turn_id"])].speaker in explicit)
                ]
                rare_seed_terms = frozenset(
                    term for turn_id in eligible_seed_ids
                    for term in terms_by_turn.get(turn_id, ())
                    if document_frequency[term] <= args.rare_df)
                for bonus in bonuses:
                    def closure_gain(item: dict) -> float:
                        if args.require_explicit_speaker and not explicit:
                            return 0.0
                        turn_id = str(item["turn_id"])
                        turn = turns[turn_id]
                        if explicit and turn.speaker not in explicit:
                            return 0.0
                        shared = rare_seed_terms & terms_by_turn.get(turn_id, ())
                        if len(shared) < args.min_shared_terms:
                            return 0.0
                        # IDF orders multiple valid witnesses but the entire
                        # closure stays under one finite bonus.  It can rerank a
                        # close boundary; it cannot overwhelm all base channels.
                        strength = sum(math.log((len(turns) + 1) /
                                                (document_frequency[term] + 1))
                                       for term in shared)
                        return bonus * min(1.0, strength / 4.0)

                    softened = bool(row["trace"].get(
                        "proof_priority_softened"))
                    proof_bonus = float(row["trace"].get(
                        "proof_priority_bonus") or 0.5)

                    def order_key(item: dict) -> tuple:
                        score = (float(item.get("fused_score") or 0.0)
                                 + closure_gain(item))
                        if softened:
                            score += proof_bonus * bool(item.get("mandatory"))
                            return (-score, int(item.get("rank") or math.inf),
                                    str(item["turn_id"]))
                        return (-int(bool(item.get("mandatory"))), -score,
                                int(item.get("rank") or math.inf),
                                str(item["turn_id"]))

                    ordered = sorted(candidates, key=order_key)
                    selected = {str(item["turn_id"])
                                for item in ordered[:args.turn_limit]}
                    hit = gold_ids <= selected
                    label = f"seed_{seed_count}_bonus_{bonus:g}"
                    for counter in counters:
                        counter[f"{label}_all_hit"] += hit
                        counter[f"{label}_gain"] += hit and not observed_hit
                        counter[f"{label}_loss"] += observed_hit and not hit
                        if judged is not None:
                            counter[f"{label}_wrong_all_hit_gain"] += (
                                not judged_correct and hit and not observed_hit)
                            counter[f"{label}_correct_all_hit_loss"] += (
                                judged_correct and observed_hit and not hit)

    store.close()
    payload = {
        "scope": "full_longmemeval500_locomo1540",
        "turn_limit": args.turn_limit,
        "rare_df": args.rare_df,
        "min_shared_terms": args.min_shared_terms,
        "require_explicit_speaker": args.require_explicit_speaker,
        "seed_counts": list(seed_counts),
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
