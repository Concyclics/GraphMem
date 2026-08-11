#!/usr/bin/env python3
"""Full-corpus gate for a narrow declarative-prompt -> named-response closure.

The graph candidate reservoir already contains almost every annotated gold turn,
but relation questions can rank the prompt side of a dialogue highly while the
answering turn sits just below the fixed packing boundary.  This simulation
propagates score only across one immediate, cross-speaker boundary and only
when the responder is explicitly named in the question.  It does not use gold
labels to compute the score; labels and verdicts are consulted only afterward
to report gains and losses.
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


def _index(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return {str(row["question_id"]): row for line in handle
                if line.strip() and (row := json.loads(line))}


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _explicit_speakers(query: str, speakers: set[str]) -> frozenset[str]:
    query_words = _words(query)
    return frozenset(
        speaker for speaker in speakers
        if (speaker_words := _words(speaker))
        and not speaker_words <= {"user", "assistant", "system"}
        and speaker_words <= query_words)


def _csv(text: str, cast):
    return tuple(cast(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judge-lme", type=Path)
    parser.add_argument("--judge-locomo", type=Path)
    parser.add_argument("--seed-counts", default="4,8,16,32")
    parser.add_argument("--bonuses", default="0.25,0.5,0.75,1,1.5")
    parser.add_argument("--min-query-overlap", default="1,2,3")
    parser.add_argument("--windows", default="1,2,3,4")
    parser.add_argument("--turn-limit", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seed_counts = _csv(args.seed_counts, int)
    bonuses = _csv(args.bonuses, float)
    overlaps = _csv(args.min_query_overlap, int)
    windows = _csv(args.windows, int)
    variants = tuple((seed_count, bonus, overlap, window)
                     for seed_count in seed_counts
                     for bonus in bonuses for overlap in overlaps
                     for window in windows)
    questions = {row.question_id: row for row in load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold))}
    judges = _index(args.judge_lme)
    judges.update(_index(args.judge_locomo))
    store = SQLiteGraphStore(args.source_db, read_only=True)
    totals: dict[str, Counter] = defaultdict(Counter)
    strata: dict[str, Counter] = defaultdict(Counter)
    cached_memory = ""
    turns = {}
    by_position = {}
    speakers: set[str] = set()

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
                speakers = {turn.speaker for turn in memory_turns}
            gold_ids = {
                turn_id for ref in question.gold_turns
                if (turn_id := by_position.get((ref.session_id, ref.turn_index)))
            }
            if not gold_ids:
                continue
            candidates = list(row["candidate_scores"])
            candidates_by_id = {
                str(item["turn_id"]): item for item in candidates}
            explicit = _explicit_speakers(question.query, speakers)
            query_terms = content_terms(question.query)
            softened = bool(row["trace"].get("proof_priority_softened"))
            proof_bonus = float(row["trace"].get("proof_priority_bonus") or 0.5)
            judged = judges.get(question_id)
            judged_correct = bool(judged and judged.get("correct"))
            benchmark = str(row["benchmark"])
            counters = (totals[benchmark], strata[str(row["stratum"])])
            baseline_hit = gold_ids <= set(row["retrieved_turn_ids"])
            for counter in counters:
                counter["questions"] += 1
                counter["baseline_all_hit"] += baseline_hit
                counter["questions_with_explicit_speaker"] += bool(explicit)
                if judged is not None:
                    counter["judged_correct"] += judged_correct
                    counter["judged_wrong"] += not judged_correct

            for seed_count, bonus, min_overlap, window in variants:
                gains: dict[str, float] = defaultdict(float)
                admitted_pairs = 0
                for seed in candidates[:seed_count]:
                    source_id = str(seed["turn_id"])
                    source = turns.get(source_id)
                    if source is None:
                        continue
                    # This arm isolates the missing declarative case.  Existing
                    # dialogue closure already knows how to follow a question.
                    if "?" in source.raw_text:
                        continue
                    if len(query_terms & content_terms(source.raw_text)) < min_overlap:
                        continue
                    direct = max(float(seed.get(name) or 0.0) for name in (
                        "exact_score", "bm25_score", "dense_score"))
                    if direct <= 0:
                        continue
                    for distance in range(1, window + 1):
                        response_id = by_position.get(
                            (source.session_id, source.turn_index + distance))
                        response = turns.get(response_id) if response_id else None
                        if (response is None or response.speaker == source.speaker
                                or response.speaker not in explicit
                                or response.turn_id not in candidates_by_id):
                            continue
                        gains[response.turn_id] = max(
                            gains[response.turn_id], bonus * direct / distance)
                        admitted_pairs += 1

                def key(item: dict) -> tuple:
                    score = (float(item.get("fused_score") or 0.0)
                             + gains.get(str(item["turn_id"]), 0.0))
                    if softened:
                        score += proof_bonus * bool(item.get("mandatory"))
                        return (-score, int(item.get("rank") or math.inf))
                    return (-int(bool(item.get("mandatory"))), -score,
                            int(item.get("rank") or math.inf))

                selected = {str(item["turn_id"]) for item in
                            sorted(candidates, key=key)[:args.turn_limit]}
                hit = gold_ids <= selected
                label = f"s{seed_count}_b{bonus:g}_o{min_overlap}_w{window}"
                for counter in counters:
                    counter[f"{label}_all_hit"] += hit
                    counter[f"{label}_gain"] += hit and not baseline_hit
                    counter[f"{label}_loss"] += baseline_hit and not hit
                    counter[f"{label}_promoted_questions"] += bool(gains)
                    counter[f"{label}_admitted_pairs"] += admitted_pairs
                    if judged is not None:
                        counter[f"{label}_wrong_all_hit_gain"] += (
                            not judged_correct and hit and not baseline_hit)
                        counter[f"{label}_correct_all_hit_loss"] += (
                            judged_correct and baseline_hit and not hit)

    store.close()
    payload = {
        "scope": "full_longmemeval500_locomo1540",
        "turn_limit": args.turn_limit,
        "variants": [
            {"label": f"s{seed_count}_b{bonus:g}_o{overlap}_w{window}",
             "seed_count": seed_count, "bonus": bonus,
             "min_query_overlap": overlap, "window": window}
            for seed_count, bonus, overlap, window in variants],
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
