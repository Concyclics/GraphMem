#!/usr/bin/env python3
"""Does ranking the mandatory proof-unit turns by relevance recover lost evidence?

`pack()` puts every mandatory turn ahead of every ranked candidate, in the order
its proof units happen to declare them.  That is harmless when the mandatory set
fits the budget.  It is the *entire* selection when it does not, and for LoCoMo it
never fits: a question produces 109-125 proof units and 95-104 mandatory turns
against a 16-32 turn budget, so the pack measures 100% mandatory and the lexical
and dense candidate ranking never gets a single seat.

The declaration order comes from binding order and carries no relevance signal,
so today the pack is an arbitrary truncation of ~100 turns down to 32.  This
measures sorting that truncation by the score the candidate pool already carries.

Both arms run the real navigator -- an earlier attempt to simulate the control
offline from `candidate_scores` reproduced only 0.00-0.38 of the actual packed
set, precisely because it ignored the mandatory list.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import QueryBudget  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.retrieval import GraphNavigator  # noqa: E402
from graphmem.retrieval.navigator import HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--profile", default="h10")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--max-questions", type=int)
    # Shard by memory, not by question: the navigator rebuilds its read view
    # whenever the memory changes, so splitting a memory across shards would pay
    # that cost once per shard.
    parser.add_argument("--embedding", action="store_true",
                        help="production always runs with the dense channel on")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v5/v5_6_budget220k_quoted.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SQLiteGraphStore(args.source_db, read_only=True)
    records = [row for row in load_full_questions(args.lme, args.locomo,
                                                  load_gold_turns(args.gold))
               if row.question.gold_turns]
    if args.max_questions:
        records = records[:args.max_questions]
    records.sort(key=lambda row: (row.question.memory_id, row.question.question_id))
    if args.shards > 1:
        memories = sorted({row.question.memory_id for row in records})
        mine = {key for index, key in enumerate(memories) if index % args.shards == args.shard}
        records = [row for row in records if row.question.memory_id in mine]
        print(f"shard {args.shard}/{args.shards}: {len(mine)} memories, {len(records)} questions",
              flush=True)

    budget = QueryBudget(max_evidence_turns=args.max_evidence_turns)
    # The dense channel has to be on.  Run without it, this A/B measured
    # +0.130 turn_all_hit; the full answer run, which always passes --embedding,
    # measured 0.551 -> 0.536 and judged accuracy fell on both benchmarks.  The
    # arms were internally comparable either way, but `fused_score` without the
    # dense term is a different ordering than the one production truncates by,
    # so the lexical-only result did not transfer.
    embedding = (QwenEmbeddingIndex(store, load_config(args.config), record_usage=False)
                 if args.embedding else None)
    search = embedding.search if embedding else None
    arms = {
        "control": GraphNavigator(store, dense_search=search,
                                  harness_profile=HarnessProfile(args.profile)),
        "rank_mandatory": GraphNavigator(store, dense_search=search,
                                         harness_profile=HarnessProfile(args.profile),
                                         rank_mandatory=True),
    }
    stats: dict[str, dict[str, Counter]] = {arm: defaultdict(Counter) for arm in arms}
    paired: list[tuple[str, int, int]] = []
    mandatory_sizes: list[int] = []
    turns: dict[str, object] = {}
    current = None
    started = time.perf_counter()

    for index, record in enumerate(records, 1):
        question = record.question
        if question.memory_id != current:
            current = question.memory_id
            turns = {turn.turn_id: turn for turn in store.turns(current)}
        gold = {(row.session_id, row.turn_index) for row in question.gold_turns}
        outcome: dict[str, int] = {}
        for name, navigator in arms.items():
            try:
                result = navigator.navigate(question.memory_id, question.query, budget)
            except Exception:  # noqa: BLE001 - one bad memory must not stop the sweep
                outcome = {}
                break
            hit = {(turns[t].session_id, turns[t].turn_index)
                   for t in result.retrieved_turn_ids if t in turns}
            counter = stats[name][question.stratum]
            counter["n"] += 1
            counter["all_hit"] += int(gold <= hit)
            counter["recall_num"] += len(hit & gold)
            counter["recall_den"] += len(gold)
            outcome[name] = int(gold <= hit)
            if name == "control":
                mandatory_sizes.append(sum(
                    1 for unit in result.proof_units if unit.mandatory
                    for _ in unit.source_turn_ids))
        if len(outcome) == 2:
            paired.append((question.stratum, outcome["control"], outcome["rank_mandatory"]))
        if index % 200 == 0:
            print(f"  {index}/{len(records)}  ({time.perf_counter()-started:.0f}s)", flush=True)
    store.close()

    print(f"\nmandatory turns per question: mean={statistics.mean(mandatory_sizes):.1f} "
          f"vs budget {args.max_evidence_turns}")
    print(f"\n{'stratum':28}{'n':>5}{'control':>18}{'rank_mandatory':>20}{'delta':>9}")
    print(f"{'':28}{'':5}{'all_hit recall':>18}{'all_hit recall':>20}{'all_hit':>9}")
    summary: dict[str, dict] = {}
    for stratum in sorted(stats["control"]):
        c, r = stats["control"][stratum], stats["rank_mandatory"][stratum]
        ca, ra = c["all_hit"] / max(1, c["n"]), r["all_hit"] / max(1, r["n"])
        cr = c["recall_num"] / max(1, c["recall_den"])
        rr = r["recall_num"] / max(1, r["recall_den"])
        print(f"{stratum:28}{c['n']:5d}{ca:11.3f}{cr:7.3f}{ra:13.3f}{rr:7.3f}{ra-ca:+9.3f}")
        summary[stratum] = {"n": c["n"], "control_all_hit": ca, "rank_all_hit": ra,
                            "control_recall": cr, "rank_recall": rr, "delta_all_hit": ra - ca}

    wins = sum(1 for _, c, r in paired if r > c)
    losses = sum(1 for _, c, r in paired if r < c)
    print(f"\npaired over {len(paired)} questions: rank wins {wins}, loses {losses}, "
          f"ties {len(paired)-wins-losses}")
    args.output.mkdir(parents=True, exist_ok=True)
    # Raw counters, not just the ratios, so shards can be merged exactly.
    raw = {arm: {stratum: dict(counter) for stratum, counter in strata.items()}
           for arm, strata in stats.items()}
    name = f"mandatory_rank_{args.shard}.json" if args.shards > 1 else "mandatory_rank.json"
    (args.output / name).write_text(
        json.dumps({"budget": args.max_evidence_turns,
                    "mandatory_mean": statistics.mean(mandatory_sizes) if mandatory_sizes else 0,
                    "mandatory_n": len(mandatory_sizes),
                    "summary": summary, "wins": wins, "losses": losses, "raw": raw}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
