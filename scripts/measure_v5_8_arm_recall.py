#!/usr/bin/env python3
"""Does a Phase A build knob change what retrieval can find?

Phase A scored its arms on `semantic_terminal_turn_coverage`, which counts the
turns some fact cites divided by all turns in the memory.  That number cannot
distinguish a fact that cites the turn a question needs from a fact that cites a
turn nobody asks about, and it rises whenever the model simply emits more facts:
the quote arm went from 1.84 to 2.95 facts per scene and bought 0.087 coverage,
which is *less* new ground per fact than the baseline managed.  So coverage
cannot decide which knob to keep.

This runs the real navigator against each arm's graph and scores the packed
turns against gold, which is the thing the answer stage actually sees.  Still no
judge and no answer LLM -- retrieval is deterministic, so one pass per arm is a
complete measurement rather than a sample.

Two confounds are removed first.

Dense channel.  The arm databases carry only predicate embeddings, written by
the canonicalizer, and their count tracks the arm's fact count (2,562 for the
baseline, 4,108 for the quote arm).  The turn embeddings the navigator's dense
channel searches were never written at all.  Left alone, an arm's dense channel
would differ from its neighbour's for a reason that has nothing to do with the
knob, so every arm is indexed to full turn coverage before it is scored.  Turn
text is identical across arms, so this makes the dense channel identical too and
the graph the only thing that varies.  It also matches production, which always
answers with `--embedding`.

Question set.  Each arm holds the same 10 memories, and only questions whose
memory is present and whose gold turns are annotated are scored, so the arms are
paired question by question.
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
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DB",
                        help="repeatable, e.g. --arm B0_core=../artifacts/.../graphmem.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v5/v5_8_units.json"),
                        help="retrieval settings only; identical for every arm")
    parser.add_argument("--profile", default="h10")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--no-embedding", action="store_true",
                        help="skip the dense channel; the lexical-only ordering is a "
                             "different one from the one production truncates by")
    return parser.parse_args()


def score(db: Path, records, config, args) -> dict:
    """Index turn embeddings, navigate every question, count gold turns packed."""
    store = SQLiteGraphStore(db)
    memories = sorted({row.question.memory_id for row in records})
    search = None
    if not args.no_embedding:
        index = QwenEmbeddingIndex(store, config, record_usage=False)
        indexed = sum(index.index_memory(memory_id) for memory_id in memories)
        print(f"    indexed {indexed} turn embeddings", flush=True)
        search = index.search
    navigator = GraphNavigator(store, dense_search=search,
                               harness_profile=HarnessProfile(args.profile))
    budget = QueryBudget(max_evidence_turns=args.max_evidence_turns)

    stats: dict[str, Counter] = defaultdict(Counter)
    per_question: dict[str, int] = {}
    turns: dict[str, object] = {}
    current = None
    started = time.perf_counter()
    for index_, record in enumerate(records, 1):
        question = record.question
        if question.memory_id != current:
            current = question.memory_id
            turns = {turn.turn_id: turn for turn in store.turns(current)}
        result = navigator.navigate(question.memory_id, question.query, budget)
        hit = {(turns[t].session_id, turns[t].turn_index)
               for t in result.retrieved_turn_ids if t in turns}
        gold = {(row.session_id, row.turn_index) for row in question.gold_turns}
        counter = stats[question.stratum]
        counter["n"] += 1
        counter["all_hit"] += int(gold <= hit)
        counter["recall_num"] += len(hit & gold)
        counter["recall_den"] += len(gold)
        per_question[question.question_id] = int(gold <= hit)
        if index_ % 200 == 0:
            print(f"    {index_}/{len(records)}  ({time.perf_counter()-started:.0f}s)", flush=True)
    store.close()
    return {"strata": {k: dict(v) for k, v in stats.items()}, "per_question": per_question}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    arms = dict(spec.split("=", 1) for spec in args.arm)

    records = [row for row in load_full_questions(args.lme, args.locomo,
                                                  load_gold_turns(args.gold))
               if row.question.gold_turns]
    # Every arm holds the same 10 memories; read them off the first database so
    # the question set is defined by the data, not by a hardcoded list.
    probe = SQLiteGraphStore(Path(next(iter(arms.values()))), read_only=True)
    present = {str(row["memory_id"]) for row in probe._read("SELECT memory_id FROM conversations")}
    probe.close()
    records = [row for row in records if row.question.memory_id in present]
    records.sort(key=lambda row: (row.question.memory_id, row.question.question_id))
    print(f"{len(records)} gold-annotated questions over {len(present)} memories", flush=True)

    results: dict[str, dict] = {}
    for name, db in arms.items():
        print(f"\n=== {name} ===", flush=True)
        results[name] = score(Path(db), records, config, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    strata = sorted({s for row in results.values() for s in row["strata"]})
    print(f"\n{'stratum':26}{'n':>5}" + "".join(f"{name:>18}" for name in arms))
    print(f"{'':26}{'':5}" + "".join(f"{'all_hit recall':>18}" for _ in arms))
    for stratum in strata:
        counters = [results[name]["strata"].get(stratum, {}) for name in arms]
        n = max(int(c.get("n", 0)) for c in counters)
        cells = ""
        for c in counters:
            a = c.get("all_hit", 0) / max(1, c.get("n", 0))
            r = c.get("recall_num", 0) / max(1, c.get("recall_den", 0))
            cells += f"{a:11.3f}{r:7.3f}"
        print(f"{stratum:26}{n:5d}{cells}")

    totals = {}
    for name in arms:
        c = Counter()
        for counter in results[name]["strata"].values():
            c.update(counter)
        totals[name] = c
        print(f"\n{name:16} overall all_hit={c['all_hit']/max(1,c['n']):.4f} "
              f"recall={c['recall_num']/max(1,c['recall_den']):.4f}")

    base = next(iter(arms))
    for name in list(arms)[1:]:
        a, b = results[base]["per_question"], results[name]["per_question"]
        shared = set(a) & set(b)
        wins = sum(1 for q in shared if b[q] > a[q])
        losses = sum(1 for q in shared if b[q] < a[q])
        print(f"paired vs {base}: {name} wins {wins}, loses {losses}, "
              f"ties {len(shared)-wins-losses}")


if __name__ == "__main__":
    main()
