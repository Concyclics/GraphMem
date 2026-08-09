#!/usr/bin/env python3
"""Phase B: which retrieval scheme, at what latency?

Phase A varied the *graph* and held retrieval fixed.  This holds the graph fixed
and varies retrieval, which is the other half of the ablation and the cheap half:
every arm here reuses one already-built database, so an arm costs a pass over the
question set rather than a rebuild.

Two things are measured per arm:

* accuracy -- `all_hit` (every gold turn packed) and turn `recall`, scored the
  same way `measure_v5_8_arm_recall.py` scores it, so the numbers are comparable
  across the two phases.  This is *retrieval* accuracy; no judge and no answer
  LLM run here.
* latency -- `NavigationResult.stage_latency_ms`, which the navigator already
  records on both the legacy and the harness path.  Reported as p50/p95 of the
  `total` key plus a per-stage mean, because the interesting question is not
  "is h10 slower" but "which stage does h10 spend it in".

The dense channel is indexed once, before any arm runs, and the resulting
`search` is shared.  Indexing mutates the database, so doing it per arm would
charge the first arm for work the rest inherit and make the latency column a
measurement of iteration order.
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
from graphmem.retrieval.navigator import HarnessProfile, NavigatorVariant  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


#: name -> navigator kwargs plus the evidence budget.  ``profile=None`` selects
#: the legacy path, where ``variant`` is what decides the scheme.
ARMS: dict[str, dict] = {
    # Axis A: the harness ladder, at the current default budget.
    "h0_n5@32":   {"profile": "h0",  "turns": 32},
    "h5_algebra@32": {"profile": "h5", "turns": 32},
    "h8_reservoir@32": {"profile": "h8", "turns": 32},
    "h9_facts@32": {"profile": "h9",  "turns": 32},
    "h10_ast@32": {"profile": "h10", "turns": 32},
    # Axis B: the evidence budget, on the richest scheme.
    "h10_ast@48": {"profile": "h10", "turns": 48},
    "h10_ast@64": {"profile": "h10", "turns": 64},
    # Axis C: legacy navigator variants, for the floor.
    "n1_raw_fusion@32": {"profile": None, "variant": "n1_raw_fusion", "turns": 32},
    "n5_set_cover@32": {"profile": None, "variant": "n5_set_cover", "turns": 32},
    # Axis D: hop decay and per-expansion pruning.  The graph term was a flag,
    # so a node at the hop cap scored what a seed scored, and nothing pruned an
    # expansion: 25.9% of a 32-turn pack went to graph-only turns yielding 0.4%.
    "h10_decay0.5@32": {"profile": "h10", "turns": 32, "decay": 0.5},
    "h10_decay0.3@32": {"profile": "h10", "turns": 32, "decay": 0.3},
    "h10_beam4@32": {"profile": "h10", "turns": 32, "beam": 4},
    "h10_beam2@32": {"profile": "h10", "turns": 32, "beam": 2},
    "h10_d0.5_b4@32": {"profile": "h10", "turns": 32, "decay": 0.5, "beam": 4},
    "h10_d0.3_b2@32": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2},
    "h10_d0.3_b2@64": {"profile": "h10", "turns": 64, "decay": 0.3, "beam": 2},
    # Axis W: the fusion itself.  Every arm below is h10 with decay+beam already
    # on, so W0 is the control and each W moves exactly one term.  The funnel
    # says gold is 100% in the reservoir and ranks at p50=10 / p90=202 against a
    # 32-turn pack, so the whole remaining gap is here.
    "W0_control": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2},
    "W1_operand_cap1": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "weights": {"operand_cap": 1}},
    "W2_operand_cap2": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "weights": {"operand_cap": 2}},
    "W3_operand_off": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                       "weights": {"operand": 0.0}},
    "W4_operand_half": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "weights": {"operand": 0.2}},
    "W5_exact1.0": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                    "weights": {"exact": 1.0}},
    "W6_graph0.4": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                    "weights": {"graph": 0.4}},
    "W7_binding_off": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                       "weights": {"binding": 0.0}},
    "W8_lexical_only": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "weights": {"operand": 0.0, "graph": 0.0, "binding": 0.0}},
    # Axis R: route sessions as units, then spend the pack inside them.  Single
    # session questions (78.7% of the set) score 0.6895 against an oracle of
    # 1.0000, so the whole deficit there is that sessions are never ranked.
    "R0_control": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2},
    "R1_route3": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "route": 3},
    "R2_route5": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "route": 5},
    "R3_route8": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "route": 8},
    "R4_route5_quota": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "route": 5, "quota": True},
    "R5_route8_quota": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "route": 8, "quota": True},
    "R6_route3_quota": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "route": 3, "quota": True},
    "R7_route5_q64": {"profile": "h10", "turns": 64, "decay": 0.3, "beam": 2,
                      "route": 5, "quota": True},
    "R8_h0_route5": {"profile": None, "variant": "n5_set_cover", "turns": 32, "route": 5},
    # Axis B: the evidence budget is capped at 32 turns while the 5000-token cap
    # runs at 25-29% -- a packed turn averages 45 tokens, so ~110 turns fit.
    # 32->48->64 was monotone (+5.5pp) and nothing has tested past it.
    "h0@32": {"profile": None, "variant": "n5_set_cover", "turns": 32},
    "h0@64": {"profile": None, "variant": "n5_set_cover", "turns": 64},
    "h0@96": {"profile": None, "variant": "n5_set_cover", "turns": 96},
    "h0@128": {"profile": None, "variant": "n5_set_cover", "turns": 128},
    "h10@96": {"profile": "h10", "turns": 96, "decay": 0.3, "beam": 2},
    "h10@128": {"profile": "h10", "turns": 128, "decay": 0.3, "beam": 2},
    # Every confirmed winner stacked: budget + routing + decay + beam + exact.
    "STACK@64": {"profile": "h10", "turns": 64, "decay": 0.3, "beam": 2, "route": 8,
                 "weights": {"exact": 1.0}},
    "STACK@128": {"profile": "h10", "turns": 128, "decay": 0.3, "beam": 2, "route": 8,
                  "weights": {"exact": 1.0}},
    # Axis F: isolate the h0/h10 gap.  The two pipelines differ in five places at
    # once, so "h10 loses by 13pp" has never been an ablation of Query IR or the
    # algebra.  Session flooding is the one difference attribution kept pointing
    # at; if h10+flood reaches h0, the gap is pool construction and Query IR was
    # never on trial.
    "F0_h10": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2},
    "F1_h10_flood8": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "flood": 8},
    "F2_h10_flood5": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "flood": 5},
    "F3_h10_flood8_q": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                        "flood": 8, "quota": True},
    "F4_h0": {"profile": None, "variant": "n5_set_cover", "turns": 32},
    "F5_h10_flood8@64": {"profile": "h10", "turns": 64, "decay": 0.3, "beam": 2, "flood": 8},
    "F6_h10_setcover": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True},
    "F7_h10_sc_route8": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2,
                         "setcover": True, "route": 8},
    "F8_h10_sc@64": {"profile": "h10", "turns": 64, "decay": 0.3, "beam": 2, "setcover": True},
    # Axis Q: isolate Query IR.  Packer and pool construction are now identical
    # across every arm (set_cover + route8), so what varies is only how much of
    # the compiled query reaches the pipeline.  Q0 is the full IR; each arm below
    # removes one channel the IR feeds.
    "Q0_full_IR": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True, "route": 8},
    "Q1_no_operand_score": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True, "route": 8, "weights": {"operand": 0.0}},
    "Q2_no_binding_score": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True, "route": 8, "weights": {"binding": 0.0}},
    "Q3_no_structured_ch": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True, "route": 8, "channels": ["source_projection", "lexical", "dense"]},
    "Q4_no_IR_signals": {"profile": "h10", "turns": 32, "decay": 0.3, "beam": 2, "setcover": True, "route": 8, "weights": {"operand": 0.0, "binding": 0.0},
                         "channels": ["source_projection", "lexical", "dense"]},
    "Q5_h5_algebra_sc": {"profile": "h5", "turns": 32, "decay": 0.3, "beam": 2,
                         "setcover": True, "route": 8},
    "Q6_h4_sched_sc": {"profile": "h4", "turns": 32, "decay": 0.3, "beam": 2,
                       "setcover": True, "route": 8},
    "Q7_h2_postings_sc": {"profile": "h2", "turns": 32, "decay": 0.3, "beam": 2,
                          "setcover": True, "route": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True,
                        help="one built graph; every arm reads it unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/v5/v5_8_units.json"))
    parser.add_argument("--arm", action="append", default=None,
                        help="repeatable; subset of the built-in arm names")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N questions (calibration runs)")
    parser.add_argument("--no-embedding", action="store_true")
    return parser.parse_args()


def score(store, records, spec, search) -> dict:
    navigator = GraphNavigator(
        store,
        dense_search=search,
        harness_profile=HarnessProfile(spec["profile"]) if spec["profile"] else None,
        variant=NavigatorVariant(spec["variant"]) if spec.get("variant") else NavigatorVariant.N5_SET_COVER,
        graph_hop_decay=float(spec.get("decay", 1.0)),
        expansion_beam=int(spec.get("beam", 0)),
        fusion_weights=spec.get("weights"),
        session_router_k=int(spec.get("route", 0)),
        session_flood_k=int(spec.get("flood", 0)),
        harness_set_cover=bool(spec.get("setcover", False)),
        fact_channels=spec.get("channels"),
        per_session_quota=bool(spec.get("quota", False)),
    )
    budget = QueryBudget(max_evidence_turns=spec["turns"])

    stats: dict[str, Counter] = defaultdict(Counter)
    per_question: dict[str, int] = {}
    totals: list[float] = []
    stages: dict[str, list[float]] = defaultdict(list)
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
        for stage, value in result.stage_latency_ms.items():
            stages[stage].append(float(value))
            if stage == "total":
                totals.append(float(value))
        if index_ % 200 == 0:
            print(f"    {index_}/{len(records)}  ({time.perf_counter()-started:.0f}s)", flush=True)

    totals.sort()
    def pct(p: float) -> float:
        return totals[min(len(totals) - 1, int(len(totals) * p))] if totals else 0.0
    return {
        "strata": {k: dict(v) for k, v in stats.items()},
        "per_question": per_question,
        "latency_ms": {
            "mean": statistics.fmean(totals) if totals else 0.0,
            "p50": pct(0.50), "p95": pct(0.95), "max": totals[-1] if totals else 0.0,
            "wall_seconds": time.perf_counter() - started,
        },
        "stage_mean_ms": {k: statistics.fmean(v) for k, v in sorted(stages.items()) if k != "total"},
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    names = args.arm or list(ARMS)
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; known: {list(ARMS)}")

    records = [row for row in load_full_questions(args.lme, args.locomo,
                                                  load_gold_turns(args.gold))
               if row.question.gold_turns]
    store = SQLiteGraphStore(args.db)
    present = {str(row["memory_id"]) for row in store._read("SELECT memory_id FROM conversations")}
    records = [row for row in records if row.question.memory_id in present]
    records.sort(key=lambda row: (row.question.memory_id, row.question.question_id))
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} gold-annotated questions over {len(present)} memories", flush=True)

    search = None
    if not args.no_embedding:
        index = QwenEmbeddingIndex(store, config, record_usage=False)
        indexed = sum(index.index_memory(memory_id) for memory_id in sorted(present))
        print(f"indexed {indexed} turn embeddings (shared by every arm)", flush=True)
        search = index.search

    results: dict[str, dict] = {}
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        results[name] = score(store, records, ARMS[name], search)
        row = results[name]
        overall = Counter()
        for counter in row["strata"].values():
            overall.update(counter)
        print(f"  all_hit={overall['all_hit']/max(1,overall['n']):.4f} "
              f"recall={overall['recall_num']/max(1,overall['recall_den']):.4f} "
              f"p50={row['latency_ms']['p50']:.1f}ms p95={row['latency_ms']['p95']:.1f}ms",
              flush=True)
    store.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'arm':22}{'all_hit':>9}{'recall':>8}{'p50ms':>9}{'p95ms':>9}{'mean':>9}")
    for name in names:
        row = results[name]
        overall = Counter()
        for counter in row["strata"].values():
            overall.update(counter)
        lat = row["latency_ms"]
        print(f"{name:22}{overall['all_hit']/max(1,overall['n']):9.4f}"
              f"{overall['recall_num']/max(1,overall['recall_den']):8.4f}"
              f"{lat['p50']:9.1f}{lat['p95']:9.1f}{lat['mean']:9.1f}")

    base = names[0]
    print(f"\npaired vs {base}:")
    for name in names[1:]:
        a, b = results[base]["per_question"], results[name]["per_question"]
        shared = set(a) & set(b)
        wins = sum(1 for q in shared if b[q] > a[q])
        losses = sum(1 for q in shared if b[q] < a[q])
        # McNemar exact, two-sided, on the discordant pairs only.
        n = wins + losses
        if n:
            from math import comb
            k = min(wins, losses)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
        else:
            p = 1.0
        print(f"  {name:22} wins {wins:4d}  loses {losses:4d}  "
              f"ties {len(shared)-wins-losses:4d}  McNemar p={p:.4f}")


if __name__ == "__main__":
    main()
