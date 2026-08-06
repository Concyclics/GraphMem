#!/usr/bin/env python3
"""Pre-check the V5.8 unit rebuild on a sample before spending the full 92 minutes.

Four numbers, three of them with a known baseline and one that decides whether
the design can work at all.

    A  tokens/memory max            <= 220,000   (V5.6: 210,801, headroom 9,199)
    B  finish_reason=length share   <  0.5%      (V5.6: 4.40%)
    C  summary is prose, not a triple bag        (V5.6: 0.00 -- it is a bag)
    D  entity strings recur across scenes        (no baseline; this is the gate)

D is the one that matters.  A scene summary and an entity list are only worth
their output tokens if the *same* entity is written the *same* way in two
different sessions -- that recurrence is the only thing that links the 2.68
sessions a LoCoMo cat1 question needs, and cat1 is the worst-routed category at
session_all_hit 0.592.  V5.7 failed in exactly this shape: a per-fact category
field produced 2,012 distinct strings over 2,222 facts because 129 independent
extraction calls cannot see each other's vocabulary.  Entities are a friendlier
target than categories -- a proper noun is copied, not invented -- but that is an
argument, not a measurement, which is what this script is for.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller  # noqa: E402
from graphmem.build.canonicalize import PredicateCanonicalizer  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.devset import ingest_questions  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402

GATES = {"tokens_max": 220_000, "truncation_rate": 0.005,
         "summary_prose": 0.80, "entity_cross_session": 0.30}

#: Function words a sentence has and a concatenation of subject-predicate-object
#: fragments does not.  The V5.6 summaries -- "Obsess has website URL
#: https://obsessvr.com/ Vertebrae has website URL ..." -- score near zero.
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "that", "which", "while", "after", "before", "when",
    "her", "his", "their", "its", "they", "it", "she", "he", "as", "than",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--memories", type=int, default=20)
    parser.add_argument("--memory-workers", type=int, default=5)
    parser.add_argument("--locomo-share", type=float, default=0.5,
                        help="LoCoMo is where the official turn-level gold is, and where the "
                             "cross-session routing this rebuild targets actually lives")
    return parser.parse_args()


def prose_score(text: str) -> float:
    """Share of tokens that are function words.  Prose ~0.3+, a triple bag ~0.05."""
    tokens = re.findall(r"[a-z']+", text.casefold())
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in FUNCTION_WORDS) / len(tokens)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = args.output_root / f"unit_gate_{args.config.stem}_{stamp}"
    root.mkdir(parents=True)

    records = load_full_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    by_memory: dict[str, list] = defaultdict(list)
    for record in records:
        by_memory[record.question.memory_id].append(record.question)
    locomo = sorted(key for key in by_memory if key.startswith("locomo"))
    lme = sorted(key for key in by_memory if not key.startswith("locomo"))
    want_locomo = min(len(locomo), int(args.memories * args.locomo_share))
    selected = locomo[:want_locomo] + lme[:args.memories - want_locomo]

    store = SQLiteGraphStore(root / "graphmem.sqlite")
    ingest_questions(store, [by_memory[key][0] for key in selected])
    print(f"ingested {len(selected)} memories "
          f"({want_locomo} LoCoMo, {len(selected)-want_locomo} LongMemEval)", flush=True)

    def build(memory_id: str):
        distiller = QwenSemanticDistiller(store, config, "v5.8-units")
        pipeline = GraphBuildPipeline(
            store, dataset_hash="v5.8-units", distiller=distiller,
            predicate_canonicalizer=PredicateCanonicalizer(store, config))
        started = time.perf_counter()
        return memory_id, pipeline.build(memory_id, config), time.perf_counter() - started

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.memory_workers)) as pool:
        for memory_id, manifest, seconds in pool.map(build, selected):
            total = int(dict(manifest.build_token_usage).get("total_tokens", 0))
            diagnostics = dict(manifest.build_diagnostics)
            rows.append({"memory_id": memory_id, "total_tokens": total,
                         "seconds": round(seconds, 1),
                         # Coverage is the quality axis that costs nothing to read
                         # and carries no judge noise, so a token ablation can be
                         # scored without spending a benchmark run on it.
                         "coverage": diagnostics.get("semantic_terminal_turn_coverage"),
                         "facts_per_scene": diagnostics.get("facts_per_scene_mean"),
                         "fallback_scenes": diagnostics.get("extraction_fallback_scenes")})
            print(f"  {memory_id}: {total:,} tokens in {seconds:.0f}s", flush=True)

    truncated = store._read(
        "SELECT COUNT(*) FROM llm_calls WHERE cached=0 AND stage LIKE 'scene_semantic%' "
        "AND response_json LIKE '%\"length\"%'")[0][0]
    calls = store._read(
        "SELECT COUNT(*) FROM llm_calls WHERE cached=0 AND stage LIKE 'scene_semantic%'")[0][0]

    prose, lengths, cross, per_memory_entities = [], [], [], []
    samples: list[str] = []
    for memory_id in selected:
        scenes = [node for node in store.nodes(memory_id) if node.node_type == NodeType.SCENE]
        # Scene summaries live on the SCENE node; entities ride the routing
        # fields, so read them back off the graph rather than the packets.
        entity_sessions: dict[str, set[str]] = defaultdict(set)
        for node in scenes:
            summary = str(node.summary or "")
            if summary:
                prose.append(prose_score(summary)); lengths.append(len(summary))
                if len(samples) < 6:
                    samples.append(summary[:150])
            session = str(node.attributes.get("session_id", ""))
            for name in node.attributes.get("entities", ()) or ():
                entity_sessions[str(name).casefold()].add(session)
        if entity_sessions:
            per_memory_entities.append(len(entity_sessions))
            cross.append(sum(1 for sessions in entity_sessions.values() if len(sessions) > 1)
                         / len(entity_sessions))

    totals = [row["total_tokens"] for row in rows]
    measured = {
        "tokens_max": max(totals), "tokens_mean": statistics.mean(totals),
        "truncation_rate": truncated / calls if calls else 0.0,
        "summary_prose": statistics.mean(prose) if prose else 0.0,
        "summary_chars_mean": statistics.mean(lengths) if lengths else 0.0,
        "entity_cross_session": statistics.mean(cross) if cross else 0.0,
        "entities_per_memory": statistics.mean(per_memory_entities) if per_memory_entities else 0.0,
        "coverage": statistics.mean(float(row["coverage"] or 0) for row in rows),
        "facts_per_scene": statistics.mean(float(row["facts_per_scene"] or 0) for row in rows),
        "seconds_per_memory": statistics.mean(row["seconds"] for row in rows),
    }
    verdict = {
        "A_tokens": measured["tokens_max"] <= GATES["tokens_max"],
        "B_truncation": measured["truncation_rate"] < GATES["truncation_rate"],
        "C_prose": measured["summary_prose"] >= 0.15,
        "D_entity_cross_session": measured["entity_cross_session"] >= GATES["entity_cross_session"],
    }
    summary = {"config": str(args.config), "config_hash": config_hash(config),
               "memories": len(rows), "gates": GATES, "measured": measured,
               "verdict": verdict, "passed": all(verdict.values()), "rows": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nsample scene summaries:")
    for text in samples:
        print(f"  {text!r}")
    print("\n" + json.dumps({k: summary[k] for k in ("measured", "verdict", "passed")}, indent=2))
    if not verdict["D_entity_cross_session"]:
        print("\nD FAILED: entity strings do not recur across sessions, so they cannot link the "
              "2.68 sessions a cat1 question needs. Do not spend the full rebuild.", flush=True)
    store.close()


if __name__ == "__main__":
    main()
