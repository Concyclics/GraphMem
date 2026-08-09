#!/usr/bin/env python3
"""Build the LongMemEval memories that carry turn-level gold, so LME retrieval
can be measured for the first time.

Every all_hit number measured so far is LoCoMo: LME gives each question its own
memory, turn-level gold exists for exactly 100 of the 500 questions (50
multi_session, 50 temporal_reasoning), and only one of those landed in the
10-memory samples.  So `lme_multi_session` judging at 0.489 currently cannot be
split into "the index missed the evidence" and "the answer stage had it and
missed anyway" -- which is the difference between rebuilding routing and
rebuilding the answer path.

These memories are a subset of the full 510, so the work is not thrown away.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller  # noqa: E402
from graphmem.build.canonicalize import PredicateCanonicalizer  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.devset import ingest_questions  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v5/v5_8_final.json")
    parser.add_argument("--lme", type=Path,
                        default=Path("../artifacts/data/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path,
                        default=Path("../artifacts/data/locomo10_graphmem.json"))
    parser.add_argument("--gold", type=Path,
                        default=Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"))
    parser.add_argument("--memory-workers", type=int, default=25)
    parser.add_argument("--include-locomo", action="store_true",
                        help="also build the 10 LoCoMo memories, for a single "
                             "database that covers both benchmarks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    records = load_full_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    by_memory: dict[str, list] = defaultdict(list)
    for record in records:
        if record.question.gold_turns:
            by_memory[record.question.memory_id].append(record.question)
    selected = sorted(key for key in by_memory if not key.startswith("locomo"))
    if args.include_locomo:
        selected += sorted(key for key in by_memory if key.startswith("locomo"))
    strata = defaultdict(int)
    for key in selected:
        strata[by_memory[key][0].stratum] += 1
    print(f"{len(selected)} memories: " +
          ", ".join(f"{k}={v}" for k, v in sorted(strata.items())), flush=True)

    store = SQLiteGraphStore(args.output / "graphmem.sqlite")
    ingest_questions(store, [by_memory[key][0] for key in selected])

    def build(memory_id: str):
        distiller = QwenSemanticDistiller(store, config, "v5.8-lme-gold")
        pipeline = GraphBuildPipeline(
            store, dataset_hash="v5.8-lme-gold", distiller=distiller,
            predicate_canonicalizer=PredicateCanonicalizer(store, config))
        started = time.perf_counter()
        try:
            manifest = pipeline.build(memory_id, config)
        except Exception as error:  # noqa: BLE001 - one bad memory must not stop the run
            return memory_id, None, time.perf_counter() - started, repr(error)[:200]
        return memory_id, manifest, time.perf_counter() - started, None

    rows, failures = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.memory_workers)) as pool:
        for index, (memory_id, manifest, seconds, error) in enumerate(
                pool.map(build, selected), 1):
            if error is not None:
                failures.append({"memory_id": memory_id, "error": error})
                print(f"  [{index}/{len(selected)}] {memory_id} FAILED: {error}", flush=True)
                continue
            total = int(dict(manifest.build_token_usage).get("total_tokens", 0))
            diagnostics = dict(manifest.build_diagnostics)
            rows.append({"memory_id": memory_id, "total_tokens": total,
                         "seconds": round(seconds, 1),
                         "coverage": diagnostics.get("semantic_terminal_turn_coverage"),
                         "fallback_scenes": diagnostics.get("extraction_fallback_scenes")})
            if index % 10 == 0:
                print(f"  [{index}/{len(selected)}] {total:,} tokens "
                      f"({time.perf_counter()-started:.0f}s elapsed)", flush=True)

    truncated = store._read(
        "SELECT COUNT(*) FROM llm_calls WHERE cached=0 AND stage LIKE 'scene_semantic%' "
        "AND response_json LIKE '%\"length\"%'")[0][0]
    calls = store._read(
        "SELECT COUNT(*) FROM llm_calls WHERE cached=0 AND stage LIKE 'scene_semantic%'")[0][0]
    totals = [row["total_tokens"] for row in rows] or [0]
    summary = {
        "config": str(args.config), "config_hash": config_hash(config),
        "memories": len(rows), "failures": failures,
        "tokens_max": max(totals), "tokens_mean": statistics.mean(totals),
        "truncation_rate": truncated / calls if calls else 0.0,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} built, {len(failures)} failed, "
          f"tokens max={summary['tokens_max']:,} mean={summary['tokens_mean']:,.0f}, "
          f"truncation={summary['truncation_rate']:.4f}, "
          f"{summary['wall_seconds']:.0f}s", flush=True)
    store.close()


if __name__ == "__main__":
    main()
