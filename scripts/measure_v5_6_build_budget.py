#!/usr/bin/env python3
"""Build a few memories cold under a config and report the real token cost.

``ModelConfig.semantic_average_tokens_per_memory`` declared 220,000 from V5
onward and had no consumer, so the only way to know what a build actually costs
is to run one and read the ledger.  This builds into a throwaway database with a
cold cache, so the numbers are cold-build cost, not cache-warm cost.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller  # noqa: E402
from graphmem.build.canonicalize import PredicateCanonicalizer  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.eval.devset import ingest_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--memories", type=int, default=4)
    parser.add_argument("--memory-workers", type=int, default=2)
    parser.add_argument("--benchmark", choices=("longmemeval", "locomo", "both"),
                        default="longmemeval")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = args.output_root / f"build_budget_{args.label or args.config.stem}_{stamp}"
    root.mkdir(parents=True)

    questions = load_dev_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    if args.benchmark != "both":
        questions = [row for row in questions if row.benchmark == args.benchmark]
    # One question per memory; memories are the build unit.
    by_memory: dict[str, object] = {}
    for question in questions:
        by_memory.setdefault(question.memory_id, question)
    selected = [by_memory[key] for key in sorted(by_memory)][:args.memories]

    store = SQLiteGraphStore(root / "graphmem.sqlite")
    ingest_questions(store, selected)
    print(f"ingested {len(selected)} memories into {root/'graphmem.sqlite'}", flush=True)

    def build(question):
        distiller = QwenSemanticDistiller(store, config, "v5.6-budget")
        pipeline = GraphBuildPipeline(
            store, dataset_hash="v5.6-budget", distiller=distiller,
            predicate_canonicalizer=PredicateCanonicalizer(store, config))
        started = time.perf_counter()
        manifest = pipeline.build(question.memory_id, config)
        return question.memory_id, manifest, (time.perf_counter() - started)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.memory_workers)) as pool:
        for memory_id, manifest, seconds in pool.map(build, selected):
            usage = dict(manifest.build_token_usage)
            diagnostics = dict(manifest.build_diagnostics)
            ledger = diagnostics.get("build_token_budget") or {}
            total = int(usage.get("total_tokens", 0))
            rows.append({
                "memory_id": memory_id, "total_tokens": total,
                "seconds": round(seconds, 1),
                "stage_tokens": diagnostics.get("cold_equivalent_stage_tokens"),
                "ledger": ledger,
                "extraction_scenes": diagnostics.get("extraction_scenes"),
                "extraction_fallback_scenes": diagnostics.get("extraction_fallback_scenes"),
                "semantic_terminal_turn_coverage": diagnostics.get("semantic_terminal_turn_coverage"),
                "facts_per_scene_mean": diagnostics.get("facts_per_scene_mean"),
            })
            print(f"  {memory_id}: {total:,} tokens in {seconds:.0f}s "
                  f"(degraded={ledger.get('degraded_calls')}, skipped={ledger.get('skipped_scenes')})",
                  flush=True)

    totals = sorted(row["total_tokens"] for row in rows)
    ceiling = config.models.semantic_max_tokens_per_memory
    summary = {
        "config": str(args.config), "config_hash": config_hash(config),
        "ceiling": ceiling, "memories": len(rows),
        "tokens_per_memory": {
            "mean": statistics.mean(totals), "p50": statistics.median(totals),
            "max": max(totals), "min": min(totals),
        },
        "over_ceiling": sum(1 for value in totals if ceiling and value > ceiling),
        "mean_fact_coverage": statistics.mean(
            float(row["semantic_terminal_turn_coverage"] or 0) for row in rows),
        "total_fallback_scenes": sum(int(row["extraction_fallback_scenes"] or 0) for row in rows),
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
