#!/usr/bin/env python3
"""Build every memory the full benchmark needs, resumably.

510 memories: LongMemEval 500 plus LoCoMo 10.  About 110 already exist in the
frozen artifact, and their extraction is byte-identical, so the copy is seeded
from it and only the missing ~400 are built.

Resume is by ``graph_versions``: a memory that already carries a graph is
skipped.  A run killed halfway therefore restarts at the memory it died on with
every earlier memory and every cached LLM call intact -- at ~229K tokens per
memory a non-resumable build would be an unacceptable thing to lose.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
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
    parser.add_argument("--seed-db", type=Path, help="frozen artifact to copy already-built graphs from")
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--memory-workers", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--max-memories", type=int)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_concurrency:
        config = replace(config, models=replace(config.models, max_concurrency=args.max_concurrency))

    args.target_db.parent.mkdir(parents=True, exist_ok=True)
    if args.seed_db and not args.target_db.exists():
        print(f"seeding {args.target_db} from {args.seed_db}", flush=True)
        shutil.copy2(args.seed_db, args.target_db)

    questions = load_full_questions(args.lme, args.locomo, load_gold_turns(args.gold))
    store = SQLiteGraphStore(args.target_db)

    # Ingest only memories the database does not already hold.  Re-ingesting a
    # present memory rewrites every one of its turns, and each rewrite clears the
    # FTS row by a full scan of an unindexed virtual table: measured at 4.8GB of
    # writes and ~15s per memory, against 0.06s for a genuinely new one.  The
    # seed artifact overlaps LongMemEval 500 by about 100 memories, so this is
    # the difference between ~30 seconds and over two hours -- and it makes
    # resume cheap for free.
    present = {row[0] for row in store._read("SELECT memory_id FROM conversations")}
    fresh = [row.question for row in questions if row.question.memory_id not in present]
    print(f"ingesting {len({row.memory_id for row in fresh})} new memories "
          f"({len(present)} already present)", flush=True)
    if fresh:
        ingest_questions(store, fresh)

    built = {row[0] for row in store._read(
        "SELECT memory_id FROM graph_versions WHERE graph_checksum != ''")}
    wanted = [row[0] for row in store._read(
        "SELECT memory_id FROM conversations ORDER BY memory_id")]
    pending = [item for item in wanted if item not in built]
    if args.max_memories:
        pending = pending[:args.max_memories]
    print(f"memories: {len(wanted)} total, {len(built)} already built, {len(pending)} to build",
          flush=True)

    started = time.perf_counter()
    results: list[dict] = []
    failures: list[dict] = []

    def build(memory_id: str) -> dict:
        distiller = QwenSemanticDistiller(store, config, "v5.6-full")
        pipeline = GraphBuildPipeline(
            store, dataset_hash="v5.6-full", distiller=distiller,
            predicate_canonicalizer=PredicateCanonicalizer(store, config))
        tick = time.perf_counter()
        manifest = pipeline.build(memory_id, config)
        usage = dict(manifest.build_token_usage)
        return {"memory_id": memory_id, "tokens": int(usage.get("total_tokens", 0)),
                "seconds": round(time.perf_counter() - tick, 1),
                "nodes": manifest.node_count, "edges": manifest.edge_count}

    with ThreadPoolExecutor(max_workers=max(1, args.memory_workers)) as pool:
        futures = {pool.submit(build, item): item for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            memory_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - one bad memory must not end the run
                failures.append({"memory_id": memory_id, "error": repr(error)[:400]})
                print(f"[build error] {memory_id}: {error!r}"[:300], flush=True)
                continue
            if index % 10 == 0:
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                print(f"  built {index}/{len(pending)}  {elapsed/60:.1f}m elapsed, "
                      f"~{(len(pending)-index)/max(rate,1e-9)/60:.0f}m left", flush=True)

    tokens = sorted(row["tokens"] for row in results) or [0]
    report = {
        "config": str(args.config), "config_hash": config_hash(config),
        "target_db": str(args.target_db),
        "memories_total": len(wanted), "memories_preexisting": len(built),
        "memories_built": len(results), "failures": failures,
        "wall_minutes": round((time.perf_counter() - started) / 60, 1),
        "tokens_per_memory": {
            "mean": statistics.mean(tokens), "p50": statistics.median(tokens),
            "p95": tokens[max(0, int(0.95 * len(tokens)) - 1)], "max": max(tokens)},
        "tokens_total": sum(tokens),
        "memory_workers": args.memory_workers, "max_concurrency": config.models.max_concurrency,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": report, "rows": results}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2)[:2000])
    store.close()
    if failures:
        raise SystemExit(f"{len(failures)} memories failed; rerun to resume")


if __name__ == "__main__":
    main()
