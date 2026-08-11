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
import math
import os
import sqlite3
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller  # noqa: E402
from graphmem.build.canonicalize import PredicateCanonicalizer  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_gold_turns  # noqa: E402
from graphmem.eval.devset import ingest_questions  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-db", type=Path, help="frozen artifact to copy already-built graphs from")
    parser.add_argument("--seed-report", type=Path,
                        help="build report paired with seed-db; carries diagnostics for copied memories")
    parser.add_argument("--seed-relation-embedding-db", type=Path,
                        help="immutable relation-vector cache used to seed a new sidecar")
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", help="override the build profile, e.g. b5")
    parser.add_argument(
        "--llm-model",
        help="override the build model without changing the frozen build profile")
    parser.add_argument(
        "--llm-base-url",
        help="override the build-model OpenAI-compatible base URL")
    parser.add_argument(
        "--llm-api-key-env", default="",
        help="environment variable containing the build-model API key")
    parser.add_argument(
        "--llm-request-profile", choices=("qwen", "openai"), default="qwen",
        help="select Qwen-local or OpenAI request fields for semantic extraction")
    parser.add_argument(
        "--llm-request-timeout-seconds", type=float, default=120.0,
        help="fail one external request so the resumable build can recover")
    parser.add_argument("--relation-mask-propagation", action="store_true")
    parser.add_argument("--rare-lexical-relation", action="store_true")
    parser.add_argument(
        "--predicate-family-state-relations", action="store_true",
        help=("join state routing by owner-role + morphological predicate family "
              "+ polarity + modality instead of exact predicate phrase"))
    parser.add_argument(
        "--enabled-relation-signals",
        help="comma-separated relation signal allow-list; empty disables all signals")
    parser.add_argument(
        "--embedding", action="store_true",
        help="index every source turn with the configured embedding model")
    parser.add_argument(
        "--relation-embedding-db", type=Path,
        help=("separate cache for RoutingCard and atomic-summary vectors; "
              "keeps online turn search free of graph-node embeddings"))
    parser.add_argument("--memory-workers", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--max-memories", type=int)
    parser.add_argument(
        "--development-set", action="store_true",
        help="expect the frozen 100 LongMemEval + 100 LoCoMo question subset")
    parser.add_argument("--require-zero-retries", action="store_true")
    parser.add_argument("--require-complete-diagnostics", action="store_true")
    parser.add_argument(
        "--rebuild-missing-diagnostics", action="store_true",
        help=("when resuming after an interrupted process, rebuild only graph "
              "versions absent from the existing report so every published "
              "Memory receives a same-run diagnostic row"))
    parser.add_argument(
        "--frozen-semantic-cache-only", action="store_true",
        help=("never call the semantic LLM; replay Full-arm responses by exact "
              "cache key or source-batch fingerprint"))
    parser.add_argument(
        "--frozen-semantic-source-report", type=Path,
        help=("Full-arm build report whose per-Memory budget-skipped call count "
              "is replayed as deterministic extraction fallback"))
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.frozen_semantic_source_report) != bool(
            args.frozen_semantic_cache_only):
        raise ValueError(
            "--frozen-semantic-cache-only and --frozen-semantic-source-report "
            "must be supplied together")
    frozen_fallback_calls: dict[str, int] = {}
    if args.frozen_semantic_source_report:
        frozen_payload = json.loads(args.frozen_semantic_source_report.read_text(
            encoding="utf-8"))
        frozen_fallback_calls = {
            str(row["memory_id"]): int(
                row.get("build_quality", {}).get("budget_skipped_scenes", 0))
            for row in frozen_payload.get("rows", ())
        }
    config = load_config(args.config)
    if args.llm_model or args.llm_base_url:
        config = replace(config, models=replace(
            config.models,
            llm_model=args.llm_model or config.models.llm_model,
            llm_base_url=args.llm_base_url or config.models.llm_base_url))
    if args.max_concurrency:
        config = replace(config, models=replace(config.models, max_concurrency=args.max_concurrency))
    if args.llm_request_timeout_seconds <= 0:
        raise ValueError("--llm-request-timeout-seconds must be positive")
    if args.profile:
        config = replace(config, profile=args.profile)
    if (args.relation_mask_propagation or args.rare_lexical_relation
            or args.predicate_family_state_relations):
        config = replace(config, edges=replace(
            config.edges,
            relation_mask_propagation=(
                args.relation_mask_propagation
                or config.edges.relation_mask_propagation),
            rare_lexical_relation=(
                args.rare_lexical_relation
                or config.edges.rare_lexical_relation),
            predicate_family_state_relations=(
                args.predicate_family_state_relations
                or config.edges.predicate_family_state_relations)))
    if args.enabled_relation_signals is not None:
        enabled = tuple(value.strip() for value in
                        args.enabled_relation_signals.split(",") if value.strip())
        config = replace(
            config, edges=replace(config.edges,
                                  enabled_relation_signals=enabled))
        # Re-run frozen-dataclass validation after ``replace`` has materialised
        # both layers; this makes invalid experiment labels fail before ingest.
        config.__post_init__()

    args.target_db.parent.mkdir(parents=True, exist_ok=True)
    def sqlite_backup(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as reader:
            with sqlite3.connect(target) as writer:
                reader.backup(writer)

    if args.seed_db and not args.target_db.exists():
        print(f"seeding {args.target_db} from {args.seed_db}", flush=True)
        sqlite_backup(args.seed_db, args.target_db)
    if (args.seed_relation_embedding_db and args.relation_embedding_db
            and not args.relation_embedding_db.exists()):
        print("seeding relation embeddings "
              f"{args.relation_embedding_db} from {args.seed_relation_embedding_db}",
              flush=True)
        sqlite_backup(args.seed_relation_embedding_db,
                      args.relation_embedding_db)

    questions = load_full_questions(
        args.lme, args.locomo, load_gold_turns(args.gold),
        expect_lme=(100 if args.development_set else 500),
        expect_locomo=(100 if args.development_set else 1540))
    store = SQLiteGraphStore(args.target_db)
    relation_store = (SQLiteGraphStore(args.relation_embedding_db)
                      if args.relation_embedding_db else None)

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

    wanted = [row[0] for row in store._read(
        "SELECT memory_id FROM conversations ORDER BY memory_id")]
    prior_report_rows: list[dict] = []
    prior_report_summary: dict = {}
    if args.report.exists():
        try:
            prior_payload = json.loads(args.report.read_text(encoding="utf-8"))
            prior_report_rows = list(prior_payload.get("rows", ()))
            prior_report_summary = dict(prior_payload.get("summary", {}))
        except (OSError, ValueError, TypeError):
            prior_report_rows = []
    built = {row[0] for row in store._read(
        "SELECT memory_id FROM graph_versions WHERE graph_checksum != ''")}
    if args.rebuild_missing_diagnostics and prior_report_rows:
        diagnosed = {str(row.get("memory_id") or "")
                     for row in prior_report_rows}
        replay = sorted(built - diagnosed)
        if replay:
            # Deleting the publication marker is sufficient: ``replace_graph``
            # atomically replaces every old graph/evidence row for the exact
            # Memory.  Conversation, frozen semantic cache and vectors remain
            # untouched.  This is intentionally unavailable without both the
            # explicit flag and an existing report.
            with store.transaction() as db:
                db.executemany(
                    "DELETE FROM graph_versions WHERE memory_id=?",
                    ((memory_id,) for memory_id in replay))
            built.difference_update(replay)
            print(f"replaying {len(replay)} graph versions missing diagnostics",
                  flush=True)
    pending = [item for item in wanted if item not in built]
    if args.max_memories:
        pending = pending[:args.max_memories]
    print(f"memories: {len(wanted)} total, {len(built)} already built, {len(pending)} to build",
          flush=True)

    started = time.perf_counter()
    results: list[dict] = []
    failures: list[dict] = []
    global_llm_limit = max(1, config.models.max_concurrency)
    memory_workers = max(1, args.memory_workers)
    per_memory_llm_workers = max(1, min(
        16, global_llm_limit // memory_workers))
    request_gate = threading.BoundedSemaphore(global_llm_limit)
    print(
        "LLM concurrency: "
        f"{memory_workers} memory workers x {per_memory_llm_workers} inner workers, "
        f"global cap {global_llm_limit}",
        flush=True,
    )

    build_api_key = "local"
    if args.llm_api_key_env:
        build_api_key = os.environ.get(args.llm_api_key_env, "")
        if not build_api_key:
            raise ValueError(
                f"missing build API key environment variable: {args.llm_api_key_env}")

    def build(memory_id: str) -> dict:
        from openai import OpenAI
        client = OpenAI(
            base_url=config.models.llm_base_url,
            api_key=build_api_key,
            timeout=args.llm_request_timeout_seconds,
            max_retries=0)
        distiller = QwenSemanticDistiller(
            store, config, "v5.6-full", client=client,
            request_gate=request_gate,
            worker_limit=per_memory_llm_workers,
            request_profile=args.llm_request_profile,
            frozen_cache_only=args.frozen_semantic_cache_only,
            frozen_fallback_calls=frozen_fallback_calls.get(memory_id, 0))
        if args.embedding:
            QwenEmbeddingIndex(store, config, batch_size=128).index_memory(
                memory_id)
        relation_index = (QwenEmbeddingIndex(
            relation_store, config, batch_size=128)
                          if relation_store is not None else None)
        pipeline = GraphBuildPipeline(
            store, dataset_hash="v5.6-full", distiller=distiller,
            predicate_canonicalizer=PredicateCanonicalizer(store, config),
            coarsen_vector_provider=(
                relation_index.embed_graph_nodes if relation_index else None),
            relation_vector_provider=(
                relation_index.embed_graph_nodes if relation_index else None))
        tick = time.perf_counter()
        manifest = pipeline.build(memory_id, config)
        usage = dict(manifest.build_token_usage)
        diagnostics = dict(manifest.build_diagnostics)
        method = dict(diagnostics.get("method", {}))
        cir = dict(method.get("cir", {}))
        budget_diagnostics = dict(diagnostics.get("build_token_budget") or {})
        return {"memory_id": memory_id,
                "input_tokens": int(usage.get("uncached_input_tokens", 0)),
                "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "tokens": (int(usage.get("uncached_input_tokens", 0))
                           + int(usage.get("output_tokens", 0))),
                "seconds": round(time.perf_counter() - tick, 1),
                "nodes": manifest.node_count, "edges": manifest.edge_count,
                "build_quality": {
                    "extraction_scenes": int(
                        diagnostics.get("extraction_scenes", 0)),
                    "extraction_success_scenes": int(
                        diagnostics.get("extraction_success_scenes", 0)),
                    "extraction_fallback_scenes": int(
                        diagnostics.get("extraction_fallback_scenes", 0)),
                    "extraction_retry_calls": int(
                        diagnostics.get("extraction_retry_calls", 0)),
                    "budget_degraded": bool(
                        int(budget_diagnostics.get("degraded_calls", 0))
                        or int(budget_diagnostics.get("skipped_scenes", 0))),
                    "budget_degraded_calls": int(
                        budget_diagnostics.get("degraded_calls", 0)),
                    "budget_skipped_scenes": int(
                        budget_diagnostics.get("skipped_scenes", 0)),
                    "budget_diagnostics": budget_diagnostics,
                },
                # Preserve the candidate funnel in the build artifact.  Edge
                # provenance alone can show what survived materialisation, but
                # cannot distinguish a disabled signal from one whose proposed
                # candidates were all removed by bounded pruning.
                "relation_candidate_diagnostics": {
                    "enabled_relation_signals": list(
                        method.get("enabled_relation_signals", ())),
                    "coarse_candidate_pairs": int(
                        cir.get("coarse_candidate_pairs", 0)),
                    "gated_child_pairs": int(cir.get("gated_child_pairs", 0)),
                    "atomic_relation_candidates_generated": int(
                        cir.get("atomic_relation_candidates_generated", 0)),
                    "atomic_relation_pairs_proposed": int(
                        cir.get("atomic_relation_pairs_proposed", 0)),
                    "relation_mask_pairs": int(
                        cir.get("relation_mask_pairs", 0)),
                    "relation_mask_counts": dict(
                        cir.get("relation_mask_counts", {})),
                    "atomic_candidate_source_counts": dict(
                        cir.get("atomic_candidate_source_counts", {})),
                    "atomic_candidate_signal_counts": dict(
                        cir.get("atomic_candidate_signal_counts", {})),
                    "accepted_pairs": int(cir.get("accepted_pairs", 0)),
                }}

    with ThreadPoolExecutor(max_workers=memory_workers) as pool:
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

    def nearest_stats(values, unit: str) -> dict:
        rows = sorted(int(value) for value in values)
        def nearest(p: float) -> int:
            return rows[max(0, math.ceil(p * len(rows)) - 1)] if rows else 0
        return {
            "count": len(rows), "mean": statistics.mean(rows) if rows else 0,
            "p50": nearest(0.50), "p95": nearest(0.95),
            "p99": nearest(0.99), "max": max(rows, default=0),
            "unit": unit, "percentile_method": "nearest_rank",
        }

    # Summarize every memory represented in the authority, including current-
    # version memories copied into a resumable seed.  Reporting only this
    # process's pending suffix previously made a 510-memory run look like a
    # 400-memory cost distribution.
    ledger_rows = store._read(
        "SELECT memory_id,"
        "SUM(CAST(json_extract(usage_json,'$.uncached_input_tokens') AS INTEGER)) input_tokens,"
        "SUM(CAST(json_extract(usage_json,'$.cached_input_tokens') AS INTEGER)) cached_input_tokens,"
        "SUM(CAST(json_extract(usage_json,'$.output_tokens') AS INTEGER)) output_tokens,"
        "SUM(retry_count) retry_count "
        "FROM llm_calls GROUP BY memory_id")
    ledger = []
    for row in ledger_rows:
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        ledger.append({
            "memory_id": str(row["memory_id"]),
            "input_tokens": input_tokens,
            "cached_input_tokens": int(row["cached_input_tokens"] or 0),
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "retry_count": int(row["retry_count"] or 0),
        })
    token_gate = int(config.models.semantic_max_tokens_per_memory)
    over_gate = [row["memory_id"] for row in ledger
                 if token_gate and row["total_tokens"] > token_gate]
    token_stats = {
        "input": nearest_stats(
            (row["input_tokens"] for row in ledger), "tokens_per_memory"),
        "output": nearest_stats(
            (row["output_tokens"] for row in ledger), "tokens_per_memory"),
        "total": nearest_stats(
            (row["total_tokens"] for row in ledger), "tokens_per_memory"),
    }
    seed_rows: list[dict] = []
    if args.seed_report and args.seed_report.exists():
        seed_rows = list(json.loads(args.seed_report.read_text(
            encoding="utf-8")).get("rows", ()))
    diagnostics_by_memory = {
        str(row["memory_id"]): row for row in (*seed_rows, *prior_report_rows)
        if str(row.get("memory_id") or "") in set(wanted)
    }
    diagnostics_by_memory.update({str(row["memory_id"]): row for row in results})
    all_build_rows = [diagnostics_by_memory[memory_id]
                      for memory_id in wanted if memory_id in diagnostics_by_memory]
    prior_preexisting = int(prior_report_summary.get(
        "memories_preexisting", len(built)))
    prior_built = int(prior_report_summary.get("memories_built", 0))
    cumulative_built = min(
        max(0, len(wanted) - prior_preexisting), prior_built + len(results))
    report = {
        "config": str(args.config), "config_hash": config_hash(config),
        "target_db": str(args.target_db),
        "memories_total": len(wanted),
        "memories_preexisting": prior_preexisting,
        "memories_built": cumulative_built, "failures": failures,
        "wall_minutes": round(
            float(prior_report_summary.get("wall_minutes", 0.0))
            + (time.perf_counter() - started) / 60, 1),
        "tokens_per_memory": token_stats["total"],
        "build_token_stats": token_stats,
        "tokens_total": sum(row["total_tokens"] for row in ledger),
        "token_ledger_memories": len(ledger),
        "cached_input_tokens": sum(row["cached_input_tokens"] for row in ledger),
        "retry_count": sum(row["retry_count"] for row in ledger),
        "token_gate": token_gate,
        "token_gate_violations": over_gate,
        "embedding": args.embedding,
        "relation_embedding_db": (
            str(args.relation_embedding_db)
            if args.relation_embedding_db else None),
        "relation_mask_propagation": config.edges.relation_mask_propagation,
        "rare_lexical_relation": config.edges.rare_lexical_relation,
        "predicate_family_state_relations": (
            config.edges.predicate_family_state_relations),
        "enabled_relation_signals": list(config.edges.enabled_relation_signals),
        "memory_workers": memory_workers,
        "llm_workers_per_memory": per_memory_llm_workers,
        "max_concurrency": global_llm_limit,
        "llm_model": config.models.llm_model,
        "llm_base_url": config.models.llm_base_url,
        "llm_api_key_env": args.llm_api_key_env or None,
        "llm_request_profile": args.llm_request_profile,
        "llm_request_timeout_seconds": args.llm_request_timeout_seconds,
        "frozen_semantic_cache_only": args.frozen_semantic_cache_only,
        "frozen_semantic_source_report": (
            str(args.frozen_semantic_source_report)
            if args.frozen_semantic_source_report else None),
        "build_diagnostic_memories": len(all_build_rows),
        "fallback_and_degradation": {
            "extraction_fallback_scenes": sum(int(
                row.get("build_quality", {}).get(
                    "extraction_fallback_scenes", 0)) for row in all_build_rows),
            "extraction_retry_calls": sum(int(
                row.get("build_quality", {}).get(
                    "extraction_retry_calls", 0)) for row in all_build_rows),
            "budget_degraded_memories": sum(bool(
                row.get("build_quality", {}).get(
                    "budget_degraded", False)) for row in all_build_rows),
            "budget_degraded_calls": sum(int(
                row.get("build_quality", {}).get(
                    "budget_degraded_calls", 0)) for row in all_build_rows),
            "budget_skipped_scenes": sum(int(
                row.get("build_quality", {}).get(
                    "budget_skipped_scenes", 0)) for row in all_build_rows),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": report, "rows": all_build_rows,
                                       "token_ledger": ledger}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2)[:2000])
    store.close()
    if relation_store is not None:
        relation_store.close()
    if failures:
        raise SystemExit(f"{len(failures)} memories failed; rerun to resume")
    if over_gate:
        raise SystemExit(f"{len(over_gate)} memories exceeded the build token gate")
    if args.require_zero_retries and report["retry_count"]:
        raise SystemExit(
            f"build retry gate failed: {report['retry_count']} request retries")
    if (args.require_complete_diagnostics
            and report["build_diagnostic_memories"] != len(wanted)):
        raise SystemExit(
            "build diagnostics incomplete: "
            f"{report['build_diagnostic_memories']}/{len(wanted)} memories")


if __name__ == "__main__":
    main()
