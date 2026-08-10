#!/usr/bin/env python3
"""Navigate, then answer, then emit judge-ready rows.

This is the first V5 stage that produces an answer at all, so it is also the
first chance to check whether ``turn_all_hit`` -- the proxy every h0..h9
conclusion rests on -- predicts judged correctness.  ``answers.jsonl`` is
written in the field shape ``evaluate_mem0_judge.py`` expects; ``retrieval.jsonl``
carries the per-question retrieval metrics so the two can be joined afterwards
by ``analyze_v5_6_proxy_validity.py``.

Gold labels are read here, in the runner, and are never passed to
``graphmem.answer``.  The answer stage sees only the question and the graph.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import AnswerConfig, AnswerStage, prompt_contract  # noqa: E402
from graphmem.config import config_hash, load_config  # noqa: E402
from graphmem.domain import dataclass_dict  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.eval.metrics import navigation_metrics  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="h9")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int,
                        help="override retrieval evidence-token budget")
    parser.add_argument("--max-answer-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=0,
                        help="0 omits the API output cap; positive values enable an ablation cap")
    parser.add_argument("--answer-model",
                        help="answer backbone; defaults to models.llm_model")
    parser.add_argument("--answer-base-url",
                        help="answer endpoint; defaults to models.llm_base_url")
    parser.add_argument("--answer-api-key-env",
                        help="environment variable containing the answer endpoint key")
    parser.add_argument("--answer-request-profile",
                        choices=("qwen", "openai", "omit"), default="qwen")
    parser.add_argument("--packing-model",
                        help="tokenizer model used only to enforce the evidence budget")
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="explicit seed sent to the answer service")
    parser.add_argument("--span-window", type=int, default=-1,
                        help="-1 renders whole turns; >=0 renders cited spans widened by N chars")
    parser.add_argument("--evidence-order",
                        choices=("chronological", "relevance", "adaptive", "topological"),
                        default="chronological",
                        help="render by source time, retrieval rank, or query-directed order")
    parser.add_argument("--no-closed-form", action="store_true")
    parser.add_argument(
        "--candidate-answer-injection", action="store_true",
        help=("inject the algebraic draft into the answer prompt; off by default "
              "because incorrect drafts can anchor the answer model"))
    parser.add_argument("--no-h10-owner-rescue", action="store_true")
    parser.add_argument("--no-h10-traversal", action="store_true")
    parser.add_argument(
        "--no-hierarchical-routing", action="store_true",
        help=("disable coarse-to-fine seed routing while retaining the same "
              "QueryIR, candidate reservoirs and answer contract; intended "
              "for graph-structure ablations"))
    parser.add_argument("--no-manifest-collection-key", action="store_true")
    parser.add_argument("--rank-mandatory", action="store_true",
                        help="order mandatory proof-unit turns by candidate score before the "
                             "turn cap truncates them; measured +16.0pp turn_all_hit on "
                             "locomo_cat4 and +17.4pp on cat2, no effect on LongMemEval")
    parser.add_argument("--obligation-aware-packing", action="store_true",
                        help="V5.10 greedy obligation/span packer; preserves frozen profiles when off")
    parser.add_argument("--precision-aware-packing", action="store_true",
                        help="use adaptive evidence limits, operand floors and MMR optional fill")
    parser.add_argument("--candidate-pool-limit", type=int, default=0,
                        help="0 keeps the full id reservoir; positive values expose a bounded "
                             "candidate precision/recall operating point")
    parser.add_argument("--raw-fallback-reserve", type=int, default=0,
                        help="reserve at most this many top-scoring raw fallback turns; "
                             "0 leaves every fallback optional")
    parser.add_argument("--span-pack-window", type=int, default=96,
                        help="character context charged around each selected evidence span")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--embedding-db", type=Path,
                        help="read turn vectors from a separate immutable SQLite sidecar")
    parser.add_argument("--dense-sidecar-dir", type=Path,
                        help="versioned per-memory FAISS/NumPy turn-vector indexes")
    parser.add_argument("--dense-backend",
                        choices=("auto", "numpy_exact", "faiss_flat"), default="auto")
    parser.add_argument("--query-embedding-cache", type=Path,
                        help="persistent WAL cache; defaults to the run directory")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--question-id", action="append", default=[],
                        help="run one or more exact question ids; may be repeated")
    parser.add_argument("--answer-workers", type=int, default=32)
    parser.add_argument("--label", default="")
    parser.add_argument("--run-root", type=Path,
                        help="Exact output directory; enables deterministic resume paths")
    parser.add_argument("--resume", action="store_true",
                        help="Skip question ids already present in both checkpoint JSONL files")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Navigate/answer/append this many questions per durable batch")
    parser.add_argument("--native-seed-fusion", action="store_true")
    parser.add_argument("--queryir-soft-fallback", action="store_true")
    parser.add_argument("--queryir-soft-fallback-threshold", type=float, default=0.80)
    parser.add_argument("--source-time-normalization", action="store_true",
                        help="experimental: materialize source-anchored relative time in evidence")
    parser.add_argument("--precision-grounded-prompt", action="store_true",
                        help="use the opt-in V5.20 direct-evidence answer contract")
    parser.add_argument("--obligation-aware-relations", action="store_true")
    parser.add_argument("--graph-hop-decay", type=float, default=1.0)
    parser.add_argument("--expansion-beam", type=int, default=4)
    parser.add_argument(
        "--rare-lexical-relations", action="store_true",
        help=("admit lexical_rare-only coarse edges and their QueryIR bonus; "
              "off is the paired control on the same lexical graph"))
    parser.add_argument(
        "--query-gated-rare-lexical", action="store_true",
        help=("admit lexical_rare-only graph bridges only for multi-fact, temporal, "
              "state-history, or collection QueryIR plans"))
    parser.add_argument("--full", action="store_true",
                        help="score LongMemEval 500 + LoCoMo Cat 1-4 (2,040) instead of the "
                             "frozen 200-question development set")
    parser.add_argument("--shard", type=int, default=0,
                        help="this process handles memories where hash(memory) %% shards == shard")
    parser.add_argument("--shards", type=int, default=1,
                        help="split by MEMORY, not by question: navigation builds one graph view "
                             "per memory and sharding by question would rebuild it in every process")
    parser.add_argument("--navigate-workers", type=int, default=1,
                        help="navigation is CPU-bound and holds the store lock; >1 helps only "
                             "when the dense channel dominates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or args.profile
    root = (args.run_root if args.run_root else
            args.output_root / (f"v5_6_answer_{label}_{stamp}"
                                + (f"_shard{args.shard}" if args.shards > 1 else "")))
    root.mkdir(parents=True, exist_ok=args.resume)

    # The authority graph is opened read-only; answers are cached in a separate
    # sidecar so a scoring run can never mutate the frozen build.
    store = SQLiteGraphStore(args.source_db, read_only=True)
    cache_db = root / "answer_cache.sqlite"
    cache_store = SQLiteGraphStore(cache_db)

    config = load_config(args.config)
    gold = load_gold_turns(args.gold)
    if args.full:
        # FullQuestion proxies attribute access to its DevQuestion, so everything
        # downstream keeps working; the extra flags ride along for the metrics.
        full_rows = load_full_questions(args.lme, args.locomo, gold)
        questions = [row.question for row in full_rows]
        flags = {row.question.question_id: row for row in full_rows}
    else:
        questions = load_dev_questions(args.lme, args.locomo, gold)
        flags = {}
    if args.question_id:
        requested = set(args.question_id)
        questions = [row for row in questions if row.question_id in requested]
        missing = requested - {row.question_id for row in questions}
        if missing:
            raise ValueError(f"unknown question ids: {sorted(missing)}")
    if args.shards > 1:
        # Shard on memory_id so each process touches a disjoint set of graphs and
        # builds each view exactly once; sharding on question_id would make every
        # process load every memory.
        import hashlib as _hashlib
        def _bucket(memory_id: str) -> int:
            return int(_hashlib.sha256(memory_id.encode()).hexdigest()[:8], 16) % args.shards
        questions = [row for row in questions if _bucket(row.memory_id) == args.shard]
        print(f"shard {args.shard}/{args.shards}: {len(questions)} questions, "
              f"{len({row.memory_id for row in questions})} memories", flush=True)
    if args.max_questions:
        questions = questions[:args.max_questions]

    budget = replace(
        config.query_budget, max_evidence_turns=args.max_evidence_turns,
        **({"max_evidence_tokens": args.max_evidence_tokens}
           if args.max_evidence_tokens else {}),
        **({"max_answer_tokens": args.max_answer_tokens}
           if args.max_answer_tokens else {}))
    answer_config = AnswerConfig(
        span_window=(args.span_pack_window if args.obligation_aware_packing
                     and args.span_window < 0 else
                     (None if args.span_window < 0 else args.span_window)),
        closed_form_enabled=not args.no_closed_form,
        candidate_answer_injection=args.candidate_answer_injection,
        evidence_order=args.evidence_order,
        normalize_relative_time=args.source_time_normalization,
        precision_grounding=args.precision_grounded_prompt,
        max_output_tokens=(args.max_output_tokens or None),
        sampling_seed=args.sampling_seed)

    embedding_store = None
    embedding_options = {
        "record_usage": False,
        "query_cache_path": (args.query_embedding_cache
                             or root / "query_embedding_cache.sqlite"),
        "dense_sidecar_dir": args.dense_sidecar_dir,
        "dense_backend": args.dense_backend,
    }
    if args.embedding_db:
        embedding_store = SQLiteGraphStore(args.embedding_db, read_only=True)
        embedding = QwenEmbeddingIndex(embedding_store, config, **embedding_options)
    else:
        embedding = QwenEmbeddingIndex(
            store, config, **embedding_options) if args.embedding else None
    navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                               dense_search_many=(embedding.search_many if embedding else None),
                               harness_profile=HarnessProfile(args.profile),
                               rank_mandatory=args.rank_mandatory,
                               h10_owner_rescue=not args.no_h10_owner_rescue,
                               h10_traversal=not args.no_h10_traversal,
                               hierarchical_routing=(
                                   not args.no_hierarchical_routing),
                               manifest_collection_key=not args.no_manifest_collection_key,
                               obligation_aware_packing=args.obligation_aware_packing,
                               precision_aware_packing=args.precision_aware_packing,
                               candidate_pool_limit=args.candidate_pool_limit,
                               span_pack_window=args.span_pack_window,
                               raw_fallback_reserve=args.raw_fallback_reserve,
                               obligation_aware_relations=args.obligation_aware_relations,
                               native_seed_fusion=args.native_seed_fusion,
                               queryir_soft_fallback=args.queryir_soft_fallback,
                               queryir_soft_fallback_threshold=(
                                   args.queryir_soft_fallback_threshold),
                               graph_hop_decay=args.graph_hop_decay,
                               expansion_beam=args.expansion_beam,
                               rare_lexical_relations=(
                                   args.rare_lexical_relations),
                               query_gated_rare_lexical=(
                                   args.query_gated_rare_lexical))

    stage = AnswerStage(
        store, config, "v5.6-answer", answer_config=answer_config,
        require_exact_tokenizer=True, cache_store=cache_store,
        answer_model=args.answer_model, answer_base_url=args.answer_base_url,
        answer_api_key_env=args.answer_api_key_env,
        answer_request_profile=args.answer_request_profile,
        packing_model=args.packing_model)
    edge_source_cache: dict[str, dict[str, str]] = {}

    def traversed_signals(memory_id: str, result) -> dict[str, int]:
        sources = edge_source_cache.get(memory_id)
        if sources is None:
            sources = {edge.edge_id: edge.source for edge in store.edges(memory_id)}
            edge_source_cache[memory_id] = sources
        counts: dict[str, int] = {}
        for step in result.proof:
            source = sources.get(step.edge_id, "")
            marker = "relation_mask:"
            if marker not in source:
                continue
            mask = source.split(marker, 1)[1].split("|", 1)[0]
            for signal in filter(None, mask.split(",")):
                counts[signal] = counts.get(signal, 0) + 1
        return dict(sorted(counts.items()))

    def prepare(row):
        question, result, _metric = row
        question_date = str(question.raw.get("question_date") or "") or None
        # result.algebra is populated only by AST-executing profiles; passing it
        # is what lets the closed-form composer emit a count without an LLM.
        return stage.prepare(
            question.question_id, question.query, result, budget,
            question_date=question_date, algebra=result.algebra)

    answer_path = root / "answers.jsonl"
    retrieval_path = root / "retrieval.jsonl"
    prepared_path = root / "prepared_answers.jsonl"

    def load_rows(path: Path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    completed = set()
    if args.resume:
        answer_checkpoint = load_rows(answer_path)
        retrieval_checkpoint = load_rows(retrieval_path)
        prepared_checkpoint = load_rows(prepared_path)
        answered = {str(row["question_id"]) for row in answer_checkpoint}
        retrieved = {str(row["dev_question_id"]) for row in retrieval_checkpoint}
        prepared_ids = {str(row["question_id"]) for row in prepared_checkpoint}
        completed = answered & retrieved & prepared_ids
        # Three JSONLs cannot be appended atomically.  If a process died after
        # writing only one or two of them, retain exactly one complete row per
        # ID and rerun the torn suffix (the answer cache prevents a second API
        # charge).  Without this compaction, resume silently duplicated answer
        # rows and inflated Token percentiles.
        order = [row.question_id for row in questions if row.question_id in completed]
        for path, key, checkpoint in (
            (answer_path, "question_id", answer_checkpoint),
            (retrieval_path, "dev_question_id", retrieval_checkpoint),
            (prepared_path, "question_id", prepared_checkpoint),
        ):
            by_id = {str(row[key]): row for row in checkpoint
                     if str(row[key]) in completed}
            path.write_text("".join(
                json.dumps(by_id[question_id], ensure_ascii=True) + "\n"
                for question_id in order), encoding="utf-8")
    remaining = [row for row in questions if row.question_id not in completed]
    batch_size = max(1, args.checkpoint_every)
    print(
        f"running {len(remaining)} remaining / {len(questions)} questions with "
        f"{args.profile}; checkpoint batch={batch_size}", flush=True)
    # Navigation/packing is CPU-bound, while answer generation is GPU-bound.
    # Stream prepared requests into a bounded answer pool instead of preparing
    # all 2,040 prompts first: the latter left the GPU idle for the entire
    # navigation phase.  The bound prevents frozen prompts from accumulating
    # without limit when the model is slower than the producer.
    pending_checkpoints = []
    checkpointed = 0

    def checkpoint(rows) -> None:
        nonlocal checkpointed
        rows.sort(key=lambda item: item[0])
        with answer_path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(item[1], ensure_ascii=True) + "\n" for item in rows)
        with retrieval_path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(item[2], ensure_ascii=True) + "\n" for item in rows)
        with prepared_path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(item[3], ensure_ascii=True) + "\n" for item in rows)
        checkpointed += len(rows)
        rows.clear()
        print(
            f"  checkpointed {len(completed) + checkpointed}/{len(questions)}",
            flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.answer_workers)) as pool:
        futures = {}

        def finish_future(future) -> None:
            index, question, result, metric, prepared_answer = futures.pop(future)
            answer = future.result()
            raw = question.raw
            answer_row = {
                "question_id": question.question_id,
                "question": question.query,
                "question_type": str(raw.get("question_type")
                                     or f"locomo_category_{raw.get('locomo_category')}"),
                "question_date": raw.get("question_date") or "",
                "answer": raw.get("answer", ""),
                "gold_answer": raw.get("answer", ""),
                "prediction": answer.prediction,
                "benchmark": question.benchmark,
                "stratum": question.stratum,
                "answer_model": answer.answer_model,
                "prompt_payload_hash": answer.prompt_payload_hash,
                "traversed_relation_signals": traversed_signals(
                    question.memory_id, result),
            }
            row_flags = flags.get(question.question_id)
            retrieval_row = {
                "dev_question_id": question.question_id, "stratum": question.stratum,
                # turn_all_hit is vacuously true where no gold turns are annotated,
                # so the full-set report must be able to exclude those rows.
                "has_turn_gold": bool(row_flags.has_turn_gold) if row_flags else True,
                "is_abstention": bool(row_flags.is_abstention) if row_flags else False,
                "benchmark": question.benchmark, "configuration": label,
                "prompt_tokens": answer.prompt_tokens,
                "api_prompt_tokens": answer.api_prompt_tokens,
                "evidence_tokens": answer.evidence_tokens,
                "completion_tokens": answer.completion_tokens,
                "answer_total_tokens": answer.api_total_tokens,
                "answer_latency_ms": answer.latency_ms,
                "answer_cached": answer.cached,
                "answer_model": answer.answer_model,
                "prompt_payload_hash": answer.prompt_payload_hash,
                "traversed_relation_signals": traversed_signals(
                    question.memory_id, result),
                "answer_finish_reason": answer.finish_reason,
                "packed_turns": len(answer.evidence_turn_ids),
                "closed_form": answer.closed_form,
                "budget_relaxed": answer.budget_relaxed,
                "answer_warnings": list(answer.warnings),
                "evidence_layout": answer.trace.get("evidence_layout", ""),
                "evidence_chain_count": answer.trace.get(
                    "evidence_chain_count", 0),
                "evidence_chain_turns": answer.trace.get(
                    "evidence_chain_turns", 0),
                "evidence_graph_group_count": answer.trace.get(
                    "evidence_graph_group_count", 0),
                "evidence_graph_turns": answer.trace.get(
                    "evidence_graph_turns", 0),
                "evidence_auxiliary_turns": answer.trace.get(
                    "evidence_auxiliary_turns", 0),
                **{key: value for key, value in metric.items() if key != "question_id"},
            }
            pending_checkpoints.append((
                index, answer_row, retrieval_row, prepared_answer.to_record()))
            if len(pending_checkpoints) >= batch_size:
                checkpoint(pending_checkpoints)

        max_in_flight = max(1, args.answer_workers) * 2
        for index, question in enumerate(remaining):
            result = navigator.navigate(question.memory_id, question.query, budget)
            metric = navigation_metrics(question, result, store)
            prepared_answer = prepare((question, result, metric))
            future = pool.submit(stage.complete, prepared_answer)
            futures[future] = (
                index, question, result, metric, prepared_answer)
            if len(futures) >= max_in_flight:
                done, _pending = wait(tuple(futures),
                                      return_when=FIRST_COMPLETED)
                for completed_future in done:
                    finish_future(completed_future)
        for future in as_completed(tuple(futures)):
            finish_future(future)
    if pending_checkpoints:
        checkpoint(pending_checkpoints)

    answer_rows = load_rows(answer_path)
    retrieval_rows = load_rows(retrieval_path)
    prepared_rows = load_rows(prepared_path)
    expected_ids = {str(row.question_id) for row in questions}
    answer_ids = {str(row["question_id"]) for row in answer_rows}
    retrieval_ids = {str(row["dev_question_id"]) for row in retrieval_rows}
    prepared_ids = {str(row["question_id"]) for row in prepared_rows}
    if answer_ids != expected_ids or retrieval_ids != expected_ids \
            or prepared_ids != expected_ids:
        raise RuntimeError(
            "answer artifact incomplete: "
            f"expected={len(expected_ids)} answers={len(answer_ids)} "
            f"retrieval={len(retrieval_ids)} prepared={len(prepared_ids)}")
    if (len(answer_rows) != len(expected_ids)
            or len(retrieval_rows) != len(expected_ids)
            or len(prepared_rows) != len(expected_ids)):
        raise RuntimeError("answer artifacts contain duplicate question IDs")
    # Completion-order checkpoints are ideal for recovery; canonical question
    # order is ideal for byte-stable artifacts and dual-model replay.
    question_order = [str(row.question_id) for row in questions]
    ordered_payloads = []
    for path, key, rows in (
        (answer_path, "question_id", answer_rows),
        (retrieval_path, "dev_question_id", retrieval_rows),
        (prepared_path, "question_id", prepared_rows),
    ):
        by_id = {str(row[key]): row for row in rows}
        ordered = [by_id[question_id] for question_id in question_order]
        path.write_text("".join(
            json.dumps(row, ensure_ascii=True) + "\n" for row in ordered),
            encoding="utf-8")
        ordered_payloads.append(ordered)
    answer_rows, retrieval_rows, prepared_rows = ordered_payloads
    prepared_hashes = {str(row["question_id"]): str(
        row.get("prompt_payload_hash") or "") for row in prepared_rows}
    prompt_hash_mismatches = [
        str(row["question_id"]) for row in answer_rows
        if str(row.get("prompt_payload_hash") or "")
        != prepared_hashes.get(str(row["question_id"]))
    ]
    if prompt_hash_mismatches:
        raise RuntimeError(
            "answer/prepared prompt hash mismatch for "
            f"{len(prompt_hash_mismatches)} questions")
    # Emit benchmark-specific judge inputs for both development and full runs.
    # Previously only the shard merger did this, which made a single-process
    # precision ablation require an ad-hoc filtering step before judging.
    for benchmark, filename in (
        ("longmemeval", "answers_longmemeval.jsonl"),
        ("locomo", "answers_locomo.jsonl"),
    ):
        (root / filename).write_text("".join(
            json.dumps(row, ensure_ascii=True) + "\n"
            for row in answer_rows if row.get("benchmark") == benchmark),
            encoding="utf-8")

    def token_stats(values):
        ordered = sorted(int(value) for value in values)
        def nearest(p):
            return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0
        return {
            "count": len(ordered),
            "mean": sum(ordered) / max(1, len(ordered)),
            "p50": nearest(0.50), "p95": nearest(0.95),
            "p99": nearest(0.99), "max": max(ordered, default=0),
            "unit": "tokens_per_question",
            "percentile_method": "nearest_rank",
        }

    tokens = sorted(row["prompt_tokens"] for row in retrieval_rows)
    manifest = {
        "profile": args.profile, "label": label, "questions": len(answer_rows),
        "source_db": str(args.source_db), "config_hash": config_hash(config),
        "answer_prompt_hash": prompt_contract(
            answer_config.normalize_relative_time,
            answer_config.precision_grounding,
            answer_config.evidence_order == "topological")[2],
        "span_window": answer_config.span_window,
        "evidence_order": answer_config.evidence_order,
        "source_time_normalization": answer_config.normalize_relative_time,
        "precision_grounded_prompt": answer_config.precision_grounding,
        "obligation_aware_packing": args.obligation_aware_packing,
        "precision_aware_packing": args.precision_aware_packing,
        "candidate_pool_limit": args.candidate_pool_limit,
        "raw_fallback_reserve": args.raw_fallback_reserve,
        "obligation_aware_relations": args.obligation_aware_relations,
        "graph_traversal": not args.no_h10_traversal,
        "hierarchical_routing": not args.no_hierarchical_routing,
        "native_seed_fusion": args.native_seed_fusion,
        "dense_search": embedding is not None,
        "embedding_db": str(args.embedding_db) if args.embedding_db else None,
        "queryir_soft_fallback": args.queryir_soft_fallback,
        "queryir_soft_fallback_threshold": args.queryir_soft_fallback_threshold,
        "graph_hop_decay": args.graph_hop_decay,
        "expansion_beam": args.expansion_beam,
        "rare_lexical_relations": args.rare_lexical_relations,
        "query_gated_rare_lexical": args.query_gated_rare_lexical,
        "span_pack_window": args.span_pack_window,
        "closed_form_enabled": answer_config.closed_form_enabled,
        "candidate_answer_injection": answer_config.candidate_answer_injection,
        "max_output_tokens": answer_config.max_output_tokens,
        "sampling_seed": answer_config.sampling_seed,
        "answer_model": stage.answer_model,
        "answer_base_url": args.answer_base_url or config.models.llm_base_url,
        "answer_request_profile": args.answer_request_profile,
        "packing_model": args.packing_model or config.models.llm_model,
        "output_truncated": sum(
            row.get("answer_finish_reason") == "length" for row in retrieval_rows),
        "budget": dataclass_dict(budget),
        "token_counter": stage.counter.describe(),
        "prompt_tokens": {
            "mean": sum(tokens) / max(1, len(tokens)),
            "p50": tokens[len(tokens) // 2] if tokens else 0,
            "p95": tokens[max(0, int(0.95 * len(tokens)) - 1)] if tokens else 0,
            "max": max(tokens, default=0),
            "over_soft_budget": sum(1 for row in tokens if row > budget.max_answer_tokens),
        },
        "answer_api_tokens": {
            "prompt": token_stats(row.get("api_prompt_tokens", 0)
                                  for row in retrieval_rows),
            "completion": token_stats(row.get("completion_tokens", 0)
                                      for row in retrieval_rows),
            "total": token_stats(row.get("answer_total_tokens", 0)
                                 for row in retrieval_rows),
        },
        "answer_api_tokens_by_benchmark": {
            benchmark: {
                "prompt": token_stats(row.get("api_prompt_tokens", 0)
                                      for row in retrieval_rows
                                      if row.get("benchmark") == benchmark),
                "completion": token_stats(row.get("completion_tokens", 0)
                                          for row in retrieval_rows
                                          if row.get("benchmark") == benchmark),
                "total": token_stats(row.get("answer_total_tokens", 0)
                                     for row in retrieval_rows
                                     if row.get("benchmark") == benchmark),
            } for benchmark in ("longmemeval", "locomo")
        },
        "prepared_answers": str(prepared_path),
        "prepared_prompt_hashes": len({
            str(row.get("prompt_payload_hash")) for row in retrieval_rows}),
        "prompt_identity_audit": {
            "question_ids_match": True,
            "prompt_hash_mismatches": len(prompt_hash_mismatches),
        },
        "answer_api_usage_sums": {
            "prompt": sum(int(row.get("api_prompt_tokens") or 0)
                          for row in retrieval_rows),
            "completion": sum(int(row.get("completion_tokens") or 0)
                              for row in retrieval_rows),
            "total": sum(int(row.get("answer_total_tokens") or 0)
                         for row in retrieval_rows),
        },
        "answer_api_usage_additivity_ok": all(
            int(row.get("api_prompt_tokens") or 0)
            + int(row.get("completion_tokens") or 0)
            == int(row.get("answer_total_tokens") or 0)
            for row in retrieval_rows),
        "closed_form_rate": sum(row["closed_form"] for row in retrieval_rows) / max(1, len(retrieval_rows)),
        "generated_at": stamp,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {root}")
    store.close()
    if embedding_store is not None:
        embedding_store.close()
    cache_store.close()


if __name__ == "__main__":
    main()
