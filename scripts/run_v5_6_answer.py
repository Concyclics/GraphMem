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
import sys
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="explicit seed sent to the answer service")
    parser.add_argument("--span-window", type=int, default=-1,
                        help="-1 renders whole turns; >=0 renders cited spans widened by N chars")
    parser.add_argument("--evidence-order",
                        choices=("chronological", "relevance", "adaptive"),
                        default="chronological",
                        help="render by source time, retrieval rank, or query-directed order")
    parser.add_argument("--no-closed-form", action="store_true")
    parser.add_argument("--no-h10-owner-rescue", action="store_true")
    parser.add_argument("--no-h10-traversal", action="store_true")
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
    parser.add_argument("--span-pack-window", type=int, default=96,
                        help="character context charged around each selected evidence span")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--embedding-db", type=Path,
                        help="read turn vectors from a separate immutable SQLite sidecar")
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
    parser.add_argument("--obligation-aware-relations", action="store_true")
    parser.add_argument("--graph-hop-decay", type=float, default=1.0)
    parser.add_argument("--expansion-beam", type=int, default=4)
    parser.add_argument(
        "--rare-lexical-relations", action="store_true",
        help=("admit lexical_rare-only coarse edges and their QueryIR bonus; "
              "off is the paired control on the same lexical graph"))
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
        evidence_order=args.evidence_order,
        normalize_relative_time=args.source_time_normalization,
        max_output_tokens=(args.max_output_tokens or None),
        sampling_seed=args.sampling_seed)

    embedding_store = None
    if args.embedding_db:
        embedding_store = SQLiteGraphStore(args.embedding_db, read_only=True)
        embedding = QwenEmbeddingIndex(embedding_store, config, record_usage=False)
    else:
        embedding = QwenEmbeddingIndex(
            store, config, record_usage=False) if args.embedding else None
    navigator = GraphNavigator(store, dense_search=embedding.search if embedding else None,
                               harness_profile=HarnessProfile(args.profile),
                               rank_mandatory=args.rank_mandatory,
                               h10_owner_rescue=not args.no_h10_owner_rescue,
                               h10_traversal=not args.no_h10_traversal,
                               manifest_collection_key=not args.no_manifest_collection_key,
                               obligation_aware_packing=args.obligation_aware_packing,
                               precision_aware_packing=args.precision_aware_packing,
                               candidate_pool_limit=args.candidate_pool_limit,
                               span_pack_window=args.span_pack_window,
                               obligation_aware_relations=args.obligation_aware_relations,
                               native_seed_fusion=args.native_seed_fusion,
                               queryir_soft_fallback=args.queryir_soft_fallback,
                               queryir_soft_fallback_threshold=(
                                   args.queryir_soft_fallback_threshold),
                               graph_hop_decay=args.graph_hop_decay,
                               expansion_beam=args.expansion_beam,
                               rare_lexical_relations=(
                                   args.rare_lexical_relations))

    stage = AnswerStage(store, config, "v5.6-answer", answer_config=answer_config,
                        require_exact_tokenizer=True, cache_store=cache_store)

    def run(row):
        question, result, _metric = row
        question_date = str(question.raw.get("question_date") or "") or None
        # result.algebra is populated only by AST-executing profiles; passing it
        # is what lets the closed-form composer emit a count without an LLM.
        return stage.answer(question.question_id, question.query, result, budget,
                            question_date=question_date, algebra=result.algebra)

    answer_path = root / "answers.jsonl"
    retrieval_path = root / "retrieval.jsonl"

    def load_rows(path: Path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    completed = set()
    if args.resume:
        answered = {str(row["question_id"]) for row in load_rows(answer_path)}
        retrieved = {str(row["dev_question_id"]) for row in load_rows(retrieval_path)}
        completed = answered & retrieved
    remaining = [row for row in questions if row.question_id not in completed]
    batch_size = max(1, args.checkpoint_every)
    print(
        f"running {len(remaining)} remaining / {len(questions)} questions with "
        f"{args.profile}; checkpoint batch={batch_size}", flush=True)
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        navigations = []
        for question in batch:
            result = navigator.navigate(question.memory_id, question.query, budget)
            navigations.append(
                (question, result, navigation_metrics(question, result, store)))
        with ThreadPoolExecutor(max_workers=max(1, args.answer_workers)) as pool:
            answers = list(pool.map(run, navigations))

        answer_rows, retrieval_rows = [], []
        for (question, result, metric), answer in zip(navigations, answers):
            raw = question.raw
            answer_rows.append({
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
            })
            row_flags = flags.get(question.question_id)
            retrieval_rows.append({
                "dev_question_id": question.question_id, "stratum": question.stratum,
                # turn_all_hit is vacuously true where no gold turns are annotated,
                # so the full-set report must be able to exclude those rows.
                "has_turn_gold": bool(row_flags.has_turn_gold) if row_flags else True,
                "is_abstention": bool(row_flags.is_abstention) if row_flags else False,
                "benchmark": question.benchmark, "configuration": label,
                "prompt_tokens": answer.prompt_tokens,
                "evidence_tokens": answer.evidence_tokens,
                "completion_tokens": answer.completion_tokens,
                "answer_finish_reason": answer.finish_reason,
                "packed_turns": len(answer.evidence_turn_ids),
                "closed_form": answer.closed_form,
                "budget_relaxed": answer.budget_relaxed,
                "answer_warnings": list(answer.warnings),
                **{key: value for key, value in metric.items() if key != "question_id"},
            })
        with answer_path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(row, ensure_ascii=True) + "\n" for row in answer_rows)
        with retrieval_path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(row, ensure_ascii=True) + "\n" for row in retrieval_rows)
        done = len(completed) + min(start + len(batch), len(remaining))
        print(f"  checkpointed {done}/{len(questions)}", flush=True)

    answer_rows = load_rows(answer_path)
    retrieval_rows = load_rows(retrieval_path)
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

    tokens = sorted(row["prompt_tokens"] for row in retrieval_rows)
    manifest = {
        "profile": args.profile, "label": label, "questions": len(answer_rows),
        "source_db": str(args.source_db), "config_hash": config_hash(config),
        "answer_prompt_hash": prompt_contract(
            answer_config.normalize_relative_time)[2],
        "span_window": answer_config.span_window,
        "evidence_order": answer_config.evidence_order,
        "source_time_normalization": answer_config.normalize_relative_time,
        "obligation_aware_packing": args.obligation_aware_packing,
        "precision_aware_packing": args.precision_aware_packing,
        "candidate_pool_limit": args.candidate_pool_limit,
        "obligation_aware_relations": args.obligation_aware_relations,
        "native_seed_fusion": args.native_seed_fusion,
        "dense_search": embedding is not None,
        "embedding_db": str(args.embedding_db) if args.embedding_db else None,
        "queryir_soft_fallback": args.queryir_soft_fallback,
        "queryir_soft_fallback_threshold": args.queryir_soft_fallback_threshold,
        "graph_hop_decay": args.graph_hop_decay,
        "expansion_beam": args.expansion_beam,
        "rare_lexical_relations": args.rare_lexical_relations,
        "span_pack_window": args.span_pack_window,
        "closed_form_enabled": answer_config.closed_form_enabled,
        "max_output_tokens": answer_config.max_output_tokens,
        "sampling_seed": answer_config.sampling_seed,
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
