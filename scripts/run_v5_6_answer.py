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
from graphmem.answer.aggregation import AGGREGATION_LEDGER_SCHEMA_VERSION  # noqa: E402
from graphmem.config import (  # noqa: E402
    config_hash, load_config, load_runtime_config, runtime_config_hash,
)
from graphmem.domain import dataclass_dict  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.eval.fullset import load_full_questions  # noqa: E402
from graphmem.eval.metrics import navigation_metrics  # noqa: E402
from graphmem.retrieval import GraphNavigator, HarnessProfile  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def effective_runtime_navigator_options(
        runtime_config, *, disable_hierarchical_routing: bool = False,
        disable_graph_traversal: bool = False,
        research_overrides: dict | None = None) -> dict:
    """Translate a deployed runtime profile into an audited experiment arm.

    Runtime profiles are the authority for production defaults, but a paired
    structural ablation must be able to disable hierarchy and relation
    traversal independently.  Keeping the override here prevents the CLI from
    claiming an arm was disabled while ``GraphNavigator`` still receives the
    value frozen in the runtime JSON.
    """

    options = runtime_config.retrieval.navigator_options()
    # ``h10_traversal`` is a research switch and therefore absent from the
    # deployed schema; materialize its production default so every arm has an
    # explicit, hashable two-factor contract.
    options.setdefault("h10_traversal", True)
    options.update(dict(research_overrides or {}))
    if disable_hierarchical_routing:
        options["hierarchical_routing"] = False
    if disable_graph_traversal:
        options["h10_traversal"] = False
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path,
        help=("optional deployed query-plane profile; when supplied it is the "
              "authority for retrieval options and query budgets"))
    parser.add_argument("--profile", default="h9")
    parser.add_argument("--max-evidence-turns", type=int, default=32)
    parser.add_argument("--max-evidence-tokens", type=int,
                        help="override retrieval evidence-token budget")
    parser.add_argument("--max-answer-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=0,
                        help="0 omits the API output cap; positive values enable an ablation cap")
    parser.add_argument(
        "--answer-policy", choices=("legacy", "v5_54", "v5_63"),
        default="legacy",
        help=("core answer/readout contract; v5_54 enables the validated "
              "typed, aggregation, inference and graph-block routes without "
              "offline prompt materializers"))
    parser.add_argument(
        "--answer-plan", action="store_true",
        help=("append the opt-in V5.56 deterministic temporal/state binding "
              "index after the validated readout policy"))
    parser.add_argument("--answer-plan-max-candidates", type=int, default=5)
    parser.add_argument("--answer-plan-excerpt-chars", type=int, default=440)
    parser.add_argument(
        "--answer-plan-kind", action="append",
        choices=("date_difference", "relative_time", "age_projection",
                 "latest_state", "temporal_lookup", "temporal_order"),
        help=("eligible V5.56 AnswerPlan route; repeat to select several; "
              "defaults to the conservative date-difference and explicit "
              "temporal-order routes"))
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
                        choices=("chronological", "relevance", "adaptive",
                                 "topological_plain", "topological",
                                 "topological_recency"),
                        default="chronological",
                        help=("render by source time, retrieval rank, query-directed order, "
                              "or graph topology with/without graph labels"))
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
    parser.add_argument(
        "--embedding-request-model",
        help=("served embedding-model alias; storage/cache identity still comes "
              "from the runtime/config embedding model"))
    parser.add_argument("--embedding-db", type=Path,
                        help="read turn vectors from a separate immutable SQLite sidecar")
    parser.add_argument(
        "--relation-embedding-db", type=Path,
        help=("read existing graph-node vectors for exact CanonicalFact search; "
              "no new build or embedding calls are performed"))
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
    parser.add_argument(
        "--prepare-only", action="store_true",
        help=("run full navigation and freeze PreparedAnswer prompts without "
              "calling the answer model; used for full-corpus Token/retention gates"))
    parser.add_argument(
        "--save-candidate-scores", action="store_true",
        help=("include the complete candidate ranking in retrieval JSONL for "
              "offline score/packing audits"))
    parser.add_argument("--native-seed-fusion", action="store_true")
    parser.add_argument(
        "--relational-view-scoring", action="store_true",
        help=("keep every QueryIR view candidate but weight ranking by view "
              "completeness; owner-only matches no longer act as full relation hits"))
    parser.add_argument(
        "--query-relation-view", action="store_true",
        help=("add a deterministic owner/head-stripped relation view without "
              "an embedding or model call"))
    parser.add_argument(
        "--relational-view-named-speakers-only", action="store_true",
        help=("apply relational view scoring only to named multi-party "
              "transcripts; generic user/assistant memories remain frozen"))
    parser.add_argument(
        "--relational-consensus-bonus", type=float, default=0.0,
        help=("score rank-sensitive agreement across relation-bearing QueryIR "
              "views; zero preserves max-per-channel fusion"))
    parser.add_argument(
        "--dialogue-response-closure", action="store_true",
        help=("pair a query-relevant dialogue prompt with its immediate "
              "cross-speaker response before the fixed-size evidence pack"))
    parser.add_argument(
        "--proof-priority-bonus", type=float,
        help=("finite score bonus for proof-unit candidates; omitted keeps "
              "the historical hard mandatory-first ordering"))
    parser.add_argument("--proof-priority-flood-threshold", type=int, default=0)
    parser.add_argument("--dialogue-response-flood-threshold", type=int, default=0)
    parser.add_argument("--speaker-owner-bonus", type=float, default=0.0)
    parser.add_argument("--query-witness-bonus", type=float, default=0.0)
    parser.add_argument("--query-witness-seed-count", type=int, default=16)
    parser.add_argument("--query-witness-rare-df", type=int, default=4)
    parser.add_argument("--query-witness-min-shared-terms", type=int, default=2)
    parser.add_argument("--queryir-soft-fallback", action="store_true")
    parser.add_argument("--queryir-soft-fallback-threshold", type=float, default=0.80)
    parser.add_argument(
        "--exact-lookup-fast-path", action="store_true",
        help=("guard scalar LOOKUP queries with direct source/fact confidence; "
              "low-confidence requests retain hierarchical graph traversal"))
    parser.add_argument(
        "--exact-lookup-turn-limit", type=int, default=16,
        help="maximum evidence turns after a high-confidence direct lookup")
    parser.add_argument(
        "--exact-lookup-priority", action="store_true",
        help=("promote exact source/fact witnesses inside normal graph packing "
              "without truncating traversal or the evidence budget"))
    parser.add_argument("--exact-lookup-priority-min-score", type=float,
                        default=1.50)
    parser.add_argument("--exact-lookup-priority-bonus", type=float, default=1.0)
    parser.add_argument(
        "--exact-lookup-priority-named-speakers-only", action="store_true",
        help=("enable exact priority only for named multi-party transcripts; "
              "generic user/assistant memories retain the normal graph path"))
    parser.add_argument("--source-time-normalization", action="store_true",
                        help="experimental: materialize source-anchored relative time in evidence")
    parser.add_argument("--precision-grounded-prompt", action="store_true",
                        help="use the opt-in V5.20 direct-evidence answer contract")
    parser.add_argument("--aggregation-ledger", action="store_true",
                        help="append a structured operand ledger for aggregation queries")
    parser.add_argument("--aggregation-ledger-limit", type=int, default=24,
                        help="maximum packed source turns indexed in an aggregation ledger")
    parser.add_argument(
        "--aggregation-execution-card", action="store_true",
        help=("replace rendered candidate snippets with a compact operation "
              "card; candidate IDs remain trace-only and are not reserved"))
    parser.add_argument(
        "--aggregation-operand-worksheet", action="store_true",
        help=("add the opt-in bounded V5.60 operand worksheet to compact "
              "aggregation cards; disabled in the validated default"))
    parser.add_argument(
        "--aggregation-source-reserve", action="store_true",
        help=("during aggregation prompt re-packing, preserve generic user "
              "source turns before optional assistant prose; named multi-party "
              "memories remain symmetric"))
    parser.add_argument(
        "--aggregation-source-reserve-operation", action="append",
        choices=("sum", "count_distinct", "date_difference", "difference",
                 "mean", "minimum", "maximum", "unit_rate"),
        help=("aggregation operation eligible for direct-source reservation; "
              "repeat to select several; defaults to sum and count_distinct"))
    parser.add_argument(
        "--preference-synthesis-prompt", action="store_true",
        help=("route advice/recommendation wording to a grounded synthesis "
              "contract without using benchmark labels"))
    parser.add_argument(
        "--exact-grounding-footer", action="store_true",
        help=("repeat exact entity/relation and missing-fact checks after the "
              "evidence block so long contexts cannot dilute them"))
    parser.add_argument(
        "--question-date-mode",
        choices=("always", "query_relative", "never"), default="always",
        help=("show the global question date always, only for deictic query "
              "phrases, or never; source-time annotations remain unchanged"))
    parser.add_argument(
        "--question-recency-footer", action="store_true",
        help=("repeat the original question and source-time rule after the "
              "evidence block without increasing the evidence budget"))
    parser.add_argument(
        "--compact-topological-prompt", action="store_true",
        help=("replace the verbose graph-label glossary with its compact, "
              "semantically equivalent contract to reclaim answer tokens"))
    parser.add_argument(
        "--query-focus-index", action="store_true",
        help=("repeat bounded query-centered excerpts from the full text of "
              "already-packed anonymous turns; does not add evidence turns"))
    parser.add_argument("--query-focus-limit", type=int, default=4)
    parser.add_argument(
        "--query-focus-excerpt-chars", type=int,
        help=("override the answer policy's excerpt length; v5_54 defaults to "
              "360 and v5_63 uses its validated 480-character setting"))
    parser.add_argument(
        "--focused-prompt-scope", choices=("all", "default"), default="all",
        help=("apply contextual date, recency footer and compact topology to "
              "all prompts or only prompts without aggregation/preference contracts"))
    parser.add_argument("--obligation-aware-relations", action="store_true")
    parser.add_argument("--graph-hop-decay", type=float, default=1.0)
    parser.add_argument("--expansion-beam", type=int, default=4)
    parser.add_argument(
        "--hierarchy-descent-beam", type=int, default=1,
        help=("number of structurally reranked children retained at each level "
              "after a relation edge enters a new coarse region"))
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
    parser.add_argument(
        "--lme-type", action="append", default=[],
        help=("with --full, retain only this LongMemEval question_type; may be "
              "repeated, for example multi-session and temporal-reasoning"))
    parser.add_argument(
        "--locomo-category", action="append", type=int, default=[],
        help=("with --full, retain only this LoCoMo category; may be repeated; "
              "the default remains categories 1-4"))
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
    runtime_config = (
        load_runtime_config(args.runtime_config)
        if args.runtime_config is not None else None)
    runtime_dense_options = None
    if runtime_config is not None:
        args.profile = runtime_config.retrieval.harness_profile
        runtime_dense_options = runtime_config.retrieval.embedding_options()
        if runtime_dense_options is not None:
            args.embedding = True
            if runtime_dense_options["dense_sidecar_dir"]:
                args.dense_sidecar_dir = Path(str(
                    runtime_dense_options["dense_sidecar_dir"]))
            if runtime_dense_options["query_cache_path"]:
                args.query_embedding_cache = Path(str(
                    runtime_dense_options["query_cache_path"]))
            args.dense_backend = str(runtime_dense_options["dense_backend"])
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
        lme_types = tuple(dict.fromkeys(args.lme_type)) or None
        locomo_categories = (tuple(dict.fromkeys(args.locomo_category))
                             or (1, 2, 3, 4))
        full_rows = load_full_questions(
            args.lme, args.locomo, gold,
            lme_types=lme_types, locomo_categories=locomo_categories,
            expect_lme=(None if lme_types else 500),
            expect_locomo=(None if args.locomo_category else 1540))
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

    budget = (runtime_config.query_budget if runtime_config is not None else
              replace(
                  config.query_budget,
                  max_evidence_turns=args.max_evidence_turns,
                  **({"max_evidence_tokens": args.max_evidence_tokens}
                     if args.max_evidence_tokens else {}),
                  **({"max_answer_tokens": args.max_answer_tokens}
                     if args.max_answer_tokens else {})))
    if args.answer_policy == "v5_63":
        v563_overrides = ({
            "max_output_tokens": args.max_output_tokens,
        } if args.max_output_tokens else {})
        if args.query_focus_excerpt_chars is not None:
            v563_overrides["query_focus_excerpt_chars"] = (
                args.query_focus_excerpt_chars)
        answer_config = AnswerConfig.v5_63(**v563_overrides,
            sampling_seed=args.sampling_seed,
            query_focus_index_limit=args.query_focus_limit,
            answer_plan_enabled=args.answer_plan,
            answer_plan_max_candidates=args.answer_plan_max_candidates,
            answer_plan_excerpt_chars=args.answer_plan_excerpt_chars,
            answer_plan_kinds=tuple(
                args.answer_plan_kind
                or ("date_difference", "temporal_order")))
    elif args.answer_policy == "v5_54":
        answer_config = AnswerConfig.v5_54(**({
            "max_output_tokens": args.max_output_tokens,
        } if args.max_output_tokens else {}), sampling_seed=args.sampling_seed,
            query_focus_index_enabled=args.query_focus_index,
            query_focus_index_limit=args.query_focus_limit,
            query_focus_excerpt_chars=(
                args.query_focus_excerpt_chars
                if args.query_focus_excerpt_chars is not None else 360),
            aggregation_operand_worksheet_enabled=(
                args.aggregation_operand_worksheet),
            answer_plan_enabled=args.answer_plan,
            answer_plan_max_candidates=args.answer_plan_max_candidates,
            answer_plan_excerpt_chars=args.answer_plan_excerpt_chars,
            answer_plan_kinds=tuple(
                args.answer_plan_kind
                or ("date_difference", "temporal_order")))
    else:
        answer_config = AnswerConfig(
            span_window=(args.span_pack_window if args.obligation_aware_packing
                         and args.span_window < 0 else
                         (None if args.span_window < 0 else args.span_window)),
            closed_form_enabled=not args.no_closed_form,
            candidate_answer_injection=args.candidate_answer_injection,
            evidence_order=args.evidence_order,
            normalize_relative_time=args.source_time_normalization,
            precision_grounding=args.precision_grounded_prompt,
            aggregation_ledger_enabled=args.aggregation_ledger,
            aggregation_ledger_limit=args.aggregation_ledger_limit,
            aggregation_execution_card=args.aggregation_execution_card,
            aggregation_operand_worksheet_enabled=(
                args.aggregation_operand_worksheet),
            aggregation_source_reserve_enabled=args.aggregation_source_reserve,
            aggregation_source_reserve_operations=tuple(
                args.aggregation_source_reserve_operation
                or ("sum", "count_distinct", "unit_rate")),
            preference_synthesis_enabled=args.preference_synthesis_prompt,
            exact_grounding_footer=args.exact_grounding_footer,
            question_date_mode=args.question_date_mode,
            question_recency_footer=args.question_recency_footer,
            compact_topological_contract=args.compact_topological_prompt,
            query_focus_index_enabled=args.query_focus_index,
            query_focus_index_limit=args.query_focus_limit,
            query_focus_excerpt_chars=(
                args.query_focus_excerpt_chars
                if args.query_focus_excerpt_chars is not None else 360),
            focused_prompt_scope=args.focused_prompt_scope,
            answer_plan_enabled=args.answer_plan,
            answer_plan_max_candidates=args.answer_plan_max_candidates,
            answer_plan_excerpt_chars=args.answer_plan_excerpt_chars,
            answer_plan_kinds=tuple(
                args.answer_plan_kind
                or ("date_difference", "temporal_order")),
            max_output_tokens=(args.max_output_tokens or None),
            sampling_seed=args.sampling_seed)

    embedding_store = None
    relation_embedding_store = None
    embedding_options = {
        "record_usage": False,
        "query_cache_path": (args.query_embedding_cache
                             or root / "query_embedding_cache.sqlite"),
        "dense_sidecar_dir": args.dense_sidecar_dir,
        "dense_backend": args.dense_backend,
    }
    if runtime_dense_options is not None:
        embedding_options.update(runtime_dense_options)
        embedding_options["record_usage"] = False
    if args.embedding_request_model:
        embedding_options["request_model_id"] = args.embedding_request_model
    if args.embedding_db:
        embedding_store = SQLiteGraphStore(args.embedding_db, read_only=True)
        embedding = QwenEmbeddingIndex(embedding_store, config, **embedding_options)
    else:
        embedding = QwenEmbeddingIndex(
            store, config, **embedding_options) if args.embedding else None
    if args.relation_embedding_db:
        relation_embedding_store = SQLiteGraphStore(
            args.relation_embedding_db, read_only=True)
        if embedding is None:
            raise ValueError("--relation-embedding-db requires --embedding")
    fact_dense_search = (
        (lambda memory_id, query, item_ids, limit:
         embedding.search_items(
             memory_id, query, item_ids, limit,
             source_store=relation_embedding_store))
        if embedding is not None and relation_embedding_store is not None
        else None)
    navigator_common = {
        "dense_search": embedding.search if embedding else None,
        "dense_search_many": embedding.search_many if embedding else None,
        "fact_dense_search": fact_dense_search,
    }
    if runtime_config is not None:
        navigator_options = effective_runtime_navigator_options(
            runtime_config,
            disable_hierarchical_routing=args.no_hierarchical_routing,
            disable_graph_traversal=args.no_h10_traversal,
            research_overrides={
                "rank_mandatory": args.rank_mandatory,
                "h10_owner_rescue": not args.no_h10_owner_rescue,
                "manifest_collection_key": not args.no_manifest_collection_key,
                "raw_fallback_reserve": args.raw_fallback_reserve,
                "query_gated_rare_lexical": args.query_gated_rare_lexical,
                "relational_consensus_bonus": args.relational_consensus_bonus,
            })
        # These research-only switches are intentionally absent from the
        # stable runtime schema. They remain off unless explicitly requested.
        navigator = GraphNavigator(
            store, **navigator_common, **navigator_options)
    else:
        navigator = GraphNavigator(store, **navigator_common,
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
                               relational_view_scoring=(
                                   args.relational_view_scoring),
                               query_relation_view=args.query_relation_view,
                               relational_view_named_speakers_only=(
                                   args.relational_view_named_speakers_only),
                               relational_consensus_bonus=(
                                   args.relational_consensus_bonus),
                               dialogue_response_closure=(
                                   args.dialogue_response_closure),
                               dialogue_response_flood_threshold=(
                                   args.dialogue_response_flood_threshold),
                               proof_priority_bonus=args.proof_priority_bonus,
                               proof_priority_flood_threshold=(
                                   args.proof_priority_flood_threshold),
                               speaker_owner_bonus=args.speaker_owner_bonus,
                               query_witness_bonus=args.query_witness_bonus,
                               query_witness_seed_count=(
                                   args.query_witness_seed_count),
                               query_witness_rare_df=args.query_witness_rare_df,
                               query_witness_min_shared_terms=(
                                   args.query_witness_min_shared_terms),
                               queryir_soft_fallback=args.queryir_soft_fallback,
                               queryir_soft_fallback_threshold=(
                                   args.queryir_soft_fallback_threshold),
                               exact_lookup_fast_path=(
                                   args.exact_lookup_fast_path),
                               exact_lookup_turn_limit=(
                                   args.exact_lookup_turn_limit),
                               exact_lookup_priority=args.exact_lookup_priority,
                               exact_lookup_priority_min_score=(
                                   args.exact_lookup_priority_min_score),
                               exact_lookup_priority_bonus=(
                                   args.exact_lookup_priority_bonus),
                               exact_lookup_priority_named_speakers_only=(
                                   args.exact_lookup_priority_named_speakers_only),
                               graph_hop_decay=args.graph_hop_decay,
                               expansion_beam=args.expansion_beam,
                               hierarchy_descent_beam=(
                                   args.hierarchy_descent_beam),
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

    def navigate_rows(question_rows):
        """Bounded parallel navigation shared by prepare and answer modes."""

        indexed = iter(enumerate(question_rows))

        def navigate_one(item):
            index, question = item
            result = navigator.navigate(
                question.memory_id, question.query, budget)
            metric = navigation_metrics(question, result, store)
            return index, question, result, metric

        workers = max(1, args.navigate_workers)
        if workers == 1:
            for item in indexed:
                yield navigate_one(item)
            return
        # Keep only one wave of graph reads resident.  Full-corpus memories are
        # large, so submitting all 2,040 jobs would retain completed views and
        # prompts faster than the answer stage can drain them.
        with ThreadPoolExecutor(max_workers=workers) as navigation_pool:
            futures = {}
            for _ in range(workers):
                try:
                    item = next(indexed)
                except StopIteration:
                    break
                futures[navigation_pool.submit(navigate_one, item)] = None
            while futures:
                done, _pending = wait(
                    tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    yield future.result()
                    try:
                        item = next(indexed)
                    except StopIteration:
                        continue
                    futures[navigation_pool.submit(navigate_one, item)] = None

    answer_path = root / "answers.jsonl"
    retrieval_path = root / "retrieval.jsonl"
    prepared_path = root / "prepared_answers.jsonl"

    def load_rows(path: Path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    if args.prepare_only:
        # A retrieval/Prompt gate must cover the same full question list as the
        # answer experiment while spending zero answer-generation Tokens.  Keep
        # this branch in the canonical runner so it cannot silently drift from
        # the production navigation, AnswerConfig, tokenizer or prompt renderer.
        existing_prepared = load_rows(prepared_path) if args.resume else []
        existing_retrieval = load_rows(retrieval_path) if args.resume else []
        prepared_by_id = {
            str(row["question_id"]): row for row in existing_prepared}
        retrieval_by_id = {
            str(row["dev_question_id"]): row for row in existing_retrieval}
        completed_ids = set(prepared_by_id) & set(retrieval_by_id)
        pending = [row for row in questions if row.question_id not in completed_ids]
        print(
            f"preparing {len(pending)} remaining / {len(questions)} questions "
            f"with {args.profile}; answer calls disabled", flush=True)
        with (prepared_path.open("a", encoding="utf-8") as prepared_handle,
              retrieval_path.open("a", encoding="utf-8") as retrieval_handle):
            for completed_count, (_index, question, result, metric) in enumerate(
                    navigate_rows(pending), 1):
                prepared_answer = prepare((question, result, metric))
                prepared_row = prepared_answer.to_record()
                row_flags = flags.get(question.question_id)
                retrieval_row = {
                    "dev_question_id": question.question_id,
                    "stratum": question.stratum,
                    "has_turn_gold": (
                        bool(row_flags.has_turn_gold) if row_flags else True),
                    "is_abstention": (
                        bool(row_flags.is_abstention) if row_flags else False),
                    "benchmark": question.benchmark,
                    "configuration": label,
                    "prompt_tokens": prepared_answer.packing_prompt_tokens,
                    "evidence_tokens": prepared_answer.evidence_tokens,
                    "packed_turns": len(prepared_answer.evidence_turn_ids),
                    "aggregation_ledger": prepared_answer.trace.get(
                        "aggregation_ledger"),
                    "aggregation_source_reserve_turns": (
                        prepared_answer.trace.get(
                            "aggregation_source_reserve_turns", 0)),
                    "evidence_layout": prepared_answer.trace.get(
                        "evidence_layout", ""),
                    "execution_mode": result.trace.get(
                        "execution_mode", "hierarchical_graph"),
                    "exact_lookup": result.trace.get("exact_lookup", {}),
                    "dual_lane_rank": result.trace.get("dual_lane_rank", {}),
                    "dual_lane_active": result.trace.get(
                        "dual_lane_active", False),
                    "dual_lane_named_transcript": result.trace.get(
                        "dual_lane_named_transcript", False),
                    "dual_lane_legacy_operator_route": result.trace.get(
                        "dual_lane_legacy_operator_route", False),
                    "dual_lane_precision_packed": result.trace.get(
                        "dual_lane_precision_packed", 0),
                    "dual_lane_coverage_packed": result.trace.get(
                        "dual_lane_coverage_packed", 0),
                    "traversed_relation_signals": traversed_signals(
                        question.memory_id, result),
                    "retrieved_turn_ids": list(result.retrieved_turn_ids),
                    **({"candidate_scores": [{
                        "turn_id": candidate.turn_id,
                        "rank": rank,
                        "exact_score": candidate.exact_score,
                        "bm25_score": candidate.bm25_score,
                        "dense_score": candidate.dense_score,
                        "graph_score": candidate.graph_score,
                        "binding_score": candidate.binding_score,
                        "operand_ids": list(candidate.operand_ids),
                        "session_score": candidate.session_score,
                        "adjacency_score": candidate.adjacency_score,
                        "source_channels": list(candidate.source_channels),
                        "fused_score": candidate.fused_score,
                        "relational_consensus_score": (
                            candidate.relational_consensus_score),
                        "mandatory": candidate.mandatory,
                        "packed": candidate.turn_id in set(
                            result.retrieved_turn_ids),
                    } for rank, candidate in enumerate(
                        result.candidate_scores, 1)]}
                       if args.save_candidate_scores else {}),
                    **{key: value for key, value in metric.items()
                       if key != "question_id"},
                }
                prepared_handle.write(
                    json.dumps(prepared_row, ensure_ascii=True) + "\n")
                retrieval_handle.write(
                    json.dumps(retrieval_row, ensure_ascii=True) + "\n")
                prepared_by_id[question.question_id] = prepared_row
                retrieval_by_id[question.question_id] = retrieval_row
                if completed_count % max(1, args.checkpoint_every) == 0:
                    prepared_handle.flush()
                    retrieval_handle.flush()
                    print(
                        f"  prepared {len(completed_ids) + completed_count}/{len(questions)}",
                        flush=True)

        question_order = [str(row.question_id) for row in questions]
        prepared_rows = [prepared_by_id[question_id] for question_id in question_order]
        retrieval_rows = [retrieval_by_id[question_id] for question_id in question_order]
        prepared_path.write_text("".join(
            json.dumps(row, ensure_ascii=True) + "\n" for row in prepared_rows),
            encoding="utf-8")
        retrieval_path.write_text("".join(
            json.dumps(row, ensure_ascii=True) + "\n" for row in retrieval_rows),
            encoding="utf-8")
        manifest = {
            "schema_version": "graphmem-v5.22-full-prepare-gate-v1",
            "profile": args.profile,
            "label": label,
            "questions": len(questions),
            "full": bool(args.full),
            "source_db": str(args.source_db),
            "config_hash": config_hash(config),
            "runtime_config": (str(args.runtime_config)
                               if args.runtime_config is not None else None),
            "runtime_config_hash": (runtime_config_hash(runtime_config)
                                    if runtime_config is not None else None),
            "answer_calls": 0,
            "answer_generation_tokens": 0,
            "answer_policy": args.answer_policy,
            "core_readout_policy": answer_config.readout_policy,
            "answer_plan": answer_config.answer_plan_enabled,
            "answer_plan_max_candidates": (
                answer_config.answer_plan_max_candidates),
            "answer_plan_excerpt_chars": (
                answer_config.answer_plan_excerpt_chars),
            "answer_plan_kinds": list(answer_config.answer_plan_kinds),
            "graph_traversal": navigator.h10_traversal,
            "hierarchical_routing": navigator.hierarchical_routing,
            "structural_ablation_override": {
                "disable_graph_traversal": bool(args.no_h10_traversal),
                "disable_hierarchical_routing": bool(
                    args.no_hierarchical_routing),
            },
            "candidate_scores_saved": args.save_candidate_scores,
            "evidence_order": answer_config.evidence_order,
            "source_time_normalization": (
                answer_config.normalize_relative_time),
            "question_date_mode": answer_config.question_date_mode,
            "question_recency_footer": (
                answer_config.question_recency_footer),
            "compact_topological_prompt": (
                answer_config.compact_topological_contract),
            "query_focus_index": answer_config.query_focus_index_enabled,
            "query_focus_limit": answer_config.query_focus_index_limit,
            "query_focus_excerpt_chars": (
                answer_config.query_focus_excerpt_chars),
            "temporal_query_focus": (
                answer_config.temporal_query_focus_enabled),
            "preference_focus_strategy": (
                answer_config.preference_focus_strategy),
            "aggregation_operand_worksheet_selective": (
                answer_config.aggregation_operand_worksheet_selective),
            "focused_prompt_scope": answer_config.focused_prompt_scope,
            "exact_grounding_footer": (
                answer_config.exact_grounding_footer),
            "aggregation_ledger_limit": answer_config.aggregation_ledger_limit,
            "aggregation_execution_card": (
                answer_config.aggregation_execution_card),
            "aggregation_source_reserve": (
                answer_config.aggregation_source_reserve_enabled),
            "aggregation_source_reserve_operations": list(
                answer_config.aggregation_source_reserve_operations),
            "exact_lookup_priority": navigator.exact_lookup_priority,
            "exact_lookup_priority_min_score": (
                navigator.exact_lookup_priority_min_score),
            "exact_lookup_priority_bonus": navigator.exact_lookup_priority_bonus,
            "exact_lookup_priority_named_speakers_only": (
                navigator.exact_lookup_priority_named_speakers_only),
            "navigate_workers": max(1, args.navigate_workers),
            "native_seed_fusion": navigator.native_seed_fusion,
            "relational_view_scoring": navigator.relational_view_scoring,
            "query_relation_view": navigator.query_relation_view,
            "relational_view_named_speakers_only": (
                navigator.relational_view_named_speakers_only),
            "relational_consensus_bonus": navigator.relational_consensus_bonus,
            "dual_lane_packing": navigator.dual_lane_packing,
            "dual_lane_operator_aware": navigator.dual_lane_operator_aware,
            "dual_lane_precision_head": navigator.dual_lane_precision_head,
            "dual_lane_rrf_k": navigator.dual_lane_rrf_k,
            "dual_lane_proof_reserve": navigator.dual_lane_proof_reserve,
            "speaker_owner_bonus": navigator.speaker_owner_bonus,
            "query_witness_bonus": navigator.query_witness_bonus,
            "query_witness_seed_count": navigator.query_witness_seed_count,
            "query_witness_rare_df": navigator.query_witness_rare_df,
            "query_witness_min_shared_terms": (
                navigator.query_witness_min_shared_terms),
            "max_evidence_turns": budget.max_evidence_turns,
            "max_evidence_tokens": budget.max_evidence_tokens,
            "max_answer_tokens": budget.max_answer_tokens,
            "prepared_answers": str(prepared_path),
            "retrieval": str(retrieval_path),
        }
        (root / "prepare_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        store.close()
        cache_store.close()
        if embedding_store is not None:
            embedding_store.close()
        if relation_embedding_store is not None:
            relation_embedding_store.close()
        return

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
                "execution_mode": result.trace.get(
                    "execution_mode", "hierarchical_graph"),
                "exact_lookup": result.trace.get("exact_lookup", {}),
                "dual_lane_rank": result.trace.get("dual_lane_rank", {}),
                "dual_lane_active": result.trace.get(
                    "dual_lane_active", False),
                "dual_lane_named_transcript": result.trace.get(
                    "dual_lane_named_transcript", False),
                "dual_lane_legacy_operator_route": result.trace.get(
                    "dual_lane_legacy_operator_route", False),
                "dual_lane_precision_packed": result.trace.get(
                    "dual_lane_precision_packed", 0),
                "dual_lane_coverage_packed": result.trace.get(
                    "dual_lane_coverage_packed", 0),
                "aggregation_ledger": answer.trace.get("aggregation_ledger"),
                **{key: value for key, value in metric.items() if key != "question_id"},
            }
            pending_checkpoints.append((
                index, answer_row, retrieval_row, prepared_answer.to_record()))
            if len(pending_checkpoints) >= batch_size:
                checkpoint(pending_checkpoints)

        # One request per worker is already enough to saturate vLLM.  Keeping a
        # second full wave delayed the first durable checkpoint until 512 slow
        # navigation calls had completed, even when the first 200+ answers were
        # already sitting in memory.  Bound at one wave so full-corpus runs
        # become resumable earlier without reducing model concurrency.
        max_in_flight = max(1, args.answer_workers)
        for index, question, result, metric in navigate_rows(remaining):
            prepared_answer = prepare((question, result, metric))
            future = pool.submit(stage.complete, prepared_answer)
            futures[future] = (
                index, question, result, metric, prepared_answer)
            # Drain answers while navigation continues.  Previously completed
            # requests stayed in ``futures`` until the set reached 256; with
            # navigation slower than vLLM this postponed the first durable
            # checkpoint until roughly request 512 despite idle answer slots.
            done = {item for item in futures if item.done()}
            if not done and len(futures) >= max_in_flight:
                done, _pending = wait(
                    tuple(futures), return_when=FIRST_COMPLETED)
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
    def focused_contract_flags(mode: str) -> tuple[bool, bool, bool]:
        enabled = (answer_config.focused_prompt_scope == "all"
                   or mode == "default")
        return (
            enabled and answer_config.question_date_mode != "always",
            enabled and answer_config.question_recency_footer,
            enabled and answer_config.compact_topological_contract,
        )
    default_contract_flags = focused_contract_flags("default")
    aggregation_contract_flags = focused_contract_flags("aggregation")
    preference_contract_flags = focused_contract_flags("preference")
    manifest = {
        "profile": args.profile, "label": label, "questions": len(answer_rows),
        "answer_policy": answer_config.readout_policy,
        "answer_plan": answer_config.answer_plan_enabled,
        "answer_plan_max_candidates": (
            answer_config.answer_plan_max_candidates),
        "answer_plan_excerpt_chars": (
            answer_config.answer_plan_excerpt_chars),
        "answer_plan_kinds": list(answer_config.answer_plan_kinds),
        "runtime_config": (str(args.runtime_config)
                           if args.runtime_config is not None else None),
        "runtime_config_hash": (runtime_config_hash(runtime_config)
                                if runtime_config is not None else None),
        "question_filters": {
            "lme_types": list(args.lme_type),
            "locomo_categories": list(args.locomo_category),
        },
        "source_db": str(args.source_db), "config_hash": config_hash(config),
        "answer_prompt_hash": prompt_contract(
            answer_config.normalize_relative_time,
            answer_config.precision_grounding,
            answer_config.evidence_order in {
                "topological", "topological_recency"},
            answer_config.aggregation_ledger_enabled,
            answer_config.preference_synthesis_enabled,
            answer_config.exact_grounding_footer,
            *default_contract_flags)[2],
        "answer_prompt_hashes_observed": sorted({
            str(row.get("prompt_hash") or "") for row in prepared_rows
            if str(row.get("prompt_hash") or "")}),
        "answer_prompt_hash_by_mode": {
            "default": prompt_contract(
                answer_config.normalize_relative_time,
                answer_config.precision_grounding,
                answer_config.evidence_order in {
                    "topological", "topological_recency"}, False, False,
                answer_config.exact_grounding_footer,
                *default_contract_flags)[2],
            "aggregation": (prompt_contract(
                answer_config.normalize_relative_time,
                answer_config.precision_grounding,
                answer_config.evidence_order in {
                    "topological", "topological_recency"}, True, False,
                answer_config.exact_grounding_footer,
                *aggregation_contract_flags)[2]
                if answer_config.aggregation_ledger_enabled else None),
            "preference_synthesis": (prompt_contract(
                answer_config.normalize_relative_time,
                answer_config.precision_grounding,
                answer_config.evidence_order in {
                    "topological", "topological_recency"}, False, True,
                answer_config.exact_grounding_footer,
                *preference_contract_flags)[2]
                if answer_config.preference_synthesis_enabled else None),
        },
        "span_window": answer_config.span_window,
        "evidence_order": answer_config.evidence_order,
        "source_time_normalization": answer_config.normalize_relative_time,
        "question_date_mode": answer_config.question_date_mode,
        "question_recency_footer": answer_config.question_recency_footer,
        "compact_topological_prompt": (
            answer_config.compact_topological_contract),
        "query_focus_index": answer_config.query_focus_index_enabled,
        "query_focus_limit": answer_config.query_focus_index_limit,
        "query_focus_excerpt_chars": (
            answer_config.query_focus_excerpt_chars),
        "temporal_query_focus": answer_config.temporal_query_focus_enabled,
        "preference_focus_strategy": answer_config.preference_focus_strategy,
        "focused_prompt_scope": answer_config.focused_prompt_scope,
        "precision_grounded_prompt": answer_config.precision_grounding,
        "aggregation_ledger": answer_config.aggregation_ledger_enabled,
        "aggregation_ledger_schema_version": (
            AGGREGATION_LEDGER_SCHEMA_VERSION
            if answer_config.aggregation_ledger_enabled else None),
        "aggregation_ledger_limit": answer_config.aggregation_ledger_limit,
        "aggregation_execution_card": (
            answer_config.aggregation_execution_card),
        "aggregation_operand_worksheet": (
            answer_config.aggregation_operand_worksheet_enabled),
        "aggregation_operand_worksheet_selective": (
            answer_config.aggregation_operand_worksheet_selective),
        "aggregation_source_reserve": (
            answer_config.aggregation_source_reserve_enabled),
        "aggregation_source_reserve_operations": list(
            answer_config.aggregation_source_reserve_operations),
        "preference_synthesis_prompt": (
            answer_config.preference_synthesis_enabled),
        "exact_grounding_footer": answer_config.exact_grounding_footer,
        "obligation_aware_packing": args.obligation_aware_packing,
        "precision_aware_packing": args.precision_aware_packing,
        "dual_lane_packing": navigator.dual_lane_packing,
        "dual_lane_operator_aware": navigator.dual_lane_operator_aware,
        "dual_lane_precision_head": navigator.dual_lane_precision_head,
        "dual_lane_rrf_k": navigator.dual_lane_rrf_k,
        "dual_lane_proof_reserve": navigator.dual_lane_proof_reserve,
        "candidate_pool_limit": args.candidate_pool_limit,
        "raw_fallback_reserve": args.raw_fallback_reserve,
        "obligation_aware_relations": args.obligation_aware_relations,
        "graph_traversal": navigator.h10_traversal,
        "hierarchical_routing": navigator.hierarchical_routing,
        "structural_ablation_override": {
            "disable_graph_traversal": bool(args.no_h10_traversal),
            "disable_hierarchical_routing": bool(
                args.no_hierarchical_routing),
        },
        "native_seed_fusion": navigator.native_seed_fusion,
        "relational_view_scoring": navigator.relational_view_scoring,
        "query_relation_view": navigator.query_relation_view,
        "relational_view_named_speakers_only": (
            navigator.relational_view_named_speakers_only),
        "relational_consensus_bonus": navigator.relational_consensus_bonus,
        "dialogue_response_closure": args.dialogue_response_closure,
        "dialogue_response_flood_threshold": (
            args.dialogue_response_flood_threshold),
        "proof_priority_bonus": args.proof_priority_bonus,
        "proof_priority_flood_threshold": (
            args.proof_priority_flood_threshold),
        "speaker_owner_bonus": navigator.speaker_owner_bonus,
        "query_witness_bonus": navigator.query_witness_bonus,
        "query_witness_seed_count": navigator.query_witness_seed_count,
        "query_witness_rare_df": navigator.query_witness_rare_df,
        "query_witness_min_shared_terms": (
            navigator.query_witness_min_shared_terms),
        "dense_search": embedding is not None,
        "embedding_db": str(args.embedding_db) if args.embedding_db else None,
        "relation_embedding_db": (
            str(args.relation_embedding_db)
            if args.relation_embedding_db else None),
        "queryir_soft_fallback": args.queryir_soft_fallback,
        "queryir_soft_fallback_threshold": args.queryir_soft_fallback_threshold,
        "exact_lookup_fast_path": args.exact_lookup_fast_path,
        "exact_lookup_turn_limit": args.exact_lookup_turn_limit,
        "exact_lookup_priority": navigator.exact_lookup_priority,
        "exact_lookup_priority_min_score": (
            navigator.exact_lookup_priority_min_score),
        "exact_lookup_priority_bonus": navigator.exact_lookup_priority_bonus,
        "exact_lookup_priority_named_speakers_only": (
            navigator.exact_lookup_priority_named_speakers_only),
        "graph_hop_decay": args.graph_hop_decay,
        "expansion_beam": args.expansion_beam,
        "hierarchy_descent_beam": args.hierarchy_descent_beam,
        "rare_lexical_relations": args.rare_lexical_relations,
        "query_gated_rare_lexical": args.query_gated_rare_lexical,
        "navigate_workers": max(1, args.navigate_workers),
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
    if relation_embedding_store is not None:
        relation_embedding_store.close()
    cache_store.close()


if __name__ == "__main__":
    main()
