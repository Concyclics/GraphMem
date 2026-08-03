#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.pipeline import DemoConfig, run_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GraphMem LLM token demo.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--question-type", default="all")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "direct_session_k16_compact_no_compress",
            "direct_session_k16_compact_graphmem",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-cache-dir", type=Path)
    parser.add_argument(
        "--deepseek-model", "--llm-model", dest="deepseek_model",
        default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument(
        "--deepseek-base-url", "--llm-base-url",
        dest="deepseek_base_url",
        default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"),
        help="OpenAI-compatible LLM base URL (local vLLM: http://127.0.0.1:8001/v1).",
    )
    parser.add_argument(
        "--llm-api-key-env", default="SGAO_API_KEY",
        help="Environment variable containing the OpenAI-compatible LLM API key.",
    )
    parser.add_argument(
        "--llm-request-profile", choices=["deepseek", "openai", "omit"],
        default="openai",
        help="Map no-reasoning to DeepSeek thinking.disabled, OpenAI reasoning_effort=none, or no field.",
    )
    parser.add_argument(
        "--llm-local",
        action="store_true",
        help="Use local vLLM for build/answer (port 8001, dummy API key).",
    )
    parser.add_argument("--llm-local-port", type=int, default=8001)
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"))
    parser.add_argument("--tree-mode", choices=["legacy_kway", "direct_session", "hierarchical_state_graph_v2", "hierarchical_hypergraph_v3", "hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"])
    parser.add_argument("--fanout-k", type=int, default=16)
    parser.add_argument("--max-group-rough-tokens", type=int, default=6000)
    parser.add_argument("--leaf-top-k", type=int, default=14)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--root-candidate-k", type=int, default=8)
    parser.add_argument("--global-leaf-top-k", type=int, default=24)
    parser.add_argument("--qa-summary-top-k", type=int, default=4)
    parser.add_argument("--per-session-leaf-k", type=int, default=2)
    parser.add_argument("--enable-coverage-rerank", action="store_true")
    parser.add_argument("--coverage-rerank-lambda", type=float, default=0.75)
    parser.add_argument("--coverage-rerank-pool-k", type=int, default=80)
    parser.add_argument("--graph-neighbor-k", type=int, default=2)
    parser.add_argument("--qa-context-token-budget", type=int, default=10000)
    parser.add_argument("--qa-max-tokens", type=int, default=1024)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--question-workers", type=int, default=2)
    parser.add_argument("--summary-workers", type=int, default=32)
    parser.add_argument("--max-inflight-deepseek", type=int, default=32)
    parser.add_argument(
        "--summary-schema",
        choices=["minimal_memory_v1", "compact_memory_v2", "multilingual_memory_v1", "graphmem_v2", "graphmem_v3", "graphmem_v3_6", "graphmem_v4_0", "graphmem_v4_1_query"],
    )
    parser.add_argument(
        "--summarizer-kind",
        choices=["auto", "none", "llmlingua2", "qwen_local"],
        default="auto",
    )
    parser.add_argument("--summarizer-base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--summarizer-model")
    parser.add_argument("--summary-token-budget", type=int, default=1280)
    parser.add_argument("--build-leaf-text", choices=["auto", "raw", "user_only"], default="auto")
    parser.add_argument(
        "--retrieval-leaf-text",
        choices=["auto", "raw", "user_only"],
        default="auto",
    )
    parser.add_argument("--compressor-chunk-rough-tokens", type=int, default=384)
    parser.add_argument("--raw-group-summary-max-tokens", type=int, default=1024)
    parser.add_argument("--session-summary-max-tokens", type=int, default=2048)
    parser.add_argument("--legacy-internal-summary-max-tokens", type=int, default=896)
    parser.add_argument("--llmlingua-model")
    parser.add_argument("--llmlingua-device-map")
    parser.add_argument("--use-llmlingua2", action="store_true")
    parser.add_argument("--enable-speaker-profiles", action="store_true")
    parser.add_argument("--enable-speaker-neighbor-window", action="store_true")
    parser.add_argument("--enable-speaker-retrieval-text", action="store_true")
    parser.add_argument(
        "--disable-explicit-speaker-retrieval-boost",
        action="store_true",
        help="Do not auto-inflate leaf/global/per-session retrieval k for speaker-labeled data.",
    )
    parser.add_argument(
        "--disable-leaf-enrichment",
        action="store_true",
        help="Skip per-leaf facts/keywords emitted during compact_memory_v2 session summary.",
    )
    parser.add_argument(
        "--disable-lossless-root-summary",
        action="store_true",
        help="Store LLM-rendered compact summary on roots instead of full child dialogue.",
    )
    parser.add_argument("--enable-typed-root-edges", action="store_true")
    parser.add_argument("--enable-multilevel-summary-retrieval", action="store_true")
    parser.add_argument("--enable-llm-root-edges", action="store_true")
    parser.add_argument("--enable-llm-leaf-edges", action="store_true")
    parser.add_argument("--enable-leaf-graph-expansion", action="store_true")
    parser.add_argument(
        "--enable-compute-plan",
        action="store_true",
        help="Level-2 program-aided arithmetic: LLM emits a JSON compute plan, code executes it.",
    )
    parser.add_argument(
        "--force-enhanced-retrieval",
        action="store_true",
        help="Force enhanced retrieval logic regardless of variant defaults.",
    )
    parser.add_argument(
        "--force-enhanced-qa",
        action="store_true",
        help="Force enhanced QA prompt regardless of variant defaults.",
    )
    parser.add_argument(
        "--enable-answer-note-extraction",
        action="store_true",
        help="Run stage-A note extraction before final QA.",
    )
    parser.add_argument(
        "--answer-note-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for note extraction stage (default: 1024).",
    )
    parser.add_argument(
        "--disable-answer-use-notes-for-qa",
        action="store_true",
        help="Extract notes but keep final QA on raw context.",
    )
    parser.add_argument(
        "--answer-include-raw-context-with-notes",
        action="store_true",
        help="When QA uses notes, append raw evidence as fallback context.",
    )
    parser.add_argument("--llm-root-edge-max-tokens", type=int, default=256)
    parser.add_argument("--llm-root-edge-neighbors-per-relation", type=int, default=2)
    parser.add_argument("--llm-root-edge-min-shared", type=int, default=1)
    parser.add_argument("--llm-root-edge-anchor-limit", type=int, default=8)
    parser.add_argument("--llm-leaf-edge-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-leaf-edge-max-snippet-chars", type=int, default=1024)
    parser.add_argument(
        "--llm-leaf-edge-min-confidence",
        type=float,
        default=0.8,
        help="Drop LLM leaf edges below this confidence (default: 0.8).",
    )
    parser.add_argument(
        "--llm-leaf-edge-max-edges-per-leaf",
        type=int,
        default=3,
        help="Cap non-temporal LLM edges per leaf; keep highest confidence (default: 3).",
    )
    parser.add_argument(
        "--llm-leaf-edge-max-edges-per-session",
        type=int,
        default=16,
        help="Cap LLM leaf edges per session after confidence filter (default: 16).",
    )
    parser.add_argument("--llm-leaf-edge-max-leaves-per-session", type=int, default=48)
    parser.add_argument("--leaf-graph-neighbor-k", type=int, default=2)
    parser.add_argument("--leaf-graph-expansion-budget", type=int, default=4)
    parser.add_argument(
        "--enable-graph-search",
        action="store_true",
        help="HippoRAG-style retrieval: embedding seeds + PPR over root/leaf edges.",
    )
    parser.add_argument("--graph-search-seed-roots", type=int, default=6)
    parser.add_argument("--graph-search-seed-leaves", type=int, default=10)
    parser.add_argument("--graph-search-ppr-damping", type=float, default=0.85)
    parser.add_argument("--graph-search-ppr-iterations", type=int, default=25)
    parser.add_argument("--graph-search-embedding-blend", type=float, default=0.1)
    parser.add_argument(
        "--graph-search-seed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Embedding only picks PPR seeds; PPR runs on the full graph and leaves are "
            "selected by global graph score (default: on)."
        ),
    )
    parser.add_argument(
        "--graph-search-structural-root-leaf-weight",
        type=float,
        default=0.1,
        help="Root↔leaf structural weight (default: 0.1; avoid high values).",
    )
    parser.add_argument(
        "--graph-search-session-coverage",
        type=int,
        default=0,
        help="Free-select: guarantee ≥1 leaf from each of the top-N sessions (0=off, default).",
    )
    parser.add_argument("--graph-search-session-min-leaves", type=int, default=3)
    parser.add_argument("--graph-search-max-sessions", type=int, default=8)
    parser.add_argument("--graph-search-per-session-leaf-cap", type=int, default=4)
    parser.add_argument("--no-graph-search-protect-leaves", action="store_true")
    parser.add_argument(
        "--enable-graph-first-retrieval",
        action="store_true",
        help=(
            "Graph-primary retrieval: PPR/graph scores drive leaf selection with a "
            "global embedding safety pool (recommended over --enable-graph-search)."
        ),
    )
    parser.add_argument(
        "--graph-first-embedding-blend",
        type=float,
        default=0.25,
        help="Final leaf score weight on embedding rank (default: 0.25; graph dominates).",
    )
    parser.add_argument(
        "--graph-first-session-coverage",
        type=int,
        default=2,
        help="Guarantee at least one leaf from each of the top-N graph-ranked sessions.",
    )
    parser.add_argument(
        "--graph-first-candidate-pool-k",
        type=int,
        default=80,
        help="Max graph-expanded leaves in the candidate pool before final selection.",
    )
    parser.add_argument(
        "--enable-fusion-retrieval",
        action="store_true",
        help=(
            "Triple-pass leaf ranking: semantic embedding + BM25 keyword + entity overlap, "
            "fused with RRF (composable with graph-first)."
        ),
    )
    parser.add_argument(
        "--fusion-method",
        choices=["rrf", "weighted"],
        default="rrf",
        help="How to combine semantic/keyword/entity ranks (default: rrf).",
    )
    parser.add_argument("--fusion-rrf-k", type=int, default=60)
    parser.add_argument("--fusion-weight-semantic", type=float, default=1.0)
    parser.add_argument("--fusion-weight-keyword", type=float, default=1.0)
    parser.add_argument("--fusion-weight-entity", type=float, default=1.0)
    parser.add_argument(
        "--no-fusion-query-adaptive-weights",
        action="store_true",
        help="Disable query-type weight boosts for keyword/entity passes.",
    )
    parser.add_argument(
        "--disable-typed-retrieval",
        action="store_true",
        help="Disable query-time typed anchor matching for root ranking and PPR seeds.",
    )
    parser.add_argument(
        "--typed-retrieval-embedding-blend",
        type=float,
        default=0.55,
        help="Blend between embedding (1.0) and typed anchor overlap (0.0) for roots/seeds.",
    )
    parser.add_argument(
        "--disable-protected-fusion",
        action="store_true",
        help="Allow fusion to fully rerank leaves instead of protecting semantic top-K.",
    )
    parser.add_argument(
        "--fusion-semantic-protect-k",
        type=int,
        default=10,
        help="When protected fusion is on, keep this many semantic-top leaves at the front.",
    )
    parser.add_argument(
        "--disable-query-type-retrieval-boost",
        action="store_true",
        help="Disable question-type-aware retrieval budget boost (list/date questions).",
    )
    parser.add_argument(
        "--list-question-extra-leaf-budget",
        type=int,
        default=6,
        help="Extra leaf budget for list/set-style questions.",
    )
    parser.add_argument(
        "--temporal-question-extra-leaf-budget",
        type=int,
        default=4,
        help="Extra leaf budget for temporal/date questions.",
    )
    parser.add_argument(
        "--disable-dual-channel-candidate-merge",
        action="store_true",
        help="Disable semantic+structured dual-channel candidate merge.",
    )
    parser.add_argument(
        "--dual-channel-structured-pool-k",
        type=int,
        default=24,
        help="Structured channel top-k merged into semantic ranking.",
    )
    parser.add_argument(
        "--enable-iterative-leaf-denoise",
        action="store_true",
        help="Iteratively kick low-relevance selected leaves and backfill from ranked tail.",
    )
    parser.add_argument(
        "--iterative-leaf-denoise-max-rounds",
        type=int,
        default=3,
        help="Maximum denoise rounds (default: 3).",
    )
    parser.add_argument(
        "--iterative-leaf-denoise-max-kick-per-round",
        type=int,
        default=5,
        help="Maximum leaves kicked each round (default: 5).",
    )
    parser.add_argument(
        "--iterative-leaf-denoise-min-relevance-ratio",
        type=float,
        default=0.35,
        help="Kick leaves below max_relevance * ratio (default: 0.35).",
    )
    parser.add_argument(
        "--iterative-leaf-denoise-protect-top-k",
        type=int,
        default=3,
        help="Protect this many top-ranked pool leaves from kicking (default: 3).",
    )
    parser.add_argument(
        "--iterative-leaf-denoise-keep-structured-top-k",
        type=int,
        default=2,
        help="Always protect this many structured-signal leaves during LLM denoise.",
    )
    parser.add_argument(
        "--typed-root-neighbors-per-relation",
        type=int,
        default=1,
        help="Max typed root edges kept per relation type (entity/update/time/...).",
    )
    parser.add_argument(
        "--typed-root-max-edges-per-root",
        type=int,
        default=6,
        help="Cap non-semantic/non-keyword typed edges per root node.",
    )
    parser.add_argument(
        "--typed-root-min-edge-score",
        type=float,
        default=0.76,
        help="Drop typed root edges below this score before PPR.",
    )
    parser.add_argument(
        "--typed-root-semantic-support-min-cosine",
        type=float,
        default=0.25,
        help="Require at least this embedding cosine for typed root bridges.",
    )
    parser.add_argument(
        "--disable-typed-root-semantic-support",
        action="store_true",
        help="Allow typed root edges even when session embeddings are dissimilar.",
    )
    parser.add_argument("--reasoning-effort", choices=["none"], default="none")
    parser.add_argument("--build-budget-tokens", type=int, default=300000)
    parser.add_argument("--answer-budget-tokens", type=int, default=10000)
    parser.add_argument("--v2-fact-extraction-max-tokens", type=int, default=3072)
    parser.add_argument("--v2-consolidation-max-tokens", type=int, default=3072)
    parser.add_argument("--v2-card-k", type=int, default=6)
    parser.add_argument("--v2-fact-k", type=int, default=14)
    parser.add_argument("--v2-leaf-k", type=int, default=12)
    parser.add_argument("--v2-context-token-budget", type=int, default=8200)
    parser.add_argument("--v2-semantic-k", type=int, default=3)
    parser.add_argument("--v2-semantic-floor", type=float, default=0.55)
    parser.add_argument("--v3-session-extraction-max-tokens", type=int, default=3072)
    parser.add_argument("--v3-context-token-budget", type=int, default=3600)
    parser.add_argument("--v36-session-extraction-max-tokens", type=int, default=4096)
    parser.add_argument("--v36-context-token-budget", type=int, default=8000)
    parser.add_argument("--v36-answer-hard-budget-tokens", type=int, default=10500)
    parser.add_argument("--v41-normal-context-target", type=int, default=8400)
    parser.add_argument("--v41-complex-context-target", type=int, default=9200)
    parser.add_argument("--v41-planner-prompt-max", type=int, default=700)
    parser.add_argument("--v41-planner-output-max", type=int, default=256)
    parser.add_argument("--v41-query-target-tokens", type=int, default=10000)
    parser.add_argument("--v41-query-hard-limit-tokens", type=int, default=13000)
    parser.add_argument("--disable-v41-planner", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "For V3, persist the index and retrieval ledger without calling the "
            "built-in base answer; use with the graph navigator as the sole reader."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mock-services",
        action="store_true",
        help="Use deterministic local mock LLM, embedding, and compressor implementations.",
    )
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-embedding", action="store_true")
    parser.add_argument("--mock-compressor", action="store_true")
    parser.add_argument("--mock-summarizer", action="store_true")
    return parser.parse_args()


DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def main() -> None:
    args = parse_args()
    deepseek_base_url = args.deepseek_base_url
    deepseek_model = args.deepseek_model
    if args.llm_local:
        deepseek_base_url = deepseek_base_url or f"http://127.0.0.1:{args.llm_local_port}/v1"
        deepseek_model = deepseek_model or DEFAULT_LOCAL_LLM_MODEL
        os.environ.setdefault(args.llm_api_key_env, "local-llm")
    config = DemoConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        memory_cache_dir=args.memory_cache_dir,
        question_type=args.question_type,
        variants=tuple(args.variants),
        deepseek_model=deepseek_model,
        deepseek_base_url=deepseek_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_request_profile=args.llm_request_profile,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        tree_mode=args.tree_mode,
        fanout_k=args.fanout_k,
        max_group_rough_tokens=args.max_group_rough_tokens,
        leaf_top_k=args.leaf_top_k,
        root_top_k=args.root_top_k,
        root_candidate_k=args.root_candidate_k,
        global_leaf_top_k=args.global_leaf_top_k,
        qa_summary_top_k=args.qa_summary_top_k,
        per_session_leaf_k=args.per_session_leaf_k,
        enable_coverage_rerank=args.enable_coverage_rerank,
        coverage_rerank_lambda=args.coverage_rerank_lambda,
        coverage_rerank_pool_k=args.coverage_rerank_pool_k,
        graph_neighbor_k=args.graph_neighbor_k,
        qa_context_token_budget=args.qa_context_token_budget,
        qa_max_tokens=args.qa_max_tokens,
        compression_ratio=args.compression_ratio,
        max_questions=args.max_questions,
        question_workers=args.question_workers,
        summary_workers=args.summary_workers,
        max_inflight_deepseek=args.max_inflight_deepseek,
        summary_schema=args.summary_schema,
        summarizer_kind=args.summarizer_kind,
        summarizer_base_url=args.summarizer_base_url,
        summarizer_model=args.summarizer_model,
        summary_token_budget=args.summary_token_budget,
        build_leaf_text=args.build_leaf_text,
        retrieval_leaf_text=args.retrieval_leaf_text,
        compressor_chunk_rough_tokens=args.compressor_chunk_rough_tokens,
        raw_group_summary_max_tokens=args.raw_group_summary_max_tokens,
        session_summary_max_tokens=args.session_summary_max_tokens,
        legacy_internal_summary_max_tokens=args.legacy_internal_summary_max_tokens,
        resume=args.resume,
        mock_services=args.mock_services,
        mock_llm=args.mock_llm,
        mock_embedding=args.mock_embedding,
        mock_compressor=args.mock_compressor,
        mock_summarizer=args.mock_summarizer,
        llmlingua_model=args.llmlingua_model,
        llmlingua_device_map=args.llmlingua_device_map,
        use_llmlingua2=args.use_llmlingua2,
        enable_speaker_profiles=args.enable_speaker_profiles,
        enable_speaker_neighbor_window=args.enable_speaker_neighbor_window,
        enable_speaker_retrieval_text=args.enable_speaker_retrieval_text,
        enable_explicit_speaker_retrieval_boost=not args.disable_explicit_speaker_retrieval_boost,
        enable_leaf_enrichment=not args.disable_leaf_enrichment,
        enable_lossless_root_summary=not args.disable_lossless_root_summary,
        enable_typed_root_edges=args.enable_typed_root_edges,
        enable_multilevel_summary_retrieval=args.enable_multilevel_summary_retrieval,
        enable_llm_root_edges=args.enable_llm_root_edges,
        llm_root_edge_max_tokens=args.llm_root_edge_max_tokens,
        llm_root_edge_neighbors_per_relation=args.llm_root_edge_neighbors_per_relation,
        llm_root_edge_min_shared=args.llm_root_edge_min_shared,
        llm_root_edge_anchor_limit=args.llm_root_edge_anchor_limit,
        enable_llm_leaf_edges=args.enable_llm_leaf_edges,
        enable_leaf_graph_expansion=args.enable_leaf_graph_expansion,
        llm_leaf_edge_max_tokens=args.llm_leaf_edge_max_tokens,
        llm_leaf_edge_max_snippet_chars=args.llm_leaf_edge_max_snippet_chars,
        llm_leaf_edge_min_confidence=args.llm_leaf_edge_min_confidence,
        llm_leaf_edge_max_edges_per_leaf=args.llm_leaf_edge_max_edges_per_leaf,
        llm_leaf_edge_max_edges_per_session=args.llm_leaf_edge_max_edges_per_session,
        llm_leaf_edge_max_leaves_per_session=args.llm_leaf_edge_max_leaves_per_session,
        leaf_graph_neighbor_k=args.leaf_graph_neighbor_k,
        leaf_graph_expansion_budget=args.leaf_graph_expansion_budget,
        enable_graph_search=args.enable_graph_search,
        graph_search_seed_roots=args.graph_search_seed_roots,
        graph_search_seed_leaves=args.graph_search_seed_leaves,
        graph_search_ppr_damping=args.graph_search_ppr_damping,
        graph_search_ppr_iterations=args.graph_search_ppr_iterations,
        graph_search_embedding_blend=args.graph_search_embedding_blend,
        graph_search_seed_only=args.graph_search_seed_only,
        graph_search_structural_root_leaf_weight=args.graph_search_structural_root_leaf_weight,
        graph_search_session_coverage=args.graph_search_session_coverage,
        graph_search_session_min_leaves=args.graph_search_session_min_leaves,
        graph_search_max_sessions=args.graph_search_max_sessions,
        graph_search_per_session_leaf_cap=args.graph_search_per_session_leaf_cap,
        graph_search_protect_leaves=not args.no_graph_search_protect_leaves,
        enable_graph_first_retrieval=args.enable_graph_first_retrieval,
        graph_first_embedding_blend=args.graph_first_embedding_blend,
        graph_first_session_coverage=args.graph_first_session_coverage,
        graph_first_candidate_pool_k=args.graph_first_candidate_pool_k,
        enable_fusion_retrieval=args.enable_fusion_retrieval,
        fusion_method=args.fusion_method,
        fusion_rrf_k=args.fusion_rrf_k,
        fusion_weight_semantic=args.fusion_weight_semantic,
        fusion_weight_keyword=args.fusion_weight_keyword,
        fusion_weight_entity=args.fusion_weight_entity,
        fusion_query_adaptive_weights=not args.no_fusion_query_adaptive_weights,
        enable_typed_retrieval=not args.disable_typed_retrieval,
        typed_retrieval_embedding_blend=args.typed_retrieval_embedding_blend,
        enable_protected_fusion=not args.disable_protected_fusion,
        fusion_semantic_protect_k=args.fusion_semantic_protect_k,
        enable_query_type_retrieval_boost=not args.disable_query_type_retrieval_boost,
        list_question_extra_leaf_budget=args.list_question_extra_leaf_budget,
        temporal_question_extra_leaf_budget=args.temporal_question_extra_leaf_budget,
        enable_dual_channel_candidate_merge=not args.disable_dual_channel_candidate_merge,
        dual_channel_structured_pool_k=args.dual_channel_structured_pool_k,
        enable_iterative_leaf_denoise=args.enable_iterative_leaf_denoise,
        iterative_leaf_denoise_max_rounds=args.iterative_leaf_denoise_max_rounds,
        iterative_leaf_denoise_max_kick_per_round=args.iterative_leaf_denoise_max_kick_per_round,
        iterative_leaf_denoise_min_relevance_ratio=args.iterative_leaf_denoise_min_relevance_ratio,
        iterative_leaf_denoise_protect_top_k=args.iterative_leaf_denoise_protect_top_k,
        iterative_leaf_denoise_keep_structured_top_k=args.iterative_leaf_denoise_keep_structured_top_k,
        typed_root_neighbors_per_relation=args.typed_root_neighbors_per_relation,
        typed_root_max_edges_per_root=args.typed_root_max_edges_per_root,
        typed_root_min_edge_score=args.typed_root_min_edge_score,
        typed_root_semantic_support_min_cosine=args.typed_root_semantic_support_min_cosine,
        typed_root_require_semantic_support=not args.disable_typed_root_semantic_support,
        enable_compute_plan=args.enable_compute_plan,
        enable_answer_note_extraction=args.enable_answer_note_extraction,
        answer_note_max_tokens=args.answer_note_max_tokens,
        answer_use_notes_for_qa=not args.disable_answer_use_notes_for_qa,
        answer_include_raw_context_with_notes=args.answer_include_raw_context_with_notes,
        force_enhanced_retrieval=args.force_enhanced_retrieval,
        force_enhanced_qa=args.force_enhanced_qa,
        reasoning_effort=args.reasoning_effort,
        build_budget_tokens=args.build_budget_tokens,
        answer_budget_tokens=args.answer_budget_tokens,
        v2_fact_extraction_max_tokens=args.v2_fact_extraction_max_tokens,
        v2_consolidation_max_tokens=args.v2_consolidation_max_tokens,
        v2_card_k=args.v2_card_k,
        v2_fact_k=args.v2_fact_k,
        v2_leaf_k=args.v2_leaf_k,
        v2_context_token_budget=args.v2_context_token_budget,
        v2_semantic_k=args.v2_semantic_k,
        v2_semantic_floor=args.v2_semantic_floor,
        v3_session_extraction_max_tokens=args.v3_session_extraction_max_tokens,
        v3_context_token_budget=args.v3_context_token_budget,
        v36_session_extraction_max_tokens=args.v36_session_extraction_max_tokens,
        v36_context_token_budget=args.v36_context_token_budget,
        v36_answer_hard_budget_tokens=args.v36_answer_hard_budget_tokens,
        v41_normal_context_target=args.v41_normal_context_target,
        v41_complex_context_target=args.v41_complex_context_target,
        v41_planner_prompt_max=args.v41_planner_prompt_max,
        v41_planner_output_max=args.v41_planner_output_max,
        v41_query_target_tokens=args.v41_query_target_tokens,
        v41_query_hard_limit_tokens=args.v41_query_hard_limit_tokens,
        v41_enable_planner=not args.disable_v41_planner,
        retrieval_only=args.retrieval_only,
    )
    aggregates = run_demo(config)
    for aggregate in aggregates:
        print(
            f"{aggregate.variant}: questions={aggregate.question_count} "
            f"llm_tokens={aggregate.total_deepseek_tokens} "
            f"calls={aggregate.deepseek_call_count}"
        )
    print(f"summary={config.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
