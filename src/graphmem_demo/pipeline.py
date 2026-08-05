from __future__ import annotations

import csv
import hashlib
import json
import queue
import re
import threading
import time
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .clients import (
    OpenAICompatibleClient,
    EmbeddingClient,
    LLMLinguaCompressor,
    LocalSummarizer,
    MockCompressor,
    MockDeepSeekClient,
    MockEmbeddingClient,
    MockLocalSummarizer,
    NoOpCompressor,
    cosine_similarity,
    rough_token_count,
)
from .data import build_leaf_nodes, group_by_session, load_longmemeval_cases
from .llm_root_edges import (
    build_llm_anchor_edges,
    llm_root_anchor_messages,
    parse_llm_root_anchors,
)
from .llm_leaf_edges import (
    build_session_leaf_edges,
    llm_session_leaf_edge_messages,
    parse_llm_leaf_edges,
)
from .fusion_retrieval import FusionRetrievalConfig, compute_fusion_scores
from .graph_retrieval import (
    GraphFirstConfig,
    GraphSearchConfig,
    graph_first_retrieve,
    graph_search_retrieve,
)
from .root_graph_edges import RootGraphEdgePolicy, build_root_graph
from .typed_retrieval import rank_roots_hybrid
from .models import (
    GRAPHMEM_V2_SCHEMA,
    AtomicFactNode,
    DeepSeekCallRecord,
    GraphEdge,
    LeafNode,
    QuestionCase,
    QuestionStats,
    RetrievedContext,
    RoutingCardNode,
    StateChain,
    SummaryNode,
    VariantStats,
)
from .retrieval_cues import proper_name_cues
from .hierarchical_v2 import (
    CONSOLIDATION_VERSION as V2_CONSOLIDATION_VERSION,
    PROMPT_VERSION as V2_PROMPT_VERSION,
    answer_messages as v2_answer_messages,
    apply_answer_constraint as apply_v2_answer_constraint,
    apply_consolidation as apply_v2_consolidation,
    build_graph_edges as build_v2_graph_edges,
    build_state_chains as build_v2_state_chains,
    consolidation_messages as v2_consolidation_messages,
    parse_session_extraction,
    prompt_hash as v2_prompt_hash,
    provider_token_estimate,
    expand_query as expand_v2_query,
    retrieve as retrieve_v2,
    session_extraction_messages as v2_session_extraction_messages,
    validate_provenance as validate_v2_provenance,
)
from .v3 import (
    GRAPHMEM_V3_SCHEMA,
    V3_BUILD_VERSION,
    V3_PROMPT_VERSION,
    V3_RETRIEVAL_VERSION,
    V3Index,
    answer_messages as v3_answer_messages,
    authoritative_catalog_answer as v3_authoritative_catalog_answer,
    build_hypergraph as build_v3_hypergraph,
    build_query_frame as build_v3_query_frame,
    build_turn_nodes as build_v3_turn_nodes,
    parse_session_extraction as parse_v3_session_extraction,
    retrieve as retrieve_v3,
    session_extraction_messages as v3_session_extraction_messages,
)
from .v3.build import clone_index as clone_v3_index
from .v3.build import prompt_hash as v3_prompt_hash
from .v3.runtime import build_index as build_v3_index
from .v3.query_planning import query_views as v3_query_views
from .v3.schema import index_from_dict as v3_index_from_dict
from .v36 import (
    GRAPHMEM_V36_SCHEMA, V36_BUILD_VERSION, V36_PROMPT_VERSION,
    V36Index, answer_messages as v36_answer_messages,
    build_query_ir as build_v36_query_ir, clone_index as clone_v36_index,
    prompt_hash as v36_prompt_hash, query_views as v36_query_views,
    retrieve as retrieve_v36,
)
from .v36.persistence import (
    persist_retrieval_index as persist_v36_retrieval_index,
    persist_vector_matrix as persist_v36_vector_matrix,
)
from .v36.runtime import build_index as build_v36_index
from .v36.schema import index_from_dict as v36_index_from_dict
from .v4 import (
    GRAPHMEM_V4_SCHEMA,
    V4_BUILD_VERSION,
    CapabilityViewV4,
    answer_messages as v4_answer_messages,
    build_capability_view as build_v4_capability_view,
    build_query_ir as build_v4_query_ir,
    capability_view_from_dict,
    query_views as v4_query_views,
    retrieve as retrieve_v4,
    validate_capability_view as validate_v4_capability_view,
)
from .v41 import (
    GRAPHMEM_V41_SCHEMA, V41_POLICY_VERSION, QueryPolicyV41, QuerySidecarV41,
    answer_messages as v41_answer_messages,
    build_query_plan as build_v41_query_plan,
    build_sidecar as build_v41_sidecar,
    parse_planner_result as parse_v41_planner_result,
    planner_messages as v41_planner_messages,
    persist_sidecar as persist_v41_sidecar,
    query_views as v41_query_views, retrieve as retrieve_v41,
    trim_latest_addition as trim_v41_latest_addition,
)
from .stats import (
    aggregate_variant_stats,
    build_question_stats,
    build_stats_payload,
    query_stats_payload,
)


@dataclass(frozen=True)
class VariantSpec:
    tree_mode: str
    compression: bool
    graph: bool
    fanout_k: int | None = None
    summary_max_tokens: int | None = None
    summary_schema: str = "minimal_memory_v1"
    build_leaf_text: str = "raw"
    retrieval_leaf_text: str = "raw"
    raw_question_types: tuple[str, ...] = ()
    hybrid_retrieval: bool = False
    local_summary: bool = False
    default_summarizer_model: str | None = None
    enhanced_retrieval: bool = False
    enhanced_qa: bool = False


VARIANT_SPECS = {
    "hierarchical_hybrid_graph_v4_1_query": VariantSpec(
        "hierarchical_hybrid_graph_v4_1_query", False, True,
        summary_schema="graphmem_v4_1_query", hybrid_retrieval=True,
        enhanced_retrieval=True, enhanced_qa=True,
    ),
    "hierarchical_hybrid_graph_v4_0": VariantSpec(
        "hierarchical_hybrid_graph_v4_0", False, True,
        summary_schema="graphmem_v4_0", hybrid_retrieval=True,
        enhanced_retrieval=True, enhanced_qa=True,
    ),
    "hierarchical_role_graph_v3_6": VariantSpec(
        "hierarchical_role_graph_v3_6", False, True,
        summary_schema="graphmem_v3_6", hybrid_retrieval=True,
        enhanced_retrieval=True, enhanced_qa=True,
    ),
    "hierarchical_hypergraph_v3": VariantSpec(
        "hierarchical_hypergraph_v3", False, True,
        summary_schema="graphmem_v3", hybrid_retrieval=True,
        enhanced_retrieval=True, enhanced_qa=True,
    ),
    "hierarchical_state_graph_v2": VariantSpec(
        "hierarchical_state_graph_v2", False, True,
        summary_schema="graphmem_v2", hybrid_retrieval=True,
        enhanced_retrieval=True, enhanced_qa=True,
    ),
    "raw_rag": VariantSpec("raw_rag", False, False),
    "summary_tree_k4_no_compress": VariantSpec("legacy_kway", False, False, fanout_k=4),
    "summary_tree_k4_graphmem": VariantSpec("legacy_kway", True, True, fanout_k=4),
    "direct_session_k16_no_compress": VariantSpec("direct_session", False, False, fanout_k=16),
    "direct_session_k16_graphmem": VariantSpec("direct_session", True, True, fanout_k=16),
    "direct_session_k16_compact_no_compress": VariantSpec(
        "direct_session",
        False,
        False,
        fanout_k=16,
        summary_schema="compact_memory_v2",
        build_leaf_text="raw",
        retrieval_leaf_text="raw",
        hybrid_retrieval=True,
    ),
    "direct_session_k16_compact_graphmem": VariantSpec(
        "direct_session",
        True,
        True,
        fanout_k=16,
        summary_schema="compact_memory_v2",
        build_leaf_text="raw",
        retrieval_leaf_text="raw",
        hybrid_retrieval=True,
    ),
    "qwen35_2b_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    "qwen35_08b_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-0.8B",
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    "qwen35_2b_summary_graphmem_no_retrieval_enhance": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=False,
        enhanced_qa=True,
    ),
    "qwen35_2b_summary_graphmem_no_qa_enhance": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=True,
        enhanced_qa=False,
    ),
    "single_llm_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_max_tokens=2048,
        summary_schema="compact_memory_v2",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        raw_question_types=("single-session-assistant", "single-session-preference"),
        hybrid_retrieval=True,
        local_summary=False,
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    # Keep old names for existing run directories and resume workflows.
    "summary_tree_no_compress": VariantSpec("legacy_kway", False, False),
    "token_efficient_graphmem": VariantSpec("legacy_kway", True, True),
}
V4_VARIANTS = frozenset({
    "hierarchical_hybrid_graph_v4_0",
    "hierarchical_hybrid_graph_v4_1_query",
})
ROLE_GRAPH_VARIANTS = frozenset({
    "hierarchical_role_graph_v3_6", *V4_VARIANTS,
})

def _is_v4_variant(variant: str) -> bool:
    return variant in V4_VARIANTS

def _is_v41_variant(variant: str) -> bool:
    return variant == "hierarchical_hybrid_graph_v4_1_query"

VARIANTS = set(VARIANT_SPECS)


@dataclass
class DemoConfig:
    data_path: Path
    output_dir: Path
    memory_cache_dir: Path | None = None
    persist_memory_artifacts: bool = True
    question_type: str = "multi-session"
    variants: tuple[str, ...] = (
        "direct_session_k16_compact_no_compress",
        "direct_session_k16_compact_graphmem",
    )
    deepseek_model: str | None = None
    deepseek_base_url: str | None = None
    llm_api_key_env: str = "SGAO_API_KEY"
    llm_request_profile: str = "openai"
    llm_timeout_sec: float = 180.0
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen3-Embedding-0.6B"
    tree_mode: str | None = None
    fanout_k: int = 16
    max_group_rough_tokens: int = 6000
    leaf_top_k: int = 14
    root_top_k: int = 4
    root_candidate_k: int = 8
    global_leaf_top_k: int = 24
    qa_summary_top_k: int = 4
    per_session_leaf_k: int = 2
    enable_coverage_rerank: bool = False
    coverage_rerank_lambda: float = 0.75
    coverage_rerank_pool_k: int = 80
    graph_neighbor_k: int = 2
    qa_context_token_budget: int = 18000
    qa_max_tokens: int = 4096
    compression_ratio: float = 0.5
    max_questions: int = 10
    question_workers: int = 2
    summary_workers: int = 32
    max_inflight_deepseek: int = 32
    summary_schema: str | None = None
    summarizer_kind: str = "auto"
    summarizer_base_url: str = "http://127.0.0.1:8003/v1"
    summarizer_model: str | None = None
    # Deprecated: summary jobs now use per-stage max token caps (raw_group/session/legacy).
    summary_token_budget: int = 320
    build_leaf_text: str = "auto"
    retrieval_leaf_text: str = "auto"
    compressor_chunk_rough_tokens: int = 384
    raw_group_summary_max_tokens: int = 2048
    session_summary_max_tokens: int = 2048
    legacy_internal_summary_max_tokens: int = 2048
    resume: bool = False
    mock_services: bool = False
    mock_llm: bool = False
    mock_embedding: bool = False
    mock_compressor: bool = False
    mock_summarizer: bool = False
    llmlingua_model: str | None = None
    llmlingua_device_map: str | None = None
    use_llmlingua2: bool = False
    enable_speaker_profiles: bool = False
    enable_speaker_neighbor_window: bool = False
    enable_speaker_retrieval_text: bool = False
    # Compatibility toggle: when enabled, datasets with explicit speaker labels
    # get larger retrieval budgets (leaf/global/per-session caps).
    enable_explicit_speaker_retrieval_boost: bool = True
    # When True with compact_memory_v2, session summary also emits per-leaf facts/keywords.
    enable_leaf_enrichment: bool = True
    # When True, SummaryNode.summary stores full child dialogue (lossless); LLM JSON is
    # metadata only (anchors / leaf enrichment), never the canonical summary body.
    enable_lossless_root_summary: bool = True
    enable_typed_root_edges: bool = False
    enable_multilevel_summary_retrieval: bool = False
    enable_llm_root_edges: bool = False
    llm_root_edge_max_tokens: int = 2048
    llm_root_edge_neighbors_per_relation: int = 2
    llm_root_edge_min_shared: int = 1
    llm_root_edge_anchor_limit: int = 8
    # Within-session leaf graph: deterministic turn neighbors + optional LLM semantic links.
    enable_llm_leaf_edges: bool = False
    enable_leaf_graph_expansion: bool = False
    llm_leaf_edge_max_tokens: int = 1024
    llm_leaf_edge_max_snippet_chars: int = 1024
    llm_leaf_edge_min_confidence: float = 0.8
    llm_leaf_edge_max_edges_per_leaf: int = 3
    llm_leaf_edge_max_edges_per_session: int = 16
    llm_leaf_edge_max_leaves_per_session: int = 48
    leaf_graph_neighbor_k: int = 2
    leaf_graph_expansion_budget: int = 4
    # HippoRAG-style graph search: embedding picks seeds, PPR walks edges (root/leaf).
    enable_graph_search: bool = False
    graph_search_seed_roots: int = 6
    graph_search_seed_leaves: int = 10
    graph_search_ppr_damping: float = 0.85
    graph_search_ppr_iterations: int = 25
    graph_search_embedding_blend: float = 0.1
    # When True, embedding only picks PPR seeds; PPR runs on the full leaf/root graph,
    # and leaves are selected by global PPR/blend score (no Phase-1 diversify budget lock).
    graph_search_seed_only: bool = True
    # Weak root↔leaf links. High values (e.g. 0.9) equalize PPR inside a session.
    graph_search_structural_root_leaf_weight: float = 0.1
    # Free-select: guarantee ≥1 leaf from each of the top-N graph-ranked sessions (0 = off).
    graph_search_session_coverage: int = 0
    graph_search_session_min_leaves: int = 3
    graph_search_max_sessions: int = 8
    graph_search_per_session_leaf_cap: int = 4
    graph_search_protect_leaves: bool = True
    # Graph-primary retrieval: PPR/graph drives selection; global embedding pool is backup.
    enable_graph_first_retrieval: bool = False
    graph_first_embedding_blend: float = 0.25
    graph_first_session_coverage: int = 2
    graph_first_candidate_pool_k: int = 80
    # Triple-pass retrieval fusion: semantic + BM25 keyword + entity overlap (RRF by default).
    enable_fusion_retrieval: bool = False
    fusion_method: str = "rrf"
    fusion_rrf_k: int = 60
    fusion_weight_semantic: float = 1.0
    fusion_weight_keyword: float = 1.0
    fusion_weight_entity: float = 1.0
    fusion_query_adaptive_weights: bool = True
    enable_typed_retrieval: bool = True
    typed_retrieval_embedding_blend: float = 0.55
    enable_protected_fusion: bool = True
    fusion_semantic_protect_k: int = 10
    # Query-shape-aware retrieval budget and dual-channel candidate merge.
    enable_query_type_retrieval_boost: bool = True
    list_question_extra_leaf_budget: int = 6
    temporal_question_extra_leaf_budget: int = 4
    enable_dual_channel_candidate_merge: bool = True
    dual_channel_structured_pool_k: int = 24
    # Iterative post-retrieval denoise: kick low-relevance leaves and backfill from tail.
    enable_iterative_leaf_denoise: bool = False
    iterative_leaf_denoise_max_rounds: int = 3
    iterative_leaf_denoise_max_kick_per_round: int = 5
    iterative_leaf_denoise_min_relevance_ratio: float = 0.35
    iterative_leaf_denoise_protect_top_k: int = 3
    iterative_leaf_denoise_keep_structured_top_k: int = 2
    # Typed root graph edges: additive high-confidence bridges, pruned for cross-topic noise.
    typed_root_neighbors_per_relation: int = 1
    typed_root_max_edges_per_root: int = 6
    typed_root_min_edge_score: float = 0.76
    typed_root_entity_min_shared_specific: int = 1
    typed_root_entity_min_shared_generic: int = 2
    typed_root_time_min_shared: int = 1
    typed_root_state_min_shared: int = 1
    typed_root_event_min_shared: int = 2
    typed_root_keyword_min_shared: int = 2
    typed_root_update_min_actions: int = 2
    typed_root_update_min_entities: int = 1
    typed_root_corpus_keyword_min_shared: int = 2
    typed_root_require_semantic_support: bool = True
    typed_root_semantic_support_min_cosine: float = 0.25
    typed_root_filter_generic_entities: bool = True
    # Stage-A reader: extract structured notes from retrieved evidence before QA.
    enable_answer_note_extraction: bool = False
    answer_note_max_tokens: int = 1024
    # Stage-B reader: when notes are available, answer from notes first.
    answer_use_notes_for_qa: bool = True
    answer_include_raw_context_with_notes: bool = False
    # Program-aided arithmetic (Level 2): on arithmetic-looking questions, ask the LLM for a
    # JSON "compute plan" (operation + operands it read from the evidence), execute it
    # deterministically in code, and feed the computed results back into the answer. This
    # keeps the LLM responsible for *what* to compute while code does the actual math.
    # Adds one extra LLM call per gated question. Opt-in (A/B with --enable-compute-plan).
    enable_compute_plan: bool = False
    # Force stricter reasoning/answer prompts regardless of variant defaults.
    force_enhanced_retrieval: bool = False
    force_enhanced_qa: bool = False
    reasoning_effort: str = "none"
    build_budget_tokens: int = 300_000
    answer_budget_tokens: int = 10_000
    v2_fact_extraction_max_tokens: int = 3072
    v2_consolidation_max_tokens: int = 3072
    v2_card_k: int = 6
    v2_fact_k: int = 14
    v2_leaf_k: int = 14
    v2_context_token_budget: int = 7600
    v2_semantic_k: int = 3
    v2_semantic_floor: float = 0.55
    v3_session_extraction_max_tokens: int = 3072
    v3_context_token_budget: int = 3600
    v36_session_extraction_max_tokens: int = 4096
    v36_llm_session_cap: int = 0
    v36_context_token_budget: int = 8000
    v36_answer_hard_budget_tokens: int = 10500
    v41_normal_context_target: int = 8400
    v41_complex_context_target: int = 9200
    v41_planner_prompt_max: int = 700
    v41_planner_output_max: int = 256
    v41_query_target_tokens: int = 10000
    v41_query_hard_limit_tokens: int = 13000
    v41_enable_planner: bool = True
    # Full-benchmark runs may retain an already-generated over-budget answer so
    # it can be reported instead of aborting/retrying the entire shard. Budget
    # pass/fail metrics remain unchanged; strict enforcement is the default.
    v41_record_query_budget_overflow: bool = False
    # Persist the V3 index and retrieval ledger without spending tokens on the
    # built-in base answer. The graph navigator can then be the sole reader.
    retrieval_only: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.variants) - VARIANTS
        if unknown:
            raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
        if self.fanout_k < 2:
            raise ValueError("fanout_k must be at least 2")
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if self.question_workers < 1:
            raise ValueError("question_workers must be at least 1")
        if self.summary_workers < 0 or self.max_inflight_deepseek < 0:
            raise ValueError("summary_workers and max_inflight_deepseek cannot be negative")
        if self.llm_timeout_sec <= 0:
            raise ValueError("llm_timeout_sec must be positive")
        if self.tree_mode is not None and self.tree_mode not in {"legacy_kway", "direct_session", "hierarchical_state_graph_v2", "hierarchical_hypergraph_v3", "hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
            raise ValueError("tree_mode must be legacy_kway, direct_session, hierarchical_state_graph_v2, or hierarchical_hypergraph_v3, or hierarchical_role_graph_v3_6, or hierarchical_hybrid_graph_v4_0, or hierarchical_hybrid_graph_v4_1_query")
        if self.summary_schema not in {
            None,
            "minimal_memory_v1",
            "compact_memory_v2",
            "multilingual_memory_v1",
            "graphmem_v2",
            "graphmem_v3",
            "graphmem_v3_6",
            "graphmem_v4_0",
            "graphmem_v4_1_query",
        }:
            raise ValueError(
                "summary_schema must be a supported memory schema through graphmem_v4_1_query"
            )
        if self.summarizer_kind not in {"auto", "none", "llmlingua2", "qwen_local"}:
            raise ValueError("summarizer_kind must be auto, none, llmlingua2, or qwen_local")
        if self.qa_context_token_budget < 1000:
            raise ValueError("qa_context_token_budget must be at least 1000")
        if self.qa_max_tokens < 128:
            raise ValueError("qa_max_tokens must be at least 128")
        if self.llm_root_edge_max_tokens < 64:
            raise ValueError("llm_root_edge_max_tokens must be at least 64")
        if self.llm_root_edge_neighbors_per_relation < 1:
            raise ValueError("llm_root_edge_neighbors_per_relation must be at least 1")
        if self.llm_root_edge_min_shared < 1:
            raise ValueError("llm_root_edge_min_shared must be at least 1")
        if self.llm_root_edge_anchor_limit < 1:
            raise ValueError("llm_root_edge_anchor_limit must be at least 1")
        if self.llm_leaf_edge_max_tokens < 64:
            raise ValueError("llm_leaf_edge_max_tokens must be at least 64")
        if self.llm_leaf_edge_max_snippet_chars < 64:
            raise ValueError("llm_leaf_edge_max_snippet_chars must be at least 64")
        if not 0.0 <= self.llm_leaf_edge_min_confidence <= 1.0:
            raise ValueError("llm_leaf_edge_min_confidence must be in [0, 1]")
        if self.llm_leaf_edge_max_edges_per_leaf < 1:
            raise ValueError("llm_leaf_edge_max_edges_per_leaf must be at least 1")
        if self.llm_leaf_edge_max_edges_per_session < 1:
            raise ValueError("llm_leaf_edge_max_edges_per_session must be at least 1")
        if self.llm_leaf_edge_max_leaves_per_session < 2:
            raise ValueError("llm_leaf_edge_max_leaves_per_session must be at least 2")
        if self.leaf_graph_neighbor_k < 1:
            raise ValueError("leaf_graph_neighbor_k must be at least 1")
        if self.leaf_graph_expansion_budget < 0:
            raise ValueError("leaf_graph_expansion_budget cannot be negative")
        if self.graph_search_seed_roots < 1 or self.graph_search_seed_leaves < 1:
            raise ValueError("graph_search_seed_roots and graph_search_seed_leaves must be at least 1")
        if not 0.0 < self.graph_search_ppr_damping < 1.0:
            raise ValueError("graph_search_ppr_damping must be in (0, 1)")
        if self.graph_search_ppr_iterations < 1:
            raise ValueError("graph_search_ppr_iterations must be at least 1")
        if not 0.0 <= self.graph_search_embedding_blend <= 1.0:
            raise ValueError("graph_search_embedding_blend must be in [0, 1]")
        if self.graph_search_session_min_leaves < 1:
            raise ValueError("graph_search_session_min_leaves must be at least 1")
        if self.graph_search_max_sessions < 1:
            raise ValueError("graph_search_max_sessions must be at least 1")
        if self.graph_search_per_session_leaf_cap < 1:
            raise ValueError("graph_search_per_session_leaf_cap must be at least 1")
        if self.graph_search_structural_root_leaf_weight < 0.0:
            raise ValueError("graph_search_structural_root_leaf_weight cannot be negative")
        if self.graph_search_session_coverage < 0:
            raise ValueError("graph_search_session_coverage cannot be negative")
        if not 0.0 <= self.graph_first_embedding_blend <= 1.0:
            raise ValueError("graph_first_embedding_blend must be in [0, 1]")
        if self.graph_first_session_coverage < 0:
            raise ValueError("graph_first_session_coverage cannot be negative")
        if self.graph_first_candidate_pool_k < 1:
            raise ValueError("graph_first_candidate_pool_k must be at least 1")
        if not 0.0 <= self.typed_retrieval_embedding_blend <= 1.0:
            raise ValueError("typed_retrieval_embedding_blend must be in [0, 1]")
        if self.fusion_semantic_protect_k < 0:
            raise ValueError("fusion_semantic_protect_k cannot be negative")
        if self.list_question_extra_leaf_budget < 0:
            raise ValueError("list_question_extra_leaf_budget cannot be negative")
        if self.temporal_question_extra_leaf_budget < 0:
            raise ValueError("temporal_question_extra_leaf_budget cannot be negative")
        if self.dual_channel_structured_pool_k < 1:
            raise ValueError("dual_channel_structured_pool_k must be at least 1")
        if self.iterative_leaf_denoise_max_rounds < 1:
            raise ValueError("iterative_leaf_denoise_max_rounds must be at least 1")
        if self.iterative_leaf_denoise_max_kick_per_round < 1:
            raise ValueError("iterative_leaf_denoise_max_kick_per_round must be at least 1")
        if not 0.0 <= self.iterative_leaf_denoise_min_relevance_ratio <= 1.0:
            raise ValueError("iterative_leaf_denoise_min_relevance_ratio must be in [0, 1]")
        if self.iterative_leaf_denoise_protect_top_k < 0:
            raise ValueError("iterative_leaf_denoise_protect_top_k cannot be negative")
        if self.iterative_leaf_denoise_keep_structured_top_k < 0:
            raise ValueError("iterative_leaf_denoise_keep_structured_top_k cannot be negative")
        if self.answer_note_max_tokens < 128:
            raise ValueError("answer_note_max_tokens must be at least 128")
        if self.typed_root_neighbors_per_relation < 1:
            raise ValueError("typed_root_neighbors_per_relation must be at least 1")
        if self.typed_root_max_edges_per_root < 1:
            raise ValueError("typed_root_max_edges_per_root must be at least 1")
        if not 0.0 <= self.typed_root_min_edge_score <= 1.0:
            raise ValueError("typed_root_min_edge_score must be in [0, 1]")
        if not 0.0 <= self.typed_root_semantic_support_min_cosine <= 1.0:
            raise ValueError("typed_root_semantic_support_min_cosine must be in [0, 1]")
        if self.fusion_method not in {"rrf", "weighted"}:
            raise ValueError("fusion_method must be rrf or weighted")
        if self.fusion_rrf_k < 1:
            raise ValueError("fusion_rrf_k must be at least 1")
        for weight_name in (
            "fusion_weight_semantic",
            "fusion_weight_keyword",
            "fusion_weight_entity",
        ):
            if getattr(self, weight_name) < 0.0:
                raise ValueError(f"{weight_name} cannot be negative")
        for field_name in ("build_leaf_text", "retrieval_leaf_text"):
            if getattr(self, field_name) not in {"auto", "raw", "user_only"}:
                raise ValueError(f"{field_name} must be auto, raw, or user_only")
        if min(
            self.root_candidate_k,
            self.qa_summary_top_k,
            self.per_session_leaf_k,
        ) < 1:
            raise ValueError("V2 retrieval k values must be at least 1")
        # global_leaf_top_k == 0 is a valid "disable the global leaf candidate pool" setting.
        if self.global_leaf_top_k < 0:
            raise ValueError("global_leaf_top_k cannot be negative")
        if not 0.0 <= self.coverage_rerank_lambda <= 1.0:
            raise ValueError("coverage_rerank_lambda must be in [0, 1]")
        if self.coverage_rerank_pool_k < 2:
            raise ValueError("coverage_rerank_pool_k must be at least 2")

        if self.v36_session_extraction_max_tokens < 256:
            raise ValueError("v36_session_extraction_max_tokens must be at least 256")
        if self.v36_llm_session_cap < 0:
            raise ValueError("v36_llm_session_cap cannot be negative")
        if self.v36_context_token_budget < 1000:
            raise ValueError("v36_context_token_budget must be at least 1000")
        if self.v36_answer_hard_budget_tokens < self.answer_budget_tokens:
            raise ValueError("V3.6 hard answer budget cannot be below target budget")

        if self.reasoning_effort != "none":
            raise ValueError("reasoning_effort must be none")
        if self.llm_request_profile not in {"deepseek", "openai", "qwen", "omit"}:
            raise ValueError("llm_request_profile must be deepseek, openai, qwen, or omit")
        if not self.llm_api_key_env:
            raise ValueError("llm_api_key_env cannot be empty")
        if self.build_budget_tokens < 1 or self.answer_budget_tokens < 1:
            raise ValueError("token budgets must be positive")
        if not 0.0 <= self.v2_semantic_floor <= 1.0 or self.v2_semantic_k < 1:
            raise ValueError("invalid V2 semantic graph settings")
        if min(self.v2_card_k, self.v2_fact_k, self.v2_leaf_k) < 1:
            raise ValueError("V2 retrieval limits must be positive")
        if self.v2_context_token_budget < 1000:
            raise ValueError("v2_context_token_budget must be at least 1000")
        if self.v3_session_extraction_max_tokens < 256:
            raise ValueError("v3_session_extraction_max_tokens must be at least 256")
        if self.v3_context_token_budget < 1000:
            raise ValueError("v3_context_token_budget must be at least 1000")

    def use_mock_llm(self) -> bool:
        return self.mock_services or self.mock_llm

    def use_mock_embedding(self) -> bool:
        return self.mock_services or self.mock_embedding

    def use_mock_compressor(self) -> bool:
        return self.mock_services or self.mock_compressor

    def use_mock_summarizer(self) -> bool:
        return self.mock_services or self.mock_summarizer


@dataclass
class CaseRun:
    leaves: list[LeafNode]
    summaries: list[SummaryNode]
    edges: list[GraphEdge]
    retrieval: RetrievedContext
    answer: str
    llm_records: list[DeepSeekCallRecord]
    stats: QuestionStats
    answer_notes: list[dict[str, str]] = field(default_factory=list)
    answer_note_parse_error: str | None = None
    answer_used_notes: bool = False
    facts: list[AtomicFactNode] = field(default_factory=list)
    routing_cards: list[RoutingCardNode] = field(default_factory=list)
    state_chains: list[StateChain] = field(default_factory=list)
    index_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    v3_index: V3Index | None = None
    v36_index: V36Index | None = None
    v4_capability_view: CapabilityViewV4 | None = None
    v41_sidecar: QuerySidecarV41 | None = None


@dataclass
class MemoryBuild:
    leaves: list[LeafNode]
    summaries: list[SummaryNode]
    roots: list[SummaryNode]
    edges: list[GraphEdge]
    llm_records: list[DeepSeekCallRecord]
    metrics: "BuildMetrics"
    build_latency_sec: float
    facts: list[AtomicFactNode] = field(default_factory=list)
    routing_cards: list[RoutingCardNode] = field(default_factory=list)
    state_chains: list[StateChain] = field(default_factory=list)
    v3_index: V3Index | None = None
    v36_index: V36Index | None = None
    v4_capability_view: CapabilityViewV4 | None = None
    v41_sidecar: QuerySidecarV41 | None = None


@dataclass
class BuildMetrics:
    ready_job_counts: list[dict[str, Any]] = field(default_factory=list)
    summary_parse_error_count: int = 0
    summary_truncation_count: int = 0
    peak_inflight_deepseek: int = 0
    index_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    _active_deepseek: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin_call(self) -> None:
        with self._lock:
            self._active_deepseek += 1
            self.peak_inflight_deepseek = max(
                self.peak_inflight_deepseek, self._active_deepseek
            )

    def end_call(self) -> None:
        with self._lock:
            self._active_deepseek -= 1


class InflightLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(limit) if limit else None

    @contextmanager
    def track(self, metrics: BuildMetrics):
        if self._semaphore is not None:
            self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        metrics.begin_call()
        try:
            yield
        finally:
            metrics.end_call()
            with self._lock:
                self._active -= 1
            if self._semaphore is not None:
                self._semaphore.release()


@dataclass(frozen=True)
class SummaryJob:
    session_id: str
    session_date: str | None
    children: list[LeafNode | SummaryNode]
    stage: str
    level: int
    group_number: int
    summary_mode: str
    max_tokens: int


def run_demo(
    config: DemoConfig,
    *,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> list[VariantStats]:
    cases = load_longmemeval_cases(
        config.data_path, question_type=config.question_type, max_questions=config.max_questions
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    has_injected_services = any(
        service is not None for service in (llm, embedder, compressor, summarizer)
    )

    aggregates: list[VariantStats] = []
    for variant in config.variants:
        variant_llm, variant_embedder, variant_compressor, variant_summarizer = _complete_services(
            config, variant, llm, embedder, compressor, summarizer
        )
        variant_started = time.perf_counter()
        limiter = InflightLimiter(config.max_inflight_deepseek)
        variant_dir = config.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        if not config.resume:
            _reset_jsonl_outputs(variant_dir)
        stats = _read_question_stats(variant_dir / "question_stats.jsonl") if config.resume else []
        completed = {item.question_id for item in stats}

        pending_cases = [case for case in cases if case.question_id not in completed]
        if _has_memory_cache_keys(pending_cases):
            for case, run, embedding_records, compression_records in _run_cases_with_memory_cache(
                config,
                pending_cases,
                variant,
                variant_dir,
                limiter,
                allow_memory_cache_read=(
                    config.resume or config.memory_cache_dir is not None
                ),
                llm=llm,
                embedder=embedder,
                compressor=compressor,
                summarizer=summarizer,
            ):
                stats.append(run.stats)
                _write_case_outputs(
                    variant_dir, case, run, embedding_records, compression_records,
                    config.persist_memory_artifacts,
                )
                _print_case_progress(run.stats)
        elif config.question_workers > 1 and not has_injected_services:
            with ThreadPoolExecutor(max_workers=config.question_workers) as executor:
                futures = {
                    executor.submit(
                        _run_case_with_fresh_services, config, case, variant, limiter
                    ): case
                    for case in pending_cases
                }
                for future in as_completed(futures):
                    case = futures[future]
                    run, embedding_records, compression_records = future.result()
                    stats.append(run.stats)
                    _write_case_outputs(
                        variant_dir, case, run, embedding_records, compression_records,
                        config.persist_memory_artifacts,
                    )
                    _print_case_progress(run.stats)
        else:
            for case in pending_cases:
                embedding_start = len(variant_embedder.records)
                compression_start = len(variant_compressor.records)
                summarizer_start = len(variant_summarizer.records)
                run = run_case(
                    config,
                    case,
                    variant,
                    variant_llm,
                    variant_embedder,
                    variant_compressor,
                    limiter,
                    summarizer=variant_summarizer,
                )
                stats.append(run.stats)
                _write_case_outputs(
                    variant_dir,
                    case,
                    run,
                    variant_embedder.records[embedding_start:],
                    [
                        *variant_compressor.records[compression_start:],
                        *variant_summarizer.records[summarizer_start:],
                    ],
                    config.persist_memory_artifacts,
                )
                _print_case_progress(run.stats)

        aggregate = aggregate_variant_stats(stats, variant)
        aggregate.metadata.update(
            {
                "question_workers": config.question_workers,
                "summary_workers": config.summary_workers,
                "max_inflight_deepseek": config.max_inflight_deepseek,
                "peak_inflight_deepseek": limiter.peak,
                "summary_schema": config.summary_schema or VARIANT_SPECS[variant].summary_schema,
                "report_run_wall_time_sec": time.perf_counter() - variant_started,
            }
        )
        aggregates.append(aggregate)
        stage_totals = _deepseek_stage_totals(_read_jsonl(variant_dir / "llm_calls.jsonl"))
        local_summary_totals = _local_summary_stage_totals(
            _read_jsonl(variant_dir / "compression_stats.jsonl")
        )
        build_payload = build_stats_payload(stats, aggregate)
        build_payload["deepseek_token_by_stage"] = stage_totals
        build_payload["local_summarizer_by_stage"] = local_summary_totals
        query_payload = query_stats_payload(stats, aggregate)
        query_payload["deepseek_token_by_stage"] = stage_totals
        query_payload["local_summarizer_by_stage"] = local_summary_totals
        _write_json(variant_dir / "build_stats.json", build_payload)
        _write_json(variant_dir / "query_stats.json", query_payload)
        if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
            _write_json(
                variant_dir / "index_diagnostics.json",
                _v36_index_diagnostics_payload(
                    _read_jsonl(variant_dir / "index_diagnostics.jsonl")
                ),
            )
        if variant in {
            "direct_session_k16_compact_graphmem",
            "qwen35_2b_summary_graphmem",
            "qwen35_08b_summary_graphmem",
            "qwen35_2b_summary_graphmem_no_retrieval_enhance",
            "qwen35_2b_summary_graphmem_no_qa_enhance",
        }:
            _write_manual_eval_template(variant_dir)

    _write_summary(config.output_dir, aggregates)
    return aggregates


def _complete_services(
    config: DemoConfig,
    variant: str,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    spec = _variant_spec(config, variant)
    use_llmlingua2 = (
        config.use_llmlingua2
        or config.summarizer_kind == "llmlingua2"
        or (config.summarizer_kind == "auto" and spec.compression)
    )
    use_compressor = spec.compression and config.summarizer_kind != "none" and not _uses_local_summary(config, spec)
    summarizer_model = config.summarizer_model or spec.default_summarizer_model or "Qwen/Qwen3.5-2B"
    return (
        llm
        or (MockDeepSeekClient() if config.use_mock_llm() else OpenAICompatibleClient(
            model=config.deepseek_model,
            base_url=config.deepseek_base_url,
            api_key_env=config.llm_api_key_env,
            request_profile=config.llm_request_profile,
            timeout_sec=config.llm_timeout_sec,
        )),
        embedder
        or (
            MockEmbeddingClient()
            if config.use_mock_embedding()
            else EmbeddingClient(config.embedding_base_url, config.embedding_model)
        ),
        compressor
        or (
            MockCompressor(config.compression_ratio)
            if config.use_mock_compressor()
            else NoOpCompressor()
            if not use_compressor
            else LLMLinguaCompressor(
                ratio=config.compression_ratio,
                model_name=config.llmlingua_model,
                device_map=config.llmlingua_device_map,
                use_llmlingua2=use_llmlingua2,
            )
        ),
        summarizer
        or (
            MockLocalSummarizer(summarizer_model)
            if config.use_mock_summarizer()
            else LocalSummarizer(config.summarizer_base_url, summarizer_model)
        ),
    )


def _run_case_with_fresh_services(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
) -> tuple[CaseRun, list[Any], list[Any]]:
    return _run_case_with_services(config, case, variant, limiter)


def _record_start(service: Any) -> int:
    records = getattr(service, "records", None)
    return len(records) if records is not None else 0


def _records_since(service: Any, start: int) -> list[Any]:
    records = getattr(service, "records", None)
    if records is None:
        return []
    return list(records[start:])


def _close_owned_services(*services: tuple[Any, bool]) -> None:
    """Close HTTP pools created by this worker, never caller-injected services."""
    closed: set[int] = set()
    for service, owned in services:
        if not owned or service is None or id(service) in closed:
            continue
        close = getattr(getattr(service, "client", None), "close", None)
        if callable(close):
            close()
            closed.add(id(service))


def _memory_cache_path(
    config: DemoConfig,
    variant_dir: Path,
    case: QuestionCase,
    variant: str,
) -> Path:
    if not case.memory_cache_key:
        raise ValueError("memory_cache_key is required for memory cache path")
    key = _safe_cache_part(case.memory_cache_key)
    fingerprint = _memory_cache_fingerprint(config, case, variant)
    cache_variant = (
        "hierarchical_hybrid_graph_v4_0"
        if _is_v41_variant(variant) else variant
    )
    if config.memory_cache_dir is not None:
        return config.memory_cache_dir / cache_variant / f"{key}-{fingerprint[:16]}.json"
    return variant_dir / "memory_cache" / f"{key}-{fingerprint[:16]}.json"


def _v41_compatible_cache_candidates(
    expected_path: Path, case: QuestionCase,
) -> list[Path]:
    """Find legacy V4 build caches for a query-only V4.1 policy.

    Query policy edits must not force an LLM graph rebuild.  The fallback is
    intentionally narrow: the explicit memory key, V4 schema/version, and a
    structural V3.6 index must all match, and callers only accept one valid
    candidate.  Other variants retain exact fingerprint semantics.
    """
    key = _safe_cache_part(case.memory_cache_key or "")
    if not key or not expected_path.parent.exists():
        return []
    candidates: list[Path] = []
    for path in sorted(expected_path.parent.glob(f"{key}-*.json")):
        if path == expected_path:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (
            payload.get("version") == 6
            and payload.get("schema_version") == GRAPHMEM_V4_SCHEMA
            and payload.get("memory_cache_key") == case.memory_cache_key
            and isinstance(payload.get("v36_index"), dict)
        ):
            candidates.append(path)
    return candidates


def _safe_cache_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe[:96] or "memory"


def _memory_cache_fingerprint(config: DemoConfig, case: QuestionCase, variant: str) -> str:
    spec = _variant_spec(config, variant)
    data_payload = {
        "haystack_session_ids": case.haystack_session_ids,
        "haystack_dates": case.haystack_dates,
        "haystack_sessions": case.haystack_sessions,
    }
    data_hash = hashlib.sha256(
        json.dumps(data_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
        payload = {
            "version": (5 if _is_v4_variant(variant) else 4), "variant": ("hierarchical_hybrid_graph_v4_0" if _is_v4_variant(variant) else variant),
            "schema_version": (GRAPHMEM_V4_SCHEMA if _is_v4_variant(variant) else GRAPHMEM_V36_SCHEMA),
            "v4_build_version": (V4_BUILD_VERSION if _is_v4_variant(variant) else None),
            "v36_prompt_hash": v36_prompt_hash(),
            "v36_prompt_version": V36_PROMPT_VERSION,
            "v36_build_version": V36_BUILD_VERSION,
            "v36_session_extraction_max_tokens": config.v36_session_extraction_max_tokens,
            "v36_llm_session_cap": config.v36_llm_session_cap,
            "deepseek_model": config.deepseek_model,
            "embedding_model": config.embedding_model,
            "data_hash": data_hash,
            "llm_base_url": config.deepseek_base_url.rstrip("/") if config.deepseek_base_url else None,
            "llm_request_profile": config.llm_request_profile,
            "llm_api_key_env": config.llm_api_key_env,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    if variant == "hierarchical_hypergraph_v3":
        payload = {
            "version": 3, "variant": variant, "schema_version": GRAPHMEM_V3_SCHEMA,
            "v3_prompt_hash": v3_prompt_hash(), "v3_prompt_version": V3_PROMPT_VERSION,
            "v3_build_version": V3_BUILD_VERSION,
            "v3_session_extraction_max_tokens": config.v3_session_extraction_max_tokens,
            "deepseek_model": config.deepseek_model, "embedding_model": config.embedding_model,
            "data_hash": data_hash,
        }
        payload.update({
            "llm_base_url": (
                config.deepseek_base_url.rstrip("/")
                if config.deepseek_base_url else None
            ),
            "llm_request_profile": config.llm_request_profile,
            "llm_api_key_env": config.llm_api_key_env,
        })
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    if variant == "hierarchical_state_graph_v2":
        payload = {
            "version": 3, "variant": variant, "schema_version": GRAPHMEM_V2_SCHEMA,
            "v2_prompt_hash": v2_prompt_hash(), "v2_prompt_version": V2_PROMPT_VERSION,
            "v2_consolidation_version": V2_CONSOLIDATION_VERSION,
            "v2_fact_extraction_max_tokens": config.v2_fact_extraction_max_tokens,
            "v2_consolidation_max_tokens": config.v2_consolidation_max_tokens,
            "v2_semantic_k": config.v2_semantic_k, "v2_semantic_floor": config.v2_semantic_floor,
            "deepseek_model": config.deepseek_model, "embedding_model": config.embedding_model,
            "data_hash": data_hash,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    payload = {
        "version": 1,
        "variant": variant,
        "tree_mode": spec.tree_mode,
        "compression": spec.compression,
        "graph": spec.graph,
        "fanout_k": spec.fanout_k or config.fanout_k,
        "summary_schema": _summary_schema(config, spec),
        "summary_max_tokens": _summary_max_tokens(config, spec),
        "summarizer_kind": config.summarizer_kind,
        "summarizer_model": config.summarizer_model or spec.default_summarizer_model,
        "uses_local_summary": _uses_local_summary(config, spec),
        "deepseek_model": config.deepseek_model,
        "embedding_model": config.embedding_model,
        "build_leaf_text": _effective_leaf_text_mode(
            config.build_leaf_text, spec, case, phase="build"
        ),
        "skip_compression_for_raw_build": True,
        "retrieval_leaf_text": _effective_leaf_text_mode(
            config.retrieval_leaf_text, spec, case, phase="retrieval"
        ),
        "max_group_rough_tokens": config.max_group_rough_tokens,
        "raw_group_summary_max_tokens": config.raw_group_summary_max_tokens,
        "session_summary_max_tokens": config.session_summary_max_tokens,
        "legacy_internal_summary_max_tokens": config.legacy_internal_summary_max_tokens,
        "compression_ratio": config.compression_ratio,
        "compressor_chunk_rough_tokens": config.compressor_chunk_rough_tokens,
        "llmlingua_model": config.llmlingua_model,
        "use_llmlingua2": config.use_llmlingua2,
        "enable_speaker_profiles": config.enable_speaker_profiles,
        "enable_speaker_retrieval_text": config.enable_speaker_retrieval_text,
        "enable_typed_root_edges": config.enable_typed_root_edges,
        "enable_multilevel_summary_retrieval": config.enable_multilevel_summary_retrieval,
        "enable_llm_root_edges": config.enable_llm_root_edges,
        "llm_root_edge_max_tokens": config.llm_root_edge_max_tokens,
        "llm_root_edge_neighbors_per_relation": config.llm_root_edge_neighbors_per_relation,
        "llm_root_edge_min_shared": config.llm_root_edge_min_shared,
        "llm_root_edge_anchor_limit": config.llm_root_edge_anchor_limit,
        "enable_llm_leaf_edges": config.enable_llm_leaf_edges,
        "enable_leaf_graph_expansion": config.enable_leaf_graph_expansion,
        "llm_leaf_edge_max_snippet_chars": config.llm_leaf_edge_max_snippet_chars,
        "llm_leaf_edge_min_confidence": config.llm_leaf_edge_min_confidence,
        "llm_leaf_edge_max_edges_per_leaf": config.llm_leaf_edge_max_edges_per_leaf,
        "llm_leaf_edge_max_edges_per_session": config.llm_leaf_edge_max_edges_per_session,
        "leaf_graph_neighbor_k": config.leaf_graph_neighbor_k,
        "leaf_graph_expansion_budget": config.leaf_graph_expansion_budget,
        "enable_graph_search": config.enable_graph_search,
        "enable_graph_first_retrieval": config.enable_graph_first_retrieval,
        "enable_fusion_retrieval": config.enable_fusion_retrieval,
        "enable_typed_retrieval": config.enable_typed_retrieval,
        "enable_answer_note_extraction": config.enable_answer_note_extraction,
        "answer_note_max_tokens": config.answer_note_max_tokens,
        "answer_use_notes_for_qa": config.answer_use_notes_for_qa,
        "answer_include_raw_context_with_notes": config.answer_include_raw_context_with_notes,
        "typed_retrieval_embedding_blend": config.typed_retrieval_embedding_blend,
        "enable_protected_fusion": config.enable_protected_fusion,
        "fusion_semantic_protect_k": config.fusion_semantic_protect_k,
        "fusion_method": config.fusion_method,
        "graph_search_session_min_leaves": config.graph_search_session_min_leaves,
        "graph_search_max_sessions": config.graph_search_max_sessions,
        "graph_neighbor_k": config.graph_neighbor_k,
        "leaf_retrieval_text_version": 3,
        "temporal_normalization_version": 1,
        "enable_leaf_enrichment": config.enable_leaf_enrichment,
        "enable_lossless_root_summary": config.enable_lossless_root_summary,
        "summary_retrieval_text_version": 5,
        "summary_anchor_terms_version": 4,
        "keyword_edge_version": 4,
        "root_graph_edge_policy_version": 1,
        "data_hash": data_hash,
    }
    if variant == "hierarchical_state_graph_v2":
        payload.update({
            "version": 2,
            "schema_version": GRAPHMEM_V2_SCHEMA,
            "v2_prompt_hash": v2_prompt_hash(),
            "v2_prompt_version": V2_PROMPT_VERSION,
            "v2_consolidation_version": V2_CONSOLIDATION_VERSION,
            "v2_fact_extraction_max_tokens": config.v2_fact_extraction_max_tokens,
            "v2_consolidation_max_tokens": config.v2_consolidation_max_tokens,
            "v2_semantic_k": config.v2_semantic_k,
            "v2_semantic_floor": config.v2_semantic_floor,
        })
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_memory_cache(
    path: Path,
    memory: MemoryBuild,
    case: QuestionCase,
    variant: str,
    config: DemoConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v36_payload = None
    v36_vector_cache = None
    if memory.v36_index is not None:
        # Embeddings dominate cache JSON size and float-to-decimal serialization
        # time. Persist them as float32 rows and keep the structural JSON small.
        # Nodes are private to this build until the cache write returns, so the
        # temporary clearing below is safe and avoids asdict deep-copying vectors.
        vector_directory = path.with_suffix(path.suffix + ".vectors")
        persist_v36_vector_matrix(vector_directory, memory.v36_index)
        searchable_nodes = [
            *memory.v36_index.turns, *memory.v36_index.frames,
            *memory.v36_index.routing_cards,
            *memory.v36_index.evidence_groups,
        ]
        saved_embeddings = [node.embedding for node in searchable_nodes]
        try:
            for node in searchable_nodes:
                node.embedding = None
            v36_payload = asdict(memory.v36_index)
        finally:
            for node, embedding in zip(searchable_nodes, saved_embeddings):
                node.embedding = embedding
        v36_vector_cache = vector_directory.name
    payload = {
        "version": 6 if _is_v4_variant(variant) else (5 if variant == "hierarchical_role_graph_v3_6" else (3 if variant == "hierarchical_hypergraph_v3" else (2 if variant == "hierarchical_state_graph_v2" else 1))),
        "schema_version": GRAPHMEM_V4_SCHEMA if _is_v4_variant(variant) else (GRAPHMEM_V36_SCHEMA if variant == "hierarchical_role_graph_v3_6" else (GRAPHMEM_V3_SCHEMA if variant == "hierarchical_hypergraph_v3" else (GRAPHMEM_V2_SCHEMA if variant == "hierarchical_state_graph_v2" else "graphmem_v1"))),
        "memory_cache_key": case.memory_cache_key,
        "fingerprint": _memory_cache_fingerprint(config, case, variant),
        "source_question_id": case.question_id,
        "leaves": [asdict(leaf) for leaf in memory.leaves],
        "summaries": [asdict(summary) for summary in memory.summaries],
        "facts": [asdict(fact) for fact in memory.facts],
        "routing_cards": [asdict(card) for card in memory.routing_cards],
        "state_chains": [asdict(chain) for chain in memory.state_chains],
        "v3_index": asdict(memory.v3_index) if memory.v3_index is not None else None,
        "v36_index": v36_payload,
        "v4_capability_view": (
            asdict(memory.v4_capability_view)
            if memory.v4_capability_view is not None else None
        ),
        "v36_vector_cache": v36_vector_cache,
        "root_ids": [root.node_id for root in memory.roots],
        "edges": [asdict(edge) for edge in memory.edges],
        "llm_records": [asdict(record) for record in memory.llm_records],
        "metrics": {
            "ready_job_counts": memory.metrics.ready_job_counts,
            "summary_parse_error_count": memory.metrics.summary_parse_error_count,
            "summary_truncation_count": memory.metrics.summary_truncation_count,
            "peak_inflight_deepseek": memory.metrics.peak_inflight_deepseek,
            "index_diagnostics": memory.metrics.index_diagnostics,
        },
        "build_latency_sec": memory.build_latency_sec,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_memory_cache(path: Path) -> MemoryBuild | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") not in {1, 2, 3, 4, 5, 6}:
            return None
        leaves = [LeafNode(**row) for row in payload.get("leaves", [])]
        summaries = [SummaryNode(**row) for row in payload.get("summaries", [])]
        facts = [AtomicFactNode(**row) for row in payload.get("facts", [])]
        routing_cards = [RoutingCardNode(**row) for row in payload.get("routing_cards", [])]
        state_chains = [StateChain(**row) for row in payload.get("state_chains", [])]
        v3_payload = payload.get("v3_index")
        v3_index = v3_index_from_dict(v3_payload) if isinstance(v3_payload, dict) else None
        v36_payload = payload.get("v36_index")
        v36_index = v36_index_from_dict(v36_payload) if isinstance(v36_payload, dict) else None
        v4_view_payload = payload.get("v4_capability_view")
        v4_capability_view = (
            capability_view_from_dict(v4_view_payload)
            if isinstance(v4_view_payload, dict) else None
        )
        vector_cache_name = payload.get("v36_vector_cache")
        if v36_index is not None and isinstance(vector_cache_name, str) and vector_cache_name:
            vector_directory = path.parent / vector_cache_name
            ids_paths = sorted(vector_directory.glob("*.ids.json"))
            if len(ids_paths) != 1:
                return None
            ids_path = ids_paths[0]
            stem = ids_path.name[:-len(".ids.json")]
            matrix_path = vector_directory / f"{stem}.npy"
            node_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            if len(node_ids) != len(matrix):
                return None
            node_by_id = {
                node.node_id: node for node in [
                    *v36_index.turns, *v36_index.frames,
                    *v36_index.routing_cards, *v36_index.evidence_groups,
                ]
            }
            if any(node_id not in node_by_id for node_id in node_ids):
                return None
            for position, node_id in enumerate(node_ids):
                node_by_id[node_id].embedding = matrix[position].tolist()
        summary_by_id = {summary.node_id: summary for summary in summaries}
        roots = [
            summary_by_id[root_id]
            for root_id in payload.get("root_ids", [])
            if root_id in summary_by_id
        ]
        edges = [GraphEdge(**row) for row in payload.get("edges", [])]
        if facts and routing_cards:
            deterministic_edges = build_v2_graph_edges(leaves, routing_cards, facts, state_chains)
            edges = _merge_graph_edges(edges, deterministic_edges)
        llm_records = [
            DeepSeekCallRecord(**row) for row in payload.get("llm_records", [])
        ]
        metrics_payload = payload.get("metrics") or {}
        metrics = BuildMetrics(
            ready_job_counts=list(metrics_payload.get("ready_job_counts") or []),
            summary_parse_error_count=int(
                metrics_payload.get("summary_parse_error_count") or 0
            ),
            summary_truncation_count=int(
                metrics_payload.get("summary_truncation_count") or 0
            ),
            peak_inflight_deepseek=int(metrics_payload.get("peak_inflight_deepseek") or 0),
            index_diagnostics=list(metrics_payload.get("index_diagnostics") or []),
        )
        return MemoryBuild(
            leaves=leaves,
            summaries=summaries,
            roots=roots,
            edges=edges,
            llm_records=llm_records,
            metrics=metrics,
            build_latency_sec=float(payload.get("build_latency_sec") or 0.0),
            facts=facts,
            routing_cards=routing_cards,
            state_chains=state_chains,
            v3_index=v3_index,
            v36_index=v36_index,
            v4_capability_view=v4_capability_view,
        )
    except Exception:
        return None


def _run_case_with_services(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> tuple[CaseRun, list[Any], list[Any]]:
    case_llm, case_embedder, case_compressor, case_summarizer = _complete_services(
        config, variant, llm, embedder, compressor, summarizer
    )
    try:
        embedding_start = _record_start(case_embedder)
        compression_start = _record_start(case_compressor)
        summarizer_start = _record_start(case_summarizer)
        run = run_case(
            config,
            case,
            variant,
            case_llm,
            case_embedder,
            case_compressor,
            limiter,
            summarizer=case_summarizer,
        )
        return (
            run,
            _records_since(case_embedder, embedding_start),
            [
                *_records_since(case_compressor, compression_start),
                *_records_since(case_summarizer, summarizer_start),
            ],
        )
    finally:
        _close_owned_services(
            (case_llm, llm is None),
            (case_embedder, embedder is None),
            (case_compressor, compressor is None),
            (case_summarizer, summarizer is None),
        )


def _has_memory_cache_keys(cases: list[QuestionCase]) -> bool:
    return any(case.memory_cache_key for case in cases)


def _completed_question_results(futures: dict[Any, QuestionCase]) -> Any:
    """Yield every successful question result before reporting batch failures."""
    failures: list[tuple[str, Exception]] = []
    for future in as_completed(futures):
        case = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            failures.append((case.question_id, exc))
            continue
        yield case, result
    if failures:
        failed_ids = ", ".join(question_id for question_id, _ in failures[:8])
        if len(failures) > 8:
            failed_ids += f", ... (+{len(failures) - 8} more)"
        raise RuntimeError(
            f"{len(failures)} question worker(s) failed after successful "
            f"results were yielded: {failed_ids}"
        ) from failures[0][1]


def _run_cases_with_memory_cache(
    config: DemoConfig,
    cases: list[QuestionCase],
    variant: str,
    variant_dir: Path,
    limiter: InflightLimiter,
    *,
    allow_memory_cache_read: bool,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> Any:
    has_injected_services = any(
        service is not None for service in (llm, embedder, compressor, summarizer)
    )
    grouped: dict[str, list[QuestionCase]] = {}
    for case in cases:
        key = case.memory_cache_key or f"question:{case.question_id}"
        grouped.setdefault(key, []).append(case)

    # Cache-aware execution used to serialize this outer loop.  LongMemEval
    # assigns every question its own cache key, so --question-workers only
    # parallelized questions that happened to share one key and had no effect
    # on the benchmark build.  Run independent cache groups concurrently; the
    # recursive call receives exactly one group and therefore executes the
    # existing single-build/share-within-group path below.
    if (
        config.question_workers > 1
        and not has_injected_services
        and len(grouped) > 1
    ):
        worker_count=min(config.question_workers,len(grouped))
        group_config = replace(
            config,
            question_workers=max(1, config.question_workers // worker_count),
        )

        output_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def run_group(group_cases: list[QuestionCase]) -> None:
            try:
                for item in _run_cases_with_memory_cache(
                    group_config,group_cases,variant,variant_dir,limiter,
                    allow_memory_cache_read=allow_memory_cache_read,
                    llm=llm,embedder=embedder,compressor=compressor,
                    summarizer=summarizer,
                ):
                    output_queue.put(("item", item))
            except BaseException as error:
                output_queue.put(("error", error))
            finally:
                output_queue.put(("done", None))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures={
                executor.submit(run_group,group_cases):group_cases[0].question_id
                for group_cases in grouped.values()
            }
            remaining = len(futures)
            first_error: BaseException | None = None
            while remaining:
                kind, payload = output_queue.get()
                if kind == "item":
                    yield payload
                elif kind == "error":
                    if first_error is None:
                        first_error = payload
                        # Pending groups have not acquired resources and can be
                        # cancelled immediately. Running groups are allowed to
                        # publish their final item/done rows before the error is
                        # re-raised, so no producer can block on an abandoned
                        # queue during executor shutdown.
                        for future in futures:
                            if future.cancel():
                                remaining -= 1
                else:
                    remaining -= 1
            for future in futures:
                if not future.cancelled():
                    future.result()
            if first_error is not None:
                raise first_error
        return

    for group_cases in grouped.values():
        build_case = group_cases[0]
        memory_cache_key = build_case.memory_cache_key
        if not memory_cache_key:
            case = group_cases[0]
            run, embedding_records, compression_records = _run_case_with_services(
                config,
                case,
                variant,
                limiter,
                llm=llm,
                embedder=embedder,
                compressor=compressor,
                summarizer=summarizer,
            )
            yield case, run, embedding_records, compression_records
            continue

        group_build_started = time.perf_counter()
        memory_cache_path = _memory_cache_path(config, variant_dir, build_case, variant)
        memory = (
            _load_memory_cache(memory_cache_path)
            if allow_memory_cache_read and memory_cache_path.exists()
            else None
        )
        if (
            memory is None
            and allow_memory_cache_read
            and _is_v41_variant(variant)
        ):
            compatible_paths = _v41_compatible_cache_candidates(
                memory_cache_path, build_case,
            )
            compatible_memories: list[MemoryBuild] = []
            for compatible_path in compatible_paths:
                candidate = _load_memory_cache(compatible_path)
                if candidate is None or candidate.v36_index is None:
                    continue
                expected_sessions = set(build_case.haystack_session_ids)
                cached_sessions = {
                    card.session_id
                    for card in candidate.v36_index.routing_cards
                }
                if (
                    expected_sessions and cached_sessions
                    and not expected_sessions.issubset(cached_sessions)
                ):
                    continue
                compatible_memories.append(candidate)
            if len(compatible_memories) == 1:
                memory = compatible_memories[0]
        loaded_from_cache = memory is not None
        build_embedding_records: list[Any] = []
        build_compression_records: list[Any] = []
        if memory is None:
            build_llm, build_embedder, build_compressor, build_summarizer = _complete_services(
                config, variant, llm, embedder, compressor, summarizer
            )
            try:
                build_embedding_start = _record_start(build_embedder)
                build_compression_start = _record_start(build_compressor)
                build_summarizer_start = _record_start(build_summarizer)
                memory = build_memory(
                    config,
                    build_case,
                    variant,
                    build_llm,
                    build_embedder,
                    build_compressor,
                    limiter,
                    summarizer=build_summarizer,
                )
                build_embedding_records = _records_since(build_embedder, build_embedding_start)
                build_compression_records = [
                    *_records_since(build_compressor, build_compression_start),
                    *_records_since(build_summarizer, build_summarizer_start),
                ]
                _write_memory_cache(memory_cache_path, memory, build_case, variant, config)
            finally:
                _close_owned_services(
                    (build_llm, llm is None),
                    (build_embedder, embedder is None),
                    (build_compressor, compressor is None),
                    (build_summarizer, summarizer is None),
                )
        if variant == "hierarchical_hybrid_graph_v4_1_query" and memory.v41_sidecar is None:
            memory.v41_sidecar = build_v41_sidecar(memory.v36_index)

        build_record_question_id: str | None = build_case.question_id
        if loaded_from_cache and variant in {"hierarchical_state_graph_v2", "hierarchical_hypergraph_v3", "hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
            # A partial resume in the same output directory may already contain
            # the original build owner. Do not attach saved build records to the
            # first pending question a second time.
            existing_build_owners = {
                item.question_id
                for item in _read_question_stats(variant_dir / "question_stats.jsonl")
                if item.build_total_tokens > 0
            }
            cached_build_owner = next(
                (
                    record.question_id
                    for record in memory.llm_records
                    if record.stage.startswith("build_")
                ),
                None,
            )
            if cached_build_owner in existing_build_owners:
                build_record_question_id = None

        worker_count = min(config.question_workers, len(group_cases))
        if has_injected_services:
            worker_count = 1
        if worker_count <= 1:
            for case in group_cases:
                include_build_records = (
                    case.question_id == build_record_question_id
                    and (not loaded_from_cache or variant in {"hierarchical_state_graph_v2", "hierarchical_hypergraph_v3", "hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"})
                )
                run, embedding_records, compression_records = _run_case_with_cached_memory(
                    config,
                    case,
                    variant,
                    limiter,
                    memory,
                    include_build_records,
                    group_build_started if include_build_records else None,
                    llm=llm,
                    embedder=embedder,
                )
                if include_build_records:
                    embedding_records = [*build_embedding_records, *embedding_records]
                    compression_records = [*build_compression_records, *compression_records]
                yield case, run, embedding_records, compression_records
            continue

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_case_with_cached_memory,
                    config,
                    case,
                    variant,
                    limiter,
                    memory,
                    (case.question_id == build_record_question_id and (not loaded_from_cache or variant in {"hierarchical_state_graph_v2", "hierarchical_hypergraph_v3", "hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"})),
                    (
                        group_build_started
                        if case.question_id == build_record_question_id
                        else None
                    ),
                    llm=llm,
                    embedder=embedder,
                ): case
                for case in group_cases
            }
            for case, result in _completed_question_results(futures):
                run, embedding_records, compression_records = result
                if not loaded_from_cache and case.question_id == build_record_question_id:
                    embedding_records = [*build_embedding_records, *embedding_records]
                    compression_records = [*build_compression_records, *compression_records]
                yield case, run, embedding_records, compression_records


def _run_case_with_cached_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
    memory: MemoryBuild,
    include_build_records: bool,
    case_started: float | None = None,
    llm: Any | None = None,
    embedder: Any | None = None,
) -> tuple[CaseRun, list[Any], list[Any]]:
    case_llm, case_embedder, _compressor, _summarizer = _complete_services(
        config, variant, llm, embedder
    )
    try:
        embedding_start = _record_start(case_embedder)
        run = run_case_with_memory(
            config,
            case,
            variant,
            memory,
            case_llm,
            case_embedder,
            limiter,
            include_build_records=include_build_records,
            case_started=case_started,
        )
        if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
            _attach_v36_offline_gold_metrics(case, run)
        return run, _records_since(case_embedder, embedding_start), []
    finally:
        _close_owned_services(
            (case_llm, llm is None),
            (case_embedder, embedder is None),
        )


def _memory_artifact_dir(variant_dir: Path, case: QuestionCase) -> Path:
    if not case.memory_cache_key:
        return variant_dir
    digest = hashlib.sha256(case.memory_cache_key.encode("utf-8")).hexdigest()[:16]
    label = _safe_cache_part(case.memory_cache_key)[:80]
    return variant_dir / "memory_indices" / f"{label}-{digest}"


def _write_case_outputs(
    variant_dir: Path,
    case: QuestionCase,
    run: CaseRun,
    embedding_records: list[Any],
    compression_records: list[Any],
    persist_memory_artifacts: bool = True,
) -> None:
    _append_jsonl(variant_dir / "llm_calls.jsonl", [asdict(item) for item in run.llm_records])
    _append_jsonl(
        variant_dir / "embedding_calls.jsonl",
        [asdict(item) for item in embedding_records],
    )
    _append_jsonl(
        variant_dir / "compression_stats.jsonl",
        [asdict(item) for item in compression_records],
    )
    # A LoCoMo conversation has many questions sharing one immutable memory.
    # Persist that index only with its build-owner question; query-only cases
    # still write retrieval, answer, and token records below.
    persist_shared_index = persist_memory_artifacts and (
        not case.memory_cache_key or run.stats.build_total_tokens > 0
    )
    if persist_shared_index:
        artifact_dir = _memory_artifact_dir(variant_dir, case)
        if run.v36_index is not None:
            _append_jsonl(
                variant_dir / "nodes.jsonl",
                [_node_row(item) for item in [
                    *run.v36_index.turns, *run.v36_index.frames,
                    *run.v36_index.routing_cards, *run.v36_index.evidence_groups,
                ]],
            )
            _append_jsonl(
                variant_dir / "edges.jsonl",
                [asdict(item) for item in run.v36_index.edges],
            )
            _append_jsonl(
                variant_dir / "state_chains.jsonl",
                [asdict(item) for item in run.v36_index.state_chains],
            )
            _append_jsonl(
                variant_dir / "coverage.jsonl",
                [asdict(item) for item in run.v36_index.coverage],
            )
            persist_v36_retrieval_index(
                artifact_dir / "retrieval.sqlite", run.v36_index
            )
            persist_v36_vector_matrix(
                artifact_dir / "vectors", run.v36_index
            )
            if run.v4_capability_view is not None:
                _append_jsonl(
                    variant_dir / "v4_capability_views.jsonl",
                    [{
                        "schema_version": GRAPHMEM_V4_SCHEMA,
                        "base_schema_version": GRAPHMEM_V36_SCHEMA,
                        "build_version": V4_BUILD_VERSION,
                        "question_id": case.question_id,
                        "capability_view": asdict(run.v4_capability_view),
                    }],
                )
            if run.v41_sidecar is not None:
                persist_v41_sidecar(
                    artifact_dir / "retrieval_v41.sqlite", run.v41_sidecar
                )
        if run.v3_index is not None:
            _append_jsonl(
                variant_dir / "nodes.jsonl",
                [_node_row(item) for item in [
                    *run.v3_index.turns, *run.v3_index.claims, *run.v3_index.events,
                    *run.v3_index.event_entities,
                    *run.v3_index.episodes, *run.v3_index.themes,
                    *run.v3_index.event_frames, *run.v3_index.operands,
                ]],
            )
            _append_jsonl(variant_dir / "episodes.jsonl", [{key: value for key, value in asdict(item).items() if key != "embedding"} for item in run.v3_index.episodes])
            _append_jsonl(variant_dir / "themes.jsonl", [{key: value for key, value in asdict(item).items() if key != "embedding"} for item in run.v3_index.themes])
            _append_jsonl(variant_dir / "event_frames.jsonl", [{key: value for key, value in asdict(item).items() if key != "embedding"} for item in run.v3_index.event_frames])
            _append_jsonl(variant_dir / "operands.jsonl", [{key: value for key, value in asdict(item).items() if key not in {"embedding", "object_embedding"}} for item in run.v3_index.operands])
            _append_jsonl(variant_dir / "hyperedges.jsonl", [{key: value for key, value in asdict(item).items() if key != "embedding"} for item in run.v3_index.hyperedges])
            _append_jsonl(variant_dir / "state_chains.jsonl", [asdict(item) for item in run.v3_index.state_chains])
        _append_jsonl(
            variant_dir / "nodes.jsonl",
            [_node_row(item) for item in [*run.leaves, *run.summaries, *run.routing_cards, *run.facts]],
        )
        _append_jsonl(
            variant_dir / "state_chains.jsonl", [asdict(item) for item in run.state_chains]
        )
        _append_jsonl(variant_dir / "index_diagnostics.jsonl", run.index_diagnostics)
        _append_jsonl(variant_dir / "edges.jsonl", [asdict(item) for item in run.edges])
    _append_jsonl(variant_dir / "question_stats.jsonl", [asdict(run.stats)])
    _append_jsonl(variant_dir / "retrieval_results.jsonl", [asdict(run.retrieval)])
    _append_jsonl(
        variant_dir / "answers.jsonl",
        [
            {
                "question_id": case.question_id,
                "variant": run.stats.variant,
                "question": case.question,
                "question_type": case.question_type,
                "question_date": case.question_date,
                "gold_answer": case.answer,
                "prediction": run.answer,
                "answer_notes": run.answer_notes,
                "answer_note_parse_error": run.answer_note_parse_error,
                "answer_used_notes": run.answer_used_notes,
                "answer_session_ids": case.answer_session_ids,
                "retrieved_answer_session_hit": run.retrieval.answer_session_hit,
                "retrieved_answer_session_any_hit": run.retrieval.answer_session_hit,
                "retrieved_answer_session_all_hit": run.retrieval.answer_session_all_hit,
                "retrieved_answer_session_recall": run.retrieval.answer_session_recall,
            }
        ],
    )


def _print_case_progress(stats: QuestionStats) -> None:
    print(
        f"{stats.variant}: question={stats.question_id} "
        f"calls={stats.deepseek_call_count} llm_tokens={stats.total_deepseek_tokens}",
        flush=True,
    )


def run_case(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    compressor: Any,
    limiter: InflightLimiter | None = None,
    summarizer: Any | None = None,
) -> CaseRun:
    case_started = time.perf_counter()
    limiter = limiter or InflightLimiter(config.max_inflight_deepseek)
    memory = build_memory(
        config,
        case,
        variant,
        llm,
        embedder,
        compressor,
        limiter,
        summarizer=summarizer,
    )
    run = run_case_with_memory(
        config,
        case,
        variant,
        memory,
        llm,
        embedder,
        limiter,
        include_build_records=True,
        case_started=case_started,
    )
    if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
        _attach_v36_offline_gold_metrics(case, run)
    return run


def _build_v36_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    build_started: float,
) -> MemoryBuild:
    def chat(*, stage: str, messages: list[dict[str, str]], max_tokens: int, json_mode: bool) -> Any:
        return _tracked_chat(
            llm, limiter, metrics, question_id=case.question_id, variant=variant,
            stage=stage, thinking_mode="none", messages=messages,
            max_tokens=max_tokens, json_mode=json_mode,
        )

    def embed(nodes: list[Any], attr: str, target_attr: str = "embedding") -> None:
        if nodes:
            _embed_nodes(
                nodes, embedder, case.question_id, variant,
                attr=attr, target_attr=target_attr,
            )

    checkpoint_root = (
        config.memory_cache_dir
        if config.memory_cache_dir is not None
        else config.output_dir / variant / "memory_cache"
    )
    checkpoint_namespace = hashlib.sha256(json.dumps(
        {
            "schema": (GRAPHMEM_V4_SCHEMA if _is_v4_variant(variant) else GRAPHMEM_V36_SCHEMA),
            "prompt_version": (V4_BUILD_VERSION if _is_v4_variant(variant) else V36_PROMPT_VERSION),
            "model": config.deepseek_model,
            "base_url": config.deepseek_base_url,
            "session_max_tokens": config.v36_session_extraction_max_tokens,
            "build_budget_tokens": config.build_budget_tokens,
        },
        sort_keys=True, ensure_ascii=True,
    ).encode("utf-8")).hexdigest()[:16]
    checkpoint_key = (
        f"{_safe_cache_part(case.memory_cache_key or case.question_id)}-"
        f"{checkpoint_namespace}"
    )
    checkpoint_dir = checkpoint_root / ".v36_call_checkpoints" / checkpoint_key
    result = build_v36_index(
        case=case, variant=variant, chat=chat, embed=embed,
        max_tokens=config.v36_session_extraction_max_tokens,
        workers=min(32, config.summary_workers or 32),
        build_budget_tokens=config.build_budget_tokens,
        checkpoint_dir=checkpoint_dir,
        llm_session_cap=config.v36_llm_session_cap,
    )
    metrics.summary_parse_error_count += result.parse_error_count
    metrics.index_diagnostics.extend(result.diagnostics)
    v4_capability_view = None
    if _is_v4_variant(variant):
        v4_capability_view = build_v4_capability_view(result.index)
        capability_errors = validate_v4_capability_view(result.index, v4_capability_view)
        if capability_errors:
            raise ValueError(f"V4 capability validation failed: {capability_errors[:8]}")
        metrics.index_diagnostics.append({
            "stage": "v4_capability_projection",
            **v4_capability_view.diagnostics,
            "topology_mode": v4_capability_view.topology_mode,
        })
    build_total = sum(
        record.total_tokens for record in result.records
        if not record.excluded_from_budget
    )
    if build_total > config.build_budget_tokens:
        raise RuntimeError(
            f"build token budget exceeded for {case.question_id}: "
            f"{build_total}>{config.build_budget_tokens}"
        )
    return MemoryBuild(
        leaves=[], summaries=[], roots=[], edges=[], llm_records=result.records,
        metrics=metrics, build_latency_sec=time.perf_counter() - build_started,
        v36_index=result.index, v4_capability_view=v4_capability_view,
    )


def _build_v3_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    build_started: float,
) -> MemoryBuild:
    def chat(*, stage: str, messages: list[dict[str, str]], max_tokens: int, json_mode: bool) -> Any:
        return _tracked_chat(
            llm, limiter, metrics, question_id=case.question_id, variant=variant,
            stage=stage, thinking_mode="none", messages=messages,
            max_tokens=max_tokens, json_mode=json_mode,
        )

    def embed(nodes: list[Any], attr: str, target_attr: str = "embedding") -> None:
        _embed_nodes(
            nodes, embedder, case.question_id, variant,
            attr=attr, target_attr=target_attr,
        )

    result = build_v3_index(
        case=case, variant=variant, chat=chat, embed=embed,
        max_tokens=config.v3_session_extraction_max_tokens,
        workers=min(32, config.summary_workers or 32),
        build_budget_tokens=config.build_budget_tokens,
    )
    metrics.summary_parse_error_count += result.parse_error_count
    metrics.index_diagnostics.extend(result.diagnostics)
    build_total = sum(
        record.total_tokens for record in result.records
        if not record.excluded_from_budget
    )
    if build_total > config.build_budget_tokens:
        raise RuntimeError(
            f"build token budget exceeded for {case.question_id}: "
            f"{build_total}>{config.build_budget_tokens}"
        )
    return MemoryBuild(
        leaves=[], summaries=[], roots=[], edges=[], llm_records=result.records,
        metrics=metrics, build_latency_sec=time.perf_counter() - build_started,
        v3_index=result.index,
    )


def _build_v2_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    leaves: list[LeafNode],
    llm: Any,
    embedder: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    build_started: float,
) -> MemoryBuild:
    llm_records: list[DeepSeekCallRecord] = []
    for leaf in leaves:
        leaf.schema_version = GRAPHMEM_V2_SCHEMA
    grouped = group_by_session(leaves)
    session_dates = dict(zip(case.haystack_session_ids, case.haystack_dates))

    def extract(session_id: str, session_leaves: list[LeafNode]) -> tuple[RoutingCardNode, list[AtomicFactNode], str | None, DeepSeekCallRecord]:
        result = _tracked_chat(
            llm, limiter, metrics, question_id=case.question_id, variant=variant,
            stage="build_fact_extraction", thinking_mode="none",
            messages=v2_session_extraction_messages(session_id, session_dates.get(session_id), session_leaves),
            max_tokens=config.v2_fact_extraction_max_tokens, json_mode=True,
        )
        card, facts, error = parse_session_extraction(
            result.text, question_id=case.question_id, session_id=session_id,
            session_date=session_dates.get(session_id), leaves=session_leaves,
        )
        return card, facts, error, result.record

    extracted: list[tuple[str, RoutingCardNode, list[AtomicFactNode], str | None, DeepSeekCallRecord]] = []
    workers = min(32, config.summary_workers or 32, max(1, len(grouped)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract, session_id, values): session_id for session_id, values in grouped.items()}
        for future in as_completed(futures):
            session_id = futures[future]
            card, facts, error, record = future.result()
            extracted.append((session_id, card, facts, error, record))
    extracted.sort(key=lambda row: case.haystack_session_ids.index(row[0]))
    cards = [row[1] for row in extracted]
    facts = [fact for row in extracted for fact in row[2]]
    for observation_order, fact in enumerate(facts):
        fact.observation_order = observation_order
    llm_records.extend(row[4] for row in extracted)
    metrics.summary_parse_error_count += sum(row[3] is not None for row in extracted)
    for session_id, _card, session_facts, error, record in extracted:
        metrics.index_diagnostics.append({
            "question_id":case.question_id,"variant":variant,"stage":"session_extraction",
            "session_id":session_id,"leaf_count":len(grouped[session_id]),"fact_count":len(session_facts),
            "fallback_fact_count":sum(fact.confidence<=0.45 for fact in session_facts),
            "parse_error":error,"finish_reason":record.finish_reason,
            "prompt_tokens":record.prompt_tokens,"completion_tokens":record.completion_tokens,
            "total_tokens":record.total_tokens,
        })

    extraction_total = sum(record.total_tokens for record in llm_records)
    proposed_edges: list[GraphEdge] = []
    consolidation_messages = v2_consolidation_messages(facts) if facts else []
    consolidation_estimate = (
        provider_token_estimate("\n".join(message["content"] for message in consolidation_messages))
        + config.v2_consolidation_max_tokens
    )
    consolidation_allowed = bool(facts) and (
        extraction_total + consolidation_estimate <= config.build_budget_tokens - 2_000
    )
    if consolidation_allowed:
        consolidation = _tracked_chat(
            llm, limiter, metrics, question_id=case.question_id, variant=variant,
            stage="build_fact_consolidation", thinking_mode="none",
            messages=consolidation_messages,
            max_tokens=config.v2_consolidation_max_tokens, json_mode=True,
        )
        llm_records.append(consolidation.record)
        proposed_edges, _aliases, error = apply_v2_consolidation(consolidation.text, facts)
        if error:
            metrics.summary_parse_error_count += 1
        metrics.index_diagnostics.append({
            "question_id":case.question_id,"variant":variant,"stage":"question_consolidation",
            "fact_count":len(facts),"allowed":True,"estimated_tokens":consolidation_estimate,
            "parse_error":error,"finish_reason":consolidation.record.finish_reason,
            "prompt_tokens":consolidation.record.prompt_tokens,"completion_tokens":consolidation.record.completion_tokens,
            "total_tokens":consolidation.record.total_tokens,"accepted_edge_count":len(proposed_edges),
        })
    else:
        metrics.index_diagnostics.append({
            "question_id":case.question_id,"variant":variant,"stage":"question_consolidation",
            "fact_count":len(facts),"allowed":False,"estimated_tokens":consolidation_estimate,
            "reason":"soft_budget_guard","extraction_total_tokens":extraction_total,
        })

    _embed_nodes(leaves, embedder, case.question_id, variant, attr="retrieval_text")
    _embed_nodes(cards, embedder, case.question_id, variant, attr="retrieval_text")
    _embed_nodes(facts, embedder, case.question_id, variant, attr="retrieval_text")
    chains, state_edges = build_v2_state_chains(facts)
    edges = build_v2_graph_edges(
        leaves, cards, facts, chains,
        semantic_k=config.v2_semantic_k, semantic_floor=config.v2_semantic_floor,
    )
    edges = _merge_graph_edges([*edges, *state_edges], proposed_edges)
    provenance_errors = validate_v2_provenance(facts, leaves, edges)
    if provenance_errors:
        raise ValueError(f"V2 provenance validation failed: {provenance_errors[:8]}")
    build_total = sum(record.total_tokens for record in llm_records if not record.excluded_from_budget)
    if build_total > config.build_budget_tokens:
        raise RuntimeError(f"build token budget exceeded for {case.question_id}: {build_total}>{config.build_budget_tokens}")
    return MemoryBuild(
        leaves=leaves, summaries=[], roots=[], edges=edges, llm_records=llm_records,
        metrics=metrics, build_latency_sec=time.perf_counter()-build_started,
        facts=facts, routing_cards=cards, state_chains=chains,
    )


def build_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    compressor: Any,
    limiter: InflightLimiter,
    summarizer: Any | None = None,
) -> MemoryBuild:
    metrics = BuildMetrics()
    spec = _variant_spec(config, variant)
    llm_records: list[DeepSeekCallRecord] = []
    build_started = time.perf_counter()
    leaves = build_leaf_nodes(case)
    if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
        return _build_v36_memory(
            config, case, variant, llm, embedder, limiter, metrics, build_started
        )
    if variant == "hierarchical_hypergraph_v3":
        return _build_v3_memory(
            config, case, variant, llm, embedder, limiter, metrics, build_started
        )
    if variant == "hierarchical_state_graph_v2":
        return _build_v2_memory(
            config, case, variant, leaves, llm, embedder, limiter, metrics, build_started
        )
    retrieval_leaf_mode = _effective_leaf_text_mode(
        config.retrieval_leaf_text, spec, case, phase="retrieval"
    )
    leaf_enrichment = _leaf_enrichment_enabled(config, spec)
    if spec.tree_mode == "raw_rag" or not leaf_enrichment:
        _embed_nodes(
            leaves,
            embedder,
            case.question_id,
            variant,
            attr=_leaf_embedding_attr(config, case, retrieval_leaf_mode),
        )
    summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    edges: list[GraphEdge] = []

    if spec.tree_mode != "raw_rag":
        summaries, roots = _build_summary_roots(
            config,
            case,
            leaves,
            variant,
            spec,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
        if config.enable_speaker_profiles:
            speaker_profiles = _build_speaker_profile_roots(
                config,
                case,
                roots,
                variant,
                spec,
                llm,
                llm_records,
                metrics,
                limiter,
            )
            summaries.extend(speaker_profiles)
            roots.extend(speaker_profiles)
        if leaf_enrichment:
            _embed_nodes(
                leaves,
                embedder,
                case.question_id,
                variant,
                attr=_leaf_embedding_attr(config, case, retrieval_leaf_mode),
            )
        _embed_nodes(summaries, embedder, case.question_id, variant, attr="retrieval_text")
        if spec.graph:
            graph_roots = summaries if config.enable_multilevel_summary_retrieval else roots
            edges = _build_root_graph(
                graph_roots,
                config.graph_neighbor_k,
                edge_policy=_root_graph_edge_policy(
                    config,
                    enable_typed_edges=_effective_typed_root_edges(config),
                ),
            )
            if config.enable_llm_root_edges:
                llm_edges = _build_llm_root_edges(
                    config,
                    case,
                    variant,
                    graph_roots,
                    llm,
                    llm_records,
                    metrics,
                    limiter,
                )
                edges = _merge_graph_edges(edges, llm_edges)
            if config.enable_llm_leaf_edges or config.enable_leaf_graph_expansion:
                leaf_edges = _build_session_leaf_graph_edges(
                    config,
                    case,
                    variant,
                    leaves,
                    llm,
                    llm_records,
                    metrics,
                    limiter,
                )
                edges = _merge_graph_edges(edges, leaf_edges)
    build_latency = time.perf_counter() - build_started
    return MemoryBuild(
        leaves=leaves,
        summaries=summaries,
        roots=roots,
        edges=edges,
        llm_records=llm_records,
        metrics=metrics,
        build_latency_sec=build_latency,
    )


def _clone_v2_nodes(memory: MemoryBuild, question_id: str):
    id_map: dict[str, str] = {}
    def remap(node_id: str) -> str:
        if node_id not in id_map:
            suffix = node_id.split(":", 1)[1] if ":" in node_id else node_id
            id_map[node_id] = f"{question_id}:{suffix}"
        return id_map[node_id]
    leaves = [replace(leaf, node_id=remap(leaf.node_id), question_id=question_id,
                      compact_facts=list(leaf.compact_facts), anchor_terms=dict(leaf.anchor_terms),
                      embedding=leaf.embedding) for leaf in memory.leaves]
    facts = [replace(fact, node_id=remap(fact.node_id), question_id=question_id,
                     source_leaf_ids=[remap(value) for value in fact.source_leaf_ids],
                     observation_order=(fact.observation_order if fact.observation_order >= 0 else index),
                     embedding=fact.embedding)
             for index, fact in enumerate(memory.facts)]
    cards = [replace(card, node_id=remap(card.node_id), question_id=question_id,
                     fact_ids=[remap(value) for value in card.fact_ids], leaf_ids=[remap(value) for value in card.leaf_ids],
                     embedding=card.embedding) for card in memory.routing_cards]
    chains = [replace(chain, chain_id=remap(chain.chain_id), question_id=question_id,
                      current_fact_ids=[remap(value) for value in chain.current_fact_ids],
                      history_fact_ids=[remap(value) for value in chain.history_fact_ids],
                      update_order=[remap(value) for value in chain.update_order]) for chain in memory.state_chains]
    edges = [replace(edge, src=remap(edge.src), dst=remap(edge.dst), provenance=dict(edge.provenance)) for edge in memory.edges]
    return leaves, cards, facts, chains, edges


_V3_OBJECT_EMBED_LOCK = threading.Lock()


def _attach_v36_offline_gold_metrics(case: QuestionCase, run: CaseRun) -> None:
    """Attach benchmark-only recall diagnostics after the online answer is immutable."""
    gold = set(case.answer_session_ids)
    retrieved = {
        re.sub(r"__occ\d+$", "", session_id)
        for session_id in run.retrieval.retrieved_session_ids
    }
    hits = len(retrieved & gold)
    recall = hits / len(gold) if gold else 0.0
    run.retrieval.answer_session_hit = bool(hits)
    run.retrieval.answer_session_all_hit = bool(gold) and hits == len(gold)
    run.retrieval.answer_session_recall = recall
    run.retrieval.retrieved_answer_session_count = hits
    run.retrieval.gold_answer_session_count = len(gold)
    run.stats.retrieved_answer_session_hit = bool(hits)
    run.stats.retrieved_answer_session_all_hit = bool(gold) and hits == len(gold)
    run.stats.retrieved_answer_session_recall = recall
    run.stats.retrieved_answer_session_count = hits
    run.stats.gold_answer_session_count = len(gold)


def _run_v36_case_with_memory(
    config: DemoConfig, case: QuestionCase, variant: str, memory: MemoryBuild,
    llm: Any, embedder: Any, limiter: InflightLimiter, *, include_build_records: bool,
    case_started: float,
) -> CaseRun:
    if memory.v36_index is None:
        raise ValueError("role-graph memory cache is missing v36_index")
    is_v4 = _is_v4_variant(variant)
    is_v41 = _is_v41_variant(variant)
    v4_capability_view = memory.v4_capability_view
    if is_v4 and v4_capability_view is None:
        v4_capability_view = build_v4_capability_view(memory.v36_index)
        capability_errors = validate_v4_capability_view(
            memory.v36_index, v4_capability_view
        )
        if capability_errors:
            raise ValueError(f"V4 capability validation failed: {capability_errors[:8]}")
    answer_metrics = BuildMetrics()
    llm_records = (
        [replace(record, question_id=case.question_id) for record in memory.llm_records]
        if include_build_records else []
    )
    # V4/V4.1 indices are immutable at query time. Keep build-time IDs stable
    # so shared capability projections and sidecars address the same nodes.
    index = (
        memory.v36_index
        if is_v4
        else clone_v36_index(memory.v36_index, case.question_id)
    )
    query_ir = (
        build_v4_query_ir(case.question) if is_v4
        else build_v36_query_ir(case.question)
    )
    v41_plan = build_v41_query_plan(query_ir) if is_v41 else None
    query_views = (
        v41_query_views(query_ir, v41_plan)
        if is_v41 and v41_plan is not None
        else v4_query_views(query_ir) if is_v4
        else v36_query_views(query_ir)
    )
    query_vectors = embedder.embed(
        query_views, question_id=case.question_id, variant=variant
    )
    v41_policy = QueryPolicyV41(
        normal_context_target=config.v41_normal_context_target,
        complex_context_target=config.v41_complex_context_target,
        planner_prompt_max=config.v41_planner_prompt_max,
        planner_output_max=config.v41_planner_output_max,
        query_target=config.v41_query_target_tokens,
        query_hard_limit=config.v41_query_hard_limit_tokens,
    )
    if is_v41:
        assert v4_capability_view is not None
        try:
            sidecar = memory.v41_sidecar
            if sidecar is None:
                sidecar = build_v41_sidecar(index)
                memory.v41_sidecar = sidecar
        except Exception as error:
            sidecar = None
            retrieval = retrieve_v4(
                case=case, variant=variant, index=index,
                capability_view=v4_capability_view,
                query_vectors=query_vectors,
                token_budget=config.v41_normal_context_target,
            )
            retrieval.variant = variant
            retrieval.retrieval_trace["v41_sidecar_fallback"] = {
                "active": True,
                "reason": type(error).__name__,
                "fallback_variant": "hierarchical_hybrid_graph_v4_0",
            }
        else:
            retrieval = retrieve_v41(
                case=case, variant=variant, index=index,
                capability_view=v4_capability_view, sidecar=sidecar,
                query_ir=query_ir, query_vectors=query_vectors,
                token_budget=config.v41_complex_context_target, policy=v41_policy,
            )
        if (
            sidecar is not None
            and config.v41_enable_planner
            and retrieval.retrieval_trace.get("planner_required") is True
            and v41_plan is not None
        ):
            planner_call = _tracked_chat(
                llm, limiter, answer_metrics, question_id=case.question_id,
                variant=variant, stage="answer_query_planner",
                thinking_mode="none",
                messages=v41_planner_messages(
                    case, query_ir, v41_plan,
                    retrieval.retrieval_trace.get("v41_evidence_certificate") or {},
                    retrieval.retrieval_trace.get("v41_planner_evidence") or [],
                ),
                max_tokens=config.v41_planner_output_max, json_mode=True,
            )
            llm_records.append(planner_call.record)
            planner_result = parse_v41_planner_result(planner_call.text)
            offered_planner_source_ids = {
                str(row.get("source_turn_id") or "")
                for row in (
                    retrieval.retrieval_trace.get("v41_planner_evidence") or []
                )
            }
            planner_result.selected_source_ids = [
                source_id for source_id in planner_result.selected_source_ids
                if source_id in offered_planner_source_ids
            ]
            planner_result.member_candidates = [
                row for row in planner_result.member_candidates
                if row.get("source_turn_id") in offered_planner_source_ids
            ]
            remaining_context = max(
                config.v41_normal_context_target,
                config.v41_complex_context_target - planner_call.record.total_tokens,
            )
            retrieval = retrieve_v41(
                case=case, variant=variant, index=index,
                capability_view=v4_capability_view, sidecar=sidecar,
                query_ir=query_ir, query_vectors=query_vectors,
                token_budget=remaining_context, policy=v41_policy,
                planner=planner_result,
            )
            retrieval.retrieval_trace["planner_token_usage"] = {
                "cache_miss_input_tokens": planner_call.record.prompt_cache_miss_tokens,
                "cache_hit_input_tokens": planner_call.record.prompt_cache_hit_tokens,
                "output_tokens": planner_call.record.completion_tokens,
                "total_tokens": planner_call.record.total_tokens,
            }
    elif is_v4:
        assert v4_capability_view is not None
        retrieval = retrieve_v4(
            case=case, variant=variant, index=index,
            capability_view=v4_capability_view, query_vectors=query_vectors,
            token_budget=config.v36_context_token_budget,
        )
    else:
        retrieval = retrieve_v36(
            case=case, variant=variant, index=index, query_vectors=query_vectors,
            token_budget=config.v36_context_token_budget,
        )
    answer_started = time.perf_counter()
    answer_text = ""
    if config.retrieval_only:
        retrieval.retrieval_trace["answer_mode"] = "retrieval_only"
    else:
        # Operators are evidence-producing tools only. The single backbone
        # answer call validates semantic scope, provenance and lifecycle for
        # every V3.6 question; no deterministic hint bypasses this boundary.
        answer_text = None
    answer_messages_value = (
        v41_answer_messages(case, retrieval) if is_v41
        else v4_answer_messages(case, retrieval) if is_v4
        else v36_answer_messages(case, retrieval)
    )
    if is_v41 and not config.retrieval_only:
        planner_tokens = int(
            (retrieval.retrieval_trace.get("planner_token_usage") or {}).get("total_tokens") or 0
        )
        answer_algebra = str(
            (retrieval.retrieval_trace.get("v41_query_augmentation") or {}).get(
                "answer_algebra"
            ) or ""
        )
        complex_query = bool(planner_tokens) or answer_algebra in {
            "collection", "temporal_comparison", "state_update",
            "multi_hop_explanation",
        }
        preflight_total_limit = min(
            config.v41_query_hard_limit_tokens,
            12000 if complex_query else config.v41_query_target_tokens,
        )
        max_prompt_tokens = max(
            1000, preflight_total_limit - planner_tokens - min(512, config.qa_max_tokens)
        )
        estimate = provider_token_estimate("\n".join(
            message.get("content", "") for message in answer_messages_value
        ))
        while estimate > max_prompt_tokens:
            if trim_v41_latest_addition(retrieval) is None:
                break
            answer_messages_value = v41_answer_messages(case, retrieval)
            estimate = provider_token_estimate("\n".join(
                message.get("content", "") for message in answer_messages_value
            ))
        retrieval.retrieval_trace["v41_preflight_budget"] = {
            "provider_prompt_estimate": estimate,
            "max_prompt_tokens": max_prompt_tokens,
            "preflight_total_limit": preflight_total_limit,
            "complex_query": complex_query,
            "planner_tokens": planner_tokens,
            "trimmed_source_ids": retrieval.retrieval_trace.get(
                "v41_budget_trimmed_source_ids", []
            ),
        }
    if not config.retrieval_only and answer_text is None:
        result = _tracked_chat(
            llm, limiter, answer_metrics, question_id=case.question_id, variant=variant,
            stage="answer_qa", thinking_mode="none",
            messages=answer_messages_value,
            max_tokens=min(512, config.qa_max_tokens),
        )
        llm_records.append(result.record)
        answer_text = result.text.strip()
        retrieval.retrieval_trace["answer_mode"] = "llm_from_role_complete_evidence"
    answer_latency = time.perf_counter() - answer_started
    # Gold annotations are intentionally unavailable to the online V3.6 path.
    # Offline reporting attaches session-recall metrics after the answer is fixed.
    hit_count = 0
    build_metrics = memory.metrics if include_build_records else BuildMetrics()
    stats = build_question_stats(
        question_id=case.question_id, variant=variant,
        session_count=len(case.haystack_session_ids), leaf_count=len(index.turns),
        summary_count=(len(index.frames) + len(index.routing_cards) + len(index.evidence_groups)),
        edge_count=len(index.edges), records=llm_records,
        build_latency_sec=memory.build_latency_sec if include_build_records else 0.0,
        retrieval_latency_sec=retrieval.latency_sec, answer_latency_sec=answer_latency,
        answer_session_hit=retrieval.answer_session_hit,
        answer_session_all_hit=retrieval.answer_session_all_hit,
        answer_session_recall=retrieval.answer_session_recall,
        retrieved_answer_session_count=0, gold_answer_session_count=0,
        wall_time_sec=time.perf_counter() - case_started,
        summary_parse_error_count=build_metrics.summary_parse_error_count,
        summary_truncation_count=build_metrics.summary_truncation_count,
        ready_job_counts=build_metrics.ready_job_counts,
        peak_inflight_deepseek=max(
            build_metrics.peak_inflight_deepseek, answer_metrics.peak_inflight_deepseek
        ),
        build_budget_tokens=config.build_budget_tokens,
        answer_budget_tokens=(
            config.v41_query_hard_limit_tokens if is_v41
            else config.v36_answer_hard_budget_tokens
        ),
    )
    if is_v41:
        query_records = [
            record for record in llm_records
            if record.stage in {"answer_query_planner", "answer_qa"}
            and not record.excluded_from_budget
        ]
        query_usage = {
            "cache_miss_input_tokens": sum(record.prompt_cache_miss_tokens for record in query_records),
            "cache_hit_input_tokens": sum(record.prompt_cache_hit_tokens for record in query_records),
            "output_tokens": sum(record.completion_tokens for record in query_records),
            "reasoning_tokens": sum(record.reasoning_tokens for record in query_records),
            "total_tokens": sum(record.total_tokens for record in query_records),
        }
        query_usage.update({
            "over_10k": query_usage["total_tokens"] > 10000,
            "over_12k": query_usage["total_tokens"] > 12000,
            "over_13k": query_usage["total_tokens"] > 13000,
        })
        retrieval.retrieval_trace["v41_query_token_usage"] = query_usage
    retrieval.retrieval_trace["answer_target_budget_pass"] = (
        stats.answer_total_tokens <= (
            config.v41_query_target_tokens if is_v41 else config.answer_budget_tokens
        )
    )
    retrieval.retrieval_trace["answer_target_budget_tokens"] = (
        config.v41_query_target_tokens if is_v41 else config.answer_budget_tokens
    )
    retrieval.retrieval_trace["answer_hard_budget_tokens"] = (
        config.v41_query_hard_limit_tokens if is_v41
        else config.v36_answer_hard_budget_tokens
    )
    retrieval.retrieval_trace["answer_hard_budget_enforced"] = not (
        is_v41 and config.v41_record_query_budget_overflow
    )
    if (
        not stats.build_budget_pass
        or (
            not stats.answer_budget_pass
            and not (is_v41 and config.v41_record_query_budget_overflow)
        )
    ):
        raise RuntimeError(
            f"role-graph query budget exceeded for {case.question_id}: "
            f"build={stats.build_total_tokens}, answer={stats.answer_total_tokens}"
        )
    if not stats.token_accounting_valid:
        raise RuntimeError(f"invalid LLM token accounting for {case.question_id}")
    return CaseRun(
        leaves=[], summaries=[], edges=[], retrieval=retrieval, answer=answer_text,
        llm_records=llm_records, stats=stats,
        index_diagnostics=list(memory.metrics.index_diagnostics), v36_index=index,
        v4_capability_view=v4_capability_view,
        v41_sidecar=(sidecar if is_v41 else None),
    )


def _run_v3_case_with_memory(
    config: DemoConfig, case: QuestionCase, variant: str, memory: MemoryBuild,
    llm: Any, embedder: Any, limiter: InflightLimiter, *, include_build_records: bool,
    case_started: float,
) -> CaseRun:
    if memory.v3_index is None:
        raise ValueError("V3 memory cache is missing v3_index")
    # Older V3 caches predate the object-only semantic channel. Backfill once
    # on the shared immutable memory with local embeddings; no LLM tokens are used.
    if any(item.object_embedding is None for item in memory.v3_index.operands):
        with _V3_OBJECT_EMBED_LOCK:
            missing = [
                item for item in memory.v3_index.operands
                if item.object_embedding is None
            ]
            if missing:
                _embed_nodes(
                    missing, embedder, case.question_id, variant,
                    attr="object_text", target_attr="object_embedding",
                )
    answer_metrics = BuildMetrics()
    llm_records = (
        [replace(record, question_id=case.question_id) for record in memory.llm_records]
        if include_build_records
        else []
    )
    index = clone_v3_index(memory.v3_index, case.question_id)
    query_texts = v3_query_views(build_v3_query_frame(case.question))
    query_vectors = embedder.embed(
        query_texts, question_id=case.question_id, variant=variant
    )
    query_vector = query_vectors[0]
    retrieval = retrieve_v3(
        case=case, variant=variant, index=index, query_vector=query_vector,
        query_vectors=query_vectors,
        token_budget=config.v3_context_token_budget,
    )
    answer_started = time.perf_counter()
    answer_text = ""
    if config.retrieval_only:
        retrieval.retrieval_trace["answer_mode"] = "retrieval_only"
    else:
        answer_text = v3_authoritative_catalog_answer(retrieval.retrieval_trace)
    if not config.retrieval_only and answer_text is None:
        result = _tracked_chat(
            llm, limiter, answer_metrics, question_id=case.question_id, variant=variant,
            stage="answer_qa", thinking_mode="none",
            messages=v3_answer_messages(
                case, retrieval,
                max_prompt_tokens=max(
                    3000, int((config.answer_budget_tokens - min(512, config.qa_max_tokens) - 300) * 0.9)
                ),
            ),
            max_tokens=min(512, config.qa_max_tokens),
        )
        llm_records.append(result.record)
        answer_text = result.text.strip()
        retrieval.retrieval_trace["answer_mode"] = "llm_from_evidence"
    elif not config.retrieval_only:
        retrieval.retrieval_trace["answer_mode"] = "authoritative_operator"
    answer_latency = time.perf_counter() - answer_started
    gold = set(case.answer_session_ids)
    hit_count = len(set(retrieval.retrieved_session_ids) & gold)
    retrieval.answer_session_hit = bool(hit_count)
    retrieval.answer_session_all_hit = bool(gold) and hit_count == len(gold)
    retrieval.answer_session_recall = hit_count / len(gold) if gold else 0.0
    retrieval.retrieved_answer_session_count = hit_count
    retrieval.gold_answer_session_count = len(gold)
    build_metrics = memory.metrics if include_build_records else BuildMetrics()
    stats = build_question_stats(
        question_id=case.question_id, variant=variant,
        session_count=len(case.haystack_session_ids), leaf_count=len(index.turns),
        summary_count=len(index.claims) + len(index.events) + len(index.episodes) + len(index.themes),
        edge_count=len(index.hyperedges), records=llm_records,
        build_latency_sec=memory.build_latency_sec if include_build_records else 0.0,
        retrieval_latency_sec=retrieval.latency_sec, answer_latency_sec=answer_latency,
        answer_session_hit=retrieval.answer_session_hit,
        answer_session_all_hit=retrieval.answer_session_all_hit,
        answer_session_recall=retrieval.answer_session_recall,
        retrieved_answer_session_count=hit_count, gold_answer_session_count=len(gold),
        wall_time_sec=time.perf_counter() - case_started,
        summary_parse_error_count=build_metrics.summary_parse_error_count,
        summary_truncation_count=build_metrics.summary_truncation_count,
        ready_job_counts=build_metrics.ready_job_counts,
        peak_inflight_deepseek=max(
            build_metrics.peak_inflight_deepseek, answer_metrics.peak_inflight_deepseek
        ),
        build_budget_tokens=config.build_budget_tokens,
        answer_budget_tokens=config.answer_budget_tokens,
    )
    if not stats.build_budget_pass or not stats.answer_budget_pass:
        raise RuntimeError(
            f"V3 token budget exceeded for {case.question_id}: "
            f"build={stats.build_total_tokens}, answer={stats.answer_total_tokens}"
        )
    if not stats.token_accounting_valid:
        raise RuntimeError(f"invalid LLM token accounting for {case.question_id}")
    return CaseRun(
        leaves=[], summaries=[], edges=[], retrieval=retrieval, answer=answer_text,
        llm_records=llm_records, stats=stats,
        index_diagnostics=list(memory.metrics.index_diagnostics), v3_index=index,
    )


def _run_v2_case_with_memory(
    config: DemoConfig, case: QuestionCase, variant: str, memory: MemoryBuild,
    llm: Any, embedder: Any, limiter: InflightLimiter, *, include_build_records: bool,
    case_started: float,
) -> CaseRun:
    answer_metrics = BuildMetrics()
    llm_records = list(memory.llm_records) if include_build_records else []
    leaves, cards, facts, chains, edges = _clone_v2_nodes(memory, case.question_id)
    query_vector = embedder.embed([expand_v2_query(case.question)], question_id=case.question_id, variant=variant)[0]
    retrieval = retrieve_v2(
        case=case, variant=variant, leaves=leaves, cards=cards, facts=facts, chains=chains,
        edges=edges, query_vector=query_vector, card_k=config.v2_card_k, fact_k=config.v2_fact_k,
        leaf_k=config.v2_leaf_k, token_budget=config.v2_context_token_budget,
    )
    answer_started = time.perf_counter()
    result = _tracked_chat(
        llm, limiter, answer_metrics, question_id=case.question_id, variant=variant,
        stage="answer_qa", thinking_mode="none", messages=v2_answer_messages(case, retrieval),
        max_tokens=min(512, config.qa_max_tokens),
    )
    llm_records.append(result.record)
    answer_text, answer_guard = apply_v2_answer_constraint(
        case.question, retrieval, result.text
    )
    retrieval.retrieval_trace["answer_guard"] = answer_guard
    answer_latency = time.perf_counter() - answer_started
    # Gold support-session IDs are used only now, after the answer has been produced,
    # for offline recall reporting. They never influence planning, retrieval, packing, or QA.
    gold = set(case.answer_session_ids)
    hit_count = len(set(retrieval.retrieved_session_ids) & gold)
    retrieval.answer_session_hit = bool(hit_count)
    retrieval.answer_session_all_hit = bool(gold) and hit_count == len(gold)
    retrieval.answer_session_recall = hit_count / len(gold) if gold else 0.0
    retrieval.retrieved_answer_session_count = hit_count
    retrieval.gold_answer_session_count = len(gold)
    build_metrics = memory.metrics if include_build_records else BuildMetrics()
    stats = build_question_stats(
        question_id=case.question_id, variant=variant, session_count=len(case.haystack_session_ids),
        leaf_count=len(leaves), summary_count=len(cards)+len(facts), edge_count=len(edges), records=llm_records,
        build_latency_sec=memory.build_latency_sec if include_build_records else 0.0,
        retrieval_latency_sec=retrieval.latency_sec, answer_latency_sec=answer_latency,
        answer_session_hit=retrieval.answer_session_hit, answer_session_all_hit=retrieval.answer_session_all_hit,
        answer_session_recall=retrieval.answer_session_recall,
        retrieved_answer_session_count=hit_count, gold_answer_session_count=len(gold),
        wall_time_sec=time.perf_counter()-case_started,
        summary_parse_error_count=build_metrics.summary_parse_error_count,
        summary_truncation_count=build_metrics.summary_truncation_count,
        ready_job_counts=build_metrics.ready_job_counts,
        peak_inflight_deepseek=max(build_metrics.peak_inflight_deepseek, answer_metrics.peak_inflight_deepseek),
        build_budget_tokens=config.build_budget_tokens, answer_budget_tokens=config.answer_budget_tokens,
    )
    if not stats.answer_budget_pass:
        raise RuntimeError(f"answer token budget exceeded for {case.question_id}: {stats.answer_total_tokens}>{config.answer_budget_tokens}")
    if not stats.token_accounting_valid:
        raise RuntimeError(f"invalid LLM token accounting for {case.question_id}")
    return CaseRun(
        leaves=leaves, summaries=[], edges=edges, retrieval=retrieval, answer=answer_text,
        llm_records=llm_records, stats=stats, facts=facts, routing_cards=cards, state_chains=chains,
        index_diagnostics=list(memory.metrics.index_diagnostics),
    )


def run_case_with_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    memory: MemoryBuild,
    llm: Any,
    embedder: Any,
    limiter: InflightLimiter,
    *,
    include_build_records: bool,
    case_started: float | None = None,
) -> CaseRun:
    case_started = case_started if case_started is not None else time.perf_counter()
    spec = _variant_spec(config, variant)
    answer_metrics = BuildMetrics()
    if variant in {"hierarchical_role_graph_v3_6", "hierarchical_hybrid_graph_v4_0", "hierarchical_hybrid_graph_v4_1_query"}:
        return _run_v36_case_with_memory(
            config, case, variant, memory, llm, embedder, limiter,
            include_build_records=include_build_records, case_started=case_started,
        )
    if variant == "hierarchical_hypergraph_v3":
        return _run_v3_case_with_memory(
            config, case, variant, memory, llm, embedder, limiter,
            include_build_records=include_build_records, case_started=case_started,
        )
    if variant == "hierarchical_state_graph_v2":
        return _run_v2_case_with_memory(
            config, case, variant, memory, llm, embedder, limiter,
            include_build_records=include_build_records, case_started=case_started,
        )
    llm_records: list[DeepSeekCallRecord] = list(memory.llm_records) if include_build_records else []
    leaves, summaries, roots, edges = _clone_memory_for_case(memory, case.question_id)
    retrieval_roots = summaries if config.enable_multilevel_summary_retrieval else roots
    retrieval = _retrieve(
        config=config,
        case=case,
        variant=variant,
        leaves=leaves,
        roots=retrieval_roots,
        edges=edges,
        embedder=embedder,
        graph_enabled=spec.graph,
        hybrid_retrieval=spec.hybrid_retrieval,
        enhanced_retrieval=spec.enhanced_retrieval,
        enhanced_qa=spec.enhanced_qa,
    )
    if config.enable_iterative_leaf_denoise and retrieval.leaf_node_ids:
        denoise_started = time.perf_counter()
        query_vector = embedder.embed([case.question], question_id=case.question_id, variant=variant)[0]
        ranked_pool_leaves = _rank_leaves_for_config(
            config,
            leaves,
            query_vector,
            case.question,
            enhanced=spec.enhanced_retrieval,
        )
        leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
        selected_leaves = [
            leaf_by_id[node_id]
            for node_id in retrieval.leaf_node_ids
            if node_id in leaf_by_id
        ]
        selected_roots = [
            root
            for root in retrieval_roots
            if root.node_id in set(retrieval.summary_node_ids)
        ]
        if selected_leaves:
            selected_leaves = _iterative_kick_and_backfill_leaves_with_llm(
                config=config,
                case=case,
                variant=variant,
                llm=llm,
                limiter=limiter,
                metrics=answer_metrics,
                llm_records=llm_records,
                selected_leaves=selected_leaves,
                ranked_pool_leaves=ranked_pool_leaves,
            )
            selected_roots, selected_leaves = _fit_context_budget(
                selected_roots,
                selected_leaves,
                _evidence_context_budget(config, case, spec.enhanced_qa),
            )
            retrieved_sessions = sorted({leaf.session_id for leaf in selected_leaves})
            answer_sessions = set(case.answer_session_ids)
            answer_session_count = len(set(retrieved_sessions) & answer_sessions)
            gold_answer_session_count = len(answer_sessions)
            retrieval.summary_node_ids = [root.node_id for root in selected_roots]
            retrieval.leaf_node_ids = [leaf.node_id for leaf in selected_leaves]
            retrieval.retrieved_session_ids = retrieved_sessions
            retrieval.answer_session_hit = bool(answer_session_count)
            retrieval.answer_session_all_hit = (
                gold_answer_session_count > 0 and answer_session_count == gold_answer_session_count
            )
            retrieval.answer_session_recall = (
                answer_session_count / gold_answer_session_count if gold_answer_session_count else 0.0
            )
            retrieval.retrieved_answer_session_count = answer_session_count
            retrieval.gold_answer_session_count = gold_answer_session_count
            retrieval.context_text = _context_text(selected_roots, selected_leaves)
            retrieval.latency_sec += time.perf_counter() - denoise_started
    answer_started = time.perf_counter()
    answer_context = retrieval.context_text
    raw_answer_context = answer_context
    answer_notes: list[dict[str, str]] = []
    answer_note_parse_error: str | None = None
    answer_used_notes = False
    if config.enable_compute_plan and _is_arithmetic_question(case):
        plan_result = _tracked_chat(
            llm,
            limiter,
            answer_metrics,
            question_id=case.question_id,
            variant=variant,
            stage="answer_qa",
            thinking_mode="none",
            messages=_compute_plan_messages(case, retrieval.context_text),
            max_tokens=config.qa_max_tokens,
            json_mode=True,
        )
        llm_records.append(plan_result.record)
        computed_block = _execute_compute_plan(_extract_json_object(plan_result.text or ""))
        if computed_block:
            answer_context = f"{retrieval.context_text}\n\n{computed_block}"
            raw_answer_context = answer_context
    if config.enable_answer_note_extraction:
        note_result = _tracked_chat(
            llm,
            limiter,
            answer_metrics,
            question_id=case.question_id,
            variant=variant,
            stage="answer_note_extraction",
            thinking_mode="none",
            messages=_answer_note_messages(case, answer_context),
            max_tokens=min(config.qa_max_tokens, config.answer_note_max_tokens),
            json_mode=True,
        )
        llm_records.append(note_result.record)
        answer_notes, answer_note_parse_error = _parse_answer_notes(note_result.text)
        if config.answer_use_notes_for_qa and answer_notes:
            answer_context = _answer_context_from_notes(
                case,
                answer_notes,
                raw_context=raw_answer_context,
                include_raw_context=config.answer_include_raw_context_with_notes,
            )
            answer_used_notes = True
    reference_date = _reference_date_from_retrieval(
        case,
        leaves,
        retrieval_roots,
        retrieval.leaf_node_ids,
        retrieval.summary_node_ids,
    )
    answer_result = _tracked_chat(
        llm,
        limiter,
        answer_metrics,
        question_id=case.question_id,
        variant=variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=_answer_messages(
            case,
            answer_context,
            enhanced=spec.enhanced_qa,
            reference_date=reference_date,
        ),
        max_tokens=config.qa_max_tokens,
    )
    llm_records.append(answer_result.record)
    answer_latency = time.perf_counter() - answer_started
    build_metrics = memory.metrics if include_build_records else BuildMetrics()
    stats = build_question_stats(
        question_id=case.question_id,
        variant=variant,
        session_count=len(case.haystack_session_ids),
        leaf_count=len(leaves),
        summary_count=len(summaries),
        edge_count=len(edges),
        records=llm_records,
        build_latency_sec=memory.build_latency_sec if include_build_records else 0.0,
        retrieval_latency_sec=retrieval.latency_sec,
        answer_latency_sec=answer_latency,
        answer_session_hit=retrieval.answer_session_hit,
        answer_session_all_hit=retrieval.answer_session_all_hit,
        answer_session_recall=retrieval.answer_session_recall,
        retrieved_answer_session_count=retrieval.retrieved_answer_session_count,
        gold_answer_session_count=retrieval.gold_answer_session_count,
        wall_time_sec=time.perf_counter() - case_started,
        summary_parse_error_count=build_metrics.summary_parse_error_count,
        summary_truncation_count=build_metrics.summary_truncation_count,
        ready_job_counts=build_metrics.ready_job_counts,
        peak_inflight_deepseek=max(
            build_metrics.peak_inflight_deepseek,
            answer_metrics.peak_inflight_deepseek,
        ),
    )
    return CaseRun(
        leaves=leaves,
        summaries=summaries,
        edges=edges,
        retrieval=retrieval,
        answer=answer_result.text,
        llm_records=llm_records,
        stats=stats,
        answer_notes=answer_notes,
        answer_note_parse_error=answer_note_parse_error,
        answer_used_notes=answer_used_notes,
    )


def _clone_memory_for_case(
    memory: MemoryBuild,
    question_id: str,
) -> tuple[list[LeafNode], list[SummaryNode], list[SummaryNode], list[GraphEdge]]:
    id_map: dict[str, str] = {}

    def remap(node_id: str) -> str:
        existing = id_map.get(node_id)
        if existing is not None:
            return existing
        suffix = node_id.split(":", 1)[1] if ":" in node_id else node_id
        mapped = f"{question_id}:{suffix}"
        id_map[node_id] = mapped
        return mapped

    leaves = [
        LeafNode(
            node_id=remap(leaf.node_id),
            question_id=question_id,
            session_id=leaf.session_id,
            session_date=leaf.session_date,
            turn_index=leaf.turn_index,
            raw_text=leaf.raw_text,
            user_text=leaf.user_text,
            message_count=leaf.message_count,
            retrieval_text=leaf.retrieval_text or leaf.raw_text,
            compact_facts=list(leaf.compact_facts),
            anchor_terms=dict(leaf.anchor_terms),
            embedding=list(leaf.embedding) if leaf.embedding is not None else None,
        )
        for leaf in memory.leaves
    ]
    summaries = [
        SummaryNode(
            node_id=remap(summary.node_id),
            question_id=question_id,
            session_id=summary.session_id,
            session_date=summary.session_date,
            level=summary.level,
            child_ids=[remap(node_id) for node_id in summary.child_ids],
            leaf_ids=[remap(node_id) for node_id in summary.leaf_ids],
            summary=summary.summary,
            retrieval_text=summary.retrieval_text or summary.summary,
            anchor_terms=summary.anchor_terms or _summary_anchor_terms(
                summary.parsed_summary,
                summary.raw_summary_text or summary.summary,
                summary.session_date,
            ),
            summary_mode=summary.summary_mode,
            summary_schema_version=summary.summary_schema_version,
            parsed_summary=summary.parsed_summary,
            raw_summary_text=summary.raw_summary_text,
            truncated=summary.truncated,
            parse_error=summary.parse_error,
            source_level=summary.source_level,
            embedding=list(summary.embedding) if summary.embedding is not None else None,
        )
        for summary in memory.summaries
    ]
    summary_by_id = {summary.node_id: summary for summary in summaries}
    root_ids = {remap(root.node_id) for root in memory.roots}
    roots = [summary_by_id[node_id] for node_id in root_ids if node_id in summary_by_id]
    edges = [
        GraphEdge(
            src=remap(edge.src), dst=remap(edge.dst), score=edge.score,
            relation=edge.relation, directed=edge.directed, confidence=edge.confidence,
            provenance=dict(edge.provenance), schema_version=edge.schema_version,
        )
        for edge in memory.edges
    ]
    return leaves, summaries, roots, edges


def _variant_spec(config: DemoConfig, variant: str) -> VariantSpec:
    spec = VARIANT_SPECS[variant]
    if config.tree_mode is not None and spec.tree_mode != "raw_rag":
        spec = VariantSpec(
            tree_mode=config.tree_mode,
            compression=spec.compression,
            graph=spec.graph,
            fanout_k=None,
            summary_max_tokens=spec.summary_max_tokens,
            summary_schema=spec.summary_schema,
            build_leaf_text=spec.build_leaf_text,
            retrieval_leaf_text=spec.retrieval_leaf_text,
            raw_question_types=spec.raw_question_types,
            hybrid_retrieval=spec.hybrid_retrieval,
            local_summary=spec.local_summary,
            default_summarizer_model=spec.default_summarizer_model,
            enhanced_retrieval=spec.enhanced_retrieval,
            enhanced_qa=spec.enhanced_qa,
        )
    if config.force_enhanced_retrieval or config.force_enhanced_qa:
        spec = replace(
            spec,
            enhanced_retrieval=spec.enhanced_retrieval or config.force_enhanced_retrieval,
            enhanced_qa=spec.enhanced_qa or config.force_enhanced_qa,
        )
    return spec


def _summary_schema(config: DemoConfig, spec: VariantSpec) -> str:
    return config.summary_schema or spec.summary_schema


def _summary_max_tokens(config: DemoConfig, spec: VariantSpec) -> int:
    return spec.summary_max_tokens or config.session_summary_max_tokens


def _uses_local_summary(config: DemoConfig, spec: VariantSpec) -> bool:
    if config.summarizer_kind == "qwen_local":
        return True
    if config.summarizer_kind in {"none", "llmlingua2"}:
        return False
    return spec.local_summary


def _leaf_text_mode(config_mode: str, variant_mode: str) -> str:
    return variant_mode if config_mode == "auto" else config_mode


def _effective_leaf_text_mode(config_mode: str, spec: VariantSpec, case: QuestionCase, phase: str) -> str:
    variant_mode = spec.build_leaf_text if phase == "build" else spec.retrieval_leaf_text
    mode = _leaf_text_mode(config_mode, variant_mode)
    if config_mode == "auto" and _has_explicit_speakers(case):
        return "raw"
    if config_mode == "auto" and case.question_type in spec.raw_question_types:
        return "raw"
    return mode


def _has_explicit_speakers(case: QuestionCase) -> bool:
    return any(
        bool(str(message.get("speaker", "")).strip())
        for session in case.haystack_sessions
        for message in session
    )


def _should_compress_summary_input(
    config: DemoConfig,
    spec: VariantSpec,
    case: QuestionCase,
) -> bool:
    """Skip LLMLingua before summarization when build uses raw leaf text.

    Raw dialogue is already the fidelity target; compressing it before the
    summary LLM was the main source of truncation/parse regressions (subset50:
    27 truncations with raw+compress vs 9 on the older shorter-input baseline).
    """
    if not spec.compression:
        return False
    if config.summarizer_kind == "none" or _uses_local_summary(config, spec):
        return False
    build_mode = _effective_leaf_text_mode(config.build_leaf_text, spec, case, phase="build")
    return build_mode != "raw"


def _leaf_text_attr(mode: str) -> str:
    return "user_text" if mode == "user_only" else "raw_text"


def _leaf_embedding_attr(config: DemoConfig, case: QuestionCase, mode: str) -> str:
    if mode == "user_only":
        return "user_text"
    if config.enable_speaker_retrieval_text and _has_explicit_speakers(case):
        return "retrieval_text"
    return "raw_text"


def _tracked_chat(
    llm: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    **kwargs: Any,
) -> Any:
    with limiter.track(metrics):
        return llm.chat(**kwargs)


def _build_summary_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    if spec.tree_mode == "direct_session":
        return _build_direct_session_roots(
            config,
            case,
            leaves,
            variant,
            spec,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
    return _build_legacy_kway_roots(
        config,
        case,
        leaves,
        variant,
        spec,
        llm,
        compressor,
        llm_records,
        metrics,
        limiter,
        summarizer,
        )


def _build_speaker_profile_roots(
    config: DemoConfig,
    case: QuestionCase,
    roots: list[SummaryNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
) -> list[SummaryNode]:
    speakers = _explicit_speakers(case)
    if not speakers or not roots or spec.tree_mode == "raw_rag":
        return []

    timeline = _speaker_profile_timeline(roots, config.max_group_rough_tokens)
    profiles: list[SummaryNode] = []
    for speaker in speakers:
        result = _tracked_chat(
            llm,
            limiter,
            metrics,
            question_id=case.question_id,
            variant=variant,
            stage="build_summary_speaker_profile",
            thinking_mode="none",
            messages=_speaker_profile_messages(speaker, timeline),
            max_tokens=max(512, _summary_max_tokens(config, spec)),
            json_mode=True,
        )
        llm_records.append(result.record)
        parsed, parse_error = _parse_summary(result.text, "compact_memory_v2")
        rendered_summary = _render_summary(parsed, result.text, "compact_memory_v2")
        leaf_ids: list[str] = []
        for root in roots:
            leaf_ids.extend(root.leaf_ids)
        profiles.append(
            SummaryNode(
                node_id=f"{case.question_id}:profile:{_safe_node_part(speaker)}",
                question_id=case.question_id,
                session_id=f"profile:{speaker}",
                session_date=case.question_date,
                level=99,
                child_ids=[root.node_id for root in roots],
                leaf_ids=sorted(set(leaf_ids)),
                summary=rendered_summary,
                retrieval_text=_summary_retrieval_text(
                    rendered_summary,
                    parsed,
                    timeline,
                    case.question_date,
                ),
                anchor_terms=_summary_anchor_terms(parsed, timeline, case.question_date),
                summary_mode="speaker_profile",
                summary_schema_version="compact_memory_v2",
                parsed_summary=parsed,
                raw_summary_text=result.text,
                truncated=result.record.finish_reason == "length",
                parse_error=parse_error,
                source_level=99,
            )
        )
    return profiles


def _explicit_speakers(case: QuestionCase) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for session in case.haystack_sessions:
        for message in session:
            speaker = str(message.get("speaker", "")).strip()
            if not speaker or speaker in seen:
                continue
            speakers.append(speaker)
            seen.add(speaker)
    return speakers


def _safe_node_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug or "speaker"


def _speaker_profile_timeline(roots: list[SummaryNode], rough_token_limit: int) -> str:
    chunks: list[str] = []
    total = 0
    for root in sorted(roots, key=lambda item: (item.session_date or "", item.session_id)):
        chunk = (
            f"[Session {root.session_id} | {root.session_date or 'unknown'}]\n"
            f"{root.summary}"
        )
        token_count = rough_token_count(chunk)
        if chunks and total + token_count > rough_token_limit:
            break
        chunks.append(chunk)
        total += token_count
    return "\n\n".join(chunks)


def _speaker_profile_messages(speaker: str, timeline: str) -> list[dict[str, str]]:
    system_prompt = (
        "Build a speaker-specific long-term memory profile from timeline summaries. "
        "Return JSON only: {\"m\":[\"short profile fact\"],\"k\":[\"keyword\"]}. "
        "Only include facts about the named speaker. Do not transfer facts from another speaker. "
        "Preserve stable identity, relationships, family, work or education goals, preferences, "
        "recurring activities, dated events, counts, and current status. Include uncertainty when "
        "the timeline does not explicitly support a value. Use at most 18 short m strings and "
        "12 keywords."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Speaker: {speaker}\n\nTimeline summaries:\n{timeline}",
        },
    ]


def _build_legacy_kway_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    all_summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    fanout_k = spec.fanout_k or config.fanout_k
    current: dict[str, list[LeafNode | SummaryNode]] = {
        session_id: list(session_leaves)
        for session_id, session_leaves in group_by_session(leaves).items()
    }
    level = 1
    while current:
        jobs: list[SummaryJob] = []
        for session_id, nodes in current.items():
            for group_number, children in enumerate(_chunks(nodes, fanout_k)):
                is_leaf_stage = isinstance(children[0], LeafNode)
                jobs.append(
                    SummaryJob(
                        session_id=session_id,
                        session_date=dates.get(session_id),
                        children=children,
                        stage="build_summary_leaf" if is_leaf_stage else "build_summary_internal",
                        level=level,
                        group_number=group_number,
                        summary_mode="legacy_kway",
                        max_tokens=(
                            config.raw_group_summary_max_tokens
                            if is_leaf_stage
                            else config.legacy_internal_summary_max_tokens
                        ),
                    )
                )
        next_nodes = _run_summary_jobs(
            config,
            case,
            variant,
            spec,
            jobs,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
            leaves_by_id={leaf.node_id: leaf for leaf in leaves},
        )
        grouped = _group_summaries_by_session(next_nodes)
        current = {}
        for session_id, session_nodes in grouped.items():
            all_summaries.extend(session_nodes)
            if len(session_nodes) == 1:
                roots.append(session_nodes[0])
            else:
                current[session_id] = session_nodes
        level += 1
    return all_summaries, roots


def _build_direct_session_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    all_summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    pending_merge: dict[str, list[SummaryNode]] = {}
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    fanout_k = spec.fanout_k or config.fanout_k
    session_summary_max_tokens = _summary_max_tokens(config, spec)
    build_leaf_mode = _effective_leaf_text_mode(config.build_leaf_text, spec, case, phase="build")
    first_jobs: list[SummaryJob] = []
    direct_session_ids: set[str] = set()
    for session_id, session_leaves in group_by_session(leaves).items():
        raw_groups = _raw_leaf_groups(
            session_leaves,
            fanout_k,
            config.max_group_rough_tokens,
            build_leaf_mode,
        )
        direct = len(raw_groups) == 1
        if direct:
            direct_session_ids.add(session_id)
        for group_number, children in enumerate(raw_groups):
            first_jobs.append(
                SummaryJob(
                    session_id=session_id,
                    session_date=dates.get(session_id),
                    children=children,
                    stage="build_summary_session_direct" if direct else "build_summary_raw_group",
                    level=1,
                    group_number=group_number,
                    summary_mode="direct_session",
                    max_tokens=(
                        session_summary_max_tokens
                        if direct
                        else config.raw_group_summary_max_tokens
                    ),
                )
            )
    first_nodes = _run_summary_jobs(
        config,
        case,
        variant,
        spec,
        first_jobs,
        llm,
        compressor,
        llm_records,
        metrics,
        limiter,
        summarizer,
        leaves_by_id={leaf.node_id: leaf for leaf in leaves},
    )
    all_summaries.extend(first_nodes)
    for session_id, session_nodes in _group_summaries_by_session(first_nodes).items():
        if session_id in direct_session_ids:
            roots.extend(session_nodes)
            continue
        pending_merge[session_id] = session_nodes

    level = 2
    while pending_merge:
        jobs: list[SummaryJob] = []
        for session_id, nodes in pending_merge.items():
            for group_number, children in enumerate(_chunks(nodes, fanout_k)):
                jobs.append(
                    SummaryJob(
                        session_id=session_id,
                        session_date=dates.get(session_id),
                        children=children,
                        stage="build_summary_session_merge",
                        level=level,
                        group_number=group_number,
                        summary_mode="direct_session",
                        max_tokens=session_summary_max_tokens,
                    )
                )
        merge_nodes = _run_summary_jobs(
            config,
            case,
            variant,
            spec,
            jobs,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
            leaves_by_id={leaf.node_id: leaf for leaf in leaves},
        )
        all_summaries.extend(merge_nodes)
        pending_merge = {}
        for session_id, session_nodes in _group_summaries_by_session(merge_nodes).items():
            if len(session_nodes) == 1:
                roots.append(session_nodes[0])
            else:
                pending_merge[session_id] = session_nodes
        level += 1
    return all_summaries, roots


def _run_summary_jobs(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    spec: VariantSpec,
    jobs: list[SummaryJob],
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
    *,
    leaves_by_id: dict[str, LeafNode] | None = None,
) -> list[SummaryNode]:
    if not jobs:
        return []
    metrics.ready_job_counts.append(
        {
            "level": jobs[0].level,
            "job_count": len(jobs),
            "stages": {
                stage: sum(job.stage == stage for job in jobs)
                for stage in sorted({job.stage for job in jobs})
            },
        }
    )
    worker_count = len(jobs) if config.summary_workers == 0 else min(config.summary_workers, len(jobs))
    if worker_count == 1:
        results = [
            _summarize_job(
                config,
                case,
                variant,
                spec,
                job,
                llm,
                compressor,
                metrics,
                limiter,
                summarizer,
                leaves_by_id=leaves_by_id,
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda job: _summarize_job(
                        config,
                        case,
                        variant,
                        spec,
                        job,
                        llm,
                        compressor,
                        metrics,
                        limiter,
                        summarizer,
                        leaves_by_id=leaves_by_id,
                    ),
                    jobs,
                )
            )
    for node, record in results:
        if record is not None:
            llm_records.append(record)
        metrics.summary_parse_error_count += int(node.parse_error is not None)
        metrics.summary_truncation_count += int(node.truncated)
    return [node for node, _ in results]


def _summary_needs_recovery(node: SummaryNode) -> bool:
    return node.truncated or node.parse_error is not None


def _job_children_are_splittable_leaves(job: SummaryJob) -> bool:
    return len(job.children) > 1 and all(isinstance(child, LeafNode) for child in job.children)


def _merge_compact_parsed_summaries(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, list[str]] = {"m": [], "k": []}
    seen: dict[str, set[str]] = {"m": set(), "k": set()}
    for parsed in (left, right):
        if not parsed:
            continue
        for key in ("m", "k"):
            for item in _summary_string_list(parsed.get(key)):
                if item in seen[key]:
                    continue
                seen[key].add(item)
                merged[key].append(item)
    return merged


def _merge_split_summary_nodes(
    left: SummaryNode,
    right: SummaryNode,
    job: SummaryJob,
    schema: str,
    child_text: str,
    *,
    config: DemoConfig,
    spec: VariantSpec,
    case: QuestionCase,
) -> SummaryNode:
    parsed = (
        _merge_compact_parsed_summaries(left.parsed_summary, right.parsed_summary)
        if schema == "compact_memory_v2"
        else left.parsed_summary or right.parsed_summary
    )
    summary_text = "\n\n".join(
        block
        for block in (left.raw_summary_text or "", right.raw_summary_text or "")
        if block.strip()
    )
    rendered_summary, summary_parse_error = _render_root_summary_body(
        config,
        spec,
        case,
        job,
        parsed=parsed,
        summary_text=summary_text,
        schema=schema,
    )
    merged_has_content = bool(rendered_summary.strip())
    return SummaryNode(
        node_id=(
            f"{left.question_id}:{job.session_id}:summary:"
            f"{job.stage}:l{job.level}:g{job.group_number}"
        ),
        question_id=left.question_id,
        session_id=job.session_id,
        session_date=job.session_date,
        level=job.level,
        child_ids=[*left.child_ids, *right.child_ids],
        leaf_ids=sorted(set(left.leaf_ids) | set(right.leaf_ids)),
        summary=rendered_summary,
        retrieval_text=_summary_retrieval_text(
            rendered_summary,
            parsed,
            child_text,
            job.session_date,
        ),
        anchor_terms=_summary_anchor_terms(parsed, child_text, job.session_date),
        summary_mode=job.summary_mode,
        summary_schema_version=schema,
        parsed_summary=parsed,
        raw_summary_text=summary_text,
        truncated=left.truncated or right.truncated,
        parse_error=summary_parse_error if not merged_has_content else None,
        source_level=job.level,
    )


def _lossless_root_summary_text(
    children: list[LeafNode | SummaryNode],
    leaf_text_mode: str,
    session_date: str | None,
) -> str:
    blocks: list[str] = []
    if session_date:
        blocks.append(f"Session date: {session_date}")
    for index, child in enumerate(children, start=1):
        if isinstance(child, LeafNode):
            blocks.append(f"[Child {index}]\n{_leaf_text(child, leaf_text_mode)}")
        elif child.summary.strip():
            blocks.append(child.summary.strip())
    return "\n\n".join(blocks)


def _render_root_summary_body(
    config: DemoConfig,
    spec: VariantSpec,
    case: QuestionCase,
    job: SummaryJob,
    *,
    parsed: dict[str, Any] | None,
    summary_text: str,
    schema: str,
    parse_error: str | None = None,
) -> tuple[str, str | None]:
    leaf_mode = _leaf_text_mode(config.build_leaf_text, spec.build_leaf_text)
    if config.enable_lossless_root_summary:
        body = _lossless_root_summary_text(job.children, leaf_mode, job.session_date)
        if body.strip():
            return body, None
        return _render_summary(parsed, summary_text, schema), parse_error
    return _render_summary(parsed, summary_text, schema), parse_error


def _summarize_job_once(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    spec: VariantSpec,
    job: SummaryJob,
    llm: Any,
    compressor: Any,
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
    *,
    max_tokens: int,
    leaves_by_id: dict[str, LeafNode] | None = None,
) -> tuple[SummaryNode, DeepSeekCallRecord | None]:
    schema = _summary_schema(config, spec)
    include_leaf_enrichment = _job_supports_leaf_enrichment(config, spec, job)
    child_text = _child_text(
        job.children,
        _leaf_text_mode(config.build_leaf_text, spec.build_leaf_text),
    )
    if _should_compress_summary_input(config, spec, case):
        child_text = compressor.compress(
            child_text,
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            chunk_rough_tokens=config.compressor_chunk_rough_tokens,
        )
    messages = _summary_messages(
        job.session_id,
        job.session_date,
        job.stage,
        child_text,
        schema,
        include_leaf_enrichment=include_leaf_enrichment,
    )
    if _uses_local_summary(config, spec):
        if summarizer is None:
            raise RuntimeError("qwen_local summarizer is required for this variant")
        summary_result = summarizer.summarize(
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            messages=messages,
            max_tokens=max_tokens,
            json_mode=True,
        )
        summary_text = summary_result.text
        record: DeepSeekCallRecord | None = None
        finish_reason = None
    else:
        result = _tracked_chat(
            llm,
            limiter,
            metrics,
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            thinking_mode="none",
            messages=messages,
            max_tokens=max_tokens,
            json_mode=True,
        )
        summary_text = result.text
        record = result.record
        finish_reason = result.record.finish_reason
    parsed, parse_error = _parse_summary(summary_text, schema)
    rendered_summary, summary_parse_error = _render_root_summary_body(
        config,
        spec,
        case,
        job,
        parsed=parsed,
        summary_text=summary_text,
        schema=schema,
        parse_error=parse_error,
    )
    if include_leaf_enrichment and leaves_by_id is not None:
        _apply_leaf_enrichment_from_parsed(job.children, parsed, leaves_by_id)
    effective_parse_error = summary_parse_error if summary_parse_error is not None else parse_error
    if config.enable_lossless_root_summary and rendered_summary.strip():
        effective_parse_error = None
    node = SummaryNode(
        node_id=(
            f"{case.question_id}:{job.session_id}:summary:"
            f"{job.stage}:l{job.level}:g{job.group_number}"
        ),
        question_id=case.question_id,
        session_id=job.session_id,
        session_date=job.session_date,
        level=job.level,
        child_ids=[child.node_id for child in job.children],
        leaf_ids=_leaf_ids(job.children),
        summary=rendered_summary,
        retrieval_text=_summary_retrieval_text(
            rendered_summary,
            parsed,
            child_text,
            job.session_date,
        ),
        anchor_terms=_summary_anchor_terms(parsed, child_text, job.session_date),
        summary_mode=job.summary_mode,
        summary_schema_version=schema,
        parsed_summary=parsed,
        raw_summary_text=summary_text,
        truncated=finish_reason == "length",
        parse_error=effective_parse_error,
        source_level=job.level,
    )
    return node, record


def _summarize_job(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    spec: VariantSpec,
    job: SummaryJob,
    llm: Any,
    compressor: Any,
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
    *,
    leaves_by_id: dict[str, LeafNode] | None = None,
) -> tuple[SummaryNode, DeepSeekCallRecord | None]:
    schema = _summary_schema(config, spec)
    child_text = _child_text(
        job.children,
        _leaf_text_mode(config.build_leaf_text, spec.build_leaf_text),
    )
    retry_cap = _summary_max_tokens(config, spec)
    node, record = _summarize_job_once(
        config,
        case,
        variant,
        spec,
        job,
        llm,
        compressor,
        metrics,
        limiter,
        summarizer,
        max_tokens=job.max_tokens,
        leaves_by_id=leaves_by_id,
    )
    if _summary_needs_recovery(node) and job.max_tokens < retry_cap:
        boosted_tokens = min(max(job.max_tokens * 2, retry_cap), retry_cap)
        if boosted_tokens > job.max_tokens:
            retry_node, retry_record = _summarize_job_once(
                config,
                case,
                variant,
                spec,
                job,
                llm,
                compressor,
                metrics,
                limiter,
                summarizer,
                max_tokens=boosted_tokens,
                leaves_by_id=leaves_by_id,
            )
            previous = (0 if _summary_needs_recovery(node) else 1, len(node.summary or ""))
            candidate = (0 if _summary_needs_recovery(retry_node) else 1, len(retry_node.summary or ""))
            if candidate >= previous:
                node, record = retry_node, retry_record or record
    if _summary_needs_recovery(node) and _job_children_are_splittable_leaves(job):
        midpoint = len(job.children) // 2
        left_job = replace(
            job,
            children=job.children[:midpoint],
            max_tokens=min(job.max_tokens, retry_cap),
        )
        right_job = replace(
            job,
            children=job.children[midpoint:],
            max_tokens=min(job.max_tokens, retry_cap),
        )
        left_node, left_record = _summarize_job(
            config,
            case,
            variant,
            spec,
            left_job,
            llm,
            compressor,
            metrics,
            limiter,
            summarizer,
            leaves_by_id=leaves_by_id,
        )
        right_node, right_record = _summarize_job(
            config,
            case,
            variant,
            spec,
            right_job,
            llm,
            compressor,
            metrics,
            limiter,
            summarizer,
            leaves_by_id=leaves_by_id,
        )
        node = _merge_split_summary_nodes(
            left_node,
            right_node,
            job,
            schema,
            child_text,
            config=config,
            spec=spec,
            case=case,
        )
        record = right_record or left_record or record
    return node, record


def _retrieve(
    *,
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    embedder: Any,
    graph_enabled: bool,
    hybrid_retrieval: bool,
    enhanced_retrieval: bool,
    enhanced_qa: bool,
) -> RetrievedContext:
    started = time.perf_counter()
    query_vector = embedder.embed([case.question], question_id=case.question_id, variant=variant)[0]
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    selected_roots: list[SummaryNode] = []
    selected_leaves: list[LeafNode]
    retrieval_edges: list[GraphEdge] = []
    protected_leaf_ids: set[str] = set()

    if hybrid_retrieval:
        selected_roots, selected_leaves, retrieval_edges, protected_leaf_ids = _retrieve_hybrid(
            config,
            case,
            leaves,
            roots,
            edges,
            query_vector,
            graph_enabled,
            enhanced_retrieval,
        )
    elif roots:
        ranked_roots = _rank_roots_for_config(config, roots, query_vector, case.question)
        root_ids = [root.node_id for root in ranked_roots[: config.root_top_k]]
        if graph_enabled:
            root_ids, retrieval_edges = _expand_root_ids(root_ids, edges, config.graph_neighbor_k)
        selected_roots = [root_by_id[node_id] for node_id in root_ids if node_id in root_by_id]
        candidate_leaf_ids = {leaf_id for root in selected_roots for leaf_id in root.leaf_ids}
        candidate_leaves = [leaf_by_id[node_id] for node_id in candidate_leaf_ids if node_id in leaf_by_id]
        ranked_leaves = _rank_leaves_for_config(
            config,
            candidate_leaves or leaves,
            query_vector,
            case.question,
            enhanced=enhanced_retrieval,
        )
        if config.enable_coverage_rerank:
            ranked_leaves = _coverage_rerank_leaves(
                ranked_leaves,
                query_vector,
                case.question,
                enhanced=enhanced_retrieval,
                lambda_weight=config.coverage_rerank_lambda,
                pool_k=config.coverage_rerank_pool_k,
            )
        ranked_leaves = _apply_dual_channel_merge_for_config(
            config,
            ranked_leaves,
            candidate_leaves or leaves,
            case.question,
            case.question_type,
        )
        selected_leaves = ranked_leaves[: _effective_leaf_top_k(config, case)]
        if config.enable_leaf_graph_expansion and graph_enabled:
            selected_leaves, leaf_graph_edges = _apply_leaf_graph_expansion(
                config,
                case,
                selected_leaves,
                edges,
                leaf_by_id,
            )
            retrieval_edges.extend(leaf_graph_edges)
    else:
        ranked_leaves = _rank_leaves_for_config(
            config,
            leaves,
            query_vector,
            case.question,
            enhanced=enhanced_retrieval,
        )
        if config.enable_coverage_rerank:
            ranked_leaves = _coverage_rerank_leaves(
                ranked_leaves,
                query_vector,
                case.question,
                enhanced=enhanced_retrieval,
                lambda_weight=config.coverage_rerank_lambda,
                pool_k=config.coverage_rerank_pool_k,
            )
        ranked_leaves = _apply_dual_channel_merge_for_config(
            config,
            ranked_leaves,
            leaves,
            case.question,
            case.question_type,
        )
        selected_leaves = ranked_leaves[: _effective_leaf_top_k(config, case)]
        if config.enable_leaf_graph_expansion and graph_enabled:
            selected_leaves, leaf_graph_edges = _apply_leaf_graph_expansion(
                config,
                case,
                selected_leaves,
                edges,
                leaf_by_id,
            )
            retrieval_edges.extend(leaf_graph_edges)
    selected_roots, selected_leaves = _fit_context_budget(
        selected_roots,
        selected_leaves,
        _evidence_context_budget(config, case, enhanced_qa),
        protected_leaf_ids=protected_leaf_ids if config.graph_search_protect_leaves else None,
    )
    retrieved_sessions = sorted({leaf.session_id for leaf in selected_leaves})
    answer_sessions = set(case.answer_session_ids)
    answer_session_count = len(set(retrieved_sessions) & answer_sessions)
    gold_answer_session_count = len(answer_sessions)
    answer_session_recall = (
        answer_session_count / gold_answer_session_count if gold_answer_session_count else 0.0
    )
    context = _context_text(selected_roots, selected_leaves)
    return RetrievedContext(
        question_id=case.question_id,
        variant=variant,
        summary_node_ids=[root.node_id for root in selected_roots],
        leaf_node_ids=[leaf.node_id for leaf in selected_leaves],
        edge_count=len(retrieval_edges),
        context_text=context,
        answer_session_hit=bool(answer_session_count),
        retrieved_session_ids=retrieved_sessions,
        latency_sec=time.perf_counter() - started,
        answer_session_all_hit=(
            gold_answer_session_count > 0 and answer_session_count == gold_answer_session_count
        ),
        answer_session_recall=answer_session_recall,
        retrieved_answer_session_count=answer_session_count,
        gold_answer_session_count=gold_answer_session_count,
    )


def _retrieve_hybrid(
    config: DemoConfig,
    case: QuestionCase | str,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    query_vector: list[float],
    graph_enabled: bool,
    enhanced: bool,
) -> tuple[list[SummaryNode], list[LeafNode], list[GraphEdge], set[str]]:
    question = case.question if isinstance(case, QuestionCase) else case
    question_type = case.question_type if isinstance(case, QuestionCase) else ""
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    roots_by_session = {root.session_id: root for root in roots}
    ranked_roots = _rank_roots_for_config(config, roots, query_vector, question)
    rank_leaves_fn = lambda leaves, query_vector, question, enhanced=enhanced: _rank_leaves_for_config(
        config,
        leaves,
        query_vector,
        question,
        enhanced=enhanced,
    )

    if config.enable_graph_first_retrieval and graph_enabled:
        graph_result = graph_first_retrieve(
            leaves=leaves,
            roots=roots,
            edges=edges,
            query_vector=query_vector,
            question=question,
            search_config=_graph_first_config(config, case),
            rank_leaves_fn=rank_leaves_fn,
            enhanced=enhanced,
        )
        return _graph_retrieval_return(
            config,
            case,
            graph_result,
            leaves,
            question,
            question_type,
            enhanced=enhanced,
        )

    if config.enable_graph_search and graph_enabled:
        retrieval_edges: list[GraphEdge] = []
        if config.graph_search_seed_only:
            graph_leaves = leaves
        else:
            root_ids = [root.node_id for root in ranked_roots[: config.root_candidate_k]]
            root_ids, retrieval_edges = _expand_root_ids(root_ids, edges, config.graph_neighbor_k)
            candidate_leaf_ids: set[str] = set()
            for root_id in root_ids:
                root = root_by_id.get(root_id)
                if root is not None:
                    candidate_leaf_ids.update(root.leaf_ids)
            global_leaf_ids = _global_leaf_ids_for_hybrid(
                config,
                case,
                leaves,
                query_vector,
                question,
                enhanced=enhanced,
                rank_leaves_fn=rank_leaves_fn,
            )
            graph_leaves = [
                leaf_by_id[leaf_id]
                for leaf_id in candidate_leaf_ids | global_leaf_ids
                if leaf_id in leaf_by_id
            ] or leaves
        graph_result = graph_search_retrieve(
            leaves=graph_leaves,
            roots=roots,
            edges=edges,
            query_vector=query_vector,
            question=question,
            search_config=_graph_search_config(config, case),
            rank_leaves_fn=rank_leaves_fn,
            enhanced=enhanced,
        )
        return _graph_retrieval_return(
            config,
            case,
            graph_result,
            leaves,
            question,
            question_type,
            enhanced=enhanced,
            retrieval_edges=retrieval_edges,
        )

    root_ids = [root.node_id for root in ranked_roots[: config.root_candidate_k]]
    retrieval_edges: list[GraphEdge] = []
    if graph_enabled:
        root_ids, retrieval_edges = _expand_root_ids(root_ids, edges, config.graph_neighbor_k)
    candidate_leaf_ids: set[str] = set()
    for root_id in root_ids:
        root = root_by_id.get(root_id)
        if root is not None:
            candidate_leaf_ids.update(root.leaf_ids)
    global_leaf_ids = _global_leaf_ids_for_hybrid(
        config,
        case,
        leaves,
        query_vector,
        question,
        enhanced=enhanced,
        rank_leaves_fn=rank_leaves_fn,
    )
    candidate_leaves = [
        leaf_by_id[leaf_id]
        for leaf_id in candidate_leaf_ids | global_leaf_ids
        if leaf_id in leaf_by_id
    ]
    ranked_leaves = _coverage_rerank_for_config(
        config,
        rank_leaves_fn(candidate_leaves or leaves, query_vector, question, enhanced=enhanced),
        query_vector,
        question,
        enhanced=enhanced,
    )
    ranked_leaves = _apply_dual_channel_merge_for_config(
        config,
        ranked_leaves,
        candidate_leaves or leaves,
        question,
        question_type,
    )
    seed_roots = (
        [root_by_id[root_id] for root_id in root_ids if root_id in root_by_id]
        if enhanced
        else ranked_roots[: config.root_top_k]
    )
    root_seed_leaves = _root_seed_leaves(
        seed_roots,
        leaf_by_id,
        query_vector,
        question,
        enhanced,
    )
    selected_leaves = _diversify_leaves(
        ranked_leaves,
        limit=_effective_leaf_top_k(config, case),
        per_session_k=_effective_per_session_leaf_k(config, case),
        seed_leaves=root_seed_leaves,
    )
    if enhanced:
        selected_leaves = _expand_leaves_if_enhanced(
            config,
            case,
            selected_leaves,
            leaves,
            question,
            question_type,
            enhanced=enhanced,
        )
    context_sessions = {leaf.session_id for leaf in selected_leaves}
    summary_rank = {root.node_id: index for index, root in enumerate(ranked_roots)}
    selected_roots = sorted(
        [roots_by_session[session_id] for session_id in context_sessions if session_id in roots_by_session],
        key=lambda root: (summary_rank.get(root.node_id, len(summary_rank)), root.node_id),
    )
    if enhanced and len(selected_roots) < config.qa_summary_top_k:
        selected_root_ids = {root.node_id for root in selected_roots}
        for root in ranked_roots:
            if root.node_id in selected_root_ids:
                continue
            selected_roots.append(root)
            selected_root_ids.add(root.node_id)
            if len(selected_roots) >= config.qa_summary_top_k:
                break
    selected_roots = selected_roots[: config.qa_summary_top_k]
    if config.enable_leaf_graph_expansion and graph_enabled:
        selected_leaves, leaf_graph_edges = _apply_leaf_graph_expansion(
            config,
            case,
            selected_leaves,
            edges,
            leaf_by_id,
        )
        retrieval_edges.extend(leaf_graph_edges)
    return selected_roots, selected_leaves, retrieval_edges, set()


def _effective_leaf_top_k(config: DemoConfig, case: QuestionCase | str) -> int:
    effective = config.leaf_top_k
    if (
        config.enable_explicit_speaker_retrieval_boost
        and isinstance(case, QuestionCase)
        and _has_explicit_speakers(case)
    ):
        effective = max(effective, 24)
    if config.enable_query_type_retrieval_boost and isinstance(case, QuestionCase):
        if _is_list_or_set_question(case.question, case.question_type):
            effective += config.list_question_extra_leaf_budget
        elif _is_temporal_question(case.question, case.question_type):
            effective += config.temporal_question_extra_leaf_budget
    return effective


def _effective_global_leaf_top_k(config: DemoConfig, case: QuestionCase | str) -> int:
    if (
        config.enable_explicit_speaker_retrieval_boost
        and isinstance(case, QuestionCase)
        and _has_explicit_speakers(case)
    ):
        return max(config.global_leaf_top_k, 48)
    return config.global_leaf_top_k


def _effective_per_session_leaf_k(config: DemoConfig, case: QuestionCase | str) -> int:
    if (
        config.enable_explicit_speaker_retrieval_boost
        and isinstance(case, QuestionCase)
        and _has_explicit_speakers(case)
    ):
        return max(config.per_session_leaf_k, 3)
    return config.per_session_leaf_k


def _expand_selected_session_context(
    selected_leaves: list[LeafNode],
    leaves: list[LeafNode],
    question: str,
    question_type: str,
    limit: int,
    *,
    explicit_speaker: bool = False,
) -> list[LeafNode]:
    if not selected_leaves or limit <= len(selected_leaves) // 2:
        return selected_leaves
    grouped = group_by_session(leaves)
    expanded: list[LeafNode] = []
    seen: set[str] = set()

    def add(leaf: LeafNode) -> bool:
        if leaf.node_id in seen:
            return False
        expanded.append(leaf)
        seen.add(leaf.node_id)
        return len(expanded) >= limit

    if explicit_speaker:
        for leaf in selected_leaves:
            if add(leaf):
                return expanded
        sibling_sets: list[tuple[list[LeafNode], int]] = []
        for leaf in selected_leaves:
            siblings = sorted(grouped.get(leaf.session_id, []), key=lambda item: item.turn_index)
            index = next((idx for idx, item in enumerate(siblings) if item.node_id == leaf.node_id), -1)
            if index < 0:
                continue
            sibling_sets.append((siblings, index))
        for offset in (-1, 1, 2, 3):
            for siblings, index in sibling_sets:
                neighbor_index = index + offset
                if 0 <= neighbor_index < len(siblings) and add(siblings[neighbor_index]):
                    return expanded
        return expanded

    previous_chat = question_type == "single-session-assistant" or bool(
        re.search(
            r"previous (conversation|chat)|our previous|looking back|remind me what|finally decided",
            question,
            flags=re.IGNORECASE,
        )
    )
    if not previous_chat:
        return selected_leaves

    top_session = selected_leaves[0].session_id
    for sibling in sorted(grouped.get(top_session, []), key=lambda item: item.turn_index)[:8]:
        if add(sibling):
            return expanded

    for leaf in selected_leaves:
        if add(leaf):
            return expanded
        siblings = sorted(grouped.get(leaf.session_id, []), key=lambda item: item.turn_index)
        index = next((idx for idx, item in enumerate(siblings) if item.node_id == leaf.node_id), -1)
        if index < 0:
            continue
        window = siblings[max(0, index - 1) : min(len(siblings), index + 5)]
        for sibling in window:
            if add(sibling):
                return expanded

    for leaf in selected_leaves:
        if add(leaf):
            break
    return expanded


def _diversify_leaves(
    ranked_leaves: list[LeafNode],
    *,
    limit: int,
    per_session_k: int,
    seed_leaves: list[LeafNode] | None = None,
) -> list[LeafNode]:
    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    per_session: dict[str, int] = {}
    for leaf in seed_leaves or []:
        if leaf.node_id in selected_ids:
            continue
        if per_session.get(leaf.session_id, 0) >= per_session_k:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session[leaf.session_id] = per_session.get(leaf.session_id, 0) + 1
        if len(selected) >= limit:
            return selected
    for leaf in ranked_leaves:
        if leaf.node_id in selected_ids:
            continue
        if per_session.get(leaf.session_id, 0) >= per_session_k:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session[leaf.session_id] = per_session.get(leaf.session_id, 0) + 1
        if len(selected) >= limit:
            return selected
    for leaf in ranked_leaves:
        if leaf.node_id in selected_ids:
            continue
        selected.append(leaf)
        if len(selected) >= limit:
            break
    return selected


def _root_seed_leaves(
    roots: list[SummaryNode],
    leaf_by_id: dict[str, LeafNode],
    query_vector: list[float],
    question: str = "",
    enhanced: bool = False,
) -> list[LeafNode]:
    seed_leaves: list[LeafNode] = []
    for root in roots:
        children = [leaf_by_id[leaf_id] for leaf_id in root.leaf_ids if leaf_id in leaf_by_id]
        ranked_children = _rank_leaves(children, query_vector, question, enhanced=enhanced)
        if ranked_children:
            seed_leaves.append(ranked_children[0])
    return seed_leaves


def _root_graph_edge_policy(
    config: DemoConfig,
    *,
    enable_typed_edges: bool,
) -> RootGraphEdgePolicy:
    return RootGraphEdgePolicy(
        graph_neighbor_k=config.graph_neighbor_k,
        enable_typed_edges=enable_typed_edges,
        typed_neighbors_per_relation=config.typed_root_neighbors_per_relation,
        keyword_neighbors_per_root=config.graph_neighbor_k,
        semantic_neighbors_per_root=config.graph_neighbor_k,
        typed_min_score=config.typed_root_min_edge_score,
        typed_max_per_root=config.typed_root_max_edges_per_root,
        entity_min_shared_specific=config.typed_root_entity_min_shared_specific,
        entity_min_shared_generic=config.typed_root_entity_min_shared_generic,
        time_min_shared=config.typed_root_time_min_shared,
        state_min_shared=config.typed_root_state_min_shared,
        event_min_shared=config.typed_root_event_min_shared,
        typed_keyword_min_shared=config.typed_root_keyword_min_shared,
        update_min_actions=config.typed_root_update_min_actions,
        update_min_entities=config.typed_root_update_min_entities,
        corpus_keyword_min_shared=config.typed_root_corpus_keyword_min_shared,
        require_semantic_support=config.typed_root_require_semantic_support,
        semantic_support_min_cosine=config.typed_root_semantic_support_min_cosine,
        filter_generic_entities=config.typed_root_filter_generic_entities,
    )


def _build_root_graph(
    roots: list[SummaryNode],
    graph_neighbor_k: int,
    *,
    enable_typed_edges: bool = False,
    prefer_typed_edges: bool = False,
    edge_policy: RootGraphEdgePolicy | None = None,
) -> list[GraphEdge]:
    del prefer_typed_edges  # deprecated: typed edges are always additive now
    if edge_policy is None:
        edge_policy = RootGraphEdgePolicy(
            graph_neighbor_k=graph_neighbor_k,
            enable_typed_edges=enable_typed_edges,
            keyword_neighbors_per_root=graph_neighbor_k,
            semantic_neighbors_per_root=graph_neighbor_k,
        )
    return build_root_graph(roots, edge_policy)


def _build_llm_root_edges(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    roots: list[SummaryNode],
    llm: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
) -> list[GraphEdge]:
    llm_anchors: dict[str, dict[str, list[str]]] = {}
    for root in roots:
        result = _tracked_chat(
            llm,
            limiter,
            metrics,
            question_id=case.question_id,
            variant=variant,
            stage="build_root_edge_anchor",
            thinking_mode="none",
            messages=llm_root_anchor_messages(root),
            max_tokens=config.llm_root_edge_max_tokens,
            json_mode=True,
        )
        llm_records.append(result.record)
        parsed = parse_llm_root_anchors(
            result.text,
            max_items_per_key=config.llm_root_edge_anchor_limit,
        )
        if parsed:
            llm_anchors[root.node_id] = parsed
    return build_llm_anchor_edges(
        roots,
        llm_anchors,
        neighbors_per_relation=config.llm_root_edge_neighbors_per_relation,
        min_shared=config.llm_root_edge_min_shared,
    )


def _build_session_leaf_graph_edges(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    leaves: list[LeafNode],
    llm: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
) -> list[GraphEdge]:
    leaves_by_session = group_by_session(leaves)
    llm_edges_by_session: dict[str, list[GraphEdge]] = {}
    if config.enable_llm_leaf_edges:
        for session_id, session_leaves in leaves_by_session.items():
            if len(session_leaves) < 2:
                continue
            capped = sorted(session_leaves, key=lambda item: item.turn_index)[
                : config.llm_leaf_edge_max_leaves_per_session
            ]
            result = _tracked_chat(
                llm,
                limiter,
                metrics,
                question_id=case.question_id,
                variant=variant,
                stage="build_leaf_edge_anchor",
                thinking_mode="none",
                messages=llm_session_leaf_edge_messages(
                    session_id,
                    capped,
                    max_snippet_chars=config.llm_leaf_edge_max_snippet_chars,
                ),
                max_tokens=config.llm_leaf_edge_max_tokens,
                json_mode=True,
            )
            llm_records.append(result.record)
            valid_ids = {leaf.node_id for leaf in capped}
            parsed = parse_llm_leaf_edges(
                result.text,
                valid_leaf_ids=valid_ids,
                min_confidence=config.llm_leaf_edge_min_confidence,
                max_edges_per_leaf=config.llm_leaf_edge_max_edges_per_leaf,
                max_edges_per_session=config.llm_leaf_edge_max_edges_per_session,
            )
            if parsed:
                llm_edges_by_session[session_id] = parsed
    return build_session_leaf_edges(
        leaves_by_session,
        enable_deterministic=True,
        llm_edges_by_session=llm_edges_by_session,
    )


def _leaf_only_edges(edges: list[GraphEdge], leaf_by_id: dict[str, LeafNode]) -> list[GraphEdge]:
    return [
        edge
        for edge in edges
        if edge.src in leaf_by_id and edge.dst in leaf_by_id
    ]


def _expand_leaf_ids(
    seed_leaf_ids: list[str],
    edges: list[GraphEdge],
    leaf_by_id: dict[str, LeafNode],
    neighbor_k: int,
) -> tuple[list[str], list[GraphEdge]]:
    if not seed_leaf_ids or not edges:
        return list(seed_leaf_ids), []
    expanded = list(seed_leaf_ids)
    seen = set(seed_leaf_ids)
    used_edges: list[GraphEdge] = []
    neighbor_counts = {leaf_id: 0 for leaf_id in seed_leaf_ids}
    for edge in sorted(edges, key=_edge_expansion_sort_key, reverse=True):
        for source, destination in ((edge.src, edge.dst), (edge.dst, edge.src)):
            if source not in neighbor_counts:
                continue
            if destination in seen:
                continue
            source_leaf = leaf_by_id.get(source)
            destination_leaf = leaf_by_id.get(destination)
            if source_leaf is None or destination_leaf is None:
                continue
            if source_leaf.session_id != destination_leaf.session_id:
                continue
            if neighbor_counts[source] >= neighbor_k:
                continue
            expanded.append(destination)
            seen.add(destination)
            used_edges.append(edge)
            neighbor_counts[source] += 1
            neighbor_counts.setdefault(destination, 0)
    return expanded, used_edges


def _apply_leaf_graph_expansion(
    config: DemoConfig,
    case: QuestionCase,
    selected_leaves: list[LeafNode],
    edges: list[GraphEdge],
    leaf_by_id: dict[str, LeafNode],
) -> tuple[list[LeafNode], list[GraphEdge]]:
    if not config.enable_leaf_graph_expansion or config.leaf_graph_expansion_budget <= 0:
        return selected_leaves, []
    leaf_edges = _leaf_only_edges(edges, leaf_by_id)
    if not leaf_edges:
        return selected_leaves, []
    seed_ids = [leaf.node_id for leaf in selected_leaves]
    expanded_ids, used_edges = _expand_leaf_ids(
        seed_ids,
        leaf_edges,
        leaf_by_id,
        config.leaf_graph_neighbor_k,
    )
    selected_ids = {leaf.node_id for leaf in selected_leaves}
    base_limit = _effective_leaf_top_k(config, case)
    max_total = base_limit + config.leaf_graph_expansion_budget
    result = list(selected_leaves)
    added = 0
    for leaf_id in expanded_ids:
        if leaf_id in selected_ids:
            continue
        if len(result) >= max_total or added >= config.leaf_graph_expansion_budget:
            break
        leaf = leaf_by_id.get(leaf_id)
        if leaf is None:
            continue
        result.append(leaf)
        selected_ids.add(leaf_id)
        added += 1
    return result, used_edges


def _merge_graph_edges(base_edges: list[GraphEdge], extra_edges: list[GraphEdge]) -> list[GraphEdge]:
    merged = list(base_edges)
    seen: set[tuple[str, str, str]] = {
        (edge.src, edge.dst, edge.relation) for edge in merged
    }
    for edge in extra_edges:
        key = (edge.src, edge.dst, edge.relation)
        if key in seen:
            continue
        merged.append(edge)
        seen.add(key)
    return merged


def _expand_root_ids(
    root_ids: list[str],
    edges: list[GraphEdge],
    graph_neighbor_k: int,
) -> tuple[list[str], list[GraphEdge]]:
    expanded = list(root_ids)
    seen = set(root_ids)
    used_edges: list[GraphEdge] = []
    neighbor_counts = {root_id: 0 for root_id in root_ids}
    for edge in sorted(edges, key=_edge_expansion_sort_key, reverse=True):
        for source, destination in ((edge.src, edge.dst), (edge.dst, edge.src)):
            if source in neighbor_counts and destination not in seen:
                if neighbor_counts[source] >= graph_neighbor_k:
                    continue
                expanded.append(destination)
                seen.add(destination)
                used_edges.append(edge)
                neighbor_counts[source] += 1
    return expanded, used_edges


_EDGE_RELATION_EXPANSION_BONUS = {
    "keyword_neighbor": 0.06,
    "semantic_neighbor": 0.03,
    "temporal_neighbor": 0.01,
    "time_neighbor": 0.04,
    "state_neighbor": 0.05,
    "entity_neighbor": 0.04,
    "event_neighbor": 0.03,
    "update_neighbor": 0.02,
}


def _edge_expansion_sort_key(edge: GraphEdge) -> tuple[float, float, str, str, str]:
    return (
        edge.score + _EDGE_RELATION_EXPANSION_BONUS.get(edge.relation, 0.0),
        edge.score,
        edge.relation,
        edge.src,
        edge.dst,
    )


def _embed_nodes(
    nodes: list[Any], embedder: Any, question_id: str, variant: str, attr: str,
    target_attr: str = "embedding",
) -> None:
    if not nodes:
        return
    vectors = embedder.embed(
        [getattr(node, attr) for node in nodes], question_id=question_id, variant=variant
    )
    for node, vector in zip(nodes, vectors):
        setattr(node, target_attr, vector)


def _effective_typed_root_edges(config: DemoConfig) -> bool:
    if config.enable_typed_root_edges:
        return True
    if not config.enable_typed_retrieval:
        return False
    return config.enable_graph_first_retrieval or config.enable_graph_search


def _rank_roots_for_config(
    config: DemoConfig,
    roots: list[SummaryNode],
    query_vector: list[float],
    question: str,
) -> list[SummaryNode]:
    if config.enable_typed_retrieval and question.strip():
        return rank_roots_hybrid(
            roots,
            query_vector,
            question,
            embedding_blend=config.typed_retrieval_embedding_blend,
        )
    return _rank_nodes(roots, query_vector)


def _rank_nodes(nodes: list[Any], query_vector: list[float]) -> list[Any]:
    return sorted(
        nodes,
        key=lambda node: (cosine_similarity(node.embedding, query_vector), node.node_id),
        reverse=True,
    )


def _fusion_retrieval_config(config: DemoConfig) -> FusionRetrievalConfig:
    method = config.fusion_method if config.fusion_method in {"rrf", "weighted"} else "rrf"
    protect_k = (
        config.fusion_semantic_protect_k
        if config.enable_protected_fusion and config.enable_fusion_retrieval
        else 0
    )
    return FusionRetrievalConfig(
        method=method,
        rrf_k=config.fusion_rrf_k,
        weight_semantic=config.fusion_weight_semantic,
        weight_keyword=config.fusion_weight_keyword,
        weight_entity=config.fusion_weight_entity,
        query_adaptive_weights=config.fusion_query_adaptive_weights,
        protect_semantic_top_k=protect_k,
    )


def _leaf_scores_for_config(
    config: DemoConfig,
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
) -> dict[str, float]:
    if config.enable_fusion_retrieval:
        return compute_fusion_scores(
            leaves,
            query_vector,
            question,
            config=_fusion_retrieval_config(config),
        )
    query_terms = _important_query_terms(question)
    update_query = _is_update_sensitive_question(question)
    return {
        leaf.node_id: _leaf_rank_score(
            leaf,
            query_vector,
            query_terms=query_terms,
            update_query=update_query,
            enhanced=enhanced,
        )
        for leaf in leaves
    }


def _rank_leaves_for_config(
    config: DemoConfig,
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
) -> list[LeafNode]:
    if not config.enable_fusion_retrieval and not enhanced:
        return _rank_nodes(leaves, query_vector)
    scores = _leaf_scores_for_config(
        config, leaves, query_vector, question, enhanced=enhanced
    )
    return sorted(
        leaves,
        key=lambda leaf: (scores.get(leaf.node_id, 0.0), leaf.node_id),
        reverse=True,
    )


def _explicit_speaker_enhanced(config: DemoConfig, case: QuestionCase | str) -> bool:
    return (
        isinstance(case, QuestionCase)
        and _has_explicit_speakers(case)
        and config.enable_speaker_neighbor_window
    )


def _expand_leaves_if_enhanced(
    config: DemoConfig,
    case: QuestionCase | str,
    selected_leaves: list[LeafNode],
    leaves: list[LeafNode],
    question: str,
    question_type: str,
    *,
    enhanced: bool,
) -> list[LeafNode]:
    if not enhanced:
        return selected_leaves
    return _expand_selected_session_context(
        selected_leaves,
        leaves,
        question,
        question_type,
        _effective_leaf_top_k(config, case),
        explicit_speaker=_explicit_speaker_enhanced(config, case),
    )


def _graph_first_config(config: DemoConfig, case: QuestionCase | str) -> GraphFirstConfig:
    return GraphFirstConfig(
        seed_roots=config.graph_search_seed_roots,
        seed_leaves=config.graph_search_seed_leaves,
        ppr_damping=config.graph_search_ppr_damping,
        ppr_iterations=config.graph_search_ppr_iterations,
        embedding_blend=config.graph_first_embedding_blend,
        structural_root_leaf_weight=config.graph_search_structural_root_leaf_weight,
        leaf_limit=_effective_leaf_top_k(config, case),
        root_limit=config.qa_summary_top_k,
        global_leaf_top_k=_effective_global_leaf_top_k(config, case),
        per_session_leaf_k=_effective_per_session_leaf_k(config, case),
        session_coverage=config.graph_first_session_coverage,
        per_session_leaf_cap=config.graph_search_per_session_leaf_cap,
        max_activated_sessions=config.graph_search_max_sessions,
        candidate_pool_k=config.graph_first_candidate_pool_k,
        use_typed_retrieval=config.enable_typed_retrieval,
        typed_embedding_blend=config.typed_retrieval_embedding_blend,
    )


def _graph_search_config(config: DemoConfig, case: QuestionCase | str) -> GraphSearchConfig:
    return GraphSearchConfig(
        seed_roots=config.graph_search_seed_roots,
        seed_leaves=config.graph_search_seed_leaves,
        ppr_damping=config.graph_search_ppr_damping,
        ppr_iterations=config.graph_search_ppr_iterations,
        embedding_blend=config.graph_search_embedding_blend,
        session_min_leaves=config.graph_search_session_min_leaves,
        max_activated_sessions=config.graph_search_max_sessions,
        per_session_leaf_cap=config.graph_search_per_session_leaf_cap,
        leaf_limit=_effective_leaf_top_k(config, case),
        root_limit=config.qa_summary_top_k,
        structural_root_leaf_weight=config.graph_search_structural_root_leaf_weight,
        seed_only=config.graph_search_seed_only,
        free_leaf_select=config.graph_search_seed_only,
        session_coverage=config.graph_search_session_coverage,
        use_typed_retrieval=config.enable_typed_retrieval,
        typed_embedding_blend=config.typed_retrieval_embedding_blend,
    )


def _coverage_rerank_for_config(
    config: DemoConfig,
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
) -> list[LeafNode]:
    if not config.enable_coverage_rerank:
        return leaves
    relevance = (
        _leaf_scores_for_config(
            config,
            leaves[: config.coverage_rerank_pool_k],
            query_vector,
            question,
            enhanced=enhanced,
        )
        if config.enable_fusion_retrieval
        else None
    )
    return _coverage_rerank_leaves(
        leaves,
        query_vector,
        question,
        enhanced=enhanced,
        lambda_weight=config.coverage_rerank_lambda,
        pool_k=config.coverage_rerank_pool_k,
        relevance_scores=relevance,
    )


def _iterative_kick_and_backfill_leaves(
    selected_leaves: list[LeafNode],
    ranked_pool_leaves: list[LeafNode],
    relevance_scores: dict[str, float],
    *,
    max_rounds: int,
    max_kick_per_round: int,
    min_relevance_ratio: float,
    protect_top_k: int,
    protected_leaf_ids: set[str] | None = None,
) -> list[LeafNode]:
    if not selected_leaves:
        return selected_leaves
    if not ranked_pool_leaves:
        return selected_leaves

    target_k = len(selected_leaves)
    selected_ids = {leaf.node_id for leaf in selected_leaves}
    pool_by_id = {leaf.node_id: leaf for leaf in ranked_pool_leaves}
    protected_ids = set(protected_leaf_ids or set())
    for leaf in ranked_pool_leaves[:protect_top_k]:
        if leaf.node_id in selected_ids:
            protected_ids.add(leaf.node_id)

    max_score = max((relevance_scores.get(leaf.node_id, 0.0) for leaf in ranked_pool_leaves), default=0.0)
    threshold = max_score * min_relevance_ratio
    if threshold <= 0:
        return selected_leaves

    selected = list(selected_leaves)
    for _ in range(max_rounds):
        kick_candidates = [
            leaf
            for leaf in selected
            if leaf.node_id not in protected_ids
            and relevance_scores.get(leaf.node_id, 0.0) < threshold
        ]
        kick_candidates.sort(key=lambda leaf: relevance_scores.get(leaf.node_id, 0.0))
        kick = kick_candidates[:max_kick_per_round]
        if not kick:
            break
        kick_ids = {leaf.node_id for leaf in kick}

        selected = [leaf for leaf in selected if leaf.node_id not in kick_ids]
        selected_ids -= kick_ids

        refill: list[LeafNode] = []
        for leaf in ranked_pool_leaves:
            if len(refill) >= len(kick):
                break
            if leaf.node_id in selected_ids:
                continue
            refill.append(pool_by_id[leaf.node_id])
            selected_ids.add(leaf.node_id)

        if not refill:
            break
        selected.extend(refill)

        selected.sort(
            key=lambda leaf: (relevance_scores.get(leaf.node_id, 0.0), -leaf.turn_index, leaf.node_id),
            reverse=True,
        )
        selected = selected[:target_k]
        selected_ids = {leaf.node_id for leaf in selected}

    return selected


def _iterative_kick_and_backfill_leaves_with_llm(
    *,
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    llm_records: list[DeepSeekCallRecord],
    selected_leaves: list[LeafNode],
    ranked_pool_leaves: list[LeafNode],
) -> list[LeafNode]:
    if not selected_leaves or not ranked_pool_leaves:
        return selected_leaves

    target_k = len(selected_leaves)
    pool_rank = {leaf.node_id: index for index, leaf in enumerate(ranked_pool_leaves)}
    selected: list[LeafNode] = sorted(
        selected_leaves,
        key=lambda leaf: (pool_rank.get(leaf.node_id, 10**9), leaf.node_id),
    )
    selected_ids = {leaf.node_id for leaf in selected}
    protected_ids = {
        leaf.node_id
        for leaf in ranked_pool_leaves[: config.iterative_leaf_denoise_protect_top_k]
        if leaf.node_id in selected_ids
    }
    structured_selected = _rank_structured_leaf_channel(
        config,
        selected,
        case.question,
        case.question_type,
    )
    for leaf in structured_selected[: config.iterative_leaf_denoise_keep_structured_top_k]:
        protected_ids.add(leaf.node_id)

    for _ in range(config.iterative_leaf_denoise_max_rounds):
        payload = _llm_pick_kick_leaf_indices(
            case=case,
            variant=variant,
            llm=llm,
            limiter=limiter,
            metrics=metrics,
            llm_records=llm_records,
            selected_leaves=selected,
            max_kick_per_round=config.iterative_leaf_denoise_max_kick_per_round,
            protected_leaf_ids=protected_ids,
        )
        kick_indices = payload.get("kick_indices") or []
        kick_set = {
            int(index)
            for index in kick_indices
            if isinstance(index, int) and 0 <= index < len(selected)
        }
        if not kick_set:
            break
        kick_ids = {selected[index].node_id for index in sorted(kick_set)}
        kick_ids -= protected_ids
        if not kick_ids:
            break

        selected = [leaf for leaf in selected if leaf.node_id not in kick_ids]
        selected_ids = {leaf.node_id for leaf in selected}

        refill: list[LeafNode] = []
        for leaf in ranked_pool_leaves:
            if len(selected) + len(refill) >= target_k:
                break
            if leaf.node_id in selected_ids:
                continue
            refill.append(leaf)
            selected_ids.add(leaf.node_id)
        selected.extend(refill)
        selected = sorted(
            selected,
            key=lambda leaf: (pool_rank.get(leaf.node_id, 10**9), leaf.node_id),
        )[:target_k]
        selected_ids = {leaf.node_id for leaf in selected}

    return selected


def _llm_pick_kick_leaf_indices(
    *,
    case: QuestionCase,
    variant: str,
    llm: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    llm_records: list[DeepSeekCallRecord],
    selected_leaves: list[LeafNode],
    max_kick_per_round: int,
    protected_leaf_ids: set[str],
) -> dict[str, Any]:
    result = _tracked_chat(
        llm,
        limiter,
        metrics,
        question_id=case.question_id,
        variant=variant,
        stage="answer_retrieval_denoise",
        thinking_mode="none",
        messages=_retrieval_denoise_messages(
            case,
            selected_leaves,
            max_kick_per_round=max_kick_per_round,
            protected_leaf_ids=protected_leaf_ids,
        ),
        max_tokens=2048,
        json_mode=True,
    )
    llm_records.append(result.record)
    payload = _extract_json_object(result.text or "")
    if not isinstance(payload, dict):
        return {"kick_indices": []}
    if "kick_indices" not in payload:
        return {"kick_indices": []}
    raw_indices = payload.get("kick_indices")
    if not isinstance(raw_indices, list):
        return {"kick_indices": []}
    indices: list[int] = []
    for value in raw_indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(selected_leaves):
            continue
        indices.append(index)
    if len(indices) > max_kick_per_round:
        indices = indices[:max_kick_per_round]
    return {"kick_indices": sorted(set(indices))}


def _retrieval_denoise_messages(
    case: QuestionCase,
    selected_leaves: list[LeafNode],
    *,
    max_kick_per_round: int,
    protected_leaf_ids: set[str],
) -> list[dict[str, str]]:
    temporal_question = _is_temporal_question(case.question, case.question_type)
    list_question = _is_list_or_set_question(case.question, case.question_type)
    system_content = (
        "You are filtering noisy retrieval evidence for QA. "
        "Given the question and current selected leaves, identify leaves that are clearly irrelevant "
        "to answering the question. Kick at most the allowed number, and it is valid to kick none. "
        "Never kick protected leaves. "
        "Preserve evidence diversity: for temporal questions keep enough time-anchored leaves, and for "
        "list/set questions keep leaves covering different concrete items. "
        "Return strict JSON only with keys: kick_indices (array[int]) and rationale (string)."
    )
    rows = []
    for index, leaf in enumerate(selected_leaves):
        protected = "yes" if leaf.node_id in protected_leaf_ids else "no"
        structured_score = _structured_signal_score(leaf, case.question, case.question_type)
        tags: list[str] = []
        if structured_score > 0:
            tags.append("structured_match")
        text = (leaf.retrieval_text or leaf.raw_text or "").lower()
        if re.search(r"yesterday|today|last|week|month|year|before|after|ago|\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}", text):
            tags.append("time_anchor")
        if re.search(r",| and | or |including|plus|以及|还有|包括|、", text):
            tags.append("list_like")
        rows.append(
            {
                "index": index,
                "node_id": leaf.node_id,
                "protected": protected,
                "tags": tags,
                "session_id": leaf.session_id,
                "session_date": leaf.session_date or "",
                "text": _shorten_text_for_denoise(leaf.retrieval_text or leaf.raw_text, 420),
            }
        )
    user_content = json.dumps(
        {
            "question_id": case.question_id,
            "question_type": case.question_type,
            "question_date": case.question_date or "",
            "question": case.question,
            "question_mode": {
                "temporal_question": temporal_question,
                "list_or_set_question": list_question,
            },
            "max_kick_per_round": max_kick_per_round,
            "selected_leaves": rows,
            "rules": [
                "Kick only clearly irrelevant leaves.",
                "If uncertain, keep the leaf.",
                "Do not kick protected=yes leaves.",
                "kick_indices length must be <= max_kick_per_round.",
                "For temporal questions keep at least one time_anchor leaf if available.",
                "For list/set questions keep multiple list_like/structured_match leaves if available.",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _shorten_text_for_denoise(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _global_leaf_ids_for_hybrid(
    config: DemoConfig,
    case: QuestionCase | str,
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
    rank_leaves_fn: Any,
) -> set[str]:
    global_ranked = _coverage_rerank_for_config(
        config,
        rank_leaves_fn(leaves, query_vector, question, enhanced=enhanced),
        query_vector,
        question,
        enhanced=enhanced,
    )
    return {
        leaf.node_id
        for leaf in global_ranked[: _effective_global_leaf_top_k(config, case)]
    }


def _graph_retrieval_return(
    config: DemoConfig,
    case: QuestionCase | str,
    graph_result: Any,
    leaves: list[LeafNode],
    question: str,
    question_type: str,
    *,
    enhanced: bool,
    retrieval_edges: list[GraphEdge] | None = None,
) -> tuple[list[SummaryNode], list[LeafNode], list[GraphEdge], set[str]]:
    selected_leaves = _expand_leaves_if_enhanced(
        config,
        case,
        graph_result.selected_leaves,
        leaves,
        question,
        question_type,
        enhanced=enhanced,
    )
    edges = list(retrieval_edges or []) + list(graph_result.used_edges)
    return graph_result.selected_roots, selected_leaves, edges, set(graph_result.graph_leaf_ids)


def _rank_leaves(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
) -> list[LeafNode]:
    if not enhanced:
        return _rank_nodes(leaves, query_vector)
    query_terms = _important_query_terms(question)
    update_query = _is_update_sensitive_question(question)
    return sorted(
        leaves,
        key=lambda leaf: (
            _leaf_rank_score(
                leaf,
                query_vector,
                query_terms=query_terms,
                update_query=update_query,
                enhanced=enhanced,
            ),
            leaf.node_id,
        ),
        reverse=True,
    )


def _leaf_rank_score(
    leaf: LeafNode,
    query_vector: list[float],
    *,
    query_terms: set[str],
    update_query: bool,
    enhanced: bool,
) -> float:
    base = cosine_similarity(leaf.embedding, query_vector)
    if not enhanced:
        return base
    return (
        base
        + _lexical_overlap_score(leaf.raw_text, query_terms)
        + _update_signal_score(leaf.raw_text if update_query else "")
    )


def _coverage_rerank_leaves(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
    lambda_weight: float,
    pool_k: int,
    relevance_scores: dict[str, float] | None = None,
) -> list[LeafNode]:
    if len(leaves) < 2:
        return leaves
    if pool_k < len(leaves):
        rerank_pool = leaves[:pool_k]
        remainder = leaves[pool_k:]
    else:
        rerank_pool = leaves
        remainder = []
    query_terms = _important_query_terms(question)
    update_query = _is_update_sensitive_question(question)
    if relevance_scores is None:
        relevance = {
            leaf.node_id: _leaf_rank_score(
                leaf,
                query_vector,
                query_terms=query_terms,
                update_query=update_query,
                enhanced=enhanced,
            )
            for leaf in rerank_pool
        }
    else:
        relevance = {leaf.node_id: relevance_scores.get(leaf.node_id, 0.0) for leaf in rerank_pool}
    by_id = {leaf.node_id: leaf for leaf in rerank_pool}
    ordered_pool = sorted(
        rerank_pool, key=lambda leaf: (relevance[leaf.node_id], leaf.node_id), reverse=True
    )
    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    while len(selected) < len(ordered_pool):
        best_leaf: LeafNode | None = None
        best_score = float("-inf")
        for leaf in ordered_pool:
            if leaf.node_id in selected_ids:
                continue
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    cosine_similarity(leaf.embedding, chosen.embedding) for chosen in selected
                )
            mmr_score = lambda_weight * relevance[leaf.node_id] - (1.0 - lambda_weight) * redundancy
            if (
                mmr_score > best_score
                or (
                    mmr_score == best_score
                    and best_leaf is not None
                    and leaf.node_id > best_leaf.node_id
                )
            ):
                best_score = mmr_score
                best_leaf = leaf
        if best_leaf is None:
            break
        selected.append(by_id[best_leaf.node_id])
        selected_ids.add(best_leaf.node_id)
    if len(selected) == len(rerank_pool):
        return selected + remainder
    residual_pool = [leaf for leaf in rerank_pool if leaf.node_id not in selected_ids]
    return selected + residual_pool + remainder


def _is_temporal_question(question: str, question_type: str = "") -> bool:
    if question_type == "category_2":
        return True
    return bool(
        re.search(
            r"\bwhen\b|what time|date|yesterday|today|last week|last month|last friday|before|after|ago|weekend|month|year|哪天|什么时候|日期|上周|上个月|昨天|前",
            question,
            flags=re.IGNORECASE,
        )
    )


def _is_list_or_set_question(question: str, question_type: str = "") -> bool:
    if question_type == "category_1":
        return True
    return bool(
        re.search(
            r"what (activities|events|books|types|ways)|where has|in what ways|who supports|what does|which|list|哪些|什么活动|哪些活动|哪些事件|哪些书|哪些类型",
            question,
            flags=re.IGNORECASE,
        )
    )


def _structured_signal_score(leaf: LeafNode, question: str, question_type: str = "") -> float:
    text = (leaf.retrieval_text or leaf.raw_text or "").lower()
    score = 0.0
    terms = _important_query_terms(question)
    if terms:
        score += min(0.6, 0.08 * sum(term in text for term in terms))
    if _is_temporal_question(question, question_type):
        if re.search(
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}:\d{2}\b|yesterday|today|last|week|month|year|before|after|ago|昨天|上周|上个月|之前|之后",
            text,
            flags=re.IGNORECASE,
        ):
            score += 0.35
    if _is_list_or_set_question(question, question_type):
        if re.search(
            r",| and | or |also|including|plus|with|以及|还有|包括|和|、",
            text,
            flags=re.IGNORECASE,
        ):
            score += 0.2
    return score


def _rank_structured_leaf_channel(
    config: DemoConfig,
    candidate_leaves: list[LeafNode],
    question: str,
    question_type: str = "",
) -> list[LeafNode]:
    scored = [
        (leaf, _structured_signal_score(leaf, question, question_type))
        for leaf in candidate_leaves
    ]
    scored = [item for item in scored if item[1] > 0.0]
    scored.sort(key=lambda item: (item[1], item[0].node_id), reverse=True)
    return [leaf for leaf, _ in scored[: config.dual_channel_structured_pool_k]]


def _merge_ranked_leaf_channels(
    primary_ranked: list[LeafNode],
    structured_ranked: list[LeafNode],
) -> list[LeafNode]:
    if not structured_ranked:
        return primary_ranked
    primary_copy = list(primary_ranked)
    structured_copy = list(structured_ranked)
    merged: list[LeafNode] = []
    seen: set[str] = set()
    while primary_copy or structured_copy:
        for _ in range(2):
            if not primary_copy:
                break
            leaf = primary_copy.pop(0)
            if leaf.node_id in seen:
                continue
            merged.append(leaf)
            seen.add(leaf.node_id)
        if structured_copy:
            leaf = structured_copy.pop(0)
            if leaf.node_id not in seen:
                merged.append(leaf)
                seen.add(leaf.node_id)
    return merged


def _apply_dual_channel_merge_for_config(
    config: DemoConfig,
    ranked_leaves: list[LeafNode],
    candidate_leaves: list[LeafNode],
    question: str,
    question_type: str = "",
) -> list[LeafNode]:
    if not config.enable_dual_channel_candidate_merge:
        return ranked_leaves
    structured_ranked = _rank_structured_leaf_channel(
        config,
        candidate_leaves,
        question,
        question_type,
    )
    return _merge_ranked_leaf_channels(ranked_leaves, structured_ranked)


def _important_query_terms(question: str) -> set[str]:
    stop = {
        "how",
        "many",
        "much",
        "what",
        "when",
        "where",
        "did",
        "have",
        "the",
        "and",
        "for",
        "currently",
        "total",
        "recently",
        "多少",
        "几个",
        "什么",
        "当前",
        "现在",
        "最近",
        "总共",
    }
    terms = {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) > 2 and token.lower() not in stop
    }
    return terms


def _lexical_overlap_score(text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    lowered = text.lower()
    hits = sum(term in lowered for term in query_terms)
    return min(0.18, hits * 0.04)


def _is_update_sensitive_question(question: str) -> bool:
    return bool(
        re.search(
            r"currently|how many|how much|total|recent|since|now|current|arrive|arrival|leave|left|cost|spent|最近|当前|现在|多少|几个|总共|花了|到达|离开",
            question,
            flags=re.IGNORECASE,
        )
    )


def _update_signal_score(text: str) -> float:
    if not text:
        return 0.0
    patterns = (
        r"cancel|subscribe|subscription|currently|now|current|no longer|instead|changed",
        r"buy|bought|purchase|purchased|cost|spent|total|\$\d+",
        r"arrive|arrival|reach|reached|leave|left|\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
        r"取消|订阅|现在|当前|不再|购买|买了|花了|总共|到达|抵达|离开",
    )
    hits = sum(bool(re.search(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
    return min(0.24, hits * 0.06)


_SUMMARY_CUE_STOPWORDS = {
    "assistant",
    "build",
    "child",
    "counts",
    "date",
    "dates",
    "events",
    "facts",
    "keywords",
    "memory",
    "session",
    "summary",
    "updates",
    "user",
}

_SUMMARY_ANCHOR_KEYS = (
    "entities",
    "times",
    "quantities",
    "actions",
    "state_phrases",
    "keywords",
)


_SUMMARY_ACTION_CUES = (
    "accepted",
    "arrived",
    "arrival",
    "attended",
    "bought",
    "canceled",
    "cancelled",
    "changed",
    "completed",
    "cost",
    "current",
    "currently",
    "decided",
    "delivered",
    "flight",
    "joined",
    "left",
    "moved",
    "ordered",
    "planned",
    "purchased",
    "read",
    "recommended",
    "replaced",
    "returned",
    "spent",
    "subscribed",
    "subscription",
    "visited",
)


def _body_keyword_cues(text: str, *, limit: int = 16) -> list[str]:
    terms = sorted(
        _summary_term_set(text),
        key=lambda item: (-len(item), item),
    )
    return terms[:limit]


def _summary_retrieval_text(
    rendered_summary: str,
    parsed: dict[str, Any] | None,
    source_text: str,
    session_date: str | None,
) -> str:
    anchors = _summary_anchor_terms(parsed, source_text, session_date)
    cues = _summary_search_cues(parsed, source_text, anchors=anchors)
    anchor_text = _summary_anchor_text(anchors)
    blocks = []
    if session_date:
        blocks.append(f"Session date: {session_date}")
    if rendered_summary.strip():
        blocks.append(rendered_summary.strip())
    if anchor_text:
        blocks.append("Anchor terms:\n" + anchor_text)
    if cues:
        blocks.append("Search cues: " + "; ".join(cues))
    return "\n".join(blocks)


def _summary_anchor_terms(
    parsed: dict[str, Any] | None,
    source_text: str,
    session_date: str | None,
    limit_per_type: int = 32,
) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = {
        key: []
        for key in _SUMMARY_ANCHOR_KEYS
    }
    keyword_candidates: list[str] = []
    if parsed:
        for key in ("keywords", "k"):
            keyword_candidates.extend(_summary_string_list(parsed.get(key)))
    anchors["keywords"] = _dedupe_cues(keyword_candidates, limit_per_type)
    if len(anchors["keywords"]) < 4:
        anchors["keywords"] = _dedupe_cues(
            [*anchors["keywords"], *_body_keyword_cues(source_text)],
            limit_per_type,
        )
    anchors["entities"] = _dedupe_cues(_proper_name_cues(source_text), limit_per_type)
    time_candidates = _numeric_time_cues(source_text)
    if session_date:
        time_candidates.append(session_date)
    anchors["times"] = _dedupe_cues(time_candidates, limit_per_type)
    anchors["quantities"] = _dedupe_cues(_quantity_cues(source_text), limit_per_type)
    lowered = source_text.lower()
    anchors["actions"] = _dedupe_cues(
        [cue for cue in _SUMMARY_ACTION_CUES if cue in lowered],
        limit_per_type,
    )
    anchors["state_phrases"] = _dedupe_cues(
        [
            *_state_phrase_cues(source_text),
            *_speaker_attribute_cues(source_text),
            *_action_object_cues(source_text),
        ],
        limit_per_type,
    )
    return {key: value for key, value in anchors.items() if value}


def _summary_anchor_text(anchors: dict[str, list[str]]) -> str:
    lines: list[str] = []
    labels = {
        "entities": "Entities",
        "times": "Times",
        "quantities": "Quantities",
        "actions": "Actions",
        "state_phrases": "State phrases",
        "keywords": "Keywords",
    }
    for key in _SUMMARY_ANCHOR_KEYS:
        values = anchors.get(key) or []
        if values:
            lines.append(f"{labels[key]}: " + "; ".join(values[:16]))
    return "\n".join(lines)


def _summary_search_cues(
    parsed: dict[str, Any] | None,
    source_text: str,
    limit: int = 48,
    *,
    anchors: dict[str, list[str]] | None = None,
) -> list[str]:
    candidates: list[str] = []
    if anchors is not None:
        for key in _SUMMARY_ANCHOR_KEYS:
            candidates.extend(anchors.get(key) or [])
        return _dedupe_cues(candidates, limit)
    if parsed:
        for key in ("keywords", "k"):
            candidates.extend(_summary_string_list(parsed.get(key)))
    candidates.extend(_proper_name_cues(source_text))
    candidates.extend(_numeric_time_cues(source_text))
    candidates.extend(_quantity_cues(source_text))
    candidates.extend(_state_phrase_cues(source_text))
    lowered = source_text.lower()
    candidates.extend(cue for cue in _SUMMARY_ACTION_CUES if cue in lowered)
    return _dedupe_cues(candidates, limit)


def _proper_name_cues(text: str) -> list[str]:
    return proper_name_cues(text, stopwords=_SUMMARY_CUE_STOPWORDS)


def _state_phrase_cues(text: str) -> list[str]:
    cues: list[str] = []
    patterns = (
        r"\b(?:currently|now|still)\s+(?:reading|devouring|using|keeping|storing|wearing|owning|have|having)\s+([^.;\n|]{3,90})",
        r"\b(?:stored|keeping|kept|storing)\s+(?:it|them|my\s+[^.;\n|]{2,40}?)\s+(?:in|on|under|at)\s+([^.;\n|]{3,80})",
        r"\b(?:moved|switched|changed|replaced)\s+(?:to|into|from)\s+([^.;\n|]{3,80})",
        r"\b(?:subscribed to|canceled|cancelled|bought|purchased|ordered)\s+([^.;\n|]{3,80})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            cue = _normalize_summary_state_phrase(match.group(1))
            if cue:
                cues.append(cue)
    return cues


def _speaker_attribute_cues(text: str) -> list[str]:
    cues: list[str] = []
    speaker_pattern = r"[A-Z][A-Za-z.'-]{1,40}"
    verb_pattern = (
        r"is|was|has|had|wants?|likes?|loves?|enjoys?|prefers?|works|volunteers|"
        r"reads?|ran|went|moved|camped|attended|signed|joined|started|finished|"
        r"plans?|planned|studies|studying|pursues?|pursuing"
    )
    for match in re.finditer(
        rf"\b({speaker_pattern})\s+({verb_pattern})\b\s+([^.;\n|]{{2,90}})",
        text,
    ):
        speaker = match.group(1).strip()
        verb = match.group(2).strip()
        obj = _normalize_summary_state_phrase(match.group(3))
        if obj:
            cues.append(f"{speaker} {verb} {obj}")
    for match in re.finditer(
        rf"\b({speaker_pattern})'s\s+([^.;\n|]{{3,90}})",
        text,
    ):
        speaker = match.group(1).strip()
        attr = _normalize_summary_state_phrase(match.group(2))
        if attr:
            cues.append(f"{speaker}'s {attr}")
    return cues


def _action_object_cues(text: str) -> list[str]:
    cues: list[str] = []
    action_pattern = (
        r"went to|going to|go to|attended|ran|signed up for|planning(?: on)?|"
        r"moved from|moved to|read|recommended|camped|camping|visited|made|"
        r"painted|created|joined|started|finished|volunteered at|works at|"
        r"pursue|pursuing|studying"
    )
    for match in re.finditer(
        rf"\b({action_pattern})\b\s+([^.;\n|]{{3,90}})",
        text,
        flags=re.IGNORECASE,
    ):
        action = re.sub(r"\s+", " ", match.group(1).strip().lower())
        obj = _normalize_summary_state_phrase(match.group(2))
        if obj:
            cues.append(f"{action} {obj}")
    return cues


def _normalize_summary_state_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?*\"'()[]"))
    value = re.sub(
        r"\s+(?:and|but|because|which|that|while|so|by the way)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:my|a|an|the)\s+", "", value, flags=re.IGNORECASE)
    if len(value) < 3 or value.casefold() in _SUMMARY_CUE_STOPWORDS:
        return ""
    return value[:90]


def _numeric_time_cues(text: str) -> list[str]:
    patterns = (
        r"\$\s?\d+(?:[,.]\d+)?",
        r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b",
        r"\b\d{1,2}\s?(?:AM|PM|am|pm)\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
        r"\b\d+(?:[,.]\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
    )
    cues: list[str] = []
    for pattern in patterns:
        cues.extend(match.group(0).strip() for match in re.finditer(pattern, text))
    return cues


def _quantity_cues(text: str) -> list[str]:
    patterns = (
        r"\$\s?\d+(?:[,.]\d+)?",
        r"\b\d+(?:[,.]\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|years?|people|persons|items|tickets|books|episodes|classes|sessions|miles|km|kilometers?)\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:minutes?|hours?|days?|weeks?|months?|years?|people|persons|items|tickets|books|episodes|classes|sessions)\b",
    )
    cues: list[str] = []
    for pattern in patterns:
        cues.extend(match.group(0).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return cues


def _dedupe_cues(candidates: list[str], limit: int) -> list[str]:
    cues: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cue = re.sub(r"\s+", " ", str(candidate).strip(" \t\r\n.,;:"))
        if len(cue) < 2:
            continue
        key = cue.casefold()
        if key in seen or key in _SUMMARY_CUE_STOPWORDS:
            continue
        seen.add(key)
        cues.append(cue)
        if len(cues) >= limit:
            break
    return cues


def _summary_term_set(text: str) -> set[str]:
    terms = {
        token.casefold()
        for token in re.findall(r"[\w&.'-]+", text)
        if len(token) >= 3 and token.casefold() not in _SUMMARY_CUE_STOPWORDS
    }
    terms.update(cue.casefold() for cue in _numeric_time_cues(text))
    return terms


def _time_normalization_rule(
    session_date: str | None,
    *,
    for_leaf_facts: bool = False,
) -> str:
    if session_date and for_leaf_facts:
        return (
            "Resolve every fuzzy or relative time mention in leaf facts using the provided session "
            "date. Write facts with explicit YYYY-MM-DD date(s) as the primary time reference. "
            "Examples: 'Attended workshop on 2023-07-14' instead of 'Attended workshop yesterday'; "
            "'Trip during 2023-05-01 to 2023-05-07' for last week; 'Moved in 2022' for last year "
            "when the session date implies that calendar year. For ranges like last week or "
            "two weekends ago, write the resolved start/end dates. Do not leave relative-only "
            "time words in facts when session date is known."
        )
    if session_date:
        return (
            "Normalize fuzzy time mentions (e.g., today, yesterday, last week, this Friday, "
            "two weekends ago) using the provided session date. Keep the original fuzzy phrase "
            "and append the resolved date in parentheses when possible, for example "
            "'last week (2023-05-01 to 2023-05-07)' or 'yesterday (2023-05-11)'. Prefer "
            "explicit YYYY-MM-DD formatting for resolved dates."
        )
    return (
        "Preserve explicit time/date phrases; if session date is unknown, do not invent "
        "absolute dates from fuzzy time mentions."
    )


def _summary_messages(
    session_id: str,
    session_date: str | None,
    stage: str,
    child_text: str,
    schema: str,
    *,
    include_leaf_enrichment: bool = False,
) -> list[dict[str, str]]:
    time_normalization_rule = _time_normalization_rule(
        session_date,
        for_leaf_facts=include_leaf_enrichment and schema == "compact_memory_v2",
    )
    if schema == "multilingual_memory_v1":
        system_prompt = (
            "Extract multilingual user memory as JSON only with exactly these keys: "
            '{"facts":[],"events":[],"counts":[],"dates":[],"updates":[],"keywords":[]}. '
            "Use short atomic strings in the original language when possible. Preserve numbers, "
            "dates, times, costs, negations, cancellations, purchases, subscriptions, arrivals, "
            "departures, current state, and updates. Ignore assistant filler and generic advice. "
            "Use at most 8 facts/events total, 6 counts/dates/updates total, and 10 keywords. "
            + time_normalization_rule
        )
    elif schema == "compact_memory_v2":
        if include_leaf_enrichment:
            system_prompt = (
                'Extract memory as JSON only: {"m":["session-level memory fact"],'
                '"k":["session keyword"],'
                '"leaves":[{"i":1,"f":["short fact for child 1"],'
                '"k":["keyword"]}]}. '
                "For each [Child N] block, add one leaves entry with matching i=N. "
                "Each leaf f must capture user AND assistant facts from that child only. "
                "Session m/k are global highlights across all children. Keep user facts, "
                "preferences, plans, purchases, visits, events, counts, costs, dates, "
                "negations, and updates. Also keep assistant-provided answers, "
                "recommendations, named entities, methods, options, tables, numbers, and "
                "rubrics that could answer a later 'previous conversation' question. "
                "Use at most 16 m strings, 16 session k strings, 4 f and 6 k per leaf. "
                "Drop only unrelated filler and repeated wording. "
                + time_normalization_rule
            )
        else:
            system_prompt = (
                'Extract memory as JSON only: {"m":["short memory fact or update"],'
                '"k":["keyword"]}. Keep user facts, preferences, plans, purchases, visits, '
                "events, counts, costs, dates, negations, and updates. Also keep assistant-provided "
                "answers, recommendations, named entities, methods, options, tables, numbers, and "
                "rubrics that could answer a later 'previous conversation' question. Use at most 16 "
                "short atomic m strings and 16 keywords. Drop only unrelated filler and repeated wording. "
                + time_normalization_rule
            )
    else:
        system_prompt = (
            "Extract compact memory as a JSON object only. Use this schema exactly: "
            '{"compact_summary":"short session or group summary","facts":["short durable memory fact"],'
            '"updates":["short update or contradiction"],"time_anchors":["short temporal anchor"],'
            '"keywords":["keyword"]}. Use short strings. Keep user facts, preferences, plans, '
            "purchases, visits, events, updates, negations, and temporal anchors. Drop assistant "
            "filler and repeated wording. Return empty arrays when a list has no content. "
            + time_normalization_rule
        )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Build stage: {stage}\nSession: {session_id}\n"
                f"Date: {session_date or 'unknown'}\nChild memory:\n{child_text}"
            ),
        },
    ]


_ARITHMETIC_QUESTION_TYPES = {"temporal-reasoning"}

_ARITHMETIC_KEYWORDS = (
    "how many",
    "how much",
    "how long",
    "how old",
    "how often",
    "total",
    "sum",
    "average",
    "combined",
    "altogether",
    "in total",
    "number of",
    "count",
    "times",
    "days ago",
    "weeks ago",
    "months ago",
    "years ago",
    "how many days",
    "how many weeks",
    "how many months",
    "how many years",
    "since",
    "elapsed",
    "difference",
    "older",
    "younger",
    "longer",
    "shorter",
    "more than",
    "less than",
    "duration",
    "cost",
    "spend",
    "spent",
    "price",
    "percent",
)

# Supported deterministic operations for the Level-2 compute plan. Operands are supplied
# by the LLM (the values it read from the evidence); code only performs the math.
_COMPUTE_PLAN_OPS = (
    "diff_days",
    "elapsed_weeks",
    "elapsed_months",
    "add",
    "subtract",
    "multiply",
    "divide",
    "count",
    "min",
    "max",
)


def _is_arithmetic_question(case: QuestionCase) -> bool:
    if case.question_type in _ARITHMETIC_QUESTION_TYPES:
        return True
    text = (case.question or "").lower()
    return any(keyword in text for keyword in _ARITHMETIC_KEYWORDS)


def _compute_plan_messages(case: QuestionCase, context: str) -> list[dict[str, str]]:
    ops = ", ".join(_COMPUTE_PLAN_OPS)
    system_content = (
        "You convert a memory QA question into a deterministic compute plan. Read the "
        "question and evidence, then decide whether answering requires arithmetic "
        "(elapsed time between dates, sums, totals, counts, comparisons, differences). "
        "CODE will perform the math, so you must NOT compute results yourself \u2014 your only "
        "job is to pick the CORRECT operands from the evidence. Wrong operands are worse "
        "than no plan, so be careful.\n"
        "Return STRICT JSON only with this shape:\n"
        '{"steps": [{"op": "<op>", "label": "<short name>", "unit": "<unit>", '
        '"round": "<rounding>", "args": {...}}]}\n'
        f"Allowed op values: {ops}.\n"
        "EVERY operand must be an object that cites its evidence, NOT a bare value:\n"
        '  {"value": <date-or-number>, "source": "<verbatim quote or \'question date\'>"}\n'
        "The source must be a verbatim snippet copied from the evidence line you read the "
        "value from, and it MUST contain the EVENT/SUBJECT WORDS \u2014 not just a date or "
        "timestamp. A bare date like '2022/03/21 (Mon) 15:54' is NOT acceptable; quote the "
        "clause that names what happened, e.g. \"attended a baking class on 2022/03/21\". "
        "This is how the answer step verifies you picked the right event. If you cannot "
        'quote an explicit value tied to the question\'s subject, do NOT invent one \u2014 '
        'return {"steps": []}.\n'
        "Argument conventions:\n"
        '- diff_days/elapsed_weeks/elapsed_months: args {"a": <operand>, "b": <operand>}; '
        "dates as YYYY-MM-DD or exactly as shown. Result is |a-b|.\n"
        '- add/subtract/multiply/divide: args {"a": <operand>, "b": <operand>}; '
        'add also accepts {"values": [<operand>, ...]}.\n'
        '- count: args {"items": [<operand>, ...]} returns the list length.\n'
        '- min/max: args {"values": [<operand>, ...]}.\n'
        "Two extra rules that prevent the most common mistakes:\n"
        "1. ANCHOR: for any 'X ago', 'since', 'how long ago/until', or relative-time "
        "question, one operand MUST be the question date (use source \"question date\"); "
        "never use a session timestamp as the anchor. The other operand is the dated event "
        "that matches the question's subject \u2014 if several events have dates, do NOT just "
        "take the first or most recent; pick the one whose text actually matches the "
        "subject, and quote that subject text in source so the choice is auditable.\n"
        "2. UNIT/ROUND: set \"unit\" to the unit the question asks for (days, weeks, "
        "months, or years) and \"round\" to how the answer should be reported: use "
        "\"nearest\" for natural phrasing like 'how many weeks ago' (so 20 days -> 3 "
        "weeks), \"exact\" when a precise count is wanted, \"floor\" only if the question "
        "says 'full/complete'. Defaults: weeks/months -> nearest, days -> exact.\n"
        'If NO arithmetic is needed, return {"steps": []}. Return JSON only, no prose.'
    )
    user_content = (
        f"Question date: {case.question_date or 'unknown'}\n"
        f"Question: {case.question}\n\nRetrieved memory evidence:\n{context}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _operand_value(value: Any) -> Any:
    """Operands may be a bare scalar or an object {"value": ..., "source": ...}."""
    if isinstance(value, dict):
        return value.get("value")
    return value


def _operand_source(value: Any) -> str | None:
    if isinstance(value, dict):
        src = value.get("source")
        return str(src).strip() if src not in (None, "") else None
    return None


def _coerce_number(value: Any) -> float | None:
    value = _operand_value(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d[\d,]*\.?\d*", value.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _coerce_date(value: Any) -> date | None:
    value = _operand_value(value)
    if isinstance(value, str):
        return _parse_plan_date(value)
    return None


def _parse_plan_date(value: str | None) -> date | None:
    if not value:
        return None
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


_UNIT_DAYS = {"days": 1.0, "weeks": 7.0, "months": 30.44, "years": 365.25}

# Map the legacy op name to the unit it implies, so callers can keep using the old ops.
_OP_DEFAULT_UNIT = {"diff_days": "days", "elapsed_weeks": "weeks", "elapsed_months": "months"}


def _render_elapsed(days: int, unit: str, rounding: str) -> str:
    """Render an elapsed-time result, leading with the requested unit/rounding and
    always keeping the exact day count so the answer model can sanity-check."""
    unit = unit if unit in _UNIT_DAYS else "days"
    raw = days / _UNIT_DAYS[unit]
    if unit == "days":
        return f"{days} days"
    if rounding == "exact":
        primary = f"{raw:.2f} {unit}"
    elif rounding == "floor":
        primary = f"{int(raw)} {unit} (rounded down)"
    else:  # nearest (default for weeks/months/years)
        primary = f"~{round(raw)} {unit}"
    return f"{primary} (= {days} days exact; {raw:.2f} {unit})"


def _execute_compute_step(
    op: str, args: dict[str, Any], *, unit: str | None, rounding: str | None
) -> str | None:
    """Execute a single deterministic compute step. Returns a human-readable result
    string, or None if the operands are insufficient/invalid."""
    if op in {"diff_days", "elapsed_weeks", "elapsed_months"}:
        date_a = _coerce_date(args.get("a"))
        date_b = _coerce_date(args.get("b"))
        if date_a is None or date_b is None:
            return None
        days = abs((date_a - date_b).days)
        chosen_unit = unit or _OP_DEFAULT_UNIT.get(op, "days")
        chosen_round = rounding or ("exact" if chosen_unit == "days" else "nearest")
        return _render_elapsed(days, chosen_unit, chosen_round)
    if op in {"add", "min", "max"}:
        values = args.get("values")
        if values is None and op == "add":
            a = _coerce_number(args.get("a"))
            b = _coerce_number(args.get("b"))
            values = [v for v in (a, b) if v is not None]
        numbers = [n for n in (_coerce_number(v) for v in (values or [])) if n is not None]
        if not numbers:
            return None
        result = {"add": sum(numbers), "min": min(numbers), "max": max(numbers)}[op]
        return _format_number(result)
    if op in {"subtract", "multiply", "divide"}:
        a = _coerce_number(args.get("a"))
        b = _coerce_number(args.get("b"))
        if a is None or b is None:
            return None
        if op == "subtract":
            return _format_number(a - b)
        if op == "multiply":
            return _format_number(a * b)
        if b == 0:
            return None
        return _format_number(a / b)
    if op == "count":
        items = args.get("items")
        if not isinstance(items, list):
            return None
        return str(len(items))
    return None


def _operand_trace(args: dict[str, Any]) -> str:
    """Echo the operands (with their cited sources) so the answer model can verify them."""
    parts: list[str] = []
    for key in ("a", "b"):
        if key in args:
            raw = _operand_value(args[key])
            src = _operand_source(args[key])
            parts.append(f"{key}={raw}" + (f" [{src}]" if src else ""))
    for key in ("values", "items"):
        seq = args.get(key)
        if isinstance(seq, list):
            rendered = ", ".join(
                f"{_operand_value(v)}" + (f" [{_operand_source(v)}]" if _operand_source(v) else "")
                for v in seq
            )
            parts.append(f"{key}=[{rendered}]")
    return "; ".join(parts)


def _execute_compute_plan(plan: dict[str, Any] | None) -> str:
    """Render a deterministic results block from a parsed compute plan, or '' if empty."""
    if not isinstance(plan, dict):
        return ""
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return ""
    lines: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        op = str(step.get("op") or "").strip()
        if op not in _COMPUTE_PLAN_OPS:
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        unit = str(step.get("unit") or "").strip().lower() or None
        rounding = str(step.get("round") or "").strip().lower() or None
        result = _execute_compute_step(op, args, unit=unit, rounding=rounding)
        if result is None:
            continue
        label = str(step.get("label") or f"{op}_{index}").strip()
        trace = _operand_trace(args)
        line = f"- {label}: {result}"
        if trace:
            line += f"\n    operands: {trace}"
        lines.append(line)
    if not lines:
        return ""
    header = (
        "Precomputed results: CODE did the math exactly from the operands you supplied. "
        "Trust the arithmetic, but first VERIFY each operand below against the evidence "
        "(check the cited source matches the question's subject); if an operand is wrong, "
        "ignore that line and reason from the evidence instead."
    )
    return header + "\n" + "\n".join(lines)


def _answer_note_messages(case: QuestionCase, context: str) -> list[dict[str, str]]:
    system_content = (
        "Extract concise evidence notes for downstream answering. Return JSON object only with key "
        '"notes" as a list. Each note object must contain: '
        "session_id (string), date (string), fact (string), value (string), "
        "entities (array of strings), evidence_quote (string). "
        "Use empty strings or empty arrays if unknown. Keep only details relevant to the question. "
        "Do not infer unseen facts."
    )
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"Question: {case.question}\n\nRetrieved memory evidence:\n{context}"
            ),
        },
    ]


def _parse_answer_notes(text: str, *, max_notes: int = 24) -> tuple[list[dict[str, str]], str | None]:
    if not text.strip():
        return [], "empty_note_response"
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        payload = _extract_json_object(text)
        if payload is None:
            return [], f"invalid_json: {error}"
    if not isinstance(payload, dict):
        return [], "note_json_must_be_object"
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list):
        return [], "missing_notes_list"
    parsed: list[dict[str, str]] = []
    for entry in raw_notes:
        if not isinstance(entry, dict):
            continue
        fact = _summary_string(entry.get("fact"))
        if not fact:
            continue
        entities = _summary_string_list(entry.get("entities"))
        parsed.append(
            {
                "session_id": _summary_string(entry.get("session_id")),
                "date": _summary_string(entry.get("date")),
                "fact": fact,
                "value": _summary_string(entry.get("value")),
                "entities": ", ".join(entities[:6]),
                "evidence_quote": _summary_string(entry.get("evidence_quote")),
            }
        )
        if len(parsed) >= max_notes:
            break
    if not parsed:
        return [], "no_valid_notes"
    return parsed, None


def _answer_context_from_notes(
    case: QuestionCase,
    notes: list[dict[str, str]],
    *,
    raw_context: str,
    include_raw_context: bool,
) -> str:
    compact_notes = []
    for index, note in enumerate(notes, start=1):
        compact_notes.append(
            {
                "id": f"n{index}",
                "session_id": note.get("session_id", ""),
                "date": note.get("date", ""),
                "fact": note.get("fact", ""),
                "value": note.get("value", ""),
                "entities": note.get("entities", ""),
                "evidence_quote": note.get("evidence_quote", ""),
            }
        )
    note_block = json.dumps({"question": case.question, "notes": compact_notes}, ensure_ascii=False, indent=2)
    if include_raw_context:
        return (
            "Structured evidence notes (primary source for reasoning):\n"
            f"{note_block}\n\n"
            "Raw evidence (fallback only when notes are insufficient):\n"
            f"{raw_context}"
        )
    return "Structured evidence notes:\n" + note_block


def _answer_messages(
    case: QuestionCase,
    context: str,
    *,
    enhanced: bool = False,
    reference_date: str | None = None,
) -> list[dict[str, str]]:
    final_value_rule = (
        "Output format must be exactly two sections: (1) 'Evidence facts:' with brief supporting "
        "facts, then (2) one line 'Final answer: <value>'. The Final answer line must contain only "
        "the requested final value (or list/count/date span) with no extra explanation. For relative "
        "time/date answers, keep the original phrase and append resolved date(s) in parentheses, e.g. "
        "'the Friday before 15 July 2023 (2023-07-14)'."
    )
    if enhanced:
        system_content = (
            "Answer the user memory question from the supplied evidence only. First write a short "
            "'Evidence facts:' section with the facts you used, then write 'Final answer:'. For "
            "time, money, count, and total questions, explicitly perform the arithmetic or elapsed "
            "time calculation when the evidence provides the needed values. For current-state or "
            "knowledge-update questions, treat the latest known value as current unless later "
            "evidence contradicts it; do not say insufficient just because there is no newer update. "
            "Treat user memory statements as authoritative even if a later statement is a recollection; "
            "prefer the later dated value for update questions. If an earlier total is followed by a "
            "later addition and no later total is stated, add the increment to the earlier total. "
            "For relative date words such as today, yesterday, last Monday, last week, or next month, "
            "resolve them using the date shown in that evidence item's session header. For ordering "
            "questions, list each event with its date, then sort by date before writing the final order. "
            "For questions asking how many times an event happened, if the event is not mentioned, "
            "answer that the information is insufficient or not mentioned; do not convert absence of "
            "evidence into a numeric zero. "
            "If the evidence is in structured notes JSON, reason from those notes first and use "
            "evidence_quote/session_id fields for traceability. "
            "For preference or advice questions, use the evidence as user-specific constraints and "
            "give a useful personalized answer; do not require that the exact recommendation already "
            "appears in memory. Preserve negative preferences and avoidances, such as avoiding phone "
            "or TV use when the evidence says those hurt sleep. For questions about a previous "
            "conversation, assistant messages and "
            "assistant-provided tables, recommendations, names, methods, and numbers are valid "
            "evidence. If the user asks what was finally decided or chosen, prefer the last accepted "
            "or named option in the relevant conversation over earlier suggestions. If a required "
            "value is truly missing, say the information is insufficient and "
            "name what is missing. Do not invent unstated facts. Scan all supplied evidence before "
            "the final answer; do not ignore an explicit phrase like 'it cost me', 'my new X', or "
            "'my previous role as Y'. For previous-conversation questions, later turns override "
            "earlier drafts: a user's praise, acceptance, or repeated use of a name should be "
            "treated as the final choice, and a later named table should override an earlier table "
            "with generic Agent labels. For recommendation questions, if no exact local event or "
            "publication list is present, still answer with tailored categories, venues, search "
            "targets, or conference/publication areas grounded in the user's interests and "
            "avoidances; do not abstain just because the evidence lacks a live event calendar. "
            "For count and total questions, include every explicit service, brand, doctor, trip, "
            "or cost item mentioned in relevant evidence, including restaurants or delivery "
            "platforms used for convenience. Before giving a numeric count, write the counted "
            "items as a short list and check whether each item satisfies the wording of the "
            "question. Do not count recommendations, examples, budgets, price ranges, or future "
            "plans unless the question explicitly asks about planned items. For currently-own or "
            "currently-use questions, exclude items only considered, suggested, replaced, canceled, "
            "returned, or not yet acquired. For attended/visited/completed questions, exclude "
            "missed, planned, suggested, or merely discussed events. Count each explicitly named "
            "person, baby, item, device, appointment, trip, or event separately when the evidence "
            "states separate entities, including twins or multiple named items in the same turn. "
            "For holiday/date questions, if a session date is the "
            "holiday and the user describes a flight or event as today/recent in that session, use "
            "that dated evidence unless another retrieved fact directly contradicts it. "
            + final_value_rule
        )
    else:
        system_content = (
            "Answer the user memory question from the supplied evidence. If evidence is "
            "insufficient, say so. Compute direct counts, totals, elapsed times, and clock "
            "times when the evidence gives the needed values or time anchors. Keep the answer "
            "concise and state the evidence-based calculation when one is needed. If structured "
            "notes are provided, use them as the primary evidence source. "
            + final_value_rule
        )
    reference_line = ""
    if reference_date:
        reference_line = f"These conversations took place around {reference_date}.\n"
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"{reference_line}"
                f"Question: {case.question}\n\nRetrieved memory evidence:\n{context}"
            ),
        },
    ]


def _evidence_context_budget(config: DemoConfig, case: QuestionCase, enhanced_qa: bool) -> int:
    overhead = rough_token_count(
        "\n".join(message["content"] for message in _answer_messages(case, "", enhanced=enhanced_qa))
    )
    # Reserve enough room for completion plus tokenizer mismatch. The context
    # budget uses a rough local counter, while provider accounting is model-side.
    answer_margin = max(2400, config.qa_max_tokens + 1400)
    return max(1000, config.qa_context_token_budget - overhead - answer_margin)


def _leaf_enrichment_enabled(config: DemoConfig, spec: VariantSpec) -> bool:
    return config.enable_leaf_enrichment and _summary_schema(config, spec) == "compact_memory_v2"


def _job_supports_leaf_enrichment(
    config: DemoConfig,
    spec: VariantSpec,
    job: SummaryJob,
) -> bool:
    return _leaf_enrichment_enabled(config, spec) and all(
        isinstance(child, LeafNode) for child in job.children
    )


def _merge_string_lists(*lists: list[str], limit: int = 16) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for values in lists:
        for value in values:
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            merged.append(value)
            if len(merged) >= limit:
                return merged
    return merged


def _parse_leaf_enrichment_entries(
    parsed: dict[str, Any] | None,
) -> dict[int, tuple[list[str], list[str]]]:
    if not parsed:
        return {}
    entries = parsed.get("leaves")
    if not isinstance(entries, list):
        return {}
    result: dict[int, tuple[list[str], list[str]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        index_value = entry.get("i")
        if index_value is None:
            index_value = entry.get("index")
        if index_value is None:
            index_value = entry.get("child")
        try:
            child_index = int(index_value)
        except (TypeError, ValueError):
            continue
        facts = _summary_string_list(entry.get("f") or entry.get("facts"))
        keywords = _summary_string_list(entry.get("k") or entry.get("keywords"))
        if not facts and not keywords:
            continue
        previous = result.get(child_index, ([], []))
        result[child_index] = (
            _merge_string_lists(previous[0], facts),
            _merge_string_lists(previous[1], keywords),
        )
    return result


def _leaf_retrieval_text(
    raw_text: str,
    compact_facts: list[str],
    anchor_terms: dict[str, list[str]],
    session_date: str | None,
) -> str:
    blocks: list[str] = []
    if session_date:
        blocks.append(f"Session date: {session_date}")
    if compact_facts:
        blocks.append("Facts: " + "; ".join(compact_facts))
    anchor_text = _summary_anchor_text(anchor_terms)
    if anchor_text:
        blocks.append("Anchor terms:\n" + anchor_text)
    if raw_text.strip():
        blocks.append(raw_text.strip())
    return "\n".join(blocks)


_PARENTHESES_DATE_PATTERN = re.compile(
    r"\((\d{4}-\d{2}-\d{2}(?: to \d{4}-\d{2}-\d{2})?)\)"
)
_RELATIVE_TIME_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    (re.compile(r"\blast week\b", re.IGNORECASE), "last week"),
    (re.compile(r"\blast month\b", re.IGNORECASE), "last month"),
    (re.compile(r"\blast year\b", re.IGNORECASE), "last year"),
)


def _format_iso_date(value: date) -> str:
    return value.isoformat()


def _resolve_relative_time_phrase(phrase: str, anchor: date) -> str | None:
    normalized = phrase.casefold()
    if normalized == "today":
        return _format_iso_date(anchor)
    if normalized == "yesterday":
        return _format_iso_date(anchor - timedelta(days=1))
    if normalized == "last week":
        end = anchor - timedelta(days=1)
        start = end - timedelta(days=6)
        return f"{_format_iso_date(start)} to {_format_iso_date(end)}"
    if normalized == "last month":
        first_of_month = anchor.replace(day=1)
        end = first_of_month - timedelta(days=1)
        start = end.replace(day=1)
        return f"{_format_iso_date(start)} to {_format_iso_date(end)}"
    if normalized == "last year":
        return str(anchor.year - 1)
    return None


def _promote_parenthetical_dates_in_fact(fact: str) -> str:
    match = _PARENTHESES_DATE_PATTERN.search(fact)
    if not match:
        return fact
    resolved = match.group(1)
    without_parens = (fact[: match.start()] + fact[match.end() :]).strip()
    without_parens = re.sub(r"\s{2,}", " ", without_parens).strip(" ,;")
    if not without_parens:
        return resolved
    if resolved in without_parens:
        return without_parens
    return f"{without_parens} ({resolved})"


def _normalize_temporal_compact_facts(
    facts: list[str],
    session_date: str | None,
) -> list[str]:
    anchor = _parse_plan_date(session_date)
    normalized: list[str] = []
    for fact in facts:
        updated = _promote_parenthetical_dates_in_fact(fact)
        if anchor is not None:
            for pattern, label in _RELATIVE_TIME_FACT_PATTERNS:
                if not pattern.search(updated):
                    continue
                resolved = _resolve_relative_time_phrase(label, anchor)
                if not resolved:
                    continue
                if resolved in updated:
                    updated = pattern.sub(resolved, updated)
                else:
                    updated = pattern.sub(f"{label} ({resolved})", updated)
        normalized.append(updated.strip())
    return normalized


def _chronological_date_key(value: str | None) -> tuple[int, str]:
    parsed = _parse_plan_date(value)
    if parsed is None:
        return (1, value or "")
    return (0, parsed.isoformat())


def _sort_summaries_chronologically(summaries: list[SummaryNode]) -> list[SummaryNode]:
    return sorted(
        summaries,
        key=lambda summary: (
            _chronological_date_key(summary.session_date),
            summary.session_id,
        ),
    )


def _sort_leaves_chronologically(leaves: list[LeafNode]) -> list[LeafNode]:
    return sorted(
        leaves,
        key=lambda leaf: (
            _chronological_date_key(leaf.session_date),
            leaf.turn_index,
            leaf.session_id,
        ),
    )


def _latest_reference_date(
    case: QuestionCase,
    summaries: list[SummaryNode] | None = None,
    leaves: list[LeafNode] | None = None,
) -> str | None:
    candidates: list[date] = []
    for summary in summaries or []:
        parsed = _parse_plan_date(summary.session_date)
        if parsed is not None:
            candidates.append(parsed)
    for leaf in leaves or []:
        parsed = _parse_plan_date(leaf.session_date)
        if parsed is not None:
            candidates.append(parsed)
    if candidates:
        return _format_iso_date(max(candidates))
    return case.question_date or None


def _reference_date_from_retrieval(
    case: QuestionCase,
    leaves: list[LeafNode],
    summaries: list[SummaryNode],
    leaf_node_ids: list[str],
    summary_node_ids: list[str],
) -> str | None:
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    summary_by_id = {summary.node_id: summary for summary in summaries}
    selected_leaves = [leaf_by_id[node_id] for node_id in leaf_node_ids if node_id in leaf_by_id]
    selected_summaries = [
        summary_by_id[node_id] for node_id in summary_node_ids if node_id in summary_by_id
    ]
    return _latest_reference_date(case, selected_summaries, selected_leaves)


def _apply_leaf_enrichment_to_node(
    leaf: LeafNode,
    facts: list[str],
    keywords: list[str],
) -> None:
    facts = _normalize_temporal_compact_facts(facts, leaf.session_date)
    leaf.compact_facts = _merge_string_lists(leaf.compact_facts, facts, limit=8)
    rule_anchors = _summary_anchor_terms(None, leaf.raw_text, leaf.session_date)
    if keywords:
        rule_anchors["keywords"] = _merge_string_lists(
            rule_anchors.get("keywords", []),
            keywords,
            limit=12,
        )
    merged_anchors: dict[str, list[str]] = dict(leaf.anchor_terms)
    for key, values in rule_anchors.items():
        merged_anchors[key] = _merge_string_lists(
            merged_anchors.get(key, []),
            values,
            limit=16,
        )
    leaf.anchor_terms = {key: value for key, value in merged_anchors.items() if value}
    leaf.retrieval_text = _leaf_retrieval_text(
        leaf.raw_text,
        leaf.compact_facts,
        leaf.anchor_terms,
        leaf.session_date,
    )


def _apply_leaf_enrichment_from_parsed(
    children: list[LeafNode | SummaryNode],
    parsed: dict[str, Any] | None,
    leaves_by_id: dict[str, LeafNode],
) -> None:
    enrichment_by_index = _parse_leaf_enrichment_entries(parsed)
    if not enrichment_by_index:
        return
    for index, child in enumerate(children, start=1):
        if not isinstance(child, LeafNode):
            continue
        payload = enrichment_by_index.get(index)
        if payload is None:
            continue
        facts, keywords = payload
        leaf = leaves_by_id.get(child.node_id, child)
        _apply_leaf_enrichment_to_node(leaf, facts, keywords)


def _child_text(children: list[LeafNode | SummaryNode], leaf_text_mode: str) -> str:
    chunks = []
    for index, child in enumerate(children, start=1):
        text = _leaf_text(child, leaf_text_mode) if isinstance(child, LeafNode) else child.summary
        chunks.append(f"[Child {index}]\n{text}")
    return "\n\n".join(chunks)


def _leaf_text(leaf: LeafNode, mode: str) -> str:
    return leaf.user_text if mode == "user_only" else leaf.raw_text


def _leaf_ids(children: list[LeafNode | SummaryNode]) -> list[str]:
    ids: list[str] = []
    for child in children:
        ids.extend([child.node_id] if isinstance(child, LeafNode) else child.leaf_ids)
    return ids


def _raw_leaf_groups(
    leaves: list[LeafNode],
    fanout_k: int,
    max_group_rough_tokens: int,
    leaf_text_mode: str,
) -> list[list[LeafNode]]:
    groups: list[list[LeafNode]] = []
    current: list[LeafNode] = []
    current_tokens = 0
    for leaf in leaves:
        leaf_tokens = rough_token_count(_leaf_text(leaf, leaf_text_mode))
        would_exceed_budget = (
            max_group_rough_tokens > 0
            and current
            and current_tokens + leaf_tokens > max_group_rough_tokens
        )
        if len(current) >= fanout_k or would_exceed_budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(leaf)
        current_tokens += leaf_tokens
    if current:
        groups.append(current)
    return groups


def _group_summaries_by_session(nodes: list[SummaryNode]) -> dict[str, list[SummaryNode]]:
    grouped: dict[str, list[SummaryNode]] = {}
    for node in nodes:
        grouped.setdefault(node.session_id, []).append(node)
    return grouped


def _parse_summary(text: str, schema: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        extracted = _extract_json_object(text)
        if extracted is None:
            return None, f"invalid_json: {error}"
        payload = extracted
    if not isinstance(payload, dict):
        return None, "summary_json_must_be_object"

    if schema == "multilingual_memory_v1":
        parsed = {
            "facts": _summary_string_list(payload.get("facts")),
            "events": _summary_string_list(payload.get("events")),
            "counts": _summary_string_list(payload.get("counts")),
            "dates": _summary_string_list(payload.get("dates")),
            "updates": _summary_string_list(payload.get("updates")),
            "keywords": _summary_string_list(payload.get("keywords")),
        }
    elif schema == "compact_memory_v2":
        parsed: dict[str, Any] = {
            "m": _summary_string_list(payload.get("m")),
            "k": _summary_string_list(payload.get("k")),
        }
        leaf_entries = payload.get("leaves")
        if isinstance(leaf_entries, list):
            normalized_leaves: list[dict[str, Any]] = []
            for entry in leaf_entries:
                if not isinstance(entry, dict):
                    continue
                index_value = entry.get("i")
                if index_value is None:
                    index_value = entry.get("index")
                if index_value is None:
                    index_value = entry.get("child")
                try:
                    child_index = int(index_value)
                except (TypeError, ValueError):
                    continue
                facts = _summary_string_list(entry.get("f") or entry.get("facts"))
                keywords = _summary_string_list(entry.get("k") or entry.get("keywords"))
                if not facts and not keywords:
                    continue
                normalized_leaves.append({"i": child_index, "f": facts, "k": keywords})
            if normalized_leaves:
                parsed["leaves"] = normalized_leaves
    else:
        parsed = {
            "compact_summary": _summary_string(payload.get("compact_summary")),
            "facts": _summary_string_list(payload.get("facts")),
            "updates": _summary_string_list(payload.get("updates")),
            "time_anchors": _summary_string_list(payload.get("time_anchors")),
            "keywords": _summary_string_list(payload.get("keywords")),
        }
    return parsed, None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _summary_string(value: Any) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _summary_string_list(value: Any) -> list[str]:
    if isinstance(value, (str, int, float, bool)):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in (_summary_string(item) for item in value) if item]


def _render_summary(parsed: dict[str, Any] | None, raw_text: str, schema: str) -> str:
    if parsed is None:
        return raw_text.strip()
    if schema == "multilingual_memory_v1":
        blocks: list[str] = []
        for label, key in (
            ("Facts", "facts"),
            ("Events", "events"),
            ("Counts", "counts"),
            ("Dates", "dates"),
            ("Updates", "updates"),
            ("Keywords", "keywords"),
        ):
            values = parsed[key]
            if values:
                blocks.append(f"{label}: " + "; ".join(values))
        return "\n".join(blocks)
    if schema == "compact_memory_v2":
        memory = "; ".join(parsed["m"])
        keywords = "; ".join(parsed["k"])
        return "\n".join(
            block for block in (f"Memory: {memory}" if memory else "", f"Keywords: {keywords}" if keywords else "") if block
        )
    blocks: list[str] = []
    if parsed["compact_summary"]:
        blocks.append(parsed["compact_summary"])
    for label, key in (
        ("Facts", "facts"),
        ("Updates", "updates"),
        ("Times", "time_anchors"),
        ("Keywords", "keywords"),
    ):
        values = parsed[key]
        if values:
            blocks.append(f"{label}: " + "; ".join(values))
    return "\n".join(blocks)


def _context_text(summaries: list[SummaryNode], leaves: list[LeafNode]) -> str:
    ordered_summaries = _sort_summaries_chronologically(summaries)
    ordered_leaves = _sort_leaves_chronologically(leaves)
    blocks = []
    if ordered_summaries:
        blocks.append("Relevant session summaries:")
        blocks.extend(
            f"- Session {summary.session_id} ({summary.session_date or 'unknown'}): {summary.summary}"
            for summary in ordered_summaries
        )
    blocks.append("Raw evidence:")
    blocks.extend(
        f"[Session {leaf.session_id} | {leaf.session_date or 'unknown'} | turn {leaf.turn_index}]\n{leaf.raw_text}"
        for leaf in ordered_leaves
    )
    return "\n\n".join(blocks)


def _fit_context_budget(
    summaries: list[SummaryNode],
    leaves: list[LeafNode],
    token_budget: int,
    *,
    protected_leaf_ids: set[str] | None = None,
) -> tuple[list[SummaryNode], list[LeafNode]]:
    protected_leaf_ids = protected_leaf_ids or set()
    kept_summaries = list(summaries)
    kept_leaves = list(leaves)

    def over_budget() -> bool:
        return rough_token_count(_context_text(kept_summaries, kept_leaves)) > token_budget

    while len(kept_leaves) > 1 and over_budget():
        removed = False
        for index in range(len(kept_leaves) - 1, -1, -1):
            if kept_leaves[index].node_id in protected_leaf_ids:
                continue
            kept_leaves.pop(index)
            removed = True
            break
        if not removed:
            break

    while kept_summaries and over_budget():
        kept_summaries.pop()

    while len(kept_leaves) > 1 and over_budget():
        kept_leaves.pop()
    while kept_summaries and over_budget():
        kept_summaries.pop()
    return kept_summaries, kept_leaves


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _node_row(node: LeafNode | SummaryNode | RoutingCardNode | AtomicFactNode) -> dict[str, Any]:
    row = asdict(node)
    if isinstance(node, LeafNode):
        row["node_type"] = "leaf"
    elif isinstance(node, RoutingCardNode):
        row["node_type"] = "routing_card"
    elif isinstance(node, AtomicFactNode):
        row["node_type"] = "atomic_fact"
    elif getattr(node, "schema_version", "") == GRAPHMEM_V36_SCHEMA:
        row["node_type"] = {
            "TurnNodeV36": "turn", "RoleFrameNode": "role_frame",
            "RoutingCard": "routing_card", "EvidenceGroup": "evidence_group",
        }.get(type(node).__name__, type(node).__name__.casefold())
    elif getattr(node, "schema_version", "") == GRAPHMEM_V3_SCHEMA:
        row["node_type"] = {
            "TurnNode": "turn", "ClaimNode": "claim", "EventNode": "event",
            "EventEntityNode": "event_entity",
            "EpisodeNode": "episode", "ThemeNode": "theme",
        }.get(type(node).__name__, type(node).__name__.casefold())
    else:
        row["node_type"] = "summary"
    row.pop("embedding", None)
    row.pop("object_embedding", None)
    return row


def _reset_jsonl_outputs(directory: Path) -> None:
    for name in (
        "llm_calls.jsonl",
        "embedding_calls.jsonl",
        "compression_stats.jsonl",
        "nodes.jsonl",
        "episodes.jsonl",
        "themes.jsonl",
        "hyperedges.jsonl",
        "state_chains.jsonl",
        "coverage.jsonl",
        "index_diagnostics.jsonl",
        "edges.jsonl",
        "question_stats.jsonl",
        "retrieval_results.jsonl",
        "answers.jsonl",
        "manual_eval.jsonl",
    ):
        (directory / name).write_text("", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_question_stats(path: Path) -> list[QuestionStats]:
    return [QuestionStats(**row) for row in _read_jsonl(path)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _deepseek_stage_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        stage = str(row["stage"])
        stage_total = totals.setdefault(
            stage,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
        )
        stage_total["calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            stage_total[field] += int(row.get(field) or 0)
    return totals


def _local_summary_stage_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}
    for row in rows:
        compressor_name = str(row.get("compressor") or "")
        if not compressor_name.startswith("qwen_local:"):
            continue
        stage = str(row["stage"])
        stage_total = totals.setdefault(
            stage,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_sec": 0.0,
                "failure_count": 0,
            },
        )
        stage_total["calls"] = int(stage_total["calls"]) + 1
        prompt_tokens = int(row.get("origin_tokens") or 0)
        completion_tokens = int(row.get("compressed_tokens") or 0)
        stage_total["prompt_tokens"] = int(stage_total["prompt_tokens"]) + prompt_tokens
        stage_total["completion_tokens"] = int(stage_total["completion_tokens"]) + completion_tokens
        stage_total["total_tokens"] = int(stage_total["total_tokens"]) + prompt_tokens + completion_tokens
        stage_total["latency_sec"] = float(stage_total["latency_sec"]) + float(
            row.get("latency_sec") or 0.0
        )
        stage_total["failure_count"] = int(stage_total["failure_count"]) + int(
            bool(row.get("error_status"))
        )
    return totals


def _v36_index_diagnostics_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        marker = json.dumps(row, sort_keys=True, ensure_ascii=True)
        if marker not in seen:
            seen.add(marker)
            unique.append(row)
    sessions = [row for row in unique if row.get("stage") == "v36_session_extraction"]
    indexes = [row for row in unique if row.get("stage") == "v36_index"]
    failures = [
        row for row in sessions
        if row.get("parse_error") in {
            "invalid_json", "empty_frames", "coverage_gap",
        }
    ]
    return {
        "schema_version": GRAPHMEM_V36_SCHEMA,
        "summary": {
            "session_count": len(sessions),
            "parse_validation_failure_count": len(failures),
            "parse_validation_failure_rate": len(failures) / len(sessions) if sessions else 0.0,
            "invalid_json_count": sum(
                row.get("parse_error") == "invalid_json" for row in failures
            ),
            "empty_frames_count": sum(
                row.get("parse_error") == "empty_frames" for row in failures
            ),
            "coverage_gap_count": sum(
                row.get("parse_error") == "coverage_gap" for row in failures
            ),
            "local_lossless_frame_count": sum(
                int(row.get("local_lossless_frame_count") or 0) for row in sessions
            ),
            "lossless_only_turn_count": sum(int(row.get("lossless_only_count") or 0) for row in sessions),
            "provider_output_cap_violation_count": sum(not bool(row.get("provider_output_cap_honored", True)) for row in sessions),
            "node_count": sum(int(row.get("turn_count") or 0) + int(row.get("frame_count") or 0) + int(row.get("routing_card_count") or 0) + int(row.get("evidence_group_count") or 0) for row in indexes),
            "edge_count": sum(int(row.get("edge_count") or 0) for row in indexes),
            "max_node_degree": max((int(row.get("max_node_degree") or 0) for row in indexes), default=0),
            "participant_edge_count": sum(int(row.get("participant_edge_count") or 0) for row in indexes),
            "temporal_scope_edge_count": sum(int(row.get("temporal_scope_edge_count") or 0) for row in indexes),
        },
        "rows": unique,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_summary(output_dir: Path, aggregates: list[VariantStats]) -> None:
    rows = [asdict(aggregate) for aggregate in aggregates]
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "variant",
        "question_count",
        "build_cache_miss_input_tokens",
        "build_cache_hit_input_tokens",
        "build_output_tokens",
        "build_total_tokens",
        "build_budget_max_tokens",
        "build_budget_pass_count",
        "answer_cache_miss_input_tokens",
        "answer_cache_hit_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "answer_budget_max_tokens",
        "answer_budget_pass_count",
        "reasoning_tokens",
        "total_deepseek_tokens",
        "deepseek_call_count",
        "avg_tokens_per_question",
        "token_budget_avg_under_300k",
        "over_build_budget_question_ids",
        "over_answer_budget_question_ids",
        "retrieval_answer_session_hit_rate",
        "retrieval_answer_session_all_hit_rate",
        "avg_retrieved_answer_session_recall",
        "summary_count",
        "edge_count",
        "build_calls_per_session",
        "summary_parse_error_count",
        "summary_truncation_count",
        "wall_time_sec",
    ]
    markdown = [
        "# GraphMem Token Demo Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _write_manual_eval_template(variant_dir: Path) -> None:
    prior = {
        row["question_id"]: row
        for row in _read_jsonl(variant_dir / "manual_eval.jsonl")
        if row.get("question_id")
    }
    rows: list[dict[str, Any]] = []
    for answer in _read_jsonl(variant_dir / "answers.jsonl"):
        existing = prior.get(answer["question_id"], {})
        rows.append(
            {
                "question_id": answer["question_id"],
                "question": answer["question"],
                "gold_answer": answer["gold_answer"],
                "prediction": answer["prediction"],
                "strict_correct": existing.get("strict_correct"),
                "relaxed_correct": existing.get("relaxed_correct"),
                "error_type": existing.get("error_type"),
                "notes": existing.get("notes", ""),
            }
        )
    path = variant_dir / "manual_eval.jsonl"
    path.write_text("", encoding="utf-8")
    _append_jsonl(path, rows)
    strict_rows = [row for row in rows if isinstance(row["strict_correct"], bool)]
    relaxed_rows = [row for row in rows if isinstance(row["relaxed_correct"], bool)]
    markdown = [
        "# GraphMem Manual Evaluation",
        "",
        "Strict: numeric answers must state the gold value; insufficient gold must be explicit.",
        "Relaxed: report semantic matches separately; no LLM judge token is used.",
        "",
        f"- Rows: {len(rows)}",
        f"- Strict judged: {len(strict_rows)}",
        f"- Strict accuracy: {_manual_accuracy(strict_rows, 'strict_correct')}",
        f"- Relaxed judged: {len(relaxed_rows)}",
        f"- Relaxed accuracy: {_manual_accuracy(relaxed_rows, 'relaxed_correct')}",
        "",
        "| question_id | strict | relaxed | error_type | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    markdown.extend(
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|")
            for field in ("question_id", "strict_correct", "relaxed_correct", "error_type", "notes")
        )
        + " |"
        for row in rows
    )
    (variant_dir / "manual_eval.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _manual_accuracy(rows: list[dict[str, Any]], field: str) -> str:
    if not rows:
        return "pending"
    return f"{sum(bool(row[field]) for row in rows) / len(rows):.3f}"
