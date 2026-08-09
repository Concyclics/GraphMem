from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .domain import QueryBudget, canonical_json


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    dataset_hash: str
    model_id: str
    prompt_hash: str
    schema_version: str
    config_hash: str
    stage: str

    def key(self) -> str:
        values = asdict(self)
        if any(not str(value).strip() for value in values.values()):
            raise ValueError("cache identity fields must all be non-empty")
        return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    llm_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm_base_url: str = "http://127.0.0.1:8002/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    thinking_enabled: bool = False
    max_concurrency: int = 384
    refine_input_tokens_per_endpoint: int = 96
    refine_output_tokens: int = 256
    bridge_refine_output_tokens: int = 512
    semantic_batch_scenes: int = 4
    semantic_batch_input_tokens: int = 12000
    semantic_scene_input_tokens: int = 16000
    semantic_turn_input_chars: int = 0
    semantic_batch_output_tokens: int = 4096
    semantic_average_tokens_per_memory: int = 180000
    semantic_max_facts_per_scene: int = 12
    semantic_summary_tokens: int = 64
    semantic_repair_output_tokens: int = 4096
    semantic_constrained_json: bool = False
    semantic_individual_repair: bool = False
    semantic_extraction_mode: str = "legacy_batch"
    semantic_max_retries: int = 0
    semantic_retry_output_tokens: int = 1024
    semantic_compile_summary: bool = False
    # Hard per-memory build ceiling, enforced by BuildTokenLedger.  0 disables
    # enforcement and restores the pre-V5.6 unbounded behaviour.
    # ``semantic_average_tokens_per_memory`` above has declared 220000 since V5
    # and never had a consumer; this is the enforced counterpart.
    semantic_max_tokens_per_memory: int = 0
    # Fraction of the ceiling that may be spent before the fact cap is reduced.
    semantic_budget_degrade_at: float = 0.75
    # Emit the exact-quote evidence field.  It costs ~26% of extraction output
    # and only refines the span inside a turn the fact already cites, which the
    # projection re-derives from the value deterministically.
    semantic_quote_evidence: bool = True
    # Hard character ceiling on the predicate field, enforced by guided decoding
    # rather than asked for in the prompt.  Extraction currently writes whole
    # propositions into `p` -- mean 4.7 words, p95 10, often duplicating `v`
    # ("demonstrated high efficacy rates" / "high efficacy rates") -- so no two
    # predicates ever coincide, 96% of collections are singletons, and neither
    # embedding clustering nor an LLM vocabulary can merge them: they are
    # distinct propositions, not variants of one relation.  0 disables.
    semantic_predicate_max_chars: int = 0
    # Basis for the ledger's per-call output reservation.  0 keeps the old
    # behaviour of reserving `semantic_batch_output_tokens`, which makes the
    # output ceiling and the build budget the same knob: raising the ceiling to
    # 32768 to stop truncation reserved 32768 per call, exhausted the 220,000
    # per-memory budget in about seven calls, and drove extraction into fallback
    # on 100 scenes with 0.28 facts per scene.  Setting an expected value
    # decouples them -- the ceiling becomes a runaway guard, the budget is spent
    # against what calls actually cost.  Measured output on this corpus is
    # ~600 tokens per call.
    semantic_expected_output_tokens: int = 0
    # When a call costs more than its reservation, whether the ledger degrades
    # subsequent calls.  Off means the ceiling is advisory for a single call and
    # only the running total governs.
    semantic_fallback_on_overrun: bool = True
    # Character ceiling on the per-scene summary sentence, enforced by guided
    # decoding.  0 restores the pre-V5.8 behaviour, where `semantic_compile_summary`
    # concatenates fact triples and the routing cards built from them read as
    # duplicated term soup.  A sentence is what a question embedding can match.
    semantic_scene_summary_chars: int = 0
    # Ask extraction for the named entities each scene mentions.  LoCoMo cat1
    # spreads its evidence over 2.68 sessions and is the worst-routed category
    # (session_all_hit 0.592); an entity is what links those sessions.
    semantic_scene_entities: bool = False
    # V5.10 lossless atomic extraction.  Off by default so every frozen V5.8
    # and V5.9 artifact retains its exact prompt, schema, and cache identity.
    # When enabled, a deterministic scan emits high-salience information units
    # and the extractor must account for every unit with either a grounded fact
    # or an explicit unresolved reason.
    semantic_atomic_coverage: bool = False
    # Replace the fixed per-scene fact cap with
    # ceil(alpha * units + beta * entities + gamma * temporal_units), bounded
    # by the floor below and ``semantic_adaptive_fact_cap_max``.
    semantic_adaptive_fact_cap: bool = False
    semantic_adaptive_fact_cap_max: int = 24
    semantic_fact_cap_alpha: float = 0.50
    semantic_fact_cap_beta: float = 0.25
    semantic_fact_cap_gamma: float = 0.50
    # Minimum fraction of information units that must be accounted for.  A
    # failed contract is retried once (when retries are enabled), then exposed
    # as a raw-source fallback rather than silently certified as complete.
    semantic_min_unit_coverage: float = 0.95
    semantic_raw_fallback_on_low_coverage: bool = True
    # Losslessly split overlong turns into sentence-aligned source spans.  This
    # uses ``semantic_turn_input_chars`` as the chunk size; no middle text is
    # replaced by a truncation marker.
    semantic_sentence_chunking: bool = False


@dataclass(frozen=True, slots=True)
class SceneConfig:
    min_turns: int = 2
    max_turns: int = 8
    topic_similarity_threshold: float = 0.55
    max_events_per_scene: int = 3
    coreference_margin: float = 0.08
    refine_batch_size: int = 24
    llm_semantic_extraction: bool = False
    llm_hierarchy_compression: bool = False


@dataclass(frozen=True, slots=True)
class CoarsenConfig:
    fanout: int = 8
    max_levels: int = 3
    summary_tokens: int = 320
    cross_session_merge: bool = True
    #: Second pass over the whole memory that merges mention strings into keys
    #: spanning more than one session.  Extraction runs per scene and cannot see
    #: another scene's vocabulary, which is why the entity layer measured 1,305
    #: names of which only 60 reach two sessions -- and the eight widest of those
    #: are speaker names, constant within a memory and so discriminating nothing.
    #: Off by default: this is an arm until it is measured, not a default.
    entity_merge: bool = False
    #: A key that reaches one session joins nothing; a key that reaches most of
    #: them routes nowhere.  Both ends are cut.  The upper bound is a share
    #: rather than a count so it does not encode any dataset's session count.
    entity_merge_min_sessions: int = 2
    entity_merge_max_session_share: float = 0.25
    #: Surfaces shorter than this are ambient words rather than referents.
    entity_merge_min_chars: int = 4
    #: 0 disables the dense step, leaving normalisation-only merging.  Above 0,
    #: two surfaces whose Qwen3-Embedding vectors exceed this cosine are merged;
    #: no new model is introduced, and no generative call is made.
    entity_merge_embedding_threshold: float = 0.0
    # Report/CIR path.  False preserves every frozen B0--B5 graph; true builds
    # an arbitrary-depth semantic hierarchy and compact structural provenance.
    recursive_hierarchy: bool = False
    compact_routing_provenance: bool = True
    # V5.10 report arm: real HNSW balanced graph coarsening.  The frozen default
    # remains the lexical bounded partition.
    assignment_method: str = "bounded_semantic_partition"
    hnsw_dimension: int = 256
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100


@dataclass(frozen=True, slots=True)
class EdgeConfig:
    embedding_k: int = 8
    max_candidates_per_node: int = 24
    max_degree_per_relation: int = 12
    low_threshold: float = 0.45
    high_threshold: float = 0.78
    refine_mode: str = "ambiguous_only"
    refine_batch_size: int = 24
    max_refine_calls_per_1000_turns: int = 20
    # Zero keeps the frozen pre-V5.10 path unbounded.  V5.10 admits only the
    # highest-value ambiguous candidates, with both a hub-degree bound and a
    # graph-size-proportional global bound before any LLM call is scheduled.
    max_refine_candidates_per_node: int = 0
    max_refine_candidates_per_1000_nodes: int = 0
    graph_variant: str = "g0"
    temporal_normalization: bool = False
    cross_session_portals: bool = False
    parent_gated_relations: bool = False
    relation_candidate_method: str = "bounded_sparse"
    cross_session_neighbor_quota: int = 0
    typed_relation_restoration: bool = False
    typed_relation_min_confidence: float = 0.82
    # V5.14 experimental path.  Coarse candidate edges carry a relation-signal
    # mask (semantic/entity/time/state/rare-lexical) into their child scopes instead of
    # propagating one untyped cosine.  Disabled by default so frozen V5.13
    # snapshots and config hashes retain their historical behaviour.
    relation_mask_propagation: bool = False
    # The 200-question V5.14 audit found that adding entity/state/time postings
    # to the already-unioned lexical+dense atomic candidates improved pair
    # recall by only 0.053pp and changed no all-hit decision.  Keep it as a
    # separate research arm; parent-mask routing does not require it.
    atomic_relation_multiview: bool = False
    # Rare lexical terms are measured by session document frequency over raw
    # source text.  They are a long-document relation feature, not a replacement
    # for query BM25.  Requiring three shared terms reproduces the high-recall,
    # bounded LongMemEval operating point while rejecting one-word coincidences.
    # The full V5.15 gate raised session-pair all-hit from 63% to 93% but changed
    # neither 32-turn nor 48-turn packed accuracy, so promotion remains opt-in.
    rare_lexical_relation: bool = False
    rare_lexical_df_share: float = 0.05
    rare_lexical_min_shared: int = 3
    # Independent quotas are intentionally combined by union.  A shared total
    # top-k made dense/lexical similarity crowd out sparse entity, temporal and
    # state signals on the hard multi-hop development set.
    relation_view_quotas: Mapping[str, int] = field(default_factory=lambda: {
        "lexical": 8,
        "semantic": 8,
        "entity": 4,
        "state": 4,
        "temporal": 2,
        "collection": 2,
        "rare_lexical": 6,
    })
    predicate_embedding_threshold: float = 0.92
    # Predicates only ever merge inside one of these slots.  "slot" is the V5.4
    # behaviour, (owner, scope, value_type, polarity), which leaves 51% of
    # predicates structurally ineligible before the threshold is consulted;
    # "owner" widens to (owner, polarity), making 78.6% eligible.
    predicate_cluster_scope: str = "slot"
    # "mutual_pair" is the V5.4 rule: merge only when two labels are each
    # other's nearest neighbour, so a family of five similar predicates yields
    # at most a couple of merges.  "agglomerative" takes the transitive closure
    # of every above-threshold pair.
    predicate_cluster_mode: str = "mutual_pair"
    portal_degree_cap: int = 2
    relation_degree_caps: Mapping[str, int] = field(default_factory=lambda: {
        "same_event": 4, "same_activity": 8, "state_next": 4,
        "state_transition": 4, "temporal_before": 8, "shared_entity": 32,
        "shared_value": 32, "collection_co_member": 128,
        "has_fact": 64, "fact_value": 64, "coreference": 32,
        "coarse_related": 16,
        "same_entity_state": 8, "temporal_continuation": 8,
        "causal": 4, "contradiction_update": 8,
    })


@dataclass(frozen=True, slots=True)
class StorageConfig:
    runtime_mode: str = "sqlite_snapshot"
    sqlite_path: str = "artifacts/v5/graphmem.sqlite"
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_batch_nodes: int = 1000
    neo4j_batch_edges: int = 2000


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeConfig:
    """Supported online-retrieval knobs for a deployed query plane.

    Build configuration and query-serving configuration deliberately remain
    separate: changing worker/cache sizing must not invalidate a frozen graph's
    build cache identity.  These defaults are the measured V5.11 balanced path,
    not the historical ablation defaults on :class:`GraphNavigator`.
    """

    harness_profile: str = "h11"
    graph_hop_decay: float = 0.3
    expansion_beam: int = 2
    fusion_weights: Mapping[str, float] = field(default_factory=dict)
    session_router_k: int = 0
    per_session_quota: bool = False
    session_flood_k: int = 0
    hierarchical_routing: bool = True
    hierarchy_root_beam: int = 2
    hierarchy_child_beam: int = 4
    hierarchy_descent_beam: int = 1
    rare_lexical_relations: bool = False
    hierarchy_operator_aware: bool = True
    obligation_aware_packing: bool = True
    precision_aware_packing: bool = False
    # 0 keeps the full id-only reservoir. Positive values expose a genuine
    # candidate precision/recall operating point before evidence packing.
    candidate_pool_limit: int = 0
    span_pack_window: int = 96
    obligation_aware_relations: bool = False
    native_seed_fusion: bool = True
    # At low compiler confidence, retain the AST operator but union/relax its
    # seed filters with the legacy parse instead of hard-excluding evidence.
    queryir_soft_fallback: bool = False
    queryir_soft_fallback_threshold: float = 0.80
    read_pool_size: int = 1
    snapshot_cache_bytes: int = 256 * 1024 * 1024
    snapshot_cache_memories: int = 8
    metadata_cache_memories: int = 8
    compiled_cache_dir: str = "artifacts/v5_11/compiled_memory_views"
    compiled_cache_admission: bool = True

    def __post_init__(self) -> None:
        if self.harness_profile not in {
                "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h8",
                "h9", "h10", "h11"}:
            raise ValueError("unsupported retrieval harness_profile")
        if not 0.0 <= self.graph_hop_decay <= 1.0:
            raise ValueError("graph_hop_decay must be in [0, 1]")
        positive = {
            "expansion_beam": self.expansion_beam,
            "hierarchy_root_beam": self.hierarchy_root_beam,
            "hierarchy_child_beam": self.hierarchy_child_beam,
            "hierarchy_descent_beam": self.hierarchy_descent_beam,
            "read_pool_size": self.read_pool_size,
            "snapshot_cache_bytes": self.snapshot_cache_bytes,
            "snapshot_cache_memories": self.snapshot_cache_memories,
            "metadata_cache_memories": self.metadata_cache_memories,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"positive retrieval runtime values required: {positive}")
        if self.session_router_k < 0 or self.session_flood_k < 0:
            raise ValueError("session routing limits cannot be negative")
        if self.candidate_pool_limit < 0:
            raise ValueError("candidate_pool_limit cannot be negative")
        if self.span_pack_window < 0:
            raise ValueError("span_pack_window cannot be negative")
        if any(float(value) < 0 for value in self.fusion_weights.values()):
            raise ValueError("fusion weights cannot be negative")
        if not 0.0 <= self.queryir_soft_fallback_threshold <= 1.0:
            raise ValueError("queryir_soft_fallback_threshold must be in [0, 1]")

    def navigator_options(
        self, *, compiled_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Translate the stable config schema to ``GraphNavigator`` options."""
        cache_dir = (self.compiled_cache_dir if compiled_cache_dir is None
                     else str(compiled_cache_dir))
        options: dict[str, Any] = {
            "harness_profile": self.harness_profile,
            "graph_hop_decay": self.graph_hop_decay,
            "expansion_beam": self.expansion_beam,
            "session_router_k": self.session_router_k,
            "per_session_quota": self.per_session_quota,
            "session_flood_k": self.session_flood_k,
            "hierarchical_routing": self.hierarchical_routing,
            "hierarchy_root_beam": self.hierarchy_root_beam,
            "hierarchy_child_beam": self.hierarchy_child_beam,
            "hierarchy_descent_beam": self.hierarchy_descent_beam,
            "rare_lexical_relations": self.rare_lexical_relations,
            "hierarchy_operator_aware": self.hierarchy_operator_aware,
            "obligation_aware_packing": self.obligation_aware_packing,
            "precision_aware_packing": self.precision_aware_packing,
            "candidate_pool_limit": self.candidate_pool_limit,
            "span_pack_window": self.span_pack_window,
            "obligation_aware_relations": self.obligation_aware_relations,
            "native_seed_fusion": self.native_seed_fusion,
            "queryir_soft_fallback": self.queryir_soft_fallback,
            "queryir_soft_fallback_threshold": self.queryir_soft_fallback_threshold,
            "read_pool_size": self.read_pool_size,
            "snapshot_cache_bytes": self.snapshot_cache_bytes,
            "snapshot_cache_memories": self.snapshot_cache_memories,
            "metadata_cache_memories": self.metadata_cache_memories,
            "compiled_cache_admission": self.compiled_cache_admission,
        }
        if self.fusion_weights:
            options["fusion_weights"] = dict(self.fusion_weights)
        if cache_dir:
            options["compiled_cache_dir"] = cache_dir
        return options


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """Bounded multi-process serving and lifecycle settings."""

    workers: int = 8
    start_method: str = "spawn"
    max_queued: int = 32
    per_tenant_outstanding: int = 8
    affinity_replicas: int = 2
    warm_replicas: int = 1
    retry_broken_worker: int = 1
    worker_cpu_ids: tuple[int, ...] = ()
    request_timeout_seconds: float = 5.0
    precompile_on_start: bool = True
    precompile_workers: int = 4
    sidecar_refresh_seconds: float = 10.0
    host: str = "127.0.0.1"
    port: int = 8090

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worker_cpu_ids", tuple(int(value) for value in self.worker_cpu_ids))
        if self.workers <= 0 or self.per_tenant_outstanding <= 0:
            raise ValueError("workers and per_tenant_outstanding must be positive")
        if self.max_queued < 0 or self.retry_broken_worker < 0:
            raise ValueError("queue and retry settings cannot be negative")
        if not 1 <= self.affinity_replicas <= self.workers:
            raise ValueError("affinity_replicas must be in [1, workers]")
        if not 1 <= self.warm_replicas <= self.affinity_replicas:
            raise ValueError("warm_replicas must be in [1, affinity_replicas]")
        if self.worker_cpu_ids:
            if len(self.worker_cpu_ids) != self.workers:
                raise ValueError("worker_cpu_ids must contain one CPU ID per worker")
            if len(set(self.worker_cpu_ids)) != self.workers:
                raise ValueError("worker_cpu_ids must be unique")
            if any(value < 0 for value in self.worker_cpu_ids):
                raise ValueError("worker_cpu_ids cannot contain negative values")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.precompile_workers <= 0 or self.sidecar_refresh_seconds < 0:
            raise ValueError("invalid compiled-sidecar lifecycle settings")
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError("invalid serving address")

    def pool_options(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "start_method": self.start_method,
            "max_queued": self.max_queued,
            "per_tenant_outstanding": self.per_tenant_outstanding,
            "affinity_replicas": self.affinity_replicas,
            "retry_broken_worker": self.retry_broken_worker,
            "worker_cpu_ids": self.worker_cpu_ids or None,
        }


@dataclass(frozen=True, slots=True)
class GraphMemRuntimeConfig:
    """Independent, hashable configuration for the online query plane."""

    schema_version: str = "graphmem-runtime-v5.11"
    profile: str = "v5_11_balanced"
    retrieval: RetrievalRuntimeConfig = field(default_factory=RetrievalRuntimeConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    query_budget: QueryBudget = field(default_factory=lambda: QueryBudget(
        max_evidence_turns=32, max_evidence_tokens=5000))

    def __post_init__(self) -> None:
        if self.schema_version != "graphmem-runtime-v5.11":
            raise ValueError("unsupported GraphMem runtime schema version")
        if not self.profile.strip():
            raise ValueError("runtime profile must be non-empty")


@dataclass(frozen=True, slots=True)
class GraphMemV5Config:
    schema_version: str = "graphmem-v5"
    profile: str = "full_balanced"
    random_seed: int = 42
    models: ModelConfig = field(default_factory=ModelConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    coarsen: CoarsenConfig = field(default_factory=CoarsenConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    query_budget: QueryBudget = field(default_factory=QueryBudget)

    def __post_init__(self) -> None:
        if self.schema_version != "graphmem-v5":
            raise ValueError("unsupported GraphMem schema version")
        if self.models.thinking_enabled:
            raise ValueError("GraphMem V5 memory backbone thinking must remain disabled")
        if self.storage.runtime_mode not in {
            "neo4j_direct", "neo4j_cached", "sqlite_snapshot"
        }:
            raise ValueError("invalid graph runtime mode")
        if self.edges.refine_mode not in {
            "none", "ambiguous_only", "high_value_only", "all_bounded_candidates"
        }:
            raise ValueError("invalid edge refine mode")
        if self.edges.graph_variant not in {"g0", "g1", "g2", "g3", "g4", "g5"}:
            raise ValueError("invalid semantic graph variant")
        if self.edges.predicate_cluster_scope not in {"slot", "owner", "memory"}:
            raise ValueError("invalid predicate cluster scope")
        if self.edges.predicate_cluster_mode not in {"mutual_pair", "agglomerative"}:
            raise ValueError("invalid predicate cluster mode")
        if self.models.semantic_extraction_mode not in {
                "legacy_batch", "strict_single", "strict_pair", "strict_batch"}:
            raise ValueError("invalid semantic extraction mode")
        if self.models.semantic_max_retries not in {0, 1}:
            raise ValueError("semantic_max_retries must be 0 or 1")
        if self.models.semantic_turn_input_chars < 0:
            raise ValueError("semantic_turn_input_chars cannot be negative")
        if self.models.semantic_predicate_max_chars < 0:
            raise ValueError("semantic_predicate_max_chars cannot be negative")
        if self.models.semantic_max_tokens_per_memory < 0:
            raise ValueError("semantic_max_tokens_per_memory cannot be negative")
        if self.models.semantic_adaptive_fact_cap_max < self.models.semantic_max_facts_per_scene:
            raise ValueError(
                "semantic_adaptive_fact_cap_max cannot be below semantic_max_facts_per_scene")
        if any(value < 0 for value in (
                self.models.semantic_fact_cap_alpha,
                self.models.semantic_fact_cap_beta,
                self.models.semantic_fact_cap_gamma)):
            raise ValueError("semantic fact-cap coefficients cannot be negative")
        if not 0.0 <= self.models.semantic_min_unit_coverage <= 1.0:
            raise ValueError("semantic_min_unit_coverage must be in [0, 1]")
        if not 0.0 < self.models.semantic_budget_degrade_at <= 1.0:
            raise ValueError("semantic_budget_degrade_at must be in (0, 1]")
        if not 0.0 <= self.edges.predicate_embedding_threshold <= 1.0:
            raise ValueError("predicate embedding threshold must be in [0, 1]")
        if not 0 <= self.edges.low_threshold < self.edges.high_threshold <= 1:
            raise ValueError("edge thresholds must satisfy 0 <= low < high <= 1")
        if not 0 < self.coarsen.entity_merge_max_session_share <= 1:
            raise ValueError("entity merge session share must be in (0, 1]")
        if not 0 <= self.coarsen.entity_merge_embedding_threshold <= 1:
            raise ValueError("entity merge embedding threshold must be in [0, 1]")
        if self.coarsen.entity_merge_min_sessions < 2:
            raise ValueError("an entity merge key must span at least two sessions")
        if self.coarsen.recursive_hierarchy and self.coarsen.fanout < 2:
            raise ValueError("recursive hierarchy fanout must be at least 2")
        if self.coarsen.recursive_hierarchy and self.coarsen.max_levels < 2:
            raise ValueError("recursive hierarchy requires at least two levels")
        if self.coarsen.assignment_method not in {
                "bounded_semantic_partition", "hnsw"}:
            raise ValueError("invalid coarsen assignment_method")
        if self.edges.relation_candidate_method not in {"bounded_sparse", "hnsw"}:
            raise ValueError("invalid relation candidate method")
        if not 0 <= self.edges.typed_relation_min_confidence <= 1:
            raise ValueError("typed relation confidence must be in [0, 1]")
        if self.edges.cross_session_neighbor_quota < 0:
            raise ValueError("cross-session neighbour quota cannot be negative")
        positive = {
            "fanout": self.coarsen.fanout,
            "max_levels": self.coarsen.max_levels,
            "summary_tokens": self.coarsen.summary_tokens,
            "hnsw_dimension": self.coarsen.hnsw_dimension,
            "hnsw_m": self.coarsen.hnsw_m,
            "hnsw_ef_construction": self.coarsen.hnsw_ef_construction,
            "embedding_k": self.edges.embedding_k,
            "max_candidates_per_node": self.edges.max_candidates_per_node,
            "max_degree_per_relation": self.edges.max_degree_per_relation,
            "refine_batch_size": self.edges.refine_batch_size,
            "scene_min_turns": self.scenes.min_turns,
            "scene_max_turns": self.scenes.max_turns,
            "max_events_per_scene": self.scenes.max_events_per_scene,
            "coreference_batch_size": self.scenes.refine_batch_size,
            "max_concurrency": self.models.max_concurrency,
            "refine_input_tokens_per_endpoint": self.models.refine_input_tokens_per_endpoint,
            "refine_output_tokens": self.models.refine_output_tokens,
            "bridge_refine_output_tokens": self.models.bridge_refine_output_tokens,
            "semantic_batch_scenes": self.models.semantic_batch_scenes,
            "semantic_batch_input_tokens": self.models.semantic_batch_input_tokens,
            "semantic_scene_input_tokens": self.models.semantic_scene_input_tokens,
            "semantic_batch_output_tokens": self.models.semantic_batch_output_tokens,
            "semantic_average_tokens_per_memory": self.models.semantic_average_tokens_per_memory,
            "semantic_max_facts_per_scene": self.models.semantic_max_facts_per_scene,
            "semantic_summary_tokens": self.models.semantic_summary_tokens,
            "semantic_repair_output_tokens": self.models.semantic_repair_output_tokens,
            "semantic_retry_output_tokens": self.models.semantic_retry_output_tokens,
            "semantic_adaptive_fact_cap_max": self.models.semantic_adaptive_fact_cap_max,
            "portal_degree_cap": self.edges.portal_degree_cap,
            "neo4j_batch_nodes": self.storage.neo4j_batch_nodes,
            "neo4j_batch_edges": self.storage.neo4j_batch_edges,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"positive configuration values required: {positive}")
        if self.edges.max_refine_calls_per_1000_turns < 0:
            raise ValueError("refine call limit cannot be negative")
        if self.edges.max_refine_candidates_per_node < 0:
            raise ValueError("refine candidate degree limit cannot be negative")
        if self.edges.max_refine_candidates_per_1000_nodes < 0:
            raise ValueError("refine candidate density limit cannot be negative")
        if self.scenes.min_turns > self.scenes.max_turns:
            raise ValueError("scene min_turns cannot exceed max_turns")
        if not 0.0 <= self.scenes.topic_similarity_threshold <= 1.0:
            raise ValueError("scene similarity threshold must be in [0, 1]")
        if not 0.0 <= self.scenes.coreference_margin <= 1.0:
            raise ValueError("coreference margin must be in [0, 1]")


def config_hash(config: GraphMemV5Config) -> str:
    return hashlib.sha256(canonical_json(asdict(config)).encode("utf-8")).hexdigest()


def _section(cls: type[Any], payload: Mapping[str, Any] | None) -> Any:
    return cls(**dict(payload or {}))


def config_from_dict(payload: Mapping[str, Any]) -> GraphMemV5Config:
    value = dict(payload)
    return GraphMemV5Config(
        schema_version=str(value.get("schema_version", "graphmem-v5")),
        profile=str(value.get("profile", "full_balanced")),
        random_seed=int(value.get("random_seed", 42)),
        models=_section(ModelConfig, value.get("models")),
        scenes=_section(SceneConfig, value.get("scenes")),
        storage=_section(StorageConfig, value.get("storage")),
        coarsen=_section(CoarsenConfig, value.get("coarsen")),
        edges=_section(EdgeConfig, value.get("edges")),
        query_budget=_section(QueryBudget, value.get("query_budget")),
    )


def load_config(path: Path) -> GraphMemV5Config:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PyYAML is required to load YAML V5 configs") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError("V5 config root must be a mapping")
    return config_from_dict(payload)


def runtime_config_hash(config: GraphMemRuntimeConfig) -> str:
    """Hash runtime tuning independently from the immutable graph build."""
    return hashlib.sha256(canonical_json(asdict(config)).encode("utf-8")).hexdigest()


def runtime_config_from_dict(payload: Mapping[str, Any]) -> GraphMemRuntimeConfig:
    value = dict(payload)
    return GraphMemRuntimeConfig(
        schema_version=str(value.get("schema_version", "graphmem-runtime-v5.11")),
        profile=str(value.get("profile", "v5_11_balanced")),
        retrieval=_section(RetrievalRuntimeConfig, value.get("retrieval")),
        serving=_section(ServingConfig, value.get("serving")),
        query_budget=_section(QueryBudget, value.get("query_budget")),
    )


def load_runtime_config(path: Path) -> GraphMemRuntimeConfig:
    """Load a JSON/YAML query-plane profile without changing build identity."""
    source = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        payload = json.loads(source)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PyYAML is required to load YAML runtime configs") from error
        payload = yaml.safe_load(source)
    if not isinstance(payload, Mapping):
        raise ValueError("runtime config root must be a mapping")
    return runtime_config_from_dict(payload)
