from __future__ import annotations

import heapq
import itertools
import math
import re
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..domain import (
    CandidateScore,
    EvidenceCertificate,
    EvidenceMember,
    EvidenceUnit,
    NavigationResult,
    NodeType,
    ProofStep,
    QueryBudget,
    QueryOperator,
    RelationType,
    SourceTurn,
    stable_id,
)
from ..principals import build_principal_registry, resolution_stats
from ..runtime import GraphReadView, SQLiteSnapshotRuntime
from ..storage import SQLiteGraphStore
from ..tokenization import TokenCounter, resolve_token_counter
from .algebra import evaluate as evaluate_algebra
from .ast_algebra import evaluate_ast
from .operators import requires_exhaustive_scope as ops_requires_scope
from .bindings import bind_facts, bind_facts_discriminant
from .certificate import evaluate_certificate, finalize_ast_certificate
from .compiled_memory import (
    COMPILED_MEMORY_SCHEMA,
    CompiledMemoryArtifact,
    CompiledMemorySidecar,
)
from .packer import (
    adaptive_evidence_turn_limit,
    build_proof_units,
    pack as pack_proof_units,
    pack_obligation_aware,
)
from .query_ir import compile_query
from .scheduler import ScheduleResult, execute as schedule_relations
from .facts import (CHANNELS as FACT_CHANNELS, build_fact_reservoir,
                    select_active_facts)
from .seeding import DenseSearchMany, TurnSearchIndex, seed_operands


DEFAULT_BACKBONE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "that", "the", "their", "they", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
NEGATIVE_TERMS = frozenset({"not", "never", "no", "neither", "without", "didn't", "don't"})
TIME_TERMS = frozenset({
    "before", "after", "first", "last", "later", "earlier", "during", "until",
    "since", "when", "date", "day", "week", "month", "year", "long", "duration",
})
PREFERENCE_QUERY_RE = re.compile(
    r"\b(?:recommend|suggest|likely|prefer|favorite|favourite|should|"
    r"would\s+(?:like|enjoy)|good\s+fit)\b", re.I)


class NavigatorVariant(StrEnum):
    N0_LEGACY = "n0_legacy"
    N1_RAW_FUSION = "n1_raw_fusion"
    N2_PROVENANCE = "n2_provenance"
    N3_PRIORITY = "n3_priority"
    N4_CERTIFICATE = "n4_certificate"
    N5_SET_COVER = "n5_set_cover"


class HarnessProfile(StrEnum):
    H0_N5 = "h0"
    H1_TELEMETRY = "h1"
    H2_POSTINGS = "h2"
    H3_MULTI_ANCHOR = "h3"
    H4_SCHEDULER = "h4"
    H5_ALGEBRA = "h5"
    H6_PROOF_PACKING = "h6"
    # The feedback's H-series: H7 is proof-unit packing, H8 the safe-superset
    # reservoir, H9 the operator algebra.  They are added in dependency order,
    # which is why H8 lands before H7.
    H8_RESERVOIR = "h8"
    # PR4a: the reservoir principle applied to CanonicalFacts.  The proof funnel
    # showed the fact exists on 68.5% of questions but is reached on 21.0%.
    H9_FACT_RESERVOIR = "h9"
    # PR5a: executes the compiled operator AST and emits answer members with
    # their witnesses.  Aggregation questions are 40% of the development set and
    # its worst category at 51.9%; without members the closed-form composer has
    # nothing to read and fired on 0 of 200 questions.
    H10_AST = "h10"
    # V5.10: promote the AST once at the compiler boundary.  H10 keeps the
    # historical split legacy/AST execution for a frozen ablation baseline.
    H11_UNIFIED_IR = "h11"


VARIANT_RANK = {variant: index for index, variant in enumerate(NavigatorVariant)}


#: The wide-reservoir fusion, one name per term.  `operand_cap` is the number of
#: operands past which `operand` stops accumulating: the term used to be
#: `0.4 * len(operand_ids)` with no ceiling, so a turn bound to six operands
#: collected 2.4 before any lexical channel was consulted, while exact/bm25/dense
#: are each normalised to about [0, 1].
FUSION_DEFAULTS: dict[str, float] = {
    "exact": 1.2, "bm25": 1.0, "dense": 1.0, "graph": 0.8, "binding": 0.7,
    "operand": 0.4, "operand_cap": 1e9, "role": 0.25, "slot": 0.5,
    "session": 0.12, "adjacency": 1.0,
}


DenseSearch = Callable[[str, str, int], Sequence[tuple[str, float]]]
FactDenseSearch = Callable[
    [str, str, Sequence[str], int], Sequence[tuple[str, float]]]


def terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_RE.findall(text))


@lru_cache(maxsize=8_192)
def content_terms(text: str) -> frozenset[str]:
    return frozenset(token for token in terms(text) if token not in STOPWORDS and len(token) > 1)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(terms(text)) * 1.3))


def question_slots(query: str) -> tuple[str, tuple[str, ...], bool]:
    lowered = query.casefold()
    negative = any(term in terms(lowered) for term in NEGATIVE_TERMS)
    if "how many" in lowered or lowered.startswith("count"):
        return "count", ("collection_scope", "member_1", "member_2"), negative
    if any(word in lowered for word in ("before", "after", "earlier", "later", "first", "last")):
        return "temporal_comparison", ("temporal_left", "temporal_right", "ordering"), negative
    if "how long" in lowered or "duration" in lowered:
        return "duration", ("temporal_start", "temporal_end", "duration"), negative
    if any(word in lowered for word in ("what are", "which", "list", "what do")):
        return "list", ("collection_scope", "member_1", "member_2"), negative
    if any(word in lowered for word in ("change", "became", "now", "currently", "replaced")):
        return "state_change", ("prior_state", "current_state"), negative
    if any(word in lowered for word in TIME_TERMS):
        return "temporal", ("event", "time"), negative
    return "fact", ("subject", "predicate", "object"), negative


def exact_lookup_eligible(ir) -> bool:
    """Whether a plan may take the guarded direct-evidence route.

    ``LOOKUP`` alone is not enough: the compiler deliberately represents some
    plural or recommendation questions as a lookup.  Those require collection
    closure or preference synthesis and must retain the normal graph path.
    Date-qualified scalar questions remain eligible; a date is often the most
    discriminating lexical key for LoCoMo single-hop evidence.
    """
    slots = getattr(ir, "slots", None)
    return bool(
        ir.operator == QueryOperator.LOOKUP
        and len(ir.operands) == 1
        and not bool(getattr(slots, "expects_multiple", False))
        and not bool(getattr(slots, "is_count", False))
        and not bool(getattr(slots, "is_duration", False))
        and not bool(getattr(slots, "is_latest", False))
        and not PREFERENCE_QUERY_RE.search(ir.query)
)


def has_named_multi_party(turns: Sequence[SourceTurn]) -> bool:
    """Distinguish named dialogue participants from transport-only roles."""
    generic = {"", "assistant", "system", "tool", "user", "human"}
    return any((turn.speaker or "").casefold().strip() not in generic
               for turn in turns)


def _weighted_term_coverage(
    query_terms: frozenset[str], candidate_terms: frozenset[str],
    turn_index: TurnSearchIndex,
) -> float:
    """IDF-weighted query coverage on immutable per-memory postings."""
    if not query_terms:
        return 0.0
    total = max(1, len(turn_index.turns))
    weights = {
        term: math.log(1.0 + total / (1.0 + len(turn_index.postings.get(term, ()))))
        for term in query_terms
    }
    denominator = sum(weights.values())
    return (sum(weight for term, weight in weights.items()
                if term in candidate_terms) / denominator
            if denominator else 0.0)


def _exact_lookup_ranking(
    seeded,
    turn_index: TurnSearchIndex,
    query: str,
    direct_fact_turn_scores: Mapping[str, float],
) -> tuple[dict[str, float], bool, dict[str, object]]:
    """Score and gate the exact-lookup route without an LLM reranker.

    A high normalized dense score by itself is unsafe because every query has a
    top dense result.  The route only fires when a source turn covers specific
    (IDF-weighted) query terms and is corroborated by another retrieval channel
    or by a CanonicalFact carrying source provenance.  A miss therefore falls
    back to the ordinary hierarchical traversal rather than losing recall.
    """
    query_terms = content_terms(query)
    entry_by_turn = {row.turn_id: row for row in seeded.reservoir}
    candidates = tuple(dict.fromkeys((
        *seeded.source_turn_ids, *direct_fact_turn_scores.keys())))
    scores: dict[str, float] = {}
    details: dict[str, tuple[float, int, float]] = {}
    for turn_id in candidates:
        turn_terms = turn_index.turn_terms.get(turn_id, frozenset())
        coverage = _weighted_term_coverage(query_terms, turn_terms, turn_index)
        entry = entry_by_turn.get(turn_id)
        support = sum(
            1 for rank in (entry.ranks.values() if entry is not None else ())
            if rank <= 16)
        channel_peak = max(
            (float(value) for value in entry.scores.values()), default=0.0
        ) if entry is not None else 0.0
        fact_score = float(direct_fact_turn_scores.get(turn_id, 0.0))
        score = (
            1.40 * coverage
            + 0.25 * min(3, support)
            + 0.65 * fact_score
            + 0.20 * channel_peak
        )
        scores[turn_id] = score
        details[turn_id] = (coverage, support, fact_score)
    ordered = sorted(scores, key=lambda turn_id: (-scores[turn_id], turn_id))
    top_id = ordered[0] if ordered else ""
    coverage, support, fact_score = details.get(top_id, (0.0, 0, 0.0))
    fact_ordered = tuple(
        turn_id for turn_id in ordered if turn_id in direct_fact_turn_scores)
    fact_top_id = fact_ordered[0] if fact_ordered else ""
    fact_coverage, fact_support, fact_provenance_score = details.get(
        fact_top_id, (0.0, 0, 0.0))
    # The threshold is intentionally a conjunction.  It excludes topical dense
    # neighbours with no exact lexical support and owner-only fact floods.
    confident = bool(
        top_id
        and coverage >= 0.30
        and scores[top_id] >= 1.10
        and (support >= 2 or (support >= 1 and fact_score >= 0.45))
    )
    return scores, confident, {
        "top_turn_id": top_id,
        "top_score": scores.get(top_id, 0.0),
        "top_idf_coverage": coverage,
        "top_channel_support": support,
        "top_fact_score": fact_score,
        "scored_turns": len(scores),
        # Priority mode is deliberately fact-only.  The earlier implementation
        # gated on ``top_score`` across every lexical/dense seed and then added
        # the exact bonus to every seed.  It therefore activated on 11 LoCoMo
        # questions with zero direct facts and behaved like a second broad
        # lexical reranker (13 gains / 18 losses on the full benchmark).
        "fact_top_turn_id": fact_top_id,
        "fact_top_score": scores.get(fact_top_id, 0.0),
        "fact_top_idf_coverage": fact_coverage,
        "fact_top_channel_support": fact_support,
        "fact_top_provenance_score": fact_provenance_score,
        "fact_scored_turns": len(fact_ordered),
    }


class GraphNavigator:
    def __init__(
        self,
        store: SQLiteGraphStore,
        *,
        variant: NavigatorVariant | str = NavigatorVariant.N5_SET_COVER,
        dense_search: DenseSearch | None = None,
        dense_search_many: DenseSearchMany | None = None,
        fact_dense_search: FactDenseSearch | None = None,
        harness_profile: HarnessProfile | str | None = None,
        token_counter: TokenCounter | None = None,
        fact_channels: Sequence[str] | None = None,
        binding_discriminant: bool = False,
        skip_traversal_on_certificate: bool = True,
        preferred_relations=None,
        fallback_relations=None,
        rank_mandatory: bool = False,
        h10_owner_rescue: bool = True,
        h10_traversal: bool = True,
        manifest_collection_key: bool = True,
        #: Measured on 761 questions: decay alone +0.92pp, beam alone -52% mean
        #: latency and -76% p95, together +1.31pp at 122ms mean against 257ms.
        #: 1.0 / 0 restore the pre-V5.8 flag-and-no-pruning behaviour.
        graph_hop_decay: float = 0.3,
        expansion_beam: int = 2,
        #: Fusion weights, so the W-series can move one term at a time.  Defaults
        #: are the values that were inline in `_harness_rows`.  `operand_cap`
        #: bounds `operand * len(operand_ids)`, which was unbounded and is the
        #: main reason graph-only candidates scored 2.41 against 1.24.
        fusion_weights: Mapping[str, float] | None = None,
        #: Restrict candidates to the best `session_router_k` sessions, scored by
        #: BM25 of the question against each session's own text.  0 keeps the old
        #: behaviour, where sessions are never ranked as units and the 32-turn
        #: pack is drawn from a 578-turn pool spanning ~26 sessions.  Measured:
        #: single-session questions score 0.6895 against an oracle of 1.0000, so
        #: the entire deficit on 78.7% of the set is that no routing happens.
        session_router_k: int = 0,
        #: Give every routed session a floor of the evidence budget instead of
        #: letting one session's high-scoring turns take all 32 slots.
        per_session_quota: bool = False,
        #: Add *every* turn of the best `session_flood_k` sessions to the pool,
        #: which is what the legacy path does (`_navigate_legacy`, top-8).
        #: `session_router_k` filters an already-diluted pool; this one builds the
        #: pool from whole sessions, so a gold turn with no lexical match of its
        #: own still enters on its session's evidence.  The two are the only
        #: pipeline difference that survived attribution of the h0/h10 gap, so
        #: this is what isolates it from Query IR and the algebra.
        session_flood_k: int = 0,
        #: Pack with the legacy `_set_cover` instead of `_rank_pack`.  Pool size
        #: is not what separates the two pipelines -- h0 wins on 297 candidate
        #: turns against h10's 420 -- so the packer is the remaining difference.
        #: `_set_cover` picks by marginal utility with a slot-coverage gain and a
        #: bonus for a session not yet represented; `_rank_pack` sorts once.
        harness_set_cover: bool = False,
        #: Execute H10 through directional root-to-leaf routing.  Earlier
        #: profiles keep the flat postings path as frozen baselines.
        hierarchical_routing: bool = True,
        hierarchy_root_beam: int = 2,
        hierarchy_child_beam: int = 4,
        #: Structural fanout at each level after a relation lands in a new
        #: region.  It is intentionally independent of ``expansion_beam``:
        #: relation hops and hierarchy depth are separate search dimensions.
        hierarchy_descent_beam: int = 1,
        #: Admit coarse edges whose only construction signal is lexical_rare.
        #: Mixed masks retain their non-lexical attributes when this is false.
        rare_lexical_relations: bool = False,
        #: Restrict lexical-only bridges to QueryIR plans that require multiple
        #: witnesses/endpoints.  Ordinary lookup keeps lexical/dense seeding but
        #: does not spend its relation beam on same-topic lexical regions.
        query_gated_rare_lexical: bool = False,
        hierarchy_operator_aware: bool = True,
        read_pool_size: int = 4,
        snapshot_cache_bytes: int = 512 * 1024 * 1024,
        snapshot_cache_memories: int = 16,
        metadata_cache_memories: int = 16,
        #: V5.10 span/token packer.  Off keeps every frozen H0--H10 result;
        #: on selects atomic proof units by marginal obligation coverage per
        #: rendered span token and attaches spans for the answer stage.
        obligation_aware_packing: bool = False,
        precision_aware_packing: bool = False,
        candidate_pool_limit: int = 0,
        span_pack_window: int = 96,
        #: Keep a bounded number of the highest-scoring raw provenance
        #: fallbacks in the final pack.  Zero preserves the score-only pack.
        #: This is deliberately a quota rather than ``mandatory=True``: making
        #: every fallback mandatory displaced direct LME evidence, while
        #: removing the floor entirely lost useful cross-session LoCoMo turns.
        raw_fallback_reserve: int = 0,
        #: Compile QueryIR obligations into relation-specific beam priorities.
        #: Off preserves H0--H10 traversal ordering for clean ablations.
        obligation_aware_relations: bool = False,
        #: Use the immutable TurnSearchIndex for every query view instead of a
        #: SQLite FTS call per view.  Kept opt-in for accuracy/latency gating.
        native_seed_fusion: bool = False,
        #: Preserve owner/predicate views for candidate reach, but weight their
        #: ranking contribution by relational completeness instead of taking an
        #: untyped max.  This also keeps exact overlap on its native coverage
        #: scale rather than inflating the best shallow overlap to 1.0.
        relational_view_scoring: bool = False,
        #: Add a deterministic lexical view over the question relation after
        #: removing explicit owners and the requested answer head.
        query_relation_view: bool = False,
        #: The owner-wrapper defect exists in named multi-party transcripts.
        #: Generic user/assistant memories keep their frozen scoring when this
        #: guard is enabled; the condition is an input property, not a dataset
        #: or benchmark label.
        relational_view_named_speakers_only: bool = False,
        #: Score rank-sensitive agreement across relation-bearing QueryIR
        #: views.  Zero preserves the frozen max-per-channel fusion.
        relational_consensus_bonus: float = 0.0,
        #: Promote the answer turn paired with a query-relevant dialogue
        #: question.  LoCoMo frequently annotates a short pronominal response
        #: ("Luna and Oliver!") whose preceding turn contains the relation
        #: surface ("What are their names?").  Generic symmetric adjacency is
        #: too weak for that evidence to survive a 64-turn pack.
        dialogue_response_closure: bool = False,
        dialogue_response_flood_threshold: int = 0,
        #: ``None`` preserves the historical lexicographic mandatory-first
        #: ordering.  A finite value turns uncertain proof membership into a
        #: score bonus; proof-unit atomicity/certification still remain hard
        #: constraints in the packer.
        proof_priority_bonus: float | None = None,
        proof_priority_flood_threshold: int = 0,
        #: QueryIR owner signal for dialogue corpora.  When a question names a
        #: speaker explicitly, add a bounded score to that speaker's raw turns.
        #: This is a rerank only: it creates no candidates, edges or Token cost.
        speaker_owner_bonus: float = 0.0,
        #: Query-local witness closure.  Broad lexical-only graph edges remain
        #: disabled; instead, a named speaker's high-ranked turns may promote
        #: another turn from the same speaker when they share multiple
        #: memory-rare terms.  The finite bonus changes neither candidate count
        #: nor the evidence/answer Token budgets.
        query_witness_bonus: float = 0.0,
        query_witness_seed_count: int = 16,
        query_witness_rare_df: int = 4,
        query_witness_min_shared_terms: int = 2,
        queryir_soft_fallback: bool = False,
        queryir_soft_fallback_threshold: float = 0.80,
        #: Guarded single-fact route: direct source/fact ranking first, graph
        #: fallback on low confidence, and a bounded evidence pack on success.
        exact_lookup_fast_path: bool = False,
        exact_lookup_turn_limit: int = 16,
        exact_lookup_priority: bool = False,
        exact_lookup_priority_min_score: float = 1.50,
        exact_lookup_priority_bonus: float = 1.0,
        exact_lookup_priority_named_speakers_only: bool = False,
        #: Optional trusted local directory containing versioned compiled
        #: graph/turn/provenance sidecars.  SQLite remains the authority.
        compiled_cache_dir: str | Path | None = None,
        #: Frequency-aware admission prevents one-shot tenants from evicting a
        #: hot memory when a compiled sidecar can serve the cold request once.
        compiled_cache_admission: bool = True,
    ) -> None:
        self.store = store
        self.skip_traversal_on_certificate = skip_traversal_on_certificate
        self.rank_mandatory = rank_mandatory
        # H10 was added to the owner-rescue, traversal and proof-packing sets in
        # one batch.  The packing one measured -11pp on LoCoMo and is reverted;
        # these two and the manifest collection_key match were never measured
        # alone, and together they are the residual against the V5.6 baseline.
        # The graph term was `1.0 if reached else 0.0`, so a node three hops from
        # the seed contributed exactly what a seed did.  Measured consequence:
        # candidates reached by graph alone score 2.41 against 1.24 for ones with
        # a lexical match, take 25.9% of a 32-turn pack, and yield gold at 0.4%.
        # `graph_hop_decay ** hop` discounts by distance; 1.0 keeps the old flag.
        self.graph_hop_decay = graph_hop_decay
        self.expansion_beam = expansion_beam
        self.fusion = {**FUSION_DEFAULTS, **dict(fusion_weights or {})}
        self.session_router_k = session_router_k
        self.per_session_quota = per_session_quota
        self.session_flood_k = session_flood_k
        self.harness_set_cover = harness_set_cover
        self.hierarchical_routing = hierarchical_routing
        self.hierarchy_root_beam = max(1, hierarchy_root_beam)
        self.hierarchy_child_beam = max(1, hierarchy_child_beam)
        self.hierarchy_descent_beam = max(1, hierarchy_descent_beam)
        self.rare_lexical_relations = rare_lexical_relations
        self.query_gated_rare_lexical = query_gated_rare_lexical
        self.hierarchy_operator_aware = hierarchy_operator_aware
        self.h10_owner_rescue = h10_owner_rescue
        self.h10_traversal = h10_traversal
        self.manifest_collection_key = manifest_collection_key
        self.preferred_relations = preferred_relations
        self.fallback_relations = fallback_relations
        self.read_pool_size = store.enable_read_pool(read_pool_size)
        self.runtime = SQLiteSnapshotRuntime(
            store, max_cached_views=snapshot_cache_memories,
            max_cache_bytes=snapshot_cache_bytes)
        self.variant = NavigatorVariant(variant)
        self.dense_search = dense_search
        self.dense_search_many = dense_search_many
        self.fact_dense_search = fact_dense_search
        self.harness_profile = HarnessProfile(harness_profile) if harness_profile else None
        # Evidence budgets are only meaningful against the backbone's own
        # vocabulary.  The word-count estimate stays available as a labelled
        # fallback so a run without a local tokenizer still completes, but the
        # manifest then has to say the numbers are estimates.
        self.token_counter = token_counter or resolve_token_counter(DEFAULT_BACKBONE_MODEL)
        # Tokenization is deterministic for a frozen backbone and dominates the
        # evidence packer's CPU time.  Cache by source text so concurrent queries
        # do not repeatedly encode the same immutable turn.
        self._count_tokens_cached = lru_cache(maxsize=131_072)(
            self.token_counter.count)
        # Channel set for the fact reservoir, so F0-F5 can be ablated.
        self.fact_channels = tuple(fact_channels) if fact_channels is not None else FACT_CHANNELS
        # PR4b is measured but not yet promoted: on the fixed 200 it halves
        # binding coverage because operand owners are junk or unresolved, so it
        # stays opt-in until the owner resolution it depends on is fixed.
        self.binding_discriminant = binding_discriminant
        self.obligation_aware_packing = obligation_aware_packing
        self.precision_aware_packing = precision_aware_packing
        self.candidate_pool_limit = max(0, candidate_pool_limit)
        self.span_pack_window = max(0, span_pack_window)
        self.raw_fallback_reserve = max(0, raw_fallback_reserve)
        self.obligation_aware_relations = obligation_aware_relations
        self.native_seed_fusion = native_seed_fusion
        self.relational_view_scoring = relational_view_scoring
        self.query_relation_view = query_relation_view
        self.relational_view_named_speakers_only = (
            relational_view_named_speakers_only)
        if relational_consensus_bonus < 0:
            raise ValueError("relational_consensus_bonus must be non-negative")
        self.relational_consensus_bonus = relational_consensus_bonus
        self.dialogue_response_closure = dialogue_response_closure
        if dialogue_response_flood_threshold < 0:
            raise ValueError(
                "dialogue_response_flood_threshold must be non-negative")
        self.dialogue_response_flood_threshold = (
            dialogue_response_flood_threshold)
        if proof_priority_bonus is not None and proof_priority_bonus < 0:
            raise ValueError("proof_priority_bonus must be non-negative")
        self.proof_priority_bonus = proof_priority_bonus
        if proof_priority_flood_threshold < 0:
            raise ValueError("proof_priority_flood_threshold must be non-negative")
        self.proof_priority_flood_threshold = proof_priority_flood_threshold
        if speaker_owner_bonus < 0:
            raise ValueError("speaker_owner_bonus must be non-negative")
        self.speaker_owner_bonus = speaker_owner_bonus
        if query_witness_bonus < 0:
            raise ValueError("query_witness_bonus must be non-negative")
        if min(query_witness_seed_count, query_witness_rare_df,
               query_witness_min_shared_terms) <= 0:
            raise ValueError("query witness limits must be positive")
        self.query_witness_bonus = query_witness_bonus
        self.query_witness_seed_count = query_witness_seed_count
        self.query_witness_rare_df = query_witness_rare_df
        self.query_witness_min_shared_terms = query_witness_min_shared_terms
        self.queryir_soft_fallback = queryir_soft_fallback
        self.queryir_soft_fallback_threshold = max(
            0.0, min(1.0, queryir_soft_fallback_threshold))
        self.exact_lookup_fast_path = exact_lookup_fast_path
        self.exact_lookup_turn_limit = max(1, exact_lookup_turn_limit)
        self.exact_lookup_priority = exact_lookup_priority
        self.exact_lookup_priority_min_score = max(
            0.0, exact_lookup_priority_min_score)
        self.exact_lookup_priority_bonus = max(0.0, exact_lookup_priority_bonus)
        self.exact_lookup_priority_named_speakers_only = (
            exact_lookup_priority_named_speakers_only)
        self.compiled_sidecar = (
            CompiledMemorySidecar(compiled_cache_dir)
            if compiled_cache_dir is not None else None)
        self.compiled_cache_admission = compiled_cache_admission
        self._compiled_hydrations = 0
        self._compiled_admissions = 0
        self._compiled_bypasses = 0
        self._compiled_retained: dict[str, tuple[int, int]] = {}
        self._memory_frequency: dict[str, int] = {}
        self._frequency_observations = 0
        self._request_state = threading.local()
        self.metadata_cache_memories = max(1, metadata_cache_memories)
        self._metadata_lock = threading.RLock()
        self._turn_group_cache: OrderedDict[
            str, tuple[int, dict[str, tuple[str, ...]]]
        ] = OrderedDict()
        self._principal_cache: OrderedDict[str, tuple[int, object]] = OrderedDict()
        self._turn_search_cache: OrderedDict[str, TurnSearchIndex] = OrderedDict()
        self._turn_bundle_cache: OrderedDict[
            str, tuple[int, tuple[SourceTurn, ...], dict[str, SourceTurn],
                       dict[tuple[str, int], str]]
        ] = OrderedDict()
        self._evidence_index_cache: OrderedDict[
            str, tuple[int, dict[str, tuple[str, ...]],
                       dict[str, tuple[str, ...]],
                       dict[str, tuple[EvidenceMember, ...]]]
        ] = OrderedDict()
        self._query_witness_cache: OrderedDict[
            str, tuple[int, dict[str, frozenset[str]], Counter[str]]
        ] = OrderedDict()

    def _observe_memory(self, memory_id: str) -> int:
        self._frequency_observations += 1
        if self._frequency_observations % 4096 == 0:
            self._memory_frequency = {
                key: max(1, value // 2)
                for key, value in self._memory_frequency.items()}
        value = self._memory_frequency.get(memory_id, 0) + 1
        self._memory_frequency[memory_id] = value
        return value

    def _request_compiled_artifact(
            self, memory_id: str, version: int | None = None,
    ) -> CompiledMemoryArtifact | None:
        artifact = getattr(self._request_state, "artifact", None)
        if (artifact is not None and artifact.memory_id == memory_id
                and (version is None or artifact.graph_version == version)):
            return artifact
        return None

    def _hydrate_compiled_memory(self, memory_id: str) -> GraphReadView | None:
        """Install a cold memory's validated materialized indexes as one unit."""
        sidecar = self.compiled_sidecar
        if sidecar is None:
            return None
        frequency = self._observe_memory(memory_id)
        version, checksum = self.store.graph_identity(memory_id)
        cached = self.runtime.peek(memory_id, version)
        if cached is not None and cached.graph_checksum == checksum:
            return self.runtime.touch(memory_id, version)
        # GraphNavigator can also be used concurrently without the process
        # serving layer.  The metadata lock makes sidecar hydration single-flight
        # inside that case; runtime.install provides the final atomic swap.
        with self._metadata_lock:
            cached = self.runtime.peek(memory_id, version)
            if cached is not None and cached.graph_checksum == checksum:
                return self.runtime.touch(memory_id, version)
            artifact = sidecar.load(memory_id, version, checksum)
            if artifact is None:
                return None
            lru_keys = self.runtime.lru_keys()
            stats = self.runtime.cache_stats()
            capacity_pressure = (
                len(lru_keys) >= self.runtime.max_cached_views
                or int(stats["accounted_bytes"]) + artifact.view_retained_bytes
                > self.runtime.max_cache_bytes)
            if (self.compiled_cache_admission
                    and not getattr(self._request_state, "force_admit", False)
                    and capacity_pressure and lru_keys
                    and frequency <= self._memory_frequency.get(lru_keys[0][0], 0)):
                # The artifact is immutable and can serve this request directly.
                # Do not let a one-shot tenant evict four coordinated hot
                # structures (view, turn index, evidence index and principals).
                self._request_state.artifact = artifact
                self._compiled_bypasses += 1
                return artifact.view
            self.runtime.install(
                artifact.view, memory_id=memory_id,
                accounted_bytes=artifact.view_retained_bytes)
            self._turn_bundle_cache[memory_id] = (
                version, artifact.turn_index.turns,
                artifact.turn_index.turn_by_id,
                artifact.turn_by_session_index)
            self._turn_search_cache[memory_id] = artifact.turn_index
            self._evidence_index_cache[memory_id] = (
                version, artifact.groups_by_turn,
                artifact.turns_by_group, artifact.members_by_group)
            self._principal_cache[memory_id] = (version, artifact.principals)
            for cache in (
                    self._turn_bundle_cache, self._turn_search_cache,
                    self._evidence_index_cache, self._principal_cache):
                cache.move_to_end(memory_id)
                while len(cache) > self.metadata_cache_memories:
                    cache.popitem(last=False)
            self._compiled_hydrations += 1
            self._compiled_admissions += 1
            self._compiled_retained[memory_id] = (
                version, artifact.total_retained_bytes)
            return artifact.view

    def precompile_memory(self, memory_id: str, *, force: bool = False,
                          account_bytes: bool = True) -> CompiledMemoryArtifact:
        """Build and atomically publish one disposable compiled sidecar."""
        sidecar = self.compiled_sidecar
        if sidecar is None:
            raise RuntimeError("compiled_cache_dir is required for precompilation")
        version, checksum = self.store.graph_identity(memory_id)
        if not force:
            cached = sidecar.load(memory_id, version, checksum)
            if cached is not None:
                return cached
        view = self.runtime.view(memory_id)
        turns_, by_id, by_session_index = self._turn_bundle(
            memory_id, view.graph_version)
        turn_index = self._turn_search_index(memory_id, turns_)
        groups_by_turn, turns_by_group, members_by_group = self._evidence_indexes(
            memory_id, view.graph_version)
        principals = self._principals(memory_id, view)
        artifact = CompiledMemoryArtifact(
            schema_version=COMPILED_MEMORY_SCHEMA,
            memory_id=memory_id,
            graph_version=view.graph_version,
            graph_checksum=view.graph_checksum,
            created_ns=time.time_ns(),
            view=view,
            turns=turn_index.turns,
            turn_by_id=turn_index.turn_by_id,
            turn_by_session_index=by_session_index,
            turn_index=turn_index,
            groups_by_turn=groups_by_turn,
            turns_by_group=turns_by_group,
            members_by_group=members_by_group,
            principals=principals,
        )
        if account_bytes:
            artifact = artifact.with_accounting()
        else:
            artifact = replace(
                artifact,
                view_retained_bytes=view.estimated_bytes,
                total_retained_bytes=view.estimated_bytes)
        if self.store.graph_identity(memory_id) != (
                artifact.graph_version, artifact.graph_checksum):
            raise RuntimeError(
                f"graph {memory_id!r} changed during sidecar compilation")
        sidecar.save(artifact)
        return artifact

    def warm_memory(self, memory_id: str, queries: Sequence[str],
                    budget: QueryBudget) -> GraphReadView:
        """Explicitly admit a memory while running representative warm queries."""
        if not queries:
            raise ValueError("at least one warm query is required")
        self._request_state.force_admit = True
        try:
            for query in queries:
                self.navigate(memory_id, query, budget)
            view = self.runtime.peek(memory_id)
            return view if view is not None else self.runtime.view(memory_id)
        finally:
            self._request_state.force_admit = False

    def cache_stats(self) -> dict[str, object]:
        """Expose view lifecycle, sidecar and metadata-cache observability."""
        retained_memories = set(self._turn_bundle_cache)
        retained_memories.update(self._turn_search_cache)
        retained_memories.update(self._evidence_index_cache)
        retained_memories.update(self._principal_cache)
        for memory_id, (version, _bytes) in self._compiled_retained.items():
            if self.runtime.peek(memory_id, version) is not None:
                retained_memories.add(memory_id)
        return {
            "runtime": self.runtime.cache_stats(),
            "metadata_entries": {
                "turn_bundles": len(self._turn_bundle_cache),
                "turn_search_indexes": len(self._turn_search_cache),
                "evidence_indexes": len(self._evidence_index_cache),
                "principal_registries": len(self._principal_cache),
            },
            "compiled": ({
                **self.compiled_sidecar.stats(),
                "hydrations": self._compiled_hydrations,
                "admissions": self._compiled_admissions,
                "bypasses": self._compiled_bypasses,
                "frequency_entries": len(self._memory_frequency),
                "retained_artifact_bytes": sum(
                    row[1] for memory_id, row in self._compiled_retained.items()
                    if memory_id in retained_memories),
            } if self.compiled_sidecar is not None else {
                "enabled": False, "hydrations": 0,
            }),
        }

    def _turn_tokens(self, turn: SourceTurn) -> int:
        return self._count_tokens_cached(turn.raw_text)

    def _turn_bundle(
        self, memory_id: str, version: int,
    ) -> tuple[tuple[SourceTurn, ...], dict[str, SourceTurn],
               dict[tuple[str, int], str]]:
        """Return one immutable raw-turn projection per visible graph version."""
        artifact = self._request_compiled_artifact(memory_id, version)
        if artifact is not None:
            return (artifact.turn_index.turns, artifact.turn_index.turn_by_id,
                    artifact.turn_by_session_index)
        with self._metadata_lock:
            cached = self._turn_bundle_cache.get(memory_id)
            if cached is None or cached[0] != version:
                turns_ = tuple(self.store.turns(memory_id))
                cached = (
                    version,
                    turns_,
                    {row.turn_id: row for row in turns_},
                    {(row.session_id, row.turn_index): row.turn_id for row in turns_},
                )
                self._turn_bundle_cache[memory_id] = cached
            self._turn_bundle_cache.move_to_end(memory_id)
            while len(self._turn_bundle_cache) > self.metadata_cache_memories:
                self._turn_bundle_cache.popitem(last=False)
            return cached[1], cached[2], cached[3]

    def _query_witness_features(
        self, memory_id: str, version: int, turns_: Sequence[SourceTurn],
    ) -> tuple[dict[str, frozenset[str]], Counter[str]]:
        """Cache immutable turn terms/DF used by query-local witness closure."""

        with self._metadata_lock:
            cached = self._query_witness_cache.get(memory_id)
            if cached is None or cached[0] != version:
                terms_by_turn = {
                    turn.turn_id: content_terms(turn.raw_text) for turn in turns_}
                document_frequency: Counter[str] = Counter(
                    term for values in terms_by_turn.values() for term in values)
                cached = (version, terms_by_turn, document_frequency)
                self._query_witness_cache[memory_id] = cached
            self._query_witness_cache.move_to_end(memory_id)
            while len(self._query_witness_cache) > self.metadata_cache_memories:
                self._query_witness_cache.popitem(last=False)
            return cached[1], cached[2]

    def _evidence_indexes(
        self, memory_id: str, version: int,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]],
               dict[str, tuple[EvidenceMember, ...]]]:
        """Compile both directions of provenance once per graph snapshot."""
        artifact = self._request_compiled_artifact(memory_id, version)
        if artifact is not None:
            return (artifact.groups_by_turn, artifact.turns_by_group,
                    artifact.members_by_group)
        with self._metadata_lock:
            cached = self._evidence_index_cache.get(memory_id)
            if cached is None or cached[0] != version:
                groups_by_turn: dict[str, list[str]] = defaultdict(list)
                turns_by_group: dict[str, tuple[str, ...]] = {}
                members_by_group: dict[str, tuple[EvidenceMember, ...]] = {}
                for group in self.store.evidence_groups(memory_id):
                    turns_by_group[group.evidence_group_id] = tuple(
                        member.turn_id for member in group.members)
                    members_by_group[group.evidence_group_id] = group.members
                    for member in group.members:
                        groups_by_turn[member.turn_id].append(group.evidence_group_id)
                cached = (
                    version,
                    {key: tuple(dict.fromkeys(value))
                     for key, value in groups_by_turn.items()},
                    turns_by_group,
                    members_by_group,
                )
                self._evidence_index_cache[memory_id] = cached
            self._evidence_index_cache.move_to_end(memory_id)
            while len(self._evidence_index_cache) > self.metadata_cache_memories:
                self._evidence_index_cache.popitem(last=False)
            return cached[1], cached[2], cached[3]

    def route_sessions(self, all_turns, query: str, limit: int) -> tuple[str, ...]:
        """Rank sessions as units by BM25 of the question against session text.

        The session score in `_top_sessions` is a max over turn scores, which is
        a by-product of turn ranking rather than a session signal: evidence
        spread over several turns of one session does not accumulate.  This
        scores the session itself, with the length normalisation that a max
        cannot have -- sessions here run from 6 to 60 turns.
        """
        if not all_turns:
            return ()
        return self._turn_search_index(all_turns[0].memory_id, all_turns).rank_sessions(
            query, limit)

    def _turn_search_index(self, memory_id: str,
                           turns_: Sequence[SourceTurn]) -> TurnSearchIndex:
        artifact = self._request_compiled_artifact(memory_id)
        if artifact is not None:
            return artifact.turn_index
        signature = ((len(turns_), turns_[-1].turn_id, turns_[-1].content_hash)
                     if turns_ else (0, "", ""))
        with self._metadata_lock:
            cached = self._turn_search_cache.get(memory_id)
            if cached is not None and cached.signature == signature:
                self._turn_search_cache.move_to_end(memory_id)
                return cached
            index = TurnSearchIndex(turns_)
            self._turn_search_cache[memory_id] = index
            bundle = self._turn_bundle_cache.get(memory_id)
            if bundle is not None:
                # One authoritative id map is enough.  TurnSearchIndex already
                # retains it, so make the bundle share that table instead of
                # holding an equal per-memory dict for every worker.
                self._turn_bundle_cache[memory_id] = (
                    bundle[0], index.turns, index.turn_by_id, bundle[3])
            self._turn_search_cache.move_to_end(memory_id)
            while len(self._turn_search_cache) > self.metadata_cache_memories:
                self._turn_search_cache.popitem(last=False)
            return index

    def _principals(self, memory_id: str, view):
        version = view.graph_version
        artifact = self._request_compiled_artifact(memory_id, version)
        if artifact is not None:
            return artifact.principals
        with self._metadata_lock:
            cached = self._principal_cache.get(memory_id)
            if cached is None or cached[0] != version:
                cached = (version, build_principal_registry(self.store, memory_id, view))
                self._principal_cache[memory_id] = cached
            self._principal_cache.move_to_end(memory_id)
            while len(self._principal_cache) > self.metadata_cache_memories:
                self._principal_cache.popitem(last=False)
            return cached[1]

    def _turn_groups(self, memory_id: str, version: int) -> dict[str, tuple[str, ...]]:
        return self._evidence_indexes(memory_id, version)[0]

    def navigate(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        try:
            if self.harness_profile in {None, HarnessProfile.H0_N5, HarnessProfile.H1_TELEMETRY}:
                result = self._navigate_legacy(memory_id, query, budget)
                if self.harness_profile == HarnessProfile.H1_TELEMETRY:
                    trace = dict(result.trace)
                    trace["exhaustion_telemetry"] = {
                        "search_budget_exhausted": bool(result.budget_exhausted and result.visited_nodes >= budget.max_visited_nodes),
                        "node_cap_reached": result.visited_nodes >= budget.max_visited_nodes,
                        "edge_cap_reached": result.visited_edges >= budget.max_visited_edges,
                        "hop_cap_reached": False,
                        "frontier_truncated": result.frontier_peak >= budget.max_frontier,
                        "pack_turn_cap_reached": len(result.packed_turn_ids) >= budget.max_evidence_turns,
                        "pack_token_cap_reached": result.evidence_tokens >= budget.max_evidence_tokens,
                        "normal_candidate_drop": len(result.dropped_turn_ids) > 0,
                    }
                    return replace(result, trace=trace)
                return result
            return self._navigate_harness(memory_id, query, budget)
        finally:
            self._request_state.artifact = None

    def _navigate_legacy(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        started = time.perf_counter()
        stage_times: dict[str, float] = {}
        compiled_view = self._hydrate_compiled_memory(memory_id)
        view = compiled_view or self.runtime.view(memory_id)
        all_turns, by_id, _by_session_index = self._turn_bundle(
            memory_id, view.graph_version)
        query_terms = content_terms(query)

        tick = time.perf_counter()
        scores = self._raw_candidates(memory_id, query, query_terms, all_turns)
        stage_times["seed_fusion"] = (time.perf_counter() - tick) * 1000
        top_sessions = self._top_sessions(scores, by_id, limit=8)
        candidate_ids = set(scores)
        for turn in all_turns:
            if turn.session_id in top_sessions:
                candidate_ids.add(turn.turn_id)

        node_relevance = self._node_relevance(view.nodes, query_terms)
        seeds = tuple(node_id for node_id, _ in sorted(
            node_relevance.items(), key=lambda item: (-item[1], item[0])
        )[: min(12, budget.max_frontier)])

        visited: list[str] = []
        proof: list[ProofStep] = []
        frontier_peak = len(seeds)
        budget_exhausted = False
        tick = time.perf_counter()
        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N3_PRIORITY]:
            visited, proof, frontier_peak, budget_exhausted = self._priority_expand(
                view, seeds, node_relevance, query, budget,
                certificate_guided=(
                    VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N4_CERTIFICATE]
                ),
            )
        elif VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N2_PROVENANCE]:
            visited = list(seeds[: budget.max_visited_nodes])
        stage_times["graph_read_view"] = (time.perf_counter() - tick) * 1000

        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N2_PROVENANCE]:
            tick = time.perf_counter()
            group_ids = view.evidence_group_ids_for_nodes(visited or seeds)
            turns_by_group = self._evidence_indexes(
                memory_id, view.graph_version)[1]
            for group_id in group_ids:
                candidate_ids.update(turns_by_group.get(group_id, ()))
            stage_times["provenance_closure"] = (time.perf_counter() - tick) * 1000

        kind, required_slots, negative_required = question_slots(query)
        candidate_rows = self._candidate_rows(
            candidate_ids, by_id, scores, query_terms, visited, view, proof
        )
        tick = time.perf_counter()
        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N5_SET_COVER]:
            packed, dropped, coverage = self._set_cover(
                candidate_rows, by_id, kind, required_slots, budget
            )
        else:
            packed, dropped, coverage = self._rank_pack(candidate_rows, by_id, budget)
        stage_times["evidence_pack"] = (time.perf_counter() - tick) * 1000

        covered = tuple(slot for slot in required_slots if coverage.get(slot))
        if negative_required and not any(
            content_terms(by_id[turn_id].raw_text) & NEGATIVE_TERMS for turn_id in packed
        ):
            missing = tuple(slot for slot in required_slots if slot not in covered) + ("negative_scope",)
        else:
            missing = tuple(slot for slot in required_slots if slot not in covered)
        certificate = EvidenceCertificate(
            question_kind=kind,
            required_slots=required_slots + (("negative_scope",) if negative_required else ()),
            covered_slots=covered + (("negative_scope",) if negative_required and "negative_scope" not in missing else ()),
            missing_slots=missing,
            complete=not missing,
            iterations=min(budget.max_iterations, max(1, len(visited) // max(1, budget.max_frontier) + 1)),
            negative_scope_required=negative_required,
        )
        evidence_tokens = sum(estimate_tokens(by_id[item].raw_text) for item in packed)
        retrieved_sessions = tuple(dict.fromkeys(by_id[item].session_id for item in packed))
        stage_times["total"] = (time.perf_counter() - started) * 1000
        graph_id = stable_id(
            "graph-artifact", memory_id, view.graph_version,
            view.graph_checksum,
        )
        return NavigationResult(
            question_id=stable_id("question", memory_id, query),
            memory_id=memory_id,
            graph_artifact_id=graph_id,
            retrieved_session_ids=retrieved_sessions,
            retrieved_turn_ids=packed,
            proof=tuple(proof),
            visited_nodes=len(visited),
            visited_edges=len(proof),
            frontier_peak=frontier_peak,
            evidence_tokens=evidence_tokens,
            budget_exhausted=budget_exhausted or len(candidate_rows) > len(packed),
            trace={
                "variant": str(self.variant),
                "top_sessions": top_sessions,
                "candidate_count": len(candidate_rows),
                "semantic_navigation_excludes_provenance_edges": True,
            },
            seed_node_ids=seeds,
            visited_path_node_ids=tuple(visited),
            slot_coverage={key: tuple(value) for key, value in coverage.items()},
            certificate=certificate,
            candidate_scores=tuple(candidate_rows),
            packed_turn_ids=packed,
            dropped_turn_ids=dropped,
            stage_latency_ms=stage_times,
            graph_only_candidate_turn_ids=tuple(row.turn_id for row in candidate_rows
                                                if row.graph_score > 0 and not any((row.exact_score,
                                                   row.bm25_score, row.dense_score))),
            relation_trace=tuple(proof),
            first_hit_relations={row.turn_id: row.relation_contributions[0]
                                 for row in candidate_rows if row.relation_contributions},
        )

    def _navigate_harness(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        """V5.5 query-only Graph Harness over an immutable V5.4 graph."""
        started = time.perf_counter(); stage_times: dict[str, float] = {}
        tick = time.perf_counter()
        compiled_view = self._hydrate_compiled_memory(memory_id)
        view = compiled_view or self.runtime.view(memory_id)
        all_turns, by_id, by_session_index = self._turn_bundle(
            memory_id, view.graph_version)
        turn_index = self._turn_search_index(memory_id, all_turns)
        stage_times["memory_hydrate"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        registry = self._principals(memory_id, view)
        compiled_ir = compile_query(query, view, registry=registry)
        profile = self.harness_profile
        if profile is HarnessProfile.H11_UNIFIED_IR:
            ir = compiled_ir.promote_ast()
            if (self.queryir_soft_fallback
                    and compiled_ir.compile_confidence
                    < self.queryir_soft_fallback_threshold):
                ir = ir.soften_with_legacy(compiled_ir)
        else:
            ir = compiled_ir
        stage_times["query_compile"] = (time.perf_counter() - tick) * 1000
        ast_profiles = {HarnessProfile.H10_AST, HarnessProfile.H11_UNIFIED_IR}
        use_postings = profile in {HarnessProfile.H2_POSTINGS, HarnessProfile.H3_MULTI_ANCHOR,
                                   HarnessProfile.H4_SCHEDULER, HarnessProfile.H5_ALGEBRA,
                                   HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                                   HarnessProfile.H9_FACT_RESERVOIR,
                                   *ast_profiles}
        use_rrf = profile in {HarnessProfile.H3_MULTI_ANCHOR, HarnessProfile.H4_SCHEDULER,
                              HarnessProfile.H5_ALGEBRA, HarnessProfile.H6_PROOF_PACKING,
                              HarnessProfile.H8_RESERVOIR, HarnessProfile.H9_FACT_RESERVOIR,
                              *ast_profiles}
        # H7 is the first profile with the wide reservoir; H2-H6 keep the narrow
        # V5.5 seeding so the ablation ladder stays interpretable.
        wide_reservoir = profile in {HarnessProfile.H8_RESERVOIR,
                                     HarnessProfile.H9_FACT_RESERVOIR,
                                     *ast_profiles}
        fact_reservoir_enabled = profile in {HarnessProfile.H9_FACT_RESERVOIR,
                                             *ast_profiles}
        execute_ast = profile in ast_profiles
        named_multi_party = has_named_multi_party(all_turns)
        relational_view_active = bool(
            not self.relational_view_named_speakers_only or named_multi_party)
        tick = time.perf_counter()
        seeded = seed_operands(self.store, view, memory_id, ir, all_turns, dense_search=self.dense_search,
                               dense_search_many=self.dense_search_many,
                               use_rrf=use_rrf, use_postings=use_postings,
                               reservoir_limit=budget.max_candidate_reservoir,
                               max_views_per_operand=budget.max_query_views_per_operand,
                               node_budget=budget.seed_nodes,
                               wide_reservoir=wide_reservoir,
                               hierarchical_routing=(execute_ast and self.hierarchical_routing),
                               hierarchy_root_beam=self.hierarchy_root_beam,
                               hierarchy_child_beam=self.hierarchy_child_beam,
                               hierarchy_operator_aware=self.hierarchy_operator_aware,
                               turn_index=turn_index,
                               native_bm25=self.native_seed_fusion,
                               relational_view_scoring=(
                                   self.relational_view_scoring
                                   and relational_view_active),
                               query_relation_view=(
                                   self.query_relation_view
                                   and relational_view_active))
        stage_times["seed_fusion"] = (time.perf_counter() - tick) * 1000
        if seeded.stats.get("hierarchical_route_ms"):
            stage_times["hierarchical_route"] = float(
                seeded.stats["hierarchical_route_ms"])
        tick = time.perf_counter()
        route_closure = (self._route_terminal_closure(view, seeded.semantic_node_ids)
                         if profile in {HarnessProfile.H2_POSTINGS, HarnessProfile.H3_MULTI_ANCHOR} else ())
        # Capped at the seed budget, not the traversal budget: handing the
        # scheduler max_visited_nodes seeds left it no room to expand.
        semantic_seeds = tuple(dict.fromkeys((*seeded.semantic_node_ids, *route_closure)))[:budget.seed_nodes]
        direct_owner_terminals = self._owner_terminal_postings(
            view, semantic_seeds, seeded.raw_scores, query
        ) if profile in {HarnessProfile.H4_SCHEDULER, HarnessProfile.H5_ALGEBRA,
                         HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                         HarnessProfile.H9_FACT_RESERVOIR,
                         *(ast_profiles if self.h10_owner_rescue else set())} else ()
        owners = {
            operand.operand_id: set().union(*(set(view.owner_alias_index.get(alias.casefold(), ()))
                                               for alias in operand.owner_aliases))
            for operand in ir.operands
        }
        exact_priority_candidate = bool(
            self.exact_lookup_priority
            and (not self.exact_lookup_priority_named_speakers_only
                 or named_multi_party))
        exact_lookup_candidate = bool(
            (self.exact_lookup_fast_path or exact_priority_candidate)
            and exact_lookup_eligible(ir))
        direct_fact_ids: tuple[str, ...] = ()
        direct_fact_turn_scores: dict[str, float] = {}
        strict_fact_turn_scores: dict[str, float] = {}
        exact_lookup_scores: dict[str, float] = {}
        exact_lookup_confident = False
        exact_lookup_trace: dict[str, object] = {
            "eligible": exact_lookup_candidate,
            "confident": False,
            "direct_fact_count": 0,
            "direct_fact_turn_count": 0,
        }
        if exact_lookup_candidate:
            query_content = content_terms(query)
            strict_fact_ids: list[str] = []
            for operand in ir.operands:
                operand_owners = tuple(sorted(owners.get(operand.operand_id, ())))
                # An owner-only posting is not exact: on LoCoMo it expands one
                # speaker to hundreds of facts.  Require the composite key here;
                # the lexical fact index below remains the owner-parser fallback.
                if operand_owners and operand.predicate_candidates:
                    strict_fact_ids.extend(view.lookup_facts(
                        owner_ids=operand_owners,
                        predicates=operand.predicate_candidates,
                        limit=12,
                        rank_terms=query_content))
            lexical_fact_ids = view.facts_for_terms(query_content, limit=16)
            dense_fact_rows = (
                self.fact_dense_search(
                    memory_id, query,
                    tuple(node.node_id for node in view.nodes.values()
                          if node.node_type == NodeType.CANONICAL_FACT),
                    24)
                if self.fact_dense_search is not None else ())
            dense_fact_scores = {
                fact_id: max(
                    0.0,
                    1.0 - rank / max(24, len(dense_fact_rows)),
                ) * 0.8 + max(0.0, min(1.0, float(similarity))) * 0.2
                for rank, (fact_id, similarity) in enumerate(
                    dense_fact_rows)
            }
            direct_fact_ids = tuple(dict.fromkeys((
                *(fact_id for fact_id, _score in dense_fact_rows),
                *strict_fact_ids, *lexical_fact_ids)))
            strict_fact_set = frozenset(strict_fact_ids)
            _groups_by_turn, direct_group_turns, _members = (
                self._evidence_indexes(memory_id, view.graph_version))
            for fact_id in direct_fact_ids:
                node = view.nodes.get(fact_id)
                if node is None:
                    continue
                attrs = node.attributes
                surface = content_terms(
                    node.summary + " " + " ".join(
                        str(attrs.get(key, "")) for key in (
                            "owner_id", "predicate", "value", "scope",
                            "collection_key")))
                fact_score = _weighted_term_coverage(
                    query_content, surface, turn_index)
                fact_score = max(
                    fact_score, dense_fact_scores.get(fact_id, 0.0))
                if fact_id in strict_fact_set:
                    fact_score += 0.20
                for group_id in view.terminal_groups_for_nodes((fact_id,)):
                    for turn_id in direct_group_turns.get(group_id, ()):
                        direct_fact_turn_scores[turn_id] = max(
                            direct_fact_turn_scores.get(turn_id, 0.0),
                            fact_score)
                        if fact_id in strict_fact_set:
                            strict_fact_turn_scores[turn_id] = max(
                                strict_fact_turn_scores.get(turn_id, 0.0),
                                fact_score)
            exact_lookup_scores, exact_lookup_confident, ranking_trace = (
                _exact_lookup_ranking(
                    seeded, turn_index, query, direct_fact_turn_scores))
            strict_ordered = tuple(sorted(
                strict_fact_turn_scores,
                key=lambda turn_id: (
                    -exact_lookup_scores.get(turn_id, 0.0), turn_id)))
            strict_top_id = strict_ordered[0] if strict_ordered else ""
            exact_lookup_trace.update({
                **ranking_trace,
                "confident": exact_lookup_confident,
                "direct_fact_count": len(direct_fact_ids),
                "direct_fact_turn_count": len(direct_fact_turn_scores),
                "strict_fact_count": len(strict_fact_set),
                "strict_fact_turn_count": len(strict_fact_turn_scores),
                "strict_top_turn_id": strict_top_id,
                "strict_top_score": exact_lookup_scores.get(
                    strict_top_id, 0.0),
                "strict_top_provenance_score": strict_fact_turn_scores.get(
                    strict_top_id, 0.0),
                "dense_fact_count": len(dense_fact_rows),
                "direct_fact_turn_ids": tuple(sorted(
                    direct_fact_turn_scores)),
                "dense_fact_ids": tuple(
                    fact_id for fact_id, _score in dense_fact_rows),
            })
        exact_lookup_fast_active = bool(
            self.exact_lookup_fast_path and exact_lookup_confident)
        exact_lookup_priority_active = bool(
            exact_priority_candidate
            and exact_lookup_trace.get("strict_top_turn_id")
            and float(exact_lookup_trace.get("strict_top_score", 0.0))
            >= self.exact_lookup_priority_min_score
            and float(exact_lookup_trace.get(
                "strict_top_provenance_score", 0.0)) >= 0.45)
        exact_lookup_trace.update({
            "fast_path_active": exact_lookup_fast_active,
            "priority_active": exact_lookup_priority_active,
            "priority_min_score": self.exact_lookup_priority_min_score,
            "priority_bonus": self.exact_lookup_priority_bonus,
            "priority_fact_only": True,
        })
        initial_bindings = bind_facts(view, owners, ir.operands, semantic_seeds)
        initial_closure = evaluate_algebra(ir.operator, initial_bindings, (item.operand_id for item in ir.operands),
                                           distinct_by=ir.distinct_by, collection_complete=False)
        initial_certificate = evaluate_certificate(ir, initial_closure, no_progress=not initial_bindings)
        stage_times["seed_binding"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        # The shortcut below skips graph traversal whenever the *initial*
        # certificate already claims completeness.  That certificate is a pre-pack
        # flag over "the operand has at least one binding" and was measured not to
        # predict correctness at all (rho=+0.103, CI [-0.04,+0.25]), so it is a
        # weak signal being used to switch off the graph.  Settable for ablation.
        if exact_lookup_fast_active:
            schedule = ScheduleResult(semantic_seeds, (), {}, {
                "node_cap_reached": False, "edge_cap_reached": False,
                "hop_cap_reached": False, "frontier_truncated": False,
            }, {node_id: 0 for node_id in semantic_seeds})
        elif (self.skip_traversal_on_certificate
                and profile in {HarnessProfile.H5_ALGEBRA, HarnessProfile.H6_PROOF_PACKING,
                                HarnessProfile.H8_RESERVOIR,
                                HarnessProfile.H9_FACT_RESERVOIR}
                and initial_certificate.complete):
            schedule = ScheduleResult(semantic_seeds, (), {}, {
                "node_cap_reached": False, "edge_cap_reached": False,
                "hop_cap_reached": False, "frontier_truncated": False,
            }, {node_id: 0 for node_id in semantic_seeds})
        else:
            schedule = schedule_relations(view, ir, semantic_seeds, budget,
                                          preferred_relations=self.preferred_relations,
                                          fallback_relations=self.fallback_relations,
                                          expansion_beam=self.expansion_beam,
                                          hierarchy_descent_beam=(
                                              self.hierarchy_descent_beam),
                                          rare_lexical_relations=(
                                              self.rare_lexical_relations),
                                          query_gated_rare_lexical=(
                                              self.query_gated_rare_lexical),
                                          obligation_aware_relations=(
                                              self.obligation_aware_relations),
                                          structured=profile in {HarnessProfile.H4_SCHEDULER,
                                                                 HarnessProfile.H5_ALGEBRA,
                                                                 HarnessProfile.H6_PROOF_PACKING,
                                                                 HarnessProfile.H8_RESERVOIR,
                                                                 HarnessProfile.H9_FACT_RESERVOIR,
                                                                 *(ast_profiles
                                                                   if self.h10_traversal else set())})
        stage_times["graph_read_view"] = (time.perf_counter() - tick) * 1000
        fact_ids = (set(semantic_seeds) | set(schedule.visited_node_ids)
                    | set(direct_owner_terminals)
                    | (set(direct_fact_ids)
                       if exact_lookup_fast_active else set())
                    | (set(strict_fact_ids)
                       if exact_lookup_priority_active else set()))
        fact_reservoir = None
        if fact_reservoir_enabled:
            # Facts the narrow path already reached are the precision core and are
            # never displaced; the reservoir only adds rescues, and only the
            # bounded active shortlist reaches binding.
            tick = time.perf_counter()
            core_facts = tuple(node_id for node_id in fact_ids if node_id in view.nodes
                               and view.nodes[node_id].node_type == NodeType.CANONICAL_FACT)
            fact_reservoir = select_active_facts(
                build_fact_reservoir(
                    view, ir, seeded,
                    self._turn_groups(memory_id, view.graph_version),
                                     core_fact_ids=core_facts,
                                     channels=self.fact_channels),
                view, ir,
                per_operand=budget.max_active_facts_per_operand,
                total=budget.max_active_facts)
            fact_ids |= set(fact_reservoir.active)
            stage_times["fact_reservoir"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        # Reconstruct complete seed-to-node provenance.  The previous map kept
        # only the final edge for ``step.dst_id``; that was enough to say a fact
        # was graph-reached, but not enough to topologically arrange its source
        # turns.  Relax paths in hop order and retain the shortest stable route.
        node_paths: dict[str, tuple[str, ...]] = {
            node_id: () for node_id in semantic_seeds
        }
        proof_by_hop = sorted(schedule.proof, key=lambda step: (
            schedule.node_hops.get(step.from_id, 1 << 20),
            schedule.node_hops.get(step.to_id, 1 << 20),
            step.edge_id, step.from_id, step.to_id))
        for step in proof_by_hop:
            parent = node_paths.get(step.from_id)
            if parent is None:
                continue
            candidate = (*parent, step.edge_id)
            existing = node_paths.get(step.to_id)
            if existing is None or (len(candidate), candidate) < (
                    len(existing), existing):
                node_paths[step.to_id] = candidate
        paths = {
            node_id: path for node_id, path in node_paths.items() if path
        }
        binding_reasons: dict[str, int] = {}
        if fact_reservoir_enabled and self.binding_discriminant:
            # A wide fact pool scored by the permissive V5.5 rule would
            # manufacture bindings, so H9 uses the dimension-based discriminant:
            # hard conflicts veto, and an ownerless operand needs corroboration.
            projected = {row.fact_id: bool(row.source_rank) for row in fact_reservoir.entries}
            bindings, binding_reasons = bind_facts_discriminant(
                view, owners, ir.operands, fact_ids, paths,
                query_terms=content_terms(query), source_projected=projected,
                temporal_constraint=(ir.slots.temporal_key if ir.slots is not None else None))
        else:
            bindings = bind_facts(view, owners, ir.operands, fact_ids, paths)
        collection_ops = {"union_distinct", "intersection_distinct", "group_by_owner", "count_distinct"}
        if str(ir.operator) in collection_ops and any(item.predicate_candidates for item in ir.operands):
            # Owner equality alone is insufficient proof for a list/count.  It
            # previously made every fact spoken by an owner mandatory and
            # crowded out lossless terminal rescue evidence.  Measured: removing
            # this blunt threshold doubles binding coverage (10% -> 20%) but
            # costs 4pp of strict all-hit, because the extra bindings are
            # low-precision and crowd the pack.  It stays until the discriminant
            # can replace it with something better than a single float.
            bindings = tuple(item for item in bindings if item.confidence >= 0.70)
        manifests = [view.nodes[item] for item in fact_ids if item in view.nodes and view.nodes[item].node_type in {
            NodeType.COLLECTION_SCOPE, NodeType.COLLECTION_MANIFEST
        }]
        collection_complete = str(ir.operator) not in collection_ops or all(
            any((not owners.get(operand.operand_id) or str(node.attributes.get("owner_id", "")) in owners[operand.operand_id])
                and (not operand.predicate_candidates or any(
                    content_terms(str(candidate)) & content_terms(str(node.attributes.get("predicate", "")))
                    for candidate in operand.predicate_candidates))
                for node in manifests)
            for operand in ir.operands
        )
        closure = evaluate_algebra(ir.operator, bindings, (item.operand_id for item in ir.operands),
                                   distinct_by=ir.distinct_by, collection_complete=collection_complete)
        # H10 runs the AST alongside the legacy closure rather than replacing it:
        # the packer and certificate still consume ``closure``, so the frozen
        # ladder's behaviour is untouched and only the answer stage sees members.
        algebra_result = None
        algebra_bindings = ()
        # Set by the AST branch when a manifest is matched; read again below by
        # the mandatory-packing step, so it must exist on every path.
        member_scope_by_operand: dict[str, set[str]] = {}
        if execute_ast and ir.ast is not None:
            # PR2b compiled the AST in shadow mode with its own operand ids, so
            # the bindings -- produced against the legacy ir.operands -- carry
            # labels the AST never mentions.  Left unmapped, every binding is
            # filtered out and the algebra returns zero members: measured as
            # closed_form_rate 0.0 with 15 facts bound.  Both operand lists are
            # built from the same owner rows in the same order, so position is
            # the correspondence.
            ast_specs = ir.ast_operands or ir.operands
            legacy_ids = [item.operand_id for item in ir.operands]
            ast_ids = [item.operand_id for item in ast_specs]
            remap = (dict(zip(legacy_ids, ast_ids))
                     if len(legacy_ids) == len(ast_ids) else {})
            ast_bindings = tuple(
                replace(row, operand_id=remap[row.operand_id])
                if row.operand_id in remap else row
                for row in bindings)
            # A count must range over a collection, not over whatever bound.
            # Counting the raw binding set produced 15 "antique items" that were
            # actually unrelated facts, reported with scope_complete=True -- a
            # confidently wrong number, which is worse than declining to answer.
            # So an operand is only closed when a manifest matches it on owner
            # *and* predicate, and the count is then restricted to that
            # manifest's members.
            closed_by_operand: dict[str, bool] = {}
            for operand, legacy in zip(ast_specs, legacy_ids + [""] * len(ast_specs)):
                # Match on the *question's* words, not on the operand's
                # predicate candidates: those are retrieved from the graph by
                # embedding similarity and contain none of the question's terms,
                # so matching against them selected nearly every manifest and a
                # count over "antique items" returned 15 unrelated facts.
                # Measured on the development set: matching on all question
                # content terms beats head+action on every axis (recall .382 vs
                # .337, all-hit .193 vs .160) for +0.8pp of coverage.
                question = frozenset(ir.slots.content_terms)
                matched = [
                    node for node in manifests
                    if node.node_type == NodeType.COLLECTION_MANIFEST
                    and bool(node.attributes.get("closed"))
                    and (not owners.get(legacy)
                         or str(node.attributes.get("owner_id", "")) in owners[legacy])
                    and question and (
                        (question & content_terms(str(node.attributes.get("predicate", ""))))
                        # collection_key is the class of thing the manifest
                        # collects, which is the noun an aggregation question
                        # actually names ("how many *model kits*").  Once the
                        # predicate leaves the chain key it is also the only
                        # field that identifies the manifest at all.
                        or (self.manifest_collection_key
                            and (question & content_terms(str(node.attributes.get("collection_key", "")))))
                        or any(question & content_terms(str(value))
                               for value in node.attributes.get("value_keys", ())))
                ]
                closed_by_operand[operand.operand_id] = bool(matched)
                scoped_members: set[str] = set()
                for node in matched:
                    scoped_members.update(str(item) for item in
                                          node.attributes.get("member_ids", ()))
                if scoped_members:
                    member_scope_by_operand[operand.operand_id] = scoped_members
            if member_scope_by_operand:
                # Scope each operand independently.  A global union lets facts
                # from Alice's matched collection satisfy Bob's operand, which
                # turns Intersection/Count into a cross-owner false positive.
                ast_bindings = tuple(row for row in ast_bindings
                                     if not closed_by_operand.get(row.operand_id, False)
                                     or row.fact_node_id in member_scope_by_operand.get(
                                         row.operand_id, set()))
            algebra_bindings = ast_bindings
            if any(closed_by_operand.values()) or not ops_requires_scope(ir.ast):
                algebra_result = evaluate_ast(ir.ast, ast_bindings,
                                              collection_closed=closed_by_operand)
            # Otherwise leave algebra unset: an aggregate whose collection is
            # unidentified has no trustworthy member list, and handing the answer
            # stage "at least 15" built from unrelated facts is worse than
            # handing it nothing.
        no_progress = not bindings
        exhausted = any(schedule.exhaustion.values())
        certificate = evaluate_certificate(ir, closure, exhausted=exhausted, no_progress=no_progress)
        stage_times["algebra_binding"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        _groups_by_turn, all_group_turns, all_group_members = (
            self._evidence_indexes(memory_id, view.graph_version))
        group_turns = {
            group_id: all_group_turns[group_id]
            for group_id in view.terminal_groups_for_nodes(tuple(fact_ids))
            if group_id in all_group_turns
        }
        edge_relation = {
            step.edge_id: str(step.relation) for step in schedule.proof
        }
        graph_path_by_turn: dict[str, tuple[str, ...]] = {}
        for node_id in sorted(fact_ids):
            path = node_paths.get(node_id, ())
            if not path:
                continue
            for group_id in view.terminal_groups_for_nodes((node_id,)):
                for turn_id in group_turns.get(group_id, ()):
                    existing = graph_path_by_turn.get(turn_id)
                    if existing is None or (len(path), path) < (
                            len(existing), existing):
                        graph_path_by_turn[turn_id] = path
        # H10 packs only the witnesses that the recursive algebra selected.
        # Making every candidate binding mandatory produced 95--104 mandatory
        # turns for a 32-turn budget and erased the benefit of evidence ranking.
        proof_bindings = algebra_result.bindings if algebra_result is not None else closure.bindings
        group_members = {
            group_id: all_group_members[group_id]
            for group_id in group_turns if group_id in all_group_members
        }
        fact_spans: dict[str, tuple[EvidenceMember, ...]] = {}
        for binding in proof_bindings:
            node = view.nodes.get(binding.fact_node_id)
            rows = node.attributes.get("evidence_spans", ()) if node else ()
            spans = []
            for row in rows if isinstance(rows, (list, tuple)) else ():
                if not isinstance(row, Mapping):
                    continue
                try:
                    spans.append(EvidenceMember(
                        str(row["turn_id"]), int(row["start"]), int(row["end"]),
                        "fact_quote"))
                except (KeyError, TypeError, ValueError):
                    continue
            if spans:
                fact_spans[binding.fact_node_id] = tuple(spans)
        active_obligations = ir.ast_obligations or ir.proof_obligations
        units = build_proof_units(
            proof_bindings, group_turns, obligations=active_obligations,
            group_members=group_members, fact_spans=fact_spans)
        pack_turn_limit = budget.max_evidence_turns
        raw_fallback_turn_ids: tuple[str, ...] = ()
        if self.obligation_aware_packing:
            raw_fallback_turn_ids = tuple(dict.fromkeys(
                turn_id
                for node_id in schedule.visited_node_ids
                if node_id in view.nodes
                for turn_id in view.nodes[node_id].attributes.get(
                    "raw_fallback_turn_ids", ())
            ))
            units = (*units, *(EvidenceUnit(
                stable_id("raw-fallback-unit", turn_id),
                (), (), (turn_id,), (), 0, False, (), "", (), False)
                for turn_id in raw_fallback_turn_ids))
        # Hydrate every terminal reached by the structured route, not only the
        # subset already accepted as an algebra result.  The latter is a proof
        # preference, while the former is the lossless CandidatePool contract.
        hydrated_turn_ids = tuple(dict.fromkeys(turn_id for turn_ids in group_turns.values() for turn_id in turn_ids))
        # Hop distance per hydrated turn, nearest wins.  Facts admitted by the
        # reservoir rather than walked to have no hop; they are charged the hop
        # cap, since nothing established how far away they are.
        turn_hop: dict[str, int] = {}
        if self.graph_hop_decay < 1.0:
            by_hop: dict[int, list[str]] = defaultdict(list)
            for node_id in fact_ids:
                by_hop[schedule.node_hops.get(node_id, budget.max_hops)].append(node_id)
            for hop in sorted(by_hop):
                for group_id in view.terminal_groups_for_nodes(tuple(by_hop[hop])):
                    for turn_id in group_turns.get(group_id, ()):
                        turn_hop.setdefault(turn_id, hop)
        # A raw fallback is a lossless candidate rescue, not proof that the
        # QueryIR obligation is satisfied.  The former implementation treated
        # every unit as mandatory, including fallback turns with no binding;
        # on a real LME count query 30 irrelevant fallback turns occupied the
        # first 30 slots and displaced both gold turns at ranks 35/37.  Only an
        # explicitly mandatory unit may bypass the upstream relevance score.
        mandatory_turn_ids = tuple(dict.fromkeys(
            turn_id for unit in units if unit.mandatory
            for turn_id in unit.source_turn_ids))
        direct_owner_turn_ids = tuple(dict.fromkeys(
            turn_id for group_id in view.terminal_groups_for_nodes(direct_owner_terminals)
            for turn_id in group_turns.get(group_id, ())
        ))
        # Raw fallbacks are candidate rescues.  They must exist in the ranked
        # pool for a bounded reserve to have any effect, but remain optional and
        # therefore cannot bypass proof relevance without consuming the quota.
        reserved_fallback_candidates = (
            raw_fallback_turn_ids if self.raw_fallback_reserve else ())
        candidate_ids = tuple(dict.fromkeys(
             (*seeded.source_turn_ids, *hydrated_turn_ids,
             *(strict_fact_turn_scores
               if exact_lookup_priority_active else ()),
             *reserved_fallback_candidates)))
        hydrated_set = frozenset(hydrated_turn_ids)
        raw_fallback_set = frozenset(raw_fallback_turn_ids)
        mandatory = set(mandatory_turn_ids)
        if self.session_flood_k:
            flooded = frozenset(self.route_sessions(all_turns, query, self.session_flood_k))
            candidate_ids = tuple(dict.fromkeys(
                (*candidate_ids, *(turn.turn_id for turn in all_turns
                                   if turn.session_id in flooded))))
        # Route before ranking.  A mandatory turn is never dropped: those come
        # from proof units the algebra already committed to, and discarding one
        # would trade a certified answer for a routing guess.
        if self.session_router_k:
            routed = frozenset(self.route_sessions(all_turns, query, self.session_router_k))
            kept = tuple(turn_id for turn_id in candidate_ids
                         if turn_id in mandatory
                         or (by_id[turn_id].session_id in routed if turn_id in by_id else False))
            if kept:
                candidate_ids = kept
        # A wider pool only helps if it is ranked at least as well as the narrow
        # one was.  The legacy scorer's lexical, session and adjacency terms are
        # what let it pick 16 useful turns out of 278; without them a 474-turn
        # pool ranks worse than the 45-turn pool it replaced.
        query_terms = content_terms(query)
        session_best: dict[str, float] = defaultdict(float)
        base_score: dict[str, float] = {}
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn: continue
            channels = seeded.raw_scores.get(turn_id, {})
            value = (float(channels.get("exact", 0.0)) * 1.2 + float(channels.get("bm25", 0.0))
                     + float(channels.get("dense", 0.0)))
            base_score[turn_id] = value
            session_best[turn.session_id] = max(session_best[turn.session_id], value)
        adjacency: dict[str, float] = defaultdict(float)
        dialogue_closure_admitted = bool(
            self.dialogue_response_closure
            and len(mandatory) > self.dialogue_response_flood_threshold)
        for turn_id, value in base_score.items():
            turn = by_id[turn_id]
            for distance in (1, 2):
                for index in (turn.turn_index - distance, turn.turn_index + distance):
                    neighbour = by_session_index.get((turn.session_id, index))
                    if neighbour:
                        adjacency[neighbour] = max(adjacency[neighbour], value * (0.35 / distance))
            if dialogue_closure_admitted:
                # Query-relevant prompt -> following answer.  The reverse floor
                # keeps a benchmark's prompt-side evidence annotation with its
                # answer when the response itself carried the lexical match.
                following_id = by_session_index.get(
                    (turn.session_id, turn.turn_index + 1))
                following = by_id.get(following_id) if following_id else None
                if (following is not None and following.speaker != turn.speaker
                        and "?" in turn.raw_text):
                    adjacency[following.turn_id] = max(
                        adjacency[following.turn_id], value)
                previous_id = by_session_index.get(
                    (turn.session_id, turn.turn_index - 1))
                previous = by_id.get(previous_id) if previous_id else None
                if (previous is not None and previous.speaker != turn.speaker
                        and "?" in previous.raw_text):
                    adjacency[previous.turn_id] = max(
                        adjacency[previous.turn_id], value * 0.75)
        # Invert binding provenance once.  The former inner loop rebuilt the
        # union of every binding's evidence turns for every candidate, making
        # ranking O(|candidates| * |bindings| * |evidence|).
        binding_operands_by_turn: dict[str, set[str]] = defaultdict(set)
        binding_confidence_by_operand: dict[str, float] = defaultdict(float)
        for binding in closure.bindings:
            binding_confidence_by_operand[binding.operand_id] = max(
                binding_confidence_by_operand[binding.operand_id],
                binding.confidence)
            for group_id in binding.evidence_group_ids:
                for turn_id in group_turns.get(group_id, ()):
                    binding_operands_by_turn[turn_id].add(binding.operand_id)
        rows: list[CandidateScore] = []
        speaker_owner_matches = 0
        relational_consensus_matches = 0
        reservoir_by_turn = {row.turn_id: row for row in seeded.reservoir}
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn: continue
            channels = seeded.raw_scores.get(turn_id, {})
            exact = float(channels.get("exact", 0.0)); bm25 = float(channels.get("bm25", 0.0)); dense = float(channels.get("dense", 0.0))
            graph = (self.graph_hop_decay ** turn_hop.get(turn_id, budget.max_hops)
                     if self.graph_hop_decay < 1.0 else 1.0) if turn_id in hydrated_set else 0.0
            text_terms = content_terms(turn.raw_text)
            slot_gain = min(1.0, len(query_terms & text_terms) / max(1, len(query_terms)))
            role_gain = (0.3 if text_terms & TIME_TERMS else 0.0) + (0.3 if text_terms & NEGATIVE_TERMS else 0.0)
            session_score = session_best.get(turn.session_id, 0.0)
            adjacency_score = adjacency.get(turn_id, 0.0)
            operand_ids = set(binding_operands_by_turn.get(turn_id, ()))
            if turn_id in direct_owner_turn_ids:
                operand_ids.update(item.operand_id for item in ir.operands if item.owner_aliases)
            operand_ids = tuple(sorted(operand_ids))
            bscore = max((binding_confidence_by_operand[item]
                          for item in operand_ids), default=0.0)
            # The richer lexical/session/adjacency terms exist because a 545-turn
            # pool needs them to rank as well as the legacy scorer ranked 278.
            # They stay off for H2-H6 so those rungs keep their frozen V5.5 values.
            w = self.fusion
            operand_gain = w["operand"] * min(len(operand_ids), w["operand_cap"])
            fused = ((w["exact"] * exact + w["bm25"] * bm25 + w["dense"] * dense
                      + w["graph"] * graph + w["binding"] * bscore + operand_gain
                      + w["role"] * role_gain + w["slot"] * slot_gain
                      + w["session"] * session_score + w["adjacency"] * adjacency_score
                      + (self.exact_lookup_priority_bonus
                         * exact_lookup_scores.get(turn_id, 0.0)
                         if (exact_lookup_priority_active
                             and turn_id in strict_fact_turn_scores) else
                         exact_lookup_scores.get(turn_id, 0.0)
                         if exact_lookup_fast_active else 0.0))
                     if wide_reservoir else
                     (exact + bm25 + dense + w["graph"] * graph + w["binding"] * bscore
                      + operand_gain))
            speaker_terms = content_terms(turn.speaker)
            speaker_owner_match = bool(
                self.speaker_owner_bonus
                and speaker_terms
                and not speaker_terms <= {"user", "assistant", "system"}
                and speaker_terms <= query_terms)
            if speaker_owner_match:
                fused += self.speaker_owner_bonus
                speaker_owner_matches += 1
            reservoir_entry = reservoir_by_turn.get(turn_id)
            relational_consensus = (
                reservoir_entry.relational_consensus
                if reservoir_entry is not None else 0.0)
            if (self.relational_consensus_bonus
                    and relational_view_active and relational_consensus):
                fused += self.relational_consensus_bonus * relational_consensus
                relational_consensus_matches += 1
            rows.append(CandidateScore(turn_id, turn.session_id, exact, bm25, dense, graph,
                                       role_gain, slot_gain,
                                       self._turn_tokens(turn), fused,
                                       tuple(name for name, value in (("exact", exact), ("bm25", bm25), ("dense", dense), ("graph", graph)) if value),
                                       session_score, adjacency_score,
                                       graph_path_ids=graph_path_by_turn.get(turn_id, ()),
                                       relation_contributions=tuple(
                                           edge_relation.get(edge_id, "")
                                           for edge_id in graph_path_by_turn.get(turn_id, ())
                                           if edge_relation.get(edge_id, "")),
                                       operand_ids=operand_ids, binding_score=bscore,
                                       obligation_gain=len(operand_ids),
                                       relational_consensus_score=relational_consensus,
                                       mandatory=turn_id in mandatory,
                                       proof_unit_ids=tuple(unit.unit_id for unit in units if turn_id in unit.source_turn_ids)))
        soften_proof_priority = bool(
            self.proof_priority_bonus is not None
            and len(mandatory) > self.proof_priority_flood_threshold)
        def sort_candidates(values: list[CandidateScore]) -> None:
            if not soften_proof_priority:
                values.sort(key=lambda row: (
                    -row.mandatory, -row.fused_score, row.turn_id))
            else:
                values.sort(key=lambda row: (
                    -(row.fused_score
                      + self.proof_priority_bonus * bool(row.mandatory)),
                    row.turn_id))

        sort_candidates(rows)
        query_witness_matches = 0
        query_witness_seed_terms = 0
        if self.query_witness_bonus:
            explicit_speakers = frozenset(
                turn.speaker for turn in all_turns
                if (speaker_terms := content_terms(turn.speaker))
                and not speaker_terms <= {"user", "assistant", "system"}
                and speaker_terms <= query_terms)
            # Owner scope is the precision boundary that distinguishes this
            # closure from the rejected broad lexical coarse edge.
            if explicit_speakers:
                terms_by_turn, document_frequency = (
                    self._query_witness_features(
                        memory_id, view.graph_version, all_turns))
                seed_ids = tuple(
                    row.turn_id for row in rows[:self.query_witness_seed_count]
                    if by_id[row.turn_id].speaker in explicit_speakers)
                rare_seed_terms = frozenset(
                    term for turn_id in seed_ids
                    for term in terms_by_turn.get(turn_id, ())
                    if document_frequency[term] <= self.query_witness_rare_df)
                query_witness_seed_terms = len(rare_seed_terms)
                reranked: list[CandidateScore] = []
                for row in rows:
                    turn = by_id[row.turn_id]
                    shared = (rare_seed_terms
                              & terms_by_turn.get(row.turn_id, ()))
                    if (turn.speaker not in explicit_speakers
                            or len(shared) < self.query_witness_min_shared_terms):
                        reranked.append(row)
                        continue
                    strength = sum(math.log(
                        (len(all_turns) + 1)
                        / (document_frequency[term] + 1)) for term in shared)
                    gain = self.query_witness_bonus * min(1.0, strength / 4.0)
                    reranked.append(replace(
                        row, fused_score=row.fused_score + gain,
                        source_channels=tuple(dict.fromkeys((
                            *row.source_channels, "query_witness")))))
                    query_witness_matches += 1
                rows = reranked
                sort_candidates(rows)
        candidate_count_before_limit = len(rows)
        if self.candidate_pool_limit:
            rows = rows[:self.candidate_pool_limit]
        # H10 is deliberately not in this set.  `rows` above is already sorted
        # mandatory-first and then by fused_score, so `_rank_pack` packs the
        # mandatory turns in relevance order.  `pack_proof_units` instead keeps
        # the order the proof units declare, which is binding order and carries
        # no relevance signal -- harmless when the mandatory set fits the budget,
        # but a LoCoMo question produces 95-104 mandatory turns against a 32-turn
        # budget, so it becomes the entire selection.  Adding H10 here as a
        # "missing profile" fix cost 11pp of LoCoMo accuracy (0.823 -> 0.714);
        # `rank_mandatory=True` recovers 0.808 by restoring exactly the ordering
        # `_rank_pack` already had.
        effective_turn_limit = (
            min(budget.max_evidence_turns, self.exact_lookup_turn_limit)
            if exact_lookup_fast_active else budget.max_evidence_turns)
        effective_budget = replace(
            budget, max_evidence_turns=effective_turn_limit)
        if self.obligation_aware_packing:
            answer_kind = (algebra_result.answer_kind if algebra_result is not None
                           else str(ir.ast_operator or ir.operator))
            pack_turn_limit = (
                effective_turn_limit if exact_lookup_fast_active else (
                    adaptive_evidence_turn_limit(
                        answer_kind, len(ir.operands), budget.max_evidence_turns,
                        query=query)
                    if self.precision_aware_packing else budget.max_evidence_turns))
            pack_budget = replace(budget, max_evidence_turns=pack_turn_limit)
            baseline_floor, _baseline_dropped, _baseline_coverage = self._rank_pack(
                rows, by_id, pack_budget, self.per_session_quota,
                reserved_turn_ids=raw_fallback_set,
                reserve_limit=self.raw_fallback_reserve)
            packed, dropped, pack_exhaustion, packed_units, packed_span_tokens = pack_obligation_aware(
                units, rows, by_id, query=query, answer_kind=answer_kind,
                max_turns=pack_turn_limit,
                max_tokens=budget.max_evidence_tokens,
                count_text_tokens=self._count_tokens_cached,
                span_window=self.span_pack_window,
                baseline_floor=baseline_floor,
                precision_aware=self.precision_aware_packing)
            units = packed_units
        elif profile in {HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                       HarnessProfile.H9_FACT_RESERVOIR}:
            packed, dropped, pack_exhaustion = pack_proof_units(units, rows, by_id,
                                                                 max_turns=effective_turn_limit,
                                                                 max_tokens=budget.max_evidence_tokens,
                                                                 token_cost=self._turn_tokens,
                                                                 rank_mandatory=self.rank_mandatory)
        else:
            if self.harness_set_cover:
                kind, required_slots, _negative = question_slots(query)
                packed, dropped, _cover = self._set_cover(
                    rows, by_id, kind, required_slots, effective_budget)
            else:
                packed, dropped, _ = self._rank_pack(
                    rows, by_id, effective_budget, self.per_session_quota,
                    reserved_turn_ids=raw_fallback_set,
                    reserve_limit=self.raw_fallback_reserve)
            pack_exhaustion = {"turn_cap_reached": len(packed) >= effective_turn_limit,
                               "token_cap_reached": sum(self._turn_tokens(by_id[item]) for item in packed) >= budget.max_evidence_tokens}
        stage_times["evidence_pack"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        if algebra_result is not None:
            certificate = finalize_ast_certificate(
                ir, algebra_result, algebra_bindings, group_turns, packed,
                units=units, exhausted=exhausted or any(pack_exhaustion.values()))
        stage_times["certificate_finalize"] = (time.perf_counter() - tick) * 1000
        stage_times["total"] = (time.perf_counter() - started) * 1000
        graph_id = stable_id(
            "graph-artifact", memory_id, view.graph_version, view.graph_checksum)
        evidence_tokens = (packed_span_tokens
                           if self.obligation_aware_packing else
                           sum(self._turn_tokens(by_id[item]) for item in packed))
        operand_coverage = {
            operand.operand_id: {
                "seed_nodes": seeded.operand_nodes.get(operand.operand_id, ()),
                "seed_turns": seeded.operand_turns.get(operand.operand_id, ()),
                "bindings": tuple(row.binding_id for row in bindings if row.operand_id == operand.operand_id),
                "packed_turns": tuple(item for item in packed if item in mandatory),
            } for operand in ir.operands
        }
        return NavigationResult(
            question_id=stable_id("question", memory_id, query), memory_id=memory_id, graph_artifact_id=graph_id,
            retrieved_session_ids=tuple(dict.fromkeys(by_id[item].session_id for item in packed)),
            retrieved_turn_ids=packed, proof=schedule.proof, visited_nodes=len(schedule.visited_node_ids),
            visited_edges=len(schedule.proof), frontier_peak=min(budget.max_frontier, len(seeded.semantic_node_ids)),
            evidence_tokens=evidence_tokens, budget_exhausted=any(schedule.exhaustion.values()) or any(pack_exhaustion.values()),
            trace={"variant": str(profile), "query_operator": str(ir.operator), "candidate_count": len(rows),
                   "semantic_navigation_excludes_provenance_edges": True, "relation_counts": schedule.relation_counts,
                   # The legacy H0/H1 path keeps the historical word-count estimate so it stays
                   # a faithful frozen baseline; the harness budgets against real tokens.
                   "token_counter": self.token_counter.describe(),
                   "seeding": dict(seeded.stats),
                   # The AST is compiled but not executed yet.  Recording it here,
                   # together with where it disagrees with the legacy classifier,
                   # measures the compiler before anything depends on it.
                   "operator_ast": compiled_ir.describe_ast(),
                   "ast_operator": (str(compiled_ir.ast_operator)
                                    if compiled_ir.ast_operator else ""),
                   "ast_diverges": compiled_ir.ast_diverges,
                   "query_ir_confidence": compiled_ir.compile_confidence,
                   "query_ir_fallback_reasons": list(compiled_ir.fallback_reasons),
                   "query_ir_soft_fallback": ir.soft_fallback_applied,
                   "query_ir_mode": ("unified_ast" if profile is HarnessProfile.H11_UNIFIED_IR
                                     else "legacy_plus_shadow_ast"),
                   "ast_operands": [item.operand_id for item in compiled_ir.ast_operands],
                   "ast_obligations": [f"{item.kind}:{item.operand_id or 'root'}"
                                       for item in compiled_ir.ast_obligations],
                   "parse_warnings": list(ir.parse_warnings),
                   "principals": registry.stats(),
                   "owner_resolution": resolution_stats(ir.resolved_owners,
                                                        ir.owner_resolution_warnings),
                   "fact_reservoir": dict(fact_reservoir.stats) if fact_reservoir else {},
                   "binding_reasons": binding_reasons,
                   "runtime_cache": self.runtime.cache_stats(),
                   "sqlite_read_pool_size": self.read_pool_size,
                   "obligation_aware_packing": self.obligation_aware_packing,
                   "obligation_aware_relations": self.obligation_aware_relations,
                   "rare_lexical_relations": self.rare_lexical_relations,
                   "query_gated_rare_lexical": self.query_gated_rare_lexical,
                   "native_seed_fusion": self.native_seed_fusion,
                   "relational_consensus_bonus": (
                       self.relational_consensus_bonus),
                   "relational_consensus_matches": (
                       relational_consensus_matches),
                   "dialogue_response_closure": self.dialogue_response_closure,
                   "dialogue_response_closure_admitted": dialogue_closure_admitted,
                   "dialogue_response_flood_threshold": (
                       self.dialogue_response_flood_threshold),
                   "proof_priority_bonus": self.proof_priority_bonus,
                   "proof_priority_softened": soften_proof_priority,
                   "proof_priority_flood_threshold": (
                       self.proof_priority_flood_threshold),
                   "speaker_owner_bonus": self.speaker_owner_bonus,
                   "speaker_owner_matches": speaker_owner_matches,
                   "query_witness_bonus": self.query_witness_bonus,
                   "query_witness_seed_count": self.query_witness_seed_count,
                   "query_witness_rare_df": self.query_witness_rare_df,
                   "query_witness_min_shared_terms": (
                       self.query_witness_min_shared_terms),
                   "query_witness_matches": query_witness_matches,
                   "query_witness_seed_terms": query_witness_seed_terms,
                   "span_pack_window": self.span_pack_window,
                   "precision_aware_packing": self.precision_aware_packing,
                   "exact_lookup_fast_path": self.exact_lookup_fast_path,
                   "exact_lookup_priority": self.exact_lookup_priority,
                   "exact_lookup_priority_named_speakers_only": (
                       self.exact_lookup_priority_named_speakers_only),
                   "exact_lookup": exact_lookup_trace,
                   "execution_mode": ("exact_lookup" if exact_lookup_fast_active
                                      else "hierarchical_graph"),
                   "raw_fallback_reserve": self.raw_fallback_reserve,
                   "raw_fallback_candidates": sum(
                       row.turn_id in raw_fallback_set for row in rows),
                   "raw_fallback_packed": sum(
                       turn_id in raw_fallback_set for turn_id in packed),
                   "adaptive_pack_turn_limit": pack_turn_limit,
                   "candidate_count_before_limit": candidate_count_before_limit,
                   "candidate_pool_limit": self.candidate_pool_limit},
            seed_node_ids=semantic_seeds, visited_path_node_ids=tuple(dict.fromkeys(
                (*schedule.visited_node_ids, *direct_owner_terminals))),
            slot_coverage={item.operand_id: tuple(row.binding_id for row in bindings if row.operand_id == item.operand_id)
                           for item in ir.operands}, certificate=certificate, candidate_scores=tuple(rows),
            packed_turn_ids=packed, dropped_turn_ids=dropped, stage_latency_ms=stage_times,
            # Diagnostic set, not a ranking.  Canonical order keeps a view
            # compiled in another process byte-for-byte comparable with one
            # compiled locally under a different hash seed.
            graph_only_candidate_turn_ids=tuple(sorted(hydrated_turn_ids)),
            relation_trace=schedule.proof,
            first_hit_relations={item.operand_id: (paths.get(item.fact_node_id, ("posting",))[0]) for item in bindings},
            search_exhaustion=schedule.exhaustion, pack_exhaustion=pack_exhaustion,
            operand_coverage=operand_coverage, proof_units=units, stop_reason=certificate.stop_reason,
            reservoir_fact_node_ids=tuple(row.fact_id for row in fact_reservoir.entries)
                                     if fact_reservoir else (),
            reached_fact_node_ids=tuple(sorted(
                node_id for node_id in fact_ids
                if node_id in view.nodes and view.nodes[node_id].node_type == NodeType.CANONICAL_FACT)),
            bound_fact_node_ids=tuple(dict.fromkeys(row.fact_node_id for row in bindings)),
            selected_fact_node_ids=tuple(dict.fromkeys(
                row.fact_node_id for row in (
                    algebra_result.bindings if algebra_result is not None else closure.bindings))),
            algebra=algebra_result,
        )

    @staticmethod
    def _route_terminal_closure(view, seed_ids: Sequence[str], *, limit: int = 48) -> tuple[str, ...]:
        """Bounded route-card refinement before terminal provenance hydration."""
        queue = [(item, 0) for item in seed_ids]; seen: set[str] = set(); result: list[str] = []
        while queue and len(seen) < limit:
            node_id, depth = queue.pop(0)
            if node_id in seen or node_id not in view.nodes:
                continue
            seen.add(node_id); node = view.nodes[node_id]
            if node.attributes.get("provenance_scope", "terminal") == "terminal":
                result.append(node_id); continue
            if depth >= 3:
                continue
            for row in view.neighbors(node_id, (RelationType.REFINES_TO, RelationType.SCENE_CONTAINS),
                                      semantic_only=True):
                queue.append((row.next_node_id, depth + 1))
        return tuple(dict.fromkeys(result))

    @staticmethod
    def _owner_terminal_postings(view, seed_ids: Sequence[str], raw_scores, query: str,
                                 *, per_owner_limit: int = 256) -> tuple[str, ...]:
        """Lossless rescue for the matched speaker/owner, without session flooding."""
        query_terms = content_terms(query); result: list[str] = []
        for node_id in seed_ids:
            node = view.nodes.get(node_id)
            if not node or node.node_type != NodeType.CANONICAL_ENTITY:
                continue
            candidates = []
            for row in view.neighbors(node_id, (RelationType.HAS_FACT,), semantic_only=True):
                target = view.nodes.get(row.next_node_id)
                # The build never emits Entity --HAS_FACT--> EvidenceGroupRef;
                # HAS_FACT from an entity reaches CanonicalFact, CollectionScope
                # and StateHead, while evidence refs hang off a Scene by
                # SCENE_CONTAINS.  Requiring EVIDENCE_GROUP_REF here meant this
                # rescue returned nothing on every question ever run.  Facts are
                # terminal and carry their own evidence groups, so they serve the
                # same purpose.
                if not target or target.node_type != NodeType.CANONICAL_FACT:
                    continue
                score = len(query_terms & content_terms(target.summary)) / max(1, len(query_terms))
                for group_id in target.all_evidence_group_ids:
                    score += 0.01 * len(raw_scores.get(group_id, {}))
                candidates.append((score, target.node_id))
            result.extend(node_id for _, node_id in sorted(candidates, key=lambda item: (-item[0], item[1]))[:per_owner_limit])
        return tuple(dict.fromkeys(result))

    def _raw_candidates(
        self, memory_id: str, query: str, query_terms: frozenset[str], turns_: Sequence[SourceTurn]
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {"exact": 0.0, "bm25": 0.0, "dense": 0.0}
        )
        for turn in turns_:
            turn_terms = content_terms(turn.raw_text)
            overlap = len(query_terms & turn_terms)
            if overlap:
                scores[turn.turn_id]["exact"] = overlap / max(1, len(query_terms))
                if " ".join(query_terms) in turn.raw_text.casefold():
                    scores[turn.turn_id]["exact"] += 0.5
        for turn_id, score in self.store.search_turns(memory_id, query, limit=96):
            scores[turn_id]["bm25"] = max(0.0, score)
        if self.dense_search:
            for turn_id, score in self.dense_search(memory_id, query, 96):
                scores[turn_id]["dense"] = float(score)
        bm25_max = max((row["bm25"] for row in scores.values()), default=0.0)
        dense_values = [row["dense"] for row in scores.values() if row["dense"]]
        dense_min = min(dense_values, default=0.0)
        dense_max = max(dense_values, default=0.0)
        for row in scores.values():
            if bm25_max > 0:
                row["bm25"] /= bm25_max
            if row["dense"] and dense_max > dense_min:
                row["dense"] = (row["dense"] - dense_min) / (dense_max - dense_min)
        return scores

    @staticmethod
    def _top_sessions(scores: Mapping[str, Mapping[str, float]], by_id: Mapping[str, SourceTurn], limit: int) -> tuple[str, ...]:
        result: dict[str, float] = defaultdict(float)
        for turn_id, channels in scores.items():
            if turn_id not in by_id:
                continue
            fused = channels["exact"] * 1.4 + channels["bm25"] + channels["dense"]
            result[by_id[turn_id].session_id] = max(result[by_id[turn_id].session_id], fused)
        return tuple(key for key, _ in sorted(result.items(), key=lambda item: (-item[1], item[0]))[:limit])

    @staticmethod
    def _node_relevance(nodes: Mapping[str, object], query_terms: frozenset[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for node_id, node in nodes.items():
            node_terms = content_terms(getattr(node, "summary", ""))
            overlap = len(query_terms & node_terms)
            if overlap:
                result[node_id] = overlap / max(1, len(query_terms)) + float(getattr(node, "confidence", 1.0)) * 0.05
        return result

    def _priority_expand(self, view, seeds, relevance, query, budget, *, certificate_guided=False):
        kind, required, _ = question_slots(query)
        uncovered = set(required)
        queue: list[tuple[float, int, str, int, tuple[ProofStep, ...]]] = []
        sequence = itertools.count()
        for node_id in seeds:
            heapq.heappush(queue, (-relevance.get(node_id, 0.0), 0, node_id, next(sequence), ()))
        visited: list[str] = []
        visited_set: set[str] = set()
        used_edges: set[str] = set()
        proof: list[ProofStep] = []
        peak = len(queue)
        while queue and len(visited) < budget.max_visited_nodes and len(used_edges) < budget.max_visited_edges:
            _, hop, node_id, _, path = heapq.heappop(queue)
            if node_id in visited_set:
                continue
            visited_set.add(node_id)
            visited.append(node_id)
            if certificate_guided:
                uncovered -= set(view.role_bitset.get(node_id, ()))
            proof.extend(step for step in path if step.edge_id not in used_edges)
            used_edges.update(step.edge_id for step in path)
            if hop >= budget.max_hops:
                continue
            for row in view.neighbors(node_id, semantic_only=True):
                if row.next_node_id in visited_set:
                    continue
                roles = view.role_bitset.get(row.next_node_id, ())
                target_roles = uncovered if certificate_guided else set(required)
                role_gain = len(target_roles & set(roles)) / max(1, len(target_roles))
                relation_gain = 0.25 if row.edge.relation in {
                    RelationType.PORTAL, RelationType.TEMPORAL_BEFORE,
                    RelationType.TEMPORAL_AFTER, RelationType.STATE_TRANSITION,
                    RelationType.MEMBER_OF,
                } else 0.0
                priority = relevance.get(row.next_node_id, 0.0) + role_gain + relation_gain - 0.08 * (hop + 1)
                step = ProofStep(row.edge.edge_id, row.edge.src_id, row.edge.relation,
                                 row.edge.dst_id, row.edge.evidence_group_id)
                heapq.heappush(queue, (-priority, hop + 1, row.next_node_id,
                                       next(sequence), (*path, step)))
            if len(queue) > budget.max_frontier:
                queue = heapq.nsmallest(budget.max_frontier, queue)
                heapq.heapify(queue)
            peak = max(peak, len(queue))
        exhausted = bool(queue) or len(visited) >= budget.max_visited_nodes or len(used_edges) >= budget.max_visited_edges
        unique_proof = {step.edge_id: step for step in proof}
        return visited, list(unique_proof.values())[: budget.max_visited_edges], peak, exhausted

    def _candidate_rows(self, candidate_ids, by_id, raw_scores, query_terms, visited, view, proof=()):
        graph_turns: set[str] = set()
        path_by_turn: dict[str, list[str]] = defaultdict(list)
        relation_by_turn: dict[str, list[str]] = defaultdict(list)
        proof_by_node = defaultdict(list)
        for step in proof:
            proof_by_node[step.dst_id].append(step)
        for group_id in view.evidence_group_ids_for_nodes(visited):
            group = self.store.evidence_group(group_id)
            if group:
                graph_turns.update(member.turn_id for member in group.members)
        for node_id in visited:
            for group_id in view.evidence_group_ids_for_nodes((node_id,)):
                group = self.store.evidence_group(group_id)
                if not group:
                    continue
                for member in group.members:
                    for step in proof_by_node.get(node_id, ()):
                        path_by_turn[member.turn_id].append(step.edge_id)
                        relation_by_turn[member.turn_id].append(str(step.relation))
        base_by_turn = {
            turn_id: channels.get("exact", 0.0) * 1.2 + channels.get("bm25", 0.0)
            + channels.get("dense", 0.0)
            for turn_id, channels in raw_scores.items() if turn_id in by_id
        }
        session_max: dict[str, float] = defaultdict(float)
        by_session_index: dict[tuple[str, int], str] = {
            (turn.session_id, turn.turn_index): turn_id for turn_id, turn in by_id.items()
        }
        for turn_id, score in base_by_turn.items():
            turn = by_id[turn_id]
            session_max[turn.session_id] = max(session_max[turn.session_id], score)
        adjacency: dict[str, float] = defaultdict(float)
        for turn_id, score in base_by_turn.items():
            turn = by_id[turn_id]
            for distance in (1, 2):
                for index in (turn.turn_index - distance, turn.turn_index + distance):
                    neighbor = by_session_index.get((turn.session_id, index))
                    if neighbor:
                        adjacency[neighbor] = max(adjacency[neighbor], score * (0.35 / distance))
        rows: list[CandidateScore] = []
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn:
                continue
            channels = raw_scores.get(turn_id, {})
            exact = float(channels.get("exact", 0.0))
            bm25 = float(channels.get("bm25", 0.0))
            dense = float(channels.get("dense", 0.0))
            graph = 1.0 if turn_id in graph_turns else 0.0
            text_terms = content_terms(turn.raw_text)
            temporal_gain = 0.3 if text_terms & TIME_TERMS else 0.0
            negative_gain = 0.3 if text_terms & NEGATIVE_TERMS else 0.0
            role_gain = temporal_gain + negative_gain
            slot_gain = min(1.0, len(query_terms & text_terms) / max(1, len(query_terms)))
            session_score = session_max.get(turn.session_id, 0.0)
            adjacency_score = adjacency.get(turn_id, 0.0)
            fused = (1.2 * exact + bm25 + dense + 0.55 * graph + 0.25 * role_gain
                     + 0.5 * slot_gain + 0.12 * session_score + adjacency_score)
            channels_used = tuple(name for name, value in (
                ("exact", exact), ("bm25", bm25), ("dense", dense), ("graph", graph)
            ) if value > 0)
            rows.append(CandidateScore(
                turn_id, turn.session_id, exact, bm25, dense, graph,
                role_gain, slot_gain, estimate_tokens(turn.raw_text), fused, channels_used,
                session_score, adjacency_score,
                tuple(dict.fromkeys(path_by_turn.get(turn_id, ()))),
                tuple(dict.fromkeys(relation_by_turn.get(turn_id, ()))),
            ))
        return sorted(rows, key=lambda row: (-row.fused_score, row.turn_id))

    @staticmethod
    def _slot_matches(kind: str, required: Sequence[str], turn: SourceTurn) -> set[str]:
        lowered = turn.raw_text.casefold()
        tokens_ = content_terms(lowered)
        matched: set[str] = set()
        if kind in {"count", "list"}:
            matched.add("collection_scope")
            if any(char.isdigit() for char in lowered) or "," in lowered or " and " in lowered:
                matched.update(slot for slot in required if slot.startswith("member_"))
            else:
                matched.add("member_1")
        elif kind.startswith("temporal") or kind == "duration":
            if tokens_ & TIME_TERMS or re.search(r"\b\d{1,4}([:/-]\d{1,2})?\b", lowered):
                matched.update(required[:2])
            if any(word in lowered for word in ("before", "after", "since", "until", "later", "earlier")):
                matched.add(required[-1])
        elif kind == "state_change":
            if any(word in lowered for word in ("was", "used to", "before", "previous")):
                matched.add("prior_state")
            if any(word in lowered for word in ("now", "currently", "became", "is")):
                matched.add("current_state")
        else:
            matched.update(required)
        return matched

    def _set_cover(self, rows, by_id, kind, required, budget):
        uncovered = set(required)
        remaining = list(rows)
        packed: list[str] = []
        coverage: dict[str, list[str]] = defaultdict(list)
        tokens_used = 0
        slot_matches = {
            row.turn_id: self._slot_matches(kind, required, by_id[row.turn_id]) for row in rows
        }
        while remaining and len(packed) < budget.max_evidence_turns:
            ranked = []
            for row in remaining:
                matched = slot_matches[row.turn_id]
                gain = len(uncovered & matched)
                diversity = (0.9 if kind in {"count", "list"} else 0.2) if by_id[row.turn_id].session_id not in {
                    by_id[item].session_id for item in packed
                } else 0.0
                utility = row.fused_score + gain * 1.25 + diversity - 0.0005 * row.token_cost
                ranked.append((utility, row, matched))
            _, selected, matched = max(ranked, key=lambda item: (item[0], item[1].fused_score, item[1].turn_id))
            remaining = [row for row in remaining if row.turn_id != selected.turn_id]
            if tokens_used + selected.token_cost > budget.max_evidence_tokens:
                continue
            packed.append(selected.turn_id)
            tokens_used += selected.token_cost
            for slot in matched:
                coverage[slot].append(selected.turn_id)
            uncovered -= matched
            # A heuristic certificate is a routing hint, not permission to throw
            # away high-scoring source evidence. Continue filling the fixed turn/
            # token budget so false-positive closure cannot reduce recall.
        dropped = tuple(row.turn_id for row in rows if row.turn_id not in packed)
        return tuple(packed), dropped, coverage

    @staticmethod
    def _rank_pack(
        rows, by_id, budget, per_session_quota: bool = False, *,
        reserved_turn_ids=(), reserve_limit: int = 0,
    ):
        packed: list[str] = []
        tokens_used = 0
        reserved = frozenset(reserved_turn_ids)
        # Spend only the explicit fallback allowance.  Rows are already sorted
        # by mandatory status and fused score, so this chooses the strongest
        # fallback evidence without recreating the old all-fallback prefix.
        for row in rows:
            if len(packed) >= min(max(0, reserve_limit), budget.max_evidence_turns):
                break
            if row.turn_id not in reserved:
                continue
            if tokens_used + row.token_cost > budget.max_evidence_tokens:
                continue
            packed.append(row.turn_id)
            tokens_used += row.token_cost
        if per_session_quota:
            # Two passes.  The first gives every session present in `rows` a
            # floor, so a session whose turns all rank just below another
            # session's still contributes; the second spends what is left by
            # rank.  Without the floor one session can take all 32 slots, which
            # is what makes a multi-session answer unreachable even when both
            # of its sessions are in the pool.
            sessions = list(dict.fromkeys(
                by_id[row.turn_id].session_id for row in rows if row.turn_id in by_id))
            quota = max(1, budget.max_evidence_turns // max(1, len(sessions)))
            taken: Counter = Counter()
            for row in rows:
                if len(packed) >= budget.max_evidence_turns:
                    break
                session_id = by_id[row.turn_id].session_id if row.turn_id in by_id else ""
                if not row.mandatory and taken[session_id] >= quota:
                    continue
                if tokens_used + row.token_cost > budget.max_evidence_tokens:
                    continue
                packed.append(row.turn_id)
                taken[session_id] += 1
                tokens_used += row.token_cost
        for row in rows:
            if len(packed) >= budget.max_evidence_turns:
                break
            if row.turn_id in packed:
                continue
            if tokens_used + row.token_cost > budget.max_evidence_tokens:
                continue
            packed.append(row.turn_id)
            tokens_used += row.token_cost
        dropped = tuple(row.turn_id for row in rows if row.turn_id not in packed)
        return tuple(packed), dropped, {}
