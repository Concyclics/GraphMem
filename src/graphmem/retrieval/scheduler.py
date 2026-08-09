from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import ProofStep, QueryBudget, RelationType
from ..text import content_terms
from .query_ir import QueryIR


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    visited_node_ids: tuple[str, ...]
    proof: tuple[ProofStep, ...]
    relation_counts: dict[str, int]
    exhaustion: dict[str, bool]
    #: Hop distance from the nearest seed, per visited node.  The scorer needs it
    #: to discount by distance: without it a node three hops out is
    #: indistinguishable from a seed, which is how 26% of the evidence budget
    #: ended up going to turns with no lexical match and a 0.4% gold yield.
    node_hops: dict[str, int] = field(default_factory=dict)


#: Measured share of edges of each type whose destination is a gold fact,
#: against the base rate of gold facts in the memory (2 LoCoMo conversations,
#: 107 questions):
#:   collection_co_member 2.27x / 1.08x   state_next 2.27x / 1.28x
#:   has_fact             1.00x / 0.77x   scene_contains 0.41x / 0.41x
#:   member_of, refines_to, participates_in, at_time   0.00x
#: A relation at or below 1.0x carries no information about which fact answers
#: the question -- it encodes structural membership, not content -- so walking
#: it spends budget that content channels would use better.
DEFAULT_PREFERRED = (
    #: First because it is the only relation that leaves the seeding session at
    #: all: every other edge here has single-session evidence, so traversal over
    #: them cannot reach a second session no matter how much budget it is given.
    RelationType.SHARED_REFERENT,
    RelationType.COARSE_RELATED,
    RelationType.SAME_ENTITY_STATE, RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL, RelationType.CONTRADICTION_UPDATE,
    RelationType.HAS_FACT, RelationType.MEMBER_OF, RelationType.PARTICIPATES_IN,
    RelationType.COLLECTION_CO_MEMBER, RelationType.AT_TIME, RelationType.STATE_NEXT,
    RelationType.TEMPORAL_BEFORE, RelationType.REFINES_TO,
)
DEFAULT_FALLBACK = (RelationType.SCENE_CONTAINS,)
#: Only relations that link facts to each other by content.  Disabling traversal
#: entirely reproduced the baseline exactly (turn_recall 0.326, turn_all_hit
#: 0.509), and keeping only these cut edges walked by 86% with no change either,
#: so containment relations are pure overhead here.  SHARED_VALUE and FACT_VALUE
#: are added by the P9 value-lattice projection and are the cross-session
#: content path the frozen graph never had.
INFORMATIVE_PREFERRED = (
    RelationType.COARSE_RELATED,
    RelationType.SAME_ENTITY_STATE, RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL, RelationType.CONTRADICTION_UPDATE,
    RelationType.COLLECTION_CO_MEMBER, RelationType.STATE_NEXT, RelationType.TEMPORAL_BEFORE,
    RelationType.SHARED_VALUE, RelationType.FACT_VALUE,
)


_TYPED_RELATIONS = (
    RelationType.COREFERENCE, RelationType.SAME_ENTITY_STATE,
    RelationType.TEMPORAL_CONTINUATION, RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
)


def _obligation_relation_bonus(ir: QueryIR, relation: RelationType, *,
                               inverse: bool) -> float:
    """Compile proof needs into a physical relation preference.

    This affects only the bounded beam order; it never declares an obligation
    complete and never bypasses evidence/provenance checks.
    """
    kinds = {row.kind for row in ir.proof_obligations}
    score = 0.0
    if kinds & {"time_endpoint", "ordering"}:
        score += {
            RelationType.TEMPORAL_CONTINUATION: 3.0,
            RelationType.TEMPORAL_BEFORE: 2.5,
            RelationType.SAME_ENTITY_STATE: 1.25,
            RelationType.CONTRADICTION_UPDATE: 1.0,
        }.get(relation, 0.0)
    if "state_history" in kinds:
        score += {
            RelationType.CONTRADICTION_UPDATE: 3.0,
            RelationType.SAME_ENTITY_STATE: 2.5,
            RelationType.STATE_NEXT: 2.0,
            RelationType.TEMPORAL_CONTINUATION: 1.5,
        }.get(relation, 0.0)
    if "collection" in kinds:
        score += {
            RelationType.COLLECTION_CO_MEMBER: 2.5,
            RelationType.COREFERENCE: 1.25,
            RelationType.COARSE_RELATED: 0.5,
        }.get(relation, 0.0)
    if len(ir.operands) > 1:
        score += {
            RelationType.COREFERENCE: 2.0,
            RelationType.COARSE_RELATED: 1.0,
            RelationType.CAUSAL: 0.75,
        }.get(relation, 0.0)
    query_terms = content_terms(ir.query)
    if query_terms & {"why", "because", "cause", "caused", "lead", "led"}:
        score += 3.0 if relation == RelationType.CAUSAL else 0.0
    if any(row.polarity == "negative" for row in ir.operands):
        score += 2.5 if relation == RelationType.CONTRADICTION_UPDATE else 0.0
    # Latest-state traversal follows the writer's old->new orientation; an
    # inverse edge is useful for history but should not beat the current value.
    if "state_history" in kinds and relation in {
            RelationType.STATE_NEXT, RelationType.TEMPORAL_CONTINUATION,
            RelationType.CONTRADICTION_UPDATE}:
        score += -0.25 if inverse else 0.5
    return score


def _coarse_signal_bonus(ir: QueryIR, source: str) -> float:
    """Use V5.14 relation-mask metadata without treating it as a fact claim."""

    prefix = "relation_mask:"
    if not source.startswith(prefix):
        return 0.0
    signals = set(source[len(prefix):].split(","))
    kinds = {row.kind for row in ir.proof_obligations}
    score = 0.0
    if signals & {"temporal_near"} and kinds & {"time_endpoint", "ordering"}:
        score += 1.25
    if "state_compatible" in signals and kinds & {
            "state_history", "time_endpoint", "ordering"}:
        score += 1.25
    if "collection_related" in signals and "collection" in kinds:
        score += 1.5
    if "shared_entity" in signals and len(ir.operands) > 1:
        score += 1.0
    if "lexical_rare" in signals and (
            len(ir.operands) > 1 or kinds & {
                "state_history", "time_endpoint", "ordering", "collection"}):
        # Rare overlap is strong for long-region -> long-region linkage but was
        # measured to be a weak short-query router.  Give it only a modest
        # cross-region bonus when the compiled plan actually needs multiple
        # facts/endpoints; ordinary lookup still relies on lexical seed scoring.
        score += 0.75
    # A semantic-only mask remains a normal coarse edge.  Its relation-specific
    # siblings should win ties, but it receives no extra proof authority.
    return score


def _relation_mask_signals(source: str) -> frozenset[str]:
    prefix = "relation_mask:"
    return (frozenset(source[len(prefix):].split(","))
            if source.startswith(prefix) else frozenset())


def execute(view, ir: QueryIR, seed_ids: tuple[str, ...], budget: QueryBudget, *,
            structured: bool,
            preferred_relations=None, fallback_relations=None,
            expansion_beam: int = 0,
            obligation_aware_relations: bool = False) -> ScheduleResult:
    """Deterministic obligation-first traversal; provenance is hydrated later.

    ``expansion_beam`` keeps only the best N neighbours of each expanded node.
    0 restores the unpruned behaviour, where every neighbour of every visited
    node was queued and the only limit was ``queue[:max_frontier]`` -- a global
    truncation of the *oldest* entries, which does not bound how much a single
    hub node contributes.  Measured effect of no beam: ~74 turns per question
    reach the candidate pool by graph alone, 8.3 of them survive into a 32-turn
    pack, and they return gold at 0.4%.
    """
    if not structured:
        return ScheduleResult(seed_ids[:budget.traversal_nodes], (), {}, {
            "node_cap_reached": len(seed_ids) > budget.traversal_nodes, "edge_cap_reached": False,
            "hop_cap_reached": False, "frontier_truncated": False,
        }, {node_id: 0 for node_id in seed_ids[:budget.traversal_nodes]})
    preferred = tuple(preferred_relations) if preferred_relations is not None else DEFAULT_PREFERRED
    if obligation_aware_relations:
        preferred = tuple(dict.fromkeys((*_TYPED_RELATIONS, *preferred)))
    fallback = tuple(fallback_relations) if fallback_relations is not None else DEFAULT_FALLBACK
    queue = [(node_id, 0, False, None) for node_id in seed_ids]
    query_terms = content_terms(ir.query)
    visited: list[str] = []; seen: set[str] = set(); proof: list[ProofStep] = []; relation_counts: dict[str, int] = {}
    node_hops: dict[str, int] = {}
    fallback_nodes = fallback_edges = 0; frontier_truncated = False; hop_cap = False
    while queue and len(visited) < budget.traversal_nodes and len(proof) < budget.max_visited_edges:
        node_id, hop, is_fallback, parent = queue.pop(0)
        if node_id in seen: continue
        seen.add(node_id); visited.append(node_id); node_hops[node_id] = hop
        if parent is not None:
            proof.append(ProofStep(parent.edge.edge_id, parent.edge.src_id, parent.edge.relation,
                                   parent.edge.dst_id, parent.edge.evidence_group_id,
                                   inverse=parent.inverse))
            relation_counts[str(parent.edge.relation)] = relation_counts.get(str(parent.edge.relation), 0) + 1
        if hop >= budget.max_hops:
            hop_cap = hop_cap or bool(view.neighbors(node_id, semantic_only=True)); continue
        allowed = fallback if is_fallback else preferred
        arrived_signals = (_relation_mask_signals(parent.edge.source)
                           if parent is not None
                           and parent.edge.relation == RelationType.COARSE_RELATED
                           else frozenset())
        typed_region_arrival = bool(arrived_signals & {
            "shared_entity", "state_compatible", "temporal_near",
            "collection_related", "lexical_rare",
        })
        if typed_region_arrival:
            # A typed coarse edge locates a related region; its value is lost if
            # the next beam keeps following region edges and never hydrates the
            # region's evidence.  Admit structural descent in the normal beam
            # for this one expansion only.  Fact destinations then win via the
            # existing fact bonus, while pure scene-similar edges retain the old
            # conservative fallback behaviour.
            allowed = tuple(dict.fromkeys((
                *allowed, RelationType.SCENE_CONTAINS,
                RelationType.REFINES_TO)))
        def destination_priority(row):
            node = view.nodes.get(row.next_node_id)
            if not node:
                return (0.0, row.edge.edge_id)
            attrs = node.attributes
            # Exact lexical surface, lazily memoized by ``content_terms``.  A
            # full per-node duplicate made small hot sets fast but inflated each
            # cached tenant view and worsened large-tenant p95/RSS.
            lexical = len(query_terms & content_terms(
                " ".join(str(attrs.get(key, "")) for key in
                         ("predicate", "scope", "collection_key", "value"))
                + " " + node.summary))
            collection = node.node_type.value in {"collection_manifest", "collection_scope"}
            collection_bonus = 4.0 if collection and ir.operator.value in {
                "count_distinct", "union_distinct", "intersection_distinct", "group_by_owner"
            } else 0.0
            fact_bonus = 1.0 if node.node_type.value == "canonical_fact" else 0.0
            relation_bonus = (_obligation_relation_bonus(
                ir, row.edge.relation, inverse=row.inverse)
                              if obligation_aware_relations else 0.0)
            signal_bonus = _coarse_signal_bonus(ir, row.edge.source)
            return (collection_bonus + lexical + fact_bonus + relation_bonus
                    + signal_bonus,
                    row.edge.edge_id)

        # Score each neighbour once; the previous key lambda called
        # destination_priority twice per comparison.
        scored = [(destination_priority(row), row) for row in
                  view.neighbors(node_id, allowed, semantic_only=True)]
        ordered = sorted(scored, key=lambda item: (-item[0][0], item[0][1]))
        # Prune per expansion, before anything is queued: a node that scores
        # below its siblings is not going to be rescued by the frontier cap,
        # which drops the oldest entries rather than the weakest ones.
        admitted = 0
        for _, row in ordered:
            if row.next_node_id in seen: continue
            fallback_edge = row.edge.relation == RelationType.SCENE_CONTAINS
            if fallback_edge and (fallback_nodes >= 16 or fallback_edges >= 32): continue
            if expansion_beam and admitted >= expansion_beam: break
            queue.append((row.next_node_id, hop + 1, fallback_edge, row))
            admitted += 1
            if fallback_edge: fallback_edges += 1
        if not queue and hop < budget.max_hops:
            for row in view.neighbors(node_id, fallback, semantic_only=True):
                if row.next_node_id not in seen and fallback_nodes < 16:
                    queue.append((row.next_node_id, hop + 1, True, row)); fallback_nodes += 1
        if len(queue) > budget.max_frontier:
            queue = queue[:budget.max_frontier]; frontier_truncated = True
    return ScheduleResult(tuple(visited), tuple(proof), relation_counts, {
        "node_cap_reached": len(visited) >= budget.traversal_nodes,
        "edge_cap_reached": len(proof) >= budget.max_visited_edges,
        "hop_cap_reached": hop_cap, "frontier_truncated": frontier_truncated,
    }, node_hops)
