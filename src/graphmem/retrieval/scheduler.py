from __future__ import annotations

from dataclasses import dataclass

from ..domain import ProofStep, QueryBudget, RelationType
from ..text import content_terms
from .query_ir import QueryIR


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    visited_node_ids: tuple[str, ...]
    proof: tuple[ProofStep, ...]
    relation_counts: dict[str, int]
    exhaustion: dict[str, bool]


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
    RelationType.COLLECTION_CO_MEMBER, RelationType.STATE_NEXT, RelationType.TEMPORAL_BEFORE,
    RelationType.SHARED_VALUE, RelationType.FACT_VALUE,
)


def execute(view, ir: QueryIR, seed_ids: tuple[str, ...], budget: QueryBudget, *,
            structured: bool,
            preferred_relations=None, fallback_relations=None) -> ScheduleResult:
    """Deterministic obligation-first traversal; provenance is hydrated later."""
    if not structured:
        return ScheduleResult(seed_ids[:budget.traversal_nodes], (), {}, {
            "node_cap_reached": len(seed_ids) > budget.traversal_nodes, "edge_cap_reached": False,
            "hop_cap_reached": False, "frontier_truncated": False,
        })
    preferred = tuple(preferred_relations) if preferred_relations is not None else DEFAULT_PREFERRED
    fallback = tuple(fallback_relations) if fallback_relations is not None else DEFAULT_FALLBACK
    queue = [(node_id, 0, False, None) for node_id in seed_ids]
    visited: list[str] = []; seen: set[str] = set(); proof: list[ProofStep] = []; relation_counts: dict[str, int] = {}
    fallback_nodes = fallback_edges = 0; frontier_truncated = False; hop_cap = False
    while queue and len(visited) < budget.traversal_nodes and len(proof) < budget.max_visited_edges:
        node_id, hop, is_fallback, parent = queue.pop(0)
        if node_id in seen: continue
        seen.add(node_id); visited.append(node_id)
        if parent is not None:
            proof.append(ProofStep(parent.edge.edge_id, parent.edge.src_id, parent.edge.relation,
                                   parent.edge.dst_id, parent.edge.evidence_group_id,
                                   inverse=parent.inverse))
            relation_counts[str(parent.edge.relation)] = relation_counts.get(str(parent.edge.relation), 0) + 1
        if hop >= budget.max_hops:
            hop_cap = hop_cap or bool(view.neighbors(node_id, semantic_only=True)); continue
        allowed = fallback if is_fallback else preferred
        query_terms = content_terms(ir.query)

        def destination_priority(row):
            node = view.nodes.get(row.next_node_id)
            if not node:
                return (0.0, row.edge.edge_id)
            attrs = node.attributes
            lexical = len(query_terms & content_terms(
                " ".join(str(attrs.get(key, "")) for key in ("predicate", "scope", "collection_key", "value"))
                + " " + node.summary))
            collection = node.node_type.value in {"collection_manifest", "collection_scope"}
            collection_bonus = 4.0 if collection and ir.operator.value in {
                "count_distinct", "union_distinct", "intersection_distinct", "group_by_owner"
            } else 0.0
            fact_bonus = 1.0 if node.node_type.value == "canonical_fact" else 0.0
            return (collection_bonus + lexical + fact_bonus, row.edge.edge_id)

        # Score each neighbour once; the previous key lambda called
        # destination_priority twice per comparison.
        scored = [(destination_priority(row), row) for row in
                  view.neighbors(node_id, allowed, semantic_only=True)]
        for _, row in sorted(scored, key=lambda item: (-item[0][0], item[0][1])):
            if row.next_node_id in seen: continue
            fallback_edge = row.edge.relation == RelationType.SCENE_CONTAINS
            if fallback_edge and (fallback_nodes >= 16 or fallback_edges >= 32): continue
            queue.append((row.next_node_id, hop + 1, fallback_edge, row))
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
    })
