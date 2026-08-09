"""Operator-aware, directional traversal of the routing hierarchy.

The V5.8 graph already stores ``REFINES_TO`` edges, but older retrieval flattened
every card's postings into one global map.  This module turns those edges into a
physical data plane: score roots, inspect only their children, retain a bounded
beam, and repeat until terminal facts/scenes are reached.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain import NodeType, QueryOperator
from ..text import content_terms, normalize_key
from .operators import requires_exhaustive_scope
from .query_ir import QueryIR


_SET_OPERATORS = {
    QueryOperator.UNION_DISTINCT,
    QueryOperator.INTERSECTION_DISTINCT,
    QueryOperator.GROUP_BY_OWNER,
    QueryOperator.COUNT_DISTINCT,
}
_TEMPORAL_OPERATORS = {
    QueryOperator.ARGMIN_TIME,
    QueryOperator.ARGMAX_TIME,
    QueryOperator.ORDINAL,
    QueryOperator.DATE_DIFFERENCE,
    QueryOperator.LATEST_STATE,
}


@dataclass(frozen=True, slots=True)
class PhysicalRoutePlan:
    query_terms: frozenset[str]
    posting_keys: tuple[str, ...]
    operator: QueryOperator
    root_beam: int = 2
    child_beam: int = 4
    max_depth: int = 8
    max_selected_nodes: int = 64
    portal_beam: int = 0
    exhaustive: bool = False
    temporal: bool = False
    adaptive: bool = True


@dataclass(frozen=True, slots=True)
class HierarchicalRouteResult:
    selected_node_ids: tuple[str, ...]
    terminal_node_ids: tuple[str, ...]
    visited_node_ids: tuple[str, ...]
    selected_by_depth: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    portal_node_ids: tuple[str, ...] = ()
    candidate_count: int = 0
    max_frontier: int = 0
    widened: bool = False
    exhausted: bool = False

    @property
    def stats(self) -> Mapping[str, Any]:
        return {
            "selected_nodes": len(self.selected_node_ids),
            "terminal_nodes": len(self.terminal_node_ids),
            "visited_nodes": len(self.visited_node_ids),
            "candidate_count": self.candidate_count,
            "levels": len(self.selected_by_depth),
            "max_frontier": self.max_frontier,
            "adaptive_widened": self.widened,
            "route_exhausted": self.exhausted,
            "portal_nodes": len(self.portal_node_ids),
        }


def compile_physical_route(
    ir: QueryIR,
    *,
    max_nodes: int,
    root_beam: int = 2,
    child_beam: int = 4,
    operator_aware: bool = True,
) -> PhysicalRoutePlan:
    ast = ir.ast
    operator = ir.ast_operator or ir.operator
    operands = ir.ast_operands or ir.operands
    keys: list[str] = []
    for operand in operands:
        keys.extend(operand.owner_aliases)
        keys.extend(operand.predicate_candidates)
        keys.extend(operand.scope_candidates)
        if operand.temporal_constraint:
            keys.append(operand.temporal_constraint)
    query_terms = content_terms(ir.query)
    keys.extend(sorted(query_terms))
    posting_keys = tuple(dict.fromkeys(
        normalized for value in keys if (normalized := normalize_key(str(value)))))
    exhaustive = (requires_exhaustive_scope(ast) if ast is not None
                  else operator in _SET_OPERATORS)
    temporal = operator in _TEMPORAL_OPERATORS
    # Set/closure queries widen at every chosen parent.  Lookup queries keep a
    # narrow beam; missing obligations can still fall back to the flat reservoir.
    effective_beam = (max(child_beam, 8)
                      if exhaustive and operator_aware else child_beam)
    return PhysicalRoutePlan(
        query_terms=query_terms,
        posting_keys=posting_keys,
        operator=operator,
        root_beam=max(1, root_beam),
        child_beam=max(1, effective_beam),
        max_selected_nodes=max(1, max_nodes),
        # Operator-aware plans may open a few sparse lexical portals and then
        # stitch their ancestor corridor into the proof.  This is analogous to
        # multiple entry points in a navigable small-world index: it rescues a
        # unique leaf term that a lossy parent summary cannot carry, without
        # flattening every matching route card into the candidate pool.
        portal_beam=(max(root_beam, 8 if exhaustive else 4)
                     if operator_aware else 0),
        exhaustive=exhaustive,
        temporal=temporal,
        adaptive=operator_aware,
    )


def _score_child(view, parent_id: str, child_id: str, plan: PhysicalRoutePlan,
                 posting_hits: Mapping[str, int]) -> float:
    node = view.nodes[child_id]
    terms = view.node_terms.get(child_id, frozenset())
    lexical = len(plan.query_terms & terms) / max(1, len(plan.query_terms))
    score = lexical * 4.0
    # A binary posting bonus makes every child containing a common word tie with
    # the child matching owner + predicate + unique anchor.  Count independent
    # parent-local key hits instead; this is still O(keys + local postings) and
    # never consults the flattened leaf index.
    score += min(4.0, 0.8 * posting_hits.get(child_id, 0))
    if plan.exhaustive and node.node_type in {
            NodeType.COLLECTION_SCOPE, NodeType.COLLECTION_MANIFEST}:
        score += 1.0
    if plan.temporal and (
            node.event_time or node.attributes.get("event_time_range")
            or node.attributes.get("observation_time_range")
            or bool(content_terms(" ".join(map(str, node.attributes.get("times", ())))))):
        score += 0.75
    # Confidence is a tie-break signal, never strong enough to beat a lexical or
    # parent-posting hit.
    return score + float(node.confidence) * 0.01


def route_hierarchy(view, plan: PhysicalRoutePlan) -> HierarchicalRouteResult:
    roots = list(view.routing_roots())
    if not roots:
        roots = list(view.lexical_nodes(
            plan.query_terms, limit=plan.root_beam, route_only=True))
    if not roots:
        return HierarchicalRouteResult((), (), (), exhausted=True)

    ranked_roots = sorted(roots, key=lambda node_id: (
        -len(plan.query_terms & view.node_terms.get(node_id, frozenset())), node_id))
    frontier = ranked_roots[:plan.root_beam]
    selected: list[str] = list(frontier)
    terminals: list[str] = []
    visited: list[str] = list(roots)
    selected_by_depth: dict[int, tuple[str, ...]] = {0: tuple(frontier)}
    candidate_count = len(roots)
    max_frontier = len(frontier)
    widened = False
    exhausted = False

    for depth in range(1, plan.max_depth + 1):
        next_frontier: list[str] = []
        depth_selected: list[str] = []
        for parent_id in frontier:
            children = list(view.hierarchy_children(parent_id))
            if not children:
                terminals.append(parent_id)
                continue
            candidate_count += len(children)
            visited.extend(children)
            posting_hits: Counter[str] = Counter()
            for key in plan.posting_keys:
                posting_hits.update(
                    view.parent_posting_children(parent_id, (key,)))
            scored = [(_score_child(view, parent_id, child_id, plan, posting_hits), child_id)
                      for child_id in children]
            scored.sort(key=lambda row: (-row[0], row[1]))
            positive = [row for row in scored if row[0] > 0.011]
            beam = plan.child_beam
            if (plan.adaptive and plan.exhaustive
                    and len(positive) < min(2, len(children))):
                beam = min(len(children), plan.child_beam * 2)
                widened = widened or beam > plan.child_beam
            admitted = (positive or scored)[:beam]
            for _score, child_id in admitted:
                if child_id in selected:
                    continue
                if len(selected) >= plan.max_selected_nodes:
                    exhausted = True
                    break
                selected.append(child_id)
                depth_selected.append(child_id)
                if view.hierarchy_children(child_id):
                    next_frontier.append(child_id)
                else:
                    terminals.append(child_id)
            if exhausted:
                break
        if depth_selected:
            selected_by_depth[depth] = tuple(depth_selected)
        if exhausted or not next_frontier:
            break
        frontier = list(dict.fromkeys(next_frontier))
        max_frontier = max(max_frontier, len(frontier))

    portal_nodes: list[str] = []
    if plan.portal_beam and len(selected) < plan.max_selected_nodes:
        portal_candidates = view.lexical_nodes(
            plan.query_terms,
            limit=max(plan.portal_beam * 8, 32),
            route_only=True,
        )
        candidate_count += len(portal_candidates)
        leaves = [
            node_id for node_id in portal_candidates
            if view.nodes[node_id].node_type == NodeType.ROUTING_CARD
            and view.nodes[node_id].attributes.get("session_id") is not None
        ][:plan.portal_beam]
        for leaf_id in leaves:
            if leaf_id not in selected and len(selected) < plan.max_selected_nodes:
                selected.append(leaf_id)
                terminals.append(leaf_id)
                portal_nodes.append(leaf_id)
            # Preserve a checkable root-to-leaf corridor for portal admission.
            cursor = leaf_id
            for _depth in range(plan.max_depth):
                parents = view.hierarchy_parents(cursor)
                if not parents:
                    break
                parent_id = parents[0]
                visited.append(parent_id)
                if (parent_id not in selected
                        and len(selected) < plan.max_selected_nodes):
                    selected.append(parent_id)
                cursor = parent_id

    return HierarchicalRouteResult(
        selected_node_ids=tuple(dict.fromkeys(selected)),
        terminal_node_ids=tuple(dict.fromkeys(terminals)),
        visited_node_ids=tuple(dict.fromkeys(visited)),
        selected_by_depth=selected_by_depth,
        candidate_count=candidate_count,
        max_frontier=max_frontier,
        widened=widened,
        exhausted=exhausted,
        portal_node_ids=tuple(portal_nodes),
    )
