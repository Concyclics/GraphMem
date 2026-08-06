"""Content edges: facts linked by what they say, not by what contains them.

Measured lift of each relation, as the share of its edges whose destination is a
gold fact against the base rate of gold facts in the memory (2 LoCoMo
conversations, 107 questions):

    collection_co_member  2.27x    state_next      2.27x
    has_fact              1.00x    scene_contains  0.41x
    member_of / refines_to / participates_in / at_time   0.00x

The pattern is exact: the only relations that carry signal are the two that link
facts *to each other by content*.  Everything at or below 1.0x encodes
containment -- which entity owns the fact, which scene holds it, which session
refines to which scene -- and containment is orthogonal to aboutness, which is
what a question asks.  Traversal over the containment graph was measured to
contribute *nothing*: disabling it entirely left turn_recall at 0.326 and
turn_all_hit at 0.509, identical to the baseline.

This module builds the missing content layer deterministically from attributes
extraction already wrote.  ``_semantic_graph`` built a version of it for the
g0-g4 variants and the lean g5 path dropped it; rebuilding it here costs no LLM
call and no re-extraction.

The cross-session ``SHARED_VALUE`` link is the point.  In the frozen graph the
only thing joining two sessions is the shared owner entity, which measured
0.00x, so a multi-session question has no content path between the sessions
holding its evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..domain import GraphEdge, GraphNode, NodeType, RelationType, stable_id
from .config import ProjectionConfig


@dataclass(frozen=True, slots=True)
class ValueLatticeRow:
    value_id: str
    value_key: str
    fact_ids: tuple[str, ...]
    session_ids: tuple[str, ...]

    @property
    def cross_session(self) -> bool:
        return len(self.session_ids) > 1


def _attribute(node: GraphNode, key: str) -> str:
    return str(node.attributes.get(key, "") or "")


def collect_values(nodes: Iterable[GraphNode]) -> dict[tuple[str, str], list[GraphNode]]:
    """Group canonical facts by (value_type, normalized value)."""
    rows: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    for node in nodes:
        if node.node_type != NodeType.CANONICAL_FACT:
            continue
        value_key = _attribute(node, "value_key")
        if not value_key:
            continue
        rows[(_attribute(node, "value_type") or "text", value_key)].append(node)
    for group in rows.values():
        group.sort(key=lambda row: (_attribute(row, "session_id"),
                                    int(row.attributes.get("turn_index", -1) or -1),
                                    row.node_id))
    return rows


def build_value_lattice(
    memory_id: str, nodes: Sequence[GraphNode], config: ProjectionConfig,
) -> tuple[list[GraphNode], list[GraphEdge], list[ValueLatticeRow]]:
    """Emit CANONICAL_VALUE nodes, FACT_VALUE edges and cross-session SHARED_VALUE."""
    if not config.value_lattice:
        return [], [], []
    value_nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    rows: list[ValueLatticeRow] = []

    for (value_type, value_key), facts in sorted(collect_values(nodes).items()):
        # A value seen once links nothing; it would add a node and an edge that
        # can never join two facts.
        if len(facts) < 2:
            continue
        evidence = tuple(dict.fromkeys(
            group for fact in facts for group in fact.all_evidence_group_ids))
        if not evidence:
            continue
        value_id = stable_id("node", memory_id, "canonical-value", value_type, value_key)
        sessions = tuple(dict.fromkeys(_attribute(fact, "session_id") for fact in facts))
        value_nodes.append(GraphNode(
            value_id, memory_id, NodeType.CANONICAL_VALUE, 0,
            str(facts[0].attributes.get("value", value_key)), evidence[0], evidence[1:],
            attributes={
                "value_type": value_type, "normalized": value_key,
                "fact_ids": tuple(fact.node_id for fact in facts),
                "session_ids": sessions, "cross_session": len(sessions) > 1,
                "roles": ("value", "route"), "provenance_scope": "route",
            }))
        for fact in facts:
            edges.append(GraphEdge(
                stable_id("edge", memory_id, "fact-value", fact.node_id, value_id),
                memory_id, fact.node_id, RelationType.FACT_VALUE, value_id,
                fact.evidence_group_id, True, 1.0, "projection_value_lattice",
                fact.evidence_group_ids))
        # SHARED_VALUE is emitted only across sessions.  Within one session the
        # facts are already adjacent through the scene, so a same-session link
        # adds degree without adding reach -- and degree is what crowds the
        # budget out.
        if len(sessions) > 1:
            capped = facts[: config.shared_value_cap]
            for index, left in enumerate(capped):
                for right in capped[index + 1:]:
                    if _attribute(left, "session_id") == _attribute(right, "session_id"):
                        continue
                    edges.append(GraphEdge(
                        stable_id("edge", memory_id, "shared-value", left.node_id, right.node_id),
                        memory_id, left.node_id, RelationType.SHARED_VALUE, right.node_id,
                        left.evidence_group_id, False, 1.0, "projection_shared_value",
                        left.evidence_group_ids))
        rows.append(ValueLatticeRow(value_id, value_key,
                                    tuple(fact.node_id for fact in facts), sessions))
    return value_nodes, edges, rows


def lattice_stats(rows: Sequence[ValueLatticeRow]) -> Mapping[str, float | int]:
    if not rows:
        return {"values": 0}
    sizes = sorted(len(row.fact_ids) for row in rows)
    cross = [row for row in rows if row.cross_session]
    return {
        "values": len(rows),
        "cross_session_values": len(cross),
        "facts_linked": sum(sizes),
        "facts_per_value_mean": sum(sizes) / len(sizes),
        "facts_per_value_max": sizes[-1],
        "sessions_per_cross_value_mean": (
            sum(len(row.session_ids) for row in cross) / len(cross) if cross else 0.0),
    }
