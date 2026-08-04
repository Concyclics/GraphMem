from __future__ import annotations

import hashlib
from dataclasses import dataclass

from graphmem_demo.v36.schema import V36Index

from ..domain import (
    EvidenceGroup,
    EvidenceMember,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
    SourceTurn,
    stable_id,
)


_RELATIONS = {
    "routing_contains": RelationType.REFINES_TO,
    "source": RelationType.HAS_EVIDENCE,
    "next_turn": RelationType.DIALOGUE_PAIR,
    "dialogue_pair": RelationType.DIALOGUE_PAIR,
    "reference": RelationType.COREFERENCE,
    "same_event": RelationType.SAME_EVENT,
    "state_transition": RelationType.STATE_TRANSITION,
    "temporal_endpoint": RelationType.AT_TIME,
    "semantic_neighbor": RelationType.MENTIONS,
    "collection_member": RelationType.CONTAINS,
    "contrast": RelationType.REPLACEMENT,
}


@dataclass(frozen=True, slots=True)
class LegacyGraph:
    turns: tuple[SourceTurn, ...]
    evidence_groups: tuple[EvidenceGroup, ...]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def adapt_v36_index(index: V36Index, memory_id: str) -> LegacyGraph:
    """Convert an immutable V3.6/V4 physical graph into V5 contracts.

    Raw text remains only on SourceTurn. Graph node summaries are compact routing
    fields so that the same objects are safe to project to Neo4j later.
    """
    turn_map: dict[str, str] = {}
    turns: list[SourceTurn] = []
    for turn in index.turns:
        turn_id = stable_id("turn", memory_id, turn.session_id, turn.turn_index)
        turn_map[turn.node_id] = turn_id
        turns.append(SourceTurn(
            turn_id=turn_id,
            memory_id=memory_id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            listener=turn.listener,
            role=turn.transport_role,
            timestamp=turn.session_date,
            raw_text=turn.text,
            content_hash=hashlib.sha256(turn.text.encode("utf-8")).hexdigest(),
        ))

    group_map: dict[str, str] = {}
    groups: list[EvidenceGroup] = []
    for group in index.evidence_groups:
        members = tuple(
            EvidenceMember(turn_map[source], 0, len(next(
                (turn.raw_text for turn in turns if turn.turn_id == turn_map[source]), ""
            )), "legacy_source")
            for source in group.source_turn_ids if source in turn_map
        )
        if not members:
            continue
        group_id = stable_id("evidence", memory_id, group.group_kind, [m.turn_id for m in members])
        group_map[group.group_id] = group_id
        groups.append(EvidenceGroup(
            evidence_group_id=group_id,
            memory_id=memory_id,
            members=members,
            content_hash=stable_id("content", [m.turn_id for m in members]),
        ))

    fallback_groups: dict[tuple[str, ...], str] = {}
    def evidence_id(source_ids: list[str]) -> str:
        source_turn_ids = tuple(turn_map[source] for source in source_ids if source in turn_map)
        if not source_turn_ids:
            raise ValueError("legacy semantic object has no valid source turn")
        if source_turn_ids in fallback_groups:
            return fallback_groups[source_turn_ids]
        group_id = stable_id("evidence", memory_id, "legacy", source_turn_ids)
        fallback_groups[source_turn_ids] = group_id
        members = tuple(EvidenceMember(turn_id, 0, len(next(
            turn.raw_text for turn in turns if turn.turn_id == turn_id
        )), "legacy_source") for turn_id in source_turn_ids)
        groups.append(EvidenceGroup(
            evidence_group_id=group_id, memory_id=memory_id, members=members,
            content_hash=stable_id("content", source_turn_ids),
        ))
        return group_id

    node_map: dict[str, str] = {}
    nodes: list[GraphNode] = []
    for card in index.routing_cards:
        node_id = stable_id("node", memory_id, "routing_card", card.session_id)
        node_map[card.card_id] = node_id
        nodes.append(GraphNode(
            node_id=node_id, memory_id=memory_id, node_type=NodeType.ROUTING_CARD,
            level=1, summary=card.routing_text[:720],
            evidence_group_id=evidence_id(card.turn_ids),
            attributes={"session_id": card.session_id},
        ))
    for frame in index.frames:
        node_id = stable_id("node", memory_id, "frame", frame.frame_id)
        node_map[frame.frame_id] = node_id
        nodes.append(GraphNode(
            node_id=node_id, memory_id=memory_id,
            node_type=(NodeType.EVENT_FRAME if frame.frame_kind == "event" else NodeType.ROLE_FRAME),
            level=0, summary=frame.retrieval_text[:720],
            evidence_group_id=evidence_id(frame.source_turn_ids),
            event_time=frame.temporal.event_time,
            state=frame.lifecycle_status,
            confidence=frame.confidence,
            attributes={"owner": frame.owner_key, "entity": frame.entity_key,
                        "predicate": frame.predicate_key, "object": frame.object_key},
        ))

    edges: list[GraphEdge] = []
    for legacy in index.edges:
        if legacy.src not in node_map or legacy.dst not in node_map:
            continue
        relation = _RELATIONS.get(legacy.relation)
        if relation is None:
            continue
        source_ids = [str(value) for value in legacy.provenance.get("source_turn_ids", [])]
        if not source_ids:
            source_ids = next((frame.source_turn_ids for frame in index.frames
                               if frame.frame_id in {legacy.src, legacy.dst}), [])
        if not source_ids:
            continue
        group_id = evidence_id(source_ids)
        src_id, dst_id = node_map[legacy.src], node_map[legacy.dst]
        edges.append(GraphEdge(
            edge_id=stable_id("edge", src_id, relation, dst_id, group_id),
            memory_id=memory_id, src_id=src_id, relation=relation, dst_id=dst_id,
            evidence_group_id=group_id, directed=legacy.directed,
            confidence=legacy.confidence, source="legacy_v4_adapter",
        ))
    return LegacyGraph(tuple(turns), tuple(groups), tuple(nodes), tuple(edges))
