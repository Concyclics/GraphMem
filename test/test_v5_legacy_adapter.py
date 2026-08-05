from __future__ import annotations

from graphmem.domain import NodeType
from graphmem.legacy import adapt_v36_index
from graphmem_demo.v36.schema import (
    EvidenceGroup,
    GraphEdgeV36,
    RoleFrameNode,
    RoutingCard,
    TurnNodeV36,
    V36Index,
)


def _index() -> V36Index:
    turn = TurnNodeV36(
        node_id="q:s:turn:0", question_id="q", session_id="s",
        session_date="2026-01-01", turn_index=0, speaker="A",
        speaker_key="a", listener="B", transport_role="user",
        text="A booked the train.", retrieval_text="A booked train",
    )
    frame = RoleFrameNode(
        frame_id="q:s:frame:0", question_id="q", session_ids=["s"],
        frame_kind="event", owner_key="a", entity_key="train",
        predicate_key="booked", object_key="train",
        source_turn_ids=[turn.node_id], retrieval_text="a booked train",
    )
    card = RoutingCard(
        card_id="q:s:card", question_id="q", session_id="s",
        speaker_keys=["a"], canonical_entities=["train"], relations=["booked"],
        key_events=["booked train"], current_states=[], time_range="2026-01-01",
        frame_ids=[frame.frame_id], turn_ids=[turn.node_id],
        routing_text="train booking",
    )
    group = EvidenceGroup(
        group_id="q:group:0", question_id="q", group_kind="single_fact",
        member_frame_ids=[frame.frame_id], source_turn_ids=[turn.node_id],
        required_roles=["event", "source"],
        completeness_mask={"event": True, "source": True},
        provenance_complete=True, confidence=1.0,
        retrieval_text="train booking", session_ids=["s"],
    )
    edge = GraphEdgeV36(
        edge_id="q:edge:0", question_id="q", src=card.card_id,
        dst=frame.frame_id, relation="routing_contains", directed=True,
        confidence=1.0, provenance={"source_turn_ids": [turn.node_id]},
    )
    return V36Index(
        turns=[turn], frames=[frame], routing_cards=[card],
        evidence_groups=[group], edges=[edge],
    )


def test_adapter_is_deterministic_and_keeps_raw_text_off_graph_nodes() -> None:
    first = adapt_v36_index(_index(), "memory")
    second = adapt_v36_index(_index(), "memory")
    assert first == second
    assert first.turns[0].raw_text == "A booked the train."
    assert {node.node_type for node in first.nodes} == {
        NodeType.ROUTING_CARD, NodeType.EVENT_FRAME,
    }
    assert all("A booked the train." not in node.summary for node in first.nodes)
    assert all(edge.evidence_group_id for edge in first.edges)


def test_adapter_output_changes_with_memory_identity() -> None:
    assert adapt_v36_index(_index(), "a").nodes != adapt_v36_index(_index(), "b").nodes
