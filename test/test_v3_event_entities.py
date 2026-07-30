from __future__ import annotations

import json
from dataclasses import asdict

from graphmem_demo.v3.build import validate_hypergraph
from graphmem_demo.v3.event_entities import (
    event_entity_candidate_payload,
    parse_event_entities,
)
from graphmem_demo.v3.schema import (
    EventNode,
    TurnNode,
    V3Index,
    index_from_dict,
)


def _turn(index: int, text: str) -> TurnNode:
    session_id = f"s{index}"
    return TurnNode(
        node_id=f"q:{session_id}:turn:0",
        question_id="q",
        session_id=session_id,
        session_date=f"2026-02-{index + 1:02d}",
        turn_index=0,
        speaker="Priya",
        speaker_key="priya",
        listener="Noah",
        transport_role="user",
        text=text,
        retrieval_text=f"speaker Priya | {text}",
        embedding=[1.0, 0.0],
    )


def _event(
    index: int,
    label: str,
    status: str,
    event_time: str | None = None,
) -> EventNode:
    session_id = f"s{index}"
    return EventNode(
        node_id=f"q:{session_id}:event:0",
        question_id="q",
        session_id=session_id,
        label=label,
        label_key=label.casefold(),
        status=status,  # type: ignore[arg-type]
        participant_keys=["priya"],
        event_time=event_time,
        source_turn_ids=[f"q:{session_id}:turn:0"],
        semantic_type_keys=["fabrication prototype", "solar kiln"],
        retrieval_text=label,
        embedding=[1.0, 0.0],
    )


def _fixture() -> tuple[list[TurnNode], list[EventNode], dict]:
    turns = [
        _turn(0, "I started building my solar kiln prototype."),
        _turn(1, "The kiln prototype is coming together."),
        _turn(2, "I finished the prototype this week."),
    ]
    events = [
        _event(0, "Priya starts a solar kiln prototype", "asserted"),
        _event(1, "Priya reports progress on the kiln prototype", "asserted"),
        _event(2, "Priya completes the prototype", "complete", "2026-02-03"),
    ]
    candidates = event_entity_candidate_payload(events, turns)
    assert candidates is not None
    return turns, events, candidates


def test_event_entity_candidates_support_multi_mention_lifecycle() -> None:
    _turns, events, candidates = _fixture()
    candidate_ids = {
        row["event_id"] for row in candidates["event_candidates"]
    }
    assert candidate_ids == {event.node_id for event in events}
    assert len(candidates["event_candidate_links"]) >= 2
    assert len(candidates["event_neighborhoods"]) == 1


def test_event_entity_parser_builds_grounded_multiary_hyperedge() -> None:
    turns, events, candidates = _fixture()
    response = {
        "event_clusters": [{
            "member_event_ids": [event.node_id for event in events],
            "canonical_label": "Priya's solar kiln prototype",
            "identity_anchors": ["solar kiln prototype"],
            "confidence": 0.94,
        }]
    }
    entities, edges = parse_event_entities(
        json.dumps(response),
        question_id="q",
        events=events,
        candidate_payload=candidates,
    )
    assert len(entities) == len(edges) == 1
    entity = entities[0]
    assert entity.member_event_ids == [event.node_id for event in events]
    assert entity.current_event_id == events[-1].node_id
    assert entity.lifecycle_status == "complete"
    assert entity.time_end == "2026-02-03"
    assert "solar" in entity.retrieval_text
    assert edges[0].relation == "event_entity_member"
    assert [row.role for row in edges[0].incidences] == [
        "identity", "mention_asserted", "mention_asserted", "mention_complete",
    ]
    index = V3Index(
        turns=turns,
        events=events,
        event_entities=entities,
        hyperedges=edges,
    )
    assert validate_hypergraph(index) == []
    restored = index_from_dict(asdict(index))
    assert restored.event_entities[0].member_event_ids == entity.member_event_ids


def test_event_entity_parser_rejects_hallucinated_or_disconnected_member() -> None:
    _turns, events, candidates = _fixture()
    hallucinated = {
        "event_clusters": [[
            [events[0].node_id, "q:missing:event:0"],
            "solar kiln prototype", ["solar kiln"], 0.99,
        ]]
    }
    entities, edges = parse_event_entities(
        json.dumps(hallucinated),
        question_id="q",
        events=events,
        candidate_payload=candidates,
    )
    assert entities == []
    assert edges == []

    disconnected = dict(candidates)
    disconnected["event_candidate_links"] = [
        candidates["event_candidate_links"][0]
    ]
    response = {
        "event_clusters": [[
            [event.node_id for event in events],
            "solar kiln prototype", ["solar kiln"], 0.99,
        ]]
    }
    entities, _edges = parse_event_entities(
        json.dumps(response),
        question_id="q",
        events=events,
        candidate_payload=disconnected,
    )
    assert entities == []


def test_event_entity_parser_rejects_sibling_terminal_occurrences() -> None:
    turns, events, candidates = _fixture()
    sibling_turn = _turn(3, "I also completed a second kiln prototype.")
    sibling = _event(
        3, "Priya completes a second kiln prototype", "complete", "2026-02-10"
    )
    turns.append(sibling_turn)
    events.append(sibling)
    candidates = event_entity_candidate_payload(events, turns)
    assert candidates is not None
    response = {
        "event_clusters": [[
            [event.node_id for event in events],
            "Priya's kiln prototypes", ["kiln prototype"], 0.99,
        ]]
    }
    entities, edges = parse_event_entities(
        json.dumps(response),
        question_id="q",
        events=events,
        candidate_payload=candidates,
    )
    assert entities == []
    assert edges == []


def test_event_decision_is_constrained_to_terminal_neighborhood() -> None:
    _turns, events, candidates = _fixture()
    neighborhood = candidates["event_neighborhoods"][0]
    target_id = neighborhood["target_event_id"]
    predecessor_id = neighborhood["candidate_predecessor_ids"][0]
    response = {
        "event_decisions": [{
            "target_event_id": target_id,
            "predecessor_event_ids": [predecessor_id],
            "canonical_label": "Priya's solar kiln prototype",
            "identity_anchors": ["solar kiln prototype"],
            "confidence": 0.95,
        }]
    }
    entities, edges = parse_event_entities(
        json.dumps(response),
        question_id="q",
        events=events,
        candidate_payload=candidates,
    )
    assert len(entities) == len(edges) == 1
    assert entities[0].current_event_id == target_id

    response["event_decisions"][0]["predecessor_event_ids"] = [target_id]
    entities, edges = parse_event_entities(
        json.dumps(response),
        question_id="q",
        events=events,
        candidate_payload=candidates,
    )
    assert entities == []
    assert edges == []
