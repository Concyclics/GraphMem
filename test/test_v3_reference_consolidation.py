from __future__ import annotations

import json

from graphmem_demo.v3.reference_consolidation import (
    _NAMED_VALUE_CUE,
    parse_reference_edges,
    event_identity_candidate_rows,
    parse_event_identity_edges,
    reference_candidate_payload,
)
from graphmem_demo.v3.schema import EventNode, TurnNode


def _turn(
    index: int,
    speaker: str,
    listener: str,
    text: str,
    embedding: list[float],
) -> TurnNode:
    return TurnNode(
        node_id=f"q:s{index}:turn:0",
        question_id="q",
        session_id=f"s{index}",
        session_date=f"2026-01-{index + 1:02d}",
        turn_index=0,
        speaker=speaker,
        speaker_key=speaker.casefold(),
        listener=listener,
        transport_role="user",
        text=text,
        retrieval_text=f"speaker {speaker} | {text} | listener {listener}",
        embedding=embedding,
    )


def test_reference_candidates_are_bounded_prior_cross_speaker_turns() -> None:
    antecedent = _turn(
        0, "Caroline", "Melanie",
        "I highly recommend Becoming Nicole; it gave me hope.", [1.0, 0.0],
    )
    same_speaker = _turn(
        1, "Melanie", "Caroline", "I read Charlotte's Web as a child.", [0.9, 0.1],
    )
    anchor = _turn(
        2, "Melanie", "Caroline",
        "I have been reading that book you recommended.", [1.0, 0.0],
    )
    payload = reference_candidate_payload([antecedent, same_speaker, anchor])
    assert payload is not None
    assert payload["anchors"][0]["node_id"] == anchor.node_id
    candidate_ids = {
        row["node_id"] for row in payload["anchors"][0]["candidates"]
    }
    assert antecedent.node_id in candidate_ids
    assert same_speaker.node_id not in candidate_ids


def test_reference_edge_parser_validates_direction_speaker_and_confidence() -> None:
    antecedent = _turn(0, "Caroline", "Melanie", "I recommend Northstar.", [1.0])
    anchor = _turn(1, "Melanie", "Caroline", "I read what you recommended.", [1.0])
    payload = {"links": [
        [anchor.node_id, antecedent.node_id, 0.92, "Northstar"],
        [antecedent.node_id, anchor.node_id, 0.99, "wrong direction"],
        [anchor.node_id, antecedent.node_id, 0.50, "low confidence duplicate"],
    ]}
    edges = parse_reference_edges(
        json.dumps(payload), question_id="q", turns=[antecedent, anchor]
    )
    assert len(edges) == 1
    assert edges[0].relation == "refers_to"
    assert edges[0].directed is True
    assert [item.role for item in edges[0].incidences] == ["antecedent", "anaphor"]
    assert "Northstar" in edges[0].retrieval_text


def test_explicit_reference_late_in_long_memory_is_not_prefix_truncated() -> None:
    turns = [
        _turn(0, "Caroline", "Melanie", "I shared several resources.", [1.0, 0.0])
    ]
    for index in range(1, 14):
        turns.append(_turn(
            index, "Melanie", "Caroline", "Your advice was useful.", [0.7, 0.3]
        ))
    strong = _turn(
        14, "Melanie", "Caroline",
        "I finished that guide you recommended.", [1.0, 0.0],
    )
    turns.append(strong)
    payload = reference_candidate_payload(turns, max_anchors=3)
    assert payload is not None
    assert strong.node_id in {row["node_id"] for row in payload["anchors"]}


def test_explicit_action_reference_rejects_topical_but_unrelated_antecedent() -> None:
    unrelated = _turn(
        0, "Caroline", "Melanie", "I hope your pottery break gets easier.", [1.0]
    )
    anchor = _turn(
        1, "Melanie", "Caroline", "I am reading that book you recommended.", [1.0]
    )
    edges = parse_reference_edges(
        json.dumps({"links": [[anchor.node_id, unrelated.node_id, 0.99, "pottery"]]}),
        question_id="q",
        turns=[unrelated, anchor],
    )
    assert edges == []


def test_named_value_cue_does_not_treat_contractions_as_quotes() -> None:
    assert _NAMED_VALUE_CUE.search('I recommend "The Long Way" by N. Author')
    assert not _NAMED_VALUE_CUE.search(
        "I'm ready to share my advice. It's a difficult process."
    )


def _event(
    index: int, session_id: str, label: str, status: str, types: list[str]
) -> EventNode:
    return EventNode(
        node_id=f"q:{session_id}:event:{index}",
        question_id="q",
        session_id=session_id,
        label=label,
        label_key=label.casefold(),
        status=status,  # type: ignore[arg-type]
        participant_keys=["melanie"],
        event_time=None,
        source_turn_ids=[f"q:{session_id}:turn:0"],
        semantic_type_keys=types,
        retrieval_text=label,
        embedding=[1.0, 0.0],
    )


def test_event_identity_candidates_are_cross_session_and_bounded() -> None:
    turns = [
        _turn(0, "Melanie", "Caroline", "I plan to adopt a child.", [1.0, 0.0]),
        _turn(1, "Melanie", "Caroline", "The adoption was finalized.", [1.0, 0.0]),
    ]
    events = [
        _event(0, "s0", "Melanie plans an adoption", "planned", ["family event", "adoption"]),
        _event(1, "s1", "Melanie completed the adoption", "complete", ["family event", "adoption"]),
    ]
    rows = event_identity_candidate_rows(events, turns, max_pairs=1)
    assert len(rows) == 1
    assert rows[0]["earlier_event_id"] == events[0].node_id
    assert rows[0]["later_event_id"] == events[1].node_id


def test_event_identity_parser_only_accepts_proposed_chronological_pair() -> None:
    events = [
        _event(0, "s0", "Melanie plans an adoption", "planned", ["adoption"]),
        _event(1, "s1", "Melanie completed the adoption", "complete", ["adoption"]),
        _event(2, "s2", "Melanie discusses adoption policy", "asserted", ["policy"]),
    ]
    candidates = [{
        "earlier_event_id": events[0].node_id,
        "later_event_id": events[1].node_id,
    }]
    output = {"event_links": [
        [events[0].node_id, events[1].node_id, 0.91, "same adoption lifecycle"],
        [events[0].node_id, events[2].node_id, 0.99, "topical only"],
        [events[1].node_id, events[0].node_id, 0.99, "backwards"],
    ]}
    edges = parse_event_identity_edges(
        json.dumps(output), question_id="q", events=events,
        candidate_pairs=candidates,
    )
    assert len(edges) == 1
    assert edges[0].relation == "same_event"
    assert [row.role for row in edges[0].incidences] == [
        "earlier_planned", "later_complete"
    ]


def test_event_identity_rejects_recurring_social_thread() -> None:
    events = [
        _event(0, "s0", "Melanie support exchange", "asserted", ["support"]),
        _event(1, "s1", "Melanie encouragement exchange", "complete", ["support"]),
    ]
    candidates = [{
        "earlier_event_id": events[0].node_id,
        "later_event_id": events[1].node_id,
    }]
    edges = parse_event_identity_edges(
        json.dumps({"event_links": [[
            events[0].node_id, events[1].node_id, 0.99,
            "same recurring support thread",
        ]]}),
        question_id="q", events=events, candidate_pairs=candidates,
    )
    assert edges == []
