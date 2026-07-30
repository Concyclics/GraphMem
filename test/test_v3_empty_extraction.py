from __future__ import annotations

import json

from graphmem_demo.v3.build import parse_session_extraction
from graphmem_demo.v3.schema import TurnNode


def test_valid_empty_extraction_keeps_l0_without_inventing_said_claims() -> None:
    turn = TurnNode(
        "q:s:turn:0", "q", "s", "2026-01-01", 0,
        "A", "a", "B", "user", "Hello.", "speaker A Hello.",
    )
    claims, events, episodes, status = parse_session_extraction(
        '{"claims":[],"events":[],"episodes":[]}',
        question_id="q",
        session_id="s",
        session_date="2026-01-01",
        turns=[turn],
    )
    assert claims == []
    assert events == []
    assert episodes == []
    assert status == "no_claims"
    assert turn.text == "Hello."


def test_sparse_valid_extraction_is_augmented_losslessly() -> None:
    turns = [
        TurnNode(
            f"q:s:turn:{index}", "q", "s", "2026-01-01", index,
            "A", "a", "B", "user", f"Durable statement {index}.",
            f"speaker A Durable statement {index}.",
        )
        for index in range(4)
    ]
    payload = (
        "{\"claims\":[[\"A\",\"owns\",\"a pump\",\"state\","
        "\"positive\",\"asserted\",\"assert\",\"equipment\",null,"
        "[\"q:s:turn:0\"]]],\"events\":[],\"episodes\":[]}"
    )
    claims, events, episodes, status = parse_session_extraction(
        payload, question_id="q", session_id="s", session_date="2026-01-01", turns=turns,
    )
    assert len(claims) == 4
    assert {source for claim in claims for source in claim.source_turn_ids} == {
        turn.node_id for turn in turns
    }
    assert claims[0].predicate == "owns"
    assert all(claim.predicate == "said" for claim in claims[1:])
    assert events == [] and episodes == []
    assert status == "undercovered_claims_augmented"


def test_nested_object_extraction_accepts_generic_source_aliases() -> None:
    turn = TurnNode(
        "q:s:turn:0", "q", "s", "2026-01-01", 0,
        "A", "a", "B", "user", "I completed the repair.",
        "speaker A completed the repair.",
    )
    payload = {
        "memory_graph": {
            "atomic_facts": {
                "fact_0": {
                    "subject": "A",
                    "predicate": "completed",
                    "fact": "the repair",
                    "kind": "event",
                    "state_op": "complete",
                    "source_turn_id": "turn:0",
                }
            },
            "event_nodes": {
                "event_0": {
                    "name": "repair completion",
                    "status": "complete",
                    "participants": ["A"],
                    "claims": [0],
                    "evidence_turn_ids": ["0"],
                }
            },
            "episode_nodes": {},
        }
    }
    claims, events, episodes, status = parse_session_extraction(
        json.dumps(payload), question_id="q", session_id="s",
        session_date="2026-01-01", turns=[turn],
    )
    assert status is None
    assert claims[0].predicate == "completed"
    assert claims[0].source_turn_ids == [turn.node_id]
    assert events[0].claim_ids == [claims[0].node_id]
    assert events[0].source_turn_ids == [turn.node_id]
    assert episodes == []
