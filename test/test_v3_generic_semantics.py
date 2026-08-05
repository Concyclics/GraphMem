from __future__ import annotations

import json

from graphmem_demo.v3.build import calibrate_event_status, parse_session_extraction
from graphmem_demo.v3.catalog import ensure_catalog
from graphmem_demo.v3.catalog_duration import duration_from_operands
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import ClaimNode, TurnNode, V3Index


def _turn() -> TurnNode:
    return TurnNode(
        "q:s:turn:0", "q", "s", "2026-02-01", 0,
        "Rin", "rin", "Sol", "user",
        "The kiln reached 900 degrees.", "Rin kiln reached 900 degrees",
    )


def test_object_shaped_extraction_is_normalized_and_grounded() -> None:
    turn = _turn()
    payload = {
        "facts": [{
            "subject": "Rin",
            "predicate": "reached",
            "value": "900 degrees",
            "kind": "quantity",
            "source_turn_ids": [turn.node_id],
            "quantity": 900,
            "unit": "degrees",
        }],
        "events": [{
            "label": "kiln firing",
            "semantic_type_keys": ["industrial_process", "kiln_firing"],
            "participants": ["Rin"],
            "sources": [turn.node_id],
            "claim_indices": [0],
        }],
        "episodes": [],
    }
    claims, events, _episodes, error = parse_session_extraction(
        json.dumps(payload),
        question_id="q",
        session_id="s",
        session_date="2026-02-01",
        turns=[turn],
    )
    assert error is None
    assert claims[0].quantity == 900
    assert claims[0].source_turn_ids == [turn.node_id]
    assert events[0].claim_ids == [claims[0].node_id]
    assert events[0].semantic_type_keys == ["industrial process", "kiln firing"]
    index = ensure_catalog(V3Index(turns=[turn], claims=claims, events=events))
    assert index.event_frames[0].semantic_type_keys == [
        "industrial process", "kiln firing",
    ]
    assert index.operands[0].event_type_keys == [
        "industrial process", "kiln firing",
    ]


def test_non_atomic_said_claim_does_not_enter_operand_catalog() -> None:
    turn = _turn()
    claim = ClaimNode(
        node_id="q:s:claim:0",
        question_id="q",
        session_id="s",
        subject="Sol",
        subject_key="sol",
        predicate="said",
        predicate_key="said",
        object="A long generic explanation.",
        object_key="long generic explanation",
        source_turn_ids=[turn.node_id],
        retrieval_text="Sol said a long generic explanation",
    )
    index = ensure_catalog(V3Index(turns=[turn], claims=[claim]))
    assert index.operands == []


def test_query_semantics_distinguish_order_quantity_and_duration() -> None:
    assert build_query_frame(
        "What is the order of the three repairs, from earliest to latest?"
    ).requested_operation == "ordering"
    assert build_query_frame(
        "Which subscription did I start using most recently?"
    ).requested_operation == "latest"
    quantity = build_query_frame(
        "How many hours of glazing did I do last week?"
    )
    assert (quantity.requested_operation, quantity.answer_form) == ("count", "number")
    assert build_query_frame(
        "How many days did it take for the parcel to arrive?"
    ).requested_operation == "duration"
    current_place = build_query_frame("Where is the current kiln log stored?")
    assert (current_place.requested_operation, current_place.answer_form) == (
        "latest", "entity",
    )
    assert build_query_frame(
        "How often does the pump run?"
    ).requested_operation == "recurrence"


def test_duration_uses_two_typed_event_dates_not_session_dates() -> None:
    frame = build_query_frame(
        "How many days did it take for the parcel to arrive after I bought it?"
    )
    operands = [
        OperandRecordV3(
            "q:operand:0", "q", "rin", "bought", "parcel", "parcel",
            event_time="2026-01-15", source_claim_ids=["c0"],
            source_turn_ids=["t0"], retrieval_text="Rin bought parcel 2026-01-15",
        ),
        OperandRecordV3(
            "q:operand:1", "q", "rin", "arrived", "parcel", "parcel",
            event_time="2026-01-20", source_claim_ids=["c1"],
            source_turn_ids=["t1"], retrieval_text="parcel arrived 2026-01-20",
        ),
    ]
    hint = duration_from_operands(
        frame,
        operands,
        lambda query, text: len(set(query.content_terms) & set(text.lower().split()))
        / max(1, len(query.content_terms)),
    )
    assert hint is not None
    assert hint["elapsed_days"] == 5
    assert hint["inclusive_days"] == 6
    assert hint["complete"] is True


def test_event_status_calibration_resolves_explicit_progress_conflict() -> None:
    assert calibrate_event_status(
        "complete", "I enjoy the robotics project and can see it come together."
    ) == "asserted"
    assert calibrate_event_status(
        "asserted", "I finally wrapped the robotics project up last month."
    ) == "complete"
    assert calibrate_event_status(
        "complete", "I attended the conference last week."
    ) == "complete"
