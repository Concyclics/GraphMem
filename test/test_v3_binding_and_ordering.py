from __future__ import annotations

from graphmem_demo.v3.binding_hints import missing_possessive_anchor_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import _node_text, _tokens, build_query_frame
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.semantic_operators import earliest_alternative_hint


def _turn(text: str) -> TurnNode:
    return TurnNode(
        node_id="t",
        question_id="q",
        session_id="s",
        session_date="2026-01-01",
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def _operand(
    operand_id: str, predicate: str, obj: str, day: str
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        event_time=day,
        observed_at=day,
        source_turn_ids=[f"turn:{operand_id}"],
        retrieval_text=f"participant | {predicate} | {obj}",
    )


def test_missing_possessive_anchor_does_not_bind_sibling_relation() -> None:
    frame = build_query_frame("What did I make for my uncle's celebration?")
    hint = missing_possessive_anchor_hint(
        frame,
        [("turn", _turn("I made a cake for my niece."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is not None
    assert hint["anchor"] == "uncle"


def test_present_possessive_anchor_does_not_force_abstention() -> None:
    frame = build_query_frame("What did I make for my uncle's celebration?")
    hint = missing_possessive_anchor_hint(
        frame,
        [("turn", _turn("I made a cake for my uncle."), 1.0, "test")],
        node_text=_node_text,
        tokenize=_tokens,
    )
    assert hint is None


def test_earliest_alternative_binds_only_named_choices() -> None:
    frame = build_query_frame(
        "Which event happened first, the volleyball league or the charity run?"
    )
    hint = earliest_alternative_hint(
        frame,
        [
            _operand("noise", "attended", "book reading", "2026-01-01"),
            _operand("volleyball", "joined volleyball league", "two months ago", "2026-02-01"),
            _operand("charity", "completed", "charity run", "2026-03-01"),
        ],
    )
    assert hint is not None
    assert hint["value"] == "volleyball league"


def test_earliest_named_people_require_relation_evidence_for_each_person() -> None:
    frame = build_query_frame("Who became a mentor first, Rowan or Casey?")
    hint = earliest_alternative_hint(
        frame,
        [
            OperandRecordV3(
                operand_id="rowan-noise", question_id="q",
                subject_key="rowan", predicate_key="visited",
                object_key="museum", object_text="museum",
                event_time="2026-01-01", observed_at="2026-01-01",
                source_turn_ids=["turn:rowan-noise"],
                retrieval_text="Rowan visited a museum",
            ),
            OperandRecordV3(
                operand_id="casey-mentor", question_id="q",
                subject_key="casey", predicate_key="became mentor",
                object_key="mentor", object_text="mentor",
                event_time="2026-02-01", observed_at="2026-02-01",
                source_turn_ids=["turn:casey-mentor"],
                retrieval_text="Casey became a mentor",
            ),
        ],
    )
    assert hint is not None
    assert hint["operation"] == "named_alternative_incomplete"
    assert hint["missing_alternatives"] == ["rowan"]
