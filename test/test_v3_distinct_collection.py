from __future__ import annotations

from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.distinct_collection import distinct_action_collection_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _operand(
    suffix: str,
    predicate: str,
    value: str,
    source: str,
    *,
    frame: str | None = None,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=f"q:operand:{suffix}", question_id="q",
        subject_key="participant 1", predicate_key=predicate,
        object_key=value.casefold(), object_text=value,
        event_frame_id=frame, modality="asserted",
        source_turn_ids=[source],
    )


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id, question_id="q", session_id=node_id,
        session_date="2024-01-01", turn_index=0, speaker="participant_1",
        speaker_key="participant 1", listener="", transport_role="user",
        text=text, retrieval_text=text,
    )


def test_distinct_collection_unions_actions_and_deduplicates_entities() -> None:
    frame = build_query_frame(
        "How many pieces of equipment did I buy, assemble, sell, or repair?"
    )
    rows = [
        _operand("a", "got", "a new workbench", "turn:a", frame="frame:a"),
        _operand("a2", "bought", "my workbench from Acme", "turn:a2"),
        _operand("b", "assembled", "that storage cabinet", "turn:b"),
        _operand("c", "fixed", "the drill press", "turn:c"),
        _operand(
            "noise", "bought",
            "guards to protect the equipment", "turn:noise", frame="frame:noise",
        ),
    ]
    scores = {
        "q:operand:a": 0.77, "q:operand:a2": 0.72,
        "q:operand:b": 0.70, "q:operand:c": 0.68,
        "q:operand:noise": 0.62, "frame:a": 0.75, "frame:noise": 0.55,
    }
    hint = distinct_action_collection_hint(
        frame, rows,
        [
            EventFrameV3(
                frame_id="frame:a", question_id="q", label="shop equipment",
                label_key="shop equipment",
            ),
            EventFrameV3(
                frame_id="frame:noise", question_id="q", label="protective supplies",
                label_key="protective supplies",
            ),
        ],
        [
            _turn("turn:a", "I got a new workbench."),
            _turn("turn:a2", "I bought my workbench from Acme."),
            _turn("turn:b", "I assembled that storage cabinet."),
            _turn("turn:c", "I fixed the drill press."),
            _turn("turn:noise", "I bought guards to protect the equipment."),
        ],
        target_semantic_similarity=lambda node: scores.get(node.node_id, 0.0),
    )
    assert hint is not None
    assert hint["value"] == 3
    assert {row["object"] for row in hint["items"]} == {
        "a new workbench", "that storage cabinet", "the drill press",
    }


def test_distinct_collection_keeps_two_entities_from_one_source_turn() -> None:
    frame = build_query_frame("How many devices did I buy?")
    rows = [
        _operand("a", "got", "a camera", "turn:both"),
        _operand("b", "got", "a microphone", "turn:both"),
    ]
    hint = distinct_action_collection_hint(
        frame, rows, [], [_turn("turn:both", "I got a camera and a microphone.")],
        target_semantic_similarity=lambda _node: 0.8,
    )
    assert hint is not None
    assert hint["value"] == 2


def test_distinct_collection_rejects_shared_modifier_with_different_head() -> None:
    frame = build_query_frame("How many glazed bowls did I buy?")
    hint = distinct_action_collection_hint(
        frame,
        [_operand("a", "bought", "six glazed plates", "turn:a")],
        [],
        [_turn("turn:a", "I bought six glazed plates.")],
        target_semantic_similarity=lambda _node: 0.92,
    )
    assert hint is None
