from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.relation_slot import relation_slot_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _overlap(frame, text):
    query = set(frame.content_terms + frame.participant_terms)
    return len(query & set(text.casefold().split())) / max(1, len(query))


def _turn(node_id, text):
    return TurnNode(
        node_id=node_id, question_id="q", session_id="s", session_date="2026-01-01",
        turn_index=0, speaker="Mira", speaker_key="mira", listener="Rowan",
        transport_role="speaker_a", text=text, retrieval_text=text,
    )


def _operand(index, predicate, value, frame_id, source_id):
    return OperandRecordV3(
        operand_id=f"o{index}", question_id="q", subject_key="mira",
        predicate_key=predicate, object_key=value.casefold(), object_text=value,
        event_frame_id=frame_id, polarity="positive", modality="asserted",
        source_turn_ids=[source_id], retrieval_text=f"Mira {predicate} {value}",
    )


def _frame(index, label, sources):
    return EventFrameV3(
        frame_id=f"f{index}", question_id="q", label=label,
        label_key=label.casefold(), participant_keys=["mira", "rowan"],
        source_turn_ids=sources, retrieval_text=label,
    )


def test_relation_location_follows_exact_event_provenance() -> None:
    frame = build_query_frame(
        "Where did Mira meet Rowan before they began collaborating?"
    )
    turns = [
        _turn("t1", "Mira met Rowan at a symposium in Utrecht and they clicked."),
        _turn("t2", "Mira later performed in Oslo with Rowan."),
    ]
    hint = relation_slot_hint(
        frame,
        [
            _operand(1, "met", "Rowan", "f1", "t1"),
            _operand(2, "performed", "Oslo", "f2", "t2"),
        ],
        [
            _frame(1, "Mira met Rowan at symposium", ["t1"]),
            _frame(2, "Mira performed in Oslo", ["t2"]),
        ],
        turns,
        query_overlap=_overlap,
    )
    assert hint is not None
    assert hint["complete"] is True
    assert hint["value"] == "at a symposium in Utrecht"
    assert hint["operand_ids"] == ["o1"]
    assert hint["event_frame_ids"] == ["f1"]
    assert hint["source_turn_ids"] == ["t1"]


def test_relation_location_skips_temporal_prepositional_phrase() -> None:
    frame = build_query_frame("Where did Mira meet Rowan?")
    hint = relation_slot_hint(
        frame,
        [_operand(1, "met", "Rowan", "f1", "t1")],
        [_frame(1, "Mira met Rowan", ["t1"])],
        [_turn("t1", "Mira met Rowan in August at the harbor.")],
        query_overlap=_overlap,
    )
    assert hint is not None
    assert hint["value"] == "at the harbor"


def test_relation_location_refuses_cross_event_attribute_borrowing() -> None:
    frame = build_query_frame("Where did Mira meet Rowan?")
    assert relation_slot_hint(
        frame,
        [
            _operand(1, "met", "Rowan", "f1", "t1"),
            _operand(2, "performed", "Oslo", "f2", "t2"),
        ],
        [
            _frame(1, "Mira met Rowan", ["t1"]),
            _frame(2, "Mira performed in Oslo", ["t2"]),
        ],
        [
            _turn("t1", "Mira met Rowan and they began collaborating."),
            _turn("t2", "Mira performed in Oslo with Rowan."),
        ],
        query_overlap=_overlap,
    ) is None


def test_relation_slot_does_not_activate_for_non_location_query() -> None:
    frame = build_query_frame("When did Mira meet Rowan?")
    assert relation_slot_hint(
        frame,
        [_operand(1, "met", "Rowan", "f1", "t1")],
        [_frame(1, "Mira met Rowan", ["t1"])],
        [_turn("t1", "Mira met Rowan at the harbor.")],
        query_overlap=_overlap,
    ) is None
