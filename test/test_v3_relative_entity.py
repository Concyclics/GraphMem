from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.relative_entity import resolve_relative_entity
from graphmem_demo.v3.retrieval import build_query_frame


def test_relative_event_frame_resolves_to_exact_object_operand() -> None:
    frame = EventFrameV3(
        frame_id="f", question_id="q", label="participant 1 got a new smoker",
        label_key="participant 1 got new smoker",
        participant_keys=["participant 1"], event_time="2023-03-15",
    )
    smoker = OperandRecordV3(
        operand_id="o", question_id="q", subject_key="participant 1",
        predicate_key="has", object_key="new smoker", object_text="new smoker",
        event_frame_id="f", polarity="positive", modality="asserted",
        source_turn_ids=["t"], confidence=0.9,
    )
    result = resolve_relative_entity(
        build_query_frame("What kitchen appliance did I buy 10 days ago?"),
        {"within_tolerance": True, "supporting_node_ids": ["f"]},
        {"f": frame, "o": smoker},
        [smoker],
        similarity=lambda _item: 0.8,
    )
    assert result == {
        "operation": "relative_time_entity_binding",
        "value": "new smoker",
        "operand_id": "o",
        "supporting_node_ids": ["f", "o"],
        "source_turn_ids": ["t"],
        "relation": "event_frame_member",
        "complete": True,
    }
