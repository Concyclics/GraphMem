from types import SimpleNamespace

from graphmem_demo.v3.location_state import location_at_time_hint
from graphmem_demo.v3.retrieval import build_query_frame


def _operand(
    node_id: str,
    subject: str,
    predicate: str,
    value: str,
    event_time: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        operand_id=node_id,
        subject_key=subject,
        predicate_key=predicate,
        object_text=value,
        event_time=event_time,
        polarity="positive",
        source_turn_ids=[f"{node_id}:source"],
    )


def test_exact_date_location_uses_latest_prior_movement() -> None:
    frame = build_query_frame("Where was Alex on July 12, 2022?")
    operands = [
        _operand("o1", "alex", "leaves for Toronto", "tomorrow", "2022-07-11"),
        _operand("o2", "alex", "plans to return", "July 20", "2022-07-20"),
        _operand("o3", "alex", "went to Nuuk", "Nuuk", "2022-07-21"),
    ]
    hint = location_at_time_hint(frame, operands)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["value"] == "Toronto"
    assert hint["valid_from"] == "2022-07-11"
    assert hint["valid_to"] == "2022-07-20"


def test_location_state_refuses_target_after_return_boundary() -> None:
    frame = build_query_frame("Where was Alex on July 20, 2022?")
    operands = [
        _operand("o1", "alex", "leaves for Toronto", "tomorrow", "2022-07-11"),
        _operand("o2", "alex", "plans to return", "July 20", "2022-07-20"),
    ]
    assert location_at_time_hint(frame, operands) is None


def test_location_state_requires_requested_subject() -> None:
    frame = build_query_frame("Where was Alex on July 12, 2022?")
    operands = [
        _operand("o1", "morgan", "leaves for Toronto", "tomorrow", "2022-07-11")
    ]
    assert location_at_time_hint(frame, operands) is None
