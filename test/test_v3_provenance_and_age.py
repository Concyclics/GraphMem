from types import SimpleNamespace

from graphmem_demo.v3.catalog_arithmetic import partitioned_scalar_total
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.state_temporal_operators import relative_age_hint


def _numeric(
    index: int,
    value: float,
    object_text: str,
    context: str,
    source: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=f"o{index}",
        question_id="q",
        subject_key="participant 1",
        predicate_key="played",
        object_key=object_text.casefold(),
        object_text=object_text,
        context_key=context,
        polarity="positive",
        modality="asserted",
        quantity=value,
        unit="hours",
        source_turn_ids=[source],
        session_ids=[f"s{index}"],
        retrieval_text=f"I played {object_text} for {value} hours {context}",
    )


def test_total_deduplicates_entity_and_scalar_projection_of_same_mention() -> None:
    frame = build_query_frame("How many hours did I play games in total?")
    operands = [
        _numeric(0, 30, "Adventure Game", "", "turn-a"),
        _numeric(1, 30, "30 hours", "Adventure Game", "turn-a"),
        _numeric(2, 10, "Puzzle Game", "", "turn-b"),
    ]
    turns = [
        SimpleNamespace(session_id="s0", text="I played Adventure Game for 30 hours."),
        SimpleNamespace(session_id="s1", text="I played Adventure Game for 30 hours."),
        SimpleNamespace(session_id="s2", text="I played Puzzle Game for 10 hours."),
    ]
    hint = partitioned_scalar_total(
        frame, operands, query_overlap=lambda _frame, _text: 1.0, turns=turns
    )
    assert hint is not None
    assert hint["value"] == 40
    assert len(hint["operand_ids"]) == 2


def test_how_old_resolves_age_expression_bound_to_event() -> None:
    frame = build_query_frame(
        "How old was I when my aunt gave me the bronze bracelet?"
    )
    node = SimpleNamespace(
        node_id="turn-1",
        text="My aunt gave me the bronze bracelet when I was 19 years old.",
        retrieval_text=(
            "My aunt gave me the bronze bracelet when I was 19 years old."
        ),
        source_turn_ids=[],
    )
    hint = relative_age_hint(
        frame,
        [("turn", node, 1.0, "scope_local_turn_primary")],
        "2026-01-01",
        tokenize=lambda value: value.casefold().split(),
        node_text=lambda value: value.retrieval_text,
    )
    assert frame.answer_form == "number"
    assert hint is not None
    assert hint["operation"] == "event_age_from_evidence_expression"
    assert hint["value"] == 19
    assert hint["source_turn_ids"] == ["turn-1"]
