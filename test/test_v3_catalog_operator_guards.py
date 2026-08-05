from graphmem_demo.v3.catalog_operators import catalog_operator_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _overlap(frame, text):
    lowered = text.casefold()
    return sum(term in lowered for term in frame.content_terms) / max(1, len(frame.content_terms))
    return len(set(frame.content_terms) & words) / max(1, len(frame.content_terms))


def _operand():
    return OperandRecordV3(
        "q:operand:0", "q", "a", "operated", "pump", "pump",
        recurrence_days=["saturday"], source_claim_ids=["c0"],
        source_turn_ids=["t0"], retrieval_text="A operated pump saturday",
    )


def test_recurrence_operator_requires_frequency_semantics() -> None:
    frame = build_query_frame("How many hours did I operate the pump last week?")
    hint = catalog_operator_hint(frame, [_operand()], query_overlap=_overlap)
    assert hint is not None
    assert hint["operation"] != "weekly_recurrence_count"


def test_recurrence_operator_handles_generic_frequency_question() -> None:
    frame = build_query_frame("How often do I operate the pump in a typical week?")
    hint = catalog_operator_hint(frame, [_operand()], query_overlap=_overlap)
    assert hint is not None
    assert hint["operation"] == "weekly_recurrence_count"
    assert hint["value"] == 1


def test_recurrence_operator_requires_explicit_week_granularity() -> None:
    frame = build_query_frame("How often does Alex inspect the pump?")
    hint = catalog_operator_hint(frame, [_operand()], query_overlap=_overlap)
    assert hint is not None
    assert hint["operation"] != "weekly_recurrence_count"


def test_recurrence_operator_requires_full_compound_entity() -> None:
    frame = build_query_frame("How often do I play table tennis with my friends in a typical week?")
    operand = OperandRecordV3(
        "q:operand:1", "q", "a", "play", "tennis", "tennis",
        recurrence_days=["sunday"], source_claim_ids=["c1"],
        source_turn_ids=["t1"], retrieval_text="A played tennis with friends on Sunday",
    )
    hint = catalog_operator_hint(frame, [operand], query_overlap=_overlap)
    assert hint is not None
    assert hint["operation"] == "exact_entity_mismatch"
    assert hint["complete"] is True
