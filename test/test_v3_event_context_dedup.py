from graphmem_demo.v3.catalog_arithmetic import arithmetic_hint
from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _overlap(frame, text):
    words = set(text.casefold().replace("_", " ").split())
    return len(set(frame.content_terms) & words) / max(1, len(frame.content_terms))


def _operand(
    index,
    predicate,
    value,
    *,
    source,
    frame_id=None,
    quantity=None,
    unit="",
):
    return OperandRecordV3(
        f"q:operand:{index}",
        "q",
        "person",
        predicate,
        value,
        value,
        event_frame_id=frame_id,
        polarity="positive",
        modality="asserted",
        quantity=quantity,
        unit=unit,
        source_claim_ids=[f"c{index}"],
        source_turn_ids=[source],
        session_ids=[source.split(":")[0]],
        retrieval_text=f"person {predicate} {value}",
    )


def test_count_collapses_direct_and_frame_projection_from_same_turn() -> None:
    frame = build_query_frame("How many times did I inspect turbines?")
    event_frames = [
        EventFrameV3(
            "f0",
            "q",
            "person inspected turbine three times",
            "person inspect turbine three times",
            source_turn_ids=["s0:t0"],
        )
    ]
    operands = [
        _operand(
            0,
            "inspected",
            "turbine",
            source="s0:t0",
            quantity=3,
            unit="times",
        ),
        _operand(
            1,
            "inspected",
            "turbine three times",
            source="s0:t0",
            frame_id="f0",
        ),
    ]
    hint = arithmetic_hint(frame, operands, event_frames, _overlap)
    assert hint is not None
    assert hint["value"] == 3


def test_count_rejects_secondary_fact_from_unrelated_coarse_event() -> None:
    frame = build_query_frame("How many ceremonies did I attend?")
    event_frames = [
        EventFrameV3(
            "f0",
            "q",
            "person recovered a missing keepsake",
            "person recover missing keepsake",
            source_turn_ids=["s0:t0"],
        )
    ]
    operands = [
        _operand(
            0,
            "attended",
            "ceremony",
            source="s0:t0",
            frame_id="f0",
        )
    ]
    assert arithmetic_hint(frame, operands, event_frames, _overlap) is None


def test_money_dedup_uses_co_source_entity_context() -> None:
    frame = build_query_frame("How much total money did I spend on equipment?")
    operands = [
        _operand(
            0,
            "paid for equipment sensor",
            "$40",
            source="s0:t0",
            quantity=40,
            unit="USD",
        ),
        _operand(1, "installed", "equipment sensor", source="s1:t0"),
        _operand(
            2,
            "cost",
            "$40",
            source="s1:t0",
            quantity=40,
            unit="USD",
        ),
        _operand(
            3,
            "paid for equipment motor",
            "$120",
            source="s2:t0",
            quantity=120,
            unit="USD",
        ),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["value"] == 160
