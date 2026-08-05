from graphmem_demo.v3.catalog_arithmetic import arithmetic_hint
from graphmem_demo.v3.catalog_schema import EventFrameV3, OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _overlap(frame, text):
    words = set(text.casefold().replace("_", " ").split())
    return len(set(frame.content_terms) & words) / max(1, len(frame.content_terms))


def _operand(
    index, predicate, value, *, quantity=None, unit="", source="t0",
    frame_id=None, sessions=None,
):
    return OperandRecordV3(
        f"q:operand:{index}", "q", "maker", predicate, value, value,
        event_frame_id=frame_id, polarity="positive", modality="asserted",
        quantity=quantity, unit=unit, source_claim_ids=[f"c{index}"],
        source_turn_ids=[source], session_ids=sessions or ["s0"],
        retrieval_text=f"maker {predicate} {value}",
    )


def test_generic_per_item_division_uses_typed_money_and_count() -> None:
    frame = build_query_frame("How much did I spend on each ceramic tile?")
    operands = [
        _operand(0, "spent", "$48 on ceramic tiles", quantity=48, unit="USD"),
        _operand(1, "purchased", "6 ceramic tiles", quantity=6, unit="tiles"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "per_item_amount"
    assert hint["value"] == 8


def test_generic_total_money_deduplicates_repeated_purchase_mentions() -> None:
    frame = build_query_frame("How much total money did I spend on kiln repairs?")
    operands = [
        _operand(0, "paid", "$25 for kiln belt replacement", quantity=25, unit="USD", source="t0"),
        _operand(1, "installed", "kiln sensor costing $40", quantity=40, unit="USD", source="t0"),
        _operand(2, "paid", "$40 for kiln sensor", quantity=40, unit="USD", source="t1"),
        _operand(3, "bought", "kiln shield for $120", quantity=120, unit="USD", source="t2"),
        _operand(4, "bought", "airline ticket for $900", quantity=900, unit="USD", source="t3", sessions=["s3"]),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "total_money"
    assert hint["value"] == 185


def test_generic_occurrence_count_sums_quantities_and_deduplicates_event_summary() -> None:
    frame = build_query_frame("How many times did I test pressure valves across all events?")
    operands = [
        _operand(0, "tested", "pressure valve A", source="t0"),
        _operand(1, "tested", "pressure valve B three times", quantity=3, unit="times", source="t1"),
        _operand(2, "tested", "pressure valves C, D, E", source="t2"),
        _operand(3, "tested", "three pressure valves", source="t2"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "event_occurrence_count"
    assert hint["value"] == 7


def test_event_identity_merges_alias_frames_with_shared_named_participant() -> None:
    frame = build_query_frame("How many ceremony events did I attend?")
    frames = [
        EventFrameV3(
            "f0", "q", "attended colleague ceremony", "a",
            participant_keys=["maker", "ria"], source_turn_ids=["t0"],
        ),
        EventFrameV3(
            "f1", "q", "attended Ria ceremony", "b",
            participant_keys=["maker", "ria"], source_turn_ids=["t1"],
        ),
    ]
    operands = [
        _operand(0, "attended", "colleague ceremony", source="t0", frame_id="f0"),
        _operand(1, "attended", "Ria ceremony", source="t1", frame_id="f1"),
    ]
    hint = arithmetic_hint(frame, operands, frames, _overlap)
    assert hint is not None
    assert hint["value"] == 1

def test_occurrence_operator_uses_latest_explicit_cumulative_total() -> None:
    frame = build_query_frame("How many regional workshops have I tried?")
    older = _operand(0, "tried", "three regional workshops recently", source="t0")
    older.observed_at = "2026-01-01"
    newer = _operand(1, "tried", "four regional workshops so far", source="t1")
    newer.observed_at = "2026-02-01"
    hint = arithmetic_hint(frame, [older, newer], [], _overlap)
    assert hint is not None
    assert hint["operation"] == "event_occurrence_count"
    assert hint["value"] == 4


def test_occurrence_count_collapses_event_relations_and_uses_already_total() -> None:
    frame = build_query_frame("How many times have I met up with Rowan from abroad?")
    frames = [
        EventFrameV3(
            "f0", "q", "met Rowan abroad", "meeting",
            participant_keys=["maker", "rowan"], source_turn_ids=["t0"],
        ),
        EventFrameV3(
            "f1", "q", "friend Rowan from abroad", "meeting",
            participant_keys=["maker", "rowan"], source_turn_ids=["t1"],
        ),
    ]
    first = _operand(40, "met", "Rowan", source="t0", frame_id="f0")
    bonded = _operand(41, "bonded with", "Rowan", source="t0", frame_id="f0")
    identity = _operand(42, "friend", "Rowan", source="t1", frame_id="f1")
    identity.context_key = "abroad"
    total = _operand(
        43, "met up", "Rowan", quantity=2, unit="times",
        source="t1", frame_id="f1",
    )
    planned = _operand(44, "planning to meet up", "Rowan", source="t2")
    turns = [
        type("Turn", (), {"node_id": "t0", "text": "I met Rowan and we bonded."})(),
        type("Turn", (), {
            "node_id": "t1",
            "text": "Rowan is my friend from abroad; we have met up twice already.",
        })(),
        type("Turn", (), {"node_id": "t2", "text": "I am planning to meet Rowan."})(),
    ]
    hint = arithmetic_hint(
        frame, [first, bonded, identity, total, planned], frames, _overlap, turns
    )
    assert hint is not None
    assert hint["operation"] == "event_occurrence_count"
    assert hint["value"] == 2


def test_occurrence_operator_defers_mixed_historical_and_current_set() -> None:
    frame = build_query_frame("How many projects have I led or am currently leading?")
    operands = [
        _operand(0, "led", "archive migration project", source="t0"),
        _operand(1, "currently leading", "sensor calibration project", source="t1"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is None


def test_generic_revenue_total_combines_direct_and_per_item_sales() -> None:
    frame = build_query_frame("What is the total money I earned selling my crafts?")
    operands = [
        _operand(20, "earned", "90 USD from craft sales", quantity=90, unit="USD", source="t20"),
        _operand(21, "sold", "8 carved tokens", quantity=8, unit="tokens", source="t21"),
        _operand(22, "sold price", "5 USD each", quantity=5, unit="USD", source="t21"),
        _operand(23, "paid", "700 USD rent", quantity=700, unit="USD", source="noise"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "aggregate_revenue_total"
    assert hint["value"] == 130


def test_fundraising_total_is_money_flow_not_event_count() -> None:
    frame = build_query_frame(
        "How much money did I raise in total through all the events?"
    )
    operands = [
        _operand(50, "raised", "$250", quantity=250, unit="USD", source="t50"),
        _operand(51, "fundraised", "$5,000", quantity=5000, unit="USD", source="t51"),
        _operand(52, "raised", "$600", quantity=600, unit="USD", source="t52"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "aggregate_revenue_total"
    assert hint["value"] == 5850


def test_money_difference_binds_initial_and_corrected_values() -> None:
    frame = build_query_frame(
        "How much more did I have to pay for the project after the initial quote?"
    )
    initial = _operand(
        60, "quoted", "$2,500 for the project", quantity=2500, unit="USD",
        source="t60",
    )
    initial.context_key = "initial project quote"
    corrected = _operand(
        61, "corrected price", "$2,800 for the project", quantity=2800,
        unit="USD", source="t61",
    )
    corrected.context_key = "updated project price"
    hint = arithmetic_hint(frame, [initial, corrected], [], _overlap)
    assert hint is not None
    assert hint["operation"] == "money_difference"
    assert hint["value"] == 300


def test_money_symbols_are_typed_currency_units() -> None:
    frame = build_query_frame(
        "How much more did I pay after the original repair quote?"
    )
    original = _operand(
        62, "quoted", "original repair quote", quantity=2500, unit="$", source="t62"
    )
    original.context_key = "original repair quote"
    final = _operand(
        63, "final price", "updated repair price", quantity=2800, unit="$", source="t63"
    )
    final.context_key = "updated repair price"
    hint = arithmetic_hint(frame, [original, final], [], _overlap)
    assert hint is not None
    assert hint["operation"] == "money_difference"
    assert hint["value"] == 300


def test_how_much_without_typed_money_never_becomes_occurrence_count() -> None:
    frame = build_query_frame("How much did I spend on the sculpture?")
    hint = arithmetic_hint(
        frame, [_operand(70, "bought", "sculpture", source="t70")], [], _overlap
    )
    assert hint is None


def test_dimensional_total_sums_same_action_mass_operands() -> None:
    frame = build_query_frame("What is the total weight of the new material I purchased?")
    operands = [
        _operand(30, "purchased", "clay batch", quantity=12, unit="pounds", source="t30"),
        _operand(31, "bought", "glaze powder", quantity=3, unit="pounds", source="t31"),
        _operand(32, "shipped", "unrelated parcel", quantity=40, unit="pounds", source="noise"),
    ]
    hint = arithmetic_hint(frame, operands, [], _overlap)
    assert hint is not None
    assert hint["operation"] == "dimensional_quantity_total"
    assert hint["value"] == 15
