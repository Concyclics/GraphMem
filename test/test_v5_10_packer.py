from __future__ import annotations

import hashlib

from graphmem.domain import (
    CandidateScore,
    EvidenceMember,
    EvidenceUnit,
    FactBinding,
    ProofObligation,
    SourceTurn,
)
from graphmem.retrieval.packer import (
    adaptive_evidence_turn_limit,
    build_proof_units,
    pack_obligation_aware,
    salient_spans,
)
from graphmem.retrieval.navigator import GraphNavigator
from graphmem.config import QueryBudget


def _turn(turn_id: str, session: str, text: str) -> SourceTurn:
    return SourceTurn(
        turn_id, "m", session, 0, "Alice", "Bob", "user", "2025-01-02",
        text, hashlib.sha256(text.encode()).hexdigest())


def _candidate(turn: SourceTurn, score: float, operands=()) -> CandidateScore:
    return CandidateScore(
        turn.turn_id, turn.session_id, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, len(turn.raw_text.split()), score, (),
        operand_ids=tuple(operands))


def _binding(binding_id: str = "b1") -> FactBinding:
    return FactBinding(
        binding_id=binding_id, operand_id="op1", fact_node_id="fact1",
        owner_id="alice", predicate="bought", scope="shopping", value_key="camera",
        event_instance_id=None, time_interval=None, evidence_group_ids=("g1",),
        confidence=0.95, value="camera", session_id="s1", turn_index=0)


def test_proof_units_carry_real_obligations_and_exact_fact_spans() -> None:
    span = EvidenceMember("t1", 12, 29, "fact_quote")
    obligations = (
        ProofObligation("need-binding", "op1", "binding"),
        ProofObligation("need-provenance", "op1", "provenance"),
        ProofObligation("need-order", None, "ordering"),
    )

    units = build_proof_units(
        (_binding(),), {"g1": ("t1",)}, obligations=obligations,
        group_members={"g1": (EvidenceMember("t1", 0, 50, "source"),)},
        fact_spans={"fact1": (span,)})

    assert len(units) == 1
    assert units[0].obligation_ids == (
        "need-binding", "need-provenance", "need-order")
    assert units[0].operand_ids == ("op1",)
    assert units[0].spans == (span,)


def test_raw_fallback_reserve_is_bounded_and_uses_upstream_rank() -> None:
    turns = {
        f"t{index}": _turn(f"t{index}", f"s{index % 2}", f"evidence {index}")
        for index in range(8)
    }
    rows = tuple(
        _candidate(turns[f"t{index}"], 10.0 - index)
        for index in range(8)
    )
    budget = QueryBudget(max_evidence_turns=4, max_evidence_tokens=100)

    packed, _dropped, _coverage = GraphNavigator._rank_pack(
        rows, turns, budget, reserved_turn_ids={"t5", "t6", "t7"},
        reserve_limit=2)

    assert packed == ("t5", "t6", "t0", "t1")

def test_salient_fallback_keeps_numeric_temporal_and_negative_sentences() -> None:
    turn = _turn(
        "t1", "s1",
        "Unrelated opening chatter. I did not buy it on Monday. "
        "The trip lasted 14 days and ended yesterday. More unrelated filler.")

    spans = salient_spans(
        turn, "How many days did the trip last?", answer_kind="duration")
    text = " ".join(turn.raw_text[span.span_start:span.span_end] for span in spans)

    assert "14 days" in text and "yesterday" in text
    assert "did not buy" in text
    assert all(span.support_type == "salient_fallback" for span in spans)


def test_obligation_packer_preserves_operand_and_session_floors_with_fewer_tokens() -> None:
    left = _turn(
        "left", "s1",
        "filler " * 80 + "Alice bought 3 cameras on Monday. " + "tail " * 80)
    right = _turn(
        "right", "s2",
        "opening " * 80 + "Bob returned 2 cameras on Friday. " + "tail " * 80)
    noise = _turn("noise", "s1", "camera " * 200)
    left_span = EvidenceMember("left", left.raw_text.index("Alice"),
                               left.raw_text.index("Monday") + len("Monday"), "fact_quote")
    right_span = EvidenceMember("right", right.raw_text.index("Bob"),
                                right.raw_text.index("Friday") + len("Friday"), "fact_quote")
    units = (
        EvidenceUnit("u1", ("o1",), ("b1",), ("left",), (), 0, True,
                     ("op1",), "alice-camera", (left_span,)),
        EvidenceUnit("u2", ("o2",), ("b2",), ("right",), (), 0, True,
                     ("op2",), "bob-camera", (right_span,)),
    )
    turns = {turn.turn_id: turn for turn in (left, right, noise)}
    candidates = (
        _candidate(noise, 9.0), _candidate(left, 2.0, ("op1",)),
        _candidate(right, 1.5, ("op2",)),
    )

    packed, dropped, flags, packed_units, used = pack_obligation_aware(
        units, candidates, turns, query="How many cameras did Alice and Bob handle?",
        answer_kind="count", max_turns=2, max_tokens=80,
        count_text_tokens=lambda text: len(text.split()), span_window=8)

    assert set(packed) == {"left", "right"}
    assert "noise" in dropped
    assert not flags["obligation_incomplete"] and not flags["operand_incomplete"]
    assert used < len(left.raw_text.split()) + len(right.raw_text.split())
    assert {unit.unit_id for unit in packed_units} >= {"u1", "u2"}


def test_atomic_multi_turn_unit_is_never_falsely_certified_when_only_one_turn_fits() -> None:
    first = _turn("t1", "s1", "Alice gave Bob the key on Monday.")
    second = _turn("t2", "s2", "Bob confirmed receiving the key on Tuesday.")
    unit = EvidenceUnit(
        "atomic", ("two-sided",), ("b1",), ("t1", "t2"), (), 0, True,
        ("op",), "key-transfer", (), True)
    candidates = (_candidate(first, 2.0, ("op",)), _candidate(second, 1.0, ("op",)))

    packed, _dropped, flags, audit_units, _used = pack_obligation_aware(
        (unit,), candidates, {"t1": first, "t2": second}, query="Who received the key?",
        answer_kind="lookup", max_turns=1, max_tokens=100,
        count_text_tokens=lambda text: len(text.split()), span_window=0)

    assert len(packed) == 1
    assert flags["atomic_unit_dropped"] and flags["obligation_incomplete"]
    audited = next(row for row in audit_units if row.unit_id == "atomic")
    assert audited.source_turn_ids == ("t1", "t2")


def test_obligation_packing_is_deterministic() -> None:
    turns = {
        f"t{index}": _turn(f"t{index}", f"s{index % 2}", f"fact {index} happened on Monday")
        for index in range(8)
    }
    units = tuple(EvidenceUnit(
        f"u{index}", (f"o{index % 2}",), (f"b{index}",), (f"t{index}",), (),
        0, True, (f"op{index % 2}",), f"m{index}") for index in range(8))
    candidates = tuple(_candidate(turn, float(index), (f"op{index % 2}",))
                       for index, turn in enumerate(turns.values()))
    kwargs = dict(query="What happened on Monday?", answer_kind="temporal",
                  max_turns=4, max_tokens=80,
                  count_text_tokens=lambda text: len(text.split()), span_window=0)

    results = {
        pack_obligation_aware(units, candidates, turns, **kwargs)[0]
        for _ in range(20)
    }

    assert len(results) == 1


def test_span_pack_is_a_monotone_extension_of_full_turn_floor() -> None:
    turns = {
        f"t{index}": _turn(
            f"t{index}", "s1",
            (("long filler " * 80) + ". " if index < 4 else "short ")
            + f"gold fact {index} happened Monday")
        for index in range(7)
    }
    candidates = tuple(_candidate(turns[f"t{index}"], 10.0 - index)
                       for index in range(7))
    # A frozen full-turn budget can skip expensive early ranks and reach t6.
    # Span compression must never discard that already-selected evidence while
    # filling the newly released capacity.
    floor = ("t0", "t6")
    packed, _dropped, _flags, _units, _used = pack_obligation_aware(
        (), candidates, turns, query="What gold fact happened Monday?",
        answer_kind="temporal", max_turns=5, max_tokens=200,
        count_text_tokens=lambda text: len(text.split()), span_window=0,
        baseline_floor=floor)

    assert set(floor) <= set(packed)


def test_adaptive_limit_keeps_collections_wide_and_lookups_focused() -> None:
    assert adaptive_evidence_turn_limit("lookup", 1, 32) == 12
    assert adaptive_evidence_turn_limit("lookup", 2, 32) == 16
    assert adaptive_evidence_turn_limit("date_difference", 2, 32) == 24
    assert adaptive_evidence_turn_limit(
        "lookup", 1, 32, query="What did I cook a couple of days ago?") == 24
    assert adaptive_evidence_turn_limit("count", 1, 32) == 24
    assert adaptive_evidence_turn_limit("union_distinct", 4, 32) == 32


def test_precision_aware_pack_can_displace_a_noisy_monotonic_floor() -> None:
    left = _turn("left", "s1", "Alice bought 3 cameras on Monday.")
    right = _turn("right", "s2", "Bob returned 2 cameras on Friday.")
    noise = _turn("noise", "s1", "camera camera camera unrelated chatter")
    units = (
        EvidenceUnit("u1", ("o1",), ("b1",), ("left",), (), 0, True,
                     ("op1",), "alice-camera"),
        EvidenceUnit("u2", ("o2",), ("b2",), ("right",), (), 0, True,
                     ("op2",), "bob-camera"),
    )
    candidates = (
        _candidate(noise, 9.0), _candidate(left, 2.0, ("op1",)),
        _candidate(right, 1.5, ("op2",)),
    )

    packed, _dropped, flags, _audit, _used = pack_obligation_aware(
        units, candidates, {row.turn_id: row for row in (left, right, noise)},
        query="How many cameras did Alice and Bob handle?", answer_kind="count",
        max_turns=2, max_tokens=100,
        count_text_tokens=lambda text: len(text.split()), span_window=0,
        baseline_floor=("noise", "left"), precision_aware=True)

    assert set(packed) == {"left", "right"}
    assert flags["precision_aware"]
