from __future__ import annotations

from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.semantic_operators import scalar_attribute_state_hint


def _operand(
    operand_id: str,
    predicate: str,
    value: str,
    observed_at: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="reactor vessel",
        predicate_key=predicate,
        object_key=value.casefold(),
        object_text=value,
        context_key="cooling loop",
        event_time=None,
        observed_at=observed_at,
        polarity="positive",
        modality="asserted",
        recurrence_days=[],
        confidence=0.95,
        source_turn_ids=[f"{operand_id}:source"],
        retrieval_text=f"reactor vessel | {predicate} | {value} | cooling loop",
    )


def test_scalar_attribute_state_is_domain_generic_and_uses_latest_value() -> None:
    frame = build_query_frame("What was the reactor vessel pressure?")
    old = _operand("q:operand:1", "pressure", "31 psi", "2026-01-02")
    new = _operand("q:operand:2", "pressure reading", "42 psi", "2026-01-09")
    unrelated = _operand(
        "q:operand:3", "coolant temperature", "88 degrees", "2026-01-10"
    )

    def overlap(_frame, text: str) -> float:
        return 2.0 if "pressure" in text else 0.1

    result = scalar_attribute_state_hint(
        frame,
        [old, new, unrelated],
        semantic_similarity=lambda item: (
            0.9 if "pressure" in item.predicate_key else 0.2
        ),
        query_overlap=overlap,
    )

    assert result is not None
    assert result["operation"] == "scalar_attribute_state"
    assert result["value"] == "42 psi"
    assert result["source_turn_ids"] == ["q:operand:2:source"]


def test_scalar_attribute_state_does_not_read_observation_clock_as_value() -> None:
    frame = build_query_frame("What was the reactor vessel pressure?")
    item = _operand("q:operand:1", "pressure", "stable", "2026-01-02 13:45")
    result = scalar_attribute_state_hint(
        frame,
        [item],
        semantic_similarity=lambda _item: 1.0,
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert result is None


def test_event_clock_question_is_not_scalar_attribute_state() -> None:
    frame = build_query_frame("What time did I reach the studio?")
    item = _operand("q:operand:1", "studio hours", "9:00 to 12:00", "2026-01-02")
    result = scalar_attribute_state_hint(
        frame,
        [item],
        semantic_similarity=lambda _item: 1.0,
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert result is None


def test_scalar_attribute_state_respects_explicit_historical_cutoff() -> None:
    frame = build_query_frame("What was the reactor vessel pressure in 2025?")
    item = _operand("q:operand:1", "pressure", "31 psi", "2026-01-02")
    result = scalar_attribute_state_hint(
        frame,
        [item],
        semantic_similarity=lambda _item: 1.0,
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert result is None


def test_scalar_attribute_state_abstains_on_newer_untyped_competing_value() -> None:
    frame = build_query_frame("What was the reactor vessel pressure?")
    old = _operand("q:operand:1", "pressure", "31 psi", "2026-01-02")
    nested = _operand(
        "q:operand:2", "goal", "keep reactor vessel pressure above 42 psi",
        "2026-01-09",
    )
    result = scalar_attribute_state_hint(
        frame,
        [old, nested],
        semantic_similarity=lambda _item: 1.0,
        query_overlap=lambda _frame, text: float("pressure" in text),
    )
    assert result is not None
    assert result["operation"] == "scalar_attribute_state_ambiguous"
    assert result["complete"] is False
    assert result["candidate_values"] == ["31 psi", "42 psi"]
    assert result["source_turn_ids"] == [
        "q:operand:1:source", "q:operand:2:source",
    ]


def test_scalar_attribute_state_resolves_newer_embedded_attribute_reference() -> None:
    frame = build_query_frame("What was the reactor vessel pressure?")
    old = _operand("q:operand:1", "pressure", "31 psi", "2026-01-02")
    nested = _operand(
        "q:operand:2", "maintenance goal",
        "stay below the existing reactor vessel pressure of 42 psi",
        "2026-01-09",
    )
    result = scalar_attribute_state_hint(
        frame,
        [old, nested],
        semantic_similarity=lambda _item: 1.0,
        query_overlap=lambda _frame, text: float("pressure" in text),
    )
    assert result is not None
    assert result["operation"] == "scalar_attribute_state"
    assert result["value"] == "42 psi"
    assert result["source_turn_ids"] == ["q:operand:2:source"]
    assert result["completion_basis"] == "embedded_attribute_scalar_reference"
