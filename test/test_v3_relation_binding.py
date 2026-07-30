from __future__ import annotations

from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.semantic_operators import (
    final_choice_hint,
    scalar_attribute_state_hint,
)


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    day: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="speaker a",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        context_key="",
        event_time=day,
        observed_at=day,
        source_turn_ids=[f"{operand_id}:source"],
        retrieval_text=f"speaker a | {predicate} | {obj}",
    )


def test_scalar_state_requires_a_scalar_typed_question() -> None:
    frame = build_query_frame("What is Speaker A's identity?")
    item = _operand("q:operand:1", "creating art", "17", "2026-01-02")
    result = scalar_attribute_state_hint(
        frame,
        [item],
        semantic_similarity=lambda _item: 0.9,
        query_overlap=lambda _frame, _text: 0.5,
    )
    assert result is None


def test_final_choice_binds_relation_before_adoption_state() -> None:
    frame = build_query_frame("What career path did Speaker A decide to pursue?")
    correct = _operand(
        "q:operand:1",
        "decided career path",
        "community counseling",
        "2026-01-02",
    )
    unrelated = _operand(
        "q:operand:2",
        "views adoption",
        "a way of giving back",
        "2026-01-09",
    )

    def overlap(_frame, text: str) -> float:
        return 1.0 if "career" in text else 0.0

    result = final_choice_hint(
        frame,
        [correct, unrelated],
        semantic_similarity=lambda _item: 0.8,
        query_overlap=overlap,
    )
    assert result is not None
    assert result["value"] == "community counseling"
    assert result["source_turn_ids"] == ["q:operand:1:source"]
