from __future__ import annotations

from graphmem_demo.v3.catalog_duration import duration_from_operands
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.semantic_operators import final_choice_hint


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    day: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="james",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        context_key="",
        event_time=day,
        observed_at=day,
        source_turn_ids=[f"{operand_id}:turn"],
        retrieval_text=f"James | {predicate} | {obj}",
    )


def _overlap(frame, text: str) -> float:
    lowered = text.casefold()
    return sum(term in lowered for term in frame.content_terms) / max(
        1, len(frame.content_terms)
    )


def test_duration_does_not_invent_endpoints_for_relation_duration() -> None:
    frame = build_query_frame(
        "How long did Evan and his partner date before getting married?"
    )
    operands = [
        _operand("q:operand:1", "told friends about getting married", "none", "2023-12-30"),
        _operand("q:operand:2", "never had checkout problem", "none", "2023-12-31"),
    ]
    assert duration_from_operands(frame, operands, _overlap) is None


def test_name_choice_binds_the_named_entity_relation() -> None:
    frame = build_query_frame("What is the name of the pup that was adopted by James?")
    adopted = _operand("q:operand:1", "has", "a pup named Ned", "2022-04-12")
    unrelated = _operand("q:operand:2", "dog name", "Max", "2022-06-16")

    result = final_choice_hint(
        frame,
        [adopted, unrelated],
        semantic_similarity=lambda _item: 0.8,
        query_overlap=_overlap,
    )
    assert result is not None
    assert result["value"] == "Ned"
    assert result["source_turn_ids"] == ["q:operand:1:turn"]
