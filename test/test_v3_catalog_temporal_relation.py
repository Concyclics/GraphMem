from __future__ import annotations

from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.catalog_temporal_relation import relative_operand_hint
from graphmem_demo.v3.retrieval import build_query_frame


def _operand(
    suffix: str,
    *,
    predicate: str,
    value: str,
    when: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=f"q:operand:{suffix}",
        question_id="q",
        subject_key="speaker a",
        predicate_key=predicate,
        object_key=value.casefold(),
        object_text=value,
        event_time=when,
        source_turn_ids=[f"q:turn:{suffix}"],
    )


def test_relative_operand_prefers_grounded_possession_over_unfulfilled_intent() -> None:
    frame = build_query_frame(
        "What new household device did I purchase before getting the countertop oven?"
    )
    prior = _operand(
        "prior",
        predicate="plans use",
        value="my new pressure cooker for dinner",
        when="2024/03/01 (Fri) 09:00",
    )
    distractor = _operand(
        "distractor",
        predicate="thinking getting",
        value="a new smartwatch or high-tech device",
        when="2024/03/01 (Fri) 16:00",
    )
    anchor = _operand(
        "anchor",
        predicate="got",
        value="countertop oven",
        when="2024/03/01 (Fri) 20:00",
    )
    scores = {
        prior.operand_id: 0.54,
        distractor.operand_id: 0.50,
        anchor.operand_id: 0.70,
    }
    hint = relative_operand_hint(
        frame,
        [prior, distractor, anchor],
        semantic_similarity=lambda item: scores[item.operand_id],
    )
    assert hint is not None
    assert hint["answer_operand_id"] == prior.operand_id
    assert hint["anchor_operand_id"] == anchor.operand_id
    assert hint["source_turn_ids"] == ["q:turn:anchor", "q:turn:prior"]


def test_relative_operand_requires_same_first_person_subject() -> None:
    frame = build_query_frame(
        "What new household device did I purchase before getting the countertop oven?"
    )
    prior = _operand(
        "prior",
        predicate="used",
        value="my new pressure cooker",
        when="2024/03/01 (Fri) 09:00",
    )
    other = _operand(
        "other",
        predicate="used",
        value="my new kitchen robot",
        when="2024/03/01 (Fri) 19:00",
    )
    other.subject_key = "speaker b"
    anchor = _operand(
        "anchor",
        predicate="got",
        value="countertop oven",
        when="2024/03/01 (Fri) 20:00",
    )
    scores = {
        prior.operand_id: 0.55,
        other.operand_id: 0.90,
        anchor.operand_id: 0.75,
    }
    hint = relative_operand_hint(
        frame,
        [prior, other, anchor],
        semantic_similarity=lambda item: scores[item.operand_id],
    )
    assert hint is not None
    assert hint["answer_operand_id"] == prior.operand_id
