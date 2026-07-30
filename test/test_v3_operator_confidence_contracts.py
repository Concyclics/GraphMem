from __future__ import annotations

from graphmem_demo.v3.catalog_duration import duration_from_operands
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import (
    _authoritative_catalog_hint,
    authoritative_catalog_answer,
    build_query_frame,
)
from graphmem_demo.v3.semantic_operators import (
    earliest_alternative_hint,
    scalar_attribute_state_hint,
)


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    observed: str,
    *,
    event_time: str | None = None,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        context_key="",
        event_time=event_time or observed,
        observed_at=observed,
        source_turn_ids=[f"{operand_id}:turn"],
        retrieval_text=f"participant 1 | {predicate} | {obj}",
    )


def _overlap(frame, text: str) -> float:
    lowered = text.casefold()
    return sum(term in lowered for term in frame.content_terms) / max(
        1, len(frame.content_terms)
    )


def test_duration_requires_two_semantically_bound_endpoints() -> None:
    frame = build_query_frame(
        "How many days ago did I harvest my first batch of fresh herbs?"
    )
    harvest = _operand("q:operand:1", "harvested", "fresh herbs", "2023-04-15")
    unrelated = _operand("q:operand:2", "collected", "rare book", "2023-04-09")
    assert duration_from_operands(frame, [harvest, unrelated], _overlap) is None


def test_scalar_value_in_a_purpose_clause_does_not_change_answer_type() -> None:
    frame = build_query_frame(
        "What was the designation on my suit that helped me find the file number?"
    )
    item = _operand("q:operand:1", "found file", "53", "2023-04-15")
    result = scalar_attribute_state_hint(
        frame,
        [item],
        semantic_similarity=lambda _item: 0.9,
        query_overlap=_overlap,
    )
    assert result is None


def test_earliest_alternative_uses_continuous_state_start() -> None:
    frame = build_query_frame(
        "Which event happened first, my attendance at a cultural festival "
        "or the start of my Spanish classes?"
    )
    festival = _operand(
        "q:operand:1", "attended", "cultural festival", "2023-05-27",
        event_time="2023-05-26",
    )
    classes = _operand(
        "q:operand:2", "has been taking", "Spanish classes for past three months",
        "2023-05-27",
    )
    result = earliest_alternative_hint(frame, [festival, classes])
    assert result is not None
    assert result["value"] == "start of my spanish classes"


def test_only_closed_operator_with_packed_provenance_is_authoritative() -> None:
    arithmetic = {
        "operation": "money_difference",
        "complete": True,
        "packed_provenance_complete": True,
        "value": 300,
    }
    assert _authoritative_catalog_hint(arithmetic) == arithmetic

    semantic_lookup = {
        "operation": "relation_slot_location",
        "complete": True,
        "packed_provenance_complete": True,
        "value": "Northport",
    }
    assert _authoritative_catalog_hint(semantic_lookup) is None
    assert _authoritative_catalog_hint({
        **arithmetic, "packed_provenance_complete": False,
    }) is None


def test_authoritative_operator_answer_is_rendered_without_model_vote() -> None:
    trace = {
        "closure_certificate": {
            "complete": True,
            "truncated": False,
            "provenance_complete": True,
            "missing_requirements": [],
        },
        "catalog_operator_hint": {
            "operation": "earliest_named_alternative",
            "complete": True,
            "packed_provenance_complete": True,
            "value": "new router",
            "proofs": [
                {"alternative": "new router", "resolved_date": "2023-01-15"},
                {"alternative": "smart thermostat", "resolved_date": "2023-02-10"},
            ],
        }
    }
    assert authoritative_catalog_answer(trace) == "new router"


def test_non_authoritative_hint_still_requires_evidence_answering() -> None:
    trace = {
        "catalog_operator_hint": {
            "operation": "relation_slot_location",
            "complete": True,
            "packed_provenance_complete": True,
            "value": "Northport",
        }
    }
    assert authoritative_catalog_answer(trace) is None


def test_closed_distinct_action_collection_is_authoritative() -> None:
    trace = {
        "closure_certificate": {
            "complete": True,
            "truncated": False,
            "provenance_complete": True,
            "missing_requirements": [],
        },
        "catalog_operator_hint": {
            "operation": "distinct_action_entity_collection",
            "complete": True,
            "packed_provenance_complete": True,
            "value": 4,
            "items": [{"object": "bench"}, {"object": "cabinet"}],
        }
    }
    assert authoritative_catalog_answer(trace) == "4"


def test_incomplete_graph_closure_cannot_short_circuit_answering() -> None:
    trace = {
        "closure_certificate": {
            "complete": False,
            "truncated": True,
            "provenance_complete": True,
            "missing_requirements": ["collection_scope", "untruncated_scope"],
        },
        "catalog_operator_hint": {
            "operation": "distinct_action_entity_collection",
            "complete": True,
            "packed_provenance_complete": True,
            "value": 12,
            "items": [{"object": "relevant item"}],
        },
    }
    assert authoritative_catalog_answer(trace) is None
