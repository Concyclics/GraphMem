from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re
from typing import Any, Callable

from .alternative_comparison import earliest_alternative_from_sources
from .combined_duration import combined_named_duration_hint
from .catalog_arithmetic import arithmetic_hint
from .clock_arithmetic import arrival_clock_hint
from .catalog_schema import EventFrameV3, OperandRecordV3
from .catalog_relation import latest_relation_hint
from .catalog_temporal_relation import relative_operand_hint
from .catalog_duration import duration_from_operands, duration_from_turns
from .dialogue_answer import dialogue_answer_hint
from .dialogue_followup import dialogue_followup_plan_hint
from .distinct_collection import distinct_action_collection_hint
from .event_onset import event_onset_from_sources
from .exact_entity_absence import exact_entity_absence_hint
from .lossless_cardinality import latest_cardinality_from_turns
from .ordinal_event import ordinal_event_hint
from .ordered_collection import ordered_action_collection_candidates
from .relation_slot import relation_slot_hint
from .scalar_comparison import scalar_comparison_hint
from .scalar_aggregate import named_scalar_average_hint
from .schema import QueryFrame
from .semantic_operators import (
    earliest_alternative_hint, final_choice_hint, frequency_state_comparison_hint,
    ordered_event_hint, scalar_attribute_state_hint,
)


def _date(value: str | None) -> tuple[int, str]:
    if not value:
        return (0, "")
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return (1, datetime.strptime(value.strip(), fmt).isoformat())
        except ValueError:
            pass

    return (1, value)

def _first_person_frequency_object(question: str) -> set[str]:
    match = re.search(
        r"\bhow often\s+(?:do|did)\s+i\s+\w+\s+(.+?)"
        r"(?:\s+(?:with|at|in|on|for|during)\b|[?]|\Z)",
        question.casefold(),
    )
    if not match:
        match = re.search(
            r"\bhow many\s+days?\s+(?:a|per|each)\s+week\s+"
            r"(?:do|did)\s+i\s+\w+\s+(.+?)(?:[?]|\Z)",
            question.casefold(),
        )
    if not match:
        return set()
    ignored = {"a", "an", "the", "my", "our"}
    return {
        term
        for term in re.findall(r"[\w\x27]+", match.group(1))
        if term not in ignored
    }


def _singular_term(term: str) -> str:
    if term.endswith(("sses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def catalog_operator_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    event_frames: list[EventFrameV3] | None = None,
    query_overlap: Callable[[QueryFrame, str], float],
    semantic_similarity: Callable[[OperandRecordV3], float] | None = None,
    object_semantic_similarity: Callable[[OperandRecordV3], float] | None = None,
    target_semantic_similarity: Callable[[Any], float] | None = None,
    turns: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Compute only type-level operations that are valid across memory datasets."""
    dialogue_answer = dialogue_answer_hint(frame, turns or [])
    if dialogue_answer is not None:
        return dialogue_answer

    exact_absence = exact_entity_absence_hint(frame, turns or [])
    if exact_absence is not None:
        return exact_absence

    arrival_clock = arrival_clock_hint(frame, turns or [])
    if arrival_clock is not None:
        return arrival_clock

    followup_plan = dialogue_followup_plan_hint(frame, turns or [])
    if followup_plan is not None:
        return followup_plan

    event_onset = event_onset_from_sources(frame, turns or [])
    if event_onset is not None:
        return event_onset

    combined_duration = combined_named_duration_hint(frame, turns or [])
    if combined_duration is not None:
        return combined_duration

    turn_duration = duration_from_turns(frame, turns or [], query_overlap)
    if turn_duration is not None:
        return turn_duration

    latest_cardinality = latest_cardinality_from_turns(frame, turns or [])
    if latest_cardinality is not None:
        return latest_cardinality

    scalar_average = named_scalar_average_hint(frame, turns or [])
    if scalar_average is not None:
        return scalar_average

    if target_semantic_similarity is not None:
        ordered_candidates = ordered_action_collection_candidates(
            frame, turns or [],
            target_semantic_similarity=target_semantic_similarity,
        )
        if ordered_candidates is not None:
            return ordered_candidates

    if target_semantic_similarity is not None:
        distinct_collection = distinct_action_collection_hint(
            frame, operands, event_frames or [], turns or [],
            target_semantic_similarity=target_semantic_similarity,
        )
        if distinct_collection is not None:
            return distinct_collection

    ranked = sorted(
        (
            (query_overlap(frame, item.retrieval_text), item)
            for item in operands
            if item.polarity != "negative"
        ),
        key=lambda row: (
            row[0],
            row[1].confidence,
            _date(row[1].event_time or row[1].observed_at),
        ),
        reverse=True,
    )
    relevant = [item for score, item in ranked if score > 0]
    if not relevant:
        return None

    ordinal_event = ordinal_event_hint(frame, operands, turns)
    if ordinal_event is not None:
        return ordinal_event

    relation_slot = relation_slot_hint(
        frame,
        operands,
        event_frames or [],
        turns or [],
        query_overlap=query_overlap,
        semantic_similarity=semantic_similarity,
    )
    if relation_slot is not None:
        return relation_slot

    scalar_comparison = scalar_comparison_hint(frame, turns or [])
    if scalar_comparison is not None:
        return scalar_comparison

    if semantic_similarity is not None:
        relative_operand = relative_operand_hint(
            frame, operands, semantic_similarity=semantic_similarity,
            turns=turns or [],
        )
        if relative_operand is not None:
            return relative_operand
        earliest = earliest_alternative_from_sources(
            frame, operands, turns or []
        ) or earliest_alternative_hint(frame, operands)
        if earliest is not None:
            return earliest
        frequency_comparison = frequency_state_comparison_hint(
            frame, operands, query_overlap=query_overlap, turns=turns
        )
        if frequency_comparison is not None:
            return frequency_comparison
        choice = final_choice_hint(
            frame, operands, semantic_similarity=semantic_similarity,
            query_overlap=query_overlap,
        )
        if choice is not None:
            return choice
        ordered = ordered_event_hint(
            frame, operands, semantic_similarity=semantic_similarity,
            object_semantic_similarity=object_semantic_similarity,
            query_overlap=query_overlap,
        )
        if ordered is not None:
            return ordered
        scalar_state = scalar_attribute_state_hint(
            frame,
            operands,
            semantic_similarity=semantic_similarity,
            query_overlap=query_overlap,
        )
        if scalar_state is not None:
            return scalar_state

    arithmetic = arithmetic_hint(
        frame, operands, event_frames or [], query_overlap, turns
    )
    if arithmetic is not None:
        return arithmetic

    relation = latest_relation_hint(
        frame, operands, query_overlap, semantic_similarity
    )
    if relation is not None:
        return relation

    if frame.requested_operation == "duration":
        hint = duration_from_operands(frame, relevant, query_overlap)
        if hint is not None:
            return hint

    frequency_question = bool(re.search(
        r"\b(?:how often|frequency|times?\s+per\s+(?:day|week|month|year)|"
        r"days?\s+(?:a|per|each)\s+week|weekly|monthly|yearly|"
        r"each\s+(?:day|week|month|year)|typical\s+(?:day|week|month|year))\b",
        frame.raw_question.casefold(),
    ))
    weekly_frequency_question = bool(re.search(
        r"\b(?:week|weekly|times?\s+per\s+week|days?\s+(?:a|per|each)\s+week)\b",
        frame.raw_question.casefold(),
    ))
    if (
        frame.requested_operation in {"count", "recurrence"}
        and frequency_question
        and weekly_frequency_question
    ):
        required_object = _first_person_frequency_object(frame.raw_question)
        if required_object:
            candidate_terms = [
                set(re.findall(
                    r"[\w\x27]+",
                    (
                        item.predicate_key + " " + item.object_text + " "
                        + item.context_key
                    ).casefold(),
                ))
                for item in relevant
            ]
            exact_match = any(required_object.issubset(terms) for terms in candidate_terms)
            partial_match = any(required_object & terms for terms in candidate_terms)
            if partial_match and not exact_match:
                return {
                    "operation": "exact_entity_mismatch",
                    "value": "insufficient evidence for the exact requested entity",
                    "required_terms": sorted(required_object),
                    "complete": True,
                    "operand_ids": [],
                    "source_turn_ids": [],
                }
        grouped: dict[tuple[str, str, str], list[OperandRecordV3]] = defaultdict(list)
        for item in relevant:
            item_terms = set(re.findall(
                r"[\w']+",
                (
                    item.predicate_key + " " + item.object_text + " "
                    + item.context_key
                ).casefold(),
            ))
            if item.recurrence_days and (
                not required_object or required_object.issubset(item_terms)
            ):
                grouped[(item.subject_key, item.predicate_key, item.context_key)].append(item)
        if grouped:
            primary_rows = max(
                grouped.values(),
                key=lambda values: (
                    max(query_overlap(frame, item.retrieval_text) for item in values),
                    len({day for item in values for day in item.recurrence_days}),
                ),
            )
            rows = list(primary_rows)
            if required_object:
                normalized_required = {
                    _singular_term(term)
                    for term in required_object
                }
                primary_subjects = {item.subject_key for item in primary_rows}
                recurrence_candidates = [
                    item for item in operands
                    if item.polarity != "negative"
                    and item.modality not in {"planned", "possible", "hypothetical"}
                ]
                recurrence_vocab = [
                    {_singular_term(term) for term in re.findall(
                        r"[\w']+",
                        (item.predicate_key + " " + item.object_text + " " + item.context_key).casefold(),
                    )}
                    for item in recurrence_candidates
                    if item.recurrence_days and item.subject_key in primary_subjects
                ]
                head = max(
                    normalized_required,
                    key=lambda term: (sum(term in vocab for vocab in recurrence_vocab), -len(term), term),
                    default="",
                )
                primary_similarity = max(
                    (semantic_similarity(item) for item in primary_rows),
                    default=0.0,
                ) if semantic_similarity is not None else 0.0
                membership = re.compile(
                    r"\b(?:attend|go|has|have|enroll|practice|start|take|train)\w*\b"
                )
                for item in recurrence_candidates:
                    if item in rows or item.subject_key not in primary_subjects:
                        continue
                    text = (item.predicate_key + " " + item.object_text + " " + item.context_key).casefold()
                    normalized_terms = {
                        _singular_term(term)
                        for term in re.findall(r"[\w']+", text)
                    }
                    if not item.recurrence_days or head not in normalized_terms or not membership.search(text):
                        continue
                    modifier_match = bool((normalized_required - {head}) & normalized_terms)
                    semantic_match = (
                        semantic_similarity is not None
                        and semantic_similarity(item) >= max(0.35, primary_similarity - 0.18)
                    )
                    if modifier_match or semantic_match:
                        rows.append(item)
            days = list(dict.fromkeys(
                day for item in rows for day in item.recurrence_days
            ))
            return {
                "operation": "weekly_recurrence_count",
                "value": len(days),
                "recurrence_days": days,
                "operand_ids": [item.operand_id for item in rows],
                "source_turn_ids": list(dict.fromkeys(
                    value for item in rows for value in item.source_turn_ids
                )),
                "complete": True,
                "completion_basis": "subject_entity_recurrence_union",
            }

    if frame.requested_operation in {"latest", "earliest", "state"}:
        timed = [
            item for item in relevant
            if item.event_time or item.observed_at
        ]
        if timed:
            reverse = frame.requested_operation != "earliest"
            item = sorted(
                timed,
                key=lambda row: _date(row.event_time or row.observed_at),
                reverse=reverse,
            )[0]
            return {
                "operation": f"catalog_{frame.requested_operation}",
                "value": item.object_text,
                "time": item.event_time or item.observed_at,
                "operand_ids": [item.operand_id],
                "source_turn_ids": item.source_turn_ids,
                "complete": False,
            }

    return {
        "operation": "catalog_candidates",
        "items": [
            {
                "operand_id": item.operand_id,
                "subject": item.subject_key,
                "predicate": item.predicate_key,
                "object": item.object_text,
                "time": item.event_time or item.observed_at,
                "polarity": item.polarity,
                "modality": item.modality,
                "source_turn_ids": item.source_turn_ids,
            }
            for item in relevant[:8]
        ],
        "complete": False,
    }
