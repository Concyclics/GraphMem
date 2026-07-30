from __future__ import annotations

import re
from typing import Any

from .schema import QueryFrame


_GENERIC = {
    "all", "and", "both", "did", "do", "does",
    "have", "in", "many", "of", "on", "participate", "the", "to",
}
_PARTICIPATION = {
    "attend", "compete", "compet", "participate", "participat",
    "perform", "win",
}


def _token(value: str) -> str:
    result = value.casefold().strip("'\"")
    irregular = {"won": "win", "went": "go"}
    if result in irregular:
        return irregular[result]
    if result.endswith("'s"):
        result = result[:-2]
    if len(result) > 5 and result.endswith("ing"):
        result = result[:-3]
    elif len(result) > 4 and result.endswith("ed"):
        result = result[:-2]
    if len(result) > 4 and result.endswith("s") and not result.endswith("ss"):
        result = result[:-1]
    return irregular.get(result, result)


def _tokens(value: str) -> set[str]:
    return {
        _token(token)
        for token in re.findall(r"[\w'-]+", value)
        if len(token) > 1
    }


def all_subjects_relation_hint(
    frame: QueryFrame,
    operands: list[Any],
) -> dict[str, Any] | None:
    """Evaluate an explicit ALL/both quantifier over named subjects.

    Missing evidence does not imply false: the operator emits a complete
    boolean only when every named subject has positive typed provenance.
    """

    question = frame.raw_question.casefold()
    if not re.search(r"\b(?:all|both)\b", question):
        return None
    participants = list(dict.fromkeys(
        _token(value) for value in frame.participant_terms
        if _token(value) not in _GENERIC
    ))
    if len(participants) < 2:
        return None
    query_terms = _tokens(question)
    target_terms = query_terms - set(participants) - _GENERIC - _PARTICIPATION
    asks_participation = bool(query_terms & _PARTICIPATION)

    proofs: dict[str, list[Any]] = {participant: [] for participant in participants}
    for item in operands:
        if (
            str(getattr(item, "polarity", "positive")) == "negative"
            or str(getattr(item, "modality", "asserted"))
            in {"planned", "possible", "hypothetical"}
        ):
            continue
        subject_terms = _tokens(str(getattr(item, "subject_key", "")))
        relation_terms = _tokens(
            " ".join((
                str(getattr(item, "predicate_key", "")),
                str(getattr(item, "object_text", "")),
                str(getattr(item, "context_key", "")),
            ))
        )
        if target_terms and not (target_terms & relation_terms):
            continue
        if asks_participation and not (
            relation_terms & _PARTICIPATION
            or {"competition", "event"} & relation_terms
        ):
            continue
        for participant in participants:
            if participant in subject_terms:
                proofs[participant].append(item)

    if any(not rows for rows in proofs.values()):
        return None
    selected = {
        participant: max(
            rows,
            key=lambda item: (
                len(target_terms & _tokens(
                    str(getattr(item, "predicate_key", ""))
                    + " "
                    + str(getattr(item, "object_text", ""))
                )),
                float(getattr(item, "confidence", 0.0)),
                str(getattr(item, "operand_id", "")),
            ),
        )
        for participant, rows in proofs.items()
    }
    operand_ids = [
        str(getattr(item, "operand_id", ""))
        for item in selected.values()
    ]
    source_ids = list(dict.fromkeys(
        source
        for item in selected.values()
        for source in getattr(item, "source_turn_ids", [])
    ))
    return {
        "operation": "all_subjects_relation",
        "value": "yes",
        "subjects": participants,
        "proofs": {
            participant: {
                "predicate": str(getattr(item, "predicate_key", "")),
                "object": str(getattr(item, "object_text", "")),
                "operand_id": str(getattr(item, "operand_id", "")),
                "source_turn_ids": list(getattr(item, "source_turn_ids", [])),
            }
            for participant, item in selected.items()
        },
        "operand_ids": operand_ids,
        "source_turn_ids": source_ids,
        "complete": bool(source_ids),
    }
