from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_IRREGULAR = {
    "won": "win",
    "tourney": "tournament",
    "tourneys": "tournament",
}
_FUNCTION = {
    "a", "an", "at", "based", "did", "do", "does", "game", "in", "is",
    "my", "of", "on", "the", "this", "to", "was", "what", "when", "which",
}


def _tokens(value: str) -> set[str]:
    rows = set()
    for raw in re.findall(r"[\w']+", value.casefold()):
        token = _IRREGULAR.get(raw, raw)
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        rows.add(_IRREGULAR.get(token, token))
    return rows


def _date(value: str | None) -> datetime:
    text = value or ""
    match = re.search(
        r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text
    )
    if match:
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            )
        except ValueError:
            pass
    return datetime.min


def _subject_matches(frame: QueryFrame, item: OperandRecordV3) -> bool:
    named = {value.casefold() for value in frame.participant_terms}
    subject = _tokens(item.subject_key)
    if named:
        return bool(named & subject)
    if re.search(r"\b(?:i|me|my)\b", frame.raw_question.casefold()):
        return "participant" not in subject or "2" not in subject
    return True


def _dialogue_window(
    session_id: str, source_ids: list[str], turns: list[Any] | None
) -> list[Any]:
    rows = [
        turn for turn in (turns or [])
        if str(getattr(turn, "session_id", "")) == session_id
    ]
    if not rows:
        by_id = {getattr(turn, "node_id", ""): turn for turn in (turns or [])}
        return [by_id[value] for value in source_ids if value in by_id]
    rows.sort(key=lambda turn: int(getattr(turn, "turn_index", 0)))
    positions = [
        index for index, turn in enumerate(rows)
        if getattr(turn, "node_id", "") in source_ids
    ]
    if not positions:
        return []
    return rows[max(0, min(positions) - 2):min(len(rows), max(positions) + 5)]


def ordinal_event_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    turns: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve an ordinal completed event, then bind its local attribute/date."""
    question = frame.raw_question.casefold()
    ordinal_match = re.search(
        r"\b(" + "|".join(_ORDINALS) + r"|[1-9](?:st|nd|rd|th))\b",
        question,
    )
    if ordinal_match is None:
        return None
    raw_ordinal = ordinal_match.group(1)
    ordinal = (
        _ORDINALS[raw_ordinal]
        if raw_ordinal in _ORDINALS
        else int(re.match(r"\d+", raw_ordinal).group(0))  # type: ignore[union-attr]
    )

    query_terms = _tokens(question)
    relation_terms = query_terms & {
        "attend", "complete", "finish", "participate", "run", "visit",
        "volunteer", "win",
    }
    event_terms = query_terms - relation_terms - set(_ORDINALS) - _FUNCTION
    if not relation_terms or not event_terms:
        return None

    candidates: list[OperandRecordV3] = []
    for item in operands:
        if (
            item.polarity == "negative"
            or item.modality in {"planned", "possible", "hypothetical"}
            or not _subject_matches(frame, item)
        ):
            continue
        terms = _tokens(
            f"{item.predicate_key} {item.object_text} {item.context_key}"
        )
        if not (relation_terms & terms) or not (event_terms & terms):
            continue
        candidates.append(item)
    if not candidates:
        return None

    by_session: dict[str, list[OperandRecordV3]] = {}
    for item in candidates:
        for session_id in item.session_ids or [""]:
            by_session.setdefault(session_id, []).append(item)
    occurrences: list[tuple[datetime, str, OperandRecordV3]] = []
    for session_id, rows in by_session.items():
        best = max(
            rows,
            key=lambda item: (
                len(_tokens(item.object_text)),
                len(event_terms & _tokens(item.object_text)),
                item.confidence,
                item.operand_id,
            ),
        )
        occurrences.append(
            (_date(best.observed_at or best.event_time), session_id, best)
        )
    occurrences.sort(key=lambda row: (row[0], row[1], row[2].operand_id))
    if ordinal > len(occurrences):
        return None
    _observed, session_id, selected = occurrences[ordinal - 1]

    local = [
        item
        for item in operands
        if session_id in item.session_ids
        and item.polarity != "negative"
        and item.modality not in {"planned", "possible", "hypothetical"}
    ]
    event_attributes = [
        item for item in local
        if relation_terms & _tokens(item.predicate_key)
        and event_terms & _tokens(
            f"{item.predicate_key} {item.object_text} {item.context_key}"
        )
    ]
    attribute = max(
        event_attributes or [selected],
        key=lambda item: (
            len(_tokens(item.object_text)),
            len(event_terms & _tokens(item.object_text)),
            item.confidence,
            item.operand_id,
        ),
    )
    source_ids = list(dict.fromkeys(
        source
        for item in {selected.operand_id: selected, attribute.operand_id: attribute}.values()
        for source in item.source_turn_ids
    ))
    context_turns = _dialogue_window(session_id, source_ids, turns)
    context_source_ids = [
        str(getattr(turn, "node_id", "")) for turn in context_turns
        if getattr(turn, "node_id", "")
    ]
    source_ids = list(dict.fromkeys([*source_ids, *context_source_ids]))
    source_text = " ".join(
        str(getattr(turn, "text", "")) for turn in context_turns
    )
    relative_match = re.search(
        r"\b(?:last|previous)\s+(?:day|week|month|year)\b|"
        r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
        source_text,
        flags=re.IGNORECASE,
    )
    asks_date = bool(re.search(r"\bwhen\b|\bwhat\s+(?:date|day|week|month|year)\b", question))
    specific_attribute_terms = (
        _tokens(attribute.object_text)
        - event_terms
        - set(_ORDINALS)
        - {"another"}
    )
    if asks_date:
        value = (
            relative_match.group(0)
            if relative_match is not None
            else (attribute.event_time or attribute.observed_at)
        )
        attribute_resolution = "typed_date"
    elif specific_attribute_terms:
        value = attribute.object_text
        attribute_resolution = "typed_attribute"
    else:
        value = None
        attribute_resolution = "dialogue_window_required"
    return {
        "operation": "ordinal_event_attribute",
        "ordinal": ordinal,
        "value": value,
        "event_value": selected.object_text,
        "event_time": selected.event_time,
        "observed_at": selected.observed_at,
        "session_id": session_id,
        "attribute_resolution": attribute_resolution,
        "operand_ids": list(dict.fromkeys(
            [selected.operand_id, attribute.operand_id]
        )),
        "source_turn_ids": source_ids,
        "evidence": source_text[:640],
        "complete": bool(value and source_ids),
    }
