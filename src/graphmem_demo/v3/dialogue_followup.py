from __future__ import annotations

import re
from typing import Any

from .build import canonical_key
from .schema import QueryFrame


_PLAN_QUERY_RE = re.compile(
    r"\bwhat\s+did\s+(?P<target>[A-Za-z][\w'-]*)\s+"
    r"(?:plan|intend|want)\w*\s+to\s+do\b",
    re.IGNORECASE,
)
_COMMITMENT_RE = re.compile(
    r"\b(?:promise\w*|can|will|send\w*|give\w*|share\w*|provide\w*|"
    r"bring\w*)\b",
    re.IGNORECASE,
)
_PLAN_VALUE_RE = re.compile(
    r"\b(?:am\s+going\s+to|['’]m\s+going\s+to|plan(?:ning)?\s+to|"
    r"intend(?:ing)?\s+to|will)\s+(?P<value>[^.!?]{1,180})",
    re.IGNORECASE,
)
_STOP = {
    "did", "do", "plan", "promi", "promise", "share", "what",
    "with", "the", "to",
}


def _token_key(value: str) -> str:
    lowered = value.casefold().strip("'\"")
    if lowered.endswith("'s"):
        lowered = lowered[:-2]
    if len(lowered) > 5 and lowered.endswith("ing"):
        lowered = lowered[:-3]
    elif len(lowered) > 4 and lowered.endswith("ed"):
        lowered = lowered[:-2]
    if len(lowered) > 4 and lowered.endswith("s") and not lowered.endswith("ss"):
        lowered = lowered[:-1]
    return lowered


def _terms(value: str) -> set[str]:
    return {
        _token_key(token)
        for token in re.findall(r"[\w'-]+", value.replace("_", " "))
        if _token_key(token) not in _STOP
    }


def _speaker(turn: Any) -> str:
    return canonical_key(str(
        getattr(turn, "speaker_key", "") or getattr(turn, "speaker", "")
    ))


def _plan_value(text: str) -> str | None:
    match = _PLAN_VALUE_RE.search(text)
    if match is None:
        return None
    value = re.split(
        r"\s+(?:-|—)\s+|\s*,?\s+\b(?:and|but|because)\b",
        match.group("value"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" \t\r\n,;:-")
    return value if 1 <= len(value.split()) <= 28 else None


def dialogue_followup_plan_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Follow an object-bearing commitment to the recipient's adjacent plan."""

    match = _PLAN_QUERY_RE.search(frame.raw_question)
    if frame.requested_operation != "lookup" or match is None:
        return None
    target = canonical_key(match.group("target"))
    participants = [canonical_key(value) for value in frame.participant_terms]
    sources = [value for value in participants if value and value != target]
    if not target or not sources:
        return None
    query_terms = _terms(frame.raw_question) - set(participants)
    by_session: dict[str, list[Any]] = {}
    for turn in turns:
        by_session.setdefault(str(getattr(turn, "session_id", "")), []).append(turn)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for session_turns in by_session.values():
        ordered = sorted(
            session_turns,
            key=lambda turn: (
                int(getattr(turn, "turn_index", -1)),
                str(getattr(turn, "node_id", "")),
            ),
        )
        for position, anchor in enumerate(ordered):
            if _speaker(anchor) not in sources:
                continue
            anchor_text = str(getattr(anchor, "text", ""))
            if not _COMMITMENT_RE.search(anchor_text):
                continue
            prior = ordered[position - 1] if position else None
            anchor_window = " ".join(
                value for value in [
                    str(getattr(prior, "text", "")) if prior is not None else "",
                    anchor_text,
                ] if value
            )
            overlap = query_terms & _terms(anchor_window)
            if query_terms and not overlap:
                continue
            for response in ordered[position + 1:position + 3]:
                if _speaker(response) != target:
                    continue
                response_text = str(getattr(response, "text", ""))
                value = _plan_value(response_text)
                if value is None:
                    continue
                source_ids = [
                    str(getattr(item, "node_id", ""))
                    for item in (prior, anchor, response)
                    if item is not None
                ]
                result = {
                    "operation": "dialogue_followup_plan",
                    "value": value,
                    "target_speaker": str(getattr(response, "speaker", "")),
                    "commitment_speaker": str(getattr(anchor, "speaker", "")),
                    "source_turn_ids": source_ids,
                    "operand_ids": [],
                    "evidence": [anchor_window, response_text],
                    "complete": True,
                    "completion_basis": "same_session_commitment_response_plan",
                }
                score = (
                    len(overlap),
                    len(query_terms & _terms(response_text)),
                    str(getattr(response, "session_date", "") or ""),
                    int(getattr(response, "turn_index", -1)),
                    str(getattr(response, "node_id", "")),
                )
                candidates.append((score, result))
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]
