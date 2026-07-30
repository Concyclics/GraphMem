from __future__ import annotations

import re
from typing import Any

from .schema import QueryFrame


def _user_turn(turn: Any) -> bool:
    transport = str(getattr(turn, "transport_role", "")).casefold()
    speaker = str(getattr(turn, "speaker_key", "")).casefold()
    return transport != "assistant" and (
        not transport
        or transport == "user"
        or speaker in {"participant 1", "participant_1", "user"}
    )


def _singular(value: str) -> str:
    value = value.casefold()
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def _tokens(value: str) -> set[str]:
    return {
        _singular(token)
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", value)
    }


def exact_entity_absence_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Prove absence only for strict possessive entity-slot questions.

    This is a global lossless lexical audit, not a semantic-neighbor decision:
    a sibling entity may provide useful contrast but cannot establish the
    requested entity's name or duration.
    """

    question = frame.raw_question
    name_match = re.search(
        r"\b(?:what|which)\s+is\s+the\s+name\s+of\s+my\s+"
        r"(?P<entity>[A-Za-z][A-Za-z' -]*?)[?]?$",
        question,
        re.IGNORECASE,
    )
    duration_match = re.search(
        r"\bhow\s+long\s+have\s+i\s+been\s+"
        r"(?P<relation>[A-Za-z]+)\s+"
        r"(?P<entity>[A-Za-z][A-Za-z' -]*?)[?]?$",
        question,
        re.IGNORECASE,
    )
    if name_match is None and duration_match is None:
        return None

    match = name_match or duration_match
    assert match is not None
    entity_phrase = match.group("entity").strip()
    entity_terms = _tokens(entity_phrase)
    if not entity_terms:
        return None
    head = next(reversed(re.findall(r"[A-Za-z][A-Za-z'-]*", entity_phrase)))
    head = _singular(head)

    rows = [
        turn for turn in turns
        if _user_turn(turn)
    ]
    if name_match is not None:
        exact_present = any(
            head in _tokens(str(getattr(turn, "text", "")))
            and re.search(
                r"\b(?:my|our|name|named|called)\b",
                str(getattr(turn, "text", "")), re.IGNORECASE,
            )
            for turn in rows
        )
    else:
        relation = _singular(duration_match.group("relation"))
        modifiers = entity_terms - {head}
        exact_present = any(
            head in _tokens(str(getattr(turn, "text", "")))
            and relation in _tokens(str(getattr(turn, "text", "")))
            and (
                not modifiers
                or modifiers.issubset(_tokens(str(getattr(turn, "text", ""))))
            )
            for turn in rows
        )
    if exact_present:
        return None

    sources: list[str] = []
    if name_match is not None:
        sibling_pattern = re.compile(
            r"\bmy\s+[A-Za-z][A-Za-z'-]*(?:'s)?\s+name\s+is\b|"
            r"\bmy\s+[A-Za-z][A-Za-z'-]*\s+(?:is\s+)?(?:named|called)\b",
            re.IGNORECASE,
        )
        sources = [
            str(getattr(turn, "node_id", ""))
            for turn in rows
            if sibling_pattern.search(str(getattr(turn, "text", "")))
        ]
    else:
        relation = _singular(duration_match.group("relation"))
        modifiers = entity_terms - {head}
        for turn in rows:
            text = str(getattr(turn, "text", ""))
            terms = _tokens(text)
            if relation in terms and (not modifiers or bool(modifiers & terms)):
                sources.append(str(getattr(turn, "node_id", "")))

    return {
        "operation": "exact_entity_mismatch",
        "value": "insufficient evidence for the exact requested entity",
        "required_entity": entity_phrase,
        "required_terms": sorted(entity_terms),
        "source_turn_ids": list(dict.fromkeys(source for source in sources if source)),
        "operand_ids": [],
        "complete": True,
        "global_lossless_scan_complete": True,
        "completion_basis": "strict_possessive_entity_head_absent_from_lossless_user_turns",
    }
