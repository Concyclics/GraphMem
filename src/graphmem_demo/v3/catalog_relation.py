from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_IRREGULAR = {
    "began": "begin", "bought": "buy", "kept": "keep", "started": "start",
    "stored": "store", "using": "use", "used": "use",
}
_RELATION_FAMILIES = (
    {"keep", "locate", "location", "place", "put", "store"},
    {"begin", "join", "start", "subscribe", "subscription", "trial", "use"},
    {"finish", "complete", "end", "stop"},
    {"like", "love", "prefer", "enjoy"},
)
_INITIATION = {"begin", "join", "start", "subscribe", "trial"}
_FUNCTION = {
    "did", "do", "does", "i", "me", "my",
    "current", "currently", "latest", "most", "now", "recent", "recently",
    "where", "which", "what", "who",
}
_DISCOURSE = {
    "advised", "asked", "discussed", "explained", "provided",
    "recommended", "suggested",
}


def _token(value: str) -> str:
    value = value.casefold().strip("'\"")
    if value in _IRREGULAR:
        return _IRREGULAR[value]
    if len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return _IRREGULAR.get(value, value)


def _tokens(value: str) -> set[str]:
    return {
        _token(item)
        for item in re.findall(r"[\w']+", value.replace("_", " ").replace("-", " "))
        if len(item) > 1
    }


def _date(value: str | None) -> datetime:
    text = value or ""
    match = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if not match:
        return datetime.min
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return datetime.min


def _relative_age_days(text: str) -> int:
    lowered = text.casefold()
    if re.search(r"\b(?:today|currently|now|recently)\b", lowered):
        return 0
    if re.search(r"\blast\s+day\b|\byesterday\b", lowered):
        return 1
    if re.search(r"\blast\s+week\b", lowered):
        return 7
    if re.search(r"\blast\s+month\b", lowered):
        return 30
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\b",
        lowered,
    )
    if not match:
        return 10**6
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    amount = int(match.group(1)) if match.group(1).isdigit() else words[match.group(1)]
    scale = {"day": 1, "week": 7, "month": 30, "year": 365}[match.group(2)]
    return amount * scale


def latest_relation_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
    semantic_similarity: Callable[[OperandRecordV3], float] | None = None,
) -> dict[str, Any] | None:
    if frame.requested_operation not in {"latest", "state"}:
        return None
    query_terms = _tokens(frame.raw_question) - _FUNCTION
    query_families = [
        family for family in _RELATION_FAMILIES if family & query_terms
    ]
    relation_terms = set().union(*query_families) if query_families else set()
    entity_terms = query_terms - relation_terms
    query_initiation = bool(query_terms & _INITIATION)
    asks_completed_relation = bool(re.search(
        r"\b(?:did|bought|purchased|acquired|received|got)\b",
        frame.raw_question.casefold(),
    ))
    ranked = []
    for item in operands:
        if item.polarity == "negative" or item.state_op in {"remove", "cancel", "retract"}:
            continue
        if asks_completed_relation and item.modality in {"planned", "possible", "hypothetical"}:
            continue
        if re.search(
            r"\b(?:advis|ask|discuss|explain|provid|recommend|suggest)",
            item.predicate_key.casefold(),
        ):
            continue
        text = " ".join([
            item.predicate_key, item.object_text, item.context_key,
        ])
        terms = _tokens(text)
        lexical = len(query_terms & terms)
        relation = sum(1 for family in query_families if family & terms)
        initiation = int(query_initiation and bool(terms & _INITIATION))
        entity_overlap = len(entity_terms & terms)
        if query_initiation and not initiation:
            continue
        if query_families and relation <= 0:
            continue
        if lexical <= 0 and relation <= 0 and initiation <= 0:
            continue
        dense = max(0.0, semantic_similarity(item)) if semantic_similarity else 0.0
        semantic = (
            2 * entity_overlap + lexical + 3 * relation + 3 * initiation
            + query_overlap(frame, item.retrieval_text)
            + (10 if query_initiation else 2) * dense
        )
        age = _relative_age_days(item.object_text + " " + (item.event_time or ""))
        ranked.append((
            entity_overlap, relation, initiation,
            _date(item.observed_at or item.event_time), -age, semantic,
            item.confidence, item,
        ))
    if not ranked:
        return None
    *_rank, item = max(
        ranked, key=lambda row: (*row[:7], row[7].operand_id)
    )
    return {
        "operation": "latest_relation_state",
        "value": item.object_text,
        "predicate": item.predicate_key,
        "event_time": item.event_time,
        "observed_at": item.observed_at,
        "modality": item.modality,
        "operand_ids": [item.operand_id],
        "source_turn_ids": list(item.source_turn_ids),
        "complete": True,
    }
