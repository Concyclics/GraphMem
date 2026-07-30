from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .build import canonical_key
from .schema import QueryFrame


_COMPARISON_RE = re.compile(
    r"\bhow\s+(?:many|much)\s+(?P<unit>[a-z]+)\s+"
    r"(?P<direction>older|younger)\s+(?:is|was|are|were)\s+"
    r"(?P<left>.+?)\s+than\s+(?P<right>.+?)[?]?$",
    re.IGNORECASE,
)
_AGE_CUE_RE = re.compile(
    r"\b(?:age|aged|birthday|years?\s+old|young|older|younger)\b",
    re.IGNORECASE,
)
_ORDINAL_BIRTHDAY_RE = re.compile(
    r"\b(?P<entity>[A-Za-z][A-Za-z' -]{0,60}?)['’]s\s+"
    r"(?P<value>\d{1,3})(?:st|nd|rd|th)\s+birthday\b",
    re.IGNORECASE,
)
_EXPLICIT_AGE_RE = re.compile(
    r"\b(?P<entity>[A-Za-z][A-Za-z' -]{0,60}?)\s+"
    r"(?:is|was|turned|turns|aged)\s+"
    r"(?P<value>\d{1,3})(?:\s+years?\s+old)?\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])(?P<value>\d{1,3})(?![\w.])")
_SELF_TERMS = {"i", "me", "my", "myself"}
_ENTITY_STOP = {
    "a", "an", "are", "is", "my", "the", "their", "was", "were",
}


@dataclass(frozen=True)
class _ScalarMention:
    entity_key: str
    value: float
    source_turn_id: str
    confidence: float
    evidence: str


def _entity_terms(value: str) -> set[str]:
    return {
        term
        for term in canonical_key(value).split()
        if term not in _ENTITY_STOP
    }


def _speaker_key(turn: Any) -> str:
    return canonical_key(
        str(getattr(turn, "speaker_key", "") or getattr(turn, "speaker", ""))
    )


def _sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]


def _age_mentions(turns: list[Any]) -> list[_ScalarMention]:
    mentions: list[_ScalarMention] = []
    for turn in turns:
        text = str(getattr(turn, "text", ""))
        source_id = str(getattr(turn, "node_id", ""))
        speaker = _speaker_key(turn)
        for sentence in _sentences(text):
            for match in _ORDINAL_BIRTHDAY_RE.finditer(sentence):
                entity = canonical_key(match.group("entity"))
                if entity:
                    mentions.append(_ScalarMention(
                        entity, float(match.group("value")), source_id, 1.0, sentence,
                    ))
            for match in _EXPLICIT_AGE_RE.finditer(sentence):
                entity = canonical_key(match.group("entity"))
                if entity and not re.search(
                    r"\b(?:this|that|it|there|what|which|who)\b",
                    entity,
                ):
                    mentions.append(_ScalarMention(
                        entity, float(match.group("value")), source_id, 0.95, sentence,
                    ))
            if not speaker or not _AGE_CUE_RE.search(sentence):
                continue
            values = [
                float(match.group("value"))
                for match in _NUMBER_RE.finditer(sentence)
                if 0 < float(match.group("value")) < 125
            ]
            if len(set(values)) == 1:
                mentions.append(_ScalarMention(
                    speaker, values[0], source_id, 0.82, sentence,
                ))
    unique: dict[tuple[str, float, str], _ScalarMention] = {}
    for mention in mentions:
        key = (mention.entity_key, mention.value, mention.source_turn_id)
        old = unique.get(key)
        if old is None or mention.confidence > old.confidence:
            unique[key] = mention
    return list(unique.values())


def _bind_entity(
    phrase: str,
    mentions: list[_ScalarMention],
) -> list[_ScalarMention]:
    raw_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", phrase)
    }
    terms = _entity_terms(phrase)
    self_reference = bool(raw_terms) and raw_terms <= _SELF_TERMS
    if self_reference:
        # Role labels are used only to ground first-person deixis, never as an
        # importance or truth signal.
        return [
            mention for mention in mentions
            if re.fullmatch(r"participant\s*1|user", mention.entity_key)
        ]
    terms -= _SELF_TERMS
    if not terms:
        return []
    return [
        mention for mention in mentions
        if terms <= _entity_terms(mention.entity_key)
        or _entity_terms(mention.entity_key) <= terms
    ]


def scalar_comparison_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Compute a typed scalar comparison from lossless entity-bound mentions."""

    match = _COMPARISON_RE.search(frame.raw_question.strip())
    if not match:
        return None
    unit = match.group("unit").casefold()
    if unit not in {"year", "years"}:
        return None
    mentions = _age_mentions(turns)
    left = _bind_entity(match.group("left"), mentions)
    right = _bind_entity(match.group("right"), mentions)
    if not left or not right:
        return None

    pairs = [
        (min(a.confidence, b.confidence), a, b)
        for a in left for b in right
        if a.entity_key != b.entity_key and a.value != b.value
    ]
    if not pairs:
        return None
    _confidence, left_value, right_value = max(
        pairs,
        key=lambda row: (
            row[0],
            row[1].confidence + row[2].confidence,
            row[1].source_turn_id,
            row[2].source_turn_id,
        ),
    )
    difference = (
        left_value.value - right_value.value
        if match.group("direction").casefold() == "older"
        else right_value.value - left_value.value
    )
    if difference < 0:
        return None
    value: int | float = int(difference) if difference.is_integer() else difference
    return {
        "operation": "scalar_comparison",
        "value": value,
        "unit": unit,
        "direction": match.group("direction").casefold(),
        "left": {
            "entity": left_value.entity_key,
            "value": left_value.value,
            "source_turn_id": left_value.source_turn_id,
            "evidence": left_value.evidence,
        },
        "right": {
            "entity": right_value.entity_key,
            "value": right_value.value,
            "source_turn_id": right_value.source_turn_id,
            "evidence": right_value.evidence,
        },
        "source_turn_ids": list(dict.fromkeys([
            left_value.source_turn_id, right_value.source_turn_id,
        ])),
        "operand_ids": [],
        "complete": True,
        "completion_basis": "lossless_entity_bound_scalar_difference",
    }
