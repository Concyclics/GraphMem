from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .build import canonical_key
from .schema import QueryFrame
from .temporal_normalize import parse_datetime


_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "a": 1.0,
    "an": 1.0,
}
_NUMBER = (
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|a|an)"
)
_DURATION_RE = re.compile(
    rf"\b(?P<number>{_NUMBER})(?P<half>\s+and\s+(?:a|one)\s+half)?\s+"
    r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_ASSERTED_COMPLETION_RE = re.compile(
    r"\b(?:complete|completed|finish|finished|read|listened|took|spent)\w*\b",
    re.IGNORECASE,
)
_NON_ASSERTED_RE = re.compile(
    r"\b(?:could|estimate|estimated|expect|expected|hope|might|plan|planned|"
    r"predict|predicted|probably|should|would)\b",
    re.IGNORECASE,
)
_SELF_REFERENCE_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.IGNORECASE)
_UNIT_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
}


@dataclass(frozen=True)
class _DurationMention:
    entity: str
    value: float
    unit: str
    source_turn_id: str
    evidence: str
    support_turn_ids: tuple[str, ...] = ()


def _sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]


def _explicit_entities(question: str) -> list[str]:
    values = [
        (left or right).strip()
        for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
        if (left or right).strip()
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = canonical_key(value)
        if key and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _speaker_compatible(frame: QueryFrame, turn: Any) -> bool:
    speaker = canonical_key(str(
        getattr(turn, "speaker_key", "") or getattr(turn, "speaker", "")
    ))
    transport = str(getattr(turn, "transport_role", "")).casefold()
    if _SELF_REFERENCE_RE.search(frame.raw_question):
        return transport == "user" or speaker in {"participant 1", "user"}
    if frame.participant_terms:
        requested = {canonical_key(value) for value in frame.participant_terms}
        return speaker in requested
    return True


def _number_value(match: re.Match[str]) -> float:
    raw = match.group("number").casefold()
    value = float(raw) if re.fullmatch(r"\d+(?:\.\d+)?", raw) else _NUMBER_WORDS[raw]
    if match.group("half"):
        value += 0.5
    return value


def _entity_in_sentence(entity: str, sentence: str) -> bool:
    entity_key = canonical_key(entity)
    sentence_key = canonical_key(sentence)
    return bool(entity_key) and re.search(
        rf"(?<!\w){re.escape(entity_key)}(?!\w)", sentence_key
    ) is not None


def _mentions_for_entity(
    entity: str,
    frame: QueryFrame,
    turns: list[Any],
) -> list[_DurationMention]:
    mentions: list[_DurationMention] = []
    for turn in turns:
        if not _speaker_compatible(frame, turn):
            continue
        source_id = str(getattr(turn, "node_id", ""))
        for sentence in _sentences(str(getattr(turn, "text", ""))):
            if (
                not _entity_in_sentence(entity, sentence)
                or not _ASSERTED_COMPLETION_RE.search(sentence)
                or _NON_ASSERTED_RE.search(sentence)
            ):
                continue
            for match in _DURATION_RE.finditer(sentence):
                mentions.append(_DurationMention(
                    entity=entity,
                    value=_number_value(match),
                    unit=match.group("unit").casefold().rstrip("s"),
                    source_turn_id=source_id,
                    evidence=sentence,
                ))
    unique: dict[tuple[float, str, str], _DurationMention] = {}
    for mention in mentions:
        unique[(mention.value, mention.unit, mention.source_turn_id)] = mention
    return list(unique.values())


def _interval_mention_for_entity(
    entity: str,
    frame: QueryFrame,
    turns: list[Any],
) -> _DurationMention | None:
    requested = re.search(
        r"\bhow many\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        frame.raw_question,
        re.IGNORECASE,
    )
    if requested is None:
        return None
    unit = requested.group(1).casefold().rstrip("s")
    unit_days = {
        "second": 1.0 / 86400.0,
        "minute": 1.0 / 1440.0,
        "hour": 1.0 / 24.0,
        "day": 1.0,
        "week": 7.0,
        "month": 30.0,
        "year": 365.0,
    }[unit]
    starts: dict[Any, tuple[str, str]] = {}
    ends: dict[Any, tuple[str, str]] = {}
    for turn in turns:
        if not _speaker_compatible(frame, turn):
            continue
        source_id = str(getattr(turn, "node_id", ""))
        observed = parse_datetime(str(getattr(turn, "session_date", "")))
        if observed is None:
            continue
        for sentence in _sentences(str(getattr(turn, "text", ""))):
            if not _entity_in_sentence(entity, sentence) or _NON_ASSERTED_RE.search(sentence):
                continue
            if re.search(r"\b(?:start|started|begin|began)\w*\b", sentence, re.IGNORECASE):
                starts[observed] = (source_id, sentence)
            if re.search(
                r"\b(?:complete|completed|finish|finished|done)\w*\b",
                sentence,
                re.IGNORECASE,
            ):
                ends[observed] = (source_id, sentence)
    pairs = [(start, end) for start in starts for end in ends if end >= start]
    if not pairs:
        return None
    start, end = max(pairs, key=lambda pair: (pair[1], -pair[0].toordinal()))
    elapsed_days = (end - start).total_seconds() / 86400.0
    value = elapsed_days / unit_days
    if abs(value - round(value)) < 1e-9:
        value = float(round(value))
    start_id, start_evidence = starts[start]
    end_id, end_evidence = ends[end]
    return _DurationMention(
        entity=entity,
        value=value,
        unit=unit,
        source_turn_id=end_id,
        evidence=f"{start_evidence} -> {end_evidence}",
        support_turn_ids=tuple(dict.fromkeys((start_id, end_id))),
    )


def _sum_mentions(mentions: list[_DurationMention]) -> tuple[float, str] | None:
    units = {mention.unit for mention in mentions}
    if len(units) == 1:
        return sum(mention.value for mention in mentions), mentions[0].unit
    if not units <= set(_UNIT_SECONDS):
        return None
    # Preserve an exact result by returning the finest unit explicitly present.
    unit = min(units, key=lambda value: _UNIT_SECONDS[value])
    total_seconds = sum(
        mention.value * _UNIT_SECONDS[mention.unit] for mention in mentions
    )
    return total_seconds / _UNIT_SECONDS[unit], unit


def combined_named_duration_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Sum independently sourced durations for explicit named alternatives.

    This is a provenance closure, not a topic rule: every quoted entity in a
    combined-duration question must bind to its own asserted completion span.
    Missing or ambiguous operands leave the operation incomplete.
    """

    question = frame.raw_question
    if (
        not (
            frame.requested_operation == "duration"
            or re.search(r"\b(?:how long|duration)\b", question, re.IGNORECASE)
        )
        or not re.search(r"\b(?:combined|in total|altogether|sum)\b", question, re.IGNORECASE)
    ):
        return None
    entities = _explicit_entities(question)
    if len(entities) < 2:
        return None
    bound: list[_DurationMention] = []
    proofs: list[dict[str, Any]] = []
    for entity in entities:
        candidates = _mentions_for_entity(entity, frame, turns)
        if not candidates:
            interval = _interval_mention_for_entity(entity, frame, turns)
            if interval is not None:
                candidates = [interval]
        values = {(item.value, item.unit) for item in candidates}
        if len(values) != 1:
            return {
                "operation": "combined_named_duration_incomplete",
                "value": None,
                "entities": entities,
                "missing_or_ambiguous_entity": entity,
                "source_turn_ids": [],
                "operand_ids": [],
                "complete": False,
            }
        mention = max(candidates, key=lambda item: item.source_turn_id)
        bound.append(mention)
        proofs.append({
            "entity": entity,
            "value": mention.value,
            "unit": mention.unit,
            "source_turn_id": mention.source_turn_id,
            "source_turn_ids": list(
                mention.support_turn_ids or (mention.source_turn_id,)
            ),
            "evidence": mention.evidence,
        })
    summed = _sum_mentions(bound)
    if summed is None:
        return None
    value, unit = summed
    normalized_value: int | float = int(value) if value.is_integer() else value
    return {
        "operation": "combined_named_duration",
        "value": normalized_value,
        "unit": unit,
        "entities": entities,
        "proofs": proofs,
        "source_turn_ids": list(dict.fromkeys(
            source_id
            for mention in bound
            for source_id in (
                mention.support_turn_ids or (mention.source_turn_id,)
            )
        )),
        "operand_ids": [],
        "complete": True,
        "completion_basis": "lossless_named_entity_duration_sum",
    }
