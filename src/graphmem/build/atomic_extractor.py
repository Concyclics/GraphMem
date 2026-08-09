from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class InformationUnit:
    """A deterministic, source-grounded extraction obligation.

    The unit is deliberately smaller than a fact.  It records the fragile
    surface that an abstractive extractor is most likely to lose (a number,
    date, negation, named entity, modality, or state change) and lets the build
    prove that the surface was either represented or explicitly rejected.
    """

    unit_id: int
    kind: str
    turn_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class TurnChunk:
    turn_id: str
    start: int
    end: int
    text: str


_NUMBER = re.compile(
    r"(?<![\w])(?:[$£€¥]\s*)?(?:\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?|zero|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve|first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth)(?:\s*(?:%|percent|dollars?|usd|euros?|"
    r"pounds?|yen|kg|kilograms?|g|grams?|km|kilometers?|miles?|hours?|minutes?|"
    r"days?|weeks?|months?|years?|times?))?(?![\w])",
    re.I,
)
_DATE = re.compile(
    r"\b(?:\d{4}[/-]\d{1,2}(?:[/-]\d{1,2})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s+\d{4})?|"
    r"today|tonight|yesterday|tomorrow|last\s+(?:night|week|month|year)|"
    r"next\s+(?:week|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b",
    re.I,
)
_DURATION = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"a|an)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:not|never|no longer|didn't|doesn't|don't|isn't|aren't|wasn't|weren't|"
    r"won't|wouldn't|can't|cannot|couldn't|hasn't|haven't|hadn't|without)\b",
    re.I,
)
_MODALITY = re.compile(
    r"\b(?:plan(?:s|ned|ning)? to|planning on|want(?:s|ed)? to|hope(?:s|d)? to|"
    r"would (?:like|love) to|going to|will|might|may|intend(?:s|ed)? to|"
    r"expect(?:s|ed)? to|consider(?:s|ed|ing)?)\b",
    re.I,
)
_STATE_CHANGE = re.compile(
    r"\b(?:start(?:s|ed|ing)?|stop(?:s|ped|ping)?|begin(?:s|ning)?|began|"
    r"finish(?:es|ed|ing)?|complete(?:s|d)?|move(?:s|d)?|join(?:s|ed)?|leave|left|"
    r"buy|bought|purchase(?:s|d)?|sell|sold|adopt(?:s|ed)?|become|became|"
    r"switch(?:es|ed)?|change(?:s|d)?|win|won|lose|lost|receive(?:s|d)?|"
    r"cancel(?:s|led)?|resume(?:s|d)?)\b",
    re.I,
)
_ENTITY = re.compile(
    r"(?<![\w])(?:[A-Z][A-Za-z0-9'&-]*(?:\s+(?:(?:of|the|and|de|van|von)\s+)?"
    r"[A-Z][A-Za-z0-9'&-]*){0,4})"
)
_QUOTED_ITEM = re.compile(r'\"([^\"\n]{2,80})\"')
_ENTITY_STOP = {
    "a", "an", "and", "but", "he", "her", "here", "his", "i", "if", "in", "it",
    "its", "my", "no", "not", "oh", "on", "she", "so", "that", "the", "their",
    "then", "there", "they", "this", "we", "well", "what", "when", "where", "who",
    "why", "yes", "you", "your", "today", "tomorrow", "yesterday",
    "another", "first", "final", "opening", "later", "meanwhile", "afterward", "at",
    "also", "however", "although", "because", "finally",
    "do", "can", "could", "would", "should", "may", "might", "must", "will",
    "are", "by", "for", "from", "how", "long", "media", "plus", "take", "speaking",
    "anyway", "almost", "give", "check", "glad", "something", "seeing", "guess",
    "cuddling", "tell", "life", "here", "hey",
    "been", "sounds", "have", "lots", "trying", "appreciate", "even", "nature",
    "specifically", "cute", "representing", "moreover", "connecting", "having",
    "cooking", "really", "anything", "yeah", "with", "thank", "women", "art",
    "every", "hope", "unconditional",
}
_PRIORITY = {
    "date": 0,
    "duration": 1,
    "number_unit": 2,
    "negation": 3,
    "modality": 4,
    "state_change": 5,
    "entity": 6,
    "item": 7,
}


def scan_information_units(turns: Sequence[Any]) -> tuple[InformationUnit, ...]:
    """Find high-salience surfaces without invoking a model.

    Overlapping numeric/date matches are collapsed in favour of the more
    specific type.  Logical operators (negation/modality/state change) remain
    separate because losing one changes the meaning of the same value.
    """

    candidates: list[tuple[int, str, str, int, int, str]] = []
    for turn_position, turn in enumerate(turns):
        text = str(turn.raw_text)
        turn_id = str(turn.turn_id)
        patterns = (
            ("date", _DATE),
            ("duration", _DURATION),
            ("number_unit", _NUMBER),
            ("negation", _NEGATION),
            ("modality", _MODALITY),
            ("state_change", _STATE_CHANGE),
        )
        for kind, pattern in patterns:
            for match in pattern.finditer(text):
                value = " ".join(match.group(0).split())
                if kind == "number_unit":
                    # Numbered assistant lists ("1. ...", "2) ...") made up a
                    # large fraction of failed atomic obligations.  They encode
                    # document layout, not a durable quantity.
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    prefix = text[line_start:match.start()]
                    suffix = text[match.end():]
                    if (not prefix.strip()
                            and re.match(r"^[.)\-:]\s+", suffix)
                            and not re.search(r"[$£€¥%]", value)):
                        continue
                if kind == "date" and value.casefold() == "may":
                    # The case-insensitive month regex also matched the modal
                    # verb in "may want".  Keep unambiguous month contexts.
                    before = text[max(0, match.start() - 20):match.start()]
                    after = text[match.end():match.end() + 12]
                    month_context = (
                        re.search(r"\b(?:in|during|since|until|by|from|through)\s*$",
                                  before, re.I)
                        or re.match(r"\s+(?:\d{1,2}(?:st|nd|rd|th)?|\d{4})\b",
                                    after, re.I)
                    )
                    if not month_context:
                        continue
                candidates.append((turn_position, kind, turn_id, match.start(), match.end(), value))
        for match in _ENTITY.finditer(text):
            value = " ".join(match.group(0).split()).strip(".,!?;:'")
            lowered = value.casefold()
            if re.match(
                r"^(?:i|you|he|she|it|that|they|we|here|there|who|what|where|how|life)'",
                lowered,
            ):
                continue
            if lowered.startswith("hey ") and len(value.split()) == 2:
                offset = value.casefold().find("hey") + 4
                value = value[offset:]
                entity_start = match.start() + offset
            else:
                entity_start = match.start()
            if len(value) < 2 or value.casefold() in _ENTITY_STOP:
                continue
            # A capitalised sentence opener is weak evidence unless it contains
            # another capitalised token or recognizable entity punctuation.
            prefix = text[:entity_start]
            sentence_opener = (
                not prefix.strip()
                or bool(re.search(
                    r"(?:^|[.!?。！？]\s+|\n)\s*(?:\d+[.)\-:]\s+)?$", prefix))
            )
            if sentence_opener:
                first_token = value.casefold().split()[0]
                if ((" " not in value and not re.search(r"[.&'-]", value))
                        or first_token in _ENTITY_STOP):
                    continue
            candidates.append((turn_position, "entity", turn_id,
                               entity_start, entity_start + len(value), value))
        for match in _QUOTED_ITEM.finditer(text):
            value = match.group(1).strip()
            if value:
                start = match.start() + match.group(0).find(value)
                candidates.append((turn_position, "item", turn_id,
                                   start, start + len(value), value))

    # Date/duration/number candidates describe the same fragile scalar when
    # their spans overlap.  Keep the most specific one; do not collapse logical
    # operators or named items into it.
    kept: list[tuple[int, str, str, int, int, str]] = []
    for candidate in sorted(candidates, key=lambda row: (
            row[0], row[3], _PRIORITY[row[1]], -(row[4] - row[3]), row[5].casefold())):
        turn_position, kind, _turn_id, start, end, _text = candidate
        scalar = kind in {"date", "duration", "number_unit"}
        if scalar and any(
            old[0] == turn_position
            and old[1] in {"date", "duration", "number_unit"}
            and max(start, old[3]) < min(end, old[4])
            for old in kept
        ):
            continue
        if kind == "entity" and any(
            old[0] == turn_position
            and old[1] in {"date", "duration", "number_unit"}
            and max(start, old[3]) < min(end, old[4])
            for old in kept
        ):
            continue
        duplicate = any(
            old[0] == turn_position and old[1] == kind
            and old[3] == start and old[4] == end
            for old in kept
        )
        if not duplicate:
            kept.append(candidate)

    return tuple(
        InformationUnit(index, kind, turn_id, start, end, text)
        for index, (_position, kind, turn_id, start, end, text) in enumerate(kept)
    )


def adaptive_fact_cap(
    units: Sequence[InformationUnit],
    *,
    floor: int,
    ceiling: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> int:
    """K_s=min(K_max, max(K_floor, ceil(alpha I + beta E + gamma T)))."""

    entities = sum(unit.kind == "entity" for unit in units)
    temporal = sum(unit.kind in {"date", "duration"} for unit in units)
    requested = math.ceil(alpha * len(units) + beta * entities + gamma * temporal)
    return min(ceiling, max(floor, requested))


def sentence_chunks(
    turn_id: str,
    text: str,
    max_chars: int,
    protected_spans: Sequence[tuple[int, int]] = (),
) -> tuple[TurnChunk, ...]:
    """Losslessly split a long turn at sentence/word boundaries.

    Unlike the old head/tail compactor, concatenating the returned source spans
    reproduces the entire input.  Whitespace stays attached to a neighbouring
    chunk, so offsets remain exact and evidence quotes resolve in the raw turn.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return (TurnChunk(turn_id, 0, len(text), text),)
    chunks: list[TurnChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        if hard_end == len(text):
            end = hard_end
        else:
            window = text[start:hard_end]
            boundaries = [match.end() for match in re.finditer(r"(?:[.!?。！？]\s+|\n+)", window)]
            end = start + (boundaries[-1] if boundaries and boundaries[-1] >= max_chars // 3 else 0)
            if end == start:
                spaces = [match.end() for match in re.finditer(r"\s+", window)]
                end = start + (spaces[-1] if spaces and spaces[-1] >= max_chars // 3 else len(window))
        if end <= start:  # defensive guard for pathological zero-width input
            end = min(len(text), start + max_chars)
        # Never split an extraction obligation between chunks.  High-salience
        # surfaces are short; extending by a few characters is cheaper than
        # emitting an obligation that the model cannot see in any one source
        # segment.
        crossing = [span_end for span_start, span_end in protected_spans
                    if span_start < end < span_end]
        if crossing:
            end = min(len(text), max(crossing))
        chunks.append(TurnChunk(turn_id, start, end, text[start:end]))
        start = end
    return tuple(chunks)


def units_for_span(
    units: Iterable[InformationUnit], turn_id: str, start: int, end: int
) -> tuple[InformationUnit, ...]:
    return tuple(
        unit for unit in units
        if unit.turn_id == turn_id and start <= unit.start and unit.end <= end
    )
