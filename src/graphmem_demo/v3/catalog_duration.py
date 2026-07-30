from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_IRREGULAR = {
    "arrived": "arrive", "began": "begin", "bought": "buy",
    "completed": "complete", "completion": "complete",
    "delivered": "deliver", "departed": "depart",
    "finished": "finish", "found": "find", "loved": "love",
    "ordered": "order", "purchased": "purchase",
    "received": "receive", "returned": "return", "started": "start",
    "submission": "submit", "submitted": "submit",
}
_ENDPOINT_FAMILIES = (
    {"arrive", "deliver", "receive"},
    {"buy", "order", "purchase"},
    {"begin", "start"},
    {"complete", "end", "finish"},
    {"submit"},
    {"depart", "leave"},
    {"return"},
)
_QUERY_GLUE = {
    "ago", "and", "between", "day", "days", "difference", "elapsed",
    "from", "had", "how", "long", "many", "pass", "passed", "since",
    "the", "time", "until", "when",
}



def _requested_duration_unit(question: str) -> str:
    match = re.search(
        r"\bhow\s+many\s+(days?|weeks?|months?|years?)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return "day"
    return match.group(1).casefold().rstrip("s")


def _convert_elapsed_days(elapsed_days: int, unit: str) -> int | float:
    scale = {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    value = elapsed_days / scale
    nearest = round(value)
    tolerance = 0.1 if unit in {"month", "year"} else 1e-9
    return nearest if abs(value - nearest) <= tolerance else round(value, 2)


def _datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _operand_datetime(item: OperandRecordV3) -> datetime | None:
    """Use the most specific typed date before the observation timestamp."""
    return (
        _datetime(item.event_time)
        or _datetime(item.object_text)
        or _datetime(item.context_key)
        or _datetime(item.observed_at)
    )


def _tokens(value: str) -> set[str]:
    result = set()
    for token in re.findall(r"[\w'-]+", value.casefold()):
        token = _IRREGULAR.get(token, token)
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
            endpoint_terms = set().union(*_ENDPOINT_FAMILIES)
            if token + "e" in endpoint_terms:
                token += "e"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
            if token.endswith("v"):
                token += "e"
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result


def _endpoint_phrases(question: str) -> tuple[set[str], set[str]] | None:
    lowered = question.casefold()
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)[?]?$", lowered)
    if not match:
        match = re.search(r"\bsince\s+(.+?)\s+when\s+(.+?)[?]?$", lowered)
    if not match:
        match = re.search(
            r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+"
            r"(?:before|after)\s+(.+?)\s+did\s+(.+?)[?]?$", lowered,
        )
    reverse = False
    if not match:
        match = re.search(
            r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+did\s+it\s+take"
            r"(?:\s+for\s+.+?\s+to)?\s+(.+?)\s+after\s+(.+?)[?]?$",
            lowered,
        )
        reverse = match is not None
    if not match:
        return None
    ignored = _QUERY_GLUE | {
        "a", "an", "at", "did", "do", "does", "i", "in", "it", "my",
        "of", "on", "one", "that", "to", "was",
    }
    left = _tokens(match.group(1)) - ignored
    right = _tokens(match.group(2)) - ignored
    if reverse:
        left, right = right, left
    return (left, right) if left and right else None


def _proper_endpoint_anchors(question: str) -> tuple[set[str], set[str]] | None:
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)[?]?$", question)
    if not match:
        match = re.search(r"\bsince\s+(.+?)\s+when\s+(.+?)[?]?$", question)
    if not match:
        match = re.search(
            r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+"
            r"(?:before|after)\s+(.+?)\s+did\s+(.+?)[?]?$",
            question, flags=re.IGNORECASE,
        )
    reverse = False
    if not match:
        match = re.search(
            r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+did\s+it\s+take"
            r"(?:\s+for\s+.+?\s+to)?\s+(.+?)\s+after\s+(.+?)[?]?$",
            question, flags=re.IGNORECASE,
        )
        reverse = match is not None
    if not match:
        return None
    anchors = []
    for phrase in match.groups():
        proper = [
            token for token in re.findall(r"\b(?:[A-Z][A-Za-z'-]*|[a-z]+[A-Z][A-Za-z'-]*)\b", phrase)
            if token not in {"I"}
        ]
        if proper:
            anchors.append(_tokens(proper[-1]))
            continue
        common = re.findall(
            r"\b(?:a|an|the|my|our|their)\s+([a-z][\w'-]+)",
            phrase.casefold(),
        )
        anchors.append(_tokens(common[-1]) if common else set())
    if reverse:
        anchors.reverse()
    return anchors[0], anchors[1]


def _rarest_terms(
    terms: set[str], operands: list[OperandRecordV3]
) -> set[str]:
    frequencies = {}
    for term in terms:
        frequencies[term] = sum(
            term in _tokens(
                item.predicate_key + " " + item.object_text + " " + item.context_key
            )
            for item in operands
        )
    positive = {term: count for term, count in frequencies.items() if count > 0}
    if not positive:
        return set()
    minimum = min(positive.values())
    return {term for term, count in positive.items() if count == minimum}


def _explicit_relative_offset_from_turns(
    frame: QueryFrame,
    turns: list[Any],
    endpoint_phrases: tuple[set[str], set[str]],
) -> dict[str, Any] | None:
    """Close a duration stated as N units before/after a shared anchor."""
    number_words = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    }
    relation_re = re.compile(
        r"\b(?P<number>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?P<unit>days?|weeks?|months?|years?)\s+"
        r"(?P<relation>before|after)\s+"
        r"(?P<anchor>[A-Za-z0-9][^,.;!?]{0,80})",
        re.IGNORECASE,
    )
    user_turns = []
    for turn in turns:
        transport = str(getattr(turn, "transport_role", "")).casefold()
        speaker = str(getattr(turn, "speaker_key", "")).casefold()
        if transport == "assistant" or (
            transport and transport != "user"
            and speaker not in {"participant 1", "participant_1", "user"}
        ):
            continue
        text = str(getattr(turn, "text", ""))
        user_turns.append((turn, text, _tokens(text)))

    left_terms, right_terms = endpoint_phrases
    glue = _QUERY_GLUE | {"a", "an", "at", "before", "after", "on", "to"}
    for target, target_text, target_terms in user_turns:
        if len(right_terms & target_terms) < min(2, len(right_terms)):
            continue
        for match in relation_re.finditer(target_text):
            anchor_terms = _tokens(match.group("anchor")) - glue
            if not anchor_terms:
                continue
            anchors = [
                turn for turn, _text, terms in user_turns
                if turn is not target
                and len(left_terms & terms) >= min(2, len(left_terms))
                and bool(anchor_terms & terms)
            ]
            if not anchors:
                continue
            raw_number = match.group("number").casefold()
            amount = int(raw_number) if raw_number.isdigit() else number_words[raw_number]
            unit = match.group("unit").casefold().rstrip("s")
            days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
            requested_unit = _requested_duration_unit(frame.raw_question)
            anchor_turn = anchors[0]
            return {
                "operation": "catalog_duration",
                "elapsed_days": days,
                "inclusive_days": days + 1,
                "value": _convert_elapsed_days(days, requested_unit),
                "unit": requested_unit,
                "relation": match.group("relation").casefold(),
                "relative_anchor": match.group("anchor").strip(),
                "source_turn_ids": [
                    str(getattr(target, "node_id", "")),
                    str(getattr(anchor_turn, "node_id", "")),
                ],
                "operand_ids": [],
                "complete": True,
                "completion_basis": "lossless_explicit_relative_offset_shared_anchor",
            }
    return None


def duration_from_turns(
    frame: QueryFrame,
    turns: list[Any],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Fallback endpoint algebra over lossless first-person turns."""

    endpoint_phrases = _endpoint_phrases(frame.raw_question)
    if frame.requested_operation != "duration" or endpoint_phrases is None:
        return None
    explicit_offset = _explicit_relative_offset_from_turns(
        frame, turns, endpoint_phrases
    )
    if explicit_offset is not None:
        return explicit_offset
    endpoint_terms = set().union(*endpoint_phrases)
    candidates: list[OperandRecordV3] = []
    for turn in turns:
        transport_role = str(getattr(turn, "transport_role", "")).casefold()
        speaker_key = str(getattr(turn, "speaker_key", "")).casefold()
        if transport_role == "assistant" or (
            transport_role and transport_role != "user"
            and speaker_key not in {"participant 1", "user"}
        ):
            continue
        node_id = str(getattr(turn, "node_id", ""))
        session_id = str(getattr(turn, "session_id", ""))
        session_date = str(getattr(turn, "session_date", ""))
        fragments = [
            value.strip()
            for value in re.split(
                r"(?<=[.!?])\s+|\n+",
                str(getattr(turn, "text", "")),
            )
            if value.strip()
        ]
        for index, fragment in enumerate(fragments):
            if len(endpoint_terms & _tokens(fragment)) < 2:
                continue
            candidates.append(OperandRecordV3(
                operand_id=f"{node_id}:lossless_endpoint:{index}",
                question_id=frame.raw_question,
                subject_key=speaker_key or "participant 1",
                predicate_key=fragment,
                object_key=fragment.casefold(),
                object_text=fragment,
                context_key="lossless endpoint evidence",
                event_time=session_date,
                observed_at=session_date,
                source_turn_ids=[node_id],
                session_ids=[session_id],
                retrieval_text=fragment,
            ))
    result = duration_from_operands(frame, candidates, query_overlap)
    if result is not None:
        result["completion_basis"] = "lossless_between_endpoint_closure"
    return result


def duration_from_operands(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    if re.search(r"\bsince\b", frame.raw_question, re.IGNORECASE) and re.search(
        r"\b(?:consecutive|in a row)\b", frame.raw_question, re.IGNORECASE,
    ):
        return None
    query = _tokens(frame.raw_question)
    endpoint_query = query - _QUERY_GLUE
    query_families = [
        family for family in _ENDPOINT_FAMILIES
        if family & _tokens(frame.raw_question)
    ]
    dated = [
        item for item in operands
        if _operand_datetime(item)
        and item.predicate_key != "said"
        and not re.search(
            r"\b(?:advis|ask|discuss|explain|offer|provid|recommend|suggest)",
            item.predicate_key.casefold(),
        )
    ]
    endpoint_phrases = _endpoint_phrases(frame.raw_question)
    if endpoint_phrases is None and len(query_families) < 2:
        # A relation-duration question is not a request to subtract any two
        # nearby dated facts.  Without two parseable, independently bound
        # endpoints, leave the answer to explicit duration evidence.
        return None
    proper_anchors = _proper_endpoint_anchors(frame.raw_question)
    endpoint_anchors = (
        (
            proper_anchors[0] or _rarest_terms(endpoint_phrases[0], dated),
            proper_anchors[1] or _rarest_terms(endpoint_phrases[1], dated),
        )
        if endpoint_phrases and proper_anchors else None
    )
    pairs = []
    for offset, left in enumerate(dated):
        left_date = _operand_datetime(left)
        for right in dated[offset + 1:]:
            right_date = _operand_datetime(right)
            if not left_date or not right_date or left_date == right_date:
                continue
            endpoint_binding = 0
            if endpoint_phrases and endpoint_anchors:
                left_operand_terms = _tokens(
                    left.predicate_key + " " + left.object_text + " " + left.context_key
                )
                right_operand_terms = _tokens(
                    right.predicate_key + " " + right.object_text + " " + right.context_key
                )
                direct = (
                    bool(endpoint_anchors[0] & left_operand_terms)
                    and bool(endpoint_anchors[1] & right_operand_terms)
                )
                reverse = (
                    bool(endpoint_anchors[0] & right_operand_terms)
                    and bool(endpoint_anchors[1] & left_operand_terms)
                )
                if not direct and not reverse:
                    continue
                if direct:
                    endpoint_binding = (
                        len(endpoint_phrases[0] & left_operand_terms)
                        + len(endpoint_phrases[1] & right_operand_terms)
                    )
                else:
                    endpoint_binding = (
                        len(endpoint_phrases[0] & right_operand_terms)
                        + len(endpoint_phrases[1] & left_operand_terms)
                    )
            action_match = (
                len(query & _tokens(left.predicate_key))
                + len(query & _tokens(right.predicate_key))
            )
            entity_match = (
                len(query & _tokens(left.object_text))
                + len(query & _tokens(right.object_text))
            )
            left_endpoint_terms = endpoint_query & _tokens(
                left.predicate_key + " " + left.object_text + " " + left.context_key
            )
            right_endpoint_terms = endpoint_query & _tokens(
                right.predicate_key + " " + right.object_text + " " + right.context_key
            )
            endpoint_union = left_endpoint_terms | right_endpoint_terms
            endpoint_balance = min(
                len(left_endpoint_terms), len(right_endpoint_terms)
            )
            shared_entity = len(_tokens(left.object_text) & _tokens(right.object_text))
            left_terms = _tokens(left.predicate_key + " " + left.object_text)
            right_terms = _tokens(right.predicate_key + " " + right.object_text)
            left_families = {
                index for index, family in enumerate(query_families) if family & left_terms
            }
            right_families = {
                index for index, family in enumerate(query_families) if family & right_terms
            }
            covered_families = len(left_families | right_families)
            if (
                len(query_families) >= 2
                and (
                    not left_families or not right_families
                    or covered_families < len(query_families)
                )
            ):
                continue
            # A duration question may name two endpoints while using only a
            # generic verb such as "passed between".  In that case both named
            # endpoint entities are stronger binding evidence than a verb
            # copied into the extracted predicate.
            named_endpoint_pair = endpoint_balance > 0 and len(endpoint_union) >= 2
            if not named_endpoint_pair and (
                action_match == 0 or (entity_match == 0 and shared_entity == 0)
            ):
                continue
            score = (
                3.0 * action_match
                + 1.5 * entity_match
                + shared_entity
                + 8.0 * len(endpoint_union)
                + 12.0 * endpoint_balance
                + 30.0 * endpoint_binding
                + 10.0 * covered_families
                + query_overlap(frame, left.retrieval_text)
                + query_overlap(frame, right.retrieval_text)
            )
            pairs.append((score, left, right, abs((right_date - left_date).days)))
    if not pairs:
        return None
    _score, left, right, elapsed = max(
        pairs, key=lambda row: (row[0], row[3], row[1].operand_id, row[2].operand_id)
    )
    requested_unit = _requested_duration_unit(frame.raw_question)
    return {
        "operation": "catalog_duration",
        "elapsed_days": elapsed,
        "inclusive_days": elapsed + 1,
        "value": _convert_elapsed_days(elapsed, requested_unit),
        "unit": requested_unit,
        "left": {
            "value": left.object_text,
            "predicate": left.predicate_key,
            "date": left.event_time or left.observed_at,
            "operand_id": left.operand_id,
        },
        "right": {
            "value": right.object_text,
            "predicate": right.predicate_key,
            "date": right.event_time or right.observed_at,
            "operand_id": right.operand_id,
        },
        "source_turn_ids": list(dict.fromkeys(
            [*left.source_turn_ids, *right.source_turn_ids]
        )),
        "complete": True,
    }
