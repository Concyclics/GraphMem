from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .action_semantics import action_family_overlap
from .schema import QueryFrame


_GENERIC = {
    "all", "and", "both", "did", "do", "does", "have", "he", "her", "his",
    "how", "i", "is", "it", "many", "me", "my", "of", "on", "person",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}
_MONTH_NUMBERS = {name: index for index, name in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
), start=1)}


def _token(value: str) -> str:
    result = value.casefold().strip("'\"")
    if result.endswith("'s"):
        result = result[:-2]
    if len(result) > 5 and result.endswith("ing"):
        result = result[:-3]
    elif len(result) > 4 and result.endswith("ed"):
        result = result[:-2]
    if len(result) > 4 and result.endswith("s") and not result.endswith("ss"):
        result = result[:-1]
    return result


def _tokens(value: str) -> set[str]:
    return {
        _token(token)
        for token in re.findall(r"[\w'-]+", value)
        if len(token) > 1
    }


def _normalized_participants(frame: QueryFrame) -> set[str]:
    return {
        _token(value)
        for value in frame.participant_terms
        if _token(value) not in _GENERIC
    }


def _normalized_date_values(value: str) -> set[str]:
    lowered = value.casefold()
    values = {lowered.replace("/", "-")}
    month_names = "|".join(_MONTH_NUMBERS)
    patterns = (
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+((?:19|20)\d{{2}})\b",
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})[,]?\s+((?:19|20)\d{{2}})\b",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, lowered)
        if not match:
            continue
        if index == 0:
            month, day, year = match.groups()
        else:
            day, month, year = match.groups()
        values.add(f"{int(year):04d}-{_MONTH_NUMBERS[month]:02d}-{int(day):02d}")
    expanded = set(values)
    for item in values:
        match = re.search(r"\b((?:19|20)\d{2})(?:-(\d{2})(?:-(\d{2}))?)?\b", item)
        if not match:
            continue
        year, month, day = match.groups()
        expanded.add(year)
        if month:
            expanded.add(f"{year}-{month}")
        if month and day:
            expanded.add(f"{year}-{month}-{day}")
    return expanded


def _preferred_date_keys(values: list[str]) -> set[str]:
    normalized = (
        set().union(*(_normalized_date_values(value) for value in values))
        if values else set()
    )
    for pattern in (
        r"(?:19|20)\d{2}-\d{2}-\d{2}",
        r"(?:19|20)\d{2}-\d{2}",
        r"(?:19|20)\d{2}",
    ):
        selected = {value for value in normalized if re.fullmatch(pattern, value)}
        if selected:
            return selected
    return set()


def relation_focus_turn_ids(
    frame: QueryFrame,
    turns: list[Any],
    *,
    anchor_limit: int = 4,
    neighbor_radius: int = 1,
) -> list[str]:
    """Return a bounded raw-turn beam for the grammatical relation slots.

    The beam is topic agnostic: it scores speaker/participant alignment,
    relation and target-term coverage, action-family agreement, and explicit
    date agreement.  It then closes over a small dialogue window so answers
    stated in a follow-up turn remain attached to their question/event anchor.
    """

    if not turns:
        return []
    participants = _normalized_participants(frame)
    query_terms = (
        set(frame.content_terms)
        | {_token(value) for value in frame.temporal_terms}
    ) - participants - _GENERIC
    if not query_terms and not frame.explicit_dates:
        return []

    documents = [_tokens(str(getattr(turn, "text", ""))) for turn in turns]
    frequencies = {
        term: sum(term in document for document in documents)
        for term in query_terms
    }
    weights = {
        term: math.log((len(documents) + 1) / (count + 1)) + 1.0
        for term, count in frequencies.items()
    }
    date_keys = _preferred_date_keys(frame.explicit_dates)
    scoped_date = any(len(value) >= 7 for value in date_keys)
    scoped_session_exists = scoped_date and any(
        date_keys & _normalized_date_values(str(getattr(turn, "session_date", "")))
        for turn in turns
    )
    media_query = bool(re.search(
        r"\b(?:image|photo|photograph|picture|show|share)\w*\b",
        frame.raw_question, re.IGNORECASE,
    ))
    candidates: list[tuple[float, int, int, str, Any]] = []
    for turn, terms in zip(turns, documents):
        text = str(getattr(turn, "text", ""))
        covered = query_terms & terms
        speaker = _token(str(getattr(turn, "speaker", "")))
        speaker_key = _token(str(getattr(turn, "speaker_key", "")))
        speaker_match = int(
            bool(participants & {speaker, speaker_key})
        )
        mention_match = int(bool(participants & terms))
        action_match = action_family_overlap(frame.raw_question, text)
        session_dates = _normalized_date_values(
            str(getattr(turn, "session_date", ""))
        )
        date_match = int(bool(date_keys & session_dates))
        if scoped_session_exists and not date_match:
            continue
        media_match = int(
            media_query and bool(re.search(
                r"\b(?:media shared|shared image|shared photo)\b", text, re.IGNORECASE,
            ))
        )
        lexical = sum(weights.get(term, 1.0) for term in covered)
        joint = int(bool(covered) and (speaker_match or mention_match or not participants))
        score = (
            lexical
            + 3.0 * speaker_match
            + 0.75 * mention_match
            + 2.0 * action_match
            + 4.0 * date_match
            + 5.0 * media_match
            + 1.5 * joint
        )
        if score <= 0:
            continue
        candidates.append((
            score,
            len(covered),
            date_match,
            str(getattr(turn, "node_id", "")),
            turn,
        ))
    if not candidates:
        return []

    candidates.sort(reverse=True, key=lambda row: row[:4])
    session_counts: Counter[str] = Counter()
    anchors: list[Any] = []
    for _score, _coverage, _date, _node_id, turn in candidates:
        session_id = str(getattr(turn, "session_id", ""))
        if session_counts[session_id] >= 3:
            continue
        anchors.append(turn)
        session_counts[session_id] += 1
        if len(anchors) >= anchor_limit:
            break

    by_session: dict[str, list[Any]] = defaultdict(list)
    for turn in turns:
        by_session[str(getattr(turn, "session_id", ""))].append(turn)
    result: list[str] = []
    for anchor in anchors:
        session_rows = sorted(
            by_session[str(getattr(anchor, "session_id", ""))],
            key=lambda row: int(getattr(row, "turn_index", 0)),
        )
        position = next(
            (
                index for index, row in enumerate(session_rows)
                if getattr(row, "node_id", "") == getattr(anchor, "node_id", "")
            ),
            None,
        )
        if position is None:
            continue
        for index in range(
            max(0, position - neighbor_radius),
            min(len(session_rows), position + neighbor_radius + 1),
        ):
            node_id = str(getattr(session_rows[index], "node_id", ""))
            if node_id and node_id not in result:
                result.append(node_id)
    return result
