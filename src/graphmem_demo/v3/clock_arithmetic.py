from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .schema import QueryFrame


_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}
_STOP = {
    "a", "an", "at", "by", "did", "do", "does", "get", "got", "i",
    "in", "on", "reach", "reached", "the", "time", "to", "what", "when",
}


def _terms(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in _STOP
    }


def _target_terms(question: str) -> set[str]:
    match = re.search(
        r"\b(?:reach(?:ed)?|arriv(?:e|ed)(?:\s+(?:at|in))?|g(?:e|o)t\s+to)\s+"
        r"(?:the\s+)?(.+?)(?:\s+(?:on|at|by|after|before)\b|[?]|$)",
        question.casefold(),
    )
    return _terms(match.group(1)) if match else set()


def _number(value: str) -> int | None:
    return int(value) if value.isdigit() else _NUMBER_WORDS.get(value.casefold())


def _duration_minutes(text: str) -> int | None:
    if not re.search(
        r"\b(?:took|take|travel|journey|drive|ride|walk|get|reach|arriv)\w*\b",
        text.casefold(),
    ):
        return None
    lowered = text.casefold()
    anchored = re.search(r"\b(?:took|take|spent)\b", lowered)
    scope = lowered
    if anchored is not None:
        scope = lowered[anchored.start():anchored.start() + 160]
        scope = re.split(
            r"\b(?:but|however|if|instead|prefer|so\s+if|would\s+like)\b",
            scope,
            maxsplit=1,
        )[0]
    total = 0
    found = False
    for raw, unit in re.findall(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)\s*(hours?|hrs?|minutes?|mins?)\b",
        scope,
    ):
        amount = _number(raw)
        if amount is None:
            continue
        total += amount * (60 if unit.startswith(("hour", "hr")) else 1)
        found = True
    return total if found and total > 0 else None


def _departure_clock(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"\b(?:i|we)\b.{0,80}\b(?:left|departed|set\s+out|headed\s+out|"
        r"started\s+(?:driving|walking|riding|travelling|traveling))\b"
        r".{0,80}?\b(?:at|around|about)\s+"
        r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b",
        text.casefold(),
    )
    if match is None:
        return None
    hour = int(match.group(1)) % 12
    minute = int(match.group(2) or 0)
    if match.group(3).replace(".", "") == "pm":
        hour += 12
    return hour, minute


def _format_clock(total_minutes: int) -> str:
    hour24, minute = divmod(total_minutes % (24 * 60), 60)
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def arrival_clock_hint(frame: QueryFrame, turns: list[Any]) -> dict[str, Any] | None:
    """Compute an arrival clock from a relation-bound departure and duration."""

    question = frame.raw_question.casefold()
    if not (
        re.search(r"\b(?:what\s+time|when)\b", question)
        and re.search(r"\b(?:reach|arriv|g(?:e|o)t\s+to)\w*\b", question)
    ):
        return None
    target = _target_terms(question)
    if not target:
        return None

    by_session: dict[str, list[Any]] = defaultdict(list)
    for turn in turns:
        by_session[str(getattr(turn, "session_id", ""))].append(turn)

    duration_rows = []
    departure_rows = []
    for session_id, rows in by_session.items():
        ordered = sorted(rows, key=lambda item: int(getattr(item, "turn_index", 0)))
        departure_rows.extend([
            (session_id, item, value)
            for item in ordered
            if (value := _departure_clock(str(getattr(item, "text", "")))) is not None
        ])
        matching = [item for item in ordered if target & _terms(str(getattr(item, "text", "")))]
        if not matching:
            continue
        duration_rows.extend([
            (session_id, item, value)
            for item in ordered
            if (value := _duration_minutes(str(getattr(item, "text", "")))) is not None
            and target & _terms(str(getattr(item, "text", "")))
        ])

    candidates = []
    weekdays = (
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday",
    )
    query_weekdays = {token for token in weekdays if token in question}
    for duration_session, duration_turn, duration in duration_rows:
        duration_text = str(getattr(duration_turn, "text", ""))
        backward_reference = bool(re.search(
            r"\b(?:last\s+time|earlier|previous(?:ly)?)\b", duration_text.casefold()
        ))
        for departure_session, departure_turn, (hour, minute) in departure_rows:
                departure_context = (
                    str(getattr(departure_turn, "session_date", "")) + " "
                    + str(getattr(departure_turn, "text", ""))
                ).casefold()
                date_match = int(bool(query_weekdays) and query_weekdays.issubset(
                    {token for token in weekdays if token in departure_context}
                ))
                same_session = duration_session == departure_session
                if not same_session and not (backward_reference and date_match):
                    continue
                endpoint = (hour * 60 + minute + duration) % (24 * 60)
                candidates.append((
                    int(same_session),
                    date_match,
                    len(target & _terms(str(getattr(duration_turn, "text", "")))),
                    endpoint,
                    f"{departure_session}->{duration_session}",
                    departure_turn,
                    duration_turn,
                    duration,
                ))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    top_score = candidates[0][:3]
    top = [item for item in candidates if item[:3] == top_score]
    endpoints = {item[3] for item in top}
    if len(endpoints) != 1:
        return {
            "operation": "arrival_clock_ambiguous",
            "value": "insufficient evidence to choose one departure-duration path",
            "source_turn_ids": [],
            "complete": True,
        }
    _, _, _, endpoint, session_id, departure_turn, duration_turn, duration = top[0]
    return {
        "operation": "arrival_clock_time",
        "value": _format_clock(endpoint),
        "duration_minutes": duration,
        "session_id": session_id,
        "source_turn_ids": list(dict.fromkeys([
            str(getattr(departure_turn, "node_id", "")),
            str(getattr(duration_turn, "node_id", "")),
        ])),
        "complete": True,
        "completion_basis": "relation_bound_departure_plus_duration",
    }
