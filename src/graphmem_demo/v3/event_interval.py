from __future__ import annotations

import calendar
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from .schema import QueryFrame


_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ),
        start=1,
    )
}
_START_RE = re.compile(
    r"\b(?:met|meet|received|was given|were given|gave|"
    r"start(?:ed)?|began|begin|took up|take up|joined)\b",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\b(?:married|finish(?:ed)?|complet(?:e|ed)|wrap(?:ped)?(?:\s+up)?|"
    r"ended|stopped|graduated)\b",
    re.IGNORECASE,
)
_STOP = {
    "about", "after", "and", "before", "did", "do", "for", "from", "get",
    "getting", "given", "her", "his", "how", "long", "on", "professor",
    "the", "their", "them", "to", "was", "were", "work", "worked",
}


def _tokens(value: str) -> set[str]:
    values: set[str] = set()
    for raw in re.findall(r"[\w'-]+", value):
        token = raw.casefold()
        if token in _STOP or len(token) <= 1:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        values.add(token)
    return values


def _observed_day(value: str | None) -> date | None:
    text = (value or "").casefold()
    iso = re.search(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        try:
            return date(*(int(part) for part in iso.groups()))
        except ValueError:
            return None
    natural = re.search(
        r"\b(\d{1,2})\s+([a-z]+),?\s+((?:19|20)\d{2})\b", text
    )
    if not natural or natural.group(2) not in _MONTHS:
        return None
    try:
        return date(
            int(natural.group(3)),
            _MONTHS[natural.group(2)],
            int(natural.group(1)),
        )
    except ValueError:
        return None


def _event_window(observed: date, text: str) -> tuple[date, date]:
    lowered = text.casefold()
    if re.search(r"\blast\s+week\b", lowered):
        return observed - timedelta(days=13), observed - timedelta(days=7)
    if re.search(r"\blast\s+month\b", lowered):
        month_index = observed.year * 12 + observed.month - 2
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        return (
            date(year, month, 1),
            date(year, month, calendar.monthrange(year, month)[1]),
        )
    if re.search(r"\byesterday\b", lowered):
        value = observed - timedelta(days=1)
        return value, value
    match = re.search(
        r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", lowered
    )
    if match:
        amount = int(match.group(1))
        scale = {"day": 1, "week": 7, "month": 30, "year": 365}[match.group(2)]
        value = observed - timedelta(days=amount * scale)
        return value, value
    return observed, observed


def event_lifecycle_duration_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Estimate a duration from a subject-bound start/end event lifecycle."""
    if frame.requested_operation != "duration":
        return None
    participants = {value.casefold() for value in frame.participant_terms}
    topic_terms = set(frame.content_terms) - participants - _STOP
    by_session: dict[str, list[Any]] = defaultdict(list)
    for turn in turns:
        by_session[str(getattr(turn, "session_id", ""))].append(turn)

    endpoints: list[dict[str, Any]] = []
    for session_turns in by_session.values():
        ordered = sorted(session_turns, key=lambda row: int(getattr(row, "turn_index", 0)))
        for position, turn in enumerate(ordered):
            speaker = str(
                getattr(turn, "speaker_key", "") or getattr(turn, "speaker", "")
            ).casefold()
            if participants and speaker not in participants:
                continue
            text = str(getattr(turn, "text", ""))
            is_start = bool(_START_RE.search(text))
            if re.search(
                r"\b(?:good|great|nice|glad|pleased)\s+to\s+meet\b",
                text,
                re.IGNORECASE,
            ):
                is_start = False
            is_end = bool(_END_RE.search(text))
            if not (is_start or is_end):
                continue
            observed = _observed_day(str(getattr(turn, "session_date", "")))
            if observed is None:
                continue
            window_text = " ".join(
                str(getattr(ordered[index], "text", ""))
                for index in range(max(0, position - 1), min(len(ordered), position + 2))
            )
            lower, upper = _event_window(observed, text)
            endpoints.append(
                {
                    "turn": turn,
                    "text": text,
                    "window_text": window_text,
                    "terms": _tokens(window_text),
                    "topic_overlap": len(topic_terms & _tokens(window_text)),
                    "lower": lower,
                    "upper": upper,
                    "is_start": is_start,
                    "is_end": is_end,
                }
            )
    starts = [row for row in endpoints if row["is_start"]]
    ends = [row for row in endpoints if row["is_end"]]
    pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for start in starts:
        for end in ends:
            if getattr(start["turn"], "node_id", None) == getattr(
                end["turn"], "node_id", None
            ):
                continue
            if start["lower"] >= end["upper"]:
                continue
            shared = start["terms"] & end["terms"]
            if end["topic_overlap"] <= 0:
                continue
            if start["topic_overlap"] <= 0 and not shared:
                continue
            union = start["terms"] | end["terms"]
            shared_ratio = len(shared) / max(1, len(union))
            score = (
                2.0 * min(start["topic_overlap"], end["topic_overlap"])
                + start["topic_overlap"]
                + end["topic_overlap"]
                + 4.0 * shared_ratio
            )
            pairs.append((score, start, end))
    if not pairs:
        return None
    score, start, end = max(
        pairs,
        key=lambda row: (
            row[0],
            row[1]["lower"],
            row[2]["upper"],
            str(getattr(row[1]["turn"], "node_id", "")),
        ),
    )
    lower_days = max(0, (end["lower"] - start["upper"]).days)
    upper_days = max(lower_days, (end["upper"] - start["lower"]).days)
    midpoint_days = round((lower_days + upper_days) / 2)
    if midpoint_days >= 45:
        unit = "months"
        value = max(1, round(midpoint_days / 30))
    elif midpoint_days >= 10:
        unit = "weeks"
        value = max(1, round(midpoint_days / 7))
    else:
        unit = "days"
        value = midpoint_days
    source_ids = [
        str(getattr(start["turn"], "node_id", "")),
        str(getattr(end["turn"], "node_id", "")),
    ]
    return {
        "operation": "event_lifecycle_duration",
        "value": value,
        "unit": unit,
        "approximate": start["lower"] != start["upper"] or end["lower"] != end["upper"],
        "lower_days": lower_days,
        "upper_days": upper_days,
        "start_evidence": start["text"][:500],
        "end_evidence": end["text"][:500],
        "source_turn_ids": source_ids,
        "pair_score": round(score, 6),
        "complete": True,
        "completion_basis": "subject_bound_start_end_lifecycle",
    }
