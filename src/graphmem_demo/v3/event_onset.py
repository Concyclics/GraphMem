from __future__ import annotations

import calendar
from datetime import date, timedelta
import re
from typing import Any

from .schema import QueryFrame


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NUMBER = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_ONSET_QUERY_RE = re.compile(
    r"\b(?:start(?:ed)?|begin|began|resume(?:d)?|restart(?:ed)?|"
    r"take|took)\b",
    re.IGNORECASE,
)
_ONSET_EVIDENCE_RE = re.compile(
    r"\b(?:start(?:ed)?|begin|began|resume(?:d)?|restart(?:ed)?|"
    r"took\s+up|joined)\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    rf"\b(?:have|has|had|i['’]ve|we['’]ve|they['’]ve|he['’]s|she['’]s)?\s*"
    rf"been\b.{{0,100}}?\bfor\s+(?P<number>{_NUMBER})\s+"
    r"(?P<unit>day|week|month|year)s?\b(?:\s+now)?",
    re.IGNORECASE,
)
_NUMBERED_AGO_RE = re.compile(
    rf"\b(?P<number>{_NUMBER})\s+(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_VAGUE_AGO_RE = re.compile(
    r"\b(?:a few|few)\s+(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_STOP = {
    "adulthood", "again", "begin", "began", "classe", "did", "his",
    "join", "joined", "resume", "resumed", "restart", "restarted",
    "start", "started", "tak", "take", "took", "when",
}
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


def _tokens(value: str) -> set[str]:
    return {
        _token_key(token)
        for token in re.findall(r"[\w'-]+", value.replace("_", " "))
    }


def _observed_day(value: str | None) -> date | None:
    text = str(value or "").casefold()
    iso = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if iso:
        try:
            return date(*(int(part) for part in iso.groups()))
        except ValueError:
            return None
    natural = re.search(r"\b(\d{1,2})\s+([a-z]+),?\s+((?:19|20)\d{2})\b", text)
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


def _amount(raw: str) -> int:
    return int(raw) if raw.isdigit() else _NUMBER_WORDS[raw.casefold()]


def _subtract_calendar(observed: date, amount: int, unit: str) -> date:
    if unit == "day":
        return observed - timedelta(days=amount)
    if unit == "week":
        return observed - timedelta(days=7 * amount)
    if unit == "month":
        month_index = observed.year * 12 + observed.month - 1 - amount
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        return date(year, month, min(observed.day, calendar.monthrange(year, month)[1]))
    year = observed.year - amount
    return date(
        year,
        observed.month,
        min(observed.day, calendar.monthrange(year, observed.month)[1]),
    )


def _speaker_matches(frame: QueryFrame, turn: Any) -> bool:
    if not frame.participant_terms:
        return True
    speaker_terms = _tokens(str(
        getattr(turn, "speaker_key", "") or getattr(turn, "speaker", "")
    ))
    return bool(speaker_terms & {_token_key(value) for value in frame.participant_terms})


def _format_onset(
    onset: date,
    *,
    unit: str,
    vague: bool,
    observed: date,
) -> str:
    if vague:
        return (
            f"a few {unit}s before "
            f"{observed.strftime('%B')} {observed.day}, {observed.year}"
        )
    if unit in {"month", "year"}:
        return onset.strftime("%B %Y") if unit == "month" else str(onset.year)
    return onset.isoformat()


def event_onset_from_sources(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Resolve an event start from relative or duration-bearing lossless turns."""

    if frame.requested_operation != "date" or not _ONSET_QUERY_RE.search(
        frame.raw_question
    ):
        return None
    participant_terms = {_token_key(value) for value in frame.participant_terms}
    activity_terms = {
        _token_key(value) for value in frame.content_terms
        if _token_key(value) not in participant_terms and _token_key(value) not in _STOP
    }
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
        for position, turn in enumerate(ordered):
            if not _speaker_matches(frame, turn):
                continue
            observed = _observed_day(str(getattr(turn, "session_date", "")))
            if observed is None:
                continue
            text = str(getattr(turn, "text", ""))
            window = ordered[max(0, position - 2):position + 1]
            window_text = " ".join(str(getattr(item, "text", "")) for item in window)
            coverage = len(activity_terms & _tokens(window_text))
            if activity_terms and coverage <= 0:
                continue
            vague = _VAGUE_AGO_RE.search(text)
            numbered = _NUMBERED_AGO_RE.search(text)
            duration = _DURATION_RE.search(text)
            if _ONSET_EVIDENCE_RE.search(text) and (vague or numbered):
                match = vague or numbered
                assert match is not None
                unit = match.group("unit").casefold()
                amount = 3 if vague else _amount(match.group("number"))
                onset = _subtract_calendar(observed, amount, unit)
                kind = "relative_start_expression"
                direct = 2
            elif duration:
                unit = duration.group("unit").casefold()
                amount = _amount(duration.group("number"))
                onset = _subtract_calendar(observed, amount, unit)
                kind = "present_perfect_duration_backshift"
                direct = 1
                vague = None
            else:
                continue
            source_ids = list(dict.fromkeys(
                str(getattr(item, "node_id", "")) for item in window
                if activity_terms & _tokens(str(getattr(item, "text", "")))
                or item is turn
            ))
            result = {
                "operation": "event_onset_from_lossless_evidence",
                "value": _format_onset(
                    onset, unit=unit, vague=bool(vague), observed=observed
                ),
                "onset_date": onset.isoformat(),
                "anchor_date": observed.isoformat(),
                "expression": match.group(0) if (match := (vague or numbered or duration)) else "",
                "source_turn_ids": source_ids,
                "operand_ids": [],
                "evidence": [str(getattr(item, "text", "")) for item in window],
                "complete": True,
                "approximate": bool(vague) or unit in {"month", "year"},
                "completion_basis": kind,
            }
            score = (
                coverage,
                direct,
                len(activity_terms & _tokens(text)),
                observed,
                int(getattr(turn, "turn_index", -1)),
                str(getattr(turn, "node_id", "")),
            )
            candidates.append((score, result))
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]
