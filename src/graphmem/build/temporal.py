from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta

from ..domain import TemporalInterval


_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%Y/%m/%d %H:%M", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
    "%I:%M %p %d %B %Y", "%d %B %Y %H:%M", "%d %B %Y",
    "%B %d %Y %H:%M", "%B %d %Y", "%B %Y", "%Y",
)
_WEEKDAYS = {name.casefold(): index for index, name in enumerate(calendar.day_name)}
_TIME_PHRASE_RE = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm)?\s+(?:on\s+)?\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}(?:\s*\([^)]*\))?(?:\s+\d{1,2}:\d{2})?|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|"
    r"(?:about\s+|around\s+)?(?:a|an|one|\d+)\s+(?:day|week|month|year)s?\s+ago|"
    r"(?:for\s+)?(?:(?:just\s+under|almost|nearly|about|around)\s+)?"
    r"(?:a|an|one|\d+)\s+(?:day|week|month|year)s?(?:\s+now)?|"
    r"today|yesterday|tomorrow|last\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"next\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I)


def extract_time_expression(text: str) -> str | None:
    match = _TIME_PHRASE_RE.search(text or "")
    return " ".join(match.group(0).split()) if match else None


def _clean(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value, flags=re.I)
    value = re.sub(r"\b(?:on|at)\b", " ", value, flags=re.I)
    return " ".join(value.replace(",", " ").split())


def _parse_absolute(value: str) -> tuple[datetime, str] | None:
    clean = _clean(value)
    for fmt in _FORMATS:
        try:
            parsed = datetime.strptime(clean, fmt)
        except ValueError:
            continue
        if fmt == "%Y":
            return parsed, "year"
        if fmt == "%B %Y":
            return parsed, "month"
        return parsed, "second" if "%S" in fmt else "minute" if "%M" in fmt else "day"
    return None


def _end_for(start: datetime, precision: str) -> datetime:
    if precision == "year":
        return start.replace(year=start.year + 1) - timedelta(microseconds=1)
    if precision == "month":
        year, month = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
        return start.replace(year=year, month=month) - timedelta(microseconds=1)
    if precision == "day":
        return start + timedelta(days=1) - timedelta(microseconds=1)
    if precision == "minute":
        return start + timedelta(minutes=1) - timedelta(microseconds=1)
    return start


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def normalize_time(raw_text: str, observed_at: str | None,
                   anchor_turn_id: str) -> TemporalInterval:
    raw = " ".join(str(raw_text or "").split())
    absolute = _parse_absolute(raw)
    if absolute:
        start, precision = absolute
        return TemporalInterval(raw, _iso(start), _iso(_end_for(start, precision)), precision,
                                "absolute", anchor_turn_id, 1.0)

    anchor_row = _parse_absolute(observed_at or "")
    anchor = anchor_row[0] if anchor_row else None
    lowered = raw.casefold()
    if anchor:
        start: datetime | None = None; precision = "day"
        if lowered == "today": start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        elif lowered == "yesterday": start = (anchor - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif lowered == "tomorrow": start = (anchor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif lowered in _WEEKDAYS:
            delta = (_WEEKDAYS[lowered] - anchor.weekday()) % 7
            start = (anchor + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif re.fullmatch(r"(?:last|next)\s+(?:" + "|".join(_WEEKDAYS) + r")", lowered):
            direction, weekday = lowered.split()
            target = _WEEKDAYS[weekday]
            if direction == "last":
                delta = (anchor.weekday() - target) % 7 or 7
                start = (anchor - timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                delta = (target - anchor.weekday()) % 7 or 7
                start = (anchor + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif lowered in {"last week", "next week"}:
            monday = (anchor - timedelta(days=anchor.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            start = monday + timedelta(days=-7 if lowered == "last week" else 7); precision = "week"
        elif lowered in {"last month", "next month"}:
            offset = -1 if lowered == "last month" else 1
            month_index = anchor.year * 12 + anchor.month - 1 + offset
            start = datetime(month_index // 12, month_index % 12 + 1, 1); precision = "month"
        elif lowered in {"last year", "next year"}:
            start = datetime(anchor.year + (-1 if lowered == "last year" else 1), 1, 1); precision = "year"
        else:
            ago = re.fullmatch(
                r"(?:about\s+|around\s+)?(?:a|an|one|(\d+))\s+(day|week|month|year)s?\s+ago",
                lowered)
            if ago:
                count = int(ago.group(1) or 1); unit = ago.group(2)
                if count > {"day": 36500, "week": 5200, "month": 1200, "year": 100}[unit]:
                    return TemporalInterval(raw, None, None, "unknown", "unresolved",
                                            anchor_turn_id, 0.0)
                if unit == "day":
                    start = (anchor - timedelta(days=count)).replace(hour=0, minute=0, second=0, microsecond=0)
                elif unit == "week":
                    start = (anchor - timedelta(days=7 * count)).replace(hour=0, minute=0, second=0, microsecond=0)
                    precision = "week"
                elif unit == "month":
                    month_index = anchor.year * 12 + anchor.month - 1 - count
                    start = datetime(month_index // 12, month_index % 12 + 1, 1); precision = "month"
                else:
                    start = datetime(anchor.year - count, 1, 1); precision = "year"
            duration = re.fullmatch(
                r"(?:for\s+)?(?:(just\s+under|almost|nearly|about|around)\s+)?"
                r"(?:a|an|one|(\d+))\s+(day|week|month|year)s?(?:\s+now)?",
                lowered)
            if duration:
                qualifier = duration.group(1) or ""
                count = int(duration.group(2) or 1); unit = duration.group(3)
                if count > {"day": 36500, "week": 5200, "month": 1200, "year": 100}[unit]:
                    return TemporalInterval(raw, None, None, "unknown", "unresolved",
                                            anchor_turn_id, 0.0)
                if unit == "day":
                    start = (anchor - timedelta(days=count)).replace(hour=0, minute=0, second=0, microsecond=0)
                elif unit == "week":
                    start = (anchor - timedelta(days=7 * count)).replace(hour=0, minute=0, second=0, microsecond=0)
                    precision = "week"
                elif unit == "month":
                    month_index = anchor.year * 12 + anchor.month - 1 - count
                    start = datetime(month_index // 12, month_index % 12 + 1, 1); precision = "month"
                else:
                    # A conversational duration describes an approximate start,
                    # not a whole calendar year. Month precision avoids turning
                    # "just under a year" into a false exact date.
                    months = 12 * count
                    if qualifier in {"just under", "almost", "nearly"}:
                        months = max(1, months - 1)
                    month_index = anchor.year * 12 + anchor.month - 1 - months
                    start = datetime(month_index // 12, month_index % 12 + 1, 1); precision = "month"
        if start:
            end = (start + timedelta(days=7) - timedelta(microseconds=1)
                   if precision == "week" else _end_for(start, precision))
            return TemporalInterval(raw, _iso(start), _iso(end), precision, "relative",
                                    anchor_turn_id, 0.9)
    return TemporalInterval(raw, None, None, "unknown", "unresolved", anchor_turn_id, 0.0)


def observed_interval(observed_at: str | None, anchor_turn_id: str) -> TemporalInterval | None:
    parsed = _parse_absolute(observed_at or "")
    if not parsed:
        return None
    start, precision = parsed
    return TemporalInterval(str(observed_at), _iso(start), _iso(_end_for(start, precision)),
                            precision, "observed", anchor_turn_id, 1.0)
