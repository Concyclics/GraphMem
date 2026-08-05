from __future__ import annotations

from datetime import datetime, timedelta
import re


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(
        r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", value
    )
    if match is not None:
        try:
            return datetime(*map(int, match.groups()))
        except ValueError:
            return None
    lowered = value.casefold()
    day_first = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
        + "|".join(sorted(_MONTHS, key=len, reverse=True))
        + r")\.?[,]?\s+((?:19|20)\d{2})\b",
        lowered,
    )
    month_first = re.search(
        r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True))
        + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+((?:19|20)\d{2})\b",
        lowered,
    )
    try:
        if day_first is not None:
            return datetime(
                int(day_first.group(3)), _MONTHS[day_first.group(2)],
                int(day_first.group(1)),
            )
        if month_first is not None:
            return datetime(
                int(month_first.group(3)), _MONTHS[month_first.group(1)],
                int(month_first.group(2)),
            )
    except ValueError:
        return None
    return None


def _closest_yearless_date(value: str, observed: datetime) -> datetime | None:
    """Resolve an English or numeric month/day against the nearest calendar year."""

    month = day = None
    named = re.search(
        r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True))
        + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        value.casefold(),
    )
    if named is not None:
        month = _MONTHS[named.group(1)]
        day = int(named.group(2))
    else:
        numeric = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?![/-]\d)", value)
        if numeric is not None:
            month, day = map(int, numeric.groups())
    if month is None or day is None:
        return None
    candidates = []
    for year in (observed.year - 1, observed.year, observed.year + 1):
        try:
            candidates.append(datetime(year, month, day))
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item - observed), item))


def resolve_evidence_time(
    event_time: str | None,
    observed_at: str | None,
) -> tuple[datetime | None, str]:
    """Resolve common relative event expressions against their source date.

    The resolver is intentionally domain-independent. Approximate expressions
    remain approximate in provenance, but receive a stable ordering point.
    """

    observed = parse_datetime(observed_at)
    explicit = parse_datetime(event_time)
    if explicit is not None:
        return explicit, "explicit"
    if observed is None:
        return None, "unresolved"
    lowered = str(event_time or "").casefold().strip()
    if not lowered or lowered in {"none", "unknown", "present"}:
        return observed, "observed_fallback"

    partial = _closest_yearless_date(str(event_time or ""), observed)
    if partial is not None:
        return partial, "anchored_partial_date"

    numbered = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\s+ago\b",
        lowered,
    )
    if numbered is not None:
        raw_amount = numbered.group(1)
        amount = (
            int(raw_amount) if raw_amount.isdigit()
            else 1 if raw_amount in {"a", "an"}
            else _NUMBER_WORDS[raw_amount]
        )
        scale = {"day": 1, "week": 7, "month": 30, "year": 365}[numbered.group(2)]
        return observed - timedelta(days=amount * scale), "anchored_relative"

    vague = re.search(r"\b(?:a few|few)\s+(day|week|month|year)s?\s+ago\b", lowered)
    if vague is not None:
        scale = {"day": 1, "week": 7, "month": 30, "year": 365}[vague.group(1)]
        return observed - timedelta(days=3 * scale), "anchored_approximate"

    if re.search(r"\byesterday\b", lowered):
        return observed - timedelta(days=1), "anchored_relative"

    weekday = next(
        (number for name, number in _WEEKDAYS.items() if re.search(rf"\b{name}\b", lowered)),
        None,
    )
    if weekday is not None and re.search(r"\blast\b", lowered):
        delta = (observed.weekday() - weekday) % 7
        return observed - timedelta(days=delta or 7), "anchored_weekday"

    last_period = re.search(r"\blast\s+(week|month|year)\b", lowered)
    if last_period is not None:
        scale = {"week": 7, "month": 30, "year": 365}[last_period.group(1)]
        return observed - timedelta(days=scale), "anchored_relative"

    return observed, "observed_fallback"
