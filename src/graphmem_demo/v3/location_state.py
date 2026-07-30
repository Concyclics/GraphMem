from __future__ import annotations

import re
from datetime import date
from typing import Any

from .schema import QueryFrame


_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_MOVE_RE = re.compile(
    r"\b(?:leave(?:s|d|ing)?(?:\s+for)?|"
    r"travel(?:s|ed|ing)?(?:\s+to)?|"
    r"go(?:es|ne|ing)?(?:\s+to)?|"
    r"visit(?:s|ed|ing)?|"
    r"arriv(?:e|es|ed|ing)(?:\s+(?:in|at))?)\s+(?P<place>.+)",
    re.IGNORECASE,
)
_RETURN_RE = re.compile(r"\b(?:return|come\s+back|back\s+home)\b", re.IGNORECASE)
_TEMPORAL_OBJECT_RE = re.compile(
    r"\b(?:today|tomorrow|yesterday|day|week|month|year|morning|evening|"
    r"future|past|unknown|session)\b",
    re.IGNORECASE,
)


def _day(value: str | None) -> date | None:
    match = _ISO_DATE_RE.search(value or "")
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _destination(item: Any) -> str | None:
    predicate = str(getattr(item, "predicate_key", "")).replace("_", " ")
    match = _MOVE_RE.search(predicate)
    if match:
        value = match.group("place").strip(" ,.;:-")
        if value:
            return value
    if not _MOVE_RE.search(predicate + " placeholder"):
        return None
    value = str(getattr(item, "object_text", "")).strip(" ,.;:-")
    if not value or _TEMPORAL_OBJECT_RE.search(value):
        return None
    return value


def location_at_time_hint(
    frame: QueryFrame,
    operands: list[Any],
) -> dict[str, Any] | None:
    """Resolve an exact-date location from typed movement-state transitions."""
    if frame.requested_operation != "location":
        return None
    target = next(
        (
            _day(value)
            for value in frame.explicit_dates
            if _day(value) is not None
        ),
        None,
    )
    if target is None:
        return None
    participants = {value.casefold() for value in frame.participant_terms}
    movements: list[tuple[date, str, Any]] = []
    boundaries: list[tuple[date, Any]] = []
    for item in operands:
        subject = str(getattr(item, "subject_key", "")).casefold()
        if participants and subject not in participants:
            continue
        if str(getattr(item, "polarity", "positive")) == "negative":
            continue
        event_day = _day(str(getattr(item, "event_time", "")))
        if event_day is None:
            continue
        predicate = str(getattr(item, "predicate_key", ""))
        if _RETURN_RE.search(predicate):
            boundaries.append((event_day, item))
            continue
        destination = _destination(item)
        if destination:
            movements.append((event_day, destination, item))
    eligible = [row for row in movements if row[0] <= target]
    if not eligible:
        return None
    latest_day = max(row[0] for row in eligible)
    latest = [row for row in eligible if row[0] == latest_day]
    destinations = {
        re.sub(r"\W+", " ", row[1].casefold()).strip(): row[1]
        for row in latest
    }
    if len(destinations) != 1:
        return None
    boundary = min(
        (row for row in boundaries if row[0] > latest_day),
        default=None,
        key=lambda row: row[0],
    )
    if boundary is not None and target >= boundary[0]:
        return None
    value = next(iter(destinations.values()))
    source_ids = list(
        dict.fromkeys(
            source_id
            for _day_value, _destination_value, item in latest
            for source_id in getattr(item, "source_turn_ids", [])
        )
    )
    return {
        "operation": "location_at_time",
        "value": value,
        "target_date": target.isoformat(),
        "valid_from": latest_day.isoformat(),
        "valid_to": boundary[0].isoformat() if boundary is not None else None,
        "operand_ids": [
            str(getattr(item, "operand_id", getattr(item, "node_id", "")))
            for _day_value, _destination_value, item in latest
        ],
        "source_turn_ids": source_ids,
        "complete": bool(source_ids),
        "completion_basis": "typed_movement_state_interval",
    }
