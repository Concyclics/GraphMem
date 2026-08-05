from __future__ import annotations

import calendar
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from .catalog_schema import EventFrameV3
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
_PLAN_RE = re.compile(
    r"\b(?:plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|"
    r"schedule(?:s|d|ing)?|agree(?:s|d|ing)?)\b",
    re.IGNORECASE,
)
_STOP = {
    "a", "about", "an", "and", "did", "do", "does", "how", "many", "of",
    "on", "or", "the", "their", "them", "they", "time", "times", "to",
    "together", "was", "were", "with",
}
_MODALITY = {
    "agree", "intend", "plan", "planned", "planning", "schedule", "scheduled",
}
_VAGUE_TIME = {
    "", "future", "later", "someday", "soon", "tbd", "unknown",
}


def _stem(raw: str) -> str:
    token = raw.casefold().strip("'\"")
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        token
        for raw in re.findall(r"[\w'-]+", value)
        if (token := _stem(raw)) not in _STOP and len(token) > 1
    }


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _same_name(left: str, right: str) -> bool:
    left_key = re.sub(r"\W+", "", left.casefold())
    right_key = re.sub(r"\W+", "", right.casefold())
    if left_key == right_key:
        return True
    return (
        min(len(left_key), len(right_key)) >= 4
        and abs(len(left_key) - len(right_key)) <= 1
        and _edit_distance(left_key, right_key) <= 1
    )


def _participants_cover(
    requested: set[str], observed: set[str],
) -> bool:
    return all(
        any(_same_name(name, candidate) for candidate in observed)
        for name in requested
    )


def _day(value: str | None) -> date | None:
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
    if natural and natural.group(2) in _MONTHS:
        try:
            return date(
                int(natural.group(3)),
                _MONTHS[natural.group(2)],
                int(natural.group(1)),
            )
        except ValueError:
            return None
    return None


def _next_month(value: date) -> tuple[int, int]:
    month_index = value.year * 12 + value.month
    year, month_zero = divmod(month_index, 12)
    return year, month_zero + 1


def _time_identity(event_time: str | None, observed_at: str | None) -> str | None:
    lowered = (event_time or "").casefold().strip()
    observed = _day(observed_at)
    if lowered in _VAGUE_TIME or lowered.startswith("future after"):
        return None
    if observed and re.search(r"\bnext\s+month\b", lowered):
        year, month = _next_month(observed)
        return f"month:{year:04d}-{month:02d}"
    if observed and re.search(r"\bnext\s+week\b", lowered):
        target = observed + timedelta(days=7)
        year, week, _weekday = target.isocalendar()
        return f"week:{year:04d}-{week:02d}"
    explicit = _day(lowered)
    if explicit:
        return f"day:{explicit.isoformat()}"
    if observed and re.search(r"\b(?:monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)\b", lowered):
        weekday_name = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lowered,
        )
        assert weekday_name is not None
        target_weekday = list(calendar.day_name).index(
            weekday_name.group(1).title()
        )
        delta = (target_weekday - observed.weekday()) % 7 or 7
        return f"day:{(observed + timedelta(days=delta)).isoformat()}"
    # Keep an explicit non-vague normalized time phrase as an identity. It
    # may be a season, holiday, or benchmark-independent named interval.
    normalized = re.sub(r"\W+", " ", lowered).strip()
    return f"phrase:{normalized}" if normalized else None


def planned_event_identity_count(
    frame: QueryFrame,
    event_frames: list[EventFrameV3],
) -> dict[str, Any] | None:
    """Count distinct planned events, not repeated mentions of a plan.

    Identity is derived from participant coverage, action semantics and the
    resolved target time. Repeated references to the same target interval are
    merged across conversations. A vague follow-up in the same conversation
    inherits that conversation's single anchored target instead of becoming a
    spurious extra event.
    """
    if frame.requested_operation != "count" or not _PLAN_RE.search(
        frame.raw_question
    ):
        return None
    requested_participants = {
        value.casefold() for value in frame.participant_terms
    }
    action_terms = (
        _tokens(frame.raw_question)
        - {_stem(value) for value in requested_participants}
        - {_stem(value) for value in _MODALITY}
    )
    if not action_terms:
        return None

    candidates: list[dict[str, Any]] = []
    for item in event_frames:
        if item.status.casefold() != "planned":
            continue
        observed_participants = {
            value.casefold() for value in item.participant_keys
        }
        if requested_participants and not _participants_cover(
            requested_participants, observed_participants
        ):
            continue
        event_terms = _tokens(f"{item.label} {item.retrieval_text}")
        overlap = {
            query_term
            for query_term in action_terms
            if any(
                query_term == event_term
                or (
                    min(len(query_term), len(event_term)) >= 4
                    and abs(len(query_term) - len(event_term)) <= 1
                    and _edit_distance(query_term, event_term) <= 1
                )
                for event_term in event_terms
            )
        }
        if not overlap:
            continue
        candidates.append(
            {
                "frame": item,
                "time_identity": _time_identity(
                    item.event_time, item.observed_at
                ),
                "action_overlap": sorted(overlap),
            }
        )
    if not candidates:
        return None

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        session_ids = row["frame"].session_ids or [""]
        for session_id in session_ids:
            by_session[str(session_id)].append(row)
    for rows in by_session.values():
        anchored = {
            row["time_identity"]
            for row in rows if row["time_identity"] is not None
        }
        if len(anchored) == 1:
            inherited = next(iter(anchored))
            for row in rows:
                if row["time_identity"] is None:
                    row["time_identity"] = inherited

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        item = row["frame"]
        identity = row["time_identity"]
        if identity is None:
            session_key = ",".join(sorted(item.session_ids)) or item.frame_id
            identity = f"session:{session_key}"
        groups[identity].append(row)

    group_rows = []
    for identity, rows in sorted(groups.items()):
        frames = [row["frame"] for row in rows]
        group_rows.append(
            {
                "identity": identity,
                "frame_ids": [item.frame_id for item in frames],
                "session_ids": list(dict.fromkeys(
                    session_id
                    for item in frames for session_id in item.session_ids
                )),
                "source_turn_ids": list(dict.fromkeys(
                    source_id
                    for item in frames for source_id in item.source_turn_ids
                )),
                "event_times": list(dict.fromkeys(
                    item.event_time for item in frames if item.event_time
                )),
                "action_overlap": sorted(set().union(*(
                    set(row["action_overlap"]) for row in rows
                ))),
            }
        )
    source_turn_ids = list(dict.fromkeys(
        source_id
        for row in group_rows for source_id in row["source_turn_ids"]
    ))
    return {
        "operation": "planned_event_identity_count",
        "value": len(group_rows),
        "groups": group_rows,
        "frame_ids": [
            row["frame"].frame_id for row in candidates
        ],
        "source_turn_ids": source_turn_ids,
        "complete": True,
        "completion_basis": "global_typed_plan_identity_closure",
    }
