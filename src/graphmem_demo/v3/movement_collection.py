from __future__ import annotations

import re
from typing import Any

from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import QueryFrame


_TRAVEL_QUERY_RE = re.compile(
    r"\b(?:travel(?:s|ed|ing)?|trip(?:s)?|visit(?:s|ed|ing)?|"
    r"journey(?:s|ed|ing)?|go|went|gone|fly|flew|drive|drove|"
    r"tour(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_PRESENCE_PREDICATE_RE = re.compile(
    r"\b(?:travel(?:s|ed|ing)?|visit(?:s|ed|ing)?|went|gone|go|"
    r"return(?:s|ed|ing)?|arriv(?:e|es|ed|ing)|depart(?:s|ed|ing)?|"
    r"flew|fly|flown|drove|drive|driven|stay(?:s|ed|ing)?|"
    r"attend(?:s|ed|ing)?|tour(?:s|ed|ing)?|vacation(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_EXCLUDED_MODALITIES = {"planned", "possible", "conditional", "unknown"}
_INTENT_RE = re.compile(
    r"\b(?:want(?:s|ed|ing)?|hope(?:s|d|ing)?|dream(?:s|ed|ing)?|"
    r"plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|wish(?:es|ed|ing)?|"
    r"book(?:s|ed|ing)?|invite(?:s|d|ing)?|will|would|someday|future)\b",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:with|for|during|after|before|where|while|and then|and)\b",
    re.IGNORECASE,
)


def _same_subject(requested: str, observed: str) -> bool:
    requested_key = re.sub(r"\W+", "", requested.casefold())
    observed_key = re.sub(r"\W+", "", observed.casefold())
    if requested_key == observed_key:
        return True
    if min(len(requested_key), len(observed_key)) < 4:
        return False
    if abs(len(requested_key) - len(observed_key)) > 1:
        return False
    mismatches = 0
    left = right = 0
    while left < len(requested_key) and right < len(observed_key):
        if requested_key[left] == observed_key[right]:
            left += 1
            right += 1
            continue
        mismatches += 1
        if mismatches > 1:
            return False
        if len(requested_key) > len(observed_key):
            left += 1
        elif len(observed_key) > len(requested_key):
            right += 1
        else:
            left += 1
            right += 1
    return mismatches + (left < len(requested_key)) + (
        right < len(observed_key)
    ) <= 1


def _year_matches(item: Any, requested_years: set[str]) -> bool:
    if not requested_years:
        return True
    values = f"{item.event_time or ''} {item.observed_at or ''}"
    observed_years = set(re.findall(r"\b(?:19|20)\d{2}\b", values))
    return bool(requested_years & observed_years)


def _clean_location(value: str) -> str | None:
    text = value.strip(" \t,.;:-")
    boundary = _BOUNDARY_RE.search(text)
    if boundary:
        text = text[:boundary.start()].strip(" \t,.;:-")
    text = re.sub(
        r"^(?:a|an|the)\s+(?:city|town|village|region|country|state)\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    words = re.findall(r"[\w'-]+", text)
    if not words or len(words) > 6:
        return None
    return " ".join(words)


def _location(item: OperandRecordV3) -> str | None:
    object_text = item.object_text.strip()
    predicate = item.predicate_key.casefold()
    preposition = re.search(
        r"\b(?:in|at|to|from)\s+(.+)$", object_text, re.IGNORECASE
    )
    if preposition:
        return _clean_location(preposition.group(1))
    if re.search(
        r"\b(?:return(?:s|ed|ing)?|visit(?:s|ed|ing)?|"
        r"arriv(?:e|es|ed|ing)|depart(?:s|ed|ing)?)\b",
        predicate,
    ):
        return _clean_location(object_text)
    context = re.search(
        r"\b(?:in|at|to|from)\s+(.+)$",
        item.context_key,
        re.IGNORECASE,
    )
    return _clean_location(context.group(1)) if context else None


def _frame_location(item: EventFrameV3) -> str | None:
    match = re.search(
        r"\b(?:in|at|to|from)\s+(.+)", item.label, re.IGNORECASE
    )
    return _clean_location(match.group(1)) if match else None


def movement_location_collection_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    event_frames: list[EventFrameV3] | None = None,
) -> dict[str, Any] | None:
    """Collect locations where a subject was physically present.

    The closure is global over typed operands and excludes plans, other
    subjects, media-only locations, invitations, bookings, and mere mentions.
    """
    if (
        frame.requested_operation not in {"list", "location"}
        or not _TRAVEL_QUERY_RE.search(frame.raw_question)
        or not frame.participant_terms
    ):
        return None
    requested_subjects = {value.casefold() for value in frame.participant_terms}
    requested_years = set(
        re.findall(r"\b(?:19|20)\d{2}\b", frame.raw_question)
    )
    candidates: list[tuple[OperandRecordV3, str]] = []
    for item in operands:
        if item.polarity.casefold() != "positive":
            continue
        if item.modality.casefold() in _EXCLUDED_MODALITIES:
            continue
        if _INTENT_RE.search(
            f"{item.predicate_key} {item.object_text} {item.event_time or ''}"
        ):
            continue
        if not any(
            _same_subject(requested, item.subject_key)
            for requested in requested_subjects
        ):
            continue
        if not _PRESENCE_PREDICATE_RE.search(item.predicate_key):
            continue
        if not _year_matches(item, requested_years):
            continue
        location = _location(item)
        if location:
            candidates.append((item, location))
    grouped: dict[str, dict[str, Any]] = {}
    for item, value in candidates:
        key = re.sub(r"\W+", " ", value.casefold()).strip()
        row = grouped.setdefault(
            key,
            {
                "value": value,
                "operand_ids": [],
                "frame_ids": [],
                "source_turn_ids": [],
                "event_times": [],
            },
        )
        row["operand_ids"].append(item.operand_id)
        row["source_turn_ids"].extend(item.source_turn_ids)
        if item.event_time:
            row["event_times"].append(item.event_time)
    for item in event_frames or []:
        if item.status.casefold() not in {"asserted", "complete"}:
            continue
        if not any(
            _same_subject(requested, observed)
            for requested in requested_subjects
            for observed in item.participant_keys
        ):
            continue
        if not _year_matches(item, requested_years):
            continue
        if _INTENT_RE.search(f"{item.label} {item.event_time or ''}"):
            continue
        if not _PRESENCE_PREDICATE_RE.search(item.label):
            continue
        location = _frame_location(item)
        if not location:
            continue
        key = re.sub(r"\W+", " ", location.casefold()).strip()
        row = grouped.setdefault(
            key,
            {
                "value": location,
                "operand_ids": [],
                "frame_ids": [],
                "source_turn_ids": [],
                "event_times": [],
            },
        )
        row["frame_ids"].append(item.frame_id)
        row["source_turn_ids"].extend(item.source_turn_ids)
        if item.event_time:
            row["event_times"].append(item.event_time)
    if not grouped:
        return None
    values = [
        {
            **row,
            "operand_ids": list(dict.fromkeys(row["operand_ids"])),
            "frame_ids": list(dict.fromkeys(row["frame_ids"])),
            "source_turn_ids": list(dict.fromkeys(row["source_turn_ids"])),
            "event_times": list(dict.fromkeys(row["event_times"])),
        }
        for _key, row in sorted(grouped.items())
    ]
    return {
        "operation": "movement_location_collection",
        "values": [row["value"] for row in values],
        "groups": values,
        "operand_ids": [
            operand_id for row in values for operand_id in row["operand_ids"]
        ],
        "frame_ids": [
            frame_id for row in values for frame_id in row["frame_ids"]
        ],
        "source_turn_ids": list(dict.fromkeys(
            source_id for row in values for source_id in row["source_turn_ids"]
        )),
        # Grounded movement-location candidates are not a complete answer.
        # A query may request a containing region, time, activity, or relative order.
        "complete": False,
        "completion_basis": "global_subject_completed_movement_candidates",
    }
