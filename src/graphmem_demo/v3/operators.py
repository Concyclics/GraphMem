from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Callable

from .schema import QueryFrame


_DATE_RE = re.compile(
    r"\b((?:19|20)\d{2})[/.-](\d{1,2})[/.-](\d{1,2})\b"
)

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NUMBERS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?\s+((?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)


def _sentence_at(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    right_candidates = [
        value for value in (
            text.find(".", end), text.find("?", end), text.find("!", end)
        )
        if value >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left + 1:right].strip()


def _dated_spans(text: str, fallback: str | None) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    fallback_match = _DATE_RE.search(fallback or "")
    fallback_year = int(fallback_match.group(1)) if fallback_match else None
    for match in _DATE_RE.finditer(text):
        day = (
            f"{int(match.group(1)):04d}-"
            f"{int(match.group(2)):02d}-"
            f"{int(match.group(3)):02d}"
        )
        spans.append((day, _sentence_at(text, match.start(), match.end())))
    for match in _MONTH_DAY_RE.finditer(text):
        year = int(match.group(3)) if match.group(3) else fallback_year
        if year is None:
            continue
        day = f"{year:04d}-{_MONTH_NUMBERS[match.group(1).casefold()]:02d}-{int(match.group(2)):02d}"
        spans.append((day, _sentence_at(text, match.start(), match.end())))
    if not spans and fallback_match:
        spans.append((
            f"{int(fallback_match.group(1)):04d}-"
            f"{int(fallback_match.group(2)):02d}-"
            f"{int(fallback_match.group(3)):02d}",
            text,
        ))
    return list(dict.fromkeys(spans))



_DURATION_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
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
    return nearest if abs(value - nearest) < 1e-9 else round(value, 2)


def _explicit_relative_duration(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
) -> dict[str, Any] | None:
    number = r"(?:\d+|" + "|".join(_DURATION_NUMBERS) + r")"
    expression = re.compile(
        rf"\b(?:(for)\s+(?:the\s+)?({number})\s+"
        rf"(day|week|month|year)s?|({number})\s+"
        rf"(day|week|month|year)s?\s+(ago)|last\s+(day|week|month|year))\b",
        re.IGNORECASE,
    )
    query_terms = set(frame.content_terms) - {
        "ago", "been", "day", "days", "how", "long", "month", "months",
        "week", "weeks", "when", "year", "years",
    }
    candidates = []
    seen = set()
    for _kind, node, score, _source in kept:
        text = node_text(node)
        coverage = query_terms & set(tokenize(text))
        if not coverage:
            continue
        provenance = tuple(sorted(getattr(node, "source_turn_ids", []) or [node.node_id]))
        for match in expression.finditer(text):
            if match.group(7):
                amount, unit, style = 1, match.group(7).casefold(), "ago"
            elif match.group(1):
                raw_amount, unit, style = match.group(2), match.group(3).casefold(), "for"
                amount = int(raw_amount) if raw_amount.isdigit() else _DURATION_NUMBERS[raw_amount.casefold()]
            else:
                raw_amount, unit, style = match.group(4), match.group(5).casefold(), "ago"
                amount = int(raw_amount) if raw_amount.isdigit() else _DURATION_NUMBERS[raw_amount.casefold()]
            key = (provenance, amount, unit, style)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((len(coverage), score, amount, unit, style, node, match.group(0), coverage))
    if not candidates:
        return None
    two_endpoint = bool(re.search(r"\b(?:before|when)\b", frame.raw_question, re.IGNORECASE))
    if not two_endpoint:
        direct = [row for row in candidates if row[4] == "for"]
        if not direct:
            return None
        row = max(direct, key=lambda value: (value[0], value[1], value[2], value[5].node_id))
        return {
            "operation": "explicit_relative_duration",
            "value": row[2], "unit": row[3],
            "expressions": [row[6]],
            "supporting_node_ids": [row[5].node_id],
            "provenance_complete": True,
        }
    pairs = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if left[3] != right[3] or left[2] == right[2]:
                continue
            union = left[7] | right[7]
            balance = min(left[0], right[0])
            complementary = int({left[4], right[4]} == {"for", "ago"})
            pairs.append((len(union), balance, complementary, left[1] + right[1], left, right))
    if not pairs:
        return None
    _union, _balance, _complementary, _score, left, right = max(pairs)
    return {
        "operation": "explicit_relative_duration",
        "value": abs(left[2] - right[2]), "unit": left[3],
        "expressions": [left[6], right[6]],
        "supporting_node_ids": list(dict.fromkeys([left[5].node_id, right[5].node_id])),
        "provenance_complete": True,
    }


def _consecutive_sequence_since(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    question_date: str | None,
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    evidence_time: Callable[[Any], str | None],
) -> dict[str, Any] | None:
    """Measure from the end of a query-bound consecutive event sequence."""
    question = frame.raw_question.casefold()
    if not (
        re.search(r"\bsince\b", question)
        and re.search(r"\b(?:consecutive|in a row)\b", question)
    ):
        return None
    sequence_length = None
    for word, number in _DURATION_NUMBERS.items():
        if re.search(rf"\b{word}\b", question):
            sequence_length = number
            break
    if sequence_length is None:
        match = re.search(r"\b(\d+)\b", question)
        sequence_length = int(match.group(1)) if match else None
    if sequence_length is None or sequence_length < 2:
        return None
    reference_match = _DATE_RE.search(question_date or "")
    if not reference_match:
        return None
    reference = date(
        int(reference_match.group(1)), int(reference_match.group(2)),
        int(reference_match.group(3)),
    )
    target_terms = set(frame.content_terms) - {
        "consecutive", "event", "events", "in", "participate", "participated",
        "pass", "passed", "row", "since", "time", "times",
    }
    if not target_terms:
        return None
    groups: dict[date, dict[str, Any]] = defaultdict(
        lambda: {"coverage": set(), "node_ids": [], "source_turn_ids": [], "evidence": []}
    )
    completed_pattern = re.compile(
        r"\b(?:attend|attended|complete|completed|did|joined|participat\w*|took part|volunteer\w*)\b",
        re.IGNORECASE,
    )
    for kind, node, _score, _source in kept:
        if kind not in {"claim", "event", "event_frame", "operand", "turn"}:
            continue
        if str(getattr(node, "modality", "asserted")).casefold() in {
            "planned", "possible", "hypothetical", "conditional",
        }:
            continue
        if str(getattr(node, "status", "asserted")).casefold() in {
            "planned", "possible", "cancelled",
        }:
            continue
        if str(getattr(node, "polarity", "positive")).casefold() == "negative":
            continue
        text = node_text(node)
        terms = set(tokenize(text))
        coverage = target_terms & terms
        if not coverage or not completed_pattern.search(text):
            continue
        raw_time = str(getattr(node, "event_time", "") or "")
        match = _DATE_RE.search(raw_time)
        if match:
            event_day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        else:
            fallback = evidence_time(node)
            spans = _dated_spans(text, fallback)
            if not spans:
                continue
            event_day = date.fromisoformat(spans[0][0])
        row = groups[event_day]
        row["coverage"].update(coverage)
        row["node_ids"].append(node.node_id)
        row["source_turn_ids"].extend(
            getattr(node, "source_turn_ids", []) or ([node.node_id] if kind == "turn" else [])
        )
        row["evidence"].append(text[:360])
    if len(groups) < sequence_length:
        return None
    candidates = []
    days = sorted(groups)
    for start in days:
        run = [start + timedelta(days=offset) for offset in range(sequence_length)]
        if not all(day in groups for day in run):
            continue
        coverage = set().union(*(groups[day]["coverage"] for day in run))
        candidates.append((len(coverage), run[-1], run))
    if not candidates:
        return None
    _coverage, endpoint, run = max(candidates, key=lambda row: (row[0], row[1]))
    elapsed_days = (reference - endpoint).days
    if elapsed_days < 0:
        return None
    unit = _requested_duration_unit(frame.raw_question)
    if unit == "month":
        value: int | float = (reference.year - endpoint.year) * 12 + reference.month - endpoint.month
    elif unit == "year":
        value = reference.year - endpoint.year
    else:
        value = _convert_elapsed_days(elapsed_days, unit)
    return {
        "operation": "duration_since_consecutive_event_sequence",
        "value": value,
        "unit": unit,
        "elapsed_days": elapsed_days,
        "sequence_dates": [day.isoformat() for day in run],
        "reference_date": reference.isoformat(),
        "supporting_node_ids": list(dict.fromkeys(
            node_id for day in run for node_id in groups[day]["node_ids"]
        ))[:12],
        "source_turn_ids": list(dict.fromkeys(
            source for day in run for source in groups[day]["source_turn_ids"]
        ))[:12],
        "provenance_complete": True,
    }


def duration_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    evidence_time: Callable[[Any], str | None],
    query_overlap: Callable[[QueryFrame, str], float],
    question_date: str | None = None,
) -> dict[str, Any] | None:
    """Compute a duration from two query-covering dated evidence clusters.

    This is a local operator over the packed evidence.  It does not scan global
    memory and does not depend on benchmark labels or gold sessions.
    """

    if frame.requested_operation != "duration":
        return None
    explicit = _explicit_relative_duration(
        frame, kept, tokenize=tokenize, node_text=node_text
    )
    if explicit is not None:
        return explicit
    sequence = _consecutive_sequence_since(
        frame, kept, question_date, tokenize=tokenize, node_text=node_text,
        evidence_time=evidence_time,
    )
    if sequence is not None:
        return sequence
    query_terms = set(frame.content_terms)
    # These describe the arithmetic request, not either endpoint.
    endpoint_terms = query_terms - {
        "day", "days", "pass", "duration", "difference", "between",
    }
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"covered_terms": set(), "supporting_node_ids": [], "evidence": []}
    )
    for _kind, node, score, _source in kept:
        observed_at = evidence_time(node)
        text = node_text(node)
        if query_overlap(frame, text) <= 0:
            continue
        event_text = str(getattr(node, "event_time", "") or "")
        event_match = _DATE_RE.search(event_text)
        if hasattr(node, "event_time") and not event_match:
            if not re.search(r"\b(?:today|yesterday)\b", event_text.casefold()):
                continue
            spans = _dated_spans(text, observed_at)
        elif event_match:
            spans = [(
                f"{int(event_match.group(1)):04d}-{int(event_match.group(2)):02d}-{int(event_match.group(3)):02d}",
                text,
            )]
        else:
            spans = _dated_spans(text, observed_at)
        for day, local_text in spans:
            covered = endpoint_terms & set(tokenize(local_text))
            if not covered:
                continue
            row = groups[day]
            row["covered_terms"].update(covered)
            row["supporting_node_ids"].append(node.node_id)
            row["evidence"].append((len(covered), score, local_text[:360]))

    if (
        groups
        and re.search(r"\bago\b", frame.raw_question, re.IGNORECASE)
        and question_date
    ):
        reference_match = _DATE_RE.search(question_date)
        if reference_match:
            reference = date(
                int(reference_match.group(1)),
                int(reference_match.group(2)),
                int(reference_match.group(3)),
            )
            eligible = [
                (day, row)
                for day, row in groups.items()
                if date.fromisoformat(day) <= reference
            ]
            if eligible:
                event_day, row = max(
                    eligible,
                    key=lambda item: (
                        len(item[1]["covered_terms"]),
                        max(item[1]["evidence"], default=(0, 0.0, ""))[:2],
                        item[0],
                    ),
                )
                elapsed_days = (
                    reference - date.fromisoformat(event_day)
                ).days
                requested_unit = _requested_duration_unit(frame.raw_question)
                return {
                    "operation": "duration_since_bound_event",
                    "event_date": event_day,
                    "reference_date": reference.isoformat(),
                    "elapsed_days": elapsed_days,
                    "inclusive_days": elapsed_days + 1,
                    "value": _convert_elapsed_days(elapsed_days, requested_unit),
                    "unit": requested_unit,
                    "supporting_node_ids": list(dict.fromkeys(
                        row["supporting_node_ids"]
                    ))[:6],
                    "provenance_complete": True,
                }
    if len(groups) < 2:
        return None
    candidates: list[tuple[int, int, str, str]] = []
    dates = sorted(groups)
    for index, left in enumerate(dates):
        for right in dates[index + 1 :]:
            union = groups[left]["covered_terms"] | groups[right]["covered_terms"]
            balance = min(
                len(groups[left]["covered_terms"]),
                len(groups[right]["covered_terms"]),
            )
            candidates.append((len(union), balance, left, right))
    if not candidates:
        return None
    _coverage, _balance, left, right = max(candidates)
    left_date = date.fromisoformat(left)
    right_date = date.fromisoformat(right)
    elapsed_days = abs((right_date - left_date).days)
    requested_unit = _requested_duration_unit(frame.raw_question)

    def endpoint(value: str) -> dict[str, Any]:
        group = groups[value]
        best_evidence = max(group["evidence"], default=(0, 0.0, ""))
        return {
            "date": value,
            "covered_terms": sorted(group["covered_terms"]),
            "supporting_node_ids": list(dict.fromkeys(group["supporting_node_ids"]))[:6],
            "evidence": best_evidence[2],
        }

    return {
        "operation": "duration_from_local_dated_evidence",
        "left": endpoint(left),
        "right": endpoint(right),
        "elapsed_days": elapsed_days,
        "inclusive_days": elapsed_days + 1,
        "value": _convert_elapsed_days(elapsed_days, requested_unit),
        "unit": requested_unit,
        "provenance_complete": True,
    }
