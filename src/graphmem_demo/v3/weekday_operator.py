from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Callable

from .schema import QueryFrame


_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[/.-](\d{1,2})[/.-](\d{1,2})\b")
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _day(value: str | None) -> date | None:
    if not value:
        return None
    match = _DATE_RE.search(value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def weekday_scope_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    question_date: str | None,
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    evidence_time: Callable[[Any], str | None],
) -> dict[str, Any] | None:
    reference = _day(question_date)
    match = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        frame.raw_question.casefold(),
    )
    if reference is None or match is None:
        return None
    target_weekday = _WEEKDAYS[match.group(1)]
    delta = (reference.weekday() - target_weekday) % 7
    target = reference - timedelta(days=delta or 7)
    query_terms = set(frame.content_terms)
    companion_query = bool(re.search(
        r"\bwho\b.{0,60}\b(?:with|accompanied)\b|\bwith whom\b",
        frame.raw_question.casefold(),
    ))
    location_query = frame.requested_operation == "location" or bool(
        re.search(r"^\s*where\b", frame.raw_question.casefold())
    )
    candidates = []
    for kind, node, score, _source in kept:
        observed = _day(evidence_time(node))
        if observed is None:
            # Relative event labels such as "past" or "recently" retain an
            # exact observation/session anchor on projected nodes.
            observed = _day(getattr(node, "observed_at", None))
        if observed is None:
            continue
        text = node_text(node)
        coverage = len(query_terms & set(tokenize(text)))
        fine = kind in {"claim", "event", "operand", "event_frame"}
        if coverage <= 0 and not fine:
            continue
        distance = abs((observed - target).days)
        specificity = 2 if fine else 1 if kind == "turn" else 0
        slot_match = int(
            (companion_query and bool(re.search(r"\bwith\b", text, re.IGNORECASE)))
            or (location_query and bool(re.search(
                r"\b(?:at|held|in|located)\b", text, re.IGNORECASE,
            )))
        )
        event_assertion = int(bool(re.search(
            r"\b(?:attend|complete|join|participat|saw|visit|went)\w*\b",
            text, flags=re.IGNORECASE,
        )) and not bool(re.search(
            r"\b(?:consider|interest|plan|recommend|suggest|think)\w*\b",
            text, flags=re.IGNORECASE,
        )))
        candidates.append((-distance, slot_match, event_assertion, specificity, coverage, score, node, observed, text))
    if not candidates:
        return None
    negative_distance, _slot_match, _event_assertion, _specificity, coverage, _score, node, observed, text = max(
        candidates
    )
    distance = -negative_distance
    return {
        "operation": "weekday_scope_from_local_evidence",
        "reference_date": reference.isoformat(),
        "expression": match.group(0),
        "target_date": target.isoformat(),
        "selected_evidence_date": observed.isoformat(),
        "distance_days": distance,
        "within_tolerance": distance <= 1,
        "matched_query_term_count": coverage,
        "supporting_node_ids": [node.node_id],
        "evidence": text[:480],
    }
