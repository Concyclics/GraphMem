from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta
from typing import Any, Callable

from .action_semantics import action_family_overlap
from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_DATE_RE = re.compile(
    r"\b((?:19|20)\d{2})[/.-](\d{1,2})[/.-](\d{1,2})\b"
)
_MONTH_BY_NAME = {
    name.casefold(): index
    for index, name in enumerate(calendar.month_name) if name
}
_KINSHIP_RE = re.compile(
    r"\b(?:aunt|aunts|brother|brothers|child|children|cousin|cousins|"
    r"daughter|daughters|father|fathers|grandchild|grandchildren|"
    r"grandfather|grandmother|grandparent|husband|mother|mothers|nephew|"
    r"niece|parent|parents|relative|relatives|sister|sisters|son|sons|"
    r"spouse|uncle|uncles|wife)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
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
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
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
        # LLM-extracted and benchmark-provided timestamps are untrusted input.
        # An invalid calendar date carries no usable ordering information.
        return None


def latest_state_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    evidence_time: Callable[[Any], str | None],
) -> dict[str, Any] | None:
    if frame.requested_operation not in {"latest", "state"}:
        return None
    attribute_terms = set(frame.content_terms) - {
        "latest", "recent", "state", "status", "now",
    }
    focus_match = re.search(
        r"\b(?:current|latest|most recent)\s+(.+?)(?:\s+(?:in|of|for|at|with)\b|[?]|$)",
        frame.raw_question.casefold(),
    )
    focus_terms = set(tokenize(focus_match.group(1))) if focus_match else set()
    documents = [set(tokenize(node_text(node))) for _kind, node, _score, _source in kept]
    document_frequency = {
        term: sum(term in document for document in documents)
        for term in attribute_terms
    }
    term_weight = {
        term: math.log((len(documents) + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    candidates: list[tuple[float, date, int, float, Any, str]] = []
    for kind, node, score, _source in kept:
        observed = _day(evidence_time(node))
        if observed is None:
            continue
        text = node_text(node)
        covered = attribute_terms & set(tokenize(text))
        if not covered:
            continue
        if focus_terms and not (focus_terms & covered):
            continue
        ranking_terms = focus_terms if focus_terms else covered
        specificity = sum(term_weight.get(term, 1.0) for term in (covered & ranking_terms))
        candidates.append((specificity, observed, len(covered), score, node, text))
    if not candidates:
        return None
    specificity, observed, coverage, _score, node, text = max(candidates)
    same_day = [
        item
        for item in candidates
        if item[0] == specificity and item[1] == observed
    ]
    return {
        "operation": "latest_state_from_local_dated_evidence",
        "observed_at": observed.isoformat(),
        "evidence": text[:480],
        "supporting_node_ids": list(dict.fromkeys(
            item[4].node_id for item in same_day
        ))[:6],
        "matched_attribute_terms": sorted(
            attribute_terms & set(tokenize(text))
        ),
    }


def _subject_alignment(frame: QueryFrame, node: Any) -> int:
    values = [
        str(getattr(node, "subject_key", "")),
        str(getattr(node, "subject", "")),
        str(getattr(node, "speaker_key", "")),
        *[str(value) for value in getattr(node, "participant_keys", [])],
    ]
    joined = " ".join(values).casefold()
    named = {value.casefold() for value in frame.participant_terms}
    if named:
        return 1 if any(value in joined for value in named) else -1
    if re.search(r"\b(?:i|me|my)\b", frame.raw_question.casefold()):
        if "participant 1" in joined or "user" in joined:
            return 1
        if "participant 2" in joined or "assistant" in joined:
            return -1
    return 0


def _turn_event_day(observed: date, text: str) -> date:
    lowered = text.casefold()
    numbered = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\s+ago\b",
        lowered,
    )
    if numbered:
        raw = numbered.group(1)
        amount = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
        factor = {"day": 1, "week": 7, "month": 30, "year": 365}[numbered.group(2)]
        delta_days = amount * factor
        if delta_days >= observed.toordinal():
            return observed
        return observed - timedelta(days=delta_days)
    if re.search(r"\byesterday\b", lowered):
        return observed - timedelta(days=1)
    named_month = re.search(
        r"\b(?:back\s+)?(?:in|during)\s+("
        + "|".join(_MONTH_BY_NAME)
        + r")\b",
        lowered,
    )
    if named_month:
        month = _MONTH_BY_NAME[named_month.group(1)]
        year = observed.year if month <= observed.month else observed.year - 1
        return date(year, month, min(observed.day, calendar.monthrange(year, month)[1]))
    vague = re.search(r"\b(?:a few|few)\s+(day|week|month|year)s?\s+ago\b", lowered)
    if vague:
        factor = {"day": 1, "week": 7, "month": 30, "year": 365}[vague.group(1)]
        return observed - timedelta(days=3 * factor)
    return observed


def _completed_action_alignment(frame: QueryFrame, text: str) -> int:
    lowered = text.casefold()
    if action_family_overlap(frame.raw_question, lowered) <= 0:
        return 0
    if not re.search(r"\b(?:did|ago|bought|purchased|acquired|received)\b", frame.raw_question.casefold()):
        return 0
    if re.search(
        r"\b(?:plan(?:s|ned)?|need(?:s|ed)?\s+to|want(?:s|ed)?\s+to|"
        r"decid(?:e|es|ed)\s+to|consider(?:s|ed|ing)?)\b",
        lowered,
    ) and not re.search(
        r"\b(?:bought|got|purchased|acquired|received|ordered|picked\s+up)\b",
        lowered,
    ):
        return -1
    if re.search(
        r"\b(?:bought|got|purchased|acquired|received|ordered|picked\s+up)\b",
        lowered,
    ):
        return 1
    return 0


def relative_time_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    question_date: str | None,
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
    evidence_time: Callable[[Any], str | None],
    semantic_similarity: Callable[[Any], float] | None = None,
) -> dict[str, Any] | None:
    reference = _day(question_date)
    if reference is None:
        return None
    match = re.search(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month|year)s?\s+ago\b",
        frame.raw_question.casefold(),
    )
    if not match:
        return None
    raw_amount = match.group(1)
    amount = (
        int(raw_amount)
        if raw_amount.isdigit()
        else 1 if raw_amount in {"a", "an"}
        else _NUMBER_WORDS[raw_amount]
    )
    unit = match.group(2)
    if unit == "day":
        target = reference - timedelta(days=amount)
        tolerance = 1
    elif unit == "week":
        target = reference - timedelta(days=7 * amount)
        tolerance = 4
    elif unit == "month":
        month_index = reference.year * 12 + reference.month - 1 - amount
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        target = date(
            year,
            month,
            min(reference.day, calendar.monthrange(year, month)[1]),
        )
        tolerance = 16
    else:
        year = reference.year - amount
        target = date(
            year,
            reference.month,
            min(reference.day, calendar.monthrange(year, reference.month)[1]),
        )
        tolerance = 32

    query_terms = set(frame.content_terms)
    requires_kinship = bool(
        re.search(r"\b(?:relative|relatives|family member|family members)\b", frame.raw_question.casefold())
    )
    candidates: list[tuple[int, int, int, float, Any, date, str]] = []
    raw_semantic_scores: dict[str, float] = {}
    for kind, node, score, _source in kept:
        observed = _day(evidence_time(node))
        if observed is None:
            continue
        text = node_text(node)
        if requires_kinship and _KINSHIP_RE.search(text) is None:
            continue
        if str(getattr(node, "modality", "")).casefold() in {
            "planned", "possible", "hypothetical"
        }:
            continue
        observed = _turn_event_day(observed, text)
        coverage = len(query_terms & set(tokenize(text)))
        semantic_score = (
            semantic_similarity(node) if semantic_similarity is not None else score
        )
        if coverage <= 0 and semantic_score <= 0.20:
            continue
        distance = abs((observed - target).days)
        specificity = (
            2 if kind in {"claim", "event", "operand", "event_frame"}
            else 1 if kind == "turn" else 0
        )
        action_match = action_family_overlap(frame.raw_question, text)
        raw_semantic_scores[getattr(node, "node_id", "")] = semantic_score
        candidates.append((-distance, specificity, semantic_score + float(action_match) + 2.0 * _subject_alignment(frame, node) + float(_completed_action_alignment(frame, text)), coverage, node, observed, text))
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda row: (row[0], row[1], row[2], row[3], getattr(row[4], "node_id", "")),
        reverse=True,
    )
    negative_distance, _specificity, _score, coverage, node, observed, text = ranked[0]
    distance = -negative_distance
    in_window = [row for row in ranked if -row[0] <= tolerance]
    subject_window = [
        row for row in in_window if _subject_alignment(frame, row[4]) > 0
    ] or in_window
    completed_window = [
        row for row in subject_window
        if _completed_action_alignment(frame, row[6]) >= 0
    ] or subject_window
    semantic_window = sorted(
        completed_window,
        key=lambda row: (
            row[2],
            row[3],
            row[1],
            raw_semantic_scores.get(getattr(row[4], "node_id", ""), 0.0),
            getattr(row[4], "node_id", ""),
        ),
        reverse=True,
    )
    if semantic_window:
        negative_distance, _specificity, _score, coverage, node, observed, text = semantic_window[0]
        distance = -negative_distance
    best_by_scope = {}
    for row in in_window:
        node_scope = tuple(getattr(row[4], "session_ids", []) or [row[4].node_id])
        best_by_scope.setdefault(node_scope, row)
    diverse = sorted(
        best_by_scope.values(),
        key=lambda row: (row[0], row[1], row[2], row[3], getattr(row[4], "node_id", "")),
        reverse=True,
    )
    candidate_node_ids = list(dict.fromkeys(
        [row[4].node_id for row in semantic_window[:8]]
        + [row[4].node_id for row in diverse]
        + [row[4].node_id for row in in_window]
    ))[:16]
    return {
        "operation": "relative_time_scope_from_local_evidence",
        "reference_date": reference.isoformat(),
        "expression": match.group(0),
        "target_date": target.isoformat(),
        "selected_evidence_date": observed.isoformat(),
        "distance_days": distance,
        "within_tolerance": distance <= tolerance,
        "matched_query_term_count": coverage,
        "action_alignment": action_family_overlap(frame.raw_question, text),
        "semantic_similarity": round(raw_semantic_scores.get(getattr(node, "node_id", ""), 0.0), 6),
        "supporting_node_ids": [node.node_id],
        "candidate_node_ids": candidate_node_ids,
        "evidence": text[:480],
    }



def relative_age_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    question_date: str | None,
    *,
    tokenize: Callable[[str], list[str]],
    node_text: Callable[[Any], str],
) -> dict[str, Any] | None:
    """Resolve how-many-units-ago from an event date anchored to its observation."""
    question = frame.raw_question.casefold()
    if re.search(r"\bhow old\b", question):
        event_terms = set(frame.content_terms) - {"old", "age", "year", "years"}
        age_pattern = re.compile(
            r"\b(?:at\s+age\s+|(?:i|he|she|they|we|[a-z]+)\s+(?:was|turned)\s+)"
            r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")(?:\s+years?\s+old)?\b",
            re.IGNORECASE,
        )
        age_candidates = []
        for kind, node, score, _source in kept:
            text = node_text(node)
            match = age_pattern.search(text)
            if match is None:
                continue
            coverage = len(event_terms & set(tokenize(text)))
            if event_terms and coverage <= 0:
                continue
            raw_age = match.group(1).casefold()
            age = int(raw_age) if raw_age.isdigit() else _NUMBER_WORDS[raw_age]
            specificity = 2 if kind in {"turn", "claim", "operand"} else 1
            age_candidates.append((coverage, specificity, score, node, age, text))
        if age_candidates:
            coverage, _specificity, _score, node, age, text = max(
                age_candidates,
                key=lambda row: (row[0], row[1], row[2], row[3].node_id),
            )
            return {
                "operation": "event_age_from_evidence_expression",
                "value": age,
                "unit": "years",
                "matched_query_term_count": coverage,
                "supporting_node_ids": [node.node_id],
                "source_turn_ids": list(getattr(node, "source_turn_ids", []))
                or [node.node_id],
                "evidence": text[:480],
                "complete": True,
            }
    unit_match = re.search(
        r"\bhow many\s+(days?|weeks?|months?|years?)\s+ago\b",
        question,
    )
    if unit_match is None and not re.search(r"\bhow long ago\b", question):
        return None
    unit = unit_match.group(1).rstrip("s") if unit_match is not None else None
    query_terms = set(frame.content_terms)
    event_terms = query_terms - {
        "ago", "how", "long", "many", "was", "were", "is", "are",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
    }
    expression = re.compile(
        r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+"
        r"(day|week|month|year)s?\s+ago\b",
        re.IGNORECASE,
    )
    direct_candidates: list[tuple[int, float, int, Any, re.Match[str]]] = []
    for kind, node, score, _source in kept:
        text = node_text(node)
        terms = set(tokenize(text))
        coverage = len(event_terms & terms)
        if event_terms and coverage <= 0:
            continue
        specificity = 2 if kind in {"turn", "claim", "operand"} else 1
        for expression_match in expression.finditer(text):
            expression_unit = expression_match.group(2).casefold()
            if unit is not None and expression_unit != unit:
                continue
            direct_candidates.append((
                coverage, score, specificity, node, expression_match,
            ))
    if direct_candidates:
        coverage, _score, _specificity, node, expression_match = max(
            direct_candidates,
            key=lambda row: (row[0], row[2], row[1], row[3].node_id),
        )
        raw_amount = expression_match.group(1).casefold()
        amount = int(raw_amount) if raw_amount.isdigit() else _NUMBER_WORDS[raw_amount]
        expression_unit = expression_match.group(2).casefold()
        return {
            "operation": "relative_age_from_evidence_expression",
            "value": f"{amount} {expression_unit}{'' if amount == 1 else 's'} ago",
            "amount": amount,
            "unit": expression_unit,
            "matched_query_term_count": coverage,
            "supporting_node_ids": [node.node_id],
            "evidence": node_text(node)[:480],
            "complete": True,
        }

    reference = _day(question_date)
    if reference is None or unit is None:
        return None
    candidates: list[tuple[float, date, Any, str, str]] = []
    for _kind, node, score, _source in kept:
        if not isinstance(node, OperandRecordV3):
            continue
        observed = _day(node.observed_at)
        event_value = (node.event_time or "").strip().casefold()
        event_day = _day(node.event_time)
        if event_day is None and observed is not None:
            if event_value == "today":
                event_day = observed
            elif event_value == "yesterday":
                event_day = observed - timedelta(days=1)
        if event_day is None:
            continue
        text = node_text(node)
        overlap = len(query_terms & set(tokenize(text))) / max(1, len(query_terms))
        if overlap <= 0:
            continue
        candidates.append((overlap + score, event_day, node, text, event_value))
    if not candidates:
        return None
    _rank, event_day, node, text, event_value = max(
        candidates, key=lambda item: (item[0], item[1], item[2].confidence)
    )
    elapsed_days = (reference - event_day).days
    if elapsed_days < 0:
        return None
    if unit == "day":
        value = elapsed_days
    elif unit == "week":
        value = elapsed_days / 7
    elif unit == "month":
        value = (reference.year - event_day.year) * 12 + reference.month - event_day.month
    else:
        value = reference.year - event_day.year
    return {
        "operation": "relative_age_from_typed_event",
        "unit": unit,
        "value": value,
        "elapsed_days": elapsed_days,
        "reference_date": reference.isoformat(),
        "event_date": event_day.isoformat(),
        "event_time_expression": event_value,
        "observed_at": node.observed_at,
        "operand_ids": [node.operand_id],
        "source_turn_ids": list(node.source_turn_ids),
        "evidence": text[:480],
        "complete": True,
    }
