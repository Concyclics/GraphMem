from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import calendar
import re
from typing import Any, Callable

from .schema import QueryFrame, TurnNode


_MONTH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DEICTIC_TEMPORAL_CUTOFF_RE = re.compile(
    r"^\s*(?:right\s+)?(?:today|now|the\s+present|present|"
    r"current\s+(?:date|time)|question\s+date|this\s+(?:moment|time))"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def date_scope_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    node_text: Callable[[Any], str],
    node_session_id: Callable[[Any], str | None],
    query_overlap: Callable[[QueryFrame, str], float],
    compact_text: Callable[[QueryFrame, str, int], str],
) -> dict[str, Any] | None:
    """Bind an explicit date-bearing turn to a query-relevant local session."""
    if frame.requested_operation != "date":
        return None
    session_relevance: dict[str, float] = defaultdict(float)
    for _kind, node, score, _source in kept:
        session_id = node_session_id(node)
        if session_id:
            session_relevance[session_id] = max(
                session_relevance[session_id],
                10.0 * query_overlap(frame, node_text(node)) + score,
            )
    candidates: list[tuple[float, str, TurnNode, str]] = []
    for kind, node, score, _source in kept:
        if kind != "turn" or not isinstance(node, TurnNode):
            continue
        matches = list(_MONTH_DATE_RE.finditer(node.text))
        if not matches:
            continue
        local = query_overlap(frame, node.text)
        session = session_relevance.get(node.session_id, 0.0)
        for match in matches:
            candidates.append((
                4.0 * session + 10.0 * local + score,
                match.group(0),
                node,
                compact_text(frame, node.text, 360),
            ))
    if not candidates:
        return None
    _score, value, node, evidence = max(
        candidates, key=lambda row: (row[0], row[2].node_id, row[1])
    )
    return {
        "operation": "explicit_date_from_local_session",
        "value": value,
        "supporting_node_ids": [node.node_id],
        "evidence": evidence,
        "complete": True,
    }


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_MONTHS = {name.casefold(): index for index, name in enumerate(calendar.month_name) if name}


def _session_date(value: str) -> date | None:
    lowered = value.casefold()
    iso = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", lowered)
    if iso:
        try:
            return date(*map(int, iso.groups()))
        except ValueError:
            return None
    natural = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)[,]?\s+((?:19|20)\d{2})\b",
        lowered,
    )
    if not natural or natural.group(2) not in _MONTHS:
        return None
    try:
        return date(int(natural.group(3)), _MONTHS[natural.group(2)], int(natural.group(1)))
    except ValueError:
        return None


def before_after_relation_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Resolve a bounded before/after relation over packed event frames.

    The operator is domain independent: it binds the grammatical anchor clause,
    keeps only frames sharing a participant and the requested answer slot, and
    chooses the nearest qualifying frame on the requested temporal side.
    """

    if frame.requested_operation != "ordering":
        return None
    match = re.search(r"\b(before|after)\b\s+(.+)$", frame.raw_question, re.IGNORECASE)
    if match is None:
        return None
    relation, anchor_clause = match.group(1).casefold(), match.group(2)
    # A deictic cutoff defines a one-place time window, not a binary event
    # relation. Binding it to any frame that happens to mention today/now
    # creates a false anchor and can pull an unrelated long episode.
    if _DEICTIC_TEMPORAL_CUTOFF_RE.fullmatch(anchor_clause):
        return None
    word_pattern = re.compile(r"[a-z0-9]+", re.IGNORECASE)
    function_words = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
        "do", "does", "for", "from", "had", "has", "have", "he", "her",
        "him", "his", "in", "is", "it", "of", "on", "or", "she", "that",
        "the", "their", "them", "they", "this", "to", "was", "were", "with",
    }

    def terms(value: str) -> set[str]:
        result: set[str] = set()
        for raw in word_pattern.findall(value):
            token = raw.casefold()
            if token in function_words or len(token) <= 1:
                continue
            result.add(token)
            if len(token) > 4 and token.endswith("ies"):
                result.add(token[:-3] + "y")
            elif len(token) > 4 and token.endswith("ing"):
                result.add(token[:-3])
            elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                result.add(token[:-1])
        return result

    anchor_terms = terms(anchor_clause)
    if not anchor_terms:
        return None
    slot_match = re.match(
        r"^\s*(?:which|what)\s+(.+?)\s+(?:was|were|did|does|do|is|are|had|has|have)\b",
        frame.raw_question,
        re.IGNORECASE,
    )
    slot_terms = terms(slot_match.group(1)) if slot_match else set()
    participant_terms = {value.casefold() for value in frame.participant_terms}
    if not slot_terms:
        slot_terms = set(frame.content_terms) - anchor_terms - participant_terms

    turn_by_id = {
        node.node_id: node
        for kind, node, _score, _source in kept
        if kind == "turn" and isinstance(node, TurnNode)
    }
    frames = [
        node for kind, node, _score, _source in kept
        if kind == "event_frame"
    ]
    if len(frames) < 2:
        return None

    def frame_text(node: Any) -> str:
        return " ".join(str(value or "") for value in (
            getattr(node, "label", ""),
            getattr(node, "retrieval_text", ""),
            " ".join(getattr(node, "semantic_type_keys", []) or []),
            " ".join(getattr(node, "participant_keys", []) or []),
            " ".join(
                turn_by_id[source_id].text
                for source_id in (getattr(node, "source_turn_ids", []) or [])
                if source_id in turn_by_id
            ),
        ))

    def frame_day(node: Any) -> date | None:
        value = _session_date(str(getattr(node, "observed_at", "") or ""))
        if value is not None:
            return value
        for source_id in getattr(node, "source_turn_ids", []) or []:
            turn = turn_by_id.get(source_id)
            if turn is not None:
                value = _session_date(str(turn.session_date or ""))
                if value is not None:
                    return value
        return None

    frame_documents = {node.node_id: terms(frame_text(node)) for node in frames}
    anchor_frequencies = {
        term: sum(term in document for document in frame_documents.values())
        for term in anchor_terms
    }
    present_anchor_terms = {
        term for term, frequency in anchor_frequencies.items() if frequency > 0
    }
    if not present_anchor_terms:
        return None
    rarest_frequency = min(anchor_frequencies[term] for term in present_anchor_terms)
    required_anchor_terms = {
        term for term in present_anchor_terms
        if anchor_frequencies[term] == rarest_frequency
    }
    anchor_rows: list[tuple[int, float, float, int, Any, date, str]] = []
    for node in frames:
        text = frame_text(node)
        document = frame_documents[node.node_id]
        required_covered = len(required_anchor_terms & document)
        weighted_coverage = sum(
            1.0 / anchor_frequencies[term]
            for term in present_anchor_terms & document
        )
        day = frame_day(node)
        if required_covered <= 0 or day is None:
            continue
        completed = int(str(getattr(node, "status", "")).casefold() == "complete")
        anchor_rows.append((
            required_covered,
            weighted_coverage,
            query_overlap(frame, text),
            completed,
            node,
            day,
            text,
        ))
    if not anchor_rows:
        return None
    _required, _weighted, _overlap, _completed, anchor, anchor_day, anchor_text = max(
        anchor_rows,
        key=lambda row: (row[0], row[1], row[2], row[3], row[5], row[4].node_id),
    )
    anchor_participants = set(getattr(anchor, "participant_keys", []) or [])
    candidate_rows: list[tuple[int, int, float, Any, date, str]] = []
    for node in frames:
        if node.node_id == anchor.node_id:
            continue
        day = frame_day(node)
        if day is None:
            continue
        if relation == "before" and day >= anchor_day:
            continue
        if relation == "after" and day <= anchor_day:
            continue
        participants = set(getattr(node, "participant_keys", []) or [])
        if anchor_participants and participants and not (anchor_participants & participants):
            continue
        text = frame_text(node)
        covered = len(slot_terms & terms(text)) if slot_terms else 1
        if covered <= 0:
            continue
        distance = abs((anchor_day - day).days)
        candidate_rows.append((covered, -distance, query_overlap(frame, text), node, day, text))
    if not candidate_rows:
        return None
    covered, negative_distance, overlap, candidate, candidate_day, candidate_text = max(
        candidate_rows, key=lambda row: (row[0], row[1], row[2], row[4], row[3].node_id)
    )
    sources = list(dict.fromkeys([
        *(getattr(anchor, "source_turn_ids", []) or []),
        *(getattr(candidate, "source_turn_ids", []) or []),
    ]))
    return {
        "operation": "before_after_local_event_relation",
        "relation": relation,
        "answer_slot_terms": sorted(slot_terms),
        "anchor_event": {
            "node_id": anchor.node_id,
            "label": getattr(anchor, "label", ""),
            "date": anchor_day.isoformat(),
            "status": getattr(anchor, "status", "unknown"),
            "source_turn_ids": list(getattr(anchor, "source_turn_ids", []) or []),
            "evidence": anchor_text,
        },
        "nearest_qualifying_event": {
            "node_id": candidate.node_id,
            "label": getattr(candidate, "label", ""),
            "date": candidate_day.isoformat(),
            "status": getattr(candidate, "status", "unknown"),
            "source_turn_ids": list(getattr(candidate, "source_turn_ids", []) or []),
            "evidence": candidate_text,
        },
        "distance_days": -negative_distance,
        "matched_answer_slot_terms": covered,
        "query_overlap": round(float(overlap), 6),
        "source_turn_ids": sources,
        "complete": True,
    }


def calendar_window_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Bind an ordinal weekend inside an explicit month to dated raw turns."""
    lowered = frame.raw_question.casefold()
    if "weekend" not in lowered:
        return None
    ordinal = next((value for word, value in _ORDINALS.items() if re.search(rf"\b{word}\b", lowered)), None)
    if ordinal is None:
        return None
    month_value = next((value for value in frame.explicit_dates if re.fullmatch(r"(?:19|20)\d{2}-\d{2}", value)), None)
    if month_value is None:
        return None
    year, month = map(int, month_value.split("-"))
    first = date(year, month, 1)
    first_saturday = first + timedelta(days=(5 - first.weekday()) % 7)
    saturday = first_saturday + timedelta(days=7 * (ordinal - 1))
    if saturday.month != month:
        return None
    sunday = saturday + timedelta(days=1)
    planning_anchor = saturday - timedelta(days=1)
    participants = {value.casefold() for value in frame.participant_terms}
    candidates: list[tuple[float, TurnNode]] = []
    for kind, node, score, _source in kept:
        if kind != "turn" or not isinstance(node, TurnNode):
            continue
        observed = _session_date(node.session_date or "")
        if observed is None or not (planning_anchor <= observed <= sunday):
            continue
        speaker_match = float(not participants or node.speaker_key.casefold() in participants)
        mention_match = float(bool(participants & {value.casefold() for value in re.findall(r"[A-Za-z]+", node.text)}))
        candidates.append((
            3.0 * speaker_match + mention_match + 4.0 * query_overlap(frame, node.text) + score,
            node,
        ))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1].node_id), reverse=True)
    selected = [node for _score, node in candidates[:3]]
    return {
        "operation": "ordinal_weekend_window",
        "ordinal": ordinal,
        "month": month_value,
        "event_window_start": saturday.isoformat(),
        "event_window_end": sunday.isoformat(),
        "planning_anchor_start": planning_anchor.isoformat(),
        "source_turn_ids": [node.node_id for node in selected],
        "evidence": [node.text for node in selected],
        "complete": True,
        "completion_basis": "explicit_month_ordinal_weekend_dated_turns",
    }


def structured_section_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, Any] | None:
    """Resolve an ordinal artifact's named section from lossless local turns."""
    if frame.requested_operation not in {"lookup", "list"}:
        return None
    ordinal_words = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    }
    ordinal = next((
        value for word, value in ordinal_words.items()
        if re.search(rf"\b{word}\b", frame.raw_question.casefold())
    ), None)
    section_match = re.search(
        r"\b(?:for|of|in)\s+the\s+([a-z][a-z0-9 _-]{0,40}?)"
        r"(?:\s+(?:in|of|from)\b|[?]|$)",
        frame.raw_question.casefold(),
    )
    if ordinal is None or section_match is None:
        return None
    section = section_match.group(1).strip()
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(section)}\s*:\s*(?:\n\s*)?([^\n]+)"
    )
    candidates: list[tuple[str, int, float, TurnNode, str]] = []
    for kind, node, score, _source in kept:
        if kind != "turn" or not isinstance(node, TurnNode):
            continue
        match = pattern.search(node.text)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            continue
        candidates.append((
            node.session_date or "",
            node.turn_index,
            query_overlap(frame, node.text) + score,
            node,
            value,
        ))
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3].node_id))
    if len(candidates) < ordinal:
        return None
    _date, _turn, _score, node, value = candidates[ordinal - 1]
    return {
        "operation": "ordinal_structured_section",
        "ordinal": ordinal,
        "section": section,
        "value": value,
        "supporting_node_ids": [node.node_id],
        "complete": True,
    }
