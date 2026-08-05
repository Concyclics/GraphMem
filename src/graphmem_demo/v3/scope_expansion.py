from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .action_semantics import action_families, has_completed_participation
from .schema import QueryFrame


_SCOPE_GENERIC = {
    "all", "amount", "count", "different", "event", "events", "how", "many",
    "number", "related", "total",
}


def _speaker_matches(frame: QueryFrame, speaker_key: str) -> bool:
    speaker_terms = set(re.findall(r"[\w']+", speaker_key.casefold()))
    named = {
        value.casefold()
        for value in frame.participant_terms
        if value.casefold() not in {"i", "me", "my"}
    }
    if named:
        return bool(named & speaker_terms)
    if re.search(r"\b(?:i|me|my)\b", frame.raw_question.casefold()):
        return (
            "participant 2" not in speaker_key.casefold()
            and "assistant" not in speaker_key.casefold()
        )
    return True


def total_scope_candidates(
    frame: QueryFrame,
    operands: list[Any],
    turns: list[Any],
    *,
    tokenize: Callable[[str], list[str]],
    similarity: Callable[[Any], float],
    limit: int = 16,
) -> tuple[list[str], list[str]]:
    """Route a total over a bounded subject/unit inverted-index slice."""
    question = frame.raw_question.casefold()
    if not re.search(r"\b(?:total|in total|altogether)\b", question):
        return [], []
    query_terms = set(tokenize(question))
    query_units = query_terms & {
        "second", "minute", "hour", "day", "week", "month", "year",
        "item", "course", "dollar", "euro", "pound",
    }
    numeric = [
        item for item in operands
        if getattr(item, "quantity", None) is not None
        and getattr(item, "polarity", "positive") != "negative"
        and getattr(item, "modality", "asserted") not in {
            "planned", "possible", "hypothetical",
        }
        and (
            not query_units
            or query_units & set(tokenize(getattr(item, "unit", "")))
        )
        and _speaker_matches(frame, getattr(item, "subject_key", ""))
    ]
    candidate_sessions = {
        session_id
        for item in numeric
        for session_id in getattr(item, "session_ids", [])
    }
    if not candidate_sessions:
        return [], []
    terms_by_session: dict[str, set[str]] = defaultdict(set)
    for item in operands:
        for session_id in getattr(item, "session_ids", []):
            if session_id in candidate_sessions:
                terms_by_session[session_id].update(
                    tokenize(getattr(item, "retrieval_text", ""))
                )
    for turn in turns:
        if getattr(turn, "session_id", "") in candidate_sessions:
            terms_by_session[turn.session_id].update(
                tokenize(getattr(turn, "retrieval_text", "") or getattr(turn, "text", ""))
            )
    semantic = set(frame.content_terms) - query_units - _SCOPE_GENERIC
    scored = []
    for session_id in candidate_sessions:
        overlap = len(semantic & terms_by_session[session_id])
        if semantic and overlap <= 0:
            continue
        session_items = [
            item for item in numeric
            if session_id in getattr(item, "session_ids", [])
        ]
        score = overlap + max((similarity(item) for item in session_items), default=0.0)
        scored.append((score, session_id))
    sessions = [
        session_id for _score, session_id
        in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True)[:limit]
    ]
    selected = set(sessions)
    operand_ids = [
        item.operand_id
        for item in numeric
        if selected & set(getattr(item, "session_ids", []))
    ]
    return sessions, operand_ids


def lossless_event_turn_candidates(
    frame: QueryFrame,
    turns: list[Any],
    *,
    tokenize: Callable[[str], list[str]],
    similarity: Callable[[Any], float],
    limit: int = 12,
) -> list[str]:
    """Find completed participation turns missed by lossy event projection."""
    if frame.requested_operation != "count":
        return []
    if "attend" not in action_families(frame.raw_question):
        return []
    meaningful = set(frame.content_terms) - _SCOPE_GENERIC - {
        "attend", "go", "participate", "visit", "volunteer",
    }
    candidates = []
    for turn in turns:
        text = getattr(turn, "text", "")
        if not has_completed_participation(text):
            continue
        if not _speaker_matches(frame, getattr(turn, "speaker_key", "")):
            continue
        overlap = len(meaningful & set(tokenize(text)))
        if meaningful and overlap <= 0:
            continue
        candidates.append((2 * overlap + similarity(turn), turn.node_id))
    return [
        node_id for _score, node_id
        in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)[:limit]
    ]
