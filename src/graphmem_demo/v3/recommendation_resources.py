from __future__ import annotations

from collections import Counter

import re
from typing import Any, Callable

from .schema import QueryFrame


_REQUEST_MODIFIERS = {
    "suggest", "recommend", "complement", "current", "setup",
    "some", "suitable", "good", "help", "helpful", "find", "think", "tips",
}


_RESOURCE_CUE = re.compile(
    r"\b(?:i|we)\s+(?:(?:already|currently|just)\s+)?"
    r"(?:have|own|use|bought|purchased|got|downloaded|installed|booked|"
    r"carry|brought)\b|"
    r"\busing\s+(?:my|our)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\w\x27-]+", re.UNICODE)


def _tokens(value: str) -> set[str]:
    return {token.casefold().strip("\x27") for token in _WORD_RE.findall(value)}


def _speaker_compatible(frame: QueryFrame, turn: Any) -> bool:
    speaker = _tokens(str(getattr(turn, "speaker_key", "")))
    if re.search(r"\b(?:i|me|my|mine|we|our|ours)\b", frame.raw_question, re.IGNORECASE):
        return (
            str(getattr(turn, "transport_role", "")).casefold() == "user"
            or "user" in speaker
            or {"participant", "1"} <= speaker
        )
    if frame.participant_terms:
        return bool(set(frame.participant_terms) & speaker)
    return True


def recommendation_scope_session_ids(
    frame: QueryFrame,
    scope_rows: list[dict[str, Any]],
) -> list[str]:
    """Select a bounded coarse scope before any lossless resource scan."""

    if frame.requested_operation != "recommendation" or not scope_rows:
        return []
    anchor_terms = set(frame.content_terms) - _REQUEST_MODIFIERS
    covered = {
        str(row["session_id"]): anchor_terms & set(row.get("covered_terms", []))
        for row in scope_rows
    }
    document_frequency = Counter(
        term for terms in covered.values() for term in terms
    )
    anchor_scores = {
        session_id: sum(1.0 / document_frequency[term] for term in terms)
        for session_id, terms in covered.items()
    }
    max_anchor = max(anchor_scores.values(), default=0.0)
    max_coverage = max(float(row.get("query_coverage", 0.0)) for row in scope_rows)
    max_posterior = max(float(row.get("posterior", 0.0)) for row in scope_rows)
    eligible = [
        str(row["session_id"])
        for row in scope_rows
        if (
            anchor_scores[str(row["session_id"])] >= max_anchor - 1e-9
            if max_anchor > 0
            else float(row.get("query_coverage", 0.0)) >= max_coverage
        )
        and float(row.get("posterior", 0.0)) >= 0.60 * max_posterior
    ]
    return list(dict.fromkeys(eligible))[:3]


def resource_evidence_text(value: str, max_chars: int = 480) -> str:
    """Return bounded sentences that explicitly state existing resources."""

    normalized = value.replace("?", ".").replace("!", ".").replace(chr(10), ".")
    sentences = [segment.strip() for segment in normalized.split(".") if segment.strip()]
    selected = [sentence for sentence in sentences if _RESOURCE_CUE.search(sentence)]
    return " ".join(selected)[:max_chars]


def recommendation_resource_turn_ids(
    frame: QueryFrame,
    turns: list[Any],
    allowed_session_ids: list[str],
    *,
    semantic_similarity: Callable[[Any], float] | None = None,
    max_items: int = 4,
) -> list[str]:
    """Project existing resources only inside a bounded routed session set."""

    if frame.requested_operation != "recommendation":
        return []
    allowed = set(allowed_session_ids[:3])
    if not allowed:
        return []
    query_terms = set(frame.content_terms + frame.participant_terms)
    ranked: list[tuple[float, int, str]] = []
    for turn in turns:
        node_id = str(getattr(turn, "node_id", ""))
        session_id = str(getattr(turn, "session_id", ""))
        text = str(getattr(turn, "text", ""))
        if (
            not node_id
            or session_id not in allowed
            or not _speaker_compatible(frame, turn)
            or not _RESOURCE_CUE.search(text)
        ):
            continue
        overlap = len(query_terms & _tokens(text)) / max(1, len(query_terms))
        semantic = max(0.0, semantic_similarity(turn)) if semantic_similarity else 0.0
        ranked.append((
            0.60 * semantic + 0.40 * overlap,
            -int(getattr(turn, "turn_index", 0)),
            node_id,
        ))
    return [
        node_id for _score, _position, node_id in sorted(ranked, reverse=True)[:max_items]
    ]
