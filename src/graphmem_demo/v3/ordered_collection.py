from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Callable

from .action_semantics import action_families
from .schema import QueryFrame
from .temporal_normalize import parse_datetime


_COMPLETED_MARKER_RE = re.compile(
    r"\b(?:already|attended|bought|completed|finished|flew|flown|"
    r"got\s+back|had|learned|participated|recently|returned|took|"
    r"tried|visited|went|worked)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_TRAVEL_COMPLETE_RE = re.compile(
    r"\b(?:i|we)(?:.ve|\s+have)?\b.{0,320}"
    r"(?:\bflew\b|\bflown\b|\bgot\s+back\b|\breturned\b|"
    r"\btook\b.{0,100}\bflight\b|\bhad\b.{0,180}\bflight\b)",
    re.IGNORECASE | re.DOTALL,
)
_PLAN_ONLY_RE = re.compile(
    r"\b(?:consider|considering|hope|might|plan|planning|want|will)\b",
    re.IGNORECASE,
)


def _tokens(value: str) -> set[str]:
    return {
        token.casefold().rstrip("s")
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", value)
    }


def ordered_action_collection_candidates(
    frame: QueryFrame,
    turns: list[Any],
    *,
    target_semantic_similarity: Callable[[Any], float],
) -> dict[str, Any] | None:
    """Expose the global lossless support for an open chronological collection.

    The fine graph can omit the answer entity while preserving the completed
    event (for example, a provider inferred from another clause in one turn).
    This operator deliberately returns an incomplete candidate closure so the
    single answer model performs only the remaining local binding/deduplication.
    """

    if frame.requested_operation != "ordering":
        return None
    match = re.search(
        r"\border\s+of\s+(?P<target>.+?)\s+i\s+(?P<action>.+?)\s+"
        r"from\s+earliest\s+to\s+latest",
        frame.raw_question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    target_phrase = match.group("target").strip()
    action_phrase = match.group("action").strip()
    requested_families = action_families(action_phrase)
    requested_terms = _tokens(action_phrase)
    target_terms = _tokens(target_phrase) - {
        "all", "different", "the", "type", "types",
    }
    if not requested_families and not requested_terms:
        return None

    rows: list[tuple[datetime, str, str]] = []
    for turn in turns:
        transport = str(getattr(turn, "transport_role", "")).casefold()
        speaker = str(getattr(turn, "speaker_key", "")).casefold()
        if transport == "assistant" or (
            transport and transport != "user"
            and speaker not in {"participant 1", "participant_1", "user"}
        ):
            continue
        text = str(getattr(turn, "text", ""))
        text_families = action_families(text)
        text_terms = _tokens(text)
        if not (
            requested_families & text_families
            or requested_terms & text_terms
        ):
            continue
        travel_completion = bool(
            "travel" in requested_families
            and _FIRST_PERSON_TRAVEL_COMPLETE_RE.search(text)
        )
        if "travel" in requested_families and not travel_completion:
            continue
        if not travel_completion and not _COMPLETED_MARKER_RE.search(text):
            continue
        if _PLAN_ONLY_RE.search(text) and not re.search(
            r"\b(?:already|flew|flown|got\s+back|had|recently|returned|took)\b",
            text,
            re.IGNORECASE,
        ):
            continue
        lexical_type = bool(target_terms & text_terms)
        semantic_type = max(0.0, target_semantic_similarity(turn)) >= 0.46
        if not travel_completion and not lexical_type and not semantic_type:
            continue
        observed = parse_datetime(str(getattr(turn, "session_date", "")))
        if observed is None:
            continue
        node_id = str(getattr(turn, "node_id", ""))
        rows.append((observed, node_id, text))

    if len(rows) < 2:
        return None
    rows.sort(key=lambda row: (row[0], row[1]))
    rows = rows[:16]
    return {
        "operation": "ordered_action_entity_candidates",
        "target_phrase": target_phrase,
        "action_phrase": action_phrase,
        "candidates": [
            {
                "date": observed.date().isoformat(),
                "source_turn_id": node_id,
                "evidence": text[:640],
            }
            for observed, node_id, text in rows
        ],
        "source_turn_ids": [node_id for _observed, node_id, _text in rows],
        "operand_ids": [],
        "complete": False,
        "completion_basis": "global_lossless_completed_action_candidate_scan",
    }
