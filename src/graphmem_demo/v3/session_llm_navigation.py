from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from graphmem_demo.clients import rough_token_count

from .structured_navigation import QueryIR, structured_navigation_summary


@dataclass(frozen=True)
class SessionSelection:
    selected_session_ids: tuple[str, ...]
    missing_slots: tuple[str, ...]
    resolved_slots: tuple[tuple[str, str], ...] = ()
    candidate_answer: str = ""
    confidence: float = 0.0
    parse_error: bool = False


def session_navigation_messages(
    *,
    question: str,
    question_date: str | None,
    session_rows: Iterable[dict[str, Any]],
    query_ir: QueryIR,
    max_prompt_rough_tokens: int = 3200,
    include_candidate: bool = False,
) -> tuple[list[dict[str, str]], list[str]]:
    """Ask an LLM to choose coarse sessions, never individual benchmark labels."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in session_rows:
        session_id = str(row.get("session_id") or "")
        if session_id:
            grouped.setdefault(session_id, []).append(row)
    valid_sessions = list(grouped)
    cards: list[str] = []
    for session_id, rows in grouped.items():
        date = next(
            (str(row.get("session_date")) for row in rows if row.get("session_date")),
            "unknown",
        )
        snippets = []
        for row in rows[:3]:
            text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
            snippets.append(text[:420])
        cards.append(
            f"SESSION={session_id} | date={date} | "
            f"query-focused lossless turns={' || '.join(snippets)}"
        )
    output_contract = (
        "JSON only with selected_session_ids (at most 8 IDs from the cards), "
        "missing_slots, resolved_slots (a compact object), candidate_answer "
        "(a concise evidence-derived proposal or empty string), and confidence "
        "(0 to 1). The proposal is auditable and must not use facts absent from "
        "the cards."
        if include_candidate
        else (
            "Return JSON only: selected_session_ids (at most 8 IDs from the "
            "cards) and missing_slots. Do not answer the question."
        )
    )
    system = (
        "Select the smallest set of conversation sessions that can answer the memory "
        "question. Use the fixed Query IR. For count, total, comparison, update, "
        "preference, or temporal questions, retain every session needed for operands, "
        "old/new states, or competing events. A query-focused snippet is only a route "
        "preview; the selected sessions will be expanded losslessly afterward. Return "
        + output_contract
    )
    prefix = (
        f"Question date: {question_date or 'unknown'}\n"
        f"Question: {question}\n"
        f"Query IR: {json.dumps(structured_navigation_summary(query_ir), ensure_ascii=False)}\n\n"
        "Coarse session cards:\n"
    )
    kept: list[str] = []
    for card in cards:
        if kept and rough_token_count(system + prefix + "\n".join([*kept, card])) > max_prompt_rough_tokens:
            break
        kept.append(card)
    kept_ids = [
        card.split(" |", 1)[0].removeprefix("SESSION=") for card in kept
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prefix + "\n".join(kept)},
    ], kept_ids


def parse_session_selection(
    text: str,
    valid_session_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    parsed = parse_session_navigation_result(text, valid_session_ids)
    return (
        parsed.selected_session_ids,
        parsed.missing_slots,
        parsed.parse_error,
    )


def parse_session_navigation_result(
    text: str,
    valid_session_ids: Iterable[str],
) -> SessionSelection:
    allowed = set(valid_session_ids)
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else "")
    except (json.JSONDecodeError, AttributeError):
        return SessionSelection((), (), parse_error=True)
    selected = tuple(dict.fromkeys(
        str(value)
        for value in payload.get("selected_session_ids", ())
        if str(value) in allowed
    ))[:8]
    missing = tuple(
        str(value)[:120] for value in payload.get("missing_slots", ())[:8]
    )
    raw_slots = payload.get("resolved_slots") or {}
    resolved_slots = (
        tuple(
            (str(key)[:80], str(value)[:240])
            for key, value in raw_slots.items()
        )[:12]
        if isinstance(raw_slots, dict)
        else ()
    )
    try:
        confidence = max(
            0.0, min(1.0, float(payload.get("confidence") or 0.0))
        )
    except (TypeError, ValueError):
        confidence = 0.0
    return SessionSelection(
        selected_session_ids=selected,
        missing_slots=missing,
        resolved_slots=resolved_slots,
        candidate_answer=str(payload.get("candidate_answer") or "")[:500],
        confidence=confidence,
        parse_error=False,
    )
