from __future__ import annotations

import re
from typing import Any

from .schema import QueryFrame


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_IRREGULAR = {
    "chose": "choose",
    "named": "name",
    "preferred": "prefer",
    "said": "say",
    "told": "tell",
}
_QUESTION_FUNCTION = {
    "a", "an", "and", "are", "as", "be", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "my",
    "of", "on", "or", "our", "the", "their", "they", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "with", "you",
    "your",
}
_INTERROGATIVE = re.compile(
    r"(?:\?|^\s*(?:what|which|who|where|when|why|how|do|does|did|is|are|was|were)\b)",
    re.IGNORECASE,
)
_EXPLICIT_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:called|named|titled)\s+[\"“'](?P<value>[^\"”']{1,120})[\"”']",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:called|named|titled)\s+(?P<value>[A-Z][^.!?]{0,100})",
    ),
    re.compile(
        r"\b(?:my\s+)?(?:answer|choice|favorite|favourite|preference)\s+"
        r"(?:is|was|would be)\s+[\"“']?(?P<value>[^\"”'.!?][^.!?]{0,100})",
        re.IGNORECASE,
    ),
)


def _lemma(value: str) -> str:
    token = value.casefold().strip("'\"")
    if token.endswith("'s"):
        token = token[:-2]
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            stem = stem[:-1]
        return _IRREGULAR.get(stem, stem)
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        if stem.endswith("r"):
            stem = stem[:-1]
        return _IRREGULAR.get(stem, stem)
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(text: str) -> set[str]:
    return {
        term
        for term in (_lemma(match.group(0)) for match in _WORD_RE.finditer(text))
        if len(term) > 1 and term not in _QUESTION_FUNCTION
    }


def _explicit_value(text: str) -> str | None:
    for pattern in _EXPLICIT_VALUE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = re.sub(r"\s+", " ", match.group("value")).strip(" \t\r\n,;:-\"'")
        value = re.split(
            r"\s+\b(?:and|because|but|so|which|who|that)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" \t\r\n,;:-\"'")
        if 1 <= len(value.split()) <= 14:
            return value
    return None


def dialogue_answer_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Resolve an explicit answer through a historical next-turn relation.

    The operator is dataset- and topic-neutral. It matches a memory question
    to the current lookup, follows only the next turn in the same dialogue,
    enforces the requested named speaker when available, and returns only an
    explicitly named value. It never treats general topical proximity as an
    answer relation.
    """

    if frame.requested_operation != "lookup":
        return None
    participant_terms = {_lemma(value) for value in frame.participant_terms}
    query_terms = _terms(frame.raw_question) - participant_terms
    if len(query_terms) < 2:
        return None
    by_session: dict[str, list[Any]] = {}
    for turn in turns:
        by_session.setdefault(str(getattr(turn, "session_id", "")), []).append(turn)
    ranked: list[tuple[tuple[Any, ...], Any, Any, str]] = []
    for session_turns in by_session.values():
        ordered = sorted(
            session_turns,
            key=lambda item: (
                int(getattr(item, "turn_index", -1)),
                str(getattr(item, "node_id", "")),
            ),
        )
        for question_turn, answer_turn in zip(ordered, ordered[1:]):
            question_text = str(getattr(question_turn, "text", ""))
            if not _INTERROGATIVE.search(question_text):
                continue
            memory_terms = _terms(question_text)
            overlap = query_terms & memory_terms
            coverage = len(overlap) / max(1, len(query_terms))
            if len(overlap) < 2 or coverage < 0.5:
                continue
            answer_speaker = str(
                getattr(answer_turn, "speaker_key", "")
                or getattr(answer_turn, "speaker", "")
            )
            answer_speaker = _lemma(answer_speaker)
            if participant_terms and answer_speaker not in participant_terms:
                continue
            value = _explicit_value(str(getattr(answer_turn, "text", "")))
            if value is None:
                continue
            score = (
                coverage,
                len(overlap),
                len(memory_terms & query_terms),
                str(getattr(answer_turn, "session_date", "") or ""),
                int(getattr(answer_turn, "turn_index", -1)),
                str(getattr(answer_turn, "node_id", "")),
            )
            ranked.append((score, question_turn, answer_turn, value))
    if not ranked:
        return None
    _score, question_turn, answer_turn, value = max(ranked, key=lambda row: row[0])
    return {
        "operation": "dialogue_answer_span",
        "value": value,
        "question_turn_id": str(getattr(question_turn, "node_id", "")),
        "source_turn_ids": [
            str(getattr(question_turn, "node_id", "")),
            str(getattr(answer_turn, "node_id", "")),
        ],
        "answer_speaker": str(getattr(answer_turn, "speaker", "")),
        "evidence": [
            str(getattr(question_turn, "text", "")),
            str(getattr(answer_turn, "text", "")),
        ],
        "complete": True,
        "completion_basis": "same_session_next_turn_explicit_named_value",
    }
