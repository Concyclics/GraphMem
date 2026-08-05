from __future__ import annotations

import re
from typing import Callable

from .schema import QueryFrame, TurnNode


_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_NUMBERED_ITEM = re.compile(r"(?m)^\s*(\d{1,3})[.)]\s+(.+?)\s*$")
_BULLET_ITEM = re.compile(r"(?m)^\s*[-*•]\s+(.+?)\s*$")
_TOPIC_GLUE = {
    "discuss", "earlier", "list", "mention", "provide", "provid", "remind", "say",
    "tell", "think", "was", "were",
}


def _topic_tokens(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[\w'-]+", text.casefold()):
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result




def _requested_position(question: str) -> int | None:
    if not re.search(
        r"\b(?:entry|entries|item|items|list|position|provided|gave|suggested|recommendations?)\b",
        question.casefold(),
    ):
        return None
    match = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)\b", question.casefold())
    if match:
        return int(match.group(1))
    lowered = question.casefold()
    for word, value in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def _items(text: str) -> dict[int, str]:
    numbered = {
        int(match.group(1)): match.group(2).strip()
        for match in _NUMBERED_ITEM.finditer(text)
    }
    if numbered:
        return numbered
    return {
        index: match.group(1).strip()
        for index, match in enumerate(_BULLET_ITEM.finditer(text), start=1)
    }


def ordinal_list_hint(
    frame: QueryFrame,
    turns: list[TurnNode],
    *,
    query_overlap: Callable[[QueryFrame, str], float],
) -> dict[str, object] | None:
    """Resolve an ordinal item from a lossless ordered-list turn.

    Candidate list turns can inherit routing evidence from nearby turns in the
    same session. This preserves exact assistant-provided lists without relying
    on benchmark labels or the list topic.
    """

    # Temporal comparisons such as "which item did I purchase first, A or B"
    # contain item/first but do not refer to a rendered ordered list.
    if frame.requested_operation not in {"list", "lookup"}:
        return None
    position = _requested_position(frame.raw_question)
    if position is None:
        return None
    topic_terms = _topic_tokens(" ".join(frame.content_terms)) - _TOPIC_GLUE
    topic_terms = {term for term in topic_terms if not re.fullmatch(r"\d+(?:st|nd|rd|th)?", term)}
    by_session: dict[str, list[TurnNode]] = {}
    for turn in turns:
        by_session.setdefault(turn.session_id, []).append(turn)
    candidates: list[tuple[float, TurnNode, str, int]] = []
    for session_turns in by_session.values():
        session_turns.sort(key=lambda row: row.turn_index)
        neighborhood_overlap = max(
            (query_overlap(frame, turn.retrieval_text) for turn in session_turns),
            default=0.0,
        )
        session_terms = set().union(*(
            _topic_tokens(turn.retrieval_text) for turn in session_turns
        ))
        topic_coverage = len(topic_terms & session_terms) / max(1, len(topic_terms))
        for turn in session_turns:
            items = _items(turn.text)
            if position not in items:
                continue
            direct_overlap = query_overlap(frame, turn.retrieval_text)
            score = (
                3.0 * direct_overlap + neighborhood_overlap
                + 6.0 * topic_coverage + min(len(items), 20) / 20
            )
            candidates.append((score, turn, items[position], len(items)))
    if not candidates:
        return None
    score, turn, value, item_count = max(
        candidates, key=lambda row: (row[0], row[3], row[1].node_id)
    )
    if score <= 0:
        return None
    return {
        "operation": "ordinal_list_item",
        "position": position,
        "value": value,
        "list_item_count": item_count,
        "source_turn_ids": [turn.node_id],
        "complete": True,
    }
