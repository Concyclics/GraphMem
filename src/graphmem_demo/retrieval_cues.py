"""Shared lexical cues for retrieval (keyword overlap, entity matching)."""
from __future__ import annotations

import re

ENTITY_CUE_STOPWORDS = {
    "how",
    "many",
    "much",
    "what",
    "when",
    "where",
    "which",
    "who",
    "did",
    "have",
    "the",
    "and",
    "for",
    "with",
    "from",
    "about",
    "your",
    "you",
    "my",
    "i",
    "me",
    "was",
    "were",
    "are",
    "is",
    "been",
    "being",
    "into",
    "than",
    "that",
    "this",
    "these",
    "those",
    "last",
    "first",
    "before",
    "after",
    "during",
    "week",
    "weeks",
    "days",
    "day",
    "month",
    "months",
    "year",
    "years",
    "ago",
    "currently",
    "recently",
    "total",
    "session",
    "assistant",
    "user",
}


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def proper_name_cues(text: str, *, stopwords: set[str] | None = None) -> list[str]:
    blocked = stopwords if stopwords is not None else ENTITY_CUE_STOPWORDS
    cues: list[str] = []
    for match in re.finditer(r'"([^"\n]{3,80})"', text):
        cue = match.group(1).strip()
        if cue and cue.lower() not in blocked:
            cues.append(cue)
    for match in re.finditer(r"(?<![A-Za-z])'([^'\n]{3,80})'(?![A-Za-z])", text):
        cue = match.group(1).strip()
        if cue and cue.lower() not in blocked:
            cues.append(cue)
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4}\b",
        text,
    ):
        cue = match.group(0).strip(" .,:;")
        if len(cue) < 3 or cue.lower() in blocked:
            continue
        cues.append(cue)
    return dedupe_preserve(cues)
