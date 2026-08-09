from __future__ import annotations

import math
import re
from functools import lru_cache


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "that", "the", "their", "they", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
})


def terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold()[:-2] if token.casefold().endswith("'s") else token.casefold()
                 for token in TOKEN_RE.findall(text))


@lru_cache(maxsize=8_192)
def content_terms(text: str) -> frozenset[str]:
    """Return content tokens, caching immutable graph/query text surfaces.

    Retrieval repeatedly compares the same fact predicates, values and turn
    text across queries.  Caching here preserves exact token semantics while
    avoiding millions of duplicate regex/casefold operations per worker.
    """
    return frozenset(token for token in terms(text) if token not in STOPWORDS and len(token) > 1)


@lru_cache(maxsize=8_192)
def normalize_key(text: str | None) -> str:
    return " ".join(terms(text or ""))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(terms(text)) * 1.3))
