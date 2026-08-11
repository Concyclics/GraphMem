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

# Relation construction needs a narrower equivalence than semantic predicate
# clustering: ``attended``/``attending`` should share a state corridor, while
# ``visited``/``wanted to visit`` must remain separated by modality.  Keeping a
# tiny deterministic stemmer here avoids loading a language toolkit in every
# online worker (and its associated RSS/PSS cost).  It intentionally handles
# only the leading lexical word and common English inflections.
_PREDICATE_FUNCTION_WORDS = frozenset({
    "a", "an", "am", "are", "be", "been", "being", "did", "do", "does",
    "had", "has", "have", "having", "is", "not", "the", "to", "was",
    "were", "will", "would",
})
_PREDICATE_IRREGULAR_ROOTS = {
    "bought": "buy", "brought": "bring", "came": "come", "did": "do",
    "done": "do", "felt": "feel", "found": "find", "gave": "give",
    "given": "give", "gone": "go", "got": "get", "had": "have",
    "has": "have", "heard": "hear", "held": "hold", "kept": "keep",
    "knew": "know", "known": "know", "led": "lead", "left": "leave",
    "lost": "lose", "made": "make", "met": "meet", "paid": "pay",
    "ran": "run", "read": "read", "said": "say", "saw": "see",
    "seen": "see", "sent": "send", "spent": "spend", "taught": "teach",
    "thought": "think", "told": "tell", "took": "take", "taken": "take",
    "went": "go", "won": "win", "wrote": "write", "written": "write",
}


@lru_cache(maxsize=8_192)
def predicate_family(text: str | None) -> str:
    """Return a conservative morphological family for a predicate phrase.

    The output is only a routing key; it never rewrites a fact or licenses an
    answer.  Polarity and modality are deliberately not folded into this value
    and must remain separate components of any state key built from it.
    """

    words = [word for word in terms(text or "")
             if word not in _PREDICATE_FUNCTION_WORDS]
    if not words:
        return ""
    word = words[0]
    if word in _PREDICATE_IRREGULAR_ROOTS:
        return _PREDICATE_IRREGULAR_ROOTS[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        root = word[:-3]
        if len(root) > 3 and root[-1] == root[-2] and root[-1] not in "lsz":
            root = root[:-1]
        if root.endswith(("at", "bl", "iz")):
            root += "e"
        return root
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        # Verbs already ending in silent e take only ``d``.  The suffix list is
        # intentionally conservative; the ordinary ``-ed`` path still covers
        # attend/visit/recommend/play without inventing an e.
        if word.endswith((
                "amed", "ared", "ated", "aved", "eived", "ired", "ized",
                "oved", "ured", "used")):
            return word[:-1]
        root = word[:-2]
        if len(root) > 3 and root[-1] == root[-2] and root[-1] not in "lsz":
            root = root[:-1]
        return root
    if word.endswith(("ches", "shes", "sses", "xes", "zes")) and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


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
