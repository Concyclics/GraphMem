"""Typed anchor retrieval: query-time structured overlap on written anchor_terms."""
from __future__ import annotations

import re
from typing import Any

from .clients import cosine_similarity
from .models import LeafNode, SummaryNode
from .retrieval_cues import ENTITY_CUE_STOPWORDS, dedupe_preserve, proper_name_cues

ANCHOR_KEYS = (
    "entities",
    "times",
    "quantities",
    "actions",
    "state_phrases",
    "keywords",
)

_DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "entities": 1.0,
    "times": 0.88,
    "state_phrases": 0.86,
    "actions": 0.82,
    "quantities": 0.80,
    "keywords": 0.65,
}

_ACTION_CUES = (
    "accepted",
    "arrived",
    "attended",
    "bought",
    "canceled",
    "cancelled",
    "changed",
    "completed",
    "currently",
    "decided",
    "ordered",
    "planned",
    "purchased",
    "read",
    "recommended",
    "spent",
    "subscribed",
    "visited",
)

_TERM_STOPWORDS = ENTITY_CUE_STOPWORDS | {
    "does",
    "not",
    "any",
    "all",
    "per",
    "many",
    "much",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
}


def query_anchor_terms(question: str, *, limit_per_type: int = 16) -> dict[str, list[str]]:
    """Parse a question into the same anchor schema used at write time."""
    anchors: dict[str, list[str]] = {key: [] for key in ANCHOR_KEYS}
    anchors["entities"] = _dedupe(_proper_name_cues(question), limit_per_type)
    anchors["times"] = _dedupe(_numeric_time_cues(question), limit_per_type)
    anchors["quantities"] = _dedupe(_quantity_cues(question), limit_per_type)
    lowered = question.lower()
    anchors["actions"] = _dedupe(
        [cue for cue in _ACTION_CUES if cue in lowered],
        limit_per_type,
    )
    anchors["state_phrases"] = _dedupe(_state_phrase_cues(question), limit_per_type)
    anchors["keywords"] = _dedupe(_important_query_terms(question), limit_per_type)
    return {key: value for key, value in anchors.items() if value}


def query_relation_weights(question: str) -> dict[str, float]:
    """Relation-sensitive weights for typed overlap scoring."""
    weights = dict(_DEFAULT_RELATION_WEIGHTS)
    lowered = question.lower()
    if proper_name_cues(question) or re.search(
        r"preference|recommend|suggest|which .+ (better|prefer)",
        lowered,
    ):
        weights["entities"] *= 1.25
        weights["state_phrases"] *= 1.15
    if re.search(
        r"\b\d+\b|how many|how much|days?|weeks?|months?|years?|ago|before|after|when|"
        r"currently|total|spent|cost|\$|pages?",
        lowered,
    ):
        weights["times"] *= 1.3
        weights["quantities"] *= 1.25
        weights["keywords"] *= 1.1
    if re.search(r"update|changed|currently|now|latest|recent|before|after", lowered):
        weights["actions"] *= 1.2
        weights["state_phrases"] *= 1.15
        weights["times"] *= 1.1
    return weights


def anchor_terms_to_sets(anchors: dict[str, list[str]] | None) -> dict[str, set[str]]:
    typed: dict[str, set[str]] = {}
    if not anchors:
        return typed
    for key, values in anchors.items():
        cleaned = {_normalize_anchor(value) for value in values if _normalize_anchor(value)}
        if cleaned:
            typed[key] = cleaned
    return typed


def summary_anchor_sets(root: SummaryNode) -> dict[str, set[str]]:
    if root.anchor_terms:
        return anchor_terms_to_sets(root.anchor_terms)
    return anchor_terms_to_sets(
        query_anchor_terms(root.retrieval_text or root.summary or "", limit_per_type=24)
    )


def leaf_anchor_sets(leaf: LeafNode) -> dict[str, set[str]]:
    anchors: dict[str, list[str]] = {
        key: list((leaf.anchor_terms or {}).get(key, []))
        for key in ANCHOR_KEYS
    }
    if leaf.compact_facts:
        anchors["keywords"] = dedupe_preserve(
            [*anchors.get("keywords", []), *leaf.compact_facts[:6]]
        )
        anchors["state_phrases"] = dedupe_preserve(
            [*anchors.get("state_phrases", []), *leaf.compact_facts[:4]]
        )
    if any(anchors.values()):
        return anchor_terms_to_sets(anchors)
    return anchor_terms_to_sets(
        query_anchor_terms(leaf.retrieval_text or leaf.raw_text or "", limit_per_type=24)
    )


def typed_overlap_score(
    node_sets: dict[str, set[str]],
    query_sets: dict[str, set[str]],
    *,
    relation_weights: dict[str, float] | None = None,
) -> float:
    if not node_sets or not query_sets:
        return 0.0
    weights = relation_weights or _DEFAULT_RELATION_WEIGHTS
    total_weight = 0.0
    weighted_hits = 0.0
    for key, weight in weights.items():
        query_values = query_sets.get(key)
        if not query_values:
            continue
        node_values = node_sets.get(key, set())
        if not node_values:
            continue
        total_weight += weight
        hits = 0.0
        for query_value in query_values:
            if query_value in node_values:
                hits += 1.0
                continue
            matched = False
            if " " in query_value:
                parts = query_value.split()
                for node_value in node_values:
                    if all(part in node_value for part in parts):
                        hits += 0.85
                        matched = True
                        break
            if matched:
                continue
            if any(query_value in node_value or node_value in query_value for node_value in node_values):
                hits += 0.7
        weighted_hits += weight * (hits / len(query_values))
    if total_weight <= 0:
        return 0.0
    return weighted_hits / total_weight


def root_typed_score(root: SummaryNode, query_anchors: dict[str, list[str]]) -> float:
    query_sets = anchor_terms_to_sets(query_anchors)
    return typed_overlap_score(
        summary_anchor_sets(root),
        query_sets,
        relation_weights=query_relation_weights(" ".join(query_anchors.get("keywords", []))),
    )


def leaf_typed_score(leaf: LeafNode, question: str) -> float:
    query_anchors = query_anchor_terms(question)
    return typed_overlap_score(
        leaf_anchor_sets(leaf),
        anchor_terms_to_sets(query_anchors),
        relation_weights=query_relation_weights(question),
    )


def hybrid_embedding_typed_score(
    embedding_score: float,
    typed_score: float,
    *,
    embedding_blend: float = 0.55,
) -> float:
    blend = min(1.0, max(0.0, embedding_blend))
    return blend * embedding_score + (1.0 - blend) * typed_score


def rank_roots_hybrid(
    roots: list[SummaryNode],
    query_vector: list[float],
    question: str,
    *,
    embedding_blend: float = 0.55,
) -> list[SummaryNode]:
    if not roots:
        return []
    query_anchors = query_anchor_terms(question)
    query_sets = anchor_terms_to_sets(query_anchors)
    relation_weights = query_relation_weights(question)

    def score(root: SummaryNode) -> float:
        embedding = cosine_similarity(root.embedding, query_vector) if root.embedding else 0.0
        typed = typed_overlap_score(
            summary_anchor_sets(root),
            query_sets,
            relation_weights=relation_weights,
        )
        return hybrid_embedding_typed_score(
            embedding,
            typed,
            embedding_blend=embedding_blend,
        )

    return sorted(roots, key=lambda root: (score(root), root.node_id), reverse=True)


def rank_leaves_hybrid(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    embedding_blend: float = 0.55,
) -> list[LeafNode]:
    if not leaves:
        return []
    query_anchors = query_anchor_terms(question)
    query_sets = anchor_terms_to_sets(query_anchors)
    relation_weights = query_relation_weights(question)

    def score(leaf: LeafNode) -> float:
        embedding = cosine_similarity(leaf.embedding, query_vector) if leaf.embedding else 0.0
        typed = typed_overlap_score(
            leaf_anchor_sets(leaf),
            query_sets,
            relation_weights=relation_weights,
        )
        return hybrid_embedding_typed_score(
            embedding,
            typed,
            embedding_blend=embedding_blend,
        )

    return sorted(leaves, key=lambda leaf: (score(leaf), leaf.node_id), reverse=True)


def personalization_scores(
    seed_roots: list[SummaryNode],
    seed_leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    embedding_blend: float = 0.55,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    query_anchors = query_anchor_terms(question)
    query_sets = anchor_terms_to_sets(query_anchors)
    relation_weights = query_relation_weights(question)

    for root in seed_roots:
        embedding = cosine_similarity(root.embedding, query_vector) if root.embedding else 0.0
        typed = typed_overlap_score(
            summary_anchor_sets(root),
            query_sets,
            relation_weights=relation_weights,
        )
        scores[root.node_id] = max(
            scores.get(root.node_id, 0.0),
            hybrid_embedding_typed_score(embedding, typed, embedding_blend=embedding_blend),
        )
    for leaf in seed_leaves:
        embedding = cosine_similarity(leaf.embedding, query_vector) if leaf.embedding else 0.0
        typed = typed_overlap_score(
            leaf_anchor_sets(leaf),
            query_sets,
            relation_weights=relation_weights,
        )
        scores[leaf.node_id] = max(
            scores.get(leaf.node_id, 0.0),
            hybrid_embedding_typed_score(embedding, typed, embedding_blend=embedding_blend),
        )
    return _normalize_scores(scores)


def _important_query_terms(question: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) > 2 and token.lower() not in _TERM_STOPWORDS
    ]


def _numeric_time_cues(text: str) -> list[str]:
    cues: list[str] = []
    for match in re.finditer(
        r"\b(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago|last\s+\w+|next\s+\w+)\b",
        text,
        flags=re.IGNORECASE,
    ):
        cues.append(match.group(0).strip())
    return cues


def _quantity_cues(text: str) -> list[str]:
    cues: list[str] = []
    for match in re.finditer(
        r"\b(?:\$?\d+(?:\.\d+)?|\d+(?:\.\d+)?%)\b",
        text,
    ):
        cues.append(match.group(0).strip())
    return cues


def _state_phrase_cues(text: str) -> list[str]:
    cues: list[str] = []
    for match in re.finditer(
        r"\b(?:currently|now|still)\s+(?:reading|using|keeping|preferring|have|having)\s+([^.;\n]{3,80})",
        text,
        flags=re.IGNORECASE,
    ):
        cue = _normalize_anchor(match.group(1))
        if cue:
            cues.append(cue)
    return cues


def _proper_name_cues(text: str) -> list[str]:
    return proper_name_cues(text, stopwords=_TERM_STOPWORDS)


def _normalize_anchor(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip(" \t\r\n.,;:")).casefold()
    if len(text) < 2 or text in _TERM_STOPWORDS:
        return ""
    return text


def _dedupe(values: list[str], limit: int) -> list[str]:
    return dedupe_preserve(values)[:limit]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value > 0)
    if total <= 0:
        return {}
    return {key: value / total for key, value in scores.items() if value > 0}
