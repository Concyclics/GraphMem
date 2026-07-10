"""Triple-pass retrieval fusion: semantic + keyword (BM25) + typed anchor overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi

from .clients import cosine_similarity
from .models import LeafNode
from .retrieval_cues import ENTITY_CUE_STOPWORDS, dedupe_preserve, proper_name_cues
from .typed_retrieval import leaf_typed_score, query_anchor_terms

_TERM_STOPWORDS = ENTITY_CUE_STOPWORDS | {
    "does",
    "not",
    "any",
    "all",
    "per",
    "night",
    "left",
    "read",
    "finished",
    "visited",
    "attend",
    "attended",
    "compared",
    "spend",
    "spent",
    "cost",
    "pages",
    "page",
    "多少",
    "几个",
    "什么",
    "当前",
    "现在",
    "最近",
    "总共",
}


@dataclass(frozen=True)
class FusionRetrievalConfig:
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    weight_semantic: float = 1.0
    weight_keyword: float = 1.0
    weight_entity: float = 1.0
    query_adaptive_weights: bool = True
    protect_semantic_top_k: int = 0


def rank_leaves_fusion(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    config: FusionRetrievalConfig,
) -> list[LeafNode]:
    if not leaves:
        return []
    fused = compute_fusion_scores(leaves, query_vector, question, config=config)
    ranked = sorted(
        leaves,
        key=lambda leaf: (fused.get(leaf.node_id, 0.0), leaf.node_id),
        reverse=True,
    )
    protect_k = max(0, config.protect_semantic_top_k)
    if protect_k <= 0:
        return ranked
    semantic_order = sorted(
        leaves,
        key=lambda leaf: (
            cosine_similarity(leaf.embedding, query_vector) if leaf.embedding else 0.0,
            leaf.node_id,
        ),
        reverse=True,
    )
    protected = semantic_order[:protect_k]
    protected_ids = {leaf.node_id for leaf in protected}
    merged: list[LeafNode] = list(protected)
    for leaf in ranked:
        if leaf.node_id in protected_ids:
            continue
        merged.append(leaf)
    return merged


def compute_fusion_scores(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    config: FusionRetrievalConfig,
) -> dict[str, float]:
    if not leaves:
        return {}
    semantic = _semantic_scores(leaves, query_vector)
    keyword = _keyword_scores(leaves, question)
    entity = _entity_scores(leaves, question)
    weights = _effective_weights(question, config)
    if config.method == "weighted":
        return _fuse_weighted(semantic, keyword, entity, weights)
    return _fuse_rrf(semantic, keyword, entity, weights, k=config.rrf_k)


def _leaf_text(leaf: LeafNode) -> str:
    return (leaf.retrieval_text or leaf.raw_text or "").strip()


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", text)
        if len(token) > 1
    ]


def _important_query_terms(question: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) > 2 and token.lower() not in _TERM_STOPWORDS
    }


def _semantic_scores(leaves: list[LeafNode], query_vector: list[float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for leaf in leaves:
        embedding = leaf.embedding or []
        scores[leaf.node_id] = cosine_similarity(embedding, query_vector) if embedding else 0.0
    return scores


def _keyword_scores(leaves: list[LeafNode], question: str) -> dict[str, float]:
    corpus = [_tokenize(_leaf_text(leaf)) for leaf in leaves]
    query_tokens = _tokenize(question)
    if not query_tokens or not any(corpus):
        return {leaf.node_id: 0.0 for leaf in leaves}
    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(query_tokens)
    return {leaf.node_id: float(raw_scores[index]) for index, leaf in enumerate(leaves)}


def _leaf_entity_terms(leaf: LeafNode) -> set[str]:
    terms: set[str] = set()
    for key in ("entities", "keywords", "state_phrases", "times", "quantities", "actions"):
        for value in (leaf.anchor_terms or {}).get(key, []):
            lowered = value.lower().strip()
            if lowered:
                terms.add(lowered)
    if terms:
        return terms
    text = _leaf_text(leaf)
    return {entity.lower() for entity in proper_name_cues(text)}


def _entity_scores(leaves: list[LeafNode], question: str) -> dict[str, float]:
    if not leaves:
        return {}
    typed_scores = {leaf.node_id: leaf_typed_score(leaf, question) for leaf in leaves}
    if any(score > 0 for score in typed_scores.values()):
        return typed_scores
    query_entities = proper_name_cues(question)
    if not query_entities:
        query_entities = [
            term
            for term in sorted(_important_query_terms(question), key=len, reverse=True)
            if len(term) >= 4 and any(char.isalpha() for char in term)
        ][:6]
    if not query_entities:
        return {leaf.node_id: 0.0 for leaf in leaves}
    scores: dict[str, float] = {}
    for leaf in leaves:
        text = _leaf_text(leaf)
        lowered = text.lower()
        leaf_entities = _leaf_entity_terms(leaf)
        hits = 0.0
        for entity in query_entities:
            entity_lower = entity.lower()
            if entity_lower in leaf_entities:
                hits += 1.0
            elif entity_lower in lowered:
                hits += 0.85
            elif " " in entity_lower and all(part in lowered for part in entity_lower.split()):
                hits += 0.7
        scores[leaf.node_id] = hits / len(query_entities)
    return scores


def _effective_weights(
    question: str,
    config: FusionRetrievalConfig,
) -> tuple[float, float, float]:
    semantic = config.weight_semantic
    keyword = config.weight_keyword
    entity = config.weight_entity
    if not config.query_adaptive_weights:
        return semantic, keyword, entity
    lowered = question.lower()
    if proper_name_cues(question) or query_anchor_terms(question).get("entities"):
        entity *= 1.6
        semantic *= 0.85
    if re.search(
        r"\b\d+\b|how many|how much|days?|weeks?|months?|years?|ago|before|after|when|"
        r"currently|total|spent|cost|\$|pages?",
        lowered,
    ):
        keyword *= 1.5
        semantic *= 0.9
    if re.search(r"preference|recommend|suggest|which .+ (better|prefer)", lowered):
        entity *= 1.2
        keyword *= 1.1
    return semantic, keyword, entity


def _rank_order(scores: dict[str, float]) -> list[str]:
    return sorted(scores.keys(), key=lambda node_id: (scores[node_id], node_id), reverse=True)


def _fuse_rrf(
    semantic: dict[str, float],
    keyword: dict[str, float],
    entity: dict[str, float],
    weights: tuple[float, float, float],
    *,
    k: int,
) -> dict[str, float]:
    w_sem, w_kw, w_ent = weights
    node_ids = set(semantic) | set(keyword) | set(entity)
    rank_maps = {
        "semantic": {node_id: rank for rank, node_id in enumerate(_rank_order(semantic), start=1)},
        "keyword": {node_id: rank for rank, node_id in enumerate(_rank_order(keyword), start=1)},
        "entity": {node_id: rank for rank, node_id in enumerate(_rank_order(entity), start=1)},
    }
    weight_map = {"semantic": w_sem, "keyword": w_kw, "entity": w_ent}
    fused: dict[str, float] = {}
    max_rank = len(node_ids) + 1
    for node_id in node_ids:
        score = 0.0
        for signal, rank_map in rank_maps.items():
            rank = rank_map.get(node_id, max_rank)
            score += weight_map[signal] / (k + rank)
        fused[node_id] = score
    return fused


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {node_id: 0.0 for node_id in scores}
    span = high - low
    return {node_id: (value - low) / span for node_id, value in scores.items()}


def _fuse_weighted(
    semantic: dict[str, float],
    keyword: dict[str, float],
    entity: dict[str, float],
    weights: tuple[float, float, float],
) -> dict[str, float]:
    w_sem, w_kw, w_ent = weights
    total = w_sem + w_kw + w_ent
    if total <= 0:
        total = 1.0
    norm_sem = _normalize_scores(semantic)
    norm_kw = _normalize_scores(keyword)
    norm_ent = _normalize_scores(entity)
    node_ids = set(norm_sem) | set(norm_kw) | set(norm_ent)
    return {
        node_id: (
            w_sem * norm_sem.get(node_id, 0.0)
            + w_kw * norm_kw.get(node_id, 0.0)
            + w_ent * norm_ent.get(node_id, 0.0)
        )
        / total
        for node_id in node_ids
    }
