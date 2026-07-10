"""Root-level graph edge construction with typed anchors and noise pruning.

Design goals (aligned with GraphMem architecture):
- Corpus keyword edges provide broad thematic recall (never replaced by typed edges).
- Typed edges (entity/update/time/state/event) are additive, high-confidence bridges.
- Noisy bridges are dropped before PPR via score floors, generic-entity filters,
  optional embedding sanity checks, and per-root caps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clients import cosine_similarity
from .models import GraphEdge, SummaryNode
from .retrieval_cues import ENTITY_CUE_STOPWORDS

_SUMMARY_TERM_STOPWORDS = ENTITY_CUE_STOPWORDS | {
    "also",
    "been",
    "being",
    "could",
    "does",
    "into",
    "just",
    "like",
    "more",
    "only",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "very",
    "will",
    "with",
    "would",
}

_GENERIC_ENTITY_TERMS = {
    "app",
    "application",
    "assistant",
    "book",
    "business",
    "community",
    "company",
    "customer",
    "data",
    "design",
    "developer",
    "event",
    "experience",
    "feature",
    "group",
    "help",
    "home",
    "information",
    "manager",
    "member",
    "office",
    "people",
    "person",
    "platform",
    "product",
    "project",
    "service",
    "session",
    "software",
    "support",
    "system",
    "team",
    "tool",
    "user",
    "website",
    "work",
}

_PROTECTED_RELATIONS = frozenset({"temporal_neighbor", "semantic_neighbor", "keyword_neighbor"})


@dataclass(frozen=True)
class RootGraphEdgePolicy:
    """Controls how root↔root edges are built and pruned."""

    graph_neighbor_k: int = 2
    enable_typed_edges: bool = False
    typed_neighbors_per_relation: int = 1
    keyword_neighbors_per_root: int = 2
    semantic_neighbors_per_root: int = 2
    typed_min_score: float = 0.76
    typed_max_per_root: int = 6
    entity_min_shared_specific: int = 1
    entity_min_shared_generic: int = 2
    time_min_shared: int = 1
    state_min_shared: int = 1
    event_min_shared: int = 2
    typed_keyword_min_shared: int = 2
    update_min_actions: int = 2
    update_min_entities: int = 1
    corpus_keyword_min_shared: int = 2
    require_semantic_support: bool = True
    semantic_support_min_cosine: float = 0.25
    filter_generic_entities: bool = True


def build_root_graph(
    roots: list[SummaryNode],
    policy: RootGraphEdgePolicy,
) -> list[GraphEdge]:
    if not roots:
        return []
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    root_by_id = {root.node_id: root for root in roots}

    temporal_roots = sorted(roots, key=lambda root: (root.session_date or "", root.session_id))
    for left, right in zip(temporal_roots, temporal_roots[1:]):
        _add_edge(edges, seen, left.node_id, right.node_id, 1.0, "temporal_neighbor")

    root_terms = {
        root.node_id: _summary_term_set(root.retrieval_text or root.summary or "")
        for root in roots
    }
    keyword_neighbors: dict[str, list[tuple[float, SummaryNode]]] = {
        root.node_id: [] for root in roots
    }
    typed_neighbors: dict[str, list[tuple[float, SummaryNode, str]]] = {
        root.node_id: [] for root in roots
    }
    root_anchors = (
        {root.node_id: _typed_anchor_sets(root) for root in roots}
        if policy.enable_typed_edges
        else {}
    )

    for index, root in enumerate(roots):
        terms = root_terms[root.node_id]
        anchors = root_anchors.get(root.node_id, {})
        for candidate in roots[index + 1 :]:
            if policy.enable_typed_edges:
                candidate_anchors = root_anchors.get(candidate.node_id, {})
                for relation, score in _typed_anchor_edge_scores(
                    anchors,
                    candidate_anchors,
                    policy=policy,
                ):
                    typed_neighbors[root.node_id].append((score, candidate, relation))
                    typed_neighbors[candidate.node_id].append((score, root, relation))

            if not terms:
                continue
            candidate_terms = root_terms[candidate.node_id]
            if not candidate_terms:
                continue
            shared = terms & candidate_terms
            if len(shared) < policy.corpus_keyword_min_shared:
                continue
            score = min(0.99, 0.45 + len(shared) / max(len(terms | candidate_terms), 1))
            keyword_neighbors[root.node_id].append((score, candidate))
            keyword_neighbors[candidate.node_id].append((score, root))

    for root in roots:
        per_relation: dict[str, int] = {}
        for score, candidate, relation in sorted(
            typed_neighbors[root.node_id],
            key=lambda item: (item[0], item[2], item[1].node_id),
            reverse=True,
        ):
            if score < policy.typed_min_score:
                continue
            if per_relation.get(relation, 0) >= policy.typed_neighbors_per_relation:
                continue
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, relation)
            per_relation[relation] = per_relation.get(relation, 0) + 1

    for root in roots:
        for score, candidate in sorted(
            keyword_neighbors[root.node_id],
            key=lambda item: (item[0], item[1].node_id),
            reverse=True,
        )[: policy.keyword_neighbors_per_root]:
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, "keyword_neighbor")

    for root in roots:
        if root.embedding is None:
            continue
        neighbors = sorted(
            (
                (cosine_similarity(root.embedding, candidate.embedding), candidate)
                for candidate in roots
                if candidate.node_id != root.node_id and candidate.embedding is not None
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, candidate in neighbors[: policy.semantic_neighbors_per_root]:
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, "semantic_neighbor")

    return prune_noisy_root_edges(edges, root_by_id, policy)


def prune_noisy_root_edges(
    edges: list[GraphEdge],
    root_by_id: dict[str, SummaryNode],
    policy: RootGraphEdgePolicy,
) -> list[GraphEdge]:
    if not edges:
        return edges

    kept: list[GraphEdge] = []
    for edge in edges:
        if edge.relation in _PROTECTED_RELATIONS:
            kept.append(edge)
            continue
        if edge.score < policy.typed_min_score:
            continue
        left = root_by_id.get(edge.src)
        right = root_by_id.get(edge.dst)
        if left is None or right is None:
            continue
        if policy.filter_generic_entities and edge.relation == "entity_neighbor":
            shared = _shared_entities(left, right)
            if shared and not _entity_edge_allowed(shared, policy):
                continue
        if policy.require_semantic_support and not _has_semantic_support(left, right, policy):
            continue
        kept.append(edge)

    return _cap_typed_edges_per_root(kept, policy.typed_max_per_root)


def _cap_typed_edges_per_root(edges: list[GraphEdge], typed_max_per_root: int) -> list[GraphEdge]:
    if typed_max_per_root < 1:
        return [edge for edge in edges if edge.relation in _PROTECTED_RELATIONS]

    protected = [edge for edge in edges if edge.relation in _PROTECTED_RELATIONS]
    typed = [
        edge
        for edge in edges
        if edge.relation not in _PROTECTED_RELATIONS
    ]
    per_root: dict[str, int] = {}
    capped_typed: list[GraphEdge] = []
    for edge in sorted(typed, key=lambda item: (item.score, item.relation), reverse=True):
        if (
            per_root.get(edge.src, 0) >= typed_max_per_root
            or per_root.get(edge.dst, 0) >= typed_max_per_root
        ):
            continue
        capped_typed.append(edge)
        per_root[edge.src] = per_root.get(edge.src, 0) + 1
        per_root[edge.dst] = per_root.get(edge.dst, 0) + 1
    return protected + capped_typed


def _typed_anchor_edge_scores(
    left: dict[str, set[str]],
    right: dict[str, set[str]],
    *,
    policy: RootGraphEdgePolicy,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    relation_specs = (
        ("entities", "entity_neighbor", 0.86, 0.04, "entity"),
        ("times", "time_neighbor", 0.76, 0.04, "time"),
        ("actions", "event_neighbor", 0.68, 0.04, "event"),
        ("state_phrases", "state_neighbor", 0.74, 0.03, "state"),
        ("keywords", "keyword_neighbor", 0.58, 0.03, "typed_keyword"),
    )
    for key, relation, base, step, kind in relation_specs:
        shared = left.get(key, set()) & right.get(key, set())
        if not _typed_shared_allowed(kind, shared, policy):
            continue
        scores.append((relation, min(0.97, base + len(shared) * step)))

    shared_actions = left.get("actions", set()) & right.get("actions", set())
    shared_entities = left.get("entities", set()) & right.get("entities", set())
    shared_keywords = left.get("keywords", set()) & right.get("keywords", set())
    entity_ok = _entity_edge_allowed(shared_entities, policy)
    if len(shared_actions) >= policy.update_min_actions and (
        len(shared_entities) >= policy.update_min_entities and entity_ok
        or len(shared_keywords) >= policy.typed_keyword_min_shared
    ):
        scores.append(("update_neighbor", min(0.94, 0.80 + 0.03 * len(shared_actions))))
    return scores


def _typed_shared_allowed(kind: str, shared: set[str], policy: RootGraphEdgePolicy) -> bool:
    if not shared:
        return False
    if kind == "entity":
        return _entity_edge_allowed(shared, policy)
    if kind == "time":
        return len(shared) >= policy.time_min_shared
    if kind == "state":
        return len(shared) >= policy.state_min_shared
    if kind == "event":
        return len(shared) >= policy.event_min_shared
    if kind == "typed_keyword":
        return len(shared) >= policy.typed_keyword_min_shared
    return False


def _entity_edge_allowed(shared: set[str], policy: RootGraphEdgePolicy) -> bool:
    if not shared:
        return False
    if not policy.filter_generic_entities:
        return len(shared) >= policy.entity_min_shared_specific
    specific = {value for value in shared if value not in _GENERIC_ENTITY_TERMS}
    if len(specific) >= policy.entity_min_shared_specific:
        return True
    return len(shared) >= policy.entity_min_shared_generic


def _shared_entities(left: SummaryNode, right: SummaryNode) -> set[str]:
    left_anchors = _typed_anchor_sets(left)
    right_anchors = _typed_anchor_sets(right)
    return left_anchors.get("entities", set()) & right_anchors.get("entities", set())


def _has_semantic_support(
    left: SummaryNode,
    right: SummaryNode,
    policy: RootGraphEdgePolicy,
) -> bool:
    if left.embedding is None or right.embedding is None:
        return True
    return cosine_similarity(left.embedding, right.embedding) >= policy.semantic_support_min_cosine


def _typed_anchor_sets(root: SummaryNode) -> dict[str, set[str]]:
    anchors = root.anchor_terms or {}
    typed: dict[str, set[str]] = {}
    for key, values in anchors.items():
        cleaned = {_normalize_anchor(value) for value in values if _normalize_anchor(value)}
        if cleaned:
            typed[key] = cleaned
    return typed


def _normalize_anchor(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip(" \t\r\n.,;:")).casefold()
    if len(text) < 2 or text in _SUMMARY_TERM_STOPWORDS:
        return ""
    return text


def _summary_term_set(text: str) -> set[str]:
    terms = {
        token.casefold()
        for token in re.findall(r"[\w&.'-]+", text)
        if len(token) >= 3 and token.casefold() not in _SUMMARY_TERM_STOPWORDS
    }
    terms.update(_numeric_time_cues(text))
    return terms


def _numeric_time_cues(text: str) -> set[str]:
    cues: set[str] = set()
    for match in re.finditer(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)?\b",
        text,
        flags=re.IGNORECASE,
    ):
        cues.add(match.group(0).casefold())
    for match in re.finditer(r"\b\d{4}/\d{2}/\d{2}\b", text):
        cues.add(match.group(0))
    return cues


def _add_edge(
    edges: list[GraphEdge],
    seen: set[tuple[str, str, str]],
    src: str,
    dst: str,
    score: float,
    relation: str,
) -> None:
    pair = tuple(sorted((src, dst)))
    key = (pair[0], pair[1], relation)
    if key in seen:
        return
    edges.append(GraphEdge(src=pair[0], dst=pair[1], score=score, relation=relation))  # type: ignore[arg-type]
    seen.add(key)
