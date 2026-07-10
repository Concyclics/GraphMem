"""Graph-based retrieval inspired by HippoRAG (Gutiérrez et al., NeurIPS 2024).

Embedding similarity picks *seed* nodes only; Personalized PageRank (PPR) propagates
activation along edges (root↔root, leaf↔leaf, root→leaf) so retrieval explores multi-hop
neighborhoods instead of staying on the initial vector-ranked path.

``graph_first_retrieve`` flips the default hybrid order: PPR/graph scores drive session
activation and leaf selection, while a global embedding pool provides a safety net.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clients import cosine_similarity
from .models import GraphEdge, LeafNode, SummaryNode
from .typed_retrieval import (
    personalization_scores as typed_personalization_scores,
    rank_roots_hybrid,
)

_EDGE_RELATION_WEIGHT = {
    "keyword_neighbor": 0.88,
    "semantic_neighbor": 0.84,
    "temporal_neighbor": 0.78,
    "time_neighbor": 0.80,
    "state_neighbor": 0.82,
    "event_neighbor": 0.78,
    "entity_neighbor": 0.76,
    "update_neighbor": 0.72,
}


@dataclass(frozen=True)
class GraphSearchConfig:
    seed_roots: int = 6
    seed_leaves: int = 10
    ppr_damping: float = 0.85
    ppr_iterations: int = 25
    embedding_blend: float = 0.35
    session_min_leaves: int = 3
    max_activated_sessions: int = 8
    per_session_leaf_cap: int = 4
    leaf_limit: int = 14
    root_limit: int = 4
    # Keep low: high values flood every leaf under a seed root and erase leaf-leaf signal.
    structural_root_leaf_weight: float = 0.1
    # When True, embedding only seeds PPR; session activation uses seeds + graph scores only.
    seed_only: bool = False
    # When True (default with seed_only), pick leaves by blended PPR score globally —
    # no Phase-1 diversify fill that spends the whole leaf budget before graph expansion.
    free_leaf_select: bool = False
    # Free-select only: guarantee ≥1 leaf from each of the top-N sessions (by graph score)
    # before filling remaining slots by global PPR. 0 = pure free select (best accuracy so far).
    session_coverage: int = 0
    use_typed_retrieval: bool = True
    typed_embedding_blend: float = 0.55


@dataclass(frozen=True)
class GraphSearchResult:
    selected_roots: list[SummaryNode]
    selected_leaves: list[LeafNode]
    used_edges: list[GraphEdge]
    graph_leaf_ids: set[str]
    activated_sessions: list[str]


@dataclass(frozen=True)
class GraphFirstConfig:
    """Graph-primary retrieval with hybrid embedding safeguards."""

    seed_roots: int = 6
    seed_leaves: int = 10
    ppr_damping: float = 0.85
    ppr_iterations: int = 25
    embedding_blend: float = 0.25
    structural_root_leaf_weight: float = 0.1
    leaf_limit: int = 14
    root_limit: int = 4
    global_leaf_top_k: int = 24
    per_session_leaf_k: int = 2
    session_coverage: int = 2
    per_session_leaf_cap: int = 4
    max_activated_sessions: int = 8
    candidate_pool_k: int = 80
    use_typed_retrieval: bool = True
    typed_embedding_blend: float = 0.55


def graph_first_retrieve(
    *,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    query_vector: list[float],
    question: str,
    search_config: GraphFirstConfig,
    rank_leaves_fn: Any,
    enhanced: bool,
) -> GraphSearchResult:
    """Graph-primary retrieval: PPR expands seeds, hybrid global pool provides backup."""
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    if not leaves:
        return GraphSearchResult([], [], [], set(), [])

    adjacency, edge_lookup = _build_adjacency(
        edges,
        roots,
        leaf_by_id,
        structural_root_leaf_weight=search_config.structural_root_leaf_weight,
    )
    all_nodes = list({*leaf_by_id.keys(), *root_by_id.keys()})
    if not all_nodes:
        return GraphSearchResult([], [], [], set(), [])

    seed_roots, ranked_roots = _rank_roots(
        roots,
        query_vector,
        question,
        use_typed=search_config.use_typed_retrieval,
        typed_blend=search_config.typed_embedding_blend,
    )
    seed_roots = seed_roots[: search_config.seed_roots]
    ranked_leaves = rank_leaves_fn(leaves, query_vector, question, enhanced=enhanced)
    seed_leaves = ranked_leaves[: search_config.seed_leaves]
    personalization = _seed_personalization(
        seed_roots,
        seed_leaves,
        query_vector,
        question,
        use_typed=search_config.use_typed_retrieval,
        typed_blend=search_config.typed_embedding_blend,
    )
    ppr_scores = personalized_pagerank(
        all_nodes,
        adjacency,
        personalization,
        damping=search_config.ppr_damping,
        max_iterations=search_config.ppr_iterations,
    )

    candidate_ids = _graph_first_candidate_ids(
        ranked_leaves,
        seed_leaves,
        ppr_scores,
        global_leaf_top_k=search_config.global_leaf_top_k,
        candidate_pool_k=search_config.candidate_pool_k,
    )
    candidate_leaves = [leaf_by_id[leaf_id] for leaf_id in candidate_ids if leaf_id in leaf_by_id]
    blended_scores = _blended_leaf_scores(
        candidate_leaves,
        ppr_scores,
        embedding_blend=search_config.embedding_blend,
    )
    ranked_sessions = _rank_sessions(
        seed_roots,
        seed_leaves,
        ranked_roots,
        ranked_leaves,
        ppr_scores,
        leaf_by_id,
        root_by_id,
        max_sessions=search_config.max_activated_sessions,
        seed_only=True,
    )
    selected_leaves, graph_leaf_ids = _select_leaves_graph_first(
        ranked_leaves,
        candidate_leaves,
        blended_scores,
        ranked_sessions,
        leaf_limit=search_config.leaf_limit,
        session_coverage=search_config.session_coverage,
        per_session_leaf_k=search_config.per_session_leaf_k,
        per_session_leaf_cap=search_config.per_session_leaf_cap,
        global_leaf_top_k=search_config.global_leaf_top_k,
    )
    selected_roots = _select_roots(
        selected_leaves,
        ranked_roots,
        ppr_scores,
        ranked_sessions,
        root_limit=search_config.root_limit,
    )
    used_edges = _collect_used_edges(selected_leaves, selected_roots, edge_lookup)
    return GraphSearchResult(
        selected_roots=selected_roots,
        selected_leaves=selected_leaves,
        used_edges=used_edges,
        graph_leaf_ids=graph_leaf_ids,
        activated_sessions=ranked_sessions,
    )


def graph_search_retrieve(
    *,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    query_vector: list[float],
    question: str,
    search_config: GraphSearchConfig,
    rank_leaves_fn: Any,
    enhanced: bool,
) -> GraphSearchResult:
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    if not leaves:
        return GraphSearchResult([], [], [], set(), [])

    adjacency, edge_lookup = _build_adjacency(
        edges,
        roots,
        leaf_by_id,
        structural_root_leaf_weight=search_config.structural_root_leaf_weight,
    )
    all_nodes = list({*leaf_by_id.keys(), *root_by_id.keys()})
    if not all_nodes:
        return GraphSearchResult([], [], [], set(), [])

    seed_roots, ranked_roots = _rank_roots(
        roots,
        query_vector,
        question,
        use_typed=search_config.use_typed_retrieval,
        typed_blend=search_config.typed_embedding_blend,
    )
    seed_roots = seed_roots[: search_config.seed_roots]
    ranked_leaves = rank_leaves_fn(leaves, query_vector, question, enhanced=enhanced)
    seed_leaves = ranked_leaves[: search_config.seed_leaves]
    personalization = _seed_personalization(
        seed_roots,
        seed_leaves,
        query_vector,
        question,
        use_typed=search_config.use_typed_retrieval,
        typed_blend=search_config.typed_embedding_blend,
    )
    ppr_scores = personalized_pagerank(
        all_nodes,
        adjacency,
        personalization,
        damping=search_config.ppr_damping,
        max_iterations=search_config.ppr_iterations,
    )

    blended_leaf_scores = _blended_leaf_scores(
        ranked_leaves,
        ppr_scores,
        embedding_blend=search_config.embedding_blend,
    )
    ranked_sessions = _rank_sessions(
        seed_roots,
        seed_leaves,
        ranked_roots,
        ranked_leaves,
        ppr_scores,
        leaf_by_id,
        root_by_id,
        max_sessions=search_config.max_activated_sessions,
        seed_only=search_config.seed_only,
    )
    free_select = search_config.free_leaf_select or search_config.seed_only
    selected_leaves, graph_leaf_ids = _select_leaves(
        ranked_leaves,
        blended_leaf_scores,
        ranked_sessions,
        leaf_limit=search_config.leaf_limit,
        session_min_leaves=search_config.session_min_leaves,
        per_session_leaf_cap=search_config.per_session_leaf_cap,
        free_select=free_select,
        session_coverage=search_config.session_coverage,
    )
    selected_roots = _select_roots(
        selected_leaves,
        ranked_roots,
        ppr_scores,
        ranked_sessions,
        root_limit=search_config.root_limit,
    )
    used_edges = _collect_used_edges(selected_leaves, selected_roots, edge_lookup)
    return GraphSearchResult(
        selected_roots=selected_roots,
        selected_leaves=selected_leaves,
        used_edges=used_edges,
        graph_leaf_ids=graph_leaf_ids,
        activated_sessions=ranked_sessions,
    )


def personalized_pagerank(
    nodes: list[str],
    adjacency: dict[str, list[tuple[str, float]]],
    personalization: dict[str, float],
    *,
    damping: float = 0.85,
    max_iterations: int = 25,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    if not nodes:
        return {}
    teleport = _normalize_scores({node: personalization.get(node, 0.0) for node in nodes})
    if not teleport:
        uniform = 1.0 / len(nodes)
        teleport = {node: uniform for node in nodes}

    scores = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(max_iterations):
        next_scores = {node: (1.0 - damping) * teleport.get(node, 0.0) for node in nodes}
        dangling_mass = 0.0
        for node in nodes:
            out_edges = adjacency.get(node, [])
            share = scores[node]
            if not out_edges:
                dangling_mass += damping * share
                continue
            total_weight = sum(weight for _, weight in out_edges)
            if total_weight <= 0:
                dangling_mass += damping * share
                continue
            for neighbor, weight in out_edges:
                if neighbor in next_scores:
                    next_scores[neighbor] += damping * share * (weight / total_weight)
        if dangling_mass:
            for node in nodes:
                next_scores[node] += dangling_mass * teleport.get(node, 0.0)
        delta = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if delta < tolerance:
            break
    return scores


def _build_adjacency(
    edges: list[GraphEdge],
    roots: list[SummaryNode],
    leaf_by_id: dict[str, LeafNode],
    *,
    structural_root_leaf_weight: float,
) -> tuple[dict[str, list[tuple[str, float]]], dict[tuple[str, str], GraphEdge]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    edge_lookup: dict[tuple[str, str], GraphEdge] = {}

    def add_edge(
        src: str,
        dst: str,
        weight: float,
        edge: GraphEdge | None = None,
        *,
        directed: bool = False,
    ) -> None:
        if src == dst:
            return
        adjacency.setdefault(src, []).append((dst, weight))
        if not directed:
            adjacency.setdefault(dst, []).append((src, weight))
        if edge is not None:
            pair = tuple(sorted((src, dst)))
            edge_lookup[pair] = edge

    for edge in edges:
        weight = max(
            0.05,
            float(edge.score) * _EDGE_RELATION_WEIGHT.get(edge.relation, 0.65),
        )
        add_edge(edge.src, edge.dst, weight, edge)

    # Weak bidirectional structural links. High weight equalizes leaves inside a session;
    # directed-only root→leaf blocks seed-leaf → root → other-session propagation.
    for root in roots:
        for leaf_id in root.leaf_ids:
            if leaf_id not in leaf_by_id:
                continue
            add_edge(root.node_id, leaf_id, structural_root_leaf_weight)

    return adjacency, edge_lookup


def _seed_personalization(
    seed_roots: list[SummaryNode],
    seed_leaves: list[LeafNode],
    query_vector: list[float],
    question: str = "",
    *,
    use_typed: bool = True,
    typed_blend: float = 0.55,
) -> dict[str, float]:
    if use_typed and question.strip():
        return typed_personalization_scores(
            seed_roots,
            seed_leaves,
            query_vector,
            question,
            embedding_blend=typed_blend,
        )
    scores: dict[str, float] = {}
    for root in seed_roots:
        if root.embedding:
            scores[root.node_id] = max(scores.get(root.node_id, 0.0), cosine_similarity(root.embedding, query_vector))
    for leaf in seed_leaves:
        if leaf.embedding:
            scores[leaf.node_id] = max(scores.get(leaf.node_id, 0.0), cosine_similarity(leaf.embedding, query_vector))
    return _normalize_scores(scores)


def _rank_roots(
    roots: list[SummaryNode],
    query_vector: list[float],
    question: str,
    *,
    use_typed: bool,
    typed_blend: float,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    if use_typed and question.strip():
        ranked = rank_roots_hybrid(
            roots,
            query_vector,
            question,
            embedding_blend=typed_blend,
        )
        return ranked, ranked
    ranked = _rank_by_embedding(roots, query_vector)
    return ranked, ranked


def _blended_leaf_scores(
    ranked_leaves: list[LeafNode],
    ppr_scores: dict[str, float],
    *,
    embedding_blend: float,
) -> dict[str, float]:
    if not ranked_leaves:
        return {}
    max_rank = max(len(ranked_leaves), 1)
    rank_scores = {
        leaf.node_id: (max_rank - index) / max_rank for index, leaf in enumerate(ranked_leaves)
    }
    max_ppr = max((ppr_scores.get(leaf.node_id, 0.0) for leaf in ranked_leaves), default=0.0) or 1.0
    blend = min(1.0, max(0.0, embedding_blend))
    blended: dict[str, float] = {}
    for leaf in ranked_leaves:
        emb = rank_scores.get(leaf.node_id, 0.0)
        graph = ppr_scores.get(leaf.node_id, 0.0) / max_ppr
        blended[leaf.node_id] = blend * emb + (1.0 - blend) * graph
    return blended


def _rank_sessions(
    seed_roots: list[SummaryNode],
    seed_leaves: list[LeafNode],
    ranked_roots: list[SummaryNode],
    ranked_leaves: list[LeafNode],
    ppr_scores: dict[str, float],
    leaf_by_id: dict[str, LeafNode],
    root_by_id: dict[str, SummaryNode],
    *,
    max_sessions: int,
    seed_only: bool = False,
) -> list[str]:
    session_scores: dict[str, float] = {}

    def bump(session_id: str, value: float) -> None:
        session_scores[session_id] = max(session_scores.get(session_id, 0.0), value)

    for index, root in enumerate(seed_roots):
        weight = 2.0 + (len(seed_roots) - index) / max(len(seed_roots), 1)
        bump(root.session_id, weight)
    for index, leaf in enumerate(seed_leaves):
        weight = 1.5 + (len(seed_leaves) - index) / max(len(seed_leaves), 1)
        bump(leaf.session_id, weight)

    if not seed_only:
        max_root_rank = max(len(ranked_roots), 1)
        for index, root in enumerate(ranked_roots[:12]):
            bump(root.session_id, 1.0 * (max_root_rank - index) / max_root_rank)

        max_leaf_rank = max(len(ranked_leaves), 1)
        for index, leaf in enumerate(ranked_leaves[:24]):
            bump(leaf.session_id, 0.6 * (max_leaf_rank - index) / max_leaf_rank)

    max_ppr = max(ppr_scores.values(), default=0.0) or 1.0
    for node_id, score in ppr_scores.items():
        session_id: str | None = None
        if node_id in root_by_id:
            session_id = root_by_id[node_id].session_id
        elif node_id in leaf_by_id:
            session_id = leaf_by_id[node_id].session_id
        if session_id is not None:
            bump(session_id, score / max_ppr)

    ranked = sorted(session_scores.items(), key=lambda item: (-item[1], item[0]))
    return [session_id for session_id, _ in ranked[:max_sessions]]


def _select_leaves(
    ranked_leaves: list[LeafNode],
    blended_scores: dict[str, float],
    ranked_sessions: list[str],
    *,
    leaf_limit: int,
    session_min_leaves: int,
    per_session_leaf_cap: int,
    free_select: bool = False,
    session_coverage: int = 0,
) -> tuple[list[LeafNode], set[str]]:
    if free_select:
        return _select_leaves_by_graph_score(
            ranked_leaves,
            blended_scores,
            ranked_sessions,
            leaf_limit=leaf_limit,
            per_session_leaf_cap=per_session_leaf_cap,
            session_coverage=session_coverage,
        )

    by_session: dict[str, list[LeafNode]] = {}
    for leaf in ranked_leaves:
        by_session.setdefault(leaf.session_id, []).append(leaf)
    for session_id in by_session:
        by_session[session_id].sort(
            key=lambda leaf: (blended_scores.get(leaf.node_id, 0.0), -leaf.turn_index),
            reverse=True,
        )

    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    graph_leaf_ids: set[str] = set()
    per_session_count: dict[str, int] = {}
    # Keep Phase-1 small so Phase-2 graph deepening can run (was min(2,...) and filled budget).
    initial_per_session = 1

    # Phase 1: light diversify — at most one leaf per session.
    for session_id in ranked_sessions:
        if len(selected) >= leaf_limit:
            break
        session_leaves = by_session.get(session_id, [])
        if not session_leaves:
            continue
        take = min(initial_per_session, per_session_leaf_cap, len(session_leaves), leaf_limit - len(selected))
        for leaf in session_leaves[:take]:
            if leaf.node_id in selected_ids:
                continue
            selected.append(leaf)
            selected_ids.add(leaf.node_id)
            per_session_count[session_id] = per_session_count.get(session_id, 0) + 1
            if len(selected) >= leaf_limit:
                return selected, graph_leaf_ids

    # Phase 2: deepen hot sessions; only these graph-expanded leaves are budget-protected.
    for session_id in ranked_sessions:
        if len(selected) >= leaf_limit:
            break
        session_leaves = by_session.get(session_id, [])
        if not session_leaves:
            continue
        target = min(session_min_leaves, per_session_leaf_cap)
        for leaf in session_leaves:
            if per_session_count.get(session_id, 0) >= target:
                break
            if leaf.node_id in selected_ids:
                continue
            selected.append(leaf)
            selected_ids.add(leaf.node_id)
            graph_leaf_ids.add(leaf.node_id)
            per_session_count[session_id] = per_session_count.get(session_id, 0) + 1
            if len(selected) >= leaf_limit:
                return selected, graph_leaf_ids

    # Phase 3: fill remaining slots by global blended score with soft per-session cap.
    for leaf in sorted(
        ranked_leaves,
        key=lambda item: (blended_scores.get(item.node_id, 0.0), -item.turn_index),
        reverse=True,
    ):
        if leaf.node_id in selected_ids:
            continue
        if len(selected) >= leaf_limit:
            break
        if per_session_count.get(leaf.session_id, 0) >= per_session_leaf_cap:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        graph_leaf_ids.add(leaf.node_id)
        per_session_count[leaf.session_id] = per_session_count.get(leaf.session_id, 0) + 1

    # Phase 4: if still under budget, ignore per-session cap (still by blended score).
    if len(selected) < leaf_limit:
        for leaf in sorted(
            ranked_leaves,
            key=lambda item: (blended_scores.get(item.node_id, 0.0), -item.turn_index),
            reverse=True,
        ):
            if leaf.node_id in selected_ids:
                continue
            selected.append(leaf)
            selected_ids.add(leaf.node_id)
            graph_leaf_ids.add(leaf.node_id)
            if len(selected) >= leaf_limit:
                break
    return selected, graph_leaf_ids


def _graph_first_candidate_ids(
    ranked_leaves: list[LeafNode],
    seed_leaves: list[LeafNode],
    ppr_scores: dict[str, float],
    *,
    global_leaf_top_k: int,
    candidate_pool_k: int,
) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def add(leaf: LeafNode) -> None:
        if leaf.node_id in seen:
            return
        seen.add(leaf.node_id)
        ordered_ids.append(leaf.node_id)

    for leaf in seed_leaves:
        add(leaf)
    for leaf in ranked_leaves[:global_leaf_top_k]:
        add(leaf)
    for leaf in sorted(
        ranked_leaves,
        key=lambda item: (ppr_scores.get(item.node_id, 0.0), item.node_id),
        reverse=True,
    )[:candidate_pool_k]:
        add(leaf)
    return ordered_ids


def _select_leaves_graph_first(
    all_ranked_leaves: list[LeafNode],
    candidate_leaves: list[LeafNode],
    blended_scores: dict[str, float],
    ranked_sessions: list[str],
    *,
    leaf_limit: int,
    session_coverage: int,
    per_session_leaf_k: int,
    per_session_leaf_cap: int,
    global_leaf_top_k: int,
) -> tuple[list[LeafNode], set[str]]:
    by_session: dict[str, list[LeafNode]] = {}
    for leaf in candidate_leaves:
        by_session.setdefault(leaf.session_id, []).append(leaf)
    for session_id in by_session:
        by_session[session_id].sort(
            key=lambda leaf: (blended_scores.get(leaf.node_id, 0.0), -leaf.turn_index),
            reverse=True,
        )

    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    graph_leaf_ids: set[str] = set()
    per_session_count: dict[str, int] = {}

    def append(leaf: LeafNode, *, graph: bool) -> bool:
        if leaf.node_id in selected_ids or len(selected) >= leaf_limit:
            return False
        if per_session_count.get(leaf.session_id, 0) >= per_session_leaf_cap:
            return False
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session_count[leaf.session_id] = per_session_count.get(leaf.session_id, 0) + 1
        if graph:
            graph_leaf_ids.add(leaf.node_id)
        return True

    coverage_n = max(0, min(session_coverage, leaf_limit, len(ranked_sessions)))
    for session_id in ranked_sessions[:coverage_n]:
        session_leaves = by_session.get(session_id) or []
        if not session_leaves:
            continue
        append(session_leaves[0], graph=True)

    for session_id in ranked_sessions:
        if len(selected) >= leaf_limit:
            break
        session_leaves = by_session.get(session_id) or []
        added = 0
        for leaf in session_leaves:
            if added >= per_session_leaf_k:
                break
            if append(leaf, graph=True):
                added += 1

    for leaf in all_ranked_leaves[:global_leaf_top_k]:
        if len(selected) >= leaf_limit:
            break
        append(leaf, graph=False)

    for leaf in sorted(
        candidate_leaves,
        key=lambda item: (blended_scores.get(item.node_id, 0.0), -item.turn_index),
        reverse=True,
    ):
        if len(selected) >= leaf_limit:
            break
        append(leaf, graph=True)

    if len(selected) < leaf_limit:
        for leaf in sorted(
            candidate_leaves,
            key=lambda item: (blended_scores.get(item.node_id, 0.0), -item.turn_index),
            reverse=True,
        ):
            if leaf.node_id in selected_ids:
                continue
            if len(selected) >= leaf_limit:
                break
            selected.append(leaf)
            selected_ids.add(leaf.node_id)
            graph_leaf_ids.add(leaf.node_id)
    return selected, graph_leaf_ids


def _select_leaves_by_graph_score(
    ranked_leaves: list[LeafNode],
    blended_scores: dict[str, float],
    ranked_sessions: list[str],
    *,
    leaf_limit: int,
    per_session_leaf_cap: int,
    session_coverage: int = 0,
) -> tuple[list[LeafNode], set[str]]:
    """Free graph search with an optional session-coverage floor.

    1) Take the best leaf from each of the top-N graph-ranked sessions (N=session_coverage).
    2) Fill remaining slots by global PPR/blend score (soft per-session cap).
    This recovers multi-session hit without the old 2-per-session budget lock.
    """
    by_session: dict[str, list[LeafNode]] = {}
    for leaf in ranked_leaves:
        by_session.setdefault(leaf.session_id, []).append(leaf)
    for session_id in by_session:
        by_session[session_id].sort(
            key=lambda leaf: (blended_scores.get(leaf.node_id, 0.0), -leaf.turn_index),
            reverse=True,
        )

    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    per_session_count: dict[str, int] = {}
    coverage_ids: set[str] = set()

    coverage_n = max(0, min(session_coverage, leaf_limit, len(ranked_sessions)))
    for session_id in ranked_sessions[:coverage_n]:
        if len(selected) >= leaf_limit:
            break
        session_leaves = by_session.get(session_id) or []
        if not session_leaves:
            continue
        leaf = session_leaves[0]
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        coverage_ids.add(leaf.node_id)
        per_session_count[session_id] = 1

    ordered = sorted(
        ranked_leaves,
        key=lambda item: (blended_scores.get(item.node_id, 0.0), -item.turn_index),
        reverse=True,
    )
    for leaf in ordered:
        if len(selected) >= leaf_limit:
            break
        if leaf.node_id in selected_ids:
            continue
        if per_session_count.get(leaf.session_id, 0) >= per_session_leaf_cap:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session_count[leaf.session_id] = per_session_count.get(leaf.session_id, 0) + 1
    if len(selected) < leaf_limit:
        for leaf in ordered:
            if leaf.node_id in selected_ids:
                continue
            selected.append(leaf)
            selected_ids.add(leaf.node_id)
            if len(selected) >= leaf_limit:
                break

    # Protect coverage leaves + top half of the remainder (budget fit pops from the end).
    protect = set(coverage_ids)
    for leaf in selected:
        if len(protect) >= max(len(coverage_ids), len(selected) // 2):
            break
        protect.add(leaf.node_id)
    return selected, protect


def _select_roots(
    selected_leaves: list[LeafNode],
    ranked_roots: list[SummaryNode],
    ppr_scores: dict[str, float],
    ranked_sessions: list[str],
    *,
    root_limit: int,
) -> list[SummaryNode]:
    sessions_with_leaves = {leaf.session_id for leaf in selected_leaves}
    roots_by_session: dict[str, list[SummaryNode]] = {}
    for root in ranked_roots:
        roots_by_session.setdefault(root.session_id, []).append(root)

    selected: list[SummaryNode] = []
    seen: set[str] = set()

    for session_id in ranked_sessions:
        if session_id not in sessions_with_leaves:
            continue
        for root in roots_by_session.get(session_id, []):
            if root.node_id in seen:
                continue
            selected.append(root)
            seen.add(root.node_id)
            break
        if len(selected) >= root_limit:
            return selected[:root_limit]

    for root in ranked_roots:
        if root.node_id in seen:
            continue
        selected.append(root)
        seen.add(root.node_id)
        if len(selected) >= root_limit:
            break

    if len(selected) < root_limit:
        for root in sorted(
            ranked_roots,
            key=lambda item: ppr_scores.get(item.node_id, 0.0),
            reverse=True,
        ):
            if root.node_id in seen:
                continue
            selected.append(root)
            seen.add(root.node_id)
            if len(selected) >= root_limit:
                break
    return selected[:root_limit]


def _collect_used_edges(
    selected_leaves: list[LeafNode],
    selected_roots: list[SummaryNode],
    edge_lookup: dict[tuple[str, str], GraphEdge],
) -> list[GraphEdge]:
    active = {node.node_id for node in selected_leaves} | {node.node_id for node in selected_roots}
    used: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for src in active:
        for dst in active:
            if src >= dst:
                continue
            pair = (src, dst)
            edge = edge_lookup.get(pair)
            if edge is None or pair in seen:
                continue
            used.append(edge)
            seen.add(pair)
    return used


def _rank_by_embedding(nodes: list[Any], query_vector: list[float]) -> list[Any]:
    return sorted(
        nodes,
        key=lambda node: (
            cosine_similarity(node.embedding, query_vector) if node.embedding else 0.0,
            getattr(node, "node_id", ""),
        ),
        reverse=True,
    )


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value > 0)
    if total <= 0:
        return {}
    return {key: value / total for key, value in scores.items() if value > 0}
