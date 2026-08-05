from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Callable, Iterable, Sequence

from ..domain import DiagnosticResult, GraphEdge, NodeType, ProofStep, QueryBudget, RelationType
from ..storage import SQLiteGraphStore
from .navigator import content_terms


class GraphDiagnosticProbe:
    """Gold-independent probe that isolates semantic-edge navigation utility."""

    MODES = frozenset({"seed_only", "relation_only", "shuffled"})
    RELATION_PRIOR = {
        RelationType.HAS_FACT: 1.00,
        RelationType.SCENE_CONTAINS: 0.95,
        RelationType.STATE_NEXT: 0.92,
        RelationType.TEMPORAL_BEFORE: 0.92,
        RelationType.TEMPORAL_AFTER: 0.92,
        RelationType.COLLECTION_CO_MEMBER: 0.88,
        RelationType.PORTAL: 0.86,
        RelationType.REFINES_TO: 0.82,
        RelationType.PARTICIPATES_IN: 0.55,
    }

    def __init__(self, store: SQLiteGraphStore,
                 dense_search: Callable[[str, str, int], Sequence[tuple[str, float]]] | None = None) -> None:
        self.store = store
        self.dense_search = dense_search
        self._turn_score_cache: dict[tuple[str, str], dict[str, float]] = {}

    def run(self, memory_id: str, query: str, budget: QueryBudget, *,
            mode: str = "relation_only", excluded_relations: Iterable[RelationType | str] = (),
            oracle_seed_ids: Sequence[str] = (), shuffle_seed: int = 42) -> DiagnosticResult:
        if mode not in self.MODES:
            raise ValueError(f"unsupported diagnostic mode: {mode}")
        nodes = {node.node_id: node for node in self.store.nodes(memory_id)}
        query_terms = content_terms(query)
        price_query = bool(query_terms & {"price", "cost", "difference", "paid", "spend", "spent"})
        temporal_query = bool(query_terms & {
            "when", "before", "after", "first", "last", "earlier", "later", "duration", "long"})
        collection_query = bool(query_terms & {"which", "what", "list", "all", "many", "count"})

        def node_relevance(node):
            lexical = len(query_terms & content_terms(node.summary)) / max(1, len(query_terms))
            value_type = str(node.attributes.get("value_type", ""))
            roles = set(node.attributes.get("roles", ()))
            slot = (0.55 if price_query and value_type in {"currency", "number"} else 0.0)
            slot += 0.45 if temporal_query and (node.event_time or "time" in roles) else 0.0
            slot += 0.35 if collection_query and "collection_scope" in roles else 0.0
            return lexical + slot

        relevance = {node_id: node_relevance(node) for node_id, node in nodes.items()}
        route_ids = {node_id for node_id, node in nodes.items()
                     if node.attributes.get("provenance_scope") == "route"}
        if not oracle_seed_ids:
            turn_to_scenes = defaultdict(list)
            turn_to_refs = defaultdict(list)
            for node_id in route_ids:
                node = nodes[node_id]
                if node.node_type == NodeType.SCENE:
                    for turn_id in node.attributes.get("turn_ids", ()):
                        turn_to_scenes[str(turn_id)].append(node_id)
            for node_id, node in nodes.items():
                if node.node_type == NodeType.EVIDENCE_GROUP_REF and node.attributes.get("turn_id"):
                    turn_to_refs[str(node.attributes["turn_id"])].append(node_id)
            cache_key = (memory_id, query)
            if cache_key not in self._turn_score_cache:
                raw_scores = defaultdict(lambda: {"exact": 0.0, "bm25": 0.0, "dense": 0.0})
                for turn in self.store.turns(memory_id):
                    overlap = len(query_terms & content_terms(turn.raw_text))
                    if overlap:
                        raw_scores[turn.turn_id]["exact"] = overlap / max(1, len(query_terms))
                for turn_id, score in self.store.search_turns(memory_id, query, limit=96):
                    raw_scores[str(turn_id)]["bm25"] = max(0.0, float(score))
                if self.dense_search:
                    for turn_id, score in self.dense_search(memory_id, query, 96):
                        raw_scores[str(turn_id)]["dense"] = float(score)
                bm25_max = max((row["bm25"] for row in raw_scores.values()), default=0.0)
                dense_values = [row["dense"] for row in raw_scores.values() if row["dense"]]
                dense_min = min(dense_values, default=0.0); dense_max = max(dense_values, default=0.0)
                self._turn_score_cache[cache_key] = {
                    turn_id: 1.4 * channels["exact"]
                    + (channels["bm25"] / bm25_max if bm25_max else 0.0)
                    + ((channels["dense"] - dense_min) / (dense_max - dense_min)
                       if channels["dense"] and dense_max > dense_min else 0.0)
                    for turn_id, channels in raw_scores.items()}
            for turn_id, fused in self._turn_score_cache[cache_key].items():
                for scene_id in turn_to_scenes.get(turn_id, ()):
                    relevance[scene_id] = max(relevance[scene_id], fused)
                for ref_id in turn_to_refs.get(turn_id, ()):
                    relevance[ref_id] = max(relevance[ref_id], fused)
        seed_limit = budget.max_frontier
        seeds = tuple(oracle_seed_ids) if oracle_seed_ids else tuple(
            node_id for node_id, score in sorted(relevance.items(), key=lambda item: (-item[1], item[0]))
            if node_id in route_ids and score > 0)[:seed_limit]
        excluded = {RelationType(item) for item in excluded_relations}
        edges = [edge for edge in self.store.edges(memory_id)
                 if edge.relation not in excluded and edge.relation not in {RelationType.HAS_EVIDENCE}]
        if mode == "shuffled":
            edges = self._shuffle(edges, shuffle_seed)
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.src_id].append(edge)
            adjacency[edge.dst_id].append(replace(edge, src_id=edge.dst_id, dst_id=edge.src_id))
        visited = list(seeds[:budget.max_visited_nodes]); seen = set(visited)
        proof = []; first_relation = {}; counts = Counter()
        # Best-first relation traversal.  The previous edge-id BFS saturated the
        # node budget according to hash order, so it did not actually measure
        # whether typed semantic edges navigate toward query-relevant terminals.
        # A queued row carries its parent edge and commits the proof only when
        # selected under budget.
        queue = []

        def enqueue_neighbors(node_id, hop):
            for edge in adjacency.get(node_id, ()):
                if edge.dst_id in seen or edge.dst_id not in nodes:
                    continue
                destination = nodes[edge.dst_id]
                terminal_bonus = 0.20 if destination.attributes.get(
                    "provenance_scope", "terminal") == "terminal" else 0.0
                priority = (4.0 * relevance[edge.dst_id]
                            + self.RELATION_PRIOR.get(edge.relation, 0.45)
                            + terminal_bonus - 0.08 * hop)
                heapq.heappush(queue, (-priority, hop, edge.dst_id, edge.edge_id, edge))

        if mode != "seed_only":
            for node_id in visited:
                enqueue_neighbors(node_id, 1)
            while queue and len(visited) < budget.max_visited_nodes and len(proof) < budget.max_visited_edges:
                _negative_score, hop, node_id, _tie_breaker, parent = heapq.heappop(queue)
                if node_id in seen:
                    continue
                seen.add(node_id); visited.append(node_id)
                if parent is not None:
                    proof.append(ProofStep(parent.edge_id, parent.src_id, parent.relation,
                                           parent.dst_id, parent.evidence_group_id))
                    first_relation[node_id] = str(parent.relation)
                    counts[str(parent.relation)] += 1
                if hop >= budget.max_hops:
                    continue
                enqueue_neighbors(node_id, hop + 1)
        turn_ids = []; first_turn_relation = {}
        terminal_types = {NodeType.SCENE, NodeType.EVENT_SKELETON, NodeType.CANONICAL_FACT,
                          NodeType.CANONICAL_VALUE, NodeType.CANONICAL_ENTITY, NodeType.TIME_ANCHOR,
                          NodeType.STATE_HEAD, NodeType.EVIDENCE_GROUP_REF}
        for node_id in visited:
            node = nodes[node_id]
            if node.attributes.get("provenance_scope", "terminal") != "terminal":
                continue
            if node.node_type not in terminal_types:
                continue
            for group_id in node.all_evidence_group_ids:
                group = self.store.evidence_group(group_id)
                if not group:
                    continue
                for member in group.members:
                    if member.turn_id not in turn_ids:
                        turn_ids.append(member.turn_id)
                        first_turn_relation[member.turn_id] = first_relation.get(node_id, "seed")
        turns = {turn.turn_id: turn for turn in self.store.turns_by_ids(turn_ids)}
        sessions = tuple(dict.fromkeys(turns[item].session_id for item in turn_ids if item in turns))
        return DiagnosticResult(memory_id, mode, seeds, tuple(visited), tuple(turn_ids), sessions,
                                tuple(proof), dict(counts), first_turn_relation,
                                bool(queue) or len(visited) >= budget.max_visited_nodes)

    @staticmethod
    def _shuffle(edges: Sequence[GraphEdge], seed: int) -> list[GraphEdge]:
        rng = random.Random(seed); by_relation = defaultdict(list)
        for edge in edges:
            by_relation[edge.relation].append(edge)
        result = []
        for relation, rows in sorted(by_relation.items(), key=lambda item: str(item[0])):
            ordered = sorted(rows, key=lambda item: item.edge_id)
            destinations = [edge.dst_id for edge in ordered]; rng.shuffle(destinations)
            result.extend(replace(edge, dst_id=dst, edge_id=edge.edge_id + ":shuffled")
                          for edge, dst in zip(ordered, destinations))
        return result
