from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from ..domain import DiagnosticResult, GraphEdge, NodeType, ProofStep, QueryBudget, RelationType
from ..storage import SQLiteGraphStore
from .navigator import content_terms


class GraphDiagnosticProbe:
    """Gold-independent probe that isolates semantic-edge navigation utility."""

    MODES = frozenset({"seed_only", "relation_only", "shuffled"})

    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store

    def run(self, memory_id: str, query: str, budget: QueryBudget, *,
            mode: str = "relation_only", excluded_relations: Iterable[RelationType | str] = (),
            oracle_seed_ids: Sequence[str] = (), shuffle_seed: int = 42) -> DiagnosticResult:
        if mode not in self.MODES:
            raise ValueError(f"unsupported diagnostic mode: {mode}")
        nodes = {node.node_id: node for node in self.store.nodes(memory_id)}
        query_terms = content_terms(query)
        relevance = {node_id: len(query_terms & content_terms(node.summary)) / max(1, len(query_terms))
                     for node_id, node in nodes.items()}
        seeds = tuple(oracle_seed_ids) if oracle_seed_ids else tuple(node_id for node_id, score in sorted(
            relevance.items(), key=lambda item: (-item[1], item[0])) if score > 0)[:budget.max_frontier]
        excluded = {RelationType(item) for item in excluded_relations}
        edges = [edge for edge in self.store.edges(memory_id)
                 if edge.relation not in excluded and edge.relation not in {RelationType.HAS_EVIDENCE}]
        if mode == "shuffled":
            edges = self._shuffle(edges, shuffle_seed)
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.src_id].append(edge)
            adjacency[edge.dst_id].append(replace(edge, src_id=edge.dst_id, dst_id=edge.src_id))
        visited = list(seeds); seen = set(seeds); proof = []; first_relation = {}; counts = Counter()
        queue = [(0, node_id) for node_id in seeds]; heapq.heapify(queue)
        if mode != "seed_only":
            while queue and len(visited) < budget.max_visited_nodes and len(proof) < budget.max_visited_edges:
                hop, node_id = heapq.heappop(queue)
                if hop >= budget.max_hops:
                    continue
                for edge in sorted(adjacency.get(node_id, ()), key=lambda item: item.edge_id):
                    if edge.dst_id in seen or edge.dst_id not in nodes:
                        continue
                    seen.add(edge.dst_id); visited.append(edge.dst_id)
                    proof.append(ProofStep(edge.edge_id, edge.src_id, edge.relation,
                                           edge.dst_id, edge.evidence_group_id))
                    first_relation[edge.dst_id] = str(edge.relation); counts[str(edge.relation)] += 1
                    heapq.heappush(queue, (hop + 1, edge.dst_id))
                    if len(visited) >= budget.max_visited_nodes or len(proof) >= budget.max_visited_edges:
                        break
        turn_ids = []; first_turn_relation = {}
        terminal_types = {NodeType.SCENE, NodeType.EVENT_SKELETON, NodeType.CANONICAL_FACT,
                          NodeType.CANONICAL_VALUE, NodeType.CANONICAL_ENTITY, NodeType.TIME_ANCHOR,
                          NodeType.STATE_HEAD}
        for node_id in visited:
            node = nodes[node_id]
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
