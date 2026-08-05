from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..domain import GraphEdge, GraphNode, NodeType, RelationType
from ..storage.sqlite import SQLiteGraphStore


PROVENANCE_RELATIONS = {RelationType.HAS_EVIDENCE}


@dataclass(frozen=True, slots=True)
class AdjacentEdge:
    edge: GraphEdge
    next_node_id: str
    inverse: bool


class GraphReadView:
    """Immutable, relation-specific adjacency compiled from canonical rows."""

    def __init__(self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = {edge.edge_id: edge for edge in edges}
        forward: dict[RelationType, dict[str, list[AdjacentEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        inverse: dict[RelationType, dict[str, list[AdjacentEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for edge in self.edges.values():
            forward[edge.relation][edge.src_id].append(
                AdjacentEdge(edge, edge.dst_id, False)
            )
            inverse[edge.relation][edge.dst_id].append(
                AdjacentEdge(edge, edge.src_id, True)
            )
            if not edge.directed:
                forward[edge.relation][edge.dst_id].append(
                    AdjacentEdge(edge, edge.src_id, False)
                )
                inverse[edge.relation][edge.src_id].append(
                    AdjacentEdge(edge, edge.dst_id, True)
                )
        self.forward = self._freeze(forward)
        self.inverse = self._freeze(inverse)
        self.entity_index: dict[str, tuple[str, ...]] = self._index("entity_id")
        self.time_index: dict[str, tuple[str, ...]] = self._index("event_time")
        self.role_bitset = {
            node_id: frozenset(str(value) for value in node.attributes.get("roles", ()))
            for node_id, node in self.nodes.items()
        }
        self.terminal_provenance_bitset = {
            node_id: frozenset(node.all_evidence_group_ids)
            for node_id, node in self.nodes.items()
            if node.attributes.get("provenance_scope", "terminal") == "terminal"
        }
        self.routing_provenance_bitset = {
            node_id: frozenset(node.all_evidence_group_ids)
            for node_id, node in self.nodes.items()
            if node.attributes.get("provenance_scope") == "route"
        }
        self.provenance_bitset = {
            node_id: frozenset(node.all_evidence_group_ids)
            for node_id, node in self.nodes.items()
        }

    @staticmethod
    def _freeze(value: dict[RelationType, dict[str, list[AdjacentEdge]]]) -> dict[
        RelationType, dict[str, tuple[AdjacentEdge, ...]]
    ]:
        return {
            relation: {
                node_id: tuple(sorted(rows, key=lambda item: item.edge.edge_id))
                for node_id, rows in by_node.items()
            }
            for relation, by_node in value.items()
        }

    def _index(self, field: str) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            value = getattr(node, field)
            if value:
                result[str(value).casefold()].append(node.node_id)
            for item in node.attributes.get(field + "s", ()):
                result[str(item).casefold()].append(node.node_id)
        return {key: tuple(sorted(set(rows))) for key, rows in result.items()}

    def neighbors(
        self,
        node_id: str,
        relations: Sequence[RelationType] | None = None,
        *,
        include_inverse: bool = True,
        semantic_only: bool = True,
    ) -> tuple[AdjacentEdge, ...]:
        allowed = tuple(relations or RelationType)
        rows: list[AdjacentEdge] = []
        for relation in allowed:
            if semantic_only and relation in PROVENANCE_RELATIONS:
                continue
            rows.extend(self.forward.get(relation, {}).get(node_id, ()))
            if include_inverse:
                rows.extend(self.inverse.get(relation, {}).get(node_id, ()))
        dedup = {(row.edge.edge_id, row.next_node_id, row.inverse): row for row in rows}
        return tuple(sorted(dedup.values(), key=lambda row: (row.edge.edge_id, row.next_node_id)))

    def evidence_group_ids_for_nodes(self, node_ids: Iterable[str], *,
                                     terminal_only: bool = True) -> tuple[str, ...]:
        source = self.terminal_provenance_bitset if terminal_only else {
            **self.routing_provenance_bitset, **self.terminal_provenance_bitset}
        groups: list[str] = []
        for node_id in node_ids:
            groups.extend(source.get(node_id, ()))
        return tuple(dict.fromkeys(groups))

    def nodes_by_type(self, node_type: NodeType) -> tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes.values() if node.node_type == node_type)


class SQLiteSnapshotRuntime:
    mode = "sqlite_snapshot"

    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store
        self._views: dict[tuple[str, int], GraphReadView] = {}

    def view(self, memory_id: str) -> GraphReadView:
        version = self.store.graph_version(memory_id)
        key = (memory_id, version)
        if key not in self._views:
            self._views = {item: view for item, view in self._views.items() if item[0] != memory_id}
            self._views[key] = GraphReadView(
                self.store.nodes(memory_id), self.store.edges(memory_id)
            )
        return self._views[key]

    def nodes(self, memory_id: str, node_ids: Sequence[str]) -> Sequence[GraphNode]:
        view = self.view(memory_id)
        return [view.nodes[node_id] for node_id in node_ids if node_id in view.nodes]

    def expand(
        self,
        memory_id: str,
        frontier: Sequence[str],
        relations: Sequence[RelationType],
        *,
        limit: int,
    ) -> Sequence[GraphEdge]:
        view = self.view(memory_id)
        result: list[GraphEdge] = []
        seen: set[str] = set()
        for node_id in frontier:
            for row in view.neighbors(node_id, relations):
                if row.edge.edge_id in seen:
                    continue
                seen.add(row.edge.edge_id)
                result.append(row.edge)
                if len(result) >= limit:
                    return result
        return result
