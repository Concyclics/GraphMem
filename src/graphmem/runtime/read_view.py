from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..domain import GraphEdge, GraphNode, NodeType, RelationType
from ..storage.sqlite import SQLiteGraphStore
from ..text import content_terms, normalize_key


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
        self._compile_query_index()

    def _compile_query_index(self) -> None:
        """Compile immutable query postings from V5 canonical attributes.

        This is intentionally an in-memory projection: it gives frozen graphs a
        typed query index without mutating their authority SQLite database.
        """
        owner_alias: dict[str, set[str]] = defaultdict(set)
        self.fact_owner_predicate_scope_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.owner_fact_index: dict[str, list[str]] = defaultdict(list)
        self.predicate_fact_index: dict[str, list[str]] = defaultdict(list)
        self.scope_fact_index: dict[str, list[str]] = defaultdict(list)
        self.value_fact_index: dict[str, list[str]] = defaultdict(list)
        self.collection_fact_index: dict[str, list[str]] = defaultdict(list)
        self.session_fact_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.routing_child_postings: dict[str, list[str]] = defaultdict(list)
        self.collection_index: dict[str, list[str]] = defaultdict(list)
        self.manifest_value_index: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            attrs = node.attributes
            if node.node_type == NodeType.CANONICAL_ENTITY:
                for alias in (node.summary, *attrs.get("aliases", ())):
                    key = normalize_key(str(alias))
                    if key:
                        owner_alias[key].add(node.node_id)
            if node.node_type == NodeType.CANONICAL_FACT:
                owner = str(attrs.get("owner_id", "")); predicate = normalize_key(str(attrs.get("predicate", "")))
                scope = normalize_key(str(attrs.get("scope", ""))); value = normalize_key(str(attrs.get("value_key", attrs.get("value", ""))))
                self.fact_owner_predicate_scope_index[(owner, predicate, scope)].append(node.node_id)
                if owner: self.owner_fact_index[owner].append(node.node_id)
                if predicate: self.predicate_fact_index[predicate].append(node.node_id)
                if scope: self.scope_fact_index[scope].append(node.node_id)
                if value: self.value_fact_index[value].append(node.node_id)
                collection = normalize_key(str(attrs.get("collection_key", "")))
                if collection: self.collection_fact_index[collection].append(node.node_id)
                session = str(attrs.get("session_id", ""))
                if session and owner and predicate:
                    self.session_fact_index[(session, owner, predicate)].append(node.node_id)
            elif node.node_type in {NodeType.COLLECTION_SCOPE, NodeType.COLLECTION_MANIFEST}:
                key = normalize_key(str(attrs.get("collection_key", "")))
                if key: self.collection_index[key].append(node.node_id)
                for value in attrs.get("member_value_keys", ()):
                    key = normalize_key(str(value))
                    if key: self.manifest_value_index[key].append(node.node_id)
            elif node.node_type == NodeType.ROUTING_CARD:
                for key, child_ids in dict(attrs.get("child_postings", {})).items():
                    normalized = normalize_key(str(key))
                    if normalized:
                        self.routing_child_postings[normalized].extend(str(row) for row in child_ids)
        self.owner_alias_index = {key: tuple(sorted(value)) for key, value in owner_alias.items()}
        self.predicate_index = tuple(sorted(self.predicate_fact_index))
        for name in (
            "fact_owner_predicate_scope_index", "owner_fact_index", "predicate_fact_index",
            "scope_fact_index", "value_fact_index", "collection_fact_index", "session_fact_index",
            "routing_child_postings", "collection_index",
            "manifest_value_index",
        ):
            value = getattr(self, name)
            setattr(self, name, {key: tuple(sorted(set(rows))) for key, rows in value.items()})

    def lookup_facts(self, *, owner_ids: Sequence[str] = (), predicates: Sequence[str] = (),
                     scopes: Sequence[str] = (), values: Sequence[str] = (), limit: int = 64) -> tuple[str, ...]:
        pools: list[set[str]] = []
        if owner_ids:
            pools.append(set().union(*(set(self.owner_fact_index.get(item, ())) for item in owner_ids)))
        predicate_keys = [normalize_key(item) for item in predicates if normalize_key(item)]
        if predicate_keys:
            matched = set().union(*(set(self.predicate_fact_index.get(item, ())) for item in predicate_keys))
            # Query compiler candidates can be a normalized stem ("travel") while
            # a fact retains a supported phrase ("travel to").  Token containment
            # is deterministic and avoids an embedding-only predicate join.
            for indexed, fact_ids in self.predicate_fact_index.items():
                indexed_terms = set(content_terms(indexed))
                if any(set(content_terms(item)) <= indexed_terms or indexed_terms <= set(content_terms(item))
                       for item in predicate_keys):
                    matched.update(fact_ids)
            pools.append(matched)
        scope_keys = [normalize_key(item) for item in scopes if normalize_key(item)]
        if scope_keys:
            pools.append(set().union(*(set(self.scope_fact_index.get(item, ())) for item in scope_keys)))
        value_keys = [normalize_key(item) for item in values if normalize_key(item)]
        if value_keys:
            pools.append(set().union(*(set(self.value_fact_index.get(item, ())) for item in value_keys)))
        if not pools:
            return ()
        result = set.intersection(*pools) if len(pools) > 1 else pools[0]
        return tuple(sorted(result)[:limit])

    def route_children(self, keys: Sequence[str], *, limit: int = 48) -> tuple[str, ...]:
        rows: list[str] = []
        for key in keys:
            rows.extend(self.routing_child_postings.get(normalize_key(key), ()))
        return tuple(dict.fromkeys(rows))[:limit]

    def collection_facts(self, collection_key: str, *, limit: int = 64) -> tuple[str, ...]:
        return self.collection_fact_index.get(normalize_key(collection_key), ())[:limit]

    def terminal_groups_for_nodes(self, node_ids: Sequence[str]) -> tuple[str, ...]:
        return self.evidence_group_ids_for_nodes(node_ids, terminal_only=True)

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
