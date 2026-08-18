from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import threading
import time
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

    def __init__(self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge], *,
                 graph_version: int = 0, graph_checksum: str = "") -> None:
        # Version metadata travels with the immutable projection.  Query code
        # can therefore identify the exact snapshot it is using without going
        # back to SQLite for a second, potentially newer authority value.
        self.graph_version = graph_version
        self.graph_checksum = graph_checksum
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
        self.provenance_bitset = {
            node_id: frozenset(node.all_evidence_group_ids)
            for node_id, node in self.nodes.items()
        }
        # Terminal/routing projections share the exact immutable frozenset with
        # the all-node map.  The previous comprehensions allocated two equal
        # provenance sets for almost every node in every worker.
        self.terminal_provenance_bitset = {
            node_id: self.provenance_bitset[node_id]
            for node_id, node in self.nodes.items()
            if node.attributes.get("provenance_scope", "terminal") == "terminal"
        }
        self.routing_provenance_bitset = {
            node_id: self.provenance_bitset[node_id]
            for node_id, node in self.nodes.items()
            if node.attributes.get("provenance_scope") == "route"
        }
        self._compile_query_index()
        # A deterministic, inexpensive weight for byte-bounded snapshot caching.
        # It deliberately includes serialized attribute/provenance payloads,
        # which dominate real V5.8 snapshots, rather than counting only objects.
        self.estimated_bytes = (
            sum(256 + len(node.summary.encode("utf-8"))
                + len(repr(node.attributes).encode("utf-8"))
                + sum(len(item) for item in node.all_evidence_group_ids)
                for node in self.nodes.values())
            + sum(192 + sum(len(item) for item in edge.all_evidence_group_ids)
                  for edge in self.edges.values())
            + 16 * sum(len(rows) for rows in self.node_term_index.values())
        )

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
        # Reverse provenance: which facts cite a given evidence group.  Source
        # turns are already fully reachable, so projecting back through their
        # groups finds facts that no owner/predicate posting would return.
        self.evidence_group_fact_index: dict[str, list[str]] = defaultdict(list)
        # A compact searchable surface per fact, so a fact can be found
        # lexically without giving every fact its own embedding.
        self.fact_term_index: dict[str, list[str]] = defaultdict(list)
        self.fact_owner_predicate_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        # Generic immutable lexical postings remove the per-query O(|V|) scan
        # previously performed by retrieval.seeding._lexical_nodes.
        self.node_term_index: dict[str, list[str]] = defaultdict(list)
        self.route_term_index: dict[str, list[str]] = defaultdict(list)
        self.node_terms: dict[str, frozenset[str]] = {}
        self.route_child_postings_by_parent: dict[
            str, dict[str, tuple[str, ...]]
        ] = {}
        for node in self.nodes.values():
            attrs = node.attributes
            searchable = " ".join(str(value) for value in (
                node.summary,
                attrs.get("owner_id", ""), attrs.get("predicate", ""),
                attrs.get("scope", ""), attrs.get("collection_key", ""),
                attrs.get("value", ""), " ".join(map(str, attrs.get("owners", ()))),
                " ".join(map(str, attrs.get("routing_atoms", ()))),
                " ".join(map(str, attrs.get("predicates", ()))),
                " ".join(map(str, attrs.get("values", ()))),
                " ".join(map(str, attrs.get("scopes", ()))),
                " ".join(map(str, attrs.get("times", ()))),
            ))
            searchable_terms = content_terms(searchable)
            self.node_terms[node.node_id] = searchable_terms
            for term in searchable_terms:
                self.node_term_index[term].append(node.node_id)
            if ("route" in self.role_bitset.get(node.node_id, ())
                    or node.node_type in {NodeType.ROUTING_CARD, NodeType.SCENE,
                                         NodeType.COLLECTION_SCOPE,
                                         NodeType.COLLECTION_MANIFEST}):
                for term in searchable_terms:
                    self.route_term_index[term].append(node.node_id)
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
                if owner and predicate:
                    self.fact_owner_predicate_index[(owner, predicate)].append(node.node_id)
                for group_id in node.all_evidence_group_ids:
                    self.evidence_group_fact_index[group_id].append(node.node_id)
                surface = " ".join((
                    node.summary, predicate, str(attrs.get("value", "")), scope, collection,
                    str(attrs.get("value_type", "")), str(attrs.get("modality", "")),
                ))
                for term in content_terms(surface):
                    self.fact_term_index[term].append(node.node_id)
            elif node.node_type in {NodeType.COLLECTION_SCOPE, NodeType.COLLECTION_MANIFEST}:
                key = normalize_key(str(attrs.get("collection_key", "")))
                if key: self.collection_index[key].append(node.node_id)
                # Collection manifests persist this field as ``value_keys``.
                # Older experimental snapshots briefly used
                # ``member_value_keys``; accept both so a reader can open either
                # schema without silently compiling an empty value index.
                values = attrs.get("value_keys", attrs.get("member_value_keys", ()))
                for value in values:
                    key = normalize_key(str(value))
                    if key: self.manifest_value_index[key].append(node.node_id)
            elif node.node_type == NodeType.ROUTING_CARD:
                parent_postings: dict[str, tuple[str, ...]] = {}
                for key, child_ids in dict(attrs.get("child_postings", {})).items():
                    normalized = normalize_key(str(key))
                    if normalized:
                        children = tuple(str(row) for row in child_ids)
                        self.routing_child_postings[normalized].extend(children)
                        parent_postings[normalized] = children
                self.route_child_postings_by_parent[node.node_id] = parent_postings

        # Directional hierarchy indexes.  REFINES_TO and SCENE_CONTAINS are
        # control-plane edges; treating their inverse as an ordinary semantic
        # neighbour lets a query jump back to the root and fan out globally.
        hierarchy_children: dict[str, list[str]] = defaultdict(list)
        hierarchy_parents: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges.values():
            if edge.relation not in {RelationType.REFINES_TO, RelationType.SCENE_CONTAINS}:
                continue
            if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
                continue
            hierarchy_children[edge.src_id].append(edge.dst_id)
            hierarchy_parents[edge.dst_id].append(edge.src_id)
        self.hierarchy_children_index = {
            key: tuple(sorted(set(rows), key=lambda item: (
                -self.nodes[item].level, item)))
            for key, rows in hierarchy_children.items()
        }
        self.hierarchy_parent_index = {
            key: tuple(sorted(set(rows), key=lambda item: (
                self.nodes[item].level, item)))
            for key, rows in hierarchy_parents.items()
        }
        cards = [node for node in self.nodes.values()
                 if node.node_type == NodeType.ROUTING_CARD]
        roots = [node.node_id for node in cards
                 if node.node_id not in self.hierarchy_parent_index]
        if not roots and cards:
            highest = max(node.level for node in cards)
            roots = [node.node_id for node in cards if node.level == highest]
        self.route_root_ids = tuple(sorted(roots))
        levels: dict[int, list[str]] = defaultdict(list)
        for node in cards:
            levels[node.level].append(node.node_id)
        self.routing_nodes_by_level = {
            level: tuple(sorted(rows)) for level, rows in levels.items()
        }
        self.owner_alias_index = {key: tuple(sorted(value)) for key, value in owner_alias.items()}
        self.predicate_index = tuple(sorted(self.predicate_fact_index))
        for name in (
            "fact_owner_predicate_scope_index", "owner_fact_index", "predicate_fact_index",
            "scope_fact_index", "value_fact_index", "collection_fact_index", "session_fact_index",
            "routing_child_postings", "collection_index",
            "manifest_value_index", "evidence_group_fact_index", "fact_term_index",
            "fact_owner_predicate_index", "node_term_index", "route_term_index",
        ):
            value = getattr(self, name)
            setattr(self, name, {key: tuple(sorted(set(rows))) for key, rows in value.items()})
        self.predicate_term_index = {
            key: content_terms(key) for key in self.predicate_fact_index
        }
        predicates_by_owner: dict[
            str, list[tuple[str, tuple[str, ...]]]
        ] = defaultdict(list)
        for (owner, predicate), fact_ids in self.fact_owner_predicate_index.items():
            predicates_by_owner[owner].append((predicate, fact_ids))
        self.fact_predicates_by_owner = {
            # Preserve the composite index's insertion order: callers truncate
            # after deduplication, so sorting here could change retrieval output.
            owner: tuple(rows)
            for owner, rows in predicates_by_owner.items()
        }
        self.collection_node_ids = tuple(
            node.node_id for node in self.nodes.values()
            if node.node_type in {NodeType.COLLECTION_SCOPE,
                                  NodeType.COLLECTION_MANIFEST})

    def _fact_rank(self, node_id: str, owner_ids: frozenset[str],
                   predicate_terms: Sequence[frozenset[str]],
                   rank_terms: frozenset[str]) -> tuple[float, str]:
        """Relevance for truncation: owner, predicate, then lexical overlap."""
        node = self.nodes.get(node_id)
        if node is None:
            return (0.0, node_id)
        attrs = node.attributes
        score = 0.0
        if owner_ids and str(attrs.get("owner_id", "")) in owner_ids:
            score += 2.0
        predicate = normalize_key(str(attrs.get("predicate", "")))
        if predicate_terms:
            indexed_terms = self.predicate_term_index.get(predicate)
            if indexed_terms is None:
                indexed_terms = content_terms(predicate)
            score += max((len(indexed_terms & item) for item in predicate_terms),
                         default=0) * 0.75
        if rank_terms:
            # ``content_terms`` is process-bounded and cached.  Compute this
            # exact ranking surface lazily instead of storing a second term set
            # for every node in every cached memory view.
            haystack = content_terms(
                " ".join(str(attrs.get(key, "")) for key in
                         ("predicate", "value", "scope", "collection_key"))
                + " " + node.summary)
            score += len(rank_terms & haystack) * 0.25
        return (score, node_id)

    def lookup_collections(self, *, owner_ids: Sequence[str] = (), predicates: Sequence[str] = (),
                           limit: int = 32) -> tuple[str, ...]:
        """Collection scopes/manifests for an owner-predicate need.

        ``collection_index`` and ``manifest_value_index`` were compiled but never
        consulted; list/count questions need them as a direct entry point.
        """
        owners = frozenset(owner_ids)
        predicate_terms = [set(content_terms(normalize_key(item))) for item in predicates
                           if normalize_key(item)]
        rows: list[tuple[float, str]] = []
        for node_id in self.collection_node_ids:
            node = self.nodes[node_id]
            attrs = node.attributes
            score = 0.0
            if owners and str(attrs.get("owner_id", "")) in owners:
                score += 2.0
            elif owners:
                continue
            if predicate_terms:
                indexed = set(content_terms(normalize_key(str(attrs.get("predicate", "")))))
                score += max((len(indexed & item) for item in predicate_terms), default=0)
            if score:
                rows.append((score, node.node_id))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return tuple(node_id for _, node_id in rows[:limit])

    def lookup_facts(self, *, owner_ids: Sequence[str] = (), predicates: Sequence[str] = (),
                     scopes: Sequence[str] = (), values: Sequence[str] = (), limit: int = 64,
                     rank_terms: frozenset[str] = frozenset()) -> tuple[str, ...]:
        pools: list[set[str]] = []
        if owner_ids:
            pools.append(set().union(*(set(self.owner_fact_index.get(item, ())) for item in owner_ids)))
        predicate_keys = [normalize_key(item) for item in predicates if normalize_key(item)]
        predicate_terms = [content_terms(item) for item in predicate_keys]
        if predicate_keys:
            matched = set().union(*(set(self.predicate_fact_index.get(item, ())) for item in predicate_keys))
            # Query compiler candidates can be a normalized stem ("travel") while
            # a fact retains a supported phrase ("travel to").  Token containment
            # is deterministic and avoids an embedding-only predicate join.
            for indexed, fact_ids in self.predicate_fact_index.items():
                indexed_terms = self.predicate_term_index[indexed]
                if any(item <= indexed_terms or indexed_terms <= item
                       for item in predicate_terms):
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
        if len(result) <= limit:
            return tuple(sorted(result))
        # Truncating by node id sorted lexicographically discards facts for no
        # reason related to the query; rank first, then cut.
        owners = frozenset(owner_ids)
        ranked = sorted(result, key=lambda node_id: (
            -self._fact_rank(node_id, owners, predicate_terms, rank_terms)[0], node_id))
        return tuple(ranked[:limit])

    def route_children(self, keys: Sequence[str], *, limit: int = 48) -> tuple[str, ...]:
        rows: list[str] = []
        for key in keys:
            rows.extend(self.routing_child_postings.get(normalize_key(key), ()))
        return tuple(dict.fromkeys(rows))[:limit]

    def routing_roots(self) -> tuple[str, ...]:
        return self.route_root_ids

    def hierarchy_children(self, node_id: str) -> tuple[str, ...]:
        """Children only; this API never follows a control edge backwards."""
        return self.hierarchy_children_index.get(node_id, ())

    def hierarchy_parents(self, node_id: str) -> tuple[str, ...]:
        return self.hierarchy_parent_index.get(node_id, ())

    def parent_posting_children(self, parent_id: str, keys: Sequence[str]) -> tuple[str, ...]:
        postings = self.route_child_postings_by_parent.get(parent_id, {})
        rows: list[str] = []
        for key in keys:
            normalized = normalize_key(str(key))
            rows.extend(postings.get(normalized, ()))
            for term in content_terms(normalized):
                rows.extend(postings.get(term, ()))
        return tuple(dict.fromkeys(row for row in rows if row in self.nodes))

    def lexical_nodes(self, query_terms: frozenset[str], *, limit: int = 64,
                      candidates: Sequence[str] | None = None,
                      route_only: bool = False) -> tuple[str, ...]:
        """Rank an inverted-index candidate set without scanning every node."""
        if not query_terms or limit <= 0:
            return ()
        index = self.route_term_index if route_only else self.node_term_index
        allowed = frozenset(candidates) if candidates is not None else None
        hits: dict[str, int] = {}
        for term in query_terms:
            for node_id in index.get(term, ()):
                if allowed is not None and node_id not in allowed:
                    continue
                hits[node_id] = hits.get(node_id, 0) + 1
        ranked = sorted(hits, key=lambda node_id: (
            -hits[node_id] / max(1, len(query_terms)),
            -float(self.nodes[node_id].confidence), node_id))
        return tuple(ranked[:limit])

    def collection_facts(self, collection_key: str, *, limit: int = 64) -> tuple[str, ...]:
        return self.collection_fact_index.get(normalize_key(collection_key), ())[:limit]

    def facts_for_evidence_groups(self, group_ids: Sequence[str], *,
                                  limit: int = 256) -> tuple[str, ...]:
        """Facts whose provenance cites any of these evidence groups."""
        rows: list[str] = []
        for group_id in group_ids:
            rows.extend(self.evidence_group_fact_index.get(group_id, ()))
        return tuple(dict.fromkeys(rows))[:limit]

    def facts_for_terms(self, query_terms: frozenset[str], *, limit: int = 64) -> tuple[str, ...]:
        """Facts ranked by how much of their searchable surface the query hits."""
        if not query_terms:
            return ()
        hits: dict[str, int] = {}
        for term in query_terms:
            for node_id in self.fact_term_index.get(term, ()):
                hits[node_id] = hits.get(node_id, 0) + 1
        ranked = sorted(hits.items(), key=lambda row: (-row[1], row[0]))
        return tuple(node_id for node_id, _ in ranked[:limit])

    def facts_for_owner_predicate(self, owner_ids: Sequence[str], predicates: Sequence[str], *,
                                  limit: int = 128) -> tuple[str, ...]:
        """Composite (owner, predicate) postings, with token-containment widening.

        A predicate candidate is often a stem ("travel") while the fact keeps a
        phrase ("travel to"), so exact key equality alone under-retrieves.
        """
        predicate_keys = [normalize_key(item) for item in predicates if normalize_key(item)]
        predicate_terms = [content_terms(item) for item in predicate_keys]
        rows: list[str] = []
        for owner in owner_ids:
            for indexed_predicate, fact_ids in self.fact_predicates_by_owner.get(
                    owner, ()):
                indexed_terms = self.predicate_term_index.get(indexed_predicate)
                if indexed_terms is None:
                    indexed_terms = content_terms(indexed_predicate)
                if not predicate_keys or any(
                        item <= indexed_terms or indexed_terms <= item
                        for item in predicate_terms):
                    rows.extend(fact_ids)
        return tuple(dict.fromkeys(rows))[:limit]

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

    def __init__(self, store: SQLiteGraphStore, *, max_cached_views: int = 16,
                 max_cache_bytes: int = 512 * 1024 * 1024) -> None:
        self.store = store
        self.max_cached_views = max(1, max_cached_views)
        self.max_cache_bytes = max(1, max_cache_bytes)
        self._views: OrderedDict[tuple[str, int], GraphReadView] = OrderedDict()
        self._view_bytes: dict[tuple[str, int], int] = {}
        self._cache_bytes = 0
        self._lock = threading.RLock()
        self._building: dict[tuple[str, int], threading.Event] = {}
        self._hits = 0
        self._misses = 0
        self._builds = 0
        self._build_waits = 0
        self._evictions = 0
        self._invalidations = 0
        self._build_ms = 0.0

    def peek(self, memory_id: str, graph_version: int | None = None) -> GraphReadView | None:
        """Inspect the retained LRU without querying SQLite or compiling a view."""
        with self._lock:
            if graph_version is not None:
                return self._views.get((memory_id, graph_version))
            for key in reversed(self._views):
                if key[0] == memory_id:
                    return self._views[key]
        return None

    def lru_keys(self) -> tuple[tuple[str, int], ...]:
        """Oldest-to-newest retained keys for an external admission policy."""
        with self._lock:
            return tuple(self._views)

    def touch(self, memory_id: str, graph_version: int) -> GraphReadView | None:
        """Record an externally validated cache hit without another SQL probe."""
        key = (memory_id, graph_version)
        with self._lock:
            view = self._views.get(key)
            if view is not None:
                self._views.move_to_end(key)
                self._hits += 1
            return view

    def _retain_locked(self, key: tuple[str, int], view: GraphReadView, *,
                       accounted_bytes: int | None = None) -> GraphReadView:
        for old_key in [item for item in self._views if item[0] == key[0]]:
            old = self._views.pop(old_key)
            del old
            self._cache_bytes -= self._view_bytes.pop(old_key)
            self._invalidations += 1
        retained_bytes = max(1, int(
            accounted_bytes if accounted_bytes is not None else view.estimated_bytes))
        self._views[key] = view
        self._view_bytes[key] = retained_bytes
        self._cache_bytes += retained_bytes
        while (len(self._views) > self.max_cached_views
               or self._cache_bytes > self.max_cache_bytes) and len(self._views) > 1:
            evicted_key, _evicted = self._views.popitem(last=False)
            self._cache_bytes -= self._view_bytes.pop(evicted_key)
            self._evictions += 1
        return view

    def install(self, view: GraphReadView, *, memory_id: str | None = None,
                accounted_bytes: int | None = None) -> GraphReadView:
        """Atomically retain an already validated immutable compiled view."""
        resolved_memory = (memory_id or
                           (next(iter(view.nodes.values())).memory_id
                            if view.nodes else ""))
        key = (resolved_memory, view.graph_version)
        if not key[0]:
            raise ValueError("cannot install an empty view without a memory id")
        with self._lock:
            cached = self._views.get(key)
            if cached is not None and cached.graph_checksum == view.graph_checksum:
                self._views.move_to_end(key)
                return cached
            return self._retain_locked(
                key, view, accounted_bytes=accounted_bytes)

    def view(self, memory_id: str) -> GraphReadView:
        version = self.store.graph_version(memory_id)
        key = (memory_id, version)
        while True:
            with self._lock:
                cached = self._views.get(key)
                if cached is not None:
                    self._hits += 1
                    self._views.move_to_end(key)
                    return cached
                event = self._building.get(key)
                if event is None:
                    event = threading.Event()
                    self._building[key] = event
                    self._misses += 1
                    owner = True
                else:
                    self._build_waits += 1
                    owner = False
            if owner:
                break
            # Single-flight: concurrent cold requests wait for one compiler
            # instead of each materializing the same graph and multiplying RAM.
            event.wait()

        build_started = time.perf_counter()
        try:
            (snapshot_version, snapshot_checksum,
             snapshot_nodes, snapshot_edges) = self.store.graph_snapshot(memory_id)
            built = GraphReadView(
                snapshot_nodes, snapshot_edges,
                graph_version=snapshot_version,
                graph_checksum=snapshot_checksum,
            )
        except BaseException:
            with self._lock:
                self._building.pop(key, event).set()
            raise
        with self._lock:
            self._builds += 1
            self._build_ms += (time.perf_counter() - build_started) * 1000
            snapshot_key = (memory_id, snapshot_version)
            # A writer may commit between the optimistic version probe and the
            # snapshot transaction.  Cache under the version actually read,
            # never under the earlier probe.
            cached_snapshot = self._views.get(snapshot_key)
            if cached_snapshot is not None:
                self._views.move_to_end(snapshot_key)
                self._building.pop(key, event).set()
                return cached_snapshot
            self._retain_locked(snapshot_key, built)
            self._building.pop(key, event).set()
            return built

    def cache_stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "views": len(self._views),
                "estimated_bytes": self._cache_bytes,
                "accounted_bytes": self._cache_bytes,
                "inflight_builds": len(self._building),
                "max_views": self.max_cached_views,
                "max_bytes": self.max_cache_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "builds": self._builds,
                "build_waits": self._build_waits,
                "evictions": self._evictions,
                "invalidations": self._invalidations,
                "build_ms": self._build_ms,
            }

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
