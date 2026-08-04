from __future__ import annotations

from typing import Any, Sequence

from ..domain import GraphEdge, GraphNode, NodeType, RelationType
from .read_view import GraphReadView


class Neo4jDirectRuntime:
    mode = "neo4j_direct"

    def __init__(self, uri: str, auth: tuple[str, str], driver: Any | None = None) -> None:
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=auth)
        self.driver = driver

    def close(self) -> None:
        self.driver.close()

    def graph_version(self, memory_id: str) -> int:
        with self.driver.session() as session:
            record = session.run(
                "MATCH (m:GraphMemProjection {memory_id:$memory_id}) RETURN m.graph_version AS version",
                memory_id=memory_id,
            ).single()
        return int(record["version"]) if record else 0

    def nodes(self, memory_id: str, node_ids: Sequence[str]) -> Sequence[GraphNode]:
        if not node_ids:
            return []
        with self.driver.session() as session:
            rows = list(session.run(
                "MATCH (n:GraphMemNode {memory_id:$memory_id}) WHERE n.node_id IN $node_ids RETURN n",
                memory_id=memory_id, node_ids=list(node_ids),
            ))
        found = {str(row["n"]["node_id"]): self._node(dict(row["n"])) for row in rows}
        return [found[node_id] for node_id in node_ids if node_id in found]

    def all_nodes(self, memory_id: str) -> list[GraphNode]:
        with self.driver.session() as session:
            return [self._node(dict(row["n"])) for row in session.run(
                "MATCH (n:GraphMemNode {memory_id:$memory_id}) RETURN n ORDER BY n.node_id",
                memory_id=memory_id,
            )]

    def all_edges(self, memory_id: str) -> list[GraphEdge]:
        with self.driver.session() as session:
            return [self._edge(dict(row["r"])) for row in session.run(
                "MATCH (:GraphMemNode {memory_id:$memory_id})-[r:GRAPHMEM_EDGE]->() "
                "RETURN r ORDER BY r.edge_id", memory_id=memory_id,
            )]

    def expand(self, memory_id: str, frontier: Sequence[str], relations: Sequence[RelationType],
               *, limit: int) -> Sequence[GraphEdge]:
        if not frontier:
            return []
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (a:GraphMemNode {memory_id:$memory_id})-[r:GRAPHMEM_EDGE]-(b:GraphMemNode) "
                "WHERE a.node_id IN $frontier AND r.relation IN $relations "
                "RETURN DISTINCT r ORDER BY r.edge_id LIMIT $limit",
                memory_id=memory_id, frontier=list(frontier),
                relations=[str(item) for item in relations], limit=limit,
            )
            return [self._edge(dict(row["r"])) for row in rows]

    @staticmethod
    def _node(row: dict[str, Any]) -> GraphNode:
        evidence = tuple(row.get("evidence_group_ids") or ())
        if not evidence:
            raise RuntimeError("projected node is missing provenance")
        return GraphNode(
            row["node_id"], row["memory_id"], NodeType(row["node_type"]), int(row["level"]),
            row.get("summary", ""), evidence[0], evidence[1:], row.get("entity_id"),
            row.get("event_time"), row.get("state"), float(row.get("confidence", 1.0)), {},
        )

    @staticmethod
    def _edge(row: dict[str, Any]) -> GraphEdge:
        evidence = tuple(row.get("evidence_group_ids") or ())
        if not evidence:
            raise RuntimeError("projected edge is missing provenance")
        return GraphEdge(
            row["edge_id"], row["memory_id"], row["src_id"], RelationType(row["relation"]),
            row["dst_id"], evidence[0], bool(row.get("directed", True)),
            float(row.get("confidence", 1.0)), row.get("source", "projected"), evidence[1:],
        )


class Neo4jCachedRuntime(Neo4jDirectRuntime):
    mode = "neo4j_cached"

    def __init__(self, uri: str, auth: tuple[str, str], driver: Any | None = None) -> None:
        super().__init__(uri, auth, driver)
        self._views: dict[tuple[str, int], GraphReadView] = {}

    def view(self, memory_id: str) -> GraphReadView:
        version = self.graph_version(memory_id)
        key = (memory_id, version)
        if key not in self._views:
            self._views = {item: value for item, value in self._views.items() if item[0] != memory_id}
            self._views[key] = GraphReadView(self.all_nodes(memory_id), self.all_edges(memory_id))
        return self._views[key]

    def nodes(self, memory_id: str, node_ids: Sequence[str]) -> Sequence[GraphNode]:
        view = self.view(memory_id)
        return [view.nodes[node_id] for node_id in node_ids if node_id in view.nodes]

    def expand(self, memory_id: str, frontier: Sequence[str], relations: Sequence[RelationType],
               *, limit: int) -> Sequence[GraphEdge]:
        view = self.view(memory_id)
        result, seen = [], set()
        for node_id in frontier:
            for row in view.neighbors(node_id, relations):
                if row.edge.edge_id in seen:
                    continue
                seen.add(row.edge.edge_id)
                result.append(row.edge)
                if len(result) >= limit:
                    return result
        return result
