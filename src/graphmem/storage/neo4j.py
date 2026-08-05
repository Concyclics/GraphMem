from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from ..domain import GraphEdge, GraphNode, canonical_json
from .sqlite import SQLiteGraphStore


def projection_checksum(nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]) -> str:
    node_rows = sorted(canonical_json({
        "node_id": node.node_id, "node_type": str(node.node_type), "level": node.level,
        "summary": node.summary[:512], "evidence_group_ids": node.all_evidence_group_ids,
        "entity_id": node.entity_id, "event_time": node.event_time, "state": node.state,
        "confidence": node.confidence,
        "provenance_scope": node.attributes.get("provenance_scope", "terminal"),
        "roles": tuple(node.attributes.get("roles", ())),
    }) for node in nodes)
    edge_rows = sorted(canonical_json({
        "edge_id": edge.edge_id, "src_id": edge.src_id, "relation": str(edge.relation),
        "dst_id": edge.dst_id, "evidence_group_ids": edge.all_evidence_group_ids,
        "directed": edge.directed, "confidence": edge.confidence, "source": edge.source,
    }) for edge in edges)
    return hashlib.sha256("\n".join([*node_rows, "--edges--", *edge_rows]).encode()).hexdigest()


class Neo4jProjector:
    """Idempotent winner-only projection; SQLite remains authoritative."""

    def __init__(self, store: SQLiteGraphStore, uri: str, auth: tuple[str, str],
                 *, node_batch: int = 1000, edge_batch: int = 2000,
                 driver: Any | None = None) -> None:
        self.store = store
        self.node_batch = node_batch
        self.edge_batch = edge_batch
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=auth)
        self.driver = driver
        self.ensure_schema()

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT graphmem_node_id IF NOT EXISTS FOR (n:GraphMemNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT graphmem_projection_memory IF NOT EXISTS FOR (m:GraphMemProjection) REQUIRE m.memory_id IS UNIQUE",
            "CREATE INDEX graphmem_node_memory IF NOT EXISTS FOR (n:GraphMemNode) ON (n.memory_id)",
            "CREATE INDEX graphmem_edge_id IF NOT EXISTS FOR ()-[r:GRAPHMEM_EDGE]-() ON (r.edge_id)",
        )
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement).consume()
            session.run("CALL db.awaitIndexes(300)").consume()

    def close(self) -> None:
        self.driver.close()

    def project_pending(self, *, max_events: int | None = None) -> int:
        query = "SELECT event_id,memory_id FROM outbox WHERE projected_at IS NULL ORDER BY event_id"
        params: tuple[Any, ...] = ()
        if max_events is not None:
            query += " LIMIT ?"
            params = (max_events,)
        rows = list(self.store._connection.execute(query, params))
        projected = 0
        for row in rows:
            self.rebuild(row["memory_id"])
            with self.store.transaction() as db:
                db.execute("UPDATE outbox SET projected_at=CURRENT_TIMESTAMP WHERE event_id=?", (row["event_id"],))
            projected += 1
        return projected

    def rebuild(self, memory_id: str) -> str:
        nodes = list(self.store.nodes(memory_id))
        edges = list(self.store.edges(memory_id))
        checksum = projection_checksum(nodes, edges)
        with self.driver.session() as session:
            session.run("MATCH (n:GraphMemNode {memory_id:$memory_id}) DETACH DELETE n",
                        memory_id=memory_id).consume()
            for start in range(0, len(nodes), self.node_batch):
                rows = [{
                    "node_id": node.node_id, "memory_id": memory_id,
                    "node_type": str(node.node_type), "level": node.level,
                    "summary": node.summary[:512],
                    "evidence_group_ids": list(node.all_evidence_group_ids),
                    "entity_id": node.entity_id, "event_time": node.event_time,
                    "state": node.state, "confidence": node.confidence,
                    "provenance_scope": node.attributes.get("provenance_scope", "terminal"),
                    "roles": list(node.attributes.get("roles", ())),
                    "graph_version": self.store.graph_version(memory_id),
                } for node in nodes[start:start + self.node_batch]]
                session.run(
                    "UNWIND $rows AS row MERGE (n:GraphMemNode {node_id:row.node_id}) SET n += row",
                    rows=rows,
                ).consume()
            for start in range(0, len(edges), self.edge_batch):
                rows = [{
                    "edge_id": edge.edge_id, "src_id": edge.src_id, "dst_id": edge.dst_id,
                    "relation": str(edge.relation),
                    "evidence_group_ids": list(edge.all_evidence_group_ids),
                    "directed": edge.directed, "confidence": edge.confidence,
                    "source": edge.source, "memory_id": memory_id,
                } for edge in edges[start:start + self.edge_batch]]
                # Relationship type stays fixed; semantic type is a short property.
                session.run(
                    "UNWIND $rows AS row MATCH (a:GraphMemNode {node_id:row.src_id}) "
                    "MATCH (b:GraphMemNode {node_id:row.dst_id}) "
                    "MERGE (a)-[r:GRAPHMEM_EDGE {edge_id:row.edge_id}]->(b) SET r += row",
                    rows=rows,
                ).consume()
            session.run(
                "MERGE (m:GraphMemProjection {memory_id:$memory_id}) SET "
                "m.graph_version=$version,m.projection_checksum=$checksum,m.node_count=$nodes,m.edge_count=$edges",
                memory_id=memory_id, version=self.store.graph_version(memory_id), checksum=checksum,
                nodes=len(nodes), edges=len(edges),
            ).consume()
            record = session.run(
                "MATCH (m:GraphMemProjection {memory_id:$memory_id}) "
                "OPTIONAL MATCH (n:GraphMemNode {memory_id:$memory_id}) "
                "WITH m,count(n) AS nodes OPTIONAL MATCH (:GraphMemNode {memory_id:$memory_id})"
                "-[r:GRAPHMEM_EDGE]->() RETURN m.projection_checksum AS checksum,nodes,count(r) AS edges",
                memory_id=memory_id,
            ).single()
        if not record or record["checksum"] != checksum or int(record["nodes"]) != len(nodes) or int(record["edges"]) != len(edges):
            raise RuntimeError("SQLite to Neo4j projection checksum/count mismatch")
        return checksum
