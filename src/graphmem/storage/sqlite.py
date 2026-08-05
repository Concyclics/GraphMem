from __future__ import annotations

import json
import math
import sqlite3
import threading
from array import array
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..domain import (
    Conversation,
    EvidenceGroup,
    EvidenceMember,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
    Session,
    SourceTurn,
    canonical_json,
)


MIGRATION_VERSION = 3


class SQLiteGraphStore:
    """Single-writer SQLite authority for raw memory and canonical graphs."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if read_only:
            self._connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True,
                                               check_same_thread=False, timeout=60.0)
        else:
            self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=60.0)
        self._connection.row_factory = sqlite3.Row
        if not read_only:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=60000")
        if not read_only:
            self._migrate()

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("cannot write through a read-only SQLiteGraphStore")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _migrate(self) -> None:
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations(
                    version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS conversations(
                    memory_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, source_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL, schema_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions(
                    session_id TEXT NOT NULL, memory_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    timestamp TEXT, content_hash TEXT NOT NULL, schema_version TEXT NOT NULL,
                    PRIMARY KEY(memory_id, session_id),
                    FOREIGN KEY(memory_id) REFERENCES conversations(memory_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS sessions_memory_idx ON sessions(memory_id, ordinal);
                CREATE TABLE IF NOT EXISTS source_turns(
                    turn_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL, speaker TEXT NOT NULL, listener TEXT NOT NULL,
                    role TEXT NOT NULL, timestamp TEXT, raw_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL, schema_version TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES conversations(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY(memory_id,session_id) REFERENCES sessions(memory_id,session_id) ON DELETE CASCADE,
                    UNIQUE(memory_id,session_id,turn_index)
                );
                CREATE INDEX IF NOT EXISTS turns_memory_idx ON source_turns(memory_id, session_id, turn_index);
                CREATE TABLE IF NOT EXISTS graph_versions(
                    memory_id TEXT PRIMARY KEY, graph_version INTEGER NOT NULL,
                    graph_checksum TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS evidence_groups(
                    evidence_group_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL, min_time TEXT, max_time TEXT,
                    schema_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_memory_idx ON evidence_groups(memory_id);
                CREATE TABLE IF NOT EXISTS evidence_members(
                    evidence_group_id TEXT NOT NULL, member_ordinal INTEGER NOT NULL,
                    turn_id TEXT NOT NULL, span_start INTEGER NOT NULL, span_end INTEGER NOT NULL,
                    support_type TEXT NOT NULL,
                    PRIMARY KEY(evidence_group_id, member_ordinal),
                    FOREIGN KEY(evidence_group_id) REFERENCES evidence_groups(evidence_group_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS graph_nodes(
                    node_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, node_type TEXT NOT NULL,
                    level INTEGER NOT NULL, summary TEXT NOT NULL, evidence_group_id TEXT NOT NULL,
                    evidence_group_ids_json TEXT NOT NULL, entity_id TEXT, event_time TEXT, state TEXT,
                    confidence REAL NOT NULL, attributes_json TEXT NOT NULL, schema_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS nodes_memory_idx ON graph_nodes(memory_id, node_type, level);
                CREATE TABLE IF NOT EXISTS graph_edges(
                    edge_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, src_id TEXT NOT NULL,
                    relation TEXT NOT NULL, dst_id TEXT NOT NULL, evidence_group_id TEXT NOT NULL,
                    evidence_group_ids_json TEXT NOT NULL, directed INTEGER NOT NULL,
                    confidence REAL NOT NULL, source TEXT NOT NULL, schema_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS edges_src_idx ON graph_edges(memory_id, src_id, relation);
                CREATE INDEX IF NOT EXISTS edges_dst_idx ON graph_edges(memory_id, dst_id, relation);
                CREATE TABLE IF NOT EXISTS llm_cache(
                    cache_key TEXT PRIMARY KEY, stage TEXT NOT NULL, request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL, usage_json TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS llm_calls(
                    call_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, stage TEXT NOT NULL,
                    cache_key TEXT NOT NULL, cached INTEGER NOT NULL, request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL, usage_json TEXT NOT NULL, latency_ms REAL NOT NULL,
                    retry_count INTEGER NOT NULL, batch_size INTEGER NOT NULL,
                    prompt_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS embeddings(
                    item_id TEXT NOT NULL, memory_id TEXT NOT NULL, model_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(item_id,model_id)
                );
                CREATE INDEX IF NOT EXISTS embeddings_memory_idx ON embeddings(memory_id,model_id);
                CREATE TABLE IF NOT EXISTS embedding_calls(
                    call_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, model_id TEXT NOT NULL,
                    item_count INTEGER NOT NULL, input_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL, heartbeat INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS outbox(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL,
                    graph_version INTEGER NOT NULL, event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL, projected_at TEXT
                );
                CREATE TABLE IF NOT EXISTS run_ledger(
                    run_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            try:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS source_turns_fts USING fts5("
                    "turn_id UNINDEXED, memory_id UNINDEXED, session_id UNINDEXED, raw_text)"
                )
            except sqlite3.OperationalError:
                pass
            db.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (MIGRATION_VERSION,),
            )

    def ingest_conversation(
        self,
        conversation: Conversation,
        sessions: Sequence[Session],
        turns: Sequence[SourceTurn],
    ) -> int:
        if any(item.memory_id != conversation.memory_id for item in (*sessions, *turns)):
            raise ValueError("all imported rows must belong to the conversation")
        with self.transaction() as db:
            db.execute(
                "INSERT INTO conversations VALUES(?,?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET "
                "dataset=excluded.dataset,source_id=excluded.source_id,content_hash=excluded.content_hash,"
                "schema_version=excluded.schema_version",
                tuple(asdict(conversation).values()),
            )
            db.executemany(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?) ON CONFLICT(memory_id,session_id) DO UPDATE SET "
                "memory_id=excluded.memory_id,ordinal=excluded.ordinal,timestamp=excluded.timestamp,"
                "content_hash=excluded.content_hash,schema_version=excluded.schema_version",
                [tuple(asdict(item).values()) for item in sessions],
            )
            self._upsert_turns(db, turns)
        return len(turns)

    def ingest_turns(self, turns: Sequence[SourceTurn]) -> int:
        with self.transaction() as db:
            self._upsert_turns(db, turns)
        return len(turns)

    def _upsert_turns(self, db: sqlite3.Connection, turns: Sequence[SourceTurn]) -> None:
        columns = (
            "turn_id,memory_id,session_id,turn_index,speaker,listener,role,timestamp,"
            "raw_text,content_hash,schema_version"
        )
        db.executemany(
            f"INSERT INTO source_turns({columns}) VALUES({','.join('?' for _ in range(11))}) "
            "ON CONFLICT(turn_id) DO UPDATE SET raw_text=excluded.raw_text,"
            "content_hash=excluded.content_hash,speaker=excluded.speaker,listener=excluded.listener,"
            "role=excluded.role,timestamp=excluded.timestamp,schema_version=excluded.schema_version",
            [tuple(asdict(item).values()) for item in turns],
        )
        try:
            db.executemany("DELETE FROM source_turns_fts WHERE turn_id=?", [(x.turn_id,) for x in turns])
            db.executemany(
                "INSERT INTO source_turns_fts(turn_id,memory_id,session_id,raw_text) VALUES(?,?,?,?)",
                [(x.turn_id, x.memory_id, x.session_id, x.raw_text) for x in turns],
            )
        except sqlite3.OperationalError:
            pass

    def conversation(self, memory_id: str) -> Conversation | None:
        row = self._connection.execute(
            "SELECT * FROM conversations WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return Conversation(**dict(row)) if row else None

    def sessions(self, memory_id: str) -> Sequence[Session]:
        rows = self._connection.execute(
            "SELECT * FROM sessions WHERE memory_id=? ORDER BY ordinal,session_id", (memory_id,)
        )
        return [Session(**dict(row)) for row in rows]

    def turns(self, memory_id: str) -> Sequence[SourceTurn]:
        rows = self._connection.execute(
            "SELECT * FROM source_turns WHERE memory_id=? ORDER BY session_id,turn_index", (memory_id,)
        )
        return [SourceTurn(**dict(row)) for row in rows]

    def turns_by_ids(self, turn_ids: Sequence[str]) -> Sequence[SourceTurn]:
        if not turn_ids:
            return []
        rows = self._connection.execute(
            f"SELECT * FROM source_turns WHERE turn_id IN ({','.join('?' for _ in turn_ids)})",
            tuple(turn_ids),
        )
        found = {row["turn_id"]: SourceTurn(**dict(row)) for row in rows}
        return [found[item] for item in turn_ids if item in found]

    def search_turns(self, memory_id: str, query: str, *, limit: int = 64) -> list[tuple[str, float]]:
        terms = [term for term in query.replace("'", " ").split() if term]
        if not terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:24])
        try:
            rows = self._connection.execute(
                "SELECT turn_id,-bm25(source_turns_fts) AS score FROM source_turns_fts "
                "WHERE source_turns_fts MATCH ? AND memory_id=? ORDER BY bm25(source_turns_fts) LIMIT ?",
                (match, memory_id, limit),
            )
            return [(row["turn_id"], float(row["score"])) for row in rows]
        except sqlite3.OperationalError:
            pattern = "%" + "%".join(terms[:8]) + "%"
            rows = self._connection.execute(
                "SELECT turn_id,1.0 AS score FROM source_turns WHERE memory_id=? "
                "AND raw_text LIKE ? LIMIT ?", (memory_id, pattern, limit)
            )
            return [(row["turn_id"], float(row["score"])) for row in rows]

    def delete_turn(self, turn_id: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM source_turns WHERE turn_id=?", (turn_id,))
            try:
                db.execute("DELETE FROM source_turns_fts WHERE turn_id=?", (turn_id,))
            except sqlite3.OperationalError:
                pass

    def delete_session(self, session_id: str) -> None:
        with self.transaction() as db:
            turn_ids = [row[0] for row in db.execute(
                "SELECT turn_id FROM source_turns WHERE session_id=?", (session_id,)
            )]
            db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            try:
                db.executemany("DELETE FROM source_turns_fts WHERE turn_id=?", [(x,) for x in turn_ids])
            except sqlite3.OperationalError:
                pass

    def delete_memory(self, memory_id: str) -> None:
        with self.transaction() as db:
            turn_ids = [row[0] for row in db.execute(
                "SELECT turn_id FROM source_turns WHERE memory_id=?", (memory_id,)
            )]
            db.execute("DELETE FROM conversations WHERE memory_id=?", (memory_id,))
            for table in ("graph_nodes", "graph_edges", "evidence_groups", "graph_versions"):
                db.execute(f"DELETE FROM {table} WHERE memory_id=?", (memory_id,))
            try:
                db.executemany("DELETE FROM source_turns_fts WHERE turn_id=?", [(x,) for x in turn_ids])
            except sqlite3.OperationalError:
                pass

    def prepare_memory_shard(self, shard_path: str | Path, memory_id: str) -> None:
        """Create/resume a self-contained raw/cache shard for one memory."""
        shard_path = Path(shard_path)
        shard = SQLiteGraphStore(shard_path)
        shard.close()
        alias = "graphmem_memory_shard"
        with self._lock:
            self._connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_path),))
            try:
                with self.transaction() as db:
                    db.execute(
                        f"INSERT OR REPLACE INTO {alias}.conversations "
                        "SELECT * FROM main.conversations WHERE memory_id=?", (memory_id,)
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO {alias}.sessions "
                        "SELECT * FROM main.sessions WHERE memory_id=?", (memory_id,)
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO {alias}.source_turns "
                        "SELECT * FROM main.source_turns WHERE memory_id=?", (memory_id,)
                    )
                    db.execute(
                        f"INSERT OR IGNORE INTO {alias}.llm_cache "
                        "SELECT cache.* FROM main.llm_cache AS cache WHERE cache.cache_key IN ("
                        "SELECT DISTINCT cache_key FROM main.llm_calls WHERE memory_id=?)",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO {alias}.embeddings "
                        "SELECT * FROM main.embeddings WHERE memory_id=?", (memory_id,)
                    )
            finally:
                self._connection.execute(f"DETACH DATABASE {alias}")

    def merge_memory_shard(self, shard_path: str | Path, memory_id: str) -> int:
        """Atomically merge one completed memory graph and its LLM ledger."""
        alias = "graphmem_memory_shard"
        with self._lock:
            self._connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(Path(shard_path)),))
            try:
                with self.transaction() as db:
                    shard_memory = db.execute(
                        f"SELECT memory_id FROM {alias}.conversations"
                    ).fetchall()
                    if [str(row[0]) for row in shard_memory] != [memory_id]:
                        raise ValueError("memory shard must contain exactly the requested memory")
                    shard_version = db.execute(
                        f"SELECT graph_checksum FROM {alias}.graph_versions WHERE memory_id=?",
                        (memory_id,),
                    ).fetchone()
                    if not shard_version:
                        raise ValueError("memory shard has no completed graph")
                    previous = db.execute(
                        "SELECT graph_version FROM main.graph_versions WHERE memory_id=?", (memory_id,)
                    ).fetchone()
                    version = int(previous[0]) + 1 if previous else 1
                    old_groups = [row[0] for row in db.execute(
                        "SELECT evidence_group_id FROM main.evidence_groups WHERE memory_id=?", (memory_id,)
                    )]
                    db.execute("DELETE FROM main.graph_nodes WHERE memory_id=?", (memory_id,))
                    db.execute("DELETE FROM main.graph_edges WHERE memory_id=?", (memory_id,))
                    db.executemany("DELETE FROM main.evidence_groups WHERE evidence_group_id=?",
                                   [(item,) for item in old_groups])
                    db.execute(
                        f"INSERT INTO main.evidence_groups SELECT * FROM {alias}.evidence_groups "
                        "WHERE memory_id=?", (memory_id,)
                    )
                    db.execute(
                        f"INSERT INTO main.evidence_members SELECT member.* FROM {alias}.evidence_members member "
                        f"JOIN {alias}.evidence_groups evidence USING(evidence_group_id) WHERE evidence.memory_id=?",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT INTO main.graph_nodes SELECT * FROM {alias}.graph_nodes WHERE memory_id=?",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT INTO main.graph_edges SELECT * FROM {alias}.graph_edges WHERE memory_id=?",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO main.llm_cache SELECT cache.* FROM {alias}.llm_cache cache "
                        f"WHERE cache.cache_key IN (SELECT cache_key FROM {alias}.llm_calls WHERE memory_id=?)",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO main.llm_calls SELECT * FROM {alias}.llm_calls WHERE memory_id=?",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO main.embeddings SELECT * FROM {alias}.embeddings WHERE memory_id=?",
                        (memory_id,),
                    )
                    db.execute(
                        f"INSERT OR REPLACE INTO main.embedding_calls SELECT * FROM {alias}.embedding_calls "
                        "WHERE memory_id=?", (memory_id,),
                    )
                    checksum = str(shard_version[0])
                    db.execute(
                        "INSERT INTO main.graph_versions(memory_id,graph_version,graph_checksum) VALUES(?,?,?) "
                        "ON CONFLICT(memory_id) DO UPDATE SET graph_version=excluded.graph_version,"
                        "graph_checksum=excluded.graph_checksum,updated_at=CURRENT_TIMESTAMP",
                        (memory_id, version, checksum),
                    )
                    db.execute(
                        "INSERT INTO main.outbox(memory_id,graph_version,event_type,payload_json) VALUES(?,?,?,?)",
                        (memory_id, version, "merge_memory_shard", canonical_json({"checksum": checksum})),
                    )
                return version
            finally:
                self._connection.execute(f"DETACH DATABASE {alias}")

    def replace_graph(
        self,
        memory_id: str,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        evidence_groups: Sequence[EvidenceGroup],
    ) -> int:
        if any(item.memory_id != memory_id for item in (*nodes, *edges, *evidence_groups)):
            raise ValueError("graph rows must share memory_id")
        from ..domain import logical_graph_checksum

        checksum = logical_graph_checksum(nodes, edges)
        with self.transaction() as db:
            previous = db.execute(
                "SELECT graph_version FROM graph_versions WHERE memory_id=?", (memory_id,)
            ).fetchone()
            version = int(previous[0]) + 1 if previous else 1
            db.execute("DELETE FROM graph_nodes WHERE memory_id=?", (memory_id,))
            db.execute("DELETE FROM graph_edges WHERE memory_id=?", (memory_id,))
            old_groups = [row[0] for row in db.execute(
                "SELECT evidence_group_id FROM evidence_groups WHERE memory_id=?", (memory_id,)
            )]
            db.executemany("DELETE FROM evidence_groups WHERE evidence_group_id=?", [(x,) for x in old_groups])
            for group in evidence_groups:
                db.execute(
                    "INSERT INTO evidence_groups VALUES(?,?,?,?,?,?)",
                    (group.evidence_group_id, memory_id, group.content_hash, group.min_time,
                     group.max_time, group.schema_version),
                )
                db.executemany(
                    "INSERT INTO evidence_members VALUES(?,?,?,?,?,?)",
                    [(group.evidence_group_id, index, member.turn_id, member.span_start,
                      member.span_end, member.support_type)
                     for index, member in enumerate(group.members)],
                )
            db.executemany(
                "INSERT INTO graph_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(x.node_id, x.memory_id, str(x.node_type), x.level, x.summary,
                  x.evidence_group_id, canonical_json(x.evidence_group_ids), x.entity_id,
                  x.event_time, x.state, x.confidence, canonical_json(x.attributes), x.schema_version)
                 for x in nodes],
            )
            db.executemany(
                "INSERT INTO graph_edges VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(x.edge_id, x.memory_id, x.src_id, str(x.relation), x.dst_id,
                  x.evidence_group_id, canonical_json(x.evidence_group_ids), int(x.directed),
                  x.confidence, x.source, x.schema_version) for x in edges],
            )
            db.execute(
                "INSERT INTO graph_versions(memory_id,graph_version,graph_checksum) VALUES(?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET graph_version=excluded.graph_version,"
                "graph_checksum=excluded.graph_checksum,updated_at=CURRENT_TIMESTAMP",
                (memory_id, version, checksum),
            )
            db.execute(
                "INSERT INTO outbox(memory_id,graph_version,event_type,payload_json) VALUES(?,?,?,?)",
                (memory_id, version, "replace_graph", canonical_json({"checksum": checksum})),
            )
        return version

    def nodes(self, memory_id: str) -> Sequence[GraphNode]:
        return [self._node(row) for row in self._connection.execute(
            "SELECT * FROM graph_nodes WHERE memory_id=? ORDER BY node_id", (memory_id,)
        )]

    def edges(self, memory_id: str) -> Sequence[GraphEdge]:
        return [self._edge(row) for row in self._connection.execute(
            "SELECT * FROM graph_edges WHERE memory_id=? ORDER BY edge_id", (memory_id,)
        )]

    def evidence_groups(self, memory_id: str) -> Sequence[EvidenceGroup]:
        result: list[EvidenceGroup] = []
        for row in self._connection.execute(
            "SELECT * FROM evidence_groups WHERE memory_id=? ORDER BY evidence_group_id", (memory_id,)
        ):
            members = tuple(EvidenceMember(
                item["turn_id"], item["span_start"], item["span_end"], item["support_type"]
            ) for item in self._connection.execute(
                "SELECT * FROM evidence_members WHERE evidence_group_id=? ORDER BY member_ordinal",
                (row["evidence_group_id"],),
            ))
            result.append(EvidenceGroup(
                row["evidence_group_id"], row["memory_id"], members, row["content_hash"],
                row["min_time"], row["max_time"], row["schema_version"],
            ))
        return result

    def evidence_group(self, evidence_group_id: str) -> EvidenceGroup | None:
        row = self._connection.execute(
            "SELECT * FROM evidence_groups WHERE evidence_group_id=?", (evidence_group_id,)
        ).fetchone()
        if not row:
            return None
        members = tuple(EvidenceMember(
            item["turn_id"], item["span_start"], item["span_end"], item["support_type"]
        ) for item in self._connection.execute(
            "SELECT * FROM evidence_members WHERE evidence_group_id=? ORDER BY member_ordinal",
            (evidence_group_id,),
        ))
        return EvidenceGroup(
            row["evidence_group_id"], row["memory_id"], members, row["content_hash"],
            row["min_time"], row["max_time"], row["schema_version"],
        )

    def graph_version(self, memory_id: str) -> int:
        row = self._connection.execute(
            "SELECT graph_version FROM graph_versions WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def graph_checksum(self, memory_id: str) -> str:
        row = self._connection.execute(
            "SELECT graph_checksum FROM graph_versions WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return str(row[0]) if row else ""

    def cache_get(self, cache_key: str) -> Mapping[str, Any] | None:
        row = self._connection.execute(
            "SELECT response_json,usage_json FROM llm_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return ({"response": json.loads(row[0]), "usage": json.loads(row[1])} if row else None)

    def cache_put(self, cache_key: str, stage: str, request: Any, response: Any,
                  usage: Any, prompt_hash: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO llm_cache(cache_key,stage,request_json,response_json,usage_json,prompt_hash) "
                "VALUES(?,?,?,?,?,?)",
                (cache_key, stage, canonical_json(request), canonical_json(response),
                 canonical_json(usage), prompt_hash),
            )

    def log_llm_call(self, *, call_id: str, memory_id: str, stage: str, cache_key: str,
                     cached: bool, request: Any, response: Any, usage: Any,
                     latency_ms: float, retry_count: int, batch_size: int,
                     prompt_hash: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO llm_calls VALUES(?,?,?,?,?,?,?,?,?,?,?, ?,CURRENT_TIMESTAMP)",
                (call_id, memory_id, stage, cache_key, int(cached), canonical_json(request),
                 canonical_json(response), canonical_json(usage), latency_ms, retry_count,
                 batch_size, prompt_hash),
            )

    def upsert_embeddings(self, memory_id: str, model_id: str,
                          rows: Sequence[tuple[str, str, Sequence[float]]]) -> int:
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO embeddings(item_id,memory_id,model_id,content_hash,dimension,vector) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(item_id,model_id) DO UPDATE SET "
                "memory_id=excluded.memory_id,content_hash=excluded.content_hash,"
                "dimension=excluded.dimension,vector=excluded.vector",
                [(item_id, memory_id, model_id, content_hash, len(vector),
                  array("f", (float(value) for value in vector)).tobytes())
                 for item_id, content_hash, vector in rows],
            )
        return len(rows)

    def embedding_hashes(self, memory_id: str, model_id: str) -> dict[str, str]:
        return {row["item_id"]: row["content_hash"] for row in self._connection.execute(
            "SELECT item_id,content_hash FROM embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, model_id),
        )}

    def search_embeddings(self, memory_id: str, model_id: str,
                          query_vector: Sequence[float], *, limit: int = 96) -> list[tuple[str, float]]:
        query = array("f", (float(value) for value in query_vector))
        query_norm = math.sqrt(sum(value * value for value in query)) or 1.0
        scores: list[tuple[str, float]] = []
        for row in self._connection.execute(
            "SELECT item_id,dimension,vector FROM embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, model_id),
        ):
            if int(row["dimension"]) != len(query):
                continue
            vector = array("f")
            vector.frombytes(row["vector"])
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            score = sum(left * right for left, right in zip(query, vector)) / (query_norm * norm)
            scores.append((row["item_id"], score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]

    def log_embedding_call(self, call_id: str, memory_id: str, model_id: str,
                           item_count: int, input_tokens: int, latency_ms: float,
                           *, heartbeat: bool = False) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO embedding_calls VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                (call_id, memory_id, model_id, item_count, input_tokens, latency_ms, int(heartbeat)),
            )

    @staticmethod
    def _node(row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            node_id=row["node_id"], memory_id=row["memory_id"], node_type=NodeType(row["node_type"]),
            level=row["level"], summary=row["summary"], evidence_group_id=row["evidence_group_id"],
            evidence_group_ids=tuple(json.loads(row["evidence_group_ids_json"])),
            entity_id=row["entity_id"], event_time=row["event_time"], state=row["state"],
            confidence=row["confidence"], attributes=json.loads(row["attributes_json"]),
            schema_version=row["schema_version"],
        )

    @staticmethod
    def _edge(row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            edge_id=row["edge_id"], memory_id=row["memory_id"], src_id=row["src_id"],
            relation=RelationType(row["relation"]), dst_id=row["dst_id"],
            evidence_group_id=row["evidence_group_id"], directed=bool(row["directed"]),
            confidence=row["confidence"], source=row["source"],
            evidence_group_ids=tuple(json.loads(row["evidence_group_ids_json"])),
            schema_version=row["schema_version"],
        )
