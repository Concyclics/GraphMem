from __future__ import annotations

import json
import hashlib
import math
import queue
import sqlite3
import threading
from array import array
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..domain import (
    Conversation,
    EvidenceGroup,
    EvidenceMember,
    GraphChecksumState,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
    Session,
    SourceTurn,
    canonical_json,
    logical_graph_checksum_state,
    logical_graph_row_digest,
)


MIGRATION_VERSION = 5


@dataclass(frozen=True, slots=True)
class GraphDeltaResult:
    """Committed row-level graph update and its newly visible version."""

    graph_version: int
    graph_checksum: str
    upserted_nodes: int
    deleted_nodes: int
    upserted_edges: int
    deleted_edges: int
    upserted_evidence_groups: int
    deleted_evidence_groups: int

    @property
    def touched_rows(self) -> int:
        return (self.upserted_nodes + self.deleted_nodes
                + self.upserted_edges + self.deleted_edges
                + self.upserted_evidence_groups + self.deleted_evidence_groups)


@dataclass(frozen=True, slots=True)
class IncrementalJobRecord:
    """Durable progress of one idempotent raw-to-route indexing job."""

    job_id: str
    memory_id: str
    session_id: str
    source_offset: int
    state: str
    payload_hash: str
    payload: Mapping[str, Any]
    expected_version: int
    graph_version: int | None
    attempts: int
    last_error: str | None


class SQLiteGraphStore:
    """Single-writer SQLite authority for raw memory and canonical graphs."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._transaction_state = threading.local()
        self._read_pool: queue.LifoQueue[sqlite3.Connection] | None = None
        self._read_pool_connections: list[sqlite3.Connection] = []
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
        with self._lock:
            for connection in self._read_pool_connections:
                connection.close()
            self._read_pool_connections.clear()
            self._read_pool = None
            self._connection.close()

    def enable_read_pool(self, size: int = 4) -> int:
        """Open independent query-only connections for concurrent snapshot reads.

        SQLite WAL permits readers to proceed while the authority connection is
        publishing a new graph.  A pool is opt-in so build-only callers and
        in-memory test databases keep the historical single-connection behavior.
        """
        if size <= 0 or str(self.path) == ":memory:":
            return 0
        with self._lock:
            if self._read_pool is not None:
                return len(self._read_pool_connections)
            pool: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=size)
            uri = f"file:{self.path.resolve()}?mode=ro"
            for _ in range(size):
                connection = sqlite3.connect(uri, uri=True, check_same_thread=False,
                                             timeout=60.0)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA busy_timeout=60000")
                self._read_pool_connections.append(connection)
                pool.put(connection)
            self._read_pool = pool
            return size

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a read query under the store lock and materialize the rows.

        One ``sqlite3.Connection`` is shared by every thread (``check_same_thread``
        is off), and the build fans out to 16 extraction workers while a writer
        may hold ``BEGIN IMMEDIATE``.  Reading off that connection without the
        lock is unsynchronized concurrent use, which SQLite reports as
        ``InterfaceError: bad parameter or other API misuse`` -- intermittently,
        since it depends on thread interleaving.

        Rows are fetched inside the lock rather than handing back a live cursor:
        a lazily-consumed cursor would escape the critical section and
        reintroduce the same race at iteration time.
        """
        pool = self._read_pool
        in_transaction = bool(getattr(self._transaction_state, "active", False))
        if pool is not None and not in_transaction:
            try:
                connection = pool.get(timeout=0.05)
            except queue.Empty:
                # Do not let a read burst occupying every pooled connection
                # starve the writer/control path indefinitely.  The authority
                # connection is separately serialized and remains a safe
                # fallback under WAL.
                connection = None
            if connection is not None:
                try:
                    return connection.execute(sql, tuple(params)).fetchall()
                finally:
                    pool.put(connection)
        with self._lock:
            return self._connection.execute(sql, tuple(params)).fetchall()

    def _read_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self._read(sql, params)
        return rows[0] if rows else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("cannot write through a read-only SQLiteGraphStore")
        with self._lock:
            try:
                self._transaction_state.active = True
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            finally:
                self._transaction_state.active = False

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
                CREATE TABLE IF NOT EXISTS graph_checksum_state(
                    memory_id TEXT PRIMARY KEY,
                    node_xor TEXT NOT NULL, node_sum TEXT NOT NULL,
                    node_count INTEGER NOT NULL,
                    edge_xor TEXT NOT NULL, edge_sum TEXT NOT NULL,
                    edge_count INTEGER NOT NULL,
                    algorithm TEXT NOT NULL DEFAULT 'graphmem-multiset-sha256-v1'
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
                CREATE TABLE IF NOT EXISTS incremental_jobs(
                    job_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_offset INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    expected_version INTEGER NOT NULL,
                    graph_version INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(memory_id,source_offset)
                );
                CREATE INDEX IF NOT EXISTS incremental_jobs_state_idx
                    ON incremental_jobs(state,memory_id,source_offset);
                CREATE TABLE IF NOT EXISTS incremental_job_events(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    graph_version INTEGER,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES incremental_jobs(job_id) ON DELETE CASCADE
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

    @staticmethod
    def _incremental_job(row: sqlite3.Row) -> IncrementalJobRecord:
        return IncrementalJobRecord(
            job_id=str(row["job_id"]), memory_id=str(row["memory_id"]),
            session_id=str(row["session_id"]), source_offset=int(row["source_offset"]),
            state=str(row["state"]), payload_hash=str(row["payload_hash"]),
            payload=json.loads(row["payload_json"]),
            expected_version=int(row["expected_version"]),
            graph_version=(int(row["graph_version"])
                           if row["graph_version"] is not None else None),
            attempts=int(row["attempts"]), last_error=row["last_error"],
        )

    def append_incremental_raw(
        self,
        *,
        job_id: str,
        session: Session,
        turns: Sequence[SourceTurn],
        source_offset: int,
        payload: Mapping[str, Any] | None = None,
    ) -> IncrementalJobRecord:
        """Atomically persist raw rows and advance RECEIVED -> RAW_DURABLE.

        Replaying an identical job is a no-op.  Reusing either the job id or
        source offset for different content is rejected, so an upstream retry
        cannot silently overwrite already acknowledged raw memory.
        """
        if self.read_only:
            raise RuntimeError("cannot append through a read-only SQLiteGraphStore")
        if source_offset < 0:
            raise ValueError("source_offset cannot be negative")
        if not turns:
            raise ValueError("an incremental append must contain at least one turn")
        if any(row.memory_id != session.memory_id or row.session_id != session.session_id
               for row in turns):
            raise ValueError("session and incremental turns must share memory/session ids")
        payload_json = canonical_json(dict(payload or {}))
        digest = hashlib.sha256(canonical_json({
            "session": asdict(session),
            "turns": [asdict(row) for row in turns],
            "payload": json.loads(payload_json),
        }).encode("utf-8")).hexdigest()
        with self.transaction() as db:
            existing = db.execute(
                "SELECT * FROM incremental_jobs WHERE job_id=? OR "
                "(memory_id=? AND source_offset=?) ORDER BY job_id=? DESC LIMIT 1",
                (job_id, session.memory_id, source_offset, job_id),
            ).fetchone()
            if existing is not None:
                record = self._incremental_job(existing)
                if record.job_id != job_id or record.payload_hash != digest:
                    raise ValueError(
                        "incremental job id/source offset was reused with different content")
                return record
            if db.execute(
                "SELECT 1 FROM conversations WHERE memory_id=?", (session.memory_id,)
            ).fetchone() is None:
                raise ValueError(f"unknown memory {session.memory_id!r}")
            version_row = db.execute(
                "SELECT graph_version FROM graph_versions WHERE memory_id=?",
                (session.memory_id,),
            ).fetchone()
            expected_version = int(version_row[0]) if version_row else 0
            db.execute(
                "INSERT INTO incremental_jobs(job_id,memory_id,session_id,source_offset,"
                "state,payload_hash,payload_json,expected_version) VALUES(?,?,?,?,?,?,?,?)",
                (job_id, session.memory_id, session.session_id, source_offset,
                 "received", digest, payload_json, expected_version),
            )
            db.execute(
                "INSERT INTO incremental_job_events(job_id,from_state,to_state,detail_json) "
                "VALUES(?,NULL,'received',?)",
                (job_id, canonical_json({"source_offset": source_offset})),
            )
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?) ON CONFLICT(memory_id,session_id) "
                "DO UPDATE SET ordinal=excluded.ordinal,timestamp=excluded.timestamp,"
                "content_hash=excluded.content_hash,schema_version=excluded.schema_version",
                tuple(asdict(session).values()),
            )
            self._upsert_turns(db, turns)
            db.execute(
                "UPDATE incremental_jobs SET state='raw_durable',updated_at=CURRENT_TIMESTAMP "
                "WHERE job_id=?", (job_id,),
            )
            db.execute(
                "INSERT INTO incremental_job_events(job_id,from_state,to_state,detail_json) "
                "VALUES(?,'received','raw_durable',?)",
                (job_id, canonical_json({"turn_count": len(turns)})),
            )
            row = db.execute(
                "SELECT * FROM incremental_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            assert row is not None
            return self._incremental_job(row)

    def incremental_job(self, job_id: str) -> IncrementalJobRecord | None:
        row = self._read_one("SELECT * FROM incremental_jobs WHERE job_id=?", (job_id,))
        return self._incremental_job(row) if row else None

    def incremental_jobs(
        self, *, state: str | None = None, limit: int = 100,
    ) -> tuple[IncrementalJobRecord, ...]:
        if limit <= 0:
            return ()
        if state is None:
            rows = self._read(
                "SELECT * FROM incremental_jobs ORDER BY memory_id,source_offset LIMIT ?",
                (limit,),
            )
        else:
            rows = self._read(
                "SELECT * FROM incremental_jobs WHERE state=? "
                "ORDER BY memory_id,source_offset LIMIT ?", (state, limit),
            )
        return tuple(self._incremental_job(row) for row in rows)

    def mark_incremental_attempt(self, job_id: str, error: str | None = None) -> None:
        with self.transaction() as db:
            before = db.total_changes
            db.execute(
                "UPDATE incremental_jobs SET attempts=attempts+1,last_error=?,"
                "updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (error, job_id),
            )
            if db.total_changes == before:
                raise KeyError(job_id)

    def transition_incremental_job(
        self, job_id: str, *, expected_state: str, next_state: str,
        graph_version: int | None = None, detail: Mapping[str, Any] | None = None,
    ) -> IncrementalJobRecord:
        """CAS a metadata-only stage; graph stages should use apply_graph_delta."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM incremental_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = self._incremental_job(row)
            if current.state != expected_state:
                raise RuntimeError(
                    f"stale incremental transition for {job_id!r}: expected "
                    f"{expected_state!r}, current state is {current.state!r}")
            db.execute(
                "UPDATE incremental_jobs SET state=?,graph_version=COALESCE(?,graph_version),"
                "last_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                (next_state, graph_version, job_id),
            )
            db.execute(
                "INSERT INTO incremental_job_events(job_id,from_state,to_state,graph_version,"
                "detail_json) VALUES(?,?,?,?,?)",
                (job_id, expected_state, next_state, graph_version,
                 canonical_json(dict(detail or {}))),
            )
            updated = db.execute(
                "SELECT * FROM incremental_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            assert updated is not None
            return self._incremental_job(updated)

    def incremental_high_watermark(self, memory_id: str) -> int:
        row = self._read_one(
            "SELECT MAX(source_offset) FROM incremental_jobs WHERE memory_id=? "
            "AND state!='received'", (memory_id,),
        )
        return int(row[0]) if row and row[0] is not None else -1

    def _upsert_turns(self, db: sqlite3.Connection, turns: Sequence[SourceTurn]) -> None:
        columns = (
            "turn_id,memory_id,session_id,turn_index,speaker,listener,role,timestamp,"
            "raw_text,content_hash,schema_version"
        )
        # Probed BEFORE the insert below: afterwards every turn is present and the
        # FTS cleanup would degrade back to deleting all of them.
        present: set[str] = set()
        for memory_id in sorted({item.memory_id for item in turns}):
            present.update(row[0] for row in db.execute(
                "SELECT turn_id FROM source_turns WHERE memory_id=?", (memory_id,)))
        db.executemany(
            f"INSERT INTO source_turns({columns}) VALUES({','.join('?' for _ in range(11))}) "
            "ON CONFLICT(turn_id) DO UPDATE SET raw_text=excluded.raw_text,"
            "content_hash=excluded.content_hash,speaker=excluded.speaker,listener=excluded.listener,"
            "role=excluded.role,timestamp=excluded.timestamp,schema_version=excluded.schema_version",
            [tuple(asdict(item).values()) for item in turns],
        )
        try:
            # turn_id is not indexed on the FTS5 table, so each delete scans it.
            # Deleting unconditionally made ingest O(n^2): 0.15s per memory on an
            # empty database and 20s per memory once 55k turns were present --
            # about 2 hours of pure CPU before the first GPU call on a 510-memory
            # corpus.  Only rows that already exist need clearing, and memory_id
            # IS indexed, so one indexed read replaces N full scans.
            stale = [(x.turn_id,) for x in turns if x.turn_id in present]
            if stale:
                db.executemany("DELETE FROM source_turns_fts WHERE turn_id=?", stale)
            db.executemany(
                "INSERT INTO source_turns_fts(turn_id,memory_id,session_id,raw_text) VALUES(?,?,?,?)",
                [(x.turn_id, x.memory_id, x.session_id, x.raw_text) for x in turns],
            )
        except sqlite3.OperationalError:
            pass

    def conversation(self, memory_id: str) -> Conversation | None:
        row = self._read_one(
            "SELECT * FROM conversations WHERE memory_id=?", (memory_id,)
        )
        return Conversation(**dict(row)) if row else None

    def sessions(self, memory_id: str) -> Sequence[Session]:
        rows = self._read(
            "SELECT * FROM sessions WHERE memory_id=? ORDER BY ordinal,session_id", (memory_id,)
        )
        return [Session(**dict(row)) for row in rows]

    def turns(self, memory_id: str) -> Sequence[SourceTurn]:
        rows = self._read(
            "SELECT * FROM source_turns WHERE memory_id=? ORDER BY session_id,turn_index", (memory_id,)
        )
        return [SourceTurn(**dict(row)) for row in rows]

    def memory_ids(self) -> tuple[str, ...]:
        """Return memories with a published graph, in deterministic order."""
        rows = self._read(
            "SELECT memory_id FROM graph_versions ORDER BY memory_id", ())
        return tuple(str(row[0]) for row in rows)

    def turns_by_ids(self, turn_ids: Sequence[str]) -> Sequence[SourceTurn]:
        if not turn_ids:
            return []
        rows = self._read(
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
            rows = self._read(
                "SELECT turn_id,-bm25(source_turns_fts) AS score FROM source_turns_fts "
                "WHERE source_turns_fts MATCH ? AND memory_id=? ORDER BY bm25(source_turns_fts) LIMIT ?",
                (match, memory_id, limit),
            )
            return [(row["turn_id"], float(row["score"])) for row in rows]
        except sqlite3.OperationalError:
            pattern = "%" + "%".join(terms[:8]) + "%"
            rows = self._read(
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
            for table in ("graph_nodes", "graph_edges", "evidence_groups",
                          "graph_versions", "graph_checksum_state"):
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

    @staticmethod
    def _write_graph_checksum_state(
        db: sqlite3.Connection, memory_id: str, state: GraphChecksumState,
    ) -> None:
        db.execute(
            "INSERT INTO graph_checksum_state VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(memory_id) DO UPDATE SET "
            "node_xor=excluded.node_xor,node_sum=excluded.node_sum,"
            "node_count=excluded.node_count,edge_xor=excluded.edge_xor,"
            "edge_sum=excluded.edge_sum,edge_count=excluded.edge_count,"
            "algorithm=excluded.algorithm",
            (memory_id, f"{state.node_xor:064x}", f"{state.node_sum:064x}",
             state.node_count, f"{state.edge_xor:064x}",
             f"{state.edge_sum:064x}", state.edge_count,
             "graphmem-multiset-sha256-v1"),
        )

    @staticmethod
    def _read_graph_checksum_state(
        db: sqlite3.Connection, memory_id: str,
    ) -> GraphChecksumState | None:
        row = db.execute(
            "SELECT node_xor,node_sum,node_count,edge_xor,edge_sum,edge_count "
            "FROM graph_checksum_state WHERE memory_id=?", (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return GraphChecksumState(
            int(str(row[0]), 16), int(str(row[1]), 16), int(row[2]),
            int(str(row[3]), 16), int(str(row[4]), 16), int(row[5]),
        )

    def replace_graph(
        self,
        memory_id: str,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        evidence_groups: Sequence[EvidenceGroup],
    ) -> int:
        if any(item.memory_id != memory_id for item in (*nodes, *edges, *evidence_groups)):
            raise ValueError("graph rows must share memory_id")
        checksum_state = logical_graph_checksum_state(nodes, edges)
        checksum = checksum_state.checksum
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
            self._write_graph_checksum_state(db, memory_id, checksum_state)
            db.execute(
                "INSERT INTO outbox(memory_id,graph_version,event_type,payload_json) VALUES(?,?,?,?)",
                (memory_id, version, "replace_graph", canonical_json({"checksum": checksum})),
            )
        return version

    def apply_graph_delta(
        self,
        memory_id: str,
        *,
        upsert_nodes: Sequence[GraphNode] = (),
        upsert_edges: Sequence[GraphEdge] = (),
        upsert_evidence_groups: Sequence[EvidenceGroup] = (),
        delete_node_ids: Sequence[str] = (),
        delete_edge_ids: Sequence[str] = (),
        delete_evidence_group_ids: Sequence[str] = (),
        expected_version: int | None = None,
        event_type: str = "apply_graph_delta",
        incremental_job_id: str | None = None,
        expected_job_state: str | None = None,
        next_job_state: str | None = None,
    ) -> GraphDeltaResult:
        """Atomically publish an affected-path graph delta.

        Readers either observe the previous committed graph or the complete new
        graph; they never observe half of a branch update.  ``expected_version``
        is a compare-and-swap guard for concurrent writers.  Incident edges are
        removed automatically with deleted nodes and all final endpoints and
        primary evidence references are validated before publication.
        """
        if self.read_only:
            raise RuntimeError("cannot publish through a read-only SQLiteGraphStore")
        upsert_nodes = tuple({item.node_id: item for item in upsert_nodes}.values())
        upsert_edges = tuple({item.edge_id: item for item in upsert_edges}.values())
        upsert_evidence_groups = tuple({
            item.evidence_group_id: item for item in upsert_evidence_groups
        }.values())
        rows = (*upsert_nodes, *upsert_edges, *upsert_evidence_groups)
        if any(item.memory_id != memory_id for item in rows):
            raise ValueError("delta rows must share memory_id")
        node_ids = tuple(dict.fromkeys(str(item) for item in delete_node_ids))
        edge_ids = tuple(dict.fromkeys(str(item) for item in delete_edge_ids))
        group_ids = tuple(dict.fromkeys(str(item) for item in delete_evidence_group_ids))
        with self.transaction() as db:
            if incremental_job_id is not None:
                if not expected_job_state or not next_job_state:
                    raise ValueError(
                        "incremental graph publication needs expected and next job states")
                job_row = db.execute(
                    "SELECT memory_id,state FROM incremental_jobs WHERE job_id=?",
                    (incremental_job_id,),
                ).fetchone()
                if job_row is None:
                    raise KeyError(incremental_job_id)
                if str(job_row[0]) != memory_id:
                    raise ValueError("incremental job and graph delta must share memory_id")
                if str(job_row[1]) != expected_job_state:
                    raise RuntimeError(
                        f"stale incremental graph transition for {incremental_job_id!r}: "
                        f"expected {expected_job_state!r}, current state is {job_row[1]!r}")
            previous = db.execute(
                "SELECT graph_version FROM graph_versions WHERE memory_id=?", (memory_id,)
            ).fetchone()
            current_version = int(previous[0]) if previous else 0
            if expected_version is not None and current_version != expected_version:
                raise RuntimeError(
                    f"stale graph delta for {memory_id!r}: expected version "
                    f"{expected_version}, current version is {current_version}")

            checksum_state = self._read_graph_checksum_state(db, memory_id)
            if checksum_state is None:
                # One-time migration path for graphs published before schema v4.
                # Subsequent deltas only read the rows named by the mutation.
                all_nodes = [self._node(row) for row in db.execute(
                    "SELECT * FROM graph_nodes WHERE memory_id=?", (memory_id,)
                ).fetchall()]
                all_edges = [self._edge(row) for row in db.execute(
                    "SELECT * FROM graph_edges WHERE memory_id=?", (memory_id,)
                ).fetchall()]
                checksum_state = logical_graph_checksum_state(all_nodes, all_edges)
                affected_node_ids = {
                    *(item.node_id for item in upsert_nodes), *node_ids,
                }
                affected_edge_ids = {
                    *(item.edge_id for item in upsert_edges), *edge_ids,
                }
                old_nodes_by_id = {
                    row.node_id: row for row in all_nodes
                    if row.node_id in affected_node_ids
                }
                old_edges_by_id = {
                    row.edge_id: row for row in all_edges
                    if (row.edge_id in affected_edge_ids
                        or row.src_id in node_ids or row.dst_id in node_ids)
                }
            else:
                affected_node_ids = tuple(dict.fromkeys((
                    *(item.node_id for item in upsert_nodes), *node_ids,
                )))
                old_nodes_by_id: dict[str, GraphNode] = {}
                if affected_node_ids:
                    placeholders = ",".join("?" for _ in affected_node_ids)
                    old_nodes_by_id = {
                        node.node_id: node
                        for node in (self._node(row) for row in db.execute(
                            f"SELECT * FROM graph_nodes WHERE memory_id=? AND "
                            f"node_id IN ({placeholders})",
                            (memory_id, *affected_node_ids),
                        ).fetchall())
                    }
                affected_edge_ids = tuple(dict.fromkeys((
                    *(item.edge_id for item in upsert_edges), *edge_ids,
                )))
                old_edges_by_id: dict[str, GraphEdge] = {}
                if affected_edge_ids:
                    placeholders = ",".join("?" for _ in affected_edge_ids)
                    old_edges_by_id.update({
                        edge.edge_id: edge
                        for edge in (self._edge(row) for row in db.execute(
                            f"SELECT * FROM graph_edges WHERE memory_id=? AND "
                            f"edge_id IN ({placeholders})",
                            (memory_id, *affected_edge_ids),
                        ).fetchall())
                    })
                if node_ids:
                    placeholders = ",".join("?" for _ in node_ids)
                    old_edges_by_id.update({
                        edge.edge_id: edge
                        for edge in (self._edge(row) for row in db.execute(
                            f"SELECT * FROM graph_edges WHERE memory_id=? AND "
                            f"(src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
                            (memory_id, *node_ids, *node_ids),
                        ).fetchall())
                    })

            deleted_edges = 0
            if edge_ids:
                placeholders = ",".join("?" for _ in edge_ids)
                before = db.total_changes
                db.execute(
                    f"DELETE FROM graph_edges WHERE memory_id=? AND edge_id IN ({placeholders})",
                    (memory_id, *edge_ids),
                )
                deleted_edges += db.total_changes - before
            if node_ids:
                placeholders = ",".join("?" for _ in node_ids)
                before = db.total_changes
                db.execute(
                    f"DELETE FROM graph_edges WHERE memory_id=? AND "
                    f"(src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
                    (memory_id, *node_ids, *node_ids),
                )
                deleted_edges += db.total_changes - before
                before = db.total_changes
                db.execute(
                    f"DELETE FROM graph_nodes WHERE memory_id=? AND node_id IN ({placeholders})",
                    (memory_id, *node_ids),
                )
                deleted_nodes = db.total_changes - before
            else:
                deleted_nodes = 0

            deleted_groups = 0
            if group_ids:
                placeholders = ",".join("?" for _ in group_ids)
                before = db.total_changes
                db.execute(
                    f"DELETE FROM evidence_groups WHERE memory_id=? AND "
                    f"evidence_group_id IN ({placeholders})",
                    (memory_id, *group_ids),
                )
                deleted_groups = db.total_changes - before

            for group in upsert_evidence_groups:
                db.execute(
                    "INSERT INTO evidence_groups VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(evidence_group_id) DO UPDATE SET "
                    "memory_id=excluded.memory_id,content_hash=excluded.content_hash,"
                    "min_time=excluded.min_time,max_time=excluded.max_time,"
                    "schema_version=excluded.schema_version",
                    (group.evidence_group_id, memory_id, group.content_hash,
                     group.min_time, group.max_time, group.schema_version),
                )
                db.execute(
                    "DELETE FROM evidence_members WHERE evidence_group_id=?",
                    (group.evidence_group_id,),
                )
                db.executemany(
                    "INSERT INTO evidence_members VALUES(?,?,?,?,?,?)",
                    [(group.evidence_group_id, index, member.turn_id,
                      member.span_start, member.span_end, member.support_type)
                     for index, member in enumerate(group.members)],
                )

            db.executemany(
                "INSERT INTO graph_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET "
                "memory_id=excluded.memory_id,node_type=excluded.node_type,"
                "level=excluded.level,summary=excluded.summary,"
                "evidence_group_id=excluded.evidence_group_id,"
                "evidence_group_ids_json=excluded.evidence_group_ids_json,"
                "entity_id=excluded.entity_id,event_time=excluded.event_time,"
                "state=excluded.state,confidence=excluded.confidence,"
                "attributes_json=excluded.attributes_json,"
                "schema_version=excluded.schema_version",
                [(x.node_id, x.memory_id, str(x.node_type), x.level, x.summary,
                  x.evidence_group_id, canonical_json(x.evidence_group_ids), x.entity_id,
                  x.event_time, x.state, x.confidence, canonical_json(x.attributes),
                  x.schema_version) for x in upsert_nodes],
            )
            db.executemany(
                "INSERT INTO graph_edges VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(edge_id) DO UPDATE SET "
                "memory_id=excluded.memory_id,src_id=excluded.src_id,"
                "relation=excluded.relation,dst_id=excluded.dst_id,"
                "evidence_group_id=excluded.evidence_group_id,"
                "evidence_group_ids_json=excluded.evidence_group_ids_json,"
                "directed=excluded.directed,confidence=excluded.confidence,"
                "source=excluded.source,schema_version=excluded.schema_version",
                [(x.edge_id, x.memory_id, x.src_id, str(x.relation), x.dst_id,
                  x.evidence_group_id, canonical_json(x.evidence_group_ids), int(x.directed),
                  x.confidence, x.source, x.schema_version) for x in upsert_edges],
            )

            # The committed graph was valid before this transaction.  It is
            # sufficient to validate new endpoints and evidence references plus
            # any evidence group explicitly removed; a full anti-join over the
            # unchanged graph would turn a four-row delta back into O(|V|+|E|).
            endpoint_ids = tuple(dict.fromkeys(
                item for edge in upsert_edges for item in (edge.src_id, edge.dst_id)
            ))
            existing_endpoints: set[str] = set()
            if endpoint_ids:
                placeholders = ",".join("?" for _ in endpoint_ids)
                existing_endpoints = {str(row[0]) for row in db.execute(
                    f"SELECT node_id FROM graph_nodes WHERE memory_id=? AND "
                    f"node_id IN ({placeholders})", (memory_id, *endpoint_ids),
                ).fetchall()}
            dangling = next((
                edge.edge_id for edge in upsert_edges
                if edge.src_id not in existing_endpoints
                or edge.dst_id not in existing_endpoints
            ), None)
            if dangling:
                raise ValueError(f"delta leaves dangling edge {dangling!r}")

            referenced_groups = tuple(dict.fromkeys(
                item.evidence_group_id for item in (*upsert_nodes, *upsert_edges)
            ))
            existing_groups: set[str] = set()
            if referenced_groups:
                placeholders = ",".join("?" for _ in referenced_groups)
                existing_groups = {str(row[0]) for row in db.execute(
                    f"SELECT evidence_group_id FROM evidence_groups WHERE memory_id=? "
                    f"AND evidence_group_id IN ({placeholders})",
                    (memory_id, *referenced_groups),
                ).fetchall()}
            missing_evidence = next((
                item.node_id for item in upsert_nodes
                if item.evidence_group_id not in existing_groups
            ), None) or next((
                item.edge_id for item in upsert_edges
                if item.evidence_group_id not in existing_groups
            ), None)
            if missing_evidence:
                raise ValueError(
                    f"delta leaves missing primary evidence for {missing_evidence!r}")
            if group_ids:
                placeholders = ",".join("?" for _ in group_ids)
                still_referenced = db.execute(
                    "SELECT row_id FROM ("
                    f"SELECT node_id AS row_id,evidence_group_id FROM graph_nodes "
                    f"WHERE memory_id=? AND evidence_group_id IN ({placeholders}) "
                    "UNION ALL "
                    f"SELECT edge_id,evidence_group_id FROM graph_edges "
                    f"WHERE memory_id=? AND evidence_group_id IN ({placeholders})"
                    ") LIMIT 1",
                    (memory_id, *group_ids, memory_id, *group_ids),
                ).fetchone()
                if still_referenced:
                    raise ValueError(
                        f"delta leaves missing primary evidence for {still_referenced[0]!r}")

            modulus = 1 << 256
            node_xor, node_sum, node_count = (
                checksum_state.node_xor, checksum_state.node_sum,
                checksum_state.node_count,
            )
            edge_xor, edge_sum, edge_count = (
                checksum_state.edge_xor, checksum_state.edge_sum,
                checksum_state.edge_count,
            )
            for node in old_nodes_by_id.values():
                digest = logical_graph_row_digest("node", node)
                node_xor ^= digest
                node_sum = (node_sum - digest) % modulus
                node_count -= 1
            for node in upsert_nodes:
                digest = logical_graph_row_digest("node", node)
                node_xor ^= digest
                node_sum = (node_sum + digest) % modulus
                node_count += 1
            for edge in old_edges_by_id.values():
                digest = logical_graph_row_digest("edge", edge)
                edge_xor ^= digest
                edge_sum = (edge_sum - digest) % modulus
                edge_count -= 1
            for edge in upsert_edges:
                digest = logical_graph_row_digest("edge", edge)
                edge_xor ^= digest
                edge_sum = (edge_sum + digest) % modulus
                edge_count += 1
            checksum_state = GraphChecksumState(
                node_xor, node_sum, node_count,
                edge_xor, edge_sum, edge_count,
            )
            checksum = checksum_state.checksum
            version = current_version + 1
            db.execute(
                "INSERT INTO graph_versions(memory_id,graph_version,graph_checksum) VALUES(?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET graph_version=excluded.graph_version,"
                "graph_checksum=excluded.graph_checksum,updated_at=CURRENT_TIMESTAMP",
                (memory_id, version, checksum),
            )
            self._write_graph_checksum_state(db, memory_id, checksum_state)
            payload = {
                "checksum": checksum,
                "upserted_nodes": len(upsert_nodes),
                "deleted_nodes": deleted_nodes,
                "upserted_edges": len(upsert_edges),
                "deleted_edges": deleted_edges,
                "upserted_evidence_groups": len(upsert_evidence_groups),
                "deleted_evidence_groups": deleted_groups,
            }
            db.execute(
                "INSERT INTO outbox(memory_id,graph_version,event_type,payload_json) VALUES(?,?,?,?)",
                (memory_id, version, event_type, canonical_json(payload)),
            )
            if incremental_job_id is not None:
                db.execute(
                    "UPDATE incremental_jobs SET state=?,graph_version=?,expected_version=?,last_error=NULL,"
                    "updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                    (next_job_state, version, version, incremental_job_id),
                )
                db.execute(
                    "INSERT INTO incremental_job_events(job_id,from_state,to_state,"
                    "graph_version,detail_json) VALUES(?,?,?,?,?)",
                    (incremental_job_id, expected_job_state, next_job_state, version,
                     canonical_json({"event_type": event_type, **payload})),
                )
        return GraphDeltaResult(
            graph_version=version, graph_checksum=checksum,
            upserted_nodes=len(upsert_nodes), deleted_nodes=deleted_nodes,
            upserted_edges=len(upsert_edges), deleted_edges=deleted_edges,
            upserted_evidence_groups=len(upsert_evidence_groups),
            deleted_evidence_groups=deleted_groups,
        )

    def nodes(self, memory_id: str) -> Sequence[GraphNode]:
        return [self._node(row) for row in self._read(
            "SELECT * FROM graph_nodes WHERE memory_id=? ORDER BY node_id", (memory_id,)
        )]

    def edges(self, memory_id: str) -> Sequence[GraphEdge]:
        return [self._edge(row) for row in self._read(
            "SELECT * FROM graph_edges WHERE memory_id=? ORDER BY edge_id", (memory_id,)
        )]

    def evidence_groups(self, memory_id: str) -> Sequence[EvidenceGroup]:
        group_rows = self._read(
            "SELECT * FROM evidence_groups WHERE memory_id=? ORDER BY evidence_group_id",
            (memory_id,))
        # Fetch members in one joined scan.  The former implementation issued
        # one SELECT per group, turning first access to a memory into hundreds of
        # SQLite round trips and dominating cold-user latency.
        members_by_group: dict[str, list[EvidenceMember]] = {}
        for item in self._read(
                """SELECT m.evidence_group_id, m.member_ordinal, m.turn_id,
                          m.span_start, m.span_end, m.support_type
                   FROM evidence_members AS m
                   JOIN evidence_groups AS g
                     ON g.evidence_group_id=m.evidence_group_id
                   WHERE g.memory_id=?
                   ORDER BY m.evidence_group_id, m.member_ordinal""",
                (memory_id,)):
            members_by_group.setdefault(
                str(item["evidence_group_id"]), []).append(EvidenceMember(
                    item["turn_id"], item["span_start"], item["span_end"],
                    item["support_type"]))
        result: list[EvidenceGroup] = []
        for row in group_rows:
            members = tuple(members_by_group.get(
                str(row["evidence_group_id"]), ()))
            result.append(EvidenceGroup(
                row["evidence_group_id"], row["memory_id"], members, row["content_hash"],
                row["min_time"], row["max_time"], row["schema_version"],
            ))
        return result

    def evidence_group(self, evidence_group_id: str) -> EvidenceGroup | None:
        row = self._read_one(
            "SELECT * FROM evidence_groups WHERE evidence_group_id=?", (evidence_group_id,)
        )
        if not row:
            return None
        members = tuple(EvidenceMember(
            item["turn_id"], item["span_start"], item["span_end"], item["support_type"]
        ) for item in self._read(
            "SELECT * FROM evidence_members WHERE evidence_group_id=? ORDER BY member_ordinal",
            (evidence_group_id,),
        ))
        return EvidenceGroup(
            row["evidence_group_id"], row["memory_id"], members, row["content_hash"],
            row["min_time"], row["max_time"], row["schema_version"],
        )

    def graph_version(self, memory_id: str) -> int:
        row = self._read_one(
            "SELECT graph_version FROM graph_versions WHERE memory_id=?", (memory_id,)
        )
        return int(row[0]) if row else 0

    def graph_checksum(self, memory_id: str) -> str:
        row = self._read_one(
            "SELECT graph_checksum FROM graph_versions WHERE memory_id=?", (memory_id,)
        )
        return str(row[0]) if row else ""

    def graph_identity(self, memory_id: str) -> tuple[int, str]:
        """Read the version/checksum pair used to validate compiled sidecars."""
        row = self._read_one(
            "SELECT graph_version,graph_checksum FROM graph_versions WHERE memory_id=?",
            (memory_id,),
        )
        return ((int(row[0]), str(row[1])) if row else (0, ""))

    def graph_snapshot(self, memory_id: str) -> tuple[
            int, str, Sequence[GraphNode], Sequence[GraphEdge]]:
        """Read version, checksum, nodes and edges in one snapshot transaction."""
        def materialize(connection: sqlite3.Connection):
            version_row = connection.execute(
                "SELECT graph_version,graph_checksum FROM graph_versions WHERE memory_id=?",
                (memory_id,),
            ).fetchone()
            node_rows = connection.execute(
                "SELECT * FROM graph_nodes WHERE memory_id=? ORDER BY node_id",
                (memory_id,),
            ).fetchall()
            edge_rows = connection.execute(
                "SELECT * FROM graph_edges WHERE memory_id=? ORDER BY edge_id",
                (memory_id,),
            ).fetchall()
            return (
                int(version_row[0]) if version_row else 0,
                str(version_row[1]) if version_row else "",
                [self._node(row) for row in node_rows],
                [self._edge(row) for row in edge_rows],
            )

        pool = self._read_pool
        in_transaction = bool(getattr(self._transaction_state, "active", False))
        if pool is not None and not in_transaction:
            try:
                connection = pool.get(timeout=0.05)
            except queue.Empty:
                connection = None
            if connection is not None:
                try:
                    connection.execute("BEGIN")
                    try:
                        result = materialize(connection)
                        connection.commit()
                        return result
                    except BaseException:
                        connection.rollback()
                        raise
                finally:
                    pool.put(connection)
        # The authority connection is serialized with the writer.  Holding the
        # same lock across all reads is an equivalent atomic snapshot and also
        # works for :memory: stores, which cannot open a second connection.
        with self._lock:
            return materialize(self._connection)

    def cache_get(self, cache_key: str) -> Mapping[str, Any] | None:
        row = self._read_one(
            "SELECT response_json,usage_json FROM llm_cache WHERE cache_key=?", (cache_key,)
        )
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
        return {row["item_id"]: row["content_hash"] for row in self._read(
            "SELECT item_id,content_hash FROM embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, model_id),
        )}

    def search_embeddings(self, memory_id: str, model_id: str,
                          query_vector: Sequence[float], *, limit: int = 96) -> list[tuple[str, float]]:
        query = array("f", (float(value) for value in query_vector))
        query_norm = math.sqrt(sum(value * value for value in query)) or 1.0
        scores: list[tuple[str, float]] = []
        for row in self._read(
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
