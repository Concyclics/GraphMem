from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np


class QueryEmbeddingCache:
    """Small process-safe persistent cache for normalized query vectors.

    Turn embeddings remain part of the authoritative graph SQLite database.
    Query vectors are derived, reusable data and intentionally live in a
    separate WAL database so read-only graph replicas can still share them.
    Cache keys include the model and query-instruction revision upstream.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS query_embeddings ("
                    "cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, "
                    "vector BLOB NOT NULL, created_at REAL NOT NULL, "
                    "last_accessed REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS cache_metadata ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO cache_metadata(key,value) VALUES(?,?)",
                    ("schema_version", str(self.SCHEMA_VERSION)),
                )
            self._initialized = True

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        ordered = tuple(dict.fromkeys(str(key) for key in keys if key))
        if not ordered:
            return {}
        found: dict[str, np.ndarray] = {}
        # Keep comfortably below SQLite's default variable limit.
        with self._connect() as connection:
            for start in range(0, len(ordered), 400):
                batch = ordered[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT cache_key,dimension,vector FROM query_embeddings "
                    f"WHERE cache_key IN ({placeholders})", batch,
                ).fetchall()
                for cache_key, dimension, blob in rows:
                    vector = np.frombuffer(blob, dtype=np.float32, count=int(dimension))
                    if vector.size == int(dimension):
                        found[str(cache_key)] = vector.copy()
            if found:
                now = time.time()
                connection.executemany(
                    "UPDATE query_embeddings SET last_accessed=?,hits=hits+1 "
                    "WHERE cache_key=?",
                    ((now, key) for key in found),
                )
        return found

    def put_many(self, vectors: Mapping[str, Sequence[float] | np.ndarray]) -> None:
        if not vectors:
            return
        now = time.time()
        rows = []
        for cache_key, value in vectors.items():
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not vector.size or not np.isfinite(vector).all() or norm <= 1e-12:
                raise ValueError(f"invalid query embedding for cache key {cache_key}")
            normalized = np.ascontiguousarray(vector / norm, dtype=np.float32)
            rows.append((str(cache_key), int(normalized.size), normalized.tobytes(), now, now))
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO query_embeddings(cache_key,dimension,vector,created_at,last_accessed) "
                "VALUES(?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET "
                "dimension=excluded.dimension,vector=excluded.vector,"
                "last_accessed=excluded.last_accessed",
                rows,
            )

    def prune(self, max_entries: int) -> int:
        if max_entries <= 0:
            return 0
        with self._connect() as connection:
            count = int(connection.execute(
                "SELECT COUNT(*) FROM query_embeddings").fetchone()[0])
            remove = max(0, count - max_entries)
            if remove:
                connection.execute(
                    "DELETE FROM query_embeddings WHERE cache_key IN ("
                    "SELECT cache_key FROM query_embeddings "
                    "ORDER BY last_accessed ASC,cache_key ASC LIMIT ?)",
                    (remove,),
                )
            return remove
