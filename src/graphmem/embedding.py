from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np

from .config import GraphMemV5Config
from .domain import SourceTurn, stable_id
from .query_embedding_cache import QueryEmbeddingCache
from .retrieval.dense_sidecar import DenseIndexCache, DenseIndexSidecar
from .storage import SQLiteGraphStore


QUERY_INSTRUCTION_REVISION = "graphmem-turn-evidence-v1"
QUERY_INSTRUCTION = (
    "Instruct: Retrieve conversation turns that provide all evidence needed to answer "
    "the memory question.\nQuery: "
)


@dataclass(slots=True)
class _EmbeddingFlight:
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class QwenEmbeddingIndex:
    """Turn-level Qwen embedding index; usage is logged outside backbone tokens."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 client: Any | None = None, batch_size: int = 64, record_usage: bool = True,
                 query_cache_path: str | Path | None = None,
                 query_cache_entries: int = 8_192,
                 memory_cache_memories: int = 16,
                 dense_sidecar_dir: str | Path | None = None,
                 dense_backend: str = "auto",
                 dense_cache_bytes: int = 256 * 1024 * 1024,
                 dense_cache_memories: int = 32,
                 model_id: str | None = None,
                 base_url: str | None = None,
                 request_model_id: str | None = None) -> None:
        self.store = store
        self.config = config
        self.model_id = model_id or config.models.embedding_model
        # Keep the stable storage/cache identity separate from an endpoint's
        # served-model alias.  vLLM deployments commonly expose the same
        # checkpoint with or without its Hugging Face namespace.
        self.request_model_id = request_model_id or self.model_id
        self.batch_size = batch_size
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_entries = max(1, int(query_cache_entries))
        self._persistent_query_cache = (
            QueryEmbeddingCache(query_cache_path) if query_cache_path else None)
        self._query_lock = threading.RLock()
        self._query_flights: dict[str, _EmbeddingFlight] = {}
        self._memory_cache: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
        self._item_memory_cache: OrderedDict[
            tuple[int, str, str, tuple[int, str, str]],
            tuple[list[str], np.ndarray],
        ] = OrderedDict()
        self._memory_cache_memories = max(1, int(memory_cache_memories))
        self._memory_lock = threading.RLock()
        self._dense_cache = (
            DenseIndexCache(
                DenseIndexSidecar(dense_sidecar_dir, backend=dense_backend),
                max_bytes=dense_cache_bytes,
                max_memories=dense_cache_memories,
            ) if dense_sidecar_dir else None
        )
        self._stats = {
            "query_memory_hits": 0,
            "query_persistent_hits": 0,
            "query_misses": 0,
            "query_batches": 0,
            "query_singleflight_waits": 0,
            "query_embedded_views": 0,
            "query_api_ms": 0.0,
            "query_api_tokens": 0,
            "query_cache_errors": 0,
        }
        self.record_usage = record_usage
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                base_url=base_url or config.models.embedding_base_url,
                api_key="local",
                max_retries=0,
            )
        self.client = client

    @property
    def stats(self) -> Mapping[str, Any]:
        with self._query_lock:
            values: dict[str, Any] = {
                **self._stats,
                "query_memory_entries": len(self._query_cache),
            }
        with self._memory_lock:
            values["memory_matrix_entries"] = len(self._memory_cache)
            values["item_matrix_entries"] = len(self._item_memory_cache)
        if self._dense_cache is not None:
            values["dense_sidecar"] = self._dense_cache.stats()
        return values

    def _put_query_memory(self, cache_key: str, vector: Sequence[float] | np.ndarray) -> None:
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized))
        if not normalized.size or not np.isfinite(normalized).all() or norm <= 1e-12:
            raise ValueError("embedding service returned an invalid query vector")
        normalized = np.ascontiguousarray(normalized / norm, dtype=np.float32)
        self._query_cache[cache_key] = normalized
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > self._query_cache_entries:
            self._query_cache.popitem(last=False)

    def _get_query_memory(self, cache_key: str) -> np.ndarray | None:
        vector = self._query_cache.get(cache_key)
        if vector is not None:
            self._query_cache.move_to_end(cache_key)
        return vector

    def index_memory(self, memory_id: str) -> int:
        turns = list(self.store.turns(memory_id))
        existing = self.store.embedding_hashes(memory_id, self.model_id)
        pending = [turn for turn in turns if existing.get(turn.turn_id) != turn.content_hash]
        indexed = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            vectors, tokens, latency = self._embed([turn.raw_text for turn in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding response cardinality mismatch")
            self.store.upsert_embeddings(memory_id, self.model_id, [
                (turn.turn_id, turn.content_hash, vector) for turn, vector in zip(batch, vectors)
            ])
            self.store.log_embedding_call(
                stable_id("embedding-call", memory_id, start, *(turn.content_hash for turn in batch)),
                memory_id, self.model_id, len(batch), tokens, latency,
            )
            indexed += len(batch)
        if indexed:
            with self._memory_lock:
                self._memory_cache.pop(memory_id, None)
        return indexed

    def import_memory(self, memory_id: str, source: SQLiteGraphStore) -> int:
        expected = {turn.turn_id: turn.content_hash for turn in self.store.turns(memory_id)}
        rows = []
        for row in source._read(
            "SELECT item_id,content_hash,dimension,vector FROM embeddings "
            "WHERE memory_id=? AND model_id=?", (memory_id, self.model_id),
        ):
            if expected.get(row["item_id"]) != row["content_hash"]:
                continue
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=int(row["dimension"]))
            rows.append((row["item_id"], row["content_hash"], vector))
        if rows:
            self.store.upsert_embeddings(memory_id, self.model_id, rows)
            with self._memory_lock:
                self._memory_cache.pop(memory_id, None)
        return len(rows)

    def embed_graph_nodes(self, memory_id: str, nodes: Sequence[Any]) -> Mapping[
            str, Sequence[float]]:
        """Embed routing-card summaries and persist them in the shared cache.

        ``GraphBuildPipeline`` accepts this method directly as its coarsening
        vector provider.  Item hashes prevent a stale card vector from surviving
        a summary or graph-compiler change.
        """

        items = [(str(node.node_id), str(node.summary)) for node in nodes]
        hashes = {item_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
                  for item_id, text in items}
        existing = self.store.embedding_hashes(memory_id, self.model_id)
        pending = [(item_id, text) for item_id, text in items
                   if existing.get(item_id) != hashes[item_id]]
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            vectors, tokens, latency = self._embed([text for _item_id, text in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding response cardinality mismatch")
            self.store.upsert_embeddings(memory_id, self.model_id, [
                (item_id, hashes[item_id], vector)
                for (item_id, _text), vector in zip(batch, vectors)
            ])
            self.store.log_embedding_call(
                stable_id("embedding-call", memory_id, "routing-card", start,
                          *(hashes[item_id] for item_id, _text in batch)),
                memory_id, self.model_id, len(batch), tokens, latency)
        result: dict[str, Sequence[float]] = {}
        expected = set(hashes)
        for row in self.store._read(
                "SELECT item_id,dimension,vector FROM embeddings "
                "WHERE memory_id=? AND model_id=?",
                (memory_id, self.model_id)):
            item_id = str(row["item_id"])
            if item_id in expected:
                result[item_id] = np.frombuffer(
                    row["vector"], dtype=np.float32,
                    count=int(row["dimension"])).copy()
        return result

    def _query_vectors(self, memory_id: str, queries: Sequence[str]) -> list[np.ndarray]:
        query_texts = [QUERY_INSTRUCTION + query for query in queries]
        keys = [hashlib.sha256((
            self.model_id + "\n" + QUERY_INSTRUCTION_REVISION + "\n" + text
        ).encode()).hexdigest() for text in query_texts]

        unresolved: list[str] = []
        with self._query_lock:
            for cache_key in dict.fromkeys(keys):
                if self._get_query_memory(cache_key) is None:
                    unresolved.append(cache_key)
                else:
                    self._stats["query_memory_hits"] += 1

        if unresolved and self._persistent_query_cache is not None:
            try:
                persisted = self._persistent_query_cache.get_many(unresolved)
            except (OSError, sqlite3.Error, ValueError):
                persisted = {}
                with self._query_lock:
                    self._stats["query_cache_errors"] += 1
            with self._query_lock:
                for cache_key, vector in persisted.items():
                    self._put_query_memory(cache_key, vector)
                    self._stats["query_persistent_hits"] += 1
            unresolved = [key for key in unresolved if key not in persisted]

        owned: list[tuple[str, str, _EmbeddingFlight]] = []
        waiting: list[_EmbeddingFlight] = []
        text_by_key = dict(zip(keys, query_texts))
        with self._query_lock:
            for cache_key in unresolved:
                if self._get_query_memory(cache_key) is not None:
                    continue
                flight = self._query_flights.get(cache_key)
                if flight is None:
                    flight = _EmbeddingFlight()
                    self._query_flights[cache_key] = flight
                    owned.append((cache_key, text_by_key[cache_key], flight))
                    self._stats["query_misses"] += 1
                else:
                    waiting.append(flight)
                    self._stats["query_singleflight_waits"] += 1

        if owned:
            owned_keys = [row[0] for row in owned]
            try:
                vectors, tokens, latency = self._embed([row[1] for row in owned])
                if len(vectors) != len(owned):
                    raise RuntimeError("embedding response cardinality mismatch")
                persisted = dict(zip(owned_keys, vectors))
                if self._persistent_query_cache is not None:
                    try:
                        self._persistent_query_cache.put_many(persisted)
                    except (OSError, sqlite3.Error, ValueError):
                        # This sidecar is a disposable optimization.  A full or
                        # temporarily locked cache must not fail retrieval after
                        # the authoritative embedding service has succeeded.
                        with self._query_lock:
                            self._stats["query_cache_errors"] += 1
                with self._query_lock:
                    for cache_key, vector in persisted.items():
                        self._put_query_memory(cache_key, vector)
                    self._stats["query_batches"] += 1
                    self._stats["query_embedded_views"] += len(owned)
                    self._stats["query_api_ms"] += latency
                    self._stats["query_api_tokens"] += tokens
                if self.record_usage:
                    self.store.log_embedding_call(
                        stable_id("embedding-call", memory_id, "query-batch", *owned_keys),
                        memory_id, self.model_id, len(owned), tokens, latency,
                    )
            except BaseException as error:
                for _cache_key, _text, flight in owned:
                    flight.error = error
                raise
            finally:
                with self._query_lock:
                    for cache_key, _text, flight in owned:
                        self._query_flights.pop(cache_key, None)
                        flight.event.set()

        for flight in waiting:
            flight.event.wait()
            if flight.error is not None:
                raise RuntimeError("concurrent query embedding failed") from flight.error

        with self._query_lock:
            result = []
            for cache_key in keys:
                vector = self._get_query_memory(cache_key)
                if vector is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("query embedding cache fill failed")
                result.append(vector)
            return result

    def _memory_matrix(self, memory_id: str) -> tuple[list[str], np.ndarray]:
        with self._memory_lock:
            cached = self._memory_cache.get(memory_id)
            if cached is not None:
                self._memory_cache.move_to_end(memory_id)
                return cached
            expected = {turn.turn_id for turn in self.store.turns(memory_id)}
            ids, vectors = [], []
            if expected:
                for row in self.store._read(
                    "SELECT item_id,dimension,vector FROM embeddings WHERE memory_id=? AND model_id=? ORDER BY item_id",
                    (memory_id, self.model_id),
                ):
                    item_id = str(row["item_id"])
                    if item_id not in expected:
                        continue
                    ids.append(item_id)
                    vectors.append(np.frombuffer(
                        row["vector"], dtype=np.float32,
                        count=int(row["dimension"])))
            matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
            if matrix.size:
                matrix /= np.maximum(
                    np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            self._memory_cache[memory_id] = (ids, matrix)
            self._memory_cache.move_to_end(memory_id)
            while len(self._memory_cache) > self._memory_cache_memories:
                self._memory_cache.popitem(last=False)
            return ids, matrix

    def _search_vector(self, memory_id: str, query_vector: np.ndarray,
                       limit: int) -> Sequence[tuple[str, float]]:
        return self._search_vectors(memory_id, ((query_vector, limit),))[0]

    def _search_vectors(
        self,
        memory_id: str,
        requests: Sequence[tuple[np.ndarray, int]],
    ) -> tuple[Sequence[tuple[str, float]], ...]:
        if not requests:
            return ()
        if self._dense_cache is not None:
            handle = self._dense_cache.get(self.store, memory_id, self.model_id)
            if handle is not None:
                return tuple(handle.search(vector, limit) for vector, limit in requests)
        ids, matrix = self._memory_matrix(memory_id)
        if not ids:
            return tuple(() for _request in requests)
        results: list[Sequence[tuple[str, float]]] = []
        for query_vector, limit in requests:
            scores = matrix @ query_vector
            count = min(limit, len(ids))
            if count <= 0:
                results.append(())
                continue
            indices = np.argpartition(-scores, count - 1)[:count]
            indices = sorted(
                indices, key=lambda index: (-float(scores[index]), ids[index]))
            results.append([(ids[index], float(scores[index])) for index in indices])
        return tuple(results)

    def search_many(self, memory_id: str,
                    requests: Sequence[tuple[str, int]]) -> tuple[
                        Sequence[tuple[str, float]], ...]:
        if not requests:
            return ()
        vectors = self._query_vectors(memory_id, [query for query, _limit in requests])
        return self._search_vectors(
            memory_id,
            tuple((vector, int(limit))
                  for vector, (_query, limit) in zip(vectors, requests)),
        )

    def search(self, memory_id: str, query: str, limit: int) -> Sequence[tuple[str, float]]:
        return self.search_many(memory_id, ((query, limit),))[0]

    def search_items(
        self,
        memory_id: str,
        query: str,
        item_ids: Sequence[str],
        limit: int,
        *,
        source_store: SQLiteGraphStore | None = None,
    ) -> Sequence[tuple[str, float]]:
        """Search an immutable subset such as CanonicalFacts by dense vector.

        Turn FAISS remains the normal hot path.  Exact lookup additionally uses
        graph-node embeddings already produced during coarsening; this cached
        NumPy projection avoids an O(number-of-nodes) Python cosine loop on every
        request and does not create any new embedding/model cost.
        """
        if limit <= 0 or not item_ids:
            return ()
        authority = source_store or self.store
        expected = tuple(sorted(set(str(item) for item in item_ids)))
        signature = (
            len(expected), expected[0] if expected else "",
            expected[-1] if expected else "",
        )
        key = (id(authority), memory_id, self.model_id, signature)
        with self._memory_lock:
            cached = self._item_memory_cache.get(key)
            if cached is None:
                allowed = frozenset(expected)
                ids: list[str] = []
                vectors: list[np.ndarray] = []
                for row in authority._read(
                    "SELECT item_id,dimension,vector FROM embeddings "
                    "WHERE memory_id=? AND model_id=? ORDER BY item_id",
                    (memory_id, self.model_id),
                ):
                    item_id = str(row["item_id"])
                    if item_id not in allowed:
                        continue
                    ids.append(item_id)
                    vectors.append(np.frombuffer(
                        row["vector"], dtype=np.float32,
                        count=int(row["dimension"])))
                matrix = (np.stack(vectors) if vectors
                          else np.empty((0, 0), dtype=np.float32))
                if matrix.size:
                    matrix /= np.maximum(
                        np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
                cached = (ids, matrix)
                self._item_memory_cache[key] = cached
                self._item_memory_cache.move_to_end(key)
                while len(self._item_memory_cache) > self._memory_cache_memories:
                    self._item_memory_cache.popitem(last=False)
            else:
                self._item_memory_cache.move_to_end(key)
        ids, matrix = cached
        if not ids:
            return ()
        vector = self._query_vectors(memory_id, (query,))[0]
        scores = matrix @ vector
        count = min(limit, len(ids))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = sorted(
            indices, key=lambda index: (-float(scores[index]), ids[index]))
        return tuple((ids[index], float(scores[index])) for index in indices)

    def _embed(self, texts: Sequence[str]) -> tuple[list[list[float]], int, float]:
        started = time.perf_counter()
        # The host GPU watchdog can terminate a low-duty-cycle embedding engine
        # even while the supervised service is healthy enough to restart.  Keep
        # a request alive across that bounded restart window, but never retry
        # schema/authentication/client errors which cannot recover by waiting.
        response = None
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                response = self.client.embeddings.create(
                    model=getattr(self, "request_model_id", self.model_id),
                    input=list(texts))
                break
            except Exception as error:
                status_code = getattr(error, "status_code", None)
                message = str(error).casefold()
                context_overflow = (
                    isinstance(status_code, int) and status_code == 400
                    and ("maximum context length" in message
                         or "context length" in message))
                if context_overflow:
                    if len(texts) == 1 and len(texts[0]) > 1:
                        midpoint = len(texts[0]) // 2
                        left, left_tokens, left_latency = self._embed(
                            (texts[0][:midpoint],))
                        right, right_tokens, right_latency = self._embed(
                            (texts[0][midpoint:],))
                        left_vector = np.asarray(left[0], dtype=np.float32)
                        right_vector = np.asarray(right[0], dtype=np.float32)
                        combined = left_vector + right_vector
                        norm = float(np.linalg.norm(combined))
                        if norm > 1e-12:
                            combined /= norm
                        return ([combined.tolist()], left_tokens + right_tokens,
                                left_latency + right_latency)
                    if len(texts) <= 1:
                        raise
                    midpoint = len(texts) // 2
                    left, left_tokens, left_latency = self._embed(texts[:midpoint])
                    right, right_tokens, right_latency = self._embed(texts[midpoint:])
                    return (left + right, left_tokens + right_tokens,
                            left_latency + right_latency)
                recoverable = (
                    error.__class__.__name__ in {"APIConnectionError", "APITimeoutError", "InternalServerError"}
                    or (isinstance(status_code, int) and status_code >= 500)
                )
                if not recoverable or attempt == 11:
                    raise
                last_error = error
                time.sleep(5.0)
        if response is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("embedding service retry exhausted") from last_error
        latency = (time.perf_counter() - started) * 1000
        ordered = sorted(response.data, key=lambda row: row.index)
        vectors = [list(row.embedding) for row in ordered]
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        return vectors, tokens, latency
