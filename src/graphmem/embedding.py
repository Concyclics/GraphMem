from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np

from .config import GraphMemV5Config
from .domain import SourceTurn, stable_id
from .storage import SQLiteGraphStore


class QwenEmbeddingIndex:
    """Turn-level Qwen embedding index; usage is logged outside backbone tokens."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 client: Any | None = None, batch_size: int = 64, record_usage: bool = True) -> None:
        self.store = store
        self.config = config
        self.model_id = config.models.embedding_model
        self.batch_size = batch_size
        self._query_cache: dict[str, list[float]] = {}
        self._memory_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self.record_usage = record_usage
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.embedding_base_url, api_key="local")
        self.client = client

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
            self._memory_cache.pop(memory_id, None)
        return len(rows)

    def search(self, memory_id: str, query: str, limit: int) -> Sequence[tuple[str, float]]:
        query_text = (
            "Instruct: Retrieve conversation turns that provide all evidence needed to answer "
            "the memory question.\nQuery: " + query
        )
        cache_key = hashlib.sha256((self.model_id + "\n" + query_text).encode()).hexdigest()
        if cache_key not in self._query_cache:
            vectors, tokens, latency = self._embed([query_text])
            self._query_cache[cache_key] = vectors[0]
            if self.record_usage:
                self.store.log_embedding_call(
                    stable_id("embedding-call", memory_id, "query", cache_key), memory_id,
                    self.model_id, 1, tokens, latency,
                )
        if memory_id not in self._memory_cache:
            ids, vectors = [], []
            for row in self.store._read(
                "SELECT item_id,dimension,vector FROM embeddings WHERE memory_id=? AND model_id=? ORDER BY item_id",
                (memory_id, self.model_id),
            ):
                ids.append(str(row["item_id"]))
                vectors.append(np.frombuffer(row["vector"], dtype=np.float32, count=int(row["dimension"])))
            matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
            if matrix.size:
                matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            self._memory_cache[memory_id] = (ids, matrix)
        ids, matrix = self._memory_cache[memory_id]
        if not ids:
            return []
        query_vector = np.asarray(self._query_cache[cache_key], dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = matrix @ query_vector
        count = min(limit, len(ids))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = sorted(indices, key=lambda index: (-float(scores[index]), ids[index]))
        return [(ids[index], float(scores[index])) for index in indices]

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
                response = self.client.embeddings.create(model=self.model_id, input=list(texts))
                break
            except Exception as error:
                status_code = getattr(error, "status_code", None)
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
