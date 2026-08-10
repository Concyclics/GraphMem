from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from graphmem.config import GraphMemV5Config
from graphmem.domain import Conversation, Session, SourceTurn, stable_id
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.retrieval.dense_sidecar import DenseIndexCache, DenseIndexSidecar
from graphmem.storage import SQLiteGraphStore


def _store(path: Path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(path)
    memory_id = "memory-one"
    session = Session("session", memory_id, 0, "2025-01-01", "session-hash")
    texts = ("Alice booked Paris", "Bob adopted a cat", "The train was cancelled")
    turns = [
        SourceTurn(
            stable_id("turn", memory_id, index), memory_id, session.session_id, index,
            "Alice", "Bob", "user", None, text,
            hashlib.sha256(text.encode()).hexdigest(),
        )
        for index, text in enumerate(texts)
    ]
    store.ingest_conversation(
        Conversation(memory_id, "test", memory_id, "memory-hash"), (session,), turns)
    vectors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.0, 0.2])
    store.upsert_embeddings(memory_id, GraphMemV5Config().models.embedding_model, [
        (turn.turn_id, turn.content_hash, vector) for turn, vector in zip(turns, vectors)
    ])
    return store


class _Embeddings:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.delay = delay

    def create(self, *, model, input):
        self.calls += 1
        self.batch_sizes.append(len(input))
        if self.delay:
            time.sleep(self.delay)
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[1.0, 0.0, 0.0])
                  for index, _text in enumerate(input)],
            usage=SimpleNamespace(prompt_tokens=len(input)),
        )


def test_query_views_are_batched_and_persisted_across_index_instances(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    cache_path = tmp_path / "query-cache.sqlite"
    first_client = _Embeddings()
    first = QwenEmbeddingIndex(
        store, GraphMemV5Config(),
        client=SimpleNamespace(embeddings=first_client),
        query_cache_path=cache_path,
        record_usage=False,
    )

    first_rows = first.search_many(
        "memory-one", (("Where did Alice travel?", 2), ("Alice booking", 1)))

    assert first_client.calls == 1
    assert first_client.batch_sizes == [2]
    assert len(first_rows) == 2
    second_client = _Embeddings()
    second = QwenEmbeddingIndex(
        store, GraphMemV5Config(),
        client=SimpleNamespace(embeddings=second_client),
        query_cache_path=cache_path,
        record_usage=False,
    )
    second_rows = second.search_many(
        "memory-one", (("Where did Alice travel?", 2), ("Alice booking", 1)))
    assert second_client.calls == 0
    assert second_rows == first_rows
    assert second.stats["query_persistent_hits"] == 2


def test_query_embedding_singleflight_collapses_concurrent_misses(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    embeddings = _Embeddings(delay=0.05)
    index = QwenEmbeddingIndex(
        store, GraphMemV5Config(),
        client=SimpleNamespace(embeddings=embeddings),
        record_usage=False,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(
            lambda _index: index.search("memory-one", "Where did Alice travel?", 2),
            range(8),
        ))
    assert embeddings.calls == 1
    assert all(row == rows[0] for row in rows)
    assert index.stats["query_singleflight_waits"] == 7


def test_search_items_reuses_existing_graph_node_vectors(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    relation_store = _store(tmp_path / "relations.sqlite")
    model_id = GraphMemV5Config().models.embedding_model
    relation_store.upsert_embeddings("memory-one", model_id, (
        ("fact:paris", "fact-hash-1", [1.0, 0.0, 0.0]),
        ("fact:cat", "fact-hash-2", [0.0, 1.0, 0.0]),
    ))
    embeddings = _Embeddings()
    index = QwenEmbeddingIndex(
        store, GraphMemV5Config(),
        client=SimpleNamespace(embeddings=embeddings), record_usage=False)

    first = index.search_items(
        "memory-one", "Where did Alice travel?",
        ("fact:paris", "fact:cat"), 2, source_store=relation_store)
    second = index.search_items(
        "memory-one", "Where did Alice travel?",
        ("fact:paris", "fact:cat"), 2, source_store=relation_store)

    assert first[0][0] == "fact:paris"
    assert second == first
    assert embeddings.calls == 1
    assert index.stats["item_matrix_entries"] == 1


def test_numpy_dense_sidecar_matches_exact_matrix_and_invalidates(tmp_path: Path) -> None:
    store = _store(tmp_path / "graph.sqlite")
    model_id = GraphMemV5Config().models.embedding_model
    sidecar = DenseIndexSidecar(tmp_path / "dense", backend="numpy_exact")
    compiled = sidecar.build(store, "memory-one", model_id)
    assert compiled["status"] == "compiled"
    assert compiled["vector_count"] == 3
    handle = sidecar.load(store, "memory-one", model_id)
    assert handle is not None

    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    expected = store.search_embeddings("memory-one", model_id, query, limit=3)
    actual = handle.search(query, 3)
    assert [row[0] for row in actual] == [row[0] for row in expected]
    assert np.allclose([row[1] for row in actual], [row[1] for row in expected])

    cache = DenseIndexCache(sidecar, max_bytes=1024 * 1024, max_memories=2)
    assert cache.get(store, "memory-one", model_id) is not None
    assert cache.get(store, "memory-one", model_id) is not None
    assert cache.stats()["hits"] == 1

    first_turn = store.turns("memory-one")[0]
    store.upsert_embeddings(
        "memory-one", model_id,
        ((first_turn.turn_id, "stale-content-hash", [0.0, 0.0, 1.0]),),
    )
    assert sidecar.load(store, "memory-one", model_id) is None
