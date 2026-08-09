"""Versioned per-memory dense indexes for the online query path.

SQLite remains authoritative.  Each sidecar contains only normalized source-
turn vectors for one memory/model/graph publication.  The stable manifest is
published last and points at an immutable versioned data file, so readers see
either the previous complete index or the next complete index, never a partial
write.  ``faiss_flat`` is exact and is preferred when FAISS is installed;
``numpy_exact`` is the dependency-free mmap fallback with identical semantics.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..storage import SQLiteGraphStore


DENSE_INDEX_SCHEMA = "graphmem-v5.18-dense-index-v1"
SUPPORTED_BACKENDS = frozenset({"auto", "numpy_exact", "faiss_flat"})


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str) -> str:
    normalized = str(backend).strip().casefold()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported dense index backend: {backend!r}")
    if normalized == "auto":
        return "faiss_flat" if faiss_available() else "numpy_exact"
    if normalized == "faiss_flat" and not faiss_available():
        raise RuntimeError(
            "faiss_flat requested but faiss is not installed; install GraphMem[dense] "
            "or use backend=auto/numpy_exact")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_stem(memory_id: str) -> str:
    digest = hashlib.sha256(memory_id.encode("utf-8")).hexdigest()
    readable = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in memory_id
    )[:48].strip("_") or "memory"
    return f"{readable}.{digest[:20]}"


def _turn_embedding_rows(
    store: SQLiteGraphStore,
    memory_id: str,
    model_id: str,
) -> tuple[list[str], list[str], np.ndarray, str]:
    expected = {turn.turn_id: turn.content_hash for turn in store.turns(memory_id)}
    rows: list[tuple[str, str, np.ndarray]] = []
    for row in store._read(
        "SELECT item_id,content_hash,dimension,vector FROM embeddings "
        "WHERE memory_id=? AND model_id=? ORDER BY item_id",
        (memory_id, model_id),
    ):
        item_id = str(row["item_id"])
        content_hash = str(row["content_hash"])
        if expected.get(item_id) != content_hash:
            continue
        dimension = int(row["dimension"])
        vector = np.frombuffer(row["vector"], dtype=np.float32, count=dimension)
        if vector.size != dimension:
            raise ValueError(f"truncated embedding vector for {item_id}")
        rows.append((item_id, content_hash, vector))
    found = {item_id for item_id, _content_hash, _vector in rows}
    missing = sorted(set(expected) - found)
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"memory {memory_id!r} is missing {len(missing)} turn embeddings ({preview})")
    ids = [row[0] for row in rows]
    hashes = [row[1] for row in rows]
    matrix = np.stack([row[2] for row in rows]) if rows else np.empty((0, 0), np.float32)
    if matrix.size:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    fingerprint = hashlib.sha256()
    fingerprint.update(model_id.encode("utf-8"))
    for item_id, content_hash in zip(ids, hashes):
        fingerprint.update(b"\0")
        fingerprint.update(item_id.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(content_hash.encode("ascii"))
    return ids, hashes, matrix, fingerprint.hexdigest()


def _turn_embedding_identity(store: SQLiteGraphStore, memory_id: str,
                             model_id: str) -> tuple[int, str]:
    expected = {turn.turn_id: turn.content_hash for turn in store.turns(memory_id)}
    present = store.embedding_hashes(memory_id, model_id)
    missing = sorted(
        item_id for item_id, content_hash in expected.items()
        if present.get(item_id) != content_hash)
    if missing:
        return len(expected), ""
    fingerprint = hashlib.sha256()
    fingerprint.update(model_id.encode("utf-8"))
    for item_id in sorted(expected):
        fingerprint.update(b"\0")
        fingerprint.update(item_id.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(expected[item_id].encode("ascii"))
    return len(expected), fingerprint.hexdigest()


@dataclass(frozen=True, slots=True)
class DenseIndexHandle:
    memory_id: str
    model_id: str
    backend: str
    graph_version: int
    graph_checksum: str
    embedding_checksum: str
    ids: tuple[str, ...]
    dimension: int
    retained_bytes: int
    data: Any

    def search(self, query_vector: Sequence[float] | np.ndarray,
               limit: int) -> list[tuple[str, float]]:
        if not self.ids or limit <= 0:
            return []
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError(
                f"query dimension {vector.size} does not match dense index {self.dimension}")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12 or not np.isfinite(vector).all():
            return []
        vector = np.ascontiguousarray(vector / norm, dtype=np.float32)
        count = min(int(limit), len(self.ids))
        if self.backend == "faiss_flat":
            scores, indices = self.data.search(vector.reshape(1, -1), count)
            rows = [
                (self.ids[int(index)], float(score))
                for index, score in zip(indices[0], scores[0]) if int(index) >= 0
            ]
            rows.sort(key=lambda row: (-row[1], row[0]))
            return rows
        scores = self.data @ vector
        indices = np.argpartition(-scores, count - 1)[:count]
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), self.ids[index]))
        return [(self.ids[index], float(scores[index])) for index in ordered]


class DenseIndexSidecar:
    """Atomic compiler/loader for disposable per-memory dense indexes."""

    def __init__(self, root: str | Path, *, backend: str = "auto") -> None:
        self.root = Path(root)
        self.backend = str(backend).strip().casefold()
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported dense index backend: {backend!r}")

    def manifest_path(self, memory_id: str) -> Path:
        return self.root / f"{_memory_stem(memory_id)}.dense.json"

    def _read_manifest(self, memory_id: str) -> Mapping[str, Any] | None:
        try:
            payload = json.loads(self.manifest_path(memory_id).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, Mapping) else None

    def is_current(self, store: SQLiteGraphStore, memory_id: str,
                   model_id: str) -> bool:
        payload = self._read_manifest(memory_id)
        if payload is None:
            return False
        graph_version, graph_checksum = store.graph_identity(memory_id)
        vector_count, embedding_checksum = _turn_embedding_identity(
            store, memory_id, model_id)
        expected_backend = self.backend
        return bool(
            payload.get("schema_version") == DENSE_INDEX_SCHEMA
            and payload.get("memory_id") == memory_id
            and payload.get("model_id") == model_id
            and int(payload.get("graph_version", -1)) == graph_version
            and payload.get("graph_checksum") == graph_checksum
            and int(payload.get("vector_count", -1)) == vector_count
            and payload.get("embedding_checksum") == embedding_checksum
            and (expected_backend == "auto" or payload.get("backend") == expected_backend)
            and (self.root / str(payload.get("data_file", ""))).is_file()
        )

    def build(self, store: SQLiteGraphStore, memory_id: str, model_id: str,
              *, force: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        if not force and self.is_current(store, memory_id, model_id):
            payload = dict(self._read_manifest(memory_id) or {})
            return {**payload, "status": "current", "elapsed_ms": 0.0}
        backend = resolve_backend(self.backend)
        graph_version, graph_checksum = store.graph_identity(memory_id)
        ids, _hashes, matrix, embedding_checksum = _turn_embedding_rows(
            store, memory_id, model_id)
        if not ids:
            raise ValueError(f"memory {memory_id!r} has no source-turn embeddings")
        dimension = int(matrix.shape[1])
        version_key = hashlib.sha256(
            f"{graph_version}\n{graph_checksum}\n{embedding_checksum}\n{backend}".encode()
        ).hexdigest()[:20]
        extension = "faiss" if backend == "faiss_flat" else "npy"
        data_name = f"{_memory_stem(memory_id)}.{version_key}.{extension}"
        self.root.mkdir(parents=True, exist_ok=True)
        data_target = self.root / data_name
        temporary: Path | None = None
        if not data_target.is_file():
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{data_name}.", suffix=".tmp",
                    dir=self.root, delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    if backend == "faiss_flat":
                        import faiss
                        index = faiss.IndexFlatIP(dimension)
                        index.add(matrix)
                        handle.close()
                        faiss.write_index(index, str(temporary))
                        with temporary.open("rb") as saved:
                            os.fsync(saved.fileno())
                    else:
                        np.save(handle, matrix, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, data_target)
            except BaseException:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                raise
        data_size = data_target.stat().st_size
        data_sha256 = _file_sha256(data_target)
        payload: dict[str, Any] = {
            "schema_version": DENSE_INDEX_SCHEMA,
            "memory_id": memory_id,
            "model_id": model_id,
            "backend": backend,
            "graph_version": graph_version,
            "graph_checksum": graph_checksum,
            "embedding_checksum": embedding_checksum,
            "vector_count": len(ids),
            "dimension": dimension,
            "ids": ids,
            "data_file": data_name,
            "data_bytes": data_size,
            "data_sha256": data_sha256,
            "created_ns": time.time_ns(),
        }
        manifest_target = self.manifest_path(memory_id)
        metadata_temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{manifest_target.name}.",
                suffix=".tmp", dir=self.root, delete=False,
            ) as handle:
                metadata_temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(metadata_temporary, 0o600)
            os.replace(metadata_temporary, manifest_target)
        except BaseException:
            if metadata_temporary is not None:
                try:
                    metadata_temporary.unlink()
                except FileNotFoundError:
                    pass
            raise
        return {
            **payload,
            "status": "compiled",
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def load(self, store: SQLiteGraphStore, memory_id: str,
             model_id: str) -> DenseIndexHandle | None:
        payload = self._read_manifest(memory_id)
        if payload is None:
            return None
        graph_version, graph_checksum = store.graph_identity(memory_id)
        vector_count, embedding_checksum = _turn_embedding_identity(
            store, memory_id, model_id)
        backend = str(payload.get("backend", ""))
        if not (
            payload.get("schema_version") == DENSE_INDEX_SCHEMA
            and payload.get("memory_id") == memory_id
            and payload.get("model_id") == model_id
            and int(payload.get("graph_version", -1)) == graph_version
            and payload.get("graph_checksum") == graph_checksum
            and int(payload.get("vector_count", -1)) == vector_count
            and payload.get("embedding_checksum") == embedding_checksum
            and backend in {"numpy_exact", "faiss_flat"}
            and (self.backend == "auto" or self.backend == backend)
        ):
            return None
        data_path = self.root / str(payload.get("data_file", ""))
        try:
            if (not data_path.is_file()
                    or data_path.stat().st_size != int(payload.get("data_bytes", -1))
                    or _file_sha256(data_path) != payload.get("data_sha256")):
                return None
            ids = tuple(str(value) for value in payload["ids"])
            dimension = int(payload["dimension"])
            if len(ids) != int(payload["vector_count"]):
                return None
            if backend == "faiss_flat":
                import faiss
                data = faiss.read_index(str(data_path))
                if data.ntotal != len(ids) or data.d != dimension:
                    return None
                retained_bytes = int(payload["data_bytes"])
            else:
                data = np.load(data_path, mmap_mode="r", allow_pickle=False)
                if data.shape != (len(ids), dimension) or data.dtype != np.float32:
                    return None
                retained_bytes = int(data.nbytes)
        except (ImportError, OSError, ValueError, TypeError, KeyError):
            return None
        return DenseIndexHandle(
            memory_id=memory_id,
            model_id=model_id,
            backend=backend,
            graph_version=graph_version,
            graph_checksum=graph_checksum,
            embedding_checksum=str(payload["embedding_checksum"]),
            ids=ids,
            dimension=dimension,
            retained_bytes=retained_bytes,
            data=data,
        )


class DenseIndexCache:
    """Byte- and count-bounded LRU over per-memory dense index handles."""

    def __init__(self, sidecar: DenseIndexSidecar, *, max_bytes: int = 256 * 1024 * 1024,
                 max_memories: int = 32) -> None:
        if max_bytes <= 0 or max_memories <= 0:
            raise ValueError("dense cache limits must be positive")
        self.sidecar = sidecar
        self.max_bytes = int(max_bytes)
        self.max_memories = int(max_memories)
        self._rows: OrderedDict[str, DenseIndexHandle] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "invalid": 0, "evictions": 0}

    def get(self, store: SQLiteGraphStore, memory_id: str,
            model_id: str) -> DenseIndexHandle | None:
        graph_version, graph_checksum = store.graph_identity(memory_id)
        with self._lock:
            cached = self._rows.get(memory_id)
            if (cached is not None and cached.model_id == model_id
                    and cached.graph_version == graph_version
                    and cached.graph_checksum == graph_checksum):
                self._rows.move_to_end(memory_id)
                self._stats["hits"] += 1
                return cached
            if cached is not None:
                self._bytes -= cached.retained_bytes
                self._rows.pop(memory_id, None)
                self._stats["invalid"] += 1
        loaded = self.sidecar.load(store, memory_id, model_id)
        if loaded is None:
            with self._lock:
                self._stats["misses"] += 1
            return None
        with self._lock:
            previous = self._rows.pop(memory_id, None)
            if previous is not None:
                self._bytes -= previous.retained_bytes
            self._rows[memory_id] = loaded
            self._bytes += loaded.retained_bytes
            while (len(self._rows) > self.max_memories or self._bytes > self.max_bytes):
                evicted_id, evicted = self._rows.popitem(last=False)
                self._bytes -= evicted.retained_bytes
                self._stats["evictions"] += 1
                if evicted_id == memory_id:
                    break
        return loaded

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._rows),
                "retained_bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "max_memories": self.max_memories,
                "backend": self.sidecar.backend,
                "directory": str(self.sidecar.root),
            }
