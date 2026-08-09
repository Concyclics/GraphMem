"""Lifecycle management for versioned compiled-memory sidecars.

SQLite is always authoritative.  This module cheaply compares the published
graph identity with a small sidecar publish record and compiles only missing or
stale views.  It is suitable both for one startup synchronization and for a
single background monitor following incremental graph publication.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import threading
import time
from typing import Iterable, Mapping

from ..retrieval import GraphNavigator
from ..retrieval.compiled_memory import CompiledMemorySidecar
from ..storage import SQLiteGraphStore


def _compile_one(
    request: tuple[str, str, str, bool, bool],
) -> dict[str, object]:
    db_path, output, memory_id, force, account_bytes = request
    started = time.perf_counter()
    store = SQLiteGraphStore(db_path, read_only=True)
    try:
        version, checksum = store.graph_identity(memory_id)
        sidecar = CompiledMemorySidecar(output)
        if not force and sidecar.is_current(memory_id, version, checksum):
            return {
                "memory_id": memory_id,
                "graph_version": version,
                "graph_checksum": checksum,
                "status": "current",
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "path": str(sidecar.path_for(memory_id)),
            }
        navigator = GraphNavigator(
            store,
            read_pool_size=1,
            metadata_cache_memories=1,
            snapshot_cache_memories=1,
            snapshot_cache_bytes=64 * 1024 * 1024,
            compiled_cache_dir=output,
        )
        artifact = navigator.precompile_memory(
            memory_id, force=True, account_bytes=account_bytes)
        path = sidecar.path_for(memory_id)
        return {
            "memory_id": memory_id,
            "graph_version": artifact.graph_version,
            "graph_checksum": artifact.graph_checksum,
            "status": "compiled",
            "nodes": len(artifact.view.nodes),
            "edges": len(artifact.view.edges),
            "turns": len(artifact.turns),
            "view_retained_bytes": artifact.view_retained_bytes,
            "total_retained_bytes": artifact.total_retained_bytes,
            "serialized_bytes": path.stat().st_size,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "path": str(path),
        }
    finally:
        store.close()


def sync_compiled_sidecars(
    db_path: str | Path,
    output: str | Path,
    *,
    memory_ids: Iterable[str] | None = None,
    workers: int = 1,
    force: bool = False,
    account_bytes: bool = True,
) -> dict[str, object]:
    """Compile every missing/stale published memory and return a manifest."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    database = Path(db_path)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    if memory_ids is None:
        store = SQLiteGraphStore(database, read_only=True)
        try:
            selected = tuple(store.memory_ids())
        finally:
            store.close()
    else:
        selected = tuple(dict.fromkeys(str(value) for value in memory_ids))
    started = time.perf_counter()
    requests = [
        (str(database), str(target), memory_id, force, account_bytes)
        for memory_id in selected
    ]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    if workers == 1:
        for request in requests:
            try:
                rows.append(_compile_one(request))
            except BaseException as error:
                failures.append({
                    "memory_id": request[2],
                    "error": f"{type(error).__name__}: {error}",
                })
    else:
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {
                pool.submit(_compile_one, request): request[2]
                for request in requests
            }
            for future in as_completed(futures):
                memory_id = futures[future]
                try:
                    rows.append(future.result())
                except BaseException as error:
                    failures.append({
                        "memory_id": memory_id,
                        "error": f"{type(error).__name__}: {error}",
                    })
    rows.sort(key=lambda row: str(row["memory_id"]))
    failures.sort(key=lambda row: row["memory_id"])
    return {
        "db_path": str(database),
        "output": str(target),
        "selected": len(selected),
        "compiled": sum(row["status"] == "compiled" for row in rows),
        "current": sum(row["status"] == "current" for row in rows),
        "failed": len(failures),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "rows": rows,
        "failures": failures,
    }


class CompiledSidecarMaintainer:
    """Single background monitor following immutable graph publications."""

    def __init__(
        self,
        db_path: str | Path,
        output: str | Path,
        *,
        refresh_seconds: float = 10.0,
        account_bytes: bool = True,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")
        self.db_path = Path(db_path)
        self.output = Path(output)
        self.refresh_seconds = refresh_seconds
        self.account_bytes = account_bytes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._runs = 0
        self._failures = 0
        self._last: Mapping[str, object] | None = None

    def sync_once(self, *, workers: int = 1,
                  force: bool = False) -> dict[str, object]:
        result = sync_compiled_sidecars(
            self.db_path,
            self.output,
            workers=workers,
            force=force,
            account_bytes=self.account_bytes,
        )
        with self._lock:
            self._runs += 1
            self._failures += int(result["failed"])
            self._last = result
        return result

    def _run(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            try:
                self.sync_once(workers=1)
            except BaseException as error:  # keep monitoring after transient I/O
                with self._lock:
                    self._runs += 1
                    self._failures += 1
                    self._last = {
                        "failed": 1,
                        "error": f"{type(error).__name__}: {error}",
                    }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="graphmem-sidecar-maintainer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "runs": self._runs,
                "failures": self._failures,
                "refresh_seconds": self.refresh_seconds,
                "running": bool(self._thread and self._thread.is_alive()),
                "last": self._last,
            }

    def __enter__(self) -> "CompiledSidecarMaintainer":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()
