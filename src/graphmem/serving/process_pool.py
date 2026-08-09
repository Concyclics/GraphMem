from __future__ import annotations

import atexit
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from typing import Iterable, Mapping, Sequence

from ..domain import NavigationResult, QueryBudget
from ..config import GraphMemV5Config
from ..embedding import QwenEmbeddingIndex
from ..retrieval import GraphNavigator
from ..storage import SQLiteGraphStore


_WORKER_STORE: SQLiteGraphStore | None = None
_WORKER_NAVIGATOR: GraphNavigator | None = None
_WORKER_EMBEDDING: QwenEmbeddingIndex | None = None


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Evidence that one serving worker has warmed an immutable snapshot."""

    pid: int
    memory_id: str
    graph_version: int
    graph_checksum: str
    cached_views: int
    estimated_bytes: int
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    compiled_hits: int = 0
    compiled_misses: int = 0
    compiled_invalid: int = 0
    compiled_hydrations: int = 0
    compiled_retained_bytes: int = 0


@dataclass(frozen=True, slots=True)
class WorkerCacheSnapshot:
    """Full per-process cache telemetry for capacity and replica accounting."""

    pid: int
    stats: Mapping[str, object]


def _close_worker() -> None:
    global _WORKER_STORE, _WORKER_NAVIGATOR, _WORKER_EMBEDDING
    if _WORKER_STORE is not None:
        _WORKER_STORE.close()
    _WORKER_STORE = None
    _WORKER_NAVIGATOR = None
    _WORKER_EMBEDDING = None


def _initialize_worker(
    db_path: str,
    navigator_options: Mapping[str, object],
    embedding_options: Mapping[str, object] | None,
    cpu_id: int | None,
) -> None:
    global _WORKER_STORE, _WORKER_NAVIGATOR, _WORKER_EMBEDDING
    if cpu_id is not None:
        try:
            os.sched_setaffinity(0, {cpu_id})
        except AttributeError as error:  # pragma: no cover - Linux serving path
            raise RuntimeError("CPU affinity is unsupported on this platform") from error
    _WORKER_STORE = SQLiteGraphStore(db_path, read_only=True)
    # A worker executes one request at a time.  One query-only connection keeps
    # WAL visibility without multiplying idle SQLite handles inside every
    # process; parallelism comes from the process shards themselves.
    options = dict(navigator_options)
    options.setdefault("read_pool_size", 1)
    if embedding_options:
        _WORKER_EMBEDDING = QwenEmbeddingIndex(
            _WORKER_STORE,
            GraphMemV5Config(),
            record_usage=False,
            **dict(embedding_options),
        )
        options["dense_search"] = _WORKER_EMBEDDING.search
        options["dense_search_many"] = _WORKER_EMBEDDING.search_many
    _WORKER_NAVIGATOR = GraphNavigator(_WORKER_STORE, **options)
    atexit.register(_close_worker)


def _navigate_worker(
    request: tuple[str, str, QueryBudget, float | None],
) -> NavigationResult:
    if _WORKER_NAVIGATOR is None:
        raise RuntimeError("GraphMem process worker was not initialized")
    memory_id, query, budget, deadline = request
    if deadline is not None and time.monotonic() >= deadline:
        raise RequestDeadlineExceeded("request expired before worker execution")
    return _WORKER_NAVIGATOR.navigate(memory_id, query, budget)


def _kill_worker() -> None:
    """Fault-injection hook: terminate the selected process without cleanup."""
    os._exit(71)


class AdmissionRejected(RuntimeError):
    """The bounded global or tenant queue has no remaining capacity."""


class RequestDeadlineExceeded(TimeoutError):
    """A queued request reached its deadline before execution began."""


class BoundedAdmissionController:
    """Thread-safe global and per-tenant outstanding-request quotas."""

    def __init__(self, global_limit: int, per_tenant_limit: int) -> None:
        if global_limit <= 0 or per_tenant_limit <= 0:
            raise ValueError("admission limits must be positive")
        self.global_limit = global_limit
        self.per_tenant_limit = per_tenant_limit
        self._global = threading.BoundedSemaphore(global_limit)
        self._tenants: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()
        self._outstanding = 0
        self._tenant_outstanding: dict[str, int] = {}
        self._submitted = self._rejected = self._completed = self._failed = 0

    def acquire(self, tenant_id: str, *, block: bool = False,
                timeout: float | None = None) -> None:
        tenant = tenant_id or "default"
        acquired_global = self._global.acquire(
            blocking=block, timeout=timeout if block else None)
        if not acquired_global:
            with self._lock:
                self._rejected += 1
            raise AdmissionRejected("global outstanding-request limit reached")
        with self._lock:
            semaphore = self._tenants.setdefault(
                tenant, threading.BoundedSemaphore(self.per_tenant_limit))
        if not semaphore.acquire(blocking=False):
            self._global.release()
            with self._lock:
                self._rejected += 1
            raise AdmissionRejected(
                f"tenant {tenant!r} outstanding limit reached")
        with self._lock:
            self._submitted += 1
            self._outstanding += 1
            self._tenant_outstanding[tenant] = (
                self._tenant_outstanding.get(tenant, 0) + 1)

    def release(self, tenant_id: str, *, failed: bool) -> None:
        tenant = tenant_id or "default"
        with self._lock:
            semaphore = self._tenants[tenant]
            self._outstanding -= 1
            self._tenant_outstanding[tenant] -= 1
            if failed:
                self._failed += 1
            else:
                self._completed += 1
        semaphore.release()
        self._global.release()

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "global_limit": self.global_limit,
                "per_tenant_limit": self.per_tenant_limit,
                "outstanding": self._outstanding,
                "tenant_outstanding": dict(self._tenant_outstanding),
                "submitted": self._submitted,
                "rejected": self._rejected,
                "completed": self._completed,
                "failed": self._failed,
            }


def _warm_worker(
    request: tuple[str, tuple[str, ...], QueryBudget],
) -> WorkerSnapshot:
    if _WORKER_NAVIGATOR is None:
        raise RuntimeError("GraphMem process worker was not initialized")
    memory_id, queries, budget = request
    view = _WORKER_NAVIGATOR.warm_memory(memory_id, queries, budget)
    navigator_stats = _WORKER_NAVIGATOR.cache_stats()
    stats = navigator_stats["runtime"]
    compiled = navigator_stats["compiled"]
    return WorkerSnapshot(
        pid=os.getpid(), memory_id=memory_id,
        graph_version=view.graph_version,
        graph_checksum=view.graph_checksum,
        cached_views=int(stats["views"]),
        estimated_bytes=int(stats["estimated_bytes"]),
        cache_hits=int(stats["hits"]),
        cache_misses=int(stats["misses"]),
        cache_evictions=int(stats["evictions"]),
        compiled_hits=int(compiled.get("hits", 0)),
        compiled_misses=int(compiled.get("misses", 0)),
        compiled_invalid=int(compiled.get("invalid", 0)),
        compiled_hydrations=int(compiled.get("hydrations", 0)),
        compiled_retained_bytes=int(compiled.get("retained_artifact_bytes", 0)),
    )


def _warm_memories_worker(
    request: tuple[tuple[tuple[str, tuple[str, ...]], ...], QueryBudget],
) -> tuple[WorkerSnapshot, ...]:
    workloads, budget = request
    return tuple(_warm_worker((memory_id, queries, budget))
                 for memory_id, queries in workloads)


def _cache_snapshot_worker() -> WorkerCacheSnapshot:
    if _WORKER_NAVIGATOR is None:
        raise RuntimeError("GraphMem process worker was not initialized")
    stats = dict(_WORKER_NAVIGATOR.cache_stats())
    try:
        cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - Linux serving path
        cpu_affinity = ()
    stats["process"] = {"cpu_affinity": cpu_affinity}
    if _WORKER_EMBEDDING is not None:
        stats["embedding"] = dict(_WORKER_EMBEDDING.stats)
    return WorkerCacheSnapshot(os.getpid(), stats)


class ProcessShardedNavigator:
    """Persistent multi-process query plane over a WAL-backed graph authority.

    Retrieval contains substantial Python ranking and packing work.  Threads can
    overlap SQLite I/O but cannot execute those CPU-bound stages in parallel due
    to the GIL.  This pool keeps one immutable graph/index cache per worker and
    dispatches independent users across processes.  Publication remains a
    single-writer operation; each worker notices the new graph version and
    atomically compiles the next snapshot on its first subsequent request.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        workers: int = 4,
        navigator_options: Mapping[str, object] | None = None,
        embedding_options: Mapping[str, object] | None = None,
        start_method: str = "spawn",
        max_queued: int = 32,
        per_tenant_outstanding: int = 8,
        affinity_replicas: int = 1,
        retry_broken_worker: int = 1,
        worker_cpu_ids: Sequence[int] | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {start_method!r}")
        self.db_path = Path(db_path)
        self.workers = workers
        self.navigator_options = dict(navigator_options or {})
        self.embedding_options = (
            dict(embedding_options) if embedding_options is not None else None)
        if max_queued < 0:
            raise ValueError("max_queued cannot be negative")
        if not 1 <= affinity_replicas <= workers:
            raise ValueError("affinity_replicas must be in [1, workers]")
        self.affinity_replicas = affinity_replicas
        if retry_broken_worker < 0:
            raise ValueError("retry_broken_worker cannot be negative")
        self.retry_broken_worker = retry_broken_worker
        if worker_cpu_ids is None:
            self.worker_cpu_ids: tuple[int | None, ...] = (None,) * workers
        else:
            normalized_cpu_ids = tuple(int(value) for value in worker_cpu_ids)
            if len(normalized_cpu_ids) != workers:
                raise ValueError("worker_cpu_ids must contain one CPU ID per worker")
            if len(set(normalized_cpu_ids)) != workers:
                raise ValueError("worker_cpu_ids must be unique")
            if any(value < 0 for value in normalized_cpu_ids):
                raise ValueError("worker_cpu_ids cannot contain negative values")
            self.worker_cpu_ids = normalized_cpu_ids
        self._shard_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._shard_outstanding = [0] * workers
        self._context = mp.get_context(start_method)
        # A shared ProcessPoolExecutor cannot target a specific child process.
        # One single-worker executor per shard makes memory affinity real and
        # keeps that memory's immutable snapshot/index hot in one process.
        self._executors = [self._new_executor(shard) for shard in range(workers)]
        self._executor_generation = [0] * workers
        self._worker_restarts = 0
        self._inflight_retries = 0
        self._admission = BoundedAdmissionController(
            workers + max_queued, per_tenant_outstanding)
        self._closed = False

    def _new_executor(self, shard: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=1, mp_context=self._context,
            initializer=_initialize_worker,
            initargs=(str(self.db_path), self.navigator_options, self.embedding_options,
                      self.worker_cpu_ids[shard]))

    def _restart_shard(self, shard: int, failed_generation: int) -> None:
        """Replace one broken executor once, even if many callbacks observe it."""
        with self._executor_lock:
            if self._closed:
                raise RuntimeError("process navigator pool is closed")
            if self._executor_generation[shard] != failed_generation:
                return
            previous = self._executors[shard]
            self._executors[shard] = self._new_executor(shard)
            self._executor_generation[shard] += 1
            self._worker_restarts += 1
            previous.shutdown(wait=False, cancel_futures=True)

    def shard_for_memory(self, memory_id: str) -> int:
        return self.affinity_shards(memory_id)[0]

    def affinity_shards(self, memory_id: str) -> tuple[int, ...]:
        """Stable rendezvous candidates for cache locality and hot-key scale."""
        ranked = sorted(range(self.workers), key=lambda shard: (
            -int.from_bytes(hashlib.blake2b(
                f"{memory_id}:{shard}".encode("utf-8"),
                digest_size=8).digest(), "big"), shard))
        return tuple(ranked[:self.affinity_replicas])

    def _admit_shard(self, memory_id: str) -> int:
        with self._shard_lock:
            shard = min(
                self.affinity_shards(memory_id),
                key=lambda item: (self._shard_outstanding[item], item))
            self._shard_outstanding[shard] += 1
            return shard

    def _release_shard(self, shard: int) -> None:
        with self._shard_lock:
            self._shard_outstanding[shard] -= 1

    def submit(
        self, memory_id: str, query: str, budget: QueryBudget, *,
        tenant_id: str = "default", deadline_monotonic: float | None = None,
        block: bool = False, timeout: float | None = None,
    ) -> Future[NavigationResult]:
        if self._closed:
            raise RuntimeError("process navigator pool is closed")
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise RequestDeadlineExceeded("request deadline has already elapsed")
        self._admission.acquire(tenant_id, block=block, timeout=timeout)
        shard = self._admit_shard(memory_id)
        public: Future[NavigationResult] = Future()
        request = (memory_id, query, budget, deadline_monotonic)
        finish_lock = threading.Lock()
        finished = False

        def finish(*, result: NavigationResult | None = None,
                   error: BaseException | None = None) -> None:
            nonlocal finished
            with finish_lock:
                if finished:
                    return
                finished = True
            self._release_shard(shard)
            self._admission.release(tenant_id, failed=error is not None)
            if public.cancelled():
                return
            if error is not None:
                public.set_exception(error)
            else:
                assert result is not None
                public.set_result(result)

        def dispatch(retries_left: int) -> None:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                finish(error=RequestDeadlineExceeded(
                    "request expired before worker retry"))
                return
            with self._executor_lock:
                executor = self._executors[shard]
                generation = self._executor_generation[shard]
            try:
                inner = executor.submit(_navigate_worker, request)
            except BrokenProcessPool as error:
                if retries_left <= 0:
                    finish(error=error)
                    return
                try:
                    self._restart_shard(shard, generation)
                    with self._executor_lock:
                        self._inflight_retries += 1
                    dispatch(retries_left - 1)
                except BaseException as restart_error:
                    finish(error=restart_error)
                return
            except BaseException as error:
                finish(error=error)
                return

            def completed(row: Future[NavigationResult]) -> None:
                try:
                    result = row.result()
                except BrokenProcessPool as error:
                    if retries_left <= 0:
                        finish(error=error)
                        return
                    try:
                        self._restart_shard(shard, generation)
                        with self._executor_lock:
                            self._inflight_retries += 1
                        dispatch(retries_left - 1)
                    except BaseException as restart_error:
                        finish(error=restart_error)
                except BaseException as error:
                    finish(error=error)
                else:
                    finish(result=result)

            inner.add_done_callback(completed)

        dispatch(self.retry_broken_worker)
        return public

    def navigate_many(
        self,
        requests: Iterable[tuple[str, str, QueryBudget]],
        *,
        chunksize: int = 1,
    ) -> list[NavigationResult]:
        if self._closed:
            raise RuntimeError("process navigator pool is closed")
        del chunksize  # affinity dispatches individual requests intentionally
        futures = [self.submit(*request, block=True) for request in requests]
        return [future.result() for future in futures]

    def warm(
        self,
        memory_id: str,
        queries: Sequence[str],
        budget: QueryBudget,
    ) -> tuple[WorkerSnapshot, ...]:
        """Warm every worker and verify that all pinned the same graph version."""
        if self._closed:
            raise RuntimeError("process navigator pool is closed")
        if not queries:
            raise ValueError("at least one warm query is required")
        futures = [executor.submit(
            _warm_worker, (memory_id, tuple(queries), budget))
            for executor in self._executors]
        snapshots = {row.pid: row for row in
                     (future.result() for future in futures)}
        if len(snapshots) != self.workers:
            raise RuntimeError(
                f"warmed {len(snapshots)} of {self.workers} process workers")
        versions = {(row.graph_version, row.graph_checksum)
                    for row in snapshots.values()}
        if len(versions) != 1:
            raise RuntimeError(f"workers pinned inconsistent graph snapshots: {versions!r}")
        return tuple(sorted(snapshots.values(), key=lambda row: row.pid))

    def warm_affinity(
        self,
        workloads: Mapping[str, Sequence[str]],
        budget: QueryBudget,
        *,
        replicas: int | None = None,
    ) -> tuple[WorkerSnapshot, ...]:
        """Prewarm each memory only on its rendezvous-affinity worker(s).

        This avoids the old all-to-all warmup, which multiplied every tenant's
        immutable indexes by the process count.  ``replicas=1`` is the lowest
        memory mode; the default follows ``affinity_replicas``.
        """
        if self._closed:
            raise RuntimeError("process navigator pool is closed")
        replica_count = self.affinity_replicas if replicas is None else replicas
        if not 1 <= replica_count <= self.affinity_replicas:
            raise ValueError("replicas must be in [1, affinity_replicas]")
        grouped: list[list[tuple[str, tuple[str, ...]]]] = [
            [] for _ in range(self.workers)]
        for memory_id, queries in workloads.items():
            if not queries:
                raise ValueError(f"memory {memory_id!r} has no warm query")
            for shard in self.affinity_shards(memory_id)[:replica_count]:
                grouped[shard].append((memory_id, tuple(queries)))
        futures = [
            (shard, self._executors[shard].submit(
                _warm_memories_worker, (tuple(rows), budget)))
            for shard, rows in enumerate(grouped) if rows]
        snapshots = tuple(
            snapshot
            for _shard, future in futures
            for snapshot in future.result())
        by_memory: dict[str, set[tuple[int, str]]] = {}
        for row in snapshots:
            by_memory.setdefault(row.memory_id, set()).add(
                (row.graph_version, row.graph_checksum))
        inconsistent = {
            memory_id: versions for memory_id, versions in by_memory.items()
            if len(versions) != 1}
        if inconsistent:
            raise RuntimeError(
                f"affinity replicas pinned inconsistent snapshots: {inconsistent!r}")
        return snapshots

    def worker_cache_stats(self) -> tuple[WorkerCacheSnapshot, ...]:
        """Collect cache hit/eviction/sidecar telemetry from every shard."""
        if self._closed:
            raise RuntimeError("process navigator pool is closed")
        rows = [executor.submit(_cache_snapshot_worker).result()
                for executor in self._executors]
        return tuple(sorted(rows, key=lambda row: row.pid))

    def admission_stats(self) -> Mapping[str, object]:
        result = self._admission.stats()
        with self._shard_lock:
            result["shard_outstanding"] = tuple(self._shard_outstanding)
        result["affinity_replicas"] = self.affinity_replicas
        with self._executor_lock:
            result["worker_restarts"] = self._worker_restarts
            result["inflight_retries"] = self._inflight_retries
        return result

    def inject_worker_crash(self, shard: int) -> Future[None]:
        """Terminate one worker for an explicit availability experiment."""
        if not 0 <= shard < self.workers:
            raise ValueError("shard is outside the worker range")
        with self._executor_lock:
            return self._executors[shard].submit(_kill_worker)

    def close(self, *, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            with self._executor_lock:
                executors = tuple(self._executors)
            for executor in executors:
                executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "ProcessShardedNavigator":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
