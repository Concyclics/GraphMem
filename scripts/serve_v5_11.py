#!/usr/bin/env python3
"""Run the bounded V5.11 multi-tenant retrieval HTTP service."""
from __future__ import annotations

import argparse
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass, replace
import enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import (  # noqa: E402
    GraphMemRuntimeConfig,
    load_runtime_config,
    runtime_config_hash,
)
from graphmem.serving import (  # noqa: E402
    AdmissionRejected,
    CompiledSidecarMaintainer,
    DenseSidecarMaintainer,
    ProcessShardedNavigator,
    RequestDeadlineExceeded,
)


def parse_cpu_ids(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("CPU range end cannot be below its start")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("CPU IDs must be unique nonnegative integers")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--runtime-config", type=Path,
        default=ROOT / "configs/v5/runtime_v5_11_balanced.json")
    parser.add_argument("--compiled-cache-dir", type=Path)
    parser.add_argument("--dense-sidecar-dir", type=Path)
    parser.add_argument("--query-embedding-cache", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--affinity-replicas", type=int)
    parser.add_argument("--cpu-ids", default="")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--sidecar-refresh-seconds", type=float)
    parser.add_argument("--no-precompile", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def effective_config(args: argparse.Namespace) -> GraphMemRuntimeConfig:
    config = load_runtime_config(args.runtime_config)
    serving = config.serving
    workers = serving.workers if args.workers is None else args.workers
    replicas = (min(serving.affinity_replicas, workers)
                if args.affinity_replicas is None else args.affinity_replicas)
    cpu_ids = parse_cpu_ids(args.cpu_ids) if args.cpu_ids else serving.worker_cpu_ids
    if cpu_ids and len(cpu_ids) != workers:
        # A portable profile intentionally leaves this empty.  When workers are
        # overridden, never silently reuse a machine-specific affinity vector.
        if args.workers is not None and not args.cpu_ids:
            cpu_ids = ()
        else:
            raise ValueError("--cpu-ids must contain one CPU ID per worker")
    serving = replace(
        serving,
        workers=workers,
        affinity_replicas=replicas,
        warm_replicas=min(serving.warm_replicas, replicas),
        worker_cpu_ids=cpu_ids,
        host=serving.host if args.host is None else args.host,
        port=serving.port if args.port is None else args.port,
        request_timeout_seconds=(serving.request_timeout_seconds
                                 if args.request_timeout is None
                                 else args.request_timeout),
        sidecar_refresh_seconds=(serving.sidecar_refresh_seconds
                                 if args.sidecar_refresh_seconds is None
                                 else args.sidecar_refresh_seconds),
        precompile_on_start=(serving.precompile_on_start
                             and not args.no_precompile),
    )
    retrieval = config.retrieval
    if args.compiled_cache_dir is not None:
        retrieval = replace(retrieval, compiled_cache_dir=str(args.compiled_cache_dir))
    if args.dense_sidecar_dir is not None:
        retrieval = replace(retrieval, dense_sidecar_dir=str(args.dense_sidecar_dir))
    if args.query_embedding_cache is not None:
        retrieval = replace(
            retrieval, query_embedding_cache_path=str(args.query_embedding_cache))
    return replace(config, retrieval=retrieval, serving=serving)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(jsonable(key)): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_cache_dir(config: GraphMemRuntimeConfig) -> Path | None:
    value = config.retrieval.compiled_cache_dir
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_runtime_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def make_handler(
    pool: ProcessShardedNavigator,
    config: GraphMemRuntimeConfig,
    maintainer: CompiledSidecarMaintainer | None,
    dense_maintainer: DenseSidecarMaintainer | None,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GraphMemRetrieval/5.11"

        def _send(self, status: int, payload: Mapping[str, object]) -> None:
            data = (json.dumps(jsonable(payload), ensure_ascii=False,
                               separators=(",", ":")) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path == "/healthz":
                self._send(200, {
                    "status": "ok",
                    "profile": config.profile,
                    "runtime_config_hash": runtime_config_hash(config),
                    "workers": config.serving.workers,
                })
                return
            if self.path == "/v1/stats":
                self._send(200, {
                    "admission": pool.admission_stats(),
                    "workers": pool.worker_cache_stats(),
                    "sidecar_maintainer": (maintainer.stats()
                                           if maintainer is not None else None),
                    "dense_sidecar_maintainer": (
                        dense_maintainer.stats()
                        if dense_maintainer is not None else None),
                })
                return
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path != "/v1/retrieve":
                self._send(404, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 1024 * 1024:
                    raise ValueError("request body must be between 1 B and 1 MiB")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, Mapping):
                    raise ValueError("request body must be a JSON object")
                memory_id = str(payload.get("memory_id", "")).strip()
                query = str(payload.get("query", "")).strip()
                tenant_id = str(payload.get("tenant_id", "default")).strip() or "default"
                if not memory_id or not query:
                    raise ValueError("memory_id and query are required")
                timeout = float(payload.get(
                    "timeout_seconds", config.serving.request_timeout_seconds))
                if not 0 < timeout <= config.serving.request_timeout_seconds:
                    raise ValueError("timeout_seconds is outside the configured bound")
                submitted = time.monotonic()
                result = pool.submit(
                    memory_id,
                    query,
                    config.query_budget,
                    tenant_id=tenant_id,
                    deadline_monotonic=submitted + timeout,
                ).result(timeout=timeout + 0.25)
                self._send(200, {
                    "result": result,
                    "end_to_end_ms": (time.monotonic() - submitted) * 1000,
                    "profile": config.profile,
                })
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._send(400, {"error": "invalid_request", "detail": str(error)})
            except AdmissionRejected as error:
                self._send(429, {"error": "admission_rejected", "detail": str(error)})
            except (RequestDeadlineExceeded, FutureTimeoutError) as error:
                self._send(504, {"error": "deadline_exceeded", "detail": str(error)})
            except BaseException as error:
                self._send(500, {
                    "error": "internal_error",
                    "detail": f"{type(error).__name__}: {error}",
                })

        def log_message(self, format_: str, *args: object) -> None:
            sys.stderr.write(
                f"{self.log_date_time_string()} {self.client_address[0]} "
                f"{format_ % args}\n")

    return Handler


def main() -> None:
    args = parse_args()
    config = effective_config(args)
    cache_dir = resolve_cache_dir(config)
    dense_dir = resolve_runtime_path(config.retrieval.dense_sidecar_dir)
    query_cache = resolve_runtime_path(
        config.retrieval.query_embedding_cache_path)
    effective = {
        "db": str(args.db.resolve()),
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_hash": runtime_config_hash(config),
        "config": config,
        "resolved_compiled_cache_dir": str(cache_dir) if cache_dir else None,
        "resolved_dense_sidecar_dir": str(dense_dir) if dense_dir else None,
        "resolved_query_embedding_cache": str(query_cache) if query_cache else None,
    }
    if args.validate_only:
        print(json.dumps(jsonable(effective), ensure_ascii=False, indent=2))
        return
    if not args.db.is_file():
        raise SystemExit(f"graph database does not exist: {args.db}")

    maintainer: CompiledSidecarMaintainer | None = None
    dense_maintainer: DenseSidecarMaintainer | None = None
    if cache_dir is not None:
        if config.serving.precompile_on_start:
            bootstrap = CompiledSidecarMaintainer(
                args.db, cache_dir,
                refresh_seconds=max(1.0, config.serving.sidecar_refresh_seconds or 1.0))
            result = bootstrap.sync_once(workers=config.serving.precompile_workers)
            print(json.dumps(jsonable({"sidecar_bootstrap": result}),
                             ensure_ascii=False), flush=True)
        if config.serving.sidecar_refresh_seconds > 0:
            maintainer = CompiledSidecarMaintainer(
                args.db, cache_dir,
                refresh_seconds=config.serving.sidecar_refresh_seconds)
            maintainer.start()

    if config.retrieval.dense_search_enabled and dense_dir is not None:
        if config.serving.precompile_on_start:
            dense_bootstrap = DenseSidecarMaintainer(
                args.db, dense_dir,
                model_id=config.retrieval.embedding_model,
                backend=config.retrieval.dense_backend,
                refresh_seconds=max(
                    1.0, config.serving.sidecar_refresh_seconds or 1.0),
            )
            dense_result = dense_bootstrap.sync_once(
                workers=config.serving.precompile_workers)
            print(json.dumps(jsonable({"dense_sidecar_bootstrap": dense_result}),
                             ensure_ascii=False), flush=True)
        if config.serving.sidecar_refresh_seconds > 0:
            dense_maintainer = DenseSidecarMaintainer(
                args.db, dense_dir,
                model_id=config.retrieval.embedding_model,
                backend=config.retrieval.dense_backend,
                refresh_seconds=config.serving.sidecar_refresh_seconds,
            )
            dense_maintainer.start()

    navigator_options = config.retrieval.navigator_options(
        compiled_cache_dir=cache_dir)
    embedding_options = config.retrieval.embedding_options(
        dense_sidecar_dir=dense_dir,
        query_embedding_cache_path=query_cache,
    )
    try:
        with ProcessShardedNavigator(
                args.db,
                navigator_options=navigator_options,
                embedding_options=embedding_options,
                **config.serving.pool_options()) as pool:
            handler = make_handler(pool, config, maintainer, dense_maintainer)
            server = ThreadingHTTPServer(
                (config.serving.host, config.serving.port), handler)
            server.daemon_threads = True
            print(json.dumps(jsonable({
                **effective,
                "listening": f"http://{config.serving.host}:{config.serving.port}",
            }), ensure_ascii=False), flush=True)
            try:
                server.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:
                pass
            finally:
                server.shutdown()
                server.server_close()
    finally:
        if maintainer is not None:
            maintainer.stop(timeout=5)
        if dense_maintainer is not None:
            dense_maintainer.stop(timeout=5)


if __name__ == "__main__":
    main()
