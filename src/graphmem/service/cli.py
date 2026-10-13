"""Installed entry point: graphmem serve / check / backup."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading

from .config import ServiceConfig
from .http import make_server
from .memory import MemoryService


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Persistent GraphMem Memory service (V5.81)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "check"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--offline", action="store_true",
                             help="persistence/API smoke mode; no semantic extraction or embeddings")
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8090)
            command.add_argument("--max-requests", type=int, default=16)
    backup = sub.add_parser("backup", help="online SQLite backup, including uncheckpointed WAL")
    backup.add_argument("--data-dir", type=Path, required=True)
    backup.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "backup":
        source = (args.data_dir / "memory.sqlite").resolve()
        with args.output.open("xb"):
            pass
        try:
            with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as db:
                with sqlite3.connect(args.output) as target:
                    db.backup(target)
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
        print(json.dumps({"backup": str(args.output.resolve())}))
        return
    config = ServiceConfig.from_file(args.config) if args.config else ServiceConfig()
    if args.data_dir:
        config = replace(config, data_dir=str(args.data_dir.resolve()))
    if args.offline:
        config = replace(config, semantic_build=False, dense_retrieval=False,
                         generate_answers=False, require_exact_tokenizer=False)
    if args.command == "check":
        # Validate packaged configuration and tokenizer without opening the DB,
        # calling model endpoints or requiring the writer lease.
        from ..tokenization import resolve_token_counter
        build, runtime = config.profiles()
        counter = resolve_token_counter(config.llm_model, require_exact=config.require_exact_tokenizer)
        print(json.dumps({"valid": True, "data_dir": str(Path(config.data_dir).resolve()),
                          "profile": runtime.profile, "readout": config.readout,
                          "semantic_build": config.semantic_build,
                          "dense_retrieval": config.dense_retrieval,
                          "token_counter": counter.describe()}, ensure_ascii=False))
        return
    api_key = os.environ.get("GRAPHMEM_API_KEY", "")
    if args.host not in {"127.0.0.1", "localhost"} and not api_key:
        parser.error("binding outside loopback requires GRAPHMEM_API_KEY")
    with MemoryService(config) as service:
        with make_server(service, args.host, args.port, api_key=api_key,
                         max_requests=args.max_requests) as server:
            previous = {}
            def stop(_signum, _frame):
                threading.Thread(target=server.shutdown, daemon=True).start()
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, stop)
            print(json.dumps({"listening": f"http://{args.host}:{server.server_port}",
                              **service.health()}, ensure_ascii=False), flush=True)
            try:
                server.serve_forever(poll_interval=0.2)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


if __name__ == "__main__":
    main()
