#!/usr/bin/env python3
"""Batch-build an exact raw-turn Mem0 OSS index for read-plane comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from v5_11_pareto_common import load_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dims", type=int, default=1024)
    parser.add_argument("--embedding-db", type=Path,
                        help="reuse existing turn vectors instead of calling the embedding API")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted build after verifying each user partition.")
    return parser.parse_args()


def memory_config(args: argparse.Namespace, collection: str) -> dict:
    return {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": collection,
            "embedding_model_dims": args.embedding_dims,
            "path": str(args.qdrant_path),
            "on_disk": False,
        }},
        "embedder": {"provider": "openai", "config": {
            "model": args.embedding_model,
            "openai_base_url": args.embedding_url,
            "api_key": "local-benchmark",
        }},
        "llm": {"provider": "openai", "config": {
            "model": "unused-infer-false",
            "api_key": "local-benchmark",
            "openai_base_url": "http://127.0.0.1:8002/v1",
        }},
        "history_db_path": str(args.state_dir / "history.db"),
    }


def main() -> None:
    args = parse_args()
    workload, _ = load_workload(args.workload)
    collection = (f"graphmem_pareto_raw{workload['memory_count']}_"
                  + workload["source"]["turn_payload_sha256"][:12])
    expected = {
        "schema_version": "mem0-raw-turn-batched-v2",
        "collection": collection,
        "source_sha256": workload["source"]["turn_payload_sha256"],
        "memory_ids": workload["memory_ids"],
        "turn_counts": workload["turn_counts"],
        "embedding_model": args.embedding_model,
        "embedding_dims": args.embedding_dims,
        "embedding_source": (str(args.embedding_db.resolve())
                             if args.embedding_db else "online_api"),
        "infer": False,
        "payload_contract": "Mem0 infer=False raw message",
    }
    manifest_path = args.qdrant_path / "graphmem_pareto_manifest.json"
    if args.reuse and manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("--reuse manifest differs from requested full workload")
        print(json.dumps({"reused": True, "collection": collection,
                          "manifest": str(manifest_path)}))
        return
    if args.qdrant_path.exists() and any(args.qdrant_path.iterdir()) and not args.resume:
        raise SystemExit(
            "qdrant path is not empty; pass --reuse/--resume or choose a fresh path")
    args.qdrant_path.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_DIR"] = str(args.state_dir)
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["OPENAI_API_KEY"] = "local-benchmark"
    from mem0 import Memory
    from mem0.memory.main import lemmatize_for_bm25
    from qdrant_client.models import FilterSelector

    memory = Memory.from_config(memory_config(args, collection))
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    embedding_con = (sqlite3.connect(
        f"file:{args.embedding_db}?mode=ro", uri=True) if args.embedding_db else None)
    total_rows = 0
    total_embedding_ms = 0.0
    total_insert_ms = 0.0
    started_all = time.monotonic()
    try:
        for memory_index, memory_id in enumerate(workload["memory_ids"], start=1):
            rows = con.execute(
                """SELECT turn_id, session_id, turn_index, role, speaker,
                          COALESCE(timestamp, ''), raw_text
                   FROM source_turns WHERE memory_id=?
                   ORDER BY session_id, turn_index, turn_id""",
                (memory_id,),
            ).fetchall()
            if len(rows) != workload["turn_counts"][memory_id]:
                raise RuntimeError(f"turn count changed for {memory_id}")
            if args.resume:
                user_filter = memory.vector_store._create_filter(
                    {"user_id": memory_id})
                existing = memory.vector_store.client.count(
                    collection_name=collection,
                    count_filter=user_filter,
                    exact=True,
                ).count
                if existing == len(rows):
                    total_rows += existing
                    print(json.dumps({
                        "memory": memory_index,
                        "of": len(workload["memory_ids"]),
                        "memory_id": memory_id,
                        "turns": len(rows),
                        "indexed_total": total_rows,
                        "resumed": "verified-complete",
                    }, ensure_ascii=False), flush=True)
                    continue
                if existing:
                    memory.vector_store.client.delete(
                        collection_name=collection,
                        points_selector=FilterSelector(filter=user_filter),
                        wait=True,
                    )
                    print(json.dumps({
                        "memory_id": memory_id,
                        "removed_partial_points": existing,
                    }, ensure_ascii=False), flush=True)
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                texts = []
                payloads = []
                ids = []
                now = datetime.now(timezone.utc).isoformat()
                for turn_id, session_id, turn_index, role, speaker, timestamp, raw_text in batch:
                    text = f"[{session_id}][{timestamp}] {speaker}: {raw_text}"
                    texts.append(text)
                    ids.append(str(uuid.uuid5(
                        uuid.NAMESPACE_URL, f"graphmem:{memory_id}:{turn_id}")))
                    payloads.append({
                        "data": text,
                        "hash": hashlib.md5(text.encode()).hexdigest(),
                        "created_at": now,
                        "updated_at": now,
                        "text_lemmatized": lemmatize_for_bm25(text),
                        "user_id": memory_id,
                        "role": role if role in {"user", "assistant"} else "user",
                        "actor_id": speaker,
                        "source": "graphmem-v5.11-pareto-raw-turn",
                        "turn_id": str(turn_id),
                        "session_id": str(session_id),
                        "turn_index": int(turn_index),
                    })
                embed_started = time.monotonic()
                if embedding_con is not None:
                    turn_ids = [str(row[0]) for row in batch]
                    placeholders = ",".join("?" for _ in turn_ids)
                    vector_rows = embedding_con.execute(
                        f"SELECT item_id,dimension,vector FROM embeddings "
                        f"WHERE model_id=? AND item_id IN ({placeholders})",
                        (args.embedding_model, *turn_ids),
                    ).fetchall()
                    by_id = {
                        str(item_id): np.frombuffer(blob, dtype=np.float32,
                                                    count=int(dimension)).tolist()
                        for item_id, dimension, blob in vector_rows
                    }
                    missing = [turn_id for turn_id in turn_ids if turn_id not in by_id]
                    if missing:
                        raise RuntimeError(
                            f"missing {len(missing)} cached embeddings for {memory_id}")
                    vectors = [by_id[turn_id] for turn_id in turn_ids]
                else:
                    vectors = memory.embedding_model.embed_batch(texts, "add")
                total_embedding_ms += (time.monotonic() - embed_started) * 1000
                insert_started = time.monotonic()
                memory.vector_store.insert(vectors=vectors, ids=ids, payloads=payloads)
                total_insert_ms += (time.monotonic() - insert_started) * 1000
                total_rows += len(batch)
            print(json.dumps({
                "memory": memory_index,
                "of": len(workload["memory_ids"]),
                "memory_id": memory_id,
                "turns": len(rows),
                "indexed_total": total_rows,
            }, ensure_ascii=False), flush=True)
    finally:
        con.close()
        if embedding_con is not None:
            embedding_con.close()
        memory.vector_store.client.close()
    elapsed = time.monotonic() - started_all
    if total_rows != workload["turn_count"]:
        raise RuntimeError(f"indexed {total_rows}, expected {workload['turn_count']}")
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "reused": False,
        "collection": collection,
        "indexed_turns": total_rows,
        "elapsed_sec": elapsed,
        "embedding_sec": total_embedding_ms / 1000,
        "insert_sec": total_insert_ms / 1000,
        "manifest": str(manifest_path),
    }
    (args.qdrant_path / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
