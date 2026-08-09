#!/usr/bin/env python3
"""Embed atomic fact summaries into a resumable sidecar for candidate audits."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.domain import NodeType  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.eval import load_dev_questions, load_gold_turns  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    hard = WORKSPACE / (
        "artifacts/development_sets/"
        "hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/atomic_augmented_dev200/"
                        "report_graph.sqlite")
    parser.add_argument("--output-db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_13/atomic_node_embeddings_dev200.sqlite")
    parser.add_argument("--lme", type=Path, default=hard /
                        "longmemeval_hard_multisession50_temporal50.json")
    parser.add_argument("--locomo", type=Path, default=hard /
                        "locomo_hard_cat1_multihop50_cat2_temporal50.json")
    parser.add_argument("--gold", type=Path, default=ROOT /
                        "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT /
                        "configs/v5/v5_10_report.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_db.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output_db} exists; pass --resume or choose another output")
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    source = SQLiteGraphStore(args.source_db, read_only=True)
    target = SQLiteGraphStore(args.output_db)
    index = QwenEmbeddingIndex(target, config, batch_size=args.batch_size)
    questions = list(load_dev_questions(
        args.lme, args.locomo, load_gold_turns(args.gold)))
    memory_ids = tuple(dict.fromkeys(sorted(
        question.memory_id for question in questions)))
    started = time.perf_counter()
    rows = []
    for ordinal, memory_id in enumerate(memory_ids, 1):
        conversation = source.conversation(memory_id)
        if conversation is None:
            continue
        target.ingest_conversation(
            conversation, source.sessions(memory_id), source.turns(memory_id))
        facts = tuple(node for node in source.nodes(memory_id)
                      if node.node_type == NodeType.CANONICAL_FACT)
        before = target._read_one(
            "SELECT count(*) FROM embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, config.models.embedding_model))[0]
        vectors = index.embed_graph_nodes(memory_id, facts)
        added = max(0, len(vectors) - int(before))
        rows.append({"memory_id": memory_id, "facts": len(facts),
                     "vectors": len(vectors), "added": added})
        print(f"{ordinal}/{len(memory_ids)} {memory_id}: "
              f"facts={len(facts)} added={added}", flush=True)
    usage = target._read_one(
        "SELECT count(*),coalesce(sum(input_tokens),0),coalesce(sum(latency_ms),0) "
        "FROM embedding_calls WHERE model_id=?",
        (config.models.embedding_model,))
    manifest = {
        "schema_version": "graphmem-v5.13-atomic-summary-embeddings-v1",
        "source_db": str(args.source_db), "output_db": str(args.output_db),
        "model": config.models.embedding_model, "batch_size": args.batch_size,
        "memories": len(rows), "facts": sum(row["facts"] for row in rows),
        "vectors": sum(row["vectors"] for row in rows),
        "embedding_calls": int(usage[0]), "embedding_input_tokens": int(usage[1]),
        "embedding_latency_ms": float(usage[2]),
        "wall_seconds": time.perf_counter() - started, "rows": rows,
    }
    manifest_path = args.output_db.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "output_db", "memories", "facts", "vectors", "embedding_calls",
        "embedding_input_tokens", "wall_seconds")}, ensure_ascii=False, indent=2))
    target.close(); source.close()


if __name__ == "__main__":
    main()
