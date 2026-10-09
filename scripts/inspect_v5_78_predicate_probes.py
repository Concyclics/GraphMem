#!/usr/bin/env python3
"""Inspect query-to-build-predicate semantic matches for selected questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config
from graphmem.domain import stable_id
from graphmem.embedding import QwenEmbeddingIndex
from graphmem.runtime import GraphReadView
from graphmem.retrieval.query_ir import compile_query
from graphmem.text import content_terms
from graphmem.storage import SQLiteGraphStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--question-id", action="append", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    args = parser.parse_args()

    wanted = set(args.question_id)
    questions = [row for row in json.loads(args.data.read_text(
        encoding="utf-8")) if str(row["question_id"]) in wanted]
    store = SQLiteGraphStore(args.db, read_only=True)
    index = QwenEmbeddingIndex(
        store, load_config(args.config), record_usage=False,
        model_id="BAAI/bge-m3", request_model_id="bge-m3",
        base_url=args.base_url, query_cache_path=args.query_cache)
    for question in questions:
        memory_id = "locomo:" + str(question["locomo_sample_id"])
        version, checksum = store.graph_identity(memory_id)
        view = GraphReadView(
            store.nodes(memory_id), store.edges(memory_id),
            graph_version=version, graph_checksum=checksum)
        by_id = {
            stable_id("predicate", memory_id, predicate): predicate
            for predicate in view.predicate_index}
        query = str(question["question"])
        ir = compile_query(query, view)
        query_terms = content_terms(query)
        owner_terms = frozenset(
            term for turn in store.turns(memory_id)
            if content_terms(turn.speaker) <= query_terms
            for term in content_terms(turn.speaker))
        relation_terms = tuple(
            term for term in (ir.slots.content_terms if ir.slots else ())
            if term not in owner_terms)
        probe_query = " ".join(relation_terms) if relation_terms else query
        response = index.client.embeddings.create(
            model="bge-m3", input=[probe_query])
        query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        for db_row in store._read(
                "SELECT item_id,dimension,vector FROM embeddings "
                "WHERE memory_id=? AND model_id=? ORDER BY item_id",
                (memory_id, "BAAI/bge-m3:predicate-v1")):
            item_id = str(db_row["item_id"])
            if item_id not in by_id:
                continue
            ids.append(item_id)
            vector = np.frombuffer(
                db_row["vector"], dtype=np.float32,
                count=int(db_row["dimension"])).copy()
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            vectors.append(vector)
        scores = np.stack(vectors) @ query_vector
        indices = sorted(range(len(ids)), key=lambda value: (
            -float(scores[value]), ids[value]))[:args.limit]
        rows = [(ids[value], float(scores[value])) for value in indices]
        print(json.dumps({
            "question_id": question["question_id"],
            "question": question["question"],
            "probe_query": probe_query,
            "answer": question.get("answer"),
            "probes": [{"predicate": by_id[item_id], "score": score}
                       for item_id, score in rows],
        }, ensure_ascii=False))
    store.close()


if __name__ == "__main__":
    main()
