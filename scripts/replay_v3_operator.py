#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import EmbeddingClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.pipeline import _load_memory_cache  # noqa: E402
from graphmem_demo.v3.query_planning import query_views  # noqa: E402
from graphmem_demo.v3.retrieval import answer_messages, build_query_frame, retrieve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay V3 retrieval/operator logic from a saved memory cache."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check-answer-pack", action="store_true")
    parser.add_argument("--token-budget", type=int, default=3600)
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
    )
    args = parser.parse_args()

    cases = {
        case.question_id: case
        for case in load_longmemeval_cases(args.data, "all", 100000)
    }
    embedder = EmbeddingClient(args.embedding_base_url, args.embedding_model)
    question_ids = list(cases) if args.all else list(args.question_id or [])
    if not question_ids:
        parser.error("provide --question-id or --all")
    for question_id in question_ids:
        case = cases[question_id]
        matches = sorted(
            args.cache_dir.glob(f"longmemeval_{question_id}-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise RuntimeError(f"no cache for {question_id}")
        memory = _load_memory_cache(matches[0])
        if memory is None or memory.v3_index is None:
            raise RuntimeError(f"invalid V3 cache: {matches[0]}")
        frame = build_query_frame(case.question)
        vectors = embedder.embed(
            query_views(frame), question_id=question_id, variant="v3_operator_replay"
        )
        result = retrieve(
            case=case,
            variant="hierarchical_hypergraph_v3",
            index=memory.v3_index,
            query_vector=vectors[0],
            query_vectors=vectors,
            token_budget=args.token_budget,
        )
        trace = result.retrieval_trace
        pack_error = None
        if args.check_answer_pack:
            try:
                answer_messages(case, result)
            except ValueError as exc:
                pack_error = str(exc)
        print(json.dumps({
            "question_id": question_id,
            "question": case.question,
            "query_frame": trace.get("query_frame"),
            "catalog_operator_hint": trace.get("catalog_operator_hint"),
            "duration_hint": trace.get("duration_hint"),
            "before_after_relation_hint": trace.get("before_after_relation_hint"),
            "relative_time_hint": trace.get("relative_time_hint"),
            "closure_certificate": trace.get("closure_certificate"),
            "answer_prepack": trace.get("answer_prepack"),
            "answer_pack_error": pack_error,
            "packed_rough_tokens": result.packed_rough_tokens,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
