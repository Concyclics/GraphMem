#!/usr/bin/env python3
"""Compare two V4 retrieval policies with one shared query embedding."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import EmbeddingClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.pipeline import _load_memory_cache, _safe_cache_part  # noqa: E402
from graphmem_demo.v36.build import clone_index  # noqa: E402
from graphmem_demo.v4 import build_capability_view, build_query_ir, query_views, retrieve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--memory-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--token-budget", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_longmemeval_cases(args.data, question_type="all")
    embedder = EmbeddingClient(args.embedding_base_url, args.embedding_model)
    loaded: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        cache_key = _safe_cache_part(case.memory_cache_key or case.question_id)
        matches = sorted(args.memory_cache_dir.glob(f"{cache_key}-*.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one cache for {cache_key}, got {matches}")
        cache_path = str(matches[0])
        memory = loaded.get(cache_path)
        if memory is None:
            memory = _load_memory_cache(matches[0])
            if memory is None or memory.v36_index is None:
                raise RuntimeError(f"invalid V4 cache: {matches[0]}")
            loaded[cache_path] = memory
        ir = build_query_ir(case.question)
        vectors = embedder.embed(
            query_views(ir), question_id=case.question_id, variant="paired_v4_ablation"
        )
        index_value = clone_index(memory.v36_index, case.question_id)
        capability_view = build_capability_view(index_value)
        os.environ.pop("GRAPHMEM_V36_DIALOGUE_CLOSURE", None)
        baseline = retrieve(
            case=case, variant="paired_v4_baseline", index=index_value,
            capability_view=capability_view, query_vectors=vectors,
            token_budget=args.token_budget,
        )
        os.environ["GRAPHMEM_V36_DIALOGUE_CLOSURE"] = "1"
        repaired = retrieve(
            case=case, variant="paired_v4_dialogue_closure", index=index_value,
            capability_view=capability_view, query_vectors=vectors,
            token_budget=args.token_budget,
        )
        os.environ.pop("GRAPHMEM_V36_DIALOGUE_CLOSURE", None)
        rows.append({
            "question_id": case.question_id,
            "question_type": case.question_type,
            "baseline": asdict(baseline),
            "repaired": asdict(repaired),
        })
        print(f"[{index}/{len(cases)}] {case.question_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
