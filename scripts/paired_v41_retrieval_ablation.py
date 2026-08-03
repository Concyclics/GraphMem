#!/usr/bin/env python3
"""Paired V4.0/V4.1 retrieval A/B using exactly one shared query embedding."""
from __future__ import annotations

import argparse
import json
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
from graphmem_demo.v4 import (  # noqa: E402
    build_capability_view, build_query_ir, query_views as v4_query_views,
    retrieve as retrieve_v4,
)
from graphmem_demo.v41 import (  # noqa: E402
    QueryPolicyV41, build_sidecar, retrieve as retrieve_v41,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--memory-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--token-budget", type=int, default=9200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_longmemeval_cases(args.data, question_type="all")
    completed: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            completed = {
                json.loads(line)["question_id"]
                for line in handle if line.strip()
            }
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    embedder = EmbeddingClient(args.embedding_base_url, args.embedding_model)
    loaded: dict[str, object] = {}
    for position, case in enumerate(cases, 1):
        if case.question_id in completed:
            continue
        cache_key = _safe_cache_part(case.memory_cache_key or case.question_id)
        matches = sorted(args.memory_cache_dir.glob(f"{cache_key}-*.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one cache for {cache_key}, got {len(matches)}"
            )
        cache_path = str(matches[0])
        memory = loaded.get(cache_path)
        if memory is None:
            memory = _load_memory_cache(matches[0])
            if memory is None or memory.v36_index is None:
                raise RuntimeError(f"invalid V4 cache: {matches[0]}")
            loaded[cache_path] = memory
        index = clone_index(memory.v36_index, case.question_id)
        view = build_capability_view(index)
        ir = build_query_ir(case.question)
        # The same vectors are passed to both policies. V4.1 domain hints act
        # only through its sidecar channels in this deterministic ablation.
        vectors = embedder.embed(
            v4_query_views(ir), question_id=case.question_id,
            variant="paired_v41_ablation",
        )
        baseline = retrieve_v4(
            case=case, variant="paired_v4_0", index=index,
            capability_view=view, query_vectors=vectors,
            token_budget=min(8400, args.token_budget),
        )
        repaired = retrieve_v41(
            case=case, variant="paired_v4_1", index=index,
            capability_view=view, sidecar=build_sidecar(index),
            query_ir=ir, query_vectors=vectors,
            token_budget=args.token_budget, policy=QueryPolicyV41(
                complex_context_target=args.token_budget,
            ),
        )
        row = {
            "question_id": case.question_id,
            "question_type": case.question_type,
            "baseline": asdict(baseline),
            "repaired": asdict(repaired),
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(cases)}] {case.question_id}", flush=True)


if __name__ == "__main__":
    main()
