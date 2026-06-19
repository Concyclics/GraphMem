#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.pipeline import DemoConfig, run_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GraphMem DeepSeek token demo.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--question-type", default="multi-session")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "direct_session_k16_compact_no_compress",
            "direct_session_k16_compact_graphmem",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepseek-model")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--tree-mode", choices=["legacy_kway", "direct_session"])
    parser.add_argument("--fanout-k", type=int, default=16)
    parser.add_argument("--max-group-rough-tokens", type=int, default=6000)
    parser.add_argument("--leaf-top-k", type=int, default=14)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--root-candidate-k", type=int, default=8)
    parser.add_argument("--global-leaf-top-k", type=int, default=24)
    parser.add_argument("--qa-summary-top-k", type=int, default=4)
    parser.add_argument("--per-session-leaf-k", type=int, default=2)
    parser.add_argument("--graph-neighbor-k", type=int, default=2)
    parser.add_argument("--qa-context-token-budget", type=int, default=10000)
    parser.add_argument("--qa-max-tokens", type=int, default=1024)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--question-workers", type=int, default=2)
    parser.add_argument("--summary-workers", type=int, default=0)
    parser.add_argument("--max-inflight-deepseek", type=int, default=0)
    parser.add_argument(
        "--summary-schema",
        choices=["minimal_memory_v1", "compact_memory_v2", "multilingual_memory_v1"],
    )
    parser.add_argument(
        "--summarizer-kind",
        choices=["auto", "none", "llmlingua2", "qwen_local"],
        default="auto",
    )
    parser.add_argument("--summarizer-base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--summarizer-model")
    parser.add_argument("--summary-token-budget", type=int, default=320)
    parser.add_argument("--build-leaf-text", choices=["auto", "raw", "user_only"], default="auto")
    parser.add_argument(
        "--retrieval-leaf-text",
        choices=["auto", "raw", "user_only"],
        default="auto",
    )
    parser.add_argument("--compressor-chunk-rough-tokens", type=int, default=384)
    parser.add_argument("--raw-group-summary-max-tokens", type=int, default=256)
    parser.add_argument("--session-summary-max-tokens", type=int, default=320)
    parser.add_argument("--legacy-internal-summary-max-tokens", type=int, default=224)
    parser.add_argument("--llmlingua-model")
    parser.add_argument("--llmlingua-device-map")
    parser.add_argument("--use-llmlingua2", action="store_true")
    parser.add_argument("--enable-speaker-profiles", action="store_true")
    parser.add_argument("--enable-speaker-neighbor-window", action="store_true")
    parser.add_argument("--enable-speaker-retrieval-text", action="store_true")
    parser.add_argument("--enable-typed-root-edges", action="store_true")
    parser.add_argument("--enable-multilevel-summary-retrieval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mock-services",
        action="store_true",
        help="Use deterministic local mock LLM, embedding, and compressor implementations.",
    )
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-embedding", action="store_true")
    parser.add_argument("--mock-compressor", action="store_true")
    parser.add_argument("--mock-summarizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        question_type=args.question_type,
        variants=tuple(args.variants),
        deepseek_model=args.deepseek_model,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        tree_mode=args.tree_mode,
        fanout_k=args.fanout_k,
        max_group_rough_tokens=args.max_group_rough_tokens,
        leaf_top_k=args.leaf_top_k,
        root_top_k=args.root_top_k,
        root_candidate_k=args.root_candidate_k,
        global_leaf_top_k=args.global_leaf_top_k,
        qa_summary_top_k=args.qa_summary_top_k,
        per_session_leaf_k=args.per_session_leaf_k,
        graph_neighbor_k=args.graph_neighbor_k,
        qa_context_token_budget=args.qa_context_token_budget,
        qa_max_tokens=args.qa_max_tokens,
        compression_ratio=args.compression_ratio,
        max_questions=args.max_questions,
        question_workers=args.question_workers,
        summary_workers=args.summary_workers,
        max_inflight_deepseek=args.max_inflight_deepseek,
        summary_schema=args.summary_schema,
        summarizer_kind=args.summarizer_kind,
        summarizer_base_url=args.summarizer_base_url,
        summarizer_model=args.summarizer_model,
        summary_token_budget=args.summary_token_budget,
        build_leaf_text=args.build_leaf_text,
        retrieval_leaf_text=args.retrieval_leaf_text,
        compressor_chunk_rough_tokens=args.compressor_chunk_rough_tokens,
        raw_group_summary_max_tokens=args.raw_group_summary_max_tokens,
        session_summary_max_tokens=args.session_summary_max_tokens,
        legacy_internal_summary_max_tokens=args.legacy_internal_summary_max_tokens,
        resume=args.resume,
        mock_services=args.mock_services,
        mock_llm=args.mock_llm,
        mock_embedding=args.mock_embedding,
        mock_compressor=args.mock_compressor,
        mock_summarizer=args.mock_summarizer,
        llmlingua_model=args.llmlingua_model,
        llmlingua_device_map=args.llmlingua_device_map,
        use_llmlingua2=args.use_llmlingua2,
        enable_speaker_profiles=args.enable_speaker_profiles,
        enable_speaker_neighbor_window=args.enable_speaker_neighbor_window,
        enable_speaker_retrieval_text=args.enable_speaker_retrieval_text,
        enable_typed_root_edges=args.enable_typed_root_edges,
        enable_multilevel_summary_retrieval=args.enable_multilevel_summary_retrieval,
    )
    aggregates = run_demo(config)
    for aggregate in aggregates:
        print(
            f"{aggregate.variant}: questions={aggregate.question_count} "
            f"deepseek_tokens={aggregate.total_deepseek_tokens} "
            f"calls={aggregate.deepseek_call_count}"
        )
    print(f"summary={config.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
