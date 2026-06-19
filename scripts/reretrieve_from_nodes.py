#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import EmbeddingClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases, speaker_retrieval_text_from_raw  # noqa: E402
from graphmem_demo.models import GraphEdge, LeafNode, SummaryNode  # noqa: E402
from graphmem_demo.pipeline import (  # noqa: E402
    DemoConfig,
    _build_root_graph,
    _embed_nodes,
    _leaf_embedding_attr,
    _effective_leaf_text_mode,
    _retrieve,
    _summary_anchor_terms,
    _summary_retrieval_text,
    _variant_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run retrieval from saved GraphMem nodes without rebuilding summaries."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="single_llm_summary_graphmem")
    parser.add_argument("--question-type", default="all")
    parser.add_argument("--max-questions", type=int, default=60)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--leaf-top-k", type=int, default=20)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--root-candidate-k", type=int, default=12)
    parser.add_argument("--global-leaf-top-k", type=int, default=40)
    parser.add_argument("--qa-summary-top-k", type=int, default=4)
    parser.add_argument("--per-session-leaf-k", type=int, default=4)
    parser.add_argument("--graph-neighbor-k", type=int, default=2)
    parser.add_argument("--qa-context-token-budget", type=int, default=10000)
    parser.add_argument(
        "--rebuild-edges",
        action="store_true",
        help="Recompute root graph edges from loaded summary retrieval text and embeddings.",
    )
    parser.add_argument("--enable-typed-root-edges", action="store_true")
    parser.add_argument("--enable-multilevel-summary-retrieval", action="store_true")
    parser.add_argument("--enable-speaker-neighbor-window", action="store_true")
    parser.add_argument("--enable-speaker-retrieval-text", action="store_true")
    parser.add_argument(
        "--retrieval-leaf-text",
        choices=["auto", "raw", "user_only"],
        default="auto",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_nodes(path: Path) -> tuple[dict[str, list[LeafNode]], dict[str, list[SummaryNode]]]:
    leaves_by_qid: dict[str, list[LeafNode]] = {}
    summaries_by_qid: dict[str, list[SummaryNode]] = {}
    for row in read_jsonl(path):
        node_type = row.pop("node_type", "")
        row.pop("embedding", None)
        if node_type == "leaf":
            leaf = LeafNode(**row)
            leaf.retrieval_text = leaf.retrieval_text or speaker_retrieval_text_from_raw(
                leaf.raw_text
            )
            leaves_by_qid.setdefault(leaf.question_id, []).append(leaf)
        elif node_type == "summary":
            summary = SummaryNode(**row)
            summary.anchor_terms = summary.anchor_terms or _summary_anchor_terms(
                summary.parsed_summary,
                summary.raw_summary_text or summary.summary,
                summary.session_date,
            )
            summary.retrieval_text = summary.retrieval_text or _summary_retrieval_text(
                summary.summary,
                summary.parsed_summary,
                summary.raw_summary_text or summary.summary,
                summary.session_date,
            )
            summaries_by_qid.setdefault(summary.question_id, []).append(summary)
    return leaves_by_qid, summaries_by_qid


def load_edges(path: Path) -> dict[str, list[GraphEdge]]:
    edges_by_qid: dict[str, list[GraphEdge]] = {}
    for row in read_jsonl(path):
        edge = GraphEdge(**row)
        qid = edge.src.split(":", 1)[0]
        edges_by_qid.setdefault(qid, []).append(edge)
    return edges_by_qid


def infer_roots(summaries: list[SummaryNode]) -> list[SummaryNode]:
    if not summaries:
        return []
    child_ids = {
        child_id
        for summary in summaries
        for child_id in summary.child_ids
    }
    roots = [summary for summary in summaries if summary.node_id not in child_ids]
    return roots or summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = args.output_dir / "retrieval_results.jsonl"
    embedding_path = args.output_dir / "embedding_calls.jsonl"
    retrieval_path.write_text("", encoding="utf-8")
    embedding_path.write_text("", encoding="utf-8")

    config = DemoConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        question_type=args.question_type,
        variants=(args.variant,),
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        leaf_top_k=args.leaf_top_k,
        root_top_k=args.root_top_k,
        root_candidate_k=args.root_candidate_k,
        global_leaf_top_k=args.global_leaf_top_k,
        qa_summary_top_k=args.qa_summary_top_k,
        per_session_leaf_k=args.per_session_leaf_k,
        graph_neighbor_k=args.graph_neighbor_k,
        qa_context_token_budget=args.qa_context_token_budget,
        retrieval_leaf_text=args.retrieval_leaf_text,
        enable_speaker_neighbor_window=args.enable_speaker_neighbor_window,
        enable_speaker_retrieval_text=args.enable_speaker_retrieval_text,
        enable_typed_root_edges=args.enable_typed_root_edges,
        enable_multilevel_summary_retrieval=args.enable_multilevel_summary_retrieval,
    )
    spec = _variant_spec(config, args.variant)
    cases = load_longmemeval_cases(args.data, args.question_type, args.max_questions)
    leaves_by_qid, summaries_by_qid = load_nodes(args.nodes)
    edges_by_qid = load_edges(args.edges)
    embedder = EmbeddingClient(args.embedding_base_url, args.embedding_model)

    for case in cases:
        leaves = leaves_by_qid.get(case.question_id, [])
        summaries = summaries_by_qid.get(case.question_id, [])
        roots = infer_roots(summaries)
        retrieval_roots = summaries if args.enable_multilevel_summary_retrieval else roots
        if not leaves:
            raise RuntimeError(f"No leaves found for {case.question_id}")
        retrieval_leaf_mode = _effective_leaf_text_mode(
            config.retrieval_leaf_text, spec, case, phase="retrieval"
        )
        embedding_start = len(embedder.records)
        _embed_nodes(
            leaves,
            embedder,
            case.question_id,
            args.variant,
            attr=_leaf_embedding_attr(config, case, retrieval_leaf_mode),
        )
        _embed_nodes(summaries, embedder, case.question_id, args.variant, attr="retrieval_text")
        edges = (
            _build_root_graph(
                retrieval_roots,
                args.graph_neighbor_k,
                enable_typed_edges=args.enable_typed_root_edges,
            )
            if args.rebuild_edges
            else edges_by_qid.get(case.question_id, [])
        )
        retrieval = _retrieve(
            config=config,
            case=case,
            variant=args.variant,
            leaves=leaves,
            roots=retrieval_roots,
            edges=edges,
            embedder=embedder,
            graph_enabled=spec.graph,
            hybrid_retrieval=spec.hybrid_retrieval,
            enhanced_retrieval=spec.enhanced_retrieval,
            enhanced_qa=spec.enhanced_qa,
        )
        append_jsonl(retrieval_path, [asdict(retrieval)])
        append_jsonl(embedding_path, [asdict(row) for row in embedder.records[embedding_start:]])
        print(
            f"{args.variant}: question={case.question_id} "
            f"leaves={len(retrieval.leaf_node_ids)} recall={retrieval.answer_session_recall:.3f}",
            flush=True,
        )
    print(f"retrieval_results={retrieval_path}")
    print(f"embedding_calls={embedding_path}")


if __name__ == "__main__":
    main()
