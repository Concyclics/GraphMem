#!/usr/bin/env python3
"""Create a size-bounded Gate A recall-failure research archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MAX_ARCHIVE_BYTES = 30 * 1024 * 1024


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def node_ids(value: Any, prefix: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        if value.startswith(prefix + ":") and any(token in value for token in (":turn:", ":frame:", ":group:", ":card")):
            found.add(value)
    elif isinstance(value, list):
        for item in value:
            found.update(node_ids(item, prefix))
    elif isinstance(value, dict):
        for item in value.values():
            found.update(node_ids(item, prefix))
    return found


def rebase(node_id: str, question_id: str, builder_id: str) -> str:
    return builder_id + node_id[len(question_id):] if node_id.startswith(question_id + ":") else node_id


def graph_node_id(row: dict[str, Any]) -> str:
    for field in ("node_id", "frame_id", "card_id", "group_id"):
        if row.get(field):
            return str(row[field])
    raise KeyError(f"Graph node has no canonical ID field: {sorted(row)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lme-data", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    research = args.run_root / "research"
    full = args.run_root / "full_fidelity"
    failures = read_jsonl(research / "turn_failure_cases.jsonl")
    failed_qids = {row["question_id"] for row in failures}
    failed_memories = {row["memory_id"] for row in failures}
    failed_builders = {row["graph_builder_question_id"] for row in failures}

    retrieval: dict[str, dict[str, Any]] = {}
    graph_roots = {
        "longmemeval": args.run_root / "lme/merged/hierarchical_hybrid_graph_v4_1_query",
        "locomo": args.run_root / "locomo/merged/hierarchical_hybrid_graph_v4_1_query",
    }
    for root in graph_roots.values():
        for row in read_jsonl(root / "retrieval_results.jsonl"):
            if row["question_id"] in failed_qids:
                retrieval[row["question_id"]] = row

    full_calls = read_jsonl(full / "llm_calls_full.jsonl.gz")
    selected_calls = [
        row for row in full_calls
        if row["question_id"] in failed_qids or row.get("memory_id") in failed_memories
    ]
    write_jsonl_gz(args.staging_dir / "calls/failed_question_and_memory_llm_calls.jsonl.gz", selected_calls)
    write_jsonl_gz(args.staging_dir / "retrieval/failed_retrieval_results.jsonl.gz", retrieval.values())
    write_jsonl_gz(args.staging_dir / "retrieval/failure_cases.jsonl.gz", failures)

    data_rows = {
        row["question_id"]: row
        for path in (args.lme_data, args.locomo_data)
        for row in json.loads(path.read_text(encoding="utf-8"))
    }
    question_samples = []
    for qid in sorted(failed_qids):
        row = data_rows[qid]
        question_samples.append({
            key: value for key, value in row.items()
            if key not in {"haystack_sessions"}
        })
    write_jsonl_gz(args.staging_dir / "samples/failure_questions_and_gold.jsonl.gz", question_samples)

    annotations = [
        row for row in read_jsonl(args.repo_root / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
        if row["question_id"] in failed_qids
    ]
    write_jsonl_gz(args.staging_dir / "samples/lme_failure_gold_turn_annotations.jsonl.gz", annotations)

    seeds_by_builder: dict[str, set[str]] = defaultdict(set)
    seed_rows = []
    for row in failures:
        qid = row["question_id"]
        builder = row["graph_builder_question_id"]
        result = retrieval[qid]
        ids = set()
        for field in (
            "gold_turn_ids", "packed_turn_ids", "candidate_gold_turn_ids",
            "missing_gold_turns", "candidate_but_dropped_gold_turns",
        ):
            ids.update(row.get(field) or [])
        ids.update(node_ids(result.get("retrieval_trace") or {}, qid))
        rebased = {rebase(item, qid, builder) for item in ids}
        seeds_by_builder[builder].update(rebased)
        seed_rows.append({
            "question_id": qid,
            "memory_id": row["memory_id"],
            "builder_question_id": builder,
            "failure_stage": row["failure_stage"],
            "original_seed_ids": sorted(ids),
            "builder_namespace_seed_ids": sorted(rebased),
        })
    write_jsonl_gz(args.staging_dir / "graphs/subgraph_seed_index.jsonl.gz", seed_rows)

    all_nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    all_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_state_chains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in graph_roots.values():
        for row in read_jsonl(root / "nodes.jsonl"):
            builder = row["question_id"]
            if builder in failed_builders:
                all_nodes[builder][graph_node_id(row)] = row
        for row in read_jsonl(root / "edges.jsonl"):
            if row["question_id"] in failed_builders:
                all_edges[row["question_id"]].append(row)
        state_path = root / "state_chains.jsonl"
        if state_path.exists():
            for row in read_jsonl(state_path):
                if row["question_id"] in failed_builders:
                    all_state_chains[row["question_id"]].append(row)

    subgraph_nodes = []
    subgraph_edges = []
    subgraph_states = []
    subgraph_stats = []
    for builder in sorted(failed_builders):
        nodes = all_nodes[builder]
        edges = all_edges[builder]
        selected = set(seeds_by_builder[builder])
        selected.update(
            node_id for node_id, node in nodes.items()
            if node.get("node_type") == "routing_card"
        )
        for _ in range(2):
            additions = set()
            for edge in edges:
                relation = edge.get("relation")
                if relation == "routing_contains" and edge.get("dst") not in selected:
                    continue
                if edge.get("src") in selected or edge.get("dst") in selected:
                    additions.update((edge.get("src"), edge.get("dst")))
            selected.update(item for item in additions if item)
        selected_edges = [
            edge for edge in edges
            if edge.get("src") in selected and edge.get("dst") in selected
        ]
        selected_nodes = [nodes[node_id] for node_id in sorted(selected) if node_id in nodes]
        memory_id = next(row["memory_id"] for row in failures if row["graph_builder_question_id"] == builder)
        subgraph_nodes.extend({"memory_id": memory_id, **row} for row in selected_nodes)
        subgraph_edges.extend({"memory_id": memory_id, **row} for row in selected_edges)
        relevant_states = [
            row for row in all_state_chains[builder]
            if node_ids(row, builder) & selected
        ]
        subgraph_states.extend({"memory_id": memory_id, **row} for row in relevant_states)
        subgraph_stats.append({
            "memory_id": memory_id,
            "builder_question_id": builder,
            "seed_nodes": len(seeds_by_builder[builder]),
            "selected_nodes": len(selected_nodes),
            "selected_edges": len(selected_edges),
            "selected_state_chains": len(relevant_states),
            "full_nodes": len(nodes),
            "full_edges": len(edges),
        })
    write_jsonl_gz(args.staging_dir / "graphs/evidence_subgraph_nodes.jsonl.gz", subgraph_nodes)
    write_jsonl_gz(args.staging_dir / "graphs/evidence_subgraph_edges.jsonl.gz", subgraph_edges)
    write_jsonl_gz(args.staging_dir / "graphs/evidence_subgraph_state_chains.jsonl.gz", subgraph_states)
    write_jsonl_gz(args.staging_dir / "graphs/evidence_subgraph_stats.jsonl.gz", subgraph_stats)

    copy_files = {
        research / "research_summary.json": "research/research_summary.json",
        research / "token_breakdown.csv": "research/token_breakdown.csv",
        research / "feature_trace_contribution.csv": "research/feature_trace_contribution.csv",
        research / "retrieval_channel_gold_coverage.csv": "research/retrieval_channel_gold_coverage.csv",
        research / "graph_expansion_relation_contribution.csv": "research/graph_expansion_relation_contribution.csv",
        research / "turn_reference_metrics.csv": "research/turn_reference_metrics.csv",
        research / "graphmem_v5_gate_a_research_report.html": "research/graphmem_v5_gate_a_research_report.html",
        args.repo_root / "docs/V5_GATE_A_TOKEN_RECALL_RESEARCH_REPORT.md": "research/V5_GATE_A_TOKEN_RECALL_RESEARCH_REPORT.md",
        args.repo_root / "V5_AUDIT.md": "research/V5_AUDIT.md",
        args.repo_root / "configs/v5/gate_a.json": "config/gate_a.json",
        args.repo_root / "src/graphmem_demo/v36/build.py": "code/v36_build.py",
        args.repo_root / "src/graphmem_demo/v36/runtime.py": "code/v36_runtime.py",
        args.repo_root / "src/graphmem_demo/v41/retrieval.py": "code/v41_retrieval.py",
        args.repo_root / "scripts/build_v5_gate_a_research_bundle.py": "code/build_research_bundle.py",
        args.repo_root / "scripts/build_v5_gate_a_full_fidelity_bundle.py": "code/build_full_fidelity_bundle.py",
        args.repo_root / "scripts/package_v5_gate_a_failure_research.py": "code/package_failure_research.py",
    }
    for source, relative in copy_files.items():
        if source.exists():
            target = args.staging_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    readme = f"""# GraphMem Gate A recall-failure research package

This bounded package contains all **{len(failures)} exact-turn all-hit failures**
from the frozen 200-question run, covering **{len(failed_memories)} memories**.

## Suggested analysis order

1. `research/V5_GATE_A_TOKEN_RECALL_RESEARCH_REPORT.md`
2. `retrieval/failure_cases.jsonl.gz`
3. `retrieval/failed_retrieval_results.jsonl.gz`
4. `graphs/subgraph_seed_index.jsonl.gz`
5. `graphs/evidence_subgraph_*.jsonl.gz`
6. `calls/failed_question_and_memory_llm_calls.jsonl.gz`

The graph export is an evidence-centered two-hop subgraph, not the complete
110-memory graph collection. It includes all failure trace node IDs, gold,
candidate, missing and packed turns, all routing cards for affected memories,
and two-hop provenance/relation closure. Full graphs and embedding binaries are
intentionally excluded to keep the archive below 30 MiB.

Build-call prompts and outputs are exact when `raw_output_available=true`.
Legacy planner raw responses were not persisted; planner rows retain reconstructed
input and parsed/post-filtered output with explicit fidelity markers.

This archive contains benchmark text, answers, gold annotations and model
responses. Treat it as restricted offline research data; never expose it to
online build/query code.
"""
    (args.staging_dir / "README.md").write_text(readme, encoding="utf-8")

    packaged_files = sorted(path for path in args.staging_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "graphmem-v5-gate-a-failure-research-v1",
        "scope": {
            "failed_questions": len(failures),
            "failed_memories": len(failed_memories),
            "selected_llm_calls": len(selected_calls),
            "subgraph_nodes": len(subgraph_nodes),
            "subgraph_edges": len(subgraph_edges),
            "subgraph_state_chains": len(subgraph_states),
        },
        "size_limit_bytes": MAX_ARCHIVE_BYTES,
        "files": {
            str(path.relative_to(args.staging_dir)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in packaged_files
            if path.name != "manifest.json"
        },
    }
    (args.staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", compresslevel=9) as archive:
        archive.add(args.staging_dir, arcname="graphmem_gate_a_failure_research")
    if args.archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Archive is {args.archive.stat().st_size} bytes, above {MAX_ARCHIVE_BYTES}"
        )


if __name__ == "__main__":
    main()
