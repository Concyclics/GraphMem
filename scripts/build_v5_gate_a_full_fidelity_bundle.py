#!/usr/bin/env python3
"""Export full-fidelity Gate A calls and graph contents for offline research.

Build-call responses are recovered from immutable V3.6 checkpoints. Their
inputs are reconstructed from the frozen canonical SourceTurns and prompt code,
then verified against each checkpoint signature. Query-planner raw responses
were not persisted by the legacy runner; those rows include reconstructed input
and the final parsed/filtered planner result with an explicit fidelity marker.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from graphmem_demo.v36.build import session_extraction_messages
from graphmem_demo.v36.runtime import (
    _candidate_pairs,
    _checkpoint_signature,
    _consolidation_messages,
)
from graphmem_demo.v36.schema import QuantityValue, RoleFrameNode, TemporalValue, TurnNodeV36
from graphmem_demo.v41.retrieval import planner_messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def turn_from_dict(row: dict[str, Any]) -> TurnNodeV36:
    allowed = set(TurnNodeV36.__dataclass_fields__)
    values = {key: value for key, value in row.items() if key in allowed}
    values["embedding"] = None
    return TurnNodeV36(**values)


def frame_from_dict(row: dict[str, Any]) -> RoleFrameNode:
    allowed = set(RoleFrameNode.__dataclass_fields__)
    values = {key: value for key, value in row.items() if key in allowed}
    values["quantity"] = QuantityValue(**(values.get("quantity") or {}))
    values["temporal"] = TemporalValue(**(values.get("temporal") or {}))
    values["embedding"] = None
    return RoleFrameNode(**values)


def load_checkpoint_map(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cache_name in ("memory_cache_lme", "memory_cache_locomo"):
        checkpoint_root = root / cache_name / ".v36_call_checkpoints"
        for path in checkpoint_root.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            call_id = (payload.get("record") or {}).get("call_id")
            if call_id:
                payload["_checkpoint_path"] = str(path)
                result[call_id] = payload
    return result


def cache_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for benchmark, cache_name in (("longmemeval", "memory_cache_lme"), ("locomo", "memory_cache_locomo")):
        directory = root / cache_name / "hierarchical_hybrid_graph_v4_0"
        files.extend((benchmark, path) for path in sorted(directory.glob("*.json")))
    return files


def graph_rows_by_builder(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]].append(row)
    return rows


def gzip_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, gzip.open(target, "wb", compresslevel=6) as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--lme-data", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    question_rows = read_jsonl(args.research_dir / "question_research_log.jsonl")
    question_meta = {row["question_id"]: row for row in question_rows}
    builder_to_memory = {
        row["graph_builder_question_id"]: row["memory_id"] for row in question_rows
    }
    data_rows = {
        row["question_id"]: row
        for path in (args.lme_data, args.locomo_data)
        for row in json.loads(path.read_text(encoding="utf-8"))
    }

    calls: list[dict[str, Any]] = []
    retrieval: dict[str, dict[str, Any]] = {}
    benchmark_roots = {
        "longmemeval": args.run_root / "lme/merged/hierarchical_hybrid_graph_v4_1_query",
        "locomo": args.run_root / "locomo/merged/hierarchical_hybrid_graph_v4_1_query",
    }
    for benchmark, root in benchmark_roots.items():
        for row in read_jsonl(root / "llm_calls.jsonl"):
            row["benchmark"] = benchmark
            calls.append(row)
        for row in read_jsonl(root / "retrieval_results.jsonl"):
            retrieval[row["question_id"]] = row
    calls_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        calls_by_qid[row["question_id"]].append(row)
    checkpoints = load_checkpoint_map(args.run_root)

    reconstructed_build: dict[str, dict[str, Any]] = {}
    canonical_cache_by_builder: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    for benchmark, path in cache_files(args.run_root):
        cache = json.loads(path.read_text(encoding="utf-8"))
        builder_qid = cache["source_question_id"]
        canonical_cache_by_builder[builder_qid] = (benchmark, path, cache)
        index = cache["v36_index"]
        turns = [turn_from_dict(row) for row in index["turns"]]
        frames = [frame_from_dict(row) for row in index["frames"]]
        sessions: dict[str, list[TurnNodeV36]] = defaultdict(list)
        for turn in turns:
            sessions[turn.session_id].append(turn)
        for values in sessions.values():
            values.sort(key=lambda item: item.turn_index)

        build_calls = [row for row in calls_by_qid[builder_qid] if row["stage"].startswith("build_")]
        session_caps = {
            int(row["max_tokens"]) for row in build_calls
            if row["stage"] == "build_v36_session" and row.get("max_tokens") is not None
        }
        signatures: dict[str, tuple[str, list[dict[str, str]], int]] = {}
        for session_id, session_turns in sessions.items():
            base = session_extraction_messages(
                session_id, session_turns[0].session_date if session_turns else None, session_turns
            )
            for cap in session_caps:
                messages = [dict(item) for item in base]
                messages[0]["content"] += (
                    f" Output JSON budget: at most {cap} tokens. "
                    "Use compact strings; cover durable facts before optional detail."
                )
                signatures[_checkpoint_signature(messages, cap)] = (session_id, messages, cap)

        identity_pairs = _candidate_pairs(frames)
        identity_messages = _consolidation_messages(frames, identity_pairs) if identity_pairs else []
        for call in build_calls:
            checkpoint = checkpoints.get(call["call_id"])
            if checkpoint is None:
                raise RuntimeError(f"Missing checkpoint for {call['call_id']}")
            if call["stage"] == "build_v36_session":
                matched = signatures.get(checkpoint["signature"])
                if matched is None:
                    raise RuntimeError(f"Session prompt signature mismatch for {call['call_id']}")
                session_id, messages, cap = matched
                reconstructed_build[call["call_id"]] = {
                    "session_id": session_id,
                    "input_messages": messages,
                    "max_tokens": cap,
                    "output_text": checkpoint["text"],
                    "checkpoint_signature": checkpoint["signature"],
                    "input_fidelity": "byte_exact_verified_by_checkpoint_signature",
                    "output_fidelity": "exact_checkpoint_response_text",
                }
            elif call["stage"] == "build_v36_identity_consolidation":
                cap = int(call["max_tokens"])
                signature = _checkpoint_signature(identity_messages, cap)
                if signature != checkpoint["signature"]:
                    raise RuntimeError(f"Identity prompt signature mismatch for {call['call_id']}")
                reconstructed_build[call["call_id"]] = {
                    "session_id": None,
                    "input_messages": identity_messages,
                    "max_tokens": cap,
                    "output_text": checkpoint["text"],
                    "checkpoint_signature": checkpoint["signature"],
                    "candidate_pair_count": len(identity_pairs),
                    "input_fidelity": "byte_exact_verified_by_checkpoint_signature",
                    "output_fidelity": "exact_checkpoint_response_text",
                }

    full_call_rows = []
    question_call_index: dict[str, dict[str, Any]] = {}
    for call in calls:
        qid = call["question_id"]
        base = {
            "question_id": qid,
            "benchmark": call["benchmark"],
            "memory_id": question_meta.get(qid, {}).get("memory_id") or builder_to_memory.get(qid),
            "stage": call["stage"],
            "record": {key: value for key, value in call.items() if key != "benchmark"},
        }
        if call["call_id"] in reconstructed_build:
            base.update(reconstructed_build[call["call_id"]])
            base["raw_output_available"] = True
        else:
            result = retrieval[qid]
            trace = result.get("retrieval_trace") or {}
            case = SimpleNamespace(
                question=data_rows[qid]["question"],
                question_date=data_rows[qid].get("question_date"),
            )
            query_ir = SimpleNamespace(**trace["query_ir"])
            augmentation = SimpleNamespace(**trace["v41_query_augmentation"])
            messages = planner_messages(
                case,
                query_ir,
                augmentation,
                trace.get("v41_evidence_certificate") or {},
                trace.get("v41_planner_evidence") or [],
            )
            base.update(
                {
                    "session_id": None,
                    "input_messages": messages,
                    "max_tokens": call.get("max_tokens"),
                    "output_text": None,
                    "parsed_output": trace.get("planner_result"),
                    "raw_output_available": False,
                    "input_fidelity": "reconstructed_from_frozen_trace_and_prompt_code_unverified",
                    "output_fidelity": "parsed_and_post_filtered_result_only_raw_response_not_persisted",
                }
            )
        base["input_sha256"] = hashlib.sha256(
            json.dumps(base["input_messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if base.get("output_text") is not None:
            base["output_sha256"] = hashlib.sha256(base["output_text"].encode("utf-8")).hexdigest()
        full_call_rows.append(base)

    for qid, rows in calls_by_qid.items():
        question_call_index[qid] = {
            "question_id": qid,
            "benchmark": rows[0]["benchmark"],
            "memory_id": question_meta.get(qid, {}).get("memory_id") or builder_to_memory.get(qid),
            "call_ids": [row["call_id"] for row in rows],
            "stages": [row["stage"] for row in rows],
            "call_count": len(rows),
        }
    for qid, meta in question_meta.items():
        question_call_index.setdefault(
            qid,
            {
                "question_id": qid,
                "benchmark": meta["benchmark"],
                "memory_id": meta["memory_id"],
                "call_ids": [],
                "stages": [],
                "call_count": 0,
            },
        )
    write_jsonl_gz(args.output_dir / "llm_calls_full.jsonl.gz", full_call_rows)
    write_jsonl_gz(
        args.output_dir / "question_call_index.jsonl.gz",
        (question_call_index[qid] for qid in sorted(question_call_index)),
    )

    graph_index_rows = []
    memory_directory: dict[str, Path] = {}
    for position, (builder_qid, memory_id) in enumerate(sorted(builder_to_memory.items())):
        label = f"memory_{position:03d}_{hashlib.sha256(memory_id.encode()).hexdigest()[:10]}"
        directory = args.output_dir / "graphs" / label
        directory.mkdir(parents=True, exist_ok=True)
        memory_directory[builder_qid] = directory
        benchmark, cache_path, cache = canonical_cache_by_builder[builder_qid]
        canonical_target = directory / "canonical_memory_artifact.json.gz"
        gzip_copy(cache_path, canonical_target)
        graph_index_rows.append(
            {
                "memory_id": memory_id,
                "builder_question_id": builder_qid,
                "benchmark": benchmark,
                "graph_directory": f"graphs/{label}",
                "canonical_memory_artifact": f"graphs/{label}/{canonical_target.name}",
                "schema_version": cache.get("schema_version"),
                "fingerprint": cache.get("fingerprint"),
                "node_counts": question_meta[builder_qid]["node_counts"],
                "edge_relation_counts": question_meta[builder_qid]["edge_relation_counts"],
                "vector_cache_reference": cache.get("v36_vector_cache"),
            }
        )

    for benchmark, root in benchmark_roots.items():
        for filename in ("nodes.jsonl", "edges.jsonl", "state_chains.jsonl"):
            rows = graph_rows_by_builder(root / filename)
            for builder_qid, values in rows.items():
                if builder_qid in memory_directory:
                    write_jsonl_gz(memory_directory[builder_qid] / f"{filename}.gz", values)
    write_jsonl_gz(args.output_dir / "memory_graph_index.jsonl.gz", graph_index_rows)
    write_jsonl_gz(
        args.output_dir / "question_to_memory_graph.jsonl.gz",
        (
            {
                "question_id": row["question_id"],
                "benchmark": row["benchmark"],
                "memory_id": row["memory_id"],
                "builder_question_id": row["graph_builder_question_id"],
                "graph_directory": next(
                    item["graph_directory"] for item in graph_index_rows
                    if item["builder_question_id"] == row["graph_builder_question_id"]
                ),
            }
            for row in question_rows
        ),
    )

    (args.output_dir / "README.md").write_text(
        """# GraphMem V5 Gate A full-fidelity research bundle

> Restricted research data: this directory contains benchmark conversation
> text, exact build prompts, and exact build-model responses.

## Files

- `llm_calls_full.jsonl.gz`: all 5,266 LLM-call records. The 5,118 build rows
  contain signature-verified `input_messages` and exact checkpoint
  `output_text`. The 148 planner rows contain reconstructed input and the
  persisted parsed/post-filtered result; their raw response was never saved.
- `question_call_index.jsonl.gz`: 200-question to call-ID/stage mapping.
- `memory_graph_index.jsonl.gz`: 110 canonical memory graphs and inventories.
- `question_to_memory_graph.jsonl.gz`: maps every question to its graph,
  including LoCoMo shared-conversation reuse.
- `graphs/*/canonical_memory_artifact.json.gz`: complete cached V4 memory
  artifact, including V3.6 index, inverted indexes, capability view, coverage,
  diagnostics, and call metadata.
- `graphs/*/nodes.jsonl.gz`, `edges.jsonl.gz`, `state_chains.jsonl.gz`: complete
  analysis-friendly graph projections with original node/edge content.
- `manifest.json`: fidelity notes plus SHA-256 and byte size for every file.

Embedding vector binaries are not duplicated; each memory index records its
`vector_cache_reference` back to the frozen Gate A artifact.
""",
        encoding="utf-8",
    )

    files = sorted(path for path in args.output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "graphmem-v5-gate-a-full-fidelity-v1",
        "warning": "Contains benchmark conversation text, exact build prompts, and exact build-model responses. Treat as restricted research data.",
        "coverage": {
            "questions": len(question_rows),
            "memories": len(graph_index_rows),
            "llm_calls": len(full_call_rows),
            "exact_build_inputs_and_outputs": sum(row["raw_output_available"] for row in full_call_rows),
            "planner_inputs_reconstructed_raw_outputs_unavailable": sum(not row["raw_output_available"] for row in full_call_rows),
        },
        "fidelity_notes": {
            "build_calls": "Input messages are verified by checkpoint signature; output_text is the exact saved response.",
            "planner_calls": "Input messages are reconstructed from frozen trace and prompt code; only parsed/post-filtered output survived.",
            "graphs": "Canonical memory artifacts plus complete node, edge, and state-chain exports. Embedding vector binaries remain referenced in the frozen run and are not duplicated.",
        },
        "files": {
            str(path.relative_to(args.output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.name != "manifest.json"
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
