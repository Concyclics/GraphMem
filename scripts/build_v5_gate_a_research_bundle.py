#!/usr/bin/env python3
"""Build an audit-first research bundle from a frozen Gate A run.

The bundle intentionally contains IDs, counts, ranks, hashes, and metrics rather
than copied conversation text. It can therefore be shared for offline analysis
without creating a second ungoverned copy of the benchmark conversations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FEATURE_FIELDS = (
    "graph_expansion",
    "v4_capability_supplements",
    "v41_scene_window_node_ids",
    "v41_late_scene_window_node_ids",
    "v41_answer_bearing_source_ids",
    "v41_reply_bound_evidence",
    "v41_semantic_turn_evidence",
    "v41_owner_lifecycle_dense_source_ids",
    "v41_collection_source_ids",
    "v41_lossless_overlay_source_ids",
    "v41_temporal_operator_source_ids",
    "v41_planner_exposed_unpacked_source_ids",
    "v41_planner_selected_evidence",
    "v41_source_additions",
    "v41_global_exact_recovery_source_ids",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def stats(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": max(vals),
    }


def turn_ids(value: Any) -> set[str]:
    """Recursively collect canonical turn IDs from heterogeneous trace values."""
    found: set[str] = set()
    if isinstance(value, str):
        if ":turn:" in value:
            found.add(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if key in {"source_turn_id", "source_id"} and isinstance(nested, str):
                if ":turn:" in nested:
                    found.add(nested)
            else:
                found.update(turn_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(turn_ids(nested))
    return found


def session_from_turn_id(node_id: str) -> str:
    return node_id.rsplit(":turn:", 1)[0].split(":", 1)[1]


def normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def packed_evidence_blocks(context_text: str) -> dict[str, str]:
    matches = list(re.finditer(r"\[SOURCE_EVIDENCE ([^\]]+)\]\n", context_text or ""))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context_text)
        blocks[match.group(1)] = context_text[match.end():end]
    return blocks


def parse_locomo_gold(question_id: str, evidence: list[str]) -> list[str]:
    refs: list[str] = []
    for item in evidence:
        for session, one_based_turn in re.findall(r"D(\d+):(\d+)", item):
            refs.append(f"{question_id}:session_{int(session)}:turn:{int(one_based_turn) - 1}")
    return sorted(set(refs))


def load_questions(lme_path: Path, locomo_path: Path, annotation_path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(annotation_path):
        annotations[row["question_id"]].append(row)

    for row in json.loads(lme_path.read_text(encoding="utf-8")):
        qid = row["question_id"]
        sessions = dict(zip(row["haystack_session_ids"], row["haystack_sessions"]))
        span_refs = []
        for ref in annotations.get(qid, []):
            raw_turn = sessions[ref["session_id"]][ref["turn_index"]]["content"]
            span_text = raw_turn[ref["span_start"]:ref["span_end"]]
            span_refs.append(
                {
                    "turn_id": f"{qid}:{ref['session_id']}:turn:{ref['turn_index']}",
                    "span_start": ref["span_start"],
                    "span_end": ref["span_end"],
                    "span_sha256": hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    "support_role": ref["support_role"],
                    "confidence": ref["confidence"],
                    "_span_text": span_text,
                }
            )
        role_map = {
            f"{qid}:{ref['session_id']}:turn:{ref['turn_index']}": ref["support_role"]
            for ref in annotations.get(qid, [])
        }
        gold_turns = sorted(
            {
                f"{qid}:{ref['session_id']}:turn:{ref['turn_index']}"
                for ref in annotations.get(qid, [])
            }
        )
        questions[qid] = {
            "benchmark": "longmemeval",
            "stratum": "lme_multi_session" if row["question_type"] == "multi-session" else "lme_temporal",
            "question": row["question"],
            "gold_session_ids": sorted(set(row["answer_session_ids"])),
            "gold_turn_ids": gold_turns,
            "gold_turn_roles": Counter(ref["support_role"] for ref in annotations.get(qid, [])),
            "gold_turn_role_map": role_map,
            "annotation_confidence": Counter(ref["confidence"] for ref in annotations.get(qid, [])),
            "gold_span_refs": span_refs,
            "memory_id": qid,
        }

    for row in json.loads(locomo_path.read_text(encoding="utf-8")):
        qid = row["question_id"]
        gold_turns = parse_locomo_gold(qid, row.get("locomo_evidence", []))
        questions[qid] = {
            "benchmark": "locomo",
            "stratum": "locomo_multihop" if int(row["locomo_category"]) == 1 else "locomo_temporal",
            "question": row["question"],
            "gold_session_ids": sorted({session_from_turn_id(ref) for ref in gold_turns}),
            "gold_turn_ids": gold_turns,
            "gold_turn_roles": {},
            "gold_turn_role_map": {ref: "official_evidence" for ref in gold_turns},
            "annotation_confidence": {"official": len(gold_turns)},
            "gold_span_refs": [],
            "memory_id": f"locomo_sample_{int(row['locomo_sample_index']):02d}",
        }
    return questions


def add_graph_inventory(
    root: Path,
    benchmark: str,
    node_counts: dict[str, Counter[str]],
    edge_counts: dict[str, Counter[str]],
    source_bound_turns: dict[str, set[str]],
) -> tuple[Counter[str], Counter[str]]:
    global_nodes: Counter[str] = Counter()
    global_edges: Counter[str] = Counter()
    with (root / "nodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = row["question_id"]
            node_type = row.get("node_type", "unknown")
            node_counts[qid][node_type] += 1
            global_nodes[node_type] += 1
    with (root / "edges.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = row["question_id"]
            relation = row.get("relation", "unknown")
            edge_counts[qid][relation] += 1
            global_edges[relation] += 1
            if relation == "source" and ":turn:" in row.get("dst", ""):
                source_bound_turns[qid].add(row["dst"])
    return global_nodes, global_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lme-data", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--lme-gold-turns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.lme_data, args.locomo_data, args.lme_gold_turns)
    roots = {
        "longmemeval": args.run_root / "lme/merged/hierarchical_hybrid_graph_v4_1_query",
        "locomo": args.run_root / "locomo/merged/hierarchical_hybrid_graph_v4_1_query",
    }
    retrieval: dict[str, dict[str, Any]] = {}
    qstats: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    node_counts: dict[str, Counter[str]] = defaultdict(Counter)
    edge_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_bound_turns: dict[str, set[str]] = defaultdict(set)
    global_node_counts: dict[str, Counter[str]] = {}
    global_edge_counts: dict[str, Counter[str]] = {}

    for benchmark, root in roots.items():
        for row in read_jsonl(root / "retrieval_results.jsonl"):
            retrieval[row["question_id"]] = row
        for row in read_jsonl(root / "question_stats.jsonl"):
            qstats[row["question_id"]] = row
        for row in read_jsonl(root / "index_diagnostics.jsonl"):
            row["benchmark"] = benchmark
            diagnostics.append(row)
        for row in read_jsonl(root / "llm_calls.jsonl"):
            row["benchmark"] = benchmark
            llm_calls.append(row)
        nc, ec = add_graph_inventory(root, benchmark, node_counts, edge_counts, source_bound_turns)
        global_node_counts[benchmark] = nc
        global_edge_counts[benchmark] = ec

    builder_by_memory: dict[str, str] = {}
    for qid, stat in qstats.items():
        if qid in questions and stat.get("build_total_tokens"):
            builder_by_memory[questions[qid]["memory_id"]] = qid

    feature_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "active_questions": set(),
            "emitted_turns": set(),
            "packed_emitted_turns": set(),
            "gold_refs": set(),
            "packed_gold_refs": set(),
        }
    )
    channel_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "candidate_refs": set(),
            "packed_refs": set(),
            "candidate_gold_refs": set(),
            "packed_gold_refs": set(),
            "questions": set(),
        }
    )
    question_rows: list[dict[str, Any]] = []

    for qid, meta in sorted(questions.items()):
        result = retrieval[qid]
        trace = result.get("retrieval_trace", {})
        stat = qstats[qid]
        graph_qid = builder_by_memory.get(meta["memory_id"], qid)
        graph_source_turns = {
            f"{qid}:{node.split(':', 1)[1]}" for node in source_bound_turns.get(graph_qid, set())
        }
        gold_turns = set(meta["gold_turn_ids"])
        gold_sessions = set(meta["gold_session_ids"])
        packed_turns = set(trace.get("packed_source_turn_ids") or result.get("evidence_leaf_ids", []))
        packed_turns = {item for item in packed_turns if ":turn:" in item}
        packed_sessions = {session_from_turn_id(item) for item in packed_turns}
        context_blocks = packed_evidence_blocks(result.get("context_text", ""))
        gold_span_refs = []
        for ref in meta["gold_span_refs"]:
            public_ref = {key: value for key, value in ref.items() if not key.startswith("_")}
            block = context_blocks.get(ref["turn_id"], "")
            public_ref["packed"] = bool(ref["turn_id"] in packed_turns)
            public_ref["span_present_in_context"] = bool(
                ref["_span_text"] and normalized_text(ref["_span_text"]) in normalized_text(block)
            )
            gold_span_refs.append(public_ref)
        retrieved_sessions = set(result.get("retrieved_session_ids", []))
        coarse_ranked = set(trace.get("coarse_ranked_ids", []))
        selected_cards = set(trace.get("selected_card_ids", []))
        fine_ranked = set(trace.get("fine_ranked_ids", []))
        v41_channels = (trace.get("v41_candidate_trace") or {}).get("channels", {})
        fine_channels = trace.get("fine_channels", {})
        coarse_channels = trace.get("coarse_channels", {})
        feature_turn_sets = {field: turn_ids(trace.get(field)) for field in FEATURE_FIELDS}
        all_feature_turns = set().union(*feature_turn_sets.values()) if feature_turn_sets else set()
        candidate_turns = fine_ranked | set(fine_channels) | set(v41_channels) | all_feature_turns

        for field, emitted in feature_turn_sets.items():
            if trace.get(field):
                feature_rollup[field]["active_questions"].add(qid)
            feature_rollup[field]["emitted_turns"].update(emitted)
            feature_rollup[field]["packed_emitted_turns"].update(emitted & packed_turns)
            feature_rollup[field]["gold_refs"].update({(qid, ref) for ref in gold_turns & emitted})
            feature_rollup[field]["packed_gold_refs"].update({(qid, ref) for ref in gold_turns & emitted & packed_turns})

        merged_turn_channels: dict[str, set[str]] = defaultdict(set)
        for node_id, channels in fine_channels.items():
            merged_turn_channels[node_id].update(channels)
        for node_id, channels in v41_channels.items():
            merged_turn_channels[node_id].update(channels)
        for node_id, channels in merged_turn_channels.items():
            for channel in channels:
                channel_rollup[channel]["candidate_refs"].add((qid, node_id))
                if node_id in packed_turns:
                    channel_rollup[channel]["packed_refs"].add((qid, node_id))
        for card, channels in coarse_channels.items():
            for channel in channels:
                key = f"coarse::{channel}"
                channel_rollup[key]["candidate_refs"].add((qid, card))
                if card in selected_cards:
                    channel_rollup[key]["packed_refs"].add((qid, card))
        for gold in gold_turns:
            channels = set((fine_channels.get(gold) or {}).keys()) | set((v41_channels.get(gold) or {}).keys())
            for channel in channels:
                channel_rollup[channel]["questions"].add(qid)
                channel_rollup[channel]["candidate_gold_refs"].add((qid, gold))
                if gold in packed_turns:
                    channel_rollup[channel]["packed_gold_refs"].add((qid, gold))
        for session in gold_sessions:
            card = f"{qid}:{session}:card"
            for channel in (coarse_channels.get(card) or {}):
                key = f"coarse::{channel}"
                channel_rollup[key]["candidate_refs"].add((qid, card))
                if card in selected_cards or session in retrieved_sessions:
                    channel_rollup[key]["packed_refs"].add((qid, card))
                channel_rollup[key]["questions"].add(qid)
                channel_rollup[key]["candidate_gold_refs"].add((qid, card))
                if card in selected_cards or session in retrieved_sessions:
                    channel_rollup[key]["packed_gold_refs"].add((qid, card))

        gold_found = gold_turns & packed_turns
        gold_candidate = gold_turns & candidate_turns
        session_found = gold_sessions & retrieved_sessions
        packed_session_found = gold_sessions & packed_sessions
        turn_any = bool(gold_found) if gold_turns else False
        turn_all = gold_turns <= packed_turns if gold_turns else False
        session_all = gold_sessions <= retrieved_sessions if gold_sessions else False
        candidate_all = gold_turns <= candidate_turns if gold_turns else False
        source_exists_all = gold_turns <= graph_source_turns if gold_turns else False
        coarse_candidate_sessions = {
            session for session in gold_sessions if f"{qid}:{session}:card" in coarse_ranked
        }
        if turn_all:
            failure = "success"
        elif not session_all:
            failure = "session_routing_miss"
        elif not candidate_all:
            failure = "within_session_candidate_miss"
        else:
            failure = "pack_drop"

        feature_detail = {
            field: {
                "active": bool(trace.get(field)),
                "emitted_turn_count": len(emitted),
                "gold_turn_ids": sorted(gold_turns & emitted),
                "packed_gold_turn_ids": sorted(gold_turns & emitted & packed_turns),
            }
            for field, emitted in feature_turn_sets.items()
        }
        gold_channel_detail = {}
        for gold in sorted(gold_turns):
            gold_channel_detail[gold] = {
                "fine_channels": fine_channels.get(gold, {}),
                "v41_channels": v41_channels.get(gold, {}),
                "fine_rank": (trace.get("fine_ranked_ids") or []).index(gold) + 1 if gold in (trace.get("fine_ranked_ids") or []) else None,
                "packed": gold in packed_turns,
                "source_frame_bound": gold in graph_source_turns,
            }

        question_rows.append(
            {
                "question_id": qid,
                "benchmark": meta["benchmark"],
                "stratum": meta["stratum"],
                "memory_id": meta["memory_id"],
                "question": meta["question"],
                "query_kind": result.get("query_kind"),
                "gold_session_ids": sorted(gold_sessions),
                "retrieved_session_ids": sorted(retrieved_sessions),
                "gold_session_count": len(gold_sessions),
                "retrieved_session_count": len(retrieved_sessions),
                "session_any_hit": bool(session_found),
                "session_all_hit": session_all,
                "session_recall": len(session_found) / len(gold_sessions) if gold_sessions else 0.0,
                "packed_session_any_hit": bool(packed_session_found),
                "packed_session_all_hit": gold_sessions <= packed_sessions if gold_sessions else False,
                "packed_session_recall": len(packed_session_found) / len(gold_sessions) if gold_sessions else 0.0,
                "gold_turn_ids": sorted(gold_turns),
                "gold_turn_roles": dict(meta["gold_turn_roles"]),
                "gold_turn_role_map": meta["gold_turn_role_map"],
                "annotation_confidence": dict(meta["annotation_confidence"]),
                "gold_span_refs": gold_span_refs,
                "gold_span_any_hit": any(ref["span_present_in_context"] for ref in gold_span_refs) if gold_span_refs else None,
                "gold_span_all_hit": all(ref["span_present_in_context"] for ref in gold_span_refs) if gold_span_refs else None,
                "gold_span_recall": (
                    sum(ref["span_present_in_context"] for ref in gold_span_refs) / len(gold_span_refs)
                    if gold_span_refs else None
                ),
                "packed_turn_ids": sorted(packed_turns),
                "gold_turn_count": len(gold_turns),
                "packed_turn_count": len(packed_turns),
                "turn_any_hit": turn_any,
                "turn_all_hit": turn_all,
                "turn_recall": len(gold_found) / len(gold_turns) if gold_turns else 0.0,
                "turn_precision": len(gold_found) / len(packed_turns) if packed_turns else 0.0,
                "candidate_turn_any_hit": bool(gold_candidate),
                "candidate_turn_all_hit": candidate_all,
                "candidate_turn_recall": len(gold_candidate) / len(gold_turns) if gold_turns else 0.0,
                "candidate_gold_turn_ids": sorted(gold_candidate),
                "coarse_candidate_session_recall": len(coarse_candidate_sessions) / len(gold_sessions) if gold_sessions else 0.0,
                "gold_turn_source_frame_bound_all": source_exists_all,
                "failure_stage": failure,
                "missing_gold_sessions": sorted(gold_sessions - retrieved_sessions),
                "missing_gold_turns": sorted(gold_turns - packed_turns),
                "candidate_but_dropped_gold_turns": sorted((gold_turns & candidate_turns) - packed_turns),
                "packed_rough_tokens": result.get("packed_rough_tokens", 0),
                "graph_builder_question_id": graph_qid,
                "node_counts": dict(node_counts.get(graph_qid, {})),
                "edge_relation_counts": dict(edge_counts.get(graph_qid, {})),
                "build_prompt_tokens": stat.get("build_prompt_tokens", 0),
                "build_completion_tokens": stat.get("build_completion_tokens", 0),
                "build_total_tokens": stat.get("build_total_tokens", 0),
                "query_total_tokens": stat.get("answer_total_tokens", 0),
                "retrieval_latency_sec": stat.get("retrieval_latency_sec", 0),
                "summary_parse_error_count": stat.get("summary_parse_error_count", 0),
                "summary_truncation_count": stat.get("summary_truncation_count", 0),
                "coarse_gold_card_channels": {
                    session: coarse_channels.get(f"{qid}:{session}:card", {}) for session in sorted(gold_sessions)
                },
                "gold_turn_channel_detail": gold_channel_detail,
                "feature_contribution_detail": feature_detail,
                "trace_flags": {
                    "planner_required": trace.get("planner_required"),
                    "planner_applied": trace.get("planner_applied"),
                    "answer_target_budget_pass": trace.get("answer_target_budget_pass"),
                    "completeness_certificate_complete": (trace.get("completeness_certificate") or {}).get("complete"),
                    "source_binding_complete": (trace.get("source_binding_certificate") or {}).get("binding_complete"),
                },
                "source_artifact": str(roots[meta["benchmark"]] / "retrieval_results.jsonl"),
            }
        )

    build_rows: list[dict[str, Any]] = []
    for qid, stat in sorted(qstats.items()):
        if not stat.get("build_total_tokens"):
            continue
        meta = questions[qid]
        identity = [row for row in llm_calls if row["question_id"] == qid and row["stage"] == "build_v36_identity_consolidation"]
        session_calls = [row for row in llm_calls if row["question_id"] == qid and row["stage"] == "build_v36_session"]
        related_diag = [row for row in diagnostics if row.get("question_id") == qid]
        build_rows.append(
            {
                "memory_id": meta["memory_id"],
                "builder_question_id": qid,
                "benchmark": meta["benchmark"],
                "stratum": meta["stratum"],
                "session_count": stat.get("session_count", 0),
                "node_counts": dict(node_counts.get(qid, {})),
                "edge_relation_counts": dict(edge_counts.get(qid, {})),
                "leaf_count": stat.get("leaf_count", 0),
                "summary_count": stat.get("summary_count", 0),
                "edge_count": stat.get("edge_count", 0),
                "build_calls": len(session_calls) + len(identity),
                "session_build_calls": len(session_calls),
                "identity_calls": len(identity),
                "build_prompt_tokens": stat.get("build_prompt_tokens", 0),
                "build_completion_tokens": stat.get("build_completion_tokens", 0),
                "build_total_tokens": stat.get("build_total_tokens", 0),
                "tokens_per_session": stat.get("build_total_tokens", 0) / max(1, stat.get("session_count", 0)),
                "tokens_per_source_turn": stat.get("build_total_tokens", 0) / max(1, stat.get("leaf_count", 0)),
                "identity_total_tokens": sum(row.get("total_tokens", 0) for row in identity),
                "identity_input_tokens": sum(row.get("prompt_tokens", 0) for row in identity),
                "identity_output_tokens": sum(row.get("completion_tokens", 0) for row in identity),
                "parse_error_sessions": sum(bool(row.get("parse_error")) for row in related_diag),
                "truncated_sessions": sum(row.get("finish_reason") == "length" for row in related_diag),
                "lossless_only_turns": sum(row.get("lossless_only_count", 0) for row in related_diag),
                "frame_count": sum(row.get("frame_count", 0) for row in related_diag),
                "coverage_count": sum(row.get("coverage_count", 0) for row in related_diag),
                "build_latency_sec": stat.get("build_latency_sec", 0),
            }
        )

    session_rows = [
        {
            "benchmark": row["benchmark"],
            "question_id": row["question_id"],
            "session_id": row.get("session_id"),
            "turn_count": row.get("turn_count", 0),
            "frame_count": row.get("frame_count", 0),
            "coverage_count": row.get("coverage_count", 0),
            "lossless_only_count": row.get("lossless_only_count", 0),
            "parse_error": row.get("parse_error"),
            "extraction_mode": row.get("extraction_mode"),
            "prompt_tokens": row.get("prompt_tokens", 0),
            "completion_tokens": row.get("completion_tokens", 0),
            "total_tokens": row.get("total_tokens", 0),
            "finish_reason": row.get("finish_reason"),
            "requested_completion_cap": row.get("requested_completion_cap", 0),
        }
        for row in diagnostics
        if row.get("stage") == "v36_session_extraction"
    ]

    token_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in llm_calls:
        key = (row["benchmark"], row["stage"])
        token_counter[key].update(
            calls=1,
            input_tokens=row.get("prompt_tokens", 0),
            output_tokens=row.get("completion_tokens", 0),
            total_tokens=row.get("total_tokens", 0),
            reasoning_tokens=row.get("reasoning_tokens", 0),
            length_finishes=row.get("finish_reason") == "length",
            errors=bool(row.get("error_status")),
            prompt_cache_hit_tokens=row.get("prompt_cache_hit_tokens", 0),
            prompt_cache_miss_tokens=row.get("prompt_cache_miss_tokens", 0),
            retries=row.get("retry_count", 0),
        )
    grand_tokens = sum(counter["total_tokens"] for counter in token_counter.values())
    token_rows = []
    for (benchmark, stage), counter in sorted(token_counter.items()):
        row = {"benchmark": benchmark, "stage": stage, **counter}
        row["share_of_all_tokens"] = counter["total_tokens"] / grand_tokens if grand_tokens else 0.0
        row["avg_tokens_per_call"] = counter["total_tokens"] / counter["calls"] if counter["calls"] else 0.0
        matching_calls = [
            item for item in llm_calls
            if item["benchmark"] == benchmark and item["stage"] == stage
        ]
        latency = stats(item.get("latency_sec", 0) for item in matching_calls)
        row.update({f"latency_sec_{key}": value for key, value in latency.items()})
        token_rows.append(row)

    llm_call_rows = []
    for row in llm_calls:
        meta = questions[row["question_id"]]
        llm_call_rows.append(
            {
                "benchmark": row["benchmark"],
                "question_id": row["question_id"],
                "memory_id": meta["memory_id"],
                "call_id": row.get("call_id"),
                "variant": row.get("variant"),
                "stage": row.get("stage"),
                "model": row.get("model"),
                "thinking_mode": row.get("thinking_mode"),
                "prompt_tokens": row.get("prompt_tokens", 0),
                "completion_tokens": row.get("completion_tokens", 0),
                "total_tokens": row.get("total_tokens", 0),
                "prompt_cache_hit_tokens": row.get("prompt_cache_hit_tokens", 0),
                "prompt_cache_miss_tokens": row.get("prompt_cache_miss_tokens", 0),
                "reasoning_tokens": row.get("reasoning_tokens", 0),
                "latency_sec": row.get("latency_sec", 0),
                "retry_count": row.get("retry_count", 0),
                "error_status": row.get("error_status"),
                "finish_reason": row.get("finish_reason"),
                "max_tokens": row.get("max_tokens"),
                "response_format": row.get("response_format"),
                "breakdown_inferred": row.get("breakdown_inferred"),
                "excluded_from_budget": row.get("excluded_from_budget"),
            }
        )

    gold_ref_to_features: dict[tuple[str, str], set[str]] = defaultdict(set)
    for feature, values in feature_rollup.items():
        for ref in values["packed_gold_refs"]:
            gold_ref_to_features[ref].add(feature)
    feature_rows = []
    for feature, values in sorted(feature_rollup.items()):
        packed_emitted = len(values["packed_emitted_turns"])
        packed_gold = len(values["packed_gold_refs"])
        feature_rows.append(
            {
                "feature": feature,
                "active_questions": len(values["active_questions"]),
                "emitted_unique_turns": len(values["emitted_turns"]),
                "packed_emitted_turns": packed_emitted,
                "gold_refs_covered": len(values["gold_refs"]),
                "packed_gold_refs_covered": packed_gold,
                "packed_gold_precision": packed_gold / packed_emitted if packed_emitted else 0.0,
                "exclusive_packed_gold_refs": sum(
                    len(gold_ref_to_features[ref]) == 1 for ref in values["packed_gold_refs"]
                ),
                "interpretation": "observed trace contribution; not a causal ablation",
            }
        )
    gold_ref_to_channels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for channel, values in channel_rollup.items():
        for ref in values["packed_gold_refs"]:
            gold_ref_to_channels[ref].add(channel)
    channel_rows = []
    for channel, values in sorted(channel_rollup.items()):
        candidate_count = len(values["candidate_refs"])
        packed_count = len(values["packed_refs"])
        candidate_gold = len(values["candidate_gold_refs"])
        packed_gold = len(values["packed_gold_refs"])
        channel_rows.append(
            {
                "channel": channel,
                "questions_with_gold_candidate": len(values["questions"]),
                "candidate_refs": candidate_count,
                "packed_refs": packed_count,
                "candidate_gold_refs": candidate_gold,
                "packed_gold_refs": packed_gold,
                "candidate_gold_precision": candidate_gold / candidate_count if candidate_count else 0.0,
                "packed_gold_precision": packed_gold / packed_count if packed_count else 0.0,
                "exclusive_packed_gold_refs": sum(
                    len(gold_ref_to_channels[ref]) == 1 for ref in values["packed_gold_refs"]
                ),
                "interpretation": "overlapping attribution; channels are not mutually exclusive",
            }
        )

    relation_rollup: dict[str, Counter[str]] = defaultdict(Counter)
    for row in question_rows:
        result = retrieval[row["question_id"]]
        gold = set(row["gold_turn_ids"])
        packed = set(row["packed_turn_ids"])
        for edge in result.get("retrieval_trace", {}).get("graph_expansion", []):
            relation = edge.get("relation", "unknown")
            relation_rollup[relation]["expansions"] += 1
            destination = edge.get("to")
            if destination in gold:
                relation_rollup[relation]["gold_destinations"] += 1
            if destination in packed:
                relation_rollup[relation]["packed_destinations"] += 1
            if destination in gold and destination in packed:
                relation_rollup[relation]["packed_gold_destinations"] += 1
    relation_rows = [
        {"relation": relation, **counter}
        for relation, counter in sorted(relation_rollup.items())
    ]

    turn_ref_rollup: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in question_rows:
        candidate = set(row["candidate_gold_turn_ids"])
        packed = set(row["packed_turn_ids"])
        detail = row["gold_turn_channel_detail"]
        for ref in row["gold_turn_ids"]:
            role = row["gold_turn_role_map"].get(ref, "unknown")
            counter = turn_ref_rollup[(row["benchmark"], row["stratum"], role)]
            counter["gold_refs"] += 1
            counter["candidate_refs"] += bool(detail[ref]["fine_channels"] or detail[ref]["v41_channels"] or ref in candidate)
            counter["packed_refs"] += ref in packed
            counter["source_frame_bound_refs"] += bool(detail[ref]["source_frame_bound"])
    turn_ref_rows = [
        {
            "benchmark": benchmark,
            "stratum": stratum,
            "support_role": role,
            **counter,
            "candidate_recall": counter["candidate_refs"] / counter["gold_refs"],
            "packed_recall": counter["packed_refs"] / counter["gold_refs"],
            "source_frame_bound_rate": counter["source_frame_bound_refs"] / counter["gold_refs"],
        }
        for (benchmark, stratum, role), counter in sorted(turn_ref_rollup.items())
    ]

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        by_stratum[row["stratum"]].append(row)
    retrieval_summary = {}
    for stratum, rows in sorted(by_stratum.items()):
        retrieval_summary[stratum] = {
            "questions": len(rows),
            "session_any_hit": sum(row["session_any_hit"] for row in rows) / len(rows),
            "session_all_hit": sum(row["session_all_hit"] for row in rows) / len(rows),
            "session_mean_recall": sum(row["session_recall"] for row in rows) / len(rows),
            "packed_session_any_hit": sum(row["packed_session_any_hit"] for row in rows) / len(rows),
            "packed_session_all_hit": sum(row["packed_session_all_hit"] for row in rows) / len(rows),
            "packed_session_mean_recall": sum(row["packed_session_recall"] for row in rows) / len(rows),
            "turn_any_hit": sum(row["turn_any_hit"] for row in rows) / len(rows),
            "turn_all_hit": sum(row["turn_all_hit"] for row in rows) / len(rows),
            "turn_mean_recall": sum(row["turn_recall"] for row in rows) / len(rows),
            "turn_mean_precision": sum(row["turn_precision"] for row in rows) / len(rows),
            "candidate_turn_all_hit": sum(row["candidate_turn_all_hit"] for row in rows) / len(rows),
            "failure_stages": dict(Counter(row["failure_stage"] for row in rows)),
            "packed_tokens": stats(row["packed_rough_tokens"] for row in rows),
        }
        span_rows = [row for row in rows if row["gold_span_refs"]]
        if span_rows:
            retrieval_summary[stratum]["span_any_hit"] = sum(row["gold_span_any_hit"] for row in span_rows) / len(span_rows)
            retrieval_summary[stratum]["span_all_hit"] = sum(row["gold_span_all_hit"] for row in span_rows) / len(span_rows)
            retrieval_summary[stratum]["span_mean_recall"] = sum(row["gold_span_recall"] for row in span_rows) / len(span_rows)

    build_summary = {
        benchmark: {
            "memories": len(rows),
            "build_total_tokens": sum(row["build_total_tokens"] for row in rows),
            "build_tokens_per_memory": stats(row["build_total_tokens"] for row in rows),
            "tokens_per_session": stats(row["tokens_per_session"] for row in rows),
            "tokens_per_source_turn": stats(row["tokens_per_source_turn"] for row in rows),
            "summary_to_leaf_ratio": sum(row["summary_count"] for row in rows) / max(1, sum(row["leaf_count"] for row in rows)),
            "parse_error_session_rate": sum(row["parse_error_sessions"] for row in rows) / max(1, sum(row["session_count"] for row in rows)),
            "truncated_session_rate": sum(row["truncated_sessions"] for row in rows) / max(1, sum(row["session_count"] for row in rows)),
        }
        for benchmark in roots
        for rows in [[row for row in build_rows if row["benchmark"] == benchmark]]
    }

    summary = {
        "schema_version": "graphmem-v5-gate-a-research-v1",
        "run_root": str(args.run_root),
        "questions": len(question_rows),
        "memories": len(build_rows),
        "llm_calls": sum(row["calls"] for row in token_rows),
        "llm_tokens": grand_tokens,
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in token_rows),
        "token_breakdown": token_rows,
        "build_summary": build_summary,
        "retrieval_summary": retrieval_summary,
        "graph_inventory": {
            benchmark: {
                "node_types": dict(global_node_counts[benchmark]),
                "edge_relations": dict(global_edge_counts[benchmark]),
            }
            for benchmark in roots
        },
        "failure_stage_counts": dict(Counter(row["failure_stage"] for row in question_rows)),
        "session_turn_crosstab": {
            f"session_all={str(session_all).lower()},turn_all={str(turn_all).lower()}": sum(
                row["session_all_hit"] == session_all and row["turn_all_hit"] == turn_all
                for row in question_rows
            )
            for session_all in (False, True)
            for turn_all in (False, True)
        },
        "data_sources": {
            "lme_data": {"path": str(args.lme_data), "sha256": sha256(args.lme_data)},
            "locomo_data": {"path": str(args.locomo_data), "sha256": sha256(args.locomo_data)},
            "lme_gold_turns": {"path": str(args.lme_gold_turns), "sha256": sha256(args.lme_gold_turns)},
        },
        "metric_notes": {
            "turn_match": "exact canonical (question_id, session_id, zero-based turn_index)",
            "locomo_mapping": "official Dn:m evidence maps to session_n and zero-based turn m-1",
            "precision_denominator": "unique packed source turns after deduplication",
            "feature_attribution": "trace overlap only; requires ablation for causal value",
        },
    }

    failure_rows = [row for row in question_rows if row["failure_stage"] != "success"]
    write_jsonl(args.output_dir / "question_research_log.jsonl", question_rows)
    write_jsonl(args.output_dir / "memory_build_log.jsonl", build_rows)
    write_jsonl(args.output_dir / "session_build_log.jsonl", session_rows)
    write_jsonl(args.output_dir / "llm_call_log.jsonl", llm_call_rows)
    write_jsonl(args.output_dir / "turn_failure_cases.jsonl", failure_rows)
    write_csv(args.output_dir / "token_breakdown.csv", token_rows)
    write_csv(args.output_dir / "feature_trace_contribution.csv", feature_rows)
    write_csv(args.output_dir / "retrieval_channel_gold_coverage.csv", channel_rows)
    write_csv(args.output_dir / "graph_expansion_relation_contribution.csv", relation_rows)
    write_csv(args.output_dir / "turn_reference_metrics.csv", turn_ref_rows)
    (args.output_dir / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle_files = (
        "feature_trace_contribution.csv",
        "graph_expansion_relation_contribution.csv",
        "llm_call_log.jsonl",
        "memory_build_log.jsonl",
        "question_research_log.jsonl",
        "research_summary.json",
        "retrieval_channel_gold_coverage.csv",
        "session_build_log.jsonl",
        "token_breakdown.csv",
        "turn_failure_cases.jsonl",
        "turn_reference_metrics.csv",
    )
    manifest = {
        "schema_version": "graphmem-v5-gate-a-research-bundle-v1",
        "producer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "files": {
            name: {
                "sha256": sha256(args.output_dir / name),
                "bytes": (args.output_dir / name).stat().st_size,
            }
            for name in bundle_files
        },
    }
    (args.output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
