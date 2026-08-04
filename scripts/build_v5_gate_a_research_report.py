#!/usr/bin/env python3
"""Create the canonical portable-report payload for the Gate A research bundle."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATUM_LABELS = {
    "lme_multi_session": "LME multi-session",
    "lme_temporal": "LME temporal",
    "locomo_multihop": "LoCoMo Cat1 multi-hop",
    "locomo_temporal": "LoCoMo Cat2 temporal",
}

STAGE_LABELS = {
    ("longmemeval", "build_v36_session"): "LME · session extraction",
    ("longmemeval", "build_v36_identity_consolidation"): "LME · identity consolidation",
    ("locomo", "build_v36_session"): "LoCoMo · session extraction",
    ("locomo", "build_v36_identity_consolidation"): "LoCoMo · identity consolidation",
    ("longmemeval", "answer_query_planner"): "LME · query planner",
    ("locomo", "answer_query_planner"): "LoCoMo · query planner",
}

FEATURE_LABELS = {
    "graph_expansion": "Graph expansion",
    "v41_semantic_turn_evidence": "Semantic-turn evidence",
    "v41_scene_window_node_ids": "Scene window",
    "v41_late_scene_window_node_ids": "Late scene window",
    "v41_planner_selected_evidence": "Planner-selected evidence",
    "v4_capability_supplements": "Capability supplements",
    "v41_reply_bound_evidence": "Reply-bound evidence",
    "v41_source_additions": "Source additions",
    "v41_answer_bearing_source_ids": "Answer-bearing source IDs",
    "v41_owner_lifecycle_dense_source_ids": "Owner lifecycle dense",
    "v41_collection_source_ids": "Collection sources",
    "v41_lossless_overlay_source_ids": "Lossless overlay",
    "v41_temporal_operator_source_ids": "Temporal operator",
    "v41_planner_exposed_unpacked_source_ids": "Planner-exposed unpacked",
    "v41_global_exact_recovery_source_ids": "Global exact recovery",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def source(source_id: str, label: str, path: str, tables: list[str], description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "python",
            "sql": "SELECT * FROM reviewed_gate_a_research_bundle",
            "description": description,
            "tables_used": tables,
            "filters": ["Frozen Gate A Qwen3-30B retrieval-only run", "200-question development set"],
            "metric_definitions": [
                "Turn all-hit = questions where every exact gold turn ID occurs in deduplicated packed source turns / questions.",
                "Candidate turn all-hit = questions where every gold turn occurs in the union of fine-ranked, V4.1-channel, and feature-emitted turn candidates / questions.",
                "Token share = provider-reported stage tokens / 27,623,686 total backbone tokens.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.research_dir / "research_summary.json").read_text(encoding="utf-8"))
    token_rows = []
    for row in read_csv(args.research_dir / "token_breakdown.csv"):
        token_rows.append(
            {
                "benchmark": row["benchmark"],
                "stage": row["stage"],
                "stage_label": STAGE_LABELS[(row["benchmark"], row["stage"])],
                "calls": number(row["calls"]),
                "input_tokens": number(row["input_tokens"]),
                "output_tokens": number(row["output_tokens"]),
                "total_tokens": number(row["total_tokens"]),
                "share": number(row["share_of_all_tokens"]),
                "length_finishes": number(row["length_finishes"]),
                "reasoning_tokens": number(row["reasoning_tokens"]),
            }
        )
    token_rows.sort(key=lambda row: row["total_tokens"], reverse=True)

    recall_rows = []
    recall_long = []
    failure_rows = []
    for stratum, row in summary["retrieval_summary"].items():
        base = {
            "stratum": stratum,
            "stratum_label": STRATUM_LABELS[stratum],
            "questions": row["questions"],
            "session_all_hit": row["session_all_hit"],
            "packed_session_all_hit": row["packed_session_all_hit"],
            "candidate_turn_all_hit": row["candidate_turn_all_hit"],
            "turn_all_hit": row["turn_all_hit"],
            "turn_any_hit": row["turn_any_hit"],
            "turn_mean_recall": row["turn_mean_recall"],
            "turn_mean_precision": row["turn_mean_precision"],
            "packed_tokens_mean": row["packed_tokens"]["mean"],
        }
        recall_rows.append(base)
        for metric, label in (
            ("session_all_hit", "Session all-hit"),
            ("candidate_turn_all_hit", "Candidate turn all-hit"),
            ("turn_all_hit", "Final turn all-hit"),
        ):
            recall_long.append({**base, "metric": label, "rate": base[metric]})
        for stage in ("session_routing_miss", "within_session_candidate_miss", "pack_drop", "success"):
            failure_rows.append(
                {
                    "stratum": stratum,
                    "stratum_label": STRATUM_LABELS[stratum],
                    "stage": stage,
                    "stage_label": stage.replace("_", " ").title(),
                    "questions": row["failure_stages"].get(stage, 0),
                    "denominator": row["questions"],
                }
            )

    feature_rows = []
    for row in read_csv(args.research_dir / "feature_trace_contribution.csv"):
        packed_outputs = int(row["packed_emitted_turns"])
        packed_gold = int(row["packed_gold_refs_covered"])
        feature_rows.append(
            {
                "feature": row["feature"],
                "feature_label": FEATURE_LABELS.get(row["feature"], row["feature"]),
                "active_questions": int(row["active_questions"]),
                "emitted_unique_turns": int(row["emitted_unique_turns"]),
                "packed_outputs": packed_outputs,
                "gold_refs_covered": int(row["gold_refs_covered"]),
                "packed_gold_refs": packed_gold,
                "packed_gold_precision": float(row["packed_gold_precision"]),
                "exclusive_packed_gold_refs": int(row["exclusive_packed_gold_refs"]),
                "interpretation": row["interpretation"],
            }
        )
    feature_rows.sort(key=lambda row: (row["packed_gold_refs"], row["exclusive_packed_gold_refs"]), reverse=True)

    role_rows = []
    for row in read_csv(args.research_dir / "turn_reference_metrics.csv"):
        role_rows.append(
            {
                "benchmark": row["benchmark"],
                "stratum_label": STRATUM_LABELS[row["stratum"]],
                "support_role": row["support_role"],
                "gold_refs": int(row["gold_refs"]),
                "candidate_refs": int(row["candidate_refs"]),
                "packed_refs": int(row["packed_refs"]),
                "candidate_recall": float(row["candidate_recall"]),
                "packed_recall": float(row["packed_recall"]),
                "source_frame_bound_rate": float(row["source_frame_bound_rate"]),
            }
        )

    build_tokens = sum(row["total_tokens"] for row in token_rows if row["stage"].startswith("build_"))
    headline = [{
        "total_tokens": summary["llm_tokens"],
        "build_token_share": build_tokens / summary["llm_tokens"],
        "lme_turn_all_hit": (
            summary["retrieval_summary"]["lme_multi_session"]["turn_all_hit"]
            + summary["retrieval_summary"]["lme_temporal"]["turn_all_hit"]
        ) / 2,
        "locomo_turn_all_hit": (
            summary["retrieval_summary"]["locomo_multihop"]["turn_all_hit"]
            + summary["retrieval_summary"]["locomo_temporal"]["turn_all_hit"]
        ) / 2,
        "reasoning_tokens": summary["reasoning_tokens"],
    }]

    sources = [
        source(
            "research_bundle",
            "Gate A reviewed research bundle",
            "research/research_summary.json",
            [
                "research_summary.json",
                "question_research_log.jsonl",
                "memory_build_log.jsonl",
                "session_build_log.jsonl",
            ],
            "Recomputes exact-turn, token, graph, feature, and failure-stage metrics from the frozen run logs.",
        ),
        source(
            "gold_turn_assets",
            "Versioned exact-turn evaluation assets",
            "eval/longmemeval_v5_dev100_gold_turns.jsonl",
            ["LongMemEval manual gold turns", "LoCoMo official locomo_evidence"],
            "Maps evidence to canonical question/session/zero-based-turn identifiers without embedding conversation text.",
        ),
    ]

    title = "GraphMem V5 Gate A：Token、图特征与精确证据召回诊断"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "A portable, source-backed diagnosis of build cost, graph utilization, and exact turn-level recall.",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cards": [
            {"id": "total_tokens", "dataset": "headline", "sourceId": "research_bundle", "description": "All provider-reported input and output tokens across 5,266 calls.", "metrics": [{"label": "Backbone tokens", "field": "total_tokens", "format": "number"}]},
            {"id": "build_share", "dataset": "headline", "sourceId": "research_bundle", "description": "Session extraction plus identity consolidation.", "metrics": [{"label": "Build token share", "field": "build_token_share", "format": "percent"}]},
            {"id": "lme_turn_all", "dataset": "headline", "sourceId": "gold_turn_assets", "description": "Exact all-gold-turn hit rate across the 100-question LongMemEval development set.", "metrics": [{"label": "LME turn all-hit", "field": "lme_turn_all_hit", "format": "percent"}]},
            {"id": "locomo_turn_all", "dataset": "headline", "sourceId": "gold_turn_assets", "description": "Official evidence-turn all-hit across LoCoMo Cat1 and Cat2.", "metrics": [{"label": "LoCoMo turn all-hit", "field": "locomo_turn_all_hit", "format": "percent"}]},
            {"id": "reasoning_tokens", "dataset": "headline", "sourceId": "research_bundle", "description": "Thinking was disabled and audited for every provider call.", "metrics": [{"label": "Reasoning tokens", "field": "reasoning_tokens", "format": "number"}]},
        ],
        "charts": [
            {
                "id": "token_by_stage",
                "title": "Build stages dominate token spend",
                "subtitle": "Provider-reported input + output tokens; each row retains call and truncation context.",
                "type": "bar",
                "dataset": "token_breakdown",
                "sourceId": "research_bundle",
                "encodings": {
                    "x": {"field": "stage_label", "type": "nominal", "label": "Stage"},
                    "y": {"field": "total_tokens", "type": "quantitative", "label": "Tokens", "format": "number"},
                },
            },
            {
                "id": "recall_funnel",
                "title": "Session success does not guarantee exact evidence turns",
                "subtitle": "All-hit at session, candidate-turn, and final packed-turn stages; 50 questions per stratum.",
                "type": "bar",
                "dataset": "recall_long",
                "sourceId": "gold_turn_assets",
                "encodings": {
                    "x": {"field": "stratum_label", "type": "nominal", "label": "Stratum"},
                    "y": {"field": "rate", "type": "quantitative", "label": "All-hit", "format": "percent"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                },
            },
            {
                "id": "failures_by_stage",
                "title": "LoCoMo failures concentrate before packing",
                "subtitle": "Mutually exclusive per-question failure classification plus successes.",
                "type": "bar",
                "dataset": "failure_stages",
                "sourceId": "research_bundle",
                "encodings": {
                    "x": {"field": "stratum_label", "type": "nominal", "label": "Stratum"},
                    "y": {"field": "questions", "type": "quantitative", "label": "Questions", "format": "number"},
                    "color": {"field": "stage_label", "type": "nominal", "label": "Outcome"},
                },
            },
            {
                "id": "feature_gold_coverage",
                "title": "Observed feature contributions are broad and overlapping",
                "subtitle": "Packed exact-gold references attributed to each trace feature; this is not causal lift.",
                "type": "bar",
                "dataset": "feature_contribution",
                "sourceId": "research_bundle",
                "encodings": {
                    "x": {"field": "feature_label", "type": "nominal", "label": "Trace feature"},
                    "y": {"field": "packed_gold_refs", "type": "quantitative", "label": "Packed gold refs", "format": "number"},
                },
            },
        ],
        "tables": [
            {
                "id": "token_table",
                "title": "Exact token ledger",
                "dataset": "token_breakdown",
                "sourceId": "research_bundle",
                "columns": [
                    {"field": "stage_label", "label": "Benchmark / stage"},
                    {"field": "calls", "label": "Calls", "format": "number"},
                    {"field": "input_tokens", "label": "Input", "format": "number"},
                    {"field": "output_tokens", "label": "Output", "format": "number"},
                    {"field": "total_tokens", "label": "Total", "format": "number"},
                    {"field": "share", "label": "Run share", "format": "percent"},
                    {"field": "length_finishes", "label": "Length finishes", "format": "number"},
                ],
            },
            {
                "id": "recall_table",
                "title": "Exact navigation metrics by stratum",
                "dataset": "recall",
                "sourceId": "gold_turn_assets",
                "columns": [
                    {"field": "stratum_label", "label": "Stratum"},
                    {"field": "session_all_hit", "label": "Session all", "format": "percent"},
                    {"field": "candidate_turn_all_hit", "label": "Candidate turn all", "format": "percent"},
                    {"field": "turn_all_hit", "label": "Final turn all", "format": "percent"},
                    {"field": "turn_any_hit", "label": "Turn any", "format": "percent"},
                    {"field": "turn_mean_recall", "label": "Mean recall", "format": "percent"},
                    {"field": "turn_mean_precision", "label": "Mean precision", "format": "percent"},
                    {"field": "packed_tokens_mean", "label": "Mean packed tokens", "format": "number"},
                ],
            },
            {
                "id": "feature_table",
                "title": "Feature trace attribution",
                "dataset": "feature_contribution",
                "sourceId": "research_bundle",
                "columns": [
                    {"field": "feature_label", "label": "Feature"},
                    {"field": "active_questions", "label": "Active Qs", "format": "number"},
                    {"field": "packed_outputs", "label": "Packed outputs", "format": "number"},
                    {"field": "packed_gold_refs", "label": "Packed gold refs", "format": "number"},
                    {"field": "packed_gold_precision", "label": "Gold precision", "format": "percent"},
                    {"field": "exclusive_packed_gold_refs", "label": "Exclusive gold refs", "format": "number"},
                ],
            },
            {
                "id": "role_table",
                "title": "Gold-turn recall by support role",
                "dataset": "turn_roles",
                "sourceId": "gold_turn_assets",
                "columns": [
                    {"field": "stratum_label", "label": "Stratum"},
                    {"field": "support_role", "label": "Support role"},
                    {"field": "gold_refs", "label": "Gold refs", "format": "number"},
                    {"field": "candidate_recall", "label": "Candidate recall", "format": "percent"},
                    {"field": "packed_recall", "label": "Packed recall", "format": "percent"},
                    {"field": "source_frame_bound_rate", "label": "Frame-bound", "format": "percent"},
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {"id": "executive_summary", "type": "markdown", "sourceId": "research_bundle", "body": "## Executive Summary\n\nThe frozen 200-question retrieval-only run spends **99.41%** of 27.62M backbone tokens during graph construction, yet exact evidence navigation remains the bottleneck. LongMemEval session all-hit is 82%, while exact turn all-hit is 71%; LoCoMo falls from 69% session all-hit to 5% official evidence-turn all-hit. The immediate opportunity is to replace broad LLM extraction with deterministic local structure plus selective refinement, then repair query-time raw-turn candidate generation and frame→source closure."},
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["total_tokens", "build_share", "lme_turn_all", "locomo_turn_all", "reasoning_tokens"]},
            {"id": "token_heading", "type": "markdown", "sourceId": "research_bundle", "body": "## Token Cost Anatomy\n\nSession extraction consumes 86.22% of all tokens and identity consolidation another 13.19%; both query planners together consume only 0.59%. Output tokens account for 42.71% of the total, and 68.43% of LongMemEval session extractions finish at the length cap. Cost optimization should start with extraction schema and selective refinement, not planner removal."},
            {"id": "token_chart", "type": "chart", "chartId": "token_by_stage"},
            {"id": "token_detail", "type": "table", "tableId": "token_table"},
            {"id": "graph_heading", "type": "markdown", "sourceId": "research_bundle", "body": "## Graph Construction and Use\n\nThe LongMemEval graphs contain 145,503 nodes and 207,196 edges. Source, routing-contains, next-turn, and dialogue-pair edges make up 93.75% of all edges. Of 1,438 observed graph expansions, 1,260 follow source edges; all 51 direct expansions to packed gold turns use source. The graph currently behaves mainly as a provenance back-link rather than a strong cross-session typed navigator."},
            {"id": "recall_heading", "type": "markdown", "sourceId": "gold_turn_assets", "body": "## Exact Turn-Level Recall\n\nLongMemEval exact turn-level evaluation is available now: 217 manually reviewed references use canonical `(question_id, session_id, zero-based turn_index)` matching. LoCoMo uses 232 official evidence-turn references. Session-level metrics materially overstate evidence completeness, especially for LoCoMo Cat1 and Cat2."},
            {"id": "recall_chart", "type": "chart", "chartId": "recall_funnel"},
            {"id": "recall_detail", "type": "table", "tableId": "recall_table"},
            {"id": "failure_heading", "type": "markdown", "sourceId": "research_bundle", "body": "## Why All-Hit Is Low\n\nLongMemEval has 29 exact-turn failures: 15 session-routing misses, 3 within-session candidate misses, and 11 pack drops. LoCoMo has 95 failures: 31 routing misses and 64 within-session candidate misses, with no observed candidate-complete pack drops. For LoCoMo, approximately 89.2% of official evidence turns already have a source-bound frame, so the primary gap is exposing and ranking the existing frame/turn evidence at query time."},
            {"id": "failure_chart", "type": "chart", "chartId": "failures_by_stage"},
            {"id": "feature_heading", "type": "markdown", "sourceId": "research_bundle", "body": "## Useful and Removable Features\n\nObserved traces favor lossless SourceTurn provenance, frame→source closure, semantic-turn evidence, scene windows, and the low-cost planner. Source additions and answer-bearing additions are wide but low precision; sidecar-inverted, structured, role-relation, late-scene, and similar overlapping channels should be isolated first. These are ablation priorities, not proven causal effects: feature outputs overlap and a channel may contribute indirectly through frames or groups."},
            {"id": "feature_chart", "type": "chart", "chartId": "feature_gold_coverage"},
            {"id": "feature_detail", "type": "table", "tableId": "feature_table"},
            {"id": "role_heading", "type": "markdown", "sourceId": "gold_turn_assets", "body": "## Evidence Roles and Query Types\n\nLongMemEval multi-session negative-scope references have only 45.5% packed recall. Count questions reach 40% turn all-hit, temporal-order 50%, and entity questions 33%, while duration reaches 87%. This is consistent with a closure problem: count, list, absence, and temporal-boundary questions require complete evidence sets rather than one highly similar turn."},
            {"id": "role_detail", "type": "table", "tableId": "role_table"},
            {"id": "methodology", "type": "markdown", "sourceId": "research_bundle", "body": "## Scope and Methodology\n\nThis is a descriptive audit of one frozen configuration: Qwen3-30B retrieval-only over 100 LongMemEval and 100 LoCoMo development questions. Token totals come from provider logs; packed turns are deduplicated before precision/recall; failures are assigned to the earliest mutually exclusive stage. LongMemEval gold turns are human-reviewed, while LoCoMo `Dn:m` references map to session `n` and zero-based turn `m−1`."},
            {"id": "limitations", "type": "markdown", "sourceId": "research_bundle", "body": "## Limitations\n\nFeature and channel attribution is overlapping trace coverage, not causal lift. Strict full-span containment is only a conservative compression audit because packed text can preserve a sufficient subclause without the entire annotated span. The present run has no paired ablations, so no feature is certified safe to delete. Query-time code must continue to remain isolated from answers, gold sessions, gold turns, and category labels."},
            {"id": "recommendations", "type": "markdown", "sourceId": "research_bundle", "body": "## Recommendations\n\n1. Replace global identity LLM consolidation with deterministic normalization plus ambiguous-only refinement.\n2. Make session extraction local-first; invoke Qwen only for bounded ambiguous EventFrame/RoleFrame fields.\n3. Enforce short schemas and frame caps to eliminate length-truncated JSON.\n4. Within selected sessions, rerank raw SourceTurns with exact/BM25/dense and force frame→source closure.\n5. Add set-coverage slots for count/list/negative-scope and temporal endpoints.\n6. Run fixed-navigator paired ablations before deleting any channel, reporting exact turn all-hit, post-pack recall, token, visited nodes, and latency."},
            {"id": "questions", "type": "markdown", "body": "## Further Research Questions\n\n- Can deterministic extraction remove at least 80% of session-generation output while preserving temporal endpoint and collection closure?\n- Why do LoCoMo source-bound frames fail to enter fine candidates: embedding text, role filtering, quota, ranking, or optional-stage ordering?\n- What is the minimum sufficient scene window, and how much of late-scene coverage is redundant?\n- Should count/list retrieval optimize explicit set coverage instead of top-k node similarity?\n- Why are semantic-neighbor edges rarely consumed by the current navigator?"},
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": manifest["generatedAt"],
            "status": "ready",
            "datasets": {
                "headline": headline,
                "token_breakdown": token_rows,
                "recall": recall_rows,
                "recall_long": recall_long,
                "failure_stages": failure_rows,
                "feature_contribution": feature_rows,
                "turn_roles": role_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    chart_map = {
        "token_by_stage": {"question": "Where are backbone tokens spent?", "dataset": "token_breakdown", "encodings": "x=stage_label, y=total_tokens"},
        "recall_funnel": {"question": "Where does evidence completeness fall between session and packed turns?", "dataset": "recall_long", "encodings": "x=stratum, y=rate, color=metric"},
        "failures_by_stage": {"question": "At which earliest stage does each question fail exact all-hit?", "dataset": "failure_stages", "encodings": "x=stratum, y=questions, color=outcome"},
        "feature_gold_coverage": {"question": "Which trace features overlap packed exact-gold references?", "dataset": "feature_contribution", "encodings": "x=feature, y=packed_gold_refs"},
        "caveat": "Feature coverage is overlapping descriptive attribution, not causal ablation lift.",
    }
    (args.output.parent / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
