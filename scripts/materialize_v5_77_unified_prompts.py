#!/usr/bin/env python3
"""Build one immutable Graph+source-focus prompt per LoCoMo question."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer.ensemble import question_from_messages  # noqa: E402
from graphmem.answer.verified_readout import (  # noqa: E402
    OBLIGATION_CONTRACTS, build_unified_source_prompt,
)
from graphmem.answer.relation_labels import relation_route_hint  # noqa: E402
from graphmem.config import load_config  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.embedding import QwenEmbeddingIndex  # noqa: E402
from graphmem.retrieval.flat_plan import (  # noqa: E402
    SOURCE_FOCUS_VERSION, append_source_focus_witnesses,
    build_source_focus_plan, build_temporal_focus_plan,
    compile_obligation_query_view, compile_query_obligations,
    compile_source_query_views, query_route,
)
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--locomo-data", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument(
        "--retrieval", type=Path,
        help=("optional retrieval JSONL with candidate relation provenance; "
              "adds deterministic route-family labels to graph witnesses"))
    parser.add_argument(
        "--relation-label-scope", choices=("all", "semantic"), default="all",
        help=("label all graph-provenance candidates, or only the gated "
              "semantic fact/predicate witnesses"))
    parser.add_argument(
        "--relation-label-routes", default="",
        help=("optional comma-separated QueryIR routes allowed to expose "
              "relation labels; empty enables every route"))
    parser.add_argument(
        "--relation-label-obligations", default="",
        help=("optional comma-separated obligation tags allowed to expose "
              "relation labels; combines with route allowlists by OR"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--graph-turns", type=int, default=32)
    parser.add_argument("--focus-turns", type=int, default=32)
    parser.add_argument(
        "--adaptive-focus-turns", type=int, default=0,
        help=("larger focus budget used only when the frozen controller trace "
              "contains --adaptive-focus-reason; zero disables escalation"))
    parser.add_argument(
        "--adaptive-focus-reason", default="queryir_or_closure_escalation",
        help="controller reason that authorizes the larger focus budget")
    parser.add_argument(
        "--preserve-semantic-overlay", action="store_true",
        help=("keep the normal graph-turn limit for the frozen base and append "
              "only the bounded witnesses recorded by semantic_witness_overlay"))
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--packing-model", default=(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"))
    parser.add_argument("--dense-focus", action="store_true")
    parser.add_argument(
        "--focus-session-diversity", action="store_true",
        help=("reserve source-focus seeds across sessions for QueryIR closure "
              "risks before filling by score"))
    parser.add_argument(
        "--dense-focus-extra-turns", type=int, default=0,
        help=("append this many graph/primary-novel witnesses from a dense "
              "source view without replacing the lexical focus rank"))
    parser.add_argument(
        "--multi-view-dense-extra-turns", type=int, default=0,
        help=("append this many witnesses from one QueryIR-guided dense "
              "paraphrase view; query embeddings are batched"))
    parser.add_argument(
        "--obligation-dense-extra-turns", type=int, default=0,
        help=("append this many witnesses from a composable QueryIR-obligation "
              "dense view without replacing any existing lane"))
    parser.add_argument(
        "--obligation-dense-tags",
        default="reasoning_chain,exhaustive_set,alias_resolution",
        help=("comma-separated obligation tags allowed to activate the "
              "obligation-dense lane"))
    parser.add_argument(
        "--temporal-focus-extra-turns", type=int, default=0,
        help=("append this many normalized event/source-time witnesses when "
              "the query contains a resolvable absolute time"))
    parser.add_argument(
        "--temporal-focus-coarse-extra-turns", type=int, default=0,
        help=("larger reserve for week/month/year query intervals; zero "
              "keeps the base temporal reserve"))
    parser.add_argument(
        "--temporal-transition-extra-turns", type=int, default=0,
        help=("append a separate duration/event-transition or expanded "
              "temporal-concept view without replacing the base temporal "
              "witnesses"))
    parser.add_argument(
        "--relation-focus-extra-turns", type=int, default=0,
        help=("append this many witnesses from a conservative relation-word "
              "family view, without changing exact graph semantics"))
    parser.add_argument(
        "--relation-focus-closure-extra-turns", type=int, default=0,
        help=("larger relation-family reserve for list/count/multi-hop routes; "
              "zero keeps the base relation reserve"))
    parser.add_argument(
        "--relation-concept-extra-turns", type=int, default=0,
        help=("append this many QueryIR concept-coverage witnesses after the "
              "ordinary direct/dialogue relation lane"))
    parser.add_argument(
        "--relation-concept-closure-extra-turns", type=int, default=0,
        help=("larger QueryIR concept reserve for list/count/multi-hop "
              "closure risk; zero keeps the base concept reserve"))
    parser.add_argument(
        "--expanded-relation-concept-extra-turns", type=int, default=0,
        help=("append a separate broader QueryIR synonym/direction witness "
              "only when its seed order differs from the base concept view"))
    parser.add_argument(
        "--promote-focus-overlap", action="store_true",
        help=("move focus witnesses already present in the graph view into "
              "the final focus section without duplicating source turns"))
    parser.add_argument(
        "--rank-focus-sessions", action="store_true",
        help=("order source-focus session clusters by frozen seed rank while "
              "preserving source order inside each session"))
    parser.add_argument(
        "--focus-capsule-turns", type=int, default=0,
        help=("repeat this many top query-view source turns in a compact "
              "navigation capsule at the end of the prompt"))
    parser.add_argument(
        "--focus-capsule-graph-only", action="store_true",
        help=("repeat only high-focus turns currently buried in the graph "
              "section; focus-section turns are already near the answer"))
    parser.add_argument(
        "--focus-capsule-routes", default="",
        help=("optional comma-separated QueryIR routes allowed to emit the "
              "capsule; empty enables every route"))
    parser.add_argument(
        "--focus-capsule-obligations", default="",
        help=("optional comma-separated obligation tags that can enable the "
              "capsule even when the primary route is not allowlisted"))
    parser.add_argument(
        "--focus-capsule-max-chars", type=int, default=0,
        help=("maximum exact-source characters per repeated capsule turn; "
              "zero inherits the ordinary evidence-block limit"))
    parser.add_argument(
        "--focus-navigation-map", action="store_true",
        help=("append a compact route-aware map from retrieval-view types to "
              "visible source-memory IDs; the map contains no summaries"))
    parser.add_argument(
        "--focus-navigation-map-routes", default="",
        help=("optional comma-separated QueryIR routes allowed to emit the "
              "navigation map; empty enables every route"))
    parser.add_argument(
        "--focus-navigation-map-obligations", default="",
        help=("optional comma-separated obligation tags allowed to emit the "
              "navigation map; combines with route allowlists by OR"))
    parser.add_argument(
        "--query-obligation-contracts", action="store_true",
        help=("render composable question-only QueryIR duties and enable the "
              "bounded derivation contract where required"))
    parser.add_argument(
        "--focus-lossless-extra-chars", type=int, default=0,
        help=("bounded extra source characters used to restore complete short "
              "turns in the focus lane; zero preserves compact spans"))
    parser.add_argument(
        "--morph-focus-extra-turns", type=int, default=0,
        help=("append this many graph-novel witnesses from an independent "
              "morphology-normalized lexical view"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_57_lossless_atomic.json")
    parser.add_argument("--embedding-base-url",
                        default="http://127.0.0.1:8003/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--dense-sidecar-dir", type=Path)
    parser.add_argument(
        "--query-embedding-cache", type=Path,
        help=("optional persistent query-vector cache; avoids recomputing "
              "unchanged query views while replaying ranking policies"))
    parser.add_argument(
        "--query-embedding-prewarm-batch-size", type=int, default=256,
        help=("batch size used to prewarm immutable query views before "
              "per-question ranking"))
    args = parser.parse_args()
    if args.graph_turns < 0 or args.focus_turns < 0:
        raise ValueError("turn limits must be non-negative")
    if (args.adaptive_focus_turns
            and args.adaptive_focus_turns < args.focus_turns):
        raise ValueError("adaptive-focus-turns cannot be below focus-turns")
    if args.morph_focus_extra_turns < 0:
        raise ValueError("morph-focus-extra-turns must be non-negative")
    if args.dense_focus_extra_turns < 0:
        raise ValueError("dense-focus-extra-turns must be non-negative")
    if args.multi_view_dense_extra_turns < 0:
        raise ValueError("multi-view-dense-extra-turns must be non-negative")
    if args.obligation_dense_extra_turns < 0:
        raise ValueError("obligation-dense-extra-turns must be non-negative")
    if args.temporal_focus_extra_turns < 0:
        raise ValueError("temporal-focus-extra-turns must be non-negative")
    if args.temporal_focus_coarse_extra_turns < 0:
        raise ValueError(
            "temporal-focus-coarse-extra-turns must be non-negative")
    if args.temporal_transition_extra_turns < 0:
        raise ValueError(
            "temporal-transition-extra-turns must be non-negative")
    if args.relation_focus_extra_turns < 0:
        raise ValueError("relation-focus-extra-turns must be non-negative")
    if args.relation_focus_closure_extra_turns < 0:
        raise ValueError(
            "relation-focus-closure-extra-turns must be non-negative")
    if args.focus_capsule_turns < 0:
        raise ValueError("focus-capsule-turns must be non-negative")
    if args.focus_capsule_max_chars < 0:
        raise ValueError("focus-capsule-max-chars must be non-negative")
    if args.relation_concept_extra_turns < 0:
        raise ValueError(
            "relation-concept-extra-turns must be non-negative")
    if args.relation_concept_closure_extra_turns < 0:
        raise ValueError(
            "relation-concept-closure-extra-turns must be non-negative")
    if args.expanded_relation_concept_extra_turns < 0:
        raise ValueError(
            "expanded-relation-concept-extra-turns must be non-negative")
    if args.query_embedding_prewarm_batch_size <= 0:
        raise ValueError(
            "query-embedding-prewarm-batch-size must be positive")
    known_routes = {
        "aggregate", "inference", "list", "lookup", "multi_hop",
        "state", "temporal",
    }
    route_options = {
        "focus capsule": args.focus_capsule_routes,
        "focus navigation map": args.focus_navigation_map_routes,
        "relation label": args.relation_label_routes,
    }
    parsed_routes: dict[str, tuple[str, ...]] = {}
    for label, raw_routes in route_options.items():
        parsed = tuple(dict.fromkeys(
            value.strip() for value in raw_routes.split(",")
            if value.strip()))
        unknown = set(parsed) - known_routes
        if unknown:
            raise ValueError(f"unknown {label} routes: {sorted(unknown)}")
        parsed_routes[label] = parsed
    focus_capsule_routes = parsed_routes["focus capsule"]
    focus_navigation_map_routes = parsed_routes["focus navigation map"]
    relation_label_routes = parsed_routes["relation label"]
    obligation_options = {
        "obligation dense": args.obligation_dense_tags,
        "focus capsule": args.focus_capsule_obligations,
        "focus navigation map": args.focus_navigation_map_obligations,
        "relation label": args.relation_label_obligations,
    }
    parsed_obligations: dict[str, tuple[str, ...]] = {}
    known_obligations = set(OBLIGATION_CONTRACTS)
    for label, raw_values in obligation_options.items():
        parsed = tuple(dict.fromkeys(
            value.strip() for value in raw_values.split(",")
            if value.strip()))
        unknown = set(parsed) - known_obligations
        if unknown:
            raise ValueError(
                f"unknown {label} obligations: {sorted(unknown)}")
        parsed_obligations[label] = parsed
    obligation_dense_tags = parsed_obligations["obligation dense"]
    focus_capsule_obligations = parsed_obligations["focus capsule"]
    focus_navigation_map_obligations = parsed_obligations[
        "focus navigation map"]
    relation_label_obligations = parsed_obligations["relation label"]
    if args.dense_focus and args.dense_focus_extra_turns:
        raise ValueError(
            "--dense-focus and --dense-focus-extra-turns are mutually exclusive")

    prepared = rows(args.prepared)
    cases = {str(row["question_id"]): row for row in json.loads(
        args.locomo_data.read_text(encoding="utf-8"))
        if int(row["locomo_category"]) in {1, 2, 3, 4}}
    if args.expected and len(prepared) != args.expected:
        raise RuntimeError(f"expected {args.expected} prompts, got {len(prepared)}")
    if len({str(row["question_id"]) for row in prepared}) != len(prepared):
        raise RuntimeError("duplicate question IDs")
    retrieval = ({
        str(row["dev_question_id"]): row for row in rows(args.retrieval)
    } if args.retrieval is not None else {})
    if args.retrieval is not None and set(retrieval) != {
            str(row["question_id"]) for row in prepared}:
        raise RuntimeError("retrieval/prepared question IDs do not match")

    args.output_root.mkdir(parents=True, exist_ok=True)
    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model)
    embedding = None
    embedding_prewarm: dict[str, int | float] | None = None
    if (args.dense_focus or args.dense_focus_extra_turns
            or args.multi_view_dense_extra_turns
            or args.obligation_dense_extra_turns):
        embedding = QwenEmbeddingIndex(
            store, load_config(args.config), record_usage=False,
            model_id=args.embedding_model,
            request_model_id=args.embedding_model,
            base_url=args.embedding_base_url,
            query_cache_path=args.query_embedding_cache,
            dense_sidecar_dir=args.dense_sidecar_dir,
            dense_backend="auto", dense_cache_memories=10)
        prewarm_primary_views: list[str] = []
        prewarm_auxiliary_views: list[str] = []
        for source in prepared:
            question = question_from_messages(source["messages"])
            obligation_tags = frozenset(
                compile_query_obligations(question).tags)
            prewarm_primary_views.append(question)
            if args.multi_view_dense_extra_turns:
                prewarm_auxiliary_views.extend(
                    compile_source_query_views(question)[1:])
            if (args.obligation_dense_extra_turns
                    and obligation_tags.intersection(obligation_dense_tags)):
                prewarm_auxiliary_views.extend(
                    compile_obligation_query_view(question)[1:])
        embedding_prewarm = dict(embedding.prewarm_queries(
            (*prewarm_primary_views, *prewarm_auxiliary_views),
            batch_size=args.query_embedding_prewarm_batch_size))
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    adaptive_focus_questions = 0
    morph_focus_witnesses = 0
    dense_focus_witnesses = 0
    multi_view_dense_witnesses = 0
    obligation_dense_questions = 0
    obligation_dense_witnesses = 0
    temporal_focus_witnesses = 0
    temporal_focus_coarse_questions = 0
    temporal_transition_witnesses = 0
    temporal_transition_questions = 0
    relation_focus_witnesses = 0
    relation_focus_closure_questions = 0
    relation_concept_witnesses = 0
    relation_concept_closure_questions = 0
    expanded_relation_concept_witnesses = 0
    expanded_relation_concept_questions = 0
    try:
        for index, source in enumerate(prepared, 1):
            question_id = str(source["question_id"])
            case = cases[question_id]
            question = question_from_messages(source["messages"])
            obligations = compile_query_obligations(question)
            route = obligations.route
            obligation_tags = frozenset(obligations.tags)
            memory_turns = tuple(store.turns(str(source["memory_id"])))
            turns = {turn.turn_id: turn for turn in memory_turns}
            controller = dict(source.get("trace", {})).get(
                "adaptive_answer_budget", {})
            controller_reasons = frozenset(map(
                str, controller.get("reasons", ())))
            focus_turns = args.focus_turns
            focus_escalated = bool(
                args.adaptive_focus_turns
                and args.adaptive_focus_reason in controller_reasons)
            if focus_escalated:
                focus_turns = args.adaptive_focus_turns
                adaptive_focus_questions += 1
            graph_limit = args.graph_turns
            overlay = dict(source.get("trace", {})).get(
                "semantic_witness_overlay", {})
            if args.preserve_semantic_overlay:
                if not overlay:
                    raise RuntimeError(
                        f"{question_id}: missing semantic_witness_overlay trace")
                graph_limit = (
                    min(args.graph_turns, int(overlay["base_turns"]))
                    + int(overlay["inserted"]))
            dense_scores = None
            expanded_dense_scores = None
            obligation_dense_scores = None
            if embedding is not None:
                query_views = [question]
                expanded_dense_index = None
                obligation_dense_index = None
                if args.multi_view_dense_extra_turns:
                    for view in compile_source_query_views(question)[1:]:
                        if view not in query_views:
                            expanded_dense_index = len(query_views)
                            query_views.append(view)
                obligation_dense_enabled = bool(
                    args.obligation_dense_extra_turns
                    and obligation_tags.intersection(obligation_dense_tags))
                if obligation_dense_enabled:
                    for view in compile_obligation_query_view(question)[1:]:
                        if view not in query_views:
                            obligation_dense_index = len(query_views)
                            query_views.append(view)
                dense_results = embedding.search_many(
                    str(source["memory_id"]), tuple(
                        (view, min(96, len(memory_turns)))
                        for view in query_views))
                dense_scores = dict(dense_results[0])
                if expanded_dense_index is not None:
                    expanded_dense_scores = dict(
                        dense_results[expanded_dense_index])
                if obligation_dense_index is not None:
                    obligation_dense_scores = dict(
                        dense_results[obligation_dense_index])
                    obligation_dense_questions += 1
            focus = build_source_focus_plan(
                question=question, turns=turns,
                max_turns=focus_turns or 1,
                dense_scores=(dense_scores if args.dense_focus else None),
                session_diversity=args.focus_session_diversity)
            morph_extra_ids: tuple[str, ...] = ()
            if args.morph_focus_extra_turns:
                morph = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=focus_turns or 1, morphological=True)
                focus = append_source_focus_witnesses(
                    focus, morph,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=args.morph_focus_extra_turns,
                    question=question, turns=turns)
                morph_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                morph_focus_witnesses += len(morph_extra_ids)
            relation_extra_ids: tuple[str, ...] = ()
            relation_limit = args.relation_focus_extra_turns
            relation_closure_escalated = False
            if (relation_limit
                    or args.relation_focus_closure_extra_turns):
                relation = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(32, focus_turns or 1),
                    relation_families=True,
                    relation_concept_coverage=False)
                if (args.relation_focus_closure_extra_turns
                        and relation.trace.get("source_closure_risk")):
                    relation_limit = max(
                        relation_limit,
                        args.relation_focus_closure_extra_turns)
                    relation_closure_escalated = True
                    relation_focus_closure_questions += 1
            if relation_limit:
                focus = append_source_focus_witnesses(
                    focus, relation,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=relation_limit,
                    question=question, turns=turns)
                relation_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                relation_focus_witnesses += len(relation_extra_ids)
            relation_concept_extra_ids: tuple[str, ...] = ()
            relation_concept = None
            relation_concept_limit = args.relation_concept_extra_turns
            relation_concept_closure_escalated = False
            if (relation_concept_limit
                    or args.relation_concept_closure_extra_turns):
                relation_concept = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(32, focus_turns or 1),
                    relation_families=True,
                    relation_concept_coverage=True)
                if (args.relation_concept_closure_extra_turns
                        and relation_concept.trace.get("source_closure_risk")):
                    relation_concept_limit = max(
                        relation_concept_limit,
                        args.relation_concept_closure_extra_turns)
                    relation_concept_closure_escalated = True
                    relation_concept_closure_questions += 1
            if relation_concept_limit:
                focus = append_source_focus_witnesses(
                    focus, relation_concept,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=relation_concept_limit,
                    question=question, turns=turns)
                relation_concept_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                relation_concept_witnesses += len(
                    relation_concept_extra_ids)
            expanded_relation_concept_extra_ids: tuple[str, ...] = ()
            if args.expanded_relation_concept_extra_turns:
                expanded_relation_concept = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(32, focus_turns or 1),
                    relation_families=True,
                    relation_concept_coverage=True,
                    expanded_relation_concepts=True)
                if relation_concept is None:
                    relation_concept = build_source_focus_plan(
                        question=question, turns=turns,
                        max_turns=max(32, focus_turns or 1),
                        relation_families=True,
                        relation_concept_coverage=True)
                if (expanded_relation_concept.trace.get(
                        "expanded_relation_query_triggered")
                        and expanded_relation_concept.seed_turn_ids
                        != relation_concept.seed_turn_ids):
                    focus = append_source_focus_witnesses(
                        focus, expanded_relation_concept,
                        excluded_turn_ids=source.get(
                            "evidence_turn_ids", ())[:graph_limit],
                        max_extra_turns=(
                            args.expanded_relation_concept_extra_turns),
                        question=question, turns=turns)
                    expanded_relation_concept_extra_ids = tuple(
                        focus.trace["auxiliary_extra_turn_ids"])
                    expanded_relation_concept_witnesses += len(
                        expanded_relation_concept_extra_ids)
                    expanded_relation_concept_questions += 1
            dense_extra_ids: tuple[str, ...] = ()
            if args.dense_focus_extra_turns:
                dense = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=focus_turns or 1,
                    dense_scores=dense_scores)
                focus = append_source_focus_witnesses(
                    focus, dense,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=args.dense_focus_extra_turns)
                dense_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                dense_focus_witnesses += len(dense_extra_ids)
            multi_view_dense_extra_ids: tuple[str, ...] = ()
            if (args.multi_view_dense_extra_turns
                    and expanded_dense_scores is not None):
                multi_view_dense = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=focus_turns or 1,
                    dense_scores=expanded_dense_scores)
                focus = append_source_focus_witnesses(
                    focus, multi_view_dense,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=args.multi_view_dense_extra_turns)
                multi_view_dense_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                multi_view_dense_witnesses += len(
                    multi_view_dense_extra_ids)
            obligation_dense_extra_ids: tuple[str, ...] = ()
            if (args.obligation_dense_extra_turns
                    and obligation_dense_scores is not None):
                obligation_dense = build_source_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(32, focus_turns or 1),
                    dense_scores=obligation_dense_scores,
                    view_kind="obligation_dense")
                focus = append_source_focus_witnesses(
                    focus, obligation_dense,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=args.obligation_dense_extra_turns)
                obligation_dense_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                obligation_dense_witnesses += len(
                    obligation_dense_extra_ids)
            temporal_extra_ids: tuple[str, ...] = ()
            temporal = None
            temporal_limit = args.temporal_focus_extra_turns
            temporal_coarse_escalated = False
            if (temporal_limit
                    or args.temporal_focus_coarse_extra_turns):
                temporal = build_temporal_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(
                        focus_turns or 1,
                        32 if args.temporal_focus_coarse_extra_turns else 1),
                    dense_scores=dense_scores)
                if (args.temporal_focus_coarse_extra_turns
                        and temporal.trace.get("query_time_precision")
                        in {"week", "month", "year"}):
                    temporal_limit = max(
                        temporal_limit,
                        args.temporal_focus_coarse_extra_turns)
                    temporal_coarse_escalated = True
                    temporal_focus_coarse_questions += 1
            if temporal_limit:
                focus = append_source_focus_witnesses(
                    focus, temporal,
                    excluded_turn_ids=source.get(
                        "evidence_turn_ids", ())[:graph_limit],
                    max_extra_turns=temporal_limit)
                temporal_extra_ids = tuple(
                    focus.trace["auxiliary_extra_turn_ids"])
                temporal_focus_witnesses += len(temporal_extra_ids)
            temporal_transition_extra_ids: tuple[str, ...] = ()
            if args.temporal_transition_extra_turns:
                temporal_transition = build_temporal_focus_plan(
                    question=question, turns=turns,
                    max_turns=max(focus_turns or 1, 32),
                    dense_scores=dense_scores,
                    expanded_relation_concepts=True)
                if temporal is None:
                    temporal = build_temporal_focus_plan(
                        question=question, turns=turns,
                        max_turns=max(focus_turns or 1, 32),
                        dense_scores=dense_scores)
                if (temporal_transition.trace.get(
                        "expanded_temporal_query_triggered")
                        and temporal_transition.seed_turn_ids
                        and temporal_transition.seed_turn_ids
                        != temporal.seed_turn_ids):
                    focus = append_source_focus_witnesses(
                        focus, temporal_transition,
                        excluded_turn_ids=source.get(
                            "evidence_turn_ids", ())[:graph_limit],
                        max_extra_turns=(
                            args.temporal_transition_extra_turns))
                    temporal_transition_extra_ids = tuple(
                        focus.trace["auxiliary_extra_turn_ids"])
                    temporal_transition_witnesses += len(
                        temporal_transition_extra_ids)
                    temporal_transition_questions += 1
            retrieval_row = retrieval.get(question_id, {})
            semantic_witness_ids = frozenset(map(str, (
                *retrieval_row.get("semantic_fact_witness_turn_ids", ()),
                *retrieval_row.get(
                    "semantic_predicate_witness_turn_ids", ()),
            )))
            relation_hints = {
                str(candidate["turn_id"]): relation_route_hint(
                    candidate.get("source_channels", ()),
                    candidate.get("relation_contributions", ()))
                for candidate in retrieval_row.get("candidate_scores", ())
                if (args.relation_label_scope == "all" or
                    str(candidate["turn_id"]) in semantic_witness_ids)
            }
            relation_hints = {
                turn_id: hint for turn_id, hint in relation_hints.items()
                if hint}
            def policy_enabled(
                    enabled: bool, routes: tuple[str, ...],
                    obligation_allowlist: tuple[str, ...]) -> bool:
                if not enabled:
                    return False
                if not routes and not obligation_allowlist:
                    return True
                return (route in routes or bool(
                    obligation_tags.intersection(obligation_allowlist)))

            relation_labels_enabled = bool(
                args.retrieval is not None and policy_enabled(
                    True, relation_label_routes,
                    relation_label_obligations))
            if not relation_labels_enabled:
                relation_hints = {}
            navigation_map_enabled = policy_enabled(
                args.focus_navigation_map, focus_navigation_map_routes,
                focus_navigation_map_obligations)
            capsule_enabled = policy_enabled(
                bool(args.focus_capsule_turns), focus_capsule_routes,
                focus_capsule_obligations)
            rendered_obligations = (
                obligations.presentation_tags()
                if args.query_obligation_contracts else ())
            rendered = build_unified_source_prompt(
                question=question,
                question_date=str(case.get("question_date") or ""),
                graph_turn_ids=source.get("evidence_turn_ids", ()),
                focus_plan=focus, turns=turns,
                graph_limit=graph_limit,
                focus_limit=(focus_turns + len(morph_extra_ids)
                             + len(relation_extra_ids)
                             + len(relation_concept_extra_ids)
                             + len(expanded_relation_concept_extra_ids)
                             + len(dense_extra_ids)
                             + len(multi_view_dense_extra_ids)
                             + len(obligation_dense_extra_ids)
                             + len(temporal_extra_ids)
                             + len(temporal_transition_extra_ids)),
                relation_hints=relation_hints,
                promote_focus_overlap=args.promote_focus_overlap,
                focus_lossless_extra_chars=args.focus_lossless_extra_chars,
                rank_focus_sessions=args.rank_focus_sessions,
                focus_capsule_turns=(
                    args.focus_capsule_turns if capsule_enabled else 0),
                focus_capsule_graph_only=args.focus_capsule_graph_only,
                focus_capsule_routes=(),
                focus_capsule_max_chars=args.focus_capsule_max_chars,
                focus_navigation_map=navigation_map_enabled,
                query_obligations=rendered_obligations)
            messages = list(rendered.messages)
            payload_hash = hashlib.sha256(
                canonical_json(messages).encode()).hexdigest()
            prompt_tokens = sum(counter.count_many(
                [str(message["content"]) for message in messages]))
            row = dict(source)
            row.update({
                "messages": messages,
                "evidence_turn_ids": list(rendered.evidence_turn_ids),
                "dropped_turn_ids": [],
                "evidence_tokens": prompt_tokens,
                "packing_prompt_tokens": prompt_tokens,
                "prompt_hash": hashlib.sha256((
                    SOURCE_FOCUS_VERSION + str(messages[0]["content"])
                ).encode()).hexdigest(),
                "prompt_payload_hash": payload_hash,
                "trace": {
                    **dict(source.get("trace", {})),
                    "unified_source_readout": dict(rendered.trace),
                },
            })
            output.append(row)
            audit.append({
                "question_id": question_id,
                "query_route": route,
                "query_obligations": list(obligations.tags),
                "rendered_query_obligations": list(rendered_obligations),
                "source_prompt_hash": source.get("prompt_payload_hash"),
                "unified_prompt_hash": payload_hash,
                "source_turns": len(source.get("evidence_turn_ids", ())),
                "unified_turns": len(rendered.evidence_turn_ids),
                "graph_turns": int(rendered.trace["graph_turns"]),
                "focus_turns": int(rendered.trace["focus_turns"]),
                "focus_turn_limit": focus_turns,
                "focus_escalated": focus_escalated,
                "focus_session_diversity": bool(
                    focus.trace.get("session_diversity")),
                "morph_focus_extra_turns": len(morph_extra_ids),
                "relation_focus_extra_turns": len(relation_extra_ids),
                "relation_focus_extra_limit": relation_limit,
                "relation_focus_closure_escalated": (
                    relation_closure_escalated),
                "relation_concept_extra_turns": len(
                    relation_concept_extra_ids),
                "relation_concept_extra_limit": relation_concept_limit,
                "relation_concept_closure_escalated": (
                    relation_concept_closure_escalated),
                "expanded_relation_concept_extra_turns": len(
                    expanded_relation_concept_extra_ids),
                "dense_focus_extra_turns": len(dense_extra_ids),
                "multi_view_dense_extra_turns": len(
                    multi_view_dense_extra_ids),
                "obligation_dense_extra_turns": len(
                    obligation_dense_extra_ids),
                "temporal_focus_extra_turns": len(temporal_extra_ids),
                "temporal_focus_extra_limit": temporal_limit,
                "temporal_focus_coarse_escalated": (
                    temporal_coarse_escalated),
                "temporal_transition_extra_turns": len(
                    temporal_transition_extra_ids),
                "relation_labels_enabled": relation_labels_enabled,
                "focus_navigation_map_enabled": navigation_map_enabled,
                "focus_capsule_enabled": capsule_enabled,
                "prompt_tokens": prompt_tokens,
            })
            if index % 200 == 0:
                print(f"materialized {index}/{len(prepared)}", flush=True)
    finally:
        store.close()

    for name, values in (("prepared_answers.jsonl", output),
                         ("audit.jsonl", audit)):
        (args.output_root / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in values),
            encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.77-unified-source-prompt-v1",
        "questions": len(output),
        "one_prompt_per_question": True,
        "uses_gold_or_judge": False,
        "graph_turn_limit": args.graph_turns,
        "preserve_semantic_overlay": args.preserve_semantic_overlay,
        "focus_turn_limit": args.focus_turns,
        "adaptive_focus_turn_limit": args.adaptive_focus_turns or None,
        "adaptive_focus_reason": (
            args.adaptive_focus_reason if args.adaptive_focus_turns else None),
        "adaptive_focus_questions": adaptive_focus_questions,
        "relation_path_labels": args.retrieval is not None,
        "relation_label_scope": (
            args.relation_label_scope if args.retrieval is not None else None),
        "relation_label_routes": list(relation_label_routes),
        "relation_label_obligations": list(relation_label_obligations),
        "query_obligation_contracts": args.query_obligation_contracts,
        "dense_focus": args.dense_focus,
        "focus_session_diversity": args.focus_session_diversity,
        "dense_focus_extra_turn_limit": args.dense_focus_extra_turns,
        "dense_focus_witnesses": dense_focus_witnesses,
        "multi_view_dense_extra_turn_limit": (
            args.multi_view_dense_extra_turns),
        "multi_view_dense_witnesses": multi_view_dense_witnesses,
        "obligation_dense_extra_turn_limit": (
            args.obligation_dense_extra_turns),
        "obligation_dense_tags": list(obligation_dense_tags),
        "obligation_dense_questions": obligation_dense_questions,
        "obligation_dense_witnesses": obligation_dense_witnesses,
        "query_embedding_prewarm_batch_size": (
            args.query_embedding_prewarm_batch_size),
        "query_embedding_prewarm": embedding_prewarm,
        "temporal_focus_extra_turn_limit": args.temporal_focus_extra_turns,
        "temporal_focus_coarse_extra_turn_limit": (
            args.temporal_focus_coarse_extra_turns),
        "temporal_focus_coarse_questions": temporal_focus_coarse_questions,
        "temporal_focus_witnesses": temporal_focus_witnesses,
        "temporal_transition_extra_turn_limit": (
            args.temporal_transition_extra_turns),
        "temporal_transition_questions": temporal_transition_questions,
        "temporal_transition_witnesses": temporal_transition_witnesses,
        "promote_focus_overlap": args.promote_focus_overlap,
        "rank_focus_sessions": args.rank_focus_sessions,
        "focus_capsule_turn_limit": args.focus_capsule_turns,
        "focus_capsule_graph_only": args.focus_capsule_graph_only,
        "focus_capsule_routes": list(focus_capsule_routes),
        "focus_capsule_obligations": list(focus_capsule_obligations),
        "focus_capsule_max_chars": args.focus_capsule_max_chars or None,
        "focus_navigation_map": args.focus_navigation_map,
        "focus_navigation_map_routes": list(focus_navigation_map_routes),
        "focus_navigation_map_obligations": list(
            focus_navigation_map_obligations),
        "focus_lossless_extra_chars": args.focus_lossless_extra_chars,
        "morph_focus_extra_turn_limit": args.morph_focus_extra_turns,
        "morph_focus_witnesses": morph_focus_witnesses,
        "relation_focus_extra_turn_limit": args.relation_focus_extra_turns,
        "relation_focus_closure_extra_turn_limit": (
            args.relation_focus_closure_extra_turns),
        "relation_focus_closure_questions": (
            relation_focus_closure_questions),
        "relation_focus_witnesses": relation_focus_witnesses,
        "relation_concept_extra_turn_limit": (
            args.relation_concept_extra_turns),
        "relation_concept_closure_extra_turn_limit": (
            args.relation_concept_closure_extra_turns),
        "relation_concept_closure_questions": (
            relation_concept_closure_questions),
        "relation_concept_witnesses": relation_concept_witnesses,
        "expanded_relation_concept_extra_turn_limit": (
            args.expanded_relation_concept_extra_turns),
        "expanded_relation_concept_questions": (
            expanded_relation_concept_questions),
        "expanded_relation_concept_witnesses": (
            expanded_relation_concept_witnesses),
        "unique_prompt_hashes": len({row["prompt_payload_hash"] for row in output}),
        "inputs": {
            "prepared": {"path": str(args.prepared), "sha256": digest(args.prepared)},
            "locomo_data": {"path": str(args.locomo_data), "sha256": digest(args.locomo_data)},
            "source_db": {"path": str(args.source_db), "sha256": digest(args.source_db)},
            "retrieval": ({
                "path": str(args.retrieval), "sha256": digest(args.retrieval)
            } if args.retrieval is not None else None),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
