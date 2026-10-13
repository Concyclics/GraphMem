"""Online V5.81 Graph + Source-focus composition, independent of benchmark files.

Uses the same planners/renderers as materialize_v5_77_unified_prompts. The graph
prefix comes from the live adaptive navigator; source views have separate,
bounded reserves. Presentation never changes the graph evidence selection.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Mapping

from .relation_labels import relation_route_hint
from .stage import PreparedAnswer
from .verified_readout import build_unified_source_prompt
from ..domain import NavigationResult, SourceTurn, canonical_json
from ..retrieval.flat_plan import (
    append_source_focus_witnesses, build_source_focus_plan,
    build_temporal_focus_plan, compile_query_obligations,
)
from ..tokenization import TokenCounter


VERSION = "graphmem-v5.81-online-source-focus-v1"


def prepare_optimized_readout(
    prepared: PreparedAnswer, *, question: str, question_date: str | None,
    turns: Mapping[str, SourceTurn], navigation: NavigationResult,
    counter: TokenCounter, max_prompt_tokens: int,
    embedding: Any | None = None,
) -> PreparedAnswer:
    """Freeze one source-grounded prompt without previous answers or labels."""
    if prepared.deterministic_prediction or not prepared.evidence_turn_ids:
        return prepared
    obligations = compile_query_obligations(question)
    graph = prepared.evidence_turn_ids
    focus = build_source_focus_plan(question=question, turns=turns, max_turns=16)
    lanes: dict[str, list[str]] = {}

    def append(name, auxiliary, limit, *, contextual=False):
        nonlocal focus
        options = {"question": question, "turns": turns} if contextual else {}
        focus = append_source_focus_witnesses(
            focus, auxiliary, excluded_turn_ids=graph,
            max_extra_turns=limit, **options)
        lanes[name] = list(focus.trace.get("auxiliary_extra_turn_ids", ()))

    append("morphological", build_source_focus_plan(
        question=question, turns=turns, max_turns=16, morphological=True), 2,
        contextual=True)
    relation = build_source_focus_plan(
        question=question, turns=turns, max_turns=32,
        relation_families=True, relation_concept_coverage=False)
    append("relation", relation, 4 if relation.trace.get("source_closure_risk") else 2,
           contextual=True)
    concept = build_source_focus_plan(
        question=question, turns=turns, max_turns=32,
        relation_families=True, relation_concept_coverage=True)
    append("concept", concept, 2 if concept.trace.get("source_closure_risk") else 1,
           contextual=True)
    expanded = build_source_focus_plan(
        question=question, turns=turns, max_turns=32, relation_families=True,
        relation_concept_coverage=True, expanded_relation_concepts=True)
    if (expanded.trace.get("expanded_relation_query_triggered")
            and expanded.seed_turn_ids != concept.seed_turn_ids):
        append("expanded_concept", expanded, 1, contextual=True)
    dense_scores = None
    if embedding is not None:
        # The index persists query vectors and batches misses through the
        # existing query cache; no new LLM call is introduced by this lane.
        dense_scores = dict(embedding.search(prepared.memory_id, question, min(96, len(turns))))
        append("dense", build_source_focus_plan(
            question=question, turns=turns, max_turns=16, dense_scores=dense_scores), 2)
    temporal = build_temporal_focus_plan(
        question=question, turns=turns, max_turns=32, dense_scores=dense_scores)
    append("temporal", temporal,
           3 if temporal.trace.get("query_time_precision") in {"week", "month", "year"} else 2)
    transition = build_temporal_focus_plan(
        question=question, turns=turns, max_turns=32, dense_scores=dense_scores,
        expanded_relation_concepts=True)
    if (transition.trace.get("expanded_temporal_query_triggered")
            and transition.seed_turn_ids and transition.seed_turn_ids != temporal.seed_turn_ids):
        append("temporal_transition", transition, 2)

    semantic_ids = set(navigation.trace.get("semantic_fact_witness_turn_ids", ()))
    semantic_ids.update(navigation.trace.get("semantic_predicate_witness_turn_ids", ()))
    hints = {}
    if obligations.route == "inference":
        hints = {row.turn_id: relation_route_hint(row.source_channels, row.relation_contributions)
                 for row in navigation.candidate_scores if row.turn_id in semantic_ids}
        hints = {key: value for key, value in hints.items() if value}
    focus_limit = 16 + sum(len(ids) for ids in lanes.values())
    initial_focus_limit = focus_limit
    while True:
        rendered = build_unified_source_prompt(
            question=question, question_date=question_date or "", turns=turns,
            graph_turn_ids=graph, graph_limit=len(graph), focus_plan=focus,
            focus_limit=focus_limit, relation_hints=hints,
            focus_lossless_extra_chars=1600, focus_capsule_turns=1,
            focus_capsule_graph_only=True, focus_capsule_max_chars=360,
            focus_capsule_routes=("lookup", "list", "temporal"),
            focus_navigation_map=obligations.route == "inference",
            query_obligations=obligations.presentation_tags())
        tokens = sum(counter.count_many([str(row["content"]) for row in rendered.messages]))
        if tokens <= max_prompt_tokens:
            break
        if focus_limit == 0:
            # Preserve the already bounded graph prompt instead of deleting
            # graph witnesses merely to fit additional presentation material.
            return replace(prepared, trace={**prepared.trace, "optimized_readout": {
                "version": VERSION, "fallback": "v5_73_prompt_budget",
                "query_obligations": list(obligations.tags)}})
        focus_limit -= 1
    if not set(graph).issubset(rendered.evidence_turn_ids):
        raise RuntimeError("optimized readout removed graph evidence")
    evidence_tokens = sum(counter.count_many(list(rendered.evidence_blocks)))
    return replace(
        prepared, messages=rendered.messages, evidence_turn_ids=rendered.evidence_turn_ids,
        dropped_turn_ids=tuple(t for t in prepared.dropped_turn_ids
                               if t not in rendered.evidence_turn_ids),
        evidence_tokens=evidence_tokens, packing_prompt_tokens=tokens,
        prompt_payload_hash=hashlib.sha256(canonical_json(rendered.messages).encode()).hexdigest(),
        prompt_hash=hashlib.sha256((VERSION + str(rendered.messages[0]["content"])).encode()).hexdigest(),
        trace={"base_readout": dict(prepared.trace), "prompt_version": VERSION,
               "unified_source_readout": dict(rendered.trace),
               "optimized_readout": {
                   "version": VERSION, "query_obligations": list(obligations.tags),
                   "presentation_tags": list(obligations.presentation_tags()),
                   "graph_turns": len(graph), "focus_limit": focus_limit,
                   "focus_budget_reduced": focus_limit < initial_focus_limit,
                   "auxiliary_candidates": lanes, "dense_focus": embedding is not None,
                   "token_counter": counter.describe(), "uses_gold_or_judge": False}})
