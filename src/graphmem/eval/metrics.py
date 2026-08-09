from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from typing import Any, Mapping, Sequence, TypeVar

from ..domain import NavigationResult
from ..storage import SQLiteGraphStore
from .devset import DevQuestion


T = TypeVar("T", bound=Hashable)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def retrieval_set_metrics(
    gold: set[T], predicted: set[T], *, prefix: str = "",
) -> dict[str, float | int | bool]:
    """Score one retrieved set against the official evidence annotations.

    Official benchmark evidence is sufficient but not guaranteed exhaustive.
    Precision reported here is therefore an *annotation-scoped lower bound*:
    an unlabelled but genuinely useful context turn is counted as a false
    positive.  It is still the right paired signal for detecting a method that
    raises recall by returning the entire memory, provided every arm uses the
    same annotations.
    """

    hits = len(gold & predicted)
    precision = _ratio(hits, len(predicted))
    recall = _ratio(hits, len(gold))
    return {
        f"{prefix}turns": len(predicted),
        f"{prefix}turn_hits": hits,
        f"{prefix}turn_false_positives": len(predicted - gold),
        f"{prefix}turn_precision": precision,
        f"{prefix}turn_recall": recall,
        f"{prefix}turn_f1": _f1(precision, recall),
        f"{prefix}turn_any_hit": bool(hits),
        # Preserve the historical vacuous value for unannotated questions;
        # callers must use ``has_turn_gold`` before aggregating quality.
        f"{prefix}turn_all_hit": gold <= predicted,
    }


def ranked_retrieval_metrics(
    gold: set[T], ranked: Iterable[T], *, cutoffs: Sequence[int] = (8, 16, 32),
    prefix: str = "candidate_",
) -> dict[str, float | int | bool]:
    """Expose ranking quality before an arbitrarily large reservoir saturates.

    ``all_hit`` over the full candidate reservoir is uninformative once the
    reservoir approaches the whole memory.  Top-K precision/recall and the rank
    of the last required gold item reveal whether the ordering can actually
    feed a bounded evidence pack.
    """

    ordered = tuple(dict.fromkeys(ranked))
    result: dict[str, float | int | bool] = {}
    ranks = [index for index, item in enumerate(ordered, 1) if item in gold]
    result[f"{prefix}first_gold_reciprocal_rank"] = _ratio(1, min(ranks)) if ranks else 0.0
    result[f"{prefix}last_gold_rank"] = max(ranks) if len(ranks) == len(gold) and gold else 0
    result[f"{prefix}average_precision"] = (
        sum(
            _ratio(hit_index, rank)
            for hit_index, rank in enumerate(ranks, 1)
        ) / len(gold) if gold else 0.0
    )
    r_cutoff = len(gold)
    result[f"{prefix}r_precision"] = (
        _ratio(len(gold & set(ordered[:r_cutoff])), r_cutoff)
        if r_cutoff else 0.0)
    for cutoff in cutoffs:
        result.update(retrieval_set_metrics(
            gold, set(ordered[:cutoff]), prefix=f"{prefix}top{cutoff}_"))
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(ordered[:cutoff], 1) if item in gold)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(gold), cutoff) + 1))
        result[f"{prefix}ndcg_at_{cutoff}"] = _ratio(dcg, ideal_dcg)
    return result


def navigation_metrics(question: DevQuestion, result: NavigationResult,
                       store: SQLiteGraphStore) -> dict[str, Any]:
    turns = {turn.turn_id: turn for turn in store.turns(question.memory_id)}
    predicted = {
        (turns[turn_id].session_id, turns[turn_id].turn_index)
        for turn_id in result.retrieved_turn_ids if turn_id in turns
    }
    gold = {(row.session_id, row.turn_index) for row in question.gold_turns}
    predicted_sessions = set(result.retrieved_session_ids)
    gold_sessions = set(question.gold_sessions)
    turn_hits = predicted & gold
    session_hits = predicted_sessions & gold_sessions
    candidate_ranked_ids = tuple(dict.fromkeys(
        row.turn_id for row in result.candidate_scores))
    candidate_ids = set(candidate_ranked_ids)
    candidate_turns = {
        (turns[turn_id].session_id, turns[turn_id].turn_index)
        for turn_id in candidate_ids if turn_id in turns
    }
    candidate_hits = candidate_turns & gold
    graph_turns: set[tuple[str, int]] = set()
    node_map = {node.node_id: node for node in store.nodes(question.memory_id)}
    for node_id in result.visited_path_node_ids:
        node = node_map.get(node_id)
        if not node:
            continue
        if node.attributes.get("provenance_scope", "terminal") != "terminal":
            continue
        for group_id in node.all_evidence_group_ids:
            group = store.evidence_group(group_id)
            if not group:
                continue
            for member in group.members:
                if member.turn_id in turns:
                    turn = turns[member.turn_id]
                    graph_turns.add((turn.session_id, turn.turn_index))
    if gold <= predicted:
        failure_stage = "success"
    elif not session_hits:
        failure_stage = "seed_miss"
    elif not gold_sessions <= predicted_sessions:
        failure_stage = "routing_miss"
    elif not gold <= candidate_turns:
        failure_stage = "within_session_candidate_miss"
    elif result.budget_exhausted:
        failure_stage = "pack_drop_or_budget_exhausted"
    else:
        failure_stage = "pack_drop"
    turn_metrics = retrieval_set_metrics(gold, predicted)
    candidate_metrics = retrieval_set_metrics(
        gold, candidate_turns, prefix="candidate_")
    graph_metrics = retrieval_set_metrics(
        gold, graph_turns, prefix="graph_reachable_")
    ranked_candidate_refs = tuple(
        (turns[turn_id].session_id, turns[turn_id].turn_index)
        for turn_id in candidate_ranked_ids if turn_id in turns)
    session_precision = _ratio(len(session_hits), len(predicted_sessions))
    session_recall = _ratio(len(session_hits), len(gold_sessions))
    candidate_count = len(candidate_turns)
    candidate_hits_count = len(candidate_hits)
    return {
        "question_id": question.question_id,
        "memory_id": question.memory_id,
        "benchmark": question.benchmark,
        "stratum": question.stratum,
        "session_any_hit": bool(session_hits),
        "session_all_hit": gold_sessions <= predicted_sessions,
        "session_recall": session_recall,
        "session_precision": session_precision,
        "session_f1": _f1(session_precision, session_recall),
        **turn_metrics,
        **candidate_metrics,
        **graph_metrics,
        **ranked_retrieval_metrics(gold, ranked_candidate_refs),
        # Selectivity distinguishes a focused retrieval from a full-memory
        # reservoir even when both have identical recall/all-hit.
        "candidate_selectivity": _ratio(candidate_count, len(turns)),
        "candidate_reduction": 1.0 - _ratio(candidate_count, len(turns)),
        "pack_selectivity_vs_candidate": _ratio(len(predicted), candidate_count),
        "pack_compression_vs_candidate": 1.0 - _ratio(len(predicted), candidate_count),
        "candidate_to_pack_gold_retention": _ratio(len(turn_hits), candidate_hits_count),
        "candidate_to_pack_recall_loss": (
            float(candidate_metrics["candidate_turn_recall"])
            - float(turn_metrics["turn_recall"])),
        "candidate_to_pack_precision_gain": (
            float(turn_metrics["turn_precision"])
            - float(candidate_metrics["candidate_turn_precision"])),
        "gold_to_candidate_expansion": _ratio(candidate_count, len(gold)),
        "gold_to_pack_expansion": _ratio(len(predicted), len(gold)),
        "path_provenance_complete": all(step.evidence_group_id for step in result.proof),
        "proof_length": len(result.proof),
        "failure_stage": failure_stage,
        "gold_turns": len(gold),
        "retrieved_turns": len(predicted),
        "visited_nodes": result.visited_nodes,
        "visited_edges": result.visited_edges,
        "frontier_peak": result.frontier_peak,
        "evidence_tokens": result.evidence_tokens,
        "budget_exhausted": result.budget_exhausted,
        "certificate_complete": bool(result.certificate and result.certificate.complete),
        "stopped_by_certificate": bool(result.stop_reason and str(result.stop_reason) == "certificate"),
        "search_node_cap_reached": bool(result.search_exhaustion.get("node_cap_reached")),
        "search_edge_cap_reached": bool(result.search_exhaustion.get("edge_cap_reached")),
        "search_hop_cap_reached": bool(result.search_exhaustion.get("hop_cap_reached")),
        "search_frontier_truncated": bool(result.search_exhaustion.get("frontier_truncated")),
        "pack_turn_cap_reached": bool(result.pack_exhaustion.get("turn_cap_reached")),
        "pack_token_cap_reached": bool(result.pack_exhaustion.get("token_cap_reached")),
        **{f"latency_{key}_ms": value for key, value in result.stage_latency_ms.items()},
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in sorted({str(row["stratum"]) for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        result[stratum] = _aggregate(subset)
    result["overall"] = _aggregate(rows)
    stratum_hits = [result[key]["turn_all_hit"] for key in result if key != "overall"]
    result["equal_stratum_turn_all_hit"] = sum(stratum_hits) / max(1, len(stratum_hits))
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    mean_fields = (
        "session_any_hit", "session_all_hit", "session_recall", "session_precision",
        "session_f1", "turn_any_hit", "turn_all_hit", "turn_recall", "turn_precision",
        "turn_f1", "visited_nodes", "visited_edges",
        "frontier_peak", "evidence_tokens", "budget_exhausted", "certificate_complete",
        "stopped_by_certificate", "search_node_cap_reached", "search_edge_cap_reached",
        "search_hop_cap_reached", "search_frontier_truncated", "pack_turn_cap_reached",
        "pack_token_cap_reached",
        "candidate_turn_all_hit", "candidate_turn_recall", "candidate_turn_precision",
        "candidate_turn_f1", "graph_reachable_turn_recall",
        "graph_reachable_turn_precision", "graph_reachable_turn_f1",
        "candidate_selectivity", "candidate_reduction", "pack_selectivity_vs_candidate",
        "pack_compression_vs_candidate", "candidate_to_pack_gold_retention",
        "candidate_to_pack_recall_loss", "candidate_to_pack_precision_gain",
        "gold_to_candidate_expansion", "gold_to_pack_expansion",
        "candidate_first_gold_reciprocal_rank", "candidate_last_gold_rank",
        "candidate_average_precision", "candidate_r_precision",
        "candidate_ndcg_at_8", "candidate_ndcg_at_16", "candidate_ndcg_at_32",
        "candidate_top8_turn_all_hit", "candidate_top8_turn_recall",
        "candidate_top8_turn_precision", "candidate_top8_turn_f1",
        "candidate_top16_turn_all_hit", "candidate_top16_turn_recall",
        "candidate_top16_turn_precision", "candidate_top16_turn_f1",
        "candidate_top32_turn_all_hit", "candidate_top32_turn_recall",
        "candidate_top32_turn_precision", "candidate_top32_turn_f1",
        "path_provenance_complete", "proof_length",
    )
    result: dict[str, float | int] = {"questions": len(rows), **{
        field: sum(float(row.get(field, 0.0)) for row in rows) / max(1, len(rows))
        for field in mean_fields
    }}
    # Micro metrics prevent a question with one gold turn and one with ten gold
    # turns from contributing identical weight to every count-based conclusion.
    for prefix in ("", "candidate_", "graph_reachable_"):
        gold_total = sum(int(row.get("gold_turns", 0)) for row in rows)
        predicted_total = sum(int(row.get(f"{prefix}turns", 0)) for row in rows)
        hit_total = sum(int(row.get(f"{prefix}turn_hits", 0)) for row in rows)
        precision = _ratio(hit_total, predicted_total)
        recall = _ratio(hit_total, gold_total)
        result[f"micro_{prefix}turn_precision"] = precision
        result[f"micro_{prefix}turn_recall"] = recall
        result[f"micro_{prefix}turn_f1"] = _f1(precision, recall)
    return result
