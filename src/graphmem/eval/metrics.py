from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain import NavigationResult
from ..storage import SQLiteGraphStore
from .devset import DevQuestion


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
    candidate_ids = {row.turn_id for row in result.candidate_scores}
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
    return {
        "question_id": question.question_id,
        "memory_id": question.memory_id,
        "benchmark": question.benchmark,
        "stratum": question.stratum,
        "session_any_hit": bool(session_hits),
        "session_all_hit": gold_sessions <= predicted_sessions,
        "session_recall": len(session_hits) / max(1, len(gold_sessions)),
        "turn_any_hit": bool(turn_hits),
        "turn_all_hit": gold <= predicted,
        "turn_recall": len(turn_hits) / max(1, len(gold)),
        "turn_precision": len(turn_hits) / max(1, len(predicted)),
        "candidate_turn_all_hit": gold <= candidate_turns,
        "candidate_turn_recall": len(candidate_hits) / max(1, len(gold)),
        "graph_reachable_turn_recall": len(graph_turns & gold) / max(1, len(gold)),
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
        "session_any_hit", "session_all_hit", "session_recall", "turn_any_hit",
        "turn_all_hit", "turn_recall", "turn_precision", "visited_nodes", "visited_edges",
        "frontier_peak", "evidence_tokens", "budget_exhausted", "certificate_complete",
        "stopped_by_certificate", "search_node_cap_reached", "search_edge_cap_reached",
        "search_hop_cap_reached", "search_frontier_truncated", "pack_turn_cap_reached",
        "pack_token_cap_reached",
        "candidate_turn_all_hit", "candidate_turn_recall", "graph_reachable_turn_recall",
        "path_provenance_complete", "proof_length",
    )
    return {"questions": len(rows), **{
        field: sum(float(row[field]) for row in rows) / max(1, len(rows))
        for field in mean_fields
    }}
