from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..domain import SourceTurn
from ..text import content_terms
from .query_ir import QueryIR


DenseSearch = Callable[[str, str, int], Sequence[tuple[str, float]]]


@dataclass(frozen=True, slots=True)
class SeedResult:
    semantic_node_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    raw_scores: Mapping[str, Mapping[str, float]]
    operand_nodes: Mapping[str, tuple[str, ...]]
    operand_turns: Mapping[str, tuple[str, ...]]


def _rrf(ranked: Mapping[str, Sequence[str]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rows in ranked.values():
        for rank, item in enumerate(rows, 1):
            scores[item] += 1.0 / (k + rank)
    return scores


def seed_operands(store, view, memory_id: str, ir: QueryIR, turns: Sequence[SourceTurn], *,
                  dense_search: DenseSearch | None, use_rrf: bool, use_postings: bool) -> SeedResult:
    query_terms = content_terms(ir.query)
    raw_scores: dict[str, dict[str, float]] = defaultdict(lambda: {"exact": 0.0, "bm25": 0.0, "dense": 0.0})
    exact = sorted((len(query_terms & content_terms(turn.raw_text)), turn.turn_id) for turn in turns)
    exact_ids = [turn_id for score, turn_id in sorted(exact, reverse=True) if score][:96]
    for rank, turn_id in enumerate(exact_ids, 1): raw_scores[turn_id]["exact"] = 1.0 / rank
    bm25_ids = []
    for rank, (turn_id, score) in enumerate(store.search_turns(memory_id, ir.query, limit=96), 1):
        bm25_ids.append(turn_id); raw_scores[turn_id]["bm25"] = 1.0 / rank
    dense_ids = []
    if dense_search:
        for rank, (turn_id, score) in enumerate(dense_search(memory_id, ir.query, 96), 1):
            dense_ids.append(turn_id); raw_scores[turn_id]["dense"] = 1.0 / rank
    operand_nodes: dict[str, tuple[str, ...]] = {}; operand_turns: dict[str, tuple[str, ...]] = {}
    all_nodes: list[str] = []; all_turns: list[str] = []
    for operand in ir.operands:
        owner_ids = set().union(*(set(view.owner_alias_index.get(alias.casefold(), ()))
                                  for alias in operand.owner_aliases))
        # Owner entities are first-class graph anchors: a scheduler can reach a
        # manifest even when the direct fact posting is incomplete.
        fact_ids = list(sorted(owner_ids))
        fact_ids.extend(view.lookup_facts(owner_ids=tuple(owner_ids), predicates=operand.predicate_candidates, limit=24))
        if use_postings:
            keys = (*operand.owner_aliases, *operand.predicate_candidates, *sorted(query_terms))
            fact_ids.extend(view.route_children(keys, limit=24))
        node_ids = tuple(dict.fromkeys(node_id for node_id in fact_ids if node_id in view.nodes))[:48]
        operand_nodes[operand.operand_id] = node_ids; all_nodes.extend(node_ids)
        channels = {"exact": exact_ids[:8], "bm25": bm25_ids[:8], "dense": dense_ids[:8]}
        ranked = _rrf(channels) if use_rrf else {item: 1.0 for rows in channels.values() for item in rows}
        # RRF ranks the union; it must not silently erase a channel's bounded
        # rescue quota.  Each channel contributes at most eight turns.
        selected = [item for item, _ in sorted(ranked.items(), key=lambda row: (-row[1], row[0]))[:24]]
        if use_rrf:
            # A selected session is only a scope prior.  Retain its best local
            # turns, never the complete session as the legacy navigator did.
            session_scores: dict[str, float] = defaultdict(float)
            for turn_id in selected:
                turn = next((row for row in turns if row.turn_id == turn_id), None)
                if turn:
                    session_scores[turn.session_id] = max(session_scores[turn.session_id], ranked.get(turn_id, 0.0))
            top_sessions = [item for item, _ in sorted(session_scores.items(), key=lambda row: (-row[1], row[0]))[:4]]
            for session_id in top_sessions:
                local = []
                for turn in turns:
                    if turn.session_id != session_id:
                        continue
                    lexical = len(query_terms & content_terms(turn.raw_text)) / max(1, len(query_terms))
                    channel = sum(raw_scores.get(turn.turn_id, {}).values())
                    local.append((channel + lexical, turn.turn_id))
                selected.extend(turn_id for _, turn_id in sorted(local, key=lambda row: (-row[0], row[1]))[:8])
        selected = tuple(dict.fromkeys(selected))
        operand_turns[operand.operand_id] = selected; all_turns.extend(selected)
    return SeedResult(tuple(dict.fromkeys(all_nodes))[:48], tuple(dict.fromkeys(all_turns)), raw_scores,
                      operand_nodes, operand_turns)
