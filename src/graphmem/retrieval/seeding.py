"""Candidate seeding for the V5.6 harness.

V5.5 ran the three source channels once over the whole question, kept only the
top eight of each, fused them to 24, and truncated the semantic nodes of *all*
operands to a shared 48.  Measured against the frozen graph that left a
candidate pool averaging 45 turns where the legacy navigator had 278, and the
pool was never once a superset of the legacy one (see docs/V5_6_REBASELINE.md).

The reservoir here is deliberately wide and holds ids only.  It reproduces the
legacy pool by construction, then adds per-operand views on top, so candidate
coverage can no longer regress at this stage.  Narrowing is the packer's job,
where it is budget-driven and measured.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..domain import SourceTurn, stable_id
from ..text import content_terms, normalize_key
from .query_ir import QueryIR


DenseSearch = Callable[[str, str, int], Sequence[tuple[str, float]]]

# Depths that reproduce the legacy navigator's reach.
LEGACY_BM25_DEPTH = 96
LEGACY_DENSE_DEPTH = 96
# The legacy navigator floods the eight strongest sessions using its own score
# scale and its own tokenizer.  Reproducing that ranking exactly is brittle, so
# the reservoir floods a wider band and lets the margin absorb the difference.
LEGACY_SESSION_FANOUT = 8
SESSION_FANOUT_MARGIN = 6
# Depths for the narrower per-operand views layered on top.
VIEW_BM25_DEPTH = 48
VIEW_DENSE_DEPTH = 48
# Turn-level channels only.  Typed postings retrieve graph nodes, not turns, so
# they contribute through the semantic seeds rather than through this scoreboard.
CHANNELS = ("exact", "bm25", "dense")


@dataclass(frozen=True, slots=True)
class QueryView:
    """One retrieval angle on the question.

    A single whole-question string cannot express "John + volunteer" and
    "Maria + volunteer" as separate needs, which is why V5.5 gave every operand
    the same source ranking and differed only on graph postings.
    """

    view_id: str
    kind: str
    text: str
    operand_id: str | None = None
    dense: bool = False


@dataclass(frozen=True, slots=True)
class ReservoirEntry:
    turn_id: str
    session_id: str
    scores: Mapping[str, float]
    ranks: Mapping[str, int]
    view_ids: tuple[str, ...]
    operand_ids: tuple[str, ...]
    rrf: float
    parity: bool


@dataclass(frozen=True, slots=True)
class SeedResult:
    semantic_node_ids: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    raw_scores: Mapping[str, Mapping[str, float]]
    operand_nodes: Mapping[str, tuple[str, ...]]
    operand_turns: Mapping[str, tuple[str, ...]]
    reservoir: tuple[ReservoirEntry, ...] = ()
    views: tuple[QueryView, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)


def _rrf(ranked: Mapping[str, Sequence[str]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rows in ranked.values():
        for rank, item in enumerate(rows, 1):
            scores[item] += 1.0 / (k + rank)
    return scores


def _phrase(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_views(ir: QueryIR, *, max_per_operand: int) -> tuple[QueryView, ...]:
    """Compose the retrieval angles for a compiled question, deterministically.

    Fields that only PR2's compiler fills (scope, answer slot, temporal phrase)
    are read defensively so this degrades to owner/predicate views until then.
    """
    views: list[QueryView] = [QueryView(stable_id("view", ir.query, "full"), "full_query", ir.query,
                                        None, dense=True)]
    for operand in ir.operands:
        owner = _phrase(*operand.owner_aliases)
        predicates = tuple(operand.predicate_candidates[:2])
        scopes = tuple(getattr(operand, "scope_candidates", ()) or ())
        answer_slot = str(getattr(operand, "answer_slot", "") or "")
        temporal = str(getattr(operand, "temporal_constraint", "") or "")
        event = str(getattr(operand, "event_identity", "") or "")
        rows: list[tuple[str, str, bool]] = []
        if owner:
            rows.append(("owner", owner, False))
        for predicate in predicates:
            rows.append(("predicate", predicate, False))
        if owner and predicates:
            # The one composed view worth an embedding call: it is the closest
            # thing to the operand's actual information need.
            rows.append(("owner_predicate", _phrase(owner, predicates[0]), True))
        if predicates and answer_slot:
            rows.append(("predicate_answer_slot", _phrase(predicates[0], answer_slot), False))
        if predicates and scopes:
            rows.append(("scope_predicate", _phrase(scopes[0], predicates[0]), False))
        if temporal or event:
            rows.append(("temporal_event", _phrase(temporal, event), False))
        if not rows:
            continue
        seen: set[str] = set()
        kept = 0
        for kind, text, dense in rows:
            key = normalize_key(text)
            if not key or key in seen or kept >= max_per_operand:
                continue
            seen.add(key); kept += 1
            views.append(QueryView(stable_id("view", ir.query, operand.operand_id, kind),
                                   kind, text, operand.operand_id, dense))
    return tuple(views)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class _ChannelRunner:
    """Runs exact/BM25/dense for a view and keeps both true scores and ranks."""

    def __init__(self, store, memory_id: str, turns: Sequence[SourceTurn],
                 turn_terms: Mapping[str, frozenset[str]], dense_search: DenseSearch | None) -> None:
        self.store, self.memory_id, self.turns = store, memory_id, turns
        self.turn_terms, self.dense_search = turn_terms, dense_search

    def exact(self, text: str, *, limit: int | None) -> list[tuple[str, float]]:
        query_terms = content_terms(text)
        if not query_terms:
            return []
        lowered = " ".join(sorted(query_terms))
        rows: list[tuple[float, str]] = []
        for turn in self.turns:
            overlap = len(query_terms & self.turn_terms[turn.turn_id])
            if not overlap:
                continue
            score = overlap / len(query_terms)
            if lowered and lowered in turn.raw_text.casefold():
                score += 0.5
            rows.append((score, turn.turn_id))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return [(turn_id, score) for score, turn_id in (rows[:limit] if limit else rows)]

    def bm25(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        return [(turn_id, max(0.0, score))
                for turn_id, score in self.store.search_turns(self.memory_id, text, limit=limit)]

    def dense(self, text: str, *, limit: int) -> list[tuple[str, float]]:
        if not self.dense_search:
            return []
        return [(turn_id, float(score)) for turn_id, score in
                self.dense_search(self.memory_id, text, limit)]


def _operand_owner_ids(view, operand) -> tuple[str, ...]:
    rows: set[str] = set()
    for alias in operand.owner_aliases:
        rows.update(view.owner_alias_index.get(alias.casefold(), ()))
    return tuple(sorted(rows))


def _operand_nodes(view, operand, owner_ids: Sequence[str], query_terms: frozenset[str], *,
                   use_postings: bool, limit: int) -> tuple[str, ...]:
    """Structured seeds for one operand, ranked rather than sliced alphabetically."""
    node_ids: list[str] = list(owner_ids)
    node_ids.extend(view.lookup_facts(
        owner_ids=tuple(owner_ids), predicates=operand.predicate_candidates,
        scopes=tuple(getattr(operand, "scope_candidates", ()) or ()),
        limit=limit, rank_terms=query_terms))
    collections = getattr(view, "lookup_collections", None)
    if collections is not None:
        node_ids.extend(collections(owner_ids=tuple(owner_ids),
                                    predicates=operand.predicate_candidates, limit=limit))
    if use_postings:
        keys = (*operand.owner_aliases, *operand.predicate_candidates, *sorted(query_terms))
        node_ids.extend(view.route_children(keys, limit=limit))
    return tuple(dict.fromkeys(node_id for node_id in node_ids if node_id in view.nodes))[:limit]


def _lexical_nodes(view, query_terms: frozenset[str], *, limit: int) -> tuple[str, ...]:
    """Nodes whose summary overlaps the question, regardless of node type.

    The structured postings only reach nodes that carry a matching owner or
    predicate attribute.  The legacy navigator seeded purely on summary overlap,
    and that is where the last of its provenance reach comes from.
    """
    if not query_terms:
        return ()
    rows: list[tuple[float, str]] = []
    for node_id, node in view.nodes.items():
        overlap = len(query_terms & content_terms(node.summary))
        if overlap:
            rows.append((overlap / len(query_terms) + float(node.confidence) * 0.05, node_id))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return tuple(node_id for _, node_id in rows[:limit])


def _interleave(rows: Mapping[str, Sequence[str]], limit: int) -> tuple[str, ...]:
    """Round-robin merge so no single operand can consume the shared budget."""
    result: list[str] = []
    seen: set[str] = set()
    cursors = {key: 0 for key in rows}
    while len(result) < limit:
        advanced = False
        for key in sorted(rows):
            index = cursors[key]
            items = rows[key]
            while index < len(items) and items[index] in seen:
                index += 1
            cursors[key] = index + 1
            if index < len(items):
                advanced = True
                seen.add(items[index]); result.append(items[index])
                if len(result) >= limit:
                    break
        if not advanced:
            break
    return tuple(result)


def _seed_narrow(store, view, memory_id: str, ir: QueryIR, turns: Sequence[SourceTurn], *,
                 dense_search: DenseSearch | None, use_rrf: bool, use_postings: bool) -> SeedResult:
    """The frozen V5.5 seeding, kept verbatim so H2-H6 stay interpretable.

    An ablation ladder is only readable if its earlier rungs do not move, so the
    wide reservoir is a new rung rather than a change to the existing ones.
    """
    query_terms = content_terms(ir.query)
    turn_by_id = {turn.turn_id: turn for turn in turns}
    turns_by_session: dict[str, list[SourceTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_session[turn.session_id].append(turn)
    raw_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: {"exact": 0.0, "bm25": 0.0, "dense": 0.0})
    exact = sorted((len(query_terms & content_terms(turn.raw_text)), turn.turn_id) for turn in turns)
    exact_ids = [turn_id for score, turn_id in sorted(exact, reverse=True) if score][:96]
    for rank, turn_id in enumerate(exact_ids, 1):
        raw_scores[turn_id]["exact"] = 1.0 / rank
    bm25_ids = []
    for rank, (turn_id, _score) in enumerate(store.search_turns(memory_id, ir.query, limit=96), 1):
        bm25_ids.append(turn_id); raw_scores[turn_id]["bm25"] = 1.0 / rank
    dense_ids = []
    if dense_search:
        for rank, (turn_id, _score) in enumerate(dense_search(memory_id, ir.query, 96), 1):
            dense_ids.append(turn_id); raw_scores[turn_id]["dense"] = 1.0 / rank
    operand_nodes: dict[str, tuple[str, ...]] = {}
    operand_turns: dict[str, tuple[str, ...]] = {}
    all_nodes: list[str] = []; all_turns: list[str] = []
    for operand in ir.operands:
        owner_ids = set().union(*(set(view.owner_alias_index.get(alias.casefold(), ()))
                                  for alias in operand.owner_aliases)) if operand.owner_aliases else set()
        fact_ids = list(sorted(owner_ids))
        fact_ids.extend(view.lookup_facts(owner_ids=tuple(owner_ids),
                                          predicates=operand.predicate_candidates, limit=24))
        if use_postings:
            keys = (*operand.owner_aliases, *operand.predicate_candidates, *sorted(query_terms))
            fact_ids.extend(view.route_children(keys, limit=24))
        node_ids = tuple(dict.fromkeys(node_id for node_id in fact_ids if node_id in view.nodes))[:48]
        operand_nodes[operand.operand_id] = node_ids; all_nodes.extend(node_ids)
        channels = {"exact": exact_ids[:8], "bm25": bm25_ids[:8], "dense": dense_ids[:8]}
        ranked = _rrf(channels) if use_rrf else {
            item: 1.0 for rows in channels.values() for item in rows}
        selected = [item for item, _ in sorted(ranked.items(), key=lambda row: (-row[1], row[0]))[:24]]
        if use_rrf:
            session_scores: dict[str, float] = defaultdict(float)
            for turn_id in selected:
                turn = turn_by_id.get(turn_id)
                if turn:
                    session_scores[turn.session_id] = max(session_scores[turn.session_id],
                                                          ranked.get(turn_id, 0.0))
            top_sessions = [item for item, _ in sorted(session_scores.items(),
                                                       key=lambda row: (-row[1], row[0]))[:4]]
            for session_id in top_sessions:
                local = []
                for turn in turns_by_session.get(session_id, ()):
                    lexical = len(query_terms & content_terms(turn.raw_text)) / max(1, len(query_terms))
                    channel = sum(raw_scores.get(turn.turn_id, {}).values())
                    local.append((channel + lexical, turn.turn_id))
                selected.extend(turn_id for _, turn_id in
                                sorted(local, key=lambda row: (-row[0], row[1]))[:8])
        selected = tuple(dict.fromkeys(selected))
        operand_turns[operand.operand_id] = selected; all_turns.extend(selected)
    return SeedResult(tuple(dict.fromkeys(all_nodes))[:48], tuple(dict.fromkeys(all_turns)),
                      raw_scores, operand_nodes, operand_turns, (), (),
                      {"mode": "narrow_v5_5", "flooded_sessions": 0})


def seed_operands(store, view, memory_id: str, ir: QueryIR, turns: Sequence[SourceTurn], *,
                  dense_search: DenseSearch | None, use_rrf: bool, use_postings: bool,
                  reservoir_limit: int = 576, max_views_per_operand: int = 6,
                  node_budget: int = 96, wide_reservoir: bool = True,
                  session_fanout: int = 0) -> SeedResult:
    # session_fanout <= 0 means "every session holding a channel hit".
    if not wide_reservoir:
        return _seed_narrow(store, view, memory_id, ir, turns, dense_search=dense_search,
                            use_rrf=use_rrf, use_postings=use_postings)
    turn_by_id = {turn.turn_id: turn for turn in turns}
    turns_by_session: dict[str, list[SourceTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_session[turn.session_id].append(turn)
    turn_terms = {turn.turn_id: content_terms(turn.raw_text) for turn in turns}
    query_terms = content_terms(ir.query)
    runner = _ChannelRunner(store, memory_id, turns, turn_terms, dense_search)

    views = build_views(ir, max_per_operand=max_views_per_operand) if wide_reservoir else (
        QueryView(stable_id("view", ir.query, "full"), "full_query", ir.query, None, dense=True),)

    # --- per-view channel results -------------------------------------------------
    scores: dict[str, dict[str, float]] = defaultdict(lambda: {name: 0.0 for name in CHANNELS})
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    view_hits: dict[str, list[str]] = defaultdict(list)
    # Membership must be tracked separately from strength: min-max normalization
    # maps the weakest hit in every channel to exactly 0.0, so a score test would
    # silently drop the tail of each channel.
    channel_hits: set[str] = set()
    per_view_ranked: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for row in views:
        is_full = row.operand_id is None
        exact = runner.exact(row.text, limit=None if is_full else VIEW_BM25_DEPTH)
        bm25 = runner.bm25(row.text, limit=LEGACY_BM25_DEPTH if is_full else VIEW_BM25_DEPTH)
        dense = (runner.dense(row.text, limit=LEGACY_DENSE_DEPTH if is_full else VIEW_DENSE_DEPTH)
                 if row.dense else [])
        for channel, rows in (("exact", exact), ("bm25", bm25), ("dense", dense)):
            if not rows:
                continue
            normalized = _normalize(dict(rows))
            for rank, (turn_id, _score) in enumerate(rows, 1):
                if turn_id not in turn_by_id:
                    continue
                # Keep the channel's true (normalized) strength, not 1/rank: the
                # legacy path min-max normalizes and the two scales must agree.
                scores[turn_id][channel] = max(scores[turn_id][channel], normalized[turn_id])
                previous = ranks[turn_id].get(channel)
                ranks[turn_id][channel] = rank if previous is None else min(previous, rank)
                view_hits[turn_id].append(row.view_id)
                channel_hits.add(turn_id)
            per_view_ranked[row.view_id][channel] = [turn_id for turn_id, _ in rows]

    # --- legacy parity core -------------------------------------------------------
    # Every turn the frozen navigator would have considered stays reachable.
    parity: set[str] = set(channel_hits)
    if wide_reservoir:
        session_strength: dict[str, float] = defaultdict(float)
        for turn_id in parity:
            turn = turn_by_id.get(turn_id)
            if not turn:
                continue
            row = scores[turn_id]
            fused = row["exact"] * 1.4 + row["bm25"] + row["dense"]
            session_strength[turn.session_id] = max(session_strength[turn.session_id], fused)
        # Any session the legacy navigator floods must contain a channel hit,
        # because that is where its session score comes from.  Flooding every
        # session with a hit therefore dominates its top-8 by construction,
        # without having to reproduce its score scale or tokenizer.
        ordered_sessions = [session for session, _ in sorted(
            session_strength.items(), key=lambda item: (-item[1], item[0]))]
        top_sessions = ordered_sessions if session_fanout <= 0 else ordered_sessions[:session_fanout]
        for session_id in top_sessions:
            parity.update(turn.turn_id for turn in turns_by_session.get(session_id, ()))

    # --- per-operand structured seeds and turn quotas ------------------------------
    operand_nodes: dict[str, tuple[str, ...]] = {}
    operand_turns: dict[str, tuple[str, ...]] = {}
    operand_of_turn: dict[str, set[str]] = defaultdict(set)
    per_operand_node_budget = max(24, node_budget // max(1, len(ir.operands)))
    for operand in ir.operands:
        owner_ids = _operand_owner_ids(view, operand)
        operand_nodes[operand.operand_id] = _operand_nodes(
            view, operand, owner_ids, query_terms, use_postings=use_postings,
            limit=per_operand_node_budget)
        channels: dict[str, list[str]] = defaultdict(list)
        for row in views:
            if row.operand_id not in (None, operand.operand_id):
                continue
            for channel, ids in per_view_ranked.get(row.view_id, {}).items():
                channels[f"{row.view_id}:{channel}"] = list(ids)
        ranked = _rrf(channels) if use_rrf else {
            item: 1.0 for rows in channels.values() for item in rows}
        selected = [item for item, _ in sorted(ranked.items(), key=lambda item: (-item[1], item[0]))]
        operand_turns[operand.operand_id] = tuple(selected)
        for turn_id in selected:
            operand_of_turn[turn_id].add(operand.operand_id)

    # --- assemble the reservoir ---------------------------------------------------
    fused_rrf = _rrf({f"{view_id}:{channel}": ids
                      for view_id, rows in per_view_ranked.items()
                      for channel, ids in rows.items()})
    pool = set(scores) | parity
    entries: list[ReservoirEntry] = []
    for turn_id in pool:
        turn = turn_by_id.get(turn_id)
        if not turn:
            continue
        entries.append(ReservoirEntry(
            turn_id, turn.session_id,
            dict(scores.get(turn_id, {name: 0.0 for name in CHANNELS})),
            dict(ranks.get(turn_id, {})),
            tuple(dict.fromkeys(view_hits.get(turn_id, ()))),
            tuple(sorted(operand_of_turn.get(turn_id, ()))),
            fused_rrf.get(turn_id, 0.0), turn_id in parity,
        ))
    # Parity members are never evicted; the cap only bounds the extra reach.
    entries.sort(key=lambda row: (not row.parity, -row.rrf, -len(row.operand_ids), row.turn_id))
    if len(entries) > reservoir_limit:
        keep = [row for row in entries if row.parity]
        extra = [row for row in entries if not row.parity]
        entries = keep + extra[:max(0, reservoir_limit - len(keep))]

    node_sources = dict(operand_nodes)
    if wide_reservoir:
        node_sources["~lexical"] = _lexical_nodes(view, query_terms,
                                                  limit=max(12, node_budget // 4))
    node_ids = _interleave(node_sources, node_budget)
    stats = {
        "views": len(views), "dense_views": sum(1 for row in views if row.dense),
        "flooded_sessions": len(top_sessions) if wide_reservoir else 0,
        "channel_hits": len(channel_hits),
        "reservoir_size": len(entries), "parity_size": len(parity),
        "reservoir_capped": len(pool) > len(entries),
        "operand_node_budget": per_operand_node_budget,
    }
    return SeedResult(node_ids, tuple(row.turn_id for row in entries),
                      {row.turn_id: row.scores for row in entries},
                      operand_nodes, operand_turns, tuple(entries), views, stats)
