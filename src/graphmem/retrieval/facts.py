"""Semantic fact reservoir and bounded active shortlist.

The proof funnel showed where evidence is actually lost: a CanonicalFact
exists for the gold turn on 68.5% of questions but is *reached as a node* on
only 21.0%.  Memories hold 450-500 facts while the node budget is 96, so the
right fact is simply never retrieved.

This is the turn-side problem of PR1 one level up, and it takes the same
shape: a wide id-only reservoir that guarantees reachability, then a bounded
active shortlist that is what actually enters binding, scheduling and algebra.
Widening the reservoir alone would only move the noise downstream, so the two
layers are separate and separately measured.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..domain import NodeType, OperandSpec, RelationType
from ..text import content_terms, normalize_key


# Reservoir depths, per operand unless noted.  Ids only, so these are cheap.
SOURCE_PROJECTED_PER_OPERAND = 128
STRUCTURED_PER_OPERAND = 128
ROUTING_PER_OPERAND = 64
GLOBAL_LEXICAL = 64
RESERVOIR_SOFT_LIMIT = 384
RESERVOIR_HARD_LIMIT = 768

# Active shortlist: what binding and the scheduler actually see.
ACTIVE_PER_OPERAND = 32
ACTIVE_SHARED_BRIDGE = 16
ACTIVE_GLOBAL_FALLBACK = 16
ACTIVE_TOTAL = 96

#: Channels enabled by default.  "routing" is deliberately absent: the channel
#: was broken (it filtered Scene ids to CANONICAL_FACT and returned nothing on
#: every question ever run) and fixing it made retrieval *worse* -- on 100 LoCoMo
#: questions turn_recall fell 0.520 -> 0.486 and turn_all_hit 0.410 -> 0.390,
#: because its 58 facts per question displace better source-projection and
#: lexical candidates inside the reservoir's soft limit.  The fix is kept so the
#: channel works when the routing keys are worth reading; today they are not.
#: Re-measure this A/B before enabling it.
CHANNELS = ("source_projection", "structured", "lexical", "dense")
ALL_CHANNELS = ("source_projection", "structured", "routing", "lexical", "dense")


@dataclass(frozen=True, slots=True)
class FactReservoirEntry:
    """One reachable fact, with no source text hydrated."""

    fact_id: str
    owner_id: str = ""
    predicate: str = ""
    scope: str = ""
    value_key: str = ""
    value_type: str = ""
    polarity: str = "positive"
    modality: str = "asserted"
    session_id: str = ""
    collection_key: str = ""
    evidence_group_ids: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    operand_ids: tuple[str, ...] = ()
    # Best rank of a source turn this fact was projected from, per operand.
    source_rank: int = 0
    core: bool = False
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class FactReservoir:
    entries: tuple[FactReservoirEntry, ...] = ()
    by_operand: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    active: tuple[str, ...] = ()
    active_by_operand: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    stats: Mapping[str, Any] = field(default_factory=dict)

    def entry(self, fact_id: str) -> FactReservoirEntry | None:
        return next((row for row in self.entries if row.fact_id == fact_id), None)


def _rrf(ranked: Mapping[str, Sequence[str]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rows in ranked.values():
        for rank, item in enumerate(rows, 1):
            scores[item] += 1.0 / (k + rank)
    return scores


def _fact_row(view, fact_id: str) -> FactReservoirEntry | None:
    node = view.nodes.get(fact_id)
    if node is None or node.node_type != NodeType.CANONICAL_FACT:
        return None
    attrs = node.attributes
    return FactReservoirEntry(
        fact_id=fact_id,
        owner_id=str(attrs.get("owner_id", "") or ""),
        predicate=normalize_key(str(attrs.get("predicate", "") or "")),
        scope=normalize_key(str(attrs.get("scope", "") or "")),
        value_key=normalize_key(str(attrs.get("value_key", attrs.get("value", "")) or "")),
        value_type=str(attrs.get("value_type", "") or ""),
        polarity=str(attrs.get("polarity", "positive") or "positive"),
        modality=str(attrs.get("modality", "asserted") or "asserted"),
        session_id=str(attrs.get("session_id", "") or ""),
        collection_key=normalize_key(str(attrs.get("collection_key", "") or "")),
        evidence_group_ids=tuple(node.all_evidence_group_ids),
    )


def _operand_owner_ids(view, operand: OperandSpec) -> tuple[str, ...]:
    rows: set[str] = set()
    for alias in operand.owner_aliases:
        rows.update(view.owner_alias_index.get(alias.casefold(), ()))
    return tuple(sorted(rows))


def turn_group_index(store, memory_id: str) -> dict[str, tuple[str, ...]]:
    """turn_id -> evidence groups citing it, built once per memory."""
    rows: dict[str, list[str]] = defaultdict(list)
    for group in store.evidence_groups(memory_id):
        for member in group.members:
            rows[member.turn_id].append(group.evidence_group_id)
    return {key: tuple(dict.fromkeys(value)) for key, value in rows.items()}


def _source_projected(view, turn_ids: Sequence[str], groups_by_turn: Mapping[str, tuple[str, ...]],
                      *, limit: int) -> tuple[list[str], dict[str, int]]:
    """turn -> evidence group -> citing facts.

    This is the channel that matters most: source turns are already 100%
    reachable, so projecting back through their provenance finds facts that no
    owner/predicate posting would return, without needing the query compiler to
    have guessed the right predicate.
    """
    rows: list[str] = []
    best_rank: dict[str, int] = {}
    for rank, turn_id in enumerate(turn_ids, 1):
        groups = groups_by_turn.get(turn_id, ())
        if not groups:
            continue
        for fact_id in view.facts_for_evidence_groups(groups, limit=limit):
            if fact_id not in best_rank:
                best_rank[fact_id] = rank
                rows.append(fact_id)
            if len(rows) >= limit:
                return rows, best_rank
    return rows, best_rank


def build_fact_reservoir(view, ir, seeded, groups_by_turn, *, core_fact_ids: Sequence[str] = (),
                         channels: Sequence[str] = CHANNELS,
                         source_projected_limit: int = SOURCE_PROJECTED_PER_OPERAND,
                         structured_limit: int = STRUCTURED_PER_OPERAND,
                         routing_limit: int = ROUTING_PER_OPERAND,
                         lexical_limit: int = GLOBAL_LEXICAL,
                         soft_limit: int = RESERVOIR_SOFT_LIMIT,
                         hard_limit: int = RESERVOIR_HARD_LIMIT,
                         dense_fact_search=None) -> FactReservoir:
    """Collect reachable facts per operand.  Ids only; nothing is hydrated."""
    enabled = set(channels)
    query_terms = content_terms(ir.query)
    per_operand_channels: dict[str, dict[str, list[str]]] = defaultdict(dict)
    contributions: dict[str, int] = defaultdict(int)
    source_ranks: dict[str, dict[str, int]] = {}

    for operand in ir.operands:
        rows: dict[str, list[str]] = {}
        owner_ids = _operand_owner_ids(view, operand)
        if "source_projection" in enabled:
            turn_ids = seeded.operand_turns.get(operand.operand_id) or seeded.source_turn_ids
            found, ranks = _source_projected(view, tuple(turn_ids), groups_by_turn,
                                             limit=source_projected_limit)
            source_ranks[operand.operand_id] = ranks
            if found:
                rows["source_projection"] = found
                contributions["source_projection"] += len(found)
        if "structured" in enabled:
            found = list(view.facts_for_owner_predicate(
                owner_ids, operand.predicate_candidates, limit=structured_limit))
            found.extend(view.lookup_facts(
                owner_ids=owner_ids, predicates=operand.predicate_candidates,
                scopes=tuple(operand.scope_candidates or ()), limit=structured_limit,
                rank_terms=query_terms))
            found = list(dict.fromkeys(found))[:structured_limit]
            if found:
                rows["structured"] = found
                contributions["structured"] += len(found)
        if "routing" in enabled:
            keys = (*operand.owner_aliases, *operand.predicate_candidates, *sorted(query_terms))
            # Routing postings never contain facts.  A level-1 card's
            # ``child_postings`` maps a term to *Scene* ids (pipeline
            # ``_packet_record`` sets ``child_id = packet.scene_id``), and level
            # 2/3 cards map to routing-card ids.  Filtering the result to
            # CANONICAL_FACT therefore matched nothing and this channel returned
            # zero rows on every question ever run.  Descend one hop instead:
            # the facts hang off the scene by SCENE_CONTAINS / HAS_FACT.
            found: list[str] = []
            for node_id in view.route_children(keys, limit=routing_limit * 2):
                node = view.nodes.get(node_id)
                if node is None:
                    continue
                if node.node_type == NodeType.CANONICAL_FACT:
                    found.append(node_id)
                    continue
                for row in view.neighbors(node_id, (RelationType.SCENE_CONTAINS,
                                                    RelationType.HAS_FACT,
                                                    RelationType.REFINES_TO),
                                          semantic_only=True):
                    child = view.nodes.get(row.next_node_id)
                    if child is not None and child.node_type == NodeType.CANONICAL_FACT:
                        found.append(row.next_node_id)
            found = list(dict.fromkeys(found))[:routing_limit]
            if found:
                rows["routing"] = found
                contributions["routing"] += len(found)
        if "lexical" in enabled:
            operand_terms = query_terms | content_terms(" ".join(
                (*operand.owner_aliases, *operand.predicate_candidates)))
            found = list(view.facts_for_terms(operand_terms, limit=lexical_limit))
            if found:
                rows["lexical"] = found
                contributions["lexical"] += len(found)
        if "dense" in enabled and dense_fact_search is not None:
            found = list(dense_fact_search(operand))
            if found:
                rows["dense"] = found
                contributions["dense"] += len(found)
        per_operand_channels[operand.operand_id] = rows

    core = set(core_fact_ids)
    entries: dict[str, FactReservoirEntry] = {}
    by_operand: dict[str, tuple[str, ...]] = {}
    for operand in ir.operands:
        rows = per_operand_channels[operand.operand_id]
        fused = _rrf(rows)
        ordered = [fact_id for fact_id, _ in
                   sorted(fused.items(), key=lambda row: (-row[1], row[0]))]
        kept: list[str] = []
        for fact_id in ordered:
            row = _fact_row(view, fact_id)
            if row is None:
                continue
            kept.append(fact_id)
            channel_ranks = {name: ids.index(fact_id) + 1
                             for name, ids in rows.items() if fact_id in ids}
            existing = entries.get(fact_id)
            operand_ids = tuple(dict.fromkeys(
                (*(existing.operand_ids if existing else ()), operand.operand_id)))
            rank = source_ranks.get(operand.operand_id, {}).get(fact_id, 0)
            if existing and existing.source_rank and rank:
                rank = min(rank, existing.source_rank)
            elif existing and existing.source_rank:
                rank = existing.source_rank
            entries[fact_id] = replace(
                row, channels=tuple(sorted(channel_ranks)), channel_ranks=channel_ranks,
                operand_ids=operand_ids, core=fact_id in core, source_rank=rank,
                score=max(fused.get(fact_id, 0.0), existing.score if existing else 0.0))
        by_operand[operand.operand_id] = tuple(kept)

    # Core facts are never dropped by the reservoir cap.
    rows_sorted = sorted(entries.values(), key=lambda row: (not row.core, -row.score, row.fact_id))
    capped = rows_sorted[:hard_limit]
    stats = {
        "reservoir_size": len(capped),
        "reservoir_capped": len(rows_sorted) > len(capped),
        "soft_limit_exceeded": len(rows_sorted) > soft_limit,
        "core_facts": sum(1 for row in capped if row.core),
        "channel_contributions": dict(contributions),
        "channels_enabled": sorted(enabled),
        "per_operand_reservoir": {key: len(value) for key, value in by_operand.items()},
    }
    return FactReservoir(tuple(capped), by_operand, (), {}, stats)


# A fact projected from a highly ranked source turn is admitted on that basis
# alone: source turns are 100% reachable, and that projection is the channel
# that recovers facts the query compiler could not name.
SOURCE_RANK_ADMIT = 32


def _admissible(entry: FactReservoirEntry, operand: OperandSpec, owner_ids: frozenset[str],
                query_terms: frozenset[str]) -> tuple[bool, str]:
    """Can this rescue fact plausibly serve the operand?

    A wide reservoir is only safe if the shortlist is selective.  A fact that
    matches nothing about the operand would just be noise competing for the
    budget, so it stays in the reservoir and out of the active set.
    """
    strong_source = bool(entry.source_rank) and entry.source_rank <= SOURCE_RANK_ADMIT
    if owner_ids and entry.owner_id and entry.owner_id not in owner_ids:
        # Owner attribution in the graph can be wrong (speaker vs subject), so a
        # mismatch demotes rather than vetoes when the source evidence is strong.
        if not strong_source:
            return False, "owner_mismatch"
        return True, "source_rank_over_owner_mismatch"
    if owner_ids and entry.owner_id in owner_ids:
        return True, "owner_match"
    if strong_source:
        return True, "source_rank"
    predicate_terms = set(content_terms(entry.predicate))
    for candidate in operand.predicate_candidates:
        candidate_terms = set(content_terms(candidate))
        if candidate_terms and (candidate_terms <= predicate_terms or predicate_terms <= candidate_terms):
            return True, "predicate_match"
    if operand.scope_candidates and entry.scope in {normalize_key(item)
                                                    for item in operand.scope_candidates}:
        return True, "scope_match"
    if operand.value_type and entry.value_type == operand.value_type:
        return True, "value_type_match"
    if query_terms & (predicate_terms | set(content_terms(entry.value_key))):
        return True, "lexical_match"
    return False, "no_signal"


def select_active_facts(reservoir: FactReservoir, view, ir, *,
                        per_operand: int = ACTIVE_PER_OPERAND,
                        shared_bridge: int = ACTIVE_SHARED_BRIDGE,
                        global_fallback: int = ACTIVE_GLOBAL_FALLBACK,
                        total: int = ACTIVE_TOTAL) -> FactReservoir:
    """Bounded, round-robin shortlist. Core facts first, then admissible rescues."""
    by_id = {row.fact_id: row for row in reservoir.entries}
    query_terms = content_terms(ir.query)
    reasons: dict[str, int] = defaultdict(int)

    ranked_per_operand: dict[str, list[str]] = {}
    for operand in ir.operands:
        owner_ids = frozenset(_operand_owner_ids(view, operand))
        rows: list[tuple[int, float, str]] = []
        for fact_id in reservoir.by_operand.get(operand.operand_id, ()):
            entry = by_id.get(fact_id)
            if entry is None:
                continue
            if entry.core:
                rows.append((0, -entry.score, fact_id))
                reasons["core"] += 1
                continue
            ok, reason = _admissible(entry, operand, owner_ids, query_terms)
            reasons[reason] += 1
            if ok:
                rows.append((1, -entry.score, fact_id))
        rows.sort()
        ranked_per_operand[operand.operand_id] = [fact_id for _, _, fact_id in rows]

    selected: list[str] = []
    seen: set[str] = set()
    active_by_operand: dict[str, list[str]] = {key: [] for key in ranked_per_operand}
    cursors = {key: 0 for key in ranked_per_operand}
    # Round robin, so a first operand can never consume the shared budget.
    while len(selected) < total:
        advanced = False
        for operand_id in sorted(ranked_per_operand):
            if len(active_by_operand[operand_id]) >= per_operand or len(selected) >= total:
                continue
            rows = ranked_per_operand[operand_id]
            index = cursors[operand_id]
            while index < len(rows) and rows[index] in seen:
                index += 1
            cursors[operand_id] = index + 1
            if index < len(rows):
                advanced = True
                fact_id = rows[index]
                seen.add(fact_id); selected.append(fact_id)
                active_by_operand[operand_id].append(fact_id)
        if not advanced:
            break

    # Facts serving more than one operand bridge an intersection; keep a few.
    bridges = [row.fact_id for row in reservoir.entries
               if len(row.operand_ids) > 1 and row.fact_id not in seen][:shared_bridge]
    for fact_id in bridges:
        if len(selected) >= total + shared_bridge:
            break
        seen.add(fact_id); selected.append(fact_id)
    fallback = [row.fact_id for row in reservoir.entries if row.fact_id not in seen][:global_fallback]
    selected.extend(fallback[:max(0, total + shared_bridge + global_fallback - len(selected))])

    stats = dict(reservoir.stats)
    stats.update({
        "active_facts": len(selected),
        "active_core": sum(1 for fact_id in selected if by_id[fact_id].core),
        "active_rescue": sum(1 for fact_id in selected if not by_id[fact_id].core),
        "active_bridges": len(bridges),
        "active_fallback": len(fallback),
        "admissibility_reasons": dict(reasons),
        "per_operand_active": {key: len(value) for key, value in active_by_operand.items()},
        "active_truncated": any(len(rows) > per_operand for rows in ranked_per_operand.values()),
    })
    return FactReservoir(reservoir.entries, reservoir.by_operand, tuple(selected),
                         {key: tuple(value) for key, value in active_by_operand.items()}, stats)
