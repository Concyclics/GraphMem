from __future__ import annotations

from typing import Callable, Iterable, Mapping

from ..domain import CandidateScore, EvidenceUnit, FactBinding, SourceTurn, stable_id
from ..text import estimate_tokens


def build_proof_units(bindings: Iterable[FactBinding], group_turns: Mapping[str, tuple[str, ...]]) -> tuple[EvidenceUnit, ...]:
    rows: list[EvidenceUnit] = []
    for binding in bindings:
        turn_ids = tuple(dict.fromkeys(turn_id for group_id in binding.evidence_group_ids
                                       for turn_id in group_turns.get(group_id, ())))
        if not turn_ids:
            continue
        rows.append(EvidenceUnit(
            stable_id("proof-unit", binding.binding_id), (binding.operand_id,), (binding.binding_id,), turn_ids,
            binding.relation_path, 0, True,
        ))
    return tuple(rows)


def pack(units: Iterable[EvidenceUnit], candidates: Iterable[CandidateScore], turns: Mapping[str, SourceTurn], *,
         max_turns: int, max_tokens: int,
         token_cost: Callable[[SourceTurn], int] | None = None,
         rank_mandatory: bool = False,
         ) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, bool]]:
    cost_of = token_cost or (lambda turn: estimate_tokens(turn.raw_text))
    candidates = list(candidates)
    packed: list[str] = []; used = 0
    # Mandatory turns must keep the order their proof units declare them in.
    # Collecting them into a set and iterating it made both the packed order and,
    # under the turn cap, the packed membership vary with PYTHONHASHSEED.
    mandatory: list[str] = []; seen: set[str] = set()
    for unit in units:
        for turn_id in unit.source_turn_ids:
            if turn_id not in seen:
                seen.add(turn_id); mandatory.append(turn_id)
    if rank_mandatory:
        # Declaration order is binding order, which says nothing about relevance.
        # That is harmless when the mandatory set fits the budget, and it is the
        # entire selection when it does not: a LoCoMo question produces 95-104
        # mandatory turns against a 16-32 turn budget, so the pack is 100%
        # mandatory and `rest` -- the lexical and dense ranking -- never gets a
        # single seat.  Ordering by the score the candidate pool already computed
        # turns an arbitrary truncation into a ranked one.  turn_id breaks ties so
        # the result stays stable across PYTHONHASHSEED, which is what the
        # declaration order was protecting in the first place.
        score_of = {row.turn_id: row.fused_score for row in candidates}
        mandatory.sort(key=lambda turn_id: (-score_of.get(turn_id, 0.0), turn_id))
    rest = [row.turn_id for row in candidates if row.turn_id not in seen]
    ordered = mandatory + rest
    turn_cap = token_cap = False
    for turn_id in ordered:
        turn = turns.get(turn_id)
        if not turn:
            continue
        cost = cost_of(turn)
        if len(packed) >= max_turns:
            turn_cap = True; break
        if used + cost > max_tokens:
            token_cap = True; continue
        packed.append(turn_id); used += cost
    dropped = tuple(row.turn_id for row in candidates if row.turn_id not in packed)
    return tuple(packed), dropped, {"turn_cap_reached": turn_cap, "token_cap_reached": token_cap}
