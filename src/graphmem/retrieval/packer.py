from __future__ import annotations

from collections import defaultdict
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
         ) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, bool]]:
    cost_of = token_cost or (lambda turn: estimate_tokens(turn.raw_text))
    packed: list[str] = []; used = 0
    # Mandatory turns must keep the order their proof units declare them in.
    # Collecting them into a set and iterating it made both the packed order and,
    # under the turn cap, the packed membership vary with PYTHONHASHSEED.
    mandatory: list[str] = []; seen: set[str] = set()
    for unit in units:
        for turn_id in unit.source_turn_ids:
            if turn_id not in seen:
                seen.add(turn_id); mandatory.append(turn_id)
    ordered = mandatory + [row.turn_id for row in candidates if row.turn_id not in seen]
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
