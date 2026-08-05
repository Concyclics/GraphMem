from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

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
         max_turns: int, max_tokens: int) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, bool]]:
    packed: list[str] = []; used = 0; mandatory = set()
    for unit in units:
        mandatory.update(unit.source_turn_ids)
    ordered = list(dict.fromkeys(mandatory)) + [row.turn_id for row in candidates if row.turn_id not in mandatory]
    turn_cap = token_cap = False
    for turn_id in ordered:
        turn = turns.get(turn_id)
        if not turn:
            continue
        cost = estimate_tokens(turn.raw_text)
        if len(packed) >= max_turns:
            turn_cap = True; break
        if used + cost > max_tokens:
            token_cap = True; continue
        packed.append(turn_id); used += cost
    dropped = tuple(row.turn_id for row in candidates if row.turn_id not in packed)
    return tuple(packed), dropped, {"turn_cap_reached": turn_cap, "token_cap_reached": token_cap}
