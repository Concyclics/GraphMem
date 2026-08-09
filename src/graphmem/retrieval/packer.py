from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Callable, Iterable, Mapping, Sequence

from ..domain import (
    CandidateScore,
    EvidenceMember,
    EvidenceUnit,
    FactBinding,
    ProofObligation,
    SourceTurn,
    stable_id,
)
from ..text import content_terms, estimate_tokens


def build_proof_units(
    bindings: Iterable[FactBinding],
    group_turns: Mapping[str, tuple[str, ...]],
    *,
    obligations: Sequence[ProofObligation] = (),
    group_members: Mapping[str, Sequence[EvidenceMember]] | None = None,
    fact_spans: Mapping[str, Sequence[EvidenceMember]] | None = None,
) -> tuple[EvidenceUnit, ...]:
    """Compile binding witnesses into atomic packing units.

    V5.9 retained turn ids only, so the answer renderer had a span mode but no
    spans to consume.  V5.10 carries exact fact spans when the graph has them,
    falls back to evidence-group members for older graphs, and records the real
    QueryIR obligation ids instead of using an operand id as a surrogate.
    """

    members = group_members or {}
    spans_by_fact = fact_spans or {}
    obligations_by_operand: dict[str | None, list[str]] = defaultdict(list)
    for obligation in obligations:
        if obligation.required:
            obligations_by_operand[obligation.operand_id].append(obligation.obligation_id)
    rows: list[EvidenceUnit] = []
    for binding in bindings:
        turn_ids = tuple(dict.fromkeys(turn_id for group_id in binding.evidence_group_ids
                                       for turn_id in group_turns.get(group_id, ())))
        if not turn_ids:
            continue
        spans = tuple(dict.fromkeys(
            tuple(spans_by_fact.get(binding.fact_node_id, ()))
            or tuple(member for group_id in binding.evidence_group_ids
                     for member in members.get(group_id, ()))
        ))
        obligation_ids = tuple(dict.fromkeys((
            *obligations_by_operand.get(binding.operand_id, ()),
            *obligations_by_operand.get(None, ()),
        ))) or (binding.operand_id,)
        rows.append(EvidenceUnit(
            stable_id("proof-unit", binding.binding_id), obligation_ids,
            (binding.binding_id,), turn_ids, binding.relation_path, 0, True,
            (binding.operand_id,), binding.value_key, spans,
        ))
    return tuple(rows)


_SENTENCE_RE = re.compile(r"[^\n.!?。！？]+(?:[.!?。！？]+|\n+|$)")
_NUMBER_RE = re.compile(r"(?:[$£€¥]\s*)?\b\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+)?\b|\b(?:first|second|third|fourth|fifth)\b", re.I)
_TIME_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|before|after|since|until|ago|last|next|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|\d{4}[/-]\d{1,2})\b", re.I)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|didn't|doesn't|don't|isn't|wasn't|won't|can't|cannot|without)\b",
    re.I)
_STATUS_RE = re.compile(
    r"\b(?:started|stopped|finished|completed|cancelled|planned|currently|now|"
    r"became|bought|sold|moved|joined|left|won|lost|received|returned)\b", re.I)


def adaptive_evidence_turn_limit(
    answer_kind: str, operand_count: int, requested: int, *, query: str = "",
) -> int:
    """Bound distractors without starving genuinely exhaustive operators.

    A universal 32-turn pack gives a one-turn lookup only 3.1% annotated
    precision.  Direct lookups need a small ambiguity set, temporal operators
    need endpoints, and collection operators need the widest scope.  The limit
    is deterministic and never exceeds the caller's declared budget.
    """

    kind = str(answer_kind).casefold()
    exhaustive = any(token in kind for token in (
        "count", "list", "union", "intersection", "group", "collection"))
    temporal = any(token in kind for token in (
        "temporal", "duration", "date_difference", "ordering", "argmin", "argmax"))
    # The current algebra compiler still labels many temporal lookups as a
    # generic lookup.  A 12-turn cap on those queries caused an 8pp answer
    # regression on the LME temporal stratum even though aggregate F1 rose.
    # Use query language only to choose a budget floor, never to filter facts.
    temporal = temporal or bool(re.search(
        r"\b(?:when|what\s+time|how\s+long|how\s+many\s+"
        r"(?:days?|weeks?|months?|years?)|before|after|first|last|earlier|later|"
        r"ago|yesterday|tomorrow|past\s+weekend|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        query, re.I))
    if exhaustive:
        target = max(24, 8 * max(1, operand_count))
    elif temporal:
        target = max(24, 6 * max(1, operand_count))
    else:
        target = max(12, 8 * max(1, operand_count))
    return max(1, min(requested, target))


def _sentences(text: str) -> tuple[tuple[int, int, str], ...]:
    rows = tuple((match.start(), match.end(), match.group(0))
                 for match in _SENTENCE_RE.finditer(text) if match.group(0).strip())
    return rows or ((0, len(text), text),)


def salient_spans(
    turn: SourceTurn,
    query: str,
    *,
    answer_kind: str = "lookup",
    max_spans: int = 3,
) -> tuple[EvidenceMember, ...]:
    """Select deterministic sentence spans when the index has no exact span.

    One best lexical sentence is always kept.  Numeric/date/negation/state
    sentences receive independent floors for operators that need them, so a
    high-overlap preamble cannot displace the actual endpoint or count value.
    """

    query_terms = content_terms(query)
    rows = []
    temporal = answer_kind in {"temporal", "duration", "date_difference", "ordering"}
    aggregate = answer_kind in {"count", "list", "count_distinct", "union_distinct"}
    stateful = answer_kind in {"state_change", "latest_state"}
    for start, end, text in _sentences(turn.raw_text):
        terms = content_terms(text)
        lexical = len(query_terms & terms) / max(1, len(query_terms))
        critical = {
            "number": bool(_NUMBER_RE.search(text)),
            "time": bool(_TIME_RE.search(text)),
            "negative": bool(_NEGATION_RE.search(text)),
            "status": bool(_STATUS_RE.search(text)),
        }
        score = lexical * 8.0
        score += 2.0 * critical["negative"]
        score += (3.0 if temporal else 0.8) * critical["time"]
        score += (3.0 if aggregate else 0.8) * critical["number"]
        score += (3.0 if stateful else 0.8) * critical["status"]
        rows.append((score, start, end, critical))
    selected: set[tuple[int, int]] = set()
    if rows:
        best = max(rows, key=lambda row: (row[0], -row[1], -row[2]))
        selected.add((best[1], best[2]))
    required_kinds = ["negative"]
    if temporal:
        required_kinds.append("time")
    if aggregate:
        required_kinds.append("number")
    if stateful:
        required_kinds.append("status")
    for kind in required_kinds:
        candidates = [row for row in rows if row[3][kind]]
        if candidates:
            best = max(candidates, key=lambda row: (row[0], -row[1], -row[2]))
            selected.add((best[1], best[2]))
    for row in sorted(rows, key=lambda item: (-item[0], item[1], item[2])):
        if len(selected) >= max_spans:
            break
        selected.add((row[1], row[2]))
    return tuple(EvidenceMember(turn.turn_id, start, end, "salient_fallback")
                 for start, end in sorted(selected)[:max_spans] if end > start)


def _usable_spans(
    turn: SourceTurn,
    exact: Sequence[EvidenceMember],
    query: str,
    answer_kind: str,
) -> tuple[EvidenceMember, ...]:
    clipped = tuple(EvidenceMember(
        turn.turn_id, max(0, member.span_start), min(len(turn.raw_text), member.span_end),
        member.support_type)
        for member in exact
        if member.turn_id == turn.turn_id
        and max(0, member.span_start) < min(len(turn.raw_text), member.span_end))
    # A whole-turn source member is provenance, not an informative span.
    if clipped and any(member.span_start > 0 or member.span_end < len(turn.raw_text)
                       for member in clipped):
        return clipped
    return salient_spans(turn, query, answer_kind=answer_kind)


def _span_text(turn: SourceTurn, spans: Sequence[EvidenceMember], window: int) -> str:
    intervals = sorted((max(0, span.span_start - window),
                        min(len(turn.raw_text), span.span_end + window)) for span in spans)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return " ... ".join(turn.raw_text[start:end].strip() for start, end in merged)


def pack_obligation_aware(
    units: Iterable[EvidenceUnit],
    candidates: Iterable[CandidateScore],
    turns: Mapping[str, SourceTurn],
    *,
    query: str,
    answer_kind: str,
    max_turns: int,
    max_tokens: int,
    count_text_tokens: Callable[[str], int],
    span_window: int = 96,
    baseline_floor: Sequence[str] = (),
    precision_aware: bool = False,
) -> tuple[
    tuple[str, ...], tuple[str, ...], dict[str, bool], tuple[EvidenceUnit, ...], int
]:
    """Greedy obligation/set-cover packing under a span-token budget.

    Selection is turn-level but proof units remain atomic: all source turns of a
    selected unit fit or none are admitted.  The objective rewards new
    obligations, operands, sessions, answer members, and critical scalar/time/
    polarity/status evidence, then discounts redundant text and token cost.
    """

    units = tuple(units)
    candidates = tuple(candidates)
    candidate_by_turn = {row.turn_id: row for row in candidates}
    candidate_rank = {row.turn_id: rank for rank, row in enumerate(candidates)}
    exact_by_turn: dict[str, list[EvidenceMember]] = defaultdict(list)
    for unit in units:
        for span in unit.spans:
            exact_by_turn[span.turn_id].append(span)
    # Span extraction and exact tokenization are deliberately lazy.  A memory
    # may expose hundreds of candidate turns while the pack admits only 16;
    # eagerly tokenizing the entire reservoir made the supposedly lightweight
    # packer slower than full-turn packing by almost 2x.
    spans_by_turn: dict[str, tuple[EvidenceMember, ...]] = {}
    rendered_by_turn: dict[str, str] = {}
    costs: dict[str, int] = {}

    def spans_of(turn_id: str) -> tuple[EvidenceMember, ...]:
        if turn_id not in spans_by_turn:
            spans_by_turn[turn_id] = _usable_spans(
                turns[turn_id], exact_by_turn.get(turn_id, ()), query, answer_kind)
        return spans_by_turn[turn_id]

    def rendered(turn_id: str) -> str:
        if turn_id not in rendered_by_turn:
            rendered_by_turn[turn_id] = _span_text(
                turns[turn_id], spans_of(turn_id), span_window)
        return rendered_by_turn[turn_id]

    def cost_of(turn_id: str) -> int:
        if turn_id not in costs:
            # ``NavigationResult.evidence_tokens`` historically measures only
            # evidence text.  Prompt labels/separators are renderer overhead and
            # were not included by the full-turn baseline, so including a fixed
            # +8 here made short LoCoMo turns look more expensive after packing.
            span_cost = count_text_tokens(rendered(turn_id))
            full_cost = candidate_by_turn[turn_id].token_cost
            if span_cost > full_cost:
                spans_by_turn[turn_id] = (
                    EvidenceMember(turn_id, 0, len(turns[turn_id].raw_text),
                                   "full_turn_cheaper"),)
                rendered_by_turn[turn_id] = turns[turn_id].raw_text
                span_cost = full_cost
            costs[turn_id] = span_cost
        return costs[turn_id]
    packed: list[str] = []
    selected_units: list[EvidenceUnit] = []
    optional_units: list[EvidenceUnit] = []
    used_tokens = 0
    covered_obligations: set[str] = set()
    covered_operands: set[str] = set()
    covered_sessions: set[str] = set()
    covered_members: set[str] = set()
    selected_unit_rank: dict[str, int] = {}
    atomic_dropped = False
    token_rejected = False

    def admit(turn_ids: Sequence[str], unit: EvidenceUnit | None = None) -> bool:
        nonlocal used_tokens, token_rejected
        new_turns = tuple(sorted(
            (turn_id for turn_id in dict.fromkeys(turn_ids) if turn_id not in packed),
            key=lambda turn_id: (candidate_rank.get(turn_id, 10**9), turn_id)))
        if not new_turns:
            return True
        if any(turn_id not in candidate_by_turn or turn_id not in turns
               for turn_id in new_turns):
            return False
        if len(packed) + len(new_turns) > max_turns:
            return False
        cost = sum(cost_of(turn_id) for turn_id in new_turns)
        if used_tokens + cost > max_tokens:
            token_rejected = True
            return False
        packed.extend(new_turns)
        used_tokens += cost
        covered_sessions.update(turns[turn_id].session_id for turn_id in new_turns)
        if unit is not None:
            covered_obligations.update(unit.obligation_ids)
            covered_operands.update(unit.operand_ids)
            if unit.member_key:
                covered_members.add(unit.member_key)
            selected_unit_rank[unit.unit_id] = len(selected_unit_rank)
            selected_units.append(replace(
                unit,
                spans=tuple(span for turn_id in unit.source_turn_ids
                            for span in spans_of(turn_id)),
                token_cost=sum(cost_of(turn_id) for turn_id in unit.source_turn_ids),
                rank=selected_unit_rank[unit.unit_id],
            ))
        return True

    def admit_optional(row: CandidateScore) -> bool:
        if not admit((row.turn_id,)):
            return False
        cost = cost_of(row.turn_id)
        covered_operands.update(row.operand_ids)
        optional_units.append(EvidenceUnit(
            stable_id("packed-span-unit", row.turn_id), (), (), (row.turn_id,),
            row.graph_path_ids, cost, False, row.operand_ids, "",
            spans_of(row.turn_id), False,
            len(selected_units) + len(optional_units)))
        return True

    # Monotonicity floor: retain every turn selected by the frozen full-turn
    # pack, then spend the capacity released by span compression.  Without this
    # floor, a span pack takes ranks 1..K while the token-constrained baseline
    # can skip expensive ranks and reach a relevant lower-ranked turn; a real
    # dev example lost rank-40 gold evidence despite admitting seven more turns.
    if not precision_aware:
        for turn_id in baseline_floor:
            row = candidate_by_turn.get(turn_id)
            if row is not None:
                admit_optional(row)

    # Reserve only the minimum number of high-ranked complete proof units needed
    # to cover QueryIR obligations/operands.  The former cost-normalized greedy
    # reordered the entire reservoir in favour of short turns: it saved tokens
    # but displaced 3/4 of previously hit gold evidence in the first smoke run.
    # Here proof coverage is a floor and the retrieval ranking remains the prior.
    viable = [unit for unit in units if unit.source_turn_ids and all(
        turn_id in candidate_by_turn and turn_id in turns
        for turn_id in unit.source_turn_ids)]
    required_obligations = {item for unit in units for item in unit.obligation_ids}
    required_operands = {item for unit in units for item in unit.operand_ids}
    uncovered_obligations = set(required_obligations)
    uncovered_operands = set(required_operands)
    reserve_capacity = max_turns if precision_aware else max_turns // 2
    reserve_limit = (max(1, min(
        reserve_capacity,
        max(4, len(required_operands) * 2))) if max_turns else 0)
    considered: set[str] = set()
    while (uncovered_obligations or uncovered_operands) and len(considered) < len(viable):
        choices = [unit for unit in viable if unit.unit_id not in considered and (
            set(unit.obligation_ids) & uncovered_obligations
            or set(unit.operand_ids) & uncovered_operands)]
        if not choices or len(selected_unit_rank) >= reserve_limit:
            break
        selected = min(choices, key=lambda unit: (
            min(candidate_rank[turn_id] for turn_id in unit.source_turn_ids),
            len(unit.source_turn_ids), unit.rank, unit.unit_id))
        considered.add(selected.unit_id)
        if admit(selected.source_turn_ids, selected):
            uncovered_obligations -= set(selected.obligation_ids)
            uncovered_operands -= set(selected.operand_ids)
        elif selected.atomic:
            atomic_dropped = True

    # Fill the remainder in the exact upstream order.  Multi-turn proof units
    # are admitted atomically when possible; if not, an individual high-ranked
    # turn may still be included, but the certificate remains incomplete.
    multi_unit_by_turn: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for unit in viable:
        if unit.atomic and len(unit.source_turn_ids) > 1:
            for turn_id in unit.source_turn_ids:
                multi_unit_by_turn[turn_id].append(unit)
    # The upstream fused ranking is already query/operand-aware.  A dynamic MMR
    # scan changed ~16% of the pack but rescued only 0.040 gold turns/question
    # while losing 0.035 and added ~89 ms on dev200.  Keep the proof floor above
    # and use the existing rank for optional fill; bounded stopping supplies the
    # precision gain without a second O(KN) online reranker.
    fill_candidates = candidates
    for row in fill_candidates:
        if len(packed) >= max_turns:
            break
        if row.turn_id in packed or row.turn_id not in turns:
            continue
        atomic_units = sorted(multi_unit_by_turn.get(row.turn_id, ()), key=lambda unit: (
            max(candidate_rank[item] for item in unit.source_turn_ids), unit.unit_id))
        admitted_atomic = False
        for unit in atomic_units:
            if unit.unit_id in selected_unit_rank:
                admitted_atomic = True
                break
            if admit(unit.source_turn_ids, unit):
                admitted_atomic = True
                break
            atomic_dropped = True
        if admitted_atomic:
            continue
        admit_optional(row)

    packed_set = set(packed)
    dropped = tuple(row.turn_id for row in candidates if row.turn_id not in packed_set)
    # A unit reached incidentally by ranked fill also discharges its proof only
    # when every source turn is present.
    for unit in units:
        if unit.source_turn_ids and set(unit.source_turn_ids) <= packed_set:
            covered_obligations.update(unit.obligation_ids)
            covered_operands.update(unit.operand_ids)
            if unit.member_key:
                covered_members.add(unit.member_key)
    atomic_dropped = atomic_dropped or any(
        unit.atomic and not set(unit.source_turn_ids) <= packed_set
        and bool(set(unit.source_turn_ids) & set(candidate_by_turn))
        for unit in units)
    flags = {
        "turn_cap_reached": len(packed) >= max_turns and bool(dropped),
        "token_cap_reached": used_tokens >= max_tokens or token_rejected,
        "atomic_unit_dropped": atomic_dropped,
        "obligation_incomplete": not required_obligations <= covered_obligations,
        "operand_incomplete": not required_operands <= covered_operands,
        "span_packing": True,
        "precision_aware": precision_aware,
    }
    selected_by_id = {unit.unit_id: unit for unit in selected_units}
    audit_units = tuple(
        selected_by_id.get(unit.unit_id, replace(
            unit, rank=len(units) + unit.rank))
        for unit in units)
    return (tuple(packed), dropped, flags,
            (*audit_units, *optional_units), used_tokens)


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
