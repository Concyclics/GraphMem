"""Relation algebra over the compiled operator AST.

``evaluate()`` in ``algebra.py`` returns only a flat binding list, so the packer
cannot tell an answer member from a candidate and the answer stage has nothing
to count.  Measured consequence: the closed-form composer fired on **0 of 200**
questions, and aggregation questions -- 40% of the development set -- scored
51.9%, the worst of any category.

This module produces ``AlgebraResult`` with explicit ``AnswerMember`` rows, each
carrying the bindings that witness it.  ``algebra.evaluate`` is left untouched:
it is what H0-H9 execute and the frozen ladder must not move.

Scope honesty is the point of the design.  A count is only exact when the
collection it ranges over is closed; otherwise it is a floor, and
``AlgebraResult.scope_complete`` says which.  Claiming an exact count over a
collection the graph only partially holds is how an aggregation answer becomes
confidently wrong.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Mapping, Sequence

from ..domain import (
    AlgebraResult, AnswerMember, FactBinding, QueryOperator, StateResult,
    TemporalEndpoint, TruthValue,
)
from . import operators as ops


def _distinct_key(binding: FactBinding, distinct_by: str) -> str:
    if distinct_by == "event_instance":
        return binding.event_instance_id or binding.value_key
    if distinct_by == "date":
        key = binding.time_interval
        return (key.start or "") if key else ""
    if distinct_by == "session":
        return binding.session_id
    if distinct_by == "owner_value":
        return f"{binding.owner_id or ''}:{binding.value_key}"
    return binding.value_key


def _member(key: str, witnesses: Sequence[FactBinding]) -> AnswerMember:
    # Witness order is fixed by binding_id so a member renders identically on
    # every run; the first witness supplies the displayed value.
    rows = sorted(witnesses, key=lambda item: item.binding_id)
    head = rows[0]
    return AnswerMember(
        member_key=key, value=head.value or head.value_key, value_key=head.value_key,
        owner_id=head.owner_id, value_type=head.value_type,
        witness_binding_ids=tuple(row.binding_id for row in rows),
        operand_ids=tuple(dict.fromkeys(row.operand_id for row in rows)),
    )


def _group(bindings: Sequence[FactBinding], distinct_by: str) -> list[AnswerMember]:
    buckets: dict[str, list[FactBinding]] = defaultdict(list)
    for binding in bindings:
        key = _distinct_key(binding, distinct_by)
        if key:
            buckets[key].append(binding)
    return [_member(key, rows) for key, rows in sorted(buckets.items())]


def _by_operand(bindings: Sequence[FactBinding]) -> dict[str, list[FactBinding]]:
    grouped: dict[str, list[FactBinding]] = defaultdict(list)
    for binding in bindings:
        grouped[binding.operand_id].append(binding)
    return grouped


def _dedup_bindings(bindings: Sequence[FactBinding]) -> tuple[FactBinding, ...]:
    return tuple(sorted({row.binding_id: row for row in bindings}.values(),
                        key=lambda row: row.binding_id))


def _dedup_members(members: Sequence[AnswerMember]) -> tuple[AnswerMember, ...]:
    """Merge equal answer members without dropping witnesses from a child plan."""
    buckets: dict[str, list[AnswerMember]] = defaultdict(list)
    for member in members:
        buckets[member.member_key].append(member)
    result: list[AnswerMember] = []
    for key, rows in sorted(buckets.items()):
        head = rows[0]
        result.append(AnswerMember(
            member_key=key,
            value=head.value,
            value_key=head.value_key,
            owner_id=head.owner_id,
            value_type=head.value_type,
            witness_binding_ids=tuple(dict.fromkeys(
                binding_id for row in rows for binding_id in row.witness_binding_ids)),
            operand_ids=tuple(dict.fromkeys(
                operand_id for row in rows for operand_id in row.operand_ids)),
        ))
    return tuple(result)


def _build_result(
    node: ops.OperatorNode,
    source_bindings: Sequence[FactBinding],
    members: Sequence[AnswerMember],
    *,
    scope_complete: bool,
    count: int | None = None,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    endpoints: Sequence[TemporalEndpoint] = (),
    state: StateResult | None = None,
    degradations: Sequence[str] = (),
    numeric_total: float | None = None,
    unit: str = "",
    truth_value: TruthValue | None = None,
) -> AlgebraResult:
    stable_members = _dedup_members(members)
    output_ids = tuple(dict.fromkeys(
        binding_id for row in stable_members for binding_id in row.witness_binding_ids))
    # Temporal and state operators can carry a witness outside the ordinary
    # member rendering.  Keep it in the selected binding set and certificate.
    output_ids = tuple(dict.fromkeys((
        *output_ids,
        *(row.binding_id for row in endpoints),
        *((state.current_binding_id,) if state is not None else ()),
    )))
    by_id = {row.binding_id: row for row in source_bindings}
    selected = tuple(by_id[binding_id] for binding_id in output_ids if binding_id in by_id)
    required = ops.operand_ids(node)
    complete = tuple(operand_id for operand_id in required
                     if any(row.operand_id == operand_id for row in source_bindings))
    return AlgebraResult(
        operator=ops.root_operator(node),
        bindings=_dedup_bindings(selected),
        output_binding_ids=output_ids,
        members=stable_members,
        count=count,
        groups=dict(groups or {}),
        temporal_endpoints=tuple(endpoints),
        state_result=state,
        witness_map={row.member_key: row.witness_binding_ids for row in stable_members},
        complete_operands=complete,
        scope_complete=scope_complete,
        answer_kind=ops.answer_kind(node),
        degradations=tuple(dict.fromkeys(degradations)),
        numeric_total=numeric_total,
        unit=unit,
        truth_value=truth_value,
    )


_SCALAR_RE = re.compile(
    r"(?<![\w.])(?P<currency>[$£€¥])?\s*"
    r"(?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?)(?![\w.])")


def _numeric_scalar(binding: FactBinding) -> tuple[float, str] | None:
    """Parse one unambiguous scalar from an extracted atomic value."""

    text = binding.value or binding.value_key
    matches = tuple(_SCALAR_RE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    try:
        value = float(match.group("number").replace(",", ""))
    except ValueError:
        return None
    unit = {
        "$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY",
    }.get(
        match.group("currency") or "",
        "currency" if binding.value_type == "currency" else "",
    )
    return value, unit


def _children_complete(children: Sequence[AlgebraResult]) -> bool:
    return bool(children) and all(row.scope_complete and not row.degradations for row in children)


def evaluate_ast(node: ops.OperatorNode, bindings: Sequence[FactBinding], *,
                 collection_closed: Mapping[str, bool] | None = None) -> AlgebraResult:
    """Recursively execute an operator AST and retain witness provenance.

    ``collection_closed`` maps an operand id to whether the collection backing it
    is closed, which the projection's COLLECTION_MANIFEST supplies.  Absent that,
    no aggregate may claim exactness.

    Earlier versions switched only on the root operator.  That made
    ``Count(Intersection(A, B))`` count the union of A and B and made
    ``DateDifference(A, B)`` choose two endpoints from a global pool.  Each node
    below now consumes the result of its actual children, which is the physical
    contract promised by QueryIR.
    """
    closed = dict(collection_closed or {})
    all_bindings = _dedup_bindings(bindings)

    def exhaustive_complete(current: ops.OperatorNode) -> bool:
        required = ops.operand_ids(current)
        return bool(required) and all(closed.get(operand_id, False) for operand_id in required)

    def run(current: ops.OperatorNode) -> AlgebraResult:
        if isinstance(current, ops.FactSet):
            rows = tuple(row for row in all_bindings if row.operand_id == current.operand_id)
            members = _group(rows, "value")
            return _build_result(current, rows, members, scope_complete=bool(rows))

        children = tuple(run(child) for child in ops.children_of(current))
        child_bindings = _dedup_bindings(tuple(
            binding for child in children for binding in child.bindings))
        child_degradations = tuple(
            degradation for child in children for degradation in child.degradations)

        if isinstance(current, ops.Lookup):
            child = children[0]
            return _build_result(current, child_bindings, child.members,
                                 scope_complete=child.scope_complete,
                                 degradations=child_degradations)

        if isinstance(current, ops.UnionDistinct):
            members = _group(child_bindings, current.distinct_by)
            complete = _children_complete(children) and exhaustive_complete(current)
            return _build_result(current, child_bindings, members,
                                 scope_complete=complete,
                                 degradations=child_degradations)

        if isinstance(current, ops.IntersectionDistinct):
            keys = [set(member.member_key for member in child.members) for child in children]
            common = set.intersection(*keys) if keys else set()
            witnesses: list[FactBinding] = []
            for child in children:
                member_ids = {
                    binding_id
                    for member in child.members if member.member_key in common
                    for binding_id in member.witness_binding_ids
                }
                witnesses.extend(row for row in child.bindings if row.binding_id in member_ids)
            members = _group(witnesses, current.distinct_by)
            # Rebuild via the child member keys when a distinct key is not the
            # binding's value key (for example event_instance).
            if current.distinct_by != "value":
                members = []
                by_binding = {row.binding_id: row for row in witnesses}
                for key in sorted(common):
                    ids = tuple(dict.fromkeys(
                        binding_id for child in children for member in child.members
                        if member.member_key == key for binding_id in member.witness_binding_ids))
                    rows = [by_binding[item] for item in ids if item in by_binding]
                    if rows:
                        members.append(_member(key, rows))
            complete = _children_complete(children) and exhaustive_complete(current)
            return _build_result(current, witnesses, members,
                                 scope_complete=complete,
                                 degradations=child_degradations)

        if isinstance(current, ops.GroupByOwner):
            members = _group(child_bindings, "owner_value")
            owners: dict[str, list[str]] = defaultdict(list)
            for member in members:
                owners[member.owner_id or ""].append(member.value)
            groups = {owner: tuple(dict.fromkeys(values))
                      for owner, values in sorted(owners.items())}
            complete = _children_complete(children) and exhaustive_complete(current)
            return _build_result(current, child_bindings, members,
                                 scope_complete=complete, groups=groups,
                                 degradations=child_degradations)

        if isinstance(current, ops.CountDistinct):
            child = children[0]
            # Distinctness belongs to Count, not to the FactSet leaf.  Keeping
            # the child's value-grouped members would collapse two occurrences
            # with different event_instance ids before Count sees them.
            members = _group(child.bindings, current.distinct_by)
            complete = child.scope_complete and exhaustive_complete(current)
            return _build_result(current, child.bindings, members,
                                 scope_complete=complete, count=len(members),
                                 degradations=child_degradations)

        if isinstance(current, ops.Sum):
            child = children[0]
            parsed: list[tuple[FactBinding, float, str]] = []
            degradations = list(child_degradations)
            for binding in child.bindings:
                scalar = _numeric_scalar(binding)
                if scalar is not None:
                    parsed.append((binding, scalar[0], scalar[1]))
            if len(parsed) != len(child.bindings):
                degradations.append("non_scalar_sum_member")
            units = {unit for _binding, _value, unit in parsed if unit}
            if len(units) > 1:
                degradations.append("mixed_sum_units")
            total = sum(value for _binding, value, _unit in parsed) if parsed else None
            selected = tuple(binding for binding, _value, _unit in parsed)
            members = [_member(binding.binding_id, [binding]) for binding in selected]
            complete = bool(parsed) and child.scope_complete and exhaustive_complete(current)
            return _build_result(
                current, selected, members,
                scope_complete=complete,
                numeric_total=total,
                unit=(next(iter(units)) if len(units) == 1 else current.unit),
                degradations=degradations,
            )

        if isinstance(current, ops.ExistsAll):
            child_operands = tuple(
                ops.operand_ids(child) for child in ops.children_of(current))
            states: list[TruthValue] = []
            for child, operand_ids in zip(children, child_operands):
                if child.members:
                    states.append(TruthValue.TRUE)
                elif operand_ids and all(closed.get(item, False) for item in operand_ids):
                    states.append(TruthValue.FALSE)
                else:
                    states.append(TruthValue.UNKNOWN)
            truth = (
                TruthValue.FALSE if TruthValue.FALSE in states
                else TruthValue.TRUE if states and all(
                    item == TruthValue.TRUE for item in states)
                else TruthValue.UNKNOWN)
            members = (tuple(member for child in children for member in child.members)
                       if truth == TruthValue.TRUE else ())
            return _build_result(current, child_bindings, members,
                                 scope_complete=truth != TruthValue.UNKNOWN,
                                 degradations=child_degradations,
                                 truth_value=truth)

        if isinstance(current, (ops.Ordinal, ops.ArgMinTime, ops.ArgMaxTime)):
            child = children[0]
            resolved = [row for row in child.bindings
                        if row.time_interval and row.time_interval.resolved]
            degradations = list(child_degradations)
            members: list[AnswerMember] = []
            picked: FactBinding | None = None
            if not resolved:
                degradations.append("no_resolved_time_interval")
            else:
                ordered = sorted(resolved, key=lambda row: (
                    row.time_interval.sort_key, row.binding_id))
                if isinstance(current, ops.Ordinal) and current.order == "descending":
                    ordered.reverse()
                index = (current.index - 1 if current.index > 0 else current.index
                         ) if isinstance(current, ops.Ordinal) else (
                             -1 if isinstance(current, ops.ArgMaxTime) else 0)
                try:
                    picked = ordered[index]
                except IndexError:
                    degradations.append("ordinal_out_of_range")
                if picked is not None:
                    members = [_member(_distinct_key(picked, ops.distinct_by(current)), [picked])]
                if len(resolved) < len(child.bindings):
                    degradations.append("unresolved_intervals_excluded")
            complete = bool(picked) and child.scope_complete and exhaustive_complete(current)
            return _build_result(current, (picked,) if picked else (), members,
                                 scope_complete=complete,
                                 degradations=degradations)

        if isinstance(current, ops.DateDifference):
            left, right = children
            left_rows = sorted((row for row in left.bindings
                                if row.time_interval and row.time_interval.resolved),
                               key=lambda row: (row.time_interval.sort_key, row.binding_id))
            right_rows = sorted((row for row in right.bindings
                                 if row.time_interval and row.time_interval.resolved),
                                key=lambda row: (row.time_interval.sort_key, row.binding_id))
            degradations = list(child_degradations)
            endpoints: tuple[TemporalEndpoint, ...] = ()
            members: list[AnswerMember] = []
            selected: tuple[FactBinding, ...] = ()
            if left_rows and right_rows:
                # Respect the AST sides.  For a repeated operand, first/last
                # preserves the natural interval semantics while still avoiding
                # a global pool across unrelated operands.
                first, last = left_rows[0], right_rows[-1]
                endpoints = (
                    TemporalEndpoint("start", first.time_interval, first.binding_id),
                    TemporalEndpoint("end", last.time_interval, last.binding_id),
                )
                selected = (first, last)
                members = [_member(_distinct_key(row, ops.distinct_by(current)), [row])
                           for row in selected]
            else:
                degradations.append("date_difference_needs_two_endpoints")
            complete = len(endpoints) == 2 and left.scope_complete and right.scope_complete
            return _build_result(current, selected, members,
                                 scope_complete=complete, endpoints=endpoints,
                                 degradations=degradations)

        if isinstance(current, ops.LatestState):
            child = children[0]
            ordered = sorted(child.bindings, key=lambda row: (
                row.time_interval.sort_key if row.time_interval else
                ((1,) + ("",) * 2 + (99, 99, 0.0)),
                row.session_id, row.turn_index, row.binding_id))
            state: StateResult | None = None
            members: list[AnswerMember] = []
            current_binding: tuple[FactBinding, ...] = ()
            degradations = list(child_degradations)
            if ordered:
                current_row = ordered[-1]
                current_value = current_row.value or current_row.value_key
                contenders = [row for row in ordered[:-1] if (
                    (row.time_interval is not None
                     and current_row.time_interval is not None
                     and row.time_interval.overlaps(current_row.time_interval))
                    or (row.session_id == current_row.session_id
                        and row.turn_index == current_row.turn_index))]
                conflict = any(
                    (row.value or row.value_key) != current_value
                    or row.polarity != current_row.polarity
                    for row in contenders)
                changed = any(
                    (row.value or row.value_key) != current_value
                    for row in ordered[:-1])
                if conflict:
                    contradiction_status = "conflicting_latest"
                    degradations.append("conflicting_latest_state")
                elif changed:
                    contradiction_status = "superseded_history"
                else:
                    contradiction_status = "none"
                interval = current_row.time_interval
                interval_uncertainty = (
                    "unresolved" if interval is None or not interval.resolved
                    else "exact" if interval.precision in {"second", "minute", "day"}
                    and interval.confidence >= 0.99
                    else "bounded")
                current_binding = (current_row,)
                state = StateResult(
                    owner_id=current_row.owner_id,
                    predicate=current_row.predicate,
                    current_value=current_value,
                    current_binding_id=current_row.binding_id,
                    prior_binding_ids=tuple(row.binding_id for row in ordered[:-1]),
                    superseded=len(ordered) > 1,
                    contradiction_status=contradiction_status,
                    interval_uncertainty=interval_uncertainty,
                )
                members = [_member(_distinct_key(current_row, ops.distinct_by(current)),
                                   [current_row])]
            complete = bool(state) and child.scope_complete and exhaustive_complete(current)
            return _build_result(current, current_binding, members,
                                 scope_complete=complete, state=state,
                                 degradations=degradations)

        raise TypeError(f"unsupported operator node: {type(current).__name__}")

    return run(node)
