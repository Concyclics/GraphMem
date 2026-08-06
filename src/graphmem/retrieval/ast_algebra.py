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
from typing import Mapping, Sequence

from ..domain import (
    AlgebraResult, AnswerMember, FactBinding, QueryOperator, StateResult, TemporalEndpoint,
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


def evaluate_ast(node: ops.OperatorNode, bindings: Sequence[FactBinding], *,
                 collection_closed: Mapping[str, bool] | None = None) -> AlgebraResult:
    """Execute the AST and return members with their witnesses.

    ``collection_closed`` maps an operand id to whether the collection backing it
    is closed, which the projection's COLLECTION_MANIFEST supplies.  Absent that,
    no aggregate may claim exactness.
    """
    closed = dict(collection_closed or {})
    required = ops.operand_ids(node)
    grouped = _by_operand(bindings)
    scoped = [binding for binding in bindings
              if not required or binding.operand_id in set(required)]
    complete = tuple(key for key in required if grouped.get(key))
    distinct_by = ops.distinct_by(node)
    kind = ops.answer_kind(node)
    operator = ops.root_operator(node)
    degradations: list[str] = []

    def scope_ok(members: Sequence[AnswerMember]) -> bool:
        if not members:
            return False
        if set(required) - set(complete):
            return False
        if ops.requires_exhaustive_scope(node) and not all(
                closed.get(key, False) for key in required):
            return False
        return True

    members: list[AnswerMember] = []
    count: int | None = None
    groups: dict[str, tuple[str, ...]] = {}
    endpoints: tuple[TemporalEndpoint, ...] = ()
    state: StateResult | None = None

    if operator == QueryOperator.INTERSECTION_DISTINCT and len(required) > 1:
        common: set[str] | None = None
        for key in required:
            values = {_distinct_key(item, distinct_by) for item in grouped.get(key, ())}
            common = values if common is None else common & values
        common = common or set()
        witnesses = [item for item in scoped if _distinct_key(item, distinct_by) in common]
        members = _group(witnesses, distinct_by)
        # An intersection member is only proved if every operand witnesses it.
        members = [row for row in members if len(set(row.operand_ids)) >= len(required)]
        if not members and common:
            degradations.append("intersection_witness_incomplete")

    elif operator == QueryOperator.COUNT_DISTINCT:
        members = _group(scoped, distinct_by)
        count = len(members)

    elif operator == QueryOperator.GROUP_BY_OWNER:
        members = _group(scoped, "owner_value")
        owners: dict[str, list[str]] = defaultdict(list)
        for row in members:
            owners[row.owner_id or ""].append(row.value)
        groups = {owner: tuple(dict.fromkeys(values)) for owner, values in sorted(owners.items())}

    elif operator == QueryOperator.EXISTS_ALL:
        members = _group(scoped, distinct_by)
        # Existence is per-operand: "did both A and B happen" needs a witness for
        # each, not one witness twice.
        if set(required) - set(complete):
            members = []

    elif operator in {QueryOperator.ARGMIN_TIME, QueryOperator.ARGMAX_TIME, QueryOperator.ORDINAL}:
        resolved = [item for item in scoped if item.time_interval and item.time_interval.resolved]
        if not resolved:
            degradations.append("no_resolved_time_interval")
        else:
            ordered = sorted(resolved, key=lambda item: (item.time_interval.sort_key,
                                                         item.binding_id))
            index = ops.ordinal_index(node) if operator == QueryOperator.ORDINAL else (
                -1 if operator == QueryOperator.ARGMAX_TIME else 0)
            try:
                picked = ordered[index]
            except IndexError:
                picked = None
                degradations.append("ordinal_out_of_range")
            if picked is not None:
                members = [_member(_distinct_key(picked, distinct_by), [picked])]
            if len(resolved) < len(scoped):
                degradations.append("unresolved_intervals_excluded")

    elif operator == QueryOperator.DATE_DIFFERENCE:
        resolved = [item for item in scoped if item.time_interval and item.time_interval.resolved]
        ordered = sorted(resolved, key=lambda item: (item.time_interval.sort_key, item.binding_id))
        if len(ordered) >= 2:
            endpoints = (TemporalEndpoint("start", ordered[0].time_interval, ordered[0].binding_id),
                         TemporalEndpoint("end", ordered[-1].time_interval, ordered[-1].binding_id))
            members = [_member(_distinct_key(row, distinct_by), [row])
                       for row in (ordered[0], ordered[-1])]
        else:
            degradations.append("date_difference_needs_two_endpoints")

    elif operator == QueryOperator.LATEST_STATE:
        ordered = sorted(scoped, key=lambda item: (
            item.time_interval.sort_key if item.time_interval else ((1,) + ("",) * 2 + (99, 99, 0.0)),
            item.session_id, item.turn_index, item.binding_id))
        if ordered:
            current = ordered[-1]
            state = StateResult(
                owner_id=current.owner_id, predicate=current.predicate,
                current_value=current.value or current.value_key,
                current_binding_id=current.binding_id,
                prior_binding_ids=tuple(row.binding_id for row in ordered[:-1]),
                superseded=len(ordered) > 1)
            members = [_member(_distinct_key(current, distinct_by), [current])]

    else:  # LOOKUP, UNION_DISTINCT
        members = _group(scoped, distinct_by)

    output = tuple(dict.fromkeys(
        binding_id for row in members for binding_id in row.witness_binding_ids))
    # Sort rather than filter in input order: the caller's binding order varies
    # with channel interleaving, and an AlgebraResult that changes with it would
    # make the packer's output depend on retrieval order.
    selected = sorted((binding for binding in bindings if binding.binding_id in set(output)),
                      key=lambda item: item.binding_id)
    return AlgebraResult(
        operator=operator, bindings=tuple(selected), output_binding_ids=output,
        members=tuple(members), count=count, groups=groups,
        temporal_endpoints=endpoints, state_result=state,
        witness_map={row.member_key: row.witness_binding_ids for row in members},
        complete_operands=complete, scope_complete=scope_ok(members),
        answer_kind=kind, degradations=tuple(degradations),
    )
