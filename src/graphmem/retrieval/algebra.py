from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..domain import FactBinding, QueryOperator
from .bindings import by_operand


@dataclass(frozen=True, slots=True)
class ClosureResult:
    bindings: tuple[FactBinding, ...]
    output_binding_ids: tuple[str, ...]
    complete_operands: tuple[str, ...]
    collection_complete: bool


def evaluate(operator: QueryOperator, bindings: Iterable[FactBinding], operand_ids: Iterable[str], *,
             distinct_by: str = "value", collection_complete: bool | None = None) -> ClosureResult:
    grouped = by_operand(bindings); required = tuple(operand_ids)
    complete = tuple(key for key in required if grouped.get(key))
    selected: list[FactBinding] = []
    if operator == QueryOperator.INTERSECTION_DISTINCT and len(required) > 1:
        common: set[str] | None = None
        for key in required:
            values = {item.event_instance_id if distinct_by == "event_instance" else item.value_key
                      for item in grouped.get(key, ())}
            common = values if common is None else common & values
        for key in required:
            selected.extend(item for item in grouped.get(key, ())
                            if (item.event_instance_id if distinct_by == "event_instance" else item.value_key) in (common or set()))
    elif operator in {QueryOperator.ARGMIN_TIME, QueryOperator.ARGMAX_TIME, QueryOperator.ORDINAL}:
        # Order by the normalized interval, never by its serialized mapping: canonical
        # JSON sorts keys, so str(interval) starts with anchor_turn_id and the old
        # comparison ranked events by turn-id hash.
        candidates = [item for items in grouped.values() for item in items
                      if item.time_interval and item.time_interval.resolved]
        if candidates:
            ordered = sorted(candidates, key=lambda item: (item.time_interval.sort_key, item.binding_id))
            selected = [ordered[-1] if operator == QueryOperator.ARGMAX_TIME else ordered[0]]
    else:
        seen: set[tuple[str, str]] = set()
        for item in (item for items in grouped.values() for item in items):
            key = (item.operand_id, item.event_instance_id if distinct_by == "event_instance" else item.value_key)
            if key not in seen:
                selected.append(item); seen.add(key)
    selected = sorted(selected, key=lambda item: (-item.confidence, item.binding_id))
    if collection_complete is None:
        collection_complete = operator not in {
            QueryOperator.UNION_DISTINCT, QueryOperator.INTERSECTION_DISTINCT,
            QueryOperator.GROUP_BY_OWNER, QueryOperator.COUNT_DISTINCT,
        }
    return ClosureResult(tuple(selected), tuple(item.binding_id for item in selected), complete,
                         collection_complete)
