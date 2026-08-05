from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..domain import OperandSpec, ProofObligation, QueryOperator, stable_id
from ..text import content_terms, normalize_key, terms

if TYPE_CHECKING:
    from ..runtime.read_view import GraphReadView


@dataclass(frozen=True, slots=True)
class QueryIR:
    query: str
    operator: QueryOperator
    operands: tuple[OperandSpec, ...]
    proof_obligations: tuple[ProofObligation, ...]
    ordering: str | None = None
    distinct_by: str = "value"


def _operator(query: str) -> QueryOperator:
    lowered = query.casefold()
    if "how many" in lowered or lowered.startswith("count "):
        return QueryOperator.COUNT_DISTINCT
    if "both" in lowered or "in common" in lowered or "shared" in lowered:
        return QueryOperator.INTERSECTION_DISTINCT
    if "how long" in lowered or "duration" in lowered or "difference" in lowered:
        return QueryOperator.DATE_DIFFERENCE
    if any(word in lowered for word in ("latest", "currently", "now", "current")):
        return QueryOperator.LATEST_STATE
    if any(word in lowered for word in ("first", "second", "third", "fourth", "fifth")):
        return QueryOperator.ORDINAL
    if any(word in lowered for word in ("before", "after", "earlier", "later")):
        return QueryOperator.ARGMIN_TIME if "first" in lowered or "earlier" in lowered else QueryOperator.ARGMAX_TIME
    if any(word in lowered for word in ("list", "what are", "which", "what do")):
        return QueryOperator.UNION_DISTINCT
    if query.casefold().startswith(("did ", "does ", "was ", "were ", "has ", "have ")):
        return QueryOperator.EXISTS_ALL
    return QueryOperator.LOOKUP


def _query_owners(query: str, view: "GraphReadView") -> tuple[tuple[str, tuple[str, ...]], ...]:
    lowered = f" {normalize_key(query)} "
    matches: list[tuple[str, tuple[str, ...], int]] = []
    for alias, owner_ids in view.owner_alias_index.items():
        if alias and f" {alias} " in lowered:
            matches.append((alias, owner_ids, len(alias.split())))
    selected: dict[tuple[str, ...], tuple[str, int]] = {}
    for alias, owner_ids, width in sorted(matches, key=lambda row: (-row[2], row[0])):
        key = tuple(owner_ids)
        if key not in selected:
            selected[key] = (alias, width)
    return tuple((alias, owner_ids) for owner_ids, (alias, _) in sorted(selected.items()))


def _predicate_candidates(query: str, owners: tuple[tuple[str, tuple[str, ...]], ...],
                          view: "GraphReadView") -> tuple[str, ...]:
    remaining = set(content_terms(query))
    for alias, _ in owners:
        remaining -= set(terms(alias))
    scored: list[tuple[int, str]] = []
    for predicate in view.predicate_index:
        overlap = len(remaining & set(content_terms(predicate)))
        if overlap:
            scored.append((overlap, predicate))
    return tuple(value for _, value in sorted(scored, key=lambda row: (-row[0], row[1]))[:4])


def compile_query(query: str, view: "GraphReadView") -> QueryIR:
    operator = _operator(query)
    owners = _query_owners(query, view)
    predicates = _predicate_candidates(query, owners, view)
    distinct_by = "event_instance" if any(word in query.casefold() for word in ("times", "occasions", "events")) else "value"
    multiplicity = "exhaustive_set" if operator in {
        QueryOperator.UNION_DISTINCT, QueryOperator.INTERSECTION_DISTINCT, QueryOperator.COUNT_DISTINCT,
    } else "at_least_one"
    rows = owners or (("", ()),)
    operands = tuple(OperandSpec(
        operand_id=stable_id("operand", query, index, alias),
        owner_aliases=(alias,) if alias else (), predicate_candidates=predicates,
        multiplicity=multiplicity, distinct_by=distinct_by,
    ) for index, (alias, _) in enumerate(rows))
    obligations: list[ProofObligation] = []
    for operand in operands:
        obligations.extend((
            ProofObligation(stable_id("obligation", operand.operand_id, "binding"), operand.operand_id, "binding"),
            ProofObligation(stable_id("obligation", operand.operand_id, "provenance"), operand.operand_id, "provenance"),
        ))
        if operand.multiplicity == "exhaustive_set":
            obligations.append(ProofObligation(
                stable_id("obligation", operand.operand_id, "collection"), operand.operand_id, "collection"))
    ordering = "ascending" if any(word in query.casefold() for word in ("before", "earlier", "first")) else None
    return QueryIR(query, operator, operands, tuple(obligations), ordering, distinct_by)
