from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

from ..domain import OperandSpec, ProofObligation, QueryOperator, stable_id
from ..text import content_terms, normalize_key, terms
from .operators import (
    ArgMaxTime,
    ArgMinTime,
    CountDistinct,
    DateDifference,
    ExistsAll,
    FactSet,
    GroupByOwner,
    IntersectionDistinct,
    LatestState,
    Lookup,
    OperatorNode,
    Ordinal,
    UnionDistinct,
    describe,
    operand_ids,
    requires_exhaustive_scope,
    root_operator,
    walk,
)
from ..principals import PrincipalRegistry, ResolvedOwner, resolve_query_owners
from .slots import QuerySlots, parse_slots

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
    # --- V5.6 operator AST, shadow only ---------------------------------------
    # ``operator``/``operands`` above stay exactly what V5.5 executed so the
    # ablation ladder does not move.  The AST is compiled alongside and traced;
    # a later profile switches execution over to it.  Keeping both lets the
    # divergence be measured before it is acted on.
    ast: OperatorNode | None = None
    ast_operands: tuple[OperandSpec, ...] = ()
    ast_obligations: tuple[ProofObligation, ...] = ()
    slots: QuerySlots | None = None
    parse_warnings: tuple[str, ...] = ()
    # Principal-aware owner resolution, also shadow: the legacy owner strings
    # above still drive execution until this is validated on its own.
    resolved_owners: tuple[ResolvedOwner, ...] = ()
    owner_resolution_warnings: tuple[str, ...] = ()
    # Confidence controls a safety union, never an answer shortcut.  A low
    # value means AST/owner/type filters may be wrong, so retrieval should keep
    # the AST operator but soften operand constraints with the legacy parse.
    compile_confidence: float = 1.0
    fallback_reasons: tuple[str, ...] = ()
    soft_fallback_applied: bool = False

    @property
    def ast_operator(self) -> QueryOperator | None:
        """The operator the AST would execute, for divergence reporting."""
        return root_operator(self.ast) if self.ast is not None else None

    @property
    def ast_diverges(self) -> bool:
        return self.ast is not None and self.ast_operator != self.operator

    def describe_ast(self) -> str:
        return describe(self.ast) if self.ast is not None else ""

    def promote_ast(self) -> "QueryIR":
        """Return one executable IR whose public fields all name the AST plan.

        V5.6--V5.10 compiled the AST in shadow while seeding, binding and the
        certificate still consumed legacy fields.  Promoting once at the
        compiler boundary removes positional operand remapping and guarantees
        every downstream stage sees identical operand ids and obligations.
        The original instance remains available to the caller for divergence
        telemetry and frozen-profile comparisons.
        """
        if self.ast is None:
            return self
        operator = root_operator(self.ast)
        operands = self.ast_operands or self.operands
        obligations = self.ast_obligations or self.proof_obligations
        distinct_by = (operands[0].distinct_by if operands else self.distinct_by)
        ordering = self.ordering
        if self.slots is not None and self.slots.ordinal_order:
            ordering = self.slots.ordinal_order
        return replace(
            self, operator=operator, operands=operands,
            proof_obligations=obligations, ordering=ordering,
            distinct_by=distinct_by)

    def soften_with_legacy(self, legacy: "QueryIR") -> "QueryIR":
        """Keep the AST plan while unioning/relaxing uncertain seed filters.

        This is intentionally not a second unbounded retrieval.  Corresponding
        operands retain the promoted stable ids and multiplicity, but owner,
        predicate and scope views are unioned.  A temporal/polarity constraint
        that only one compiler inferred becomes a score hint rather than a hard
        binding veto by being removed from the executable operand.
        """

        if len(self.operands) != len(legacy.operands):
            return replace(
                self, soft_fallback_applied=True,
                fallback_reasons=tuple(dict.fromkeys((
                    *self.fallback_reasons, "operand_cardinality_mismatch"))))
        operands = tuple(replace(
            promoted,
            owner_aliases=tuple(dict.fromkeys((
                *promoted.owner_aliases, *old.owner_aliases))),
            predicate_candidates=tuple(dict.fromkeys((
                *promoted.predicate_candidates, *old.predicate_candidates))),
            scope_candidates=tuple(dict.fromkeys((
                *promoted.scope_candidates, *old.scope_candidates))),
            temporal_constraint=(
                promoted.temporal_constraint
                if promoted.temporal_constraint == old.temporal_constraint else None),
            polarity=(promoted.polarity
                      if promoted.polarity == old.polarity else None),
        ) for promoted, old in zip(self.operands, legacy.operands))
        return replace(self, operands=operands, soft_fallback_applied=True)


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


def compose_operator(slots: QuerySlots, operands: Sequence[OperandSpec]) -> OperatorNode:
    """Build the operator tree from parsed slots.

    Composition is the whole point: "how many places have A and B both visited"
    is a count *over an intersection*, and collapsing it to either one alone
    loses the question.
    """
    leaves = tuple(FactSet(item.operand_id) for item in operands)
    if not leaves:
        raise ValueError("cannot compose an operator without operands")
    multi = len(leaves) > 1
    shared = slots.quantifier in {"both", "either"}
    per_owner = slots.quantifier in {"each", "every", "respectively"}

    def combined() -> OperatorNode:
        """How several operands relate before any outer operator applies."""
        if not multi:
            return leaves[0]
        if shared:
            return IntersectionDistinct(leaves, distinct_by=slots.distinct_by)
        if per_owner:
            return GroupByOwner(leaves, distinct_by=slots.distinct_by)
        return UnionDistinct(leaves, distinct_by=slots.distinct_by)

    # Existence outranks the quantifier: "Do both A and B have pets?" asks for a
    # witness per owner, not for the values they share.
    if slots.is_existence:
        return ExistsAll(leaves) if multi and (shared or per_owner) else (
            ExistsAll(leaves) if multi else ExistsAll((leaves[0],)))
    if slots.is_duration or slots.temporal_relation == "between":
        left, right = (leaves[0], leaves[1]) if multi else (leaves[0], leaves[0])
        return DateDifference(left, right)
    if slots.is_count:
        return CountDistinct(combined(), distinct_by=slots.distinct_by)
    if slots.ordinal_index is not None:
        return Ordinal(combined(), index=slots.ordinal_index, order=slots.ordinal_order)
    if slots.is_latest:
        return LatestState(combined())
    if slots.temporal_relation == "before":
        return ArgMinTime(combined())
    if slots.temporal_relation == "after":
        return ArgMaxTime(combined())
    if per_owner and multi:
        return GroupByOwner(leaves, distinct_by=slots.distinct_by)
    if shared and multi:
        return IntersectionDistinct(leaves, distinct_by=slots.distinct_by)
    if slots.is_list:
        # A plural answer head ("which cities") is a set question even for one
        # owner, and a set question carries a collection obligation.
        if multi or slots.expects_multiple:
            return UnionDistinct(leaves, distinct_by=slots.distinct_by)
        return Lookup(leaves[0])
    return Lookup(leaves[0]) if not multi else UnionDistinct(leaves, distinct_by=slots.distinct_by)


def _ast_operands(query: str, slots: QuerySlots, owners, predicates, scopes) -> tuple[OperandSpec, ...]:
    """Operand specs with the slots V5.5 never filled in.

    These are shadow copies: ``QueryIR.operands`` keeps the legacy shape so the
    reservoir and packer see exactly what they saw before.
    """
    rows = owners or (("", ()),)
    exhaustive = slots.is_count or slots.is_list or slots.quantifier in {"both", "each", "all", "every"}
    return tuple(OperandSpec(
        operand_id=stable_id("ast-operand", query, index, alias),
        owner_aliases=(alias,) if alias else (),
        predicate_candidates=predicates,
        scope_candidates=scopes,
        value_type=slots.value_type,
        temporal_constraint=slots.temporal_phrase or None,
        polarity=slots.polarity,
        multiplicity="exhaustive_set" if exhaustive else "at_least_one",
        distinct_by=slots.distinct_by,
    ) for index, (alias, _) in enumerate(rows))


def _ast_obligations(ast: OperatorNode, operands: Sequence[OperandSpec]) -> tuple[ProofObligation, ...]:
    by_id = {item.operand_id: item for item in operands}
    rows: list[ProofObligation] = []
    for operand_id in operand_ids(ast):
        operand = by_id.get(operand_id)
        if operand is None:
            continue
        for kind in ("binding", "provenance"):
            rows.append(ProofObligation(
                stable_id("ast-obligation", operand_id, kind), operand_id, kind))
    if requires_exhaustive_scope(ast):
        for operand_id in operand_ids(ast):
            rows.append(ProofObligation(
                stable_id("ast-obligation", operand_id, "collection"), operand_id, "collection"))
    # Extra obligations belong to every physical operator, not only the root.
    # For example Count(Ordinal(FactSet(...))) still needs an ordering proof.
    for index, node in enumerate(walk(ast)):
        kind = _AST_EXTRA_OBLIGATION.get(type(node))
        if kind:
            rows.append(ProofObligation(
                stable_id("ast-obligation", index, type(node).__name__, kind), None, kind))
    return tuple(rows)


_AST_EXTRA_OBLIGATION = {
    DateDifference: "time_endpoint",
    LatestState: "state_history",
    Ordinal: "ordering",
    ArgMinTime: "ordering",
    ArgMaxTime: "ordering",
}


def _scope_candidates(query: str, view: "GraphReadView") -> tuple[str, ...]:
    """Scopes mentioned by the question; V5.5 left this permanently empty."""
    query_terms = content_terms(query)
    if not query_terms:
        return ()
    scored: list[tuple[int, str]] = []
    for scope in getattr(view, "scope_fact_index", {}):
        overlap = len(query_terms & content_terms(scope))
        if overlap:
            scored.append((overlap, scope))
    return tuple(value for _, value in sorted(scored, key=lambda row: (-row[0], row[1]))[:2])


def compile_query(query: str, view: "GraphReadView", *,
                  registry: PrincipalRegistry | None = None) -> QueryIR:
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
    # Shadow compile: parse the slots and compose the AST alongside the legacy
    # decision, but execute neither of the AST's operands nor its operator yet.
    slots = parse_slots(query)
    scopes = _scope_candidates(query, view)
    resolved_owners: tuple[ResolvedOwner, ...] = ()
    owner_warnings: tuple[str, ...] = ()
    if registry is not None:
        resolved_owners, owner_warnings = resolve_query_owners(query, registry)
    # AST operands prefer the principal-aware resolution when it produced one;
    # the legacy alias list stays untouched so H0-H9 execution cannot move.
    ast_owner_rows = (tuple((row.mention_text, row.canonical_entity_ids)
                            for row in resolved_owners) or owners)
    ast_operands = _ast_operands(query, slots, ast_owner_rows, predicates, scopes)
    ast = compose_operator(slots, ast_operands)
    ast_operator = root_operator(ast)
    fallback_reasons: list[str] = []
    confidence = 1.0
    if ast_operator != operator:
        fallback_reasons.append("legacy_ast_operator_divergence")
        confidence -= 0.30
    if slots.warnings:
        fallback_reasons.extend(f"parse:{warning}" for warning in slots.warnings)
        confidence -= min(0.30, 0.10 * len(slots.warnings))
    if owner_warnings:
        fallback_reasons.extend(f"owner:{warning}" for warning in owner_warnings)
        confidence -= min(0.35, 0.12 * len(owner_warnings))
    if (slots.is_count or slots.expects_multiple) and not predicates:
        fallback_reasons.append("exhaustive_query_without_predicate_match")
        confidence -= 0.15
    confidence = max(0.0, min(1.0, confidence))
    return QueryIR(query, operator, operands, tuple(obligations), ordering, distinct_by,
                   ast=ast, ast_operands=ast_operands,
                   ast_obligations=_ast_obligations(ast, ast_operands), slots=slots,
                   parse_warnings=slots.warnings, resolved_owners=resolved_owners,
                   owner_resolution_warnings=owner_warnings,
                   compile_confidence=confidence,
                   fallback_reasons=tuple(dict.fromkeys(fallback_reasons)))
