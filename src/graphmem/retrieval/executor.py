"""Typed, deterministic execution decision over the QueryIR algebra.

The result is shadow-safe by default: callers may log it or attach it as an
unverified prompt proposal.  ``safe_to_bypass`` becomes true only after a
post-pack certificate proves that every selected witness survived.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain import AlgebraResult, EvidenceCertificate, TruthValue


@dataclass(frozen=True, slots=True)
class TypedExecutionResult:
    answer_kind: str
    value: Any
    unit: str
    provenance_binding_ids: tuple[str, ...]
    interval_uncertainty: str
    contradiction_status: str
    safe_to_bypass: bool
    reason_codes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        if isinstance(self.value, bool):
            return "yes" if self.value else "no"
        if isinstance(self.value, tuple):
            return ", ".join(map(str, self.value))
        if self.value is None:
            return ""
        if self.answer_kind == "sum":
            rendered = (str(int(self.value))
                        if isinstance(self.value, float) and self.value.is_integer()
                        else str(self.value))
            return f"{rendered} {self.unit}".strip()
        return str(self.value)


def inspect_execution(
    result: AlgebraResult | None,
    certificate: EvidenceCertificate | None = None,
) -> TypedExecutionResult | None:
    if result is None:
        return None
    kind = result.answer_kind
    value: Any = None
    unit = ""
    uncertainty = "not_applicable"
    contradiction = "none"
    reasons = list(result.degradations)

    if kind == "count":
        value = result.count
        unit = "items"
        # A one-row manifest is not a closed-world proof by itself.  It may be
        # correct, but cannot authorize bypass without an independent closure
        # witness, which the current AlgebraResult does not yet carry.
        if value == 1:
            reasons.append("single_member_closed_world_unproven")
    elif kind in {"list", "ordinal"}:
        values = tuple(dict.fromkeys(
            member.value for member in result.members if member.value))
        value = values[0] if kind == "ordinal" and values else values
    elif kind == "group":
        value = tuple(
            f"{owner}: {', '.join(values)}"
            for owner, values in sorted(result.groups.items()))
    elif kind == "existence":
        truth = result.truth_value
        if truth is None:
            truth = TruthValue.TRUE if result.members else TruthValue.UNKNOWN
        value = (True if truth == TruthValue.TRUE
                 else False if truth == TruthValue.FALSE else None)
        if truth == TruthValue.UNKNOWN:
            reasons.append("open_world_unknown")
        elif truth == TruthValue.FALSE and not (
                certificate is not None
                and certificate.negative_scope_required
                and certificate.post_pack_complete):
            reasons.append("negative_scope_unproven")
    elif kind == "sum":
        value = result.numeric_total
        unit = result.unit
        if value is None:
            reasons.append("unresolved_numeric_sum")
    elif kind == "date_difference":
        endpoints = tuple(result.temporal_endpoints)
        if len(endpoints) == 2:
            value = endpoints[1].key.days_between(endpoints[0].key)
            unit = "days"
            precisions = {row.key.precision for row in endpoints}
            uncertainty = (
                "exact" if precisions <= {"second", "minute", "day"}
                and min(row.key.confidence for row in endpoints) >= 0.99
                else "bounded")
        if value is None:
            reasons.append("unresolved_temporal_difference")
    elif kind == "state":
        state = result.state_result
        if state is not None:
            value = state.current_value
            uncertainty = state.interval_uncertainty
            contradiction = state.contradiction_status
            if contradiction == "conflicting_latest":
                reasons.append("conflicting_latest_state")
    else:
        return None

    witnesses = tuple(dict.fromkeys((
        *result.output_binding_ids,
        *(row.binding_id for row in result.temporal_endpoints),
    )))
    if not result.scope_complete:
        reasons.append("scope_incomplete")
    if certificate is None:
        reasons.append("certificate_absent")
    elif not certificate.post_pack_complete:
        reasons.append("post_pack_certificate_incomplete")
    if value is None or value == ():
        reasons.append("empty_typed_value")
    reasons = list(dict.fromkeys(reasons))
    safe = not reasons and kind in {
        "count", "sum", "list", "group", "existence", "date_difference",
        "state", "ordinal"}
    return TypedExecutionResult(
        answer_kind=kind, value=value, unit=unit,
        provenance_binding_ids=witnesses,
        interval_uncertainty=uncertainty,
        contradiction_status=contradiction,
        safe_to_bypass=safe, reason_codes=tuple(reasons))
