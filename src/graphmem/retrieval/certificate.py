from __future__ import annotations

from typing import Mapping, Sequence

from ..domain import (
    AlgebraResult,
    CertificateStatus,
    EvidenceCertificate,
    EvidenceUnit,
    FactBinding,
    StopReason,
)
from .algebra import ClosureResult
from .query_ir import QueryIR


def evaluate_certificate(ir: QueryIR, closure: ClosureResult, *, exhausted: bool = False,
                         no_progress: bool = False) -> EvidenceCertificate:
    required = tuple(item.obligation_id for item in ir.proof_obligations)
    operand_status = {}
    covered: list[str] = []
    for operand in ir.operands:
        has_binding = operand.operand_id in closure.complete_operands
        operand_status[operand.operand_id] = "complete" if has_binding else "missing_binding"
        for obligation in ir.proof_obligations:
            if obligation.operand_id != operand.operand_id:
                continue
            if obligation.kind in {"binding", "provenance"} and has_binding:
                covered.append(obligation.obligation_id)
            elif obligation.kind == "collection" and has_binding and closure.collection_complete:
                covered.append(obligation.obligation_id)
    missing = tuple(item for item in required if item not in covered)
    if not missing:
        status = CertificateStatus.COMPLETE; reason = StopReason.CERTIFICATE
    elif exhausted:
        status = CertificateStatus.BUDGET_EXHAUSTED; reason = StopReason.NODE_CAP
    elif no_progress:
        status = CertificateStatus.NO_PROGRESS; reason = StopReason.NO_PROGRESS
    elif any("collection" in item for item in missing):
        status = CertificateStatus.INCOMPLETE_COLLECTION; reason = None
    else:
        status = CertificateStatus.INCOMPLETE_OPERAND; reason = None
    return EvidenceCertificate(
        question_kind=str(ir.operator), required_slots=required, covered_slots=tuple(covered),
        missing_slots=missing, complete=not missing, iterations=1, status=status,
        operand_status=operand_status, stop_reason=reason,
        phase_status={
            "logical_compile": True,
            "pre_pack_closure": not missing,
            "witness_packed": False,
            "post_pack_complete": False,
        },
        unsatisfied_phase="pre_pack_closure" if missing else "witness_packed",
    )


def finalize_ast_certificate(
    ir: QueryIR,
    algebra: AlgebraResult,
    bindings: Sequence[FactBinding],
    group_turns: Mapping[str, tuple[str, ...]],
    packed_turn_ids: Sequence[str],
    *,
    units: Sequence[EvidenceUnit] = (),
    exhausted: bool = False,
) -> EvidenceCertificate:
    """Certify the AST result *after* the evidence budget has been applied.

    A pre-pack closure is not an answer certificate: the packer may remove one
    half of an intersection or a temporal endpoint.  This function recomputes
    every AST obligation against the final turn set and is deliberately strict:
    each output binding must retain at least one complete evidence group.
    """
    obligations = ir.ast_obligations or ir.proof_obligations
    required = tuple(row.obligation_id for row in obligations if row.required)
    packed = frozenset(packed_turn_ids)
    by_id = {row.binding_id: row for row in bindings}

    witness_ids = tuple(dict.fromkeys((
        *algebra.output_binding_ids,
        *(row.binding_id for row in algebra.temporal_endpoints),
        *((algebra.state_result.current_binding_id,) if algebra.state_result else ()),
        *(algebra.state_result.prior_binding_ids if algebra.state_result else ()),
    )))

    def binding_is_packed(binding_id: str) -> bool:
        binding = by_id.get(binding_id)
        if binding is None:
            return False
        groups = [frozenset(group_turns.get(group_id, ()))
                  for group_id in binding.evidence_group_ids]
        # One fully retained provenance group is sufficient to verify a binding;
        # partial spans are not.  Empty/missing groups cannot certify anything.
        return any(group and group <= packed for group in groups)

    packed_witnesses = frozenset(
        binding_id for binding_id in witness_ids if binding_is_packed(binding_id))
    complete_operands = frozenset(algebra.complete_operands)
    output_by_operand: dict[str, tuple[str, ...]] = {}
    for operand_id in complete_operands:
        output_by_operand[operand_id] = tuple(
            binding_id for binding_id in witness_ids
            if binding_id in by_id and by_id[binding_id].operand_id == operand_id)

    covered: list[str] = []
    operand_status: dict[str, str] = {}
    for operand in (ir.ast_operands or ir.operands):
        output_ids = output_by_operand.get(operand.operand_id, ())
        bound = operand.operand_id in complete_operands
        provenance = bool(output_ids) and all(item in packed_witnesses for item in output_ids)
        operand_status[operand.operand_id] = (
            "complete" if bound and provenance else
            "missing_provenance" if bound else "missing_binding")

    for obligation in obligations:
        satisfied = False
        if obligation.kind == "binding":
            satisfied = bool(obligation.operand_id in complete_operands)
        elif obligation.kind == "provenance":
            ids = output_by_operand.get(obligation.operand_id or "", ())
            satisfied = bool(ids) and all(item in packed_witnesses for item in ids)
        elif obligation.kind == "collection":
            satisfied = algebra.scope_complete
        elif obligation.kind == "time_endpoint":
            ids = tuple(row.binding_id for row in algebra.temporal_endpoints)
            satisfied = len(ids) == 2 and all(item in packed_witnesses for item in ids)
        elif obligation.kind == "ordering":
            ids = algebra.output_binding_ids
            satisfied = bool(ids) and all(
                item in packed_witnesses
                and item in by_id
                and by_id[item].time_interval is not None
                and by_id[item].time_interval.resolved
                for item in ids)
        elif obligation.kind == "state_history":
            state = algebra.state_result
            ids = ((state.current_binding_id, *state.prior_binding_ids) if state else ())
            satisfied = bool(ids) and algebra.scope_complete and all(
                item in packed_witnesses for item in ids)
        if satisfied:
            covered.append(obligation.obligation_id)

    missing = tuple(item for item in required if item not in covered)
    logical_complete = bool(ir.ast is not None and not ir.parse_warnings)
    pre_pack_complete = bool(
        logical_complete and algebra.scope_complete and not algebra.degradations)
    witness_complete = bool(witness_ids) and all(
        item in packed_witnesses for item in witness_ids)
    post_pack_complete = bool(pre_pack_complete and witness_complete and not missing)

    dropped_units = tuple(
        unit.unit_id for unit in units
        if unit.mandatory and not frozenset(unit.source_turn_ids) <= packed)
    if post_pack_complete:
        status = CertificateStatus.COMPLETE
        reason = StopReason.CERTIFICATE
        unsatisfied = None
    elif exhausted:
        status = CertificateStatus.BUDGET_EXHAUSTED
        reason = StopReason.NODE_CAP
        unsatisfied = "post_pack_complete"
    elif not pre_pack_complete:
        status = (CertificateStatus.INCOMPLETE_COLLECTION
                  if not algebra.scope_complete else CertificateStatus.INCOMPLETE_OPERAND)
        reason = None
        unsatisfied = "pre_pack_closure"
    else:
        status = CertificateStatus.INCOMPLETE_OPERAND
        reason = None
        unsatisfied = "witness_packed"

    return EvidenceCertificate(
        question_kind=str(algebra.operator),
        required_slots=required,
        covered_slots=tuple(covered),
        missing_slots=missing,
        complete=pre_pack_complete,
        iterations=1,
        status=status,
        operand_status=operand_status,
        stop_reason=reason,
        phase_status={
            "logical_compile": logical_complete,
            "pre_pack_closure": pre_pack_complete,
            "witness_packed": witness_complete,
            "post_pack_complete": post_pack_complete,
        },
        post_pack_complete=post_pack_complete,
        unsatisfied_phase=unsatisfied,
        dropped_mandatory_unit_ids=dropped_units,
    )
