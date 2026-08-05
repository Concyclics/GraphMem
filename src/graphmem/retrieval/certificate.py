from __future__ import annotations

from ..domain import CertificateStatus, EvidenceCertificate, StopReason
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
    )
