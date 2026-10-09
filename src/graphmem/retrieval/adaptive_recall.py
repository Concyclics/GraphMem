"""QueryIR/closure-gated evidence expansion.

The normal retrieval path keeps a validated precision-oriented evidence pack.
This module decides whether that pack is structurally under-specified before a
larger source-turn budget is allowed.  The decision deliberately consumes only
query-plan and evidence-certificate state; benchmark labels, gold turns,
answers, and judge results are not part of the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ..domain import EvidenceCertificate, ProofObligation, QueryOperator
from .query_ir import QueryIR


_TEMPORAL_RE = re.compile(
    r"\b(?:when|what\s+(?:date|time)|how\s+long|before|after|since|until|"
    r"first|last|latest|earliest|earlier|later|ago|elapsed|"
    r"days?|weeks?|months?|years?)\b",
    re.I,
)
_AGGREGATE_RE = re.compile(
    r"\b(?:how\s+many|how\s+much|how\s+often|count|number\s+of|total|"
    r"combined|average|sum|altogether|in\s+all)\b",
    re.I,
)
_MULTI_HOP_RE = re.compile(
    r"\b(?:both|in\s+common|shared|respectively|relationship\s+between|"
    r"based\s+on|because\s+of|result(?:ed)?\s+in|lead(?:ing)?\s+to|"
    r"who\s+.*\s+(?:that|which)|what\s+.*\s+(?:that|which))\b",
    re.I,
)
_INFERENCE_RE = re.compile(
    r"\b(?:would|could|might|likely|prefer|open\s+to|benefit\s+from|"
    r"infer(?:red|ence)?|imply|suggests?|indicates?|"
    r"what\s+does\s+.*\s+say\s+about)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class AdaptiveRecallPlan:
    """A bounded expansion decision for one query."""

    base_turns: int
    target_turns: int
    route: str
    reasons: tuple[str, ...] = ()
    severity: int = 0
    base_tokens: int = 0
    target_tokens: int = 0
    in_budget_repair: bool = False

    @property
    def triggered(self) -> bool:
        return bool(
            self.target_turns > self.base_turns
            or (self.base_tokens > 0
                and self.target_tokens > self.base_tokens))

    @property
    def active(self) -> bool:
        """Whether the evidence pack must be rebuilt at any budget tier."""

        return self.triggered or self.in_budget_repair


def _route(ir: QueryIR) -> str:
    query = ir.query
    slots = ir.slots
    operator = ir.ast_operator or ir.operator
    if (operator in {QueryOperator.COUNT_DISTINCT, QueryOperator.SUM}
            or (slots is not None and (slots.is_count or slots.is_sum))
            or _AGGREGATE_RE.search(query)):
        return "aggregate"
    if (operator in {
            QueryOperator.DATE_DIFFERENCE, QueryOperator.ARGMIN_TIME,
            QueryOperator.ARGMAX_TIME, QueryOperator.ORDINAL,
            QueryOperator.LATEST_STATE,
        }
            or (slots is not None and (
                slots.is_duration or slots.is_latest
                or slots.temporal_relation
                or slots.ordinal_index is not None))
            or _TEMPORAL_RE.search(query)):
        return "temporal"
    if len(ir.operands) > 1 or _MULTI_HOP_RE.search(query):
        return "multi_hop"
    if (slots is not None and slots.is_advice) or _INFERENCE_RE.search(query):
        return "inference"
    if (operator in {
            QueryOperator.UNION_DISTINCT, QueryOperator.INTERSECTION_DISTINCT,
            QueryOperator.GROUP_BY_OWNER,
        }
            or (slots is not None and slots.expects_multiple)):
        return "collection"
    return "lookup"


def _missing_kinds(
    ir: QueryIR, certificate: EvidenceCertificate,
) -> tuple[str, ...]:
    missing = frozenset(certificate.missing_slots)
    obligations: Sequence[ProofObligation] = (
        ir.ast_obligations or ir.proof_obligations)
    return tuple(dict.fromkeys(
        obligation.kind for obligation in obligations
        if obligation.required and obligation.obligation_id in missing))


def plan_adaptive_recall(
    ir: QueryIR,
    closure_certificate: EvidenceCertificate,
    packed_certificate: EvidenceCertificate,
    *,
    hard_turn_limit: int,
    base_turns: int = 64,
    medium_turns: int = 80,
    maximum_turns: int = 96,
    hard_token_limit: int = 0,
    base_tokens: int = 0,
    medium_tokens: int = 0,
    maximum_tokens: int = 0,
    compile_confidence_threshold: float = 0.80,
    minimum_severity: int = 3,
    maximum_tier_severity: int = 6,
    enable_in_budget_repair: bool = False,
    repair_minimum_severity: int = 2,
    lookup_minimum_severity: int = 0,
    expand_queryir_soft_fallback: bool = False,
    base_token_cap_reached: bool = False,
    candidate_count: int = 0,
) -> AdaptiveRecallPlan:
    """Choose a larger evidence budget only for an observable plan deficit.

    A larger context is not a generic confidence fallback.  Expansion requires
    either a complex query whose logical/evidence closure is incomplete, or an
    uncertain QueryIR compilation combined with an incomplete packed witness
    set.  The returned target remains bounded by the caller's hard budget and
    the available candidate reservoir.
    """

    hard = max(1, int(hard_turn_limit))
    base = min(hard, max(1, int(base_turns)))
    medium = min(hard, max(base, int(medium_turns)))
    maximum = min(hard, max(medium, int(maximum_turns)))
    if candidate_count:
        hard = min(hard, max(1, int(candidate_count)))
        base = min(base, hard)
        medium = min(medium, hard)
        maximum = min(maximum, hard)

    hard_token = max(0, int(hard_token_limit))
    if hard_token:
        base_token = min(
            hard_token, max(1, int(base_tokens or hard_token)))
        medium_token = min(
            hard_token, max(base_token, int(medium_tokens or hard_token)))
        maximum_token = min(
            hard_token, max(medium_token, int(maximum_tokens or hard_token)))
    else:
        base_token = medium_token = maximum_token = 0

    route = _route(ir)
    missing_kinds = _missing_kinds(ir, packed_certificate)
    closure_incomplete = not closure_certificate.complete
    pack_incomplete = not packed_certificate.post_pack_complete
    queryir_uncertain = bool(
        ir.soft_fallback_applied
        or ir.compile_confidence < compile_confidence_threshold
        or ir.parse_warnings
        or ir.owner_resolution_warnings)
    complex_route = route in {
        "aggregate", "temporal", "multi_hop", "inference", "collection"}

    reasons: list[str] = []
    severity = 0
    if closure_incomplete:
        reasons.append("pre_pack_closure_incomplete")
        severity += 2
    if pack_incomplete:
        reasons.append("packed_witness_incomplete")
        severity += 1
    if queryir_uncertain:
        reasons.append("queryir_uncertain")
        severity += 1
    if len(ir.operands) > 1:
        reasons.append("multiple_operands")
        severity += 1
    critical_missing = frozenset(missing_kinds) & {
        "time_endpoint", "ordering", "state_history", "collection",
        "binding", "provenance",
    }
    if critical_missing:
        reasons.extend(f"missing_{kind}" for kind in sorted(critical_missing))
        severity += 1

    # Generic one-fact lookups are deliberately excluded unless compilation is
    # itself uncertain.  Extra topical turns are particularly harmful there.
    standard_eligible = bool(
        candidate_count > base
        and pack_incomplete
        and severity >= minimum_severity
        and ((complex_route and (closure_incomplete or critical_missing))
             or (queryir_uncertain and (closure_incomplete or complex_route))))
    lookup_eligible = bool(
        lookup_minimum_severity > 0
        and candidate_count > base
        and pack_incomplete
        and route == "lookup"
        and severity >= lookup_minimum_severity)
    soft_fallback_eligible = bool(
        expand_queryir_soft_fallback
        and candidate_count > base
        and pack_incomplete
        and ir.soft_fallback_applied)
    if lookup_eligible:
        reasons.append("lookup_witness_expansion")
    if soft_fallback_eligible:
        reasons.append("queryir_soft_fallback_expansion")
    eligible = standard_eligible or lookup_eligible or soft_fallback_eligible
    token_eligible = bool(
        hard_token
        and base_token_cap_reached
        and pack_incomplete
        and severity >= repair_minimum_severity)
    repair_eligible = bool(
        enable_in_budget_repair
        and candidate_count > base
        and pack_incomplete
        and severity >= repair_minimum_severity
        # A generic missing-provenance flag does not identify which validated
        # tail item is safe to replace.  Require an independently observable
        # compiler uncertainty signal for same-budget surgery; measured broad
        # repair lost three complete gold packs for every one it rescued.
        and queryir_uncertain)
    if not eligible and not token_eligible:
        if repair_eligible:
            reasons.append("in_budget_witness_repair")
        return AdaptiveRecallPlan(
            base, base, route, tuple(dict.fromkeys(reasons)), severity,
            base_tokens=base_token, target_tokens=base_token,
            in_budget_repair=repair_eligible)

    # The maximum tier is reserved for compound deficits.  A single weak
    # diagnostic is common in otherwise answerable packs and does not justify
    # doubling the evidence tail; the threshold is configurable and audited.
    severe = severity >= maximum_tier_severity
    target = (maximum if severe else medium) if eligible else base
    # Turn and Token escalation are deliberately orthogonal.  A logical gap
    # may justify looking farther into the ranked reservoir without granting a
    # larger rendered context; evidence Tokens grow only after the first pack
    # demonstrably hit its Token cap.
    target_token = (
        maximum_token if severe else medium_token
    ) if token_eligible else base_token
    if token_eligible:
        reasons.append("evidence_token_cap_reached")
    return AdaptiveRecallPlan(
        base, max(base, target), route, tuple(dict.fromkeys(reasons)), severity,
        base_tokens=base_token, target_tokens=target_token,
        in_budget_repair=False)
