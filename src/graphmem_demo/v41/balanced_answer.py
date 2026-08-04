"""Balanced single-call answer policy for frozen GraphMem V4.1."""
from __future__ import annotations

import json
from typing import Any

from ..models import QuestionCase, RetrievedContext
from .agentic_answer import _candidate_sources, _operator_ledger

_ALGEBRA = {
    "collection": "Silently enumerate all exact scoped members or operands. Deduplicate true repeats, keep distinct conjoined items, apply add/remove/cancel/replacement, then return the requested list/count/aggregate.",
    "dialogue_lookup": "Pair the exact request with its reply. Return the requested concrete slot from the reply, not the request, topic, or neighboring scene.",
    "state_update": "Bind one owner/entity/attribute/context. Select latest valid or requested historical state and preserve planned/completed/cancelled lifecycle.",
    "temporal_lookup": "Bind the exact event. Resolve relative time from its source date and return the requested date/age/duration type, not a nearby timestamp.",
    "temporal_comparison": "Bind both exact event endpoints, normalize their times, then order/subtract in the question's direction.",
    "preference_recommendation": "Preserve owner attribution, positive/negative preferences, established practices, and the requested recommendation scope.",
    "multi_hop": "Keep all hops on one coherent entity/event chain and return the final requested slot.",
    "inferential_profile": "Make one narrow ordinary inference from direct premises; do not infer sensitive identity or diagnosis without direct evidence.",
    "reference_identity": "Resolve all discriminating clues in one coherent scene and return the concrete identity.",
}

_HIGH_CONFIDENCE = {
    "dialogue_attribute_item_match", "relative_time_from_lossless_source",
    "relative_anchor_source_lookup", "latest_valid_state",
    "event_identity_collection_members", "temporal_order_from_lossless_sources",
    "temporal_sequence_from_lossless_sources", "time_difference_from_lossless_sources",
    "duration_total", "source_bound_explicit_date", "age_arithmetic_from_lossless_sources",
    "explicit_operand_currency_sum", "labeled_collection_subtotal_sum",
    "transaction_sum_from_lossless_sources", "record_time_extreme",
    "scoped_completed_event_members", "scoped_completed_duration_total",
    "latest_scalar_state_from_lossless_sources", "latest_labeled_currency_state",
    "weekly_schedule_distinct_days", "current_role_duration_from_lossless_sources",
    "relative_duration_at_event", "relative_value_multiplier_from_lossless_sources",
    "dialogue_final_choice_from_lossless_sources", "currency_extreme_entity_from_lossless_sources",
}


def _trusted_ledger(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _operator_ledger(trace) if row.get("operation") in _HIGH_CONFIDENCE][:6]


def answer_messages(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, str]]:
    trace = retrieval.retrieval_trace or {}
    ir = trace.get("v41_query_augmentation") or {}
    algebra = str(ir.get("answer_algebra") or "direct_fact")
    focused = _candidate_sources(case, retrieval, limit=8)
    payload = {
        "question_date": case.question_date,
        "question": case.question,
        "answer_algebra": algebra,
        "algebra_instruction": _ALGEBRA.get(algebra, "Bind exact owner, entity, relation, scope, lifecycle, and requested value type."),
        "query_ir": {k: ir.get(k) for k in ("target_entities", "target_relation", "target_owner", "requested_value_type", "temporal_constraints", "state_constraints", "collection_constraints", "polarity", "required_roles", "alternative_entities", "event_identity_terms", "scope_boundary") if ir.get(k) not in (None, "", [], {})},
        "source_certified_algebra": _trusted_ledger(trace),
        "focused_lossless_sources": focused,
        "full_graphmem_evidence": retrieval.context_text,
    }
    system = (
        "Answer one memory question from GraphMem V4.1 evidence. Lossless source turns are authoritative; frames and cards are navigation aids. Focused sources are duplicated only to make relevant evidence visible and do not replace the full evidence. Do not abstain because a diagnostic role is missing; answer whenever exact evidence or all computable operands are present. A source-certified algebra row is a high-priority candidate, but verify its entity and scope against its cited source. In particular, copy a dialogue_attribute_item_match concrete candidate and execute a relative_time_from_lossless_source value/unit when their source matches the question. Preserve speaker ownership, relation direction, negation, lifecycle, exact names, numbers, units, and dates. Never transfer a value from a sibling entity or event. Use insufficient evidence only after checking the full evidence and finding an exact binding or required operand truly absent. Reason silently and output only the concise final answer."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
