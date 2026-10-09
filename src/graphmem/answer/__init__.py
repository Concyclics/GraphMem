"""Answer composition for GraphMem V5.6.

Retrieval produces evidence; this package turns evidence into an answer.  It is
the only part of the read path allowed to make a generative call, and it makes
exactly one per question.

Nothing here may import ``graphmem.eval`` or read a gold label.
"""
from __future__ import annotations

from .composer import AnswerDraft, compose
from .aggregation import AggregationLedger, aggregation_operation, build_aggregation_ledger
from .prompts import (
    PROMPT_HASH, PROMPT_VERSION, build_answer_messages,
    is_preference_synthesis_query, prompt_contract, question_needs_global_date,
)
from .rendering import (
    AnswerConfig, RenderedEvidence, render_evidence, render_turn, resolve_evidence_order,
)
from .readout_policy import (
    ReadoutPolicyError, V5_54_POLICY, apply_readout_policy,
    apply_v5_54_readout,
)
from .stage import AnswerResult, AnswerStage, PreparedAnswer
from .evidence_card import (
    TypedEvidenceCard, build_typed_evidence_card, evidence_card_route,
)
from .ensemble import (
    ENSEMBLE_SCHEMA_VERSION, build_candidate_verifier_messages,
    build_ensemble_family_messages, combine_candidate_families,
    cross_family_consensus_choice, ensemble_route, parse_verifier_choice,
)
from .budget_controller import (
    BUDGET_CONTROLLER_SCHEMA_VERSION, BUDGET_POLICIES,
    AnswerBudgetDecision, answer_signature, candidate_signatures,
    decide_answer_budget, modal_candidate_choice, reliable_candidate_choice,
    stage_normalized_candidate_choice,
)
from .incremental import (
    INCREMENTAL_PROMPT_VERSION, MERGED_EXPANSION_PROMPT_VERSION,
    IncrementalEvidencePlan, build_incremental_answer_messages,
    build_merged_expansion_messages, plan_incremental_evidence,
)
from .verified_readout import (
    FOCUS_SYSTEM_PROMPT, RenderedVerifiedPrompt,
    build_source_focus_prompt, build_unified_source_prompt,
    build_verified_prompt,
)

__all__ = [
    "AggregationLedger", "AnswerConfig", "AnswerDraft", "AnswerResult", "AnswerStage", "PreparedAnswer",
    "PROMPT_HASH",
    "PROMPT_VERSION", "RenderedEvidence", "build_answer_messages", "compose",
    "aggregation_operation", "build_aggregation_ledger",
    "is_preference_synthesis_query", "prompt_contract", "render_evidence",
    "question_needs_global_date",
    "FOCUS_SYSTEM_PROMPT", "RenderedVerifiedPrompt",
    "build_source_focus_prompt", "build_unified_source_prompt",
    "build_verified_prompt",
    "ReadoutPolicyError", "V5_54_POLICY", "apply_readout_policy",
    "apply_v5_54_readout",
    "TypedEvidenceCard", "build_typed_evidence_card", "evidence_card_route",
    "ENSEMBLE_SCHEMA_VERSION", "build_candidate_verifier_messages",
    "build_ensemble_family_messages", "combine_candidate_families",
    "cross_family_consensus_choice", "ensemble_route",
    "parse_verifier_choice",
    "BUDGET_CONTROLLER_SCHEMA_VERSION", "BUDGET_POLICIES",
    "AnswerBudgetDecision", "answer_signature", "candidate_signatures",
    "decide_answer_budget", "reliable_candidate_choice",
    "modal_candidate_choice", "stage_normalized_candidate_choice",
    "INCREMENTAL_PROMPT_VERSION", "IncrementalEvidencePlan",
    "MERGED_EXPANSION_PROMPT_VERSION", "build_incremental_answer_messages",
    "build_merged_expansion_messages", "plan_incremental_evidence",
    "render_turn", "resolve_evidence_order",
]
