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

__all__ = [
    "AggregationLedger", "AnswerConfig", "AnswerDraft", "AnswerResult", "AnswerStage", "PreparedAnswer",
    "PROMPT_HASH",
    "PROMPT_VERSION", "RenderedEvidence", "build_answer_messages", "compose",
    "aggregation_operation", "build_aggregation_ledger",
    "is_preference_synthesis_query", "prompt_contract", "render_evidence",
    "question_needs_global_date",
    "ReadoutPolicyError", "V5_54_POLICY", "apply_readout_policy",
    "apply_v5_54_readout",
    "render_turn", "resolve_evidence_order",
]
