"""Answer composition for GraphMem V5.6.

Retrieval produces evidence; this package turns evidence into an answer.  It is
the only part of the read path allowed to make a generative call, and it makes
exactly one per question.

Nothing here may import ``graphmem.eval`` or read a gold label.
"""
from __future__ import annotations

from .composer import AnswerDraft, compose
from .prompts import PROMPT_HASH, PROMPT_VERSION, build_answer_messages, prompt_contract
from .rendering import (
    AnswerConfig, RenderedEvidence, render_evidence, render_turn, resolve_evidence_order,
)
from .stage import AnswerResult, AnswerStage, PreparedAnswer

__all__ = [
    "AnswerConfig", "AnswerDraft", "AnswerResult", "AnswerStage", "PreparedAnswer",
    "PROMPT_HASH",
    "PROMPT_VERSION", "RenderedEvidence", "build_answer_messages", "compose",
    "prompt_contract", "render_evidence", "render_turn", "resolve_evidence_order",
]
