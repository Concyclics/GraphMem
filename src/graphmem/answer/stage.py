"""The answer stage: one bounded, cached, deterministic call per question.

Contracts this stage holds:

* **One generative call per question, at temperature 0, thinking disabled.**
  Retrieval must still make zero generative calls; this is the only such call
  in the read path.
* **The budget is enforced, not reported.**  Evidence is rendered under
  ``max_answer_tokens``; if the assembled prompt still exceeds it, optional
  turns are dropped and the budget is relaxed at most to
  ``max_answer_tokens_hard``.  Beyond that the question fails loudly rather
  than silently overspending.
* **Answers are cached by prompt bytes.**  Re-running a scored configuration
  costs nothing and returns byte-identical text.
* **No gold, ever.**  This module must stay importable without ``graphmem.eval``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config
from ..domain import (
    AlgebraResult, NavigationResult, QueryBudget, SourceTurn, canonical_json, stable_id,
)
from ..storage import SQLiteGraphStore
from ..tokenization import resolve_token_counter
from .composer import AnswerDraft, compose
from .prompts import PROMPT_HASH, PROMPT_VERSION, build_answer_messages
from .rendering import AnswerConfig, RenderedEvidence, render_evidence


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question_id: str
    memory_id: str
    prediction: str
    evidence_turn_ids: tuple[str, ...]
    dropped_turn_ids: tuple[str, ...]
    evidence_tokens: int
    prompt_tokens: int
    completion_tokens: int
    closed_form: bool
    draft_text: str = ""
    draft_certified: bool = False
    cached: bool = False
    budget_relaxed: bool = False
    prompt_hash: str = PROMPT_HASH
    latency_ms: float = 0.0
    warnings: tuple[str, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)


class AnswerStage:
    """Renders packed evidence and produces one answer per question."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, *, answer_config: AnswerConfig | None = None,
                 client: Any | None = None, require_exact_tokenizer: bool = True) -> None:
        self.store = store
        self.config = config
        self.dataset_hash = dataset_hash
        self.answer_config = answer_config or AnswerConfig()
        self.counter = resolve_token_counter(config.models.llm_model,
                                             require_exact=require_exact_tokenizer)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
        self.client = client
        self._turn_cache: dict[str, dict[str, SourceTurn]] = {}
        self._session_order: dict[str, dict[str, int]] = {}

    # -- evidence ---------------------------------------------------------

    def _turns(self, memory_id: str) -> dict[str, SourceTurn]:
        cached = self._turn_cache.get(memory_id)
        if cached is None:
            cached = {turn.turn_id: turn for turn in self.store.turns(memory_id)}
            # Single-memory cache: navigation walks memories in order and a full
            # corpus of turn text will not fit in memory at 510 graphs.
            self._turn_cache = {memory_id: cached}
            self._session_order = {memory_id: {
                session.session_id: session.ordinal
                for session in self.store.sessions(memory_id)}}
        return cached

    def render(self, result: NavigationResult, budget: QueryBudget,
               max_tokens: int | None = None) -> RenderedEvidence:
        turn_map = self._turns(result.memory_id)
        packed = result.packed_turn_ids or result.retrieved_turn_ids
        spans = {
            unit_span.turn_id: tuple(
                span for unit in result.proof_units for span in unit.spans
                if span.turn_id == unit_span.turn_id)
            for unit in result.proof_units for unit_span in unit.spans
        }
        mandatory = tuple(dict.fromkeys(
            turn_id for unit in result.proof_units if unit.mandatory
            for turn_id in unit.source_turn_ids))
        return render_evidence(
            [turn_map[turn_id] for turn_id in packed if turn_id in turn_map],
            config=self.answer_config, counter=self.counter,
            max_tokens=max_tokens if max_tokens is not None else budget.max_answer_tokens,
            session_order=self._session_order.get(result.memory_id),
            spans_by_turn=spans, mandatory_turn_ids=mandatory,
        )

    # -- answering --------------------------------------------------------

    def answer(self, question_id: str, question: str, result: NavigationResult,
               budget: QueryBudget, *, question_date: str | None = None,
               algebra: AlgebraResult | None = None) -> AnswerResult:
        started = time.perf_counter()
        warnings: list[str] = []
        draft: AnswerDraft | None = (
            compose(algebra, result.certificate) if self.answer_config.closed_form_enabled else None)

        evidence = self.render(result, budget)
        if evidence.mandatory_dropped:
            warnings.append("mandatory_turn_dropped_for_budget")
        messages = build_answer_messages(
            question=question, question_date=question_date, evidence_text=evidence.text,
            candidate_answer=draft.text if draft else None)
        prompt_tokens = self._prompt_tokens(messages)
        relaxed = False
        if prompt_tokens > budget.max_answer_tokens:
            # The rendered evidence fit, but the prompt scaffolding pushed the
            # total over.  Re-render against the remaining headroom before
            # touching the hard ceiling.
            overhead = prompt_tokens - evidence.tokens
            evidence = self.render(result, budget,
                                   max_tokens=max(1, budget.max_answer_tokens - overhead))
            messages = build_answer_messages(
                question=question, question_date=question_date, evidence_text=evidence.text,
                candidate_answer=draft.text if draft else None)
            prompt_tokens = self._prompt_tokens(messages)
            if prompt_tokens > budget.max_answer_tokens:
                relaxed = True
                warnings.append("answer_budget_relaxed_to_hard_ceiling")
        if prompt_tokens > budget.max_answer_tokens_hard:
            raise RuntimeError(
                f"answer prompt for {question_id} is {prompt_tokens} tokens, above the hard "
                f"ceiling {budget.max_answer_tokens_hard}")

        text, completion_tokens, cached = self._call(question_id, result.memory_id, messages)
        prediction = " ".join(text.split())
        if not prediction:
            warnings.append("empty_prediction")
        return AnswerResult(
            question_id=question_id, memory_id=result.memory_id, prediction=prediction,
            evidence_turn_ids=evidence.turn_ids, dropped_turn_ids=evidence.dropped_turn_ids,
            evidence_tokens=evidence.tokens, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            closed_form=bool(draft and draft.certified),
            draft_text=draft.text if draft else "",
            draft_certified=bool(draft and draft.certified),
            cached=cached, budget_relaxed=relaxed, latency_ms=(time.perf_counter() - started) * 1000,
            warnings=tuple(warnings),
            trace={
                "prompt_version": PROMPT_VERSION,
                "span_window": self.answer_config.span_window,
                "packed_turns": len(evidence.turn_ids),
                "evidence_truncated": evidence.truncated,
                "token_counter": self.counter.describe(),
                "draft_kind": draft.answer_kind if draft else None,
                "draft_degradations": list(draft.degradations) if draft else [],
            },
        )

    def _prompt_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(self.counter.count_many([str(row["content"]) for row in messages]))

    def _call(self, question_id: str, memory_id: str,
              messages: Sequence[Mapping[str, str]]) -> tuple[str, int, bool]:
        request = {
            "model": self.config.models.llm_model, "messages": list(messages),
            "temperature": 0, "max_tokens": self.answer_config.max_output_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        identity = CacheIdentity(
            self.dataset_hash, self.config.models.llm_model, PROMPT_HASH,
            self.config.schema_version,
            hashlib.sha256(canonical_json({
                "span_window": self.answer_config.span_window,
                "include_dates": self.answer_config.include_dates,
                "include_speaker": self.answer_config.include_speaker,
                "max_output_tokens": self.answer_config.max_output_tokens,
            }).encode()).hexdigest(),
            "answer:" + hashlib.sha256(canonical_json(request["messages"]).encode()).hexdigest(),
        )
        key = identity.key()
        started = time.perf_counter()
        cached = self.store.cache_get(key)
        if cached:
            response, usage, is_cached = cached["response"], dict(cached["usage"]), True
            completion = int(usage.get("output_tokens", 0))
            usage = {"cached_input_tokens": int(usage.get("uncached_input_tokens", 0)),
                     "uncached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                     "total_tokens": int(usage.get("uncached_input_tokens", 0))}
        else:
            completion_result = self.client.chat.completions.create(**request)
            message = completion_result.choices[0].message
            if getattr(message, "reasoning_content", None):
                raise RuntimeError("answer stage returned reasoning content")
            response = {"content": message.content or "",
                        "model": getattr(completion_result, "model", ""),
                        "finish_reason": getattr(completion_result.choices[0], "finish_reason", None)}
            raw = getattr(completion_result, "usage", None)
            prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
            completion = int(getattr(raw, "completion_tokens", 0) or 0)
            usage = {"cached_input_tokens": 0, "uncached_input_tokens": prompt,
                     "output_tokens": completion, "reasoning_tokens": 0,
                     "total_tokens": prompt + completion}
            self.store.cache_put(key, "answer", request, response, usage, PROMPT_HASH)
            is_cached = False
        occurrence = self.store._read_one(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?",
            (memory_id, key))[0]
        self.store.log_llm_call(
            call_id=stable_id("llm-call", memory_id, key, is_cached, occurrence),
            memory_id=memory_id, stage="answer", cache_key=key, cached=is_cached,
            request=request, response=response, usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000, retry_count=0, batch_size=1,
            prompt_hash=PROMPT_HASH)
        return str(response.get("content", "")), completion, is_cached
