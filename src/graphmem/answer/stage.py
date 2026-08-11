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
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config
from ..domain import (
    AlgebraResult, NavigationResult, QueryBudget, SourceTurn, canonical_json, stable_id,
)
from ..storage import SQLiteGraphStore
from ..tokenization import resolve_token_counter
from .composer import AnswerDraft, compose
from .aggregation import AggregationLedger, build_aggregation_ledger
from ..retrieval.executor import inspect_execution
from .prompts import (
    PROMPT_HASH, build_answer_messages, is_preference_synthesis_query,
    prompt_contract, question_needs_global_date,
)
from .rendering import (
    AnswerConfig, RenderedEvidence, render_evidence, resolve_evidence_order,
)
from .readout_policy import apply_readout_policy


def _aggregation_source_reserve_ids(
    turns: Mapping[str, SourceTurn], turn_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return direct source statements only for generic two-role transcripts.

    Named multi-party datasets use the chat role as transport metadata, so
    preferring ``role=user`` there would systematically erase one participant.
    """

    generic_speakers = {"", "assistant", "system", "tool", "user", "human"}
    rows = [turns[turn_id] for turn_id in turn_ids if turn_id in turns]
    if any((turn.speaker or "").casefold().strip() not in generic_speakers
           for turn in rows):
        return ()
    return tuple(
        turn.turn_id for turn in rows
        if (turn.role or "").casefold().strip() in {"user", "human"})


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
    finish_reason: str = ""
    draft_text: str = ""
    draft_certified: bool = False
    cached: bool = False
    budget_relaxed: bool = False
    prompt_hash: str = PROMPT_HASH
    latency_ms: float = 0.0
    api_prompt_tokens: int = 0
    api_total_tokens: int = 0
    answer_model: str = ""
    prompt_payload_hash: str = ""
    warnings: tuple[str, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedAnswer:
    """Frozen answer-model request produced after navigation and packing."""

    question_id: str
    memory_id: str
    messages: tuple[Mapping[str, str], ...]
    evidence_turn_ids: tuple[str, ...]
    dropped_turn_ids: tuple[str, ...]
    evidence_tokens: int
    packing_prompt_tokens: int
    closed_form: bool
    draft_text: str
    draft_certified: bool
    budget_relaxed: bool
    prompt_hash: str
    prompt_payload_hash: str
    warnings: tuple[str, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)
    deterministic_prediction: str = ""
    preparation_latency_ms: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "memory_id": self.memory_id,
            "messages": [dict(row) for row in self.messages],
            "evidence_turn_ids": list(self.evidence_turn_ids),
            "dropped_turn_ids": list(self.dropped_turn_ids),
            "evidence_tokens": self.evidence_tokens,
            "packing_prompt_tokens": self.packing_prompt_tokens,
            "closed_form": self.closed_form,
            "draft_text": self.draft_text,
            "draft_certified": self.draft_certified,
            "budget_relaxed": self.budget_relaxed,
            "prompt_hash": self.prompt_hash,
            "prompt_payload_hash": self.prompt_payload_hash,
            "warnings": list(self.warnings),
            "trace": dict(self.trace),
            "deterministic_prediction": self.deterministic_prediction,
            "preparation_latency_ms": self.preparation_latency_ms,
        }

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> "PreparedAnswer":
        return cls(
            question_id=str(row["question_id"]),
            memory_id=str(row["memory_id"]),
            messages=tuple(dict(item) for item in row.get("messages", ())),
            evidence_turn_ids=tuple(map(str, row.get("evidence_turn_ids", ()))),
            dropped_turn_ids=tuple(map(str, row.get("dropped_turn_ids", ()))),
            evidence_tokens=int(row.get("evidence_tokens", 0)),
            packing_prompt_tokens=int(row.get("packing_prompt_tokens", 0)),
            closed_form=bool(row.get("closed_form")),
            draft_text=str(row.get("draft_text", "")),
            draft_certified=bool(row.get("draft_certified")),
            budget_relaxed=bool(row.get("budget_relaxed")),
            prompt_hash=str(row.get("prompt_hash", PROMPT_HASH)),
            prompt_payload_hash=str(row["prompt_payload_hash"]),
            warnings=tuple(map(str, row.get("warnings", ()))),
            trace=dict(row.get("trace", {})),
            deterministic_prediction=str(row.get("deterministic_prediction", "")),
            preparation_latency_ms=float(row.get("preparation_latency_ms", 0.0)),
        )


class AnswerStage:
    """Renders packed evidence and produces one answer per question."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, *, answer_config: AnswerConfig | None = None,
                 client: Any | None = None, require_exact_tokenizer: bool = True,
                 cache_store: SQLiteGraphStore | None = None,
                 answer_model: str | None = None,
                 answer_base_url: str | None = None,
                 answer_api_key_env: str | None = None,
                 answer_request_profile: str = "qwen",
                 packing_model: str | None = None) -> None:
        self.store = store
        # Scoring runs open the authority graph read-only, so answers and their
        # call ledger go to a separate writable sidecar.  Defaulting to ``store``
        # keeps the single-database case (and the tests) unchanged.
        self.cache_store = cache_store if cache_store is not None else store
        self.config = config
        self.dataset_hash = dataset_hash
        self.answer_config = answer_config or AnswerConfig()
        self.answer_model = answer_model or config.models.llm_model
        self.answer_request_profile = answer_request_profile
        if self.answer_request_profile not in {"qwen", "openai", "omit"}:
            raise ValueError("answer_request_profile must be qwen, openai or omit")
        self.counter = resolve_token_counter(packing_model or config.models.llm_model,
                                             require_exact=require_exact_tokenizer)
        if client is None:
            from openai import OpenAI
            api_key = (os.environ.get(answer_api_key_env, "")
                       if answer_api_key_env else "local")
            if answer_api_key_env and not api_key:
                raise RuntimeError(f"{answer_api_key_env} is required for answer calls")
            client = OpenAI(
                base_url=answer_base_url or config.models.llm_base_url,
                api_key=api_key)
        self.client = client
        self._turn_cache: dict[str, tuple[dict[str, SourceTurn], dict[str, int]]] = {}
        self._cache_lock = threading.Lock()

    # -- evidence ---------------------------------------------------------

    def _turns(self, memory_id: str) -> tuple[dict[str, SourceTurn], dict[str, int]]:
        """Turn map and session order for one memory, safely under concurrency.

        The previous single-entry cache replaced the whole dict on every miss.
        With answers rendered by 32 workers across different memories, one thread
        could hold memory A's turn map while another had already swapped
        ``_session_order`` to memory B, so A's turns were ordered by B's sessions
        -- or by nothing, when the lookup missed.  Both are returned together now
        and the cache is keyed and bounded rather than overwritten.
        """
        with self._cache_lock:
            entry = self._turn_cache.get(memory_id)
            if entry is None:
                entry = (
                    {turn.turn_id: turn for turn in self.store.turns(memory_id)},
                    {session.session_id: session.ordinal
                     for session in self.store.sessions(memory_id)},
                )
                # A full corpus of turn text will not fit at 510 graphs, so keep
                # a small working set rather than one entry or everything.
                if len(self._turn_cache) >= 8:
                    self._turn_cache.pop(next(iter(self._turn_cache)))
                self._turn_cache[memory_id] = entry
            return entry

    def render(self, result: NavigationResult, budget: QueryBudget,
               max_tokens: int | None = None, *, question: str = "",
               reserved_turn_ids: Sequence[str] = ()) -> RenderedEvidence:
        turn_map, session_order = self._turns(result.memory_id)
        packed = result.packed_turn_ids or result.retrieved_turn_ids
        spans = {
            unit_span.turn_id: tuple(
                span for unit in result.proof_units for span in unit.spans
                if span.turn_id == unit_span.turn_id)
            for unit in result.proof_units for unit_span in unit.spans
        }
        mandatory = tuple(dict.fromkeys(
            (*(
                turn_id for unit in result.proof_units if unit.mandatory
                for turn_id in unit.source_turn_ids),
             *reserved_turn_ids)))
        evidence_order = resolve_evidence_order(
            self.answer_config.evidence_order, question,
            str(result.trace.get("query_operator") or ""))
        render_config = replace(self.answer_config, evidence_order=evidence_order)
        ordered = [turn_id for turn_id in packed if turn_id in turn_map]
        prefixes: dict[str, str] = {}
        layout_stats = {"chain_count": 0, "chain_turns": 0,
                        "graph_group_count": 0, "graph_turns": 0,
                        "auxiliary_turns": 0}
        if evidence_order in {
                "topological_plain", "topological", "topological_recency"}:
            ordered, prefixes, layout_stats = self._topological_layout(
                result, ordered, turn_map, session_order,
                strongest_last=evidence_order == "topological_recency")
            if evidence_order == "topological_plain":
                # Pure graph-rerank ablation: preserve the topology-derived
                # order and block statistics, but do not reveal graph labels
                # or a graph-specific instruction to the answer backbone.
                prefixes = {}
        rendered = render_evidence(
            [turn_map[turn_id] for turn_id in ordered],
            config=render_config, counter=self.counter,
            max_tokens=max_tokens if max_tokens is not None else budget.max_answer_tokens,
            session_order=session_order,
            spans_by_turn=spans, mandatory_turn_ids=mandatory,
            prefixes_by_turn=prefixes,
        )
        return replace(
            rendered, layout_mode=evidence_order,
            chain_count=layout_stats["chain_count"],
            chain_turns=layout_stats["chain_turns"],
            graph_group_count=layout_stats["graph_group_count"],
            graph_turns=layout_stats["graph_turns"],
            auxiliary_turns=layout_stats["auxiliary_turns"])

    @staticmethod
    def _topological_layout(result: NavigationResult, packed: Sequence[str],
                            turns: Mapping[str, SourceTurn],
                            session_order: Mapping[str, int], *,
                            strongest_last: bool = False) -> tuple[
                                list[str], dict[str, str], dict[str, int]]:
        """Group packed evidence by proof topology without filtering turns.

        Proof units sharing a QueryIR operand form one chain.  Within a chain,
        root-to-leaf depth precedes source chronology.  Operand-associated
        candidates follow their proof chain. Remaining turns use small
        relevance-anchor windows. Blocks of every kind compete by their best
        upstream rank, so a weak graph binding cannot precede a strong unbound
        source turn merely because it has topology metadata.
        """

        packed_set = frozenset(packed)
        candidate_rank = {
            row.turn_id: rank for rank, row in enumerate(result.candidate_scores)
        }
        candidate_by_turn = {
            row.turn_id: row for row in result.candidate_scores
        }
        proof_units = [
            unit for unit in result.proof_units
            if unit.binding_ids and packed_set.intersection(unit.source_turn_ids)
        ]
        def proof_key(unit) -> tuple[str, str]:
            operand = (unit.operand_ids[0] if unit.operand_ids else "unbound")
            # A shared first edge is a real connected branch. Seed facts have
            # no path, so keep distinct members/bindings separate instead of
            # manufacturing one enormous operand block.
            branch = (f"edge:{unit.relation_path_ids[0]}"
                      if unit.relation_path_ids else
                      f"seed:{unit.member_key or unit.binding_ids[0]}")
            return operand, branch

        operand_keys: list[tuple[str, str]] = []
        for unit in proof_units:
            key = proof_key(unit)
            if key not in operand_keys:
                operand_keys.append(key)
        proof_operands = {key[0] for key in operand_keys}
        for turn_id in packed:
            row = candidate_by_turn.get(turn_id)
            for operand in (row.operand_ids if row else ()):
                if operand in proof_operands:
                    continue
                key = (operand, "support")
                if key not in operand_keys:
                    operand_keys.append(key)
        chain_of = {key: index + 1 for index, key in enumerate(operand_keys)}
        chain_best = {chain: 1 << 30 for chain in chain_of.values()}
        for unit in proof_units:
            key = proof_key(unit)
            chain = chain_of[key]
            chain_best[chain] = min(
                chain_best[chain],
                *(candidate_rank.get(turn_id, 1 << 30)
                  for turn_id in unit.source_turn_ids))
        for turn_id in packed:
            row = candidate_by_turn.get(turn_id)
            for operand in (row.operand_ids if row else ()):
                keys = [key for key in operand_keys if key[0] == operand]
                if not keys:
                    continue
                path_head = (f"edge:{row.graph_path_ids[0]}"
                             if row and row.graph_path_ids else "")
                exact_branch = [key for key in keys if key[1] == path_head]
                key = min(exact_branch or keys,
                          key=lambda item: (chain_best[chain_of[item]],
                                            chain_of[item]))
                chain = chain_of[key]
                chain_best[chain] = min(
                    chain_best[chain], candidate_rank.get(turn_id, 1 << 30))

        assignments: dict[str, tuple] = {}
        prefixes: dict[str, str] = {}
        for unit in sorted(proof_units, key=lambda item: (
                chain_of.get(proof_key(item), 1 << 20),
                len(item.relation_path_ids), tuple(item.relation_path_ids),
                item.rank, item.unit_id)):
            key = proof_key(unit)
            chain = chain_of[key]
            depth = len(unit.relation_path_ids)
            for turn_id in unit.source_turn_ids:
                if turn_id not in packed_set or turn_id not in turns:
                    continue
                turn = turns[turn_id]
                chronological = (
                    session_order.get(turn.session_id, 1 << 30),
                    turn.session_id, turn.turn_index, turn.turn_id)
                value = (chain_best[chain], 0, chain, 0, depth,
                         tuple(unit.relation_path_ids),
                         unit.rank, chronological)
                if turn_id not in assignments or value < assignments[turn_id]:
                    assignments[turn_id] = value
                    prefixes[turn_id] = f"[CHAIN {chain} step={depth}]"

        for turn_id in packed:
            if turn_id in assignments or turn_id not in turns:
                continue
            row = candidate_by_turn.get(turn_id)
            matching = [key for key in operand_keys
                        if row is not None and key[0] in row.operand_ids]
            if not matching:
                continue
            path_head = (f"edge:{row.graph_path_ids[0]}"
                         if row and row.graph_path_ids else "")
            exact_branch = [key for key in matching if key[1] == path_head]
            key = min(exact_branch or matching,
                      key=lambda item: (chain_best[chain_of[item]],
                                        chain_of[item]))
            chain = chain_of[key]
            turn = turns[turn_id]
            chronological = (
                session_order.get(turn.session_id, 1 << 30),
                turn.session_id, turn.turn_index, turn.turn_id)
            assignments[turn_id] = (
                chain_best[chain], 0, chain, 1,
                candidate_rank.get(turn_id, 1 << 30),
                chronological)
            prefixes[turn_id] = f"[CHAIN {chain} support]"

        # A graph-reached candidate can be useful even when permissive/partial
        # QueryIR binding did not attach an operand.  Group those turns by their
        # first traversal edge and sort each branch root-to-leaf.  This is the
        # missing topology for the 71/200 questions that had no proof chain at
        # all in the first presentation ablation.
        graph_rows: dict[str, list[str]] = {}
        for turn_id in packed:
            if turn_id in assignments:
                continue
            row = candidate_by_turn.get(turn_id)
            if row is None or not row.graph_path_ids:
                continue
            graph_rows.setdefault(row.graph_path_ids[0], []).append(turn_id)
        graph_keys = sorted(graph_rows, key=lambda key: (
            min(candidate_rank.get(turn_id, 1 << 30)
                for turn_id in graph_rows[key]), key))
        graph_group = {key: index + 1 for index, key in enumerate(graph_keys)}
        graph_best = {
            key: min(candidate_rank.get(turn_id, 1 << 30)
                     for turn_id in graph_rows[key])
            for key in graph_keys
        }
        for key in graph_keys:
            group = graph_group[key]
            for turn_id in graph_rows[key]:
                row = candidate_by_turn[turn_id]
                turn = turns[turn_id]
                path = tuple(row.graph_path_ids)
                chronological = (
                    session_order.get(turn.session_id, 1 << 30),
                    turn.session_id, turn.turn_index, turn.turn_id)
                assignments[turn_id] = (
                    graph_best[key], 1, group, len(path), path,
                    candidate_rank.get(turn_id, 1 << 30), chronological)
                prefixes[turn_id] = (
                    f"[GRAPH {group} step={len(path)}]")

        auxiliary = [turn_id for turn_id in packed
                     if turn_id not in assignments and turn_id in turns]
        remaining_aux = set(auxiliary)
        aux_groups: list[tuple[int, str, list[str]]] = []
        for anchor_id in sorted(auxiliary, key=lambda turn_id: (
                candidate_rank.get(turn_id, 1 << 30), turn_id)):
            if anchor_id not in remaining_aux:
                continue
            anchor = turns[anchor_id]
            members = sorted(
                (turn_id for turn_id in remaining_aux
                 if turns[turn_id].session_id == anchor.session_id
                 and abs(turns[turn_id].turn_index - anchor.turn_index) <= 2),
                key=lambda turn_id: (
                    turns[turn_id].turn_index,
                    candidate_rank.get(turn_id, 1 << 30), turn_id))
            if not members:
                members = [anchor_id]
            remaining_aux.difference_update(members)
            aux_groups.append((candidate_rank.get(anchor_id, 1 << 30),
                               anchor_id, members))
        for group, (anchor_rank, _anchor_id, members) in enumerate(
                aux_groups, start=1):
            for turn_id in members:
                turn = turns[turn_id]
                rank = candidate_rank.get(turn_id, 1 << 30)
                assignments[turn_id] = (
                    anchor_rank, 2, group, turn.turn_index,
                    rank, turn.turn_id)
                # Group id and member rank are sufficient to recover both the
                # local packet and its relevance.  Repeating ``group=`` plus
                # the identical anchor rank on every member cost hundreds of
                # prompt tokens at 64 turns without adding topology.
                prefixes[turn_id] = f"[AUX {group} rank={rank + 1}]"

        if strongest_last:
            assignments = {
                turn_id: (-value[0], *value[1:])
                for turn_id, value in assignments.items()
            }
        ordered = sorted(
            (turn_id for turn_id in packed if turn_id in assignments),
            key=lambda turn_id: assignments[turn_id])
        chain_turns = sum(prefix.startswith("[CHAIN")
                          for prefix in prefixes.values())
        graph_turns = sum(prefix.startswith("[GRAPH")
                          for prefix in prefixes.values())
        auxiliary_turns = sum(prefix.startswith("[AUX")
                              for prefix in prefixes.values())
        return ordered, prefixes, {
            "chain_count": len(chain_of),
            "chain_turns": chain_turns,
            "graph_group_count": len(graph_keys),
            "graph_turns": graph_turns,
            "auxiliary_turns": auxiliary_turns,
        }

    # -- answering --------------------------------------------------------

    def answer(self, question_id: str, question: str, result: NavigationResult,
               budget: QueryBudget, *, question_date: str | None = None,
               algebra: AlgebraResult | None = None) -> AnswerResult:
        return self.complete(self.prepare(
            question_id, question, result, budget,
            question_date=question_date, algebra=algebra))

    def prepare(self, question_id: str, question: str, result: NavigationResult,
                budget: QueryBudget, *, question_date: str | None = None,
                algebra: AlgebraResult | None = None) -> PreparedAnswer:
        """Freeze navigation, evidence packing and exact prompt bytes."""

        started = time.perf_counter()
        warnings: list[str] = []
        draft: AnswerDraft | None = (
            compose(algebra, result.certificate)
            if self.answer_config.closed_form_enabled else None)
        typed_execution = inspect_execution(algebra, result.certificate)
        evidence_order = resolve_evidence_order(
            self.answer_config.evidence_order, question,
            str(result.trace.get("query_operator") or ""))
        preference_synthesis = (
            self.answer_config.preference_synthesis_enabled
            and is_preference_synthesis_query(question))
        evidence = self.render(result, budget, question=question)
        turn_map, _session_order = self._turns(result.memory_id)
        def make_ledger(rendered: RenderedEvidence) -> AggregationLedger | None:
            if not self.answer_config.aggregation_ledger_enabled:
                return None
            return build_aggregation_ledger(
                question, turn_map, rendered.turn_ids,
                limit=self.answer_config.aggregation_ledger_limit,
                execution_card=self.answer_config.aggregation_execution_card)
        ledger = make_ledger(evidence)
        focused_prompt = (
            self.answer_config.focused_prompt_scope == "all"
            or (ledger is None and not preference_synthesis))
        effective_question_date_mode = (
            self.answer_config.question_date_mode
            if focused_prompt else "always")
        include_question_date = (
            effective_question_date_mode == "always"
            or (effective_question_date_mode == "query_relative"
                and question_needs_global_date(question)))
        contextual_question_date = effective_question_date_mode != "always"
        question_recency_footer = (
            focused_prompt and self.answer_config.question_recency_footer)
        compact_topological_contract = (
            focused_prompt
            and self.answer_config.compact_topological_contract)
        aggregation_source_reserve: tuple[str, ...] = ()
        if (ledger is not None
                and not self.answer_config.aggregation_execution_card
                and self.answer_config.aggregation_source_reserve_enabled
                # Direct source statements improve operand closure for
                # enumeration, but add distractors to already-local temporal
                # and arithmetic comparisons.  Keep the reserve on the two
                # operations for which the full paired gate was positive.
                and ledger.operation in set(
                    self.answer_config.aggregation_source_reserve_operations)):
            aggregation_source_reserve = _aggregation_source_reserve_ids(
                turn_map, evidence.turn_ids)
        if evidence.mandatory_dropped:
            warnings.append("mandatory_turn_dropped_for_budget")
        if (self.answer_config.deterministic_bypass_enabled
                and typed_execution is not None
                and typed_execution.safe_to_bypass):
            prepared = PreparedAnswer(
                question_id=question_id, memory_id=result.memory_id,
                messages=(), evidence_turn_ids=evidence.turn_ids,
                dropped_turn_ids=evidence.dropped_turn_ids,
                evidence_tokens=evidence.tokens, packing_prompt_tokens=0,
                closed_form=True, draft_text=typed_execution.text,
                draft_certified=True, budget_relaxed=False,
                prompt_hash=PROMPT_HASH,
                prompt_payload_hash=hashlib.sha256(b"[]").hexdigest(),
                warnings=tuple(warnings),
                deterministic_prediction=typed_execution.text,
                preparation_latency_ms=(time.perf_counter() - started) * 1000,
                trace={"deterministic_bypass": True, "typed_execution": {
                    "kind": typed_execution.answer_kind,
                    "unit": typed_execution.unit,
                    "interval_uncertainty": typed_execution.interval_uncertainty,
                    "contradiction_status": typed_execution.contradiction_status,
                    "provenance_binding_ids": list(
                        typed_execution.provenance_binding_ids),
                    "reason_codes": list(typed_execution.reason_codes)}})
            return apply_readout_policy(
                prepared, self.counter, self.answer_config.readout_policy)

        if ledger is not None and ledger.result_certified:
            prompt_version, _prompt_text, prompt_hash = prompt_contract(
                self.answer_config.normalize_relative_time,
                self.answer_config.precision_grounding,
                evidence_order in {"topological", "topological_recency"}, True,
                False, self.answer_config.exact_grounding_footer,
                contextual_question_date,
                question_recency_footer, compact_topological_contract)
            payload_hash = hashlib.sha256(canonical_json({
                "operation": ledger.operation,
                "operands": list(ledger.deterministic_operands),
                "result": ledger.deterministic_result,
                "schema_version": ledger.schema_version,
            }).encode()).hexdigest()
            prepared = PreparedAnswer(
                question_id=question_id, memory_id=result.memory_id,
                messages=(), evidence_turn_ids=evidence.turn_ids,
                dropped_turn_ids=evidence.dropped_turn_ids,
                evidence_tokens=evidence.tokens, packing_prompt_tokens=0,
                closed_form=True, draft_text=ledger.deterministic_result,
                draft_certified=True, budget_relaxed=False,
                prompt_hash=prompt_hash, prompt_payload_hash=payload_hash,
                warnings=tuple(warnings),
                deterministic_prediction=ledger.deterministic_result,
                preparation_latency_ms=(time.perf_counter() - started) * 1000,
                trace={
                    "prompt_version": prompt_version,
                    "deterministic_bypass": True,
                    "deterministic_bypass_source": "aggregation_ledger",
                    "aggregation_ledger": {
                        "schema_version": ledger.schema_version,
                        "operation": ledger.operation,
                        "candidate_turn_ids": list(ledger.candidate_turn_ids),
                        "numeric_candidate_count": ledger.numeric_candidate_count,
                        "result_certified": True,
                        "deterministic_operands": list(
                            ledger.deterministic_operands),
                        "deterministic_result": ledger.deterministic_result,
                    },
                    "evidence_order": self.answer_config.evidence_order,
                    "resolved_evidence_order": evidence_order,
                    "packed_turns": len(evidence.turn_ids),
                })
            return apply_readout_policy(
                prepared, self.counter, self.answer_config.readout_policy)

        messages = build_answer_messages(
            question=question, question_date=question_date,
            evidence_text=evidence.text,
            candidate_answer=(
                draft.text if (draft is not None
                               and self.answer_config.candidate_answer_injection)
                else None),
            normalize_relative_time=self.answer_config.normalize_relative_time,
            precision_grounding=self.answer_config.precision_grounding,
            topological_layout=evidence_order in {
                "topological", "topological_recency"},
            aggregation_ledger=(ledger.text if ledger else None),
            aggregation_ledger_contract=bool(ledger),
            preference_synthesis=preference_synthesis,
            exact_grounding_footer=(
                self.answer_config.exact_grounding_footer),
            include_question_date=include_question_date,
            question_recency_footer=question_recency_footer,
            compact_topological_contract=compact_topological_contract)
        prompt_version, _prompt_text, prompt_hash = prompt_contract(
            self.answer_config.normalize_relative_time,
            self.answer_config.precision_grounding,
            evidence_order in {"topological", "topological_recency"},
            bool(ledger), preference_synthesis,
            self.answer_config.exact_grounding_footer,
            contextual_question_date,
            question_recency_footer, compact_topological_contract)
        prompt_tokens = self._prompt_tokens(messages)
        relaxed = False
        if prompt_tokens > budget.max_answer_tokens:
            overhead = prompt_tokens - evidence.tokens
            evidence = self.render(
                result, budget,
                max_tokens=max(1, budget.max_answer_tokens - overhead),
                question=question,
                reserved_turn_ids=tuple(dict.fromkeys((
                    *((ledger.candidate_turn_ids
                       if ledger is not None
                       and not self.answer_config.aggregation_execution_card
                       else ())),
                    *aggregation_source_reserve))))
            ledger = make_ledger(evidence)
            messages = build_answer_messages(
                question=question, question_date=question_date,
                evidence_text=evidence.text,
                candidate_answer=(
                    draft.text if (draft is not None
                                   and self.answer_config.candidate_answer_injection)
                    else None),
                normalize_relative_time=self.answer_config.normalize_relative_time,
                precision_grounding=self.answer_config.precision_grounding,
                topological_layout=evidence_order in {
                    "topological", "topological_recency"},
                aggregation_ledger=(ledger.text if ledger else None),
                aggregation_ledger_contract=bool(ledger),
                preference_synthesis=preference_synthesis,
                exact_grounding_footer=(
                    self.answer_config.exact_grounding_footer),
                include_question_date=include_question_date,
                question_recency_footer=question_recency_footer,
                compact_topological_contract=compact_topological_contract)
            prompt_tokens = self._prompt_tokens(messages)
            if prompt_tokens > budget.max_answer_tokens:
                relaxed = True
                warnings.append("answer_budget_relaxed_to_hard_ceiling")
        if prompt_tokens > budget.max_answer_tokens_hard:
            raise RuntimeError(
                f"answer prompt for {question_id} is {prompt_tokens} tokens, above the hard "
                f"ceiling {budget.max_answer_tokens_hard}")

        prompt_payload_hash = hashlib.sha256(
            canonical_json(messages).encode()).hexdigest()
        prepared = PreparedAnswer(
            question_id=question_id, memory_id=result.memory_id,
            messages=tuple(dict(row) for row in messages),
            evidence_turn_ids=evidence.turn_ids,
            dropped_turn_ids=evidence.dropped_turn_ids,
            evidence_tokens=evidence.tokens,
            packing_prompt_tokens=prompt_tokens,
            closed_form=bool(draft and draft.certified),
            draft_text=draft.text if draft else "",
            draft_certified=bool(draft and draft.certified),
            budget_relaxed=relaxed, prompt_hash=prompt_hash,
            prompt_payload_hash=prompt_payload_hash,
            warnings=tuple(warnings),
            preparation_latency_ms=(time.perf_counter() - started) * 1000,
            trace={
                "prompt_version": prompt_version,
                "focused_prompt_scope": self.answer_config.focused_prompt_scope,
                "focused_prompt_applied": focused_prompt,
                "question_date_mode": effective_question_date_mode,
                "question_date_included": include_question_date,
                "question_recency_footer": question_recency_footer,
                "compact_topological_contract": compact_topological_contract,
                "span_window": self.answer_config.span_window,
                "evidence_order": self.answer_config.evidence_order,
                "resolved_evidence_order": evidence_order,
                "packed_turns": len(evidence.turn_ids),
                "evidence_truncated": evidence.truncated,
                "evidence_layout": evidence.layout_mode,
                "evidence_chain_count": evidence.chain_count,
                "evidence_chain_turns": evidence.chain_turns,
                "evidence_graph_group_count": evidence.graph_group_count,
                "evidence_graph_turns": evidence.graph_turns,
                "evidence_auxiliary_turns": evidence.auxiliary_turns,
                "token_counter": self.counter.describe(),
                "draft_kind": draft.answer_kind if draft else None,
                "draft_degradations": list(draft.degradations) if draft else [],
                "deterministic_bypass": False,
                "typed_execution": ({
                    "kind": typed_execution.answer_kind,
                    "unit": typed_execution.unit,
                    "interval_uncertainty": typed_execution.interval_uncertainty,
                    "contradiction_status": typed_execution.contradiction_status,
                    "safe_to_bypass": typed_execution.safe_to_bypass,
                    "reason_codes": list(typed_execution.reason_codes),
                } if typed_execution is not None else None),
                "aggregation_ledger": ({
                    "schema_version": ledger.schema_version,
                    "operation": ledger.operation,
                    "execution_card": (
                        self.answer_config.aggregation_execution_card),
                    "candidate_turn_ids": list(ledger.candidate_turn_ids),
                    "numeric_candidate_count": ledger.numeric_candidate_count,
                    "result_certified": ledger.result_certified,
                    "deterministic_operands": list(
                        ledger.deterministic_operands),
                    "deterministic_result": ledger.deterministic_result,
                } if ledger is not None else None),
                "aggregation_source_reserve_turns": len(
                    aggregation_source_reserve),
                "preference_synthesis": preference_synthesis,
            })
        prepared = apply_readout_policy(
            prepared, self.counter, self.answer_config.readout_policy)
        return replace(
            prepared,
            preparation_latency_ms=(time.perf_counter() - started) * 1000,
        )

    def complete(self, prepared: PreparedAnswer) -> AnswerResult:
        """Complete a frozen request with the configured answer backbone."""

        started = time.perf_counter()
        warnings = list(prepared.warnings)
        if prepared.deterministic_prediction:
            return AnswerResult(
                question_id=prepared.question_id, memory_id=prepared.memory_id,
                prediction=prepared.deterministic_prediction,
                evidence_turn_ids=prepared.evidence_turn_ids,
                dropped_turn_ids=prepared.dropped_turn_ids,
                evidence_tokens=prepared.evidence_tokens, prompt_tokens=0,
                completion_tokens=0, closed_form=True,
                finish_reason="deterministic",
                draft_text=prepared.draft_text, draft_certified=True,
                budget_relaxed=prepared.budget_relaxed,
                prompt_hash=prepared.prompt_hash,
                prompt_payload_hash=prepared.prompt_payload_hash,
                answer_model=self.answer_model,
                latency_ms=prepared.preparation_latency_ms,
                warnings=tuple(warnings), trace=prepared.trace)

        text, api_prompt, completion, api_total, cached, finish_reason = self._call(
            prepared.question_id, prepared.memory_id,
            prepared.messages, prepared.prompt_hash)
        prediction = " ".join(text.split())
        if not prediction:
            warnings.append("empty_prediction")
        if finish_reason == "length":
            warnings.append("answer_output_truncated")
        return AnswerResult(
            question_id=prepared.question_id, memory_id=prepared.memory_id,
            prediction=prediction,
            evidence_turn_ids=prepared.evidence_turn_ids,
            dropped_turn_ids=prepared.dropped_turn_ids,
            evidence_tokens=prepared.evidence_tokens,
            prompt_tokens=prepared.packing_prompt_tokens,
            completion_tokens=completion, closed_form=prepared.closed_form,
            finish_reason=finish_reason, draft_text=prepared.draft_text,
            draft_certified=prepared.draft_certified, cached=cached,
            budget_relaxed=prepared.budget_relaxed,
            latency_ms=(prepared.preparation_latency_ms
                        + (time.perf_counter() - started) * 1000),
            api_prompt_tokens=api_prompt,
            api_total_tokens=api_total,
            answer_model=self.answer_model,
            prompt_payload_hash=prepared.prompt_payload_hash,
            warnings=tuple(warnings), prompt_hash=prepared.prompt_hash,
            trace={**dict(prepared.trace), "finish_reason": finish_reason})

    def _prompt_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(self.counter.count_many([str(row["content"]) for row in messages]))

    def _call(self, question_id: str, memory_id: str,
              messages: Sequence[Mapping[str, str]], prompt_hash: str,
              ) -> tuple[str, int, int, int, bool, str]:
        request = {
            "model": self.answer_model, "messages": list(messages),
            "temperature": 0, "seed": self.answer_config.sampling_seed,
        }
        if self.answer_request_profile == "qwen":
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}}
        elif self.answer_request_profile == "openai":
            request["reasoning_effort"] = "none"
        if self.answer_config.max_output_tokens is not None:
            output_key = ("max_completion_tokens"
                          if (self.answer_request_profile == "openai"
                              and self.answer_model.casefold().startswith("gpt-5"))
                          else "max_tokens")
            request[output_key] = self.answer_config.max_output_tokens
        identity = CacheIdentity(
            self.dataset_hash, self.answer_model, prompt_hash,
            self.config.schema_version,
            hashlib.sha256(canonical_json({
                "span_window": self.answer_config.span_window,
                "include_dates": self.answer_config.include_dates,
                "include_speaker": self.answer_config.include_speaker,
                "evidence_order": self.answer_config.evidence_order,
                "normalize_relative_time": self.answer_config.normalize_relative_time,
                "precision_grounding": self.answer_config.precision_grounding,
                "candidate_answer_injection": (
                    self.answer_config.candidate_answer_injection),
                "max_output_tokens": self.answer_config.max_output_tokens,
                "sampling_seed": self.answer_config.sampling_seed,
                "request_profile": self.answer_request_profile,
            }).encode()).hexdigest(),
            "answer:" + hashlib.sha256(canonical_json(request["messages"]).encode()).hexdigest(),
        )
        key = identity.key()
        started = time.perf_counter()
        cached = self.cache_store.cache_get(key)
        if cached:
            response, usage, is_cached = cached["response"], dict(cached["usage"]), True
            prompt = int(usage.get("uncached_input_tokens", 0))
            completion = int(usage.get("output_tokens", 0))
            api_total = int(usage.get("total_tokens", prompt + completion))
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
            api_total = int(getattr(raw, "total_tokens", 0)
                            or (prompt + completion))
            usage = {"cached_input_tokens": 0, "uncached_input_tokens": prompt,
                     "output_tokens": completion, "reasoning_tokens": 0,
                     "total_tokens": prompt + completion}
            self.cache_store.cache_put(key, "answer", request, response, usage, prompt_hash)
            is_cached = False
        occurrence = self.cache_store._read_one(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?",
            (memory_id, key))[0]
        self.cache_store.log_llm_call(
            call_id=stable_id("llm-call", memory_id, key, is_cached, occurrence),
            memory_id=memory_id, stage="answer", cache_key=key, cached=is_cached,
            request=request, response=response, usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000, retry_count=0, batch_size=1,
            prompt_hash=prompt_hash)
        return (str(response.get("content", "")), prompt, completion, api_total,
                is_cached, str(response.get("finish_reason") or ""))
