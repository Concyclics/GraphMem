"""Incremental answer prompts for an adaptive evidence budget.

The second answer pass must not resend a complete expanded evidence pack.  It
receives only evidence that was absent from the first pass, a small set of
source-backed anchors from that pass, and the first-pass answers as explicitly
fallible proposals.  This keeps the extra request proportional to the new
information discovered by retrieval rather than to the whole memory window.

This module is deliberately evaluator-independent: neither the planner nor the
prompt builder accepts gold turns, reference answers, or judge verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..domain import SourceTurn
from .budget_controller import answer_signature
from .rendering import AnswerConfig, render_turn


INCREMENTAL_PROMPT_VERSION = "graphmem-v5.76-incremental-evidence-v2"
MERGED_EXPANSION_PROMPT_VERSION = "graphmem-v5.76-merged-expansion-v1"


@dataclass(frozen=True, slots=True)
class IncrementalEvidencePlan:
    """A deterministic delta between the first and expanded evidence packs."""

    added_turn_ids: tuple[str, ...]
    anchor_turn_ids: tuple[str, ...]
    removed_turn_ids: tuple[str, ...]
    retained_turn_ids: tuple[str, ...]

    @property
    def transmitted_turn_ids(self) -> tuple[str, ...]:
        return (*self.anchor_turn_ids, *self.added_turn_ids)


def plan_incremental_evidence(
    base_turn_ids: Sequence[str],
    expanded_turn_ids: Sequence[str],
    *,
    preferred_anchor_ids: Sequence[str] = (),
    max_anchors: int = 4,
) -> IncrementalEvidencePlan:
    """Return a stable evidence delta without duplicating source turns.

    ``expanded_turn_ids`` order is retained because it already carries the
    navigator's graph-layout decision.  Preferred anchors normally come from
    the typed evidence card.  When none survive in the base pack, the last
    base turns are used because the validated readout layout places its
    strongest evidence nearest the generation boundary.
    """

    if max_anchors < 0:
        raise ValueError("max_anchors must be non-negative")
    base = tuple(dict.fromkeys(str(item) for item in base_turn_ids if item))
    expanded = tuple(dict.fromkeys(
        str(item) for item in expanded_turn_ids if item))
    base_set = set(base)
    expanded_set = set(expanded)
    added = tuple(item for item in expanded if item not in base_set)
    retained = tuple(item for item in expanded if item in base_set)
    removed = tuple(item for item in base if item not in expanded_set)

    anchors: list[str] = []
    for item in preferred_anchor_ids:
        turn_id = str(item)
        if (turn_id in base_set and turn_id not in added
                and turn_id not in anchors):
            anchors.append(turn_id)
            if len(anchors) == max_anchors:
                break
    if len(anchors) < max_anchors:
        # Walk backwards so the nearest-to-generation evidence is admitted
        # first, then restore its original presentation order.
        fallback = []
        for turn_id in reversed(base):
            if turn_id not in anchors and turn_id not in added:
                fallback.append(turn_id)
                if len(anchors) + len(fallback) == max_anchors:
                    break
        anchors.extend(reversed(fallback))
    return IncrementalEvidencePlan(
        added_turn_ids=added,
        anchor_turn_ids=tuple(anchors[:max_anchors]),
        removed_turn_ids=removed,
        retained_turn_ids=retained,
    )


def _unique_proposals(candidates: Sequence[Mapping[str, Any] | str],
                      limit: int) -> tuple[str, ...]:
    if limit <= 0:
        raise ValueError("proposal limit must be positive")
    rows: list[str] = []
    signatures: set[str] = set()
    for candidate in candidates:
        text = (str(candidate.get("prediction") or "")
                if isinstance(candidate, Mapping) else str(candidate))
        text = " ".join(text.split())
        signature = answer_signature(text)
        if not text or not signature or signature in signatures:
            continue
        signatures.add(signature)
        rows.append(text)
        if len(rows) == limit:
            break
    return tuple(rows)


def build_incremental_answer_messages(
    *,
    question: str,
    first_pass_candidates: Sequence[Mapping[str, Any] | str],
    added_turns: Sequence[SourceTurn],
    anchor_turns: Sequence[SourceTurn] = (),
    route: str = "lookup",
    escalation_reasons: Sequence[str] = (),
    missing_obligation_count: int = 0,
    answer_config: AnswerConfig | None = None,
    proposal_limit: int = 4,
) -> tuple[dict[str, str], ...]:
    """Build a compact, source-grounded continuation request.

    Candidate answers summarize an earlier source-grounded pass but are never
    represented as source turns.  Crucially, the omitted base context is not
    treated as negative evidence: a continuation preserves a sound proposal
    unless the delta contradicts it or supplies a more exact/complete answer.
    No hidden chain of thought is requested or exposed.
    """

    clean_question = " ".join(str(question).split())
    if not clean_question:
        raise ValueError("question must be non-empty")
    config = answer_config or AnswerConfig.v5_63()
    proposals = _unique_proposals(first_pass_candidates, proposal_limit)
    anchor_rows = tuple(render_turn(turn, config) for turn in anchor_turns)
    added_rows = tuple(render_turn(turn, config) for turn in added_turns)
    if not added_rows:
        raise ValueError("incremental pass requires at least one new source turn")

    diagnostics = [f"route={route or 'lookup'}"]
    if missing_obligation_count:
        diagnostics.append(
            f"unresolved_bindings={int(missing_obligation_count)}")
    safe_reasons = [
        " ".join(str(reason).replace("_", " ").split())
        for reason in escalation_reasons if str(reason).strip()
    ]
    if safe_reasons:
        diagnostics.append("trigger=" + ", ".join(safe_reasons[:4]))

    system = (
        "You are the incremental readout stage of a conversation-memory "
        "system. Answer the exact question from source evidence. Earlier "
        "answers below are provisional results produced from an earlier base "
        "evidence pass; they are not new source evidence. The full base "
        "context is intentionally not repeated, so its absence here is not a "
        "contradiction or a reason to abstain. BASE ANCHOR and NEW EVIDENCE "
        "rows are primary source turns. Use NEW EVIDENCE to validate, correct, "
        "or complete the proposals. Preserve the best earlier proposal unless "
        "a source turn explicitly conflicts with it, makes it more precise, "
        "or supplies a missing operand needed by the exact question. Keep "
        "the exact subject, relation, polarity, completion status, quantity, "
        "unit, and time scope. Resolve relative dates from each source turn's "
        "own timestamp. For multi-hop, comparison, count, or temporal "
        "questions, bind every required operand before answering and do not "
        "treat an unobserved operand as zero. Return one concise final answer "
        "only; do not output analysis or candidate numbers."
    )
    sections = [
        f"Question: {clean_question}",
        "\nFirst-pass proposals (fallible; not source evidence):",
    ]
    sections.extend(
        f"[PROPOSAL {index}] {text}"
        for index, text in enumerate(proposals, start=1))
    if not proposals:
        sections.append("[PROPOSAL] No usable first-pass answer.")
    sections.append("\nBudget-controller diagnostics: " + "; ".join(diagnostics))
    if anchor_rows:
        sections.append("\nBase evidence anchors:")
        sections.extend(
            f"[BASE ANCHOR {index}] {text}"
            for index, text in enumerate(anchor_rows, start=1))
    sections.append("\nNew evidence discovered by expanded retrieval:")
    sections.extend(
        f"[NEW EVIDENCE {index}] {text}"
        for index, text in enumerate(added_rows, start=1))
    sections.extend((
        "\nResolution rule: prefer exact source statements over proposals; "
        "combine anchors and new evidence only when the question requires it. "
        "Answer 'insufficient information' only when the earlier proposals "
        "were already insufficient or mutually incompatible and the added "
        "source evidence still cannot resolve them; do not abstain merely "
        "because the unrepeated base context is omitted.",
        f"Question (answer this exact relation): {clean_question}",
        "Final answer:",
    ))
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(sections)},
    )


def build_merged_expansion_messages(
    *,
    base_messages: Sequence[Mapping[str, str]],
    question: str,
    added_turns: Sequence[SourceTurn],
    route: str = "lookup",
    escalation_reasons: Sequence[str] = (),
    missing_obligation_count: int = 0,
    answer_config: AnswerConfig | None = None,
) -> tuple[dict[str, str], ...]:
    """Append deduplicated expansion evidence to a first-pass *prompt*.

    Unlike :func:`build_incremental_answer_messages`, this is evaluated before
    any answer generation.  The model is called exactly once over the 32-turn
    precision core plus the source turns newly discovered by the expanded
    retrieval.  It is the preferred production path whenever the budget gate
    depends solely on QueryIR and evidence-closure telemetry.
    """

    clean_question = " ".join(str(question).split())
    if not clean_question:
        raise ValueError("question must be non-empty")
    if not added_turns:
        raise ValueError("merged expansion requires at least one new source turn")
    messages = [dict(message) for message in base_messages]
    if not messages:
        raise ValueError("base messages must be non-empty")
    system_index = next((index for index, message in enumerate(messages)
                         if message.get("role") == "system"), None)
    user_index = next((index for index in range(len(messages) - 1, -1, -1)
                       if messages[index].get("role") == "user"), None)
    if system_index is None or user_index is None:
        raise ValueError("base messages require system and user roles")
    config = answer_config or AnswerConfig.v5_63()
    evidence = tuple(render_turn(turn, config) for turn in added_turns)
    safe_reasons = [
        " ".join(str(reason).replace("_", " ").split())
        for reason in escalation_reasons if str(reason).strip()
    ]
    diagnostics = [f"route={route or 'lookup'}"]
    if missing_obligation_count:
        diagnostics.append(
            f"unresolved_bindings={int(missing_obligation_count)}")
    if safe_reasons:
        diagnostics.append("trigger=" + ", ".join(safe_reasons[:4]))

    messages[system_index]["content"] = (
        str(messages[system_index].get("content") or "")
        + "\n\nThe ADAPTIVE EXPANSION block contains additional primary "
          "source turns selected after a QueryIR/evidence-closure check. It "
          "supplements the precision-core memories and does not replace them. "
          "Use both blocks, reject nearby-topic noise, and change a direct "
          "precision-core fact only when an exact added source turn requires it."
    )
    expansion = [
        "\n\nAdaptive expansion diagnostics: " + "; ".join(diagnostics),
        "ADAPTIVE EXPANSION (new primary source turns; no duplicates from "
        "the precision core):",
    ]
    expansion.extend(
        f"[EXPANDED SOURCE {index}] {text}"
        for index, text in enumerate(evidence, start=1))
    expansion.extend((
        "Use the precision core and expansion evidence as one deduplicated "
        "evidence set. Silently verify every required entity, event, operand, "
        "state, or temporal endpoint before answering.",
        f"Question (answer this exact relation): {clean_question}",
        "Return one concise final answer only.",
    ))
    messages[user_index]["content"] = (
        str(messages[user_index].get("content") or "")
        + "\n".join(expansion))
    return tuple(messages)
