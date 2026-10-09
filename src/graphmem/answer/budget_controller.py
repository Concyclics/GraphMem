"""Observable two-stage budgets for retrieval and answer generation.

The controller is intentionally label-free: it consumes only the first-pass
retrieval trace and answers emitted for that request.  It never receives a
gold answer or judge verdict.  A caller can therefore audit every escalation
and reproduce the same decision online and in benchmark replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


BUDGET_CONTROLLER_SCHEMA_VERSION = "graphmem-v5.76-budget-controller-v1"
BUDGET_POLICIES = ("balanced", "pareto", "accuracy")

_PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def answer_signature(text: str) -> str:
    """Return a conservative surface-equivalence signature.

    This only removes case, Unicode-width, punctuation and whitespace
    differences.  It does *not* collapse distinct entities, numbers, dates or
    polarity, because those differences are precisely the uncertainty signal
    that should trigger a second pass.
    """

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def candidate_signatures(
    candidates: Sequence[Mapping[str, Any]], *, prefix: int = 2,
) -> tuple[str, ...]:
    if prefix <= 0:
        raise ValueError("candidate disagreement prefix must be positive")
    signatures = tuple(
        answer_signature(str(row.get("prediction") or ""))
        for row in candidates[:prefix])
    return tuple(value for value in signatures if value)


def reliable_candidate_choice(
    candidates: Sequence[Mapping[str, Any]],
    *,
    single_stage_consensus_ratio: float = 0.75,
) -> int | None:
    """Return a zero-based choice only for independently repeated support.

    Agreement across the 32-turn and expanded evidence stages takes priority.
    Within a single stage, a strict 3/4-style consensus is accepted.  All
    weaker or tied cases remain unresolved for evidence-aware verification.
    """

    if not 0.5 < single_stage_consensus_ratio <= 1.0:
        raise ValueError("single-stage consensus ratio must be in (0.5, 1]")
    groups: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        signature = answer_signature(str(candidate.get("prediction") or ""))
        if not signature:
            continue
        group = groups.setdefault(signature, {
            "first": index, "count": 0, "stages": set(), "families": set()})
        group["count"] += 1
        group["stages"].add(str(candidate.get("budget_stage") or "unknown"))
        group["families"].add(str(candidate.get("family") or "unknown"))
    if not groups:
        return None
    cross_stage = [group for group in groups.values()
                   if len(group["stages"]) >= 2]
    if cross_stage:
        winner = max(cross_stage, key=lambda group: (
            len(group["stages"]), len(group["families"]),
            group["count"], -group["first"]))
        return int(winner["first"])
    required = max(2, math.ceil(
        single_stage_consensus_ratio * max(1, len(candidates))))
    eligible = [group for group in groups.values()
                if group["count"] >= required]
    if not eligible:
        return None
    winner = max(eligible, key=lambda group: (
        group["count"], len(group["families"]), -group["first"]))
    return int(winner["first"])


def modal_candidate_choice(
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    """Select the most frequent normalized answer, ties by source order."""

    signatures = [answer_signature(str(row.get("prediction") or ""))
                  for row in candidates]
    if not signatures or not any(signatures):
        raise ValueError("at least one non-empty candidate is required")
    counts = Counter(value for value in signatures if value)
    winner = max(counts, key=lambda value: (
        counts[value], -signatures.index(value)))
    return signatures.index(winner)


def stage_normalized_candidate_choice(
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    """Vote within each budget stage before combining unequal sample counts."""

    signatures = [answer_signature(str(row.get("prediction") or ""))
                  for row in candidates]
    if not signatures or not any(signatures):
        raise ValueError("at least one non-empty candidate is required")
    stage_totals: Counter[str] = Counter()
    stage_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for signature, candidate in zip(signatures, candidates):
        if not signature:
            continue
        stage = str(candidate.get("budget_stage") or "unknown")
        stage_totals[stage] += 1
        stage_votes[stage][signature] += 1
    scores: Counter[str] = Counter()
    stage_support: Counter[str] = Counter()
    for stage, votes in stage_votes.items():
        for signature, count in votes.items():
            scores[signature] += count / stage_totals[stage]
            stage_support[signature] += 1
    winner = max(scores, key=lambda value: (
        scores[value], stage_support[value],
        sum(votes[value] for votes in stage_votes.values()),
        -signatures.index(value)))
    return signatures.index(winner)


@dataclass(frozen=True, slots=True)
class AnswerBudgetDecision:
    """Second-pass decision derived entirely from observable runtime state."""

    policy: str
    expand: bool
    target_turns: int
    target_tokens: int
    target_candidates: int
    retain_first_pass: bool
    reasons: tuple[str, ...] = ()
    retrieval_severity: int = 0
    candidate_unique: int = 0


def decide_answer_budget(
    retrieval_trace: Mapping[str, Any],
    first_pass_candidates: Sequence[Mapping[str, Any]],
    *,
    policy: str = "balanced",
    disagreement_prefix: int = 2,
    base_turns: int = 32,
    medium_turns: int = 64,
    maximum_turns: int = 80,
    base_tokens: int = 2200,
    medium_tokens: int = 3400,
    maximum_tokens: int = 4500,
    base_candidates: int = 2,
    expanded_candidates: int = 8,
) -> AnswerBudgetDecision:
    """Choose whether to run an additive second answer/retrieval pass.

    ``balanced`` expands on an existing QueryIR/closure escalation signal or
    candidate disagreement. ``pareto`` uses retrieval-only signals: the normal
    closure trigger, severity-2 scalar lookup gaps, or QueryIR soft fallback.
    ``accuracy`` treats any severity-2 closure deficit or candidate
    disagreement as sufficient risk. In every mode the first-pass answers
    remain in the final candidate pool; expansion never overwrites a valid
    short-context answer.
    """

    if policy not in BUDGET_POLICIES:
        raise ValueError(f"unsupported budget policy: {policy}")
    if not (0 < base_turns <= medium_turns <= maximum_turns):
        raise ValueError("turn tiers must satisfy 0 < base <= medium <= max")
    if not (0 < base_tokens <= medium_tokens <= maximum_tokens):
        raise ValueError("token tiers must satisfy 0 < base <= medium <= max")
    if not (0 < base_candidates <= expanded_candidates):
        raise ValueError(
            "candidate tiers must satisfy 0 < base <= expanded")

    severity = int(retrieval_trace.get("adaptive_recall_severity", 0) or 0)
    retrieval_triggered = bool(
        retrieval_trace.get("adaptive_recall_triggered", False))
    queryir_uncertain = bool(
        retrieval_trace.get("query_ir_soft_fallback", False))
    token_cap = bool(retrieval_trace.get("pack_token_cap_reached", False))
    signatures = candidate_signatures(
        first_pass_candidates, prefix=disagreement_prefix)
    unique = len(set(signatures))
    disagreement = unique > 1

    reasons: list[str] = []
    if retrieval_triggered:
        reasons.append("queryir_or_closure_escalation")
    if disagreement and policy in {"balanced", "accuracy"}:
        reasons.append("answer_disagreement")
    if policy == "accuracy" and severity >= 2:
        reasons.append("closure_severity_ge_2")
    route = str(retrieval_trace.get("adaptive_recall_route") or "")
    pareto_lookup = bool(
        policy == "pareto" and route == "lookup" and severity >= 2)
    if pareto_lookup:
        reasons.append("lookup_witness_severity_ge_2")
    pareto_queryir = bool(policy == "pareto" and queryir_uncertain)
    if pareto_queryir:
        reasons.append("queryir_soft_fallback")
    if token_cap:
        reasons.append("evidence_token_cap_reached")

    expand = bool(
        retrieval_triggered
        or (disagreement and policy in {"balanced", "accuracy"})
        or pareto_lookup
        or pareto_queryir
        or (policy == "accuracy" and severity >= 2))
    severe = severity >= 6
    target_turns = (maximum_turns if severe else medium_turns) if expand else base_turns
    # Token growth remains independent of logical/answer uncertainty.  A
    # larger evidence allowance is granted only after the base pack hit its
    # measured cap; otherwise expansion searches farther under the same Token
    # ceiling.
    target_tokens = (
        (maximum_tokens if severe else medium_tokens)
        if expand and token_cap else base_tokens)
    return AnswerBudgetDecision(
        policy=policy,
        expand=expand,
        target_turns=target_turns,
        target_tokens=target_tokens,
        target_candidates=(expanded_candidates if expand else base_candidates),
        retain_first_pass=True,
        reasons=tuple(dict.fromkeys(reasons)),
        retrieval_severity=severity,
        candidate_unique=unique,
    )
