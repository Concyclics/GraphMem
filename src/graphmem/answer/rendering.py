"""Turn packed evidence into prompt text under an exact token budget.

Two things here are load-bearing:

* **Order is content.**  The rendered block is part of a cached prompt, so the
  same navigation must always produce the same bytes.  Turns are ordered by
  ``(session_order, turn_index, turn_id)`` with an explicit id tiebreak, never
  by set iteration.
* **The budget is enforced with the backbone's own vocabulary.**  The V5.4
  word-count heuristic runs ~25% high, so a budget enforced with it is neither
  tight nor safe.  ``QueryBudget.max_answer_tokens`` has had no consumer at all
  until now; this module is it.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from ..build.temporal import extract_time_expressions, normalize_time
from ..domain import EvidenceMember, SourceTurn
from ..tokenization import TokenCounter

#: ``span_window=None`` renders the whole turn.  0 renders only the cited span.
FULL_TURN: None = None

_TEMPORAL_OPERATORS = frozenset({
    "argmin_time", "argmax_time", "date_difference", "latest_state", "ordinal",
})
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(?:when|date|time|days?|weeks?|months?|years?|before|after|since|until|"
    r"during|first|last|latest|earliest|recent(?:ly)?|ago|long|older|old|age|"
    r"then|next|previous|currently|now|past|weekend|summer|spring|winter|fall|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|20[12]\d)\b",
    re.I,
)


def resolve_evidence_order(configured: str, question: str = "",
                           query_operator: str = "") -> str:
    """Choose presentation order without benchmark labels or gold evidence.

    Temporal calculations depend on source order, while ordinary multi-hop
    questions benefit from putting the highest-ranked proof first.  QueryIR's
    operator is authoritative when it is typed; lexical time cues cover lookup
    questions whose requested value happens to be a date or duration.
    """
    if configured != "adaptive":
        return configured
    if query_operator.casefold() in _TEMPORAL_OPERATORS:
        return "chronological"
    return "chronological" if _TEMPORAL_QUERY_RE.search(question) else "relevance"


@dataclass(frozen=True, slots=True)
class AnswerConfig:
    """Answer-stage knobs.

    Deliberately *not* a member of ``GraphMemV5Config``: adding a field there
    changes ``config_hash``, which invalidates the refine cache and every frozen
    graph comparison.  The answer stage runs after retrieval and cannot affect a
    graph, so it carries its own config.
    """

    span_window: int | None = FULL_TURN
    include_dates: bool = True
    include_speaker: bool = True
    # ``chronological`` preserves the original answer contract.  ``relevance``
    # keeps the navigator/packer rank so the strongest evidence appears first
    # and the weakest tail is the first to be dropped under a token budget.
    # ``adaptive`` is resolved before rendering: temporal queries use the
    # former and other queries use the latter. ``topological_plain`` reorders
    # the frozen evidence set by graph topology without exposing graph labels
    # to the answer model; ``topological`` additionally renders those labels
    # and the matching prompt contract. ``topological_recency`` preserves each
    # graph block internally but places the strongest block last so the answer
    # model sees the best evidence nearest its generation boundary.
    evidence_order: str = "chronological"
    # Materialize relative phrases against the source turn timestamp in the
    # rendered evidence.  This prevents the answer model from re-anchoring
    # ``last week/month/year`` against the later question date.
    normalize_relative_time: bool = False
    # Opt-in V5.20 answer contract for noisy graph reservoirs.  Kept separate
    # from source-time normalization so prompt-only ablations remain possible.
    precision_grounding: bool = False
    # Add a query-ranked operand ledger after the evidence for explicit count,
    # sum, difference, min/max and mean questions.  The ledger never invents a
    # result when relational operand closure is uncertified.
    aggregation_ledger_enabled: bool = False
    aggregation_ledger_limit: int = 24
    # Render only an operation-specific execution card instead of duplicating
    # snippets from every ledger candidate. A bounded worksheet over the best
    # already-packed direct turns is added by the readout policy.
    aggregation_execution_card: bool = False
    # Add the V5.60 bounded source-backed operand worksheet to compact
    # aggregation cards.  The full-set prototype was not a no-regression
    # winner, so this remains an explicit experiment instead of silently
    # changing the validated V5.54 policy.
    aggregation_operand_worksheet_enabled: bool = False
    # V5.63 exposes the worksheet only when its operands pass a deterministic
    # completeness gate.  This avoids the V5.60 regression where a partial
    # worksheet looked authoritative and displaced the correct full-context
    # answer.  ``False`` preserves the original all-or-nothing experiment.
    aggregation_operand_worksheet_selective: bool = False
    # When a generic user/assistant transcript is re-packed to make room for
    # the aggregation ledger, keep the direct user statements before optional
    # assistant prose.  This is deliberately disabled for named multi-party
    # memories (for example LoCoMo), where both transport roles are equally
    # authoritative speakers.
    aggregation_source_reserve_enabled: bool = False
    aggregation_source_reserve_operations: tuple[str, ...] = (
        "sum", "count_distinct", "unit_rate")
    # Route recommendation/advice questions to a contract that treats stored
    # preferences as constraints for synthesis rather than requiring the final
    # recommendation itself to appear verbatim in memory.
    preference_synthesis_enabled: bool = False
    # ``domain_idf`` replaces a dense-dominated preference anchor with a
    # domain-scoped IDF reading index over direct packed user turns.
    preference_focus_strategy: str = "legacy"
    # Repeat the exact entity/relation and missing-fact guard after the evidence
    # block.  Long contexts can dilute the equivalent system instruction.
    exact_grounding_footer: bool = False
    # ``query_relative`` exposes the global question timestamp only when the
    # query itself contains a deictic time phrase.  Source-memory relative
    # expressions remain anchored by source-time normalization.
    question_date_mode: str = "always"
    # Repeat the question and source-time rule after long evidence so the model
    # does not answer a nearby but different relation from an earlier block.
    question_recency_footer: bool = False
    # Preserve the graph-layout semantics while reclaiming prompt tokens from
    # the verbose label glossary.  The saved budget pays for a post-evidence
    # query reminder without increasing any answer request.
    compact_topological_contract: bool = False
    # Repeat bounded, query-centered excerpts from the *full text* of turns
    # that are already present in the evidence pack.  Relation spans can point
    # at the beginning of a long list while the requested item is near its
    # tail; this reading index repairs that presentation loss without adding a
    # turn or changing retrieval coverage.
    query_focus_index_enabled: bool = False
    query_focus_index_limit: int = 4
    query_focus_excerpt_chars: int = 360
    # Allow Query Focus for explicit date-difference surfaces (ago/before/
    # between/passed/take), while retaining the temporal exclusion elsewhere.
    temporal_query_focus_enabled: bool = False
    # ``default`` applies the contextual-date/footer/compact-layout rewrite
    # only when no specialized aggregation or preference contract is active.
    # Those contracts already own the post-evidence readout semantics.
    focused_prompt_scope: str = "all"
    # Apply a validated post-packing readout policy in the core answer path.
    # ``legacy`` preserves every frozen artifact. ``v5_54`` composes the
    # label-free V5.43--V5.54 winner routes without depending on offline prompt
    # materializers.
    readout_policy: str = "legacy"
    # Compile a bounded, deterministic binding index after the validated
    # readout policy.  It quotes only already-packed evidence and never changes
    # the evidence set/order.  Kept opt-in until paired full-set validation.
    answer_plan_enabled: bool = False
    answer_plan_max_candidates: int = 5
    answer_plan_excerpt_chars: int = 440
    answer_plan_kinds: tuple[str, ...] = (
        "date_difference", "temporal_order")
    closed_form_enabled: bool = True
    # Keep the algebraic draft available for auditing, but do not place it in
    # the answer prompt unless an experiment explicitly opts in.  A noisy
    # mechanical proposal can anchor the backbone even when the retrieved
    # evidence contains the correct answer.
    candidate_answer_injection: bool = False
    # Remains false until an untouched holdout demonstrates >=99.5% precision
    # and <=0.5% false-complete for each whitelisted operator.
    deterministic_bypass_enabled: bool = False
    # ``None`` deliberately omits ``max_tokens`` from the API request.  The
    # model then ends on its stop condition or context limit; actual completion
    # usage and finish_reason are still recorded for every call.
    max_output_tokens: int | None = None
    sampling_seed: int = 0
    #: Rendered evidence is ``[session date] speaker: text``; this bounds the
    #: per-turn header so a pathological speaker label cannot eat the budget.
    max_speaker_chars: int = 48

    @classmethod
    def v5_54(cls, **overrides) -> "AnswerConfig":
        """Return the measured V5.54 answer contract.

        Retrieval budgets stay outside this answer-only configuration.  The
        matching query-plane values live in
        ``configs/v5/runtime_v5_54_accuracy64.json``.
        """

        values = {
            "span_window": 96,
            "evidence_order": "topological",
            "normalize_relative_time": True,
            "precision_grounding": False,
            "aggregation_ledger_enabled": True,
            "aggregation_ledger_limit": 32,
            "aggregation_execution_card": False,
            "aggregation_operand_worksheet_enabled": False,
            "aggregation_source_reserve_enabled": True,
            "aggregation_source_reserve_operations": (
                "sum", "count_distinct", "unit_rate"),
            "preference_synthesis_enabled": True,
            "exact_grounding_footer": False,
            "question_date_mode": "query_relative",
            "question_recency_footer": True,
            "compact_topological_contract": True,
            "focused_prompt_scope": "default",
            "closed_form_enabled": True,
            "candidate_answer_injection": False,
            "deterministic_bypass_enabled": False,
            "max_output_tokens": 2000,
            "sampling_seed": 0,
            "readout_policy": "v5_54",
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def v5_63(cls, **overrides) -> "AnswerConfig":
        """Return the validated selective-accuracy extension of V5.54.

        The extension keeps the same 64-turn retrieval budget and permits at
        most the readout policy's existing +500-token allowance per request.
        """

        values = {
            # Only the paired-positive temporal surface is enabled.  Broad
            # lookup Query Focus had small regressions and remains off.
            "query_focus_index_enabled": False,
            "temporal_query_focus_enabled": True,
            # The paired-positive temporal arm used 480-character excerpts.
            # Shortening them to 360 changed the 64-turn re-pack boundary and
            # lost endpoint context even though the same focus turns were
            # selected.
            "query_focus_excerpt_chars": 480,
            "preference_focus_strategy": "domain_idf",
            "aggregation_operand_worksheet_enabled": True,
            "aggregation_operand_worksheet_selective": True,
        }
        values.update(overrides)
        return cls.v5_54(**values)

    def __post_init__(self) -> None:
        if self.span_window is not None and self.span_window < 0:
            raise ValueError("span_window must be None or non-negative")
        if self.evidence_order not in {
                "chronological", "relevance", "adaptive",
                "topological_plain", "topological", "topological_recency"}:
            raise ValueError(
                "evidence_order must be chronological, relevance, adaptive, "
                "topological_plain, topological, or topological_recency")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be None or positive")
        if self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")
        if self.aggregation_ledger_limit <= 0:
            raise ValueError("aggregation_ledger_limit must be positive")
        if self.answer_plan_max_candidates <= 0:
            raise ValueError("answer_plan_max_candidates must be positive")
        if self.answer_plan_excerpt_chars < 80:
            raise ValueError("answer_plan_excerpt_chars must be at least 80")
        if self.query_focus_index_limit <= 0:
            raise ValueError("query_focus_index_limit must be positive")
        if self.query_focus_excerpt_chars < 120:
            raise ValueError("query_focus_excerpt_chars must be at least 120")
        if self.preference_focus_strategy not in {"legacy", "domain_idf"}:
            raise ValueError(
                "preference_focus_strategy must be legacy or domain_idf")
        if (self.aggregation_operand_worksheet_selective
                and not self.aggregation_operand_worksheet_enabled):
            raise ValueError(
                "selective aggregation worksheet requires worksheet enabled")
        allowed_plan_kinds = {
            "date_difference", "relative_time", "age_projection",
            "latest_state", "temporal_lookup", "temporal_order"}
        unknown_plan_kinds = set(self.answer_plan_kinds) - allowed_plan_kinds
        if unknown_plan_kinds:
            raise ValueError(
                f"unsupported answer_plan_kinds: {sorted(unknown_plan_kinds)}")
        if self.question_date_mode not in {"always", "query_relative", "never"}:
            raise ValueError(
                "question_date_mode must be always, query_relative, or never")
        if self.focused_prompt_scope not in {"all", "default"}:
            raise ValueError("focused_prompt_scope must be all or default")
        if self.readout_policy not in {"legacy", "v5_54"}:
            raise ValueError("readout_policy must be legacy or v5_54")
        if self.readout_policy == "v5_54":
            required = {
                "evidence_order": self.evidence_order == "topological",
                "source_time": self.normalize_relative_time,
                "aggregation_ledger": self.aggregation_ledger_enabled,
                "preference_synthesis": self.preference_synthesis_enabled,
                "query_relative_date": self.question_date_mode == "query_relative",
                "question_footer": self.question_recency_footer,
                "compact_topology": self.compact_topological_contract,
                "default_scope": self.focused_prompt_scope == "default",
                "no_candidate_injection": not self.candidate_answer_injection,
            }
            missing = [name for name, enabled in required.items() if not enabled]
            if missing:
                raise ValueError(
                    "v5_54 readout policy requires its measured base contract: "
                    + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class RenderedEvidence:
    text: str
    turn_ids: tuple[str, ...]
    dropped_turn_ids: tuple[str, ...]
    tokens: int
    truncated: bool
    #: True when the budget could only be met by dropping a mandatory turn.
    mandatory_dropped: bool = False
    layout_mode: str = ""
    chain_count: int = 0
    chain_turns: int = 0
    graph_group_count: int = 0
    graph_turns: int = 0
    auxiliary_turns: int = 0


def _clip_spans(spans: Sequence[EvidenceMember], length: int,
                window: int) -> tuple[tuple[int, int], ...]:
    """Merge cited spans widened by ``window`` characters, clipped to the turn."""
    widened = sorted(
        (max(0, span.span_start - window), min(length, span.span_end + window))
        for span in spans if span.span_start < length
    )
    merged: list[tuple[int, int]] = []
    for start, end in widened:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def render_turn(turn: SourceTurn, config: AnswerConfig,
                spans: Sequence[EvidenceMember] = ()) -> str:
    """Render one turn, restricted to its cited spans when a window is set."""
    text = turn.raw_text
    if config.span_window is not None and spans:
        segments = _clip_spans(spans, len(text), config.span_window)
        if segments:
            body = " ... ".join(text[start:end].strip() for start, end in segments)
            # A span that covers the whole turn should not gain an ellipsis.
            text = body if segments != ((0, len(text)),) else text
    header: list[str] = []
    if config.include_dates and turn.timestamp:
        header.append(f"[{turn.session_id} @ {turn.timestamp}]")
    else:
        header.append(f"[{turn.session_id}]")
    if config.include_speaker and turn.speaker:
        header.append(f"{turn.speaker[:config.max_speaker_chars]}:")
    body = " ".join(text.split())
    if config.normalize_relative_time and turn.timestamp:
        notes = _source_time_notes(body, turn)
        if notes:
            body = f"{body} {' '.join(notes)}"
    return " ".join(header) + " " + body


_SOURCE_RELATIVE_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|last|next|ago)\b", re.I)


def _source_time_notes(text: str, turn: SourceTurn) -> tuple[str, ...]:
    """Render deterministic source-anchored intervals for relative phrases."""

    rows: list[str] = []
    for phrase in extract_time_expressions(text):
        if not _SOURCE_RELATIVE_RE.search(phrase):
            continue
        interval = normalize_time(phrase, turn.timestamp, turn.turn_id)
        if interval.kind != "relative" or not interval.start:
            continue
        start = interval.start.split("T", 1)[0]
        end = (interval.end or interval.start).split("T", 1)[0]
        resolved = start if start == end else f"{start}..{end}"
        anchor = str(turn.timestamp).split(" ", 1)[0].split("T", 1)[0]
        rows.append(
            f'[source-time "{phrase}" => {resolved}; anchor={anchor}]')
    return tuple(rows)


def render_evidence(
    turns: Iterable[SourceTurn],
    *,
    config: AnswerConfig,
    counter: TokenCounter,
    max_tokens: int,
    session_order: Mapping[str, int] | None = None,
    spans_by_turn: Mapping[str, Sequence[EvidenceMember]] | None = None,
    mandatory_turn_ids: Sequence[str] = (),
    prefixes_by_turn: Mapping[str, str] | None = None,
) -> RenderedEvidence:
    """Render turns newest-context-last, dropping optional turns to fit the budget.

    Dropping is by reverse rank among optional turns, so the turns a proof
    bundle marked mandatory are the last to go.  When even the mandatory set
    exceeds ``max_tokens`` the caller is told via ``mandatory_dropped`` rather
    than being handed a silently truncated pack.
    """
    order = session_order or {}
    spans = spans_by_turn or {}
    prefixes = prefixes_by_turn or {}
    mandatory = set(mandatory_turn_ids)
    source_rows = list(turns)
    rows = (source_rows if config.evidence_order in {
        "relevance", "topological_plain", "topological",
        "topological_recency"} else
            sorted(source_rows,
                   key=lambda turn: (order.get(turn.session_id, 1 << 30),
                                     turn.session_id, turn.turn_index, turn.turn_id)))
    rendered = {
        turn.turn_id: " ".join(filter(None, (
            prefixes.get(turn.turn_id, ""),
            render_turn(turn, config, spans.get(turn.turn_id, ())),
        ))) for turn in rows
    }
    costs = dict(zip(rendered, counter.count_many(list(rendered.values()))))
    # +1 per turn for the joining newline.
    keep = [turn.turn_id for turn in rows]
    dropped: list[str] = []

    def total(ids: Sequence[str]) -> int:
        return sum(costs[item] + 1 for item in ids)

    # Most relevance-preserving layouts put the strongest block first, so the
    # historical reverse scan drops the weakest tail. ``topological_recency``
    # intentionally places the strongest block last for long-context recency;
    # applying the same reverse scan there silently deletes the evidence the
    # layout was designed to protect.  Select from the weak prefix in that mode
    # and retain the final presentation order after membership is decided.
    drop_order = (keep if config.evidence_order == "topological_recency"
                  else list(reversed(keep)))
    optional = [item for item in drop_order if item not in mandatory]
    for turn_id in optional:
        if total(keep) <= max_tokens:
            break
        keep.remove(turn_id)
        dropped.append(turn_id)
    mandatory_dropped = False
    if total(keep) > max_tokens:
        # Only mandatory turns remain and they still do not fit.
        mandatory_drop_order = (
            list(keep) if config.evidence_order == "topological_recency"
            else list(reversed(keep)))
        for turn_id in mandatory_drop_order:
            if total(keep) <= max_tokens or len(keep) <= 1:
                break
            keep.remove(turn_id)
            dropped.append(turn_id)
            mandatory_dropped = True
    kept = [turn_id for turn_id in (turn.turn_id for turn in rows) if turn_id in set(keep)]
    text = "\n".join(rendered[turn_id] for turn_id in kept)
    return RenderedEvidence(
        text=text, turn_ids=tuple(kept), dropped_turn_ids=tuple(sorted(dropped)),
        tokens=counter.count(text), truncated=bool(dropped),
        mandatory_dropped=mandatory_dropped,
    )
