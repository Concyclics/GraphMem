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
    # and the matching prompt contract.
    evidence_order: str = "chronological"
    # Materialize relative phrases against the source turn timestamp in the
    # rendered evidence.  This prevents the answer model from re-anchoring
    # ``last week/month/year`` against the later question date.
    normalize_relative_time: bool = False
    # Opt-in V5.20 answer contract for noisy graph reservoirs.  Kept separate
    # from source-time normalization so prompt-only ablations remain possible.
    precision_grounding: bool = False
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

    def __post_init__(self) -> None:
        if self.span_window is not None and self.span_window < 0:
            raise ValueError("span_window must be None or non-negative")
        if self.evidence_order not in {
                "chronological", "relevance", "adaptive",
                "topological_plain", "topological"}:
            raise ValueError(
                "evidence_order must be chronological, relevance, adaptive, "
                "topological_plain, or topological")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be None or positive")
        if self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")


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
        "relevance", "topological_plain", "topological"} else
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

    optional = [item for item in reversed(keep) if item not in mandatory]
    for turn_id in optional:
        if total(keep) <= max_tokens:
            break
        keep.remove(turn_id)
        dropped.append(turn_id)
    mandatory_dropped = False
    if total(keep) > max_tokens:
        # Only mandatory turns remain and they still do not fit.
        for turn_id in [item for item in reversed(keep)]:
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
