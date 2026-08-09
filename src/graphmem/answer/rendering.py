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
from typing import Iterable, Mapping, Sequence

from ..domain import EvidenceMember, SourceTurn
from ..tokenization import TokenCounter

#: ``span_window=None`` renders the whole turn.  0 renders only the cited span.
FULL_TURN: None = None


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
    closed_form_enabled: bool = True
    # Remains false until an untouched holdout demonstrates >=99.5% precision
    # and <=0.5% false-complete for each whitelisted operator.
    deterministic_bypass_enabled: bool = False
    max_output_tokens: int = 256
    #: Rendered evidence is ``[session date] speaker: text``; this bounds the
    #: per-turn header so a pathological speaker label cannot eat the budget.
    max_speaker_chars: int = 48

    def __post_init__(self) -> None:
        if self.span_window is not None and self.span_window < 0:
            raise ValueError("span_window must be None or non-negative")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class RenderedEvidence:
    text: str
    turn_ids: tuple[str, ...]
    dropped_turn_ids: tuple[str, ...]
    tokens: int
    truncated: bool
    #: True when the budget could only be met by dropping a mandatory turn.
    mandatory_dropped: bool = False


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
    return " ".join(header) + " " + " ".join(text.split())


def render_evidence(
    turns: Iterable[SourceTurn],
    *,
    config: AnswerConfig,
    counter: TokenCounter,
    max_tokens: int,
    session_order: Mapping[str, int] | None = None,
    spans_by_turn: Mapping[str, Sequence[EvidenceMember]] | None = None,
    mandatory_turn_ids: Sequence[str] = (),
) -> RenderedEvidence:
    """Render turns newest-context-last, dropping optional turns to fit the budget.

    Dropping is by reverse rank among optional turns, so the turns a proof
    bundle marked mandatory are the last to go.  When even the mandatory set
    exceeds ``max_tokens`` the caller is told via ``mandatory_dropped`` rather
    than being handed a silently truncated pack.
    """
    order = session_order or {}
    spans = spans_by_turn or {}
    mandatory = set(mandatory_turn_ids)
    rows = sorted(turns, key=lambda turn: (order.get(turn.session_id, 1 << 30),
                                           turn.session_id, turn.turn_index, turn.turn_id))
    rendered = {turn.turn_id: render_turn(turn, config, spans.get(turn.turn_id, ()))
                for turn in rows}
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
