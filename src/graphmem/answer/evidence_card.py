"""Bounded source-only evidence cards for the answer boundary.

The graph pack remains the source of truth.  This module only re-reads turns
that are already in that pack and repeats a few high-value excerpts after the
long evidence body, where the answer model is less likely to lose them to
recency or unrelated graph neighbours.  It never sees a prediction or label.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Mapping, Sequence

from ..domain import CandidateScore, SourceTurn
from ..text import content_terms
from ..tokenization import TokenCounter
from .rendering import AnswerConfig, render_turn


TYPED_EVIDENCE_CARD_VERSION = "graphmem-v5.73-typed-evidence-card-v1"

_TIME_RE = re.compile(
    r"\b(?:when|date|time|day|week|month|year|before|after|since|until|first|"
    r"last|latest|earliest|ago|today|yesterday|tomorrow|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|january|february|march|april|"
    r"may|june|july|august|september|october|november|december|(?:19|20)\d{2})\b",
    re.I,
)
_AGGREGATE_RE = re.compile(
    r"\b(?:how many|how much|count|number of|total|sum|average|mean|each|per|"
    r"minimum|maximum|most|least|list|which activities|what activities)\b",
    re.I,
)
_INFERENCE_RE = re.compile(
    r"\b(?:would|might|likely|could|infer(?:red)?|suggests?|indicates?|"
    r"underlying|personality|condition|benefit from)\b",
    re.I,
)
_STATUS_RE = re.compile(
    r"\b(?:not|never|cancelled|canceled|planned|scheduled|started|stopped|"
    r"finished|completed|attended|visited|bought|sold|joined|left|returned)\b",
    re.I,
)
_NUMBER_RE = re.compile(
    r"(?:[$£€¥]\s*)?\b\d+(?:[.,]\d+)?\b|"
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.I,
)
_SENTENCE_RE = re.compile(r"[^\n.!?。！？]+(?:[.!?。！？]+|\n+|$)")


@dataclass(frozen=True, slots=True)
class TypedEvidenceCard:
    route: str
    text: str
    turn_ids: tuple[str, ...]
    tokens: int
    schema_version: str = TYPED_EVIDENCE_CARD_VERSION


def evidence_card_route(question: str, answer_kind: str = "") -> str:
    """Choose a label-free readout obligation from the query surface."""

    kind = str(answer_kind).casefold()
    if _AGGREGATE_RE.search(question) or any(
            token in kind for token in ("count", "sum", "mean", "collection")):
        return "aggregate"
    if _TIME_RE.search(question) or any(
            token in kind for token in ("time", "date", "temporal", "ordinal",
                                        "argmin", "argmax", "latest")):
        return "temporal"
    if _INFERENCE_RE.search(question):
        return "inference"
    return "lookup"


def _best_excerpt(text: str, query_terms: frozenset[str], route: str,
                  limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(clean)
                 if match.group(0).strip()]
    if not sentences:
        return clean[:limit].rsplit(" ", 1)[0]

    def score(item: tuple[int, str]) -> tuple[float, int]:
        index, sentence = item
        value = 3.0 * len(content_terms(sentence) & query_terms)
        if route == "temporal" and _TIME_RE.search(sentence):
            value += 2.0
        if route == "aggregate" and _NUMBER_RE.search(sentence):
            value += 2.0
        if _STATUS_RE.search(sentence):
            value += 0.7
        return value, -index

    ranked = sorted(enumerate(sentences), key=lambda item: (
        -score(item)[0], -score(item)[1]))
    chosen: list[tuple[int, str]] = []
    chars = 0
    for index, sentence in ranked:
        if chosen and chars + len(sentence) + 1 > limit:
            continue
        chosen.append((index, sentence))
        chars += len(sentence) + 1
        if chars >= int(limit * 0.75):
            break
    excerpt = " ".join(sentence for _index, sentence in sorted(chosen))
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0]
    return excerpt.strip()


def build_typed_evidence_card(
    question: str,
    turns: Mapping[str, SourceTurn],
    packed_turn_ids: Sequence[str],
    candidate_scores: Sequence[CandidateScore],
    *,
    answer_kind: str = "",
    config: AnswerConfig,
    counter: TokenCounter,
) -> TypedEvidenceCard | None:
    """Build a <=500-token typed card from already packed source turns."""

    available = [turn_id for turn_id in packed_turn_ids if turn_id in turns]
    if not available:
        return None
    route = evidence_card_route(question, answer_kind)
    query_terms = content_terms(question)
    by_score = {row.turn_id: row for row in candidate_scores}
    pack_rank = {turn_id: index for index, turn_id in enumerate(available)}
    document_frequency: dict[str, int] = {}
    terms_by_turn: dict[str, frozenset[str]] = {}
    for turn_id in available:
        terms = content_terms(turns[turn_id].raw_text)
        terms_by_turn[turn_id] = terms
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    count = max(1, len(available))

    def lexical(turn_id: str) -> float:
        return sum(
            math.log((count + 1.0) / (document_frequency.get(term, 0) + 0.5))
            for term in query_terms & terms_by_turn[turn_id])

    explicit_speakers = {
        turn.speaker for turn in turns.values()
        if content_terms(turn.speaker)
        and content_terms(turn.speaker) <= query_terms
        and not content_terms(turn.speaker) <= {
            "user", "assistant", "system", "human"}
    }
    scored: list[tuple[float, int, str]] = []
    for turn_id in available:
        turn = turns[turn_id]
        candidate = by_score.get(turn_id)
        value = 3.0 * lexical(turn_id)
        if candidate is not None:
            value += min(2.0, candidate.exact_score + candidate.bm25_score
                         + candidate.dense_score)
            value += min(1.5, candidate.binding_score
                         + candidate.obligation_gain
                         + 0.5 * len(candidate.operand_ids))
        if explicit_speakers and turn.speaker in explicit_speakers:
            value += 2.0
        if route == "temporal" and (_TIME_RE.search(turn.raw_text)
                                     or turn.timestamp):
            value += 1.2
        if route == "aggregate" and _NUMBER_RE.search(turn.raw_text):
            value += 1.2
        if _STATUS_RE.search(turn.raw_text):
            value += 0.4
        scored.append((value, pack_rank[turn_id], turn_id))

    selected: list[str] = []
    selected_terms: list[frozenset[str]] = []
    remaining = {turn_id for _value, _rank, turn_id in scored}
    base_score = {turn_id: value for value, _rank, turn_id in scored}
    while remaining and len(selected) < config.typed_evidence_card_limit:
        def mmr(turn_id: str) -> tuple[float, int, str]:
            terms = terms_by_turn[turn_id]
            redundancy = max((
                len(terms & prior) / max(1, len(terms | prior))
                for prior in selected_terms), default=0.0)
            # Session diversity helps multi-hop while lexical relevance remains
            # the primary selection criterion.
            session_penalty = 0.35 if any(
                turns[item].session_id == turns[turn_id].session_id
                for item in selected) else 0.0
            return (base_score[turn_id] - 1.25 * redundancy - session_penalty,
                    -pack_rank[turn_id], turn_id)

        best = max(remaining, key=mmr)
        selected.append(best)
        selected_terms.append(terms_by_turn[best])
        remaining.remove(best)

    obligations = {
        "lookup": (
            "bind the exact subject and requested relation; reject a value from "
            "another person, object, or event"),
        "temporal": (
            "bind every required endpoint; event/source-time controls the answer, "
            "while observation_time only anchors relative phrases"),
        "aggregate": (
            "enumerate exact-scope operands, preserve completion/polarity, "
            "deduplicate mentions rather than distinct events, then compute once"),
        "inference": (
            "bind the stated traits or events first, then make only the narrow "
            "inference requested by the question"),
    }
    lines = [
        "Answer-critical evidence card (source excerpts only; not a proposed answer):",
        f"Typed obligation [{route}]: {obligations[route]}.",
    ]
    kept: list[str] = []
    for turn_id in selected:
        turn = turns[turn_id]
        excerpt = _best_excerpt(
            turn.raw_text, query_terms, route,
            config.typed_evidence_card_excerpt_chars)
        rendered = render_turn(replace(turn, raw_text=excerpt), config)
        line = f"K{len(kept) + 1} source={turn_id}: {rendered}"
        if counter.count("\n".join((*lines, line))) > (
                config.typed_evidence_card_max_tokens):
            # One retry with a smaller excerpt keeps the hard token contract
            # while retaining the source-time annotation generated by render_turn.
            excerpt = _best_excerpt(
                turn.raw_text, query_terms, route,
                max(80, config.typed_evidence_card_excerpt_chars // 2))
            rendered = render_turn(replace(turn, raw_text=excerpt), config)
            line = f"K{len(kept) + 1} source={turn_id}: {rendered}"
        if counter.count("\n".join((*lines, line))) > (
                config.typed_evidence_card_max_tokens):
            continue
        lines.append(line)
        kept.append(turn_id)
    if not kept:
        return None
    text = "\n".join(lines)
    return TypedEvidenceCard(
        route=route, text=text, turn_ids=tuple(kept), tokens=counter.count(text))
