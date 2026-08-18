"""Deterministic, evidence-grounded answer-plan compiler.

The compiler is deliberately downstream of retrieval and the validated V5.54
readout policy.  It never sees benchmark labels or gold answers and it does not
remove or globally reorder evidence.  For temporal/duration/state questions it
adds a small, query-ranked binding index next to generation, while retaining
the complete frozen memory block as the authority.

This addresses a specific failure mode of no-reasoning answer backbones: the
correct source turns are present among 64 memories, but the model binds a date
or value from a nearby event.  The index quotes source evidence verbatim; all
plan fields are routing metadata, never newly generated facts.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from ..domain import canonical_json
from ..retrieval.slots import QuerySlots, parse_slots
from ..tokenization import TokenCounter


ANSWER_PLAN_VERSION = "graphmem-v5.56-structured-answer-plan-v1"

_MEMORY_MARKER = "Conversation memories:\n"
_QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
_LABEL_RE = re.compile(
    r"^\[(?:CHAIN (?P<chain>\d+) (?:step=\d+|support)|"
    r"GRAPH (?P<graph>\d+) step=\d+|"
    r"AUX (?P<aux>\d+) rank=\d+)\]\s",
    re.M,
)
_RELAXED_LABEL_RE = re.compile(
    r"\[(?:CHAIN (?P<chain>\d+) (?:step=\d+|support)|"
    r"GRAPH (?P<graph>\d+) step=\d+|"
    r"AUX (?P<aux>\d+) rank=\d+)\]\s"
)
_END_MARKERS = (
    "\n\nAggregation ledger (",
    "\n\nAggregation execution card:",
    "\n\nOutput contract:",
    "\n\nFinal check:",
    "\n\nAnswer the original Question now:",
    "\n\nInference Question:",
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_DATE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b|\[source-time\s+\")",
    re.I,
)
_WHEN_RE = re.compile(r"^(?:when|what time|what date|which date)\b", re.I)
_AGO_RE = re.compile(r"\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b", re.I)
_AGE_RE = re.compile(r"\b(?:how old|how many years (?:old|will))\b", re.I)
_DURATION_RE = re.compile(
    r"^(?:how long|how many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?))\b.*"
    r"\b(?:between|before|after|when|since|until|take|passed|elapsed)\b",
    re.I,
)
_ORDER_RE = re.compile(r"^(?:who|which)\b.*\b(?:first|earlier|later|earliest)\b", re.I)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
    "you", "your",
})


@dataclass(frozen=True, slots=True)
class PlanCandidate:
    candidate_id: str
    cluster_id: str
    turn_index: int
    turn_id: str
    excerpt: str
    query_terms: tuple[str, ...]
    has_time_anchor: bool
    score: float
    endpoint_role: str = "event"


@dataclass(frozen=True, slots=True)
class AnswerPlan:
    kind: str
    question: str
    answer_slot: str
    content_terms: tuple[str, ...]
    head_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    temporal_phrase: str
    candidates: tuple[PlanCandidate, ...]
    endpoint_status: str
    answer_status: str

    def render(self) -> str:
        candidate_ids = ",".join(row.candidate_id for row in self.candidates)
        anchors = ",".join(
            row.candidate_id for row in self.candidates if row.has_time_anchor)
        bindings = candidate_ids or "unresolved"
        temporal = anchors or "unresolved-search-full-memory"
        lines = [
            "AnswerPlan (deterministic routing metadata; quoted excerpts are evidence):",
            f"operator: {self.kind}",
            f"answer_slot: {self.answer_slot or 'value'}",
            "event_terms: " + (", ".join(self.content_terms) or "unspecified"),
            f"event_binding: {self.endpoint_status} [{bindings}]",
            f"time_anchor: candidate [{temporal}]",
            f"answer_value: {self.answer_status}",
            "Candidate bindings (verbatim; cluster order is preserved locally):",
        ]
        for row in self.candidates:
            terms = ",".join(row.query_terms) or "context"
            lines.append(
                f"{row.candidate_id} role={row.endpoint_role} cluster={row.cluster_id} "
                f"match={terms} :: {row.excerpt}")
        procedure = {
            "date_difference": (
                "Bind the two exact event endpoints independently, resolve each "
                "endpoint from its own source timestamp/[source-time], and subtract "
                "once in the requested unit."),
            "relative_time": (
                "Bind the exact event time, then compare it with the Question "
                "reference date in the requested unit; do not manufacture a second "
                "memory event."),
            "age_projection": (
                "Bind an explicit age/date baseline and the exact target event time, "
                "then project the age once."),
            "latest_state": (
                "Bind states of the exact subject/relation and apply explicit "
                "updates in event-time order."),
            "temporal_order": (
                "Bind a dated event for every named candidate before comparing "
                "their times."),
            "temporal_lookup": (
                "Bind the exact subject/event first, then return the time/value "
                "from that same memory statement."),
        }.get(self.kind, "Bind the exact subject and event before reading its value.")
        lines.extend((
            "Execution: " + procedure +
            " Never borrow a value from a nearby event.",
            "The candidate index is not exhaustive. If it lacks a binding, search "
            "the full Conversation memories above before deciding information is "
            "insufficient.",
            f"Question: {self.question}",
            "Return one concise answer only.",
        ))
        return "\n".join(lines)

    def to_trace(self) -> dict[str, Any]:
        return {
            "version": ANSWER_PLAN_VERSION,
            "kind": self.kind,
            "answer_slot": self.answer_slot,
            "content_terms": list(self.content_terms),
            "head_terms": list(self.head_terms),
            "action_terms": list(self.action_terms),
            "temporal_phrase": self.temporal_phrase,
            "endpoint_status": self.endpoint_status,
            "answer_status": self.answer_status,
            "candidates": [{
                "candidate_id": row.candidate_id,
                "cluster_id": row.cluster_id,
                "turn_index": row.turn_index,
                "turn_id": row.turn_id,
                "query_terms": list(row.query_terms),
                "has_time_anchor": row.has_time_anchor,
                "score": row.score,
                "endpoint_role": row.endpoint_role,
            } for row in self.candidates],
        }


def _terms(text: str) -> set[str]:
    result: set[str] = set()
    for value in _WORD_RE.findall(text.casefold()):
        if value in _STOPWORDS or len(value) < 3:
            continue
        result.add(value)
        if len(value) > 5 and value.endswith("ing"):
            result.add(value[:-3])
        elif len(value) > 4 and value.endswith("ed"):
            result.add(value[:-2])
        elif len(value) > 4 and value.endswith("s"):
            result.add(value[:-1])
    return result


def _question(user: str) -> str:
    match = _QUESTION_RE.search(user)
    if match is None:
        raise ValueError("answer-plan question header is absent")
    return " ".join(match.group("question").split())


def _split_evidence(user: str) -> tuple[str, str, str]:
    start = user.find(_MEMORY_MARKER)
    if start < 0:
        raise ValueError("answer-plan conversation-memory marker is absent")
    evidence_start = start + len(_MEMORY_MARKER)
    stops = [position for marker in _END_MARKERS
             if (position := user.find(marker, evidence_start)) >= 0]
    evidence_end = min(stops) if stops else len(user)
    return user[:evidence_start], user[evidence_start:evidence_end], user[evidence_end:]


def _chunks(evidence: str, expected: int) -> list[tuple[int, str, str]]:
    matches = list(_LABEL_RE.finditer(evidence))
    if len(matches) != expected or (matches and matches[0].start() != 0):
        matches = list(_RELAXED_LABEL_RE.finditer(evidence))
    if len(matches) != expected or not matches or matches[0].start() != 0:
        raise ValueError(
            f"answer-plan evidence labels={len(matches)} expected={expected}")
    rows: list[tuple[int, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        cluster = (
            f"chain:{match.group('chain')}" if match.group("chain") else
            f"graph:{match.group('graph')}" if match.group("graph") else
            f"aux:{match.group('aux')}")
        rows.append((index, cluster, evidence[match.start():end].strip()))
    return rows


def _plan_kind(slots: QuerySlots, trace: dict[str, Any]) -> str:
    operation = str((trace.get("aggregation_ledger") or {}).get("operation") or "")
    question = str(trace.get("_answer_plan_question") or "")
    # A count/sum/difference card already owns its operands.  Treating a scope
    # phrase such as "since the start of the year" as a temporal subtraction
    # was the most damaging false route in the first prototype.
    if operation and operation != "date_difference":
        return ""
    if _AGO_RE.search(question):
        return "relative_time"
    if _AGE_RE.search(question):
        return "age_projection"
    if operation == "date_difference" or _DURATION_RE.search(question):
        return "date_difference"
    if slots.is_latest:
        return "latest_state"
    if _WHEN_RE.search(question):
        return "temporal_lookup"
    if _ORDER_RE.search(question):
        return "temporal_order"
    return ""


def _strip_question_frame(text: str) -> str:
    return re.sub(
        r"^\s*(?:how\s+long|how\s+many\s+\w+|what\s+time|when)\s+",
        "", text.strip(" ?."), flags=re.I)


def _event_clauses(question: str) -> tuple[str, str] | None:
    """Return two explicit event clauses; never invent a reference endpoint."""

    text = _strip_question_frame(question)
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", text, re.I)
    if between:
        return between.group(1).strip(), between.group(2).strip()
    relation = re.search(r"\b(before|after|when)\b", text, re.I)
    if relation:
        left = _strip_question_frame(text[:relation.start()])
        right = _strip_question_frame(text[relation.end():])
        # "did it take for me to X" carries no event content before X.
        left = re.sub(r"^(?:did\s+it\s+take\s+for\s+me\s+to|had\s+i\s+been|"
                      r"have\s+i\s+been|did\s+i)\s+", "", left, flags=re.I)
        if len(_terms(left)) >= 2 and len(_terms(right)) >= 2:
            return left, right
    return None


def compile_answer_plan(
    prepared: Any, *, max_candidates: int = 5, excerpt_chars: int = 440,
    enabled_kinds: tuple[str, ...] | None = None,
) -> AnswerPlan | None:
    """Compile one plan from frozen prompt bytes, never from evaluation data."""

    if not prepared.messages or max_candidates <= 0:
        return None
    user = str(prepared.messages[-1]["content"])
    question = _question(user)
    slots = parse_slots(question)
    plan_trace = dict(prepared.trace)
    plan_trace["_answer_plan_question"] = question
    kind = _plan_kind(slots, plan_trace)
    if not kind or (enabled_kinds is not None and kind not in enabled_kinds):
        return None
    if kind == "temporal_order":
        max_candidates = min(max_candidates, 3)
    event_clauses = _event_clauses(question) if kind == "date_difference" else None
    if kind == "date_difference" and event_clauses is None:
        # One stored event plus the question date ("months ago/since") is not
        # a two-memory relation and was empirically harmed by duplicate excerpts.
        return None
    _head, evidence, _tail = _split_evidence(user)
    chunks = _chunks(evidence, len(prepared.evidence_turn_ids))
    question_terms = set(slots.content_terms) or _terms(question)
    chunk_terms = [_terms(row[2]) for row in chunks]
    frequency = Counter(term for values in chunk_terms for term in values)
    count = max(1, len(chunks))

    def score_rows(terms_for_role: set[str], role: str):
        scored: list[tuple[float, int, str, str, tuple[str, ...], bool, str]] = []
        for (index, cluster, text), values in zip(chunks, chunk_terms):
            overlap = tuple(sorted(terms_for_role & values))
            if not overlap:
                continue
            idf = sum(math.log((count + 1) / (frequency[item] + 1)) + 1
                      for item in overlap)
            coverage = len(overlap) / max(1, len(terms_for_role))
            time_anchor = bool(_DATE_RE.search(text))
            score = round(2.0 * coverage + idf + 0.2 * time_anchor, 8)
            scored.append((score, index, cluster, text, overlap, time_anchor, role))
        return scored

    if event_clauses is not None:
        left_terms, right_terms = map(_terms, event_clauses)
        shared = left_terms & right_terms
        left_distinct = left_terms - shared
        right_distinct = right_terms - shared
        # Each endpoint must have a discriminative lexical witness; otherwise
        # the plan would only restate a broad topic such as "museum" twice.
        if not left_distinct or not right_distinct:
            return None
        left_rows = [row for row in score_rows(left_terms, "endpoint_A")
                     if set(row[4]) & left_distinct]
        right_rows = [row for row in score_rows(right_terms, "endpoint_B")
                      if set(row[4]) & right_distinct]
        if not left_rows or not right_rows:
            return None
        per_endpoint = max(1, max_candidates // 2)
        selected = (sorted(left_rows, key=lambda row: (-row[0], row[1]))[:per_endpoint]
                    + sorted(right_rows, key=lambda row: (-row[0], row[1]))[:per_endpoint])
        # A single source turn may state both events, but two separately dated
        # endpoints require at least two source turns before the plan claims a
        # relational binding.
        if len({row[1] for row in selected}) < 2:
            return None
    else:
        selected = score_rows(question_terms, "candidate")
        selected = sorted(selected, key=lambda row: (-row[0], row[1]))[:max_candidates]

    if not selected:
        return None
    selected.sort(key=lambda row: row[1])

    candidates = tuple(PlanCandidate(
        candidate_id=f"B{position}", cluster_id=row[2], turn_index=row[1],
        turn_id=str(prepared.evidence_turn_ids[row[1]]),
        excerpt=" ".join(row[3].split())[:excerpt_chars],
        query_terms=row[4], has_time_anchor=row[5], score=row[0],
        endpoint_role=row[6],
    ) for position, row in enumerate(selected, 1))
    anchored = sum(row.has_time_anchor for row in candidates)
    endpoint_status = (
        "candidate-supported" if candidates else "unresolved")
    answer_status = (
        "derived-from-two-bound-endpoints" if kind == "date_difference" else
        "derived-from-event-and-question-reference-time" if kind == "relative_time" else
        "derived-from-age-baseline-and-event-time" if kind == "age_projection" else
        "derived-from-latest-valid-state" if kind == "latest_state" else
        "derived-by-ordering-bound-events" if kind == "temporal_order" else
        "supported-by-the-bound-event")
    # Do not claim endpoint completeness.  That was the source of the V5.55
    # false-abstention regression.
    if kind in {"date_difference", "age_projection"} and anchored < 2:
        endpoint_status = "candidate-partial-search-full-memory"
    answer_slot = (
        "duration" if kind in {"date_difference", "relative_time"} else
        "age" if kind == "age_projection" else
        "entity" if kind == "temporal_order" else slots.answer_slot)
    return AnswerPlan(
        kind=kind, question=question, answer_slot=answer_slot,
        content_terms=tuple(slots.content_terms),
        head_terms=tuple(slots.head_terms), action_terms=tuple(slots.action_terms),
        temporal_phrase=slots.temporal_phrase, candidates=candidates,
        endpoint_status=endpoint_status, answer_status=answer_status)


def apply_answer_plan(
    prepared: Any, counter: TokenCounter, *, max_candidates: int = 5,
    excerpt_chars: int = 440, max_prompt_tokens: int | None = None,
    enabled_kinds: tuple[str, ...] = ("date_difference", "temporal_order"),
) -> Any:
    """Append a bounded plan; return the source prompt when it is ineligible."""

    plan = compile_answer_plan(
        prepared, max_candidates=max_candidates, excerpt_chars=excerpt_chars,
        enabled_kinds=enabled_kinds)
    if plan is None:
        return prepared
    messages = [dict(row) for row in prepared.messages]
    appendix = "\n\n" + plan.render()
    messages[-1]["content"] = messages[-1]["content"].rstrip() + appendix
    prompt_tokens = sum(counter.count(row["content"]) for row in messages)
    if max_prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
        # Retry with fewer/shorter candidates before falling back unchanged.
        if max_candidates > 2:
            return apply_answer_plan(
                prepared, counter, max_candidates=max(2, max_candidates - 2),
                excerpt_chars=min(excerpt_chars, 300),
                max_prompt_tokens=max_prompt_tokens,
                enabled_kinds=enabled_kinds)
        return prepared
    trace = dict(prepared.trace)
    current = str(trace.get("prompt_version") or "")
    trace["prompt_version"] = "+".join(filter(None, (
        current, ANSWER_PLAN_VERSION)))
    trace["answer_plan"] = plan.to_trace()
    trace["answer_plan_source_payload_hash"] = prepared.prompt_payload_hash
    rows = tuple(messages)
    prompt_hash = hashlib.sha256(
        (trace["prompt_version"] + rows[0]["content"]).encode()).hexdigest()
    payload_hash = hashlib.sha256(canonical_json(rows).encode()).hexdigest()
    return replace(
        prepared, messages=rows, packing_prompt_tokens=prompt_tokens,
        prompt_hash=prompt_hash, prompt_payload_hash=payload_hash, trace=trace)


__all__ = [
    "ANSWER_PLAN_VERSION", "AnswerPlan", "PlanCandidate",
    "apply_answer_plan", "compile_answer_plan",
]
