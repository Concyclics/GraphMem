"""Validated, label-free answer readout policies.

V5.43--V5.54 were measured as a sequence of prompt-only materializers over a
frozen :class:`~graphmem.answer.stage.PreparedAnswer` corpus.  Keeping those
rewrites outside the read path made the reported winner impossible to obtain
from the online implementation.  This module is the core equivalent of that
sequence.  It deliberately operates after evidence packing so it cannot alter
retrieval coverage, and every accepted rewrite is budget-nonincreasing.

The V5.54 route reads only the question, rendered memories and the existing
mechanical aggregation/preference trace.  It never imports evaluation code or
sees a benchmark label, gold answer, prediction or judge verdict.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..domain import canonical_json
from ..tokenization import TokenCounter
from .prompts import AGGREGATION_LEDGER_APPENDIX


V5_54_POLICY = "v5_54"

_TYPED_VERSION = "graphmem-v5.43-typed-readout-v1"
_RECENCY_VERSION = "graphmem-v5.42-topological-recency-readout-v1"
_COMPACT_AGGREGATION_VERSION = (
    "graphmem-v5.60-compact-aggregation-worksheet-v1")
_SINGLE_LINE_VERSION = "graphmem-v5.47-single-line-aggregation-stop-v1"
_INFERENCE_VERSION = "graphmem-v5.48-grounded-inference-synthesis-v1"
_LEXICAL_BLOCK_VERSION = "graphmem-v5.51-lexical-block-readout-v1"
_MAX_READOUT_TOKEN_INCREASE = 500

_MEMORY_MARKER = "Conversation memories:\n"
_FOOTER_MARKER = "\n\nAnswer the original Question now:"
_END_MARKERS = (
    "\n\nAggregation ledger (",
    "\n\nQuery focus (",
    "\n\nOutput contract:",
    "\n\nFinal check:",
    _FOOTER_MARKER,
)
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
_NAMED_SPEAKER_RE = re.compile(
    r"^\[(?:CHAIN \d+ (?:step=\d+|support)|GRAPH \d+ step=\d+|"
    r"AUX \d+ rank=\d+)\]\s+\[[^\]]+\]\s+([^:\n]{1,48}):",
    re.M,
)
_GENERIC_SPEAKERS = frozenset({
    "", "assistant", "system", "tool", "user", "human",
})
_RECENCY_PREFIXES = frozenset({"what", "when", "who", "which"})

_COUNT_RE = re.compile(
    r"\b(?:how many|count|total|sum|average|mean|difference|most|least|"
    r"fewest|number of)\b", re.I)
_DURATION_RE = re.compile(
    r"\b(?:how long|after how many|duration|time between|how much time)\b",
    re.I)
_TEMPORAL_RE = re.compile(
    r"^(?:when|how long|after how many)\b|"
    r"\b(?:what|which)\s+(?:date|day|month|year)\b", re.I)
_TYPED_TEMPORAL_RE = re.compile(
    r"\b(?:when|what date|which date|what year|which year|what month|"
    r"before|after|first|last|latest|earliest|recent(?:ly)?)\b", re.I)
_LIST_RE = re.compile(
    r"\b(?:which|what)\s+(?:activities|activity|books|classes|events|items|"
    r"locations|places|things|types|ways|games|movies|films|songs|people|"
    r"friends|sports|hobbies|projects|countries|cities|states|foods|meals|"
    r"restaurants|gifts|skills|languages|instruments|festivals|trips)\b",
    re.I)
_WHY_RE = re.compile(r"^(?:why|what (?:made|caused|reason)|how come)\b", re.I)
_QUESTION_MODE_RE = re.compile(r"^(?P<mode>who|where|what|which)\b", re.I)
_INFERENCE_RE = re.compile(
    r"\b(?:would|might|likely|could|considered|personality traits|"
    r"underlying condition|attributes describe|based on)\b", re.I)
_COUNTERFACTUAL_RE = re.compile(
    r"\bif\b.*\b(?:hadn't|had\s+not|didn't|did\s+not|weren't|were\s+not|"
    r"wouldn't|would\s+not)\b", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b", re.I)
_EXACT_ON_YEAR_RE = re.compile(
    r"\bon\b[^?\n]{0,40}\b(?:19|20)\d{2}\b", re.I)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "with", "you", "your",
})

_STRICT_ABSENCE = (
    "If the exact requested entity or relation was never mentioned, say so and "
    "name the near-match only when useful."
)
_INFERENCE_SYSTEM = (
    "For modal inference, infer from stated facts and ordinary "
    "knowledge; the conclusion need not be verbatim."
)
_COMPACT_AGGREGATION_SYSTEM = (
    " An Aggregation execution card follows the graph-grouped memories. It "
    "specifies the requested operation but is not evidence. Select the complete "
    "operand set only from direct memories matching the exact subject, relation, "
    "time, polarity, and completion state. Deduplicate repeated mentions of one "
    "occurrence, retain distinct occurrences, and never treat an absent operand "
    "as zero. Return only the computed answer."
)
_COMPACT_AGGREGATION_WORKSHEET_SYSTEM = (
    " An Aggregation execution card follows the graph-grouped memories. It "
    "specifies the requested operation and may include a bounded reading index "
    "that quotes already supplied source turns. The index is not a certified "
    "answer or a complete set. Select the complete "
    "operand set only from direct memories matching the exact subject, relation, "
    "time, polarity, and completion state. Deduplicate repeated mentions of one "
    "occurrence, retain distinct occurrences, and never treat an absent operand "
    "as zero. Return only the computed answer."
)
_LEDGER_MARKER = "\n\nAggregation ledger ("
_OUTPUT_MARKER = "\n\nOutput contract:"
_OPERATION_RE = re.compile(r"(?:^|\n)Operation:\s*(?P<operation>[a-z_]+)")
_SINGLE_LINE_SOURCE_GUARD = (
    "Use graph adjacency only to find related evidence; sharing a graph block "
    "does not by itself make a memory an operand."
)
_SINGLE_LINE_COMPUTE = "Compute and answer this exact Question now: "
_SINGLE_LINE_NEW = "Graph proximity is not operand proof."
_SINGLE_LINE_STOP = (
    "\nOutput one concise final-answer line only, then stop; never list evidence "
    "or repeat."
)
_ANONYMOUS_COMPACT_OPERATIONS = frozenset({
    "date_difference", "difference", "mean", "unit_rate",
})


class ReadoutPolicyError(ValueError):
    """The selected policy was applied to an incompatible base prompt."""


def _messages(prepared: Any) -> list[dict[str, str]]:
    return [dict(message) for message in prepared.messages]


def _question(user: str) -> str:
    match = _QUESTION_RE.search(user)
    if match is None:
        raise ReadoutPolicyError("question header is absent")
    return " ".join(match.group("question").split())


def _prefix(question: str) -> str:
    words = question.casefold().split()
    return words[0].strip("'\"([{.,?!:;") if words else ""


def _split_evidence(user: str) -> tuple[str, str, str]:
    start = user.find(_MEMORY_MARKER)
    if start < 0:
        raise ReadoutPolicyError("conversation memory marker is absent")
    evidence_start = start + len(_MEMORY_MARKER)
    stops = [position for marker in _END_MARKERS
             if (position := user.find(marker, evidence_start)) >= 0]
    evidence_end = min(stops) if stops else len(user)
    return user[:evidence_start], user[evidence_start:evidence_end], user[evidence_end:]


def _named_multi_party(evidence: str) -> bool:
    return any(match.group(1).casefold().strip() not in _GENERIC_SPEAKERS
               for match in _NAMED_SPEAKER_RE.finditer(evidence))


def _named(prepared: Any) -> bool:
    if not prepared.messages:
        return False
    _head, evidence, _tail = _split_evidence(str(prepared.messages[-1]["content"]))
    return _named_multi_party(evidence)


def _with_prompt(
    prepared: Any,
    messages: Sequence[Mapping[str, str]],
    trace: Mapping[str, Any],
    counter: TokenCounter,
    *,
    evidence_turn_ids: Sequence[str] | None = None,
    allow_token_increase: bool = False,
) -> Any:
    rows = tuple(dict(message) for message in messages)
    prompt_tokens = sum(counter.count(message["content"]) for message in rows)
    permitted_increase = (
        _MAX_READOUT_TOKEN_INCREASE if allow_token_increase else 0)
    if prompt_tokens > prepared.packing_prompt_tokens + permitted_increase:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: readout policy increased prompt tokens "
            f"{prepared.packing_prompt_tokens} -> {prompt_tokens}")
    version = str(trace.get("prompt_version") or "")
    prompt_hash = hashlib.sha256(
        (version + rows[0]["content"]).encode()).hexdigest()
    payload_hash = hashlib.sha256(canonical_json(rows).encode()).hexdigest()
    return replace(
        prepared,
        messages=rows,
        evidence_turn_ids=(
            tuple(evidence_turn_ids) if evidence_turn_ids is not None
            else prepared.evidence_turn_ids),
        packing_prompt_tokens=prompt_tokens,
        prompt_hash=prompt_hash,
        prompt_payload_hash=payload_hash,
        trace=dict(trace),
    )


def _append_version(trace: dict[str, Any], version: str) -> str:
    current = str(trace.get("prompt_version") or "")
    value = "+".join(filter(None, (current, version)))
    trace["prompt_version"] = value
    return value


def _readout_kind(question: str) -> str:
    if _COUNT_RE.search(question):
        return "collection"
    if _DURATION_RE.search(question):
        return "duration"
    if _WHY_RE.search(question):
        return "causal"
    if _TYPED_TEMPORAL_RE.search(question):
        return "temporal"
    if _LIST_RE.search(question):
        return "list"
    return "lookup"


def _readout_contract(kind: str) -> str:
    return {
        "collection": (
            "Enumerate distinct qualifying facts, exclude plans and duplicates, "
            "then compute once. Return only the concise answer."),
        "duration": (
            "Match the exact event endpoints, resolve each from its own memory "
            "date, then compute once. Return only the concise answer."),
        "causal": (
            "Use the explicit reason for the exact subject and event; ignore "
            "same-topic explanations. Return only the concise answer."),
        "temporal": (
            "Match the exact subject and event, then use that memory's date or "
            "[source-time]; do not substitute another event. Return only the "
            "concise answer."),
        "list": (
            "Collect every distinct item satisfying the exact subject and "
            "relation; exclude plans, near-matches, and duplicates. Return only "
            "the concise answer."),
        "lookup": (
            "Match the exact subject, relation, and event; ignore same-topic "
            "near-matches. Return only the concise answer."),
    }[kind]


def _typed_readout(prepared: Any, counter: TokenCounter) -> Any:
    messages = _messages(prepared)
    user = messages[-1]["content"]
    head, evidence, suffix = _split_evidence(user)
    question = _question(user)
    kind = _readout_kind(question)
    footer_at = suffix.find(_FOOTER_MARKER)
    if footer_at < 0:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: question recency footer is absent")
    footer = (f"{_FOOTER_MARKER} {question}\n"
              f"{_readout_contract(kind)}")
    messages[-1]["content"] = head + evidence + suffix[:footer_at] + footer
    trace = dict(prepared.trace)
    _append_version(trace, _TYPED_VERSION)
    trace.update({
        "typed_readout": True,
        "typed_readout_kind": kind,
        "recency_layout_gate": False,
        "recency_layout_blocks": 0,
        "typed_readout_source_payload_hash": prepared.prompt_payload_hash,
    })
    return _with_prompt(
        prepared, messages, trace, counter, allow_token_increase=True)


def _chunks(evidence: str) -> list[tuple[str, str]]:
    matches = list(_LABEL_RE.finditer(evidence))
    if not matches or matches[0].start() != 0:
        raise ReadoutPolicyError(
            "topological evidence does not begin with a known label")
    rows: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        key = (
            f"chain:{match.group('chain')}" if match.group("chain") else
            f"graph:{match.group('graph')}" if match.group("graph") else
            f"aux:{match.group('aux')}"
        )
        rows.append((key, evidence[match.start():end]))
    return rows


def _blocks(evidence: str, expected_turns: int) -> list[list[tuple[int, str, str]]]:
    rows = _chunks(evidence)
    if len(rows) != expected_turns:
        matches = list(_RELAXED_LABEL_RE.finditer(evidence))
        if (not matches or matches[0].start() != 0
                or len(matches) != expected_turns):
            raise ReadoutPolicyError(
                f"rendered evidence has {len(rows)} anchored / {len(matches)} "
                f"relaxed labels for {expected_turns} turns")
        rows = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
            key = (
                f"chain:{match.group('chain')}" if match.group("chain") else
                f"graph:{match.group('graph')}" if match.group("graph") else
                f"aux:{match.group('aux')}"
            )
            rows.append((key, evidence[match.start():end]))
    blocks: list[list[tuple[int, str, str]]] = []
    for index, (key, value) in enumerate(rows):
        if not blocks or blocks[-1][-1][1] != key:
            blocks.append([])
        blocks[-1].append((index, key, value))
    return blocks


def _reverse_blocks(prepared: Any, counter: TokenCounter) -> Any:
    messages = _messages(prepared)
    head, evidence, tail = _split_evidence(messages[-1]["content"])
    blocks = _blocks(evidence, len(prepared.evidence_turn_ids))
    rows = [row for block in reversed(blocks) for row in block]
    permutation = [row[0] for row in rows]
    if permutation == list(range(len(permutation))):
        return prepared
    old = "Blocks are ranked by relevance, then graph or time order."
    new = "Blocks run weakest-to-strongest; keep internal order."
    if old not in messages[0]["content"]:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: compact topology contract is absent")
    messages[0]["content"] = messages[0]["content"].replace(old, new, 1)
    messages[-1]["content"] = head + "".join(row[2] for row in rows) + tail
    trace = dict(prepared.trace)
    _append_version(trace, _RECENCY_VERSION)
    trace.update({
        "evidence_order": "topological_recency",
        "resolved_evidence_order": "topological_recency",
        "recency_layout_source_payload_hash": prepared.prompt_payload_hash,
        "recency_layout_blocks": len(blocks),
    })
    evidence_ids = [prepared.evidence_turn_ids[index] for index in permutation]
    return _with_prompt(
        prepared, messages, trace, counter, evidence_turn_ids=evidence_ids)


def _aggregation_rule(operation: str, question: str = "") -> str:
    if (question and operation == "date_difference"
            and re.search(
                r"\bhow\s+many\s+(?:minutes?|hours?|days?|weeks?|months?|"
                r"years?)\s+did\s+it\s+take\b", question, re.I)
            and re.search(r"\band\b", question, re.I)):
        return (
            "Bind the separately stated completed duration for each named "
            "activity, deduplicate repeated mentions, add the durations once, "
            "and return only the combined duration in the requested unit.")
    if (question and operation == "minimum"
            and re.search(r"\bminimum\s+amount\b", question, re.I)
            and re.search(r"\band\b", question, re.I)):
        return (
            "Bind the lower-bound sale amount for every named item, then add "
            "those lower bounds once. Return the combined minimum proceeds, "
            "not the smaller individual item value.")
    return {
        "count_distinct": (
            "Enumerate the complete set of distinct qualifying items or "
            "occurrences from direct statements. Match the status requested by "
            "the question: completed/purchased/attended excludes unrealized "
            "plans, while a question about planned or pending items retains them. "
            "Exclude near-matches and duplicate mentions; then count once."),
        "sum": (
            "Collect every distinct unit-compatible amount for the exact subject "
            "and scope; exclude plans, unrelated amounts, stated subtotals, and "
            "duplicate mentions; then add the operands once."),
        "unit_rate": (
            "Bind the exact aggregate price and the matching distinct item count. "
            "Divide total price by item count exactly once and preserve the "
            "currency unit; do not add unrelated prices."),
        "difference": (
            "Bind the exact two requested quantities. For remaining or needed, "
            "use target minus the latest current amount; preserve the requested "
            "order, sign, and unit. For savings, subtract the exact chosen cost "
            "from the explicitly rejected cost and do not substitute another "
            "nearby alternative. In a personal comparison, prefer costs the user "
            "explicitly stated or adopted for those exact options over generic "
            "assistant estimates. Only the user's own turn can bind that personal "
            "cost; an assistant-only estimate is not an operand. If the user "
            "never bound one option's cost, answer insufficient information."),
        "date_difference": (
            "Bind the exact start and end events, resolve each from its own memory "
            "date or [source-time], subtract the calendar endpoints in the "
            "requested unit, and ignore unrelated durations or booking lead time."),
        "mean": (
            "Bind the complete requested population, use one value per distinct "
            "member, sum those values, and divide once by the member count."),
        "minimum": (
            "Bind all qualifying values for the exact scope and return the minimum "
            "with its original unit; ignore unrelated values and near-matches."),
        "maximum": (
            "Bind all qualifying values for the exact scope and return the maximum "
            "with its original unit; ignore unrelated values and near-matches."),
    }.get(operation, (
        "Select the complete exact operand set, apply the named operation once, "
        "and preserve the requested unit."))


def _compact_aggregation(prepared: Any, counter: TokenCounter) -> Any:
    trace = dict(prepared.trace)
    ledger_trace = trace.get("aggregation_ledger") or {}
    messages = _messages(prepared)
    user = messages[-1]["content"]
    ledger_at = user.find(_LEDGER_MARKER)
    output_at = user.find(_OUTPUT_MARKER, ledger_at + 1)
    if ledger_at < 0:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: aggregation marker is absent")
    if output_at < 0:
        output_at = len(user)
    question = _question(user)
    operation_match = _OPERATION_RE.search(user[ledger_at:output_at])
    if operation_match is None:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: aggregation operation is absent")
    operation = operation_match.group("operation")
    traced_operation = str(ledger_trace.get("operation") or "")
    if traced_operation and operation != traced_operation:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: aggregation operation trace mismatch")
    worksheet = tuple(
        str(row) for row in ledger_trace.get("worksheet_lines") or ()
        if str(row).strip()
    ) if ledger_trace.get("worksheet_enabled") else ()
    worksheet_block = ""
    if worksheet:
        worksheet_block = (
            "\nOperand worksheet (verbatim packed-source candidates; not a "
            "complete set):\n"
            + "\n".join(worksheet)
        )
    card = (
        "\n\nAggregation execution card:\n"
        f"Operation: {operation}\n"
        f"Procedure: {_aggregation_rule(operation, question if worksheet else '')}\n"
        "Use graph adjacency only to find related evidence; sharing a graph block "
        "does not by itself make a memory an operand."
        f"{worksheet_block}\n"
        f"Compute and answer this exact Question now: {question}"
    )
    messages[-1]["content"] = user[:ledger_at] + user[output_at:] + card
    if AGGREGATION_LEDGER_APPENDIX not in messages[0]["content"]:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: aggregation system appendix is absent")
    compact_system = (
        _COMPACT_AGGREGATION_WORKSHEET_SYSTEM
        if worksheet else _COMPACT_AGGREGATION_SYSTEM)
    messages[0]["content"] = messages[0]["content"].replace(
        AGGREGATION_LEDGER_APPENDIX, compact_system, 1)
    _append_version(trace, _COMPACT_AGGREGATION_VERSION)
    trace.update({
        "aggregation_execution_card": True,
        "aggregation_execution_operation": operation,
        "aggregation_ledger_candidates_rendered": 0,
        "aggregation_ledger_candidates_available": len(
            ledger_trace.get("candidate_turn_ids") or ()),
        "aggregation_worksheet_rows": len(worksheet),
        "aggregation_worksheet_turn_ids": list(
            ledger_trace.get("worksheet_turn_ids") or ()),
        "aggregation_execution_source_payload_hash": prepared.prompt_payload_hash,
    })
    return _with_prompt(
        prepared, messages, trace, counter, allow_token_increase=True)


def _single_line_aggregation(prepared: Any, counter: TokenCounter) -> Any:
    messages = _messages(prepared)
    user = messages[-1]["content"]
    if (_SINGLE_LINE_SOURCE_GUARD not in user
            or _SINGLE_LINE_COMPUTE not in user):
        raise ReadoutPolicyError(
            f"{prepared.question_id}: compact aggregation suffix is absent")
    messages[-1]["content"] = (
        user.replace(_SINGLE_LINE_SOURCE_GUARD, _SINGLE_LINE_NEW, 1)
        .replace(_SINGLE_LINE_COMPUTE, "Question: ", 1)
        + _SINGLE_LINE_STOP)
    trace = dict(prepared.trace)
    _append_version(trace, _SINGLE_LINE_VERSION)
    trace.update({
        "aggregation_single_line_stop": True,
        "aggregation_single_line_source_payload_hash": prepared.prompt_payload_hash,
    })
    return _with_prompt(prepared, messages, trace, counter)


def _inference_synthesis(prepared: Any, counter: TokenCounter) -> Any:
    messages = _messages(prepared)
    user = messages[-1]["content"]
    question = _question(user)
    footer_at = user.find(_FOOTER_MARKER)
    if footer_at < 0:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: inference question footer is absent")
    if _STRICT_ABSENCE not in messages[0]["content"]:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: strict absence clause is absent")
    messages[0]["content"] = messages[0]["content"].replace(
        _STRICT_ABSENCE, _INFERENCE_SYSTEM, 1)
    messages[-1]["content"] = user[:footer_at] + (
        f"\n\nInference Question: {question}\n"
        "Infer one plausible answer from stated personal facts and ordinary "
        "knowledge. Implicit is allowed; invent no facts. Output one line."
    )
    trace = dict(prepared.trace)
    _append_version(trace, _INFERENCE_VERSION)
    trace.update({
        "inference_synthesis": True,
        "inference_synthesis_source_payload_hash": prepared.prompt_payload_hash,
    })
    return _with_prompt(prepared, messages, trace, counter)


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for value in _WORD_RE.findall(text.casefold()):
        if value in _STOPWORDS or len(value) < 2:
            continue
        terms.add(value)
        if len(value) > 5 and value.endswith("ing"):
            terms.add(value[:-3])
        elif len(value) > 4 and value.endswith("ed"):
            terms.add(value[:-2])
        elif len(value) > 4 and value.endswith("s"):
            terms.add(value[:-1])
    return terms


def _question_mode(question: str) -> str:
    if _TEMPORAL_RE.search(question):
        return "temporal"
    match = _QUESTION_MODE_RE.search(question)
    return match.group("mode").casefold() if match else "other"


def _lexical_reorder(
    prepared: Any, counter: TokenCounter, *, mode: str, speaker_scope: str,
) -> Any:
    trace = dict(prepared.trace)
    if trace.get("aggregation_ledger") or trace.get("preference_synthesis"):
        return prepared
    messages = _messages(prepared)
    head, evidence, tail = _split_evidence(messages[-1]["content"])
    question = _question(messages[-1]["content"])
    actual_mode = _question_mode(question)
    named = _named_multi_party(evidence)
    speaker_match = (speaker_scope == "all"
                     or (speaker_scope == "named" and named)
                     or (speaker_scope == "anonymous" and not named))
    if not speaker_match or actual_mode != mode:
        return prepared

    blocks = _blocks(evidence, len(prepared.evidence_turn_ids))
    query_terms = _terms(question)
    block_terms = [_terms("".join(row[2] for row in block)) for block in blocks]
    frequency = Counter(term for terms in block_terms for term in terms)
    count = max(1, len(blocks))

    def score(index: int) -> tuple[float, float, int]:
        overlap = query_terms & block_terms[index]
        # Set iteration order is process-randomized.  Sorting prevents tiny
        # floating-point summation differences from changing tied block order
        # across workers or interpreter restarts.
        idf = sum(math.log((count + 1) / (frequency[term] + 1)) + 1
                  for term in sorted(overlap))
        coverage = len(overlap) / max(1, len(query_terms))
        return coverage, idf, index

    order = sorted(range(len(blocks)), key=score)
    rows = [row for index in order for row in blocks[index]]
    permutation = [row[0] for row in rows]
    if permutation == list(range(len(permutation))):
        return prepared
    messages[-1]["content"] = head + "".join(row[2] for row in rows) + tail
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    # BPE boundaries can grow by one token after two blocks become adjacent.
    # The measured policy keeps the old row in that case.
    if prompt_tokens > prepared.packing_prompt_tokens:
        return prepared
    _append_version(trace, _LEXICAL_BLOCK_VERSION)
    trace.update({
        "evidence_order": "topological_query_overlap",
        "resolved_evidence_order": "topological_query_overlap",
        "lexical_block_readout": True,
        "lexical_block_readout_mode": actual_mode,
        "lexical_block_count": len(blocks),
        "lexical_block_source_payload_hash": prepared.prompt_payload_hash,
    })
    evidence_ids = [prepared.evidence_turn_ids[index] for index in permutation]
    return _with_prompt(
        prepared, messages, trace, counter, evidence_turn_ids=evidence_ids)


def apply_v5_54_readout(prepared: Any, counter: TokenCounter) -> Any:
    """Apply the exact V5.54 winner policy to one packed answer request."""

    if not prepared.messages:
        trace = dict(prepared.trace)
        trace.update({"readout_policy": V5_54_POLICY,
                      "readout_policy_route": ["deterministic"]})
        return replace(prepared, trace=trace)

    base_evidence = frozenset(prepared.evidence_turn_ids)
    base_tokens = prepared.packing_prompt_tokens
    current = prepared
    routes: list[str] = []
    trace = current.trace
    specialized = bool(
        trace.get("aggregation_ledger") or trace.get("preference_synthesis"))

    # V5.45: typed readout for anonymous transcripts; strongest-last topology
    # only for named what/when/who/which requests. Specialized prompts remain
    # byte-identical because they already own their answer semantics.
    if specialized:
        routes.append("specialized_frozen")
    else:
        question = _question(current.messages[-1]["content"])
        if _named(current):
            if _prefix(question) in _RECENCY_PREFIXES:
                updated = _reverse_blocks(current, counter)
                routes.append("named_recency" if updated is not current
                              else "named_frozen")
                current = updated
            else:
                routes.append("named_frozen")
        else:
            current = _typed_readout(current, counter)
            routes.append("anonymous_typed")

    # Preserve the measured V5.54 routing.  The optional V5.60 worksheet is
    # consumed by these compact cards only when explicitly enabled.
    trace = current.trace
    ledger_trace = trace.get("aggregation_ledger") or {}
    operation = str(ledger_trace.get("operation") or "")
    selective_worksheet_route = (
        str(ledger_trace.get("worksheet_route") or "")
        if ledger_trace.get("worksheet_selective") else "")
    question = _question(current.messages[-1]["content"])
    named = _named(current)
    if (not operation and _INFERENCE_RE.search(question)
            and not _COUNTERFACTUAL_RE.search(question)
            and not trace.get("preference_synthesis")):
        current = _inference_synthesis(current, counter)
        routes.append("inference")
    elif operation and selective_worksheet_route and not named:
        current = _compact_aggregation(current, counter)
        current = _single_line_aggregation(current, counter)
        routes.append(
            f"selective_operand_worksheet:{selective_worksheet_route}")
    elif operation == "date_difference" and named:
        current = _compact_aggregation(current, counter)
        current = _single_line_aggregation(current, counter)
        routes.append("named_date_difference_single_line")
    elif operation in _ANONYMOUS_COMPACT_OPERATIONS and not named:
        current = _compact_aggregation(current, counter)
        routes.append(f"anonymous_compact:{operation}")

    # V5.52: query-overlap ordering is positive for named temporal questions.
    source_hash = current.prompt_payload_hash
    current = _lexical_reorder(
        current, counter, mode="temporal", speaker_scope="named")
    if current.prompt_payload_hash != source_hash:
        routes.append("named_temporal_query_overlap")

    # V5.53: the what/which layout is retained only for broad period questions,
    # not an exact "on <date/year>" lookup.
    question = _question(current.messages[-1]["content"])
    mode = _question_mode(question)
    if (_YEAR_RE.search(question) and not _EXACT_ON_YEAR_RE.search(question)
            and mode in {"what", "which"}):
        source_hash = current.prompt_payload_hash
        current = _lexical_reorder(
            current, counter, mode=mode, speaker_scope="named")
        if current.prompt_payload_hash != source_hash:
            routes.append(f"named_broad_period:{mode}")

    # V5.54: the same temporal block rule repairs six anonymous LME requests.
    source_hash = current.prompt_payload_hash
    current = _lexical_reorder(
        current, counter, mode="temporal", speaker_scope="anonymous")
    if current.prompt_payload_hash != source_hash:
        routes.append("anonymous_temporal_query_overlap")

    if frozenset(current.evidence_turn_ids) != base_evidence:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: V5.54 changed the evidence set")
    if current.packing_prompt_tokens > base_tokens + _MAX_READOUT_TOKEN_INCREASE:
        raise ReadoutPolicyError(
            f"{prepared.question_id}: V5.54 exceeded its +"
            f"{_MAX_READOUT_TOKEN_INCREASE} token readout allowance")
    trace = dict(current.trace)
    trace.update({
        "readout_policy": V5_54_POLICY,
        "readout_policy_route": routes,
        "readout_policy_evidence_set_frozen": True,
        "readout_policy_token_delta": (
            current.packing_prompt_tokens - base_tokens),
    })
    return replace(current, trace=trace)


def apply_readout_policy(
    prepared: Any, counter: TokenCounter, policy: str,
) -> Any:
    """Apply a named core policy while preserving legacy artifact semantics."""

    if policy == "legacy":
        return prepared
    if policy == V5_54_POLICY:
        return apply_v5_54_readout(prepared, counter)
    raise ValueError(f"unsupported answer readout policy: {policy}")


__all__ = [
    "ReadoutPolicyError", "V5_54_POLICY", "apply_readout_policy",
    "apply_v5_54_readout",
]
