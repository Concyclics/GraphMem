"""Prompt-only typed evidence readout for frozen ``PreparedAnswer`` rows.

This module deliberately does not inspect gold answers, predictions, or judge
labels.  It converts the already-packed evidence into a numbered view, asks an
answer model to compile a small source-grounded workspace, and renders a second
prompt that independently checks a fallible baseline answer against that
workspace and the cited source turns.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


EVIDENCE_START = re.compile(r"(?=\[(?:AUX|CHAIN|GRAPH)\s)")
EVIDENCE_ID = re.compile(r"\bE(\d{1,3})\b", re.I)
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}")
STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "could",
    "does", "from", "have", "into", "might", "other", "their", "there",
    "these", "they", "this", "what", "when", "where", "which", "with",
    "would", "your", "were", "been", "that", "than", "some", "many",
}


READOUT_SYSTEM = """You are a source-grounded evidence compiler, not the final
answerer. Inspect every numbered memory before deciding that evidence is
missing. Bind facts to the exact subject, relation, object, event, polarity,
completion status, and time requested by the question. Same-topic facts about
a different person, object, or event are distractors. Return one JSON object
only, with this schema:
{
  "query": {"subject": "", "relation": "", "answer_type": "", "time_scope": "", "operation": ""},
  "witnesses": [{"evidence_id": "E01", "subject": "", "relation": "", "value": "", "event_time": "", "status": "", "support": "brief source-grounded excerpt"}],
  "rejected_candidates": [{"evidence_id": "E02", "reason": "wrong subject/object/event/time/status"}],
  "computation": {"distinct_items": [], "formula": "", "result": ""},
  "missing_slots": [],
  "confidence": "high|medium|low"
}
Never invent an evidence ID or use the fallible baseline answer as evidence."""


ROUTE_READOUT = {
    "count": (
        "Enumerate one witness per distinct matching completed event or active "
        "item. Exclude plans, suggestions, negations, cancellations, and repeated "
        "mentions. Show the deduplicated items and arithmetic in computation."
    ),
    "temporal": (
        "Separate event time from observation/conversation time. Resolve explicit "
        "source-time annotations as written, compare all matching candidates, and "
        "retain both endpoints for durations or ordering."
    ),
    "multi_hop": (
        "Build a connected witness chain and fill every required operand. Do not "
        "bridge two memories merely because they share a topic or entity name."
    ),
    "preference": (
        "Extract the user's own preferences, habits, constraints, and prior "
        "successes. Distinguish them from assistant suggestions. A useful response "
        "may personalize a recommendation without a previously named product."
    ),
    "inference": (
        "Separate directly stated facts from a minimal supported inference. Include "
        "the most specific source facts needed for the inference and avoid "
        "unnecessary abstention when those facts identify a likely answer."
    ),
    "lookup": (
        "Prefer the most direct exact witness. Reject near-matches that change the "
        "speaker, owner, relation, object, event, or requested attribute."
    ),
}


ANSWER_SYSTEM = """You are the final evidence verifier. Derive the answer from
the typed workspace and cited source memories before comparing it with the
fallible baseline. The baseline and workspace are proposals, not evidence.
Use an answer only when the cited memories support the exact subject, relation,
object, event, polarity, status, and time. If a direct or derivable witness is
present, do not answer "insufficient information". Reject same-topic
near-matches. Return only the concise final answer, with no explanation or
citations."""


ROUTE_ANSWER = {
    "count": (
        "Internally enumerate distinct matching completed events/items, apply "
        "updates and removals, deduplicate repeated mentions, then return the count."
    ),
    "temporal": (
        "Use event/source-time rather than mention time. Check the requested event "
        "and all endpoints before returning a date, duration, or ordering."
    ),
    "multi_hop": (
        "Verify every operand and intermediate relation in one connected chain; "
        "do not answer from one locally similar fact."
    ),
    "preference": (
        "Personalize using the user's demonstrated preferences and constraints. "
        "Do not require a previously named product unless the question asks for it."
    ),
    "inference": (
        "Make the narrowest reasonable inference supported by the cited facts. "
        "Minimal common knowledge is allowed, but invented conversational facts are not."
    ),
    "lookup": (
        "Return the value bound to the exact queried subject and relation; a value "
        "from another person or event is wrong even when its wording is similar."
    ),
}


AUDIT_SYSTEM = """You are an adversarial evidence executor. The supplied typed
readout is an untrusted draft: it may stop early, overlook a later witness,
bind a fact to the wrong person or event, or perform incomplete arithmetic.
Rescan every numbered source memory before accepting or correcting it. Evidence
rank, graph role, repetition, and the draft's confidence are not proof.

Return exactly one JSON object with this schema:
{
  "candidate_ledger": [{"evidence_id": "E01", "subject": "", "relation": "", "value": "", "event_time": "", "status": "", "decision": "include|exclude", "reason": ""}],
  "corrections": ["brief correction to the draft"],
  "operation": {"operands": [], "formula": "", "result": ""},
  "missing_slots": [],
  "final_answer": "concise answer string",
  "confidence": "high|medium|low"
}

The final answer must answer the requested field, not restate the question or
the audit. Include all coordinated values when the question asks for a list.
Never invent an evidence ID. Do not treat the draft or a previous answer as
source evidence."""


ROUTE_AUDIT = {
    "count": (
        "Build an exhaustive ledger from all memories, including plausible items "
        "that are ultimately excluded. Normalize aliases, deduplicate repeated "
        "mentions, respect completion/cancellation and the requested time window, "
        "then recompute the count, sum, maximum, or minimum from explicit operands."
    ),
    "temporal": (
        "Enumerate every candidate event and its source/event time. Resolve each "
        "relative expression from its own source date, enforce first/last/before/"
        "after constraints, and show the exact date or duration operands."
    ),
    "multi_hop": (
        "Write one operand per required hop, identify a source witness for each, "
        "and verify that subjects and objects join into one connected chain. Scan "
        "for alternate witnesses before declaring a slot missing."
    ),
    "preference": (
        "Collect the user's own positive and negative preferences, constraints, "
        "and demonstrated choices across all memories, then give the narrowest "
        "personalized answer the question requests."
    ),
    "inference": (
        "Aggregate all directly stated traits relevant to the inference. Prefer a "
        "specific, minimally inferred answer supported by those traits, and include "
        "compatible alternatives when the question permits more than one."
    ),
    "lookup": (
        "Search all memories for exact and paraphrased forms of the requested "
        "relation. Check attribution, ownership, object, event, and requested "
        "attribute; include every explicitly requested coordinated value."
    ),
}


def route_question(question: str, stratum: str = "") -> str:
    """Choose a gold-independent readout route from question text and metadata."""

    text = " ".join(question.casefold().split())
    kind = stratum.casefold()
    if re.search(
        r"\b(how many|how much|how often|number of|total|times has|times did)\b",
        text,
    ):
        return "count"
    if "preference" in kind or re.search(
        r"\b(recommend|suggest|would i (?:like|prefer)|tips on what)\b", text
    ):
        return "preference"
    if "locomo_cat3" in kind or re.search(
        r"\b(might|likely|potentially|would .* benefit|can be inferred)\b", text
    ):
        return "inference"
    if "temporal" in kind or "locomo_cat2" in kind or re.search(
        r"\b(when|what date|which month|how long|first|last|before|after|during)\b",
        text,
    ):
        return "temporal"
    if "multi_session" in kind or "multi-session" in kind or "locomo_cat1" in kind:
        return "multi_hop"
    return "lookup"


def extract_evidence_blocks(messages: Sequence[Mapping[str, str]]) -> list[str]:
    """Extract the rendered evidence turns without carrying answer instructions."""

    user = "\n".join(
        str(row.get("content") or "") for row in messages
        if str(row.get("role") or "") == "user"
    )
    marker = "Conversation memories:"
    if marker not in user:
        return []
    body = user.split(marker, 1)[1]
    for end_marker in (
        "\n\nAnswer the original Question now:",
        "\nAnswer the original Question now:",
    ):
        if end_marker in body:
            body = body.split(end_marker, 1)[0]
            break
    parts = EVIDENCE_START.split(body)
    return [
        " ".join(part.strip().split())
        for part in parts
        if re.match(r"^\[(?:AUX|CHAIN|GRAPH)\s", part.strip())
    ]


def numbered_evidence(blocks: Sequence[str]) -> list[str]:
    return [f"[E{index:02d}] {block}" for index, block in enumerate(blocks, 1)]


def make_readout_messages(
    *, question: str, question_date: str, route: str,
    evidence_blocks: Sequence[str],
) -> list[dict[str, str]]:
    route_instruction = ROUTE_READOUT.get(route, ROUTE_READOUT["lookup"])
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n"
        f"Execution route: {route}\n"
        f"Route contract: {route_instruction}\n\n"
        "Numbered conversation memories:\n"
        + "\n".join(numbered_evidence(evidence_blocks))
    )
    return [
        {"role": "system", "content": READOUT_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _cited_indices(value: Any) -> list[int]:
    result: list[int] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"witnesses", "evidence_id", "source_evidence_ids"}:
                result.extend(_cited_indices(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            result.extend(_cited_indices(item))
    elif isinstance(value, str):
        result.extend(int(match.group(1)) for match in EVIDENCE_ID.finditer(value))
    return result


def select_compact_evidence(
    *, question: str, readout: Mapping[str, Any], evidence_blocks: Sequence[str],
    max_blocks: int = 20, fallback_blocks: int = 6,
) -> list[tuple[int, str]]:
    """Keep cited witnesses, lexical matches, then a small rank-order fallback."""

    if max_blocks <= 0:
        return []
    selected: list[int] = []
    for one_based in _cited_indices(readout):
        index = one_based - 1
        if 0 <= index < len(evidence_blocks) and index not in selected:
            selected.append(index)

    terms = {
        token.casefold() for token in WORD.findall(question)
        if len(token) >= 4 and token.casefold() not in STOPWORDS
    }
    lexical = sorted(
        range(len(evidence_blocks)),
        key=lambda index: (
            -sum(term in evidence_blocks[index].casefold() for term in terms),
            index,
        ),
    )
    for index in lexical:
        if index not in selected:
            selected.append(index)
        if len(selected) >= max_blocks:
            break
    for index in range(min(fallback_blocks, len(evidence_blocks))):
        if index not in selected:
            selected.append(index)
        if len(selected) >= max_blocks:
            break
    return [(index + 1, evidence_blocks[index]) for index in selected[:max_blocks]]


def make_answer_messages(
    *, question: str, question_date: str, route: str, baseline: str | None,
    readout_text: str, compact_evidence: Sequence[tuple[int, str]],
) -> list[dict[str, str]]:
    route_instruction = ROUTE_ANSWER.get(route, ROUTE_ANSWER["lookup"])
    evidence = "\n".join(
        f"[E{index:02d}] {block}" for index, block in compact_evidence
    ) or "(no source memory selected; verify that evidence is genuinely missing)"
    baseline_section = (
        f"Fallible baseline answer:\n{baseline}\n\n"
        if baseline is not None
        else "No baseline answer is supplied. Derive the result independently.\n\n"
    )
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n"
        f"Execution route: {route}\n"
        f"Route contract: {route_instruction}\n\n"
        f"{baseline_section}"
        f"Typed readout proposal:\n{readout_text}\n\n"
        f"Cited and query-matched source memories:\n{evidence}\n\n"
        "Independently derive and verify the answer now. Return only the concise final answer."
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_audit_messages(
    *, question: str, question_date: str, route: str, baseline: str | None,
    readout_text: str, evidence_blocks: Sequence[tuple[int, str]],
) -> list[dict[str, str]]:
    """Build an exhaustive second-pass audit over a frozen evidence set."""

    route_instruction = ROUTE_AUDIT.get(route, ROUTE_AUDIT["lookup"])
    evidence = "\n".join(
        f"[E{index:02d}] {block}" for index, block in evidence_blocks
    ) or "(no source memory was supplied)"
    baseline_section = (
        f"Fallible previous answer (not evidence):\n{baseline}\n\n"
        if baseline is not None else ""
    )
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n"
        f"Execution route: {route}\n"
        f"Adversarial route contract: {route_instruction}\n\n"
        f"{baseline_section}"
        f"Untrusted first-pass readout:\n{readout_text}\n\n"
        f"Complete frozen source-memory set:\n{evidence}\n\n"
        "Audit the draft against every source memory and emit the JSON object now."
    )
    return [
        {"role": "system", "content": AUDIT_SYSTEM},
        {"role": "user", "content": user},
    ]
