"""Diverse local answer families and a source-grounded candidate selector.

Two four-sample requests share the exact frozen evidence but use complementary
readout contracts.  The verifier may select an emitted candidate; it may not
synthesize a ninth answer.  No function in this module reads benchmark labels,
gold answers, judge verdicts, or a previous baseline prediction.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from ..domain import canonical_json


ENSEMBLE_SCHEMA_VERSION = "graphmem-v5.73-split-prompt-ensemble-v1"
FAMILIES = ("extractive", "audit")

_QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
_COUNT_RE = re.compile(
    r"\b(?:how many|how much|number of|count|total|average|mean|minimum|"
    r"maximum|most|least|each|per)\b", re.I)
_TEMPORAL_RE = re.compile(
    r"\b(?:when|what date|which date|day|week|month|year|how long|before|"
    r"after|first|last|latest|earliest|ago|during)\b", re.I)
_INFERENCE_RE = re.compile(
    r"\b(?:would|might|likely|could|infer(?:red)?|underlying|personality|"
    r"condition|benefit from|suggests?)\b", re.I)
_PREFERENCE_RE = re.compile(
    r"\b(?:recommend|suggest|advice|tips|would i (?:like|prefer))\b", re.I)

_FAMILY_SYSTEM = {
    "extractive": (
        "\n\nIndependent readout A (source binding): Start from the Answer-critical "
        "evidence card, then verify it against all supplied memories. Bind the "
        "exact subject, relation, value, status, and event/source-time. Reject "
        "same-topic near-matches. For a multi-hop question, require one cited "
        "source fact for every join. Produce only one concise final answer; do "
        "not describe the search or quote evidence."
    ),
    "audit": (
        "\n\nIndependent readout B (adversarial execution): Rescan all supplied "
        "memories before answering. Actively test the most plausible answer "
        "against wrong-person, wrong-relation, wrong-event, observation-time, "
        "planned-versus-completed, duplicate-event, and omitted-operand errors. "
        "For counts/lists, construct the exact qualifying event set internally. "
        "Return only one concise final answer; do not expose the audit."
    ),
}

_ROUTE_SYSTEM = {
    "aggregate": (
        " Internally enumerate every exact-scope operand. Preserve separately "
        "dated occurrences, collapse only repeated mentions of the same event, "
        "exclude plans/cancellations when completion is requested, and compute "
        "exactly once."
    ),
    "temporal": (
        " Separate event/source-time from the observation date. Use [source-time] "
        "as written and bind every endpoint before choosing a date, duration, or "
        "order."
    ),
    "inference": (
        " Bind conversational premises first. Stable ordinary knowledge may name "
        "the narrow concept those premises entail, but may not invent a personal "
        "fact or event."
    ),
    "preference": (
        " Treat user-stated preferences and constraints as grounded inputs to a "
        "useful recommendation; do not require the recommendation to occur "
        "verbatim in memory."
    ),
    "lookup": (
        " Prefer a direct exact witness and return the value attached to the "
        "queried person and relation, not a nearby entity or attribute."
    ),
}


def normalize_answer(text: str) -> str:
    return " ".join(str(text or "").split())


def question_from_messages(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        match = _QUESTION_RE.search(str(message.get("content") or ""))
        if match is not None:
            return " ".join(match.group("question").split())
    raise ValueError("prepared messages do not contain a Question header")


def ensemble_route(question: str) -> str:
    if _COUNT_RE.search(question):
        return "aggregate"
    if _PREFERENCE_RE.search(question):
        return "preference"
    if _INFERENCE_RE.search(question):
        return "inference"
    if _TEMPORAL_RE.search(question):
        return "temporal"
    return "lookup"


def build_ensemble_family_messages(
    messages: Sequence[Mapping[str, str]], family: str,
) -> tuple[dict[str, str], ...]:
    """Return one independent prompt family over unchanged evidence bytes."""

    if family not in FAMILIES:
        raise ValueError(f"unknown answer family: {family}")
    rows = [dict(message) for message in messages]
    if not rows:
        raise ValueError("an ensemble prompt requires non-empty messages")
    question = question_from_messages(rows)
    route = ensemble_route(question)
    system_index = next((index for index, row in enumerate(rows)
                         if row.get("role") == "system"), None)
    if system_index is None:
        rows.insert(0, {"role": "system", "content": ""})
        system_index = 0
    rows[system_index]["content"] = (
        str(rows[system_index].get("content") or "")
        + _FAMILY_SYSTEM[family] + _ROUTE_SYSTEM[route])
    # A short family tag makes the two requests byte-distinct without changing
    # evidence order or presenting any answer candidate.
    rows[-1]["content"] = (
        str(rows[-1].get("content") or "")
        + f"\n\nExecute independent {family} readout for the original Question now.")
    return tuple(rows)


def prompt_payload_hash(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(canonical_json(list(messages)).encode()).hexdigest()


def build_candidate_verifier_messages(
    base_messages: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Ask a local verifier to select, never rewrite, one emitted answer."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    question = question_from_messages(base_messages)
    route = ensemble_route(question)
    rendered = []
    for index, candidate in enumerate(candidates, 1):
        answer = normalize_answer(str(candidate.get("prediction") or ""))
        if not answer:
            raise ValueError(f"candidate {index} is empty")
        support_families = tuple(map(str, candidate.get(
            "support_families", (candidate.get("family", "unknown"),))))
        support_count = int(candidate.get("support_count", 1))
        rendered.append(
            f"C{index} [family={candidate.get('family', 'unknown')}; "
            f"sample={candidate.get('family_rank', candidate.get('rank', index))}; "
            f"support={support_count}; independent_families="
            f"{','.join(support_families)}]: {answer}")
    system = (
        "You are a source-grounded candidate selector. Re-read the supplied "
        "conversation memories and Answer-critical evidence card, then select "
        "the single candidate that best answers the exact question. Check "
        "subject, relation, event, polarity, completion status, event/source-time, "
        "and every count/list operand. Agreement across independently prompted "
        "families is a reliability signal, but repeated samples within one family "
        "and wording confidence are not source evidence. You must not synthesize, "
        "repair, combine, or paraphrase "
        "an answer. Return JSON only: {\"choice\": <1-based integer>, "
        "\"reason_code\": \"exact_witness|complete_operands|correct_time|"
        "supported_inference|least_unsupported\"}."
        + _ROUTE_SYSTEM[route]
    )
    # Preserve the exact base request, then append candidates as an isolated
    # selection task.  A prior baseline prediction is never provided.
    base_system = "\n".join(
        str(message.get("content") or "") for message in base_messages
        if str(message.get("role") or "") == "system")
    rows = [dict(message) for message in base_messages
            if str(message.get("role") or "") != "system"]
    rows.append({
        "role": "user",
        "content": (
            "Candidate answers (untrusted; select one exact string, do not write "
            "a new answer):\n" + "\n".join(rendered)
            + f"\n\nOriginal Question: {question}"
        ),
    })
    rows.insert(0, {
        "role": "system",
        "content": system + ("\n\nBase answer contract:\n" + base_system
                             if base_system else ""),
    })
    return tuple(rows)


def combine_candidate_families(
    direct: Sequence[Mapping[str, Any]],
    structured: Sequence[Mapping[str, Any]],
    *,
    per_family: int = 4,
) -> tuple[dict[str, Any], ...]:
    """Build a deterministic direct/structured 4+4 candidate sequence."""

    if per_family <= 0:
        raise ValueError("per_family must be positive")
    if len(direct) < per_family or len(structured) < per_family:
        raise ValueError("both candidate families must contain per_family rows")
    combined: list[dict[str, Any]] = []
    for family, source in (("direct_v563", direct),
                           ("structured_v573", structured)):
        for family_rank, candidate in enumerate(source[:per_family], 1):
            prediction = normalize_answer(str(candidate.get("prediction") or ""))
            if not prediction:
                raise ValueError(f"{family} candidate {family_rank} is empty")
            digest = hashlib.sha256(prediction.encode()).hexdigest()
            declared = str(candidate.get("prediction_sha256") or digest)
            if declared != digest:
                raise ValueError(
                    f"{family} candidate {family_rank} has an invalid digest")
            combined.append({
                "rank": len(combined) + 1,
                "family": family,
                "family_rank": family_rank,
                "prediction": prediction,
                "prediction_sha256": digest,
                "finish_reason": str(candidate.get("finish_reason") or ""),
                "source_family": str(candidate.get("family") or "direct"),
                "source_rank": int(candidate.get("rank", family_rank)),
            })
    return tuple(combined)


def cross_family_consensus_choice(
    candidates: Sequence[Mapping[str, Any]],
) -> int | None:
    """Return the first index of the strongest independently shared answer."""

    groups: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        digest = str(candidate.get("prediction_sha256") or "")
        if not digest:
            prediction = normalize_answer(str(candidate.get("prediction") or ""))
            if not prediction:
                continue
            digest = hashlib.sha256(prediction.encode()).hexdigest()
        group = groups.setdefault(digest, {
            "first": index, "count": 0, "families": set()})
        group["count"] += 1
        group["families"].add(str(candidate.get("family") or "unknown"))
    shared = [group for group in groups.values()
              if len(group["families"]) >= 2]
    if not shared:
        return None
    winner = max(shared, key=lambda group: (
        len(group["families"]), group["count"], -group["first"]))
    return int(winner["first"])


def parse_verifier_choice(text: str, candidate_count: int) -> int | None:
    """Return a zero-based candidate index from strict JSON or a safe fallback."""

    raw = str(text or "").strip()
    payload: Any = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.S)
        if match is not None:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    choice = payload.get("choice") if isinstance(payload, Mapping) else None
    try:
        index = int(choice) - 1
    except (TypeError, ValueError):
        match = re.search(r"\bC?(\d{1,2})\b", raw)
        index = int(match.group(1)) - 1 if match else -1
    return index if 0 <= index < candidate_count else None


def deterministic_candidate_fallback(
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    """Choose the modal normalized answer, breaking ties by original order."""

    normalized = [normalize_answer(str(row.get("prediction") or ""))
                  for row in candidates]
    counts = Counter(normalized)
    winner = max(counts, key=lambda value: (counts[value], -normalized.index(value)))
    return normalized.index(winner)
