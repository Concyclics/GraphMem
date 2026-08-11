"""The answer contract.

The prompt text is frozen behind ``PROMPT_HASH``.  Every ablation arm compares
answers produced under one prompt; editing this file mid-study silently
invalidates every arm that ran before the edit, so the hash is asserted into
each run manifest.

The contract is benchmark-neutral: it names no dataset, no question type
taxonomy and no gold field.  ``graphmem.answer`` must stay importable without
``graphmem.eval``.
"""
from __future__ import annotations

import hashlib
import re

PROMPT_VERSION = "graphmem-v5.6-answer-v1"

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied conversation memories. "
    "Prefer exact statements in the original turns over summaries. Keep speaker, "
    "entity, action, polarity, completion status, units, and dates exact. "
    "For counts, totals, comparisons, updates, or event order, use all relevant "
    "memories, deduplicate repeated mentions, and compute the requested result. "
    "When asked how many more are needed, remain, or must be earned, compute "
    "target minus the latest current amount; do not return the target itself. "
    "For current quantities, apply additions, removals, cancellations, and "
    "replacements to distinct named items before counting active items. "
    "For most/frequency questions, count completed occurrences, expand explicit "
    "multiplicities such as 'each way' or 'per batch', and exclude plans or "
    "recommendations. "
    "For previous-versus-current questions, order states by their evidence dates "
    "and report the older state before the latest state. "
    "Resolve relative dates from each memory's conversation date. Observation "
    "dates anchor relative expressions but are not necessarily event dates. If "
    "the memories state that an item was used in or brought to an event, treat "
    "its possession or arrival as preceding that event even if both were "
    "discussed later. "
    "If the exact requested entity or relation was never mentioned, say so and "
    "name the near-match only when useful. "
    "A computed candidate shown under 'Candidate answer' is a fallible "
    "mechanical proposal, not evidence: use its value only when the cited "
    "memories support every part of it, and otherwise ignore it. "
    "Return only the concise final answer."
)

SOURCE_TIME_PROMPT_VERSION = "graphmem-v5.12-answer-source-time-v1"
SOURCE_TIME_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT.replace(
    "Resolve relative dates from each memory's conversation date. Observation ",
    "Resolve relative dates inside each memory from that memory's conversation "
    "date, never from the question date. The question date anchors only relative "
    "phrases in the question itself. A [source-time ...] annotation is the "
    "deterministic resolution of a memory phrase and must be used as written. Observation ",
)

# Query-side precision contract for wide graph reservoirs.  This is an opt-in
# prompt rather than a replacement for the frozen V5.6/V5.12 contracts: old
# artifacts must remain replayable under their original prompt hashes.
GROUNDED_PROMPT_VERSION = "graphmem-v5.20-grounded-answer-v1"
GROUNDED_PROMPT_APPENDIX = (
    " First isolate the smallest set of memories that directly supports the "
    "requested answer; ignore same-topic memories that concern another entity, "
    "another event, or a different state. Do not treat an assistant suggestion, "
    "hypothetical example, unaccepted plan, or image caption alone as proof that "
    "the user performed, owned, preferred, or completed something. Prefer an "
    "explicit first-person statement or a clearly confirmed fact. Exact wording "
    "of the requested relation is not required when the statement directly "
    "entails it, but topical similarity alone is never sufficient. For a count "
    "or list, silently form a ledger of distinct relevant events or items, keep "
    "separate recurring occurrences such as two named weekdays, and remove only "
    "duplicate mentions of the same occurrence. For a duration, date, or order, "
    "silently identify each endpoint and its date before calculating or sorting. "
    "When facts conflict, use the newest explicit update for a current-state "
    "question and the event-time fact for a historical question. Before answering, "
    "verify the final entity, number, unit, polarity, and date against the direct "
    "evidence. Do not expose the ledger, evidence review, or calculation steps. "
    "Return the final answer once, in one or at most two concise sentences."
)
GROUNDED_OUTPUT_CONTRACT = (
    "Output contract: Return only the final answer in one or at most two concise "
    "sentences. Do not quote or list evidence, show calculations, add a rationale, "
    "or repeat the conclusion."
)
TOPOLOGICAL_LAYOUT_VERSION = "graphmem-v5.20-topological-evidence-v1"
TOPOLOGICAL_LAYOUT_APPENDIX = (
    " Evidence is arranged into graph-derived blocks. [CHAIN k step=d] lines "
    "belong to one QueryIR operand and follow its root-to-leaf relation path; "
    "[CHAIN k support] lines are nearby evidence for that operand. "
    "[GRAPH g step=d] lines share a graph traversal branch even when QueryIR "
    "did not bind them to an operand. [AUX] lines are unbound supporting "
    "context grouped in a small window around a high-relevance anchor. All "
    "blocks are ordered by their best query-relevance rank before their "
    "internal graph/temporal order. These labels are navigation hints, not "
    "facts or confidence guarantees: verify the exact entity, event, status, "
    "number, and date in the memory text. Read all blocks needed by the "
    "question, but do not combine facts merely because they share a block."
)
COMPACT_TOPOLOGICAL_LAYOUT_VERSION = (
    "graphmem-v5.38-compact-topological-evidence-v1")
COMPACT_TOPOLOGICAL_LAYOUT_APPENDIX = (
    " Memories are graph-grouped: [CHAIN] follows one QueryIR path, [GRAPH] "
    "one traversal branch, and [AUX] contextual support. Blocks are ranked by "
    "relevance, then graph or time order. Labels are navigation hints only: "
    "verify the facts and do not merge memories only because they share a block."
)
COMPACT_TOPOLOGICAL_LABELS_VERSION = (
    "graphmem-v5.39-compact-topological-labels-v1")
COMPACT_TOPOLOGICAL_LABELS_APPENDIX = (
    " Memories are graph-grouped: [Ck.d] is QueryIR chain k at step d "
    "([Ck.S] is nearby support), [Gk.d] is graph branch k at step d, and "
    "[Ak.r] is auxiliary group k at retrieval rank r. Blocks are ranked by "
    "relevance, then graph or time order. Labels are navigation hints only: "
    "verify the facts and do not merge memories only because they share a block."
)
QUERY_FOCUS_INDEX_VERSION = "graphmem-v5.39-query-focus-index-v1"
QUERY_FOCUS_INDEX_APPENDIX = (
    " A Query focus index after the memories repeats short, exact excerpts "
    "from turns already present in the packed evidence. It is a reading aid, "
    "not extra evidence or a proposed answer; verify each excerpt in its full "
    "memory block."
)
AGGREGATION_LEDGER_VERSION = "graphmem-v5.21-aggregation-ledger-v1"
AGGREGATION_LEDGER_APPENDIX = (
    " For an aggregation question, the user message may contain an Aggregation "
    "ledger after the memories. It is a deterministic index over already cited "
    "source turns, not additional evidence and not a certified answer. Select "
    "only operands satisfying the exact entity, event, time, polarity, and "
    "completion constraints. Deduplicate mentions of one occurrence but retain "
    "distinct occurrences. Never infer numeric zero from a missing operand. "
    "When the ledger says its result is unavailable, perform the named arithmetic "
    "yourself only after verifying the complete operand set in the source turns."
)
PREFERENCE_SYNTHESIS_VERSION = "graphmem-v5.21-preference-synthesis-v1"
PREFERENCE_SYNTHESIS_APPENDIX = (
    " This is a preference, advice, or recommendation request. Treat the "
    "memories as grounded constraints and examples, not as a requirement that "
    "the final suggestion must already appear verbatim. You may synthesize a "
    "new recommendation that is compatible with the user's demonstrated "
    "preferences, possessions, habits, goals, and negative constraints. Do not "
    "invent a user preference or personal fact, and do not claim the user has "
    "already tried or owns a newly suggested item. Prefer a specific, useful "
    "answer over abstaining merely because no memory states the recommendation "
    "itself. Keep the connection to the grounded constraints concise."
)
EXACT_GROUNDING_FOOTER_VERSION = "graphmem-v5.23-exact-grounding-footer-v1"
EXACT_GROUNDING_FOOTER = (
    "Final check: require exact entity and relation matches; if any required "
    "fact is unsupported, answer insufficient information. Return only the answer."
)
QUESTION_RECENCY_FOOTER_VERSION = "graphmem-v5.38-question-recency-footer-v1"
CONTEXTUAL_QUESTION_DATE_VERSION = "graphmem-v5.38-contextual-question-date-v1"
_QUESTION_DATE_CUE_RE = re.compile(
    r"\b(?:ago|today|yesterday|tomorrow|now|currently|current|recently|"
    r"this\s+(?:day|week|month|year)|past\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|years?)|(?:last|next|previous)\s+"
    r"(?:day|week|month|year)|as\s+of|so\s+far)\b",
    re.I,
)

_PREFERENCE_QUERY_RE = re.compile(
    r"\b(?:can|could|would) you (?:recommend|suggest|give|help)\b|"
    r"\bdo you have (?:any|some) (?:\w+\s+){0,3}"
    r"(?:tips?|advice|suggestions?|recommendations?|ideas?)\b|"
    r"\b(?:any|some) (?:\w+\s+){0,3}"
    r"(?:tips?|advice|suggestions?|recommendations?|ideas?)\b|"
    r"\bwhat (?:should|could) i\b|\bwhat do you think\b|"
    r"\bdo you think\b|\bcould there be a reason\b",
    re.I,
)


def is_preference_synthesis_query(question: str) -> bool:
    """Detect an advice request from its wording, without benchmark labels."""

    return bool(_PREFERENCE_QUERY_RE.search(" ".join(question.split())))


def question_needs_global_date(question: str) -> bool:
    """Whether a relative phrase in the question needs the question timestamp.

    A source turn's relative expressions are already normalized from its own
    timestamp.  Showing the later global question date for every lookup makes
    long-context models re-anchor source phrases such as ``next month`` to the
    wrong conversation.  Keep the global anchor only when the query itself is
    deictic.
    """

    return bool(_QUESTION_DATE_CUE_RE.search(" ".join(question.split())))


def _query_operation_contract(question: str) -> str:
    """Name the arithmetic contract when the question form fixes it.

    Kept from V3 because 'how many more' questions are answered with the target
    rather than the delta often enough to be worth stating explicitly.
    """
    normalized = " ".join(question.casefold().split())
    if (re.search(r"\bhow many more\b|\bhow many\b.*\b(?:remain|remaining|left)\b", normalized)
            or re.search(r"\bhow many\b.*\bneed(?:ed)?\b.*\b(?:earn|gain|add|save|accumulate|reach)\b",
                         normalized)):
        return "scalar_delta: answer = target amount - latest current amount"
    return "derive the exact answer form requested by the question"


def build_answer_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_text: str,
    candidate_answer: str | None = None,
    normalize_relative_time: bool = False,
    precision_grounding: bool = False,
    topological_layout: bool = False,
    aggregation_ledger: str | None = None,
    aggregation_ledger_contract: bool = False,
    preference_synthesis: bool = False,
    exact_grounding_footer: bool = False,
    include_question_date: bool = True,
    question_recency_footer: bool = False,
    compact_topological_contract: bool = False,
    compact_topological_labels: bool = False,
    query_focus_index: str | None = None,
) -> list[dict[str, str]]:
    sections = ([f"Question date: {question_date or 'unknown'}"]
                if include_question_date else []) + [
        f"Question: {question}",
        "",
        f"Query operation: {_query_operation_contract(question)}",
    ]
    if candidate_answer:
        sections += ["", f"Candidate answer (unverified proposal): {candidate_answer}"]
    sections += ["", "Conversation memories:", evidence_text]
    if query_focus_index:
        sections += ["", query_focus_index]
    if aggregation_ledger:
        # Recency is intentional: aggregation errors persisted when all gold
        # turns were packed but their operands were scattered through 64 turns.
        sections += ["", aggregation_ledger]
    if precision_grounding:
        # Repeat the short format contract *after* the long evidence block.  A
        # system-only instruction before 5K-12K evidence was often ignored by
        # Qwen, which then emitted evidence lists or degenerated into repetition.
        sections += ["", GROUNDED_OUTPUT_CONTRACT]
    if exact_grounding_footer and not preference_synthesis:
        sections += ["", EXACT_GROUNDING_FOOTER]
    if question_recency_footer:
        sections += [
            "",
            f"Answer the original Question now: {question}",
            "Resolve relative time inside each memory only from that memory's "
            "own date or [source-time] annotation. Return the concise answer "
            "once; do not quote evidence or repeat the conclusion.",
        ]
    return [
        {"role": "system", "content": prompt_contract(
            normalize_relative_time, precision_grounding,
            topological_layout,
            aggregation_ledger_contract or bool(aggregation_ledger),
            preference_synthesis, exact_grounding_footer,
            not include_question_date, question_recency_footer,
            compact_topological_contract, compact_topological_labels,
            bool(query_focus_index))[1]},
        {"role": "user", "content": "\n".join(sections)},
    ]


PROMPT_HASH = hashlib.sha256(
    (PROMPT_VERSION + ANSWER_SYSTEM_PROMPT).encode("utf-8")
).hexdigest()


def prompt_contract(normalize_relative_time: bool = False,
                    precision_grounding: bool = False,
                    topological_layout: bool = False,
                    aggregation_ledger: bool = False,
                    preference_synthesis: bool = False,
                    exact_grounding_footer: bool = False,
                    contextual_question_date: bool = False,
                    question_recency_footer: bool = False,
                    compact_topological_contract: bool = False,
                    compact_topological_labels: bool = False,
                    query_focus_index: bool = False,
                    ) -> tuple[str, str, str]:
    """Return the exact version/text/hash for an answer configuration."""

    version, prompt = (
        (SOURCE_TIME_PROMPT_VERSION, SOURCE_TIME_SYSTEM_PROMPT)
        if normalize_relative_time else (PROMPT_VERSION, ANSWER_SYSTEM_PROMPT))
    if precision_grounding:
        version = (GROUNDED_PROMPT_VERSION
                   + ("-source-time" if normalize_relative_time else ""))
        prompt += GROUNDED_PROMPT_APPENDIX
    if topological_layout:
        if compact_topological_labels:
            version += "+" + COMPACT_TOPOLOGICAL_LABELS_VERSION
            prompt += COMPACT_TOPOLOGICAL_LABELS_APPENDIX
        elif compact_topological_contract:
            version += "+" + COMPACT_TOPOLOGICAL_LAYOUT_VERSION
            prompt += COMPACT_TOPOLOGICAL_LAYOUT_APPENDIX
        else:
            version += "+" + TOPOLOGICAL_LAYOUT_VERSION
            prompt += TOPOLOGICAL_LAYOUT_APPENDIX
    if aggregation_ledger:
        version += "+" + AGGREGATION_LEDGER_VERSION
        prompt += AGGREGATION_LEDGER_APPENDIX
    if preference_synthesis:
        version += "+" + PREFERENCE_SYNTHESIS_VERSION
        prompt += PREFERENCE_SYNTHESIS_APPENDIX
    if exact_grounding_footer:
        # The text lives after evidence in the user message for recency; adding
        # the version here still makes the prompt-contract hash configuration
        # specific and prevents incompatible artifacts from being merged.
        version += "+" + EXACT_GROUNDING_FOOTER_VERSION
    if contextual_question_date:
        version += "+" + CONTEXTUAL_QUESTION_DATE_VERSION
    if question_recency_footer:
        version += "+" + QUESTION_RECENCY_FOOTER_VERSION
    if query_focus_index:
        version += "+" + QUERY_FOCUS_INDEX_VERSION
        prompt += QUERY_FOCUS_INDEX_APPENDIX
    return version, prompt, hashlib.sha256((version + prompt).encode("utf-8")).hexdigest()
