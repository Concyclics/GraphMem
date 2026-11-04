"""Frozen reader constants from legacy_graphmem_answer_prompts.py."""

import re
from datetime import datetime

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied conversation memories. "
    "Prefer exact statements in the original turns over summaries. Keep speaker, "
    "entity, action, polarity, completion status, units, and dates exact. "
    "For counts, totals, comparisons, updates, or event order, use all relevant "
    "memories, deduplicate repeated mentions, and compute the requested result. "
    "When asked how many more are needed, remain, or must be earned, compute "
    "target minus the latest current amount; do not return the target itself. "
    "For an each-item or per-item cost, divide the exact aggregate price by the "
    "matching distinct item count once; do not add the price and count. "
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

SOURCE_TIME_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT.replace(
    "Resolve relative dates from each memory's conversation date. Observation ",
    "Resolve relative dates inside each memory from that memory's conversation "
    "date, never from the question date. The question date anchors only relative "
    "phrases in the question itself. A [source-time ...] annotation is the "
    "deterministic resolution of a memory phrase and must be used as written. Observation ",
)

PREFERENCE_SYNTHESIS_APPENDIX = (
    " This is a preference, advice, or recommendation request. Treat the "
    "memories as grounded constraints and examples, not as a requirement that "
    "the final suggestion must already appear verbatim. You may synthesize a "
    "new recommendation that is compatible with the user's demonstrated "
    "preferences, possessions, habits, goals, and negative constraints. First "
    "identify the exact user-stated ingredients, equipment, possessions, "
    "or constraints in the memories and visibly use them in the recommendation; "
    "do not claim that personal context is missing when it is present. Do not "
    "invent a user preference or personal fact, and do not claim the user has "
    "already tried or owns a newly suggested item. Prefer a specific, useful "
    "answer over abstaining merely because no memory states the recommendation "
    "itself. Keep the connection to the grounded constraints concise."
)

_QUESTION_DATE_CUE_RE = re.compile(
    r"\b(?:ago|today|yesterday|tomorrow|now|currently|current|recently|"
    r"this\s+(?:day|week|month|year)|past\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|years?)|(?:last|next|previous)\s+"
    r"(?:day|week|month|year)|as\s+of|so\s+far)\b",
    re.I,
)


def question_needs_global_date(question: str) -> bool:
    """Whether a relative phrase in the question needs the question timestamp.

    A source turn's relative expressions are already normalized from its own
    timestamp.  Showing the later global question date for every lookup makes
    long-context models re-anchor source phrases such as ``next month`` to the
    wrong conversation.  Keep the global anchor only when the query itself is
    deictic.
    """

    return bool(_QUESTION_DATE_CUE_RE.search(" ".join(question.split())))
