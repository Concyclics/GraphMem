"""Public question predicates and source-grounded instructions for LME."""

import re
from .astra_stage2_shared import INSTRUCTION, request_body

NUMERIC = re.compile(
    r"^how (?:many|much)\b|\b(?:average|total number|total amount)\b", re.I
)
TOPICS = {
    "relatives": re.compile(r"\b(?:siblings?|brothers?|sisters?)\b", re.I),
    "attendance": re.compile(
        r"\b(?:workshops?|lectures?|conferences?|seminars?|symposium|symposia)\b", re.I
    ),
    "projects": re.compile(r"\bprojects?\b", re.I),
}
SOURCE_TOPICS = {
    **TOPICS,
    "projects": re.compile(r"\b(?:projects?|led|leading)\b", re.I),
}


def topic_names(question):
    return [name for name, pattern in TOPICS.items() if pattern.search(question)]


def route(question):
    numeric = bool(NUMERIC.search(question))
    if numeric and topic_names(question):
        return "topic_count_sources"
    if re.search(
        r"\b(?:most|least|highest|lowest|largest|smallest)\b", question, re.I
    ) and re.search(
        r"\b(?:money|spend|spent|spending|cost|paid|amount|payment|expense)\b",
        question,
        re.I,
    ):
        return "ranked_spending"
    if numeric:
        return "numeric_answer"
    if re.search(
        r"\b(?:poster|undergrad|undergraduate|research paper|thesis)\b", question, re.I
    ):
        return "named_episode"
    return None


NUMERIC_GUIDANCE = """Resolve the requested quantity before answering. Keep the exact person, object, role and completed action fixed. Privately reconcile the relevant original user statements and bind every operand to that scope.
For counts, count distinct objects or completed events, not mentions, repeated descriptions, future plans or inferred occurrences of a routine. Repeated descriptions of the same item, couple or outcome may refer to one event despite different report dates; count a second occurrence only when the records establish a separate event. Match leadership of a whole project separately from leading one component or participating in it. When counting alternatives considered before choosing a named target, exclude that chosen target from the prior alternatives.
For a balance or remaining requirement, return the additional amount still needed, not the redemption threshold. For comparisons of routines, use the stated usual pattern for each named day, not an isolated late wake-up or a suggested schedule. For elapsed time, identify the actual start of the requested activity and the specific endpoint; enrollment, an intention, a related earlier activity and a restart are not interchangeable. For an average, include all and only the named people.
Preserve explicit reports even when their dates are imprecise. Do not invent dates or silently reinterpret them to fit a total. If a timestamp conflicts with the user's explicit retrospective description, distinguish the reported result from the uncertain timing with a brief qualifier rather than silently discarding the report. If only part of a requested total is known, clearly label that supported amount as partial; missing evidence is not zero.
Return the requested result and unit concisely. Do not lead with an intermediate value, threshold, operand list or a different interpretation; include a short qualification only when needed to avoid a false claim."""

RANK_GUIDANCE = """Compare the requested spending across all relevant user-reported amounts. Group transactions by the actual store or entity and add distinct purchases when the question asks cumulative spending; do not confuse the largest single transaction with the largest total. Preserve discounts, fees and shared purchases according to what the question asks. A report described as "last month" does not supply an exact day: do not silently exclude the whole report by assuming a day or narrower period. When timing is ambiguous, identify the largest supported reported amount and its entity with a short timing qualification. Do not invent a transaction or ignore an explicit incompatible date. Return the entity asked for first, with only the amount or brief qualification needed to clarify it."""

EPISODE_GUIDANCE = """Match the exact educational or research episode before answering: academic level, course versus thesis, project topic, artifact (paper, poster, presentation), role, and venue must refer to the same event. A university mentioned for another visit, degree or presentation is not automatically the venue of this event. Combine records that explicitly establish the linkage. If a relevant date is only a conference's submission date or deadline rather than a confirmed personal action date, state that associated date with its correct attribution and clarify that the personal exact date is not established. If the requested event or venue is absent, say so without borrowing a nearby event's answer."""


def instruction(branch):
    guide = (
        NUMERIC_GUIDANCE
        if branch in ["numeric_answer", "topic_count_sources"]
        else RANK_GUIDANCE
        if branch == "ranked_spending"
        else EPISODE_GUIDANCE
    )
    return INSTRUCTION + "\n\nQUESTION-SPECIFIC READING GUIDANCE:\n" + guide


def payload(row):
    result = request_body(row)
    result["instructions"] = instruction(row["branch"])
    return result
