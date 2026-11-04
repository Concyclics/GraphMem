"""Narrow task semantics after the broader prompt pilot; development rules."""

import re
from .branch_prompt_v1_shared import route as broad_route, BASE

ATTRIBUTE = re.compile(r"^\s*(?:what|where|which|how often)\b", re.I)
EVENT_DURATION = re.compile(
    r"^\s*how\s+(?:long|(?:many|much)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?))\b.*\bwhen\b",
    re.I,
)
COMPONENTS = re.compile(
    r"\b(?:hours?|days?|weeks?|months?|years?|money|amount|cost|spend|spent|earned|earn)\b",
    re.I,
)


def route(question):
    b = broad_route(question)
    if b == "dated_state" and ATTRIBUTE.search(question):
        return "state_attribute"
    if b == "time_endpoints" and EVENT_DURATION.search(question):
        return "event_anchored_duration"
    if b == "quantity_total" and COMPONENTS.search(question):
        return "component_sum"
    return None


GUIDANCE = {
    "state_attribute": """Read the requested attribute as a BEFORE -> CHANGE -> AFTER timeline. Quote the short statements that establish each version. Resolve the update by its event time and meaning. "Previously" within an old record refers to that record's earlier state; the current question's "previous" asks for the state before the relevant change. Return the old version for "previous" or "original", and the latest established version for "current" or "now". If both are asked, label both. Keep the same person, object and attribute throughout. Do not answer with a related role, location or object.""",
    "event_anchored_duration": """This question asks for a duration at the moment of a SECOND EVENT. First name the starting event and the second event. Quote the two short time statements and identify their shared reference moment. Resolve "ago" in each original statement before comparing. If the start was X weeks before the shared reference and the second event was Y weeks before it, the requested duration is X minus Y, not X. If an offset refers to another already past event, combine those offsets first. A message date is the reference for its wording, not automatically the date of every event in it. Use the second event as the endpoint rather than the question date. Directly stated durations or ages can establish an interval without absolute calendar dates. Give the requested interval, with uncertainty only when the evidence itself is approximate.""",
    "component_sum": """Resolve this sum in two short steps. First list ALL required components and their values, using the question's named items or ordinary category. Scan the earlier records as well as the latest one; an earlier stage or small purchase remains part of the total. Second add the listed values and check that the final answer equals that expression. A cumulative amount already includes its named components, so do not add them again. Repeated mentions are not new purchases or stages. Do not silently add requirements to the question's category or narrow a vague time window to an invented exact cutoff. If a named component is not established, identify it as missing and do not present a partial sum as a complete answer.""",
}


def system(branch):
    return BASE + "\n\nTASK-SPECIFIC CHECK:\n" + GUIDANCE[branch]
