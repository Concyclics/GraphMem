"""Clarify elapsed-time answers using only public question text and records."""

import re
from .astra_stage2_shared import request_body
from .astra_lme_branches_shared import instruction as prior_instruction

ELAPSED = re.compile(
    r"^how many (?:days?|weeks?|months?|years?)\b.*\b(?:ago|since|between|passed|had|have|has)\b",
    re.I,
)
GUIDANCE = """For this elapsed-time question, identify the specific event or pair of events being asked about. "Met" or another repeated activity does not mean "first ever" unless the question says first: use the relevant dated encounter, and distinguish an earlier introduction if necessary.
Calculate the interval between the supported dates. When the only explicit anchors are a report date or a planned event date, give the interval between those named anchors and clearly label them as reported or planned; also state that the exact actual-event interval is unconfirmed. Do not replace a computable, useful qualified interval with only "unknown". Conversely, never present a report date or plan as a confirmed actual occurrence. If both an explicit start date and a conflicting duration appear, state the interval from the explicit start date with the conflict qualified, instead of silently choosing one account.
Put the directly relevant interval and unit first, then the shortest qualification needed to preserve what the records establish. Do not invent an endpoint, erase a genuine contradiction, assume a first occurrence, or use a related but different activity."""


def route(question):
    return "elapsed_time_anchors" if ELAPSED.search(question) else None


def instruction(branch):
    if branch == "elapsed_time_anchors":
        return (
            prior_instruction("numeric_answer")
            + "\n\nELAPSED-TIME CLARIFICATION:\n"
            + GUIDANCE
        )
    return prior_instruction(branch)


def payload(row):
    body = request_body(row)
    body["instructions"] = instruction(row["branch"])
    return body
