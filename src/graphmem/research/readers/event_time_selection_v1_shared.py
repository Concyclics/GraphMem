"""Question-only event chronology; no outcome/identity-based routing."""

import re
from .branch_prompt_v1_shared import BASE
from .branch_prompt_v2_shared import route as task_route
from .quantity_evidence_v2_shared import route as quantity_route
from .activity_count_route_shared import route as activity_route
from .budget_agreement_shared import ROLE
from .contract_reader_shared import ASSISTANT
from .projected_chat_route_shared import PREVIOUS_CHAT

LATEST = re.compile(r"\bmost recent(?:ly)?\b|\blatest\b", re.I)
ORDER = re.compile(
    r"\border of\b|\bin the order\b|\bfrom (?:earliest|first) to (?:latest|last)\b",
    re.I,
)


def route(question):
    if (
        task_route(question)
        or quantity_route(question)
        or activity_route(question)
        or ROLE.search(question)
        or ASSISTANT.search(question)
        or PREVIOUS_CHAT.search(question)
    ):
        return None
    if ORDER.search(question):
        return "event_order"
    if LATEST.search(question):
        return "latest_event"
    return None


def system(branch, question):
    assert branch in ("latest_event", "event_order")
    return (
        BASE
        + """
EVENT TIMELINE
The date of a message is not the date of every event it recalls. First identify ALL events matching the requested subject, action and completion status. A start, a continued activity, a later recollection and a future plan are different events.
For each qualifying event, write in <work>: event | short original time phrase | date of that source | derived EVENT date or interval. Convert relative expressions using the date of the message containing them. Within that record, a longer "ago" duration points to an EARLIER event. If two statements use different reference dates, normalize each before comparing. For a service's start, subtract the stated duration of use from that record's date; the last mention of using it is not its start date. For a completed visit or trip, use when it happened, not when it was later mentioned.
Scan all relevant user records for additional events and later corrections. Link repeated mentions of the same event; include distinct events even if they share a category. Include only events inside the requested time window. Preserve approximate dates as intervals rather than discarding the event for lacking an exact day. Do not fill an unestablished named event with a different one.
Then sort by EVENT time from oldest to newest, checking the direction explicitly. For "most recent" or "latest", choose the LAST matching event on this timeline and answer its requested attribute. For an order, return the complete ordered list. The final answer must follow the timeline you computed."""
    )
