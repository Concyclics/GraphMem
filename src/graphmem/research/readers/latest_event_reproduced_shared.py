"""Whole latest-event family, selected after two complete development replays.

This is a post-diagnostic one-answer-margin selection. The earlier per-branch
net-two and combined-policy gates stay recorded as failed. No question identity,
reference, grade or prior answer is used to choose this inference path.
"""

from .event_time_selection_v1_shared import route as parent_route, system


def route(question):
    return "latest_event" if parent_route(question) == "latest_event" else None
