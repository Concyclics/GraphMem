"""Accepted prompt plans using only a question, its date and retrieved originals.

This module neither reads benchmark IDs/labels nor loads a model. The caller
uses the existing answer server. Try a plan's primary request, then its repair
only if the output is malformed/truncated. If both fail, try the next plan,
ultimately falling back to the existing calendar/base reader. Charge every
request actually executed, and both existing query embedding views once.
"""

from datetime import datetime
from .decomposition_shared import payload
from .branch_prompt_v2_shared import route as task_route, system as task_system
from .context_operation_v1_shared import system as event_system
from .missing_operand_guard_shared import missing_named_operands
from .quantity_evidence_v2_shared import (
    route as quantity_route,
    system as quantity_system,
    catalog as quantity_catalog,
)
from .components_evidence_v3_shared import (
    route as component_route,
    system as component_system,
    catalog as component_catalog,
)
from .activity_count_route_shared import route as activity_route
from .count_navigation_v2_shared import system as count_system, catalog as count_catalog
from .latest_event_reproduced_shared import (
    route as latest_event_route,
    system as latest_event_system,
)


def render(question, question_date, turns):
    """Preserve the evaluated role ordering without experiment-preparation imports."""
    ordered = sorted(
        turns,
        key=lambda t: (
            datetime.strptime(t["timestamp"], "%Y/%m/%d (%a) %H:%M"),
            t["display_session"],
            t["turn_index"],
        ),
    )
    blocks = []
    for role in ["assistant", "user"]:
        lines = [role.upper() + " RECORDS"]
        last = None
        for t in ordered:
            if t["speaker"] != role:
                continue
            if last != t["session_id"]:
                lines.append(f"\nSession {t['display_session']} — {t['timestamp']}")
                last = t["session_id"]
            lines.append(f"[turn {t['turn_index'] + 1}] {role}: {t['raw_text']}")
        blocks.append("\n".join(lines))
    return (
        "The following records are grouped by speaker; every record keeps its source timestamp.\n\n<records>\n"
        + "\n\n".join(blocks)
        + "\n</records>\n\nCURRENT QUESTION: "
        + question
        + "\nQuestion date: "
        + question_date
        + "\nMake a complete ledger of the required distinct items or operands, then compute the answer."
    )


def _spec(branch, text, system):
    return {
        "branch": branch,
        "payload": payload(system, text, 1800),
        "repair_payload": payload(
            system + "\nWrite a complete concise <final> block now.", text, 512
        ),
    }


def plans(question, question_date, selected_original_turns):
    """Return ordered prompt overrides; an empty list delegates to the base reader."""
    text = render(question, question_date, selected_original_turns)
    tail = "Make a complete ledger of the required distinct items or operands, then compute the answer."
    assert text.endswith(tail)
    text = (
        text[: -len(tail)]
        + "Follow the task-specific steps and give the supported answer."
    )
    out = []
    if missing_named_operands(question, selected_original_turns):
        out.append(
            _spec(
                "missing_named_operand", text, event_system("two_event_order", question)
            )
        )
    elif quantity_route(question):
        extended = (
            text
            + "\n\nLITERAL USER QUANTITY MENTIONS (navigation, not a complete inventory):\n"
            + quantity_catalog(question, selected_original_turns)
        )
        extended += "\n\nCURRENT QUESTION: " + question
        out.append(
            _spec(
                "typed_user_quantities",
                extended,
                quantity_system("typed_user_quantities", question),
            )
        )
    elif component_route(question):
        extended = (
            text
            + "\n\nLITERAL USER QUANTITY MENTIONS (navigation, not a complete inventory):\n"
            + component_catalog(question, selected_original_turns)
        )
        extended += "\n\nCURRENT QUESTION: " + question
        out.append(
            _spec(
                "component_evidence",
                extended,
                component_system("component_evidence", question),
            )
        )
    elif activity_route(question):
        extended = (
            text
            + "\n\nLITERAL USER SENTENCES RELATED TO THE COUNT (navigation, not a complete inventory):\n"
            + count_catalog(question, selected_original_turns)
        )
        extended += "\n\nCURRENT QUESTION: " + question
        out.append(
            _spec(
                "activity_count", extended, count_system("count_navigation", question)
            )
        )
    elif latest_event_route(question):
        extended = text + "\n\nCURRENT QUESTION: " + question
        out.append(
            _spec(
                "latest_event", extended, latest_event_system("latest_event", question)
            )
        )
    task = task_route(question)
    if task:
        out.append(_spec(task, text, task_system(task)))
    return out
