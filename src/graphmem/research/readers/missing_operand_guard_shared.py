"""Route a missing literal named operand to the existing two-event reader.

Literal absence only triggers a reader check; it never determines the answer.
The reader still sees every original and can recognize aliases or paraphrases.
"""

import re
from .context_operation_v1_shared import route

PAIR = re.compile(r"\bfirst\s*,?\s+(.+?)\s+or\s+(.+?)\??$", re.I)
NAME = re.compile(r"\b[A-Z][A-Za-z]+\b")
STOP = {"The", "An", "Of", "To", "In", "On", "At", "And", "Or"}


def missing_named_operands(question, turns):
    if route(question) != "two_event_order":
        return []
    pair = PAIR.search(question)
    if pair is None:
        return []
    # All roles, not just user records, to avoid rejecting an alias inferred
    # from a conversational reply. The actual reader verifies the event.
    text = "\n".join(t["raw_text"] for t in turns)
    missing = []
    for side in pair.groups():
        names = [w for w in NAME.findall(side) if w not in STOP]
        absent = [
            w for w in names if not re.search(r"\b" + re.escape(w) + r"\b", text, re.I)
        ]
        if absent:
            missing.append(
                {"operand": side.rstrip("?"), "absent_name_tokens": sorted(set(absent))}
            )
    return missing
