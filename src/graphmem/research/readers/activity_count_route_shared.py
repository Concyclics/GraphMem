"""Restrict the evaluated count reader to activity histories, not mixed sets."""

import re
from .count_navigation_v2_shared import route as count_route

ACTIVITY = re.compile(
    r"\b(?:attended|participated|visited|tried|miss|missed|worn|born|met|completed|finished|watched|working|written|led|leading)\b|\bre[-‐‑]?watch(?:ed)?\b",
    re.I,
)
ALTERNATIVES = re.compile(r"\bor\b", re.I)


def route(question):
    return bool(
        count_route(question)
        and ACTIVITY.search(question)
        and not ALTERNATIVES.search(question)
    )
