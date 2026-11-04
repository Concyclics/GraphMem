"""Frozen reader constants from prepare_count_policy.py."""

import re
from datetime import datetime

COUNT = re.compile(r"\b(how many|how much|total|altogether|combined)\b", re.I)

INTERVAL = re.compile(
    r"\bhow (?:many|much) (?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b.{0,90}\b(?:passed|elapsed|since|between|older|younger)\b",
    re.I,
)
