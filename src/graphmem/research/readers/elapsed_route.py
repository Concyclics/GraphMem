"""Frozen reader constants from prepare_elapsed_ago_policy.py."""

import re
from datetime import datetime

AGO = re.compile(
    r"\bhow\s+(?:many|much)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.I,
)
