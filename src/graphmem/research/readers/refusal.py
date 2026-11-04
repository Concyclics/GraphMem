"""Frozen reader constants from prepare_locomo_refusal_policy.py."""

import re
from datetime import datetime

REFUSAL = re.compile(
    r"\b(?:unknown|unspecified|not specified|not mentioned|not provided|not stated|not available|"
    r"not known|not enough|insufficient|no evidence|no information|no record|no mention|"
    r"cannot determine|cannot be determined|can\x27t determine|unable to determine|"
    r"not confirmed|not been confirmed|not explicitly|not explicitly stated)\b",
    re.I,
)
