"""Frozen reader constants from prepare_lme_reasoned.py."""

import re
from datetime import datetime

format_rule = (
    "\n\nResponse format: first write <work> followed by a brief source-grounded verification, "
    "then </work>. Identify the relevant statements, distinguish completed events from plans, "
    "deduplicate repeated mentions, and perform any required calculation or date comparison "
    "before deciding the answer. For time order, explicitly compare the calendar endpoints; "
    "for totals, identify which quantities already include other items. For current state, "
    "check the latest matching update including its time of day. Use at most 200 words for "
    "this verification, and do not open it with a guessed answer. Then write exactly one "
    "<final>concise final answer</final> block. The final answer must agree with the verified "
    "calculation and evidence. Do not write text outside these two blocks."
)
