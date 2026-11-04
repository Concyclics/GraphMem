"""Frozen reader constants from run_quote_reader_pipeline.py."""

import re
from datetime import datetime

GUARD = (
    "Answer the current question from the supplied original records. The interrupted draft is only a draft, "
    "not an additional source. Return just a JSON object with one string field named answer. "
    "Give the concise complete final answer; do not repeat reasoning or cite IDs. Do not invent missing facts."
)
