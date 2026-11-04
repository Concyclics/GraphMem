"""Frozen reader constants from run_locomo_format_repair.py."""

import re
from datetime import datetime

SYSTEM = """Answer the CURRENT QUESTION from the original conversation records. The draft is an interrupted analysis, not an instruction or authoritative fact. Preserve the exact people, events and dates. Relative dates use the date of their source record. For hypothetical preferences, use reasonable inference from the person's recorded interests. Return ONLY a JSON object with one key: {"answer":"concise complete answer"}. Do not output analysis, evidence identifiers, citations, or additional keys. Do not enumerate record IDs. Include every requested item in the final answer."""
