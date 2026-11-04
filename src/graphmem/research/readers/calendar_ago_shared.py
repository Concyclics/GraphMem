"""Deterministic readout from a verified exact-date candidate."""

from datetime import date
import math
import re
from .calendar_source_resolver import source_date, shift_months

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "candidate_id": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["reasoning", "candidate_id", "answer"],
    "additionalProperties": False,
}


def compute(row, candidate_id):
    candidates = {c["id"]: c for c in row["candidates"]}
    if candidate_id not in candidates:
        return None
    c = candidates[candidate_id]
    if c["start"] != c["end"]:
        return None
    records = {r["id"]: r for r in row["records"]}
    record = records[c["source_id"]]
    assert record["speaker"] == "user" and c["phrase"] in record["raw_text"]
    event, endpoint = date.fromisoformat(c["start"]), source_date(row["question_date"])
    days = (endpoint - event).days
    if days < 0:
        return None
    unit = re.search(
        r"\bhow\s+(?:many|much)\s+(days?|weeks?|months?|years?)\s+ago\b",
        row["question"],
        re.I,
    )
    if not unit:
        return None
    unit = unit.group(1).lower()
    if unit.startswith("day"):
        return f"{days} days ago."
    if unit.startswith("week"):
        if days % 7 == 0:
            return f"{days // 7} weeks ({days} days) ago."
        return f"About {math.floor(days / 7 + 0.5)} weeks ({days} days) ago."
    months = (endpoint.year - event.year) * 12 + endpoint.month - event.month
    values = range(max(0, months - 1), months + 2)
    nearest = min(
        values, key=lambda value: abs((shift_months(event, value) - endpoint).days)
    )
    if unit.startswith("month"):
        return f"About {nearest} months ({days} days) ago."
    return f"About {math.floor(nearest / 12 + 0.5)} years ({days} days) ago."
