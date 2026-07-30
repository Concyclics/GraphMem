from __future__ import annotations

import re
from typing import Any


def resolve_source_ids(value: Any, valid: set[str]) -> list[str]:
    """Resolve only exact IDs or unique turn references inside one session."""
    if isinstance(value, (str, int)):
        rows = [value]
    elif isinstance(value, list):
        rows = value
    else:
        rows = []
    by_index: dict[int, list[str]] = {}
    for node_id in valid:
        match = re.search(r":turn:(\d+)$", node_id)
        if match:
            by_index.setdefault(int(match.group(1)), []).append(node_id)
    resolved: list[str] = []
    for item in rows:
        raw = str(item).strip()
        if raw in valid:
            resolved.append(raw)
            continue
        suffix_matches = [
            node_id for node_id in valid
            if raw and node_id.endswith(":" + raw)
        ]
        if len(suffix_matches) == 1:
            resolved.append(suffix_matches[0])
            continue
        match = re.fullmatch(r"(?:turn[:_\s-]*)?(\d+)", raw, flags=re.IGNORECASE)
        if match:
            candidates = by_index.get(int(match.group(1)), [])
            if len(candidates) == 1:
                resolved.append(candidates[0])
    return list(dict.fromkeys(resolved))
