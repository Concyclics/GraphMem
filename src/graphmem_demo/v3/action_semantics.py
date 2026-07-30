from __future__ import annotations

import re
from collections.abc import Iterable


_ACTION_FAMILIES = {
    "acquire": {
        "acquire", "acquired", "buy", "bought", "get", "got", "order",
        "ordered", "pick", "picked", "purchase", "purchased", "receive",
        "received",
    },
    "attend": {
        "attend", "attended", "go", "gone", "participate", "participated",
        "visit", "visited", "volunteer", "volunteered", "went",
    },
    "complete": {
        "complete", "completed", "finish", "finished",
    },
    "project_work": {
        "assemble", "assembled", "build", "built", "complete", "completed",
        "finish", "finished", "work", "worked", "working",
    },
    "prepare": {
        "bake", "baked", "baking", "cook", "cooked", "cooking",
        "make", "made", "making", "prepare", "prepared", "preparing",
    },
    "encounter": {
        "chat", "chatted", "conversation", "converse", "conversed",
        "encounter", "encountered", "meet", "meets", "met", "talk", "talked",
    },
    "recommend": {
        "advice", "advise", "advised", "recommend", "recommendation",
        "recommendations", "recommended", "suggest", "suggested", "suggestion",
        "suggestions",
    },
    "remove": {
        "cancel", "cancelled", "canceled", "discard", "discarded", "remove",
        "removed", "lose", "lost", "return", "returned", "sell", "sold",
    },
    "repair": {
        "clean", "cleaned", "fix", "fixed", "lubricate", "lubricated",
        "maintain", "maintained", "mend", "mended", "repair", "repaired",
        "replace", "replaced", "restore", "restored", "service", "serviced",
        "tune", "tuned",
    },
    "travel": {
        "fly", "flew", "flight", "flown", "travel", "traveled",
        "travelled",
    },
    "use": {
        "operate", "operated", "play", "played", "spend", "spent", "take",
        "took", "use", "used", "wear", "wearing", "wore",
    },
}

_TOKEN_TO_FAMILY = {
    token: family
    for family, tokens in _ACTION_FAMILIES.items()
    for token in tokens
}


def action_families(value: str | Iterable[str]) -> set[str]:
    """Return small, benchmark-independent event-action equivalence classes."""
    if isinstance(value, str):
        # “got back from” denotes completed travel/attendance, not acquisition.
        normalized = re.sub(r"\bgot\s+back\s+from\b", " went ", value.casefold())
        tokens = re.findall(r"[\w']+", normalized)
    else:
        tokens = [str(item).casefold() for item in value]
    normalized: set[str] = set(tokens)
    for token in tokens:
        if len(token) > 5 and token.endswith("ing"):
            base = token[:-3]
            normalized.update({base, base + "e"})
        elif len(token) > 4 and token.endswith("ed"):
            base = token[:-2]
            normalized.update({base, base + "e"})
    return {
        family
        for token in normalized
        if (family := _TOKEN_TO_FAMILY.get(token)) is not None
    }


def action_family_overlap(left: str | Iterable[str], right: str | Iterable[str]) -> int:
    return len(action_families(left) & action_families(right))


def has_completed_participation(value: str) -> bool:
    lowered = value.casefold()
    if re.search(
        r"\b(?:i|we)\b.{0,50}\b(?:plan|want|will)\w*\b.{0,100}"
        r"\b(?:attend|go|participat|visit|volunteer)\w*\b",
        lowered,
    ):
        return False
    return bool(re.search(
        r"\b(?:i|we)\b.{0,240}\b(?:attended|went|visited|volunteered|participated|"
        r"got\s+back\s+from)\b",
        lowered,
    ))
