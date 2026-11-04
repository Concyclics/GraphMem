"""Question-only count route and literal lexical navigation over user records."""

import re
from .branch_prompt_v1_shared import BASE
from .quantity_evidence_v2_shared import route as quantity_route
from .branch_prompt_v2_shared import route as task_route
from .budget_agreement_shared import ROLE
from .contract_reader_shared import ASSISTANT
from .projected_chat_route_shared import PREVIOUS_CHAT

COUNT = re.compile(r"^\s*how many\b|\btotal number\b", re.I)
DURATION = re.compile(
    r"^\s*how many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.I
)
SCOPE = re.compile(
    r"\b(?:total|different|types|currently|initially|simultaneously|including|excluding|or|and|collection|re-watch)\b|\b(?:have I|do I (?:have|own|use))\b",
    re.I,
)
STOP = set(
    "a an the i my me we our have has had do does did how many what was is are were been be to of in on at for from by with and or both all as that this it its their they them he she his her you your can could would should will each most total number different types currently initially simultaneously including excluding now so far since before after past last recently years year months month weeks week days day times time items item use used using own have".split()
)


def route(question):
    if (
        quantity_route(question)
        or task_route(question)
        or ROLE.search(question)
        or ASSISTANT.search(question)
        or PREVIOUS_CHAT.search(question)
    ):
        return None
    return (
        "count_navigation"
        if COUNT.search(question)
        and not DURATION.search(question)
        and SCOPE.search(question)
        else None
    )


def words(text):
    text = re.sub(r"(?<=\w)[-‐‑](?=\w)", "", text.lower())
    out = set()
    for w in re.findall(r"[a-z]+|\d+", text):
        if w in STOP:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ied"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        if len(w) > 2 and w not in STOP:
            out.add(w)
    return out


def catalog_entries(question, turns):
    query = words(question)
    candidates = []
    for t in turns:
        if t["speaker"] != "user":
            continue
        for pos, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", t["raw_text"])):
            sentence = sentence.strip()
            overlap = len(query & words(sentence))
            if not sentence or overlap == 0:
                continue
            candidates.append(
                {
                    "turn_id": t["turn_id"],
                    "display_session": t["display_session"],
                    "turn_index": t["turn_index"],
                    "timestamp": t["timestamp"],
                    "sentence_index": pos,
                    "quote": sentence,
                    "score": overlap,
                }
            )
    chosen, chars, per_turn = [], 0, {}
    for c in sorted(
        candidates,
        key=lambda c: (
            -c["score"],
            c["display_session"],
            c["turn_index"],
            c["sentence_index"],
        ),
    ):
        if len(chosen) >= 16:
            break
        if chars + len(c["quote"]) > 2800 or per_turn.get(c["turn_id"], 0) >= 2:
            continue
        chosen.append(c)
        chars += len(c["quote"])
        per_turn[c["turn_id"]] = per_turn.get(c["turn_id"], 0) + 1
    return sorted(
        chosen,
        key=lambda c: (c["display_session"], c["turn_index"], c["sentence_index"]),
    )


def catalog(question, turns):
    return "\n".join(
        f"Session {c['display_session']}, turn {c['turn_index'] + 1}, user ({c['timestamp']}): {c['quote']}"
        for c in catalog_entries(question, turns)
    )


def system(branch, question):
    assert branch == "count_navigation"
    return (
        BASE
        + """
COUNTING TASK: First state what one counted unit is and the actual eligibility rule in this question. In <work>, make a compact complete list of the qualifying objects or occurrences, or the supported totals and genuine changes. Then return the count or arithmetic result of that list. Read the full user history; the repeated user sentences at the end are navigation, not an exhaustive inventory.
Use ordinary category membership and the user's own statements. Do not demand an exact repetition of the question's wording when the user's description already establishes it, or add an unasked restriction. Distinct types are counted once per type, while distinct occurrences of the same activity count separately. A later description of the same occurrence is a duplicate, not a new event. A later acquisition is an addition to an earlier inventory unless it was already included; a later explicit total replaces the earlier total. Follow the requested time and completion scope.
For AND/OR categories or actions, collect the qualifying groups separately, then count each physical object once across those groups. Respect explicit inclusions and exclusions even when they extend beyond normal ownership or completion. Include planned actions only when the question asks for plans. For an initial count use the initial quantities rather than later losses or survivors. Require every named category when a total is requested; if a necessary item is unestablished, report the missing information instead of zero or a partial total. The final numeric answer must match the listed objects or arithmetic."""
    )
