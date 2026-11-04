"""Move the existing deterministic source expansion before the answer call.

The route sees public question text only. Ranking and source-token limits are
unchanged from the existing lexical/graph reread retriever; no model is added.
"""

import json
import re
import numpy as np

from ..features import Lexical, norm
from .calendar_source_resolver import RELATIVE, resolve, source_date


def route(question):
    if re.search(r"\bgraduat\w*\b", question, re.I) and re.search(
        r"\b(?:age|older|younger|years?)\b", question, re.I
    ):
        return "age_since_education"
    if re.search(r"\b(?:bought|ordered|purchas\w*)\b", question, re.I) and re.search(
        r"\b(?:receiv\w*|deliver\w*|arriv\w*)\b", question, re.I
    ):
        return "purchase_to_receipt"
    if re.search(
        r"\b(?:been living|have i lived|resid\w*)\b", question, re.I
    ) and re.search(
        r"\b(?:how long|how many (?:years?|months?|weeks?|days?))\b", question, re.I
    ):
        return "residence_duration"
    return None


def supplemental_ids(question, turns, graph_scores, tokenizer):
    assert route(question)
    return retrieve_supplemental(question, turns, graph_scores, tokenizer)


def retrieve_supplemental(question, turns, graph_scores, tokenizer):
    lexical = Lexical([t["speaker"] + " " + t["raw_text"] for t in turns])
    target_assistant = bool(
        re.search(
            r"\b(?:did|have) you (?:suggest|recommend|tell|advise)|\byou (?:suggested|recommended|told)\b",
            question,
            re.I,
        )
    )
    speaker = "assistant" if target_assistant else "user"
    owner = np.array([t["speaker"] == speaker for t in turns])
    assert len(graph_scores) == len(turns)
    mixed = (
        0.55 * norm(graph_scores) + 0.45 * norm(lexical.score(question)) + 0.2 * owner
    )
    order = list(map(int, np.argsort(-mixed, kind="stable")))
    extra = []
    if re.search(
        r"\b(age|aged|older|younger|birthday|born)\b|\byears old\b", question, re.I
    ):
        age = lexical.score("my age birthday born years old") + 0.5 * owner
        extra = list(map(int, np.argsort(-age, kind="stable")[:6]))
    candidates = list(dict.fromkeys(order[:12] + extra + order[12:40]))
    selected, used = [], 0
    for i in candidates:
        turn = turns[i]
        if turn["speaker"] != speaker and sum(owner[j] for j in selected) < 12:
            continue
        text = f"[Session {turn['display_session']}, turn {turn['turn_index'] + 1} @ {turn['timestamp']}] {turn['speaker']}: {turn['raw_text']}\n"
        anchor, notes = source_date(turn["timestamp"]), []
        for match in RELATIVE.finditer(turn["raw_text"]):
            start, end = resolve(match.group(), anchor)
            span = str(start) + (".." + str(end) if end != start else "")
            notes.append(
                f"[source-time: {json.dumps(match.group(), ensure_ascii=False)} = {span}; own session date {anchor}]"
            )
            if len(notes) == 3:
                break
        if notes:
            text += "\n".join(notes) + "\n"
        cost = len(tokenizer.encode(text, add_special_tokens=False).ids)
        if used + cost > 5200:
            continue
        selected.append(i)
        used += cost
        if len(selected) >= 24:
            break
    assert selected
    return [turns[i]["turn_id"] for i in selected]
