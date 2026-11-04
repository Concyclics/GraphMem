"""Repeat literal user quantities to help a single existing reader bind operands."""

import re
from .context_operation_v1_shared import route as prior_route, subtype, GUIDANCE
from .branch_prompt_v1_shared import BASE
from .branch_prompt_v2_shared import (
    route as component_route,
    GUIDANCE as COMPONENT_GUIDANCE,
)

NUMBER = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|half|dozen)\b",
    re.I,
)
WORDS = re.compile(r"[a-z]+", re.I)
STOP = set(
    "a an the i my me am was were is are do did does have had has how what much many more less than then when will would to from at in on of for and or by with instead it this that old older years year need save saved".split()
)


def route(question):
    return (
        "component_evidence" if component_route(question) == "component_sum" else None
    )


def stems(text):
    return {
        w[:-1] if len(w) > 4 and w.endswith("s") else w
        for w in WORDS.findall(text.lower())
        if w not in STOP
    }


def catalog_entries(question, turns):
    query = stems(question)
    kind = "stated_difference"
    extra = {
        "ages": re.compile(
            r"\b(?:age|aged|old|born|birthday|graduat\w*|college|university)\b", re.I
        ),
        "reward_balance": re.compile(
            r"\b(?:points?|stars?|balance|redeem\w*|earned|reward\w*)\b", re.I
        ),
        "stated_difference": re.compile(
            r"[$£€¥]|\b(?:cost|price|paid|spend|spent|fare|amount|total|minutes?|hours?|days?|weeks?|months?|years?|miles?|pounds?|kilograms?)\b",
            re.I,
        ),
    }[kind]
    candidates = []
    for t in turns:
        if t["speaker"] != "user":
            continue
        for pos, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", t["raw_text"])):
            sentence = sentence.strip()
            if not sentence or not NUMBER.search(sentence):
                continue
            overlap = len(query & stems(sentence))
            relevant = extra.search(sentence) is not None
            if overlap == 0 and not relevant:
                continue
            candidates.append(
                {
                    "turn_id": t["turn_id"],
                    "display_session": t["display_session"],
                    "turn_index": t["turn_index"],
                    "timestamp": t["timestamp"],
                    "sentence_index": pos,
                    "quote": sentence,
                    "score": 4 * overlap + 2 * relevant,
                }
            )
    # CPU lexical navigation only; no original source is removed from the reader.
    chosen, chars = [], 0
    for c in sorted(
        candidates,
        key=lambda c: (
            -c["score"],
            c["display_session"],
            c["turn_index"],
            c["sentence_index"],
        ),
    ):
        if len(chosen) >= 24:
            break
        if chars + len(c["quote"]) > 6500:
            continue
        chosen.append(c)
        chars += len(c["quote"])
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
    assert branch == "component_evidence"
    return (
        BASE
        + "\n"
        + COMPONENT_GUIDANCE["component_sum"]
        + """
Before adding, check that every named item and the requested final stage are actually established. A total through a final stage cannot be answered with only an earlier stage if the final stage is absent. When the question asks time spent across stages, count the actual durations the user reported for each distinct stage; do not replace them with a typical total duration or invent overlap/credit transfers. When it names several works or purchases, bind a value to each exact title/item before summing. A completed activity, a plan and a different version of an item are distinct. The repeated user-quantity sentences are navigation only, so consult the full records for context and any missing components. Quote the relevant original values, show one compact expression covering the complete requested scope, and return that sum. If a necessary named component is not established, state which one is missing rather than give a partial sum.
"""
    )
