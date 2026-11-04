"""Repeat literal user quantities to help a single existing reader bind operands."""

import re
from .context_operation_v1_shared import route as prior_route, subtype, GUIDANCE
from .branch_prompt_v1_shared import BASE

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
        "typed_user_quantities" if prior_route(question) == "typed_difference" else None
    )


def stems(text):
    return {
        w[:-1] if len(w) > 4 and w.endswith("s") else w
        for w in WORDS.findall(text.lower())
        if w not in STOP
    }


def catalog_entries(question, turns):
    query = stems(question)
    kind = subtype(question)
    extra = {
        "ages": re.compile(
            r"\b(?:age|aged|old|born|birthday|graduat\w*|college|university)\b", re.I
        ),
        "reward_balance": re.compile(
            r"\b(?:points?|stars?|balance|redeem\w*|earned|reward\w*)\b", re.I
        ),
        "stated_difference": re.compile(
            r"[$£€¥]|\b(?:cost|price|paid|spend|spent|fare|amount|total|minutes?|hours?|miles?|pounds?|kilograms?)\b",
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
    assert branch == "typed_user_quantities"
    return (
        BASE
        + """
This is a recall-and-calculate task about what the USER reported. It is not a request for current advice or for correcting their reported figures. Even an implausible reported amount must be recalled faithfully. Generic assistant estimates, alternative scenarios and suggested prices cannot replace the user's actual operands. A value for an unmentioned alternative remains missing even if the assistant offered a typical estimate. Identify both required quantities before calculating; never substitute a calendar year for an age. The navigation list repeats literal user sentences, but may be incomplete: the full original records are also available. Use the relevant source sentences to establish the operands, write one calculation, and put its result in <final> without revising it to a different interpretation.
"""
        + GUIDANCE[subtype(question)]
    )
