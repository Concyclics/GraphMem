"""Query-role routing and conservative answer agreement; never correctness routing."""

import re
import unicodedata
from decimal import Decimal

ROLE = re.compile(
    r"\b(?:you (?:created|wrote|outlined|listed|explained|designed|composed|drafted|generated|gave|provided|recommended|suggested)|we (?:outlined|discussed|created|wrote|agreed)|(?:our|the) (?:previous|earlier|last|past) (?:conversation|discussion)|previous conversation|conversation (?:where|in which) you)\b",
    re.I,
)
WORDS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ],
        range(13),
    )
)


def normalized_answer(answer):
    text = unicodedata.normalize("NFKC", answer).casefold().strip()
    text = re.sub(r"^(?:answer|final answer)\s*:\s*", "", text)
    text = text.strip(" \t\r\n\"'").rstrip(".!")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\s+", " ", text)
    if text in WORDS:
        return "number:" + str(WORDS[text])
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        value = Decimal(text)
        return "number:" + ("0" if value == 0 else format(value.normalize(), "f"))
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(
        r"\b(days|weeks|months|years|hours|minutes|seconds)\b",
        lambda m: m[1][:-1],
        text,
    )
    return text


def agrees(a, b):
    left, right = normalized_answer(a), normalized_answer(b)
    return bool(left) and left == right
