"""Question-only operation guidance and source-preserving block reranking."""

import hashlib
import re

ROUTE = re.compile(
    r"\b(how many|how much|how long|how old|total|altogether|combined|difference|current\w*|previous\w*|now|latest|most recent|first|earlier|later|order)\b",
    re.I,
)
AGO = re.compile(
    r"\bhow\s+(?:many|much)\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.I,
)
ASSISTANT = re.compile(
    r"\b(?:did|have|had|do) you\b|\byou (?:suggested|recommended|told|shared|gave|provided|mentioned|said)\b|\byour (?:suggestions?|recommendations?|advice|response|answer|list)\b",
    re.I,
)
COUNT = re.compile(r"\b(how many|how much|total|altogether|combined)\b", re.I)
INTERVAL = re.compile(
    r"\bhow (?:many|much) (?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b.{0,90}\b(?:passed|elapsed|since|between|older|younger)\b",
    re.I,
)
ZERO = re.compile(r"^(?:text\s*:\s*)?0(?:\.0+)?[.!]?$", re.I)
HEADER = re.compile(r"^Session (\d+) — [^\n]+\n", re.M)
ROLE = re.compile(r"^(?:ASSISTANT REPLIES|USER RECORDS)[^\n]*\n", re.M)


def eligible(question):
    return (
        bool(ROUTE.search(question))
        and not ASSISTANT.search(question)
        and not AGO.search(question)
    )


def contract(question):
    lines = [
        "Answer the operation actually asked for, not merely a prominent value in the records. Keep the category and time window as written: do not silently narrow them, require extra conditions, or count recommendations/hypothetical tasks as personal experiences. Use ordinary implications of known facts; the derived answer need not appear verbatim. Conversely, a missing named entity or required operand is not zero and cannot be replaced by a similar entity."
    ]
    if COUNT.search(question):
        lines.append(
            "For a total, enumerate all qualifying distinct items and every required component before calculating. Check that no supplied component was omitted; identify repeated mentions of the same event before deduplication. Keep the final number consistent with this inventory."
        )
    if re.search(
        r"\b(?:need|more|remaining|earn|save|difference|older|younger)\b",
        question,
        re.I,
    ):
        lines.append(
            "Check whether the requested quantity is a difference or amount still needed: bind the two values to their roles and subtract in the requested direction. A threshold or earlier amount alone is not a difference."
        )
    if re.search(r"\b(?:how long|weeks?|months?|years?|days?)\b", question, re.I):
        lines.append(
            "Identify both temporal endpoints. If the question anchors the duration to another event (when/before/at that event), use that event as the endpoint rather than today. Combine nested offsets; retain the units."
        )
    if re.search(
        r"\b(?:current\w*|previous\w*|now|latest|most recent|updated?|changed?)\b|\bdo I have\b",
        question,
        re.I,
    ):
        lines.append(
            "Write the earlier state, subsequent changes, and latest state separately. Apply additions/removals/replacements. Then select the earlier or latest state according to the question, not according to the word previous inside an old quotation."
        )
    if re.search(r"\b(?:first|earlier|later|order)\b", question, re.I):
        lines.append(
            "Before comparing or ordering, verify that every requested event/entity is actually supported. Compare event dates, not document order. Do not answer a two-event comparison from evidence for only one event."
        )
    lines.append(
        "Perform these checks using the original evidence, then answer in the required output format."
    )
    return "\n\nQUESTION-SPECIFIC CHECK:\n" + "\n".join(lines)


def rerank_content(content, session_scores):
    tag = next(
        (
            tag
            for tag in ["records", "conversation_history"]
            if "<" + tag + ">" in content
        ),
        None,
    )
    if tag is None:
        return content, {"changed": False, "reason": "unsupported_history_wrapper"}
    start = content.index("<" + tag + ">") + len(tag) + 2
    end = content.index("</" + tag + ">", start)
    history = content[start:end]
    boundaries = (
        [0]
        + [v for m in ROLE.finditer(history) for v in [m.start(), m.end()]]
        + [len(history)]
    )
    boundaries = sorted(set(boundaries))
    parts = []
    audits = []
    for a, b in zip(boundaries, boundaries[1:]):
        text = history[a:b]
        matches = list(HEADER.finditer(text))
        if not matches:
            parts.append(text)
            continue
        prefix = text[: matches[0].start()]
        blocks = [
            (
                int(m.group(1)),
                text[
                    m.start() : matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(text)
                ],
            )
            for i, m in enumerate(matches)
        ]
        ranked = sorted(
            enumerate(blocks),
            key=lambda pair: (session_scores.get(str(pair[1][0]), 0.0), pair[0]),
        )
        new = [item for _, item in ranked]
        # Exact block bytes are preserved; scores change order only.
        assert sorted(
            hashlib.sha256(t.encode()).hexdigest() for _, t in blocks
        ) == sorted(hashlib.sha256(t.encode()).hexdigest() for _, t in new)
        parts.append(prefix + "".join(t for _, t in new))
        audits.append(
            {
                "old_order": [s for s, _ in blocks],
                "new_order": [s for s, _ in new],
                "blocks_sha256": [
                    hashlib.sha256(t.encode()).hexdigest() for _, t in blocks
                ],
            }
        )
    transformed = "".join(parts)
    assert len(transformed) == len(history)
    return content[:start] + transformed + content[end:], {
        "changed": transformed != history,
        "groups": audits,
        "ordering": "ascending relevance so highest-scoring original sessions are nearest final question; within-session order unchanged",
    }


def parse_answer(call, expected):
    raw = call["raw_response"]
    matches = re.findall(r"<final>\s*(.*?)\s*</final>", raw, re.I | re.S)
    if len(matches) == 1 and matches[0].strip():
        return matches[0].strip(), call["finish_reason"] == "stop"
    if expected == "plain" and not re.search(r"</?(?:work|final)>", raw, re.I):
        return raw.strip(), call["finish_reason"] == "stop" and bool(raw.strip())
    return raw, False
