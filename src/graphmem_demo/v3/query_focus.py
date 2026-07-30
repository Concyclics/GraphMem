"""Query-conditioned contraction of an already retrieved V3 subgraph."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .action_semantics import action_family_overlap
from .build import canonical_key


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_BLOCK_HEADER_RE = re.compile(
    r"^\[(?:TURN|CLAIM|EVENT|EVENT_FRAME|OPERAND|EPISODE|THEME)\b",
    re.IGNORECASE,
)
_BLOCK_ID_RE = re.compile(
    r"^\[(?:TURN|CLAIM|EVENT|EVENT_FRAME|OPERAND|EPISODE|THEME)\s+([^\s|\]]+)",
    re.IGNORECASE,
)
_SOURCE_IDS_RE = re.compile(r"\bsources=([^\]]+)", re.IGNORECASE)
_SPEAKER_RE = re.compile(r"\bspeaker=([^\]|]+)", re.IGNORECASE)
_CROSS_SPEAKER_ANAPHORA_RE = re.compile(
    r"\b(?:that|this|the)\b.{0,48}\byou\b|"
    r"\byou\b.{0,24}\b(?:advised|recommended|suggested)\b",
    re.IGNORECASE,
)
_QUESTION_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its",
    "many", "much", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "would",
}


@dataclass(frozen=True)
class AnswerSlot:
    kind: str
    instruction: str


def infer_answer_slot(question: str) -> AnswerSlot:
    """Infer the requested semantic slot from benchmark-independent syntax."""

    q = " ".join(_tokens(question))
    if re.search(r"\bhow many\b|\bwhat (?:number|amount)\b", q):
        return AnswerSlot("number", "Return a number or auditable counted set.")
    if re.match(r"^how long\b", q):
        return AnswerSlot("duration", "Return the requested elapsed duration or combined duration.")
    if re.match(r"^(?:when|what date|what time)\b", q):
        return AnswerSlot(
            "time",
            "Return the requested date, time, or time range. Resolve relative time "
            "against the cited source header and include that anchor in the answer.",
        )
    if re.match(r"^where\b", q):
        return AnswerSlot("location", "Return the requested place, not an activity or date.")
    if re.match(r"^who\b", q):
        return AnswerSlot("person", "Return the requested person or people.")
    if re.match(r"^why\b", q):
        return AnswerSlot("cause", "Return the stated cause or motivation.")
    if re.search(r"^how (?:did|does|do|was|were)\b.*\bfeel\b", q):
        return AnswerSlot("emotion", "Return the person's emotion or reaction.")
    if re.match(r"^how\b", q):
        return AnswerSlot("manner", "Return the requested manner, method, or reaction.")
    if re.match(r"^what (?:type|kind|sort)\b", q):
        return AnswerSlot("category", "Return the requested type or category of the named object.")
    if re.match(r"^what does\b.*\bmake\b", q):
        return AnswerSlot("effect", "Return the resulting state or effect, not a role or activity.")
    if re.match(r"^(?:what|which)\b", q):
        return AnswerSlot(
            "entity_or_attribute",
            "Return the exact requested object or fine-grained attribute. When a typed "
            "projection omits a modifier or complement preserved in its lossless raw source, "
            "the raw source controls that attribute.",
        )
    return AnswerSlot("proposition", "Return the proposition requested by the question.")


def should_use_focused_capsule(*, answer_form: str, operation: str) -> bool:
    """Avoid narrowing requests whose answer must remain open-ended or exhaustive."""

    return answer_form not in {"recommendation", "list"} and operation != "ordering"


def _token_key(token: str) -> str:
    value = token.casefold().strip("'\"")
    if value.endswith("'s"):
        value = value[:-2]
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _tokens(text: str) -> list[str]:
    return [_token_key(token) for token in _WORD_RE.findall(text)]


def _question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in _tokens(question):
        if token in _QUESTION_STOP or len(token) < 2:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _score_block(
    block: str,
    *,
    terms: list[str],
    term_frequency: dict[str, int],
    question: str,
) -> float:
    lowered = canonical_key(block)
    block_terms = set(_tokens(block))
    score = 0.0
    covered = 0
    for term in terms:
        if term not in block_terms and term not in lowered:
            continue
        covered += 1
        score += 1.0 / max(1, term_frequency.get(term, 1)) ** 0.5
    if terms:
        score += 2.2 * covered / len(terms)
    score += 1.2 * action_family_overlap(question, block)
    if re.match(r"^(?:what|which)\b", canonical_key(question)) and re.search(
        r"[\"'][A-Z][^\"'\n]{1,80}[\"']|\b[A-Z][\w'-]+\s+by\s+[A-Z][\w'-]+",
        block,
    ):
        score += 0.9
    for left, right in zip(terms, terms[1:]):
        if re.search(rf"\b{re.escape(left)}\b.{{0,48}}\b{re.escape(right)}\b", lowered):
            score += 0.55
    if _BLOCK_HEADER_RE.match(block.strip()):
        score += 0.25
    if block.lstrip().upper().startswith(("[CLAIM", "[OPERAND", "[EVENT_FRAME")):
        score += 0.3
    q = canonical_key(question)
    if re.search(r"\b(?:plan|planned|planning|intend|intended)\b", q):
        if re.search(r"\b(?:planned|planning|will|going to|intend)\b", lowered):
            score += 0.85
    if re.search(r"\bfavou?rite\b", q):
        if re.search(
            r"\b(?:favou?rite|love|loved|enjoy|enjoyed|awesome|blast|highly recommend)\b",
            lowered,
        ):
            score += 1.15
    asks_about_media = bool(
        re.search(r"\b(?:photo|picture|image|look like|shown|shared)\b", q)
    )
    if "[media shared" in lowered and not asks_about_media:
        score -= 0.45
    return score


def focused_evidence_capsule(
    question: str,
    context: str,
    *,
    max_blocks: int = 10,
    max_chars: int = 3000,
) -> str:
    """Return a bounded, provenance-preserving contraction of packed evidence."""

    blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
    if not blocks:
        return ""
    terms = _question_terms(question)
    term_frequency = {
        term: sum(
            1
            for block in blocks
            if term in set(_tokens(block)) or term in canonical_key(block)
        )
        for term in terms
    }
    ranked = sorted(
        enumerate(blocks),
        key=lambda item: (
            -_score_block(
                item[1],
                terms=terms,
                term_frequency=term_frequency,
                question=question,
            ),
            item[0],
        ),
    )
    selected: list[tuple[int, str]] = []
    used = 0
    seen: set[str] = set()
    block_by_id: dict[str, tuple[int, str]] = {}
    for index, block in enumerate(blocks):
        match = _BLOCK_ID_RE.match(block)
        if match:
            block_by_id[match.group(1).strip()] = (index, block)

    def speaker(block: str) -> str:
        match = _SPEAKER_RE.search(block.split("\n", 1)[0])
        return canonical_key(match.group(1)) if match else ""

    def append_block(original_index: int, block: str) -> bool:
        nonlocal used
        normalized = canonical_key(block)
        if normalized in seen or len(selected) >= max_blocks:
            return False
        cost = len(block) + (2 if selected else 0)
        if selected and used + cost > max_chars:
            return False
        if not selected and cost > max_chars:
            block = block[:max_chars]
            cost = len(block)
        selected.append((original_index, block))
        seen.add(normalized)
        used += cost
        return True

    for original_index, block in ranked:
        if not append_block(original_index, block):
            continue
        header = block.split("\n", 1)[0]
        source_match = _SOURCE_IDS_RE.search(header)
        follows_lossless_source = block.lstrip().upper().startswith(
            ("[CLAIM", "[EVENT ", "[EVENT_FRAME", "[OPERAND")
        )
        if source_match and follows_lossless_source:
            for source_id in source_match.group(1).split(","):
                source = block_by_id.get(source_id.strip())
                if source is not None:
                    append_block(*source)
        if _CROSS_SPEAKER_ANAPHORA_RE.search(block):
            current_speaker = speaker(block)
            antecedents = [
                (index, candidate)
                for index, candidate in enumerate(blocks)
                if candidate != block
                and action_family_overlap(block, candidate) > 0
                and (
                    not current_speaker
                    or not speaker(candidate)
                    or speaker(candidate) != current_speaker
                )
            ]
            if antecedents:
                antecedent = max(
                    antecedents,
                    key=lambda item: (
                        _score_block(
                            item[1], terms=terms,
                            term_frequency=term_frequency, question=question,
                        ),
                        -item[0],
                    ),
                )
                append_block(*antecedent)
        if len(selected) >= max_blocks:
            break
    # Present coarse/typed routing evidence first and lossless grounding last.
    # This preserves coarse content as evidence while making raw source turns
    # the final authority for an omitted complement or fine attribute.
    ordered = [item for item in selected if not item[1].lstrip().upper().startswith("[TURN")]
    ordered.extend(
        item for item in selected if item[1].lstrip().upper().startswith("[TURN")
    )
    return "\n\n".join(block for _index, block in ordered)
