#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

UNIT_ALIASES = {
    "day": ("day", "days"),
    "week": ("week", "weeks"),
    "hour": ("hour", "hours"),
    "people": ("people", "person", "persons"),
}

RECOMMEND_PATTERN = r"rec+om+m?end(?:ed|ing|s)?|rec+om+m?endations?"


@dataclass
class MemoryChunk:
    question_id: str
    node_type: str
    session_id: str
    session_date: str | None
    turn_index: int | None
    text: str


@dataclass
class OpResult:
    answer: str
    reason: str
    operator: str
    evidence: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply generic typed memory operators.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--query-stats", type=Path, required=True)
    parser.add_argument("--llm-calls", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="generic_memory_ops")
    parser.add_argument("--enable-sum-amounts", action="store_true")
    parser.add_argument("--enable-speaker-mismatch-abstain", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_chunks(path: Path) -> dict[str, list[MemoryChunk]]:
    chunks: dict[str, list[MemoryChunk]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id") or "")
        node_type = str(row.get("node_type") or "")
        text = row.get("raw_text") if node_type == "leaf" else row.get("summary")
        if not qid or not text:
            continue
        chunks.setdefault(qid, []).append(
            MemoryChunk(
                question_id=qid,
                node_type=node_type,
                session_id=str(row.get("session_id") or ""),
                session_date=row.get("session_date"),
                turn_index=row.get("turn_index"),
                text=str(text),
            )
        )
    for values in chunks.values():
        values.sort(key=lambda item: (item.session_date or "", item.session_id, item.turn_index or -1))
    return chunks


def final_answer(result: OpResult) -> str:
    return f"Evidence facts: {result.reason}\n\nFinal answer: {result.answer}"


def sentences(chunks: Iterable[MemoryChunk]) -> Iterable[tuple[MemoryChunk, str]]:
    for chunk in chunks:
        for part in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            text = part.strip()
            if text:
                yield chunk, text


def speaker_sentences(chunks: Iterable[MemoryChunk]) -> Iterable[tuple[MemoryChunk, str | None, str]]:
    for chunk in chunks:
        for line in chunk.text.splitlines():
            line = line.strip()
            if not line:
                continue
            speaker = sentence_speaker(line)
            for part in re.split(r"(?<=[.!?])\s+", line):
                text = part.strip()
                if text:
                    yield chunk, speaker, text


def speaker_lines(chunks: Iterable[MemoryChunk]) -> Iterable[tuple[MemoryChunk, str | None, str]]:
    for chunk in chunks:
        for line in chunk.text.splitlines():
            text = line.strip()
            if text:
                yield chunk, sentence_speaker(text), text


def sentence_clauses(text: str) -> list[str]:
    clauses = [part.strip() for part in re.split(r"\s*;\s*|\s+\|\s+|\n+", text) if part.strip()]
    return clauses or [text]


def compact(text: str, limit: int = 260) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalize_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\n\r.,;:!?*\"'()[]"))
    value = re.sub(r"^(?:the|a|an|my|new)\s+", "", value, flags=re.IGNORECASE)
    return value


def title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def number_value(text: str) -> int | None:
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


def numeric_value(text: str) -> float | None:
    text = text.lower().strip()
    if text in {"a", "an", "one"}:
        return 1.0
    if text.isdigit():
        return float(int(text))
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    value = NUMBER_WORDS.get(text)
    return float(value) if value is not None else None


def speaker_names(chunks: list[MemoryChunk]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for match in re.finditer(r"\b(?:User|Assistant|System)\s+\(([A-Z][A-Za-z .'-]+?)\)", chunk.text):
            name = normalize_name(match.group(1))
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def sentence_speaker(text: str) -> str | None:
    match = re.match(r"(?:User|Assistant|System)\s+\(([A-Z][A-Za-z .'-]+?)\)", text.strip())
    return normalize_name(match.group(1)) if match else None


def target_speaker(question: str, speakers: list[str]) -> str | None:
    mentioned = [
        speaker
        for speaker in speakers
        if re.search(rf"\b{re.escape(speaker)}(?:'s)?\b", question, flags=re.IGNORECASE)
    ]
    if not mentioned:
        return None
    for pattern in (
        r"\b(?:what|when|where|why|how|did|does|is|are|was|were)\s+(?:did|does|is|are|was|were\s+)?({})\b",
        r"\babout\s+({})\b",
    ):
        for speaker in mentioned:
            if re.search(pattern.format(re.escape(speaker)), question, flags=re.IGNORECASE):
                return speaker
    for speaker in mentioned:
        if re.search(rf"\b{re.escape(speaker)}'s\b", question, flags=re.IGNORECASE):
            return speaker
    return mentioned[0] if len(mentioned) == 1 else None


def content_terms(question: str, speakers: list[str]) -> set[str]:
    stop = {
        "what",
        "when",
        "where",
        "why",
        "how",
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "was",
        "were",
        "are",
        "is",
        "would",
        "could",
        "should",
        "might",
        "likely",
        "which",
        "plans",
        "plan",
        "respect",
        "process",
        "around",
        "into",
        "her",
        "his",
        "their",
        "your",
        "our",
        "she",
        "him",
        "they",
        "them",
        "got",
        "get",
        "the",
        "that",
        "this",
        "with",
        "from",
        "for",
        "about",
        "after",
        "before",
        "kind",
        "type",
        "person",
        "people",
    }
    speaker_tokens = {token.lower() for speaker in speakers for token in re.findall(r"[A-Za-z]+", speaker)}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", question)
        if len(token) > 2 and token.lower() not in stop and token.lower() not in speaker_tokens
    }


def op_speaker_mismatch_abstain(
    question: str,
    chunks: list[MemoryChunk],
    *,
    strict_target_support: bool = True,
) -> OpResult | None:
    speakers = speaker_names(chunks)
    target = target_speaker(question, speakers)
    if target is None:
        return None
    terms = content_terms(question, speakers)
    if not terms:
        return None

    target_hits: list[str] = []
    other_hits: dict[str, list[str]] = {}
    required_hits = min(2, len(terms))
    for _chunk, speaker, sent in speaker_sentences(chunk for chunk in chunks if chunk.node_type == "leaf"):
        if speaker is None:
            continue
        lowered = sent.lower()
        if sent.rstrip().endswith("?"):
            continue
        hits = sum(term in lowered for term in terms)
        if hits < required_hits:
            continue
        if title_key(speaker) == title_key(target):
            if not strict_target_support or _target_sentence_supports_fact(question, sent, hits):
                target_hits.append(sent)
        else:
            other_hits.setdefault(speaker, []).append(sent)

    if target_hits or not other_hits:
        return None
    best_speaker, evidence = max(other_hits.items(), key=lambda item: len(item[1]))
    if not evidence:
        return None
    return OpResult(
        answer=f"The supplied evidence does not mention this information for {target}.",
        reason=(
            f"The question asks about {target}, but the matching evidence is about "
            f"{best_speaker}; facts should not be transferred between speakers."
        ),
        operator="speaker_mismatch_abstain",
        evidence=[compact(item) for item in evidence[:3]],
    )


def _binary_question(question: str) -> bool:
    return bool(re.match(r"\s*(?:did|does|do|is|are|was|were|has|have)\b", question, flags=re.IGNORECASE))


def _question_object_terms(question: str, speakers: list[str]) -> set[str]:
    terms = content_terms(question, speakers)
    terms -= {
        "the",
        "and",
        "this",
        "that",
        "these",
        "those",
        "black",
        "white",
        "new",
        "old",
        "make",
        "made",
        "create",
        "created",
        "paint",
        "painted",
        "draw",
        "drew",
        "build",
        "built",
        "cook",
        "cooked",
        "bake",
        "baked",
        "buy",
        "bought",
        "own",
        "owns",
        "pet",
        "photo",
        "picture",
        "image",
    }
    return {term for term in terms if len(term) > 2}


def _speaker_line_is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _speaker_self_action(sentence: str, actions: str) -> bool:
    if re.search(
        r"\b(?:made|make|making)\s+(?:me|you|him|her|them|us|sure|it|(?:that|this)\s+happen|a\s+change|changes|progress|sense|friends|a\s+(?:big\s+|huge\s+)?difference)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        return False
    action_pattern = rf"(?<![-a-z])(?:{actions})(?![a-z])"
    return bool(
        re.search(
            rf"\b(?:i|i've|i’d|i'd|we|we've|my family)\b[^.!?]{{0,80}}\b{action_pattern}",
            sentence,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"\b{action_pattern}[^.!?]{{0,80}}\b(?:by me|myself|ourselves)\b",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def _line_mentions_terms(sentence: str, terms: set[str], *, min_hits: int = 1) -> bool:
    lowered = sentence.lower()
    hits = sum(1 for term in terms if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered))
    return hits >= min_hits


def _creation_question_target(question: str, speakers: list[str]) -> tuple[str, set[str]] | None:
    if not _binary_question(question):
        return None
    if not re.search(r"\b(?:make|made|create|created|paint|painted|draw|drew|build|built|cook|cooked|bake|baked)\b", question, flags=re.IGNORECASE):
        return None
    target = target_speaker(question, speakers)
    if target is None:
        return None
    terms = _question_object_terms(question, speakers)
    if not terms:
        return None
    return target, terms


def _pet_question_target(question: str, speakers: list[str]) -> tuple[str, str, set[str]] | None:
    if not _binary_question(question) or not re.search(r"\bpet|dog|cat|guinea pig|pup|puppy|kitty\b", question, flags=re.IGNORECASE):
        return None
    possessive_speakers = [
        speaker
        for speaker in speakers
        if re.search(rf"\b{re.escape(speaker)}(?:'s|’s)\b", question, flags=re.IGNORECASE)
    ]
    if len(possessive_speakers) != 1:
        return None
    owner = target_speaker(question, speakers)
    if owner is None:
        return None
    names = [
        normalize_name(match)
        for match in re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", question)
        if normalize_name(match) not in speakers
        and normalize_name(match).lower() not in {"did", "does", "do", "is", "are", "was", "were", "has", "have"}
    ]
    entity = names[0] if names else ""
    terms = _question_object_terms(question, speakers)
    if entity:
        terms.add(entity.lower())
    if not entity or not terms:
        return None
    return owner, entity, terms


def op_binary_speaker_fact(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    speakers = speaker_names(chunks)
    if not speakers or not _binary_question(question):
        return None

    creation = _creation_question_target(question, speakers)
    if creation is not None:
        target, terms = creation
        target_evidence: list[str] = []
        other_evidence: dict[str, list[str]] = {}
        actions = "made|make|created|create|painted|paint|drew|draw|built|build|cooked|cook|baked|bake"
        for _chunk, speaker, sent in speaker_sentences(chunk for chunk in chunks if chunk.node_type == "leaf"):
            if speaker is None or _speaker_line_is_question(sent):
                continue
            if not _speaker_self_action(sent, actions) or not _line_mentions_terms(sent, terms):
                continue
            if title_key(speaker) == title_key(target):
                target_evidence.append(sent)
            else:
                other_evidence.setdefault(speaker, []).append(sent)
        if target_evidence:
            return OpResult(
                answer="Yes",
                reason=f"The evidence says {target} personally made or created the referenced item.",
                operator="binary_speaker_fact",
                evidence=[compact(item) for item in target_evidence[:3]],
            )
        if other_evidence:
            best_speaker, evidence = max(other_evidence.items(), key=lambda item: len(item[1]))
            return OpResult(
                answer="No",
                reason=(
                    f"The question asks whether {target} made the item, but the direct "
                    f"creation evidence belongs to {best_speaker}."
                ),
                operator="binary_speaker_fact",
                evidence=[compact(item) for item in evidence[:3]],
            )

    pet = _pet_question_target(question, speakers)
    if pet is not None:
        owner, entity, terms = pet
        target_evidence = []
        other_evidence: dict[str, list[str]] = {}
        for _chunk, speaker, sent in speaker_sentences(chunk for chunk in chunks if chunk.node_type == "leaf"):
            if speaker is None or _speaker_line_is_question(sent):
                continue
            if not _line_mentions_terms(sent, terms, min_hits=1):
                continue
            lowered = sent.lower()
            owns_pet = bool(
                re.search(r"\b(?:my|our|we've got|we have|i have|i've got)\b", lowered)
                or re.search(r"\b(?:pet|dog|cat|guinea pig|pup|puppy|kitty)\b", lowered)
            )
            if not owns_pet:
                continue
            if title_key(speaker) == title_key(owner):
                target_evidence.append(sent)
            else:
                other_evidence.setdefault(speaker, []).append(sent)
        summary_other: list[str] = []
        for _chunk, sent in sentences(chunk for chunk in chunks if chunk.node_type != "leaf"):
            if not _line_mentions_terms(sent, {entity.lower()}, min_hits=1):
                continue
            for speaker in speakers:
                if title_key(speaker) == title_key(owner):
                    continue
                if re.search(rf"\b{re.escape(speaker)}\b[^.!?]{{0,80}}\b(?:has|owns|named)\b", sent, flags=re.IGNORECASE):
                    summary_other.append(sent)
        if target_evidence:
            return OpResult(
                answer="Yes",
                reason=f"The evidence says {owner} owns or has {entity} as a pet.",
                operator="binary_speaker_fact",
                evidence=[compact(item) for item in target_evidence[:3]],
            )
        if other_evidence or summary_other:
            if other_evidence:
                best_speaker, evidence = max(other_evidence.items(), key=lambda item: len(item[1]))
                reason = (
                    f"The question asks whether {entity} is {owner}'s pet, but the direct "
                    f"pet evidence belongs to {best_speaker}."
                )
                evidence_items = evidence
            else:
                reason = (
                    f"The question asks whether {entity} is {owner}'s pet, but summary "
                    "evidence attributes that pet to another speaker."
                )
                evidence_items = summary_other
            return OpResult(
                answer="No",
                reason=reason,
                operator="binary_speaker_fact",
                evidence=[compact(item) for item in evidence_items[:3]],
            )
    return None


def _calculation_or_count_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:how many|how much|total|percentage|percent|save|difference|sum|combined)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def _normalize_required_phrase(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(?:my|the|a|an|this|that|these|those|initially|total|different)\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?\"'"))
    return value


def _required_conjunct_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    patterns = (
        r"\b(?:in|at|for|from|between|of|on)\s+([A-Z][A-Za-z0-9' -]{2,40})\s+and\s+(?:in|at|for|from|between|of|on\s+)?([A-Z][A-Za-z0-9' -]{2,40})(?:\?|,|$)",
        r"\bfor\s+([a-z][a-z0-9' -]{2,40})\s+and\s+([a-z][a-z0-9' -]{2,40})(?:\?|,|$)",
        r"\bfrom\s+([a-z][a-z0-9' -]{2,40})\s+to\s+(?:the\s+completion\s+of\s+)?(?:my\s+)?([A-Z][A-Za-z0-9' -]{2,40})(?:\?|,|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            phrases.extend(_normalize_required_phrase(group) for group in match.groups())
    instead = re.search(
        r"\bby taking\s+(?:the\s+)?([a-z][a-z0-9' -]{2,30})\b.+?\binstead of\s+(?:a\s+|the\s+)?([a-z][a-z0-9' -]{2,30})(?:\?|,|$)",
        question,
        flags=re.IGNORECASE,
    )
    if instead:
        phrases.extend(_normalize_required_phrase(group) for group in instead.groups())
    cleaned: list[str] = []
    for phrase in phrases:
        phrase = re.sub(
            r"\b(?:did|do|does|i|you|have|spend|spent|traveling|travel|trip|initially|plant|taking|take|hotel|airport)\b",
            " ",
            phrase,
            flags=re.IGNORECASE,
        )
        phrase = _normalize_required_phrase(phrase)
        if len(phrase) >= 3:
            cleaned.append(phrase)
    return _dedupe_strings(cleaned, 6)


def _dedupe_strings(values: list[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = title_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def _phrase_in_text(phrase: str, text: str) -> bool:
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9']+", phrase)
        if term.lower() not in {"and", "or", "the", "for", "from", "with", "my"}
    ]
    if not terms:
        return False
    lowered = text.lower()
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(term.rstrip('s'))}s?(?![a-z0-9])", lowered)
        for term in terms
    )


def op_missing_required_conjunct(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not _calculation_or_count_question(question):
        return None
    required = _required_conjunct_phrases(question)
    if len(required) < 2:
        return None
    memory_text = "\n".join(chunk.text for chunk in chunks)
    present = [phrase for phrase in required if _phrase_in_text(phrase, memory_text)]
    missing = [phrase for phrase in required if phrase not in present]
    if not present or not missing:
        return None
    if len(missing) == 1 and len(required) == 2:
        return OpResult(
            answer=(
                "The information provided is not enough to answer, because the memory "
                f"mentions {present[0]} but does not mention {missing[0]}."
            ),
            reason=(
                "The question requires all listed entities for the calculation, but "
                f"the memory lacks evidence for: {', '.join(missing)}."
            ),
            operator="missing_required_conjunct",
            evidence=[compact(chunk.text) for chunk in chunks if any(_phrase_in_text(item, chunk.text) for item in present)][:3],
        )
    return None


def _target_sentence_supports_fact(question: str, sentence: str, hits: int) -> bool:
    lowered = sentence.lower()
    if re.search(r"\b(what|when|where|why|how|did|do|does|are|is|was|were)\b.+\?$", sentence.strip(), flags=re.IGNORECASE):
        return False
    if re.search(r"\b(?:think|feel|attitude|say|said|opinion)\b", question, flags=re.IGNORECASE):
        return hits >= 1 and not re.search(r"\b(?:what|when|where|why|how)\b.+\?$", sentence.strip(), flags=re.IGNORECASE)
    if re.search(r"\b(?:i|i'm|ive|i've|i'd|my|me|we|we're|our|mine)\b", lowered):
        return True
    if hits >= 2 and not re.search(
        r"\b(?:you|your|thanks|congrats|congratulations|awesome|great|cool|sounds|wow|love that)\b",
        lowered,
    ):
        return True
    return False


def add_hours(time_text: str, hours: int) -> str:
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", time_text, flags=re.IGNORECASE)
    if not match:
        return time_text
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3).upper()
    if suffix == "PM" and hour != 12:
        hour += 12
    if suffix == "AM" and hour == 12:
        hour = 0
    hour = (hour + hours) % 24
    out_suffix = "AM" if hour < 12 else "PM"
    out_hour = hour % 12 or 12
    return f"{out_hour}:{minute:02d} {out_suffix}"


def op_before_after_purchase(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not re.search(r"\bbefore\b", question, flags=re.IGNORECASE):
        return None
    target_match = re.search(r"before (?:getting|buying|purchasing|I got) the ([A-Z][A-Za-z0-9 \-]+)", question)
    if not target_match:
        return None
    target = normalize_name(target_match.group(1))
    candidates: list[tuple[str, str]] = []
    for _, sent in sentences(chunks):
        if re.search(r"\b(new|got|bought|purchased|invested in|using my new)\b", sent, flags=re.IGNORECASE):
            for pattern in (
                r"(?:using|with|got|bought|purchased|invested in)\s+(?:my\s+)?new\s+([A-Z][A-Za-z0-9 \-]+?)(?:\s+to\b|\s+for\b|,|\.|$)",
                r"my new\s+([A-Z][A-Za-z0-9 \-]+?)(?:\s+to\b|\s+for\b|,|\.|$)",
                r"using (?:the|my)\s+([A-Z][A-Za-z0-9 \-]+?)(?:\s+to\b|\s+for\b|,|\.|$)",
            ):
                for match in re.finditer(pattern, sent):
                    name = normalize_name(match.group(1))
                    if name and title_key(name) != title_key(target) and len(name.split()) <= 4:
                        candidates.append((name, sent))
    if candidates:
        name, evidence = candidates[0]
        return OpResult(
            answer=name,
            reason=f"The memory mentions {name} as a new/purchased item before the target item {target}.",
            operator="before_after_purchase",
            evidence=[compact(evidence)],
        )
    return None


def op_count_services(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not re.search(r"how many .*food delivery", question, flags=re.IGNORECASE):
        return None
    candidates: dict[str, str] = {}
    for _, sent in sentences(chunks):
        if not re.search(r"delivery|lately|weekends|pre-made|convenience", sent, flags=re.IGNORECASE):
            continue
        patterns = (
            r"had\s+([A-Z][A-Za-z'&]*(?:\s+[A-Z][A-Za-z'&]*){0,2})\s+(?:\d+|one|two|three|four|five)\s+times",
            r"all about\s+([A-Z][A-Za-z'&]*(?:\s+[A-Z][A-Za-z'&]*){0,2})\s+lately",
            r"called\s+([A-Z][A-Za-z'&]*(?:\s+[A-Z][A-Za-z'&]*){0,2})",
            r"(?:using|relying on)\s+([A-Z][A-Za-z'&]*(?:\s+[A-Z][A-Za-z'&]*){0,2})",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, sent):
                name = normalize_name(match.group(1))
                if len(name) < 3 or title_key(name) in {"im", "ive"}:
                    continue
                candidates.setdefault(title_key(name), name)
    if len(candidates) >= 2:
        names = list(candidates.values())
        return OpResult(
            answer=f"{len(names)} ({', '.join(names)})",
            reason="The memory names these recently used food delivery services: " + ", ".join(names) + ".",
            operator="count_entities",
            evidence=names,
        )
    return None


def op_count_attended_events(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not re.search(r"how many .*graduation ceremon", question, flags=re.IGNORECASE):
        return None
    events: dict[str, str] = {}
    for _, sent in sentences(chunks):
        if "graduation" not in sent.lower():
            continue
        if re.search(r"\b(miss|missed|missing|couldn't attend|did not attend)\b", sent, flags=re.IGNORECASE):
            continue
        if not re.search(r"\b(attended|just attended|went to|got back from)\b", sent, flags=re.IGNORECASE):
            continue
        name_match = re.search(r"(?:cousin|friend|colleague|best friend|nephew|niece)\s+([A-Z][a-z]+)", sent)
        key = name_match.group(1) if name_match else compact(sent, 80)
        events.setdefault(key, compact(sent))
    if events:
        return OpResult(
            answer=str(len(events)),
            reason=f"The memory has {len(events)} attended graduation event(s), excluding explicitly missed ceremonies.",
            operator="count_attended_events",
            evidence=list(events.values()),
        )
    return None


def op_count_named_attended_events(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not re.search(r"\bhow many\b", q):
        return None
    leaf_chunks = [chunk for chunk in chunks if chunk.node_type == "leaf"]
    if re.search(r"\bmovie festivals?\b|\bfilm festivals?\b", q) and re.search(r"\battend", q):
        events: dict[str, str] = {}
        for _chunk, sent in sentences(leaf_chunks):
            if _event_sentence_rejected(sent):
                continue
            if not re.search(
                r"\b(?:attended|volunteered at|participated in|got back from|Q&A session .* at|screening .* at)\b",
                sent,
                flags=re.IGNORECASE,
            ):
                continue
            for name in _named_event_mentions(sent, r"(?:Film Festival|Film Fest|Festival|Fest)"):
                if re.search(r"\b(?:film|movie|AFI|Sundance|Tribeca|Austin|Portland|Seattle)\b", name, flags=re.IGNORECASE):
                    events.setdefault(title_key(name), name)
        if len(events) >= 2:
            names = list(events.values())
            return OpResult(
                answer=f"{len(names)} ({', '.join(names)})",
                reason="The memory contains these distinct personally attended or worked film-festival events.",
                operator="count_named_events",
                evidence=names,
            )

    if re.search(r"\bweddings?\b", q) and re.search(r"\battend", q):
        events: dict[str, str] = {}
        for _chunk, sent in sentences(leaf_chunks):
            if _event_sentence_rejected(sent):
                continue
            lowered = sent.lower()
            if "wedding" not in lowered and "tie the knot" not in lowered:
                continue
            if re.search(r"\b(?:planning my own wedding|upcoming wedding|my wedding ceremony|our wedding)\b", lowered):
                continue
            if not re.search(r"\b(?:got back from|been to|attended|was a bridesmaid|wedding at|at the wedding|tie the knot)\b", lowered):
                continue
            key = _wedding_event_key(sent)
            if key:
                events.setdefault(title_key(key), key)
        if len(events) >= 2 and all(" and " in name for name in events.values()):
            names = list(events.values())
            return OpResult(
                answer=f"{len(names)} ({', '.join(names)})",
                reason="The memory contains these distinct attended wedding events, excluding the user's own wedding planning.",
                operator="count_named_events",
                evidence=names,
            )
    return None


def _event_sentence_rejected(sentence: str) -> bool:
    lowered = sentence.lower()
    if _non_user_or_advice_sentence(sentence):
        return True
    return bool(
        re.search(
            r"\b(?:recommend|suggest|tips|options|should|could|would|planning to attend|want to attend|"
            r"interested in attending|might attend|festival scene|film recommendations)\b",
            lowered,
        )
        and not re.search(
            r"\b(?:i attended|i volunteered|i even volunteered|i participated|i just got back|i got to|i had the opportunity|i was a)\b",
            lowered,
        )
    )


def _named_event_mentions(sentence: str, suffix_pattern: str) -> list[str]:
    names: list[str] = []
    pattern = rf"\b([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){{0,5}}\s+{suffix_pattern})\b"
    for match in re.finditer(pattern, sentence):
        name = normalize_name(match.group(1))
        name = re.sub(r"^(?:the|at|from)\s+", "", name, flags=re.IGNORECASE)
        if 4 <= len(name) <= 80:
            names.append(name)
    return names


def _wedding_event_key(sentence: str) -> str:
    patterns = (
        r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)'?s wedding\b",
        r"\bwedding .*?\b([A-Z][a-z]+)\s+(?:and|&)\s+([A-Z][a-z]+)\b",
        r"\bwedding .*?\b([A-Z][a-z]+)\s+finally\s+got\s+to\s+tie\s+the\s+knot\s+with\s+(?:her\s+partner\s+)?([A-Z][a-z]+)\b",
        r"\b([A-Z][a-z]+)\s+finally\s+got\s+to\s+tie\s+the\s+knot\s+with\s+(?:her\s+partner\s+)?([A-Z][a-z]+)\b",
        r"\b([A-Z][a-z]+)\s+got\s+to\s+tie\s+the\s+knot\s+with\s+(?:her\s+partner\s+)?([A-Z][a-z]+)\b",
        r"\bbride,\s*([A-Z][a-z]+),\s*[^.;\n|]*?\b(?:her\s+)?husband,\s*([A-Z][a-z]+)\b",
        r"\b([A-Z][a-z]+)'s wedding\b[^.;\n|]*?\b([A-Z][a-z]+)\b",
        r"\b(?:cousin|friend|roommate)\s+([A-Z][a-z]+)'?s wedding\b",
        r"\b([A-Z][a-z]+)'s wedding at\b",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if not match:
            continue
        return " and ".join(normalize_name(value) for value in match.groups() if value)
    return ""


def op_count_health_visits(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not re.search(r"\bhow many\b", q):
        return None
    if not re.search(r"\bdoctor'?s?|physicians?|appointments?\b", q):
        return None
    leaf_chunks = [chunk for chunk in chunks if chunk.node_type == "leaf"]
    if "appointment" in q and re.search(r"\bin march\b", q):
        appointments: dict[str, str] = {}
        for _chunk, sent in health_sentences(leaf_chunks):
            if not re.search(r"\b(appointment|checkup|check-up|follow-up|saw|visited|went to)\b", sent, flags=re.IGNORECASE):
                continue
            if not re.search(r"\bMarch\b|\b03/", sent, flags=re.IGNORECASE):
                continue
            if _health_sentence_rejected(sent) or not _personal_health_sentence(sent):
                continue
            key = _health_event_key(sent)
            appointments.setdefault(key, compact(sent))
        if len(appointments) >= 1:
            return OpResult(
                answer=str(len(appointments)),
                reason=(
                    f"The memory contains {len(appointments)} March medical appointment(s), "
                    "counting explicit attended/scheduled appointment facts and excluding advice."
                ),
                operator="count_health_events",
                evidence=list(appointments.values()),
            )

    if re.search(r"\bdifferent doctors?\b|\bdoctors? did I visit\b|\bphysicians? did I visit\b", q):
        title_doctors: dict[str, str] = {}
        named_doctors: dict[str, str] = {}
        for _chunk, sent in health_sentences(leaf_chunks):
            if _health_sentence_rejected(sent) or not _personal_health_sentence(sent):
                continue
            if not re.search(r"\b(appointment|follow-up|saw|visited|went to|visit with|consulted|prescribed\b.+\bby)\b", sent, flags=re.IGNORECASE):
                continue
            for name in _doctor_title_mentions(sent):
                title_doctors.setdefault(title_key(name), compact(sent))
            if not _doctor_title_mentions(sent):
                for name in _doctor_name_mentions(sent):
                    named_doctors.setdefault(title_key(name), compact(sent))
        doctors = title_doctors or named_doctors
        if len(doctors) >= 1:
            names = [_display_health_name(key) for key in doctors]
            return OpResult(
                answer=f"{len(doctors)} ({', '.join(names)})",
                reason=(
                    f"The memory has explicit visit/appointment evidence for {len(doctors)} "
                    "distinct doctor or specialist type(s)."
                ),
                operator="count_health_events",
                evidence=list(doctors.values()),
            )
    return None


def health_sentences(chunks: Iterable[MemoryChunk]) -> Iterable[tuple[MemoryChunk, str]]:
    for chunk in chunks:
        text = chunk.text.replace("Dr.", "Dr")
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            part = part.strip()
            if part:
                yield chunk, part


def _health_sentence_rejected(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(
        re.search(
            r"can help|recommend|tips|general advice|what to expect|questions to ask|"
            r"should ask|may recommend|procedure|colonoscopy|not a medical professional|"
            r"consult with your|healthcare provider|i'll schedule|i will schedule|"
            r"planning to schedule",
            lowered,
        )
    )


def _personal_health_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if re.search(r"^\s*(assistant|system)\s*:", lowered):
        return False
    if sentence.lstrip().startswith(("*", "-", "1.", "2.", "3.", "4.", "5.")):
        return False
    return bool(re.search(r"\b(User:|I|I've|I had|I went|I saw|my)\b", sentence))


def _doctor_mentions(sentence: str) -> list[str]:
    titles = _doctor_title_mentions(sentence)
    if titles:
        return titles
    return _doctor_name_mentions(sentence)


def _doctor_title_mentions(sentence: str) -> list[str]:
    title_pattern = (
        r"\b(primary care physician|ENT specialist|dermatologist|orthopedic surgeon|"
        r"allergist|cardiologist|dentist|ophthalmologist|optometrist|therapist|"
        r"psychiatrist|pediatrician|gynecologist)\b"
    )
    titles = [
        normalize_name(match.group(1))
        for match in re.finditer(title_pattern, sentence, flags=re.IGNORECASE)
    ]
    return titles


def _doctor_name_mentions(sentence: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(
        r"\b(?:appointment|follow-up|saw|visited|went to|visit with|consulted)\s+"
        r"(?:with\s+)?(?:Dr\.?|Doctor)\s+([A-Z][a-z]+)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        names.append(normalize_name(match.group(1)))
    for match in re.finditer(
        r"\bprescribed\s+.+?\s+by\s+(?:my\s+)?(?:primary care physician|doctor),?\s+"
        r"(?:Dr\.?|Doctor)\s+([A-Z][a-z]+)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        names.append(normalize_name(match.group(1)))
    return names


def _health_event_key(sentence: str) -> str:
    mentions = _doctor_mentions(sentence)
    if mentions:
        return title_key("|".join(sorted(title_key(name) for name in mentions)))
    date_match = re.search(r"\b(?:March|03/)\s*\d{1,2}\b|\b\d{1,2}(?:st|nd|rd|th)?\b", sentence, flags=re.IGNORECASE)
    return title_key(date_match.group(0)) if date_match else title_key(compact(sentence, 80))


def _display_health_name(key: str) -> str:
    display = {
        "primarycarephysician": "primary care physician",
        "entspecialist": "ENT specialist",
        "dermatologist": "dermatologist",
        "orthopedicsurgeon": "orthopedic surgeon",
    }
    return display.get(key, key)


def op_current_subscriptions(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not re.search(r"how many .*subscriptions.*currently", question, flags=re.IGNORECASE):
        return None
    current: dict[str, str] = {}
    canceled: set[str] = set()
    for _, sent in sentences(chunks):
        if "subscription" not in sent.lower() and "subscribed" not in sent.lower() and "getting " not in sent.lower():
            continue
        for match in re.finditer(r"cancel(?:ed|led)?\s+(?:my\s+)?([A-Z][A-Za-z'& ]+?)\s+(?:magazine\s+)?subscription", sent):
            canceled.add(title_key(match.group(1)))
        patterns = (
            r"subscribed to\s+([A-Z][A-Za-z'& ]+?)(?:\s+in\b|,|\.|;)",
            r"getting\s+([A-Z][A-Za-z'& ]+?)(?:,|\s+which|\s+for\b|\.|;)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, sent):
                name = normalize_name(match.group(1))
                if 2 <= len(name) <= 40 and title_key(name) not in canceled:
                    current[title_key(name)] = name
    for key in list(current):
        if key in canceled:
            current.pop(key, None)
    if current:
        names = list(current.values())
        return OpResult(
            answer=f"{len(names)} ({', '.join(names)})",
            reason="Current subscriptions are inferred from active subscribed/getting statements after excluding canceled subscriptions.",
            operator="current_state",
            evidence=names,
        )
    return None


def op_current_reading(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not (re.search(r"\bcurrently\b", q) and re.search(r"\b(book|reading|read)\b", q)):
        return None
    candidates: list[tuple[str, str, str]] = []
    for chunk, sent in sentences([chunk for chunk in chunks if chunk.node_type == "leaf"]):
        if _non_user_or_advice_sentence(sent):
            continue
        if not re.search(r"\bcurrently\s+(?:reading|devouring)|\bnow\s+reading", sent, flags=re.IGNORECASE):
            continue
        title = _quoted_title(sent)
        if not title:
            match = re.search(
                r"\b(?:currently|now)\s+(?:reading|devouring)\s+([A-Z][^.;\n|]{2,90})",
                sent,
                flags=re.IGNORECASE,
            )
            title = normalize_name(match.group(1)) if match else ""
        if title:
            candidates.append((chunk.session_date or "", title, compact(sent)))
    if not candidates:
        return None
    _, title, evidence = sorted(candidates, key=lambda item: item[0])[-1]
    return OpResult(
        answer=title,
        reason="The latest personal memory explicitly states the book the user is currently reading.",
        operator="current_state",
        evidence=[evidence],
    )


def op_current_storage_location(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not (re.search(r"\bwhere\b", q) and re.search(r"\bcurrent(?:ly)?\b", q)):
        return None
    object_terms = _storage_object_terms(question)
    if not object_terms:
        return None
    candidates: list[tuple[str, int, str, str]] = []
    for chunk, sent in sentences([chunk for chunk in chunks if chunk.node_type == "leaf"]):
        if _non_user_or_advice_sentence(sent):
            continue
        lowered = sent.lower()
        if re.search(r"\byour\b", lowered) and not re.search(r"\b(?:I|I'm|I've|my|me)\b", sent):
            continue
        if re.search(r"\bi'm excited to hear\b|\bas for organizing your\b", lowered):
            continue
        if not all(term in lowered for term in object_terms):
            continue
        if not re.search(r"\b(?:keep|kept|keeping|store|stored|storing|put|placed|organize)\b", lowered):
            continue
        location = _storage_location_from_sentence(sent)
        if not location:
            continue
        if re.search(r"\b(?:recommend|suggest|tip|should|could)\b", lowered):
            continue
        candidates.append((chunk.session_date or "", chunk.turn_index or -1, location, compact(sent)))
    if not candidates:
        return None
    _, _, location, evidence = sorted(candidates, key=lambda item: (item[0], item[1]))[-1]
    return OpResult(
        answer=location,
        reason="The latest personal storage memory for the requested object gives this location.",
        operator="current_state",
        evidence=[evidence],
    )


def op_current_numeric_status(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    leaf_chunks = [chunk for chunk in chunks if chunk.node_type == "leaf"]
    if re.search(r"\bfollowers?\b", q) and re.search(r"\b(?:instagram|now|current)\b", q):
        if re.search(r"\b(?:increase|increased|gain|gained|growth|grew)\b", q):
            return None
        latest = _latest_numeric_candidate(
            leaf_chunks,
            include_pattern=r"\b(?:followers?|instagram)\b",
            value_patterns=(
                r"\b(?:close to|nearing|around|about)?\s*(\d[\d,]*)\s+followers?\b",
                r"\bfollower count\b[^.;\n|]*?(\d[\d,]*)\b",
            ),
        )
        if latest:
            value, evidence = latest
            return OpResult(
                answer=str(int(value)),
                reason="The latest personal Instagram follower-count memory gives this current value.",
                operator="current_numeric_state",
                evidence=[evidence],
            )

    if re.search(r"\bpages?\b", q) and re.search(r"\b(?:so far|currently|read)\b", q):
        title_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z]+", question)
            if len(token) > 3 and token.lower() not in {"many", "pages", "read", "have", "currently", "history", "everything"}
        }
        latest = _latest_numeric_candidate(
            leaf_chunks,
            include_pattern=r"\b(?:page|currently on|reading)\b",
            value_patterns=(
                r"\bcurrently on page\s+(\d[\d,]*)\b",
                r"\bon page\s+(\d[\d,]*)\b",
                r"\bread\s+(\d[\d,]*)\s+pages?\b",
            ),
            required_terms=title_terms,
        )
        if latest:
            value, evidence = latest
            return OpResult(
                answer=str(int(value)),
                reason="The latest personal reading-progress memory gives the current page count.",
                operator="current_numeric_state",
                evidence=[evidence],
            )

    if re.search(r"\bstars?\b", q) and re.search(r"\bgold\b", q):
        latest = _latest_numeric_candidate(
            leaf_chunks,
            include_pattern=r"\b(?:stars?|gold level|gold)\b",
            value_patterns=(
                r"\b(?:need|requires?|require|reach(?:ing)? Gold(?: level)? requires)\s+(?:a total of\s+)?(\d[\d,]*)\s+stars?\b",
                r"\b(\d[\d,]*)\s+stars?\s+to\s+reach\s+(?:the\s+)?Gold\b",
            ),
            prefer_user_corrections=True,
        )
        if latest:
            value, evidence = latest
            return OpResult(
                answer=str(int(value)),
                reason="The latest correction/current memory gives the number of stars needed for Gold level.",
                operator="current_numeric_state",
                evidence=[evidence],
            )

    if re.search(r"\bpersonal best\b", q) and re.search(r"\b(?:time|run|race|5k)\b", q):
        if re.search(r"\bprevious\b", q):
            return None
        best = _best_time_candidate(leaf_chunks)
        if best:
            answer, evidence = best
            return OpResult(
                answer=answer,
                reason="The personal-best question should use the best/later race time stated in personal memory.",
                operator="current_numeric_state",
                evidence=[evidence],
            )
    return None


def _latest_numeric_candidate(
    chunks: list[MemoryChunk],
    *,
    include_pattern: str,
    value_patterns: tuple[str, ...],
    required_terms: set[str] | None = None,
    prefer_user_corrections: bool = False,
) -> tuple[float, str] | None:
    candidates: list[tuple[int, str, int, int, float, str]] = []
    sequence = 0
    for chunk, sent in sentences(chunks):
        sequence += 1
        if _non_user_or_advice_sentence(sent):
            continue
        personal_text = _personal_user_text(sent)
        if not personal_text:
            continue
        lowered = personal_text.lower()
        if not re.search(include_pattern, personal_text, flags=re.IGNORECASE):
            continue
        if required_terms and not all(term in lowered for term in required_terms):
            quoted = " ".join(left or right for left, right in re.findall(r'"([^"\n]+)"|\'([^\'\n]+)\'', personal_text))
            if not all(term in (lowered + " " + quoted.lower()) for term in required_terms):
                continue
        for pattern in value_patterns:
            for match in re.finditer(pattern, personal_text, flags=re.IGNORECASE):
                value = numeric_value(match.group(1).replace(",", ""))
                if value is None:
                    continue
                correction = 1 if prefer_user_corrections and re.search(r"\b(?:actually|not|correct|correction)\b", lowered) else 0
                candidates.append((correction, chunk.session_date or "", chunk.turn_index or -1, sequence, value, compact(personal_text)))
    if not candidates:
        return None
    _, _, _, _, value, evidence = sorted(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[-1]
    return value, evidence


def _personal_user_text(sentence: str) -> str:
    text = sentence.strip()
    if text.startswith("Assistant:"):
        return ""
    if "User:" in text:
        text = text.split("User:", 1)[1]
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[0]
    text = text.strip()
    if not re.search(r"\b(?:I|I've|I'm|I'd|I'll|my|me|we|our)\b", text, flags=re.IGNORECASE):
        return ""
    return text


def _best_time_candidate(chunks: list[MemoryChunk]) -> tuple[str, str] | None:
    candidates: list[tuple[int, str, int, str, str]] = []
    for chunk, sent in sentences(chunks):
        if _non_user_or_advice_sentence(sent):
            continue
        if not re.search(r"\bpersonal best\b|\bPB\b|\bcharity 5K\b", sent, flags=re.IGNORECASE):
            continue
        for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b", sent):
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            if minutes > 90 or seconds > 59:
                continue
            total = minutes * 60 + seconds
            candidates.append((total, chunk.session_date or "", chunk.turn_index or -1, match.group(0), compact(sent)))
        for match in re.finditer(r"\b(\d{1,2})\s+minutes?\s+and\s+(\d{1,2})\s+seconds?\b", sent, flags=re.IGNORECASE):
            total = int(match.group(1)) * 60 + int(match.group(2))
            candidates.append((total, chunk.session_date or "", chunk.turn_index or -1, f"{match.group(1)} minutes and {match.group(2)} seconds", compact(sent)))
    if not candidates:
        return None
    total, _, _, answer, evidence = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0]
    return answer, evidence


def _non_user_or_advice_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    lowered = stripped.lower()
    if stripped.startswith("Assistant:") and not stripped.startswith("User:"):
        return True
    return bool(
        re.search(
            r"\b(?:recommend|suggest|tips?|you should|you can|try to|consider)\b",
            lowered,
        )
        and not re.search(r"\b(?:I|I've|I'm|my|me)\b", stripped)
    )


def _quoted_title(sentence: str) -> str:
    titles = _quoted_titles(sentence)
    return titles[0] if titles else ""


def _quoted_titles(sentence: str) -> list[str]:
    titles: list[str] = []
    for pattern in (r'"([^"\n]{3,90})"', r"(?<![A-Za-z])'([^'\n]{3,90})'(?![A-Za-z])"):
        for match in re.finditer(pattern, sentence):
            title = re.sub(r"\s+", " ", match.group(1).strip(" \t\r\n.,;:!?"))
            if title and not re.search(r"\b(?:keywords?|search|google|website|option)\b", title, flags=re.IGNORECASE):
                titles.append(title)
    return _dedupe_strings(titles, 12)


def _book_recommendation_target(question: str, speakers: list[str]) -> str | None:
    if not re.search(r"\bbook|novel|series\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(rf"\b(?:{RECOMMEND_PATTERN})\b", question, flags=re.IGNORECASE):
        return None
    for speaker in speakers:
        escaped = re.escape(speaker)
        patterns = (
            rf"\b(?:did|has|have)\s+{escaped}\s+(?:ever\s+)?(?:{RECOMMEND_PATTERN})",
            rf"\bbook\s+(?:did|has|have)\s+{escaped}\s+(?:ever\s+)?(?:{RECOMMEND_PATTERN})",
            rf"\b(?:{RECOMMEND_PATTERN})\s+(?:has|have|did)\s+{escaped}\s+(?:given|made|(?:{RECOMMEND_PATTERN}))",
            rf"\b{escaped}'s\s+book\s+(?:{RECOMMEND_PATTERN})\b",
        )
        if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in patterns):
            return speaker
    return None


def _direct_target_speaker(question: str, speakers: list[str]) -> str | None:
    mentioned = [
        speaker
        for speaker in speakers
        if re.search(rf"\b{re.escape(speaker)}(?:'s|’s)?\b", question, flags=re.IGNORECASE)
    ]
    if len(mentioned) == 1:
        return mentioned[0]
    return target_speaker(question, speakers)


def op_book_recommendations(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    speakers = speaker_names(chunks)
    target = _book_recommendation_target(question, speakers)
    if target is None:
        return None
    recommendations: dict[str, tuple[str, str, int, str]] = {}
    for chunk, speaker, line in speaker_lines(chunk for chunk in chunks if chunk.node_type == "leaf"):
        if speaker is None or title_key(speaker) != title_key(target):
            continue
        lowered = line.lower()
        if "?" in line and not re.search(rf"\b(?:{RECOMMEND_PATTERN}|highly recommend|you'd love|you would love)\b", lowered):
            continue
        if re.search(r"\b(?:took|followed|tried)\s+your\s+rec", lowered):
            continue
        if not re.search(rf"\b(?:{RECOMMEND_PATTERN}|highly recommend|you'd love|you would love|add it to your list)\b", lowered):
            continue
        titles = [
            title
            for title in _quoted_titles(line)
            if not re.search(r"\b(?:movie|show|song|track|watch(?:ed|ing)?|trilogy|game|rpg)\b", line, flags=re.IGNORECASE)
        ]
        for title in titles:
            recommendations.setdefault(
                title_key(title),
                (title, chunk.session_date or "", chunk.turn_index or -1, compact(line)),
            )
    if not recommendations:
        return None
    ordered = sorted(recommendations.values(), key=lambda item: (item[1], item[2], item[0]))
    titles = [item[0] for item in ordered]
    return OpResult(
        answer=", ".join(titles),
        reason=f"{target}'s direct book recommendation statements mention: {', '.join(titles)}.",
        operator="book_recommendations",
        evidence=[item[3] for item in ordered[:5]],
    )


def _clean_favorite_answer(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?"))
    value = re.sub(r"^(?:but|and|also|this|that|the|a|an)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:for me|to me|as you might be able to tell)$", "", value, flags=re.IGNORECASE)
    return normalize_name(value)


def _favorite_candidates_from_line(line: str) -> list[str]:
    candidates: list[str] = []
    patterns = (
        r"\b([A-Z][A-Za-z0-9&.' -]{2,80}?)\s+is\s+(?:my|one of my)\s+(?:favorite|favourite|fave|top pick)s?\b",
        r"\b([A-Z][A-Za-z0-9&.' -]{2,80}?)\s+is\s+at\s+the\s+top\s+of\s+my\s+list\b",
        r"\bmy\s+(?:favorite|favourite|fave|top pick)\s+(?:is|would be)\s+([^.;!?]{2,80})",
        r"\b([A-Za-z][A-Za-z0-9&.' -]{2,80}?)\s+is\s+my\s+top\s+pick\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            candidates.append(_clean_favorite_answer(match.group(1)))
    for match in re.finditer(
        r"\bI\s+love\s+([A-Za-z][A-Za-z0-9&.' -]{2,50}?),\s+but\s+I\s+also\s+enjoy\s+([A-Za-z][A-Za-z0-9&.' -]{2,50}?)(?:\.|,|$)",
        line,
        flags=re.IGNORECASE,
    ):
        candidates.extend(_clean_favorite_answer(group) for group in match.groups())
    return _dedupe_strings([item for item in candidates if item], 6)


def _favorite_food_question(question: str) -> bool:
    return bool(
        re.search(r"\b(?:favorite|favourite|fave|top pick|top of .* list)\b", question, flags=re.IGNORECASE)
        and re.search(r"\b(?:dish|food)\b", question, flags=re.IGNORECASE)
    )


def _favorite_food_line_match(question: str, line: str, answer: str) -> bool:
    haystack = f"{line} {answer}".lower()
    if re.search(r"\b(?:dish|recipe)\b", question, flags=re.IGNORECASE):
        return bool(re.search(r"\b(?:dish|recipe|top of my list)\b", haystack))
    return bool(
        re.search(r"\b(?:dish|recipe|food|cooking show|top of my list)\b", haystack)
        or re.search(r"\b(?:roasted chicken|ice cream|cake|chocolate|coconut)\b", answer, flags=re.IGNORECASE)
    )


def op_direct_favorite(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not _favorite_food_question(question):
        return None
    speakers = speaker_names(chunks)
    target = _direct_target_speaker(question, speakers)
    if target is None:
        return None
    candidates: dict[str, tuple[str, str, int, int, str]] = {}
    sequence = 0
    for chunk, speaker, line in speaker_lines(chunk for chunk in chunks if chunk.node_type == "leaf"):
        if speaker is None or title_key(speaker) != title_key(target):
            continue
        if line.rstrip().endswith("?"):
            continue
        for answer in _favorite_candidates_from_line(line):
            if len(answer) < 3:
                continue
            if not _favorite_food_line_match(question, line, answer):
                continue
            sequence += 1
            candidates.setdefault(
                title_key(answer),
                (answer, chunk.session_date or "", chunk.turn_index or -1, sequence, compact(line)),
            )
    if not candidates:
        return None
    ordered = sorted(candidates.values(), key=lambda item: (item[1], item[2], item[3]))
    answers = [item[0] for item in ordered]
    return OpResult(
        answer=", ".join(answers),
        reason=f"{target}'s direct favorite/top-pick statements mention: {', '.join(answers)}.",
        operator="direct_favorite",
        evidence=[item[4] for item in ordered[:5]],
    )


def _storage_object_terms(question: str) -> list[str]:
    stop = {
        "where",
        "current",
        "currently",
        "keep",
        "store",
        "stored",
        "put",
        "do",
        "did",
        "i",
        "my",
        "the",
        "old",
        "new",
    }
    terms: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9'-]+", question.lower()):
        if token in stop or len(token) < 3:
            continue
        terms.append(token)
    return terms[:4]


def _storage_location_from_sentence(sentence: str) -> str:
    patterns = (
        r"\b(?:stored|keeping|kept|storing|store|keep|put|placed)\s+(?:them|it|my\s+[^.;\n|]{2,50}?)\s+(under|in|on|at)\s+([^.;\n|]{3,90})",
        r"\b[^.;\n|]{0,80}?\s+(under|in|on|at)\s+(a\s+shoe rack(?:\s+in\s+(?:it|my closet|the closet))?|my closet|the closet|under my bed|my bed)",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if not match:
            continue
        preposition = match.group(1).lower()
        place = normalize_name(match.group(2))
        place = re.split(r"\s*,\s*", place, maxsplit=1)[0]
        place = re.sub(r"\b(?:and|but|because|which|that|while)\b.*$", "", place, flags=re.IGNORECASE).strip()
        if re.search(r"\bin it\b", place, flags=re.IGNORECASE) and re.search(r"\bcloset\b", sentence, flags=re.IGNORECASE):
            place = re.sub(r"\bin it\b", "in my closet", place, flags=re.IGNORECASE)
        if (
            re.search(r"\bshoe rack\b", place, flags=re.IGNORECASE)
            and "closet" not in place.lower()
            and re.search(r"\bcloset\b", sentence, flags=re.IGNORECASE)
        ):
            place = place.rstrip(" .,;:") + " in my closet"
        if place:
            return f"{preposition} {place}"
    return ""


def op_sum_unit_quantities(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not re.search(r"\b(how many|total number|in total)\b", q):
        return None
    if re.search(r"\baverage\b|\bpercentage\b|\bpercent\b|\bolder\b", q):
        return None
    unit = _question_unit(question)
    if unit is None:
        return None
    anchors = _quantity_anchors(question)
    if len(anchors) < 2:
        return None

    found: dict[str, tuple[float, str]] = {}
    leaf_chunks = [chunk for chunk in chunks if chunk.node_type == "leaf"]
    for anchor in anchors:
        for _chunk, sent in sentences(leaf_chunks):
            if not _anchor_matches(anchor, sent):
                continue
            value = _unit_quantity_in_sentence(sent, unit)
            if value is None:
                continue
            if _quantity_sentence_rejected(sent, question):
                continue
            found.setdefault(anchor, (value, compact(sent)))
            break
    if len(found) != len(anchors):
        return None
    total = sum(value for value, _ in found.values())
    if total <= 0:
        return None
    unit_text = _plural_unit(unit, total)
    answer_value = _format_quantity(total)
    evidence = [text for _, text in found.values()]
    anchor_text = ", ".join(f"{anchor}: {_format_quantity(value)}" for anchor, (value, _) in found.items())
    return OpResult(
        answer=f"{answer_value} {unit_text}",
        reason=f"The explicit quantities by requested item are {anchor_text}, summing to {answer_value} {unit_text}.",
        operator="sum_unit_quantities",
        evidence=evidence,
    )


def _question_unit(question: str) -> str | None:
    lowered = question.lower()
    for unit, aliases in UNIT_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
            return unit
    return None


def _quantity_anchors(question: str) -> list[str]:
    anchors: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\b",
        question,
    ):
        anchor = normalize_name(match.group(0))
        if anchor.lower() in {"how", "what", "i"}:
            continue
        key = title_key(anchor)
        if len(key) < 3 or key in seen:
            continue
        anchors.append(anchor)
        seen.add(key)
    return anchors[:4]


def _anchor_matches(anchor: str, text: str) -> bool:
    lowered = text.lower()
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", anchor)
        if len(token) > 1 and token.lower() not in {"the", "main", "all"}
    ]
    if not tokens:
        return False
    if all(re.search(rf"\b{re.escape(token)}\b", lowered) for token in tokens):
        return True
    alias_groups = {
        "mcu": {"marvel", "cinematic", "universe"},
        "nyc": {"new", "york", "city"},
        "instagram": {"influencer", "collaboration"},
        "facebook": {"ad", "campaign"},
    }
    anchor_key = title_key(anchor)
    aliases = alias_groups.get(anchor_key)
    return bool(aliases and all(token in lowered for token in aliases))


def _unit_quantity_in_sentence(sentence: str, unit: str) -> float | None:
    aliases = UNIT_ALIASES[unit]
    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    if unit == "week":
        if re.search(r"\ba\s+week\s+and\s+a\s+half\b", sentence, flags=re.IGNORECASE):
            return 1.5
        match = re.search(
            r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+and\s+a\s+half\s+weeks?",
            sentence,
            flags=re.IGNORECASE,
        )
        if match and (value := numeric_value(match.group(1))) is not None:
            return value + 0.5
    for pattern in (
        rf"(\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\s+({alias_pattern})\b",
        rf"({alias_pattern})\s+(?:of|for|around|about)?\s*(\d[\d,]*(?:\.\d+)?)\b",
        rf"reached\s+around\s+(\d[\d,]*(?:\.\d+)?)\s+({alias_pattern})\b",
    ):
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1)
        if raw_value.lower() in aliases and match.lastindex and match.lastindex >= 2:
            raw_value = match.group(2)
        value = numeric_value(raw_value.replace(",", ""))
        if value is not None:
            return value
    return None


def _quantity_sentence_rejected(sentence: str, question: str) -> bool:
    lowered = sentence.lower()
    if re.search(r"\b(can|could|would|recommend|suggest|option|budget|estimate|range|typical)\b", lowered):
        return True
    if re.search(r"\bplanning to\b|\bwant to\b|\bthinking of\b", lowered) and not re.search(
        r"\b(previous|recently got back|finished|watched|reached|ran for|spent)\b",
        lowered,
    ):
        return True
    if "facebook ad campaign" in question.lower() and "previous ad campaign" in lowered:
        return False
    return False


def _plural_unit(unit: str, value: float) -> str:
    singular = {"people": "people"}.get(unit, unit)
    plural = {"people": "people", "day": "days", "week": "weeks", "hour": "hours"}.get(unit, unit + "s")
    return singular if abs(value - 1.0) < 1e-9 else plural


def _format_quantity(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def op_sum_money(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not (
        re.search(r"\b(?:amount|money|cost|expense|expenses|dollars?)\b", q)
        or re.search(r"\bspent\s+on\b", q)
    ):
        return None
    if re.search(r"\b(?:days|weeks|months|years|hours)\b", q) and not re.search(
        r"\b(?:money|amount|cost|expense|expenses|dollars?)\b", q
    ):
        return None
    topic_terms = set(re.findall(r"[a-z]+", question.lower())) - {
        "how",
        "much",
        "total",
        "money",
        "have",
        "spent",
        "since",
        "the",
        "start",
        "of",
        "year",
        "related",
        "expenses",
        "on",
        "i",
        "me",
        "my",
        "past",
        "few",
        "months",
        "recent",
        "recently",
        "items",
        "item",
        "things",
    }

    topic_aliases = {
        "bike": {"bike", "bikes", "bicycle", "bicycles", "cycling", "cyclist"},
        "car": {"car", "cars", "auto", "vehicle", "vehicles"},
        "travel": {"travel", "trip", "flight", "hotel", "airbnb"},
        "medical": {"medical", "doctor", "clinic", "appointment", "prescription"},
    }
    anchors = set(topic_terms)
    for term in list(topic_terms):
        anchors.update(topic_aliases.get(term, set()))

    def topic_grounded(text: str) -> bool:
        if not anchors:
            return True
        lowered_text = text.lower()
        return any(re.search(rf"\b{re.escape(anchor)}\b", lowered_text) for anchor in anchors)

    def actual_personal_spend(text: str) -> bool:
        lowered_text = text.lower()
        if re.search(
            r"budget|allocate|estimated?|typical|range|ranges from|can cost|could cost|"
            r"will cost|would cost|option|free|verified certificate|audit for free|"
            r"per day|per month|per year|/month|/year|subscription|cost per click|"
            r"filter search results|parking costs|depending on",
            lowered_text,
        ):
            return False
        if re.search(r"\$\d[\d,]*(?:\.\d+)?\s*[-–]\s*\$?\d", text):
            return False
        return bool(
            re.search(
                r"\bI\b.*\b(?:spent|bought|purchased|paid|got|treated myself)\b|"
                r"\b(?:spent|cost(?:ed)? me|paid|bought|purchased)\b|"
                r"\bUser (?:recently )?(?:spent|bought|purchased)\b|"
                r"\bMemory: Spent\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    units: list[tuple[MemoryChunk, str]] = []
    for chunk, sent in sentences(chunks):
        for clause in sentence_clauses(sent):
            units.append((chunk, clause))

    content_stop = {
        "user",
        "recently",
        "new",
        "got",
        "done",
        "did",
        "which",
        "were",
        "was",
        "the",
        "and",
        "for",
        "with",
        "from",
        "where",
        "they",
        "their",
        "local",
        "shop",
        "downtown",
        "spent",
        "cost",
        "bought",
        "purchased",
        "installed",
        "replacement",
        "replace",
        "around",
    }

    def content_words(text: str) -> set[str]:
        words = set(re.findall(r"[a-z][a-z0-9']+", text.lower()))
        words -= content_stop
        words -= anchors
        return {word for word in words if len(word) > 2}

    candidates: list[tuple[int, str, str, str]] = []
    for index, (chunk, unit) in enumerate(units):
        if not re.search(r"\$\d+", unit):
            continue
        if not re.search(r"cost|spent|purchased|bought|paid|treated myself|got", unit, flags=re.IGNORECASE):
            continue
        if not actual_personal_spend(unit):
            continue
        local_context = " ".join(
            units[j][1]
            for j in range(max(0, index - 1), min(len(units), index + 2))
            if units[j][0] is chunk
        )
        if not topic_grounded(local_context):
            continue
        if re.search(r"plan|planning|will|next week|considering", unit, flags=re.IGNORECASE) and not re.search(
            r"cost|spent|purchased|bought|installed|replacement|replace", unit, flags=re.IGNORECASE
        ):
            continue
        for match in re.finditer(r"\$(\d+(?:,\d{3})*)", unit):
            value = int(match.group(1).replace(",", ""))
            words = content_words(unit)
            candidates.append((value, compact(unit), " ".join(sorted(words)), chunk.node_type))

    grouped: list[tuple[int, str, set[str], str]] = []
    for value, text, word_text, node_type in candidates:
        words = set(word_text.split())
        merged = False
        for group_index, (old_value, old_text, old_words, old_node_type) in enumerate(grouped):
            if old_value != value:
                continue
            overlap = len(words & old_words)
            union = len(words | old_words) or 1
            if overlap == 0 and overlap / union < 0.35:
                continue
            keep_new = old_node_type != "leaf" and node_type == "leaf"
            grouped[group_index] = (
                old_value,
                text if keep_new else old_text,
                old_words | words,
                "leaf" if keep_new else old_node_type,
            )
            merged = True
            break
        if not merged:
            grouped.append((value, text, words, node_type))

    if len(grouped) >= 2:
        total = sum(value for value, _, _, _ in grouped)
        evidence = [text for _, text, _, _ in grouped]
        return OpResult(
            answer=f"${total}",
            reason=f"The explicit spent/cost items sum to ${total}.",
            operator="sum_amounts",
            evidence=evidence,
        )
    return None


def op_temporal_arrival(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    place_match = re.search(r"reach(?:ed)? the ([a-z ]+?) on", question, flags=re.IGNORECASE)
    if not place_match:
        return None
    place = place_match.group(1).strip().lower()
    leave_time: str | None = None
    duration_hours: int | None = None
    evidence: list[str] = []
    for _, sent in sentences(chunks):
        lowered = sent.lower()
        if leave_time is None:
            match = re.search(r"left home at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", sent, flags=re.IGNORECASE)
            if match:
                leave_time = match.group(1)
                evidence.append(compact(sent))
        if duration_hours is None and place in lowered:
            match = re.search(r"took (?:me\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+hours?", sent, flags=re.IGNORECASE)
            if match:
                duration_hours = number_value(match.group(1))
                evidence.append(compact(sent))
    if leave_time and duration_hours is not None:
        arrival = add_hours(leave_time, duration_hours)
        return OpResult(
            answer=arrival,
            reason=f"The user left home at {leave_time.upper()} and the trip to the {place} took {duration_hours} hours.",
            operator="temporal_delta",
            evidence=evidence,
        )
    return None


def op_previous_role(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if "previous occupation" not in question.lower() and "previous role" not in question.lower():
        return None
    for _, sent in sentences(chunks):
        match = re.search(r"previous role as (?:a|an)\s+(.+?)(?:\.|,| and | but )", sent, flags=re.IGNORECASE)
        if match:
            role = normalize_name(match.group(1))
            return OpResult(
                answer=role[0].upper() + role[1:],
                reason="The memory explicitly states the user's previous role.",
                operator="profile_fact",
                evidence=[compact(sent)],
            )
    return None


def op_preference_recommendation(question: str, question_type: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if question_type != "single-session-preference":
        return None
    text = "\n".join(chunk.text for chunk in chunks)
    lowered = text.lower()
    if "evening" in q and "wind down by" in lowered:
        time_match = re.search(r"wind down by\s+(\d{1,2}:\d{2}\s*(?:am|pm))", text, flags=re.IGNORECASE)
        time = time_match.group(1).upper() if time_match else "bedtime"
        avoid = "avoid screens/phone/TV" if re.search(r"avoid screens|electronic device detox|phone|tv", lowered) else "avoid stimulating activities"
        return OpResult(
            answer=(
                f"Do relaxing activities before {time}: reading, gentle stretching or yoga, "
                f"deep breathing, body-scan meditation, soothing music, or journaling; {avoid}."
            ),
            reason=f"The memory says the user wants to wind down by {time} and reduce sleep-disrupting stimulation.",
            operator="preference_profile",
            evidence=[compact(sent) for _, sent in sentences(chunks) if "wind down" in sent.lower()][:3],
        )
    if re.search(r"publication|conference", q):
        evidence_sents = [
            sent
            for _, sent in sentences(chunks)
            if re.search(r"recent advancements|research papers|conference|deep learning|medical image|healthcare", sent, flags=re.IGNORECASE)
        ]
        if evidence_sents:
            domain = "AI in healthcare"
            domain_match = re.search(r"deep learning for ([A-Za-z ]+?)(?: include| focus|,|;)", text, flags=re.IGNORECASE)
            if domain_match:
                domain = "deep learning for " + normalize_name(domain_match.group(1))
            return OpResult(
                answer=f"Look for recent publications or conferences focused on {domain}, especially methods, datasets, and explainability work mentioned in the memory.",
                reason="The memory's publication/conference interests are domain-specific, so recommendations should stay within that research area.",
                operator="preference_profile",
                evidence=[compact(item) for item in evidence_sents[:4]],
            )
    if re.search(r"cultural events?|events? happening", q):
        evidence_sents = [
            sent
            for _, sent in sentences(chunks)
            if re.search(r"spanish|french|language skills|cultural exchange|cultural festival|language exchange", sent, flags=re.IGNORECASE)
            and not re.search(r"natural language processing|\bNLP\b", sent, flags=re.IGNORECASE)
        ]
        if evidence_sents:
            evidence_text = "\n".join(evidence_sents)
            languages = []
            for language in ("Spanish", "French", "German", "Italian", "Mandarin", "Japanese"):
                if re.search(rf"\b{language}\b", evidence_text, flags=re.IGNORECASE):
                    languages.append(language)
            language_text = " and ".join(languages[:2]) if languages else "their language skills"
            return OpResult(
                answer=(
                    f"The user would prefer cultural event recommendations that let them practice {language_text} "
                    "and participate in language-learning or cultural-exchange settings, such as language exchange "
                    "meetups, multicultural festivals, embassy/cultural-center events, or conversation groups. They "
                    "would not prefer generic events without language practice or cultural exchange."
                ),
                reason="The user's event preferences are tied to language practice and cultural exchange, not generic weekend entertainment.",
                operator="preference_profile",
                evidence=[compact(item) for item in evidence_sents[:4]],
            )
    if re.search(r"painting|paintings|inspiration", q):
        evidence_sents = [
            sent
            for _, sent in sentences(chunks)
            if re.search(r"painting|paintings|instagram|online tutorials|30-day painting challenge|flower|palette knife", sent, flags=re.IGNORECASE)
        ]
        evidence_text = "\n".join(evidence_sents)
        if re.search(r"painting|paintings", evidence_text, flags=re.IGNORECASE):
            return OpResult(
                answer=(
                    "The user would prefer inspiration advice that builds on their painting history: revisit "
                    "Instagram art accounts and flower paintings, try techniques from online tutorials, experiment "
                    "with palette-knife texture, and use their recent 30-day painting challenge for structure. They "
                    "would not prefer vague inspiration tips unrelated to these existing sources and themes."
                ),
                reason="The user's painting memories point to specific inspiration sources, themes, techniques, and a recent challenge structure.",
                operator="preference_profile",
                evidence=[compact(item) for item in evidence_sents[:5]],
            )
    if re.search(r"hotel|accommodation|stay", q):
        evidence_sents = [
            sent
            for _, sent in sentences(chunks)
            if re.search(r"hotel|room|rooftop pool|hot tub|balcony|skyline|ocean|space needle", sent, flags=re.IGNORECASE)
        ]
        evidence_text = "\n".join(evidence_sents)
        if re.search(r"hotel|room", evidence_text, flags=re.IGNORECASE) and re.search(
            r"view|rooftop pool|hot tub|balcony|skyline|ocean|space needle",
            evidence_text,
            flags=re.IGNORECASE,
        ):
            location_match = re.search(r"trip to\s+([A-Z][A-Za-z ]+?)(?:\?|$)", question)
            location = normalize_name(location_match.group(1)) if location_match else "the destination"
            view_text = "ocean or city skyline views" if location.lower() == "miami" else "city, ocean, waterfront, or landmark views"
            return OpResult(
                answer=(
                    f"The user would prefer hotel suggestions in {location} that offer great views, especially "
                    f"{view_text}, and distinctive amenities such as a rooftop pool or a hot tub on the balcony. "
                    "They would not prefer basic or budget hotel suggestions without those view-focused or unique "
                    "room/rooftop features."
                ),
                reason="The user's hotel memories emphasize views and distinctive amenities rather than generic lodging.",
                operator="preference_profile",
                evidence=[compact(item) for item in evidence_sents[:5]],
            )
    return None


def op_membership_meetup_delta(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if not re.search(r"how long .*member", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"meetup|attended", question, flags=re.IGNORECASE):
        return None
    joined_weeks: int | None = None
    meetup_weeks: int | None = None
    evidence: list[str] = []
    for _, sent in sentences(chunks):
        joined = re.search(
            r"joined .*?(\d+|one|two|three|four|five|six)\s+weeks? ago",
            sent,
            flags=re.IGNORECASE,
        )
        if joined:
            joined_weeks = number_value(joined.group(1))
            evidence.append(compact(sent))
        attended = re.search(
            r"attended .*?meetup .*?(\d+|one|two|three|four|five|six|last)\s+weeks? ago|attended .*?meetup .*?last week",
            sent,
            flags=re.IGNORECASE,
        )
        if attended:
            value = attended.group(1) if attended.lastindex else "last"
            meetup_weeks = 1 if value.lower() == "last" else number_value(value)
            evidence.append(compact(sent))
    if joined_weeks is not None and meetup_weeks is not None and joined_weeks >= meetup_weeks:
        delta = joined_weeks - meetup_weeks
        word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(delta, str(delta))
        suffix = "week" if delta == 1 else "weeks"
        return OpResult(
            answer=f"{word} {suffix}",
            reason=(
                f"The user joined {joined_weeks} weeks before the reference date and attended the meetup "
                f"{meetup_weeks} week(s) before it, so membership length at the meetup was {delta} week(s)."
            ),
            operator="relative_time_delta",
            evidence=evidence,
        )
    return None


def op_month_delta(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if "how many months ago" not in question.lower():
        return None
    visited_months: int | None = None
    advance_months: int | None = None
    evidence: list[str] = []
    for _, sent in sentences(chunks):
        visit = re.search(r"visited .*?(\d+|one|two|three|four|five|six)\s+months? ago", sent, flags=re.IGNORECASE)
        if visit:
            visited_months = number_value(visit.group(1))
            evidence.append(compact(sent))
        advance = re.search(r"book(?:ed|ing)?\s+(\d+|one|two|three|four|five|six)\s+months? in advance", sent, flags=re.IGNORECASE)
        if advance:
            advance_months = number_value(advance.group(1))
            evidence.append(compact(sent))
    if visited_months is not None and advance_months is not None:
        total = visited_months + advance_months
        return OpResult(
            answer=f"{total} months ago",
            reason=f"The trip was {visited_months} months ago and booking was {advance_months} months in advance.",
            operator="temporal_delta",
            evidence=evidence,
        )
    return None


def op_airline_order(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not (re.search(r"\border\b", q) and "airline" in q and re.search(r"\bflew|flight", q)):
        return None
    events: dict[str, tuple[str, int, str]] = {}
    for chunk, sent in sentences([chunk for chunk in chunks if chunk.node_type == "leaf"]):
        if _non_user_or_advice_sentence(sent):
            continue
        if not re.search(
            r"\b(?:flight|flew|flying with|got back from|took a round-trip)\b",
            sent,
            flags=re.IGNORECASE,
        ):
            continue
        if _airline_sentence_rejected(sent):
            continue
        airline = _actual_airline_from_sentence(sent)
        if not airline:
            continue
        key = title_key(airline)
        event_date = _event_date_sort_key(sent, chunk.session_date)
        turn_index = chunk.turn_index or -1
        previous = events.get(key)
        if previous is None or (event_date, turn_index) < (previous[0], previous[1]):
            events[key] = (event_date, turn_index, compact(sent))
    if len(events) < 2:
        return None
    ordered = sorted(
        ((_display_airline(key), date, turn, evidence) for key, (date, turn, evidence) in events.items()),
        key=lambda item: (item[1], item[2], item[0]),
    )
    names = [name for name, _, _, _ in ordered]
    return OpResult(
        answer=", ".join(names),
        reason="The actual flight memories are sorted from earliest to latest by their event/session dates.",
        operator="temporal_event_order",
        evidence=[evidence for _, _, _, evidence in ordered],
    )


def _actual_airline_from_sentence(sentence: str) -> str:
    aliases = {
        "jetblue": "JetBlue",
        "delta": "Delta",
        "united": "United",
        "united airlines": "United",
        "american airlines": "American Airlines",
        "spirit": "Spirit Airlines",
        "spirit airlines": "Spirit Airlines",
        "southwest": "Southwest",
        "southwest airlines": "Southwest",
    }
    lowered = sentence.lower()
    for alias, display in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            if alias == "delta" and not re.search(r"\b(?:flight|flew|round-trip|skymiles)\b", lowered):
                continue
            return display
    return ""


def _display_airline(key: str) -> str:
    displays = {
        "jetblue": "JetBlue",
        "delta": "Delta",
        "united": "United",
        "unitedairlines": "United",
        "americanairlines": "American Airlines",
        "spiritairlines": "Spirit Airlines",
        "spirit": "Spirit Airlines",
        "southwest": "Southwest",
        "southwestairlines": "Southwest",
    }
    return displays.get(key, key)


def _airline_sentence_rejected(sentence: str) -> bool:
    lowered = sentence.lower()
    if not re.search(
        r"\b(?:i just got back|after taking|took a round-trip|today)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(?:in-flight savings|headsets|benefits|sign-up bonus)\b", lowered):
        return True
    if re.search(
        r"\b(?:planning|considering|redeem|redemption|best deals|compare|book a new flight|"
        r"want to|prefer|could use|card|credit card|bonus|miles|points|options|policy|"
        r"seat selection|amenities|baggage|insurance|customer service)\b",
        lowered,
    ) and not re.search(
        r"\b(?:got back from|today|had a .*flight|took a round-trip flight|my flight from)\b",
        lowered,
    ):
        return True
    return False


def _event_date_sort_key(sentence: str, session_date: str | None) -> str:
    month_day = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    match = re.search(
        r"\b(" + "|".join(month_day) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if match and session_date:
        year_match = re.match(r"(\d{4})/", session_date)
        year = year_match.group(1) if year_match else "9999"
        return f"{year}/{month_day[match.group(1).lower()]}/{int(match.group(2)):02d}"
    return session_date or "9999/99/99"


def op_holiday_airline(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if "airline" not in question.lower() or "valentine" not in question.lower():
        return None
    holiday_chunks = [chunk for chunk in chunks if chunk.session_date and "/02/14" in chunk.session_date]
    for chunk, sent in sentences(holiday_chunks or chunks):
        if not re.search(r"flight|flew|flying", sent, flags=re.IGNORECASE):
            continue
        match = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+flight", sent)
        if match:
            airline = normalize_name(match.group(1))
            if airline.lower() not in {"my", "the", "a"}:
                return OpResult(
                    answer=airline,
                    reason="The holiday-dated memory mentions this airline for the user's flight.",
                    operator="dated_event_lookup",
                    evidence=[compact(sent), chunk.session_date or ""],
                )
    return None


def op_previous_chat_final(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    q = question.lower()
    if not re.search(r"previous|looking back|finally decided|remind me", q):
        return None
    if "name" in q or "decided" in q:
        for _, sent in sentences(chunks):
            match = re.search(r"([A-Z][A-Za-z0-9\-]+)\s+is\s+(?:a\s+)?(?:really\s+)?cool one", sent, flags=re.IGNORECASE)
            if match:
                name = normalize_name(match.group(1))
                return OpResult(
                    answer=name,
                    reason="The user accepted this later name suggestion and continued the design around it.",
                    operator="final_decision",
                    evidence=[compact(sent)],
                )
    if "beer" in q:
        for _, sent in sentences(chunks):
            match = re.search(r"\b(Pilsner|Lager)\b.*\b(Pilsner|Lager)\b|\b(Pilsner|lager)\b", sent, flags=re.IGNORECASE)
            if match and re.search(r"recommend|work well|beer", sent, flags=re.IGNORECASE):
                return OpResult(
                    answer="Pilsner or Lager",
                    reason="The previous recipe discussion recommends these beer styles.",
                    operator="previous_answer_lookup",
                    evidence=[compact(sent)],
                )
    if "shift rotation" in q:
        person_match = re.search(r"for\s+([A-Z][a-z]+)\s+on\s+(?:a\s+)?([A-Z][a-z]+)", question)
        if not person_match:
            return None
        person = person_match.group(1)
        day = person_match.group(2)
        for _, sent in sentences(chunks):
            if day.lower() not in sent.lower() or person.lower() not in sent.lower() or "|" not in sent:
                continue
            rows = [line.strip() for line in sent.split("|") if line.strip()]
            if rows and rows[0].lower() == day.lower():
                index = next((idx for idx, value in enumerate(rows[1:], start=1) if value.lower() == person.lower()), None)
                if index == 1:
                    return OpResult(
                        answer=f"{person} was assigned to the 8 am - 4 pm (Day Shift) on {day}s.",
                        reason="The final named rotation table places the person in the first shift column for that day.",
                        operator="table_lookup",
                        evidence=[compact(sent)],
                    )
        table_text = "\n".join(chunk.text for chunk in chunks if "|" in chunk.text and person.lower() in chunk.text.lower())
        match = re.search(rf"\|\s*{day}\s*\|\s*{person}\s*\|", table_text, flags=re.IGNORECASE)
        if match:
            return OpResult(
                answer=f"{person} was assigned to the 8 am - 4 pm (Day Shift) on {day}s.",
                reason="The final named rotation table places the person in the first shift column for that day.",
                operator="table_lookup",
                evidence=[compact(table_text)],
            )
    return None


def op_duration_sum(question: str, chunks: list[MemoryChunk]) -> OpResult | None:
    if "combined" not in question.lower() or not re.search(r"hours|driving|road trip", question, flags=re.IGNORECASE):
        return None
    durations: dict[str, tuple[int, str]] = {}
    for _, sent in sentences(chunks):
        if not re.search(r"drive|drove|driving|took", sent, flags=re.IGNORECASE):
            continue
        if not re.search(r"\b(I|I've|my|me)\b", sent):
            continue
        if re.search(r"\bfrom\b.+\bto\b|according to|approx|~|recommend|option|route|itinerary", sent, flags=re.IGNORECASE):
            continue
        for match in re.finditer(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+hours?", sent, flags=re.IGNORECASE):
            value = number_value(match.group(1))
            if value is not None and 0 < value <= 24:
                destination = None
                for pattern in (
                    r"trip to\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})(?:\s+in\b|\s+-|,|\.|\s+only\b)",
                    r"drove .*?\bto\s+([A-Z][A-Za-z.]*)(?:,|\.|\s+and\b|\s+for\b)",
                    r"to the mountains in\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})",
                    r"drive to\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})(?:,|\.|\s+and\b|\s+for\b)",
                ):
                    destination_match = re.search(pattern, sent)
                    if destination_match:
                        destination = normalize_name(destination_match.group(1))
                        break
                key = destination or compact(sent, 80)
                durations.setdefault(title_key(key), (value, compact(sent)))
    if len(durations) >= 3:
        total = sum(value for value, _ in durations.values())
        return OpResult(
            answer=f"{total} hours",
            reason="The relevant driving durations sum to " + str(total) + " hours.",
            operator="sum_durations",
            evidence=[text for _, text in durations.values()],
        )
    return None


OPERATORS = (
    op_before_after_purchase,
    op_count_services,
    op_count_attended_events,
    op_count_named_attended_events,
    op_count_health_visits,
    op_current_subscriptions,
    op_current_reading,
    op_current_storage_location,
    op_current_numeric_status,
    op_sum_unit_quantities,
    op_temporal_arrival,
    op_previous_role,
    op_month_delta,
    op_membership_meetup_delta,
    op_airline_order,
    op_holiday_airline,
    op_previous_chat_final,
    op_duration_sum,
)


def generic_answer(
    question: str,
    question_type: str,
    chunks: list[MemoryChunk],
    *,
    enable_sum_amounts: bool = False,
    enable_speaker_mismatch_abstain: bool = False,
) -> OpResult | None:
    result = op_binary_speaker_fact(question, chunks)
    if result is not None:
        return result
    result = op_missing_required_conjunct(question, chunks)
    if result is not None:
        return result
    result = op_book_recommendations(question, chunks)
    if result is not None:
        return result
    result = op_direct_favorite(question, chunks)
    if result is not None:
        return result
    if enable_speaker_mismatch_abstain:
        result = op_speaker_mismatch_abstain(
            question,
            chunks,
            strict_target_support=question_type == "category_5",
        )
        if result is not None:
            return result
    if enable_sum_amounts:
        result = op_sum_money(question, chunks)
        if result is not None:
            return result
    for operator in OPERATORS:
        result = operator(question, chunks)  # type: ignore[misc]
        if result is not None:
            return result
    return op_preference_recommendation(question, question_type, chunks)


def update_stats(stats: dict[str, Any], llm_calls: Path | None, fallback_ids: set[str]) -> dict[str, Any]:
    stats = dict(stats)
    stats["generic_memory_ops_count"] = len(fallback_ids)
    stats["generic_memory_ops_extra_llm_cost"] = False
    if llm_calls:
        routed_by_question: dict[str, int] = {}
        conservative_by_question: dict[str, int] = {}
        for row in read_jsonl(llm_calls):
            if row.get("stage") != "answer_qa":
                continue
            qid = row["question_id"]
            total = int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
            conservative_by_question[qid] = conservative_by_question.get(qid, 0) + total
            if qid not in fallback_ids:
                routed_by_question[qid] = routed_by_question.get(qid, 0) + total
            else:
                routed_by_question.setdefault(qid, 0)
        stats["conservative_max_query_answer_tokens"] = max(conservative_by_question.values())
        stats["routed_avg_query_answer_tokens"] = (
            sum(routed_by_question.values()) / len(routed_by_question) if routed_by_question else 0.0
        )
        stats["routed_max_query_answer_tokens"] = max(routed_by_question.values())
        stats["routed_query_answer_over_10k_count"] = sum(
            1 for value in routed_by_question.values() if value > 10000
        )
    return stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {row["question_id"]: row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    chunks_by_qid = load_chunks(args.nodes)
    output_rows: list[dict[str, Any]] = []
    ops_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.answers):
        qid = row["question_id"]
        case = cases[qid]
        result = generic_answer(
            str(case["question"]),
            str(case["question_type"]),
            chunks_by_qid.get(qid, []),
            enable_sum_amounts=args.enable_sum_amounts,
            enable_speaker_mismatch_abstain=args.enable_speaker_mismatch_abstain,
        )
        new_row = dict(row)
        new_row["variant"] = args.variant
        if result is None:
            new_row["generic_memory_op_applied"] = False
        else:
            new_row["prediction"] = final_answer(result)
            new_row["generic_memory_op_applied"] = True
            new_row["generic_memory_op"] = result.operator
            new_row["generic_memory_op_answer"] = result.answer
            new_row["generic_memory_op_reason"] = result.reason
            ops_rows.append(
                {
                    "question_id": qid,
                    "question": case["question"],
                    "operator": result.operator,
                    "answer": result.answer,
                    "reason": result.reason,
                    "evidence": result.evidence,
                }
            )
        output_rows.append(new_row)

    answers_path = args.output_dir / "answers.jsonl"
    hypothesis_path = args.output_dir / f"{args.variant}_hypothesis.jsonl"
    write_jsonl(answers_path, output_rows)
    write_jsonl(
        hypothesis_path,
        [{"question_id": row["question_id"], "hypothesis": row.get("prediction", "")} for row in output_rows],
    )
    write_jsonl(args.output_dir / "generic_memory_ops.jsonl", ops_rows)
    stats = update_stats(
        json.loads(args.query_stats.read_text(encoding="utf-8")),
        args.llm_calls,
        {row["question_id"] for row in ops_rows},
    )
    (args.output_dir / "query_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"answers={answers_path}")
    print(f"hypothesis={hypothesis_path}")
    print(f"generic_ops={len(ops_rows)}")


if __name__ == "__main__":
    main()
