from __future__ import annotations

from datetime import datetime
import re
from typing import Callable

from .catalog_schema import OperandRecordV3
from .schema import QueryFrame


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "after", "by",
    "did", "do", "does", "for", "from", "get", "getting", "had", "has",
    "have", "i", "in", "into", "is", "it", "my", "of", "on", "or", "the",
    "to", "was", "were", "what", "which", "with",
}
_INTENTION_RE = re.compile(
    r"\b(?:consider|considering|think|thinking|plan|planning|want|wants|"
    r"hope|hoping|might|may|could)\b.*\b(?:buy|get|purchase|invest)\w*\b",
    re.IGNORECASE,
)
_POSSESSION_OR_USE_RE = re.compile(
    r"\b(?:my|our)\s+(?:new\s+)?|\b(?:own|owned|use|using|used|got|"
    r"bought|purchased|invested)\b",
    re.IGNORECASE,
)


def _terms(value: str) -> set[str]:
    result: set[str] = set()
    for raw in _WORD_RE.findall(value):
        token = raw.casefold()
        if token in _FUNCTION_WORDS or len(token) <= 1:
            continue
        result.add(token)
        if len(token) > 4 and token.endswith("ies"):
            result.add(token[:-3] + "y")
        elif len(token) > 5 and token.endswith("ing"):
            result.add(token[:-3])
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            result.add(token[:-1])
    return result


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%Y-%m-%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def relative_operand_hint(
    frame: QueryFrame,
    operands: list[OperandRecordV3],
    *,
    semantic_similarity: Callable[[OperandRecordV3], float],
    turns: list[object] | None = None,
) -> dict[str, object] | None:
    """Resolve a first-person object on one temporal side of a named anchor.

    Coarse graph routing finds the relevant catalog. This operator then binds
    the exact anchor and answer at the lossless operand layer, so an episode or
    event-frame summary never becomes the answer itself.
    """

    if frame.requested_operation != "ordering":
        return None
    relation_match = re.search(
        r"\b(before|after)\b\s+(.+?)(?:[?]|\Z)",
        frame.raw_question,
        re.IGNORECASE,
    )
    slot_match = re.match(
        r"^\s*(?:which|what)\s+(.+?)\s+did\s+i\b",
        frame.raw_question,
        re.IGNORECASE,
    )
    if relation_match is None or slot_match is None:
        return None
    relation = relation_match.group(1).casefold()
    anchor_terms = _terms(relation_match.group(2))
    slot_terms = _terms(slot_match.group(1))
    if not anchor_terms or not slot_terms:
        return None

    rows = [
        item for item in operands
        if item.polarity != "negative"
        and _timestamp(item.event_time or item.observed_at) is not None
    ]
    if len(rows) < 2:
        return None

    frequencies = {
        term: sum(
            term in _terms(
                f"{item.predicate_key} {item.object_text} {item.context_key}"
            )
            for item in rows
        )
        for term in anchor_terms
    }
    present = {term for term, count in frequencies.items() if count}
    if not present:
        return None
    anchor_candidates: list[tuple[float, datetime, OperandRecordV3]] = []
    for item in rows:
        text_terms = _terms(
            f"{item.predicate_key} {item.object_text} {item.context_key}"
        )
        covered = present & text_terms
        if not covered:
            continue
        coverage = len(covered) / len(present)
        rarity = sum(1.0 / frequencies[term] for term in covered)
        anchor_candidates.append((
            5.0 * coverage + rarity + semantic_similarity(item),
            _timestamp(item.event_time or item.observed_at) or datetime.min,
            item,
        ))
    if not anchor_candidates:
        return None
    _anchor_score, anchor_time, anchor = max(
        anchor_candidates,
        key=lambda row: (row[0], row[1], row[2].operand_id),
    )

    candidates: list[tuple[float, float, datetime, OperandRecordV3]] = []

    # Lossless fallback for an acquired/possessed entity that the extractor
    # omitted as an operand.  Mention order is used only for these undated
    # possessions; typed event time remains primary for extracted operands.
    anchor_mention_time = _timestamp(anchor.observed_at) or anchor_time
    possessive_pattern = re.compile(
        r"\bmy\s+new\s+(?P<entity>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})",
    )
    synthetic_rows: list[OperandRecordV3] = []
    for turn in turns or []:
        transport = str(getattr(turn, "transport_role", "")).casefold()
        speaker = str(getattr(turn, "speaker_key", "")).casefold()
        if transport == "assistant" or (
            transport and transport != "user"
            and speaker not in {"participant 1", "participant_1", "user"}
        ):
            continue
        observed_text = str(getattr(turn, "session_date", ""))
        observed = _timestamp(observed_text)
        if observed is None:
            continue
        if relation == "before" and observed >= anchor_mention_time:
            continue
        if relation == "after" and observed <= anchor_mention_time:
            continue
        text = str(getattr(turn, "text", ""))
        for match in possessive_pattern.finditer(text):
            entity = match.group("entity").strip()
            turn_semantic = max(0.0, semantic_similarity(turn))
            synthetic_rows.append(OperandRecordV3(
                operand_id=f"{getattr(turn, 'node_id', '')}:lossless_possession",
                question_id=frame.raw_question,
                subject_key=speaker or "participant 1",
                predicate_key="possess and use",
                object_key=entity.casefold(),
                object_text=entity,
                context_key=text,
                event_time=observed_text,
                observed_at=observed_text,
                source_turn_ids=[str(getattr(turn, "node_id", ""))],
                session_ids=[str(getattr(turn, "session_id", ""))],
                retrieval_text=f"{entity} {text}",
                confidence=turn_semantic,
            ))

    for item in [*rows, *synthetic_rows]:
        if item.operand_id == anchor.operand_id:
            continue
        synthetic = item.operand_id.endswith(":lossless_possession")
        if anchor.subject_key and item.subject_key != anchor.subject_key:
            continue
        observed = _timestamp(item.event_time or item.observed_at)
        if observed is None:
            continue
        relation_anchor_time = anchor_mention_time if synthetic else anchor_time
        if relation == "before" and observed >= relation_anchor_time:
            continue
        if relation == "after" and observed <= relation_anchor_time:
            continue
        text = f"{item.predicate_key} {item.object_text} {item.context_key}"
        item_terms = _terms(text)
        lexical = len(slot_terms & item_terms) / len(slot_terms)
        semantic = (
            max(0.0, float(item.confidence))
            if synthetic else max(0.0, semantic_similarity(item))
        )
        possessed = bool(_POSSESSION_OR_USE_RE.search(text))
        unfulfilled_intention = bool(_INTENTION_RE.search(text)) and not possessed
        if semantic < 0.25 and lexical <= 0:
            continue
        distance_hours = abs((anchor_time - observed).total_seconds()) / 3600.0
        score = (
            5.0 * semantic
            + 1.2 * lexical
            + 1.5 * float(possessed)
            + 1.25 * float(synthetic)
            - 1.5 * float(unfulfilled_intention)
            - min(distance_hours, 24.0 * 90) / (24.0 * 90)
        )
        candidates.append((score, semantic, observed, item))
    if not candidates:
        return None
    score, semantic, observed, answer = max(
        candidates,
        key=lambda row: (row[0], row[1], row[2], row[3].operand_id),
    )
    return {
        "operation": "before_after_operand_relation",
        "relation": relation,
        "value": answer.object_text,
        "answer_operand_id": answer.operand_id,
        "answer_time": observed.isoformat(timespec="minutes"),
        "anchor_operand_id": anchor.operand_id,
        "anchor_value": anchor.object_text,
        "anchor_time": anchor_time.isoformat(timespec="minutes"),
        "answer_slot_terms": sorted(slot_terms),
        "semantic_score": round(float(semantic), 6),
        "score": round(float(score), 6),
        "operand_ids": [anchor.operand_id, answer.operand_id],
        "source_turn_ids": list(dict.fromkeys([
            *anchor.source_turn_ids,
            *answer.source_turn_ids,
        ])),
        "complete": True,
        "completion_basis": "named_anchor_and_typed_lossless_operand",
    }
