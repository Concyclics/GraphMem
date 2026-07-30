from __future__ import annotations

import re
from collections import defaultdict

from .catalog_schema import EventFrameV3, OperandRecordV3
from .schema import ClaimNode, HyperEdge, HyperIncidence, V3Index


_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _canonical_key(value: str) -> str:
    return " ".join(token.casefold() for token in _WORD_RE.findall(value.replace("_", " ")))


_DAYS = {
    "monday": ("monday", "mondays", "mon"),
    "tuesday": ("tuesday", "tuesdays", "tue", "tues"),
    "wednesday": ("wednesday", "wednesdays", "wed"),
    "thursday": ("thursday", "thursdays", "thu", "thur", "thurs"),
    "friday": ("friday", "fridays", "fri"),
    "saturday": ("saturday", "saturdays", "sat"),
    "sunday": ("sunday", "sundays", "sun"),
}


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))


def _mean(vectors):
    rows = [row for row in vectors if row]
    if not rows:
        return None
    width = len(rows[0])
    rows = [row for row in rows if len(row) == width]
    return [
        sum(row[index] for row in rows) / len(rows)
        for index in range(width)
    ] if rows else None


def recurrence_days(text: str) -> list[str]:
    days = [
        day for day, aliases in _DAYS.items()
        if re.search(
            r"(?<![\w])(?:" + "|".join(map(re.escape, aliases)) + r")(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if len(days) >= 2:
        return days
    if not days:
        return []
    day = days[0]
    aliases = "|".join(map(re.escape, _DAYS[day]))
    plural_weekday = bool(re.search(
        r"(?<![\w])" + re.escape(day) + r"s(?![\w])",
        text,
        flags=re.IGNORECASE,
    ))
    local_habitual = bool(re.search(
        r"\b(?:every|each)(?:\s+other)?\s+(?:" + aliases + r")\b|"
        r"\b(?:weekly|routine|scheduled|usually|typically|regularly|recurring)"
        r".{0,20}\b(?:" + aliases + r")\b|"
        r"\b(?:" + aliases + r")\b.{0,20}"
        r"\b(?:weekly|routine|scheduled|usually|typically|regularly|recurring)\b",
        text,
        flags=re.IGNORECASE,
    ))
    return days if plural_weekday or local_habitual else []


_LOSSLESS_MONEY = re.compile(
    r"(?P<symbol>[$€£¥])\s*(?P<amount>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<word_amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<word_unit>USD|EUR|GBP|dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)
_MONEY_ACTION = re.compile(
    r"\b(?P<action>rais(?:e|ed|ing)|fundrais\w*|earn(?:ed|ing)?|"
    r"spend|spent|pay|paid|cost(?:ing)?|quot(?:e|ed)|pric(?:e|ed)|"
    r"donat(?:e|ed|ing)|sav(?:e|ed|ing)|receiv(?:e|ed|ing)|sell|sold)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_ASSERTION = re.compile(r"\b(?:i|we|my|our|mine|ours)\b", re.IGNORECASE)


def _lossless_money_projections(
    index: V3Index, existing: list[OperandRecordV3]
) -> tuple[list[ClaimNode], list[OperandRecordV3]]:
    """Backfill explicit self-ascribed currency values as sourced claim projections."""

    known = {
        (source_id, float(item.quantity), item.unit.casefold())
        for item in existing if item.quantity is not None
        for source_id in item.source_turn_ids
    }
    claim_additions: list[ClaimNode] = []
    operand_additions: list[OperandRecordV3] = []
    for turn in index.turns:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", turn.text)
        for sentence in sentences:
            action_match = _MONEY_ACTION.search(sentence)
            if (
                action_match is None
                or _FIRST_PERSON_ASSERTION.search(sentence[:action_match.start()]) is None
            ):
                continue
            for match in _LOSSLESS_MONEY.finditer(sentence):
                raw_amount = match.group("amount") or match.group("word_amount")
                try:
                    amount = float(raw_amount.replace(",", ""))
                except (AttributeError, ValueError):
                    continue
                unit = match.group("symbol") or (match.group("word_unit") or "").casefold()
                signature = (turn.node_id, amount, unit.casefold())
                if signature in known:
                    continue
                known.add(signature)
                action = action_match.group("action").casefold()
                polarity = "negative" if re.search(
                    r"\b(?:did not|didn't|never|not)\b", sentence, re.IGNORECASE
                ) else "positive"
                modality = "planned" if re.search(
                    r"\b(?:plan|intend|might|would|will)\w*\b", sentence, re.IGNORECASE
                ) else "asserted"
                context_key = _canonical_key(sentence[:240])
                retrieval_text = " | ".join((
                    turn.speaker_key, action, match.group(0), sentence[:300],
                    turn.session_date or "",
                ))
                claim = ClaimNode(
                    node_id=(
                        f"{turn.question_id}:lossless_money_claim:"
                        f"{len(index.claims) + len(claim_additions)}"
                    ),
                    question_id=turn.question_id,
                    session_id=turn.session_id,
                    subject=turn.speaker,
                    subject_key=turn.speaker_key,
                    predicate=action,
                    predicate_key=action,
                    object=match.group(0),
                    object_key=_canonical_key(match.group(0)),
                    kind="quantity",
                    polarity=polarity,
                    modality=modality,
                    context_key=context_key,
                    event_time=turn.session_date,
                    observed_at=turn.session_date,
                    quantity=amount,
                    unit=unit,
                    source_turn_ids=[turn.node_id],
                    confidence=0.9,
                    retrieval_text=retrieval_text,
                    embedding=turn.embedding,
                    observation_order=turn.turn_index,
                )
                claim_additions.append(claim)
                operand_additions.append(OperandRecordV3(
                    operand_id=(
                        f"{turn.question_id}:operand:"
                        f"{len(existing) + len(operand_additions)}"
                    ),
                    question_id=turn.question_id,
                    subject_key=turn.speaker_key,
                    predicate_key=action,
                    object_key=_canonical_key(match.group(0)),
                    object_text=match.group(0),
                    context_key=context_key,
                    polarity=polarity,
                    modality=modality,
                    event_time=turn.session_date,
                    observed_at=turn.session_date,
                    quantity=amount,
                    unit=unit,
                    source_claim_ids=[claim.node_id],
                    source_turn_ids=[turn.node_id],
                    session_ids=[turn.session_id],
                    event_type_keys=["currency transaction"],
                    confidence=0.9,
                    retrieval_text=retrieval_text,
                    embedding=turn.embedding,
                ))
    return claim_additions, operand_additions


def _observed(claim: ClaimNode, turns) -> str | None:
    return claim.observed_at or next(
        (
            turns[value].session_date for value in claim.source_turn_ids
            if value in turns and turns[value].session_date
        ),
        None,
    )


def build_catalog(index: V3Index):
    turns = {item.node_id: item for item in index.turns}
    claims = {item.node_id: item for item in index.claims}
    grouped = defaultdict(list)
    for event in index.events:
        key = event.label_key or _canonical_key(event.label)
        if key:
            grouped[(key, tuple(sorted(set(event.participant_keys))))].append(event)

    frames = []
    claim_to_frame = {}
    for rows in grouped.values():
        claim_ids = _unique(
            value for item in rows for value in item.claim_ids if value in claims
        )
        source_ids = _unique(
            value for item in rows for value in item.source_turn_ids if value in turns
        )
        statuses = _unique(item.status for item in rows)
        times = _unique(item.event_time for item in rows)
        participants = _unique(
            value for item in rows for value in item.participant_keys
        )
        semantic_types = _unique(
            value for item in rows for value in item.semantic_type_keys
        )
        frame_id = f"{rows[0].question_id}:event_frame:{len(frames)}"
        frame = EventFrameV3(
            frame_id=frame_id,
            question_id=rows[0].question_id,
            label=rows[0].label,
            label_key=rows[0].label_key or _canonical_key(rows[0].label),
            participant_keys=participants,
            status=statuses[-1] if statuses else "unknown",
            event_time=times[-1] if times else None,
            observed_at=max(
                (turns[value].session_date for value in source_ids
                 if turns[value].session_date),
                default=None,
            ),
            session_ids=_unique(turns[value].session_id for value in source_ids),
            claim_ids=claim_ids,
            event_ids=[item.node_id for item in rows],
            source_turn_ids=source_ids,
            semantic_type_keys=semantic_types,
            attributes={
                "statuses": statuses, "times": times, "participants": participants,
                "semantic_types": semantic_types,
            },
            confidence=min(item.confidence for item in rows),
            retrieval_text=" | ".join(
                _unique([rows[0].label, *semantic_types, *participants, *statuses, *times])
            ),
            embedding=_mean(
                [item.embedding for item in rows]
                + [claims[value].embedding for value in claim_ids]
            ),
        )
        frames.append(frame)
        for value in claim_ids:
            claim_to_frame.setdefault(value, frame_id)

    frame_by_id = {item.frame_id: item for item in frames}
    operands = []
    for claim in index.claims:
        if claim.predicate_key == "said":
            continue
        source_turns = [
            turns[value] for value in claim.source_turn_ids if value in turns
        ]
        days = recurrence_days(
            claim.object + " " + " ".join(item.text for item in source_turns)
        )
        observed = _observed(claim, turns)
        operands.append(OperandRecordV3(
            operand_id=f"{claim.question_id}:operand:{len(operands)}",
            question_id=claim.question_id,
            subject_key=claim.subject_key,
            predicate_key=claim.predicate_key,
            object_key=claim.object_key,
            object_text=claim.object,
            context_key=claim.context_key,
            event_frame_id=claim_to_frame.get(claim.node_id),
            state_op=claim.state_op,
            polarity=claim.polarity,
            modality=claim.modality,
            event_time=claim.event_time,
            observed_at=observed,
            recurrence_days=days,
            recurrence_count=len(days) if days else None,
            quantity=claim.quantity,
            unit=claim.unit,
            source_claim_ids=[claim.node_id],
            source_turn_ids=list(claim.source_turn_ids),
            session_ids=_unique(item.session_id for item in source_turns),
            event_type_keys=list(
                frame_by_id[claim_to_frame[claim.node_id]].semantic_type_keys
                if claim.node_id in claim_to_frame else []
            ),
            confidence=claim.confidence,
            retrieval_text=" | ".join(_unique([
                claim.subject, claim.predicate, claim.object, claim.context_key,
                claim.event_time, observed, *days,
            ])),
            embedding=claim.embedding,
        ))
    return frames, operands


def catalog_hyperedges(index: V3Index) -> list[HyperEdge]:
    edges = []
    for relation, rows in (
        ("operand_projection", index.operands),
        ("event_frame_member", index.event_frames),
    ):
        for item in rows:
            item_id = item.node_id
            members = _unique([
                item_id,
                *getattr(item, "source_claim_ids", []),
                *getattr(item, "claim_ids", []),
                *getattr(item, "event_ids", []),
                *item.source_turn_ids,
            ])
            if len(members) < 2:
                continue
            edges.append(HyperEdge(
                edge_id=f"{item.question_id}:catalog_edge:{relation}:{len(edges)}",
                question_id=item.question_id,
                relation=relation,
                incidences=[
                    HyperIncidence(
                        node_id=value,
                        role="catalog" if value == item_id else "source",
                    )
                    for value in members
                ],
                directed=True,
                confidence=item.confidence,
                provenance={"method": "deterministic_catalog_projection"},
                retrieval_text=item.retrieval_text,
                embedding=item.embedding,
            ))
    return edges


def ensure_catalog(index: V3Index) -> V3Index:
    if not index.event_frames and not index.operands:
        index.event_frames, index.operands = build_catalog(index)
    claims, operands = _lossless_money_projections(index, index.operands)
    index.claims.extend(claims)
    index.operands.extend(operands)
    existing = {item.edge_id for item in index.hyperedges}
    index.hyperedges.extend(
        item for item in catalog_hyperedges(index) if item.edge_id not in existing
    )
    return index
