from __future__ import annotations

import re
from typing import Any

from .schema import QueryFrame


_MEDIA_RE = re.compile(
    r"\[Media shared by (?P<speaker>[^;\]]+);\s*caption:\s*(?P<caption>[^\]]+)\]",
    re.IGNORECASE,
)
_GENERIC_CAPTION_TERMS = {
    "a", "an", "and", "at", "by", "caption", "drawing", "food", "group",
    "image", "in", "media", "of", "on", "one", "photo", "photograph",
    "photography", "picture", "plate", "bowl", "bowls", "scene", "shared",
    "table", "the", "their", "his", "her", "two", "with",
}
_QUERY_ATTRIBUTE_STOP = {
    "kind", "type", "october", "september", "august", "july", "june",
    "may", "april", "march", "february", "january", "november",
    "december",
}


def _token_key(value: str) -> str:
    token = value.casefold().strip("'\"")
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        _token_key(token)
        for token in re.findall(r"[\w'-]+", value)
        if token.casefold() not in _GENERIC_CAPTION_TERMS
        and not token.isdigit()
    }


def media_attribute_hint(
    frame: QueryFrame,
    turns: list[Any],
) -> dict[str, Any] | None:
    """Bind a media question to typed caption content and local description."""
    lowered_question = frame.raw_question.casefold()
    has_media_noun = bool(re.search(
        r"\b(?:image|photo|photograph|picture|screenshot|drawing)\w*\b",
        lowered_question,
    ))
    has_share_verb = bool(re.search(
        r"\b(?:share|shared|sharing|send|sent)\b", lowered_question
    ))
    has_show_verb = bool(
        re.search(r"\b(?:showed|shown|showing)\b", lowered_question)
        or re.search(
            r"\b(?:did|does|do|will|can|was|were)\b.{0,60}\bshow\b",
            lowered_question,
        )
    )
    if not (has_media_noun or has_share_verb or has_show_verb):
        return None
    participants = {value.casefold() for value in frame.participant_terms}
    relation_match = re.search(
        r"\b(?:after|before)\s+(?P<target>[^?.,;]+)",
        frame.raw_question,
        re.IGNORECASE,
    )
    relation_terms = _tokens(relation_match.group("target")) if relation_match else set()
    query_terms = {
        _token_key(term) for term in frame.content_terms
        if term not in {
            "image", "photo", "photograph", "picture", "screenshot",
            "show", "share", "shared", "send", "sent",
        }
        and term not in {value.casefold() for value in frame.participant_terms}
        and term not in frame.temporal_terms
        and term not in _QUERY_ATTRIBUTE_STOP
        and not term.isdigit()
    }
    exact_dates = {
        value for value in frame.explicit_dates
        if re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}", value)
    }
    rows: list[dict[str, Any]] = []
    for turn in turns:
        text = str(getattr(turn, "text", ""))
        for match in _MEDIA_RE.finditer(text):
            speaker = match.group("speaker").strip()
            if participants and speaker.casefold() not in participants:
                continue
            session_date = str(getattr(turn, "session_date", ""))
            normalized_date = None
            date_match = re.search(
                r"\b(\d{1,2})\s+([A-Za-z]+),?\s+((?:19|20)\d{2})\b",
                session_date,
            )
            if date_match:
                months = {
                    name: index for index, name in enumerate(
                        (
                            "january", "february", "march", "april", "may",
                            "june", "july", "august", "september", "october",
                            "november", "december",
                        ),
                        start=1,
                    )
                }
                month = months.get(date_match.group(2).casefold())
                if month:
                    normalized_date = (
                        f"{int(date_match.group(3)):04d}-{month:02d}-"
                        f"{int(date_match.group(1)):02d}"
                    )
            caption = match.group("caption").strip()
            rows.append({
                "caption": caption,
                "speaker": speaker,
                "date": normalized_date or session_date,
                "source_turn_id": str(getattr(turn, "node_id", "")),
                "session_id": str(getattr(turn, "session_id", "")),
                "turn_index": int(getattr(turn, "turn_index", -1)),
                "specific_terms": sorted(_tokens(caption)),
                "enumerated_attributes": bool(
                    "," in caption and re.search(r"\band\b", caption, re.IGNORECASE)
                ),
                "exact_date_match": normalized_date in exact_dates,
                "relation_coverage": (
                    len(relation_terms & _tokens(text)) / len(relation_terms)
                    if relation_terms else 0.0
                ),
                "query_coverage": (
                    len(query_terms & _tokens(text)) / len(query_terms)
                    if query_terms else 0.0
                ),
            })
    if not rows:
        return None
    if exact_dates and any(row["exact_date_match"] for row in rows):
        rows = [row for row in rows if row["exact_date_match"]]
    artifact_rows = [row for row in rows if row["query_coverage"] > 0]
    if query_terms and artifact_rows:
        rows = artifact_rows
    rows.sort(
        key=lambda row: (
            row["relation_coverage"],
            row["query_coverage"],
            row["turn_index"],
            row["enumerated_attributes"],
            len(row["specific_terms"]),
            row["source_turn_id"],
        ),
        reverse=True,
    )
    top = rows[0]
    relation_bound = bool(
        relation_terms
        and top["relation_coverage"] >= 0.75
        and not any(
            row["relation_coverage"] == top["relation_coverage"]
            for row in rows[1:]
        )
    )
    query_bound = bool(
        query_terms
        and top["query_coverage"] >= 0.34
        and not any(
            row["query_coverage"] >= top["query_coverage"] - 0.08
            for row in rows[1:]
        )
    )
    latest_artifact_bound = bool(
        exact_dates
        and query_terms
        and top["query_coverage"] > 0
        and not any(
            row["turn_index"] == top["turn_index"] for row in rows[1:]
        )
    )
    enumerated_bound = bool(
        exact_dates
        and top["enumerated_attributes"]
        and re.search(r"\b(?:objects?|items?|things?|which)\b", lowered_question)
        and not any(row["enumerated_attributes"] for row in rows[1:])
    )
    complete = (
        relation_bound
        or query_bound
        or latest_artifact_bound
        or enumerated_bound
        or len(rows) == 1
    )
    value = re.sub(
        r"^(?:a|an)\s+(?:photo|photograph(?:y)?|picture|image)\s+of\s+",
        "",
        top["caption"],
        flags=re.IGNORECASE,
    )
    ordered = sorted(
        [
            turn for turn in turns
            if str(getattr(turn, "session_id", "")) == top["session_id"]
        ],
        key=lambda turn: int(getattr(turn, "turn_index", -1)),
    )
    source_position = next(
        (
            index for index, turn in enumerate(ordered)
            if str(getattr(turn, "node_id", "")) == top["source_turn_id"]
        ),
        -1,
    )
    followups = []
    if source_position >= 0:
        followups = [
            turn for turn in ordered[source_position + 1:source_position + 3]
            if str(getattr(turn, "speaker", "")).casefold()
            == str(top["speaker"]).casefold()
        ]
    source_turn = ordered[source_position] if source_position >= 0 else None
    source_prose = (
        _MEDIA_RE.sub("", str(getattr(source_turn, "text", ""))).strip()
        if source_turn is not None else ""
    )
    evidence = [value]
    if source_prose:
        evidence.append(source_prose)
    evidence.extend(str(getattr(turn, "text", "")) for turn in followups)
    source_ids = list(dict.fromkeys([
        top["source_turn_id"],
        *[str(getattr(turn, "node_id", "")) for turn in followups],
        *[row["source_turn_id"] for row in rows[:8]],
    ]))
    return {
        "operation": "media_attribute",
        "value": value,
        "candidates": rows[:8],
        "source_turn_ids": source_ids,
        "evidence": evidence,
        "complete": complete,
        "completion_basis": (
            "event_relation_bound_caption"
            if relation_bound else "unique_caption"
            if len(rows) == 1 else "query_bound_caption"
            if query_bound else "exact_date_unique_enumerated_caption"
            if enumerated_bound else "exact_date_artifact_type_latest_occurrence"
            if latest_artifact_bound else "ambiguous_caption_candidates"
        ),
    }
