"""Question slot parsing for the V5.6 query compiler.

V5.5 chose a single ``QueryOperator`` by racing substring tests in a fixed
order.  That loses composition ("how many ... both" keeps only the count), and
the substring tests misfire: ``"now"`` matches inside ``"know"``, the ordinal
branch pre-empts before/after, and ``EXISTS_ALL`` is unreachable for
"Did they both ...".

Parsing the slots first and composing an operator afterwards makes both
problems go away, and makes every decision individually testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import terms


ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "last": -1, "latest": -1, "most recent": -1,
}
QUANTIFIER_WORDS = {"both", "each", "all", "every", "any", "either", "respectively"}
NEGATION_WORDS = {"not", "never", "no", "none", "neither", "without", "didn't", "don't", "doesn't"}
COUNT_HEADS = {"many", "much", "often", "count", "number", "total"}
EVENT_DISTINCT_WORDS = {"times", "occasions", "instances", "occurrences"}
LATEST_WORDS = {"currently", "current", "now", "nowadays", "latest", "today", "present"}
DURATION_WORDS = {"long", "duration", "gap", "interval", "elapsed"}
LIST_HEADS = {"what", "which", "who", "where", "list", "name"}
EXIST_LEADS = {"did", "does", "do", "was", "were", "has", "have", "had", "is", "are", "can", "could"}

ANSWER_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("place", ("where", "place", "places", "city", "cities", "country", "countries",
               "location", "locations", "venue", "restaurant")),
    ("person", ("who", "whom", "person", "people", "friend", "friends", "family")),
    ("date", ("when", "date", "day", "month", "year", "time")),
    ("count", ("many", "much", "number", "count", "total")),
    ("duration", ("long", "duration")),
    ("reason", ("why", "reason", "because")),
    ("manner", ("how",)),
)
VALUE_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("location", ("where", "city", "cities", "country", "countries", "place", "places")),
    ("time", ("when", "date", "day", "month", "year")),
    ("number", ("many", "much", "number", "count", "total", "how much")),
    ("currency", ("cost", "price", "paid", "spend", "spent", "dollars")),
)
TEMPORAL_RELATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("between", ("between",)),
    ("before", ("before", "prior", "earlier", "preceding")),
    ("after", ("after", "since", "following", "later")),
    ("until", ("until", "till")),
)
_ORDINAL_DIGITS = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I)
# "Do you know what X is?" is an indirect question, not an existence check.
# Stripping the frame first stops the leading auxiliary from hijacking it.
_INDIRECT_FRAME = re.compile(
    r"^\s*(?:do|does|did|can|could|would|will)\s+(?:you|we|i)\s+"
    r"(?:know|recall|remember|tell\s+me|say)\s*(?:me\s+)?(?:about\s+)?", re.I)
_POLITE_FRAME = re.compile(r"^\s*(?:please\s+)?(?:tell\s+me|remind\s+me|list)\s*(?:about\s+)?", re.I)
# Plural answer heads mean a list is expected even when only one owner is named.
PLURAL_HINTS = {
    "places", "cities", "countries", "locations", "venues", "restaurants", "people",
    "friends", "things", "books", "games", "pets", "items", "activities", "events",
    "topics", "hobbies", "shows", "movies", "songs", "projects", "courses", "ones",
}


@dataclass(frozen=True, slots=True)
class QuerySlots:
    """Everything the composer needs, parsed once from the question."""

    tokens: tuple[str, ...] = ()
    answer_slot: str = ""
    value_type: str | None = None
    quantifier: str = ""
    ordinal_index: int | None = None
    ordinal_order: str = "ascending"
    temporal_relation: str = ""
    temporal_phrase: str = ""
    distinct_by: str = "value"
    polarity: str = "positive"
    negation: bool = False
    is_count: bool = False
    is_existence: bool = False
    is_duration: bool = False
    is_latest: bool = False
    is_list: bool = False
    possessive: bool = False
    expects_multiple: bool = False
    indirect: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _strip_frames(query: str) -> tuple[str, bool]:
    """Remove indirect/polite question frames before parsing.

    Without this, "Do you know where Alice lives?" leads with an auxiliary and
    parses as an existence check about the listener rather than a lookup.
    """
    stripped = _INDIRECT_FRAME.sub("", query)
    indirect = stripped != query
    without_polite = _POLITE_FRAME.sub("", stripped)
    if without_polite != stripped:
        indirect = True
        stripped = without_polite
    return (stripped.strip() or query), indirect


def _has(tokens: frozenset[str], words) -> bool:
    """Whole-token membership, so 'now' never matches inside 'know'."""
    return bool(tokens & set(words))


def parse_slots(query: str) -> QuerySlots:
    stripped, indirect = _strip_frames(query)
    row = terms(stripped)
    tokens = frozenset(row)
    lowered = stripped.casefold()
    warnings: list[str] = []

    leading = row[0] if row else ""
    is_count = ("how" in tokens and _has(tokens, COUNT_HEADS)) or leading == "count"
    is_duration = "how" in tokens and _has(tokens, DURATION_WORDS)
    # "How long" is a duration question, not a count one, even though both open
    # with "how".
    if is_duration:
        is_count = False

    quantifier = next((word for word in ("both", "each", "either", "every", "all", "any",
                                         "respectively") if word in tokens), "")
    # An existence question leads with an auxiliary verb; "Did they both ..." has
    # to reach EXISTS_ALL rather than being captured by the quantifier.
    is_existence = leading in EXIST_LEADS and not is_count and not is_duration

    ordinal_index: int | None = None
    ordinal_order = "ascending"
    for word, index in ORDINAL_WORDS.items():
        if " " in word:
            if word in lowered:
                ordinal_index = index
                break
        elif word in tokens:
            ordinal_index = index
            break
    digits = _ORDINAL_DIGITS.search(query)
    if ordinal_index is None and digits:
        ordinal_index = int(digits.group(1))
    if ordinal_index == -1:
        ordinal_order = "descending"

    temporal_relation = ""
    for name, words in TEMPORAL_RELATIONS:
        if _has(tokens, words):
            temporal_relation = name
            break

    is_latest = _has(tokens, LATEST_WORDS) and ordinal_index is None
    # "latest"/"most recent" are ordinals over time, not a state lookup.
    if ordinal_index == -1:
        is_latest = False

    negation = _has(tokens, NEGATION_WORDS)
    is_list = (leading in LIST_HEADS or "list" in tokens) and not is_count and not is_duration
    # A plural answer head asks for a set even with a single owner named.
    expects_multiple = bool(tokens & PLURAL_HINTS) or quantifier in {"all", "every", "both", "each"}

    answer_slot = next((name for name, words in ANSWER_SLOTS if _has(tokens, words)), "")
    value_type = next((name for name, words in VALUE_TYPES if _has(tokens, words)), None)
    distinct_by = "event_instance" if _has(tokens, EVENT_DISTINCT_WORDS) else "value"

    from ..build.temporal import extract_time_expression

    temporal_phrase = extract_time_expression(query) or ""
    possessive = "'s" in lowered or "'" in lowered

    if is_count and is_existence:
        warnings.append("count_and_existence_both_detected")
    if ordinal_index is not None and temporal_relation in {"before", "after"}:
        warnings.append("ordinal_with_temporal_relation")

    return QuerySlots(
        tokens=row, answer_slot=answer_slot, value_type=value_type, quantifier=quantifier,
        ordinal_index=ordinal_index, ordinal_order=ordinal_order,
        temporal_relation=temporal_relation, temporal_phrase=temporal_phrase,
        distinct_by=distinct_by, polarity="negative" if negation else "positive",
        negation=negation, is_count=is_count, is_existence=is_existence,
        is_duration=is_duration, is_latest=is_latest, is_list=is_list,
        possessive=possessive, expects_multiple=expects_multiple, indirect=indirect,
        warnings=tuple(warnings),
    )
