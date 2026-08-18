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
# Open-ended advice is a synthesis request, not a request for one graph
# predicate whose wording happens to overlap the question.  Keep this wording
# router shared by retrieval and answer rendering so the two stages cannot
# silently disagree about whether a question is a recommendation.
ADVICE_QUERY_RE = re.compile(
    r"\b(?:can|could|would) you (?:recommend|suggest|give|help)\b|"
    r"\bdo you have (?:any|some) (?:\w+\s+){0,3}"
    r"(?:tips?|advice|suggestions?|recommendations?|ideas?)\b|"
    r"\b(?:any|some) (?:\w+\s+){0,3}"
    r"(?:tips?|advice|suggestions?|recommendations?|ideas?)\b|"
    r"\bwhat (?:should|could) i\b|\bwhat do you think\b|"
    r"\bdo you think\b|\bcould there be a reason\b",
    re.I,
)
# Plural answer heads mean a list is expected even when only one owner is named.
PLURAL_HINTS = {
    "places", "cities", "countries", "locations", "venues", "restaurants", "people",
    "friends", "things", "books", "games", "pets", "items", "activities", "events",
    "topics", "hobbies", "shows", "movies", "songs", "projects", "courses", "ones",
}



# Words that carry no discriminating content for matching a question against a
# collection.  Question frames, auxiliaries, pronouns and determiners all appear
# in nearly every question, so leaving them in makes every manifest "match".
QUESTION_STOPWORDS = frozenset({
    "how", "many", "much", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "did", "does", "do", "done", "was", "were", "is", "are", "am", "be", "been",
    "being", "has", "have", "had", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "from", "with", "by", "about", "into", "over", "and", "or", "but", "if", "than",
    "then", "that", "this", "these", "those", "there", "here", "it", "its", "i", "me",
    "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "total", "number", "count", "times", "time", "ever",
    "also", "just", "still", "any", "all", "some", "so", "as", "up", "out", "get",
    "got", "go", "went", "make", "made", "take", "took", "say", "said", "tell", "know",
    "please", "list", "name", "give", "show", "far", "long", "old", "now", "currently",
})
# Verb-ish suffixes used to split question content into "what happened" and
# "what it happened to", without a POS tagger or any model call.
_VERB_SUFFIXES = ("ed", "ing", "ies", "es", "s")


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
    is_sum: bool = False
    is_unit_rate: bool = False
    is_existence: bool = False
    is_duration: bool = False
    is_latest: bool = False
    is_list: bool = False
    is_advice: bool = False
    possessive: bool = False
    expects_multiple: bool = False
    indirect: bool = False
    temporal_key: object | None = None
    # Terms taken from the *question*, which is what a collection has to be
    # identified by.  Operand predicate candidates are retrieved from the graph
    # by embedding similarity, so they contain none of the question's own words:
    # "how many antique items did I inherit" produced candidates like "plans a
    # monthly family game night", which match nearly every manifest.
    content_terms: tuple[str, ...] = ()
    #: The head noun phrase being asked about ("antique items", "rollercoasters").
    head_terms: tuple[str, ...] = ()
    #: Content terms that look like the question's verb ("inherit", "spent").
    action_terms: tuple[str, ...] = ()
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


def is_advice_query(query: str) -> bool:
    """Detect an open-ended advice/recommendation request from its wording."""

    return bool(ADVICE_QUERY_RE.search(" ".join(query.split())))


def _question_terms(row: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split a question's own words into content, head-noun and action terms.

    Purely lexical and deterministic: no POS tagger, no model call.  The head is
    what follows the counting phrase ("how many *antique items*"), which is the
    collection being asked about; action terms are the residue that looks verbal
    ("inherit", "spent").  Both are needed because a collection is keyed by
    (owner, predicate, ...) and the question names the predicate, while the
    manifest's members carry the values.
    """
    content = tuple(dict.fromkeys(
        token for token in row
        if token not in QUESTION_STOPWORDS and len(token) > 2))
    head: list[str] = []
    for index, token in enumerate(row):
        if token in {"many", "much"} and index + 1 < len(row):
            # Take the noun phrase until an auxiliary or pronoun closes it.
            for candidate in row[index + 1:]:
                if candidate in QUESTION_STOPWORDS:
                    break
                head.append(candidate)
            break
    if not head:
        head = [token for token in content[:3]]
    actions = tuple(token for token in content
                    if token not in set(head) and token.endswith(_VERB_SUFFIXES))
    if not actions:
        actions = tuple(token for token in content if token not in set(head))
    return content, tuple(dict.fromkeys(head)), actions


def parse_slots(query: str) -> QuerySlots:
    stripped, indirect = _strip_frames(query)
    row = terms(stripped)
    tokens = frozenset(row)
    lowered = stripped.casefold()
    warnings: list[str] = []

    leading = row[0] if row else ""
    is_count = ("how" in tokens and _has(tokens, COUNT_HEADS)) or leading == "count"
    # A per-item price is a quotient, not a sum.  Detect it before the broad
    # "how much did I spend" rule so a total plus an item count is divided once
    # instead of instructing the answer model to add unrelated currency values.
    is_unit_rate = bool(
        "how" in tokens and "much" in tokens
        and tokens & {"each", "per"}
        and tokens & {"spent", "spend", "paid", "pay", "cost", "costs", "price"}
    )
    # A total expenditure is a numeric reduction over several facts, not the
    # number of facts and not a scalar price lookup.
    is_sum = (not is_unit_rate) and (bool(
        (tokens & {"sum", "total", "altogether", "combined"})
        and tokens & {
            "amount", "amounts", "spent", "spend", "paid", "pay",
            "cost", "costs", "sum", "total",
        }
    ) or bool(
        "how" in tokens and "much" in tokens
        and tokens & {"spent", "spend", "paid", "pay", "altogether", "total"}
    ))
    is_duration = "how" in tokens and _has(tokens, DURATION_WORDS)
    is_advice = is_advice_query(query)
    # "How long" is a duration question, not a count one, even though both open
    # with "how".
    if is_duration:
        is_count = False
    if is_sum or is_unit_rate:
        is_count = False

    quantifier = next((word for word in ("both", "each", "either", "every", "all", "any",
                                         "respectively") if word in tokens), "")
    # An existence question leads with an auxiliary verb; "Did they both ..." has
    # to reach EXISTS_ALL rather than being captured by the quantifier.
    is_existence = (
        leading in EXIST_LEADS and not is_count and not is_sum
        and not is_duration and not is_advice)

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
    # "How many days/weeks passed between ..." asks for an interval, not a
    # cardinality over events. The unit plus an explicit interval relation is a
    # deterministic discriminator from ordinary "how many days did I visit"
    # collection questions.
    temporal_units = {
        "second", "seconds", "minute", "minutes", "hour", "hours",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
    }
    temporal_count = bool(
        is_count and tokens & temporal_units
        and (temporal_relation or tokens & {"passed", "lapsed", "elapsed"}))
    if temporal_count:
        is_duration = True
        is_count = False

    is_latest = _has(tokens, LATEST_WORDS) and ordinal_index is None
    # "latest"/"most recent" are ordinals over time, not a state lookup.
    if ordinal_index == -1:
        is_latest = False

    negation = _has(tokens, NEGATION_WORDS)
    is_list = ((leading in LIST_HEADS or "list" in tokens)
               and not is_count and not is_sum and not is_duration)
    # A plural answer head asks for a set even with a single owner named.
    expects_multiple = (bool(tokens & PLURAL_HINTS)
                        or quantifier in {"all", "every", "both", "each"}
                        or is_unit_rate)

    answer_slot = next((name for name, words in ANSWER_SLOTS if _has(tokens, words)), "")
    value_type = next((name for name, words in VALUE_TYPES if _has(tokens, words)), None)
    distinct_by = "event_instance" if _has(tokens, EVENT_DISTINCT_WORDS) else "value"

    from ..build.temporal import extract_time_expression

    temporal_phrase = extract_time_expression(query) or ""
    temporal_key = None
    if temporal_phrase:
        from ..build.temporal import normalize_time
        from ..domain import TemporalKey
        temporal_key = TemporalKey.from_attribute(
            normalize_time(temporal_phrase, None, "query"))
    possessive = "'s" in lowered or "'" in lowered
    content_terms, head_terms, action_terms = _question_terms(row)

    if is_count and is_existence:
        warnings.append("count_and_existence_both_detected")
    if ordinal_index is not None and temporal_relation in {"before", "after"}:
        warnings.append("ordinal_with_temporal_relation")

    return QuerySlots(
        tokens=row, answer_slot=answer_slot, value_type=value_type, quantifier=quantifier,
        ordinal_index=ordinal_index, ordinal_order=ordinal_order,
        temporal_relation=temporal_relation, temporal_phrase=temporal_phrase,
        distinct_by=distinct_by, polarity="negative" if negation else "positive",
        negation=negation, is_count=is_count, is_sum=is_sum,
        is_unit_rate=is_unit_rate,
        is_existence=is_existence,
        is_duration=is_duration, is_latest=is_latest, is_list=is_list,
        is_advice=is_advice,
        possessive=possessive, expects_multiple=expects_multiple, indirect=indirect,
        temporal_key=temporal_key,
        content_terms=content_terms, head_terms=head_terms, action_terms=action_terms,
        warnings=tuple(warnings),
    )
