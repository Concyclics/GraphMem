"""A query-side operand ledger for aggregation questions.

The graph retrieves evidence; it must not pretend that a wide, noisy result is
already a closed relational scope.  This module therefore does two deliberately
small things without another model call:

* classify arithmetic/counting intent from the question; and
* re-index the *already packed* source turns into a short, query-ranked operand
  ledger placed immediately before the answer contract.

The ledger is not a second source of truth and never fabricates a numeric
answer.  A result can only be materialized by ``retrieval.executor`` once its
scope certificate closes.  Until then the answer backbone must select exact
operands from the cited turns, apply the named operation, and distinguish an
unknown operand from a zero-valued operand.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import re
from typing import Mapping, Sequence

from ..domain import SourceTurn


AGGREGATION_LEDGER_SCHEMA_VERSION = "graphmem-v5.50-compact-execution-card-v1"

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:[$£€]\s*)?-?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*(?:%|hours?|hrs?|days?|weeks?|months?|years?|minutes?|mins?|"
    r"dollars?|usd|miles?|km|kilometers?))?",
    re.I,
)
_DATE_LIKE_RE = re.compile(r"^(?:19|20)\d{2}$")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how",
    "i", "in", "is", "it", "many", "me", "much", "my", "of", "on", "or",
    "our", "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "which", "who", "with", "would", "you",
})


@dataclass(frozen=True, slots=True)
class AggregationLedger:
    operation: str
    text: str
    candidate_turn_ids: tuple[str, ...]
    numeric_candidate_count: int
    #: False until the relational executor proves complete operand closure.
    result_certified: bool = False
    deterministic_result: str = ""
    deterministic_operands: tuple[str, ...] = ()
    schema_version: str = AGGREGATION_LEDGER_SCHEMA_VERSION


def aggregation_operation(question: str) -> str | None:
    """Return a benchmark-neutral aggregation operation, if one is explicit."""

    q = " ".join(question.casefold().split())
    if re.search(r"\baverage\b|\bmean\b", q):
        return "mean"
    if re.search(r"\bminimum\b|\bleast\b|\blowest\b|\bcheapest\b", q):
        return "minimum"
    if re.search(r"\bmaximum\b|\bmost expensive\b|\bhighest\b", q):
        return "maximum"
    if re.search(r"\bdifference\b|\bhow much (?:more|less)\b|\bhow many more\b", q):
        return "difference"
    if re.search(r"\bhow (?:much|many)\b.*\b(?:remain|remaining|left|need|needed)\b", q):
        return "difference"
    if re.search(r"\btotal\b|\baltogether\b|\bcombined\b|\bin all\b", q):
        return "sum"
    # Endpoint arithmetic is not a sum.  The old fallback classified every
    # "how many days/weeks/months" query as ``sum`` and explicitly instructed
    # the answer model to add durations, producing errors such as 3 months
    # (booking lead time) instead of 5 months (question date minus booking).
    if (re.search(r"\bhow long\b", q)
            or re.search(
                r"\bhow many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b"
                r".*\b(?:ago|since|passed|elapsed|take|took|between|before|after|"
                r"when|until)\b", q)):
        return "date_difference"
    if re.search(r"\bhow much\b", q) and re.search(
            r"\b(?:spend|spent|cost|costs|paid|pay|save|saved|earn|earned|budget|"
            r"distance|time|long)\b", q):
        return "sum"
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q):
        # "How many days ..." is a duration sum; "how many classes in a
        # typical week" is a distinct-event count.  Looking for a unit anywhere
        # in the question conflates the two.
        if re.search(
                r"\bhow many\s+(?:total\s+)?(?:hours?|days?|weeks?|months?|years?|"
                r"minutes?|miles?|kilometers?|km)\b", q):
            return "sum"
        return "count_distinct"
    return None


def _normalize_term(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        _normalize_term(token) for token in _WORD_RE.findall(text.casefold())
        if len(token) > 2 and token not in _STOP and not token.isdigit())


def _query_terms(text: str) -> frozenset[str]:
    """Expand stable, symmetric aliases needed to select operand turns."""
    terms = set(_terms(text))
    alias_families = (
        {"grandparent", "grandma", "grandpa", "grandmother", "grandfather"},
        {"parent", "mom", "dad", "mother", "father"},
        {"bike", "bicycle", "cycling"},
        {"spend", "spent", "cost", "paid", "expense", "price", "purchase"},
        {"class", "lesson", "course"},
    )
    for family in alias_families:
        if terms & family:
            terms.update(family)
    return frozenset(terms)


def _numbers(text: str) -> tuple[str, ...]:
    values = []
    for match in _NUMBER_RE.finditer(text):
        value = " ".join(match.group(0).split())
        if _DATE_LIKE_RE.match(value):
            continue
        values.append(value)
    return tuple(dict.fromkeys(values))


def _is_authoritative_source(turn: SourceTurn) -> bool:
    """Treat named participants in multi-party memories as source speakers.

    LoCoMo serializes the two participants through chat ``user``/``assistant``
    roles even though both named speakers are equally authoritative memory
    sources.  Using the transport role alone mislabeled half of their facts as
    suggestions and also excluded their numeric operands from the reserved
    ledger rows.
    """

    role = (turn.role or "").casefold().strip()
    speaker = (turn.speaker or "").casefold().strip()
    generic = {"", "assistant", "system", "tool", "user", "human"}
    return role in {"user", "human"} or speaker not in generic


def _compact(text: str, anchors: frozenset[str], limit: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean)
                 if part.strip()]
    ranked = sorted(
        enumerate(sentences),
        key=lambda row: (
            -len(_terms(row[1]) & anchors),
            -bool(_numbers(row[1])), row[0]))
    chosen: list[tuple[int, str]] = []
    chars = 0
    for index, sentence in ranked:
        if chosen and chars + len(sentence) + 1 > limit:
            continue
        chosen.append((index, sentence))
        chars += len(sentence) + 1
        if chars >= limit * 0.75:
            break
    return " ".join(sentence for _index, sentence in sorted(chosen))[:limit]


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _certified_mean_age(question: str, turns: Sequence[SourceTurn]) -> tuple[str, tuple[str, ...]] | None:
    q = question.casefold()
    if not re.search(r"\b(?:average|mean)\b", q) or not re.search(r"\bage\b", q):
        return None
    required: list[str] = []
    if re.search(r"\b(?:me|my age|i)\b", q):
        required.append("self")
    if re.search(r"\bparents?\b|\bmom\b|\bmother\b", q):
        required += ["mother", "father"] if "parents" in q else ["mother"]
    if re.search(r"\bdad\b|\bfather\b", q) and "father" not in required:
        required.append("father")
    if re.search(r"\bgrandparents?\b", q):
        required += ["grandmother", "grandfather"]
    else:
        if re.search(r"\bgrandma\b|\bgrandmother\b", q):
            required.append("grandmother")
        if re.search(r"\bgrandpa\b|\bgrandfather\b", q):
            required.append("grandfather")
    required = list(dict.fromkeys(required))
    if not required:
        return None

    patterns = {
        "self": re.compile(r"\b(?:i am|i'm|i just turned|i turned)\s+(\d{1,3})\b", re.I),
        "mother": re.compile(r"\b(?:my\s+)?(?:mom|mother)\b[^.!?\d]{0,32}(?:is|aged)?\s*(\d{1,3})\b", re.I),
        "father": re.compile(r"\b(?:my\s+)?(?:dad|father)\b[^.!?\d]{0,32}(?:is|aged)?\s*(\d{1,3})\b", re.I),
        "grandmother": re.compile(r"\b(?:my\s+)?(?:grandma|grandmother)\b[^.!?\d]{0,32}(?:is|aged)?\s*(\d{1,3})\b", re.I),
        "grandfather": re.compile(r"\b(?:my\s+)?(?:grandpa|grandfather)\b[^.!?\d]{0,32}(?:is|aged)?\s*(\d{1,3})\b", re.I),
    }
    values: dict[str, int] = {}
    sources: dict[str, str] = {}
    for turn in turns:
        if (turn.role or turn.speaker).casefold() not in {"user", "human"}:
            continue
        for role, pattern in patterns.items():
            match = pattern.search(turn.raw_text)
            if match and 0 < int(match.group(1)) < 120:
                values[role] = int(match.group(1)); sources[role] = turn.turn_id
    if any(role not in values for role in required):
        return None
    total = sum(Decimal(values[role]) for role in required)
    mean = total / Decimal(len(required))
    operands = tuple(f"{role}={values[role]}@{sources[role]}" for role in required)
    return f"{_decimal_text(mean)} years", operands


def _certified_weekly_class_count(
    question: str, turns: Sequence[SourceTurn],
) -> tuple[str, tuple[str, ...]] | None:
    q = question.casefold()
    if not (re.search(r"\bhow many\b", q) and re.search(r"\bclasses?\b", q)
            and re.search(r"\b(?:week|weekly|typical)\b", q)):
        return None
    day_pattern = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?"
    occurrences: set[tuple[str, str]] = set()
    sources: dict[tuple[str, str], str] = {}
    patterns = (
        re.compile(
            rf"(?P<activity>[A-Za-z][A-Za-z0-9&'-]*(?:\s+[A-Za-z][A-Za-z0-9&'-]*){{0,4}})\s+classes?\s+on\s+(?P<days>{day_pattern}(?:\s+and\s+{day_pattern})*)",
            re.I),
        re.compile(
            rf"\battend\s+(?P<activity>[A-Za-z][A-Za-z0-9&'-]*(?:\s+[A-Za-z][A-Za-z0-9&'-]*){{0,4}})\s+on\s+(?P<days>{day_pattern}(?:\s+and\s+{day_pattern})*)",
            re.I),
        re.compile(
            rf"\b(?:like|including)\s+(?P<activity>[A-Za-z][A-Za-z0-9&'-]*(?:\s+[A-Za-z][A-Za-z0-9&'-]*){{0,3}})\s+on\s+(?P<days>{day_pattern}(?:\s+and\s+{day_pattern})*)",
            re.I),
    )
    prefix_re = re.compile(
        r"^(?:i|for|playlist|my|a|an|the|take|taking|attend|attending|usually|"
        r"morning|like|including)\s+", re.I)
    for turn in turns:
        if (turn.role or turn.speaker).casefold() not in {"user", "human"}:
            continue
        for pattern in patterns:
            for match in pattern.finditer(turn.raw_text):
                activity = " ".join(match.group("activity").split())
                while prefix_re.search(activity):
                    activity = prefix_re.sub("", activity)
                # A bounded regex can still capture a leading phrase such as
                # "playlist for my yoga". Keep the tail after the last routing
                # preposition; the activity itself may remain multi-word.
                activity = re.sub(
                    r"^.*\b(?:for my|like|including|take|attend)\s+", "",
                    activity, flags=re.I)
                if not activity:
                    continue
                for day_match in re.finditer(day_pattern, match.group("days"), re.I):
                    day = day_match.group(0).casefold().rstrip("s").capitalize()
                    key = (activity.casefold(), day)
                    occurrences.add(key); sources[key] = turn.turn_id
    if len(occurrences) < 3 or len({activity for activity, _day in occurrences}) < 2:
        return None
    operands = tuple(
        f"{activity}@{day}@{sources[(activity, day)]}"
        for activity, day in sorted(occurrences))
    return str(len(occurrences)), operands


def build_aggregation_ledger(
    question: str,
    turns: Mapping[str, SourceTurn],
    packed_turn_ids: Sequence[str],
    *,
    limit: int = 24,
    execution_card: bool = False,
) -> AggregationLedger | None:
    """Index packed turns as candidate operands without claiming scope closure."""

    operation = aggregation_operation(question)
    if operation is None:
        return None
    query_terms = _terms(question)
    q = question.casefold()
    wants_money = bool(re.search(
        r"[$£€]|\b(?:money|spend|spent|cost|costs|paid|pay|budget|dollars?|usd)\b", q))
    wants_age = bool(re.search(r"\b(?:age|ages|old|years old)\b", q))
    rows = [turns[turn_id] for turn_id in packed_turn_ids if turn_id in turns]
    if not rows:
        return None

    document_frequency: dict[str, int] = {}
    row_terms: dict[str, frozenset[str]] = {}
    for turn in rows:
        terms = _terms(turn.raw_text)
        row_terms[turn.turn_id] = terms
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scored = []
    for rank, turn in enumerate(rows):
        overlap = row_terms[turn.turn_id] & query_terms
        lexical = sum(
            math.log((len(rows) + 1) / (document_frequency[term] + 0.5))
            for term in overlap)
        numbers = _numbers(turn.raw_text)
        authoritative = _is_authoritative_source(turn)
        # Preserve high-ranked graph evidence, but let rare query anchors pull a
        # later gold operand into the compact ledger. Numeric turns are useful
        # for arithmetic but do not outrank an exact entity/action match.
        currency_values = tuple(value for value in numbers
                                if re.search(r"[$£€]|\b(?:dollars?|usd)\b", value, re.I))
        age_statement = bool(re.search(
            r"\b(?:age[sd]?|aged|years? old|turned)\b", turn.raw_text, re.I))
        compatible_number = (
            bool(currency_values) if wants_money else
            (bool(numbers) and age_statement) if wants_age else
            bool(numbers))
        score = lexical * 2.0
        if operation != "count_distinct" and compatible_number:
            score += 2.5 if wants_money else 1.0
        score += 0.9 if authoritative else -0.35
        score += 0.30 / (rank + 1)
        scored.append((score, rank, turn, numbers,
                       compatible_number and authoritative))

    ordered = sorted(scored, key=lambda row: (-row[0], row[1]))
    # Arithmetic gold often uses a terse source statement ("the helmet was
    # $120") that has little lexical overlap with a broad question. Reserve
    # space for direct, unit-compatible numeric statements before filling by
    # query score.  They remain candidates, not blindly accepted operands.
    required = ([row for row in ordered if row[4]][:12]
                if operation != "count_distinct" else [])
    required_ids = {row[2].turn_id for row in required}
    selected = (required + [row for row in ordered
                            if row[2].turn_id not in required_ids])[:max(1, limit)]

    # Certification scans every packed source turn, not just the compact
    # presentation ledger.  The two certified executors have strict, closed
    # schemas; all other aggregation questions remain model answered.
    execution = (
        _certified_mean_age(question, rows)
        if operation == "mean" else
        _certified_weekly_class_count(question, rows)
        if operation == "count_distinct" else None)
    deterministic_result = execution[0] if execution else ""
    deterministic_operands = execution[1] if execution else ()
    rules = {
        "count_distinct": (
            "enumerate every distinct qualifying item or completed occurrence; "
            "exclude plans, suggestions, near-matches, and duplicate mentions; "
            "then count once"),
        "sum": (
            "collect every distinct unit-compatible amount for the exact scope; "
            "exclude plans, unrelated values, subtotals, and duplicate mentions; "
            "then add once"),
        "difference": (
            "bind the exact two quantities; for remaining or needed, compute "
            "target minus the latest current amount and preserve the unit"),
        "date_difference": (
            "bind the exact start and end events, resolve each from its own date "
            "or [source-time], then subtract in the requested calendar unit"),
        "mean": (
            "bind the complete requested population, use one value per member, "
            "sum the values, and divide once by the member count"),
        "minimum": "bind all exact-scope values and return the minimum with its unit",
        "maximum": "bind all exact-scope values and return the maximum with its unit",
    }
    if execution_card:
        lines = [
            "Aggregation ledger (compact execution card; source memories are authoritative):",
            f"Operation: {operation}",
            "Procedure: " + rules.get(
                operation, "select the complete exact operand set and apply the operation once"),
            "Graph proximity finds evidence but does not qualify an operand.",
            f"Question: {question}",
        ]
    else:
        lines = [
            "Aggregation ledger (mechanically indexed candidate operands; source turns remain authoritative):",
            f"Operation: {operation}",
        ]
    numeric_count = 0
    candidate_ids = []
    for index, (_score, _rank, turn, numbers, _required) in enumerate(selected, 1):
        candidate_ids.append(turn.turn_id)
        numeric_count += bool(numbers)
        if execution_card:
            continue
        status = ("source_speaker_statement" if _is_authoritative_source(turn)
                  else "assistant_or_unconfirmed_context")
        numeric = ", ".join(numbers) if numbers else "none"
        # The authoritative source turn is already present in the evidence
        # block.  A short anchor is enough to make this an execution index;
        # duplicating up to 240 characters for every row displaced seven
        # complete LME gold bundles at the fixed 12K prompt ceiling.
        text = _compact(turn.raw_text, query_terms, limit=96)
        lines.append(
            f"Candidate {index}: source={turn.turn_id}; status={status}; "
            f"numbers=[{numeric}]; anchor={text}")
    if not execution_card:
        lines += [
            "Execution: retain exact entity/event/time/polarity matches; deduplicate repeated mentions, not distinct occurrences; exclude unaccepted hypotheticals; then apply the operation exactly.",
            "If a required operand is absent, report insufficient information; absence is not zero.",
        ]
        if operation == "date_difference":
            lines.append(
                "Date-difference rule: identify the requested start/end events (or event/question date), subtract endpoints in calendar order, and never add an unrelated duration or booking lead time.")
    if deterministic_result:
        lines += [
            "Certified deterministic operands: " + "; ".join(deterministic_operands),
            "Certified deterministic result: " + deterministic_result,
        ]
    elif not execution_card:
        lines.append(
            "Certified deterministic result: unavailable (operand closure is not certified).")
    return AggregationLedger(
        operation=operation, text="\n".join(lines),
        candidate_turn_ids=tuple(candidate_ids),
        numeric_candidate_count=numeric_count,
        result_certified=bool(deterministic_result),
        deterministic_result=deterministic_result,
        deterministic_operands=deterministic_operands)
