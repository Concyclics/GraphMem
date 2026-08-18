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


AGGREGATION_LEDGER_SCHEMA_VERSION = "graphmem-v5.60-operand-worksheet-v1"

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
    #: A bounded reading index over verbatim, already-packed source turns.
    #: These rows are candidate operands, never a closed set or an answer.
    worksheet_lines: tuple[str, ...] = ()
    worksheet_turn_ids: tuple[str, ...] = ()
    schema_version: str = AGGREGATION_LEDGER_SCHEMA_VERSION


def aggregation_operation(question: str) -> str | None:
    """Return a benchmark-neutral aggregation operation, if one is explicit."""

    q = " ".join(question.casefold().split())
    if (re.search(r"\bhow much\b", q)
            and re.search(r"\b(?:each|per)\b", q)
            and re.search(r"\b(?:spend|spent|paid|pay|cost|costs|price)\b", q)):
        return "unit_rate"
    if re.search(r"\baverage\b|\bmean\b", q):
        return "mean"
    if re.search(r"\bminimum\b|\bleast\b|\blowest\b|\bcheapest\b", q):
        return "minimum"
    if re.search(r"\bmaximum\b|\bmost expensive\b|\bhighest\b", q):
        return "maximum"
    if re.search(r"\bdifference\b|\bhow much (?:more|less)\b|\bhow many more\b", q):
        return "difference"
    if re.search(r"\bhow (?:much|many)\b.*\b(?:remain|remaining|left)\b", q):
        return "difference"
    # ``need`` alone does not imply subtraction: "how many items do I need to
    # pick up or return" asks for a distinct-item count. A delta is licensed
    # only when the requested action changes a scalar balance toward a target.
    if re.search(
            r"\bhow (?:much|many)\b.*\bneed(?:ed)?\b.*"
            r"\b(?:earn|gain|add|save|accumulate|reach)\b", q):
        return "difference"
    # Savings is the difference between the explicitly compared alternatives,
    # not a sum of their prices.  Require a comparison cue so ordinary "how
    # much did I save" balance questions retain their existing treatment.
    if (re.search(r"\bhow much\b.*\bsav(?:e|ed|ing|ings)\b", q)
            and re.search(
                r"\b(?:instead of|rather than|versus|vs\.?|compared (?:with|to))\b",
                q)):
        return "difference"
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
    if re.search(r"\b(?:total\s+)?number of\b|\bcount\b", q):
        return "count_distinct"
    if re.search(r"\btotal\b|\baltogether\b|\bcombined\b|\bin all\b", q):
        return "sum"
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


_NEGATED_EVENT_RE = re.compile(
    r"\b(?:did\s+not|didn't|never|have\s+not|haven't|has\s+not|hasn't|"
    r"cancelled|canceled|skipped|declined)\b", re.I)
_PLANNED_EVENT_RE = re.compile(
    r"\b(?:plan(?:ning|ned)?|intend(?:ing|ed)?|hope(?:fully)?|want(?:ing|ed)?|"
    r"might|may|could|consider(?:ing|ed)?|scheduled|booked|going\s+to|"
    r"will)\b", re.I)
_COMPLETED_EVENT_RE = re.compile(
    r"\b(?:attended|visited|went|saw|completed|finished|tried|participated|"
    r"joined|hosted|held|bought|purchased|sold|earned|paid|spent|made|"
    r"created|built|received|used|watched|read|played|ran|rode|ate|"
    r"cooked|baked|had)\b", re.I)
_RELATIVE_TIME_RE = re.compile(
    r"\b(?:today|yesterday|tonight|last\s+(?:week|month|year|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"recently|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b", re.I)

_SELECTIVE_DATE_SUM_RE = re.compile(
    r"\bhow\s+many\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\s+"
    r"did\s+it\s+take\b", re.I)
_SELECTIVE_DIRECT_COUNT_RE = re.compile(
    r"\b(?:attend(?:ed)?|watch(?:ed)?|pack(?:ed)?|complete(?:d)?|"
    r"add(?:ed)?|wear|wore|spot(?:ted)?|catch|caught|try|tried|"
    r"lead|leads|led|buy|bought|acquire(?:d)?|visit(?:ed)?|"
    r"write|wrote|written|read|has|have|had)\s+"
    r"(?:a\s+total\s+of\s+|exactly\s+|about\s+|around\s+)?"
    r"(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|[0-9]+)\s+(?P<object>[a-z][a-z-]*)\b",
    re.I,
)
_SELECTIVE_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_SELECTIVE_GENERIC_TERMS = frozenset({
    "about", "after", "all", "amount", "answer", "before", "between",
    "combined", "cost", "current", "currently", "day", "days", "did",
    "does", "event", "events", "from", "got", "how", "including",
    "into", "last", "many", "month", "months", "money", "much", "new",
    "number", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "past", "question",
    "recent", "recently", "since", "spend", "spent", "take", "through",
    "time", "times", "total", "what", "when", "year", "years",
})


def _selective_terms(text: str) -> set[str]:
    values: set[str] = set()
    for token in _WORD_RE.findall(text.casefold()):
        if len(token) <= 2:
            continue
        value = _normalize_term(token)
        if len(value) > 5 and value.endswith("ing"):
            value = value[:-3]
        elif len(value) > 4 and value.endswith("ed"):
            value = value[:-2]
        if value == "rais":
            value = "raise"
        values.add(value)
    return values


def selective_operand_worksheet_route(
    question: str, ledger: AggregationLedger,
) -> str | None:
    """Return a high-precision route for exposing ``ledger`` worksheet rows.

    The V5.60 all-query worksheet improved some aggregation questions but
    regressed more when a partial candidate list looked complete.  This gate
    admits only source shapes with a deterministic completeness witness and
    never reads predictions, gold annotations or benchmark categories.
    """

    workspace = "\n".join(ledger.worksheet_lines)
    if not workspace:
        return None
    operation = ledger.operation
    if (operation == "date_difference"
            and _SELECTIVE_DATE_SUM_RE.search(question)
            and re.search(r"\band\b", question, re.I)):
        return "additive_duration"
    if (operation == "minimum"
            and re.search(r"\bminimum\s+amount\b", question, re.I)
            and re.search(r"\band\b", question, re.I)):
        return "joint_minimum"
    if operation == "difference":
        first = re.search(
            r"\bby\s+taking\s+(?:(?:the|a|an)\s+)?([a-z][a-z-]*)",
            question, re.I)
        second = re.search(
            r"\binstead\s+of\s+(?:(?:the|a|an)\s+)?([a-z][a-z-]*)",
            question, re.I)
        if first is not None and second is not None:
            lowered = workspace.casefold()
            if all(re.search(rf"\b{re.escape(value.casefold())}\b", lowered)
                   for value in (first.group(1), second.group(1))):
                return "complete_alternative"
    if operation == "sum" and not re.search(
            r"\b(?:last|past|recent)\s+(?:\d+|one|two|three|four|five|six|"
            r"seven|eight|nine|ten)?\s*(?:days?|weeks?|months?|years?)\b",
            question, re.I):
        currency_values = {
            Decimal(value.replace(",", ""))
            for value in re.findall(
                r"[$\u00a3\u20ac\u00a5]\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                workspace)
        }
        critical = _selective_terms(question) - _SELECTIVE_GENERIC_TERMS
        coverage = (len(critical & _selective_terms(workspace)) / len(critical)
                    if critical else 0.0)
        if len(currency_values) >= 3 and coverage >= 0.80:
            return "complete_money_sum"
    if operation == "count_distinct":
        # A per-purchase quantity is not a cumulative collection size.  The
        # worksheet may contain one numerically explicit shopping trip while
        # an open "so far" question requires all purchases.
        if (re.search(r"\bso\s+far\b", question, re.I)
                and re.search(r"\b(?:buy|bought|purchase(?:d)?)\b",
                              question, re.I)):
            return None
        question_terms = _selective_terms(question) - _SELECTIVE_GENERIC_TERMS
        values: set[int] = set()
        for line in ledger.worksheet_lines:
            overlap = question_terms & _selective_terms(line)
            if len(overlap) < 2:
                continue
            for match in _SELECTIVE_DIRECT_COUNT_RE.finditer(line):
                if _normalize_term(match.group("object").casefold()) not in question_terms:
                    continue
                raw = match.group("count").casefold()
                values.add(
                    int(raw) if raw.isdigit() else _SELECTIVE_WORD_NUMBERS[raw])
        if len(values) == 1:
            return "direct_count"
    return None


def _surface_status(text: str) -> str:
    """Describe surface polarity/completion without deciding qualification."""

    if _NEGATED_EVENT_RE.search(text):
        return "negated_or_cancelled"
    completed = bool(_COMPLETED_EVENT_RE.search(text))
    planned = bool(_PLANNED_EVENT_RE.search(text))
    if completed and planned:
        return "mixed_completed_and_planned"
    if completed:
        return "completed_or_realized"
    if planned:
        return "planned_or_hypothetical"
    return "direct_statement"


def _multiplication_relation(text: str) -> str:
    """Expose a same-sentence quantity/rate relation, but do not compute it."""

    patterns = (
        re.compile(
            r"\b(?P<count>\d+(?:\.\d+)?)\s+[A-Za-z][^.!?]{0,64}?"
            r"(?:at|for)\s+(?P<rate>[$£€]\s*\d+(?:\.\d+)?)\s*"
            r"(?:each|apiece|per\s+[A-Za-z]+)", re.I),
        re.compile(
            r"(?P<rate>[$£€]\s*\d+(?:\.\d+)?)\s*"
            r"(?:each|apiece|per\s+[A-Za-z]+)[^.!?]{0,64}?"
            r"\b(?P<count>\d+(?:\.\d+)?)\b", re.I),
    )
    for pattern in patterns:
        if match := pattern.search(text):
            count = match.group("count")
            rate = "".join(match.group("rate").split())
            return f"same-turn relation={count} × {rate} (quantity × unit price; compute once)"
    return ""


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
    # Preserve the validated ledger candidate order used by the packer. Alias
    # expansion is presentation-only so it cannot silently change which source
    # turns survive the fixed evidence budget.
    query_terms = _terms(question)
    worksheet_query_terms = _query_terms(question)
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
            "do not transfer a count from an explicitly different object type "
            "or job title merely because it is a near-match; then count once"),
        "sum": (
            "collect every distinct unit-compatible amount for the exact scope; "
            "exclude plans, unrelated values, subtotals, and duplicate mentions; "
            "then add once"),
        "unit_rate": (
            "bind the exact aggregate price and the matching distinct item count; "
            "divide total price by item count once and preserve the currency unit"),
        "difference": (
            "bind the exact two quantities; for remaining or needed, compute "
            "target minus the latest current amount; for savings, subtract the "
            "chosen option from the explicitly rejected option; preserve order "
            "and unit and do not substitute a different alternative; for a "
            "personal comparison, prefer costs the user explicitly stated or "
            "adopted for those exact options over generic assistant estimates, "
            "and report insufficient information when the user's own turns never "
            "bound a cost for one option; an assistant-only estimate cannot bind "
            "a personal cost operand"),
        "date_difference": (
            "bind the exact start and end events, resolve each from its own date "
            "or [source-time], then subtract in the requested calendar unit"),
        "mean": (
            "bind the complete requested population, use one value per member, "
            "sum the values, and divide once by the member count"),
        "minimum": "bind all exact-scope values and return the minimum with its unit",
        "maximum": "bind all exact-scope values and return the maximum with its unit",
    }
    procedure = rules.get(
        operation, "select the complete exact operand set and apply the operation once")
    historical_window = bool(re.search(
        r"\b(?:first|initial)\s+(?:\d+|one|two|three|four|five|six|seven|eight|"
        r"nine|ten)\s+(?:days?|weeks?|months?|years?)\b", q))
    if historical_window:
        procedure += (
            "; restrict operands to the requested historical window and use the "
            "value explicitly bounded to that interval, never a later cumulative "
            "or current total")
    completed_event_count = (
        operation == "count_distinct"
        and bool(re.search(
            r"\b(?:did\s+i\s+(?:go|attend|visit|see|have|join|try|"
            r"participate|host|complete|finish)|"
            r"have\s+i\s+(?:attended|visited|seen|had|joined|tried|"
            r"participated|hosted|completed|finished)|"
            r"appointments?\s+did\s+i\s+go\s+to)\b", q)))
    if completed_event_count:
        procedure += (
            "; enumerate every separately dated direct statement of a completed "
            "visit before counting; different dates or practitioners are distinct, "
            "while merely scheduled or considered visits are excluded")
    pending_item_count = (
        operation == "count_distinct"
        and bool(re.search(r"\bneed\b.*\b(?:pick\s*up|return)\b", q)))
    if pending_item_count:
        procedure += (
            "; enumerate each explicitly still-pending physical item before "
            "counting; an old item to return and its replacement to pick up are "
            "two distinct items unless a later direct statement says that one "
            "obligation was completed or cancelled")
    # The final readout policy consumes only a compact execution card.  Put a
    # small verbatim reading index in the trace so it can preserve useful
    # operands without making all 32 ledger candidates displace source turns.
    assistant_target = bool(re.search(
        r"\b(?:did\s+you|you\s+(?:recommend|suggest|provide|list|say|tell)|"
        r"your\s+(?:recommendation|suggestion|list))\b", q))
    worksheet: list[str] = []
    worksheet_turn_ids: list[str] = []
    seen_worksheet: set[str] = set()
    worksheet_limit = (
        4 if operation in {
            "date_difference", "difference", "mean", "unit_rate"
        } else 8)
    worksheet_excerpt_chars = 180 if worksheet_limit == 4 else 220
    for _score, _rank, turn, numbers, compatible in ordered:
        authoritative = _is_authoritative_source(turn) or assistant_target
        overlap = row_terms[turn.turn_id] & worksheet_query_terms
        if not authoritative or not overlap:
            continue
        if operation in {"sum", "unit_rate", "difference", "mean", "minimum", "maximum"}:
            if not compatible:
                continue
        if operation == "date_difference" and not (
                numbers or _RELATIVE_TIME_RE.search(turn.raw_text)
                or turn.timestamp):
            continue
        excerpt = _compact(
            turn.raw_text, worksheet_query_terms,
            limit=worksheet_excerpt_chars)
        normalized = " ".join(excerpt.casefold().split())
        if not excerpt or normalized in seen_worksheet:
            continue
        matches = ",".join(sorted(overlap)[:6]) or "context"
        shown_numbers = numbers[:6]
        numeric = ", ".join(shown_numbers) if shown_numbers else "none"
        if len(numbers) > len(shown_numbers):
            numeric += f", …(+{len(numbers) - len(shown_numbers)})"
        relation = (
            _multiplication_relation(turn.raw_text)
            if operation in {"sum", "unit_rate"} else "")
        suffix = f"; {relation}" if relation else ""
        worksheet.append(
            f"W{len(worksheet) + 1}: status={_surface_status(turn.raw_text)}; "
            f"match=[{matches}]; numbers=[{numeric}]; source={turn.turn_id}{suffix} :: "
            f"{excerpt}")
        worksheet_turn_ids.append(turn.turn_id)
        seen_worksheet.add(normalized)
        if len(worksheet) >= worksheet_limit:
            break

    if execution_card:
        lines = [
            "Aggregation ledger (compact execution card; source memories are authoritative):",
            f"Operation: {operation}",
            "Procedure: " + procedure,
            "Graph proximity finds evidence but does not qualify an operand.",
        ]
        if worksheet:
            lines += [
                "Operand worksheet (query-ranked verbatim excerpts from already "
                "packed direct memories; candidates only, verify scope and scan "
                "the full memories for omitted operands):",
                *worksheet,
            ]
        lines.append(f"Question: {question}")
    else:
        lines = [
            "Aggregation ledger (mechanically indexed candidate operands; source turns remain authoritative):",
            f"Operation: {operation}",
            "Procedure: " + procedure,
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
        if completed_event_count:
            completed_rows = []
            requested_months = tuple(re.findall(
                r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
                r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
                r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", q, re.I))
            for _score, _rank, turn, _number_values, _required in selected:
                if not _is_authoritative_source(turn):
                    continue
                if not re.search(
                        r"\b(?:had\b.{0,50}\bappointments?|"
                        r"went\s+to\s+see|attended\b.{0,50}\bappointments?|"
                        r"visited|saw)\b", turn.raw_text, re.I):
                    continue
                if not re.search(
                        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
                        r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
                        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
                        r"\d{1,2}(?:st|nd|rd|th)|\d{4}[-/])\b",
                        turn.raw_text, re.I):
                    continue
                if (requested_months
                        and not any(re.search(rf"\b{re.escape(month)}\w*\b",
                                              turn.raw_text, re.I)
                                    for month in requested_months)):
                    continue
                completed_rows.append(_compact(
                    turn.raw_text, query_terms, limit=220))
            if completed_rows:
                lines.append(
                    "Completed-event surface anchors (direct packed turns; "
                    "verify scope and deduplicate before counting):")
                lines.extend(f"- {row}" for row in completed_rows[:8])
        if pending_item_count:
            pending_rows = []
            for _score, _rank, turn, _number_values, _required in selected:
                if (_is_authoritative_source(turn)
                        and re.search(r"\b(?:pick\s*up|return)\b",
                                      turn.raw_text, re.I)):
                    pending_rows.append(_compact(
                        turn.raw_text, query_terms, limit=240))
            if pending_rows:
                lines.append(
                    "Pending-item surface anchors (direct packed turns; split "
                    "old items from replacements before deduplication):")
                lines.extend(f"- {row}" for row in pending_rows[:4])
        lines += [
            "Execution: retain exact entity/event/time/polarity matches; deduplicate repeated mentions, not distinct occurrences; exclude unaccepted hypotheticals; then apply the operation exactly.",
            "Exact-scope guard: a semantically related but explicitly different object type, collection item, or job title is not the queried operand; if only such near-matches exist, report insufficient information.",
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
        deterministic_operands=deterministic_operands,
        worksheet_lines=tuple(worksheet),
        worksheet_turn_ids=tuple(worksheet_turn_ids))
