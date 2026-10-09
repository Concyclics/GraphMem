"""Lossless source-turn physical plan and regression-safe graph fusion.

The graph is an excellent navigation index, but it is not allowed to be the
only route to an immutable source turn.  This module builds a second physical
ordering from source-facing exact/BM25/dense scores, lexical IDF, dialogue
adjacency and operator-critical surface features.  Graph reachability,
bindings and relation fan-out are deliberately removed from that ordering.

The resulting flat lane is fused with a stable prefix from the graph lane.
It never creates evidence, reads a benchmark label, gold answer or judge, and
never increases the declared turn budget.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
import math
import re
from typing import Iterable, Mapping, Sequence

from ..build.temporal import (
    extract_time_expression, extract_time_expressions, normalize_time,
    observed_interval,
)
from ..domain import CandidateScore, SourceTurn
from ..text import STOPWORDS, content_terms, predicate_family, terms
from .packer import rank_query_aware_candidates
from .slots import PLURAL_HINTS, parse_slots


FLAT_PLAN_VERSION = "graphmem-v5.70-lossless-flat-plan-v1"
SOURCE_FOCUS_VERSION = "graphmem-v5.80-multilabel-source-focus-v1"

_INFERENCE_RE = re.compile(
    r"\b(?:infer(?:red|ence)?|imply|suggests?|indicates?|likely|might|"
    r"personality|financial status|socioeconomic|prefer|benefit from|"
    r"what does .* say about)\b",
    re.I,
)
_MULTI_HOP_RE = re.compile(
    r"\b(?:based on|because of|lead(?:ing)? to|result(?:ed)? in|"
    r"relationship between|connected to|in common|both|respectively|"
    r"who .* (?:that|which)|what .* (?:that|which))\b",
    re.I,
)
_AGGREGATE_RE = re.compile(
    r"\b(?:how many|how much|how often|number of|total|combined|"
    r"average|minimum|maximum|difference|in all|altogether)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(?:when|what date|what time|how long|before|after|since|until|"
    r"first|last|latest|earliest|earlier|later|ago|elapsed|"
    r"days?|weeks?|months?|years?|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b",
    re.I,
)
_STATE_RE = re.compile(
    r"\b(?:current(?:ly)?|now|latest|still|used to|previously|"
    r"changed|replaced|moved|location|status)\b",
    re.I,
)
_LIST_RE = re.compile(
    r"\b(?:which|what are|who are|list|name all|all the|every)\b",
    re.I,
)
_DERIVATION_RE = re.compile(
    r"\b(?:would|could|might|likely|potentially|suspected|considered|"
    r"appropriate|underlying|based on|in light of|say about|describe|"
    r"benefit from|can help|good (?:career|hobby|gift)|prefer|who is|"
    r"how old|popular|major holiday|coincides?|conincides?)\b|"
    r"\bchallenges?\b.*\bhow\b",
    re.I,
)
_CAUSAL_RE = re.compile(
    r"\b(?:why|because|reason|cause[ds]?|due to|led to|result(?:ed)? in|"
    r"based on|in light of|motivat(?:e|es|ed|ion))\b",
    re.I,
)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(?:if|had not|hadn't|otherwise|counterfactual)\b",
    re.I,
)
_COMPARISON_RE = re.compile(
    r"\b(?:both|each|either|neither|respectively|same|in common|"
    r"more than|less than|compared|versus|\bor\b)\b",
    re.I,
)
_GEOGRAPHIC_RE = re.compile(
    # A geographic *resolution* is narrower than a location lookup.  Enable
    # ordinary-knowledge conversion only when the requested answer head names
    # an administrative/geographic level (for example, mapping Talkeetna to
    # Alaska).  ``Where ...?`` and ``Which places ...?`` already request the
    # source location itself; treating them as derivations made otherwise
    # direct answers wander to nearby or historically mentioned places.
    r"\b(?:which|what)\s+(?:[\w'-]+\s+){0,2}"
    r"(?:city|cities|country|countries|states?|provinces?|count(?:y|ies)|"
    r"regions?|continents?|hometowns?|birthplaces?)\b",
    re.I,
)
_ALIAS_RE = re.compile(
    r"\b(?:alias|called|name[ds]?|nicknames?)\b",
    re.I,
)
_ABSTAIN_RE = re.compile(
    r"\b(?:insufficient|not (?:stated|mentioned|provided|available)|"
    r"cannot determine|can't determine|unknown|no information)\b",
    re.I,
)
_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$£€¥]\s*)?-?\d+(?:,\d{3})*(?:\.\d+)?(?:%|st|nd|rd|th)?",
    re.I,
)
_TIME_SURFACE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:today|yesterday|tomorrow|last|next|ago|before|after|since|until)\b",
    re.I,
)
_STATUS_SURFACE_RE = re.compile(
    r"\b(?:currently|now|latest|still|no longer|replaced|moved|"
    r"cancelled|canceled|completed|finished|planned|planning|will|might)\b",
    re.I,
)
_GENERIC_ANSWER_TERMS = frozenset({
    "the", "and", "that", "this", "with", "from", "into", "about",
    "because", "answer", "information", "degree", "approximately",
})
_FOCUS_IRREGULAR_NOUNS = {
    "children": "child", "feet": "foot", "geese": "goose",
    "men": "man", "mice": "mouse", "people": "person",
    "teeth": "tooth", "women": "woman",
}
_FOCUS_S_EXCEPTIONS = frozenset({
    "class", "glass", "news", "series", "species", "status",
})
_FOCUS_EQUIVALENT_NOUNS = {
    # These are deliberately narrow lexical equivalences, not a general
    # ontology.  They cover ordinary surface alternations that frequently put
    # the relation word in one dialogue turn and its value in the next.
    "canine": "dog", "pup": "dog", "puppy": "dog",
    "feline": "cat", "kitten": "cat",
    "baby": "child", "daughter": "child", "kid": "child",
    "son": "child",
}
_FOCUS_RELATION_FAMILIES = {
    # QueryIR-level relation wording.  These families are used only in an
    # additive source view; the exact/morphological ranks remain untouched.
    "club": "team", "fan": "support",
    "advice": "recommend", "advise": "recommend",
    "suggest": "recommend", "suggestion": "recommend", "tip": "recommend",
    "career": "career", "job": "career", "occupation": "career",
    "profession": "career",
    "dating": "relationship", "married": "relationship",
    "marriage": "relationship", "partner": "relationship",
    "relationship": "relationship", "single": "relationship",
    "breakup": "relationship",
    "frustration": "problem", "frustrate": "problem",
    "difficulty": "problem", "issue": "problem", "struggle": "problem",
    "image": "photo", "photograph": "photo", "picture": "photo",
    "begin": "start", "restart": "start", "resume": "start",
    "favorite": "prefer", "favourite": "prefer", "fave": "prefer",
    "preference": "prefer", "enjoy": "prefer",
    "injure": "injury", "hurt": "injury", "sprain": "injury",
    "adopt": "acquire", "buy": "acquire", "get": "acquire",
    "obtain": "acquire", "receive": "acquire",
    "call": "alias", "nickname": "alias",
    "depart": "depart", "leave": "depart",
    "recommend": "recommend",
    "encourage": "support", "support": "support",
    "balance": "balance", "manage": "balance",
    "cope": "balance", "juggle": "balance", "overwhelm": "balance",
    "celebrate": "celebrate", "chill": "celebrate",
    "party": "celebrate", "recharge": "celebrate", "relax": "celebrate",
    "locate": "location", "location": "location", "live": "location",
    "stay": "location",
    "travel": "visit", "visit": "visit",
    "collectible": "collectible", "memorabilia": "collectible",
    "souvenir": "collectible",
}
_FOCUS_CONCEPT_RELATION_FAMILIES = {
    # Broader QueryIR equivalences live in an independent additive view.  They
    # may add a witness but cannot reorder the frozen direct+packet relation
    # prefix, which keeps coverage monotonic across policy upgrades.
    "recommendation": "recommend", "reccomend": "recommend",
    "challenge": "problem", "hard": "problem", "tough": "problem",
    "pick": "start", "take": "start", "try": "start",
    "activity": "activity", "hobby": "activity", "pastime": "activity",
    "outdoor": "activity", "hike": "activity", "hiking": "activity",
    "kayak": "activity", "kayaking": "activity", "surf": "activity",
    "surfing": "activity",
    "cook": "prepare", "experiment": "prepare", "make": "prepare",
    "recipe": "prepare",
    "himalaya": "himalaya", "himalayan": "himalaya",
    "tourney": "tournament",
}
_FOCUS_QUERY_SCAFFOLD = frozenset({
    "answer", "equivalent", "information", "many", "much", "question",
    "some", "source", "thing", "things", "wording",
})
_FOCUS_RELATION_HEADS = frozenset({
    "amount", "date", "kind", "name", "number", "status", "time", "type",
})
_FOCUS_NAME_VALUE_RE = re.compile(
    r"(?:\b(?:name(?:'s|\s+is|\s+was)?|named|called)\s+(?:is\s+)?"
    r"[A-Z][\w'-]+|\b(?:baby|child|daughter|kid|son)\s+"
    r"[A-Z][\w'-]+)"
)
_FOCUS_VOCATIVE_RE = re.compile(
    r"^\s*(?:hey|hi|hello)\s*[,!:-]?\s*([A-Z][\w'-]+)\b", re.I)
_FOCUS_DURATION_SURFACE_RE = re.compile(
    r"\b(?:for|over)\s+(?:\d+|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|several|few)\s+(?:days?|weeks?|months?|years?)\b|"
    r"\bsince\s+(?:19|20)\d{2}\b",
    re.I,
)
_FOCUS_QUERY_VIEW_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bnicknames?\b", re.I),
     ("called", "short name", "addressed", "greeting")),
    (re.compile(r"\brelationship status\b", re.I),
     ("single", "dating", "married", "partner", "breakup")),
    (re.compile(r"\b(?:football|sports?)\b.*\b(?:club|team|support)", re.I),
     ("fan", "supports", "team", "club")),
    (re.compile(r"\b(?:recommend|suggest|advis|tips?)\w*\b", re.I),
     ("recommended", "suggested", "advised", "tips")),
    (re.compile(r"\b(?:children|child|kids?|sons?|daughters?|bab(?:y|ies))\b", re.I),
     ("child", "son", "daughter", "baby", "family")),
    (re.compile(r"\b(?:frustrat|problem|difficult|struggl|annoy)\w*\b", re.I),
     ("problem", "difficulty", "struggle", "annoyed")),
    (re.compile(r"\b(?:photo|picture|image|shared?)\b", re.I),
     ("photo", "picture", "image", "media", "caption", "shared")),
    (re.compile(r"\b(?:start|begin|began|resume|new activity|take up)\w*\b", re.I),
     ("started", "began", "resumed", "tried", "took up")),
    (re.compile(r"\b(?:injur|hurt|sprain|twist)\w*\b", re.I),
     ("injury", "injured", "hurt", "sprained", "twisted")),
    (re.compile(r"\b(?:favorite|prefer|likes?|loves?)\b", re.I),
     ("favorite", "prefers", "likes", "loves", "fan")),
    (re.compile(r"\b(?:feel|feeling|mood|emotion)\w*\b", re.I),
     ("felt", "feeling", "mood", "emotion")),
    (re.compile(r"\b(?:where|city|state|country|located|location)\b", re.I),
     ("located", "stayed", "visited", "traveled", "moved", "lived")),
    (re.compile(r"\b(?:job|career|profession|occupation)\b", re.I),
     ("career", "job", "profession", "work", "occupation")),
)
_FOCUS_PRESENT_PERFECT_RE = re.compile(
    r"\bwhat\s+(?:has|have)\b", re.I)


def _looks_like_plural_content(token: str) -> bool:
    return (
        token in {"children", "feet", "men", "people", "teeth", "women"}
        or (len(token) > 3 and token.endswith("s")
            and token not in STOPWORDS
            and token not in _FOCUS_S_EXCEPTIONS
            and not token.endswith(("is", "ss", "us"))))


def _focus_terms(text: str) -> tuple[str, ...]:
    """Normalize shallow morphology only for the source-facing BM25 lane.

    Graph keys intentionally preserve exact surface forms.  Focus retrieval is
    a recall backstop, where ``dogs' names`` must overlap source text such as
    ``dog`` and ``her name is Shadow``.  Keeping this normalizer local prevents
    plural folding from changing relation, state, or provenance semantics.
    """

    normalized: list[str] = []
    for surface in terms(text):
        # ``terms`` already removes singular ``'s``; trim the remaining
        # possessive apostrophe so plural possessives such as ``dogs'`` share
        # the same routing form as ``dog`` and ``puppy``.
        token = surface.rstrip("'")
        root = (
            token if token in _FOCUS_S_EXCEPTIONS
            else _FOCUS_IRREGULAR_NOUNS.get(token, predicate_family(token)))
        # ``predicate_family`` intentionally maps function-only surfaces such
        # as ``are`` and ``the`` to the empty routing key.  Empty keys must not
        # enter BM25/document-frequency accounting: they otherwise create an
        # artificial high-frequency match shared by almost every turn.
        if not root:
            continue
        if root == token:
            if len(token) > 4 and token.endswith("ies"):
                root = token[:-3] + "y"
            elif (len(token) > 3 and token.endswith("s")
                  and token not in _FOCUS_S_EXCEPTIONS
                  and not token.endswith(("ss", "us", "is"))):
                root = token[:-1]
        normalized.append(_FOCUS_EQUIVALENT_NOUNS.get(root, root))
    return tuple(normalized)


def _focus_relation_terms(text: str) -> tuple[str, ...]:
    return tuple(
        _FOCUS_RELATION_FAMILIES.get(token, token)
        for token in _focus_terms(text))


def _focus_expanded_relation_terms(text: str) -> tuple[str, ...]:
    """Use broader equivalences only in independent additive query views."""

    values: list[str] = []
    for token in _focus_terms(text):
        base = _FOCUS_RELATION_FAMILIES.get(token, token)
        values.append(_FOCUS_CONCEPT_RELATION_FAMILIES.get(base, base))
    return tuple(values)


def _relation_concept_is_declarative(
    text: str, concept: str, *, expanded: bool = False,
) -> bool:
    """Return whether a mapped relation concept occurs in a statement clause.

    Conversational turns often answer first and append a follow-up question.
    Treating every turn containing ``?`` as interrogative suppresses the
    answer clause.  Clause-local punctuation keeps that distinction without
    interpreting or generating any fact.
    """

    tokenize = (
        _focus_expanded_relation_terms if expanded
        else _focus_relation_terms)
    for match in re.finditer(r"([^.!?]+)([.!?]|$)", text):
        clause, punctuation = match.group(1), match.group(2)
        if (concept in tokenize(clause)
                and punctuation != "?"):
            return True
    return False


def _preferred_evidence_speakers(
    question: str, explicit_speakers: Iterable[str],
) -> frozenset[str]:
    """Resolve a conservative speaker direction from common query syntax.

    ``recommendations Nate received from Joanna`` should rank Joanna's source
    turns above Nate's acknowledgements, while ``nickname does Nate use``
    should rank Nate's utterance.  Returning an empty set for ambiguous syntax
    keeps the existing symmetric owner score unchanged.
    """

    speakers = tuple(dict.fromkeys(str(value) for value in explicit_speakers))
    preferred: set[str] = set()
    from_match = re.search(
        r"\b(?:received?|got|heard|learned)\b[^?]{0,80}\bfrom\s+"
        r"([A-Z][\w'-]+)\b", question)
    if from_match:
        preferred.update(
            speaker for speaker in speakers
            if speaker.casefold() == from_match.group(1).casefold())
    actor_match = re.search(
        r"\b(?:did|does|has|have|is|was|would)\s+([A-Z][\w'-]+)\b",
        question)
    if actor_match and not preferred:
        preferred.update(
            speaker for speaker in speakers
            if speaker.casefold() == actor_match.group(1).casefold())
    return frozenset(preferred)


def _vocative_alias(text: str, full_names: Iterable[str]) -> bool:
    """Detect an abbreviated direct address without treating it as a fact.

    The surface form itself remains the only answer evidence.  This predicate
    merely lets an alias/nickname query retrieve ``Hey Jo, ...`` even though
    the turn does not literally contain the word ``nickname``.
    """

    match = _FOCUS_VOCATIVE_RE.search(text)
    if match is None:
        return False
    value = match.group(1).casefold()
    names = tuple(str(name).casefold() for name in full_names)
    return any(
        value != name and len(value) >= 2 and name.startswith(value)
        for name in names)


def compile_source_query_views(question: str) -> tuple[str, ...]:
    """Compile at most one bounded semantic paraphrase view from QueryIR cues.

    The expansion contains relation vocabulary only; named entities, dates and
    the original question remain unchanged.  Embedding all views in one batch
    gives dense retrieval a chance to match ``fan`` for ``supports`` or
    ``daughter`` for ``children`` without an answer-model call.
    """

    normalized = " ".join(str(question).split())
    additions: list[str] = []
    seen: set[str] = set()
    for pattern, values in _FOCUS_QUERY_VIEW_RULES:
        if not pattern.search(normalized):
            continue
        for value in values:
            if value not in seen:
                additions.append(value)
                seen.add(value)
    if not additions:
        return (normalized,)
    expansion = (
        normalized + "\nEquivalent source wording: "
        + ", ".join(additions[:24]) + ".")
    return (normalized, expansion)


@dataclass(frozen=True, slots=True)
class FlatPlan:
    route: str
    ordered_candidates: tuple[CandidateScore, ...]
    selected_turn_ids: tuple[str, ...]
    graph_head_turn_ids: tuple[str, ...]
    flat_head_turn_ids: tuple[str, ...]
    added_flat_turn_ids: tuple[str, ...]
    removed_graph_turn_ids: tuple[str, ...]
    trace: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VerificationGate:
    eligible: bool
    reasons: tuple[str, ...]
    answer_support: float
    flat_novelty: int
    route: str


@dataclass(frozen=True, slots=True)
class SourceFocusPlan:
    """Compact source-only rank used as an independent answer view."""

    route: str
    selected_turn_ids: tuple[str, ...]
    seed_turn_ids: tuple[str, ...]
    neighbor_turn_ids: tuple[str, ...]
    trace: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class QueryObligations:
    """Composable query duties compiled from question text only."""

    route: str
    tags: tuple[str, ...]
    requires_reasoning: bool
    requires_exhaustive: bool

    def presentation_tags(self) -> tuple[str, ...]:
        """Return non-duplicative duties for generic physical routes.

        Specialized routes already render a stronger operator contract.
        Repeating broad duties (set closure, generic inference, entity and
        polarity hints) increased prompt entropy without reliably changing
        the required answer operation.  Only comparison and administrative-
        level geographic resolution alter the answer structure sufficiently
        to justify a second visible contract.  Other labels remain available
        to retrieval and tracing without being injected into the answer.
        """

        if self.route not in {"lookup", "list"}:
            return ()
        allowed = frozenset({"comparison", "geographic_resolution"})
        implied = frozenset({"direct_lookup"})
        return tuple(
            tag for tag in self.tags
            if tag in allowed and tag not in implied)


_OBLIGATION_QUERY_TERMS: Mapping[str, tuple[str, ...]] = {
    "inference": (
        "direct premises", "traits", "preferences", "goals", "behavior"),
    "causal": ("stated reason", "cause", "motivation", "consequence"),
    "counterfactual": ("condition", "alternative outcome"),
    "multi_hop": ("connected facts", "bridge evidence"),
    "multi_entity": ("evidence for each person", "shared fact"),
    "comparison": ("same", "different", "more", "less"),
    "geographic_resolution": (
        "place", "city", "state", "country", "located", "visited"),
    "temporal": ("date", "started", "ended", "before", "after", "duration"),
    "exhaustive_set": ("every distinct item", "all occurrences"),
    "alias_resolution": ("nickname", "named", "called", "addressed as"),
    "negative_existence": (
        "planned", "considered", "not yet", "never", "no longer"),
}


def append_source_focus_witnesses(
    primary: SourceFocusPlan,
    auxiliary: SourceFocusPlan,
    *, excluded_turn_ids: Sequence[str] = (), max_extra_turns: int = 4,
    question: str = "", turns: Mapping[str, SourceTurn] | None = None,
) -> SourceFocusPlan:
    """Append a bounded independent-view packet without replacing evidence.

    Turns already supplied by the graph or primary focus lane are skipped,
    making the fusion monotonic in source-turn coverage and preventing
    redundant prompt bytes.  When source turns and the question are supplied,
    the auxiliary rank is refined using query-bearing dialogue context.
    """

    if max_extra_turns < 0:
        raise ValueError("max_extra_turns must be non-negative")
    occupied = set(map(str, excluded_turn_ids))
    occupied.update(primary.selected_turn_ids)
    priority = tuple(dict.fromkeys((
        *auxiliary.seed_turn_ids,
        *auxiliary.neighbor_turn_ids,
        *auxiliary.selected_turn_ids,
    )))
    direct_priority = priority
    dialogue_packet_radius = 1
    relation_concept_view = bool(
        auxiliary.trace.get("relation_concept_coverage"))
    if (turns is not None and auxiliary.trace.get("relation_families")
            and not relation_concept_view):
        # Relation/value pairs in dialogue are frequently separated by one
        # clarification turn: ``practice first`` -> ``easy to control?`` ->
        # ``use a gamepad and timing``.  The ordinary +/-1 closure cannot make
        # the value-bearing response a candidate.  Expand *only* the
        # auxiliary relation pool to +/-2 around its frozen seeds; the caller's
        # max_extra_turns still admits just one or two source turns, so this is
        # candidate generation rather than an evidence-budget expansion.
        by_coordinate = {
            (turn.session_id, turn.turn_index): turn.turn_id
            for turn in turns.values()
        }
        packet_rows = list(priority)
        packet_seen = set(packet_rows)
        for seed_id in auxiliary.seed_turn_ids:
            seed = turns.get(seed_id)
            if seed is None:
                continue
            for delta in (-2, 2):
                neighbor_id = by_coordinate.get(
                    (seed.session_id, seed.turn_index + delta))
                if neighbor_id is None or neighbor_id in packet_seen:
                    continue
                packet_rows.append(neighbor_id)
                packet_seen.add(neighbor_id)
        priority = tuple(packet_rows)
        dialogue_packet_radius = 2
    candidates = tuple(
        turn_id for turn_id in priority
        if turn_id not in occupied and (turns is None or turn_id in turns))
    direct_candidates = tuple(
        turn_id for turn_id in direct_priority
        if turn_id not in occupied and (turns is None or turn_id in turns))
    selection = "auxiliary-rank"
    candidate_scores: dict[str, float] = {}
    lane_by_id: dict[str, str] = {}
    if relation_concept_view:
        selection = "queryir-relation-concept-coverage"
        lane_by_id = {turn_id: "relation_concept" for turn_id in candidates}
    elif question and turns and candidates:
        relation_view = bool(auxiliary.trace.get("relation_families"))
        if relation_view and dialogue_packet_radius > 1:
            # Keep a direct-relation lane in addition to the new two-step
            # packet lane.  Without this reservation, a newly admitted
            # elliptical response can replace a strong direct match rather
            # than complement it.  A closure budget of three therefore means
            # two direct witnesses plus one packet witness.
            direct_ranked, direct_scores = _contextual_auxiliary_priority(
                question=question, turns=turns,
                candidates=direct_candidates,
                seed_turn_ids=auxiliary.seed_turn_ids,
                relation_families=True, context_radius=1)
            packet_ranked, packet_scores = _contextual_auxiliary_priority(
                question=question, turns=turns, candidates=candidates,
                seed_turn_ids=auxiliary.seed_turn_ids,
                relation_families=True, context_radius=2)
            direct_quota = min(
                len(direct_ranked),
                max_extra_turns if max_extra_turns <= 1
                else min(2, max_extra_turns - 1))
            selected_rows = list(direct_ranked[:direct_quota])
            for turn_id in packet_ranked:
                if len(selected_rows) >= max_extra_turns:
                    break
                if turn_id not in selected_rows:
                    selected_rows.append(turn_id)
            for turn_id in direct_ranked[direct_quota:]:
                if len(selected_rows) >= max_extra_turns:
                    break
                if turn_id not in selected_rows:
                    selected_rows.append(turn_id)
            candidates = tuple(selected_rows) + tuple(
                turn_id for turn_id in packet_ranked
                if turn_id not in selected_rows)
            candidate_scores = {**packet_scores, **direct_scores}
            lane_by_id = {
                turn_id: (
                    "direct" if turn_id in direct_ranked[:direct_quota]
                    else "dialogue_packet")
                for turn_id in selected_rows
            }
            selection = "direct-relation-plus-dialogue-packet"
        else:
            candidates, candidate_scores = _contextual_auxiliary_priority(
                question=question, turns=turns, candidates=candidates,
                seed_turn_ids=auxiliary.seed_turn_ids,
                relation_families=relation_view)
            selection = "query-concept-dialogue-context"
    extras = candidates[:max_extra_turns]
    view_type = (
        "temporal_transition"
        if (auxiliary.trace.get("view_kind") == "temporal"
            and auxiliary.trace.get("expanded_temporal_relation"))
        else "temporal" if auxiliary.trace.get("view_kind") == "temporal"
        else "relation_concept_expanded"
        if relation_concept_view
        and auxiliary.trace.get("expanded_relation_concepts")
        else "relation_concept" if relation_concept_view
        else "relation_lexical" if auxiliary.trace.get("relation_families")
        else "morphological" if auxiliary.trace.get("morphological")
        else "obligation_dense"
        if auxiliary.trace.get("view_kind") == "obligation_dense"
        else "dense" if auxiliary.trace.get("uses_dense_scores")
        else "source")
    view_record = {
        "type": view_type,
        "route": auxiliary.route,
        "selection": selection,
        "candidate_pool_turns": len(candidates),
        "dialogue_packet_radius": dialogue_packet_radius,
        "extra_turn_ids": list(extras),
        "extra_turns": len(extras),
        "selected_scores": {
            turn_id: round(candidate_scores[turn_id], 6)
            for turn_id in extras if turn_id in candidate_scores
        },
        "selected_lanes": {
            turn_id: lane_by_id[turn_id]
            for turn_id in extras if turn_id in lane_by_id
        },
    }
    prior_views = list(primary.trace.get("auxiliary_focus_views", ()))
    return replace(
        primary,
        selected_turn_ids=(*primary.selected_turn_ids, *extras),
        trace={
            **dict(primary.trace),
            "auxiliary_focus_view": True,
            "auxiliary_focus_route": auxiliary.route,
            "auxiliary_extra_turn_ids": list(extras),
            "auxiliary_extra_turns": len(extras),
            "auxiliary_focus_views": [*prior_views, view_record],
            "auxiliary_selection": selection,
            "auxiliary_candidate_scores": {
                turn_id: round(candidate_scores[turn_id], 6)
                for turn_id in extras if turn_id in candidate_scores
            },
        })


def _contextual_auxiliary_priority(
    *, question: str, turns: Mapping[str, SourceTurn],
    candidates: Sequence[str], seed_turn_ids: Sequence[str],
    relation_families: bool = False,
    context_radius: int | None = None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Rank auxiliary witnesses by concept-bearing dialogue context.

    A source answer is often elliptical: ``Her name is Shadow`` is useful only
    together with the preceding turn that says ``other dog``.  Conversely, a
    generic ``what is the baby's name?`` shares the relation word but not the
    requested concept.  Score each candidate against its immutable +/-1
    dialogue window, prefer declarative value-bearing turns, and then perform
    a small diversity-aware greedy rank.  This changes only which original
    turns consume the auxiliary budget; it never synthesizes answer text.
    """

    ordered_turns = tuple(sorted(turns.values(), key=lambda turn: (
        turn.session_id, turn.turn_index, turn.turn_id)))
    by_coordinate = {
        (turn.session_id, turn.turn_index): turn
        for turn in ordered_turns
    }
    tokenize = _focus_relation_terms if relation_families else _focus_terms
    tokens_by_id = {
        turn.turn_id: frozenset(tokenize(
            f"{turn.speaker} {turn.raw_text}"))
        for turn in ordered_turns
    }
    document_frequency: Counter[str] = Counter(
        token for values in tokens_by_id.values() for token in values)
    count = len(ordered_turns)
    query_terms = frozenset(
        token for token in tokenize(question)
        if (len(token) > 1 and token not in STOPWORDS
            and token not in _FOCUS_QUERY_SCAFFOLD))
    if not query_terms:
        return tuple(candidates), {}
    term_weight = {
        token: 1.0 + math.log(
            (count + 1.0) / (document_frequency.get(token, 0) + 1.0))
        for token in query_terms
    }
    explicit_speakers = frozenset(
        turn.speaker for turn in ordered_turns
        if content_terms(turn.speaker)
        and content_terms(turn.speaker) <= content_terms(question))
    owner_terms = frozenset(
        token for speaker in explicit_speakers for token in tokenize(speaker))
    distinctive_terms = (
        query_terms - owner_terms - _FOCUS_RELATION_HEADS)
    relation_heads = query_terms & _FOCUS_RELATION_HEADS
    seed_rank = {
        turn_id: index for index, turn_id in enumerate(seed_turn_ids)
    }
    base_scores: dict[str, float] = {}
    context_terms_by_id: dict[str, frozenset[str]] = {}
    direct_terms_by_id: dict[str, frozenset[str]] = {}
    distinctive_context_by_id: dict[str, frozenset[str]] = {}
    for fallback_rank, turn_id in enumerate(candidates):
        turn = turns[turn_id]
        direct = tokens_by_id[turn_id] & query_terms
        context = set(direct)
        radius = (
            context_radius if context_radius is not None
            else 2 if relation_families else 1)
        for delta in range(-radius, radius + 1):
            if delta == 0:
                continue
            neighbor = by_coordinate.get(
                (turn.session_id, turn.turn_index + delta))
            if neighbor is not None:
                context.update(tokens_by_id[neighbor.turn_id] & query_terms)
        direct_terms_by_id[turn_id] = frozenset(direct)
        context_terms_by_id[turn_id] = frozenset(context)
        distinctive_context = frozenset(context) & distinctive_terms
        distinctive_context_by_id[turn_id] = distinctive_context
        direct_score = sum(
            term_weight[token] for token in sorted(direct))
        context_score = sum(
            term_weight[token] for token in sorted(context - direct))
        question_penalty = 1.4 if "?" in turn.raw_text else 0.0
        answer_form = 0.8 if "?" not in turn.raw_text else 0.0
        owner_gain = 0.8 if turn.speaker in explicit_speakers else 0.0
        rank = seed_rank.get(turn_id, len(seed_rank) + fallback_rank)
        rank_gain = 1.0 / (1.0 + rank)
        base_scores[turn_id] = (
            2.2 * direct_score + 1.6 * context_score + answer_form
            + owner_gain + rank_gain - question_penalty
            + 2.4 * sum(term_weight[token]
                        for token in sorted(distinctive_context))
            - (3.0 if distinctive_terms and not distinctive_context else 0.0))

    remaining = list(candidates)
    name_value_bearing = (
        [turn_id for turn_id in remaining
         if _FOCUS_NAME_VALUE_RE.search(turns[turn_id].raw_text)]
        if relation_families and "name" in relation_heads else [])
    if name_value_bearing:
        # Name values may be expressed without repeating the entity class
        # (``our one-year-old ... his name is Kyle``), so this gate must run
        # before the generic concept-cooccurrence filter below.
        remaining = name_value_bearing
    elif relation_families and distinctive_terms:
        # The relation-family lane is a precision backstop, not a license to
        # append generic turns that merely contain an owner or an attribute
        # head such as ``name``.  Keep a candidate only when its immutable
        # +/-1 dialogue packet covers a query concept.  If no such candidate
        # exists, retain the frozen auxiliary order rather than forcing an
        # empty view for genuinely elliptical source text.
        concept_bearing = [
            turn_id for turn_id in remaining
            if distinctive_context_by_id[turn_id]
        ]
        if concept_bearing:
            remaining = concept_bearing
    ranked: list[str] = []
    covered: set[str] = set()
    session_counts: Counter[str] = Counter()
    final_scores: dict[str, float] = {}
    while remaining:
        def marginal(turn_id: str) -> tuple[float, int, int, str]:
            turn = turns[turn_id]
            new_terms = distinctive_context_by_id[turn_id] - covered
            novelty = sum(
                term_weight[token] for token in sorted(new_terms))
            diversity = 0.65 if not session_counts[turn.session_id] else 0.0
            repeat_penalty = 0.45 * session_counts[turn.session_id]
            value = base_scores[turn_id] + 1.4 * novelty + diversity - repeat_penalty
            # Direct concept coverage breaks ties ahead of contextual-only
            # matches, followed by the frozen auxiliary order.
            return (
                value,
                len(direct_terms_by_id[turn_id] & distinctive_terms),
                -candidates.index(turn_id), turn_id)

        chosen = max(remaining, key=marginal)
        score = marginal(chosen)[0]
        ranked.append(chosen)
        final_scores[chosen] = score
        covered.update(distinctive_context_by_id[chosen])
        session_counts[turns[chosen].session_id] += 1
        remaining.remove(chosen)
    return tuple(ranked), final_scores


def preserve_graph_with_flat_packet(
    plan: FlatPlan, graph_turn_ids: Sequence[str], *, max_extra_turns: int = 8,
) -> FlatPlan:
    """Keep every graph-packed turn and append a bounded flat witness packet.

    This mode is used when offline coverage auditing shows that replacement is
    unsafe.  Prompt bytes remain governed by a separate token cap, so preserving
    evidence does not imply an unbounded context.
    """

    if max_extra_turns < 0:
        raise ValueError("max_extra_turns must be non-negative")
    graph = tuple(dict.fromkeys(map(str, graph_turn_ids)))
    graph_set = frozenset(graph)
    extras = tuple(
        turn_id for turn_id in plan.added_flat_turn_ids
        if turn_id not in graph_set)[:max_extra_turns]
    selected = (*graph, *extras)
    trace = dict(plan.trace)
    trace.update({
        "fusion_mode": "preserve-graph-plus-flat-packet",
        "graph_turns_preserved": len(graph),
        "flat_packet_turns": len(extras),
        "selected": len(selected),
        "added_flat": len(extras),
        "removed_graph": 0,
    })
    return replace(
        plan, selected_turn_ids=selected, added_flat_turn_ids=extras,
        removed_graph_turn_ids=(), trace=trace)


def query_route(question: str) -> str:
    """Compile a query-only answer route.

    Benchmark category/stratum metadata is intentionally not accepted by the
    API.  A production request therefore follows the same path as an eval row.
    """

    slots = parse_slots(question)
    if _AGGREGATE_RE.search(question) or slots.is_count or slots.is_sum:
        return "aggregate"
    if (_TEMPORAL_RE.search(question) or slots.is_duration
            or slots.temporal_relation or slots.ordinal_index is not None):
        return "temporal"
    if _STATE_RE.search(question) or slots.is_latest:
        return "state"
    if _INFERENCE_RE.search(question):
        return "inference"
    if _MULTI_HOP_RE.search(question):
        return "multi_hop"
    # Keep the physical route stable for newly detected plural WH heads.  A
    # phrase such as ``What book recommendations ...?`` needs an exhaustive
    # *duty*, but changing the primary operator also changes retrieval and
    # layout.  Legacy plural hints/quantifiers retain their established list
    # route; new answer-head coverage is carried by QueryObligations.
    legacy_multiple = (
        bool(set(slots.tokens) & PLURAL_HINTS)
        or slots.quantifier in {"all", "every", "both", "each"}
        or slots.is_unit_rate)
    if _LIST_RE.search(question) or legacy_multiple:
        return "list"
    return "lookup"


def compile_query_obligations(question: str) -> QueryObligations:
    """Compile independent answer duties without collapsing them to one route.

    ``query_route`` remains the physical primary operator for compatibility.
    These tags preserve orthogonal duties such as temporal comparison plus
    multi-entity closure, or a lookup whose value requires a geographic or
    commonsense derivation.  No benchmark category, answer or memory content
    is accepted by this compiler.
    """

    route = query_route(question)
    slots = parse_slots(question)
    tags: list[str] = []

    def add(tag: str, condition: bool = True) -> None:
        if condition and tag not in tags:
            tags.append(tag)

    add("aggregate", route == "aggregate" or slots.is_count or slots.is_sum)
    add("temporal", route == "temporal" or bool(slots.temporal_relation)
        or slots.is_duration or slots.ordinal_index is not None)
    add("latest_state", route == "state" or slots.is_latest)
    add("exhaustive_set", (
        route == "aggregate" or slots.expects_multiple
        or slots.quantifier in {"all", "both", "each", "every", "respectively"}
        or bool(re.search(
            r"\b(?:what are|who are|list|name all|all the|every)\b",
            question, re.I))))
    add("multi_entity", (
        slots.quantifier in {"both", "each", "either", "neither", "respectively"}
        or bool(re.search(
            r"\b[A-Z][\w'-]+\s+and\s+[A-Z][\w'-]+\b", question))))
    add("comparison", bool(_COMPARISON_RE.search(question)))
    add("causal", slots.answer_slot == "reason"
        or bool(_CAUSAL_RE.search(question)))
    add("counterfactual", bool(_COUNTERFACTUAL_RE.search(question)))
    add("inference", route == "inference"
        or bool(_DERIVATION_RE.search(question)))
    add("multi_hop", route == "multi_hop" or bool(_MULTI_HOP_RE.search(question))
        or "causal" in tags or "multi_entity" in tags)
    add("geographic_resolution", bool(_GEOGRAPHIC_RE.search(question)))
    add("alias_resolution", bool(_ALIAS_RE.search(question)))
    add("negative_existence", slots.negation or slots.is_existence)
    requires_reasoning = any(tag in tags for tag in (
        "inference", "causal", "counterfactual", "comparison", "multi_hop",
        "geographic_resolution", "negative_existence"))
    add("reasoning_chain", requires_reasoning)
    if not tags:
        add("direct_lookup")
    return QueryObligations(
        route=route, tags=tuple(tags),
        requires_reasoning=requires_reasoning,
        requires_exhaustive="exhaustive_set" in tags,
    )


def compile_obligation_query_view(question: str) -> tuple[str, ...]:
    """Create one bounded dense-retrieval view from composable obligations."""

    normalized = " ".join(str(question).split())
    obligations = compile_query_obligations(normalized)
    additions: list[str] = []
    seen: set[str] = set()
    for tag in obligations.tags:
        for value in _OBLIGATION_QUERY_TERMS.get(tag, ()):
            if value not in seen:
                additions.append(value)
                seen.add(value)
    if not additions:
        return (normalized,)
    return (
        normalized,
        normalized + "\nEvidence wording to retrieve: "
        + ", ".join(additions[:24]) + ".",
    )


def build_source_focus_plan(
    *, question: str, turns: Mapping[str, SourceTurn], max_turns: int = 32,
    seed_fraction: float = 2.0 / 3.0,
    dense_scores: Mapping[str, float] | None = None,
    morphological: bool = False,
    relation_families: bool = False,
    session_diversity: bool = False,
    relation_concept_coverage: bool = False,
    expanded_relation_concepts: bool = False,
    view_kind: str = "source",
) -> SourceFocusPlan:
    """Rank immutable turns with BM25/phrase signals and dialogue closure.

    This physical view is independent of graph reachability, extracted facts
    and earlier answers.  It complements rather than replaces the graph view,
    preventing a structurally popular but query-irrelevant neighborhood from
    hiding a direct source witness.
    """

    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if not 0.0 < seed_fraction <= 1.0:
        raise ValueError("seed_fraction must be in (0, 1]")
    obligations = compile_query_obligations(question)
    route = obligations.route
    base_relation_query = _focus_relation_terms(question)
    expanded_relation_query = _focus_expanded_relation_terms(question)
    expanded_relation_query_triggered = bool(
        expanded_relation_concepts
        and (base_relation_query != expanded_relation_query
             or "alias" in base_relation_query))
    ordered_turns = tuple(sorted(turns.values(), key=lambda turn: (
        turn.session_id, turn.turn_index, turn.turn_id)))
    by_id = {turn.turn_id: turn for turn in ordered_turns}
    if not ordered_turns:
        return SourceFocusPlan(
            route=route, selected_turn_ids=(), seed_turn_ids=(),
            neighbor_turn_ids=(), trace={
                "version": SOURCE_FOCUS_VERSION, "route": route,
                "query_obligations": list(obligations.tags),
                "requires_reasoning": obligations.requires_reasoning,
                "requires_exhaustive": obligations.requires_exhaustive,
                "view_kind": view_kind,
                "max_turns": max_turns, "candidates": 0, "selected": 0,
            })

    tokenize = (
        _focus_expanded_relation_terms
        if relation_families and expanded_relation_concepts
        else _focus_relation_terms if relation_families
        else _focus_terms if morphological else terms)
    query_tokens = tuple(
        token for token in tokenize(question) if len(token) > 1)
    query_terms = frozenset(query_tokens)
    query_bigrams = frozenset(zip(query_tokens, query_tokens[1:]))
    tokens_by_id = {
        turn.turn_id: tuple(tokenize(
            f"{turn.speaker} {turn.timestamp or ''} {turn.raw_text}"))
        for turn in ordered_turns
    }
    document_frequency: Counter[str] = Counter(
        token for values in tokens_by_id.values() for token in set(values))
    average_length = sum(map(len, tokens_by_id.values())) / len(ordered_turns)
    count = len(ordered_turns)
    explicit_speakers = frozenset(
        turn.speaker for turn in ordered_turns
        if content_terms(turn.speaker)
        and content_terms(turn.speaker) <= content_terms(question))
    closure_reasons: list[str] = []
    preferred_speakers = _preferred_evidence_speakers(
        question, explicit_speakers)
    query_name_surfaces = tuple(dict.fromkeys((
        *explicit_speakers,
        *(match.group(0) for match in re.finditer(
            r"\b[A-Z][\w'-]+\b", question)
          if match.group(0).casefold()
          not in {"what", "when", "where", "which", "who"}),
    )))
    if route in {"aggregate", "inference", "list", "multi_hop"}:
        closure_reasons.append(f"route:{route}")
    slots = parse_slots(question)
    if slots.is_existence:
        closure_reasons.append("existence")
    if any(_looks_like_plural_content(token) for token in terms(question)):
        closure_reasons.append("plural-slot")
    if _FOCUS_PRESENT_PERFECT_RE.search(question):
        closure_reasons.append("present-perfect")
    if len(explicit_speakers) >= 2:
        closure_reasons.append("multi-subject")
    owner_terms = frozenset(
        token for speaker in explicit_speakers for token in tokenize(speaker))
    relation_concepts = frozenset(
        token for token in query_terms
        if (token not in owner_terms and token not in STOPWORDS
            and token not in _FOCUS_QUERY_SCAFFOLD
            and token not in _FOCUS_RELATION_HEADS))
    relation_heads = query_terms & _FOCUS_RELATION_HEADS
    by_coordinate = {
        (turn.session_id, turn.turn_index): turn
        for turn in ordered_turns
    }

    scored: list[tuple[float, str]] = []
    lexical_by_id: dict[str, float] = {}
    relation_concept_by_id: dict[str, float] = {}
    direct_relation_concepts_by_id: dict[str, frozenset[str]] = {}
    for turn in ordered_turns:
        values = tokens_by_id[turn.turn_id]
        frequencies = Counter(values)
        score = 0.0
        for token in sorted(query_terms):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            inverse = math.log(
                1.0 + (count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(values) / max(1.0, average_length))
            score += inverse * frequency * 2.2 / denominator
        score += 1.5 * len(
            query_bigrams & frozenset(zip(values, values[1:])))
        if turn.speaker in explicit_speakers:
            score += 1.0
        relation_concept_score = 0.0
        direct_concepts: set[str] = set()
        if relation_families and relation_concepts:
            direct_concepts = set(values) & relation_concepts
            alias_value = bool(
                relation_concept_coverage and expanded_relation_concepts
                and "alias" in relation_concepts
                and (not preferred_speakers
                     or turn.speaker in preferred_speakers)
                and _vocative_alias(turn.raw_text, query_name_surfaces))
            if alias_value:
                direct_concepts.add("alias")
            context_concepts: set[str] = set()
            for delta in (-1, 1):
                neighbor = by_coordinate.get(
                    (turn.session_id, turn.turn_index + delta))
                if neighbor is not None:
                    context_concepts.update(
                        set(tokens_by_id[neighbor.turn_id])
                        & relation_concepts)
            context_concepts.difference_update(direct_concepts)

            def concept_weight(token: str) -> float:
                return 1.0 + math.log(
                    (count + 1.0)
                    / (document_frequency.get(token, 0) + 1.0))

            relation_concept_score = (
                3.2 * sum(
                    concept_weight(token)
                    for token in sorted(direct_concepts))
                + 1.4 * sum(
                    concept_weight(token)
                    for token in sorted(context_concepts)))
            if direct_concepts and turn.speaker in explicit_speakers:
                relation_concept_score += 2.0
            if (relation_concept_coverage and expanded_relation_concepts
                    and direct_concepts
                    and turn.speaker in preferred_speakers):
                relation_concept_score += 2.5
            if (direct_concepts and (
                    (relation_concept_coverage and any(
                        _relation_concept_is_declarative(
                            turn.raw_text, token,
                            expanded=expanded_relation_concepts)
                        for token in direct_concepts))
                    or (not relation_concept_coverage
                        and "?" not in turn.raw_text))):
                relation_concept_score += 0.8
            if (direct_concepts and relation_heads
                    and set(values) & relation_heads):
                relation_concept_score += 1.5
            # QueryIR's ``name`` slot admits common declarative value forms
            # (``his name is Kyle`` and ``daughter Sara``).  This is only a
            # rank feature over exact source text; it never extracts or
            # fabricates the value used by the answer model.
            if ("name" in relation_heads
                    and _FOCUS_NAME_VALUE_RE.search(turn.raw_text)):
                relation_concept_score += 5.0
            if alias_value:
                relation_concept_score += 5.0
            score += relation_concept_score
        direct_relation_concepts_by_id[turn.turn_id] = frozenset(
            direct_concepts)
        relation_concept_by_id[turn.turn_id] = relation_concept_score
        lexical_by_id[turn.turn_id] = score
        scored.append((score, turn.turn_id))
    scored.sort(key=lambda item: (-round(item[0], 12), item[1]))
    session_relation_concepts: dict[str, set[str]] = {}
    if relation_families:
        for turn_id, concepts in direct_relation_concepts_by_id.items():
            session_relation_concepts.setdefault(
                by_id[turn_id].session_id, set()).update(concepts)

    seed_limit = max(8, min(max_turns, round(max_turns * seed_fraction)))
    relation_concept_lane: tuple[str, ...] = ()
    if dense_scores:
        # Preserve independent lexical and semantic lanes before using their
        # fused score.  A single weighted sort tended to fill the focus budget
        # with paraphrases from one popular session and missed terse bridge
        # turns needed by multi-hop questions.
        lexical_order = tuple(
            turn_id for score, turn_id in scored if score > 0.0)
        dense_order = tuple(sorted(
            (turn.turn_id for turn in ordered_turns
             if float(dense_scores.get(turn.turn_id, 0.0)) > 0.0),
            key=lambda turn_id: (
                -float(dense_scores.get(turn_id, 0.0)), turn_id)))
        dense_weight = 2.5 if route in {"inference", "multi_hop"} else 1.8
        hybrid_order = tuple(sorted(
            (turn.turn_id for turn in ordered_turns),
            key=lambda turn_id: (
                -(lexical_by_id.get(turn_id, 0.0)
                  + dense_weight * max(
                      0.0, float(dense_scores.get(turn_id, 0.0)))),
                turn_id)))
        lane_limits = (
            max(1, round(seed_limit * 0.40)),
            max(1, round(seed_limit * 0.35)),
        )
        lane_limits += (max(0, seed_limit - sum(lane_limits)),)
        seed_rows: list[str] = []
        seed_set: set[str] = set()
        session_counts: Counter[str] = Counter()

        def admit_lane(values: Sequence[str], quota: int) -> None:
            admitted = 0
            # First pass promotes session diversity; the second fills unused
            # seats without imposing a hard session cap.
            for cap in (3, 1 << 30):
                for turn_id in values:
                    if admitted >= quota or len(seed_rows) >= seed_limit:
                        return
                    if turn_id in seed_set:
                        continue
                    session_id = by_id[turn_id].session_id
                    if session_counts[session_id] >= cap:
                        continue
                    seed_rows.append(turn_id)
                    seed_set.add(turn_id)
                    session_counts[session_id] += 1
                    admitted += 1

        for values, quota in zip(
                (lexical_order, dense_order, hybrid_order), lane_limits):
            admit_lane(values, quota)
        if len(seed_rows) < seed_limit:
            admit_lane(hybrid_order, seed_limit - len(seed_rows))
        seeds = tuple(seed_rows)
    elif relation_families and any(relation_concept_by_id.values()):
        # Reserve half of the relation-view seeds for concept-bearing packets
        # before filling from ordinary BM25.  A small per-session first pass
        # prevents a long discussion of one generic topic from hiding a terse
        # value in another session (important for exhaustive list questions).
        concept_order = tuple(sorted(
            (turn_id for turn_id, value in relation_concept_by_id.items()
             if value > 0.0),
            key=lambda turn_id: (
                -relation_concept_by_id[turn_id],
                -lexical_by_id[turn_id], turn_id)))
        lexical_order = tuple(
            turn_id for score, turn_id in scored if score > 0.0)
        concept_quota = min(
            seed_limit, max(4, round(seed_limit * 0.5)))
        seed_rows: list[str] = []
        seed_set: set[str] = set()
        session_counts: Counter[str] = Counter()

        def admit_relation(values: Sequence[str], quota: int) -> None:
            admitted = 0
            for cap in (2, 1 << 30):
                for turn_id in values:
                    if admitted >= quota or len(seed_rows) >= seed_limit:
                        return
                    if turn_id in seed_set:
                        continue
                    session_id = by_id[turn_id].session_id
                    if session_counts[session_id] >= cap:
                        continue
                    seed_rows.append(turn_id)
                    seed_set.add(turn_id)
                    session_counts[session_id] += 1
                    admitted += 1

        if relation_concept_coverage:
            # Total-score sorting over-represents repeated topic nouns.  For
            # ``How did Nate celebrate winning the international tournament?``,
            # dozens of ``win/tournament`` turns can otherwise hide the one
            # terse celebration response.  This optional third lane reserves
            # one best direct witness per QueryIR relation concept.  Callers
            # append it after the ordinary relation lane, so it cannot replace
            # direct or dialogue-packet evidence.
            concept_heads = sorted(
                relation_concepts,
                key=lambda token: (
                    document_frequency.get(token, 0), token))
            session_concept_positions: dict[tuple[str, str], list[int]] = {}
            for turn_id, concepts in direct_relation_concepts_by_id.items():
                turn = by_id[turn_id]
                for value in concepts:
                    session_concept_positions.setdefault(
                        (turn.session_id, value), []).append(turn.turn_index)
            concept_lane: list[str] = []
            for concept in concept_heads:
                options = [
                    turn_id for turn_id in concept_order
                    if concept in direct_relation_concepts_by_id[turn_id]
                    and turn_id not in concept_lane
                ]
                if not options:
                    continue

                def session_join_distance(turn_id: str) -> int:
                    turn = by_id[turn_id]
                    other_concepts = relation_concepts - {concept}
                    distances = [
                        abs(position - turn.turn_index)
                        for other_concept in other_concepts
                        for position in session_concept_positions.get(
                            (turn.session_id, other_concept), ())
                    ]
                    return min(distances, default=1 << 20)

                concept_lane.append(max(options, key=lambda turn_id: (
                    len(session_relation_concepts.get(
                        by_id[turn_id].session_id, set()) & relation_concepts),
                    _relation_concept_is_declarative(
                        by_id[turn_id].raw_text, concept,
                        expanded=expanded_relation_concepts),
                    by_id[turn_id].speaker in explicit_speakers,
                    -session_join_distance(turn_id),
                    relation_concept_by_id[turn_id],
                    lexical_by_id[turn_id], turn_id)))
                if len(concept_lane) >= concept_quota:
                    break
            relation_concept_lane = tuple(concept_lane)
            admit_relation(relation_concept_lane, len(relation_concept_lane))
            admit_relation(concept_order, concept_quota - len(seed_rows))
        else:
            admit_relation(concept_order, concept_quota)
        admit_relation(lexical_order, seed_limit - len(seed_rows))
        seeds = tuple(seed_rows)
    elif session_diversity and closure_reasons:
        lexical_order = tuple(
            turn_id for score, turn_id in scored if score > 0.0)
        seed_rows: list[str] = []
        seed_set: set[str] = set()
        session_counts: Counter[str] = Counter()
        for cap in (2, 1 << 30):
            for turn_id in lexical_order:
                if len(seed_rows) >= seed_limit:
                    break
                if turn_id in seed_set:
                    continue
                session_id = by_id[turn_id].session_id
                if session_counts[session_id] >= cap:
                    continue
                seed_rows.append(turn_id)
                seed_set.add(turn_id)
                session_counts[session_id] += 1
            if len(seed_rows) >= seed_limit:
                break
        seeds = tuple(seed_rows)
    else:
        seeds = tuple(
            turn_id for score, turn_id in scored[:seed_limit] if score > 0.0)
    selected = list(seeds)
    selected_set = set(selected)
    by_coordinate = {
        (turn.session_id, turn.turn_index): turn.turn_id
        for turn in ordered_turns
    }
    neighbors: list[str] = []
    for turn_id in seeds:
        turn = by_id[turn_id]
        for delta in (-1, 1):
            neighbor = by_coordinate.get(
                (turn.session_id, turn.turn_index + delta))
            if neighbor is None or neighbor in selected_set:
                continue
            selected.append(neighbor)
            selected_set.add(neighbor)
            neighbors.append(neighbor)
            if len(selected) >= max_turns:
                break
        if len(selected) >= max_turns:
            break
    if len(selected) < max_turns:
        for _score, turn_id in scored:
            if turn_id in selected_set:
                continue
            selected.append(turn_id)
            selected_set.add(turn_id)
            if len(selected) >= max_turns:
                break
    selected = sorted(selected[:max_turns], key=lambda turn_id: (
        by_id[turn_id].session_id, by_id[turn_id].turn_index, turn_id))
    return SourceFocusPlan(
        route=route, selected_turn_ids=tuple(selected),
        seed_turn_ids=seeds, neighbor_turn_ids=tuple(neighbors),
        trace={
            "version": SOURCE_FOCUS_VERSION, "route": route,
            "query_obligations": list(obligations.tags),
            "requires_reasoning": obligations.requires_reasoning,
            "requires_exhaustive": obligations.requires_exhaustive,
            "view_kind": view_kind,
            "max_turns": max_turns, "candidates": count,
            "lexical_seeds": len(seeds),
            "dialogue_neighbors": len(neighbors),
            "selected": len(selected), "uses_graph_scores": False,
            "uses_dense_scores": bool(dense_scores),
            "morphological": morphological,
            "relation_families": relation_families,
            "session_diversity": bool(
                session_diversity and closure_reasons),
            "source_closure_risk": bool(closure_reasons),
            "source_closure_reasons": closure_reasons,
            "relation_concept_terms": sorted(relation_concepts),
            "relation_concept_seeds": sum(
                relation_concept_by_id.get(turn_id, 0.0) > 0.0
                for turn_id in seeds),
            "relation_concepts_covered_by_seeds": sorted(set().union(*(
                direct_relation_concepts_by_id.get(turn_id, frozenset())
                for turn_id in seeds))),
            "relation_concept_lane_turn_ids": list(relation_concept_lane),
            "relation_session_conjunction": bool(
                relation_families and relation_concept_coverage),
            "relation_concept_coverage": bool(
                relation_families and relation_concept_coverage),
            "expanded_relation_concepts": bool(
                relation_families and expanded_relation_concepts),
            "expanded_relation_query_triggered": (
                expanded_relation_query_triggered),
            "focus_rank_lanes": (
                ["lexical", "dense", "hybrid"] if dense_scores
                else ["lexical"]),
            "uses_answer_or_label": False,
        },
    )


def _interval_gap_days(
    left_start: str, left_end: str | None,
    right_start: str, right_end: str | None,
) -> float:
    """Return zero for overlap, otherwise the boundary distance in days."""

    left_a = datetime.fromisoformat(left_start)
    left_b = datetime.fromisoformat(left_end or left_start)
    right_a = datetime.fromisoformat(right_start)
    right_b = datetime.fromisoformat(right_end or right_start)
    if left_a <= right_b and right_a <= left_b:
        return 0.0
    delta = right_a - left_b if left_b < right_a else left_a - right_b
    return max(0.0, delta.total_seconds() / 86_400.0)


def build_temporal_focus_plan(
    *, question: str, turns: Mapping[str, SourceTurn], max_turns: int = 16,
    dense_scores: Mapping[str, float] | None = None,
    expanded_relation_concepts: bool = False,
) -> SourceFocusPlan:
    """Build an additive source view around an explicit query-time interval.

    The graph already stores temporal nodes, but direct source retrieval must
    also understand that a turn observed on October 4 saying ``yesterday`` is
    evidence for October 3.  This view ranks immutable turns by normalized
    event-time proximity, lexical binding and speaker ownership.  It is empty
    when the question has no resolvable absolute time, so it cannot consume a
    generic auxiliary budget.
    """

    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    obligations = compile_query_obligations(question)
    route = obligations.route
    query_phrases = extract_time_expressions(question)
    target_rows = tuple(
        (phrase, normalize_time(phrase, None, "query"))
        for phrase in query_phrases)
    # A compound query such as ``the last two weeks of August 2023`` may
    # expose an unanchored relative phrase before its absolute month.  Select
    # the first independently resolvable query interval rather than treating
    # that unresolved prefix as the whole temporal constraint.
    resolved_target = next(
        ((phrase, interval) for phrase, interval in target_rows
         if interval.start), None)
    phrase = resolved_target[0] if resolved_target else (
        query_phrases[0] if query_phrases else None)
    target = resolved_target[1] if resolved_target else None
    slots = parse_slots(question)
    temporal_fallback_reasons = tuple(filter(None, (
        (f"relation:{slots.temporal_relation}"
         if slots.temporal_relation else ""),
        (f"ordinal:{slots.ordinal_index}"
         if slots.ordinal_index is not None else ""),
        ("duration" if slots.is_duration else ""),
    )))
    ordered_turns = tuple(sorted(turns.values(), key=lambda turn: (
        turn.session_id, turn.turn_index, turn.turn_id)))
    base_temporal_query = _focus_relation_terms(question)
    expanded_temporal_query = _focus_expanded_relation_terms(question)
    expanded_temporal_query_triggered = bool(
        expanded_relation_concepts
        and (((target is None or not target.start)
              and temporal_fallback_reasons)
             or (target is not None and target.start
                 and base_temporal_query != expanded_temporal_query)))
    temporal_tokenize = (
        _focus_expanded_relation_terms if expanded_relation_concepts
        else _focus_relation_terms)
    empty_trace = {
            "version": SOURCE_FOCUS_VERSION, "route": route,
            "query_obligations": list(obligations.tags),
            "requires_reasoning": obligations.requires_reasoning,
            "requires_exhaustive": obligations.requires_exhaustive,
        "view_kind": "temporal", "max_turns": max_turns,
        "candidates": len(ordered_turns), "selected": 0,
        "query_time_phrase": phrase, "query_time_resolved": False,
        "expanded_temporal_relation": expanded_relation_concepts,
        "expanded_temporal_query_triggered": (
            expanded_temporal_query_triggered),
        "temporal_fallback_gate_reasons": list(
            temporal_fallback_reasons),
        "uses_answer_or_label": False,
    }
    if not ordered_turns:
        return SourceFocusPlan(
            route=route, selected_turn_ids=(), seed_turn_ids=(),
            neighbor_turn_ids=(), trace=empty_trace)
    if target is None or not target.start:
        # ``when`` and event-transition questions often contain no absolute
        # query date.  Their answer can still be recovered from a dated source
        # turn (``I left for Canada``), a duration (``had them for 3 years``),
        # or the turn immediately after/before a query event.  Rank those
        # immutable turns directly instead of disabling the temporal lane.
        if (route != "temporal" or not expanded_relation_concepts
                or not expanded_temporal_query_triggered):
            return SourceFocusPlan(
                route=route, selected_turn_ids=(), seed_turn_ids=(),
                neighbor_turn_ids=(), trace=empty_trace)
        query_terms = frozenset(
            token for token in temporal_tokenize(question)
            if (len(token) > 1 and token not in STOPWORDS
                and token not in _FOCUS_QUERY_SCAFFOLD))
        tokens_by_id = {
            turn.turn_id: frozenset(temporal_tokenize(
                f"{turn.speaker} {turn.raw_text}"))
            for turn in ordered_turns
        }
        document_frequency: Counter[str] = Counter(
            token for values in tokens_by_id.values() for token in values)
        count = len(ordered_turns)
        explicit_speakers = frozenset(
            turn.speaker for turn in ordered_turns
            if content_terms(turn.speaker)
            and content_terms(turn.speaker) <= content_terms(question))
        owner_terms = frozenset(
            token for speaker in explicit_speakers
            for token in temporal_tokenize(speaker))
        anchor_terms = (
            query_terms - owner_terms - _FOCUS_RELATION_HEADS - {
                "after", "before", "date", "earlier", "earliest", "first",
                "last", "later", "latest", "time", "when",
            })
        by_coordinate = {
            (turn.session_id, turn.turn_index): turn
            for turn in ordered_turns
        }

        def weight(token: str) -> float:
            return 1.0 + math.log(
                (count + 1.0)
                / (document_frequency.get(token, 0) + 1.0))

        scored_fallback: list[tuple[float, str]] = []
        duration_candidates = 0
        transition_candidates = 0
        for turn in ordered_turns:
            direct = tokens_by_id[turn.turn_id] & anchor_terms
            preceding: set[str] = set()
            following: set[str] = set()
            for distance in (1, 2, 3):
                previous = by_coordinate.get(
                    (turn.session_id, turn.turn_index - distance))
                following_turn = by_coordinate.get(
                    (turn.session_id, turn.turn_index + distance))
                if previous is not None:
                    preceding.update(
                        tokens_by_id[previous.turn_id] & anchor_terms)
                if following_turn is not None:
                    following.update(
                        tokens_by_id[following_turn.turn_id] & anchor_terms)
            context = preceding | following
            duration = bool(_FOCUS_DURATION_SURFACE_RE.search(turn.raw_text))
            if duration:
                duration_candidates += 1
            directed = (
                preceding if slots.temporal_relation == "after"
                else following if slots.temporal_relation == "before"
                else set())
            if directed:
                transition_candidates += 1
            # Avoid admitting every dated dialogue turn: it must bind a query
            # concept directly/locally, or expose a duration needed by a
            # ``when`` query from the requested speaker.
            owner_duration = duration and turn.speaker in explicit_speakers
            if not direct and not context and not owner_duration:
                continue
            direct_score = 2.4 * sum(
                weight(token) for token in sorted(direct))
            context_score = 1.5 * sum(
                weight(token) for token in sorted(context - direct))
            duration_gain = 7.0 if duration else 0.0
            direction_gain = 3.5 * sum(
                weight(token) for token in sorted(directed))
            source_time_gain = (
                1.0 if turn.timestamp
                and (direct or duration or directed) else 0.0)
            owner_gain = 1.5 if turn.speaker in explicit_speakers else 0.0
            scored_fallback.append((
                direct_score + context_score + duration_gain
                + direction_gain + source_time_gain + owner_gain,
                turn.turn_id))
        # Set intersections above have no semantic iteration order.  Round
        # sub-epsilon accumulation noise before the stable turn-id tie break,
        # otherwise different PYTHONHASHSEED values can exchange the final
        # boundary witness despite identical inputs.
        scored_fallback.sort(
            key=lambda item: (-round(item[0], 12), item[1]))
        seed_limit = max(4, min(
            max_turns, round(max_turns * 2.0 / 3.0)))
        seeds = tuple(
            turn_id for score, turn_id in scored_fallback[:seed_limit]
            if score > 0.0)
        selected = list(seeds)
        selected_set = set(seeds)
        neighbors: list[str] = []
        for turn_id in seeds:
            turn = next(row for row in ordered_turns
                        if row.turn_id == turn_id)
            for delta in (-1, 1):
                neighbor = by_coordinate.get(
                    (turn.session_id, turn.turn_index + delta))
                if neighbor is None or neighbor.turn_id in selected_set:
                    continue
                selected.append(neighbor.turn_id)
                selected_set.add(neighbor.turn_id)
                neighbors.append(neighbor.turn_id)
                if len(selected) >= max_turns:
                    break
            if len(selected) >= max_turns:
                break
        by_id = {turn.turn_id: turn for turn in ordered_turns}
        selected = sorted(selected[:max_turns], key=lambda turn_id: (
            by_id[turn_id].session_id, by_id[turn_id].turn_index, turn_id))
        return SourceFocusPlan(
            route=route, selected_turn_ids=tuple(selected),
            seed_turn_ids=seeds, neighbor_turn_ids=tuple(neighbors), trace={
                **empty_trace,
                "selected": len(selected),
                "temporal_fallback": "duration-and-event-transition",
                "temporal_relation": slots.temporal_relation,
                "duration_candidates": duration_candidates,
                "transition_candidates": transition_candidates,
                "uses_dense_scores": False,
            })

    query_views = (question,)
    query_terms = frozenset(
        token for view in query_views
        for token in temporal_tokenize(view)
        if (len(token) > 1 and token not in STOPWORDS
            and token not in _FOCUS_QUERY_SCAFFOLD))
    tokens_by_id = {
        turn.turn_id: frozenset(temporal_tokenize(
            f"{turn.speaker} {turn.raw_text}"))
        for turn in ordered_turns
    }
    document_frequency: Counter[str] = Counter(
        token for values in tokens_by_id.values() for token in values)
    count = len(ordered_turns)
    explicit_speakers = frozenset(
        turn.speaker for turn in ordered_turns
        if content_terms(turn.speaker)
        and content_terms(turn.speaker) <= content_terms(question))
    dense_rank = {
        turn_id: index
        for index, turn_id in enumerate(sorted(
            (turn_id for turn_id in turns
             if float((dense_scores or {}).get(turn_id, 0.0)) > 0.0),
            key=lambda turn_id: (
                -float((dense_scores or {}).get(turn_id, 0.0)), turn_id)),
            start=1)
    }

    def proximity(gap: float, *, event: bool) -> float:
        values = ((10.0, 7.0, 5.0, 3.0, 1.0) if event
                  else (5.0, 3.5, 2.5, 1.5, 0.5))
        if gap == 0.0:
            return values[0]
        if gap <= 1.0:
            return values[1]
        if gap <= 3.0:
            return values[2]
        if gap <= 7.0:
            return values[3]
        if gap <= 31.0:
            return values[4]
        return 0.0

    scored: list[tuple[float, str, bool]] = []
    event_matches = 0
    observation_matches = 0
    for turn in ordered_turns:
        event_score = 0.0
        event_exact = False
        for expression in extract_time_expressions(turn.raw_text):
            interval = normalize_time(expression, turn.timestamp, turn.turn_id)
            if not interval.start:
                continue
            gap = _interval_gap_days(
                target.start, target.end, interval.start, interval.end)
            event_score = max(event_score, proximity(gap, event=True))
            event_exact = event_exact or gap == 0.0
        observation_score = 0.0
        observation_exact = False
        observed = observed_interval(turn.timestamp, turn.turn_id)
        if observed is not None and observed.start:
            gap = _interval_gap_days(
                target.start, target.end, observed.start, observed.end)
            observation_score = proximity(gap, event=False)
            observation_exact = gap == 0.0
        if event_score:
            event_matches += 1
        if observation_score:
            observation_matches += 1
        lexical_score = sum(
            1.0 + math.log(
                (count + 1.0) /
                (document_frequency.get(token, 0) + 1.0))
            for token in sorted(tokens_by_id[turn.turn_id] & query_terms))
        owner_gain = 1.0 if turn.speaker in explicit_speakers else 0.0
        exact_gain = 4.0 if event_exact or observation_exact else 0.0
        dense_gain = (
            5.0 / math.sqrt(dense_rank[turn.turn_id])
            if turn.turn_id in dense_rank else 0.0)
        score = (
            event_score + observation_score + exact_gain
            + 1.25 * lexical_score + dense_gain + owner_gain)
        if event_score or observation_score:
            scored.append((
                score, turn.turn_id, event_exact or observation_exact))
    exact_scored = [row for row in scored if row[2]]
    scored.sort(key=lambda item: (-round(item[0], 12), item[1]))
    seed_limit = max(4, min(max_turns, round(max_turns * 2.0 / 3.0)))
    seeds = tuple(
        turn_id for score, turn_id, _exact in scored[:seed_limit]
        if score > 0.0)
    selected = list(seeds)
    selected_set = set(seeds)
    by_coordinate = {
        (turn.session_id, turn.turn_index): turn.turn_id
        for turn in ordered_turns
    }
    by_id = {turn.turn_id: turn for turn in ordered_turns}
    neighbors: list[str] = []
    for turn_id in seeds:
        turn = by_id[turn_id]
        for delta in (-1, 1):
            neighbor = by_coordinate.get(
                (turn.session_id, turn.turn_index + delta))
            if neighbor is None or neighbor in selected_set:
                continue
            selected.append(neighbor)
            selected_set.add(neighbor)
            neighbors.append(neighbor)
            if len(selected) >= max_turns:
                break
        if len(selected) >= max_turns:
            break
    selected = sorted(selected[:max_turns], key=lambda turn_id: (
        by_id[turn_id].session_id, by_id[turn_id].turn_index, turn_id))
    return SourceFocusPlan(
        route=route, selected_turn_ids=tuple(selected),
        seed_turn_ids=seeds, neighbor_turn_ids=tuple(neighbors), trace={
            **empty_trace,
            "selected": len(selected),
            "query_time_resolved": True,
            "query_time_start": target.start,
            "query_time_end": target.end,
            "query_time_precision": target.precision,
            "query_views": len(query_views),
            "exact_interval_candidates": len(exact_scored),
            "temporal_exact_interval_boost": bool(exact_scored),
            "uses_dense_scores": bool(dense_scores),
            "event_time_candidates": event_matches,
            "observation_time_candidates": observation_matches,
        })


def _flat_score(row: CandidateScore, route: str) -> float:
    """Return a source-facing score with every graph signal removed."""

    if route == "inference":
        return (0.45 * row.exact_score + 0.65 * row.bm25_score
                + 2.35 * row.dense_score + 0.08 * row.session_score
                + 0.12 * row.adjacency_score)
    if route in {"aggregate", "temporal", "state", "list"}:
        return (1.45 * row.exact_score + 1.15 * row.bm25_score
                + 1.10 * row.dense_score + 0.08 * row.session_score
                + 0.22 * row.adjacency_score)
    return (1.55 * row.exact_score + 1.25 * row.bm25_score
            + 1.30 * row.dense_score + 0.06 * row.session_score
            + 0.16 * row.adjacency_score)


def source_facing_candidates(
    rows: Iterable[CandidateScore], *, route: str,
) -> tuple[CandidateScore, ...]:
    """Strip graph/binding features and return a deterministic flat rank."""

    flat = tuple(replace(
        row,
        graph_score=0.0,
        graph_path_ids=(),
        relation_contributions=(),
        operand_ids=(),
        binding_score=0.0,
        relation_path_score=0.0,
        obligation_gain=0.0,
        provenance_novelty=0.0,
        relational_consensus_score=0.0,
        mandatory=False,
        fused_score=_flat_score(row, route),
        source_channels=tuple(
            channel for channel in row.source_channels
            if channel in {"exact", "bm25", "dense"}),
    ) for row in rows)
    return tuple(sorted(flat, key=lambda row: (
        -row.fused_score, -row.exact_score, -row.dense_score,
        -row.bm25_score, row.turn_id)))


def _features(turns: Mapping[str, SourceTurn]) -> tuple[
        dict[str, frozenset[str]], Counter[str]]:
    terms_by_turn = {
        turn_id: content_terms(turn.raw_text) for turn_id, turn in turns.items()
    }
    frequency: Counter[str] = Counter(
        term for values in terms_by_turn.values() for term in values)
    return terms_by_turn, frequency


def _operator_critical(
    rows: Sequence[CandidateScore], turns: Mapping[str, SourceTurn], route: str,
) -> tuple[str, ...]:
    if route not in {"aggregate", "temporal", "state"}:
        return ()
    pattern = {
        "aggregate": _NUMBER_RE,
        "temporal": _TIME_SURFACE_RE,
        "state": _STATUS_SURFACE_RE,
    }[route]
    return tuple(
        row.turn_id for row in rows
        if row.turn_id in turns and pattern.search(turns[row.turn_id].raw_text)
    )


def _dialogue_neighbors(
    turn_ids: Sequence[str], turns: Mapping[str, SourceTurn],
) -> tuple[str, ...]:
    by_position = {
        (turn.session_id, turn.turn_index): turn.turn_id
        for turn in turns.values()
    }
    result: list[str] = []
    for turn_id in turn_ids:
        turn = turns.get(turn_id)
        if turn is None:
            continue
        for delta in (-1, 1):
            neighbor = by_position.get((turn.session_id, turn.turn_index + delta))
            if neighbor is not None:
                result.append(neighbor)
    return tuple(dict.fromkeys(result))


def build_flat_fusion_plan(
    *, question: str, turns: Mapping[str, SourceTurn],
    candidate_scores: Sequence[CandidateScore],
    graph_turn_ids: Sequence[str], max_turns: int = 64,
    graph_head: int = 24, flat_head: int = 24,
) -> FlatPlan:
    """Fuse graph navigation with a lossless raw-turn physical lane.

    The first ``graph_head`` turns are immutable.  Remaining seats use RRF over
    graph and flat order, with explicit reservations for flat, dialogue and
    operator-critical evidence.  The method is label-free and bounded.
    """

    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    route = query_route(question)
    available = frozenset(turns)
    graph = tuple(dict.fromkeys(
        turn_id for turn_id in graph_turn_ids if turn_id in available))
    graph_head_ids = graph[:min(graph_head, max_turns)]
    original_by_id = {row.turn_id: row for row in candidate_scores}
    structurally_witnessed = tuple(
        turn_id for turn_id in graph
        if (turn_id in frozenset(graph_head_ids)
            or (row := original_by_id.get(turn_id)) is not None and (
                row.mandatory or row.graph_score > 0.0
                or row.binding_score > 0.0 or row.operand_ids
                or "query_witness" in row.source_channels)))
    terms_by_turn, frequency = _features(turns)
    flat_rows = source_facing_candidates(candidate_scores, route=route)
    answer_kind = {
        "aggregate": "count",
        "temporal": "temporal",
        "state": "latest_state",
        "list": "list",
        "inference": "inference",
        "multi_hop": "multi_hop",
    }.get(route, "lookup")
    ranked_flat, _precision, rank_trace = rank_query_aware_candidates(
        flat_rows, turns, query=question, answer_kind=answer_kind,
        max_turns=max_turns, terms_by_turn=terms_by_turn,
        document_frequency=frequency, witness_rare_df=4)
    flat_order = tuple(
        row.turn_id for row in ranked_flat if row.turn_id in available)
    flat_head_ids = flat_order[:min(flat_head, max_turns)]

    # Graph/binding/query-witness evidence is never sacrificed for a flat
    # candidate.  The previous prototype protected only a positional head and
    # removed 26 already-complete LoCoMo witness sets.  This signal-aware floor
    # retains the graph's unique contribution while still exposing unused,
    # unstructured tail seats to the second physical plan.
    selected: list[str] = list(structurally_witnessed[:max_turns])
    selected_set = set(selected)

    def admit(values: Iterable[str], quota: int) -> None:
        admitted = 0
        for turn_id in values:
            if len(selected) >= max_turns or admitted >= quota:
                return
            if turn_id not in available or turn_id in selected_set:
                continue
            selected.append(turn_id)
            selected_set.add(turn_id)
            admitted += 1

    # A route-specific reserve avoids turning direct lookup into the wide flat
    # baseline while giving exhaustive/temporal/inference questions more room.
    reserve = {
        "lookup": 10, "state": 14, "temporal": 16,
        "aggregate": 20, "list": 20, "multi_hop": 20,
        "inference": 24,
    }[route]
    admit(flat_head_ids, reserve)
    critical = _operator_critical(ranked_flat, turns, route)
    admit(critical, 6 if route in {"aggregate", "temporal", "state"} else 0)
    neighbors = _dialogue_neighbors((*graph_head_ids[:8], *flat_head_ids[:8]), turns)
    admit(neighbors, 8)

    graph_rank = {turn_id: rank for rank, turn_id in enumerate(graph)}
    flat_rank = {turn_id: rank for rank, turn_id in enumerate(flat_order)}
    candidates = tuple(dict.fromkeys((*graph, *flat_order)))
    rrf = sorted(candidates, key=lambda turn_id: (
        -(1.0 / (60 + graph_rank.get(turn_id, len(graph) + 256))
          + 1.0 / (60 + flat_rank.get(turn_id, len(flat_order) + 256))),
        flat_rank.get(turn_id, 1 << 30),
        graph_rank.get(turn_id, 1 << 30), turn_id))
    admit(rrf, max_turns)
    admit(graph, max_turns)
    admit(flat_order, max_turns)

    selected_tuple = tuple(selected[:max_turns])
    graph_set = frozenset(graph)
    selected_set = frozenset(selected_tuple)
    added = tuple(turn_id for turn_id in selected_tuple if turn_id not in graph_set)
    removed = tuple(turn_id for turn_id in graph if turn_id not in selected_set)
    flat_lookup = {row.turn_id: row for row in flat_rows}
    selected_rows = tuple(
        flat_lookup[turn_id] for turn_id in selected_tuple if turn_id in flat_lookup)
    return FlatPlan(
        route=route, ordered_candidates=selected_rows,
        selected_turn_ids=selected_tuple,
        graph_head_turn_ids=tuple(graph_head_ids),
        flat_head_turn_ids=tuple(flat_head_ids),
        added_flat_turn_ids=added, removed_graph_turn_ids=removed,
        trace={
            "version": FLAT_PLAN_VERSION,
            "route": route,
            "max_turns": max_turns,
            "graph_candidates": len(graph),
            "flat_candidates": len(flat_order),
            "graph_head": len(graph_head_ids),
            "structurally_witnessed_floor": len(structurally_witnessed),
            "flat_reserve": reserve,
            "selected": len(selected_tuple),
            "added_flat": len(added),
            "removed_graph": len(removed),
            "rank_lanes": dict(rank_trace),
        },
    )


def answer_support_score(answer: str, turns: Iterable[SourceTurn]) -> float:
    """Measure whether a previous answer has lexical/numeric source support."""

    normalized = " ".join(answer.casefold().split())
    terms = {
        term for term in content_terms(normalized)
        if len(term) > 2 and term not in _GENERIC_ANSWER_TERMS
    }
    numbers = set(_NUMBER_RE.findall(normalized))
    source = "\n".join(turn.raw_text.casefold() for turn in turns)
    term_score = (sum(term in source for term in terms) / len(terms)
                  if terms else 1.0)
    number_score = (sum(value.casefold() in source for value in numbers)
                    / len(numbers) if numbers else 1.0)
    return min(term_score, number_score) if numbers else term_score


def verification_gate(
    *, question: str, previous_answer: str, plan: FlatPlan,
    graph_turns: Sequence[SourceTurn],
) -> VerificationGate:
    """Select a second answer pass without labels, gold or judge verdicts."""

    support = answer_support_score(previous_answer, graph_turns)
    reasons: list[str] = []
    if plan.route in {"aggregate", "temporal", "state", "multi_hop", "inference"}:
        reasons.append(f"typed_route:{plan.route}")
    if _ABSTAIN_RE.search(previous_answer):
        reasons.append("previous_abstention")
    if support < 0.60:
        reasons.append("weak_source_support")
    if len(plan.added_flat_turn_ids) >= 4:
        reasons.append("flat_graph_divergence")
    # A plain lookup with a well-supported answer remains frozen even when the
    # two ranks differ.  This is the primary regression barrier.
    eligible = bool(reasons) and not (
        plan.route == "lookup" and support >= 0.80
        and not _ABSTAIN_RE.search(previous_answer))
    return VerificationGate(
        eligible=eligible, reasons=tuple(reasons), answer_support=support,
        flat_novelty=len(plan.added_flat_turn_ids), route=plan.route)


__all__ = [
    "FLAT_PLAN_VERSION", "SOURCE_FOCUS_VERSION", "FlatPlan",
    "QueryObligations", "SourceFocusPlan", "VerificationGate",
    "answer_support_score", "append_source_focus_witnesses",
    "build_flat_fusion_plan", "build_source_focus_plan",
    "build_temporal_focus_plan", "compile_obligation_query_view",
    "compile_query_obligations", "compile_source_query_views", "query_route",
    "preserve_graph_with_flat_packet",
    "source_facing_candidates",
    "verification_gate",
]
