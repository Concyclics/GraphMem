"""Typed evidence layout and conditional final-answer verification.

This is the read side of the V5.70 dual physical plan.  It turns a bounded set
of immutable source turns into a topology-local, numbered layout and a compact
operator worksheet.  The worksheet contains routing metadata only; source
memories remain the sole factual authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from ..domain import CandidateScore, SourceTurn
from ..retrieval.flat_plan import FlatPlan, SourceFocusPlan
from ..retrieval.packer import salient_spans
from ..text import content_terms
from .rendering import AnswerConfig, render_turn


VERIFIED_READOUT_VERSION = "graphmem-v5.80-obligation-readout-v1"

_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$£€¥]\s*)?-?\d+(?:,\d{3})*(?:\.\d+)?(?:%|st|nd|rd|th)?",
    re.I,
)
_RELATIVE_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|last\s+\w+|next\s+\w+|"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago|before|after|since|until)\b",
    re.I,
)
_STATUS_RE = re.compile(
    r"\b(?:currently|now|latest|still|no longer|replaced|moved|"
    r"cancelled|canceled|completed|finished|planned|planning|"
    r"wanted|intended|might|will)\b",
    re.I,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|didn't|doesn't|hasn't|haven't|"
    r"cancelled|canceled|declined|skipped)\b",
    re.I,
)

SYSTEM_PROMPT = """You are the final source-grounded memory executor. Use only
the numbered source memories as factual evidence. A previous answer and the
execution worksheet are fallible proposals, never evidence. Before answering,
privately bind the exact subject, relation, object, owner/speaker, event status
and event time requested by the question. Reject same-topic facts about a
different person, object, event or time.

For counts and lists, scan the complete supplied evidence set, include every
matching realized item, exclude plans/negations/cancellations, and deduplicate
repeated mentions rather than distinct occurrences. For temporal questions,
separate event time from mention time and resolve each relative expression
from its own source-time annotation. For current state, apply superseding
updates in event-time order. For multi-hop questions, verify every join. For
inference, make only the narrowest ordinary inference supported by multiple
source facts; do not abstain when those facts are sufficient.

Return only the concise final answer. Do not output reasoning, citations,
JSON, a preamble, or repeat the answer."""

FOCUS_SYSTEM_PROMPT = """You are an independent source-focus readout of a
conversation-memory system. Answer the exact question using only the selected
original turns. Keep subject, relation, value, polarity, completion status,
quantity, unit and event time exact. Resolve relative dates from each turn's
source-time. For lists and counts, combine all supplied turns and deduplicate
repeated mentions; for inference, make only the narrowest ordinary inference
supported by the turns. Reject same-topic facts about another person, event or
time. Return one concise final answer only, without analysis or citations."""

UNIFIED_SYSTEM_PROMPT = """Answer the exact question using only the supplied
original conversation turns. The Graph witnesses and Source-focus witnesses
are two retrieval views of the same source corpus; neither section is more
truthful, and repeated turns have already been removed. Bind the requested
speaker, entity, relation, polarity, completion status, quantity, unit and
event time exactly. For lists and counts, scan both sections, combine every
matching item and deduplicate repeated mentions. Prefer completed events when
the question asks what was done; if the only directly bound source is a plan
or consideration, preserve that status in the answer instead of silently
dropping the item. For temporal and current-state questions, resolve relative
dates from each turn's source-time and apply superseding updates in event-time
order. For multi-hop questions, verify every join from the source turns. For
inference questions, combine the directly stated premises and make only the
narrowest ordinary inference; the answer wording need not occur verbatim.
Reject same-topic facts about another person, event or time. Return one concise
final answer only, without analysis, citations, JSON or a preamble."""

UNIFIED_RELATION_SYSTEM_APPENDIX = """

Compact {via=a>b} tags describe retrieval-route families, not additional
facts. Use them to keep related graph witnesses together while verifying every
answer value against the adjacent original conversation text."""

UNIFIED_FOCUS_CAPSULE_APPENDIX = """

The final Query-focus capsule contains exact repeats of a few numbered source
turns to reduce long-context attention loss. It is a navigation aid, not new
evidence: count each repeated source turn only once."""

UNIFIED_FOCUS_MAP_APPENDIX = """

The Query-view navigation map contains retrieval pointers, not facts or a
completeness claim. Verify referenced source text and scan both sections."""

UNIFIED_OBLIGATION_SYSTEM_APPENDIX = """

Obligation checks are instructions, not facts. Ground personal claims in the
source turns. For inference or geography, one stable ordinary-knowledge step
may connect explicit premises. Preserve uncertainty; invent no personal facts."""


ROUTE_CONTRACTS = {
    "aggregate": (
        "Construct an exhaustive include/exclude ledger; normalize aliases, "
        "deduplicate, then perform the requested count/sum/comparison once."),
    "temporal": (
        "Construct an event-time table with all candidate endpoints; enforce "
        "first/last/ordinal/before/after constraints and recompute the result."),
    "state": (
        "Bind one subject and attribute, order its values by event time, and "
        "return the latest non-superseded value requested."),
    "multi_hop": (
        "Write the subject-object value for every hop and accept the answer "
        "only when the values join into one connected chain."),
    "inference": (
        "Aggregate directly stated traits first, then make one minimal common-"
        "sense inference; do not substitute a nearby person's traits."),
    "list": (
        "Collect every coordinated value requested, preserve specificity, and "
        "do not stop after the first matching source."),
    "lookup": (
        "Prefer the exact source span bound to the requested subject and "
        "relation; verify attribution and return the requested field only."),
}

OBLIGATION_CONTRACTS = {
    "aggregate": "enumerate qualifying events before computing the scalar",
    "temporal": "bind event time separately from source observation time",
    "latest_state": "apply updates only to the same subject and attribute",
    "exhaustive_set": "keep collecting until every supplied owner/value is checked",
    "multi_entity": "reserve and verify one evidence row per named subject",
    "comparison": "compare only values with the same relation and unit",
    "causal": "separate the stated cause, outcome and merely adjacent events",
    "counterfactual": "separate observed facts from the hypothetical condition",
    "inference": "state the supporting premises, then make the narrowest inference",
    "multi_hop": "require an explicit source-supported join at every hop",
    "geographic_resolution": "resolve only the named source place to the requested level",
    "alias_resolution": "distinguish a direct name or address form from a guessed alias",
    "negative_existence": "distinguish plans/consideration from completed existence",
    "reasoning_chain": "keep premise, bridge and requested conclusion distinct",
    "direct_lookup": "return only the field directly bound to subject and relation",
}


@dataclass(frozen=True, slots=True)
class RenderedVerifiedPrompt:
    messages: tuple[Mapping[str, str], ...]
    evidence_turn_ids: tuple[str, ...]
    evidence_blocks: tuple[str, ...]
    workspace: str
    trace: Mapping[str, object]


def _topological_order(
    turn_ids: Sequence[str], turns: Mapping[str, SourceTurn], route: str,
    priority_turn_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Keep dialogue-local witnesses together without trusting graph labels."""

    rank = {turn_id: index for index, turn_id in enumerate(turn_ids)}
    priority_rank = {
        turn_id: index for index, turn_id in enumerate(priority_turn_ids)
    }
    by_session: dict[str, list[str]] = {}
    for turn_id in turn_ids:
        turn = turns.get(turn_id)
        if turn is not None:
            by_session.setdefault(turn.session_id, []).append(turn_id)
    cluster_rank = {
        session_id: min(
            priority_rank.get(turn_id, len(priority_rank) + rank[turn_id])
            for turn_id in ids)
        for session_id, ids in by_session.items()
    }
    sessions = sorted(by_session, key=lambda session_id: (
        cluster_rank[session_id], session_id))
    result: list[str] = []
    for session_id in sessions:
        ids = sorted(by_session[session_id], key=lambda turn_id: (
            turns[turn_id].turn_index, rank[turn_id], turn_id))
        result.extend(ids)
    return tuple(result)


def _surface_tags(text: str) -> tuple[str, ...]:
    tags: list[str] = []
    if _NUMBER_RE.search(text):
        tags.append("numeric")
    if _RELATIVE_RE.search(text):
        tags.append("relative-time")
    if _STATUS_RE.search(text):
        tags.append("state/status")
    if _NEGATION_RE.search(text):
        tags.append("negative/update")
    return tuple(tags)


def _bounded_excerpt(
    rendered: str, *, question: str, route: str, max_chars: int,
) -> str:
    """Bound punctuation-free turns around a query/typed anchor.

    Some source turns contain thousands of characters without a sentence
    boundary.  ``salient_spans`` correctly chooses that sentence but cannot
    make it small.  This final presentation bound keeps the source header,
    centers the body on the best query term (or typed number/time/status
    anchor), and preserves a source-time annotation when one exists.
    """

    if len(rendered) <= max_chars:
        return rendered
    header_end = rendered.find(": ")
    if header_end < 0:
        header_end = rendered.find("] ")
    header_end = min(len(rendered), header_end + 2 if header_end >= 0 else 0)
    header = rendered[:header_end]
    body = rendered[header_end:]
    lowered = body.casefold()
    terms = sorted(content_terms(question), key=lambda value: (-len(value), value))
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    typed = {
        "aggregate": _NUMBER_RE,
        "temporal": _RELATIVE_RE,
        "state": _STATUS_RE,
    }.get(route)
    if typed is not None and (match := typed.search(body)) is not None:
        positions.append(match.start())
    center = min(positions) if positions else 0
    source_note = ""
    note_start = body.rfind("[source-time ")
    if note_start >= 0:
        source_note = body[note_start:].strip()
        body = body[:note_start].rstrip()
    note_budget = min(len(source_note) + 1, max_chars // 3) if source_note else 0
    body_budget = max(80, max_chars - len(header) - note_budget - 6)
    start = max(0, center - body_budget // 3)
    end = min(len(body), start + body_budget)
    if end - start < body_budget:
        start = max(0, end - body_budget)
    excerpt = body[start:end].strip()
    if start:
        excerpt = "… " + excerpt
    if end < len(body):
        excerpt += " …"
    if source_note:
        allowed = max(0, max_chars - len(header) - len(excerpt) - 1)
        excerpt += " " + source_note[:allowed]
    return (header + excerpt)[:max_chars]


def _workspace(
    *, route: str, ordered: Sequence[str], turns: Mapping[str, SourceTurn],
    score_by_id: Mapping[str, CandidateScore], question: str,
) -> str:
    rows = [
        "Typed execution worksheet (routing metadata only; verify every item "
        "against the numbered source memories):",
        f"operator={route}",
        "obligation=" + ROUTE_CONTRACTS[route],
    ]
    candidates: list[str] = []
    for index, turn_id in enumerate(ordered, 1):
        turn = turns[turn_id]
        tags = _surface_tags(turn.raw_text)
        score = score_by_id.get(turn_id)
        # The worksheet intentionally exposes only source characteristics, not
        # graph rank or a guessed value.  Rank is not factual evidence.
        if tags or index <= 8:
            channels = ",".join(score.source_channels) if score else "source"
            candidates.append(
                f"E{index:02d}: speaker={turn.speaker or turn.role}; "
                f"source_time={turn.timestamp or 'unknown'}; "
                f"surface={','.join(tags) or 'direct'}; channels={channels}")
        if len(candidates) >= 16:
            break
    rows.extend(candidates)
    rows.extend((
        "closure=unproven: scan all numbered memories before declaring a set "
        "complete or information missing.",
        f"original_question={question}",
    ))
    return "\n".join(rows)


def build_verified_prompt(
    *, question: str, question_date: str, previous_answer: str,
    plan: FlatPlan, turns: Mapping[str, SourceTurn],
    max_spans: int = 2, max_block_chars: int = 640,
) -> RenderedVerifiedPrompt:
    """Render a bounded, source-only verifier request from a fusion plan."""

    ordered = _topological_order(plan.selected_turn_ids, turns, plan.route)
    config = AnswerConfig.v5_63()
    score_by_id = {row.turn_id: row for row in plan.ordered_candidates}
    blocks: list[str] = []
    cluster_number: dict[str, int] = {}
    cluster_steps: dict[str, int] = {}
    for turn_id in ordered:
        turn = turns[turn_id]
        if turn.session_id not in cluster_number:
            cluster_number[turn.session_id] = len(cluster_number) + 1
        cluster_steps[turn.session_id] = cluster_steps.get(turn.session_id, 0) + 1
        kind = {
            "aggregate": "count", "temporal": "temporal",
            "state": "latest_state", "list": "list",
        }.get(plan.route, plan.route)
        spans = salient_spans(
            turn, question, answer_kind=kind, max_spans=max_spans)
        rendered = _bounded_excerpt(
            render_turn(turn, config, spans), question=question,
            route=plan.route, max_chars=max_block_chars)
        blocks.append(
            f"[E{len(blocks) + 1:02d} cluster="
            f"{cluster_number[turn.session_id]} step="
            f"{cluster_steps[turn.session_id]}] {rendered}")

    workspace = _workspace(
        route=plan.route, ordered=ordered, turns=turns,
        score_by_id=score_by_id, question=question)
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n"
        f"Physical plan: graph navigation + lossless source-turn retrieval\n\n"
        "Numbered source memories (related turns are grouped locally):\n"
        + "\n".join(blocks)
        + "\n\n" + workspace
        + "\n\nFallible previous answer (keep it only if the source memories "
          "support the exact requested field):\n"
        + (previous_answer or "(empty)")
        + "\n\nIndependently verify or correct the answer now. Return only one "
          "concise final answer."
    )
    messages: tuple[Mapping[str, str], ...] = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    return RenderedVerifiedPrompt(
        messages=messages, evidence_turn_ids=ordered,
        evidence_blocks=tuple(blocks), workspace=workspace,
        trace={
            "version": VERIFIED_READOUT_VERSION,
            "route": plan.route,
            "turns": len(ordered),
            "clusters": len(cluster_number),
            "added_flat_turn_ids": list(plan.added_flat_turn_ids),
            "removed_graph_turn_ids": list(plan.removed_graph_turn_ids),
            "layout": "query-ranked-clusters/source-order-within-cluster",
            "max_block_chars": max_block_chars,
            "uses_previous_answer_as_evidence": False,
        },
    )


def build_source_focus_prompt(
    *, question: str, question_date: str, plan: SourceFocusPlan,
    turns: Mapping[str, SourceTurn], max_spans: int = 2,
    max_block_chars: int = 640,
) -> RenderedVerifiedPrompt:
    """Render a compact graph-independent candidate-generation view."""

    ordered = _topological_order(plan.selected_turn_ids, turns, plan.route)
    config = AnswerConfig.v5_63()
    blocks: list[str] = []
    for index, turn_id in enumerate(ordered, 1):
        turn = turns[turn_id]
        kind = {
            "aggregate": "count", "temporal": "temporal",
            "state": "latest_state", "list": "list",
        }.get(plan.route, plan.route)
        spans = salient_spans(
            turn, question, answer_kind=kind, max_spans=max_spans)
        rendered = _bounded_excerpt(
            render_turn(turn, config, spans), question=question,
            route=plan.route, max_chars=max_block_chars)
        blocks.append(f"[F{index:02d}] {rendered}")
    workspace = (
        f"operator={plan.route}\n"
        f"obligation={ROUTE_CONTRACTS[plan.route]}\n"
        "Treat omitted turns as unknown, not as negative evidence."
    )
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n\n"
        "Query-focused original source turns:\n" + "\n".join(blocks)
        + "\n\n" + workspace
        + "\n\nAnswer the original Question now."
    )
    messages: tuple[Mapping[str, str], ...] = (
        {"role": "system", "content": FOCUS_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )
    return RenderedVerifiedPrompt(
        messages=messages, evidence_turn_ids=ordered,
        evidence_blocks=tuple(blocks), workspace=workspace,
        trace={
            **dict(plan.trace),
            "readout_version": VERIFIED_READOUT_VERSION,
            "layout": "query-focused/source-order-within-session",
            "uses_previous_answer_as_evidence": False,
        },
    )


def build_unified_source_prompt(
    *, question: str, question_date: str,
    graph_turn_ids: Sequence[str], focus_plan: SourceFocusPlan,
    turns: Mapping[str, SourceTurn], graph_limit: int = 32,
    focus_limit: int = 32, max_spans: int = 2,
    max_block_chars: int = 640,
    relation_hints: Mapping[str, str] | None = None,
    promote_focus_overlap: bool = False,
    focus_lossless_extra_chars: int = 0,
    rank_focus_sessions: bool = False,
    focus_capsule_turns: int = 0,
    focus_capsule_graph_only: bool = False,
    focus_capsule_routes: Sequence[str] = (),
    focus_capsule_max_chars: int = 0,
    focus_navigation_map: bool = False,
    query_obligations: Sequence[str] = (),
) -> RenderedVerifiedPrompt:
    """Render one deterministic prompt from complementary retrieval views.

    The evidence budget is split before rendering so Best-of-N responses all
    consume an identical payload.  Source-focus evidence supplements the graph
    witnesses but cannot duplicate them or silently displace the graph prefix.
    """

    if graph_limit < 0 or focus_limit < 0 or graph_limit + focus_limit <= 0:
        raise ValueError("unified evidence limits must be non-negative")
    if focus_lossless_extra_chars < 0:
        raise ValueError("focus_lossless_extra_chars must be non-negative")
    if focus_capsule_turns < 0:
        raise ValueError("focus_capsule_turns must be non-negative")
    if focus_capsule_max_chars < 0:
        raise ValueError("focus_capsule_max_chars must be non-negative")
    unknown_capsule_routes = set(focus_capsule_routes) - set(ROUTE_CONTRACTS)
    if unknown_capsule_routes:
        raise ValueError(
            f"unknown focus capsule routes: {sorted(unknown_capsule_routes)}")
    obligations = tuple(dict.fromkeys(map(str, query_obligations)))
    unknown_obligations = set(obligations) - set(OBLIGATION_CONTRACTS)
    if unknown_obligations:
        raise ValueError(
            f"unknown query obligations: {sorted(unknown_obligations)}")
    available = frozenset(turns)
    graph = tuple(dict.fromkeys(
        turn_id for turn_id in graph_turn_ids if turn_id in available
    ))[:graph_limit]
    focus_candidates = tuple(dict.fromkeys(
        turn_id for turn_id in focus_plan.selected_turn_ids
        if turn_id in available))
    promoted: tuple[str, ...] = ()
    if promote_focus_overlap:
        # A strong focus witness was previously suppressed whenever the graph
        # view had already retrieved it.  That avoids duplicate source text,
        # but it also buries the best query-facing evidence among up to 64
        # graph turns.  Move (do not copy) those overlaps into the final focus
        # section so the evidence set and source authority remain unchanged.
        focus = focus_candidates[:focus_limit]
        focus_set = frozenset(focus)
        promoted = tuple(turn_id for turn_id in graph if turn_id in focus_set)
        graph = tuple(turn_id for turn_id in graph if turn_id not in focus_set)
    else:
        graph_set = frozenset(graph)
        focus = tuple(
            turn_id for turn_id in focus_candidates
            if turn_id not in graph_set
        )[:focus_limit]
    graph_ordered = _topological_order(graph, turns, focus_plan.route)
    focus_priority = tuple(dict.fromkeys((
        *focus_plan.seed_turn_ids,
        *focus_plan.neighbor_turn_ids,
        *focus_plan.selected_turn_ids,
    ))) if rank_focus_sessions else ()
    focus_ordered = _topological_order(
        focus, turns, focus_plan.route, focus_priority)
    config = AnswerConfig.v5_63()

    def compact_turn(turn_id: str) -> str:
        turn = turns[turn_id]
        kind = {
            "aggregate": "count", "temporal": "temporal",
            "state": "latest_state", "list": "list",
        }.get(focus_plan.route, focus_plan.route)
        spans = salient_spans(
            turn, question, answer_kind=kind, max_spans=max_spans)
        return _bounded_excerpt(
            render_turn(turn, config, spans), question=question,
            route=focus_plan.route, max_chars=max_block_chars)

    compact_by_id = {
        turn_id: compact_turn(turn_id)
        for turn_id in (*graph_ordered, *focus_ordered)
    }
    expanded_focus: dict[str, str] = {}
    expansion_chars = 0
    if focus_lossless_extra_chars:
        # Salient sentence windows can retain a pronoun such as "it" while
        # clipping the short preceding sentence that names the actual book,
        # person or place.  Restore complete *short* source turns only in the
        # query-focused lane, with a hard aggregate character budget.  This is
        # deterministic source text, not an LLM-generated summary.
        priority = tuple(dict.fromkeys((
            *focus_plan.seed_turn_ids,
            *focus_plan.neighbor_turn_ids,
            *focus_plan.selected_turn_ids,
        )))
        focus_set = frozenset(focus_ordered)
        for turn_id in priority:
            if turn_id not in focus_set:
                continue
            full = render_turn(turns[turn_id], config)
            if len(full) > max_block_chars:
                continue
            extra = max(0, len(full) - len(compact_by_id[turn_id]))
            if not extra or expansion_chars + extra > focus_lossless_extra_chars:
                continue
            expanded_focus[turn_id] = full
            expansion_chars += extra

    def blocks(prefix: str, ids: Sequence[str]) -> list[str]:
        values: list[str] = []
        for index, turn_id in enumerate(ids, 1):
            rendered = (
                expanded_focus.get(turn_id, compact_by_id[turn_id])
                if prefix == "F" else compact_by_id[turn_id])
            hint = str((relation_hints or {}).get(turn_id, ""))
            values.append(f"[{prefix}{index:02d}]{hint} {rendered}")
        return values

    graph_blocks = blocks("G", graph_ordered)
    focus_blocks = blocks("F", focus_ordered)
    location_by_id = {
        turn_id: f"G{index:02d}"
        for index, turn_id in enumerate(graph_ordered, 1)}
    location_by_id.update({
        turn_id: f"F{index:02d}"
        for index, turn_id in enumerate(focus_ordered, 1)})
    auxiliary_ids = tuple(
        str(turn_id)
        for view in focus_plan.trace.get("auxiliary_focus_views", ())
        for turn_id in dict(view).get("extra_turn_ids", ()))
    capsule_priority = tuple(dict.fromkeys((
        *focus_plan.seed_turn_ids,
        *auxiliary_ids,
        *focus_plan.neighbor_turn_ids,
        *focus_plan.selected_turn_ids,
    )))
    capsule_route_enabled = (
        not focus_capsule_routes or focus_plan.route in focus_capsule_routes)
    capsule_ids = (
        tuple(
            turn_id for turn_id in capsule_priority
            if (turn_id in location_by_id
                and (not focus_capsule_graph_only
                     or location_by_id[turn_id].startswith("G")))
        )[:focus_capsule_turns]
        if capsule_route_enabled else ())
    capsule_blocks: list[str] = []
    capsule_char_limit = focus_capsule_max_chars or max_block_chars
    for index, turn_id in enumerate(capsule_ids, 1):
        full = render_turn(turns[turn_id], config)
        rendered = (
            full if len(full) <= capsule_char_limit
            else _bounded_excerpt(
                full, question=question, route=focus_plan.route,
                max_chars=capsule_char_limit))
        capsule_blocks.append(
            f"[P{index:02d} repeats={location_by_id[turn_id]}] {rendered}")
    capsule_text = (
        "\n\nQuery-focus capsule (exact source repeats; count each source "
        "turn once):\n" + "\n".join(capsule_blocks)
        if capsule_blocks else "")
    view_priority = {
        "temporal": (
            "temporal_transition", "temporal", "obligation_dense",
            "relation_concept_expanded", "relation_concept",
            "relation_lexical", "dense", "morphological"),
        "aggregate": (
            "obligation_dense", "relation_concept_expanded", "relation_concept",
            "relation_lexical", "temporal", "dense", "morphological"),
        "list": (
            "obligation_dense", "relation_concept_expanded", "relation_concept",
            "relation_lexical", "temporal_transition", "dense",
            "morphological"),
        "multi_hop": (
            "obligation_dense", "relation_concept_expanded", "relation_concept",
            "relation_lexical", "temporal_transition", "dense",
            "morphological"),
        "inference": (
            "obligation_dense", "dense", "relation_concept_expanded", "relation_concept",
            "relation_lexical", "morphological"),
        "state": (
            "temporal", "obligation_dense", "relation_concept_expanded", "relation_concept",
            "relation_lexical", "dense", "morphological"),
        "lookup": (
            "obligation_dense", "relation_concept_expanded", "relation_concept", "dense",
            "relation_lexical", "temporal", "morphological"),
    }[focus_plan.route]
    auxiliary_views = tuple(
        dict(view) for view in focus_plan.trace.get(
            "auxiliary_focus_views", ()))
    view_label = {
        "temporal_transition": "event-order/duration",
        "temporal": "normalized-time",
        "relation_concept_expanded": "relation-synonym/direction",
        "relation_concept": "relation-match",
        "relation_lexical": "dialogue-closure",
        "dense": "semantic",
        "morphological": "surface-form",
        "obligation_dense": "obligation-semantic",
    }
    views_by_type: dict[str, list[str]] = {}
    for view in auxiliary_views:
        view_type = str(view.get("type", "source"))
        for turn_id in view.get("extra_turn_ids", ()):
            turn_id = str(turn_id)
            if turn_id in location_by_id:
                views_by_type.setdefault(view_type, []).append(turn_id)
    focus_map_rows: list[str] = []
    focus_map_ids: list[str] = []
    for view_type in view_priority:
        locations = tuple(dict.fromkeys(
            location_by_id[turn_id]
            for turn_id in views_by_type.get(view_type, ())))
        if not locations:
            continue
        remaining = 6 - len(focus_map_ids)
        if remaining <= 0:
            break
        locations = locations[:min(2, remaining)]
        focus_map_rows.append(
            f"- {view_label.get(view_type, view_type)}: "
            f"{','.join(locations)}")
        focus_map_ids.extend(locations)
        if len(focus_map_rows) >= 3:
            break
    focus_map_text = (
        "\n\nQuery-view navigation map (provenance pointers only):\n"
        + "\n".join(focus_map_rows)
        if focus_navigation_map and focus_map_rows else "")
    visible_obligations = tuple(
        value for value in obligations if value != "reasoning_chain")
    obligation_rows = tuple(
        f"- {value}: {OBLIGATION_CONTRACTS[value]}"
        for value in visible_obligations[:5])
    obligation_text = (
        "\nQuery obligations: " + ", ".join(visible_obligations)
        + "\nObligation checks:\n" + "\n".join(obligation_rows)
        if obligation_rows else "")
    derived_answer_enabled = bool(set(obligations) & {
        "inference", "causal", "counterfactual", "geographic_resolution"})
    user = (
        f"Question: {question}\n"
        f"Question date: {question_date or 'unknown'}\n"
        f"Query operator: {focus_plan.route}\n"
        f"Execution obligation: {ROUTE_CONTRACTS[focus_plan.route]}"
        + obligation_text + "\n\n"
        "Graph witnesses (query paths, grouped by source session):\n"
        + ("\n".join(graph_blocks) or "(none)")
        + "\n\nSource-focus witnesses (independent query view):\n"
        + ("\n".join(focus_blocks) or "(none)")
        + capsule_text
        + focus_map_text
        + "\n\nAnswer the original Question now."
    )
    relation_labels_active = bool(relation_hints)
    system_prompt = (
        UNIFIED_SYSTEM_PROMPT
        + (UNIFIED_RELATION_SYSTEM_APPENDIX if relation_labels_active else "")
        + (UNIFIED_FOCUS_CAPSULE_APPENDIX if capsule_blocks else "")
        + (UNIFIED_FOCUS_MAP_APPENDIX if focus_map_text else "")
        + (UNIFIED_OBLIGATION_SYSTEM_APPENDIX
           if derived_answer_enabled else ""))
    messages: tuple[Mapping[str, str], ...] = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    )
    evidence = (*graph_ordered, *focus_ordered)
    return RenderedVerifiedPrompt(
        messages=messages, evidence_turn_ids=evidence,
        evidence_blocks=tuple((*graph_blocks, *focus_blocks)),
        workspace=f"operator={focus_plan.route}",
        trace={
            **dict(focus_plan.trace),
            "readout_version": VERIFIED_READOUT_VERSION,
            "layout": "unified-graph32-focus32/source-order-within-view",
            "graph_turns": len(graph_ordered),
            "focus_turns": len(focus_ordered),
            "unique_turns": len(evidence),
            "focus_overlap_promoted": bool(promote_focus_overlap),
            "promoted_focus_turns": len(promoted),
            "lossless_focus_turns": len(expanded_focus),
            "lossless_focus_extra_chars": expansion_chars,
            "lossless_focus_extra_char_limit": focus_lossless_extra_chars,
            "focus_sessions_query_ranked": bool(rank_focus_sessions),
            "focus_capsule_turns": len(capsule_ids),
            "focus_capsule_turn_ids": list(capsule_ids),
            "focus_capsule_graph_only": bool(focus_capsule_graph_only),
            "focus_capsule_route_enabled": capsule_route_enabled,
            "focus_capsule_routes": list(focus_capsule_routes),
            "focus_capsule_max_chars": capsule_char_limit,
            "focus_navigation_map": bool(focus_map_text),
            "focus_navigation_map_rows": list(focus_map_rows),
            "focus_navigation_map_ids": list(focus_map_ids),
            "query_obligations": list(obligations),
            "query_obligation_rows": len(obligation_rows),
            "derived_answer_enabled": derived_answer_enabled,
            "relation_path_labels": relation_labels_active,
            "relation_labeled_turns": sum(
                turn_id in (relation_hints or {}) for turn_id in graph_ordered),
            "uses_previous_answer_as_evidence": False,
            "uses_gold_or_judge": False,
        },
    )


__all__ = [
    "FOCUS_SYSTEM_PROMPT", "RenderedVerifiedPrompt", "SYSTEM_PROMPT",
    "OBLIGATION_CONTRACTS",
    "UNIFIED_FOCUS_CAPSULE_APPENDIX", "UNIFIED_FOCUS_MAP_APPENDIX",
    "UNIFIED_RELATION_SYSTEM_APPENDIX",
    "UNIFIED_OBLIGATION_SYSTEM_APPENDIX",
    "UNIFIED_SYSTEM_PROMPT",
    "VERIFIED_READOUT_VERSION",
    "build_source_focus_prompt", "build_unified_source_prompt",
    "build_verified_prompt",
]
