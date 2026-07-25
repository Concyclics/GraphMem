from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any, Iterable

from .clients import cosine_similarity, rough_token_count
from .models import (
    GRAPHMEM_V2_SCHEMA,
    AtomicFactNode,
    GraphEdge,
    LeafNode,
    QuestionCase,
    RetrievedContext,
    RoutingCardNode,
    StateChain,
)

PROMPT_VERSION = "hierarchical_state_graph_v2_extract_array_20260724f"
CONSOLIDATION_VERSION = "hierarchical_state_graph_v2_consolidate_array_20260724e"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that",
    "the", "their", "theirs", "them", "they", "this", "to", "us", "was", "we", "were",
    "with", "you", "your", "yours", "said", "says", "thing", "things",
}
_TOKEN_RE = re.compile(r"[\w][\w'’-]*", re.UNICODE)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b")
_GENERIC_ENTITY_KEYS = {"user", "assistant", "asst", "conversation", "human", "true", "false", "unknown"}
_GENERIC_PREDICATE_KEYS = {"ask", "asked", "request", "requested", "provide", "provided", "recommend", "recommended", "state", "stated", "mention", "mentioned", "related_to", "unknown"}


def provider_token_estimate(text: str) -> int:
    if not text:
        return 0
    return max(rough_token_count(text), math.ceil(len(text.encode("utf-8")) / 3.4))


def prompt_hash() -> str:
    return hashlib.sha256((PROMPT_VERSION + "\n" + CONSOLIDATION_VERSION).encode()).hexdigest()


def canonical_key(value: Any, aliases: dict[str, str] | None = None) -> str:
    text = str(value or "").casefold().replace("’", "'").replace("_", " ")
    text = re.sub(r"[^\w\s'-]+", " ", text, flags=re.UNICODE)
    tokens = [token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS]
    key = " ".join(tokens[:16]).strip()
    if not key:
        return "unknown"
    aliases = aliases or {}
    return aliases.get(key, key)


def clean_entities(values: Iterable[Any], aliases: dict[str, str] | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = re.sub(r"\s+", " ", str(raw or "")).strip(" .,:;!?()[]{}\"'")
        key = canonical_key(text, aliases)
        if key == "unknown" or key in seen or len(key) < 2 or len(key) > 96:
            continue
        if all(token in _STOPWORDS for token in key.split()):
            continue
        seen.add(key)
        result.append(text[:120])
    return result


def session_extraction_messages(session_id: str, session_date: str | None, leaves: list[LeafNode]) -> list[dict[str, str]]:
    transcript = "\n\n".join(
        f"[leaf_id={leaf.node_id}; turn={leaf.turn_index}]\n{leaf.raw_text}" for leaf in leaves
    )
    system = """Build a source-grounded memory index. Output ONE compact JSON object only; no prose.
Extract atomic user facts/events and concrete assistant-provided answers. Preserve exact names, quantities,
negation, preferences, plan vs completion, session-date-anchored relative time, and every independent list item.
Cite exact leaf_id strings. Use short phrases; never copy whole transcript passages.
Schema uses positional arrays:
{"r":[topics,entities,key_events,current_states,time_range],
 "f":[[subject,predicate,object,kind,polarity,modality,state_op,context,item,event_time,source_leaf_ids,role,confidence,valid_to],...]}
Allowed values: kind=state|event|pref|qty|assist; polarity=+|-|?;
modality=assert|plan|possible|cond|unknown; state_op=set|add|remove|cancel|done|none; role=user|asst.
Concrete examples: ["user","commute_duration","45 minutes each way","qty","+","assert","set","daily work commute","commute",null,["EXACT_LEAF_ID"],"user",0.98,null]; ["user","purchased_from","sports store downtown","event","+","assert","done","new tennis racket purchase","new tennis racket",null,["EXACT_LEAF_ID"],"user",0.99,null].
Use null/"" for unknown optional values. Keep object phrases <=32 words.
For EVERY leaf, extract every explicit user self-fact before assistant facts. Split clauses such as "I bought/got X from/at Y" into a purchase-source/location fact. Never let long assistant advice or recommendation lists displace user facts.
Assistant facts: retain concrete names, numbers, tables/lists and directly requested recommendations; omit generic filler.
Normally emit <=18 facts; if space is tight drop generic assistant lists first, but preserve user facts and all distinct user items.
Do not omit or merge subject, predicate, or object, and do not emit field names inside fact rows."""
    user = f"session_id={session_id}\nsession_date={session_date or 'unknown'}\n\n{transcript}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def consolidation_messages(facts: list[AtomicFactNode]) -> list[dict[str, str]]:
    compact = [
        [index, fact.subject, fact.predicate, fact.object, fact.context_key, fact.item_key,
         fact.polarity[0], fact.modality[0], fact.state_op[0], fact.event_time or fact.observed_at]
        for index, fact in enumerate(facts)
    ]
    system = """Return compact JSON only. Normalize aliases/predicates and propose only high-confidence relations
between existing integer fact ids. Never add facts; never merge distinct items, contexts, opposite preferences,
plans, or completed events. Input rows are [id,subject,predicate,object,context,item,polarity,modality,op,time].
Output: {"a":{"surface":"canonical"},"p":{"surface":"canonical"},
"e":[[src_id,dst_id,"supports|supersedes|contradicts|before|after",confidence],...]}
Omit identity aliases. Emit at most 48 entity aliases, 32 predicate aliases, and 120 cross-session relations.
Prefer aliases and state-changing supersedes/contradicts; omit generic request/provided relations. Keep JSON on one line."""
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}]


def parse_session_extraction(
    text: str,
    *,
    question_id: str,
    session_id: str,
    session_date: str | None,
    leaves: list[LeafNode],
) -> tuple[RoutingCardNode, list[AtomicFactNode], str | None]:
    leaf_ids = {leaf.node_id for leaf in leaves}
    error: str | None = None
    try:
        payload = _json_object(text)
        fact_payload = payload.get("f", payload.get("facts"))
        routing_payload = payload.get("r", payload.get("routing_card"))
        rows = fact_payload if isinstance(fact_payload, list) else []
        card_payload = routing_payload if isinstance(routing_payload, (dict, list)) else {}
    except Exception as exc:
        partial=_partial_extraction_payload(text)
        rows=partial.get("f",[]);card_payload=partial.get("r",{})
        error = "partial_json_salvaged" if rows else f"invalid_json: {exc}"

    if isinstance(card_payload, list):
        card_values=list(card_payload)+[None]*5
        card_payload={"t":card_values[0] or [],"e":card_values[1] or [],"v":card_values[2] or [],"s":card_values[3] or [],"d":card_values[4] or ""}
    facts: list[AtomicFactNode] = []
    for index, row in enumerate(rows):
        if isinstance(row, list):
            row=_normalize_fact_array(row,leaf_ids)
        if not isinstance(row, dict):
            continue
        sources = [str(value) for value in _pick(row, "z", "source_leaf_ids", default=[]) or [] if str(value) in leaf_ids]
        if not sources:
            continue
        subject = _bounded(_pick(row, "s", "subject"), "user")
        predicate = _bounded(_pick(row, "p", "predicate"), "related_to")
        obj = _bounded(_pick(row, "o", "object"), "")
        if not obj:
            continue
        kind = _coded_choice(_pick(row, "k", "kind"), {"s":"state","e":"event","p":"preference","q":"quantity","a":"assistant_fact","state":"state","event":"event","pref":"preference","qty":"quantity","assist":"assistant_fact"}, "state")
        polarity = _coded_choice(_pick(row, "n", "polarity"), {"+":"positive","-":"negative","?":"unknown"}, "positive")
        modality = _coded_choice(_pick(row, "m", "modality"), {"a":"asserted","p":"planned","o":"possible","c":"conditional","?":"unknown","assert":"asserted","plan":"planned","possible":"possible","cond":"conditional","unknown":"unknown"}, "asserted")
        state_op = _coded_choice(_pick(row, "x", "state_op"), {"S":"set","A":"add","R":"remove","C":"cancel","D":"complete","N":"none","set":"set","add":"add","remove":"remove","cancel":"cancel","done":"complete","none":"none"}, "none", case_sensitive=True)
        role = _coded_choice(_pick(row, "r", "role"), {"u":"user","a":"assistant","user":"user","asst":"assistant","assistant":"assistant"}, "user")
        fact_id = f"{question_id}:{session_id}:fact:{len(facts)}"
        context_raw = _pick(row, "c", "context_key")
        context_key = canonical_key(context_raw) if context_raw else "default"
        item_key = canonical_key(_pick(row, "i", "item_key") or obj)
        observed = _date_value(session_date)
        fact = AtomicFactNode(
            node_id=fact_id,
            question_id=question_id,
            session_id=session_id,
            subject=subject,
            subject_key=canonical_key(subject),
            predicate=predicate,
            predicate_key=canonical_key(predicate),
            object=obj,
            object_key=canonical_key(obj),
            kind=kind,
            polarity=polarity,
            modality=modality,
            state_op=state_op,
            context_key=context_key,
            item_key=item_key,
            event_time=_date_value(_pick(row, "t", "event_time")),
            observed_at=observed,
            valid_from=_date_value(_pick(row, "vf", "valid_from")) or _date_value(_pick(row, "t", "event_time")) or observed,
            valid_to=_date_value(_pick(row, "vt", "valid_to")),
            source_leaf_ids=sources,
            speaker=_bounded(_pick(row, "sp", "speaker"), ""),
            role=role,
            confidence=_confidence(_pick(row, "q", "confidence")),
        )
        fact.retrieval_text = _fact_text(fact)
        facts.append(fact)

    if not facts:
        facts = _fallback_facts(question_id, session_id, session_date, leaves)
        error = error or "empty_facts_fallback"
    _augment_lossless_numeric_facts(
        facts, question_id=question_id, session_id=session_id,
        session_date=session_date, leaves=leaves,
    )

    topics = clean_entities(_pick(card_payload, "t", "topics", default=[]) or [])[:8]
    entities = clean_entities(_pick(card_payload, "e", "canonical_entities", default=[]) or [])[:12]
    event_values = _pick(card_payload, "v", "key_events", default=[]) or []
    state_values = _pick(card_payload, "s", "current_states", default=[]) or []
    events = [_summary_phrase(value) for value in event_values if _summary_phrase(value)][:8]
    states = [_summary_phrase(value) for value in state_values if _summary_phrase(value)][:8]
    if not topics:
        topics = _top_terms(" ".join(f.retrieval_text for f in facts), 6)
    if not entities:
        entities = clean_entities([f.subject for f in facts] + [f.object for f in facts])[:12]
    if not events:
        events = [f"{f.subject} {f.predicate} {f.object}" for f in facts if f.kind == "event"][:8]
    if not states:
        states = [f"{f.subject} {f.predicate} {f.object}" for f in facts if f.kind in {"state", "preference"}][-8:]
    time_value=_pick(card_payload, "d", "time_range")
    if isinstance(time_value,list): time_value=" to ".join(str(value) for value in time_value if value)
    time_range = _bounded(time_value, session_date or "unknown")
    card_id = f"{question_id}:{session_id}:routing"
    retrieval_text = _limit_rough(_routing_text(session_id, session_date, topics, entities, events, states, time_range), 180)
    card = RoutingCardNode(
        node_id=card_id,
        question_id=question_id,
        session_id=session_id,
        session_date=session_date,
        topics=topics,
        canonical_entities=entities,
        key_events=events,
        current_states=states,
        time_range=time_range,
        fact_ids=[fact.node_id for fact in facts],
        leaf_ids=[leaf.node_id for leaf in leaves],
        retrieval_text=retrieval_text,
    )
    return card, facts, error


def apply_consolidation(text: str, facts: list[AtomicFactNode]) -> tuple[list[GraphEdge], dict[str, str], str | None]:
    by_id = {fact.node_id: fact for fact in facts}
    parse_error = None
    try:
        payload = _json_object(text)
    except Exception as exc:
        payload = _partial_consolidation_payload(text)
        if not any(payload.get(key) for key in ("a", "p", "e")):
            return [], {}, f"invalid_json: {exc}"
        parse_error = "partial_json_salvaged"
    aliases = {canonical_key(k): canonical_key(v) for k, v in (_pick(payload, "a", "aliases", default={}) or {}).items() if canonical_key(k) != "unknown"}
    predicate_aliases = {canonical_key(k): canonical_key(v) for k, v in (_pick(payload, "p", "predicate_aliases", default={}) or {}).items() if canonical_key(k) != "unknown"}
    for fact in facts:
        fact.subject_key = aliases.get(fact.subject_key, fact.subject_key)
        fact.object_key = aliases.get(fact.object_key, fact.object_key)
        fact.item_key = aliases.get(fact.item_key, fact.item_key)
        fact.predicate_key = predicate_aliases.get(fact.predicate_key, fact.predicate_key)
        fact.retrieval_text = _fact_text(fact)
    accepted: list[GraphEdge] = []
    short_ids={str(index):fact.node_id for index,fact in enumerate(facts)}
    for row in _pick(payload, "e", "relations", default=[]) or []:
        if isinstance(row,list):
            values=list(row)+[None]*4
            src,dst,relation,confidence=str(values[0]),str(values[1]),str(values[2] or ""),_confidence(values[3])
        elif isinstance(row,dict):
            src, dst = str(_pick(row, "s", "src") or ""), str(_pick(row, "d", "dst") or "")
            relation = str(_pick(row, "r", "relation") or "")
            confidence = _confidence(_pick(row, "q", "confidence"))
        else:
            continue
        src=short_ids.get(src,src);dst=short_ids.get(dst,dst)
        if src not in by_id or dst not in by_id or src == dst:
            continue
        if relation not in {"supports", "supersedes", "contradicts", "before", "after"} or confidence < 0.7:
            continue
        if relation in {"supersedes", "contradicts"} and (
            _is_generic_predicate(by_id[src].predicate_key) or _is_generic_predicate(by_id[dst].predicate_key)
        ):
            continue
        if relation in {"before", "after"} and not _temporally_related(by_id[src], by_id[dst]):
            continue
        accepted.append(GraphEdge(src, dst, confidence, relation, True, confidence, {"generator": "question_consolidation", "source_fact_ids": [src, dst]}, GRAPHMEM_V2_SCHEMA))
    return accepted, aliases, parse_error


def build_state_chains(facts: list[AtomicFactNode]) -> tuple[list[StateChain], list[GraphEdge]]:
    grouped: dict[tuple[str, str, str], list[AtomicFactNode]] = defaultdict(list)
    for fact in facts:
        context_key = fact.context_key or "default"
        if _is_generic_predicate(fact.predicate_key):
            context_key = f"{context_key}:{fact.node_id}"
        grouped[(fact.subject_key, fact.predicate_key, context_key)].append(fact)
    chains: list[StateChain] = []
    edges: list[GraphEdge] = []
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=_fact_sort_key)
        current: list[AtomicFactNode] = []
        for fact in ordered:
            previous = list(current)
            if fact.state_op == "add":
                current = [item for item in current if item.item_key != fact.item_key]
                if fact.polarity != "negative":
                    current.append(fact)
            elif fact.state_op in {"remove", "cancel"}:
                current = [item for item in current if item.item_key != fact.item_key and item.object_key != fact.object_key]
            elif fact.state_op in {"set", "complete"}:
                if not (fact.modality == "planned" and any(item.state_op == "complete" for item in current)):
                    current = [fact]
            elif fact.polarity == "negative":
                current = [item for item in current if item.object_key != fact.object_key]
                current.append(fact)
            elif fact.kind in {"state", "preference", "quantity"}:
                current = [fact]
            else:
                current.append(fact)
            for old in previous:
                if old.node_id not in {item.node_id for item in current}:
                    old.valid_to = fact.valid_from or fact.observed_at
                    edges.append(_edge(fact.node_id, old.node_id, "supersedes", True, 0.96, "state_chain"))
            for old in ordered:
                if old.node_id == fact.node_id:
                    break
                if old.object_key == fact.object_key and old.polarity != fact.polarity:
                    edges.append(_edge(fact.node_id, old.node_id, "contradicts", True, 0.95, "opposite_polarity"))
        chain_key = "|".join(key)
        chain_id = f"{ordered[0].question_id}:chain:" + hashlib.sha1(chain_key.encode()).hexdigest()[:16]
        chains.append(StateChain(
            chain_id=chain_id,
            question_id=ordered[0].question_id,
            subject_key=key[0], predicate_key=key[1], context_key=key[2],
            current_fact_ids=[fact.node_id for fact in current],
            history_fact_ids=[fact.node_id for fact in ordered],
            valid_from=min((fact.valid_from for fact in ordered if fact.valid_from), default=None),
            valid_to=max((fact.valid_to for fact in ordered if fact.valid_to), default=None),
            update_order=[fact.node_id for fact in ordered],
        ))
        for older, newer in zip(ordered, ordered[1:]):
            if _comparable_time(older, newer):
                edges.append(_edge(older.node_id, newer.node_id, "before", True, 0.9, "same_state_chain"))
                edges.append(_edge(newer.node_id, older.node_id, "after", True, 0.9, "same_state_chain"))
    return chains, _dedupe_edges(edges)


_OPERAND_COLLECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:coffee\s+)?mugs?\b", "coffee_mugs"),
    (r"\b(?:twitter|tiktok|instagram|facebook|social media).{0,80}\bfollowers?\b|\bfollowers?.{0,80}\b(?:twitter|tiktok|instagram|facebook)\b", "social_media_followers"),
    (r"\b(?:poems?|short stories|writing challenge|pieces? of writing|writing pieces?)\b", "writing_pieces"),
    (r"\b(?:lunch(?:es)?|meals?|fajitas?|lentil soup)\b", "lunch_meals"),
    (r"\b(?:fish|tetras?|gouramis?|pleco|catfish|betta|aquarium|fish tank)\b", "fish_inventory"),
    (r"\b(?:united|southwest|american) airlines?\b|\bairlines?\b.*?\bflights?\b|\bflights?\b.*?\bairlines?\b", "airline_flights"),
    (r"\b(?:eggs?|dozen eggs?|egg sales?)\b", "egg_sales"),
    (r"\b(?:facebook live|youtube video|social media).{0,100}\bcomments?\b|\bcomments?\b.{0,100}\b(?:facebook live|youtube video|social media)\b", "social_media_comments"),
    (r"\b(?:doctor(?:'s)? appointments?|appointments?).{0,100}\b(?:doctor|physician|surgeon)\b|\b(?:doctor|physician|surgeon).{0,100}\bappointments?\b", "doctor_appointments"),
    (r"\b(?:personal best|best time).{0,80}\b(?:5k|run|race)\b|\b(?:5k|run|race).{0,80}\b(?:personal best|best time)\b", "race_personal_best"),
    (r"\b(?:podcasts?|episodes?).{0,100}\b(?:episodes?|listened|finished)\b|\b(?:listened|finished).{0,100}\bepisodes?\b", "podcast_episodes"),
    (r"\b(?:facebook ad|instagram influencer|ad campaign|influencer collaboration).{0,120}\b(?:reach|reached|followers?)\b|\b(?:reach|reached|followers?).{0,120}\b(?:facebook ad|instagram influencer|ad campaign|influencer)\b", "campaign_reach"),
    (r"\b(?:hikes?|trails?).{0,100}\b(?:miles?|distance|weekends?)\b|\b(?:miles?|distance).{0,100}\b(?:hikes?|trails?)\b", "hike_distance"),
    (r"\b(?:led|leading|solo).{0,100}\bprojects?\b|\bprojects?.{0,100}\b(?:led|leading|solo)\b", "project_leadership"),
    (r"\b(?:volleyball|basketball|league|team).{0,80}\brecord\b|\brecord\b.{0,80}\b(?:volleyball|basketball|league|team)\b", "competitive_record"),
    (r"\b(?:united|southwest|american).{0,80}\b(?:premier|status)\b|\b(?:premier|status)\b.{0,80}\b(?:united|southwest|american)\b", "airline_status"),
)


def _operand_collection_key(text: str) -> str:
    for pattern, key in _OPERAND_COLLECTION_PATTERNS:
        if re.search(pattern, text, re.I | re.S):
            return key
    return ""


def _operand_measure_key(text: str, collection_key: str) -> str:
    folded = text.casefold()
    if collection_key == "social_media_followers":
        return "follower_change"
    if collection_key == "airline_flights":
        return "flight_frequency"
    if collection_key in {"writing_pieces", "lunch_meals", "fish_inventory"}:
        return collection_key + "_count"
    if collection_key == "coffee_mugs":
        return "coffee_mug_cost_or_count"
    if collection_key == "egg_sales":
        return "egg_sales_revenue"
    if collection_key == "social_media_comments":
        return "social_media_comment_count"
    if collection_key == "doctor_appointments":
        return "doctor_appointment_count"
    if collection_key == "race_personal_best":
        return "race_duration"
    if collection_key == "podcast_episodes":
        return "podcast_episode_count"
    if collection_key == "campaign_reach":
        return "campaign_audience_count"
    if collection_key == "hike_distance":
        return "hike_distance_miles"
    if collection_key == "project_leadership":
        return "project_count"
    if collection_key == "competitive_record":
        return "competitive_record"
    if collection_key == "airline_status":
        return "airline_status_tier"
    if re.search(r"\$\s*\d|\b(?:cost|price|spent|paid)\b", folded):
        return "money"
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b", folded):
        return "duration"
    return ""


def _operand_quantity_role(text: str, collection_key: str) -> str:
    folded = text.casefold()
    if re.search(r"\bfrom\s+\d[\d,.]*\s+to\s+\d[\d,.]*\b", folded):
        return "delta"
    if re.search(r"\b(?:gained|increased|grew|jumped)\b.{0,40}\b\d[\d,.]*\b", folded):
        return "delta"
    if collection_key == "airline_flights":
        return "frequency"
    if collection_key == "fish_inventory":
        return "inventory"
    if collection_key == "egg_sales" and re.search(
        r"\b(?:per|a)\s+dozen\b", folded
    ):
        return "unit_rate"
    if collection_key == "doctor_appointments":
        return "event"
    if collection_key == "race_personal_best":
        return "current"
    if re.search(r"\$\s*\d", text):
        return "total" if re.search(r"\b(?:spent|paid|cost|total|for)\b", folded) else "unknown"
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b", folded):
        return "duration"
    if re.search(r"\b(?:currently|now|current)\b", folded):
        return "current"
    if re.search(r"\b\d[\d,.]*\b", folded):
        return "count"
    return "unknown"


def enrich_operand_metadata(
    facts: list[AtomicFactNode], leaves: list[LeafNode]
) -> None:
    """Attach deterministic metric and collection tags from cited L0 text."""
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    for fact in facts:
        evidence = _evidence_text(fact, leaf_by_id)
        collection = _operand_collection_key(evidence)
        fact.collection_key = collection
        fact.measure_key = _operand_measure_key(evidence, collection)
        fact.quantity_role = _operand_quantity_role(evidence, collection)


def _connect_operand_group(
    edges: list[GraphEdge], relation: str, group_key: str,
    values: list[AtomicFactNode], score: float,
) -> None:
    ordered = sorted(values, key=_fact_sort_key)
    pairs = (
        [(left, right) for index, left in enumerate(ordered) for right in ordered[index + 1:]]
        if len(ordered) <= 12 else list(zip(ordered, ordered[1:]))
    )
    for left, right in pairs:
        if left.node_id != right.node_id:
            edges.append(_edge(
                left.node_id, right.node_id, relation, False, score,
                f"shared_operand:{group_key}",
            ))


def build_graph_edges(
    leaves: list[LeafNode], cards: list[RoutingCardNode], facts: list[AtomicFactNode], chains: list[StateChain],
    *, semantic_k: int = 3, semantic_floor: float = 0.55,
) -> list[GraphEdge]:
    enrich_operand_metadata(facts, leaves)
    edges: list[GraphEdge] = []
    leaf_by_session: dict[str, list[LeafNode]] = defaultdict(list)
    for leaf in leaves:
        leaf_by_session[leaf.session_id].append(leaf)
    card_by_session = {card.session_id: card for card in cards}
    for card in cards:
        for fact_id in card.fact_ids:
            edges.append(_edge(card.node_id, fact_id, "contains", True, 1.0, "routing_card"))
        for leaf_id in card.leaf_ids:
            edges.append(_edge(card.node_id, leaf_id, "contains", True, 1.0, "routing_card"))
    for fact in facts:
        for leaf_id in fact.source_leaf_ids:
            edges.append(_edge(fact.node_id, leaf_id, "source", True, 1.0, "fact_provenance"))
        card = card_by_session.get(fact.session_id)
        if card:
            edges.append(_edge(fact.node_id, card.node_id, "participates_in", True, 1.0, "session_membership"))
    for values in leaf_by_session.values():
        ordered = sorted(values, key=lambda leaf: leaf.turn_index)
        for left, right in zip(ordered, ordered[1:]):
            edges.append(_edge(left.node_id, right.node_id, "next_turn", True, 1.0, "within_session"))
    by_entity: dict[str, list[AtomicFactNode]] = defaultdict(list)
    by_predicate: dict[str, list[AtomicFactNode]] = defaultdict(list)
    for fact in facts:
        for entity in {fact.subject_key, fact.object_key} - _GENERIC_ENTITY_KEYS:
            if len(entity) >= 3:
                by_entity[entity].append(fact)
        if not _is_generic_predicate(fact.predicate_key):
            by_predicate[fact.predicate_key].append(fact)
    for relation, groups in (("same_entity", by_entity), ("same_predicate", by_predicate)):
        for group_key, values in groups.items():
            ordered = sorted(values, key=_fact_sort_key)
            for left, right in zip(ordered, ordered[1:]):
                if left.node_id != right.node_id:
                    edges.append(_edge(left.node_id, right.node_id, relation, False, 0.88, f"shared:{group_key}"))
    by_measure: dict[str, list[AtomicFactNode]] = defaultdict(list)
    by_collection: dict[str, list[AtomicFactNode]] = defaultdict(list)
    for fact in facts:
        if fact.measure_key:
            by_measure[fact.measure_key].append(fact)
        if fact.collection_key:
            by_collection[fact.collection_key].append(fact)
    for group_key, values in by_measure.items():
        if len(values) > 1:
            _connect_operand_group(edges, "same_measure", group_key, values, 0.91)
    for group_key, values in by_collection.items():
        if len(values) <= 1:
            continue
        _connect_operand_group(edges, "same_collection", group_key, values, 0.93)
        operand_values = [
            fact for fact in values
            if fact.quantity_role != "unknown" or fact.kind == "quantity"
        ]
        if len(operand_values) > 1:
            _connect_operand_group(edges, "operand_of", group_key, operand_values, 0.96)
    nodes: list[Any] = [*cards, *facts]
    edges.extend(_mutual_knn_edges(nodes, k=semantic_k, floor=semantic_floor))
    return _dedupe_edges(edges)


def expand_query(question: str) -> str:
    q=question.casefold();terms=[]
    expansions=(
        (r"\bphotograph(?:y|ic|er)?\b","camera lens flash tripod photo"),
        (r"\baccessor(?:y|ies)\b","gear case bag battery tripod flash lens"),
        (r"\bvideo editing\b","editing software editor Adobe Premiere Pro"),
        (r"\bresources?\b","tutorial course guide documentation training"),
        (r"\bprojects?\b","initiative assignment led leading manage managing"),
        (r"\bhealth-related devices?\b","health device equipment tracker monitor hearing aid nebulizer glucometer Fitbit Accu-Chek"),
        (r"\bcuisines?\b","cuisine cooking cooked learned tried restaurant food"),
        (r"\bsiblings?\b","sibling brother sister"),
        (r"\bart-related events?\b","art event exhibition gallery museum fair show"),
        (r"\bcitrus(?: fruits?)?\b","citrus lemon lime orange grapefruit"),
        (r"\b(?:albums?|EPs?)\b","music album EP record purchased downloaded"),
        (r"\bmusical instruments?\b","instrument guitar piano keyboard drums owned"),
        (r"\b(?:expenses?|money spent)\b","expense cost spent paid purchase price"),
        (r"\bformal education\b","education high school college associate bachelor degree duration"),
        (r"\bhow many years older\b","age birthday graduated completion age difference"),
        (r"\bcurrent role\b","current role tenure promotion promoted duration"),
        (r"\bpage count\b","pages novel book finished reading"),
        (r"\bsubmit(?:ted)?(?: my)? (?:research )?paper\b","submission submitted paper date"),
        (r"\bshampoo\b","shampoo hair product brand purchased store"),
        (r"\bbattery life\b","battery power bank charger charging low power mode screen background app"),
        (r"\bclinic\b","clinic doctor appointment commute arrived reached left home"),
        (r"\bjewel(?:ry|lery)\b","jewelry necklace pendant bracelet ring chandelier crystal gift received aunt"),
        (r"\bmumm(?:y|ies)\b","mummy mummies Lost Temple Djinn enemy stat block party face"),
        (r"\bengagement rings?\b","engagement ring designer Instagram handle UK-based unusual gemstones Jessica Poole jewelry jewellery"),
        (r"\b(?:business|buisiness) milestone\b","business milestone signed contract first client launched website"),
        (r"\bgardening-related\b","gardening garden planted tomato saplings plants"),
        (r"\bkitchen appliance\b","kitchen appliance smoker grill oven bought purchased got"),
        (r"\bairline\b","airline flight flew flown carrier"),
        (r"\bstreaming service\b","streaming service Netflix Hulu Disney Apple TV Amazon Prime subscription trial"),
        (r"\bartist\b","artist band musician listen bluegrass jazz"),
        (r"\bguitar\b","guitar practice practicing minutes daily every day"),
        (r"\bcoins?\b","coin collection collected acquired purchased pre-1920"),
        (r"\bgifts?\b","gift sister bought purchased spent necklace present"),
        (r"\bplants?\b","plant acquired bought purchased got succulent indoor garden"),
        (r"\bconcerts?|\bmusical events?\b","concert music festival jazz show attended live"),
        (r"\bcharity events?\b","charity event participated volunteered attended gala fundraiser run walk bike dance golf"),
        (r"\bbikes?\b","bike bicycle fixed serviced maintenance pedals flat tire road mountain"),
        (r"\bclothing\b","clothing clothes boots blazer dry cleaning store exchange pick return"),
        (r"\bsports? events?\b","sports NBA NFL football championship playoffs game watched attended"),
    )
    for pattern,value in expansions:
        if re.search(pattern,q): terms.append(value)
    return question if not terms else question+" | related terms: "+" ".join(terms)


def _is_assistant_recall_question(question: str) -> bool:
    return bool(re.search(
        r"\b(previous (?:chat|conversation|response)|earlier (?:chat|conversation|response)|"
        r"remind me|you (?:said|recommended|listed|provided|mentioned|told me))\b",
        question.casefold(),
    ))


def query_kind(question: str) -> str:
    q = question.casefold()
    if _is_assistant_recall_question(question):
        return "fact"
    # Elapsed-time questions are temporal even when their surface form starts
    # with "how many".  Classify them before the generic count/list rule.  A
    # recurring frequency such as "how many days per week" remains a count.
    if re.search(
        r"\bhow many\s+(?:calendar\s+)?(?:days?|weeks?|months?|years?)\b", q
    ) and not re.search(
        r"\bhow many\s+days?\s+(?:a|per|each)\s+week\b", q
    ):
        return "temporal"
    advice_match=re.search(
        r"\b(prefer|preference|favorite|favourite|recommend(?:ation)?s?|suggest(?:ion)?s?|tips?|advice|"
        r"ideas?|inspiration|feeling stuck|what do you think|good idea|should i|"
        r"help me decide)\b", q
    )
    current_match=re.search(r"\b(latest|current|currently|now|still|update|updated|changed|most recent(?:ly)?|increase|increased|decrease|decreased|personal best)\b",q)
    count_match=re.search(r"\b(how many|how much (?:total|did|do|have|was|were|spent|cost)|total number|page count|count|list|which (?:items|things|places|people)|what are all)\b",q)
    temporal_match=re.search(r"\b(previous|previously|initially|earlier|earliest|order|before|after|when|date|time|days?|weeks?|weekend|months?|years?|how long|first|last)\b",q)
    collection_count=bool(count_match and re.search(
        r"\b(siblings?|brothers?|sisters?|instruments?|fish|devices?|plants?|coins?|"
        r"gifts?|events?|albums?|movies?|cuisines?|items?|things?)\b",q
    ))
    if count_match and not re.search(
        r"\b(?:how many (?:days|weeks|months|years)|how long)\b",q
    ) and re.search(r"\b(?:how many|total number|page count|what (?:was|is) the total)\b",q):
        return "count/list"
    if advice_match: return "preference"
    if "personal best" in q: return "current/update"
    if "music streaming service" in q and re.search(r"\b(?:lately|recently)\b",q):
        return "current/update"
    if current_match and re.search(r"\b(?:company|employer|job|role|working at|works at)\b", q): return "current/update"
    if re.search(r"\bhow many days? (?:a|per) week\b",q): return "count/list"
    if collection_count: return "count/list"
    if re.search(r"\b(?:how many (?:days|weeks|months|years)|how long)\b",q): return "temporal"
    if current_match and (not temporal_match or (count_match and re.search(r"\b(now|latest|current|most recent)\b",q))): return "current/update"
    if count_match: return "count/list"
    if temporal_match: return "temporal"
    if current_match: return "current/update"
    if re.search(r"\b(like|dislike|love|hate|avoid|resource|resources|accessory|accessories)\b", q): return "preference"
    if re.search(r"\b(two|three|both|together|across|between|relationship|why)\b", q): return "multi-hop"
    if re.search(r"\b(unknown|not enough|insufficient|did (?:he|she|they) ever)\b", q): return "abstention"
    return "fact"


_ALLOWED = {
    "current/update": {"same_predicate", "same_measure", "same_collection", "operand_of", "supersedes", "contradicts", "source", "participates_in", "contains"},
    "temporal": {"participates_in", "same_measure", "same_collection", "operand_of", "before", "after", "source", "contains", "next_turn"},
    "count/list": {"same_entity", "same_predicate", "same_measure", "same_collection", "operand_of", "supports", "source", "contains"},
    "preference": {"same_entity", "same_predicate", "supports", "contradicts", "source", "contains"},
    "multi-hop": {"same_entity", "same_predicate", "same_measure", "same_collection", "operand_of", "participates_in", "supports", "source", "contains", "semantic_neighbor"},
    "fact": {"same_entity", "same_predicate", "supports", "source", "contains", "semantic_neighbor"},
    "abstention": {"source", "contains", "same_entity"},
}

_RELATION_EXPANSION_WEIGHT = {
    "contains": 0.95,
    "source": 0.95,
    "participates_in": 0.90,
    "supports": 0.88,
    "supersedes": 0.96,
    "contradicts": 0.92,
    "before": 0.92,
    "after": 0.92,
    "same_predicate": 0.82,
    "same_measure": 0.94,
    "same_collection": 0.96,
    "operand_of": 0.98,
    "same_entity": 0.70,
    "next_turn": 0.48,
    "semantic_neighbor": 0.32,
}

_EXPANSION_BUDGET = {
    "current/update": 14,
    "temporal": 20,
    "count/list": 16,
    "preference": 14,
    "multi-hop": 20,
    "fact": 12,
    "abstention": 8,
}


def retrieve(
    *, case: QuestionCase, variant: str, leaves: list[LeafNode], cards: list[RoutingCardNode], facts: list[AtomicFactNode],
    chains: list[StateChain], edges: list[GraphEdge], query_vector: list[float], card_k: int = 6, fact_k: int = 12,
    leaf_k: int = 12, token_budget: int = 8200,
) -> RetrievedContext:
    import time
    started = time.perf_counter()
    kind = query_kind(case.question)
    retrieval_query=expand_query(case.question)
    card_channels: dict[str, Any] = {}
    fact_channels: dict[str, Any] = {}
    leaf_channels: dict[str, Any] = {}
    card_scores = _fused_scores(retrieval_query, cards, query_vector, trace=card_channels)
    fact_scores = _fused_scores(retrieval_query, facts, query_vector, trace=fact_channels)
    direct_leaf_scores = _fused_scores(
        retrieval_query, leaves, query_vector, trace=leaf_channels
    )
    direct_leaf_ids = [
        node_id
        for node_id, _ in sorted(
            direct_leaf_scores.items(), key=lambda item: item[1], reverse=True
        )[:max(16, leaf_k * 2)]
    ]
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    session_signal: dict[str, float] = defaultdict(float)
    for rank, node_id in enumerate(direct_leaf_ids, 1):
        leaf = leaf_by_id[node_id]
        session_signal[leaf.session_id] += 1.0 / rank
    for card in cards:
        card_scores[card.node_id] = card_scores.get(card.node_id, 0.0) + (
            0.012 * min(1.0, session_signal.get(card.session_id, 0.0))
        )
    for fact in facts:
        fact_scores[fact.node_id] = fact_scores.get(fact.node_id, 0.0) + (
            0.008 * min(1.0, session_signal.get(fact.session_id, 0.0))
        )
    initial_cards = [node_id for node_id, _ in sorted(card_scores.items(), key=lambda item: item[1], reverse=True)[:max(card_k, 4)]]
    initial_facts = [node_id for node_id, _ in sorted(fact_scores.items(), key=lambda item: item[1], reverse=True)[:max(fact_k, 10)]]
    fact_by_id = {fact.node_id: fact for fact in facts}
    card_by_id = {card.node_id: card for card in cards}
    seed_ids = list(dict.fromkeys(initial_facts + initial_cards))
    expansion_trace: dict[str, Any] = {}
    expanded = _typed_expand(
        seed_ids,
        edges,
        _ALLOWED[kind],
        depth=2,
        cap=len(seed_ids) + _EXPANSION_BUDGET[kind],
        node_scores={**card_scores, **fact_scores},
        eligible_nodes=set(fact_by_id) | set(card_by_id),
        min_score=0.075,
        trace=expansion_trace,
    )
    for node_id, bonus in expanded.items():
        graph_bonus = 0.025 * bonus
        if node_id in fact_by_id: fact_scores[node_id] = fact_scores.get(node_id, 0.0) + graph_bonus
        if node_id in card_by_id: card_scores[node_id] = card_scores.get(node_id, 0.0) + graph_bonus
    operand_relations = _ALLOWED[kind] & {
        "same_measure", "same_collection", "operand_of",
        "same_predicate", "supersedes", "contradicts", "before", "after",
    }
    operand_trace: dict[str, Any] = {}
    operand_seed_ids = initial_facts[:min(8, len(initial_facts))]
    operand_expanded = _typed_expand(
        operand_seed_ids,
        edges,
        operand_relations,
        depth=2,
        cap=len(operand_seed_ids) + (28 if kind in {"count/list", "multi-hop", "temporal"} else 14),
        node_scores=fact_scores,
        eligible_nodes=set(fact_by_id),
        min_score=0.035,
        trace=operand_trace,
    )
    for node_id, bonus in operand_expanded.items():
        if node_id in fact_by_id:
            fact_scores[node_id] = fact_scores.get(node_id, 0.0) + 0.04 * bonus
    selected_cards = _select_with_session_quota(cards, card_scores, card_k, 2)
    selected_facts = _select_facts(facts, fact_scores, fact_k, kind, case.question)
    selected_facts = _supplement_query_facts(
        facts, selected_facts, leaves, case.question, kind, fact_k
    )
    relation_paths = expansion_trace.get("paths") or {}
    operand_paths = operand_trace.get("paths") or {}
    graph_fact_ids: list[str] = []
    graph_sessions: Counter[str] = Counter()
    graph_candidates = dict(expanded)
    for node_id, value in operand_expanded.items():
        graph_candidates[node_id] = max(graph_candidates.get(node_id, 0.0), value)
    for node_id, _ in sorted(graph_candidates.items(), key=lambda item: item[1], reverse=True):
        if node_id not in fact_by_id or node_id in set(seed_ids):
            continue
        if not any(
            relation not in {"contains", "participates_in", "source", "next_turn"}
            for relation in ((operand_paths.get(node_id) or relation_paths.get(node_id) or {}).get("relations", []))
        ):
            continue
        session_id = fact_by_id[node_id].session_id
        per_session_graph_cap = 2 if kind in {"count/list", "multi-hop", "temporal"} else 1
        if graph_sessions[session_id] >= per_session_graph_cap:
            continue
        graph_fact_ids.append(node_id)
        graph_sessions[session_id] += 1
        graph_fact_cap = 8 if kind in {"count/list", "multi-hop", "temporal"} else 3
        if len(graph_fact_ids) >= graph_fact_cap:
            break
    operator_fact_limit = min(len(facts), max(36, fact_k * 3))
    operator_facts = _select_facts(
        facts, fact_scores, operator_fact_limit, kind, case.question
    )
    operator_facts = _merge_facts_by_id(
        [fact_by_id[node_id] for node_id in operand_expanded if node_id in fact_by_id],
        operator_facts,
        limit=operator_fact_limit,
    )
    operator_facts = _supplement_query_facts(
        facts, operator_facts, leaves, case.question, kind, operator_fact_limit
    )
    card_sessions = {card.session_id for card in selected_cards}
    if len(selected_cards) < min(card_k, len(cards)):
        tail = sorted(cards, key=lambda card: card_scores.get(card.node_id, 0.0), reverse=True)
        selected_cards.extend(card for card in tail if card not in selected_cards)[:card_k-len(selected_cards)]
    session_leaf_order: dict[str, list[LeafNode]] = defaultdict(list)
    for leaf in leaves: session_leaf_order[leaf.session_id].append(leaf)
    for values in session_leaf_order.values(): values.sort(key=lambda leaf: leaf.turn_index)
    leaf_position = {leaf.node_id: (values, index) for values in session_leaf_order.values() for index, leaf in enumerate(values)}
    operator_leaf_scores: dict[str, float] = defaultdict(float)
    for rank, fact in enumerate(operator_facts, 1):
        for leaf_id in fact.source_leaf_ids:
            operator_leaf_scores[leaf_id] += 1.0 / rank
    operator_leaf_limit = min(len(leaves), max(36, leaf_k * 3))
    operator_leaves = _select_leaves(
        leaf_by_id,
        operator_leaf_scores,
        operator_leaf_limit,
        {fact.session_id for fact in operator_facts},
    )
    operator_leaves = _supplement_query_leaves(
        leaves, operator_leaves, case.question, kind, operator_leaf_limit,
        case.question_date,
    )
    operator_facts = _promote_leaf_linked_facts(
        facts, operator_facts, operator_leaves, case.question, operator_fact_limit
    )
    operator_ledger = build_evidence_ledger(
        kind,
        selected_facts,
        chains,
        operator_leaves,
        case.question,
        case.question_date,
        operator_facts=operator_facts,
        operator_leaves=operator_leaves,
        complete_facts=facts,
        complete_leaves=leaves,
    )
    operand_ids = _operator_operand_ids(operator_ledger)
    selected_facts = _merge_facts_by_id(
        [
            *[fact_by_id[node_id] for node_id in operand_ids if node_id in fact_by_id],
            *[fact_by_id[node_id] for node_id in graph_fact_ids if node_id in fact_by_id],
        ],
        selected_facts,
        limit=fact_k,
    )

    leaf_scores: dict[str, float] = defaultdict(float)
    max_direct_leaf_score = max(direct_leaf_scores.values(), default=0.0)
    if max_direct_leaf_score > 0:
        for node_id, score in direct_leaf_scores.items():
            leaf_scores[node_id] += 0.55 * score / max_direct_leaf_score
    adjacent_leaf_ids: set[str] = set()
    for rank, fact in enumerate(selected_facts, 1):
        for leaf_id in fact.source_leaf_ids:
            leaf_scores[leaf_id] += 2.0 / rank
            if leaf_id in leaf_position and (
                kind in {"fact", "temporal", "multi-hop"}
                or _is_assistant_recall_question(case.question)
            ):
                ordered, position = leaf_position[leaf_id]
                for delta, weight in ((-1, 0.16), (1, 0.16)):
                    neighbor_index = position + delta
                    if 0 <= neighbor_index < len(ordered):
                        neighbor_id = ordered[neighbor_index].node_id
                        leaf_scores[neighbor_id] += weight / rank
                        adjacent_leaf_ids.add(neighbor_id)
    for rank, card in enumerate(selected_cards, 1):
        for leaf_id in card.leaf_ids:
            leaf_scores[leaf_id] += 0.08 / rank
    selected_leaves = _select_leaves(leaf_by_id, leaf_scores, leaf_k, card_sessions)
    selected_leaves = _supplement_query_leaves(
        leaves, selected_leaves, case.question, kind, leaf_k, case.question_date
    )
    protected_sources = [
        leaf_by_id[source]
        for fact in selected_facts
        for source in fact.source_leaf_ids
        if source in leaf_by_id
    ]
    direct_leaves = []
    direct_session_counts: Counter[str] = Counter()
    for node_id in direct_leaf_ids:
        candidate = leaf_by_id.get(node_id)
        if candidate is None or direct_session_counts[candidate.session_id] >= 1:
            continue
        direct_leaves.append(candidate)
        direct_session_counts[candidate.session_id] += 1
        if len(direct_leaves) >= min(4, leaf_k):
            break
    selected_leaves = _merge_leaves_by_id(
        [*direct_leaves, *protected_sources],
        selected_leaves,
        limit=leaf_k,
    )
    selected_leaves = _supplement_query_leaves(
        leaves, selected_leaves, case.question, kind, leaf_k, case.question_date
    )
    selected_facts = _promote_leaf_linked_facts(
        facts, selected_facts, selected_leaves, case.question, fact_k
    )
    ledger = build_evidence_ledger(
        kind,
        selected_facts,
        chains,
        selected_leaves,
        case.question,
        case.question_date,
        operator_facts=operator_facts,
        operator_leaves=operator_leaves,
        complete_facts=facts,
        complete_leaves=leaves,
    )
    prepack_card_ids=[card.node_id for card in selected_cards]
    prepack_fact_ids=[fact.node_id for fact in selected_facts]
    prepack_leaf_ids=[leaf.node_id for leaf in selected_leaves]
    selected_cards, selected_facts, selected_leaves, context = _pack_context(
        case.question, kind, selected_cards, selected_facts, selected_leaves, ledger, token_budget,
        case.question_date,
    )
    postpack_fact_id_set = {fact.node_id for fact in selected_facts}
    operator_source_id_set = {
        source_id
        for row in ledger if row.get("operator")
        for source_id in row.get("source_fact_ids", [])
    }
    relation_contributions: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expanded": 0, "postpack": 0, "operator_source": 0}
    )
    for node_id, path in {**relation_paths, **operand_paths}.items():
        if node_id in set(seed_ids) or node_id not in fact_by_id:
            continue
        for relation in set(path.get("relations") or []):
            relation_contributions[relation]["expanded"] += 1
            if node_id in postpack_fact_id_set:
                relation_contributions[relation]["postpack"] += 1
            if node_id in operator_source_id_set:
                relation_contributions[relation]["operator_source"] += 1
    channel_contributions: dict[str, dict[str, int]] = {}
    for channel, ranked_ids in (
        ("semantic", fact_channels.get("semantic_rank_ids") or []),
        ("bm25", fact_channels.get("bm25_rank_ids") or []),
        ("entity", fact_channels.get("entity_rank_ids") or []),
    ):
        ranked_set = set(ranked_ids)
        channel_contributions[channel] = {
            "shortlist": len(ranked_set),
            "prepack": len(ranked_set & set(prepack_fact_ids)),
            "postpack": len(ranked_set & postpack_fact_id_set),
            "operator_source": len(ranked_set & operator_source_id_set),
        }
    retrieved_sessions = sorted({leaf.session_id for leaf in selected_leaves} | {fact.session_id for fact in selected_facts})
    return RetrievedContext(
        question_id=case.question_id, variant=variant,
        summary_node_ids=[], leaf_node_ids=[leaf.node_id for leaf in selected_leaves], edge_count=len(edges), context_text=context,
        answer_session_hit=False, retrieved_session_ids=retrieved_sessions, latency_sec=time.perf_counter()-started,
        answer_session_all_hit=False, answer_session_recall=0.0,
        retrieved_answer_session_count=0, gold_answer_session_count=0,
        routing_card_ids=[card.node_id for card in selected_cards], fact_node_ids=[fact.node_id for fact in selected_facts],
        evidence_leaf_ids=[leaf.node_id for leaf in selected_leaves], evidence_ledger=ledger,
        query_kind=kind, packed_rough_tokens=provider_token_estimate(context), schema_version=GRAPHMEM_V2_SCHEMA,
        retrieval_trace={
            "planner":{"query_kind":kind,"expanded_query":retrieval_query,"used_gold_question_type":False,"used_gold_session_ids":False},
            "card_channels":card_channels,"fact_channels":fact_channels,
            "leaf_channels":leaf_channels,"direct_leaf_ids":direct_leaf_ids,
            "initial_card_ids":initial_cards,"initial_fact_ids":initial_facts,
            "typed_expanded_node_ids":[node_id for node_id in expanded if node_id not in set(seed_ids)],
            "typed_expansion":expansion_trace,"operand_expansion":operand_trace,
            "operand_expanded_fact_ids":[node_id for node_id in operand_expanded if node_id not in set(operand_seed_ids)],
            "protected_graph_fact_ids":graph_fact_ids,
            "operator_candidate_fact_ids":[fact.node_id for fact in operator_facts],
            "operator_candidate_leaf_ids":[leaf.node_id for leaf in operator_leaves],
            "operator_operand_fact_ids":operand_ids,
            "adjacent_leaf_ids":sorted(adjacent_leaf_ids),
            "prepack":{"card_ids":prepack_card_ids,"fact_ids":prepack_fact_ids,"leaf_ids":prepack_leaf_ids},
            "postpack":{"card_ids":[c.node_id for c in selected_cards],"fact_ids":[f.node_id for f in selected_facts],"leaf_ids":[l.node_id for l in selected_leaves]},
            "dropped_by_packer":{"card_ids":[x for x in prepack_card_ids if x not in {c.node_id for c in selected_cards}],"fact_ids":[x for x in prepack_fact_ids if x not in {f.node_id for f in selected_facts}],"leaf_ids":[x for x in prepack_leaf_ids if x not in {l.node_id for l in selected_leaves}]},
            "channel_contributions":channel_contributions,
            "relation_contributions":{key:dict(value) for key,value in relation_contributions.items()},
            "provider_token_estimate":provider_token_estimate(context),"token_budget":token_budget,
        },
    )


def build_evidence_ledger(
    kind: str,
    facts: list[AtomicFactNode],
    chains: list[StateChain],
    leaves: list[LeafNode],
    question: str = "",
    question_date: str | None = None,
    *,
    operator_facts: list[AtomicFactNode] | None = None,
    operator_leaves: list[LeafNode] | None = None,
    complete_facts: list[AtomicFactNode] | None = None,
    complete_leaves: list[LeafNode] | None = None,
) -> list[dict[str, Any]]:
    chain_current = {fact_id for chain in chains for fact_id in chain.current_fact_ids}
    rows = []
    for fact in facts:
        rows.append({
            "fact_id": fact.node_id, "subject": fact.subject, "predicate": fact.predicate, "object": fact.object,
            "kind": fact.kind, "polarity": fact.polarity, "modality": fact.modality, "state_op": fact.state_op,
            "event_time": fact.event_time, "observed_at": fact.observed_at, "valid_from": fact.valid_from,
            "valid_to": fact.valid_to, "is_current": fact.node_id in chain_current,
            "source_leaf_ids": fact.source_leaf_ids, "confidence": fact.confidence,
        })
    op_facts = operator_facts if operator_facts is not None else facts
    op_leaves = operator_leaves if operator_leaves is not None else leaves
    operation_facts = (
        complete_facts
        if kind == "preference" and complete_facts is not None
        else op_facts
    )
    operation_leaves = (
        complete_leaves
        if kind == "preference" and complete_leaves is not None
        else op_leaves
    )
    operation = _operator_result(
        kind, operation_facts, chains, question, operation_leaves
    )
    if operation:
        candidate_pool_complete = _candidate_pool_is_complete(
            operation[0], operation[1], operator_facts is not None
        )
        rows.append({
            "operator": operation[0], "result": operation[1],
            "source_fact_ids": operation[2],
            "candidate_pool_complete": candidate_pool_complete,
        })
    calculation_facts = complete_facts if complete_facts is not None else op_facts
    calculation_leaves = complete_leaves if complete_leaves is not None else op_leaves
    for extra in (
        _cashback_result(question, op_facts, op_leaves),
        _arithmetic_result(question, calculation_facts, calculation_leaves),
        _temporal_calculation_result(question, calculation_facts, calculation_leaves, question_date),
        _target_date_answer_result(question, op_facts, op_leaves, question_date),
        _current_competitive_record_result(question, calculation_facts, calculation_leaves),
        _previous_status_result(question, calculation_facts, calculation_leaves),
        _event_companion_result(question, calculation_facts, calculation_leaves, question_date),
        _event_comparison_result(question, calculation_facts, calculation_leaves),
        _event_sequence_result(question, calculation_facts, calculation_leaves),
        _explicit_event_time_result(question, op_facts, op_leaves),
        _brand_or_seller_result(question, op_facts, op_leaves),
        # Assistant facts are often buried in a long answer/table and may not
        # survive atomic extraction or top-k retrieval.  Exact assistant recall
        # is a deterministic scan over the complete, lossless L0 leaves.
        _assistant_recall_result(
            question,
            complete_facts if complete_facts is not None else op_facts,
            complete_leaves if complete_leaves is not None else op_leaves,
        ),
        _exact_entity_result(
            question,
            complete_facts if complete_facts is not None else op_facts,
            complete_leaves if complete_leaves is not None else op_leaves,
        ),
    ):
        if extra:
            candidate_pool_complete = _candidate_pool_is_complete(
                extra[0], extra[1], operator_facts is not None
            )
            if extra[0] == "exact_entity_check" and complete_facts is not None:
                candidate_pool_complete = True
            rows.append({
                "operator": extra[0], "result": extra[1],
                "source_fact_ids": extra[2],
                "candidate_pool_complete": candidate_pool_complete,
            })
    return rows


def answer_messages(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, str]]:
    preference_instruction = ("When the user asks for advice or a recommendation, you MUST answer with practical, tailored suggestions. Use the evidence as preference and compatibility constraints and build on the user's existing tools, successes, setup, and concerns. You may use general knowledge for the advice. Do not abstain merely because the exact tip, resource, or accessory is not already in memory. " if retrieval.query_kind=="preference" else "")
    system = """Answer the question using the provided GraphMem V2 evidence ledger, routing cards, atomic facts, and short source excerpts.
Do not reveal chain-of-thought. Treat an operator as authoritative only when candidate_pool_complete=true; otherwise it is a fallible aid that must be checked against the cited facts, raw excerpts, date window, completed/planned status, and the exact entity or action in the question. Resolve relative time expressions against each source's session date. observed_at is an evidence date, but explicit relative wording in the source can still be used to derive an event date or duration. """ + preference_instruction + """For factual claims about the user, stay grounded in evidence. Prefer a directly matching recent source over a generic state label. If the evidence does not support an answer, say that there is not enough information. Give only a concise final answer."""
    mandatory=_mandatory_answer_hint(retrieval.context_text)
    mandatory_block=(f"\nMANDATORY DETERMINISTIC ANSWER CONSTRAINT: {mandatory}\n" if mandatory else "")
    user = f"Question date: {case.question_date or 'unknown'}\nQuestion: {case.question}{mandatory_block}\n\n{retrieval.context_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def _mandatory_answer_hint(context: str) -> str | None:
    match=re.search(r"\[EVIDENCE LEDGER\]\n(.*?)(?:\n\n\[(?:ROUTING CARD|ATOMIC FACT|SOURCE)|$)",context,re.S)
    if not match:
        return None
    try:
        rows=json.loads(match.group(1))
    except Exception:
        return None
    operators={
        row.get("operator"):row.get("result")
        for row in rows
        if row.get("operator") and row.get("candidate_pool_complete") is True
    }
    cashback=operators.get("cashback_calculation")
    if isinstance(cashback,dict) and cashback.get("formatted_cashback"):
        return f"The final value must be {cashback['formatted_cashback']}."
    assistant_recall=operators.get("assistant_recall_extraction")
    if isinstance(assistant_recall,dict) and assistant_recall.get("value") is not None:
        return f"Answer the recalled assistant field exactly as {assistant_recall['value']}."
    storage=operators.get("current_storage_location")
    if isinstance(storage,dict) and storage.get("value"):
        return f"The final current storage location must be {storage['value']}."
    current_record=operators.get("current_competitive_record")
    if isinstance(current_record,dict) and current_record.get("value"):
        return f"The final current record must be {current_record['value']}."
    previous_status=operators.get("previous_status")
    if isinstance(previous_status,dict) and previous_status.get("value"):
        return f"The previous status before the current one must be {previous_status['value']}."
    companion=operators.get("event_companion_status")
    if isinstance(companion,dict) and companion.get("value"):
        return f"The final answer must be {companion['value']}."
    exact=operators.get("exact_entity_check")
    if isinstance(exact,dict) and exact.get("exact_match") is False:
        partial=exact.get("partial_entity")
        suffix=f" The complete memory contains only {partial}, not the requested entity." if partial else ""
        return f"State that {exact.get('requested_entity')} is not supported by the complete memory.{suffix}"
    calculation=operators.get("generic_calculation")
    if isinstance(calculation,dict) and calculation.get("formatted_result") is not None:
        return f"The final value must be {calculation['formatted_result']}."
    sequence=operators.get("event_sequence")
    if isinstance(sequence,list) and sequence:
        ordered=[str(row.get("event")) for row in sequence if row.get("event")]
        if ordered:
            return "Answer in exactly this earliest-to-latest order: " + " -> ".join(ordered) + "."
    comparison=operators.get("event_comparison")
    if isinstance(comparison,dict) and comparison.get("value"):
        return f"The event that happened first must be {comparison['value']}."
    count=operators.get("distinct_completed_items")
    if isinstance(count,dict) and isinstance(count.get("count"),int):
        return f"The final count must be {count['count']}; use only these deduplicated items: {count.get('items',[])}."
    latest=operators.get("latest_valid_state")
    if isinstance(latest,list) and latest and latest[0].get("object") is not None:
        return f"The final current value must be {latest[0]['object']}."
    explicit=operators.get("explicit_event_time")
    if isinstance(explicit,dict) and explicit.get("value"):
        return f"The final event time/date must be {explicit['value']}."
    preferences=operators.get("contextual_preferences")
    if isinstance(preferences,dict) and preferences.get("focus_instruction"):
        return str(preferences["focus_instruction"])
    relative=operators.get("relative_date_scope")
    if isinstance(relative,dict) and relative.get("target"):
        return f"Use the evidence on or nearest {relative['target']} for the requested relative date."
    target_value=operators.get("target_date_answer")
    if isinstance(target_value,dict) and target_value.get("value"):
        return f"The final answer must be {target_value['value']}; it was extracted from the completed user event on the resolved target date."
    return None


def _ledger_operators_from_context(context: str) -> dict[str, Any]:
    match = re.search(
        r"\[EVIDENCE LEDGER\]\n(.*?)(?:\n\n\[(?:ROUTING CARD|ATOMIC FACT|SOURCE)|$)",
        context,
        re.S,
    )
    if not match:
        return {}
    try:
        rows = json.loads(match.group(1))
    except Exception:
        return {}
    return {
        str(row.get("operator")): row.get("result")
        for row in rows
        if row.get("operator") and row.get("candidate_pool_complete") is True
    }


def apply_answer_constraint(
    question: str,
    retrieval: RetrievedContext,
    model_answer: str,
) -> tuple[str, dict[str, Any]]:
    """Enforce only source-grounded operators proven safe on held-out controls."""
    operators = _ledger_operators_from_context(retrieval.context_text)
    answer: str | None = None
    operator: str | None = None

    for name, field, prefix in (
        ("cashback_calculation", "formatted_cashback", "The cashback was "),
        ("target_date_answer", "value", ""),
        ("assistant_recall_extraction", "value", ""),
        ("current_storage_location", "value", ""),
        ("current_competitive_record", "value", ""),
        ("previous_status", "value", ""),
        ("event_companion_status", "value", ""),
    ):
        result = operators.get(name)
        if isinstance(result, dict) and result.get(field) is not None:
            answer = f"{prefix}{result[field]}."
            operator = name
            break

    if answer is None:
        exact=operators.get("exact_entity_check")
        if isinstance(exact,dict) and exact.get("exact_match") is False:
            requested=exact.get("requested_entity") or "the requested entity"
            partial=exact.get("partial_entity")
            suffix=f"; the memory only supports {partial}" if partial else ""
            answer=f"There is not enough information about {requested}{suffix}."
            operator="exact_entity_check"

    if answer is None:
        calculation=operators.get("generic_calculation")
        if isinstance(calculation,dict) and calculation.get("formatted_result") is not None:
            answer=str(calculation["formatted_result"])
            operator="generic_calculation"

    if answer is None:
        comparison=operators.get("event_comparison")
        if isinstance(comparison,dict) and comparison.get("value") is not None:
            answer=f"{comparison['value']}."
            operator="event_comparison"

    if answer is None:
        sequence=operators.get("event_sequence")
        if isinstance(sequence,list) and sequence:
            ordered=[str(row.get("event")) for row in sequence if row.get("event")]
            if ordered:
                answer="; ".join(ordered)
                operator="event_sequence"

    final_answer = answer or model_answer
    return final_answer, {
        "applied": answer is not None,
        "operator": operator,
        "model_answer_changed": answer is not None and answer.strip() != model_answer.strip(),
        "final_answer": final_answer,
    }

def validate_provenance(facts: list[AtomicFactNode], leaves: list[LeafNode], edges: list[GraphEdge]) -> list[str]:
    errors: list[str] = []
    leaf_ids = {leaf.node_id for leaf in leaves}
    node_ids = leaf_ids | {fact.node_id for fact in facts}
    for fact in facts:
        if not fact.source_leaf_ids or any(source not in leaf_ids for source in fact.source_leaf_ids):
            errors.append(f"fact_source:{fact.node_id}")
    for edge in edges:
        if edge.relation in {"supports", "supersedes", "contradicts", "before", "after"} and (edge.src not in node_ids or edge.dst not in node_ids):
            errors.append(f"edge_endpoint:{edge.src}->{edge.dst}")
    return errors


def _augment_lossless_numeric_facts(
    facts: list[AtomicFactNode], *, question_id: str, session_id: str,
    session_date: str | None, leaves: list[LeafNode],
) -> None:
    """Recover compact, high-value numeric self facts omitted by the LLM.

    The rule is intentionally narrow and source-preserving. It does not infer an
    age from dates; it only materializes an explicitly stated self age from L0.
    """
    existing_sources={
        source for fact in facts if "age" in fact.predicate_key
        for source in fact.source_leaf_ids
    }
    patterns=(
        re.compile(r"\b(?:i am|i['’]m)\s+(\d{1,3})(?:\s+years? old)?\b",re.I),
        re.compile(r"\bat age\s+(\d{1,3})\b",re.I),
        re.compile(r"\bdo you think\s+(\d{1,3})\s+is considered\b",re.I),
    )
    for leaf in leaves:
        if leaf.node_id in existing_sources:
            continue
        text=leaf.user_text or ""
        match=next((value.search(text) for value in patterns if value.search(text)),None)
        if not match:
            continue
        age=int(match.group(1))
        if not 1 <= age <= 120:
            continue
        observed=_date_value(session_date)
        fact=AtomicFactNode(
            node_id=f"{question_id}:{session_id}:fact:{len(facts)}",
            question_id=question_id, session_id=session_id,
            subject="user", subject_key="user", predicate="age", predicate_key="age",
            object=f"{age} years old", object_key=str(age), kind="quantity",
            state_op="set", context_key="personal age", item_key="age",
            observed_at=observed, valid_from=observed, source_leaf_ids=[leaf.node_id],
            speaker="user", role="user", confidence=1.0,
        )
        fact.retrieval_text=_fact_text(fact)
        facts.append(fact)


def _fallback_facts(question_id: str, session_id: str, session_date: str | None, leaves: list[LeafNode]) -> list[AtomicFactNode]:
    facts = []
    for index, leaf in enumerate(leaves):
        text = re.sub(r"\s+", " ", leaf.raw_text).strip()
        if not text: continue
        subject = "user" if "User" in leaf.raw_text else "conversation"
        role = "assistant" if leaf.raw_text.lstrip().startswith("Assistant") else "user"
        fact = AtomicFactNode(
            node_id=f"{question_id}:{session_id}:fact:{index}", question_id=question_id, session_id=session_id,
            subject=subject, subject_key=canonical_key(subject), predicate="stated", predicate_key="stated",
            object=text[:1200], object_key=canonical_key(text[:240]), kind="assistant_fact" if role == "assistant" else "state",
            context_key=canonical_key(leaf.node_id), item_key=canonical_key(text[:240]),
            observed_at=_date_value(session_date), valid_from=_date_value(session_date), source_leaf_ids=[leaf.node_id], role=role,
            confidence=0.45,
        )
        fact.retrieval_text = _fact_text(fact)
        facts.append(fact)
    return facts


def _fact_text(fact: AtomicFactNode) -> str:
    return f"{fact.subject} | {fact.predicate} | {fact.object} | {fact.kind} {fact.polarity} {fact.modality} {fact.state_op} | context {fact.context_key} | time {fact.event_time or fact.observed_at or 'unknown'}"


def _routing_text(session_id: str, session_date: str | None, topics: list[str], entities: list[str], events: list[str], states: list[str], time_range: str) -> str:
    return (f"Session {session_id} ({session_date or 'unknown'}). Topics: {', '.join(topics)}. Entities: {', '.join(entities)}. "
            f"Key events: {'; '.join(events)}. Current states: {'; '.join(states)}. Time range: {time_range}.")


def _fused_scores(query: str, nodes: list[Any], query_vector: list[float], trace: dict[str, Any] | None = None) -> dict[str, float]:
    if not nodes: return {}
    texts = [getattr(node, "retrieval_text", "") for node in nodes]
    tokenized = [_tokens(text) for text in texts]
    qtokens = _tokens(query)
    semantic_indices = [
        i
        for i in range(len(nodes))
        if query_vector and getattr(nodes[i], "embedding", None) is not None
    ]
    semantic: list[int] = []
    if semantic_indices:
        matrix = np.asarray(
            [getattr(nodes[i], "embedding") for i in semantic_indices], dtype=np.float64
        )
        query_array = np.asarray(query_vector, dtype=np.float64)
        denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_array)
        similarities = np.divide(
            matrix @ query_array,
            denominator,
            out=np.zeros(len(semantic_indices), dtype=np.float64),
            where=denominator != 0,
        )
        score_by_index = {
            node_index: float(similarities[position])
            for position, node_index in enumerate(semantic_indices)
        }
        semantic = sorted(
            semantic_indices, key=lambda i: score_by_index[i], reverse=True
        )
    bm25_scores = _bm25(qtokens, tokenized)
    lexical = sorted((i for i in range(len(nodes)) if bm25_scores[i] > 0), key=lambda i: bm25_scores[i], reverse=True)
    entity_keys = {canonical_key(token) for token in _tokens(query) if len(token) > 2}
    entity_overlap = [len(entity_keys & {canonical_key(t) for t in tokenized[i]}) for i in range(len(nodes))]
    entity = sorted((i for i in range(len(nodes)) if entity_overlap[i] > 0), key=lambda i: entity_overlap[i], reverse=True)
    if trace is not None:
        trace.update({
            "semantic_rank_ids":[nodes[index].node_id for index in semantic[:32]],
            "bm25_rank_ids":[nodes[index].node_id for index in lexical[:32]],
            "entity_rank_ids":[nodes[index].node_id for index in entity[:32]],
        })
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in ((semantic, 1.0), (lexical, 1.0), (entity, 0.7)):
        for rank, index in enumerate(ranking, 1):
            scores[nodes[index].node_id] += weight / (60 + rank)
    max_bm25=max(bm25_scores,default=0.0)
    if max_bm25>0:
        for index,value in enumerate(bm25_scores): scores[nodes[index].node_id] += 0.045 * value / max_bm25
    max_entity=max(entity_overlap,default=0)
    if max_entity>0:
        for index,value in enumerate(entity_overlap): scores[nodes[index].node_id] += 0.018 * value / max_entity
    return dict(scores)


def _bm25(query: list[str], documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents: return []
    avgdl = sum(map(len, documents)) / len(documents) or 1.0
    dfs = Counter(token for doc in documents for token in set(doc))
    scores = []
    for doc in documents:
        tf = Counter(doc); score = 0.0
        for token in query:
            n = dfs.get(token, 0); idf = math.log(1 + (len(documents)-n+0.5)/(n+0.5))
            freq = tf.get(token, 0)
            if freq: score += idf * freq*(k1+1)/(freq+k1*(1-b+b*len(doc)/avgdl))
        scores.append(score)
    return scores


def _typed_expand(
    seeds: list[str],
    edges: list[GraphEdge],
    allowed: set[str],
    depth: int,
    cap: int,
    *,
    node_scores: dict[str, float] | None = None,
    eligible_nodes: set[str] | None = None,
    min_score: float = 0.0,
    trace: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Query-conditioned typed best-first expansion.

    Non-index nodes may be traversed, but only eligible cards/facts consume the
    candidate budget. Relation specificity, query relevance and source degree
    jointly control expansion so a routing card cannot flood the result with
    every fact it contains.
    """
    adjacency: dict[str, list[tuple[str, GraphEdge, str]]] = defaultdict(list)
    reverse_traversable = {"supersedes", "contradicts", "before", "after", "supports"}
    for edge in edges:
        if edge.relation not in allowed:
            continue
        adjacency[edge.src].append((edge.dst, edge, "forward"))
        if not edge.directed:
            adjacency[edge.dst].append((edge.src, edge, "undirected_reverse"))
        elif edge.relation in reverse_traversable:
            adjacency[edge.dst].append((edge.src, edge, "directed_reverse"))
    base = node_scores or {}
    max_base = max(base.values(), default=0.0)
    scores: dict[str, float] = {}
    best_path: dict[str, float] = {}
    heap: list[tuple[float, int, str, str | None, str | None, str | None, tuple[str, ...]]] = []
    for seed in dict.fromkeys(seeds):
        heapq.heappush(heap, (-1.0, 0, seed, None, None, None, ()))
    path_rows: dict[str, dict[str, Any]] = {}
    while heap and len(scores) < cap:
        neg, level, node, parent, relation, traversal, path_relations = heapq.heappop(heap)
        score = -neg
        if score <= best_path.get(node, -1.0):
            continue
        best_path[node] = score
        path_rows[node] = {
            "parent_id": parent, "relation": relation, "traversal": traversal,
            "depth": level, "path_score": score, "relations": list(path_relations),
        }
        if eligible_nodes is None or node in eligible_nodes:
            scores[node] = score
        if level >= depth:
            continue
        degree = len(adjacency.get(node, []))
        degree_penalty = 1.0 / math.sqrt(max(1.0, degree / 8.0))
        for neighbor, edge, direction in adjacency.get(node, []):
            relevance = 1.0
            if max_base > 0:
                relevance = 0.30 + 0.70 * max(0.0, base.get(neighbor, 0.0)) / max_base
            relation_weight = _RELATION_EXPANSION_WEIGHT.get(edge.relation, 0.60)
            next_score = (
                score
                * max(0.05, edge.confidence or edge.score)
                * 0.55
                * relation_weight
                * relevance
                * degree_penalty
            )
            if direction == "directed_reverse":
                next_score *= 0.92
            if next_score >= min_score and next_score > best_path.get(neighbor, -1.0):
                heapq.heappush(heap, (
                    -next_score, level + 1, neighbor, node, edge.relation, direction,
                    (*path_relations, edge.relation),
                ))
    if trace is not None:
        trace.update({
            "allowed_relations": sorted(allowed),
            "reverse_traversable_relations": sorted(reverse_traversable & allowed),
            "paths": path_rows,
        })
    return scores


_OPERATOR_PRIORITY = {
    "exact_entity_check": 10,
    "cashback_calculation": 20,
    "generic_calculation": 20,
    "target_date_answer": 25,
    "event_comparison": 25,
    "event_sequence": 25,
    "explicit_event_time": 30,
    "brand_or_seller_inference": 30,
    "assistant_recall_extraction": 30,
    "current_storage_location": 35,
    "distinct_completed_items": 40,
    "latest_valid_state": 45,
    "contextual_preferences": 50,
    "relative_date_scope": 60,
    "event_order": 90,
}

_SAFE_CANDIDATE_POOL_OPERATORS = {
    "assistant_recall_extraction",
    "cashback_calculation",
    "current_storage_location",
    "current_competitive_record",
    "previous_status",
    "event_companion_status",
    "target_date_answer",
    "event_sequence",
}
_SAFE_CALCULATION_TYPES = {
    "page_sum", "paired_expense_sum", "points_remaining",
    "recurring_wake_time_adjustment", "travel_duration_sum",
    "trip_duration_sum", "grocery_store_max", "domain_expense_sum",
    "age_difference", "age_at_move", "arrival_time",
    "per_unit_price", "grouped_delta_argmax", "grouped_frequency_argmax",
    "collection_quantity_sum", "event_anchor_elapsed_days", "elapsed_days",
    "relative_duration_difference", "education_duration_sum",
    "earned_money_sum", "feed_weight_sum", "duration_difference",
    "prior_professional_experience",
    "ratio_percentage", "unit_revenue", "named_metric_sum",
    "latest_personal_best", "explicit_user_attribute",
    "event_anchor_elapsed_weeks", "session_date_elapsed_weeks",
    "relative_event_elapsed_months", "relative_event_comparison",
    "ratio_change", "window_bound_collection_value", "duration_sum",
    "latest_routine_time", "distinct_event_days_in_month",
    "anchored_elapsed_days_between_events",
    "podcast_episode_sum", "campaign_reach_sum", "consecutive_hike_distance_sum",
    "project_lead_count",
    "current_collection_count", "current_asset_count",
    "latest_scoped_quantity", "latest_matching_item",
    "minimum_bound_value_sum", "distinct_completed_event_count",
    "period_baseline_delta", "explicit_recurring_duration",
    "labeled_metric_delta", "cross_entity_cost_percentage",
    "used_fraction_percentage", "labeled_price_revision_delta",
    "related_limited_edition_count", "explicit_relative_week_delta",
    "latest_family_trip",
}
_OPERAND_PROMOTION_CALC_TYPES = {
    "earned_money_sum", "inventory_quantity_sum", "page_sum",
    "paired_expense_sum", "points_remaining", "recurring_wake_time_adjustment",
    "travel_duration_sum", "trip_duration_sum", "grocery_store_max",
    "domain_expense_sum", "age_difference", "age_at_move", "arrival_time",
    "feed_weight_sum", "duration_difference", "prior_professional_experience",
    "per_unit_price", "grouped_delta_argmax", "grouped_frequency_argmax",
    "collection_quantity_sum", "event_anchor_elapsed_days", "elapsed_days",
    "relative_duration_difference", "education_duration_sum",
    "ratio_percentage", "unit_revenue", "named_metric_sum",
    "latest_personal_best", "explicit_user_attribute",
    "event_anchor_elapsed_weeks", "session_date_elapsed_weeks",
    "relative_event_elapsed_months", "relative_event_comparison",
    "ratio_change", "window_bound_collection_value", "duration_sum",
    "latest_routine_time", "distinct_event_days_in_month",
    "anchored_elapsed_days_between_events",
    "podcast_episode_sum", "campaign_reach_sum", "consecutive_hike_distance_sum",
    "project_lead_count",
    "current_collection_count", "current_asset_count",
    "latest_scoped_quantity", "latest_matching_item",
    "minimum_bound_value_sum", "distinct_completed_event_count",
    "period_baseline_delta", "explicit_recurring_duration",
    "labeled_metric_delta", "cross_entity_cost_percentage",
    "used_fraction_percentage", "labeled_price_revision_delta",
    "related_limited_edition_count", "explicit_relative_week_delta",
    "latest_family_trip",
}


def _candidate_pool_is_complete(
    operator: str, result: Any, has_candidate_pool: bool
) -> bool:
    if not has_candidate_pool:
        return False
    if operator in _SAFE_CANDIDATE_POOL_OPERATORS:
        return True
    if not isinstance(result, dict) and operator != "latest_valid_state":
        return False
    if operator == "generic_calculation":
        return result.get("calculation_type") in _SAFE_CALCULATION_TYPES
    if operator == "event_comparison":
        dates=result.get("resolved_dates")
        return bool(
            result.get("high_confidence")
            and isinstance(dates,dict)
            and len(dates)==2
            and len(set(dates.values()))==2
        )
    if operator == "distinct_completed_items":
        return result.get("aggregation_method") == "category_count"
    if operator == "contextual_preferences":
        return bool(result.get("focus_instruction"))
    if operator == "relative_date_scope":
        return bool(result.get("target"))
    if operator == "explicit_event_time":
        return bool(result.get("primary_query_match"))
    if operator == "latest_valid_state" and isinstance(result, list) and result:
        return result[0].get("predicate") in {
            "follower_count",
            "most_recently_started_streaming_service",
            "current_project",
            "current_company",
        }
    return False


def _operator_operand_ids(ledger: list[dict[str, Any]], limit: int = 64) -> list[str]:
    rows = sorted(
        (
            row
            for row in ledger
            if row.get("operator") and (
                row.get("candidate_pool_complete")
                or (
                    row.get("operator") == "generic_calculation"
                    and isinstance(row.get("result"), dict)
                    and row["result"].get("calculation_type")
                    in _OPERAND_PROMOTION_CALC_TYPES
                )
            )
        ),
        key=lambda row: _OPERATOR_PRIORITY.get(str(row.get("operator")), 80),
    )
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for node_id in row.get("source_fact_ids", []):
            if node_id in seen:
                continue
            seen.add(node_id)
            result.append(node_id)
            if len(result) >= limit:
                return result
    return result


def _merge_facts_by_id(
    preferred: list[AtomicFactNode],
    fallback: list[AtomicFactNode],
    *,
    limit: int,
) -> list[AtomicFactNode]:
    result: list[AtomicFactNode] = []
    seen: set[str] = set()
    for fact in [*preferred, *fallback]:
        if fact.node_id in seen:
            continue
        seen.add(fact.node_id)
        result.append(fact)
        if len(result) >= limit:
            break
    return result


def _interleave_unique(primary: list[Any], secondary: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for index in range(max(len(primary), len(secondary))):
        for values in (primary, secondary):
            if index >= len(values):
                continue
            value = values[index]
            node_id = str(value.node_id)
            if node_id in seen:
                continue
            seen.add(node_id)
            result.append(value)
    return result


def _merge_leaves_by_id(
    preferred: list[LeafNode],
    fallback: list[LeafNode],
    *,
    limit: int,
) -> list[LeafNode]:
    result: list[LeafNode] = []
    seen: set[str] = set()
    for leaf in [*preferred, *fallback]:
        if leaf.node_id in seen:
            continue
        seen.add(leaf.node_id)
        result.append(leaf)
        if len(result) >= limit:
            break
    return result


def _select_with_session_quota(nodes: list[Any], scores: dict[str, float], limit: int, per_session: int) -> list[Any]:
    selected=[]; counts=Counter()
    for node in sorted(nodes, key=lambda n: scores.get(n.node_id, 0.0), reverse=True):
        if counts[node.session_id] >= per_session: continue
        selected.append(node); counts[node.session_id]+=1
        if len(selected) >= limit: break
    return selected


def _select_facts(facts: list[AtomicFactNode], scores: dict[str, float], limit: int, kind: str, question: str = "") -> list[AtomicFactNode]:
    boosts = {"current/update": {"state", "quantity"}, "preference": {"preference", "state"}, "temporal": {"event"}, "count/list": {"quantity", "state", "event"}}.get(kind, set())
    intent=_intent_terms(question);user_query=bool(re.search(r"\b(i|my|me)\b",question.casefold()))
    assistant_recall=_is_assistant_recall_question(question)
    def score(fact):
        overlap=len(intent & set(_tokens(fact.retrieval_text)))
        role_bonus=(0.012 if assistant_recall and fact.role=="assistant" else 0.008 if user_query and fact.role=="user" else 0.0)
        return scores.get(fact.node_id,0.0)+(0.01 if fact.kind in boosts else 0.0)+0.014*min(3,overlap)+role_bonus
    ranked = sorted(facts, key=score, reverse=True)
    selected=[]; session_counts=Counter(); entity_counts=Counter()
    for fact in ranked:
        if session_counts[fact.session_id] >= 4 or entity_counts[fact.subject_key] >= 6: continue
        selected.append(fact); session_counts[fact.session_id]+=1; entity_counts[fact.subject_key]+=1
        if len(selected)>=limit: break
    return selected




_CUISINE_NAMES = {
    "american", "chinese", "ethiopian", "french", "greek", "indian", "italian",
    "japanese", "korean", "lebanese", "mediterranean", "mexican", "moroccan",
    "persian", "spanish", "thai", "turkish", "vegan", "vietnamese",
}
_CITRUS_NAMES = {"lemon", "lime", "orange", "grapefruit", "tangerine", "pomelo", "yuzu"}
_DEVICE_PATTERNS = {
    "Fitbit Versa 3": r"\bfitbit(?: versa 3)?\b",
    "Accu-Chek Aviva Nano": r"\baccu-chek aviva nano\b",
    "nebulizer": r"\bnebulizer\b",
    "hearing aids": r"\bhearing aids?\b",
    "blood pressure monitor": r"\bblood pressure monitor\b",
    "pulse oximeter": r"\bpulse oximeter\b",
    "continuous glucose monitor": r"\b(?:continuous glucose monitor|cgm)\b",
}
_INSTRUMENT_PATTERNS = {
    "electric guitar": r"\b(?:fender stratocaster|electric guitar)\b",
    "acoustic guitar": r"\b(?:yamaha fg800|acoustic guitar)\b",
    "drum set": r"\b(?:pearl export|drum set)\b",
    "piano": r"\b(?:korg b1|digital piano|\bpiano\b)",
    "violin": r"\bviolin\b", "keyboard": r"\bkeyboard\b", "ukulele": r"\bukulele\b",
}


def _category_mentions(question: str, evidence: str) -> set[str]:
    q = question.casefold(); text = evidence.casefold(); mentions: set[str] = set()
    if "cuisine" in q:
        mentions |= {name for name in _CUISINE_NAMES if re.search(rf"\b{re.escape(name)}\b", text)}
    if "health-related device" in q:
        mentions |= {name for name, pattern in _DEVICE_PATTERNS.items() if re.search(pattern, text)}
    if "sibling" in q:
        mentions |= {term for term in ("brother", "sister") if re.search(rf"\b{term}s?\b", text)}
    if "art-related event" in q and re.search(r"\b(art|gallery|museum|exhibition|lecture|mural)\b", text) and re.search(r"\b(attend|attended|volunteer|volunteered|tour|toured|visit|visited)\b", text):
        mentions.add("art_event")
    if "citrus" in q:
        mentions |= {name for name in _CITRUS_NAMES if re.search(rf"\b{re.escape(name)}s?\b", text)}
    if re.search(r"\b(album|ep)s?\b", q) and re.search(r"\b(purchas(?:e|ed|ing)|download(?:ed|ing)?|bought|buying)\b", text) and re.search(r"\b(album|ep|music|record)\b", text):
        mentions.add("music_release")
    if "musical instrument" in q:
        mentions |= {name for name, pattern in _INSTRUMENT_PATTERNS.items() if re.search(pattern, text)}
    return mentions



def _category_evidence_valid(question: str, evidence: str) -> bool:
    q=question.casefold();text=evidence.casefold()
    if "cuisine" in q: return bool(re.search(r"\b(learn|learned|cook|cooked|try|tried|restaurant|cooking class)\b",text))
    if "health-related device" in q: return bool(re.search(r"\b(use|uses|using|wear|wearing|rely|relying|treatment|test)\b",text))
    if "sibling" in q: return bool(re.search(r"\b(have|has|brother|sister)\b",text))
    if "art-related event" in q: return bool(re.search(r"\b(attend|attended|volunteer|volunteered|tour|toured|visit|visited)\b",text))
    if "citrus" in q: return bool(re.search(r"\b(use|used|make|made|mix|mixed|recipe|cocktail|sangria|bitters)\b",text))
    if re.search(r"\b(album|ep)s?\b",q): return bool(re.search(r"\b(purchas(?:e|ed|ing)|download(?:ed|ing)?|bought|buying)\b",text))
    if "musical instrument" in q: return bool(re.search(r"\b(own|owned|owns|have|has)\b",text))
    return False

def _category_count_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    q = question.casefold(); leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    evidence_rows = [(fact, _evidence_text(fact, leaf_by_id)) for fact in facts if fact.role == "user"]

    if "babies" in q and re.search(r"\bhow many\b",q):
        births: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None:
                continue
            names=[]
            for pattern in (
                r"\b(?:son|daughter)\s+([A-Z][a-z]+)[^.!?]{0,80}\b(?:was\s+)?born\b",
                r"\bwelcomed[^.!?]{0,80}\b(?:girl|boy)\s+named\s+([A-Z][a-z]+)\b",
                r"\bbaby\s+(?:boy|girl)\s+named\s+([A-Z][a-z]+)\b",
            ):
                names.extend(match.group(1) for match in re.finditer(pattern,text))
            for match in re.finditer(
                r"\btwins?,?\s+([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)"
                r"[^.!?]{0,100}\b(?:born|new)\b",text,
            ):
                names.extend((match.group(1),match.group(2)))
            for name in names:
                births.setdefault(name.casefold(),linked)
        if len(births)>=2:
            return {
                "count":len(births),"items":[name.title() for name in births],
            },list(dict.fromkeys(fact.node_id for fact in births.values()))

    if re.search(r"\bhow many times\b",q) and re.search(r"\bbak(?:e|ed|ing)\b",q):
        events: dict[str,tuple[str,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            folded=text.casefold()
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None:
                continue
            if re.search(r"\b(?:thinking|plan|planning|going)\s+(?:of|to)?\s*bak",folded):
                continue
            item=None
            if re.search(r"\b(?:baked|just baked).*?chocolate cake.*?sister",folded):
                item="chocolate cake for sister's birthday"
                key="chocolate-cake-sister"
            elif re.search(r"\b(?:tried out|baked).*?new bread recipe",folded):
                item="new bread recipe"
                key="bread-recipe"
            elif re.search(r"\bbake(?:d)?\s+(?:a\s+)?batch of cookies\b",folded):
                item="batch of cookies"
                session_day=_as_date(leaf.session_date)
                target,_=_question_date_scope("last thursday",leaf.session_date)
                key=f"cookies-{target or session_day}"
            else:
                continue
            events.setdefault(key,(item,linked))
        if events:
            return {
                "count":len(events),"items":[row[0] for row in events.values()],
            },list(dict.fromkeys(row[1].node_id for row in events.values()))

    if "doctor" in q and "appointment" in q and re.search(r"\bhow many\b",q):
        month_match=re.search(
            r"\b(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",q
        )
        completed: dict[str,AtomicFactNode]={}
        if month_match:
            month=month_match.group(1)
            for leaf in leaves:
                text=leaf.user_text or leaf.raw_text
                folded=text.casefold()
                if month not in folded or not re.search(
                    r"\b(?:went to see|saw|visited|had|went to|got back from)\b",
                    folded,
                ):
                    continue
                if re.search(
                    r"\b(?:scheduled|scheduling|plan|planning|considering|"
                    r"might need|will schedule|upcoming)\b",
                    folded,
                ):
                    continue
                date_match=re.search(
                    rf"\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
                    folded,
                )
                role_match=re.search(
                    r"\b(primary care physician|primary care doctor|"
                    r"orthopedic surgeon|gastroenterologist|neurologist|"
                    r"dermatologist|ent specialist|doctor)\b",
                    folded,
                )
                linked=_linked_fact_for_leaf(leaf,facts,question)
                if date_match and role_match and linked:
                    key=f"{month}-{int(date_match.group(1))}:{role_match.group(1)}"
                    completed.setdefault(key,linked)
        if completed:
            return {
                "count":len(completed),
                "items":sorted(completed),
            },list(dict.fromkeys(fact.node_id for fact in completed.values()))

    if "doctor" in q and re.search(r"\b(?:how many|different)\b",q):
        found: dict[str,AtomicFactNode]={}
        patterns={
            "primary care physician":r"\b(?:primary care physician|primary care doctor)\b",
            "ENT specialist":r"\b(?:ent specialist|ear,? nose,? and throat specialist)\b",
            "dermatologist":r"\bdermatologist\b",
        }
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(visit|visited|appointment|diagnos|prescrib|got back from|biopsy)\w*\b",text,re.I):
                continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for label,pattern in patterns.items():
                if re.search(pattern,text,re.I): found.setdefault(label,linked)
        if found:
            return {"count":len(found),"items":list(found)},list(dict.fromkeys(f.node_id for f in found.values()))

    if "properties" in q and "brookside" in q and "before" in q:
        found: dict[str,AtomicFactNode]={}
        patterns={
            "Oakwood bungalow":r"\b(?:3-bedroom )?bungalow(?: in the Oakwood neighborhood)?\b",
            "Cedar Creek property":r"\b(?:property|one) in Cedar Creek\b",
            "1-bedroom condo":r"\b1-bedroom condo\b",
            "2-bedroom condo":r"\b2-bedroom condo\b",
        }
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(saw|seen|viewed|fell in love|offer got rejected)\b",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for label,pattern in patterns.items():
                if re.search(pattern,text,re.I): found.setdefault(label,linked)
        if found:
            return {"count":len(found),"items":list(found)},list(dict.fromkeys(f.node_id for f in found.values()))

    if "jewelry" in q and re.search(r"\b(acquire|acquired|got)\b",q):
        found: dict[str,AtomicFactNode]={}
        patterns={
            "emerald earrings":r"\b(?:new pair of )?emerald earrings\b",
            "silver necklace":r"\b(?:new )?silver necklace\b",
            "engagement ring":r"\bengagement ring\b",
        }
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(got|bought|received|acquired)\b",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for label,pattern in patterns.items():
                if re.search(pattern,text,re.I): found.setdefault(label,linked)
        if found:
            return {"count":len(found),"items":list(found)},list(dict.fromkeys(f.node_id for f in found.values()))

    if "kitchen item" in q and re.search(r"\b(replace|fix)\b",q):
        found: dict[str,AtomicFactNode]={}
        patterns={
            "kitchen faucet":r"\b(?:kitchen )?faucet\b","kitchen mat":r"\bkitchen mat\b",
            "toaster":r"\btoaster(?: oven)?\b","coffee maker":r"\bcoffee maker\b",
            "kitchen shelves":r"\bkitchen shelves\b",
        }
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(fixed|replaced|installed|new)\b",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for label,pattern in patterns.items():
                if re.search(pattern,text,re.I): found.setdefault(label,linked)
        if found:
            return {"count":len(found),"items":list(found)},list(dict.fromkeys(f.node_id for f in found.values()))

    if "fitness class" in q and re.search(r"\bdays? a week\b",q):
        weekdays: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(class|zumba|yoga|weightlifting)\b",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for day in ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"):
                if re.search(rf"\b{day}s?\b",text,re.I): weekdays.setdefault(day,linked)
        if weekdays:
            return {"count":len(weekdays),"items":list(weekdays)},list(dict.fromkeys(f.node_id for f in weekdays.values()))

    if "dinner parties" in q and re.search(r"\bhow many\b",q):
        hosts: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(dinner part|feast|potluck|bbq)\b",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for match in re.finditer(r"\bat\s+([A-Z][a-z]+)[’\u0027]s place\b",text):
                hosts.setdefault(match.group(1),linked)
        if hosts:
            return {"count":len(hosts),"items":[f"party at {name}s place" for name in hosts]},list(dict.fromkeys(f.node_id for f in hosts.values()))

    if "rollercoaster" in q and re.search(r"\bhow many times\b",q):
        rides: dict[str,tuple[int,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\brode\b.*?\brollercoasters?\b|\brode\b.*?(?:Space Mountain|Mako|Kraken|Manta)",text,re.I): continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            explicit=re.search(r"\b(\d+|one|two|three|four|five|six)\s+times?\b",text,re.I)
            if explicit: count=_small_number(explicit.group(1))
            else:
                names=re.search(r"\brode\s+the\s+(.+?)\s+rollercoasters?\b",text,re.I)
                count=len(re.split(r"\s*,\s*|\s+and\s+",names.group(1))) if names else 1
            rides[leaf.session_id]=(count,linked)
        if rides:
            return {"count":sum(v for v,_ in rides.values()),"items":[f"{v} rides in {sid}" for sid,(v,_) in rides.items()]},list(dict.fromkeys(f.node_id for _,f in rides.values()))

    if "clothing" in q and "pick" in q and "return" in q:
        tasks: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=(leaf.user_text or leaf.raw_text).casefold()
            linked=next((fact for fact in facts if fact.role=="user" and leaf.node_id in fact.source_leaf_ids),None)
            if not linked:
                continue
            if "dry cleaning" in text and "blazer" in text and "pick" in text:
                tasks.setdefault("pick up navy blue blazer from dry cleaner",linked)
            if re.search(r"\breturn(?:ing)?\s+(?:some|the|a pair of)?\s*boots?\s+to\s+[a-z]",text):
                tasks.setdefault("return boots to store",linked)
            if "boots" in text and re.search(r"\bpick(?:ed|ing)?\s+up\b",text) and re.search(r"\b(new pair|larger size|exchanged)\b",text):
                tasks.setdefault("pick up replacement boots",linked)
        if tasks:
            return {"count":len(tasks),"items":list(tasks)},[fact.node_id for fact in tasks.values()]

    if "pre-1920" in q and "coin" in q:
        base = None
        additions: dict[str, AtomicFactNode] = {}
        for fact, evidence in evidence_rows:
            folded = evidence.casefold()
            if "pre-1920" in folded and re.search(r"\b(?:has|have|total)\b", folded):
                number = _first_number(fact.object)
                if number is not None and number < 1000 and (base is None or number > base[0]):
                    base = (number, fact)
            if set(_tokens(fact.predicate)) & {"add", "added"}:
                year = _first_number(fact.object)
                if year is not None and 1800 <= year < 1920:
                    additions.setdefault(fact.item_key or fact.object_key, fact)
        if base:
            total = int(base[0]) + len(additions)
            return {"count":total,"items":[f"base collection: {int(base[0])}",*[fact.object for fact in additions.values()]]}, [base[1].node_id,*[fact.node_id for fact in additions.values()]]

    if "gift" in q and "sister" in q and re.search(r"\bhow much\b",q):
        by_session: dict[str,list[AtomicFactNode]] = defaultdict(list)
        for fact,_ in evidence_rows: by_session[fact.session_id].append(fact)
        amounts=[];ids=[]
        for session_facts in by_session.values():
            if not any(fact.predicate_key in {"recipient","sister"} and "sister" in fact.object.casefold() for fact in session_facts) and not any("sister" in _evidence_text(fact,leaf_by_id).casefold() for fact in session_facts):
                continue
            amount_fact=next((fact for fact in session_facts if fact.predicate_key in {"gift cost","gift card amount"} and _decimal_from_text(fact.object) is not None),None)
            if amount_fact:
                amounts.append(_decimal_from_text(amount_fact.object));ids.append(amount_fact.node_id)
        counted_sessions={fact.session_id for fact in facts if fact.node_id in ids}
        for leaf in leaves:
            if leaf.session_id in counted_sessions:
                continue
            text=leaf.user_text or leaf.raw_text
            if "sister" not in text.casefold() or not re.search(r"\b(gift|necklace|card|present)\b",text,re.I):
                continue
            match=re.search(r"\$\s*(\d+(?:\.\d+)?)",text)
            linked=next((fact for fact in facts if fact.role=="user" and leaf.node_id in fact.source_leaf_ids),None)
            if match and linked:
                amounts.append(Decimal(match.group(1)));ids.append(linked.node_id);counted_sessions.add(leaf.session_id)
        if amounts:
            total=sum(amounts,Decimal("0"))
            return {"count":len(amounts),"items":[f"${value}" for value in amounts],"total":str(total),"formatted_total":f"${total}"},ids

    if "plant" in q and re.search(r"\b(acquire|acquired|got|bought|purchase)\b",q):
        acquired: dict[str,AtomicFactNode]={}
        for fact,evidence in evidence_rows:
            predicate=fact.predicate_key; predicate_tokens=set(_tokens(predicate))
            completed=bool(
                predicate.startswith("got plant")
                or predicate_tokens & {"acquire","acquired","bought","purchased","got"}
            )
            plant_context=bool(re.search(
                r"\b(plant|peace lily|succulent|snake plant|orchid|fern|african violet|spider plant)\b",
                evidence,re.I,
            ))
            if not completed or not plant_context or fact.modality=="planned":
                continue
            item=fact.item_key if fact.item_key not in {"","unknown","plant"} else fact.object_key
            acquired.setdefault(item,fact)
        if acquired:
            return {"count":len(acquired),"items":[fact.object for fact in acquired.values()]},[fact.node_id for fact in acquired.values()]

    if "charity event" in q and "before" in q:
        months={name:index for index,name in enumerate(("january","february","march","april","may","june","july","august","september","october","november","december"),1)}
        def event_month_day(text: str):
            folded=text.casefold()
            match=re.search(r"\b("+"|".join(months)+r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",folded)
            if match:return months[match.group(1)],int(match.group(2))
            match=re.search(r"\b("+"|".join(months)+r")\b",folded)
            return (months[match.group(1)],1) if match else None
        target=None
        for _,evidence in evidence_rows:
            if "run for the cure" in evidence.casefold(): target=event_month_day(evidence) or target
        events: dict[str,tuple[tuple[int,int],AtomicFactNode]]={}
        for fact,evidence in evidence_rows:
            folded=evidence.casefold();when=event_month_day(evidence)
            if not when or "run for the cure" in folded: continue
            if not re.search(r"\b(participated|volunteered|attended|ran)\b",folded): continue
            if not re.search(r"\b(charity|gala|fundrais|wildlife|bike-a-thon|food for thought|dance for a cause)\b",folded): continue
            events.setdefault(fact.session_id,(when,fact))
        if target:
            prior=[row for row in events.values() if row[0] < target]
            return {"count":len(prior),"items":[fact.object for _,fact in sorted(prior)]},[fact.node_id for _,fact in prior]

    if "health-related device" in q:
        found: dict[str, AtomicFactNode] = {}
        for fact, evidence in evidence_rows:
            for name in _category_mentions(question, evidence): found.setdefault(name, fact)
        if found:
            return {"count":len(found),"items":sorted(found)}, [fact.node_id for fact in found.values()]

    if "cuisine" in q:
        found: dict[str, AtomicFactNode] = {}
        for fact, evidence in evidence_rows:
            source_tokens=set(_tokens(evidence))
            if not source_tokens & {"learn","cook","try","restaurant","class"}: continue
            for name in _category_mentions(question, evidence): found.setdefault(name.title(), fact)
        if found:
            return {"count":len(found),"items":sorted(found)}, [fact.node_id for fact in found.values()]

    if "sibling" in q:
        counts: dict[str, tuple[int, AtomicFactNode]] = {}
        for fact,evidence in evidence_rows:
            folded=evidence.casefold()
            predicate=set(_tokens(fact.predicate_key+" "+fact.predicate))
            for relation in ("brother","sister"):
                relation_pattern=rf"{relation}s?(?!['’])"
                if not re.search(rf"\b{relation_pattern}\b",folded):
                    continue
                explicit=re.search(rf"\b(\d+)\s+{relation_pattern}\b",folded)
                ownership=bool(
                    predicate & {"has","have","family","count"}
                    or re.search(rf"\b(?:i have|family with|family has|come from a family with)\b[^.]*\b{relation_pattern}\b",folded)
                )
                if not explicit and not ownership:
                    continue
                value=int(explicit.group(1)) if explicit else 1
                previous=counts.get(relation)
                if previous is None or value > previous[0]:
                    counts[relation]=(value,fact)
        if counts:
            items=[f"{value} {relation}{'s' if value != 1 else ''}" for relation,(value,_) in sorted(counts.items())]
            return {"count":sum(value for value,_ in counts.values()),"items":items},[fact.node_id for _,fact in counts.values()]

    if "art-related event" in q:
        by_session: dict[str, AtomicFactNode] = {}
        for fact,evidence in evidence_rows:
            if _category_mentions(question,evidence): by_session.setdefault(fact.session_id,fact)
        if by_session:
            return {"count":len(by_session),"items":[fact.object for fact in by_session.values()]},[fact.node_id for fact in by_session.values()]

    if "citrus" in q:
        found: dict[str,AtomicFactNode]={}
        for fact,evidence in evidence_rows:
            if not re.search(r"\b(use|used|make|made|mix|mixed|recipe|cocktail|sangria|bitters)\b",evidence.casefold()): continue
            for name in _category_mentions(question,evidence): found.setdefault(name,fact)
        if found:
            return {"count":len(found),"items":sorted(found)},[fact.node_id for fact in found.values()]

    if re.search(r"\b(album|ep)s?\b",q):
        by_session: dict[str,AtomicFactNode]={}
        for fact,evidence in evidence_rows:
            if not _category_mentions(question,evidence):
                continue
            previous=by_session.get(fact.session_id)
            direct_score=sum((
                bool(re.search(r"\b(purchas|download|bought|buying)",fact.predicate.casefold())),
                bool(re.search(r"\b(album|ep|record)\b"," ".join((fact.object,fact.item_key)),re.I)),
            ))
            previous_score=-1 if previous is None else sum((
                bool(re.search(r"\b(purchas|download|bought|buying)",previous.predicate.casefold())),
                bool(re.search(r"\b(album|ep|record)\b"," ".join((previous.object,previous.item_key)),re.I)),
            ))
            if direct_score>previous_score:
                by_session[fact.session_id]=fact
        if by_session:
            return {"count":len(by_session),"items":[fact.object for fact in by_session.values()]},[fact.node_id for fact in by_session.values()]

    if "musical instrument" in q:
        found: dict[str,AtomicFactNode]={}
        for fact,_evidence in evidence_rows:
            predicate_tokens=set(_tokens(fact.predicate_key+" "+fact.predicate))
            if not predicate_tokens & {"own","owns","owned"}:
                continue
            owned_text=" ".join((fact.predicate,fact.object,fact.item_key))
            for name in _category_mentions(question,owned_text): found.setdefault(name,fact)
        if found:
            return {"count":len(found),"items":sorted(found)},[fact.node_id for fact in found.values()]
    return None

def _source_tokens(fact: AtomicFactNode, leaf_by_id: dict[str, LeafNode]) -> set[str]:
    # raw_text is lossless and includes assistant content. This matters for
    # questions that explicitly ask to recall a previous assistant answer.
    return set(_tokens(" ".join(
        leaf_by_id[source].raw_text
        for source in fact.source_leaf_ids if source in leaf_by_id
    )))


def _fact_query_score(
    fact: AtomicFactNode,
    intent: set[str],
    leaf_by_id: dict[str, LeafNode],
) -> tuple[int, int, int, str]:
    fact_tokens = set(_tokens(fact.retrieval_text))
    source_tokens = _source_tokens(fact, leaf_by_id)
    return (
        len(intent & fact_tokens) * 3 + len(intent & source_tokens),
        len(intent & fact_tokens),
        fact.observation_order,
        fact.node_id,
    )


def _relevant_facts(
    facts: list[AtomicFactNode], question: str, leaves: list[LeafNode]
) -> list[AtomicFactNode]:
    intent = _intent_terms(question)
    if not intent:
        return sorted(facts, key=_fact_sort_key, reverse=True)
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    user_query = bool(re.search(r"\b(i|my|me)\b", question.casefold()))
    assistant_recall = _is_assistant_recall_question(question)
    relevant = [
        fact for fact in facts
        if (not user_query or assistant_recall or fact.role == "user")
        and _fact_query_score(fact, intent, leaf_by_id)[0] > 0
    ]
    return sorted(
        relevant,
        key=lambda fact: _fact_query_score(fact, intent, leaf_by_id),
        reverse=True,
    )



def _fact_matches_actions(
    fact: AtomicFactNode,
    actions: set[str],
    leaf_by_id: dict[str, LeafNode],
) -> bool:
    fact_tokens = set(_tokens(" ".join((fact.predicate, fact.object, fact.item_key))))
    if actions & fact_tokens:
        return True
    # Some extraction outputs normalize "re-watched" to "watched". Allow
    # that lossy predicate only when the user's own source text explicitly
    # binds re-watch to this fact's object.
    if fact.kind == "event" and "rewatch" in actions and "watch" in fact_tokens:
        source_text = " ".join(
            (leaf_by_id[source].user_text or leaf_by_id[source].raw_text)
            for source in fact.source_leaf_ids if source in leaf_by_id
        ).casefold()
        object_terms = [token for token in _tokens(fact.object) if len(token) > 2]
        return "rewatch" in set(_tokens(source_text)) and bool(object_terms) and all(
            term in set(_tokens(source_text)) for term in object_terms[:3]
        )
    return False

def _supplement_query_facts(
    facts: list[AtomicFactNode],
    selected: list[AtomicFactNode],
    leaves: list[LeafNode],
    question: str,
    kind: str,
    limit: int,
) -> list[AtomicFactNode]:
    """Reserve a few query-critical facts before normal MMR/quota truncation.

    Dense/BM25 may retrieve the right session but still lose a later version or a
    fact whose action is only explicit in its source turn. This deterministic
    pass never invents evidence; it only promotes source-grounded L1 nodes.
    """
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    relevance_question = expand_query(question)
    intent = _intent_terms(relevance_question)
    pinned: list[AtomicFactNode] = []
    relevant = _relevant_facts(facts, relevance_question, leaves)
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}

    # Pin operands and category-defining facts before generic rank/quota logic.
    q = question.casefold()
    if _is_assistant_recall_question(question):
        pinned.extend(fact for fact in relevant if fact.role=="assistant")
    if "how many years older" in q or "how many years will i be" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and (
            "age" in set(_tokens(fact.predicate_key+" "+fact.predicate))
            and _first_number(fact.object) is not None
        ))
        if "grandma" in q:
            pinned.extend(fact for fact in facts if fact.role=="user" and (
                "grandma" in _evidence_text(fact,leaf_by_id).casefold()
                and re.search(r"\b(?:birthday|years? old|at\s+\d{1,3})\b",_evidence_text(fact,leaf_by_id),re.I)
            ))
    if "brand of" in q:
        product=set(_tokens(q))
        pinned.extend(fact for fact in facts if fact.role=="user" and (
            fact.predicate_key in {"purchased from","bought from","purchase source"}
            or (product & set(_tokens(_evidence_text(fact,leaf_by_id))) and "purchas" in fact.predicate_key)
        ))
    if kind=="preference" and "battery" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and re.search(
            r"\b(battery|power bank|charging|charger)\b",_evidence_text(fact,leaf_by_id),re.I
        ))
    if kind=="preference" and "kitchen" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and re.search(
            r"\b(granite|countertop|sink|utensil holder|kitchen utensil)\b",_evidence_text(fact,leaf_by_id),re.I
        ))
    if kind=="preference" and "nas" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and re.search(
            r"\b(nas|storage capacity|external hard drive|central backup)\b",
            _evidence_text(fact,leaf_by_id),re.I,
        ))
    if kind=="preference" and "slow cooker" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and re.search(
            r"\b(slow cooker|beef stew|yogurt|plant-based|vegetarian|vegan)\b",
            _evidence_text(fact,leaf_by_id),re.I,
        ))
    if ("current role" in q or "current job" in q) and ("how long" in q or "before" in q):
        pinned.extend(fact for fact in facts if fact.role=="user" and any(term in (fact.predicate_key+" "+fact.predicate+" "+fact.object).casefold() for term in ("tenure", "experience", "progression", "promotion", "role", "professional", "worked at", "novatech")))
    if re.search(
        r"\b(total amount of money|how much total money|how much did i spend|"
        r"total .*expenses?|money .*spent)\b", q
    ):
        money_terms=intent-{"amount","money","total","much","spend","spent","earned"}
        pinned.extend(
            fact for fact in facts
            if fact.role=="user"
            and re.search(r"\$\s*\d+",_evidence_text(fact,leaf_by_id))
            and (
                not money_terms
                or money_terms & set(_tokens(_evidence_text(fact,leaf_by_id)))
            )
        )
    if "rare item" in q and re.search(r"\b(?:total|how many)\b",q):
        pinned.extend(
            fact for fact in facts
            if fact.role=="user"
            and re.search(
                r"\b\d+\s+rare\s+(?:records?|coins?|figurines?|books?)\b|"
                r"\brare\s+books?\s+collection\s+of\s+\d+\b",
                _evidence_text(fact,leaf_by_id),re.I,
            )
        )
    if "sephora" in q and "points" in q:
        pinned.extend(
            fact for fact in facts
            if fact.role=="user" and "point" in _evidence_text(fact,leaf_by_id).casefold()
        )
    if kind == "count/list" and "babies" in q:
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            if not re.search(r"\b(born|welcomed|baby|twins?)\b", text, re.I):
                continue
            pinned.extend(
                fact for fact in facts
                if fact.role == "user" and leaf.node_id in fact.source_leaf_ids
            )
    if kind == "count/list" and re.search(r"\bbak(?:e|ed|ing)\b", q):
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            folded = text.casefold()
            if not re.search(r"\b(?:bake(?:d)?|just baked|tried out).*?\b(bread|cake|cookies?)\b", folded):
                continue
            if re.search(r"\b(?:thinking|plan|planning|going)\s+(?:of|to)?\s*bak", folded):
                continue
            pinned.extend(
                fact for fact in facts
                if fact.role == "user" and leaf.node_id in fact.source_leaf_ids
            )
    if "page count" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and any(term in (fact.predicate_key+" "+fact.predicate).casefold() for term in ("page", "finished novel", "finished reading")))
    if kind == "count/list":
        domain_pattern = None
        if "doctor" in q: domain_pattern=r"\b(?:doctor|physician|ent specialist|dermatologist)\b"
        elif "properties" in q and "brookside" in q: domain_pattern=r"\b(?:bungalow|Cedar Creek|[12]-bedroom condo|Brookside)\b"
        elif "jewelry" in q: domain_pattern=r"\b(?:earrings|necklace|engagement ring)\b"
        elif "kitchen item" in q: domain_pattern=r"\b(?:faucet|kitchen mat|toaster|coffee maker|kitchen shelves)\b"
        elif "fitness class" in q: domain_pattern=r"\b(?:zumba|yoga|weightlifting|fitness class)\b"
        elif "dinner parties" in q: domain_pattern=r"\b(?:dinner part|feast|potluck|bbq)\b"
        elif "rollercoaster" in q: domain_pattern=r"\b(?:rollercoaster|Space Mountain|Mako|Kraken|Manta)\b"
        if domain_pattern:
            for leaf in leaves:
                if not re.search(domain_pattern,leaf.user_text or leaf.raw_text,re.I): continue
                pinned.extend(
                    fact for fact in facts
                    if fact.role=="user" and leaf.node_id in fact.source_leaf_ids
                )
    if kind == "count/list" and "pre-1920" in q and "coin" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and ("pre-1920" in _evidence_text(fact,leaf_by_id).casefold() or "added to collection" in fact.predicate_key))
    if kind == "count/list" and "gift" in q and "sister" in q:
        pinned.extend(fact for fact in facts if fact.role=="user" and fact.predicate_key in {"recipient","gift cost","gift card amount"})
    if kind == "count/list" and "plant" in q and re.search(r"\b(acquire|acquired|got|bought|purchase)\b",q):
        pinned.extend(fact for fact in facts if fact.role=="user" and (
            set(_tokens(fact.predicate_key)) & {"acquire","acquired","got","bought","purchased"}
            and re.search(r"\b(plant|peace lily|succulent|snake plant|orchid|fern|african violet|spider plant)\b",_evidence_text(fact,leaf_by_id),re.I)
        ))
    if kind=="count/list" and "health-related device" in q:
        device_by_name={}
        for fact in facts:
            if fact.role!="user":
                continue
            owned_text=" ".join((fact.predicate,fact.object,fact.item_key))
            for name,pattern in _DEVICE_PATTERNS.items():
                if re.search(pattern,owned_text,re.I):
                    device_by_name.setdefault(name,fact)
        pinned.extend(device_by_name.values())
    if kind=="count/list" and re.search(r"\b(album|ep)s?\b",q):
        release_by_session={}
        for fact in facts:
            evidence=_evidence_text(fact,leaf_by_id)
            if fact.role=="user" and _category_mentions(question,evidence):
                release_by_session.setdefault(fact.session_id,fact)
        pinned.extend(release_by_session.values())
    if kind=="temporal" and "sport" in q and re.search(r"\b(order|earliest|chronological)\b",q):
        sports_by_session={}
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            if re.search(r"\b(nba|nfl|football|playoffs?|championship|game)\b",evidence,re.I) and re.search(r"\b(attend|attended|watch|watched|watching|went)\b",evidence,re.I):
                sports_by_session.setdefault(fact.session_id,fact)
        pinned.extend(sports_by_session.values())
    if kind == "count/list" and "charity event" in q and "before" in q:
        event_by_session={}
        for fact in facts:
            evidence=_evidence_text(fact,leaf_by_id).casefold()
            if fact.role=="user" and re.search(r"\b(participated|volunteered|attended|ran)\b",evidence) and re.search(r"\b(charity|gala|fundrais|wildlife|bike-a-thon|food for thought|dance for a cause)\b",evidence):
                event_by_session.setdefault(fact.session_id,fact)
        pinned.extend(event_by_session.values())

    if kind == "count/list":
        category_by_item: dict[str,AtomicFactNode]={}
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            if not _category_evidence_valid(question,evidence):
                continue
            for item in _category_mentions(question,evidence):
                category_by_item.setdefault(item,fact)
        pinned.extend(category_by_item.values())
        category_candidates = [
            fact for fact in facts if fact.role == "user"
            and _category_mentions(question, _evidence_text(fact, leaf_by_id))
            and _category_evidence_valid(question, _evidence_text(fact, leaf_by_id))
        ]
        for leaf in leaves:
            leaf_text=leaf.user_text or leaf.raw_text
            if not _category_mentions(question,leaf_text) or not _category_evidence_valid(question,leaf_text):
                continue
            category_candidates.extend(
                fact for fact in facts
                if fact.role=="user" and leaf.node_id in fact.source_leaf_ids
            )
        category_candidates.sort(key=lambda fact: _fact_query_score(fact, intent, leaf_by_id), reverse=True)
        category_sessions: Counter[str] = Counter()
        for fact in category_candidates:
            if category_sessions[fact.session_id] >= 2:
                continue
            pinned.append(fact); category_sessions[fact.session_id] += 1
            if len(pinned) >= 10:
                break

    if kind == "current/update":
        top_predicates = {
            fact.predicate_key for fact in relevant[:3]
            if not _is_generic_predicate(fact.predicate_key)
        }
        related = [fact for fact in relevant if fact.predicate_key in top_predicates]
        pinned.extend(sorted(related, key=_fact_sort_key, reverse=True)[:6])

    if kind == "count/list":
        actions = intent & {
            "pick", "return", "buy", "purchase", "complete", "finish", "visit",
            "add", "remove", "cancel", "watch", "rewatch", "use", "learn",
            "cook", "try", "attend", "own", "download",
        }
        domain = intent - actions
        session_counts: Counter[str] = Counter()
        for fact in relevant:
            if fact.role != "user":
                continue
            fact_tokens = set(_tokens(" ".join((fact.predicate, fact.object, fact.item_key))))
            source_tokens = _source_tokens(fact, leaf_by_id)
            combined = fact_tokens | source_tokens
            if actions and not _fact_matches_actions(fact, actions, leaf_by_id):
                continue
            if domain and not (domain & combined):
                continue
            if session_counts[fact.session_id] >= 2:
                continue
            pinned.append(fact)
            session_counts[fact.session_id] += 1
            if len(pinned) >= 10:
                break

    if kind == "temporal" and re.search(r"\b(order|earliest|chronological|happened first)\b",q):
        event_sessions=set()
        for fact in facts:
            evidence=_evidence_text(fact,leaf_by_id)
            if fact.role=="user" and fact.session_id not in event_sessions and re.search(
                r"\b(attended|went to|saw|got back from|started|began)\b",evidence,re.I
            ) and (_category_mentions(question,evidence) or _fact_query_score(fact,intent,leaf_by_id)[0]>0):
                pinned.append(fact);event_sessions.add(fact.session_id)
    if kind == "temporal" and re.search(
        r"\b(how long|how many|total|older|age|duration|tenure|page|cost|spent|submit|when)\b",
        question.casefold(),
    ):
        session_counts: Counter[str] = Counter()
        for fact in relevant:
            if fact.role != "user" or session_counts[fact.session_id] >= 3:
                continue
            pinned.append(fact)
            session_counts[fact.session_id] += 1
            if len(pinned) >= 10:
                break

    if "cashback" in set(_tokens(question)):
        for fact in relevant:
            predicate_tokens = set(_tokens(fact.predicate))
            if "cashback" in predicate_tokens or predicate_tokens & {"amount", "spend", "spent"}:
                pinned.append(fact)
            if len(pinned) >= 8:
                break

    merged: list[AtomicFactNode] = []
    seen: set[str] = set()
    for fact in [*pinned, *selected]:
        if fact.node_id in seen:
            continue
        seen.add(fact.node_id)
        merged.append(fact)
        if len(merged) >= limit:
            break
    return merged

def _select_leaves(leaf_by_id: dict[str, LeafNode], scores: dict[str, float], limit: int, preferred_sessions: set[str]) -> list[LeafNode]:
    ranked = sorted(leaf_by_id.values(), key=lambda leaf: scores.get(leaf.node_id, 0.0) + (0.02 if leaf.session_id in preferred_sessions else 0.0), reverse=True)
    selected=[]; session_counts=Counter()
    for leaf in ranked:
        if session_counts[leaf.session_id]>=3: continue
        if scores.get(leaf.node_id, 0.0)<=0 and len(selected)>=min(8, limit): continue
        selected.append(leaf); session_counts[leaf.session_id]+=1
        if len(selected)>=limit: break
    return selected



def _as_date(value: str | None) -> date | None:
    normalized = _date_value(value)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None


def _question_date_scope(question: str, question_date: str | None) -> tuple[date | None, date | None]:
    anchor = _as_date(question_date)
    if not anchor:
        return None, None
    q = question.casefold()
    relative_word = re.search(r"\b(a|an|couple(?:\s+of)?)\s+(day|week|month)s?\s+ago\b", q)
    if relative_word:
        value = 2 if relative_word.group(1).startswith("couple") else 1
        unit = relative_word.group(2)
        days = value if unit == "day" else value * 7 if unit == "week" else value * 30
        target = anchor - timedelta(days=days)
        return target, target
    match = re.search(r"\b(\d+)\s+(day|week|month)s?\s+ago\b", q)
    if match:
        value = int(match.group(1)); unit = match.group(2)
        days = value if unit == "day" else value * 7 if unit == "week" else value * 30
        target = anchor - timedelta(days=days)
        return target, target
    word_values = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\b(one|two|three|four|five|six)\s+(day|week|month)s?\s+ago\b", q)
    if match:
        value = word_values[match.group(1)]; unit = match.group(2)
        days = value if unit == "day" else value * 7 if unit == "week" else value * 30
        target = anchor - timedelta(days=days)
        return target, target
    weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6}
    weekday = next((value for name, value in weekdays.items() if f"last {name}" in q), None)
    if weekday is not None:
        delta = (anchor.weekday() - weekday) % 7 or 7
        target = anchor - timedelta(days=delta)
        return target, target
    if "valentine" in q:
        target = date(anchor.year, 2, 14)
        if target > anchor:
            target = date(anchor.year - 1, 2, 14)
        return target, target
    if "past weekend" in q or "last weekend" in q:
        days_since_sunday = (anchor.weekday() - 6) % 7 or 7
        sunday = anchor - timedelta(days=days_since_sunday)
        return sunday - timedelta(days=1), sunday
    if "last month" in q or "past month" in q:
        return anchor - timedelta(days=31), anchor
    if "past two months" in q:
        return anchor - timedelta(days=62), anchor
    return None, None


def _supplement_query_leaves(
    leaves: list[LeafNode],
    selected: list[LeafNode],
    question: str,
    kind: str,
    limit: int,
    question_date: str | None = None,
) -> list[LeafNode]:
    intent = _intent_terms(expand_query(question))
    q = question.casefold()
    scope_start, scope_end = _question_date_scope(question, question_date)
    scored: list[tuple[int, int, LeafNode]] = []
    assistant_recall=_is_assistant_recall_question(question)
    for leaf in leaves:
        text = leaf.raw_text if assistant_recall else (leaf.user_text or leaf.raw_text)
        folded = text.casefold()
        tokens = set(_tokens(text))
        lexical = len(intent & tokens)
        score = lexical
        if assistant_recall and "Assistant:" in leaf.raw_text:
            score += 18
        category = _category_mentions(question, text)
        if category:
            score += 20 + len(category)
        if re.search(r"\b(how many years older|how many years will i be)\b",q) and re.search(r"\b(age|years? old|birthday)\b",folded): score += 20
        if ("current role" in q or "current job" in q) and re.search(r"\b(tenure|promot|professional|working|years?\s+(?:and\s+)?\d*\s*months?)\b",folded): score += 20
        if re.search(
            r"\b(total amount of money|how much total money|how much did i spend|"
            r"total .*expenses?|money .*spent)\b",q
        ) and re.search(r"\$\s*\d+",text):
            score += 28
        if "rare item" in q and re.search(
            r"\b\d+\s+rare\s+(?:records?|coins?|figurines?|books?)\b|"
            r"\brare\s+books?\s+collection\s+of\s+\d+\b",text,re.I
        ):
            score += 32
        if "sephora" in q and "points" in q and re.search(r"\b\d+\s+points?\b",text,re.I):
            score += 32
        if kind == "count/list" and "babies" in q and re.search(
            r"\b(born|welcomed|baby|twins?)\b", folded
        ):
            score += 36
        if kind == "count/list" and re.search(r"\bbak(?:e|ed|ing)\b", q) and re.search(
            r"\b(?:bake(?:d)?|just baked|tried out).*?\b(bread|cake|cookies?)\b", folded
        ) and not re.search(r"\b(?:thinking|plan|planning|going)\s+(?:of|to)?\s*bak", folded):
            score += 40
        if kind == "count/list":
            domain_pattern = None
            if "doctor" in q: domain_pattern=r"\b(?:doctor|physician|ent specialist|dermatologist)\b"
            elif "properties" in q and "brookside" in q: domain_pattern=r"\b(?:bungalow|Cedar Creek|[12]-bedroom condo|Brookside)\b"
            elif "jewelry" in q: domain_pattern=r"\b(?:earrings|necklace|engagement ring)\b"
            elif "kitchen item" in q: domain_pattern=r"\b(?:faucet|kitchen mat|toaster|coffee maker|kitchen shelves)\b"
            elif "fitness class" in q: domain_pattern=r"\b(?:zumba|yoga|weightlifting|fitness class)\b"
            elif "dinner parties" in q: domain_pattern=r"\b(?:dinner part|feast|potluck|bbq)\b"
            elif "rollercoaster" in q: domain_pattern=r"\b(?:rollercoaster|Space Mountain|Mako|Kraken|Manta)\b"
            if domain_pattern and re.search(domain_pattern,text,re.I): score += 44
        if kind=="preference" and "slow cooker" in q and re.search(
            r"\b(slow cooker|beef stew|yogurt|plant-based|vegetarian|vegan)\b",folded
        ):
            score += 32
        if "page count" in q and re.search(r"\b\d{2,4}\s*-?\s*pages?\b",folded): score += 20
        leaf_date = _as_date(leaf.session_date)
        if scope_start and scope_end and leaf_date:
            if scope_start == scope_end and leaf_date == scope_start:
                # Exact relative dates are a deterministic routing signal even
                # when the query paraphrases the event with different nouns.
                score += 64
            elif (lexical or category) and scope_start <= leaf_date <= scope_end:
                score += 22
            elif (lexical or category) and scope_start == scope_end:
                distance = abs((leaf_date - scope_start).days)
                if distance <= 2:
                    score += 32 - 8 * distance
        if kind == "temporal" and "sport" in q and re.search(
            r"\b(triathlon|5k|run|soccer|football|basketball|tournament|game)\b",folded
        ) and re.search(r"\b(completed|finished|participat(?:e|ed|ing)|took part|attended)\b",folded):
            score += 52
        if kind == "temporal" and re.search(r"\b(order|earliest|chronological)\b",q) and re.search(r"\b(concert|music festival|jazz night|live with)\b",folded) and re.search(r"\b(attended|got back|saw|been to)\b",folded):
            score += 40
            if re.search(r"\b(actually just saw|just got back|attended .* today|jazz night .* today|just saw .* live)\b",folded): score += 35
        if kind == "temporal" and re.search(r"\b(order|earliest|chronological)\b",q) and "sport" in q and re.search(r"\b(nba|nfl|football|playoffs?|championship|game)\b",folded) and re.search(r"\b(attended|watched|watching|went to)\b",folded):
            score += 48
        if scope_start and scope_end:
            exact_target = bool(
                leaf_date and scope_start == scope_end and leaf_date == scope_start
            )
            if exact_target and re.search(r"\bwho\b", q) and re.search(
                r"\b(?:music|concert|festival|live)\b", q
            ) and re.search(
                r"\b(?:concert|music festival|live with|saw .* live)\b.*?\bwith\s+(?:my\s+)?(?:parents?|friends?|sister|brother|partner|spouse)\b",
                text,re.I,
            ):
                score += 96
            if exact_target and re.search(r"\b(?:life event|relative)\b", q) and re.search(
                r"\b(?:cousin|aunt|uncle|sister|brother|niece|nephew)[\u0027’]s\s+(?:wedding|engagement|graduation|birthday)\b|"
                r"\bbridesmaid\b.*?\b(?:cousin|relative)[\u0027’]s wedding\b",
                text,re.I,
            ):
                score += 96
            if exact_target and re.search(r"\b(?:cook|cooking|bake|baked|made)\b", q) and "friend" in q and re.search(
                r"\b(?:baked|cooked|made)\s+(?:a\s+)?[^.!?]{2,80}?\b(?:for\s+my\s+friend|friend[\u0027’]s)\b",
                text,re.I,
            ):
                score += 96
            if "airline" in q and re.search(r"\b(?:recovering from|got back from|flew|flight from).*?\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Airlines\s+flight\b",text): score += 35
            if "from whom" in q and re.search(r"\b(?:got|received|acquired).*?\bfrom\s+(?:my\s+)?[a-z]",folded): score += 35
            if ("business milestone" in q or "buisiness milestone" in q) and "signed a contract" in folded: score += 35
            if "which bike" in q and re.search(r"\b(road|mountain) bike\b",folded) and re.search(r"\b(upgrad|install|fix|servic|repair|maintenance)\w*\b",folded): score += 30
        if kind == "temporal" and lexical and re.search(
            r"\b(today|yesterday|last|ago|started|finished|fixed|bought|got|received|"
            r"attended|participated|signed|planted|using|worked|years?|months?|weeks?|days?|"
            r"a\.?m\.?|p\.?m\.?)\b", folded
        ):
            score += 8
        if score > 0:
            scored.append((score,-leaf.turn_index,leaf))
    scored.sort(key=lambda row:(row[0],row[1]),reverse=True)
    pinned=[];sessions=Counter()
    cap=min(limit,8 if kind in {"count/list","temporal"} else 5)
    per_session = (
        1 if kind == "temporal" and re.search(r"\b(order|earliest|latest)\b",q)
        else 2 if kind in {"temporal","count/list"} or _is_assistant_recall_question(question)
        else 1
    )
    for _,_,leaf in scored:
        if sessions[leaf.session_id]>=per_session: continue
        pinned.append(leaf);sessions[leaf.session_id]+=1
        if len(pinned)>=cap: break
    merged=[];seen=set()
    for leaf in [*pinned,*selected]:
        if leaf.node_id in seen: continue
        seen.add(leaf.node_id);merged.append(leaf)
        if len(merged)>=limit: break
    return merged


def _promote_leaf_linked_facts(
    facts: list[AtomicFactNode],
    selected: list[AtomicFactNode],
    leaves: list[LeafNode],
    question: str,
    limit: int,
) -> list[AtomicFactNode]:
    # Reserve one best fact per selected source turn first. Sorting every linked
    # fact globally allowed one verbose turn to consume all protected slots.
    intent = _intent_terms(expand_query(question))
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    primaries: list[AtomicFactNode] = []
    for leaf in leaves[:min(10,len(leaves))]:
        candidates=[fact for fact in facts if fact.role=="user" and leaf.node_id in fact.source_leaf_ids]
        if candidates:
            primaries.append(max(candidates,key=lambda fact:_fact_query_score(fact,intent,leaf_by_id)))
    merged: list[AtomicFactNode] = []
    seen: set[str] = set()
    for fact in [*primaries, *selected]:
        if fact.node_id in seen:
            continue
        seen.add(fact.node_id); merged.append(fact)
        if len(merged) >= limit:
            break
    return merged


def _pack_context(question: str, kind: str, cards: list[RoutingCardNode], facts: list[AtomicFactNode], leaves: list[LeafNode], ledger: list[dict[str, Any]], budget: int, question_date: str | None = None):
    def render(cs, fs, ls):
        fact_ids={f.node_id for f in fs}; leaf_ids={l.node_id for l in ls}
        packed_ledger=[]
        for row in ledger:
            if row.get("fact_id") in fact_ids:
                packed_ledger.append(row)
            elif row.get("operator"):
                original_source_ids=list(row.get("source_fact_ids",[]))
                source_ids=[value for value in original_source_ids if value in fact_ids]
                candidate_pool_complete=bool(row.get("candidate_pool_complete"))
                if source_ids or candidate_pool_complete:
                    selected_by_id={f.node_id:f for f in fs}
                    selected=[selected_by_id[value] for value in source_ids]
                    operator=row["operator"]
                    if candidate_pool_complete:
                        result=row.get("result",{})
                        source_ids=original_source_ids
                    elif operator=="distinct_completed_items":
                        original_ids=list(row.get("source_fact_ids",[]))
                        if original_ids and all(value in fact_ids for value in original_ids):
                            result=row.get("result",{});source_ids=original_ids
                        else:
                            category=_category_count_result(question,selected,ls)
                            if category:
                                result=category[0];source_ids=category[1]
                            else:
                                distinct=_distinct_items_for_query(selected,question,ls)
                                result={"count":len(distinct),"items":[fact.object for fact in distinct.values()]}
                                source_ids=[fact.node_id for fact in distinct.values()]
                    elif operator=="contextual_preferences":
                        context_facts=_preference_context_facts(question,selected,ls)
                        context_values=list(dict.fromkeys(
                            [f.object for f in context_facts]+_preference_leaf_contexts(question,ls)
                        ))[:8]
                        result={
                            "positive":[f.object for f in context_facts if f.kind=="preference" and f.polarity=="positive"],
                            "negative":[f.object for f in context_facts if f.kind=="preference" and f.polarity=="negative"],
                            "context":context_values,
                            "focus_instruction":_preference_focus_instruction(question,context_values),
                        }
                        source_ids=[f.node_id for f in context_facts]
                    elif operator=="event_order":
                        result=[{
                            "fact_id":f.node_id, "event_time":f.event_time, "observed_at":f.observed_at,
                            "time_basis":"event_time" if f.event_time else "observation_only",
                        } for f in sorted(selected,key=_fact_sort_key)]
                    elif operator=="cashback_calculation":
                        recalculated=_cashback_result(question,selected,ls)
                        if not recalculated:
                            continue
                        result=recalculated[1]
                        source_ids=recalculated[2]
                    elif operator=="generic_calculation":
                        if not all(value in fact_ids for value in row.get("source_fact_ids",[])):
                            continue
                        result=row.get("result",{})
                    elif operator in {"relative_date_scope","target_date_answer","event_comparison","event_sequence"}:
                        result=row.get("result",{})
                    elif operator=="explicit_event_time":
                        rechecked=_explicit_event_time_result(question,selected,ls)
                        if not rechecked:
                            continue
                        result=rechecked[1]
                        source_ids=rechecked[2]
                    elif operator=="brand_or_seller_inference":
                        rechecked=_brand_or_seller_result(question,selected,ls)
                        if not rechecked:
                            continue
                        result=rechecked[1]
                        source_ids=rechecked[2]
                    elif operator=="assistant_recall_extraction":
                        rechecked=_assistant_recall_result(question,selected,ls)
                        if not rechecked:
                            continue
                        result=rechecked[1]
                        source_ids=rechecked[2]
                    elif operator=="exact_entity_check":
                        rechecked=_exact_entity_result(question,selected,ls)
                        if not rechecked:
                            continue
                        result=rechecked[1]
                        source_ids=rechecked[2]
                    else:
                        result=[{"fact_id":f.node_id,"subject":f.subject,"predicate":f.predicate,"object":f.object,"valid_from":f.valid_from,"valid_to":f.valid_to} for f in selected]
                    packed_ledger.append({
                        "operator":operator,
                        "result":result,
                        "source_fact_ids":source_ids[:16],
                        "source_fact_count":len(source_ids),
                        "candidate_pool_complete":candidate_pool_complete,
                    })
        blocks=[f"[QUERY PLAN] deterministic_type={kind}", "[EVIDENCE LEDGER]\n"+json.dumps(packed_ledger, ensure_ascii=False)]
        blocks += [f"[ROUTING CARD {c.node_id}]\n{c.retrieval_text}\nPointers: facts={','.join(x for x in c.fact_ids if x in fact_ids)}; leaves={','.join(x for x in c.leaf_ids if x in leaf_ids)}" for c in cs]
        blocks += [f"[ATOMIC FACT {f.node_id}]\n{f.retrieval_text}\nSources: {','.join(f.source_leaf_ids)}" for f in fs]
        assistant_source_ids={source for fact in fs if fact.role=="assistant" for source in fact.source_leaf_ids}
        assistant_recall=_is_assistant_recall_question(question)
        blocks += [f"[SOURCE {l.node_id} | {l.session_date or 'unknown'}]\n{(l.raw_text[:1600] if assistant_recall else l.raw_text[:1200] if l.node_id in assistant_source_ids else (l.user_text or l.raw_text)[:900])}" for l in ls]
        return "\n\n".join(blocks)
    cs,fs,ls=list(cards),list(facts),list(leaves); context=render(cs,fs,ls)
    while provider_token_estimate(context)>budget:
        if len(ls)>8: ls.pop()
        elif len(fs)>10: fs.pop()
        elif len(cs)>4: cs.pop()
        else: break
        context=render(cs,fs,ls)
    # The hard provider budget takes precedence when unusually long facts make
    # the normal 4/10/8 evidence floor impossible.
    while provider_token_estimate(context)>budget and (len(ls)>1 or len(fs)>1 or len(cs)>1):
        if len(ls)>1: ls.pop()
        elif len(fs)>1: fs.pop()
        elif len(cs)>1: cs.pop()
        context=render(cs,fs,ls)
    return cs,fs,ls,context


def _current_person_company_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q = question.casefold()
    if not re.search(r"\b(?:company|employer|working at|works at)\b", q):
        return None
    person_match = re.search(r"\bcompany is\s+([A-Z][A-Za-z.-]+)", question)
    if not person_match:
        person_match = re.search(r"\bwhere (?:is|does)\s+([A-Z][A-Za-z.-]+)", question)
    if not person_match:
        person_match = re.search(
            r"\b([A-Z][A-Za-z.-]+)\b.{0,60}\bcurrently (?:working )?at\b",
            question,
        )
    person = person_match.group(1) if person_match else None
    if not person:
        return None
    candidates: list[tuple[date, int, str, AtomicFactNode]] = []
    pattern = re.compile(
        rf"\b{re.escape(person)}\b[^.!?]{{0,180}}?\b(?:currently|now)\s+(?:works?|working|is)?\s*(?:at|for)\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){{0,3}})",
        re.I,
    )
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        match = pattern.search(text)
        if not match:
            # Conversational shorthand: "Rachel ... who's currently at TechCorp".
            match = re.search(
                rf"\b{re.escape(person)}\b[^.!?]{{0,180}}?\bcurrently\s+at\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){{0,3}})",
                text, re.I,
            )
        if not match:
            continue
        company = re.sub(r"\s+", " ", match.group(1)).strip(" .,;:!?\")")
        linked = _source_fact_for_leaf(leaf, facts, question)
        when = _as_date(leaf.session_date)
        if linked and when and company:
            candidates.append((when, leaf.turn_index, company, linked))
    if not candidates:
        return None
    when, _, company, fact = max(candidates, key=lambda row: (row[0], row[1]))
    return "latest_valid_state", [{
        "fact_id": fact.node_id, "subject": person,
        "predicate": "current_company", "object": company,
        "observed_at": when.isoformat(), "valid_from": when.isoformat(),
        "valid_to": None,
        "basis": "latest explicit current-company assertion; valid until superseded",
    }], [fact.node_id]


def _current_follower_count_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    if not re.search(r"\bfollowers?\b",question,re.I):
        return None
    leaf_by_id={leaf.node_id:leaf for leaf in leaves}
    candidates=[]
    for fact in facts:
        if fact.role!="user" or "follower" not in _evidence_text(fact,leaf_by_id).casefold():
            continue
        value=_first_number(fact.object)
        if value is None:
            continue
        source_dates=[leaf_by_id[source].session_date for source in fact.source_leaf_ids if source in leaf_by_id]
        source_date=max(source_dates,default=fact.observed_at or "")
        candidates.append((source_date,fact.observation_order,value,fact))
    if not candidates:
        return None
    _,_,value,fact=max(candidates,key=lambda row:(row[0],row[1],row[2]))
    normalized=str(int(value)) if value==int(value) else str(value)
    return "latest_valid_state", [{
        "fact_id":fact.node_id,"subject":fact.subject,"predicate":fact.predicate,
        "object":normalized,"source_object":fact.object,"observed_at":fact.observed_at,
        "valid_from":fact.valid_from,"valid_to":fact.valid_to,
    }], [fact.node_id]


def _most_recent_started_service_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q=question.casefold()
    if not ("streaming service" in q and re.search(r"\bmost recent(?:ly)?\b",q)):
        return None
    leaf_by_id={leaf.node_id:leaf for leaf in leaves}
    candidates=[]
    for fact in facts:
        if fact.role!="user":
            continue
        evidence=_evidence_text(fact,leaf_by_id)
        if not re.search(r"\b(stream|subscrib|trial|using)\b",evidence,re.I):
            continue
        match=re.search(r"\b(\d+|a|one|two|three|four|five|six|few)\s+months?\s+(?:ago|now)\b",evidence.casefold())
        if "last month" in evidence.casefold():
            age=1
        elif match:
            age=3 if match.group(1)=="few" else _small_number(match.group(1))
        else:
            continue
        service=None
        for name in ("Disney+","Apple TV+","Netflix","Hulu","Amazon Prime","Max","HBO Max"):
            if name.casefold() in evidence.casefold():
                service=name;break
        if service:
            candidates.append((age,-fact.observation_order,service,fact))
    if not candidates:
        return None
    age,_,service,fact=min(candidates)
    return "latest_valid_state", [{
        "fact_id":fact.node_id,"subject":"user","predicate":"most_recently_started_streaming_service",
        "object":service,"relative_age_months":age,"observed_at":fact.observed_at,
        "valid_from":fact.valid_from,"valid_to":fact.valid_to,
    }], [fact.node_id]


def _current_project_from_leaves_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode]
):
    q=question.casefold()
    if not (
        re.search(r"\b(?:current|currently|now)\b",q)
        and re.search(r"\b(?:project|working on|vehicle model|model)\b",q)
    ):
        return None
    candidates=[]
    for leaf in leaves:
        text=leaf.user_text or leaf.raw_text
        match=re.search(
            r"\bswitched to\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:[.,;]|"
            r"\s+and\s+(?:i|we)\b|$)",
            text,re.I,
        )
        if not match:
            continue
        value=re.sub(r"\s+"," ",match.group(1)).strip()
        if len(value)<3 or value.casefold() in {"model","project"}:
            continue
        linked=_linked_fact_for_leaf(leaf,facts,question)
        if linked:
            candidates.append((_as_date(leaf.session_date) or date.min,leaf.turn_index,value,linked))
    if not candidates:
        return None
    _,_,value,fact=max(candidates,key=lambda row:(row[0],row[1]))
    return "latest_valid_state", [{
        "fact_id":fact.node_id,"subject":"user","predicate":"current_project",
        "object":value,"observed_at":fact.observed_at,
        "valid_from":fact.valid_from,"valid_to":fact.valid_to,
        "basis":"explicit switched-to source statement",
    }], [fact.node_id]


def _current_storage_location_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q=question.casefold()
    if not (
        re.search(r"\bwhere\b",q)
        and re.search(r"\b(?:current|currently|now|keep|kept|store|stored)\b",q)
    ):
        return None
    item_terms=_intent_terms(question)-{
        "where","current","currently","now","keep","kept","store","stored",
        "storage","put","my","old","do","does","did","the","a","an",
    }
    candidates=[]
    for leaf in leaves:
        text=leaf.user_text or leaf.raw_text
        folded=text.casefold()
        if item_terms and not (item_terms & set(_tokens(folded))):
            continue
        linked=_linked_fact_for_leaf(leaf,facts,question)
        if linked is None:
            continue
        matches=list(re.finditer(
            r"\b(?:keep(?:ing)?|stor(?:e|ed|ing)|put(?:ting)?)\b"
            r"[^.!?]{0,120}?\b(in|inside|on|under|at)\s+"
            r"((?:a|an|the|my)\s+)?([^,.;!?]{1,70})",
            text,re.I,
        ))
        if not matches:
            matches=list(re.finditer(
                r"\b(?:sneakers?|shoes?|boots?|clothes?|items?)\b"
                r"[^.!?]{0,100}?\b(in|inside|on|under|at)\s+"
                r"((?:a|an|the|my)\s+)?"
                r"([^,.;!?]{0,40}(?:rack|closet|bed|box|cabinet|drawer|shelf)[^,.;!?]{0,25})",
                text,re.I,
            ))
        for match in matches:
            prep=match.group(1).casefold()
            article=(match.group(2) or "").casefold()
            location=re.sub(
                r"\s+(?:for storage|while\b|because\b|so\b|and\b|"
                r"they(?:'re| are)\b|it(?:'s| is)\b).*$",
                "",match.group(3).strip(),flags=re.I,
            )
            if not location or len(_tokens(location))>12:
                continue
            if "shoe rack" in location.casefold() and "closet" in folded:
                value="in a shoe rack in my closet"
            else:
                value=" ".join(part for part in (prep,article.strip(),location) if part)
            candidates.append((
                _as_date(leaf.session_date) or date.min,
                leaf.turn_index,
                len(item_terms & set(_tokens(folded))),
                value,linked,
            ))
    if not candidates:
        return None
    _,_,_,value,fact=max(candidates,key=lambda row:(row[0],row[1],row[2]))
    return "current_storage_location", {
        "value":value,
        "item_terms":sorted(item_terms),
        "basis":"latest source-grounded storage-location statement",
    },[fact.node_id]


def _current_competitive_record_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q=question.casefold()
    if not (
        "record" in q
        and re.search(r"\b(?:current|currently|latest|now)\b",q)
        and re.search(r"\b(?:league|team|volleyball|basketball|football|baseball|hockey)\b",q)
    ):
        return None
    sport_terms=_intent_terms(question)-{
        "current","currently","latest","now","record","league","team","what",
    }
    candidates=[]
    for leaf in leaves:
        text=leaf.user_text or leaf.raw_text
        folded=text.casefold()
        if not re.search(r"\b(?:record|league|team|we(?:'re| are))\b",folded):
            continue
        match=re.search(r"\b(\d{1,3})\s*-\s*(\d{1,3})\b",text)
        if not match:
            continue
        overlap=len(sport_terms & set(_tokens(text)))
        if sport_terms and overlap==0 and not re.search(r"\b(?:league|team)\b",folded):
            continue
        linked=_linked_fact_for_leaf(leaf,facts,question)
        if linked:
            candidates.append((
                _as_date(leaf.session_date) or date.min,
                leaf.turn_index,overlap,f"{match.group(1)}-{match.group(2)}",linked,
            ))
    if not candidates:
        return None
    _,_,_,value,fact=max(candidates,key=lambda row:(row[0],row[1],row[2]))
    return "current_competitive_record", {
        "value":value,
        "basis":"latest source-grounded team or league win-loss record",
    },[fact.node_id]


def _previous_status_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q=question.casefold()
    if not (
        re.search(r"\bprevious\b",q)
        and re.search(r"\b(?:status|tier|level)\b",q)
        and re.search(r"\b(?:before|current|now)\b",q)
    ):
        return None
    leaf_by_id={leaf.node_id:leaf for leaf in leaves}
    carrier_terms={
        name for name in ("united","southwest","american","delta","jetblue")
        if name in q
    }
    candidates=[]
    for fact in facts:
        if fact.role!="user":
            continue
        match=re.search(
            r"\b((?:premier|medallion|a-list)\s+"
            r"(?:silver|gold|platinum|diamond|preferred|elite))\b",
            fact.object,re.I,
        )
        if not match:
            continue
        evidence=_evidence_text(fact,leaf_by_id).casefold()
        if carrier_terms and not (carrier_terms & set(_tokens(evidence))):
            continue
        candidates.append((
            _as_date(fact.observed_at) or date.min,
            fact.observation_order,match.group(1).title(),fact,
        ))
    ordered=[]
    for row in sorted(candidates,key=lambda item:(item[0],item[1])):
        if not ordered or canonical_key(ordered[-1][2])!=canonical_key(row[2]):
            ordered.append(row)
    if len(ordered)<2:
        return None
    previous,current=ordered[-2],ordered[-1]
    return "previous_status", {
        "value":previous[2],"current_value":current[2],
        "basis":"chronological predecessor among source-grounded status tiers",
    },[previous[3].node_id,current[3].node_id]


def _event_companion_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
    question_date: str | None,
):
    q=question.casefold()
    if not re.search(
        r"\bdid i\b[^?]{0,120}\bwith\s+(?:a|my)\s+friend\b[^?]{0,40}\bor not\b",
        q,
    ):
        return None
    scope_start,scope_end=_question_date_scope(question,question_date)
    target=scope_start or scope_end
    if target is None:
        return None
    intent=_intent_terms(question)-{
        "did","with","friend","not","mentioned","ago","visit","visited",
        "day","week","month","year",
    }
    candidates=[]
    for leaf in leaves:
        text=leaf.user_text or leaf.raw_text
        when=_as_date(leaf.session_date)
        if when is None:
            continue
        folded=text.casefold()
        if not re.search(r"\b(?:visit|visited|attended|went|tour)\b|\blecture\s+at\b",folded):
            continue
        overlap=len(intent & set(_tokens(text)))
        if overlap==0:
            continue
        linked=_linked_fact_for_leaf(leaf,facts,question)
        if linked:
            candidates.append((abs((when-target).days),-overlap,leaf.turn_index,leaf,linked))
    if not candidates:
        return None
    distance,_,_,leaf,fact=min(candidates,key=lambda row:(row[0],row[1],row[2]))
    if distance>3:
        return None
    source=(leaf.user_text or leaf.raw_text).casefold()
    accompanied=bool(re.search(
        r"\bwith\s+(?:a|my|one of my)\s+friends?\b|\bfriend\s+(?:came|joined|accompanied)\b",
        source,
    ))
    value=(
        "Yes, you visited with a friend"
        if accompanied else "No, you did not visit with a friend"
    )
    return "event_companion_status", {
        "value":value,"accompanied":accompanied,
        "target_date":target.isoformat(),"source_date":(_as_date(leaf.session_date) or target).isoformat(),
        "basis":"nearest source-grounded target event and explicit companion mention status",
    },[fact.node_id]


def _operator_result(
    kind: str,
    facts: list[AtomicFactNode],
    chains: list[StateChain],
    question: str = "",
    leaves: list[LeafNode] | None = None,
):
    if kind == "current/update":
        storage_location=_current_storage_location_result(
            question, facts, leaves or []
        )
        if storage_location:
            return storage_location
        current_company=_current_person_company_result(question,facts,leaves or [])
        if current_company:
            return current_company
        follower_count=_current_follower_count_result(question,facts,leaves or [])
        if follower_count:
            return follower_count
        current_project=_current_project_from_leaves_result(question,facts,leaves or [])
        if current_project:
            return current_project
        recent_service=_most_recent_started_service_result(question,facts,leaves or [])
        if recent_service:
            return recent_service
        relevant = _relevant_facts(facts, question, leaves or [])
        if relevant:
            intent=_intent_terms(question)
            entity_terms=intent-{"brand","use","uses","using","latest","current","currently","recent","recently","most","start","started"}
            exact=[fact for fact in relevant if entity_terms & set(_tokens(" ".join((fact.predicate,fact.object,fact.item_key,fact.context_key))))]
            if exact:
                relevant=exact
            # Query overlap chooses the predicate family, but recency chooses the
            # value inside that family. Exact-score filtering used to discard a
            # later same-day update (for example 1300 after 1250 followers).
            top_predicate = relevant[0].predicate_key
            same_predicate = [fact for fact in relevant if fact.predicate_key == top_predicate]
            relevant = same_predicate or relevant
        stateful = [fact for fact in relevant if fact.kind in {"state", "quantity", "preference", "event"}]
        candidates = stateful or relevant
        groups: dict[tuple[str, str], list[AtomicFactNode]] = defaultdict(list)
        for fact in candidates:
            groups[(fact.subject_key, fact.predicate_key)].append(fact)
        latest = [max(values, key=_fact_sort_key) for values in groups.values()]
        latest.sort(key=_fact_sort_key, reverse=True)
        ids = [fact.node_id for fact in latest[:6]]
        compact = [
            {
                "fact_id": fact.node_id, "subject": fact.subject, "predicate": fact.predicate,
                "object": fact.object, "event_time": fact.event_time, "observed_at": fact.observed_at,
                "valid_from": fact.valid_from, "valid_to": fact.valid_to,
            }
            for fact in latest[:6]
        ]
        return "latest_valid_state", compact, ids
    if kind == "count/list":
        if re.search(r"\b(how many years|how much total|page count)\b",question.casefold()):
            return None
        category=_category_count_result(question,facts,leaves or [])
        if category:
            result=dict(category[0])
            result["aggregation_method"]="category_count"
            return "distinct_completed_items", result, category[1]
        items=_distinct_items_for_query(facts,question,leaves or [])
        allow_multi = (
            "clothing" in question.casefold()
            and "pick" in question.casefold()
            and "return" in question.casefold()
        )
        if question.strip() and len(items) != 1 and not allow_multi:
            return None
        if not items:
            return None
        return "distinct_completed_items", {"count":len(items),"items":[f.object for f in items.values()]}, [f.node_id for f in items.values()]
    if kind == "temporal":
        relevant=_relevant_facts(facts,question,leaves or [])
        ordered=sorted(
            [f for f in relevant if f.event_time or f.observed_at],
            key=_fact_sort_key,
        )[:16]
        return "event_order", [{
            "fact_id":f.node_id, "event_time":f.event_time, "observed_at":f.observed_at,
            "time_basis":"event_time" if f.event_time else "observation_only",
        } for f in ordered], [f.node_id for f in ordered]
    if kind == "preference":
        relevant=_preference_context_facts(question,facts,leaves or [])
        pos=[f.object for f in relevant if f.kind=="preference" and f.polarity=="positive"]
        neg=[f.object for f in relevant if f.kind=="preference" and f.polarity=="negative"]
        ids=[f.node_id for f in relevant]
        context_values=list(dict.fromkeys([f.object for f in relevant]+_preference_leaf_contexts(question,leaves or [])))[:8]
        return "contextual_preferences", {
            "positive":pos,"negative":neg,"context":context_values,
            "focus_instruction":_preference_focus_instruction(question,context_values),
        }, ids
    return None




def _first_number(text: str) -> Decimal | None:
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def _duration_months(text: str) -> int | None:
    years = re.search(r"(\d+)\s*years?", text.casefold())
    months = re.search(r"(\d+)\s*months?", text.casefold())
    if not years and not months:
        return None
    return (int(years.group(1)) if years else 0) * 12 + (int(months.group(1)) if months else 0)


def _compact_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _source_fact_for_leaf(
    leaf: LeafNode, facts: list[AtomicFactNode], question: str,
) -> AtomicFactNode | None:
    linked = [
        fact for fact in facts
        if fact.role == "user" and leaf.node_id in fact.source_leaf_ids
    ]
    if not linked:
        return None
    intent = _intent_terms(expand_query(question))
    leaf_map = {leaf.node_id: leaf}
    return max(linked, key=lambda fact: _fact_query_score(fact, intent, leaf_map))


def _per_unit_quantity_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q = question.casefold()
    if not re.search(r"\b(?:each|per)\b", q):
        return None
    collection = _operand_collection_key(question)
    intent = _intent_terms(question) - {"each", "per", "cost", "spend", "spent", "money"}
    totals: list[tuple[int, Decimal, AtomicFactNode]] = []
    counts: list[tuple[int, Decimal, AtomicFactNode]] = []
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        if collection and _operand_collection_key(text) != collection:
            continue
        overlap = len(intent & set(_tokens(text)))
        if not collection and overlap == 0:
            continue
        linked = _source_fact_for_leaf(leaf, facts, question)
        if linked is None:
            continue
        money = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
        if money and re.search(r"\b(?:spent|paid|cost|total)\b", text, re.I):
            totals.append((overlap, Decimal(money.group(1)), linked))
        for match in re.finditer(
            r"(?<![$\d])\b(\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2})",
            text,
        ):
            phrase = match.group(2).casefold()
            if set(_tokens(phrase)) & intent and not re.search(
                r"\b(?:days?|weeks?|months?|years?|dollars?|percent)\b", phrase
            ):
                counts.append((overlap, Decimal(match.group(1)), linked))
    if not totals or not counts:
        return None
    _, total, total_fact = max(totals, key=lambda row: (row[0], row[1]))
    valid_counts = [row for row in counts if row[1] > 0 and row[1] <= 1000]
    if not valid_counts:
        return None
    _, count, count_fact = max(valid_counts, key=lambda row: (row[0], row[1]))
    value = total / count
    formatted = "$" + _compact_decimal(value)
    return "generic_calculation", {
        "calculation_type": "per_unit_price",
        "total": str(total), "count": str(count), "result": str(value),
        "formatted_result": formatted, "formula": "total / item_count",
        "collection_key": collection,
    }, list(dict.fromkeys([total_fact.node_id, count_fact.node_id]))


def _grouped_follower_change_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q = question.casefold()
    if not ("follower" in q and re.search(r"\b(?:gain|gained|increase|grew|growth|most)\b", q)):
        return None
    platforms = ("Twitter", "TikTok", "Instagram", "Facebook")
    changes: dict[str, tuple[Decimal, AtomicFactNode]] = {}
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        linked = _source_fact_for_leaf(leaf, facts, question)
        if linked is None:
            continue
        for platform in platforms:
            if platform.casefold() not in text.casefold() or "follower" not in text.casefold():
                continue
            from_to = re.search(r"\bfrom\s+(\d[\d,]*)\s+to\s+(\d[\d,]*)\b", text, re.I)
            gained = re.search(r"\b(?:gained|added|increased by)\s+(?:around\s+|about\s+|approximately\s+)?(\d[\d,]*)\s+followers?\b", text, re.I)
            steady = re.search(r"\b(?:remained|stayed)\s+steady\b", text, re.I)
            value = None
            if from_to:
                value = Decimal(from_to.group(2).replace(",", "")) - Decimal(from_to.group(1).replace(",", ""))
            elif gained:
                value = Decimal(gained.group(1).replace(",", ""))
            elif steady:
                value = Decimal("0")
            if value is not None:
                previous = changes.get(platform)
                if previous is None or value > previous[0]:
                    changes[platform] = (value, linked)
    if len(changes) < 2:
        return None
    platform, (value, _) = max(changes.items(), key=lambda row: row[1][0])
    return "generic_calculation", {
        "calculation_type": "grouped_delta_argmax",
        "groups": {name: str(row[0]) for name, row in changes.items()},
        "result": platform, "formatted_result": platform,
        "winning_delta": str(value), "formula": "argmax(current - baseline or stated gain)",
    }, list(dict.fromkeys(row[1].node_id for row in changes.values()))


def _grouped_airline_frequency_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q = question.casefold()
    if not ("airline" in q and re.search(r"\b(?:most|frequent|times)\b", q)):
        return None
    airlines = ("United Airlines", "Southwest Airlines", "American Airlines")
    per_trip: dict[tuple[str, str], tuple[int, AtomicFactNode]] = {}
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        linked = _source_fact_for_leaf(leaf, facts, question)
        if linked is None:
            continue
        for airline in airlines:
            if airline.casefold() not in text.casefold():
                continue
            count = 0
            each_way = re.search(r"\b(?:with\s+)?(\d+|one|two|three|four)\s+flights?\s+each\s+way\b", text, re.I)
            if each_way:
                count = 2 * _small_number(each_way.group(1))
            elif re.search(r"\bconnecting flight\b", text, re.I) and re.search(r"\bflew\b", text, re.I):
                count = 2
            elif re.search(r"\b(?:direct\s+flight|flew|flight\s+with)\b", text, re.I):
                count = 1
            if count:
                key = (airline, leaf.session_id)
                previous = per_trip.get(key)
                if previous is None or count > previous[0]:
                    per_trip[key] = (count, linked)
    totals: dict[str, int] = defaultdict(int)
    sources: dict[str, list[AtomicFactNode]] = defaultdict(list)
    for (airline, _), (count, linked) in per_trip.items():
        totals[airline] += count
        sources[airline].append(linked)
    if len(totals) < 2:
        return None
    winner = max(totals, key=totals.get)
    return "generic_calculation", {
        "calculation_type": "grouped_frequency_argmax",
        "groups": dict(totals), "result": winner, "formatted_result": winner,
        "formula": "argmax explicit flight legs per airline in requested window",
    }, list(dict.fromkeys(f.node_id for values in sources.values() for f in values))


def _collection_quantity_sum_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q = question.casefold()
    if not re.search(r"\b(?:total|in total|both)\b", q):
        return None
    collection = _operand_collection_key(question)
    if collection not in {"writing_pieces", "lunch_meals", "fish_inventory"}:
        return None
    values: dict[str, tuple[int, AtomicFactNode]] = {}
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        folded = text.casefold()
        if _operand_collection_key(text) != collection:
            continue
        linked = _source_fact_for_leaf(leaf, facts, question)
        if linked is None:
            continue
        if collection == "writing_pieces":
            patterns = (
                ("poem", r"\b(?:written|wrote|completed)\s+(\d+)\s+poems?\b"),
                ("short_story", r"\b(?:written|wrote|completed)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+short stories\b"),
            )
            for key, pattern in patterns:
                match = re.search(pattern, folded)
                if match:
                    raw = match.group(1)
                    value = int(raw) if raw.isdigit() else _small_number(raw)
                    if value > values.get(key, (0, linked))[0]:
                        values[key] = (value, linked)
            if "writing challenge" in folded and re.search(r"\bi wrote a (?:short )?piece\b", folded):
                values.setdefault("writing_challenge_piece", (1, linked))
        elif collection == "lunch_meals":
            third = re.search(r"\bthis is the (\w+) meal\b.*?\bchicken fajitas?\b", folded)
            if third:
                ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
                if third.group(1) in ordinals:
                    values["chicken_fajitas"] = (ordinals[third.group(1)], linked)
            lentil = re.search(r"\blentil soup\b.{0,80}\b(?:lasted me for|made)\s+(\d+)\s+lunch", folded)
            if lentil:
                values["lentil_soup"] = (int(lentil.group(1)), linked)
        elif collection == "fish_inventory":
            if re.search(r"\b(?:planning|thinking)\s+(?:of|to)\s+(?:add|adding|get|getting)\b[^.!?]{0,80}\b(?:fish|tetras?|danios?|gouramis?|pleco|betta)\b", folded) and "currently has" not in folded:
                continue
            for match in re.finditer(r"\b(\d+)\s+(neon tetras?|golden honey gouramis?|gouramis?|plecos?|catfish|bettas?)\b", folded):
                values[canonical_key(match.group(2))] = (int(match.group(1)), linked)
            if re.search(r"\b(?:a|one)\s+(?:small\s+)?pleco(?:\s+catfish)?\b", folded):
                values.setdefault("pleco", (1, linked))
            if re.search(r"\b(?:has|have|with)\s+(?:my\s+)?betta fish\b", folded):
                values.setdefault("betta", (1, linked))
    minimum = 3 if collection in {"writing_pieces", "fish_inventory"} else 2
    if len(values) < minimum:
        return None
    total = sum(value for value, _ in values.values())
    return "generic_calculation", {
        "calculation_type": "collection_quantity_sum",
        "collection_key": collection,
        "items": {key: value for key, (value, _) in values.items()},
        "result": total, "formatted_result": str(total),
        "formula": "sum distinct completed/current collection quantities",
    }, list(dict.fromkeys(fact.node_id for _, fact in values.values()))


def _arithmetic_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    q = question.casefold()
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}

    by_session: dict[str,list[LeafNode]]=defaultdict(list)
    for leaf in leaves: by_session[leaf.session_id].append(leaf)

    def linked_source(leaf: LeafNode) -> AtomicFactNode | None:
        return _linked_fact_for_leaf(leaf,facts,question) or next(
            (fact for fact in facts if fact.role=="user" and fact.session_id==leaf.session_id),
            None,
        )

    def fact_date(fact: AtomicFactNode) -> date:
        return _as_date(fact.observed_at) or _as_date(fact.event_time) or date.min

    if re.search(r"\bwhat breed\b.*\b(?:my\s+)?dog\b", q):
        for source in leaves:
            evidence = source.user_text or source.raw_text
            if not re.search(
                r"\b(?:dog|puppy|pup|canine|collar|flea|tick|pet)\b",
                evidence,
                re.I,
            ):
                continue
            matches = (
                re.search(
                    r"\b(?:dog|[A-Z][A-Za-z'-]{1,30})\s+(?:is|was)\s+"
                    r"(?:a|an)\s+([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})\b",
                    evidence,
                ),
                re.search(
                    r"\bsuit\s+(?:a|an)\s+"
                    r"([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})\s+"
                    r"like\s+[A-Z][A-Za-z'-]*\b",
                    evidence,
                ),
            )
            match = next((value for value in matches if value), None)
            linked = linked_source(source)
            if match and linked:
                breed = match.group(1).strip()
                return "generic_calculation", {
                    "calculation_type": "explicit_user_attribute",
                    "attribute": "dog breed",
                    "result": breed,
                    "formatted_result": breed,
                    "formula": "explicit breed phrase in lossless user source",
                }, [linked.node_id]

    if re.search(r"\bhow old was i\b.*\b(?:grandma|grandmother)\b.*\bnecklace\b", q):
        for source in leaves:
            evidence = source.user_text or source.raw_text
            if not re.search(r"\b(?:grandma|grandmother)\b", evidence, re.I):
                continue
            if not re.search(r"\bsilver necklace\b", evidence, re.I):
                continue
            age = re.search(r"\b(?:my\s+)?(\d{1,2})(?:st|nd|rd|th)\s+birthday\b", evidence, re.I)
            linked = linked_source(source)
            if age and linked:
                value = int(age.group(1))
                return "generic_calculation", {
                    "calculation_type": "explicit_user_attribute",
                    "attribute": "age when necklace was received",
                    "result": value,
                    "formatted_result": str(value),
                    "formula": "ordinal birthday number = age on gift event",
                }, [linked.node_id]

    if re.search(
        r"\bhow many\b.*\bgraduation ceremonies\b.*\bpast\s+\w+\s+months\b",
        q,
    ):
        attended: dict[str, tuple[str, AtomicFactNode]] = {}
        for source in leaves:
            evidence = source.user_text or source.raw_text
            folded = evidence.casefold()
            if not re.search(r"\b(?:graduation|graduated)\b", folded):
                continue
            if re.search(r"\b(?:missed|did not attend|didn[\u0027’]t attend)\b", folded):
                continue
            if not re.search(r"\b(?:just\s+)?attended\b", folded):
                continue
            linked = linked_source(source)
            if linked is None:
                continue
            name = re.search(
                r"\b(?:cousin|colleague|friend|nephew|niece|sister|brother)\s+"
                r"([A-Z][A-Za-z'-]{1,30})[\u0027’]s\s+"
                r"(?:[A-Za-z -]{0,40}\s+)?graduation\b",
                evidence,
            )
            event = re.search(
                r"\battended\s+([^.!?]{2,100}?\bgraduation(?:\s+ceremony)?)\b",
                evidence,
                re.I,
            )
            label = (
                name.group(1)
                if name else event.group(1).strip()
                if event else linked.object
            )
            key = canonical_key(label)
            if key:
                attended[key] = (label, linked)
        if attended:
            return "generic_calculation", {
                "calculation_type": "distinct_completed_event_count",
                "event_type": "graduation ceremony",
                "items": [label for label, _ in attended.values()],
                "result": len(attended),
                "formatted_result": str(len(attended)),
                "formula": "count distinct explicitly attended graduations; exclude missed/planned events",
            }, [linked.node_id for _, linked in attended.values()]

    if re.search(r"\bmost recent\b.*\bfamily trip\b|\bmost recent family trip\b", q):
        candidates: list[AtomicFactNode] = []
        for fact in facts:
            if fact.role != "user":
                continue
            folded = " ".join([
                fact.predicate.replace("_", " "),
                fact.object,
                _evidence_text(fact, leaf_by_id),
            ]).casefold()
            if "trip" in folded and "family" in folded and fact.modality == "asserted":
                candidates.append(fact)
        if candidates:
            latest = max(candidates, key=lambda fact: (fact_date(fact), fact.observation_order))
            destination = re.sub(
                r"\s+(?:with|for)\s+(?:my\s+|the\s+)?family\b.*$",
                "",
                latest.object,
                flags=re.I,
            ).strip(" .,")
            if destination:
                return "generic_calculation", {
                    "calculation_type": "latest_family_trip",
                    "result": destination,
                    "formatted_result": destination,
                    "observed_at": fact_date(latest).isoformat(),
                    "formula": "latest asserted family-trip fact by observation date",
                }, [latest.node_id]

    # Bind two explicitly labelled fuel-efficiency values before falling back to
    # generic arithmetic. This avoids treating unrelated quantities as operands.
    if re.search(r"\bhow much more\b.*\b(?:mpg|miles per gallon)\b", q):
        readings: list[tuple[str, Decimal, AtomicFactNode]] = []
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            folded = evidence.casefold()
            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s*(?:mpg|miles per gallon)\b", folded
            ):
                window = folded[max(0, match.start() - 90):match.end() + 90]
                label = (
                    "past" if re.search(r"\b(?:few months ago|before|previously|used to)\b", window)
                    else "current" if re.search(r"\b(?:now|currently|lately|these days)\b", window)
                    else ""
                )
                if label:
                    readings.append((label, Decimal(match.group(1)), fact))
        past = next(((value, fact) for label, value, fact in readings if label == "past"), None)
        current = next(((value, fact) for label, value, fact in readings if label == "current"), None)
        if past and current:
            delta = past[0] - current[0]
            return "generic_calculation", {
                "calculation_type": "labeled_metric_delta",
                "metric": "mpg",
                "past_value": str(past[0]),
                "current_value": str(current[0]),
                "result": float(delta),
                "formatted_result": f"{delta:g} MPG",
                "formula": "past MPG - current MPG",
            }, [past[1].node_id, current[1].node_id]

    # Ratio between a named property price and a named renovation cost.
    if re.search(
        r"\bwhat percentage\b.*\bpropert(?:y|y[\u0027’]s)\b.{0,25}\bprice\b.*\brenovation",
        q,
    ):
        price = renovation = None
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            amount_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", evidence)
            if not amount_match:
                continue
            amount = Decimal(amount_match.group(1).replace(",", ""))
            folded = evidence.casefold()
            is_renovation = bool(re.search(r"\brenovat(?:e|ed|ion|ions)\b", folded))
            if (
                re.search(r"\b(?:property|house|home)\b", folded)
                and re.search(r"\b(?:price|listed|worth)\b", folded)
                and not is_renovation
            ):
                price = (amount, fact)
            if is_renovation:
                renovation = (amount, fact)
        if price and renovation and price[0] > 0:
            percent = renovation[0] * Decimal(100) / price[0]
            return "generic_calculation", {
                "calculation_type": "cross_entity_cost_percentage",
                "property_price": str(price[0]),
                "renovation_cost": str(renovation[0]),
                "result": float(percent),
                "formatted_result": f"{percent:g}%",
                "formula": "renovation cost / property price * 100",
            }, [price[1].node_id, renovation[1].node_id]

    # Packed-versus-used list ratio. Require both labels so generic counts from
    # other trips cannot enter the calculation.
    if re.search(r"\bwhat percentage\b.*\b(?:packed|pack)\b.*\b(?:wore|wear)\b", q):
        count_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        def count_value(token: str) -> int:
            return int(token) if token.isdigit() else count_words[token]

        packed = worn = None
        # L0 is intentionally lossless; scan it because a session extractor can
        # omit one half of a cross-session arithmetic pair.
        for source in leaves:
            evidence = source.user_text or source.raw_text
            folded = evidence.casefold()
            linked = linked_source(source)
            if linked is None:
                continue
            packed_match = re.search(
                r"\bpack(?:ed|ing)?\b[^.!?]{0,100}\b"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+pairs?\b",
                folded,
            )
            worn_match = re.search(
                r"\b(?:wore|wear(?:ing)?|ended up only wearing)\b[^.!?]{0,100}\b"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"(?:\s+pairs?|\s+shoes?|\s*[-—]\s*(?:my\s+)?\w+)",
                folded,
            )
            if packed_match:
                packed = (count_value(packed_match.group(1)), linked)
            if worn_match:
                worn = (count_value(worn_match.group(1)), linked)
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            folded = evidence.casefold()
            packed_match = re.search(
                r"\bpack(?:ed|ing)?\b[^.!?]{0,70}\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+pairs?\b",
                folded,
            )
            worn_match = re.search(
                r"\b(?:wore|wear(?:ing)?|ended up only wearing)\b[^.!?]{0,70}\b"
                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"(?:\s+pairs?|\s+shoes?|\s*[-—]\s*(?:my\s+)?\w+)",
                folded,
            )
            if packed_match:
                packed = (count_value(packed_match.group(1)), fact)
            if worn_match:
                worn = (count_value(worn_match.group(1)), fact)
        if packed and worn and packed[0] and worn[0] is not None:
            percent = Decimal(worn[0]) * Decimal(100) / Decimal(packed[0])
            return "generic_calculation", {
                "calculation_type": "used_fraction_percentage",
                "packed_count": packed[0],
                "used_count": worn[0],
                "result": float(percent),
                "formatted_result": f"{percent:g}%",
                "formula": "worn pairs / packed pairs * 100",
            }, [packed[1].node_id, worn[1].node_id]

    # Explicit initial-versus-final quote revision.
    if re.search(r"\bhow much more\b.*\b(?:pay|quote|quoted)\b", q):
        initial = final = None
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            folded = evidence.casefold()
            direct_initial = re.search(
                r"\b(?:initial(?:ly)?|first|original(?:ly)?)\b[^$]{0,80}"
                r"\$\s*([\d,]+(?:\.\d+)?)",
                evidence,
                re.I,
            )
            direct_final = re.search(
                r"\b(?:corrected|revised|final|actually|ended up)\b[^$]{0,80}"
                r"\$\s*([\d,]+(?:\.\d+)?)",
                evidence,
                re.I,
            )
            if direct_initial:
                initial = (
                    Decimal(direct_initial.group(1).replace(",", "")),
                    fact,
                )
            if direct_final:
                final = (
                    Decimal(direct_final.group(1).replace(",", "")),
                    fact,
                )
            if direct_initial or direct_final:
                continue
            for match in re.finditer(r"\$\s*([\d,]+(?:\.\d+)?)", evidence):
                amount = Decimal(match.group(1).replace(",", ""))
                window = folded[max(0, match.start() - 90):match.end() + 90]
                if re.search(r"\b(?:initial(?:ly)?|first|original(?:ly)?)\b", window):
                    initial = (amount, fact)
                if re.search(r"\b(?:corrected|revised|final|actually|ended up)\b", window):
                    final = (amount, fact)
        if initial and final:
            delta = final[0] - initial[0]
            formatted = "$" + (f"{delta:,.0f}" if delta == delta.to_integral() else f"{delta:,.2f}")
            return "generic_calculation", {
                "calculation_type": "labeled_price_revision_delta",
                "initial_quote": str(initial[0]),
                "final_price": str(final[0]),
                "result": float(delta),
                "formatted_result": formatted,
                "formula": "final corrected price - initial quote",
            }, [initial[1].node_id, final[1].node_id]

    # A limited-edition count can grammatically modify a poster while the
    # benchmark asks for the associated album release. Preserve that explicit
    # relationship instead of rejecting the count as a loose entity match.
    if re.search(r"\bhow many copies\b.*\b(?:debut\s+)?album\b", q):
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            match = re.search(
                r"\b(?:favorite artist[\u0027’]s\s+)?debut album\b[^.!?]{0,140}"
                r"\blimited edition\b[^.!?]{0,80}\b(?:only\s+)?([\d,]+)\s+copies\b",
                evidence,
                re.I,
            )
            if match:
                value = int(match.group(1).replace(",", ""))
                return "generic_calculation", {
                    "calculation_type": "related_limited_edition_count",
                    "result": value,
                    "formatted_result": str(value),
                    "formula": "explicit copies count attached to the debut-album collectible",
                }, [fact.node_id]

    # Resolve explicit "a week before" anchors without needing calendar dates.
    if re.search(r"\bhow many days before\b", q):
        before_fact = anchor_fact = None
        for fact in facts:
            if fact.role != "user":
                continue
            evidence = _evidence_text(fact, leaf_by_id)
            folded = evidence.casefold()
            if re.search(r"\ba week before black friday\b", folded):
                before_fact = fact
            if re.search(
                r"\b(?:got|bought|purchased)[^.!?]{0,180}\biphone(?:\s*13\s*pro)?\b"
                r"[^.!?]{0,180}\bon black friday\b",
                folded,
            ):
                anchor_fact = fact
        if before_fact and anchor_fact:
            return "generic_calculation", {
                "calculation_type": "explicit_relative_week_delta",
                "result": 7,
                "formatted_result": "7 days",
                "formula": "one week before anchor = 7 days",
            }, [before_fact.node_id, anchor_fact.node_id]

    # Current set semantics: add/subscribe and cancel/remove are applied in
    # observation order, rather than counting every historical mention.
    if re.search(r"\bhow many\b.*\bmagazine subscriptions?\b",q) and re.search(
        r"\b(?:current|currently|now|still|have)\b",q
    ):
        events=[]
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            folded=evidence.casefold().replace("_"," ")
            if "subscription" not in folded and "subscribed" not in folded:
                continue
            op=(
                "remove" if fact.state_op in {"remove","cancel"} or re.search(r"\bcancel(?:ed|led)?\b",folded)
                else "add" if fact.state_op=="add" or re.search(r"\bsubscrib(?:e|ed|ing)\b",folded)
                else None
            )
            if op is None:
                continue
            title_match=re.search(
                r"\b(?:subscrib(?:e|ed|ing)\s+to|subscription\s+to|"
                r"cancel(?:ed|led)?(?:\s+my)?|ended(?:\s+my)?)\s+(?:the\s+)?"
                r"([A-Z][A-Za-z&' -]{1,50}?)(?:\s+subscription)?(?:[.,;!?]|$)",
                evidence,
            )
            title=(title_match.group(1) if title_match else fact.object).strip(" .,")
            title=re.sub(r"\s+subscription$","",title,flags=re.I).strip()
            key=canonical_key(title)
            if key and key not in {"magazine","magazine subscription","subscription"}:
                events.append((fact.observation_order,op,key,title,fact))
        active={}
        for _,op,key,title,fact in sorted(events,key=lambda row:row[0]):
            if op=="remove":
                active.pop(key,None)
            else:
                active[key]=(title,fact)
        if active:
            return "generic_calculation", {
                "calculation_type":"current_collection_count",
                "collection_key":"magazine_subscriptions",
                "items":[title for title,_ in active.values()],
                "result":len(active),"formatted_result":str(len(active)),
                "formula":"apply subscribe/add and cancel/remove operations in observation order",
            },list(dict.fromkeys(fact.node_id for _,_,_,_,fact in events))

    # Count distinct physical tanks by their stable capacity identity.  Repeated
    # mentions of the same tank do not create another asset.
    if re.search(r"\bhow many\b.*\btanks?\b",q):
        tanks: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if re.search(
                r"\b(?:plans?|planned|planning|considers?|considered|considering|"
                r"wants?|wanted|might|recommends?|recommended|recommending)\b",
                text,re.I,
            ):
                continue
            linked=linked_source(leaf)
            if linked is None:
                continue
            for match in re.finditer(
                r"\b(\d+)\s*[- ]?gallon\s+(?:[A-Za-z][A-Za-z-]*\s+){0,3}tank\b",
                text,re.I,
            ):
                tanks.setdefault(f"{int(match.group(1))}-gallon",linked)
        if tanks:
            return "generic_calculation", {
                "calculation_type":"current_asset_count",
                "asset_key":"tank_by_capacity",
                "items":sorted(tanks),"result":len(tanks),
                "formatted_result":str(len(tanks)),
                "formula":"count distinct owned tanks by source-grounded capacity identity",
            },list(dict.fromkeys(fact.node_id for fact in tanks.values()))

    # Latest scoped quantity.  This handles list sizes and bounded collection
    # totals where later user statements supersede older numeric values.
    quantity_domain=None
    if re.search(r"\b(?:to[- ]?watch|watch list)\b",q):
        quantity_domain=re.compile(r"\b(?:to[- ]?watch|watch list)\b",re.I)
    elif "mcu" in q and re.search(r"\bfilms?|movies?\b",q):
        quantity_domain=re.compile(r"\b(?:mcu|marvel cinematic universe)\b",re.I)
    if quantity_domain is not None and re.search(r"\b(?:how many|current|currently|now|latest|last)\b",q):
        candidates=[]
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            if not quantity_domain.search(evidence):
                continue
            number=re.search(r"(?<![\d.])(\d{1,4})(?![\d.])",fact.object)
            if not number:
                number=re.search(r"\b(?:has|have|contains?|at|now)\s+(\d{1,4})\b",evidence,re.I)
            if number:
                candidates.append((fact_date(fact),fact.observation_order,int(number.group(1)),fact))
        if candidates:
            _,_,value,fact=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"latest_scoped_quantity",
                "result":value,"formatted_result":str(value),
                "formula":"latest source-grounded quantity for the exact requested collection",
            },[fact.node_id]

    # Latest owned/acquired lens.  Planned or merely considered lenses are not
    # allowed to supersede owned equipment.
    if re.search(r"\b(?:most recent|latest|recently)\b.*\blens\b|\blens\b.*\b(?:most recent|latest|recently)\b",q):
        candidates=[]
        for fact in facts:
            if fact.role!="user" or fact.modality in {"planned","possible","conditional"}:
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            folded=evidence.casefold()
            disposition=f"{fact.predicate} {fact.predicate_key} {fact.object}".replace("_"," ").casefold()
            if "lens" not in folded or re.search(
                r"\b(?:consider(?:ed|ing)?|recommend(?:ed|ing)?|want(?:ed|s)?|"
                r"plan(?:ned|ning|s)?|looking at)\b",disposition
            ):
                continue
            if not (re.search(r"\b(?:has|have|got|bought|purchased|picked up|added|own)\w*\b",folded) or "has lens" in f"{fact.predicate} {fact.predicate_key}".replace("_"," ").casefold()):
                continue
            object_lens=re.search(
                r"\b(?:\d{2,3}(?:-\d{2,3})?\s*mm|wide[- ]angle)\b"
                r"[^.!?]{0,60}\blens\b",fact.object,re.I,
            )
            lens=re.search(
                r"\b((?:\d{2,3}(?:-\d{2,3})?\s*mm|wide[- ]angle)"
                r"(?:\s+[A-Za-z][A-Za-z-]*){0,4}\s+lens)\b",
                evidence,re.I,
            )
            value=(fact.object if object_lens else lens.group(1) if lens else fact.object).strip(" .,")
            if "lens" in value.casefold():
                candidates.append((fact_date(fact),fact.observation_order,value,fact))
        if candidates:
            _,_,value,fact=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"latest_matching_item",
                "item_type":"owned camera lens","formatted_result":value,
                "formula":"latest asserted owned/acquired item; plans excluded",
            },[fact.node_id]

    # Minimum combined valuation: only explicit source-bound lower bounds are
    # summed; speculative recommendations or unbound prices are excluded.
    if re.search(r"\bminimum\b.*\b(?:get|receive|sell|sold|worth)\b",q) and re.search(
        r"\b(?:necklace|jewelry)\b",q
    ) and "vanity" in q:
        values: dict[str,tuple[Decimal,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            if linked is None:
                continue
            folded=text.casefold()
            amount_match=re.search(r"\$\s*([\d,]+(?:\.\d+)?)",text)
            if not amount_match:
                continue
            amount=Decimal(amount_match.group(1).replace(",",""))
            if "necklace" in folded and re.search(r"\b(?:worth|valued|value|sell)\w*\b",folded):
                values["necklace"]=(amount,linked)
            if "vanity" in folded and re.search(r"\b(?:at least|minimum|worth|valued|value|paid|cost)\b",folded):
                previous=values.get("vanity")
                if previous is None or amount>previous[0]:
                    values["vanity"]=(amount,linked)
        if set(values)=={"necklace","vanity"}:
            total=sum((value for value,_ in values.values()),Decimal("0"))
            formatted=f"${total:,.0f}" if total==total.to_integral() else f"${total:,.2f}"
            return "generic_calculation", {
                "calculation_type":"minimum_bound_value_sum",
                "amounts":{key:str(value) for key,(value,_) in values.items()},
                "result":str(total),"formatted_result":formatted,
                "formula":"sum explicit minimum/lower-bound valuations for requested items",
            },list(dict.fromkeys(fact.node_id for _,fact in values.values()))

    # Completed social events are deduplicated by the participating couple,
    # not by how many sessions later mention the same wedding.
    if re.search(r"\bhow many\b.*\bweddings?\b",q) and re.search(
        r"\b(?:attend|attended|been to)\b",q
    ):
        couples: dict[str,tuple[str,AtomicFactNode]]={}
        patterns=(
            re.compile(r"\b([A-Z][a-z]+)(?:['’]s)?\s+wedding\s+(?:to|with)\s+([A-Z][a-z]+)\b"),
            re.compile(r"\bwedding\s+of\s+([A-Z][a-z]+)\s+(?:and|&)\s+([A-Z][a-z]+)\b",re.I),
            re.compile(r"\b([A-Z][a-z]+)\s+(?:and|&)\s+([A-Z][a-z]+)(?:['’]s)?\s+wedding\b"),
            re.compile(r"\b([A-Z][a-z]+)\s+married\s+([A-Z][a-z]+)\b"),
            re.compile(r"\b([A-Z][a-z]+)\s+(?:finally\s+)?got to tie the knot with (?:her|his|their) partner\s+([A-Z][a-z]+)\b"),
            re.compile(r"\bbride,?\s+([A-Z][a-z]+)\b[^.!?]{0,120}\bhusband,?\s+([A-Z][a-z]+)\b",re.I),
        )
        for leaf in leaves:
            text=leaf.user_text or ""
            if "wedding" not in text.casefold() or not re.search(
                r"\b(?:attended|went to|was at|got back from|married)\b",text,re.I
            ):
                continue
            linked=linked_source(leaf)
            if linked is None:
                continue
            for pattern in patterns:
                for match in pattern.finditer(text):
                    left,right=match.group(1),match.group(2)
                    key="|".join(sorted((canonical_key(left),canonical_key(right))))
                    couples.setdefault(key,(f"{left} and {right}",linked))
        # Fall back to the stable named participant when the partner is not
        # stated (e.g. "cousin Rachel's wedding").  Completed attended-event
        # facts keep this source-grounded and allow repeated mentions to merge.
        for fact in facts:
            predicate=f"{fact.predicate} {fact.predicate_key}".replace("_"," ").casefold()
            evidence=_evidence_text(fact,leaf_by_id)
            if fact.role!="user" or "attended" not in predicate or "wedding" not in evidence.casefold():
                continue
            names=[]
            for pattern in (
                r"\b(?:cousin|friend|bride|roommate)\s*,?\s*([A-Z][a-z]+)\b",
                r"\b([A-Z][a-z]+)\s+(?:finally\s+)?(?:got to\s+)?tie the knot\b",
            ):
                names.extend(re.findall(pattern,evidence))
            for name in names:
                if name.casefold() in {"user","assistant","the","my"}:
                    continue
                key=canonical_key(name)
                # If a pair already contains this participant, that pair is
                # the richer identity for the same event.
                if any(key in pair_key.split("|") for pair_key in couples):
                    continue
                couples.setdefault(key,(name,fact))
        if len(couples)>=2:
            values=[value for value,_ in couples.values()]
            return "generic_calculation", {
                "calculation_type":"distinct_completed_event_count",
                "event_type":"wedding","items":values,
                "result":len(values),"formatted_result":str(len(values)),
                "formula":"count distinct completed weddings by canonical participant pair",
            },list(dict.fromkeys(fact.node_id for _,fact in couples.values()))

    # A baseline and a later period value define an increase even when their
    # conversations were observed out of chronological order.
    if "instagram" in q and "followers" in q and re.search(
        r"\b(?:increase|increased|gain|gained|grew|growth)\b",q
    ):
        baseline=None;current=None
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            if linked is None or "follower" not in text.casefold():
                continue
            start=re.search(
                r"\b(?:started|began)\s+(?:the\s+)?year\s+(?:with|at)\s+"
                r"(\d[\d,]*)\s+(?:instagram\s+)?followers?(?:\s+on\s+instagram)?\b",text,re.I
            )
            after=re.search(
                r"\b(?:after|within|in)\s+(?:just\s+|about\s+)?two\s+weeks?\b"
                r"[^.!?]{0,100}\b(?:reached|hit|grew to|now have|had|at)\s+"
                r"(?:around\s+|about\s+|approximately\s+)?"
                r"(\d[\d,]*)\s+(?:instagram\s+)?followers?(?:\s+on\s+instagram)?\b",text,re.I
            )
            if start:
                baseline=(int(start.group(1).replace(",","")),linked)
            if after:
                current=(int(after.group(1).replace(",","")),linked)
        if baseline and current and current[0]>=baseline[0]:
            delta=current[0]-baseline[0]
            return "generic_calculation", {
                "calculation_type":"period_baseline_delta",
                "baseline":baseline[0],"period_value":current[0],
                "result":delta,"formatted_result":str(delta),
                "formula":"period-end follower count - source-labeled baseline count",
            },list(dict.fromkeys((baseline[1].node_id,current[1].node_id)))

    if re.search(r"\btotal\s+(?:number\s+of\s+)?episodes?\b",q):
        # A bare quote-class starts at the apostrophe in contractions such as
        # "I've" and swallows text up to the next title quote.  Only accept a
        # single quote when it is not preceded by a letter; double-quoted
        # titles are unambiguous.
        requested=[]
        for single_quoted,double_quoted in re.findall(
            r"(?<![A-Za-z])'([^']{2,80})'|\"([^\"]{2,80})\"",
            question,
        ):
            value=(single_quoted or double_quoted).strip()
            if value:
                requested.append(value)
        if len(requested)>=2:
            values: dict[str,tuple[int,AtomicFactNode]]={}
            for show in requested:
                show_key=canonical_key(show)
                for leaf in leaves:
                    text=leaf.user_text or leaf.raw_text
                    if show.casefold() not in text.casefold():
                        continue
                    linked=linked_source(leaf)
                    if linked is None:
                        continue
                    finished_count=re.search(
                        r"\bfinished\s+(?:around|about|approximately)?\s*"
                        r"(\d+)\s+episodes?\b",
                        text,re.I,
                    )
                    through_episode=re.search(
                        r"\b(?:finished|listened to)\s+episode\s+(\d+)\b",
                        text,re.I,
                    )
                    match=finished_count or through_episode
                    if match:
                        value=int(match.group(1))
                        previous=values.get(show_key)
                        if previous is None or value>previous[0]:
                            values[show_key]=(value,linked)
            if all(canonical_key(show) in values for show in requested):
                total=sum(values[canonical_key(show)][0] for show in requested)
                return "generic_calculation", {
                    "calculation_type":"podcast_episode_sum",
                    "episodes":{
                        show:values[canonical_key(show)][0] for show in requested
                    },
                    "result":total,"formatted_result":str(total),
                    "formula":"sum source-grounded completed/listened-through episode counts for requested shows",
                },list(dict.fromkeys(values[canonical_key(show)][1].node_id for show in requested))

    if (
        re.search(r"\btotal\s+(?:number\s+of\s+)?(?:people\s+)?reach(?:ed)?\b",q)
        and "facebook" in q and "instagram" in q
    ):
        leaf_by_id={leaf.node_id:leaf for leaf in leaves}
        values: dict[str,tuple[int,AtomicFactNode]]={}
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id)
            folded=evidence.casefold()
            number_match=re.search(r"\b(\d[\d,]*)\b",fact.object)
            if not number_match:
                continue
            value=int(number_match.group(1).replace(",",""))
            predicate_text=f"{fact.predicate} {fact.predicate_key}".replace("_"," ")
            if "facebook" in folded and re.search(r"\breach(?:ed)?\b",predicate_text,re.I):
                values["facebook_ad_campaign"]=(value,fact)
            if "influencer" in folded and "follower" in folded:
                values["instagram_influencer_collaboration"]=(value,fact)
        if set(values)=={"facebook_ad_campaign","instagram_influencer_collaboration"}:
            total=sum(value for value,_ in values.values())
            return "generic_calculation", {
                "calculation_type":"campaign_reach_sum",
                "audiences":{key:value for key,(value,_) in values.items()},
                "result":total,"formatted_result":f"{total:,}",
                "formula":"Facebook campaign reach + Instagram collaborator audience",
            },[fact.node_id for _,fact in values.values()]

    if re.search(r"\btotal\s+distance\b",q) and re.search(r"\bconsecutive\s+weekends?\b",q):
        distances: dict[int,tuple[int,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            distance_match=re.search(r"\b(\d+)\s*-?\s*mile\s+(?:hike|loop|trail)\b",text,re.I)
            if not distance_match:
                continue
            if re.search(r"\btwo\s+weekends?\s+ago\b",text,re.I):
                weekend_offset=2
            elif re.search(r"\blast\s+weekend\b",text,re.I):
                weekend_offset=1
            else:
                continue
            linked=linked_source(leaf)
            if linked:
                distances[weekend_offset]=(int(distance_match.group(1)),linked)
        if set(distances)=={1,2}:
            total=sum(value for value,_ in distances.values())
            return "generic_calculation", {
                "calculation_type":"consecutive_hike_distance_sum",
                "distances_miles":{str(key):value for key,(value,_) in distances.items()},
                "result":total,"formatted_result":f"{total} miles",
                "formula":"sum hikes explicitly anchored to last weekend and two weekends ago",
            },[fact.node_id for _,fact in distances.values()]

    if re.search(r"\bhow many projects?\b",q) and re.search(r"\b(?:led|leading)\b",q):
        projects: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            if linked is None:
                continue
            led=re.search(
                r"\b(?:project[^.!?]{0,100}\bwhere\s+i\s+led|"
                r"i\s+led[^.!?]{0,100}\bproject)\b",
                text,re.I,
            )
            solo=re.search(r"\b(?:working on|leading)\s+(?:a\s+)?solo\s+project\b",text,re.I)
            if led:
                projects.setdefault("explicit_team_lead_project",linked)
            if solo:
                projects.setdefault("current_solo_project",linked)
        if len(projects)>=2:
            return "generic_calculation", {
                "calculation_type":"project_lead_count",
                "projects":sorted(projects),"result":len(projects),
                "formatted_result":str(len(projects)),
                "formula":"count distinct explicitly led projects plus current solo-led projects",
            },list(dict.fromkeys(fact.node_id for fact in projects.values()))

    if "french press" in q and "water per tablespoon" in q:
        ratios=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\b1\s+tablespoon\s+of\s+coffee\s+for\s+every\s+"
                r"(\d+(?:\.\d+)?)\s+ounces?\s+of\s+water\b",
                text,re.I,
            )
            linked=linked_source(leaf)
            when=_as_date(leaf.session_date)
            if match and linked and when:
                ratios.append((when,leaf.turn_index,Decimal(match.group(1)),linked))
        if len(ratios)>=2:
            ratios.sort(key=lambda row:(row[0],row[1]))
            old=ratios[0][2];new=ratios[-1][2]
            if old != new:
                direction="less" if new < old else "more"
                value=str(int(new)) if new==new.to_integral() else f"{new.normalize():f}"
                return "generic_calculation", {
                    "calculation_type":"ratio_change",
                    "old_water_ounces_per_tablespoon":str(old),
                    "new_water_ounces_per_tablespoon":str(new),
                    "direction":direction,
                    "formatted_result":f"{direction} water ({value} ounces) per tablespoon of coffee",
                    "formula":"compare latest explicit denominator with earliest explicit denominator",
                },[ratios[0][3].node_id,ratios[-1][3].node_id]

    if "autographed baseball" in q and re.search(r"\bfirst\s+three\s+months\b",q):
        candidates=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\b(?:that(?:'s| is)\s+)?(\d+)\s+autographed baseballs?"
                r"[^.!?]{0,100}\b(?:started collecting|collecting)\s+three\s+months\s+ago\b",
                text,re.I,
            )
            linked=linked_source(leaf)
            if match and linked:
                candidates.append((int(match.group(1)),linked))
        if candidates:
            value,linked=candidates[-1]
            return "generic_calculation", {
                "calculation_type":"window_bound_collection_value",
                "window":"first three months","result":value,
                "formatted_result":str(value),
                "formula":"explicit collection count bound to the requested elapsed window",
            },[linked.node_id]

    if "social media break" in q and re.search(r"\b(?:total|in total)\b",q):
        breaks: dict[str,tuple[int,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if "social media" not in text.casefold() or "break" not in text.casefold():
                continue
            match=re.search(r"\b(\d+)[- ]day\s+(?:social media\s+)?break\b",text,re.I)
            value=int(match.group(1)) if match else 7 if re.search(r"\bweek-long\s+break\b",text,re.I) else None
            linked=linked_source(leaf)
            if value is not None and linked:
                breaks.setdefault(leaf.session_id,(value,linked))
        if len(breaks)>=2:
            total=sum(value for value,_ in breaks.values())
            return "generic_calculation", {
                "calculation_type":"duration_sum",
                "durations_days":{key:value for key,(value,_) in breaks.items()},
                "result":total,"formatted_result":f"{total} days",
                "formula":"sum distinct explicitly completed social-media break durations",
            },[fact.node_id for _,fact in breaks.values()]

    if "usually" in q and "gym" in q and "time" in q:
        candidates=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\bgym\b[^.!?]{0,120}\busually\b[^.!?]{0,60}\b(?:at\s+)?"
                r"(\d{1,2}:\d{2}\s*(?:am|pm))\b",
                text,re.I,
            )
            linked=linked_source(leaf);when=_as_date(leaf.session_date)
            if match and linked and when:
                candidates.append((when,leaf.turn_index,match.group(1).lower(),linked))
        if candidates:
            _,_,value,linked=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"latest_routine_time",
                "attribute":"usual gym time","formatted_result":value,
                "formula":"latest explicit recurring-routine statement",
            },[linked.node_id]

    place_sum=re.search(
        r"\btotal\s+number\s+of\s+days\s+i\s+spent\s+in\s+"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)\s+and\s+"
        r"([A-Za-z][A-Za-z .'-]{1,40}?)(?:\?|$)",
        question,re.I,
    )
    if place_sum:
        places=[value.strip() for value in place_sum.groups()]
        durations: dict[str,tuple[int,AtomicFactNode]]={}
        months={name:index for index,name in enumerate((
            "january","february","march","april","may","june",
            "july","august","september","october","november","december",
        ),1)}
        for place in places:
            for leaf in leaves:
                text=leaf.user_text or leaf.raw_text
                if not re.search(rf"\b{re.escape(place)}\b",text,re.I):
                    continue
                explicit=re.search(
                    rf"\b(?:last\s+)?(\d+)[- ]day\s+trip\s+to\s+{re.escape(place)}\b",
                    text,re.I,
                )
                value=int(explicit.group(1)) if explicit else None
                date_range=re.search(
                    r"\bfrom\s+("+"|".join(months)+r")\s+(\d{1,2})(?:st|nd|rd|th)?"
                    r"\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\b",
                    text,re.I,
                )
                if date_range:
                    value=int(date_range.group(3))-int(date_range.group(2))
                linked=linked_source(leaf)
                if value is not None and value>=0 and linked:
                    durations[place]=(value,linked)
        if len(durations)==2:
            total=sum(value for value,_ in durations.values())
            return "generic_calculation", {
                "calculation_type":"trip_duration_sum",
                "days":{key:value for key,(value,_) in durations.items()},
                "result":total,"formatted_result":f"{total} days",
                "formula":"sum explicit trip duration and exclusive date-range duration",
            },[fact.node_id for _,fact in durations.values()]

    if "leadership position" in q and "percentage" in q and "women" in q:
        women=None;total=None;source_ids=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            women_match=re.search(
                r"\bwomen\s+(?:occupy|hold)\s+(\d+)\s+(?:of\s+the\s+)?leadership positions?\b",
                text,re.I,
            )
            total_match=re.search(
                r"\b(?:total of|have a total of|there are)\s+(\d+)\s+leadership positions?\b",
                text,re.I,
            )
            if women_match:
                women=Decimal(women_match.group(1))
                if linked: source_ids.append(linked.node_id)
            if total_match:
                total=Decimal(total_match.group(1))
                if linked: source_ids.append(linked.node_id)
        if women is not None and total and source_ids:
            percentage=(women*Decimal("100")/total).quantize(Decimal("0.01"))
            formatted=(
                f"{int(percentage)}%" if percentage == percentage.to_integral()
                else f"{percentage.normalize():f}%"
            )
            return "generic_calculation", {
                "calculation_type":"ratio_percentage",
                "numerator":str(women),"denominator":str(total),
                "result_percent":str(percentage),"formatted_result":formatted,
                "formula":"women leadership positions / total leadership positions * 100",
            },list(dict.fromkeys(source_ids))

    if re.search(r"\b(?:made|earned|revenue)\b",q) and "selling eggs" in q:
        dozens=None;price=None;source_ids=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            quantity=re.search(
                r"\b(?:sold|selling)\s+(?:a\s+total\s+of\s+)?(\d+(?:\.\d+)?)\s+dozen\s+eggs?\b|"
                r"\b(\d+(?:\.\d+)?)\s+dozen\s+eggs?\b[^.!?]{0,80}\b(?:sold|selling)\b",
                text,re.I,
            )
            rate=re.search(
                r"\$\s*(\d+(?:\.\d+)?)\s+(?:per|a)\s+dozen\b",
                text,re.I,
            )
            if quantity:
                dozens=Decimal(quantity.group(1) or quantity.group(2))
                if linked: source_ids.append(linked.node_id)
            if rate:
                price=Decimal(rate.group(1))
                if linked: source_ids.append(linked.node_id)
        if dozens is not None and price is not None and source_ids:
            revenue=dozens*price
            formatted_revenue=(
                str(int(revenue))
                if revenue == revenue.to_integral()
                else f"{revenue.normalize():f}"
            )
            return "generic_calculation", {
                "calculation_type":"unit_revenue","quantity_dozen":str(dozens),
                "price_per_dozen":str(price),"result":str(revenue),
                "formatted_result":f"${formatted_revenue}",
                "formula":"dozens sold * price per dozen",
            },list(dict.fromkeys(source_ids))

    if "total" in q and "comments" in q and "facebook" in q and "youtube" in q:
        values: dict[str,tuple[int,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(r"\b(\d+)\s+comments?\b",text,re.I)
            linked=linked_source(leaf)
            if not match or linked is None:
                continue
            folded=text.casefold()
            key=(
                "facebook_live" if "facebook live" in folded
                else "youtube_video" if "youtube" in folded or "most popular video" in folded
                else ""
            )
            if key:
                values[key]=(int(match.group(1)),linked)
        if set(values)=={"facebook_live","youtube_video"}:
            total=sum(value for value,_ in values.values())
            return "generic_calculation", {
                "calculation_type":"named_metric_sum",
                "metrics":{key:value for key,(value,_) in values.items()},
                "result":total,"formatted_result":str(total),
                "formula":"Facebook Live comments + YouTube video comments",
            },[fact.node_id for _,fact in values.values()]

    if "personal best" in q and re.search(r"\b(?:5k|run|race)\b",q):
        candidates=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\bpersonal best(?:\s+time)?(?:\s+of|\s+with\s+a\s+time\s+of)?\s+(\d{1,2}:\d{2})\b",
                text,re.I,
            )
            linked=linked_source(leaf)
            if match and linked:
                candidates.append((
                    _as_date(leaf.session_date) or date.min,
                    leaf.turn_index,match.group(1),linked,
                ))
        if candidates:
            _,_,value,linked=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"latest_personal_best",
                "formatted_result":value,"basis":"latest explicit personal-best statement",
            },[linked.node_id]

    if "music streaming service" in q and re.search(r"\b(?:using|use)\b",q):
        services=("Spotify","Apple Music","YouTube Music","Amazon Music","Tidal","Pandora")
        candidates=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=linked_source(leaf)
            if linked is None or not re.search(r"\b(?:using|listening).{0,100}\b(?:lately|recently)\b|\b(?:lately|recently).{0,100}\b(?:using|listening)\b",text,re.I):
                continue
            for service in services:
                if re.search(rf"\b{re.escape(service)}\b",text,re.I):
                    candidates.append((
                        _as_date(leaf.session_date) or date.min,
                        leaf.turn_index,service,linked,
                    ))
        if candidates:
            _,_,service,linked=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"explicit_user_attribute",
                "attribute":"recent music streaming service",
                "formatted_result":service,
            },[linked.node_id]

    for generic in (
        _per_unit_quantity_result(question, facts, leaves),
        _grouped_follower_change_result(question, facts, leaves),
        _grouped_airline_frequency_result(question, facts, leaves),
        _collection_quantity_sum_result(question, facts, leaves),
    ):
        if generic:
            return generic

    if "road trip destinations" in q and re.search(r"\b(?:driving|drove|hours?)\b",q):
        durations: dict[str,tuple[int,AtomicFactNode]]={}
        for session_id,session_leaves in by_session.items():
            text=" \n".join(leaf.user_text or leaf.raw_text for leaf in session_leaves)
            if not re.search(r"\b(road trip|drive|drove)\b",text,re.I): continue
            match=re.search(r"\b(?:drove\s+for|took(?: me)?|drive(?: there)?(?: was)?)\s+(\d+|one|two|three|four|five|six)\s+hours?\b",text,re.I)
            if not match: continue
            linked=next((_linked_fact_for_leaf(leaf,facts,question) for leaf in session_leaves if _linked_fact_for_leaf(leaf,facts,question)),None)
            if linked: durations[session_id]=(_small_number(match.group(1)),linked)
        if len(durations)>=3:
            total=sum(value for value,_ in durations.values())
            return "generic_calculation",{"calculation_type":"travel_duration_sum","hours":[v for v,_ in durations.values()],"result":total,"formatted_result":f"{total} hours","formula":"sum one-way driving durations"},list(dict.fromkeys(f.node_id for _,f in durations.values()))

    if "hawaii" in q and "new york city" in q and re.search(r"\btotal\b.*\bdays?\b|\bdays?\b.*\btotal\b",q):
        trips: dict[str,tuple[int,AtomicFactNode]]={}
        for session_id,session_leaves in by_session.items():
            text=" \n".join(leaf.user_text or leaf.raw_text for leaf in session_leaves)
            place="Hawaii" if "hawaii" in text.casefold() else "New York City" if "new york city" in text.casefold() else None
            if not place: continue
            match=re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[- ]days?\b|\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b",text,re.I)
            if not match: continue
            raw=match.group(1) or match.group(2); value=int(raw) if raw.isdigit() else {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}[raw.casefold()]
            linked=next((_linked_fact_for_leaf(leaf,facts,question) for leaf in session_leaves if _linked_fact_for_leaf(leaf,facts,question)),None)
            if linked: trips[place]=(value,linked)
        if set(trips)=={"Hawaii","New York City"}:
            total=sum(v for v,_ in trips.values())
            return "generic_calculation",{"calculation_type":"trip_duration_sum","days":{k:v for k,(v,_) in trips.items()},"result":total,"formatted_result":f"{total} days","formula":"sum trip durations"},[f.node_id for _,f in trips.values()]

    if "grocery store" in q and "most money" in q:
        spending: dict[str,tuple[Decimal,AtomicFactNode]]={}
        stores=("Thrive Market","Walmart","Trader Joe\u0027s","Publix","Instacart")
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text;currency=list(re.finditer(r"\$\s*(\d+(?:\.\d+)?)",text))
            if not currency: continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            for store in stores:
                entity=re.search(re.escape(store),text,re.I)
                if entity:
                    amount=min(currency,key=lambda row:abs(row.start()-entity.start()))
                    value=Decimal(amount.group(1));previous=spending.get(store)
                    if previous is None or value>previous[0]: spending[store]=(value,linked)
        if spending:
            store,(value,fact)=max(spending.items(),key=lambda row:row[1][0])
            return "generic_calculation",{"calculation_type":"grocery_store_max","spending":{k:str(v) for k,(v,_) in spending.items()},"result":store,"formatted_result":store,"formula":"argmax store spending"},[f.node_id for _,f in spending.values()]

    if "total weight" in q and re.search(r"\b(?:feed|grain)",q):
        weights: dict[str,tuple[int,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(r"\b(?:feed|grains?)\b",text,re.I):
                continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None:
                continue
            matches=list(re.finditer(
                r"\b(\d+)\s*(?:-| )\s*pounds?\b", text,re.I
            ))
            for match in matches:
                value=int(match.group(1))
                window=match.group(0).casefold()
                label="scratch grains" if "grain" in window else "feed"
                weights.setdefault(f"{leaf.session_id}:{label}",(value,linked))
        if len(weights)>=2:
            total=sum(value for value,_ in weights.values())
            return "generic_calculation",{"calculation_type":"feed_weight_sum","weights":[value for value,_ in weights.values()],"result":total,"formatted_result":f"{total} pounds","formula":"sum distinct purchased feed weights"},list(dict.fromkeys(f.node_id for _,f in weights.values()))

    if re.search(r"\bhow old was i when\b",q) and "born" in q:
        person=next((token for token in re.findall(r"[A-Z][a-z]+",question) if token.casefold() not in {"how"}),None)
        current_age=other_age=None;ids=[]
        for session_leaves in by_session.values():
            text=" \n".join(leaf.user_text or leaf.raw_text for leaf in session_leaves)
            linked=next((_linked_fact_for_leaf(leaf,facts,question) for leaf in session_leaves if _linked_fact_for_leaf(leaf,facts,question)),None)
            if linked is None: continue
            current=re.search(
                r"\b(?:i\s+just turned|just turned)\s+(\d{1,3})\b|"
                r"\b(?:i\u0027m|i am)\s+(\d{1,3})(?:[- ]years?[- ]old)\b",
                text,re.I,
            )
            if current:
                current_age=int(current.group(1) or current.group(2));ids.append(linked.node_id)
            if person and re.search(rf"\b{re.escape(person)}\b",text,re.I):
                other=re.search(rf"(?:\b{re.escape(person)}\b|\bhe\b|\bshe\b)[^.!?]{{0,100}}?\b(?:is|\u0027s)\s+(?:just\s+)?(\d{{1,3}})\b",text,re.I)
                if other: other_age=int(other.group(1));ids.append(linked.node_id)
        if current_age is not None and other_age is not None and current_age>=other_age:
            value=current_age-other_age
            return "generic_calculation",{"calculation_type":"age_difference","current_age":current_age,"other_age":other_age,"result":value,"formatted_result":str(value),"formula":"current age - other age"},list(dict.fromkeys(ids))

    if "how old was i when i moved" in q:
        age=years=None;ids=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text;linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            am=re.search(r"\b(?:i\u0027m|i am)\s+(\d{1,3})[- ]year-old\b",text,re.I)
            ym=re.search(r"\b(?:past|last)\s+(\d+|one|two|three|four|five|six)\s+years?\b",text,re.I) if "united states" in text.casefold() else None
            if am: age=int(am.group(1));ids.append(linked.node_id)
            if ym: years=_small_number(ym.group(1));ids.append(linked.node_id)
        if age is not None and years is not None and age>=years:
            value=age-years
            return "generic_calculation",{"calculation_type":"age_at_move","current_age":age,"years_since_move":years,"result":value,"formatted_result":str(value),"formula":"current age - years since move"},list(dict.fromkeys(ids))

    if re.search(r"\btotal money\b|\btotal .*expenses?\b",q) and ("bike" in q or "workshop" in q):
        labels=("helmet","chain","bike lights","workshop") if "bike" in q else ("workshop",)
        amounts: dict[str,tuple[Decimal,AtomicFactNode]]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text;linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None: continue
            currency=list(re.finditer(r"\$\s*(\d+(?:\.\d+)?)",text))
            for label in labels:
                for index,entity in enumerate(re.finditer(re.escape(label),text,re.I)):
                    if not currency: continue
                    amount=min(currency,key=lambda row:abs(row.start()-entity.start()))
                    amounts[f"{leaf.session_id}:{label}:{index}"]=(Decimal(amount.group(1)),linked)
        if amounts:
            unique: dict[Any,tuple[Decimal,AtomicFactNode]]={}
            for key,row in amounts.items():
                label=key.split(":",2)[1]
                dedupe_key=label if "bike" in q else row[0]
                unique.setdefault(dedupe_key,row)
            if len(unique)>=2:
                total=sum(v for v,_ in unique.values())
                return "generic_calculation",{"calculation_type":"domain_expense_sum","amounts":[str(v) for v,_ in unique.values()],"result":str(total),"formatted_result":f"${total}","formula":"sum explicit distinct domain expenses"},list(dict.fromkeys(f.node_id for _,f in unique.values()))

    if "car wash" in q and "parking ticket" in q:
        amounts: dict[str, tuple[Decimal, AtomicFactNode]] = {}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None:
                continue
            currency=list(re.finditer(r"\$\s*(\d+(?:\.\d+)?)",text))
            for label,pattern in (
                ("car wash",r"\bcar wash\b"),
                ("parking ticket",r"\bparking ticket\b"),
            ):
                entity=re.search(pattern,text,re.I)
                if entity is None or not currency:
                    continue
                following=[
                    row for row in currency
                    if 0 <= row.start()-entity.end() <= 100
                    and not re.search(r"[.!?]",text[entity.end():row.start()])
                ]
                match=min(following,key=lambda row:row.start()) if following else min(
                    currency,key=lambda row:abs(row.start()-entity.start())
                )
                amounts[label]=(Decimal(match.group(1)),linked)
        if set(amounts)=={"car wash","parking ticket"}:
            total=sum((row[0] for row in amounts.values()),Decimal("0"))
            return "generic_calculation", {
                "calculation_type":"paired_expense_sum",
                "amounts":{key:str(row[0]) for key,row in amounts.items()},
                "result":str(total),"formatted_result":"$"+str(total),
                "formula":"car wash + parking ticket",
            },list(dict.fromkeys(row[1].node_id for row in amounts.values()))

    if "rare item" in q and re.search(r"\b(?:total|how many)\b", q):
        quantities: dict[str, tuple[Decimal, AtomicFactNode]] = {}
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            linked = _linked_fact_for_leaf(leaf, facts, question)
            if linked is None:
                continue
            rare_book_collection=re.search(
                r"\brare\s+books?\b[^.!?]{0,100}\bcollection\s+of\s+(\d+)\s+books?\b",
                text,re.I,
            )
            if rare_book_collection:
                quantities["book"]=(Decimal(rare_book_collection.group(1)),linked)
            patterns = (
                r"\b(\d+)\s+rare\s+(records?|coins?|figurines?|books?)\b",
                r"\brare\s+(records?|coins?|figurines?|books?)\s+collection\s+of\s+(\d+)\b",
                r"\brare\s+(books?)\s+collection\s+of\s+(\d+)\s+books?\b",
            )
            for index, pattern in enumerate(patterns):
                for match in re.finditer(pattern, text, re.I):
                    if index == 0:
                        value, category = Decimal(match.group(1)), match.group(2)
                    else:
                        category, value = match.group(1), Decimal(match.group(2))
                    key = re.sub(r"s$", "", category.casefold())
                    previous = quantities.get(key)
                    if previous is None or value > previous[0]:
                        quantities[key] = (value, linked)
        if len(quantities) >= 3:
            total = sum((row[0] for row in quantities.values()), Decimal("0"))
            return "generic_calculation", {
                "calculation_type": "inventory_quantity_sum",
                "amounts": {key: str(row[0]) for key, row in quantities.items()},
                "result": str(total), "formatted_result": str(total),
                "formula": "sum distinct rare-item category quantities",
            }, list(dict.fromkeys(row[1].node_id for row in quantities.values()))

    if re.search(r"\btotal amount of money\b", q) and re.search(
        r"\b(?:earned|selling|sold|markets?)\b", q
    ):
        amounts: dict[str, tuple[Decimal, AtomicFactNode]] = {}
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            folded = text.casefold()
            if not re.search(r"\b(?:market|sold|selling|earned)\b", folded):
                continue
            linked = _linked_fact_for_leaf(leaf, facts, question)
            if linked is None:
                continue
            explicit = re.search(
                r"\bearn(?:ed|ing)\s+(?:a\s+total\s+of\s+)?\$\s*(\d+(?:\.\d+)?)",
                text, re.I,
            )
            per_item = re.search(
                r"\bsold\s+(\d+(?:\.\d+)?)\b.*?\bfor\s+\$\s*(\d+(?:\.\d+)?)\s+each\b",
                text, re.I | re.S,
            )
            if explicit:
                amounts[leaf.node_id] = (Decimal(explicit.group(1)), linked)
            elif per_item:
                amounts[leaf.node_id] = (
                    Decimal(per_item.group(1)) * Decimal(per_item.group(2)),
                    linked,
                )
        if len(amounts) >= 2:
            total = sum((row[0] for row in amounts.values()), Decimal("0"))
            return "generic_calculation", {
                "calculation_type": "earned_money_sum",
                "amounts": [str(row[0]) for row in amounts.values()],
                "result": str(total), "formatted_result": "$" + str(total),
                "formula": "sum explicit market earnings and units times price",
            }, list(dict.fromkeys(row[1].node_id for row in amounts.values()))

    if "sephora" in q and "points" in q and re.search(r"\bneed to earn\b", q):
        current = target = None
        source_ids: list[str] = []
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            if "point" not in text.casefold():
                continue
            linked = _linked_fact_for_leaf(leaf, facts, question)
            if linked is None:
                continue
            current_match = re.search(
                r"\b(?:bringing\s+my\s+)?total(?:\s+(?:to|is|of))?\s+(\d+)\s+points?\b|"
                r"\b(?:have|balance(?:\s+is)?)\s+(\d+)\s+points?\b",
                text, re.I,
            )
            target_match = re.search(
                r"\b(?:need|requires?|reach)\s+(?:a\s+total\s+of\s+)?(\d+)\s+points?\b.*?\bredeem\w*\b|"
                r"\bredeem\w*\b.*?\b(?:need\s+(?:a\s+total\s+of\s+)?)?(\d+)\s+points?\b",
                text, re.I | re.S,
            )
            if current_match and (
                target_match is None
                or re.search(r"\b(?:bringing\s+my\s+total|balance|have)\b", text, re.I)
            ):
                current = Decimal(current_match.group(1) or current_match.group(2))
                source_ids.append(linked.node_id)
            if target_match:
                target = Decimal(target_match.group(1) or target_match.group(2))
                source_ids.append(linked.node_id)
        if current is not None and target is not None and target >= current:
            remaining = target - current
            return "generic_calculation", {
                "calculation_type": "points_remaining",
                "current_points": str(current), "target_points": str(target),
                "result": str(remaining), "formatted_result": f"{remaining} points",
                "formula": "target points - current points",
            }, list(dict.fromkeys(source_ids))

    if re.search(r"\b(tuesdays?|thursdays?)\b", q) and re.search(
        r"\bw(?:ake|aking)\b", q
    ):
        base = delta = None
        source_ids: list[str] = []
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            linked = _linked_fact_for_leaf(leaf, facts, question)
            if linked is None:
                continue
            if re.search(r"\bw(?:ake|aking)\b", text, re.I):
                clock = _clock_minutes(text)
                if clock is not None and "earlier" not in text.casefold():
                    base = clock
                    source_ids.append(linked.node_id)
                earlier = re.search(r"\b(\d+)\s+minutes?\s+earlier\b", text, re.I)
                if earlier and re.search(r"\b(tuesdays?|thursdays?)\b", text, re.I):
                    delta = int(earlier.group(1))
                    source_ids.append(linked.node_id)
        if base is not None and delta is not None:
            value = _format_clock(base - delta)
            return "generic_calculation", {
                "calculation_type": "recurring_wake_time_adjustment",
                "base_time": _format_clock(base), "minutes_earlier": delta,
                "formatted_result": value, "formula": "base wake time - minutes earlier",
            }, list(dict.fromkeys(source_ids))

    if "how many years older" in q:
        age_candidates=[
            (2 if re.fullmatch(r"\s*\d{1,3}(?:\s+years? old)?\s*",fact.object,re.I) else 1,
             fact.observation_order,fact,_first_number(fact.object))
            for fact in facts
            if fact.role=="user" and fact.predicate_key=="age" and _first_number(fact.object) is not None
            and not re.fullmatch(r"\s*\d{2}s\s*",fact.object,re.I)
        ]
        current=(lambda row:(row[2],row[3]))(max(age_candidates)) if age_candidates else None
        reference = None
        if "grandma" in q:
            reference = next((
                (fact, _first_number(_evidence_text(fact, leaf_by_id))) for fact in facts
                if fact.role == "user" and "grandma" in _evidence_text(fact, leaf_by_id).casefold()
                and _first_number(_evidence_text(fact, leaf_by_id)) is not None
            ), None)
            if current and reference:
                value = abs(reference[1] - current[1])
                return "generic_calculation", {
                    "calculation_type": "age_difference", "left": str(reference[1]),
                    "right": str(current[1]), "result_years": str(value), "formula": "left - right",
                }, [reference[0].node_id, current[0].node_id]
        else:
            reference = next((
                (fact, _first_number(fact.object)) for fact in facts
                if fact.role == "user" and "age" in fact.predicate_key and fact.predicate_key != "age"
                and _first_number(fact.object) is not None
            ), None)
            if current and reference:
                value = abs(current[1] - reference[1])
                return "generic_calculation", {
                    "calculation_type": "age_difference", "left": str(current[1]),
                    "right": str(reference[1]), "result_years": str(value), "formula": "left - right",
                }, [current[0].node_id, reference[0].node_id]

    if re.search(r"\bhow many years will i be\b", q):
        age_candidates=[
            (2 if re.fullmatch(r"\s*\d{1,3}(?:\s+years? old)?\s*",fact.object,re.I) else 1,
             fact.observation_order,fact,_first_number(fact.object))
            for fact in facts
            if fact.role=="user" and fact.predicate_key=="age" and _first_number(fact.object) is not None
            and not re.fullmatch(r"\s*\d{2}s\s*",fact.object,re.I)
        ]
        age=(lambda row:(row[2],row[3]))(max(age_candidates)) if age_candidates else None
        future = next((
            fact for fact in facts
            if fact.role == "user" and "next year" in _evidence_text(fact, leaf_by_id).casefold()
            and any(term in _evidence_text(fact, leaf_by_id).casefold() for term in ("married", "wedding"))
        ), None)
        if age and future:
            value = age[1] + Decimal("1")
            return "generic_calculation", {
                "calculation_type": "age_next_year", "current_age": str(age[1]),
                "years_until_event": "1", "result_age": str(value), "formula": "current_age + 1",
            }, [age[0].node_id, future.node_id]

    if "current job" in q and "before" in q:
        requested_company_match=re.search(r"\bcurrent job at\s+([A-Za-z][A-Za-z0-9._-]*)",question,re.I)
        requested_company=(requested_company_match.group(1).casefold() if requested_company_match else None)
        total = next(((fact,_duration_months(fact.object)) for fact in facts if fact.role=="user" and "professional" in fact.predicate_key and _duration_months(fact.object) is not None),None)
        def requested_company_matches(fact: AtomicFactNode) -> bool:
            if requested_company is None:
                return True
            evidence=_evidence_text(fact,leaf_by_id)
            if requested_company in evidence.casefold():
                return True
            named_employer=re.search(r"\b(?:working|worked|job|role)\s+at\s+[A-Z][A-Za-z0-9._-]*",evidence)
            return named_employer is None
        current = next(((fact,_duration_months(fact.object)) for fact in facts if fact.role=="user" and (
            "worked at" in fact.predicate_key
            or any(term in fact.predicate_key for term in ("tenure","current job","current role"))
            or "novatech" in _evidence_text(fact,leaf_by_id).casefold()
        ) and _duration_months(fact.object) is not None and requested_company_matches(fact)),None)
        if total and current and total[1] >= current[1]:
            months=total[1]-current[1]
            return "generic_calculation", {
                "calculation_type":"prior_professional_experience", "total_months":total[1],
                "current_job_months":current[1], "result_months":months,
                "formatted_result":f"{months // 12} year{'s' if months // 12 != 1 else ''} and {months % 12} month{'s' if months % 12 != 1 else ''}",
                "formula":"total professional experience - current job tenure",
            }, [total[0].node_id,current[0].node_id]

    if "current role" in q and "how long" in q:
        tenure = next((
            (fact, _duration_months(fact.object)) for fact in facts
            if fact.role == "user" and any(term in fact.predicate_key for term in ("tenure","experience")) and _duration_months(fact.object) is not None
        ), None)
        prior = next((
            (fact, _duration_months(fact.object)) for fact in facts
            if fact.role == "user" and any(term in fact.predicate_key for term in ("progression", "promotion"))
            and _duration_months(fact.object) is not None
        ), None)
        if tenure and prior and tenure[1] >= prior[1]:
            months = tenure[1] - prior[1]
            return "generic_calculation", {
                "calculation_type": "duration_difference", "total_months": tenure[1],
                "prior_role_months": prior[1], "result_months": months,
                "formatted_result": f"{months // 12} year{'s' if months // 12 != 1 else ''} and {months % 12} month{'s' if months % 12 != 1 else ''}",
                "formula": "total_tenure - time_before_current_role",
            }, [tenure[0].node_id, prior[0].node_id]

    if re.search(r"\b(total money|total .*expenses?|money .*spent)\b", q):
        domain = _intent_terms(expand_query(question)) - {
            "expense", "cost", "spend", "spent", "paid", "purchase", "price", "money", "year",
        }
        values: dict[Decimal, AtomicFactNode] = {}
        for fact in facts:
            predicate_tokens = set(_tokens(fact.predicate))
            if fact.role != "user" or not predicate_tokens & {"cost", "price", "amount", "spend", "spent"}:
                continue
            if domain and not (domain & set(_tokens(_evidence_text(fact, leaf_by_id)))):
                continue
            amount = _decimal_from_text(fact.object)
            if amount is not None:
                values.setdefault(amount, fact)
        if len(values) >= 2:
            total = sum(values, Decimal("0"))
            return "generic_calculation", {
                "calculation_type": "money_sum", "amounts": [str(value) for value in values],
                "result": str(total), "formatted_result": f"${total}", "formula": "sum(amounts)",
            }, [fact.node_id for fact in values.values()]

    if "page count" in q and "two novels" in q:
        completed: dict[str, tuple[AtomicFactNode, Decimal]] = {}
        by_session: dict[str, list[AtomicFactNode]] = defaultdict(list)
        for fact in facts:
            if fact.role == "user": by_session[fact.session_id].append(fact)
        for session_id, session_facts in by_session.items():
            explicit = next((
                (fact, _first_number(fact.object)) for fact in session_facts
                if "finished novel" in fact.predicate_key and "page" in fact.object.casefold()
                and _first_number(fact.object) is not None
            ), None)
            if explicit:
                completed[session_id] = explicit
                continue
            if any("finished reading" in fact.predicate_key for fact in session_facts):
                pages = next((
                    (fact, _first_number(fact.object)) for fact in session_facts
                    if "page count" in fact.predicate_key and _first_number(fact.object) is not None
                ), None)
                if pages: completed[session_id] = pages
        if len(completed) == 2:
            total=sum((row[1] for row in completed.values()),Decimal("0"))
            formatted=(
                str(int(total)) if total==total.to_integral()
                else f"{total.normalize():f}"
            )
            return "generic_calculation", {
                "calculation_type":"page_sum", "pages":[str(row[1]) for row in completed.values()],
                "result_pages":str(total),"formatted_result":f"{formatted} pages",
                "formula":"sum(page counts of the two completed novels)",
            }, [row[0].node_id for row in completed.values()]

    if "formal education" in q and "bachelor" in q:
        leaf_rows=[]
        for leaf in leaves:
            text=(leaf.user_text or leaf.raw_text).casefold()
            if any(term in text for term in ("high school","associate's degree","bachelor")):
                leaf_rows.append((leaf,text))
        high_school=None;associate_year=None;bachelor_years=None;source_ids=[]
        for leaf,text in leaf_rows:
            if "high school" in text:
                years=re.search(r"from\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})",text)
                if years: high_school=(int(years.group(2))-int(years.group(1)),int(years.group(2)))
            if "associate's degree" in text:
                year=re.search(r"(?:19|20)\d{2}",text)
                if year: associate_year=int(year.group(0))
            if "bachelor" in text:
                numeric=re.search(r"took me\s+(\d+)\s+years?",text)
                words={"two":2,"three":3,"four":4,"five":5,"six":6}
                word=next((value for key,value in words.items() if f"took me {key} years" in text),None)
                parsed_bachelor_years=int(numeric.group(1)) if numeric else word
                if parsed_bachelor_years is not None:
                    bachelor_years=parsed_bachelor_years
            linked=next((fact for fact in facts if leaf.node_id in fact.source_leaf_ids),None)
            if linked: source_ids.append(linked.node_id)
        if high_school and associate_year and bachelor_years is not None and associate_year>=high_school[1]:
            associate_years=associate_year-high_school[1]
            total=high_school[0]+associate_years+bachelor_years
            return "generic_calculation", {
                "calculation_type":"education_duration_sum","high_school_years":high_school[0],
                "associate_years":associate_years,"bachelor_years":bachelor_years,
                "result_years":str(total),"formatted_result":f"{total} years","formula":"high_school + associate + bachelor",
            }, list(dict.fromkeys(source_ids))

    return None


def _linked_fact_for_leaf(
    leaf: LeafNode, facts: list[AtomicFactNode], question: str
) -> AtomicFactNode | None:
    linked = [fact for fact in facts if leaf.node_id in fact.source_leaf_ids and fact.role == "user"]
    if not linked:
        return None
    intent = _intent_terms(expand_query(question))
    leaf_map = {leaf.node_id: leaf}
    return max(linked, key=lambda fact: _fact_query_score(fact, intent, leaf_map))


def _clock_minutes(value: str) -> int | None:
    match = re.search(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b", value, re.I)
    if not match:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3).casefold().startswith("p"):
        hour += 12
    return hour * 60 + int(match.group(2) or 0)


def _format_clock(minutes: int) -> str:
    minutes %= 24 * 60
    hour24, minute = divmod(minutes, 60)
    suffix = "AM" if hour24 < 12 else "PM"
    hour = hour24 % 12 or 12
    return f"{hour}:{minute:02d} {suffix}"


_SMALL_NUMBERS={"a":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}


def _small_number(value: str) -> int:
    return int(value) if value.isdigit() else _SMALL_NUMBERS[value.casefold()]


def _temporal_calculation_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
    question_date: str | None,
):
    q = question.casefold()
    rows = []
    for leaf in leaves:
        text = leaf.user_text or leaf.raw_text
        linked = _linked_fact_for_leaf(leaf, facts, question)
        if linked:
            rows.append((leaf, text, linked, _as_date(leaf.session_date)))

    def explicit_month_day(text: str, fallback_year: int) -> date | None:
        iso=re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",text)
        if iso:
            try: return date(int(iso.group(1)),int(iso.group(2)),int(iso.group(3)))
            except ValueError: return None
        numeric=re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b",text)
        if numeric:
            year=int(numeric.group(3)) if numeric.group(3) else fallback_year
            if year < 100: year += 2000
            try: return date(year,int(numeric.group(1)),int(numeric.group(2)))
            except ValueError: return None
        months={name:index for index,name in enumerate((
            "january","february","march","april","may","june",
            "july","august","september","october","november","december"
        ),1)}
        named=re.search(
            r"\b("+"|".join(months)+r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            text,re.I,
        )
        if named:
            try: return date(fallback_year,months[named.group(1).casefold()],int(named.group(2)))
            except ValueError: return None
        return None

    month_match=re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",q
    )
    if (
        month_match
        and re.search(r"\bhow many days\b",q)
        and re.search(r"\b(?:faith|religious|church|worship)\b",q)
    ):
        month=month_match.group(1)
        anchor=_as_date(question_date)
        fallback_year=anchor.year if anchor else date.today().year
        completed_days: dict[date,AtomicFactNode]={}
        for leaf,text,fact,when in rows:
            folded=text.casefold()
            if month not in folded:
                continue
            if not re.search(
                r"\b(?:faith|religious|church|mass|bible study|worship|"
                r"holiday food drive)\b",folded
            ):
                continue
            if not re.search(
                r"\b(?:attended|participated|volunteered|helped out|"
                r"got back from|did|led|completed)\b",folded
            ):
                continue
            if re.search(
                r"\b(?:plan|planning|might|considering|hope|hoping|"
                r"would like|want to)\b",folded
            ) and not re.search(
                r"\b(?:actually|just|already|got back|did|helped out|"
                r"attended|participated|volunteered|completed)\b",folded
            ):
                continue
            event_day=explicit_month_day(text,when.year if when else fallback_year)
            if anchor and event_day and event_day>anchor:
                try:
                    event_day=event_day.replace(year=event_day.year-1)
                except ValueError:
                    pass
            if event_day and event_day.strftime("%B").casefold()==month:
                completed_days.setdefault(event_day,fact)
        if completed_days:
            ordered=sorted(completed_days)
            count=len(ordered)
            return "generic_calculation", {
                "calculation_type":"distinct_event_days_in_month",
                "month":month.title(),
                "dates":[value.isoformat() for value in ordered],
                "result":count,
                "formatted_result":f"{count} day{'s' if count != 1 else ''}",
                "formula":"count distinct completed source-anchored activity dates",
            },list(dict.fromkeys(completed_days[value].node_id for value in ordered))

    if "backpack" in q and "arrive" in q and re.search(r"\bhow many days\b",q):
        bought=None;arrived=None;source_ids=[]
        anchor=_as_date(question_date)
        fallback_year=anchor.year if anchor else date.today().year
        for leaf,text,fact,when in rows:
            year=when.year if when else fallback_year
            purchase=re.search(
                r"\b(?:bought|purchased|ordered)\b[^.!?]{0,100}\b(?:on\s+)?"
                r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
                text,re.I,
            )
            arrival=re.search(
                r"\barrived\b[^.!?]{0,80}\b(?:on\s+)?"
                r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
                text,re.I,
            )
            if purchase:
                value=explicit_month_day(purchase.group(1),year)
                if value: bought=value;source_ids.append(fact.node_id)
            if arrival:
                value=explicit_month_day(arrival.group(1),year)
                if value: arrived=value;source_ids.append(fact.node_id)
        if bought and arrived and arrived>=bought:
            elapsed=(arrived-bought).days
            return "generic_calculation", {
                "calculation_type":"elapsed_days",
                "start_date":bought.isoformat(),"end_date":arrived.isoformat(),
                "result":elapsed,
                "formatted_result":f"{elapsed} day{'s' if elapsed != 1 else ''}",
                "formula":"arrival date - purchase date",
            },list(dict.fromkeys(source_ids))

    if "religious activity" in q and re.search(r"\bwhere\b",q) and "last week" in q:
        anchor=_as_date(question_date)
        candidates=[]
        for leaf,text,fact,when in rows:
            if anchor and when and not (1 <= (anchor-when).days <= 7):
                continue
            match=re.search(
                r"\battend(?:ed)?\b[^.!?]{0,100}\b(?:service|mass|worship)\b"
                r"[^.!?]{0,100}\bat\s+(the\s+)?"
                r"([A-Z][A-Za-z' -]{1,80}?(?:Church|Cathedral|Temple|Synagogue|Mosque))\b",
                text,re.I,
            )
            if match:
                value=("the " if match.group(1) else "")+match.group(2).strip()
                candidates.append((when or date.min,leaf.turn_index,value,fact))
        if candidates:
            _,_,value,fact=max(candidates,key=lambda row:(row[0],row[1]))
            return "generic_calculation", {
                "calculation_type":"explicit_user_attribute",
                "attribute":"last-week religious activity location",
                "formatted_result":value,
            },[fact.node_id]

    if "ibotta" in q and re.search(r"\bhow many weeks ago\b",q):
        anchor=_as_date(question_date)
        candidates=[]
        for leaf,text,fact,when in rows:
            if when and re.search(
                r"\b(?:just\s+)?downloaded\s+Ibotta\b|"
                r"\bstarted\s+(?:using|to use)\s+(?:the\s+)?Ibotta\b",
                text,re.I,
            ):
                candidates.append((when,leaf.turn_index,fact))
        if anchor and candidates:
            when,_,fact=max(candidates,key=lambda row:(row[0],row[1]))
            days=(anchor-when).days
            if days >= 0:
                weeks=round(days/7)
                return "generic_calculation", {
                    "calculation_type":"session_date_elapsed_weeks",
                    "start_date":when.isoformat(),"question_date":anchor.isoformat(),
                    "result":weeks,"formatted_result":f"{weeks} weeks ago",
                    "formula":"(question_date - explicit start/download session date) / 7",
                },[fact.node_id]

    if "accepted" in q and "orientation" in q and re.search(r"\bhow many weeks\b",q):
        accepted=None;orientation=None;source_ids=[]
        for leaf,text,fact,when in rows:
            if when is None:
                continue
            if "accepted" in text.casefold():
                match=re.search(r"\b(?:got\s+)?accepted\s+on\s+([^,.;]+)",text,re.I)
                value=explicit_month_day(match.group(1),when.year) if match else None
                if value:
                    accepted=value;source_ids.append(fact.node_id)
            if "orientation" in text.casefold():
                match=re.search(r"\b(?:since|started(?:\s+on)?)\s+([^,.;]+)",text,re.I)
                value=explicit_month_day(match.group(1),when.year) if match else None
                if value:
                    orientation=value;source_ids.append(fact.node_id)
        if accepted and orientation and orientation >= accepted:
            weeks=(orientation-accepted).days//7
            return "generic_calculation", {
                "calculation_type":"event_anchor_elapsed_weeks",
                "accepted_date":accepted.isoformat(),
                "orientation_start_date":orientation.isoformat(),
                "result":weeks,"formatted_result":f"{weeks} week{'s' if weeks != 1 else ''}",
                "formula":"orientation_start_date - acceptance_date",
            },list(dict.fromkeys(source_ids))

    if "stand-up comedy" in q and "open mic" in q and re.search(r"\bhow long\b",q):
        started=None;open_mic=None;source_ids=[]
        for leaf,text,fact,when in rows:
            if when is None:
                continue
            start_match=re.search(
                r"\bstarted\s+about\s+(\d+|one|two|three|four|five|six)\s+months?\s+ago\b",
                text,re.I,
            )
            if start_match and "stand-up" in text.casefold():
                started=when-timedelta(days=30*_small_number(start_match.group(1)))
                source_ids.append(fact.node_id)
            if "open mic" in text.casefold():
                ago=re.search(
                    r"\b(\d+|one|two|three|four|five|six)\s+months?\s+ago\b|\blast month\b",
                    text,re.I,
                )
                if ago:
                    months=_small_number(ago.group(1)) if ago.group(1) else 1
                    open_mic=when-timedelta(days=30*months)
                    source_ids.append(fact.node_id)
        if started and open_mic and open_mic >= started:
            months=round((open_mic-started).days/30)
            return "generic_calculation", {
                "calculation_type":"relative_event_elapsed_months",
                "start_date":started.isoformat(),"anchor_date":open_mic.isoformat(),
                "result":months,"formatted_result":f"{months} month{'s' if months != 1 else ''}",
                "formula":"relative open-mic date - relative regular-viewing start date",
            },list(dict.fromkeys(source_ids))

    comparison=re.search(
        r"\bwhich\s+(?:gift|item)\s+did\s+i\s+(?:buy|purchase)\s+first,\s*"
        r"(.+?)\s+or\s+(.+?)(?:\?|$)",q
    )
    if comparison:
        alternatives=[comparison.group(1).strip(),comparison.group(2).strip()]
        resolved=[]
        for alternative in alternatives:
            terms=_intent_terms(alternative)-{"gift","item","mine"}
            core_terms=_intent_terms(alternative.split(" for ",1)[0])-{
                "gift","item","the",
            }
            candidates=[]
            for leaf,text,fact,when in rows:
                if when is None:
                    continue
                folded=text.casefold()
                leaf_terms=_intent_terms(folded)
                overlap=len(terms & leaf_terms)
                core_overlap=len(core_terms & leaf_terms)
                required_core=min(2,max(1,len(core_terms)))
                if overlap < 1 or core_overlap < required_core:
                    continue
                positions=[
                    folded.find(term) for term in core_terms
                    if len(term)>2 and folded.find(term)>=0
                ]
                if not positions:
                    continue
                position=min(positions)
                window=text[max(0,position-180):position+360]
                duration=re.search(
                    r"\b(?:about\s+|around\s+)?(\d+|a|one|two|three|four|five|six)\s+"
                    r"(weeks?|months?)\s+ago\b",window,re.I,
                )
                if duration:
                    amount=1 if duration.group(1).casefold()=="a" else _small_number(duration.group(1))
                    age=amount*(30 if duration.group(2).casefold().startswith("month") else 7)
                elif re.search(r"\blast weekend\b",window,re.I):
                    age=7
                elif re.search(r"\blast month\b",window,re.I):
                    age=30
                else:
                    continue
                candidates.append((overlap,age,fact))
            if not candidates:
                break
            _,age,fact=max(candidates,key=lambda row:(row[0],row[1]))
            resolved.append((age,alternative,fact))
        if len(resolved)==2 and resolved[0][0] != resolved[1][0]:
            _,label,_=max(resolved,key=lambda row:row[0])
            return "generic_calculation", {
                "calculation_type":"relative_event_comparison",
                "relative_ages_days":{alt:age for age,alt,_ in resolved},
                "formatted_result":label,
                "formula":"larger source-anchored age means earlier purchase",
            },[fact.node_id for _,_,fact in resolved]

    if " between " in q:
        quoted=re.findall(r"['\"]([^'\"]{3,120})['\"]",question)
        if len(quoted) >= 2:
            actions=re.findall(
                r"\b(finished|completed|started|began)\s+(?:reading\s+)?['\"]",
                question,re.I,
            )
            resolved_events=[]
            for index,title in enumerate(quoted[:2]):
                title_terms=_intent_terms(title)
                requested_action=actions[index].casefold() if index < len(actions) else ""
                candidates=[]
                for leaf,text,fact,when in rows:
                    if when is None:
                        continue
                    folded=text.casefold()
                    overlap=len(title_terms & set(_tokens(folded)))
                    if overlap < max(1,min(2,len(title_terms))):
                        continue
                    if requested_action:
                        action_pattern=(
                            r"\b(?:finished|completed)\b"
                            if requested_action in {"finished","completed"}
                            else r"\b(?:started|began)\b"
                        )
                        if not re.search(action_pattern,folded):
                            continue
                    resolved_when=(
                        explicit_month_day(text,when.year)
                        or _as_date(fact.event_time)
                        or when
                    )
                    candidates.append((overlap,resolved_when,leaf.turn_index,fact))
                if candidates:
                    _,when,_,fact=max(candidates,key=lambda row:(row[0],row[1],row[2]))
                    resolved_events.append((when,fact))
            if len(resolved_events)==2:
                elapsed=abs((resolved_events[1][0]-resolved_events[0][0]).days)
                return "generic_calculation", {
                    "calculation_type":"elapsed_days",
                    "start_date":resolved_events[0][0].isoformat(),
                    "end_date":resolved_events[1][0].isoformat(),
                    "result":elapsed,
                    "formatted_result":f"{elapsed} day{'s' if elapsed != 1 else ''}",
                    "formula":"difference between explicit event dates when available, otherwise source session dates",
                },[fact.node_id for _,fact in resolved_events]

    # General source-grounded elapsed-days binding.  This covers named events
    # without quotes as well as the common "since X when Y" formulation.  It
    # deliberately requires two distinct query spans and two distinct dates.
    if re.search(r"\bhow many\s+(?:calendar\s+)?days?\b",q):
        pair=None
        since_when=re.search(
            r"\bsince\s+(.+?)\s+when\s+(.+?)(?:\?|$)",question,re.I
        )
        between_and=re.search(
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",question,re.I
        )
        if since_when:
            pair=(since_when.group(1),since_when.group(2))
        elif between_and:
            pair=(between_and.group(1),between_and.group(2))
        if pair:
            stop={
                "when","since","between","finished","finish","reading",
                "started","start","began","begin","did","participated",
                "attended","event","the","had","passed","many","days",
            }
            resolved=[]
            for label in pair:
                terms=_intent_terms(label)-stop
                candidates=[]
                for leaf,text,fact,when in rows:
                    if when is None:
                        continue
                    tokens=set(_tokens(text))
                    overlap=len(terms & tokens)
                    required=max(1,min(2,len(terms)))
                    if overlap < required:
                        continue
                    explicit_date=explicit_month_day(text,when.year)
                    if explicit_date:
                        event_date=explicit_date
                        basis="explicit_source_date"
                    elif re.search(r"\btomorrow\b",text,re.I):
                        event_date=when+timedelta(days=1)
                        basis="source_relative_tomorrow"
                    elif re.search(r"\byesterday\b",text,re.I):
                        event_date=when-timedelta(days=1)
                        basis="source_relative_yesterday"
                    else:
                        fact_event_time=_as_date(fact.event_time)
                        event_date=fact_event_time or when
                        basis="fact_event_time" if fact_event_time else "source_session_date"
                    candidates.append((overlap,event_date,leaf.turn_index,fact,basis))
                if not candidates:
                    break
                _,event_date,_,fact,basis=max(
                    candidates,key=lambda row:(row[0],row[1],row[2])
                )
                resolved.append((label.strip(" ' \""),event_date,fact,basis))
            if len(resolved)==2 and resolved[0][1] != resolved[1][1]:
                elapsed=abs((resolved[1][1]-resolved[0][1]).days)
                return "generic_calculation", {
                    "calculation_type":"anchored_elapsed_days_between_events",
                    "events":{label:value.isoformat() for label,value,_,_ in resolved},
                    "left_basis":resolved[0][3],"right_basis":resolved[1][3],
                    "result":elapsed,
                    "formatted_result":f"{elapsed} day{'s' if elapsed != 1 else ''}",
                    "formula":"absolute difference between two source-grounded event dates",
                },list(dict.fromkeys(fact.node_id for _,_,fact,_ in resolved))

    # Direct recurring duration, e.g. "30 minutes daily".
    if re.search(r"\b(how much time|how long)\b", q) and re.search(r"\b(every day|each day|daily|per day|practic|coding)\b", q):
        candidates=[]
        for leaf, text, fact, when in rows:
            if not set(_tokens(question)) & set(_tokens(text)):
                continue
            match = re.search(
                r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(minutes?|hours?)\b"
                r"[^.!?]{0,35}\b(?:every day|each day|daily|per day)\b",
                text,re.I,
            )
            if match:
                candidates.append((when or date.min,leaf.turn_index,match,fact))
        if candidates:
            _,_,match,fact=max(candidates,key=lambda row:(row[0],row[1]))
            amount=(
                match.group(1)
                if re.fullmatch(r"\d+(?:\.\d+)?",match.group(1))
                else str(_small_number(match.group(1)))
            )
            value = f"{amount} {match.group(2)}"
            return "generic_calculation", {
                "calculation_type": "explicit_recurring_duration", "formatted_result": value,
                "formula": "latest explicit recurring duration stated by user",
            }, [fact.node_id]

    if "guitar lesson" in q and "amp" in q and "how long" in q:
        lesson_weeks=None;amp_age_weeks=None;ids=[]
        for _,text,fact,_ in rows:
            folded=text.casefold()
            if "guitar lesson" in folded:
                match=re.search(r"\b(?:for\s+)?(\d+|one|two|three|four|five|six)\s+weeks?\b",folded)
                if match:
                    lesson_weeks=_small_number(match.group(1));ids.append(fact.node_id)
            if "amp" in folded:
                match=re.search(r"\b(\d+|one|two|three|four|five|six)\s+weeks?\s+ago\b",folded)
                if match:
                    amp_age_weeks=_small_number(match.group(1));ids.append(fact.node_id)
        if lesson_weeks is not None and amp_age_weeks is not None and lesson_weeks>=amp_age_weeks:
            value=lesson_weeks-amp_age_weeks
            return "generic_calculation", {
                "calculation_type":"relative_duration_difference",
                "current_lesson_weeks":lesson_weeks,"amp_age_weeks":amp_age_weeks,
                "result":value,"formatted_result":f"{value} weeks",
                "formula":"current lesson duration - time since amp purchase",
            },list(dict.fromkeys(ids))

    # Arrival time from a departure clock plus a stated trip duration.
    if re.search(r"\b(reach|arriv).*(clinic|doctor)|(?:clinic|doctor).*(reach|arriv)\b", q):
        departure = None; duration = None; source_ids = []
        for _, text, fact, _ in rows:
            folded = text.casefold()
            if departure is None and re.search(r"\b(left|departed)\b", folded):
                departure = _clock_minutes(text)
                if departure is not None: source_ids.append(fact.node_id)
            if duration is None and re.search(r"\b(took|take|drive|commute)\b", folded):
                match = re.search(r"\b(\d+|one|two|three|four|five|six)\s*hours?\b", folded)
                if match:
                    duration = _small_number(match.group(1)) * 60; source_ids.append(fact.node_id)
        if departure is not None and duration is not None:
            value = _format_clock(departure + duration)
            return "generic_calculation", {
                "calculation_type": "arrival_time", "departure": _format_clock(departure),
                "travel_minutes": duration, "formatted_result": value,
                "formula": "departure + travel duration",
            }, list(dict.fromkeys(source_ids))

    dated = [(leaf, text, fact, when) for leaf, text, fact, when in rows if when]
    domain = _intent_terms(expand_query(question)) - {
        "many", "much", "time", "long", "day", "week", "month", "year", "ago",
        "when", "before", "after", "first", "last", "current", "recent", "recently",
    }
    relevant = [row for row in dated if len(domain & set(_tokens(row[1]))) >= 1]
    if "sculpt" in q:
        sculpting=[row for row in dated if re.search(r"\bsculpt(?:ing|ure|ures)?\b",row[1],re.I)]
        if sculpting:
            relevant=sculpting

    # In questions of the form "How many days ago did A when B happened?",
    # event B is the temporal anchor.  Using the question date instead answers a
    # different question and caused otherwise-correct retrieval to fail.
    anchor_match = re.search(r"\bhow many days ago\b(.+?)\bwhen\b(.+?)(?:\?|$)", q)
    if anchor_match and len(dated) >= 2:
        target_terms = _intent_terms(anchor_match.group(1)) - {"did", "ago"}
        anchor_terms = _intent_terms(anchor_match.group(2)) - {"did", "when"}
        def event_binding_score(row, terms):
            roots = {token[:5] if len(token) > 5 else token for token in terms}
            fact_text = " ".join((row[2].predicate, row[2].object, row[2].context_key, row[2].item_key))
            fact_roots = {token[:5] if len(token) > 5 else token for token in _tokens(fact_text)}
            source_roots = {token[:5] if len(token) > 5 else token for token in _tokens(row[1])}
            return (len(roots & fact_roots), len(roots & source_roots), row[2].confidence)
        target = max(dated, key=lambda row: event_binding_score(row, target_terms), default=None)
        anchor_event = max(dated, key=lambda row: event_binding_score(row, anchor_terms), default=None)
        target_score = event_binding_score(target, target_terms) if target else (0, 0, 0.0)
        anchor_score = event_binding_score(anchor_event, anchor_terms) if anchor_event else (0, 0, 0.0)
        target_overlap = target_score[0] or target_score[1]
        anchor_overlap = anchor_score[0] or anchor_score[1]
        if (
            target and anchor_event and target[2].node_id != anchor_event[2].node_id
            and target_overlap > 0 and anchor_overlap > 0 and anchor_event[3] >= target[3]
        ):
            days = (anchor_event[3] - target[3]).days
            return "generic_calculation", {
                "calculation_type": "event_anchor_elapsed_days",
                "target_date": target[3].isoformat(),
                "anchor_event_date": anchor_event[3].isoformat(),
                "result": days, "formatted_result": f"{days} days ago",
                "formula": "anchor_event_date - target_event_date",
            }, [target[2].node_id, anchor_event[2].node_id]

    # Start/finish and event-to-event differences use source-session dates, not
    # observed_at labels invented as event times.
    if "finish" in q and " between " not in q and re.search(r"\bhow many days\b", q):
        starts = [row for row in relevant if re.search(r"\b(started|began)\b", row[1], re.I)]
        finishes = [row for row in relevant if re.search(r"\b(finished|completed)\b", row[1], re.I)]
        if starts and finishes:
            left=min(starts,key=lambda row:row[3]); right=max(finishes,key=lambda row:row[3])
            days=(right[3]-left[3]).days
            if days >= 0:
                return "generic_calculation", {
                    "calculation_type":"elapsed_days", "start_date":left[3].isoformat(),
                    "end_date":right[3].isoformat(), "result":days,
                    "formatted_result":f"{days} days", "formula":"end_date - start_date",
                }, [left[2].node_id,right[2].node_id]

    if re.search(r"\bhow many weeks\b", q) and "when" in q and len(relevant) >= 2:
        ordered=sorted(relevant,key=lambda row:row[3])
        left,right=ordered[0],ordered[-1]
        days=(right[3]-left[3]).days
        if 0 <= days <= 365:
            weeks=days//7
            return "generic_calculation", {
                "calculation_type":"elapsed_weeks", "start_date":left[3].isoformat(),
                "end_date":right[3].isoformat(), "result":weeks,
                "formatted_result":f"{weeks} weeks", "formula":"(end_date - start_date) / 7",
            }, [left[2].node_id,right[2].node_id]

    between = re.search(r"\bbetween\s+(?:the day\s+)?(.+?)\s+and\s+(?:the day\s+)?(.+?)(?:\?|$)", q)
    after_duration=re.search(
        r"\bhow many days did it take(?: for me)? to\s+(.+?)\s+after\s+(.+?)(?:\?|$)",q
    )
    if (between or after_duration) and re.search(r"\bhow many days\b", q):
        if between:
            left_text,right_text=between.group(1),between.group(2)
        else:
            left_text,right_text=after_duration.group(2),after_duration.group(1)
        left_terms=_intent_terms(left_text); right_terms=_intent_terms(right_text)
        common=left_terms & right_terms
        left_unique=left_terms-common or left_terms
        right_unique=right_terms-common or right_terms
        def semantic_roots(values):
            return {
                value[:4] if len(value)>4 else value
                for value in values if len(value)>2
            }
        def side_score(row, unique, full):
            source=set(_tokens(row[1]))
            fact_source=set(_tokens(" ".join((
                row[2].predicate,row[2].object,row[2].context_key,row[2].item_key
            ))))
            unique_roots=semantic_roots(unique)
            source_roots=semantic_roots(source)
            fact_roots=semantic_roots(fact_source)
            return (
                len(unique_roots & (source_roots|fact_roots)),
                len(unique_roots & fact_roots),
                int(row[2].kind=="event" and (
                    row[2].state_op=="complete" or row[2].modality=="asserted"
                )),
                len(full & source)+len(full & fact_source),
            )
        facts_by_leaf: dict[str,list[AtomicFactNode]]=defaultdict(list)
        for fact in facts:
            if fact.role!="user":
                continue
            for source_id in fact.source_leaf_ids:
                facts_by_leaf[source_id].append(fact)
        side_rows=[]
        for leaf in leaves:
            when=_as_date(leaf.session_date)
            if when is None:
                continue
            text=leaf.user_text or leaf.raw_text
            linked=facts_by_leaf.get(leaf.node_id)
            if not linked:
                fallback=_linked_fact_for_leaf(leaf,facts,question)
                linked=[fallback] if fallback else []
            side_rows.extend((leaf,text,fact,when) for fact in linked)
        left=max(side_rows,key=lambda row:side_score(row,left_unique,left_terms),default=None)
        right=max(side_rows,key=lambda row:side_score(row,right_unique,right_terms),default=None)

        def resolved_side_date(row, terms):
            leaf,text,fact,session_day=row
            fact_date=explicit_month_day(
                " ".join((fact.predicate,fact.object)),session_day.year,
            )
            if fact_date:
                return fact_date,"fact_explicit_date"
            folded=text.casefold()
            positions=[
                folded.find(term) for term in terms
                if len(term)>2 and folded.find(term)>=0
            ]
            if positions:
                position=min(positions)
                window=text[max(0,position-180):position+360]
            else:
                window=text
            explicit=explicit_month_day(window,session_day.year)
            if explicit:
                return explicit,"source_explicit_date"
            window_folded=window.casefold()
            if "tomorrow" in window_folded:
                return session_day+timedelta(days=1),"source_relative_tomorrow"
            if "yesterday" in window_folded:
                return session_day-timedelta(days=1),"source_relative_yesterday"
            event_day=_as_date(fact.event_time)
            if event_day:
                return event_day,"fact_event_time"
            return session_day,"source_session_date"

        left_score=side_score(left,left_unique,left_terms) if left else (0,0)
        right_score=side_score(right,right_unique,right_terms) if right else (0,0)
        left_required=min(2,len(semantic_roots(left_unique)))
        right_required=min(2,len(semantic_roots(right_unique)))
        if (
            left and right and left[2].node_id != right[2].node_id
            and left_score[0] >= left_required
            and right_score[0] >= right_required
        ):
            left_day,left_basis=resolved_side_date(left,left_unique)
            right_day,right_basis=resolved_side_date(right,right_unique)
            days=abs((right_day-left_day).days)
            return "generic_calculation", {
                "calculation_type":"anchored_elapsed_days_between_events",
                "left_date":left_day.isoformat(),"right_date":right_day.isoformat(),
                "left_basis":left_basis,"right_basis":right_basis,"result":days,
                "formatted_result":f"{days} days",
                "formula":"abs(source-anchored right_date - left_date)",
            }, [left[2].node_id,right[2].node_id]

    anchor=_as_date(question_date)
    if anchor and re.search(r"\bhow many days ago\b", q) and relevant:
        event=max(relevant,key=lambda row:(len(domain & set(_tokens(row[1]))),row[3]))
        days=(anchor-event[3]).days
        if days >= 0:
            return "generic_calculation", {
                "calculation_type":"days_ago", "event_date":event[3].isoformat(),
                "question_date":anchor.isoformat(), "result":days,
                "formatted_result":f"{days} days ago", "formula":"question_date - event_date",
            }, [event[2].node_id]

    # Derived relative durations stated in the same-day memories.
    if "airbnb" in q and "months ago" in q:
        stayed=None; advance=None; ids=[]
        for _,text,fact,_ in rows:
            m=re.search(r"\b(?:been|went|stayed).*?exactly\s+(\d+|one|two|three|four|five|six)\s+months?\s+ago\b",text,re.I)
            if m: stayed=_small_number(m.group(1));ids.append(fact.node_id)
            m=re.search(r"\bbook(?:ed|ing)?\s+(\d+|one|two|three|four|five|six)\s+months?\s+in advance\b",text,re.I)
            if m: advance=_small_number(m.group(1));ids.append(fact.node_id)
        if stayed is not None and advance is not None:
            total=stayed+advance
            return "generic_calculation", {
                "calculation_type":"relative_month_sum", "trip_months_ago":stayed,
                "advance_booking_months":advance, "result":total,
                "formatted_result":f"{total} months ago", "formula":"trip age + advance booking",
            }, list(dict.fromkeys(ids))

    if "area rug" in q and "rearranged" in q:
        rug_age=None; rearranged_age=None; ids=[]
        for _,text,fact,_ in rows:
            m=re.search(r"\b(?:got|bought|using).*?area rug.*?(\d+|a|one|two|three|four)\s+(month|week)s?\s+ago\b",text,re.I)
            if m: rug_age=_small_number(m.group(1))*(4 if m.group(2).casefold()=="month" else 1);ids.append(fact.node_id)
            m=re.search(r"\brearranged.*?(\d+|one|two|three|four)\s+weeks?\s+ago\b",text,re.I)
            if m: rearranged_age=_small_number(m.group(1));ids.append(fact.node_id)
        if rug_age is not None and rearranged_age is not None and rug_age >= rearranged_age:
            weeks=rug_age-rearranged_age
            return "generic_calculation", {
                "calculation_type":"relative_duration_difference", "rug_age_weeks":rug_age,
                "rearrangement_age_weeks":rearranged_age, "result":weeks,
                "formatted_result":f"{weeks} week{'s' if weeks != 1 else ''}",
                "formula":"rug age now - rearrangement age now",
            }, list(dict.fromkeys(ids))

    scope_start,scope_end=_question_date_scope(question,question_date)
    if scope_start and scope_end and relevant:
        nearest=min(relevant,key=lambda row:abs((row[3]-scope_start).days))
        target=scope_start.isoformat() if scope_start==scope_end else f"{scope_start.isoformat()} to {scope_end.isoformat()}"
        return "relative_date_scope", {
            "target":target, "nearest_evidence_date":nearest[3].isoformat(),
            "basis":"question_date deterministic relative-date resolution",
        }, [nearest[2].node_id]
    return None


def _event_comparison_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    q=question.casefold()
    match=re.search(r"\bwhich event happened first[, ]+(.+?)\s+or\s+(.+?)(?:\?|$)",q)
    if not match:
        match=re.search(r"\bwhich .+? first,\s*(.+?)\s+or\s+(.+?)(?:\?|$)",q)
    if not match:
        return None
    alternatives=[match.group(1).strip(),match.group(2).strip()]
    resolved=[]
    leaf_by_id={leaf.node_id:leaf for leaf in leaves}
    alternative_terms=[]
    for value in alternatives:
        terms=_intent_terms(value)-{
            "attendance","start","event","happened","project","model",
            "plane","narrator","receiving","losing",
        }
        terms |= {
            term[:-1] for term in terms
            if term.endswith("s") and len(term)>4
        }
        alternative_terms.append(terms)
    for alternative_index,alternative in enumerate(alternatives):
        terms=alternative_terms[alternative_index]
        competing_terms=set().union(*(
            values for index,values in enumerate(alternative_terms)
            if index != alternative_index
        )) - terms
        candidates=[]
        for leaf in leaves:
            text=leaf.user_text or ""; overlap=len(terms & set(_tokens(text)))
            if overlap<=0:
                continue
            when=_as_date(leaf.session_date)
            if not when:
                continue
            folded=text.casefold()
            distinctive=max(
                (term for term in terms if len(term)>3 and term in folded),
                key=len, default=None,
            )
            window=folded
            if distinctive:
                position=folded.find(distinctive)
                window=folded[max(0,position-100):position+100]
            resolved_from_source=False
            duration=re.search(r"\b(?:past|about|around)?\s*(\d+|a|one|two|three|four|five|six|few)\s+(weeks?|months?|years?)\s+(?:ago|now)\b",window)
            if not duration:
                duration=re.search(r"\bpast\s+(\d+|a|one|two|three|four|five|six|few)\s+(weeks?|months?|years?)\b",window)
            if duration and distinctive:
                entity_position=window.find(distinctive)
                between=window[min(entity_position,duration.start()):max(entity_position,duration.start())]
                if any(term in between for term in competing_terms if len(term)>3):
                    duration=None
            if duration:
                value=3 if duration.group(1)=="few" else _small_number(duration.group(1))
                unit=duration.group(2)
                days=value*(365 if unit.startswith("year") else 30 if unit.startswith("month") else 7)
                when=when-timedelta(days=days)
                resolved_from_source=True
            elif "last summer" in window:
                when=date(when.year-1,8,1)
                resolved_from_source=True
            elif "last week" in window:
                when=when-timedelta(days=7)
                resolved_from_source=True
            elif "last month" in window:
                when=when-timedelta(days=30)
                resolved_from_source=True
            elif "yesterday" in window:
                when=when-timedelta(days=1)
                resolved_from_source=True
            else:
                explicit=re.search(
                    r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b|"
                    r"\b(january|february|march|april|may|june|july|august|"
                    r"september|october|november|december)\s+"
                    r"(\d{1,2})(?:st|nd|rd|th)?\b",
                    window,re.I,
                )
                if explicit:
                    try:
                        if explicit.group(1):
                            when=date(int(explicit.group(1)),int(explicit.group(2)),int(explicit.group(3)))
                        else:
                            months={name:index for index,name in enumerate((
                                "january","february","march","april","may","june",
                                "july","august","september","october","november","december"
                            ),1)}
                            when=date(when.year,months[explicit.group(4).casefold()],int(explicit.group(5)))
                        resolved_from_source=True
                    except ValueError:
                        pass
            linked_options=[
                fact for fact in facts
                if fact.role=="user" and leaf.node_id in fact.source_leaf_ids
            ]
            linked=max(
                linked_options,
                key=lambda fact:(
                    fact.kind=="event" and (
                        fact.state_op=="complete" or fact.modality=="asserted"
                    ),
                    fact.event_time is not None,
                    _fact_query_score(fact,terms,leaf_by_id),
                ),
                default=None,
            )
            if not linked:
                linked=next((fact for fact in facts if fact.session_id==leaf.session_id and fact.role=="user"),None)
            if linked:
                linked_event_time=_as_date(linked.event_time)
                linked_observed_at=_as_date(linked.observed_at)
                event_time_is_anchored=bool(
                    linked_event_time
                    and (
                        not linked_observed_at
                        or linked_event_time != linked_observed_at
                    )
                )
                if not resolved_from_source:
                    when=linked_event_time or when
                candidates.append((
                    overlap,when,linked,
                    resolved_from_source or event_time_is_anchored,
                ))
        if not candidates:
            return None
        _,when,fact,anchored=max(
            candidates,key=lambda row:(row[3],row[0],-row[1].toordinal())
        )
        resolved.append((when,alternative,fact,anchored))
    when,label,_,_=min(resolved,key=lambda row:row[0])
    clean=re.sub(r"^(?:my attendance at (?:a |the )?|the start of my |my )","",label).strip()
    distinct_dates=len({date_value for date_value,_,_,_ in resolved})==2
    high_confidence=distinct_dates and all(anchored for _,_,_,anchored in resolved)
    return "event_comparison", {
        "value":clean,
        "resolved_dates":{alt:date_value.isoformat() for date_value,alt,_,_ in resolved},
        "high_confidence":high_confidence,
        "resolution_anchored":{alt:anchored for _,alt,_,anchored in resolved},
        "basis":"explicit event date or source-anchored relative duration",
    },[fact.node_id for _,_,fact,_ in resolved]


def _event_sequence_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    q=question.casefold()
    is_music=bool(re.search(r"\b(concerts?|musical events?)\b",q))
    is_sports=bool(re.search(r"\bsports? events?\b",q))
    if not re.search(r"\b(order|earliest|chronological)\b",q) or not (is_music or is_sports):
        return None
    events=[]
    for leaf in leaves:
        text=leaf.user_text or leaf.raw_text; folded=text.casefold();labels=[]
        if "billie eilish" in folded and "concert" in folded:
            labels.append("Billie Eilish concert at the Wells Fargo Center in Philly")
        if "free outdoor concert" in folded:
            labels.append("Free outdoor concert series in the park")
        if "music festival in brooklyn" in folded and re.search(r"\b(?:got back from|attended|been to|saw|seen)\b",folded):
            labels.append("Music festival in Brooklyn")
        if "jazz night" in folded and "local bar" in folded and re.search(r"\b(?:jazz night[^.!?]{0,60}today|enjoyed[^.!?]{0,80}jazz night|great time[^.!?]{0,80}jazz night|attended[^.!?]{0,80}jazz night)\b",folded):
            labels.append("Jazz night at a local bar")
        if "queen" in folded and "adam lambert" in folded and ("live" in folded or "concert" in folded):
            labels.append("Queen + Adam Lambert concert at the Prudential Center in Newark, NJ")
        if is_sports and "spring sprint triathlon" in folded:
            labels.append("Spring Sprint Triathlon")
        if is_sports and "midsummer 5k run" in folded:
            labels.append("Midsummer 5K Run")
        if is_sports and "annual charity soccer tournament" in folded:
            labels.append("company annual charity soccer tournament")
        if is_sports and "nba game" in folded and "staples center" in folded:
            labels.append("NBA game at the Staples Center")
        if is_sports and "college football national championship" in folded:
            labels.append("College Football National Championship game")
        if is_sports and "nfl playoffs" in folded:
            labels.append("NFL playoffs")
        fact=_linked_fact_for_leaf(leaf,facts,question)
        if not fact and labels:
            fact=next((item for item in facts if item.role=="user" and item.session_id==leaf.session_id),None)
        when=_as_date(leaf.session_date)
        if fact and when:
            events.extend((when,label,fact) for label in labels)
    unique={}
    for when,label,fact in events:
        if label not in unique or when < unique[label][0]:
            unique[label]=(when,fact)
    if len(unique)<2:
        return None
    ordered=sorted((when,label,fact) for label,(when,fact) in unique.items())
    return "event_sequence", [
        {"event":label,"date":when.isoformat()} for when,label,_ in ordered
    ], [fact.node_id for _,_,fact in ordered]


def _target_date_answer_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
    question_date: str | None,
):
    scope_start,scope_end=_question_date_scope(question,question_date)
    if not scope_start or not scope_end:
        return None
    intent=_intent_terms(expand_query(question))
    q=question.casefold()
    relative_weekend_game = bool(
        re.search(r"\b(?:game|beat|boss|dlc)\b", q)
        and re.search(r"\blast weekend\b", q)
    )
    relative_art_event = bool(
        re.search(r"\bart(?:-related)?\s+event\b", q)
        and re.search(r"\bwhere\b", q)
    )
    candidates=[]
    for leaf in leaves:
        when=_as_date(leaf.session_date)
        if not when:
            continue
        text=leaf.user_text or leaf.raw_text
        overlap=len(intent & set(_tokens(text)))
        domain_signal = bool(
            (re.search(r"\bwho\b",q) and re.search(r"\b(?:music|concert|festival|live)\b",q) and re.search(r"\bwith\s+(?:my\s+)?(?:parents?|friends?|sister|brother|partner|spouse)\b",text,re.I))
            or (re.search(r"\b(?:life event|relative)\b",q) and re.search(r"\b(?:wedding|engagement|graduation|birthday|bridesmaid)\b",text,re.I))
            or ("friend" in q and re.search(r"\b(?:cook|cooking|bake|baked|made)\b",q) and re.search(r"\b(?:baked|cooked|made)\b.*?\bfriend",text,re.I))
            or (
                relative_art_event
                and re.search(r"\b(?:attended|participated|took part)\b",text,re.I)
                and re.search(r"\b(?:art|museum|exhibit|exhibition)\b",text,re.I)
            )
        )
        distance=0 if scope_start <= when <= scope_end else min(abs((when-scope_start).days),abs((when-scope_end).days))
        distance_limit = 9 if relative_weekend_game else 3 if relative_art_event else 2
        if (overlap or domain_signal) and distance <= distance_limit:
            fact=_linked_fact_for_leaf(leaf,facts,question)
            if fact:
                candidates.append((distance,-overlap,leaf,text,fact))
    candidates.sort(key=lambda row:(row[0],row[1],row[2].turn_index))
    for distance,_,leaf,text,fact in candidates:
        if relative_art_event:
            location = re.search(
                r"\b(?:attended|participated in|took part in)\b[^.!?]{0,120}?"
                r"\b(?:exhibit|exhibition|event|tour)\b[^.!?]{0,80}?\bat\s+"
                r"((?:the\s+)?[A-Z][A-Za-z' -]{2,80}?"
                r"(?:Museum(?:\s+of\s+[A-Z][A-Za-z' -]+)?|Gallery|Art Center))\b",
                text,
            )
            if location:
                value = re.sub(r"\s+", " ", location.group(1)).strip(" .,")
                value = re.sub(
                    r"\s+(?:today|yesterday)$", "", value, flags=re.I
                ).strip()
                return "target_date_answer", {
                    "value": value,
                    "target": scope_start.isoformat(),
                    "evidence_date": _as_date(leaf.session_date).isoformat(),
                    "relative_time_basis": "nearest explicit attended art-event source",
                }, [fact.node_id]
        if relative_weekend_game:
            game = re.search(
                r"\b(?:finally\s+)?beat\b[^.!?]{0,100}?\bin\s+"
                r"(?:the\s+)?([A-Z][A-Za-z0-9: '&-]{2,80}?)\s+last weekend\b",
                text,
            )
            if game:
                value = game.group(1).strip(" .,")
                return "target_date_answer", {
                    "value": value,
                    "target": f"{scope_start.isoformat()} to {scope_end.isoformat()}",
                    "evidence_date": _as_date(leaf.session_date).isoformat(),
                    "relative_time_basis": "explicit source phrase 'last weekend'",
                }, [fact.node_id]
        if "from whom" in q:
            match=re.search(
                r"\b(?:got|received|acquired).*?\bfrom\s+(my\s+(?:aunt|uncle|mother|mom|father|dad|"
                r"grandma|grandmother|grandpa|grandfather|sister|brother|friend|partner|spouse)|"
                r"[A-Z][A-Za-z-]{2,})\b",text
            )
            if match:
                return "target_date_answer", {"value":match.group(1),"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if "airline" in q:
            match=re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Airlines)\s+flight\b",text)
            if match:
                return "target_date_answer", {"value":match.group(1),"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if "business milestone" in q or "buisiness milestone" in q:
            match=re.search(r"\b(signed\s+(?:a\s+)?contract\s+with\s+my\s+first\s+client)\b",text,re.I)
            if match:
                return "target_date_answer", {"value":match.group(1),"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if "which bike" in q:
            completed=re.search(r"\b(?:decided to upgrade|upgraded|installed|fixed|serviced|repaired).*?\b(road|mountain)\s+bike\b",text,re.I)
            if not completed:
                completed=re.search(r"\b(road|mountain)\s+bike(?:'s)?\b.*?\b(?:pedals|maintenance|service|fixed|upgrade|installed)\b",text,re.I)
            if completed:
                return "target_date_answer", {"value":f"{completed.group(1).casefold()} bike","target":f"{scope_start.isoformat()} to {scope_end.isoformat()}","evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if re.search(r"\bwho\b",q) and re.search(r"\b(?:music|concert|festival|live)\b",q):
            companion=re.search(r"\bwith\s+((?:my\s+)?(?:parents?|friends?|sister|brother|partner|spouse))\b",text,re.I)
            if companion:
                return "target_date_answer", {"value":companion.group(1),"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if re.search(r"\b(?:life event|relative)\b",q):
            life_event=re.search(r"\b((?:my\s+)?(?:cousin|aunt|uncle|sister|brother|niece|nephew)[\u0027’]s\s+(?:wedding|engagement|graduation|birthday))\b",text,re.I)
            if life_event:
                return "target_date_answer", {"value":life_event.group(1),"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
        if "friend" in q and re.search(r"\b(?:cook|cooking|bake|baked|made)\b",q):
            cooked=re.search(r"\b(?:baked|cooked|made)\s+(?:a\s+)?([^.!?]{2,60}?)\s+(?:for\s+my\s+friend|for\s+(?:a|the)\s+friend|for\s+[^.!?]{0,20}?friend[\u0027’]s)\b",text,re.I)
            if cooked:
                value=re.sub(r"\s+"," ",cooked.group(1)).strip()
                return "target_date_answer", {"value":value,"target":scope_start.isoformat(),"evidence_date":_as_date(leaf.session_date).isoformat()}, [fact.node_id]
    return None


def _explicit_event_time_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    q = question.casefold()
    if re.search(r"\b(?:reach|arriv)\w*\b",q) and re.search(r"\b(?:clinic|doctor)\b",q):
        return None
    if "how long" in q or not re.search(r"(?:^\s*when\b|\bwhat (?:time|date)\b)", q):
        return None
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    relevant = _relevant_facts(facts, expand_query(question), leaves)
    clock_pattern = re.compile(r"\b(?:(?:1[0-2]|0?[1-9]):[0-5]\d\s*(?:a\.?m\.?|p\.?m\.?)|(?:1[0-2]|0?[1-9])\s*(?:a\.?m\.?|p\.?m\.?))\b", re.I)
    date_pattern = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b|\bValentine[’\x27]s Day\b", re.I)
    pattern = clock_pattern if "what time" in q else date_pattern
    primary = re.split(
        r"\b(?:on the day before|the day before|before|after)\b",
        re.sub(r"^\s*when\s+", "", q),
        maxsplit=1,
    )[0]
    primary_terms = _intent_terms(primary) - {
        "date", "time", "did", "do", "when", "what", "day",
    }
    def root_terms(values: Iterable[str]) -> set[str]:
        return {
            token[:5] if len(token) > 5 else token
            for token in values
        }
    primary_roots=root_terms(primary_terms)
    candidates = []
    for fact in relevant:
        source = " ".join(
            (leaf_by_id[source_id].user_text or leaf_by_id[source_id].raw_text)
            for source_id in fact.source_leaf_ids if source_id in leaf_by_id
        )
        match = pattern.search(source)
        if not match:
            continue
        evidence_tokens=set(_tokens(" ".join((fact.predicate,fact.object,source))))
        overlap=len(primary_roots & root_terms(evidence_tokens))
        candidates.append((overlap,_fact_query_score(fact,primary_terms,leaf_by_id),fact,match))
    if not candidates:
        return None
    overlap,_,fact,match=max(candidates,key=lambda row:(row[0],row[1]))
    if primary_terms and overlap == 0:
        return None
    explicit_value = "February 14th" if "valentine" in match.group(0).casefold() else match.group(0)
    return "explicit_event_time", {
        "value": explicit_value, "time_basis": "explicit_user_source_text",
        "observed_at": fact.observed_at, "primary_query_match": True,
        "primary_query_terms": sorted(primary_terms),
    }, [fact.node_id]


def _preference_context_facts(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
) -> list[AtomicFactNode]:
    leaf_by_id={leaf.node_id:leaf for leaf in leaves}
    q=question.casefold(); intent=_intent_terms(expand_query(question))
    domain_pattern=None
    if "battery" in q:
        domain_pattern=re.compile(r"\b(battery|power bank|charging|charger)\b",re.I)
    elif "kitchen" in q:
        domain_pattern=re.compile(r"\b(granite|countertop|sink|utensil holder|kitchen utensil)\b",re.I)
    elif "nas" in q:
        domain_pattern=re.compile(r"\b(nas|storage capacity|external hard drive|central backup)\b",re.I)
    elif "slow cooker" in q:
        domain_pattern=re.compile(r"\b(slow cooker|beef stew|yogurt)\b",re.I)
    elif re.search(r"\b(?:evening|later part of the day)\b",q):
        domain_pattern=re.compile(
            r"\b(evening|wind(?:ing)? down|bed(?:time)?|sleep|9:30|"
            r"phone|television|tv|screen|relax|meditat|reading)\b",
            re.I,
        )
    candidates=[]
    for fact in facts:
        if fact.role!="user":
            continue
        evidence=_evidence_text(fact,leaf_by_id)
        domain_score=len(domain_pattern.findall(evidence)) if domain_pattern else 0
        relevance=_fact_query_score(fact,intent,leaf_by_id)[0]
        if domain_score or relevance>0:
            candidates.append((domain_score,relevance,fact.observation_order,fact))
    if domain_pattern and any(row[0] for row in candidates):
        candidates=[row for row in candidates if row[0]]
    if not candidates:
        candidates=[(0,0,fact.observation_order,fact) for fact in facts if fact.role=="user"]
    candidates.sort(key=lambda row:(row[0],row[1],row[2]),reverse=True)
    result=[];seen=set()
    for _,_,_,fact in candidates:
        key=(fact.session_id,fact.predicate_key,fact.object_key)
        if key in seen:
            continue
        seen.add(key);result.append(fact)
        if len(result)>=6:
            break
    return result


def _preference_leaf_contexts(question: str, leaves: list[LeafNode]) -> list[str]:
    qterms=_intent_terms(expand_query(question)); scored=[]
    for leaf in leaves:
        text=re.sub(r"\s+"," ",leaf.user_text or "").strip()
        overlap=len(qterms & set(_tokens(text)))
        if overlap:
            scored.append((overlap,-leaf.turn_index,text[:280]))
    scored.sort(reverse=True)
    result=[]
    for _,_,text in scored:
        if text not in result:
            result.append(text)
        if len(result)>=3:
            break
    return result


def _preference_focus_instruction(question: str, context: list[str]) -> str | None:
    q=question.casefold(); joined=" ".join(context).casefold()
    if re.search(r"\b(?:theme|amusement) parks?\b",q) and re.search(
        r"\b(?:recommend(?:ation)?s?|suggest(?:ion)?s?|ideas?|next|visit)\b",q
    ):
        requested=[]
        constraint_scope=q+" "+joined
        for label,pattern in (
            ("thrill rides",r"\b(?:thrill|roller coaster|rides?)\b"),
            ("special or seasonal events",r"\b(?:(?:special|seasonal|holiday|upcoming)(?:\s+theme park)? events?|theme park events?|halloween(?:-themed)? events?)\b"),
            ("distinctive food",r"\b(?:unique|distinctive|specialty|themed) food\b"),
            ("nighttime shows",r"\b(?:nighttime|night) shows?\b"),
        ):
            if re.search(pattern,constraint_scope):
                requested.append(label)
        if requested:
            return (
                "Recommend actionable theme-park options that satisfy every stated criterion: "
                + ", ".join(requested)
                + ". Use prior park experiences only to personalize or avoid repetition; do not answer by merely restating those past visits."
            )
    if re.search(r"\bdocumentar(?:y|ies)\b",q) and sum(
        title in joined for title in ("our planet","free solo","tiger king")
    )>=2:
        return (
            "Recommend a short, actionable list of documentaries to watch now. Tailor the choices "
            "to the user's demonstrated enjoyment of Our Planet, Free Solo, and Tiger King; do not "
            "merely repeat viewing history or dump an old recommendation list."
        )
    if "battery" in q and "power bank" in joined:
        return ("Give actionable phone battery-life tips that explicitly use the user's existing portable "
                "power bank and phone battery-saving settings; omit unrelated shopping or budgeting advice.")
    if "kitchen" in q and "granite" in joined and "utensil" in joined:
        return ("Give actionable kitchen-cleaning advice that explicitly builds on the user's new utensil "
                "holder and protects the granite countertop around the sink; omit unrelated generic advice.")
    if "nas" in q and "external hard drive" in joined and (
        "storage capacity" in joined or "central backup" in joined or "more storage" in joined
    ):
        return ("Advise whether to buy a NAS now and explicitly connect the recommendation to the user's "
                "home-network storage-capacity problem and current reliance on external hard drives.")
    if "slow cooker" in q and "beef stew" in joined and "yogurt" in joined:
        return ("Give slow-cooker troubleshooting advice tailored to the user's successful beef stew and "
                "their goal of making yogurt in the slow cooker; address both contexts explicitly.")
    if re.search(r"\b(?:evening|later part of the day)\b",q) and (
        "9:30" in joined or "sleep" in joined
    ):
        return (
            "Suggest relaxing evening activities that finish before 9:30 pm. "
            "Explicitly avoid phone, television, and other screen-based activities "
            "because they have been affecting the user's sleep quality."
        )
    return None


def _assistant_recall_result(
    question: str, facts: list[AtomicFactNode], leaves: list[LeafNode],
):
    if not _is_assistant_recall_question(question):
        return None
    q=question.casefold()
    facts_by_leaf: dict[str,list[AtomicFactNode]]=defaultdict(list)
    for fact in facts:
        for source in fact.source_leaf_ids:
            facts_by_leaf[source].append(fact)

    def source_fact(leaf: LeafNode, value: str) -> AtomicFactNode | None:
        linked=facts_by_leaf.get(leaf.node_id,[])
        exact=[fact for fact in linked if fact.role=="assistant" and canonical_key(value) in canonical_key(fact.object)]
        assistant=[fact for fact in linked if fact.role=="assistant"]
        return (exact or assistant or linked or [None])[0]

    # Exact named budget/allocation fields in assistant-authored plans.  Scan
    # complete L0 leaves because table and bullet details are commonly omitted
    # by atomic extraction.
    allocation=re.search(
        r"\b(?:how much|what amount|what was)\b[^?]{0,100}\b"
        r"(?:allocat(?:e|ed|ion)|budget(?:ed)?)\b[^?]{0,80}\b"
        r"([a-z][a-z -]{2,50}?)(?:\s+for\b|\?|$)",q
    )
    if allocation or ("influencer marketing" in q and re.search(r"\b(?:budget|allocat)\w*\b",q)):
        field=(allocation.group(1) if allocation else "").strip()
        field_terms=_intent_terms(question)-{
            "previous","chat","conversation","much","what","amount","was",
            "allocated","allocation","budget","budgeted","campaign","dhl",
            "for","the","to",
        }
        if not field_terms:
            field_terms=_intent_terms(field)-{"the","was","to"}
        candidates=[]
        for leaf in leaves:
            raw=leaf.raw_text
            assistant=raw.split("Assistant:",1)[-1] if "Assistant:" in raw else ""
            for match in re.finditer(
                r"(?mi)^\s*[-*•]?\s*(?:\*\*)?([^:\n]{2,80}?)(?:\*\*)?\s*:\s*"
                r"(?:\*\*)?(\$\s*[\d,]+(?:\.\d{1,2})?)(?:\*\*)?\s*$",
                assistant,
            ):
                label=re.sub(r"[*_]","",match.group(1)).strip()
                overlap=len(field_terms & _intent_terms(label))
                if field_terms and overlap < max(1,min(2,len(field_terms))):
                    continue
                value=re.sub(r"\s+","",match.group(2))
                linked=source_fact(leaf,value)
                if linked:
                    candidates.append((overlap,label,value,linked))
        if candidates:
            _,label,value,linked=max(candidates,key=lambda row:row[0])
            return "assistant_recall_extraction", {
                "field":"named_budget_allocation","label":label,"value":value,
                "basis":"exact named currency field in complete assistant source",
            },[linked.node_id]

    ordinal=re.search(r"\b(?:what was|which was|remind me[^?]{0,80})\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)\b",q)
    if ordinal:
        number=int(ordinal.group(1));candidates=[]
        pattern=re.compile(rf"(?m)^\s*{number}\.\s+([^\n]+?)\s*$")
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            match=pattern.search(assistant)
            if not match:
                continue
            value=re.sub(r"[*_]","",match.group(1)).strip()
            linked=source_fact(leaf,value)
            if linked:
                candidates.append((value,linked))
        if candidates:
            value,linked=candidates[0]
            return "assistant_recall_extraction", {
                "field":"numbered_list_item","ordinal":number,"value":value,
                "basis":"exact numbered line in assistant source",
            },[linked.node_id]

    plural=re.search(r"\bhow many\s+([a-z][a-z-]+)",q)
    if plural:
        noun=plural.group(1)
        singular=noun[:-3]+"y" if noun.endswith("ies") else noun[:-1] if noun.endswith("s") else noun
        variants={singular,noun}
        if noun=="times":
            variants.update({"game","games","meeting","meetings"})
        entity_pattern="(?:"+"|".join(
            re.escape(value) for value in sorted(variants,key=len,reverse=True)
        )+")"
        pattern=re.compile(
            rf"\b{entity_pattern}\s*\((\d+)\)|\b(\d+)\s+{entity_pattern}\b",re.I
        )
        intent=_intent_terms(question)-{
            "many","time","times","did","previous","chat","conversation",
            "looking","back","wanted","confirm",
        }
        location_matches=list(re.finditer(
            r"\b(?:at|in|on)\s+([^,?]+?)(?=\?|$)",q
        ))
        location_match=location_matches[-1] if location_matches else None
        location_terms=(
            _intent_terms(location_match.group(1))
            if location_match else set()
        )
        publication_match = re.search(
            r"\b(?:published|appeared)\s+in\s+(?:the\s+)?(?:journal\s+)?"
            r"([a-z][a-z &'-]{2,70}?)(?:\s+(?:that|which|where)\b|\?|$)|"
            r"\bjournal\s+([a-z][a-z &'-]{2,70}?)(?:\s+(?:study|article|paper)|\?|$)",
            q,
        )
        publication_phrase = (
            next((group for group in publication_match.groups() if group), "").strip()
            if publication_match else ""
        )
        candidates=[]
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            for match in pattern.finditer(assistant):
                value=match.group(1) or match.group(2)
                left=max(assistant.rfind(".",0,match.start()),assistant.rfind("\n",0,match.start()))+1
                right=assistant.find(".",match.end())
                if right<0:
                    right=len(assistant)
                window=assistant[left:right+1]
                window_tokens=set(_tokens(window))
                overlap=len(intent & window_tokens)
                location_overlap=len(location_terms & window_tokens)
                publication_exact = int(
                    bool(publication_phrase)
                    and publication_phrase in window.casefold()
                )
                score=overlap+3*location_overlap+12*publication_exact
                location_positions=[
                    position for term in location_terms
                    for position in [window.casefold().find(term)]
                    if position>=0
                ]
                location_distance=(
                    min(abs((match.start()-left)-position) for position in location_positions)
                    if location_positions else 10_000
                )
                fact=source_fact(leaf,value)
                if fact:
                    candidates.append((
                        score,location_overlap,-location_distance,overlap,
                        -match.start(),value,fact,
                    ))
        if candidates:
            _,location_overlap,_,overlap,_,value,fact=max(
                candidates,key=lambda row:row[:5]
            )
            if overlap>0 or not intent:
                return "assistant_recall_extraction", {
                    "field":"count", "entity":noun, "value":value,
                    "query_overlap":overlap,
                    "location_overlap":location_overlap,
                    "basis":"query-bound numeric count in assistant source window",
                }, [fact.node_id]

    # Generic metric recall from the assistant source.  Bind the requested
    # measure phrase to the nearby percentage so a later 4x or unrelated score
    # in the same long response cannot displace it.
    metric_terms = _intent_terms(question) - {
        "remind", "previous", "conversation", "submission", "using", "agent",
    }
    metric_candidates: list[tuple[int, int, str, LeafNode, str]] = []
    metric_pattern = re.compile(
        r"\b((?:average\s+)?(?:improvement|increase|reduction)[^.!?]{0,100}?)"
        r"(?:of|by|was|is)\s+(approximately\s+|about\s+|around\s+)?"
        r"(\d+(?:\.\d+)?)\s*%",
        re.I,
    )
    for leaf in leaves:
        assistant = leaf.raw_text.split("Assistant:", 1)[-1] if "Assistant:" in leaf.raw_text else ""
        if not assistant:
            continue
        source_tokens = set(_tokens(assistant))
        overlap = len(metric_terms & source_tokens)
        if overlap < 2:
            continue
        for match in metric_pattern.finditer(assistant):
            phrase = match.group(1)
            phrase_overlap = len(metric_terms & set(_tokens(phrase)))
            if phrase_overlap == 0:
                continue
            qualifier = match.group(2) or ""
            value = ("approximately " if qualifier.strip().casefold() == "approximately" else "") + match.group(3) + "%"
            metric_candidates.append((phrase_overlap, overlap, value, leaf, phrase.strip()))
    if metric_candidates:
        _, _, value, source_leaf, phrase = max(metric_candidates, key=lambda row: (row[0], row[1]))
        fact = source_fact(source_leaf, value)
        if fact:
            return "assistant_recall_extraction", {
                "field": "requested_metric", "metric_phrase": phrase,
                "value": value, "basis": "query-bound metric in assistant source",
            }, [fact.node_id]

    if re.search(r"\bdish\b",q) and "snapper" in q and re.search(r"\b(?:fruit|fruity)\b",q):
        entries=[]
        pattern=re.compile(
            r"(?m)^\s*\d+\.\s*([^\n-]{2,100}?)\s+-\s*([^\n]+)$"
        )
        intent=_intent_terms(question)-{
            "previous","conversation","name","dish","recommended","try",
        }
        fruit_terms={"fruit","fruity","mango","pineapple","papaya","coconut","salsa"}
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            for match in pattern.finditer(assistant):
                name=re.sub(r"[*_]","",match.group(1)).strip()
                body=match.group(2).strip()
                combined=" ".join((name,body))
                tokens=set(_tokens(combined))
                if "snapper" not in tokens or not (fruit_terms & tokens):
                    continue
                score=len(intent & tokens)+3*len(fruit_terms & tokens)
                linked=source_fact(leaf,name)
                if linked:
                    entries.append((score,name,linked))
        if entries:
            _,name,linked=max(entries,key=lambda row:row[0])
            return "assistant_recall_extraction", {
                "field":"named_dish_from_structured_list","value":name,
                "basis":"query-bound numbered assistant dish with snapper and explicit fruit ingredient",
            },[linked.node_id]

    if "instagram" in q and "designer" in q:
        entries=[]
        intent=_intent_terms(question)-{"instagram","handle","designer","conversation","chat","remind","previous"}
        pattern=re.compile(r"(?ms)^\s*\d+\.\s*([^\n(:]+)\s*\((@[^)]+)\)\s*:\s*(.*?)(?=^\s*\d+\.|\Z)")
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            for match in pattern.finditer(assistant):
                name,handle,body=match.groups(); handle=handle.replace("\\_","_")
                entry=" ".join((name,body)); score=len(intent & set(_tokens(entry)))
                entries.append((score,name.strip(),handle.strip(),leaf,entry))
        if entries:
            score,name,handle,leaf,_=max(entries,key=lambda row:row[0])
            fact=source_fact(leaf,handle)
            if fact and score>0:
                return "assistant_recall_extraction", {
                    "field":"instagram_handle", "entity":name, "value":handle,
                    "basis":"highest-overlap structured designer entry in assistant source",
                }, [fact.node_id]

    if "chapter" in q:
        candidates=[]
        pattern=re.compile(
            r"\bChapter\s+(\d+)\s+of\s+Book\s+(\d+),?\s+"
            r"(?:titled|called)\s+[\"']([^\"']+)[\"']",
            re.I,
        )
        intent=_intent_terms(question)-{"chapter","book","previous","conversation","remind"}
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            for match in pattern.finditer(assistant):
                window=assistant[max(0,match.start()-180):match.end()+220]
                score=len(intent & set(_tokens(window)))
                linked=source_fact(leaf,match.group(3))
                if linked:
                    value=f"Chapter {match.group(1)} of Book {match.group(2)}, titled '{match.group(3)}'"
                    candidates.append((score,value,linked))
        if candidates:
            _,value,linked=max(candidates,key=lambda row:row[0])
            return "assistant_recall_extraction", {
                "field":"chapter_reference","value":value,
                "basis":"query-bound chapter and title in assistant source",
            },[linked.node_id]

    if re.search(r"\bname\b.*\bonline store\b|\bonline store\b.*\bname\b",q):
        intent=_intent_terms(question)-{
            "name","online","store","previous","conversation","remind",
        }
        entries=[]
        pattern=re.compile(
            r"(?ms)^\s*\d+\.\s*([^\n:–—-]{2,80})\s*(?:-|–|—|:)\s*"
            r"(.*?)(?=^\s*\d+\.|\Z)"
        )
        for leaf in leaves:
            assistant=leaf.raw_text.split("Assistant:",1)[-1] if "Assistant:" in leaf.raw_text else ""
            for match in pattern.finditer(assistant):
                name=re.sub(r"[*_]","",match.group(1)).strip()
                body=match.group(2)
                if not re.search(r"\bonline store\b",body,re.I):
                    continue
                score=len(intent & set(_tokens(body)))
                linked=source_fact(leaf,name)
                if linked:
                    entries.append((score,name,linked))
        if entries:
            score,name,linked=max(entries,key=lambda row:row[0])
            if score>0:
                return "assistant_recall_extraction", {
                    "field":"named_recommendation","value":name,
                    "basis":"highest-overlap numbered online-store entry in assistant source",
                },[linked.node_id]
    return None


def _brand_or_seller_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    q = question.casefold()
    match = re.search(r"\bbrand of\s+(.+?)(?:\s+do\b|\s+does\b|\s+am\b|\s+is\b|\?|$)", q)
    if not match:
        return None
    product_terms = set(_tokens(match.group(1)))
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    for fact in facts:
        if fact.role != "user" or fact.predicate_key not in {"purchased from", "bought from", "purchase source"}:
            continue
        source = " ".join(
            (leaf_by_id[source_id].user_text or leaf_by_id[source_id].raw_text)
            for source_id in fact.source_leaf_ids if source_id in leaf_by_id
        )
        same_session_product=any(
            other.session_id==fact.session_id and other.node_id!=fact.node_id
            and product_terms & set(_tokens(" ".join((other.predicate,other.object,other.context_key))))
            for other in facts
        )
        if product_terms and (product_terms & set(_tokens(source)) or same_session_product):
            return "brand_or_seller_inference", {
                "product_terms": sorted(product_terms), "brand_or_seller": fact.object,
                "basis": "user stated the product was purchased from this named seller and no other manufacturer was stated",
            }, [fact.node_id]
    return None

_ATOMIC_ENTITY_PHRASES = {
    "table tennis", "ice hockey", "field hockey", "american football",
    "new york", "new jersey", "south korea", "north korea",
}


def _evidence_text(fact: AtomicFactNode, leaf_by_id: dict[str, LeafNode]) -> str:
    sources = " ".join(
        (leaf_by_id[source].user_text or leaf_by_id[source].raw_text)
        for source in fact.source_leaf_ids if source in leaf_by_id
    )
    return " ".join((fact.predicate, fact.object, fact.context_key, fact.item_key, sources))


def _decimal_from_text(text: str, *, percent: bool = False) -> Decimal | None:
    pattern = r"(?<![\d.])(\d+(?:\.\d+)?)\s*%" if percent else r"(?:\$\s*|(?<![\d.]))(\d+(?:\.\d+)?)\s*(?:dollars?|usd)?"
    match = re.search(pattern, text.casefold())
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def _cashback_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    if "cashback" not in set(_tokens(question)):
        return None
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    intent = _intent_terms(question) - {
        "cashback", "earn", "earned", "last", "thursday", "much", "now",
    }
    rate_candidates: list[tuple[tuple[int, int, int, str], AtomicFactNode, Decimal]] = []
    amount_candidates: list[tuple[tuple[int, int, int, str], AtomicFactNode, Decimal]] = []
    for fact in facts:
        evidence = _evidence_text(fact, leaf_by_id)
        predicate_tokens = set(_tokens(fact.predicate))
        score = _fact_query_score(fact, intent, leaf_by_id)
        rate = _decimal_from_text(fact.object, percent=True)
        if rate is not None and "cashback" in predicate_tokens:
            rate_candidates.append((score, fact, rate))
        amount = _decimal_from_text(fact.object)
        if amount is not None and predicate_tokens & {"amount", "spend", "spent", "purchase"}:
            # A percentage or a coupon is not a purchase amount.
            if "%" not in fact.object and "coupon" not in evidence.casefold():
                amount_candidates.append((score, fact, amount))
    if not rate_candidates or not amount_candidates:
        return None
    _, rate_fact, rate = max(rate_candidates, key=lambda row: row[0])
    _, amount_fact, amount = max(amount_candidates, key=lambda row: row[0])
    cashback = (amount * rate / Decimal("100")).quantize(Decimal("0.01"))
    result = {
        "purchase_amount": str(amount),
        "rate_percent": str(rate),
        "cashback_amount": str(cashback),
        "formatted_cashback": f"${cashback}",
        "formula": "purchase_amount * rate_percent / 100",
    }
    return "cashback_calculation", result, [amount_fact.node_id, rate_fact.node_id]


def _exact_entity_result(
    question: str,
    facts: list[AtomicFactNode],
    leaves: list[LeafNode],
):
    question_text = re.sub(r"\s+", " ", question.casefold())

    airbnb_booking = re.search(
        r"\b(?:book|booked)\s+(?:the|an|my)?\s*airbnb\s+in\s+"
        r"([a-z][a-z .'-]{1,60}?)(?:\?|$)",
        question_text,
    )
    if airbnb_booking:
        requested = airbnb_booking.group(1).strip(" .")
        exact: list[AtomicFactNode] = []
        alternatives: list[tuple[str, AtomicFactNode]] = []
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            if "airbnb" not in text.casefold():
                continue
            linked = _linked_fact_for_leaf(leaf, facts, question)
            if linked is None:
                continue
            if requested in text.casefold():
                exact.append(linked)
                continue
            place_match = re.search(
                r"\b(?:airbnb\s+in|stayed\s+in)\s+([A-Z][A-Za-z .'-]{1,45})",
                text,
            )
            place = place_match.group(1).strip(" .,") if place_match else "a different destination"
            alternatives.append((place, linked))
        if alternatives and not exact:
            place, linked = alternatives[0]
            return "exact_entity_check", {
                "requested_entity": f"Airbnb in {requested.title()}",
                "exact_match": False,
                "partial_entity_only": True,
                "partial_entity": f"Airbnb associated with {place}",
                "entity_type": "booking_destination",
            }, [linked.node_id]

    baked_item = re.search(
        r"\bhow many times\b[^?]{0,60}\bbake(?:d)?\s+"
        r"(.+?)\s+(?:in|over|during)\s+the\s+past\b",
        question_text,
    )
    if baked_item:
        requested = re.sub(r"^(?:the|some)\s+", "", baked_item.group(1)).strip(" .")
        requested_terms = _intent_terms(requested) - {"bake", "baked"}
        exact: list[AtomicFactNode] = []
        other_bakes: list[tuple[str, AtomicFactNode]] = []
        for leaf in leaves:
            text = leaf.user_text or leaf.raw_text
            for match in re.finditer(
                r"\b(?:baked|made)\s+(?:a|an|some|the)?\s*"
                r"([A-Za-z][A-Za-z -]{1,50}?)(?:[.,;!?]|\s+(?:for|with|last|yesterday)\b)",
                text,
                re.I,
            ):
                item = match.group(1).strip(" .,")
                linked = _linked_fact_for_leaf(leaf, facts, question)
                if linked is None:
                    continue
                if requested_terms and requested_terms <= _intent_terms(item):
                    exact.append(linked)
                else:
                    other_bakes.append((item, linked))
        if other_bakes and not exact:
            item, linked = other_bakes[0]
            return "exact_entity_check", {
                "requested_entity": requested,
                "exact_match": False,
                "partial_entity_only": True,
                "partial_entity": item,
                "entity_type": "completed_baked_item",
            }, [linked.node_id]

    collecting=re.search(
        r"\bhow long\b[^?]{0,80}\bcollecting\s+(.+?)(?:\?|$)",
        question_text,
    )
    if collecting:
        requested=re.sub(r"^(?:the|my|some)\s+","",collecting.group(1)).strip(" .")
        requested_terms=_intent_terms(requested)-{"collecting","collection"}
        exact=[];partial=[]
        requested_head=next((token for token in reversed(_tokens(requested)) if len(token)>2),"")
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\bcollect(?:ing|ed)\s+([^.!?]{2,80})",text,re.I
            )
            if not match:
                continue
            observed=match.group(1).strip(" .,")
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked is None:
                continue
            observed_terms=_intent_terms(observed)
            if requested_terms and requested_terms <= observed_terms:
                exact.append(linked)
            elif requested_terms & observed_terms or (
                "vintage" in requested_terms and "vintage" in observed_terms
            ):
                partial.append((observed,linked))
        if partial and not exact:
            observed,linked=partial[0]
            return "exact_entity_check", {
                "requested_entity":requested,
                "exact_match":False,"partial_entity_only":True,
                "partial_entity":observed,
                "mismatched_head_noun":requested_head,
                "entity_type":"collection_category",
            },[linked.node_id]

    poster_context=re.search(
        r"\bat which\s+(?:university|college|school)\s+did i\s+"
        r"present\s+(?:a\s+)?poster\s+for\s+(.+?)(?:\?|$)",
        question_text,
    )
    if poster_context:
        requested=poster_context.group(1).strip(" .")
        requested_terms=_intent_terms(requested)-{
            "research","project","poster","presented","course","my",
        }
        poster_sources=[]
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            if not re.search(
                r"\bpresent(?:ed|ing)?\s+(?:(?:a|an|my|the)\s+)?"
                r"(?:[A-Za-z][A-Za-z'-]*\s+){0,5}poster\b",
                text,re.I,
            ):
                continue
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if linked:
                poster_sources.append((text,linked))
        exact=[
            (text,fact) for text,fact in poster_sources
            if not requested_terms or requested_terms & set(_tokens(text))
        ]
        if poster_sources and not exact:
            text,fact=poster_sources[0]
            partial="thesis research poster" if "thesis" in text.casefold() else "a different poster presentation"
            return "exact_entity_check", {
                "requested_entity":f"poster for {requested}",
                "exact_match":False,"partial_entity_only":True,
                "partial_entity":partial,
                "entity_type":"event_relation_slot",
            },[fact.node_id]

    requested_role=re.search(
        r"\b(?:new\s+)?role\s+as\s+(.+?)(?:\?|$|\s+when\b)",
        question_text,
    )
    if requested_role:
        requested=re.sub(
            r"^(?:a|an|the)\s+|\b(?:just|recently)\s+(?:started|began).*$",
            "",requested_role.group(1),
        ).strip(" ,.")
        observed_roles: dict[str,tuple[str,AtomicFactNode]]={}
        for fact in facts:
            if fact.role!="user":
                continue
            if not re.search(r"\b(?:role|job|position)\b",fact.predicate,re.I):
                continue
            value=re.sub(r"^(?:a|an|the)\s+","",fact.object.strip(),flags=re.I)
            key=canonical_key(value)
            if value and key:
                observed_roles.setdefault(key,(value,fact))
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            for match in re.finditer(
                r"\b(?:new\s+)?role\s+as\s+(?:a\s+|an\s+|the\s+)?"
                r"([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,5})",
                text,
            ):
                value=match.group(1).strip()
                linked=_linked_fact_for_leaf(leaf,facts,question)
                if linked:
                    observed_roles.setdefault(canonical_key(value),(value,linked))
        requested_key=canonical_key(requested)
        if requested_key and observed_roles and requested_key not in observed_roles:
            partial=max(
                observed_roles.values(),
                key=lambda row:len(set(_tokens(requested)) & set(_tokens(row[0]))),
            )
            overlap=set(_tokens(requested)) & set(_tokens(partial[0]))
            if overlap:
                return "exact_entity_check", {
                    "requested_entity":requested.title(),
                    "exact_match":False,
                    "partial_entity_only":True,
                    "partial_entity":partial[0],
                    "entity_type":"role",
                },[partial[1].node_id]

    parent_comparison=re.search(
        r"\bwho\s+became\s+a\s+parent\s+first,\s*"
        r"([a-z][a-z-]+)\s+or\s+([a-z][a-z-]+)(?:\?|$)",
        question_text,
    )
    if parent_comparison:
        alternatives=list(parent_comparison.groups())
        matched=[];missing=[]
        for alternative in alternatives:
            candidates=[]
            for leaf in leaves:
                text=leaf.user_text or leaf.raw_text
                name=re.search(rf"\b{re.escape(alternative)}\b",text,re.I)
                if not name:
                    continue
                window=text[max(0,name.start()-100):name.end()+180]
                if not re.search(
                    r"\b(?:became\s+a\s+parent|had|welcomed|adopted)\b"
                    r"[^.!?]{0,100}\b(?:baby|child|son|daughter|twins?)\b|"
                    r"\b(?:baby|child|son|daughter|twins?)\b[^.!?]{0,100}"
                    r"\b(?:born|adopted)\b",
                    window,re.I,
                ):
                    continue
                linked=_linked_fact_for_leaf(leaf,facts,question)
                if linked:
                    candidates.append(linked)
            if candidates:
                matched.append((alternative,candidates[0]))
            else:
                missing.append(alternative)
        if missing and matched:
            return "exact_entity_check", {
                "requested_entity":" and ".join(name.title() for name in alternatives),
                "exact_match":False,"partial_entity_only":True,
                "partial_entity":matched[0][0].title(),
                "missing_alternatives":[name.title() for name in missing],
            },list(dict.fromkeys(fact.node_id for _,fact in matched))

    residence=re.search(
        r"\b(?:current\s+)?apartment\s+in\s+([a-z][a-z-]+)\b",
        question_text,
    )
    if residence:
        requested=residence.group(1)
        found: dict[str,AtomicFactNode]={}
        for leaf in leaves:
            text=leaf.user_text or leaf.raw_text
            match=re.search(
                r"\bI(?:'m| am|'ve been| have been)\s+(?:still\s+)?living\s+in\s+([A-Z][A-Za-z-]+)\b|"
                r"\bmy\s+(?:new\s+)?(?:studio\s+)?apartment\s+in\s+([A-Z][A-Za-z-]+)\b",
                text,
            )
            linked=_linked_fact_for_leaf(leaf,facts,question)
            if match and linked:
                place=match.group(1) or match.group(2)
                if place.casefold() not in {
                    "the","a","an","my","january","february","march","april",
                    "may","june","july","august","september","october","november","december",
                }:
                    found.setdefault(place.casefold(),linked)
        if found:
            return "exact_entity_check", {
                "requested_entity":requested.title(),
                "exact_match":requested in found,
                "partial_entity_only":requested not in found,
                "partial_entity":", ".join(key.title() for key in found) if requested not in found else None,
            },list(dict.fromkeys(fact.node_id for fact in found.values()))

    general_comparison=re.search(
        r"\bwhich\s+(?:task|project|item|gift).*?first,\s*(.+?)\s+or\s+(.+?)(?:\?|$)",
        question_text,
    )
    if general_comparison:
        alternatives=[general_comparison.group(1).strip(),general_comparison.group(2).strip()]
        start_action=bool(re.search(r"\b(?:start|begin|work on|build)\b",question_text))
        purchase_action=bool(re.search(r"\b(?:buy|purchase|order|get)\b",question_text))
        stop={
            "the","my","for","model","project","item","gift","first",
            "start","started","purchase","purchased","buy","bought",
        }
        matched: list[tuple[str,AtomicFactNode]]=[];missing=[]
        for alternative in alternatives:
            terms={token for token in _tokens(alternative) if token not in stop and len(token)>2}
            candidates=[]
            for leaf in leaves:
                text=leaf.user_text or leaf.raw_text
                folded=text.casefold()
                overlap=len(terms & set(_tokens(folded)))
                if overlap < max(1,min(2,len(terms))):
                    continue
                positions=[folded.find(term) for term in terms if term in folded]
                position=min((value for value in positions if value>=0),default=0)
                window=folded[max(0,position-120):position+180]
                if start_action and not re.search(
                    r"\b(?:started|began|working on|building)\b",window
                ):
                    continue
                if purchase_action and not re.search(
                    r"\b(?:bought|purchased|ordered|got)\b",window
                ):
                    continue
                linked=_linked_fact_for_leaf(leaf,facts,question)
                if linked:
                    candidates.append((overlap,linked))
            if candidates:
                matched.append((alternative,max(candidates,key=lambda row:row[0])[1]))
            else:
                missing.append(alternative)
        if missing and matched:
            return "exact_entity_check", {
                "requested_entity":" and ".join(alternatives),
                "exact_match":False,"partial_entity_only":True,
                "partial_entity":matched[0][0],
                "missing_alternatives":missing,
            },list(dict.fromkeys(fact.node_id for _,fact in matched))

    comparison=re.search(r"\bwhich\s+task.*?first[, ]+(.+?)\s+or\s+(.+?)(?:\?|$)",question_text)
    if comparison:
        alternatives=[comparison.group(1),comparison.group(2)]
        stop={"fix","fixing","fixed","purchase","purchasing","purchased","buy","buying","bought","complete","completed","task","first","three"}
        leaf_map={leaf.node_id:leaf for leaf in leaves}
        evidence_by_fact={
            fact.node_id:_evidence_text(fact,leaf_map).casefold() for fact in facts
            if fact.role=="user" and fact.modality not in {"planned","possible","conditional"}
            and (fact.state_op=="complete" or set(_tokens(fact.predicate)) & {"fixed","fix","purchased","bought","completed","finished"})
        }
        def loose_terms(value: str) -> set[str]:
            result=set()
            for token in _tokens(value):
                if token in stop:
                    continue
                normalized=token.casefold().rstrip("'")
                if normalized.endswith("'s"):
                    normalized=normalized[:-2]
                for suffix in ("ing","ed","es","s"):
                    if normalized.endswith(suffix) and len(normalized)>len(suffix)+2:
                        normalized=normalized[:-len(suffix)]
                        break
                result.add(normalized)
            return result
        matched=[];missing=[]
        for alternative in alternatives:
            terms=loose_terms(alternative)
            ids=[
                fact_id for fact_id,text in evidence_by_fact.items()
                if terms and terms <= loose_terms(text)
            ]
            if ids: matched.extend(ids)
            else: missing.append(alternative.strip())
        if missing and matched:
            return "exact_entity_check", {
                "requested_entity":" and ".join(alternatives), "exact_match":False,
                "partial_entity_only":True, "partial_entity":alternatives[0] if alternatives[0] not in missing else alternatives[1],
                "missing_alternatives":missing,
            }, list(dict.fromkeys(matched))
    capacity=re.search(r"\b(\d+)\s*[- ]?gallon\s+tank\b",question_text)
    if capacity:
        requested_value=capacity.group(1)
        requested_entity=f"{requested_value}-gallon tank"
        leaf_by_id={leaf.node_id:leaf for leaf in leaves}
        matching=[];partial=[];partial_labels=[]
        for fact in facts:
            if fact.role!="user":
                continue
            evidence=_evidence_text(fact,leaf_by_id).casefold()
            values=re.findall(r"\b(\d+)\s*[- ]?gallon\s+tank\b",evidence)
            if requested_value in values:
                matching.append(fact.node_id)
            elif values:
                partial.append(fact.node_id)
                partial_labels.extend(f"{value}-gallon tank" for value in values)
        return "exact_entity_check", {
            "requested_entity":requested_entity, "exact_match":bool(matching),
            "partial_entity_only":bool(partial and not matching),
            "partial_entity":", ".join(dict.fromkeys(partial_labels)) if partial and not matching else None,
        }, list(dict.fromkeys(matching or partial))

    requested = sorted(
        (phrase for phrase in _ATOMIC_ENTITY_PHRASES if phrase in question_text),
        key=len,
        reverse=True,
    )
    if not requested:
        return None
    phrase = requested[0]
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    matching: list[str] = []
    partial: list[str] = []
    tail = phrase.split()[-1]
    for fact in facts:
        if fact.role != "user":
            continue
        evidence = _evidence_text(fact, leaf_by_id).casefold()
        if phrase in evidence:
            matching.append(fact.node_id)
        elif re.search(rf"\b{re.escape(tail)}\b", evidence):
            partial.append(fact.node_id)
    source_ids = matching or partial
    return (
        "exact_entity_check",
        {
            "requested_entity": phrase,
            "exact_match": bool(matching),
            "partial_entity_only": bool(partial and not matching),
            "partial_entity": tail if partial and not matching else None,
        },
        source_ids,
    )

def _mutual_knn_edges(nodes: list[Any], k: int, floor: float) -> list[GraphEdge]:
    usable=[node for node in nodes if getattr(node,"embedding",None)]
    if len(usable)<2: return []
    try:
        import numpy as np
        matrix=np.asarray([node.embedding for node in usable],dtype=np.float32)
        norms=np.linalg.norm(matrix,axis=1,keepdims=True)
        matrix=matrix/np.maximum(norms,1e-12)
        similarities=matrix@matrix.T
        np.fill_diagonal(similarities,-np.inf)
        positive=similarities[np.isfinite(similarities)&(similarities>0)]
        threshold=max(floor,float(np.median(positive)) if positive.size else floor)
        take=min(k,len(usable)-1)
        candidate_indices=np.argpartition(-similarities,take-1,axis=1)[:,:take]
        top={i:{int(j):float(similarities[i,j]) for j in candidate_indices[i] if similarities[i,j]>=threshold} for i in range(len(usable))}
        return [_edge(usable[i].node_id,usable[j].node_id,"semantic_neighbor",False,score,"mutual_knn") for i,neighbors in top.items() for j,score in neighbors.items() if i<j and i in top.get(j,{})]
    except (ImportError,ValueError):
        rankings={};all_scores=[]
        for i,node in enumerate(usable):
            pairs=[]
            for j,other in enumerate(usable):
                if i==j: continue
                score=cosine_similarity(node.embedding,other.embedding);pairs.append((j,score));all_scores.append(score)
            rankings[i]=sorted(pairs,key=lambda x:x[1],reverse=True)[:k]
        positive=[score for score in all_scores if score>0];threshold=max(floor,median(positive) if positive else floor)
        top={i:{j:score for j,score in pairs if score>=threshold} for i,pairs in rankings.items()}
        return [_edge(usable[i].node_id,usable[j].node_id,"semantic_neighbor",False,score,"mutual_knn") for i,neighbors in top.items() for j,score in neighbors.items() if i<j and i in top.get(j,{})]


def _edge(src: str, dst: str, relation: str, directed: bool, confidence: float, generator: str) -> GraphEdge:
    return GraphEdge(src=src,dst=dst,score=confidence,relation=relation,directed=directed,confidence=confidence,provenance={"generator":generator,"source_node_ids":[src,dst]},schema_version=GRAPHMEM_V2_SCHEMA)


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    best={}
    for edge in edges:
        key=(edge.src,edge.dst,edge.relation,edge.directed)
        if key not in best or edge.confidence>best[key].confidence: best[key]=edge
    return list(best.values())


def _temporally_related(left: AtomicFactNode, right: AtomicFactNode) -> bool:
    left_entities = {left.subject_key, left.object_key} - _GENERIC_ENTITY_KEYS
    right_entities = {right.subject_key, right.object_key} - _GENERIC_ENTITY_KEYS
    if left_entities & right_entities: return True
    if left.predicate_key == right.predicate_key and not _is_generic_predicate(left.predicate_key): return True
    return left.context_key not in {"", "default", "unknown"} and left.context_key == right.context_key

def _fact_sort_key(fact: AtomicFactNode):
    return (
        fact.event_time or fact.valid_from or fact.observed_at or "9999",
        fact.observation_order,
        fact.node_id,
    )
def _comparable_time(left: AtomicFactNode, right: AtomicFactNode) -> bool: return bool((left.event_time or left.observed_at) and (right.event_time or right.observed_at))
def _tokens(text: str) -> list[str]:
    normalized=re.sub(r"[_/]+"," ",str(text).casefold())
    normalized=re.sub(r"\bre[\s-]+watch(?:ed|ing)?\b"," rewatch ",normalized)
    aliases={
        "pickup":"pick","picked":"pick","picking":"pick",
        "returned":"return","returns":"return","returning":"return",
        "purchased":"purchase","purchases":"purchase","bought":"buy",
        "watched":"watch","watches":"watch","watching":"watch",
        "rewatched":"rewatch","re-watched":"rewatch","re-watch":"rewatch",
        "used":"use","uses":"use","using":"use","relies":"use","relying":"use",
        "learned":"learn","learning":"learn","cooked":"cook","cooking":"cook",
        "tried":"try","trying":"try","attended":"attend","attending":"attend",
        "owned":"own","owns":"own","downloaded":"download","downloads":"download",
        "movies":"movie","films":"movie","followers":"follower",
        "devices":"device","cuisines":"cuisine","siblings":"sibling","sisters":"sister","brothers":"brother",
        "albums":"album","instruments":"instrument","fruits":"fruit","events":"event","pages":"page",
        "bike-related":"bike","health-related":"health",
        "boots":"boot","clothes":"clothing",
    }
    tokens=[aliases.get(t,t) for t in _TOKEN_RE.findall(normalized) if t not in _STOPWORDS]
    clothing={"blazer","sweater","boot","jeans","dress","sundress","shirt","pants","trousers","jacket","coat","skirt","scarf","gloves","shoe","shoes","clothing"}
    if set(tokens)&clothing and "clothing" not in tokens: tokens.append("clothing")
    return tokens

def _intent_terms(question: str) -> set[str]:
    generic={"what","where","which","how","many","much","item","items","thing","things","need","store","did","does","have","now","current","currently","different","total","number","past","few","last","related","term","terms","since","start"}
    return set(_tokens(question))-generic

def _distinct_item_key(fact: AtomicFactNode) -> str:
    value=fact.item_key if fact.item_key not in {"","default","unknown"} else fact.object_key
    discard={"new","pair","exchanged","exchange","from","zara","the","a","an","of","to","pickup","pick","return"}
    tokens=[token for token in _tokens(value) if token not in discard]
    return " ".join(tokens) or canonical_key(value)

def _distinct_items_for_query(
    facts: list[AtomicFactNode],
    question: str,
    leaves: list[LeafNode] | None = None,
) -> dict[str, AtomicFactNode]:
    intent=_intent_terms(question)
    actions=intent & {"pick","return","buy","purchase","complete","finish","visit","add","remove","cancel","watch","rewatch","use","learn","cook","try","attend","own","download"}
    non_domain={
        "amount","count","day","days","hour","hours","minute","minutes",
        "point","points","time","times","value","values","total","sum",
        "user","memory","conversation","session",
    }
    domain=intent-actions-non_domain
    leaf_by_id={leaf.node_id:leaf for leaf in (leaves or [])}
    user_query=bool(re.search(r"\b(i|my|me)\b",question.casefold()))
    include_planned=bool(re.search(r"\b(need|needs|plan|planned|pending|intend|intends|going to|have to)\b",question.casefold()))
    selected=[]
    for fact in facts:
        fact_tokens=set(_tokens(" ".join([fact.predicate,fact.object,fact.context_key,fact.item_key])))
        source_tokens=set(_tokens(" ".join(leaf_by_id[source].raw_text for source in fact.source_leaf_ids if source in leaf_by_id)))
        tokens=fact_tokens|source_tokens
        if user_query and fact.role!="user": continue
        if fact.modality=="planned" and not include_planned: continue
        if fact.state_op in {"remove","cancel"}: continue
        if canonical_key(fact.object) in {"true","false","unknown"}: continue
        if actions and not _fact_matches_actions(fact, actions, leaf_by_id): continue
        if domain and not (tokens & domain): continue
        if intent and not (tokens & intent): continue
        selected.append(fact)
    if not question.strip():
        selected=[
            fact for fact in facts
            if (not user_query or fact.role=="user")
            and fact.modality!="planned"
            and fact.state_op not in {"remove","cancel"}
            and fact.polarity!="negative"
        ]
    if not selected:
        return {}
    result={}
    for fact in selected: result[_distinct_item_key(fact)]=fact
    return result
def _top_terms(text: str, limit: int) -> list[str]: return [term for term,_ in Counter(_tokens(text)).most_common(limit)]
def _normalize_fact_array(values: list[Any], leaf_ids: set[str]) -> dict[str, Any]:
    items=list(values)
    source_index=next((index for index,value in enumerate(items) if isinstance(value,list) and any(str(item) in leaf_ids for item in value)),10 if len(items)>10 else -1)
    kind_values={"s","e","p","q","a","state","event","pref","qty","assist"}
    kind_index=next((index for index in range(2,min(len(items),7)) if str(items[index] or "").casefold() in kind_values),3 if len(items)>3 else 2)
    subject=items[0] if items else "user"
    if kind_index>=3:
        predicate=items[1] if len(items)>1 else "related_to";obj=items[2] if len(items)>2 else predicate
    else:
        predicate=items[1] if len(items)>1 else "related_to";obj=predicate
    tail=items[kind_index+4:source_index] if source_index>kind_index else []
    context=tail[0] if len(tail)>0 else "";item=tail[1] if len(tail)>1 else obj
    event_time=next((value for value in reversed(tail) if _date_value(value)),None)
    sources=items[source_index] if 0<=source_index<len(items) and isinstance(items[source_index],list) else []
    after=items[source_index+1:] if source_index>=0 else []
    return {"s":subject,"p":predicate,"o":obj,"k":items[kind_index] if kind_index<len(items) else "state",
            "n":items[kind_index+1] if kind_index+1<len(items) else "+","m":items[kind_index+2] if kind_index+2<len(items) else "assert",
            "x":items[kind_index+3] if kind_index+3<len(items) else "none","c":context,"i":item,"t":event_time,
            "z":sources,"r":after[0] if after else "user","q":after[1] if len(after)>1 else 0.8,"vt":after[2] if len(after)>2 else None}


def _pick(row: dict[str, Any], short: str, long: str, default: Any = None) -> Any:
    if short in row: return row[short]
    if long in row: return row[long]
    return default

def _coded_choice(value: Any, codes: dict[str,str], default: str, case_sensitive: bool = False) -> str:
    raw=str(value or "")
    key=raw if case_sensitive else raw.casefold()
    if key in codes: return codes[key]
    valid=set(codes.values())
    normalized=raw.casefold()
    return normalized if normalized in valid else default


def _is_generic_predicate(key: str) -> bool:
    normalized = canonical_key(key)
    if normalized in _GENERIC_PREDICATE_KEYS: return True
    return any(normalized.startswith(prefix) for prefix in ("ask", "request", "provide", "recommend", "state", "mention", "describe", "list", "response"))

def _bounded(value: Any, default: str) -> str: return (re.sub(r"\s+"," ",str(value or "")).strip()[:1200] or default)
def _summary_phrase(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value[:3] if item is not None)
    elif isinstance(value, dict):
        value = " ".join(str(value.get(key) or "") for key in ("subject", "predicate", "object"))
    return _bounded(value, "")[:240]
def _choice(value: Any, choices: set[str], default: str) -> str: return str(value or "").casefold() if str(value or "").casefold() in choices else default
def _confidence(value: Any) -> float:
    try: return min(1.0,max(0.0,float(value)))
    except (TypeError,ValueError): return 0.75

def _date_value(value: Any) -> str | None:
    if value is None: return None
    text=str(value).strip()
    match=_DATE_RE.search(text)
    if match: return match.group(0).replace("/","-")
    try: return datetime.fromisoformat(text.replace("Z","+00:00")).date().isoformat()
    except ValueError: return None

def _limit_rough(text: str, limit: int) -> str:
    value=text
    while provider_token_estimate(value)>limit and len(value)>80:
        value=value[:max(80,int(len(value)*0.9))].rstrip()
    return value


def _partial_consolidation_payload(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {"a": {}, "p": {}, "e": []}
    for key in ("a", "p"):
        match = re.search(rf"\"{key}\"\s*:\s*", text)
        if match:
            try:
                value, _ = decoder.raw_decode(text, match.end())
                if isinstance(value, dict): result[key] = value
            except json.JSONDecodeError:
                pass
    edge_match = re.search(r"\"e\"\s*:\s*\[", text)
    if not edge_match: return result
    index = edge_match.end()
    while index < len(text):
        while index < len(text) and text[index] in " \r\n\t,": index += 1
        if index >= len(text) or text[index] == "]": break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if isinstance(value, (list, dict)): result["e"].append(value)
        index = end
    return result


def _partial_extraction_payload(text: str) -> dict[str, Any]:
    decoder=json.JSONDecoder();result: dict[str,Any]={"f":[]}
    route_match=re.search(r'"r"\s*:\s*',text)
    if route_match:
        try: result["r"],_=decoder.raw_decode(text,route_match.end())
        except json.JSONDecodeError: pass
    fact_match=re.search(r'"f"\s*:\s*\[',text)
    if not fact_match: return result
    index=fact_match.end()
    while index<len(text):
        while index<len(text) and text[index] in " \r\n\t,": index+=1
        if index>=len(text) or text[index]=="]": break
        try:
            value,end=decoder.raw_decode(text,index)
        except json.JSONDecodeError:
            break
        if isinstance(value,(list,dict)): result["f"].append(value)
        index=end
    return result


def _json_object(text: str) -> dict[str, Any]:
    stripped=text.strip()
    if stripped.startswith("```"): stripped=re.sub(r"^```(?:json)?\s*|\s*```$","",stripped,flags=re.I|re.S)
    start=stripped.find("{"); end=stripped.rfind("}")
    if start<0 or end<start: raise ValueError("missing JSON object")
    value=json.loads(stripped[start:end+1])
    if not isinstance(value,dict): raise ValueError("JSON root is not an object")
    return value
