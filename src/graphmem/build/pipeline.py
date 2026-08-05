from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..config import GraphMemV5Config, config_hash
from ..domain import (
    EvidenceGroup,
    EvidenceMember,
    GraphArtifactManifest,
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
    SourceTurn,
    canonical_json,
    logical_graph_checksum,
    stable_id,
)
from ..storage import SQLiteGraphStore
from .refine import Qwen30BRefiner, RefineCandidate
from .semantic import QwenSemanticDistiller, ScenePacket


WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
CAPITAL_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NON_ENTITY_NAMES = frozenset({
    "the", "this", "that", "these", "those", "there", "then", "when", "what",
    "where", "which", "while", "with", "without", "after", "before", "however",
    "also", "thanks", "thank", "hey", "hello", "yes", "yeah", "wow", "okay",
    "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
})
TIME_RE = re.compile(
    r"\b(?:\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?|\d{4}|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|yesterday|today|tomorrow|last week|next week)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(?:not|never|no|without|didn't|don't|doesn't|can't|cannot)\b", re.I)
STATE_RE = re.compile(r"\b(?:is|was|became|feels?|likes?|loves?|hates?|works?|lives?|owns?)\b", re.I)
VERB_RE = re.compile(r"\b([A-Za-z]+(?:ed|ing|s)|went|go|got|made|took|had|has|is|was|are|were)\b", re.I)


PROFILE_LEVEL = {f"b{index}": index for index in range(7)}


@dataclass(frozen=True, slots=True)
class _SceneSlice:
    scene_id: str
    session_id: str
    turns: tuple[SourceTurn, ...]
    summary: str


class GraphBuildPipeline:
    def __init__(self, store: SQLiteGraphStore, *, dataset_hash: str,
                 refiner: Qwen30BRefiner | None = None,
                 distiller: QwenSemanticDistiller | None = None) -> None:
        self.store = store
        self.dataset_hash = dataset_hash
        self.refiner = refiner
        self.distiller = distiller

    def build(self, memory_id: str, profile: GraphMemV5Config) -> GraphArtifactManifest:
        started = time.perf_counter()
        usage_before = self._usage(memory_id)
        turns = tuple(self.store.turns(memory_id))
        sessions = tuple(self.store.sessions(memory_id))
        if not turns or not sessions:
            raise ValueError(f"memory {memory_id!r} has no imported sessions/turns")
        level = PROFILE_LEVEL.get(profile.profile.casefold())
        if level is None:
            raise ValueError("profile must be one of B0..B6")
        if level == 6:
            raise ValueError("B6 is a legacy-adapter reference, not a V5 build profile")

        groups = [self._turn_group(turn) for turn in turns]
        group_by_turn = {group.members[0].turn_id: group for group in groups}
        scenes = self._segment(turns, profile)
        packets = (self.distiller.extract(memory_id, scenes)
                   if profile.scenes.llm_semantic_extraction and self.distiller else ())
        packet_by_scene = {packet.scene_id: packet for packet in packets}
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        scene_nodes: dict[str, GraphNode] = {}
        event_nodes: dict[str, list[GraphNode]] = defaultdict(list)
        entity_nodes: dict[str, GraphNode] = {}

        session_cards: dict[str, GraphNode] = {}
        session_compressed = {}
        if profile.scenes.llm_hierarchy_compression and self.distiller:
            requests = []
            request_sessions = []
            for session in sessions:
                children = [self._packet_record(packet_by_scene[scene.scene_id])
                            for scene in scenes if scene.session_id == session.session_id
                            and scene.scene_id in packet_by_scene]
                if children:
                    requests.append((stable_id("node", memory_id, "routing-card", 1, session.session_id),
                                     children, 128)); request_sessions.append(session.session_id)
            if requests:
                session_compressed = dict(zip(request_sessions,
                    self.distiller.compress_many(memory_id, 1, requests)))
        for session in sessions:
            session_turns = [turn for turn in turns if turn.session_id == session.session_id]
            evidence_ids = tuple(group_by_turn[turn.turn_id].evidence_group_id for turn in session_turns)
            card_summary = self._bounded_summary(" ".join(turn.raw_text for turn in session_turns), 80)
            card_attrs: dict[str, Any] = {"session_id": session.session_id, "roles": ("route",)}
            if session.session_id in session_compressed:
                compressed = session_compressed[session.session_id]
                card_summary = str(compressed["summary"]); card_attrs.update(compressed)
            card = GraphNode(
                node_id=stable_id("node", memory_id, "routing-card", 1, session.session_id),
                memory_id=memory_id, node_type=NodeType.ROUTING_CARD, level=1,
                summary=card_summary,
                evidence_group_id=evidence_ids[0], evidence_group_ids=evidence_ids[1:],
                attributes=card_attrs,
            )
            session_cards[session.session_id] = card
            nodes.append(card)

        memory_evidence = tuple(group.evidence_group_id for group in groups)
        memory_card_id = stable_id("node", memory_id, "routing-card", 3)
        l2_cards: list[GraphNode] = []
        if level >= 5:
            ordered_cards = list(session_cards.values())
            for group_index in range(0, len(ordered_cards), profile.coarsen.fanout):
                children = ordered_cards[group_index:group_index + profile.coarsen.fanout]
                evidence_ids = tuple(dict.fromkeys(
                    group_id for child in children for group_id in child.all_evidence_group_ids
                ))
                l2_summary = self._bounded_summary(" ".join(child.summary for child in children), 120)
                l2_attrs: dict[str, Any] = {
                    "child_session_ids": tuple(child.attributes["session_id"] for child in children),
                    "roles": ("route", "cross_session"),
                }
                l2_id = stable_id("node", memory_id, "routing-card", 2,
                                  tuple(child.node_id for child in children))
                if profile.scenes.llm_hierarchy_compression and self.distiller:
                    compressed = self.distiller.compress(memory_id, 2, l2_id,
                        [self._node_record(child) for child in children], 160)
                    l2_summary = str(compressed["summary"]); l2_attrs.update(compressed)
                l2_cards.append(GraphNode(
                    l2_id,
                    memory_id, NodeType.ROUTING_CARD, 2,
                    l2_summary,
                    evidence_ids[0], evidence_ids[1:],
                    attributes=l2_attrs,
                ))
            nodes.extend(l2_cards)
        memory_children = l2_cards or list(session_cards.values())
        memory_summary = self._bounded_summary(" ".join(card.summary for card in memory_children), 160)
        memory_attrs: dict[str, Any] = {"roles": ("route", "memory")}
        if profile.scenes.llm_hierarchy_compression and self.distiller:
            compressed = self.distiller.compress(memory_id, 3, memory_card_id,
                                                  [self._node_record(x) for x in memory_children], 192)
            memory_summary = str(compressed["summary"]); memory_attrs.update(compressed)
        memory_card = GraphNode(memory_card_id, memory_id, NodeType.ROUTING_CARD, 3,
            memory_summary, memory_evidence[0], memory_evidence[1:], attributes=memory_attrs)
        nodes.append(memory_card)

        if level >= 1:
            for scene in scenes:
                evidence_ids = tuple(group_by_turn[turn.turn_id].evidence_group_id for turn in scene.turns)
                scene_node = GraphNode(
                    scene.scene_id, memory_id, NodeType.SCENE, 0,
                    packet_by_scene.get(scene.scene_id, ScenePacket(scene.scene_id, scene.summary, (), ())).summary,
                    evidence_ids[0], evidence_ids[1:],
                    attributes={"session_id": scene.session_id, "turn_ids": tuple(x.turn_id for x in scene.turns),
                                "roles": ("scene",)},
                )
                scene_nodes[scene.scene_id] = scene_node
                nodes.append(scene_node)
                for event_index, event in enumerate(self._events(scene, profile)):
                    event_nodes[scene.scene_id].append(event)
                    nodes.append(event)

        if level >= 2:
            entity_nodes.update(self._entities(memory_id, scenes, event_nodes, group_by_turn))
            nodes.extend(entity_nodes.values())
            nodes.extend(self._time_and_state_nodes(memory_id, event_nodes))
            if packets:
                semantic_nodes, semantic_edges = self._semantic_graph(
                    memory_id, packets, turns, group_by_turn, profile,
                    (*session_cards.values(), *l2_cards, memory_card)
                )
                nodes.extend(semantic_nodes); edges.extend(semantic_edges)

        if level >= 3:
            edges.extend(self._hierarchy_edges(
                memory_id, memory_card, l2_cards, session_cards, scene_nodes, event_nodes
            ))
            edges.extend(self._typed_edges(memory_id, event_nodes, entity_nodes))

        refine_tokens = {"cached_input_tokens": 0, "uncached_input_tokens": 0,
                         "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        truncated: tuple[str, ...] = ()
        if level >= 4 and profile.edges.refine_mode != "none" and self.refiner:
            candidates = self._ambiguous_candidates(memory_id, scenes, event_nodes, entity_nodes)
            decisions, truncated = self.refiner.refine(memory_id, candidates)
            by_candidate = {candidate.candidate_id: candidate for candidate in candidates}
            for decision in decisions:
                if decision.decision == "NONE" or decision.candidate_id not in by_candidate:
                    continue
                candidate = by_candidate[decision.candidate_id]
                relation = RelationType(decision.decision)
                evidence = next(node for node in nodes if node.node_id == candidate.left_id).evidence_group_id
                edges.append(GraphEdge(
                    stable_id("edge", memory_id, candidate.left_id, relation, candidate.right_id),
                    memory_id, candidate.left_id, relation, candidate.right_id, evidence,
                    True, decision.confidence, decision.source,
                ))
            usage_after = self._usage(memory_id)
            refine_tokens = {key: usage_after[key] - usage_before[key] for key in usage_after}

        if level >= 5:
            if profile.coarsen.cross_session_merge and not packets:
                edges.extend(self._portal_edges(memory_id, session_cards, scene_nodes))

        nodes = self._dedup_nodes(nodes)
        edges = self._bounded_edges(self._dedup_edges(edges), profile)
        version = self.store.replace_graph(memory_id, nodes, edges, groups)
        checksum = logical_graph_checksum(nodes, edges)
        artifact_id = stable_id("graph-artifact", memory_id, self.dataset_hash,
                                config_hash(profile), checksum, version)
        usage_after = self._usage(memory_id)
        token_usage = {key: usage_after[key] - usage_before[key] for key in usage_after}
        token_usage.update({
            "truncated_candidates": len(truncated),
            "wall_time_ms": round((time.perf_counter() - started) * 1000),
        })
        return GraphArtifactManifest(
            graph_artifact_id=artifact_id, memory_id=memory_id,
            dataset_hash=self.dataset_hash, config_hash=config_hash(profile),
            graph_checksum=checksum, graph_version=version, node_count=len(nodes),
            edge_count=len(edges), evidence_group_count=len(groups),
            model_ids={"llm": profile.models.llm_model,
                       "embedding": profile.models.embedding_model},
            prompt_hashes={"selective_refine": self.refiner.prompt_hash if self.refiner else "disabled",
                           "semantic_distill": self.distiller.prompt_hash if self.distiller else "disabled"},
            build_token_usage=token_usage,
        )

    @staticmethod
    def _turn_group(turn: SourceTurn) -> EvidenceGroup:
        member = EvidenceMember(turn.turn_id, 0, max(1, len(turn.raw_text)), "source")
        group_id = stable_id("evidence", turn.memory_id, turn.turn_id, member.span_start, member.span_end)
        return EvidenceGroup(
            group_id, turn.memory_id, (member,),
            hashlib.sha256(turn.raw_text.encode("utf-8")).hexdigest(),
            turn.timestamp, turn.timestamp,
        )

    def _segment(self, turns: Sequence[SourceTurn], profile: GraphMemV5Config) -> list[_SceneSlice]:
        by_session: dict[str, list[SourceTurn]] = defaultdict(list)
        for turn in turns:
            by_session[turn.session_id].append(turn)
        result: list[_SceneSlice] = []
        for session_id in sorted(by_session):
            ordered = sorted(by_session[session_id], key=lambda item: item.turn_index)
            current: list[SourceTurn] = []
            chunks: list[list[SourceTurn]] = []
            for turn in ordered:
                should_cut = False
                if len(current) >= profile.scenes.max_turns:
                    should_cut = True
                elif len(current) >= profile.scenes.min_turns:
                    left = self._terms(" ".join(item.raw_text for item in current[-2:]))
                    right = self._terms(turn.raw_text)
                    similarity = len(left & right) / max(1, len(left | right))
                    entity_overlap = bool(self._names(" ".join(item.raw_text for item in current)) & self._names(turn.raw_text))
                    qa_pair = current[-1].role != turn.role and current[-1].listener == turn.speaker
                    should_cut = similarity < profile.scenes.topic_similarity_threshold and not entity_overlap and not qa_pair
                if should_cut:
                    chunks.append(current)
                    current = []
                current.append(turn)
            if current:
                chunks.append(current)
            if len(chunks) >= 2 and len(chunks[-1]) < profile.scenes.min_turns \
                    and len(chunks[-2]) + len(chunks[-1]) <= profile.scenes.max_turns:
                chunks[-2].extend(chunks.pop())
            result.extend(self._scene_slice(session_id, chunk) for chunk in chunks)
        return result

    def _scene_slice(self, session_id: str, turns: Sequence[SourceTurn]) -> _SceneSlice:
        scene_id = stable_id("node", turns[0].memory_id, "scene", session_id,
                             turns[0].turn_index, turns[-1].turn_index)
        return _SceneSlice(scene_id, session_id, tuple(turns),
                           self._bounded_summary(" ".join(turn.raw_text for turn in turns), 96))

    def _events(self, scene: _SceneSlice, profile: GraphMemV5Config) -> list[GraphNode]:
        candidates: list[tuple[SourceTurn, str, str, tuple[str, ...], str]] = []
        last_explicit: set[str] = set()
        for turn in scene.turns:
            sentences = re.split(r"(?<=[.!?])\s+", turn.raw_text)
            for sentence in sentences:
                verb = VERB_RE.search(sentence)
                if verb and len(sentence.split()) >= 3:
                    explicit = self._names(sentence)
                    resolved = set(explicit)
                    lowered_terms = {token.casefold() for token in WORD_RE.findall(sentence)}
                    sources = []
                    if lowered_terms & {"i", "me", "my", "mine", "we", "our"}:
                        resolved.add(turn.speaker.casefold())
                        sources.append("speaker")
                    if lowered_terms & {"you", "your", "yours"} and turn.listener:
                        resolved.add(turn.listener.casefold())
                        sources.append("listener")
                    if lowered_terms & {"he", "she", "they", "them", "their", "it"} \
                            and len(last_explicit) == 1:
                        resolved.update(last_explicit)
                        sources.append("unique_recent_entity")
                    if explicit:
                        last_explicit = explicit
                    candidates.append((turn, verb.group(1).casefold(), sentence,
                                       tuple(sorted(resolved - {"", "unknown"})),
                                       "+".join(sources) or "explicit"))
        if not candidates:
            turn = max(scene.turns, key=lambda item: len(item.raw_text))
            candidates = [(turn, "mentions", turn.raw_text,
                           tuple(sorted(self._names(turn.raw_text))), "explicit")]
        result: list[GraphNode] = []
        for index, (turn, predicate, sentence, entity_names, coreference_source) in enumerate(
            candidates[:profile.scenes.max_events_per_scene]
        ):
            group_id = stable_id("evidence", turn.memory_id, turn.turn_id, 0, max(1, len(turn.raw_text)))
            times = tuple(match.group(0).casefold() for match in TIME_RE.finditer(sentence))
            result.append(GraphNode(
                stable_id("node", turn.memory_id, "event", scene.scene_id, index, predicate, sentence),
                turn.memory_id, NodeType.EVENT_SKELETON, 0,
                self._bounded_summary(sentence, 48), group_id,
                event_time=times[0] if times else None,
                state=(STATE_RE.search(sentence).group(0).casefold() if STATE_RE.search(sentence) else None),
                attributes={
                    "scene_id": scene.scene_id, "session_id": scene.session_id,
                    "predicate": predicate, "negative_scope": bool(NEGATION_RE.search(sentence)),
                    "entity_names": entity_names,
                    "coreference_source": coreference_source,
                    "roles": ("event", "negative_scope") if NEGATION_RE.search(sentence) else ("event",),
                },
            ))
        return result

    def _entities(self, memory_id, scenes, event_nodes, group_by_turn):
        mention_groups: dict[str, list[str]] = defaultdict(list)
        mention_turns: dict[str, list[str]] = defaultdict(list)
        for scene in scenes:
            for turn in scene.turns:
                explicit = {turn.speaker.casefold(), turn.listener.casefold()} - {"", "unknown"}
                for name in explicit:
                    mention_groups[name].append(group_by_turn[turn.turn_id].evidence_group_id)
                    mention_turns[name].append(turn.turn_id)
            for event in event_nodes.get(scene.scene_id, ()):
                for name in event.attributes.get("entity_names", ()):
                    mention_groups[name].append(event.evidence_group_id)
        result = {}
        for name, group_rows in sorted(mention_groups.items()):
            groups = tuple(dict.fromkeys(group_rows))
            entity_id = stable_id("node", memory_id, "entity", name)
            result[name] = GraphNode(
                entity_id, memory_id, NodeType.CANONICAL_ENTITY, 0, name,
                groups[0], groups[1:], entity_id=entity_id,
                attributes={"aliases": (name,), "mention_turn_ids": tuple(mention_turns.get(name, ())),
                            "roles": ("entity",)},
            )
        return result

    def _time_and_state_nodes(self, memory_id, event_nodes):
        result = []
        for events in event_nodes.values():
            for event in events:
                if event.event_time:
                    result.append(GraphNode(
                        stable_id("node", memory_id, "time", event.event_time), memory_id,
                        NodeType.TIME_ANCHOR, 0, event.event_time, event.evidence_group_id,
                        event_time=event.event_time, attributes={"roles": ("time", "temporal_left", "temporal_right")},
                    ))
                if event.state:
                    result.append(GraphNode(
                        stable_id("node", memory_id, "state", event.state, event.node_id), memory_id,
                        NodeType.STATE_HEAD, 0, event.state, event.evidence_group_id,
                        state=event.state, attributes={"roles": ("prior_state", "current_state")},
                    ))
        return self._dedup_nodes(result)

    def _semantic_graph(self, memory_id, packets, turns, group_by_turn, profile, hierarchy_cards=()):
        turn_map = {turn.turn_id: turn for turn in turns}
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        facts: list[tuple[GraphNode, str, str, str]] = []
        for packet in packets:
            for fact in packet.facts:
                refs = [ref for ref in fact.evidence if ref[0] in group_by_turn]
                if not refs:
                    continue
                groups = tuple(dict.fromkeys(group_by_turn[turn_id].evidence_group_id
                                             for turn_id, _, _ in refs))
                owner_key = self._normal(fact.owner); predicate_key = self._normal(fact.predicate)
                value_key = self._normal(fact.value); scope_key = self._normal(fact.scope)
                owner_id = stable_id("node", memory_id, "semantic-owner", owner_key)
                value_id = stable_id("node", memory_id, "canonical-value", fact.value_type, value_key)
                fact_id = stable_id("node", memory_id, "canonical-fact", owner_key, predicate_key,
                                    value_key, scope_key, fact.polarity)
                nodes.setdefault(owner_id, GraphNode(owner_id, memory_id, NodeType.CANONICAL_ENTITY, 0,
                    fact.owner, groups[0], groups[1:], entity_id=owner_id,
                    attributes={"aliases": (fact.owner,), "roles": ("entity", "owner")}))
                nodes.setdefault(value_id, GraphNode(value_id, memory_id, NodeType.CANONICAL_VALUE, 0,
                    fact.value, groups[0], groups[1:], attributes={"value_type": fact.value_type,
                    "normalized": value_key, "roles": ("value",)}))
                fact_node = GraphNode(fact_id, memory_id, NodeType.CANONICAL_FACT, 0,
                    f"{fact.owner} {fact.predicate} {fact.value}", groups[0], groups[1:],
                    event_time=fact.time, confidence=fact.confidence,
                    attributes={"owner_id": owner_id, "predicate": predicate_key, "value_id": value_id,
                    "scope": scope_key, "polarity": fact.polarity, "scene_id": packet.scene_id,
                    "roles": ("fact", "predicate", "object")})
                nodes[fact_id] = fact_node; facts.append((fact_node, owner_id, value_id,
                    turn_map[refs[0][0]].session_id))
                edges.append(self._edge(memory_id, nodes[owner_id], RelationType.HAS_FACT, fact_node, "semantic"))
                edges.append(self._edge(memory_id, fact_node, RelationType.FACT_VALUE, nodes[value_id], "semantic"))
        by_value: dict[str, list[tuple[GraphNode, str]]] = defaultdict(list)
        by_activity: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
        for fact, owner_id, value_id, session_id in facts:
            by_value[value_id].append((fact, session_id))
            by_activity[(owner_id, str(fact.attributes["predicate"]))].append(fact)
        if profile.coarsen.cross_session_merge:
            for value_id, rows in by_value.items():
                sessions = {session for _, session in rows}
                if len(sessions) < 2:
                    continue
                evidence = tuple(dict.fromkeys(group for fact, _ in rows for group in fact.all_evidence_group_ids))
                region_id = stable_id("node", memory_id, "virtual-region", value_id, tuple(sorted(sessions)))
                region = GraphNode(region_id, memory_id, NodeType.VIRTUAL_REGION, 2,
                    nodes[value_id].summary, evidence[0], evidence[1:],
                    attributes={"value_id": value_id, "session_ids": tuple(sorted(sessions)),
                                "roles": ("route", "cross_session", "value")})
                nodes[region_id] = region
                for fact, _ in rows:
                    edges.append(self._edge(memory_id, region, RelationType.SHARED_VALUE, fact, "semantic"))
        for rows in by_activity.values():
            for left, right in zip(rows, rows[1:]):
                edges.append(self._edge(memory_id, left, RelationType.SAME_ACTIVITY, right, "semantic"))
        values_by_label = {self._normal(node.summary): node for node in nodes.values()
                           if node.node_type == NodeType.CANONICAL_VALUE}
        seen_aliases = set()
        for card in hierarchy_cards:
            for group in card.attributes.get("aliases", ()):
                candidates = [values_by_label.get(self._normal(value)) for value in group]
                candidates = [node for node in candidates if node]
                for left, right in zip(candidates, candidates[1:]):
                    key = tuple(sorted((left.node_id, right.node_id)))
                    if key in seen_aliases: continue
                    seen_aliases.add(key)
                    edges.append(self._edge(memory_id, left, RelationType.COREFERENCE, right, "semantic_alias"))
        return list(nodes.values()), edges

    @staticmethod
    def _normal(value):
        return " ".join(re.findall(r"[\w'-]+", value.casefold()))

    @staticmethod
    def _packet_record(packet):
        return {"child_id": packet.scene_id, "summary": " ".join(packet.summary.split()[:48]),
                "owners": tuple(dict.fromkeys(fact.owner for fact in packet.facts)),
                "predicates": tuple(dict.fromkeys(fact.predicate for fact in packet.facts)),
                "values": tuple(dict.fromkeys(fact.value for fact in packet.facts)),
                "scopes": tuple(dict.fromkeys(fact.scope for fact in packet.facts)),
                "times": tuple(dict.fromkeys(fact.time for fact in packet.facts if fact.time))}

    @staticmethod
    def _node_record(node):
        return {"child_id": node.node_id, "summary": node.summary,
                "owners": node.attributes.get("owners", ()),
                "predicates": node.attributes.get("predicates", ()),
                "values": node.attributes.get("values", ()), "scopes": node.attributes.get("scopes", ()),
                "times": node.attributes.get("times", ())}

    def _hierarchy_edges(self, memory_id, memory_card, l2_cards, session_cards, scene_nodes, event_nodes):
        edges = []
        if l2_cards:
            for l2_card in l2_cards:
                edges.append(self._edge(memory_id, memory_card, RelationType.REFINES_TO,
                                        l2_card, "deterministic"))
                for session_id in l2_card.attributes["child_session_ids"]:
                    edges.append(self._edge(memory_id, l2_card, RelationType.REFINES_TO,
                                            session_cards[str(session_id)], "deterministic"))
        else:
            for session_id, card in session_cards.items():
                edges.append(self._edge(memory_id, memory_card, RelationType.REFINES_TO,
                                        card, "deterministic"))
        for scene_id, scene in scene_nodes.items():
            card = session_cards[str(scene.attributes["session_id"])]
            edges.append(self._edge(memory_id, card, RelationType.REFINES_TO, scene, "deterministic"))
            for event in event_nodes.get(scene_id, ()):
                edges.append(self._edge(memory_id, scene, RelationType.SCENE_CONTAINS, event, "deterministic"))
        return edges

    def _typed_edges(self, memory_id, event_nodes, entity_nodes):
        edges = []
        all_events = [event for events in event_nodes.values() for event in events]
        for event in all_events:
            for name in event.attributes.get("entity_names", ()):
                if name in entity_nodes:
                    edges.append(self._edge(memory_id, event, RelationType.PARTICIPATES_IN,
                                            entity_nodes[name], "deterministic"))
        ordered = sorted((event for event in all_events if event.event_time), key=lambda item: (item.event_time, item.node_id))
        for left, right in zip(ordered, ordered[1:]):
            edges.append(self._edge(memory_id, left, RelationType.TEMPORAL_BEFORE, right, "deterministic"))
        return edges

    def _portal_edges(self, memory_id, session_cards, scene_nodes):
        edges = []
        by_entity: dict[str, list[GraphNode]] = defaultdict(list)
        for scene in scene_nodes.values():
            for name in self._names(scene.summary):
                by_entity[name].append(scene)
        for scenes in by_entity.values():
            ordered = sorted(scenes, key=lambda item: item.node_id)
            for left, right in zip(ordered, ordered[1:]):
                if left.attributes.get("session_id") != right.attributes.get("session_id"):
                    edges.append(self._edge(memory_id, left, RelationType.PORTAL, right, "deterministic"))
        return edges

    def _ambiguous_candidates(self, memory_id, scenes, event_nodes, entity_nodes):
        candidates = []
        all_events = [event for events in event_nodes.values() for event in events]
        by_name: dict[str, list[GraphNode]] = defaultdict(list)
        for event in all_events:
            for name in event.attributes.get("entity_names", ()):
                if name.startswith("participant") or name in {"user", "assistant"}:
                    continue
                by_name[name].append(event)
        degree: dict[str, int] = defaultdict(int)
        seen: set[tuple[str, str]] = set()
        for name in sorted(by_name):
            ordered = sorted(by_name[name], key=lambda item: (
                str(item.attributes.get("session_id", "")), item.node_id
            ))
            # Adjacent mentions of the same canonical entity provide a bounded
            # cross-session bridge candidate set instead of an O(n^2) clique.
            for left, right in zip(ordered, ordered[1:]):
                if left.attributes.get("session_id") == right.attributes.get("session_id"):
                    continue
                pair = tuple(sorted((left.node_id, right.node_id)))
                if pair in seen or degree[left.node_id] >= 24 or degree[right.node_id] >= 24:
                    continue
                seen.add(pair)
                degree[left.node_id] += 1
                degree[right.node_id] += 1
                candidates.append(RefineCandidate(
                    stable_id("candidate", memory_id, *pair), "edge",
                    left.node_id, right.node_id, left.summary, right.summary,
                    (str(RelationType.SAME_EVENT), str(RelationType.PORTAL), "NONE"),
                    0.05, True, True, True,
                ))
        return candidates

    @staticmethod
    def _edge(memory_id, left, relation, right, source):
        groups = tuple(dict.fromkeys((*left.all_evidence_group_ids, *right.all_evidence_group_ids)))
        return GraphEdge(
            stable_id("edge", memory_id, left.node_id, relation, right.node_id), memory_id,
            left.node_id, relation, right.node_id, groups[0], True, 1.0, source,
            evidence_group_ids=groups[1:],
        )

    @staticmethod
    def _dedup_nodes(nodes):
        return list({node.node_id: node for node in nodes}.values())

    @staticmethod
    def _dedup_edges(edges):
        return list({edge.edge_id: edge for edge in edges}.values())

    @staticmethod
    def _bounded_edges(edges, profile):
        by_relation: dict[tuple[str, RelationType], list[GraphEdge]] = defaultdict(list)
        for edge in sorted(edges, key=lambda item: (-item.confidence, item.edge_id)):
            key = (edge.src_id, edge.relation)
            cap = int(profile.edges.relation_degree_caps.get(
                str(edge.relation), profile.edges.max_degree_per_relation
            ))
            if len(by_relation[key]) < cap:
                by_relation[key].append(edge)
        return [edge for rows in by_relation.values() for edge in rows]

    def _usage(self, memory_id):
        rows = self.store._connection.execute(
            "SELECT usage_json,cached FROM llm_calls WHERE memory_id=?", (memory_id,)
        )
        totals = {"cached_input_tokens": 0, "uncached_input_tokens": 0,
                  "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        import json
        for row in rows:
            usage = json.loads(row["usage_json"])
            for key in totals:
                totals[key] += int(usage.get(key, 0))
        return totals

    @staticmethod
    def _terms(text):
        return {token.casefold() for token in WORD_RE.findall(text) if len(token) > 2}

    @staticmethod
    def _names(text):
        return {name.casefold() for name in CAPITAL_RE.findall(text)
                if name.casefold() not in NON_ENTITY_NAMES}

    @staticmethod
    def _bounded_summary(text, words):
        return " ".join(text.split()[:words])
