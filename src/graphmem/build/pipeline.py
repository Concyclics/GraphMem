from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    dataclass_dict,
    logical_graph_checksum,
    stable_id,
)
from ..storage import SQLiteGraphStore
from .coarsen import (
    ATOMIC_RELATION_NODE_TYPES,
    GatedRelationPlan,
    RelationSignal,
    RecursiveHierarchy,
    admit_llm_refined_relation,
    build_parent_gated_relations,
    build_rare_lexical_node_terms,
    build_recursive_hierarchy,
    promotable_object_value,
)
from .refine import Qwen30BRefiner, RefineCandidate
from .semantic import QwenSemanticDistiller, ScenePacket
from .temporal import extract_time_expression, normalize_time, observed_interval
from .canonicalize import PredicateCanonicalizer


WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
CAPITAL_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
ENTITY_MENTION_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][a-z]+)(?:[ '\u2019-]+(?:[A-Z]{2,}|[A-Z][a-z]+)){0,3}\b")
NON_ENTITY_NAMES = frozenset({
    "the", "this", "that", "these", "those", "there", "then", "when", "what",
    "where", "which", "while", "with", "without", "after", "before", "however",
    "also", "thanks", "thank", "hey", "hello", "yes", "yeah", "wow", "okay",
    "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
})
EVIDENCE_REF_STOPWORDS = NON_ENTITY_NAMES | frozenset({
    "and", "but", "for", "from", "have", "has", "had", "into", "just", "like",
    "more", "much", "really", "some", "than", "they", "them", "their", "were",
    "will", "would", "you", "your", "about", "could", "should", "because", "been",
    "are", "was", "is", "am", "can", "did", "does", "do", "of", "to", "in", "on",
    "at", "it", "my", "me", "we", "our", "a", "an", "i",
    "how", "why", "who", "no", "not", "use",
    "never", "appreciate", "ttyl",
    "all", "last", "take", "one", "individual", "individuals", "option",
    "metrics", "countries", "chapter",
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

DIRECTIONAL_REFINED_RELATIONS = frozenset({
    RelationType.TEMPORAL_CONTINUATION,
    RelationType.CAUSAL,
    RelationType.CONTRADICTION_UPDATE,
})



@dataclass(frozen=True, slots=True)
class _SceneSlice:
    scene_id: str
    session_id: str
    turns: tuple[SourceTurn, ...]
    summary: str


class GraphBuildPipeline:
    def __init__(self, store: SQLiteGraphStore, *, dataset_hash: str,
                 refiner: Qwen30BRefiner | None = None,
                 distiller: QwenSemanticDistiller | None = None,
                 predicate_canonicalizer: PredicateCanonicalizer | None = None,
                 coarsen_vector_provider: Callable[[
                     str, Sequence[GraphNode]], Mapping[str, Sequence[float]]] | None = None,
                 relation_vector_provider: Callable[[
                     str, Sequence[GraphNode]], Mapping[str, Sequence[float]]] | None = None) -> None:
        self.store = store
        self.dataset_hash = dataset_hash
        self.refiner = refiner
        self.distiller = distiller
        self.predicate_canonicalizer = predicate_canonicalizer
        self.coarsen_vector_provider = coarsen_vector_provider
        self.relation_vector_provider = relation_vector_provider

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
        turns_by_session: dict[str, list[SourceTurn]] = defaultdict(list)
        scenes_by_session: dict[str, list[_SceneSlice]] = defaultdict(list)
        for turn in turns:
            turns_by_session[turn.session_id].append(turn)
        for scene in scenes:
            scenes_by_session[scene.session_id].append(scene)
        packets = (self.distiller.extract(memory_id, scenes)
                   if profile.scenes.llm_semantic_extraction and self.distiller else ())
        packet_by_scene = {packet.scene_id: packet for packet in packets}
        # Computed once over every scene in the memory, before any card is
        # compiled: this is the one point in the build that sees the whole
        # vocabulary at once, which is precisely what a per-scene extraction
        # call cannot do.
        merge_aliases = self._merge_aliases(scenes, packet_by_scene, profile)
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        scene_nodes: dict[str, GraphNode] = {}
        event_nodes: dict[str, list[GraphNode]] = defaultdict(list)
        entity_nodes: dict[str, GraphNode] = {}

        lean_graph = profile.edges.graph_variant == "g5"
        session_cards: dict[str, GraphNode] = {}
        session_compressed = {}
        if profile.scenes.llm_hierarchy_compression and self.distiller:
            requests = []
            request_sessions = []
            for session in sessions:
                children = [self._packet_record(packet_by_scene[scene.scene_id])
                            for scene in scenes_by_session.get(session.session_id, ())
                            if scene.scene_id in packet_by_scene]
                if children:
                    requests.append((stable_id("node", memory_id, "routing-card", 1, session.session_id),
                                     children, 128)); request_sessions.append(session.session_id)
            if requests:
                session_compressed = dict(zip(request_sessions,
                    self.distiller.compress_many(memory_id, 1, requests)))
        for session in sessions:
            session_turns = turns_by_session.get(session.session_id, ())
            evidence_ids = tuple(group_by_turn[turn.turn_id].evidence_group_id for turn in session_turns)
            card_summary = self._bounded_summary(" ".join(turn.raw_text for turn in session_turns), 80)
            card_attrs: dict[str, Any] = {"session_id": session.session_id, "roles": ("route",)}
            if lean_graph:
                card_attrs["provenance_scope"] = "route"
            compact_route = (profile.coarsen.recursive_hierarchy
                             and profile.coarsen.compact_routing_provenance)
            if compact_route:
                card_attrs.update({
                    "provenance_scope": "route",
                    "provenance_compact": True,
                    "provenance_ref_session": session.session_id,
                })
            if lean_graph:
                children = [self._packet_record(packet_by_scene[scene.scene_id])
                            for scene in scenes_by_session.get(session.session_id, ())
                            if scene.scene_id in packet_by_scene]
                compiled = self._compile_routing(children, 80, merge_aliases)
                card_summary = compiled["summary"]; card_attrs.update(compiled)
            if session.session_id in session_compressed:
                compressed = session_compressed[session.session_id]
                card_summary = str(compressed["summary"]); card_attrs.update(compressed)
            card = GraphNode(
                node_id=stable_id("node", memory_id, "routing-card", 1, session.session_id),
                memory_id=memory_id, node_type=NodeType.ROUTING_CARD, level=1,
                summary=card_summary,
                evidence_group_id=evidence_ids[0],
                evidence_group_ids=() if compact_route else evidence_ids[1:],
                attributes=card_attrs,
            )
            session_cards[session.session_id] = card
            nodes.append(card)

        memory_evidence = tuple(group.evidence_group_id for group in groups)
        recursive_hierarchy: RecursiveHierarchy | None = None
        l2_cards: list[GraphNode] = []
        if profile.coarsen.recursive_hierarchy:
            coarsen_vectors = (
                self.coarsen_vector_provider(memory_id, tuple(session_cards.values()))
                if (profile.coarsen.assignment_method == "hnsw"
                    and self.coarsen_vector_provider is not None)
                else None)
            recursive_hierarchy = build_recursive_hierarchy(
                memory_id,
                tuple(session_cards.values()),
                fanout=profile.coarsen.fanout,
                max_levels=profile.coarsen.max_levels,
                summary_words=max(32, profile.coarsen.summary_tokens // 2),
                max_candidates=profile.edges.max_candidates_per_node,
                assignment_method=profile.coarsen.assignment_method,
                vectors=coarsen_vectors,
                hnsw_dimension=profile.coarsen.hnsw_dimension,
                hnsw_m=profile.coarsen.hnsw_m,
                hnsw_ef_construction=profile.coarsen.hnsw_ef_construction,
            )
            nodes.extend(recursive_hierarchy.parent_cards)
            memory_card = recursive_hierarchy.root
            l2_cards = [row for row in recursive_hierarchy.parent_cards if row.level == 2]
            hierarchy_cards_for_semantics = (
                *session_cards.values(), *recursive_hierarchy.parent_cards)
        else:
            memory_card_id = stable_id("node", memory_id, "routing-card", 3)
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
                    if lean_graph:
                        l2_attrs["provenance_scope"] = "route"
                    l2_id = stable_id("node", memory_id, "routing-card", 2,
                                      tuple(child.node_id for child in children))
                    if profile.scenes.llm_hierarchy_compression and self.distiller:
                        compressed = self.distiller.compress(memory_id, 2, l2_id,
                            [self._node_record(child) for child in children], 160)
                        l2_summary = str(compressed["summary"]); l2_attrs.update(compressed)
                    elif lean_graph:
                        compiled = self._compile_routing(
                            [self._node_record(child) for child in children], 120, merge_aliases)
                        l2_summary = compiled["summary"]; l2_attrs.update(compiled)
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
            if lean_graph:
                memory_attrs["provenance_scope"] = "route"
            if profile.scenes.llm_hierarchy_compression and self.distiller:
                compressed = self.distiller.compress(memory_id, 3, memory_card_id,
                                                      [self._node_record(x) for x in memory_children], 192)
                memory_summary = str(compressed["summary"]); memory_attrs.update(compressed)
            elif lean_graph:
                compiled = self._compile_routing(
                    [self._node_record(x) for x in memory_children], 160, merge_aliases)
                memory_summary = compiled["summary"]; memory_attrs.update(compiled)
            memory_card = GraphNode(memory_card_id, memory_id, NodeType.ROUTING_CARD, 3,
                memory_summary, memory_evidence[0], memory_evidence[1:], attributes=memory_attrs)
            nodes.append(memory_card)
            hierarchy_cards_for_semantics = (
                *session_cards.values(), *l2_cards, memory_card)

        if level >= 1:
            for scene in scenes:
                evidence_ids = tuple(group_by_turn[turn.turn_id].evidence_group_id for turn in scene.turns)
                packet = packet_by_scene.get(
                    scene.scene_id, ScenePacket(scene.scene_id, scene.summary, (), ()))
                scene_node = GraphNode(
                    scene.scene_id, memory_id, NodeType.SCENE, 0,
                    packet.summary,
                    evidence_ids[0], evidence_ids[1:],
                    attributes={"session_id": scene.session_id, "turn_ids": tuple(x.turn_id for x in scene.turns),
                                # Entities are what join the 2.68 sessions a LoCoMo
                                # cat1 question needs; they are only useful if the
                                # same name recurs verbatim in another session.
                                **({"entities": packet.entities} if packet.entities else {}),
                                **({
                                    "information_unit_total": len(packet.information_units),
                                    "information_unit_covered": len(packet.covered_unit_ids),
                                    "information_unit_unresolved": len(packet.unresolved_unit_ids),
                                    "information_unit_missing": len(packet.missing_unit_ids),
                                    "raw_fallback_turn_ids": packet.raw_fallback_turn_ids,
                                    "fact_cap": packet.fact_cap,
                                } if packet.information_units else {}),
                                "roles": ("scene", "route") if lean_graph else ("scene",),
                                **({"provenance_scope": "route"} if lean_graph else {})},
                )
                scene_nodes[scene.scene_id] = scene_node
                nodes.append(scene_node)
                if lean_graph:
                    for turn in scene.turns:
                        evidence_ref = self._turn_evidence_ref(
                            memory_id, scene.scene_id, turn, group_by_turn[turn.turn_id])
                        if turn.turn_id in packet.raw_fallback_turn_ids:
                            evidence_ref = replace(evidence_ref, attributes={
                                **dict(evidence_ref.attributes),
                                "raw_fallback": True,
                                "roles": tuple(dict.fromkeys((
                                    *evidence_ref.attributes.get("roles", ()),
                                    "raw_fallback", "mandatory_evidence"))),
                            })
                        nodes.append(evidence_ref)
                        edges.append(self._edge(memory_id, scene_node,
                                                RelationType.SCENE_CONTAINS, evidence_ref,
                                                "lossless_terminal_ref"))
                has_semantic_facts = bool(packet_by_scene.get(
                    scene.scene_id, ScenePacket(scene.scene_id, "", (), ())).facts)
                for event_index, event in enumerate(() if lean_graph and has_semantic_facts
                                                     else self._events(scene, profile)):
                    event_nodes[scene.scene_id].append(event)
                    nodes.append(event)

        if level >= 2:
            entity_nodes.update(self._entities(
                memory_id, scenes, event_nodes, group_by_turn, route_only=lean_graph))
            nodes.extend(entity_nodes.values())
            nodes.extend(self._time_and_state_nodes(memory_id, event_nodes))
            if packets:
                semantic_nodes, semantic_edges = self._semantic_graph(
                    memory_id, packets, turns, group_by_turn, profile, scene_nodes,
                    hierarchy_cards_for_semantics
                )
                nodes.extend(semantic_nodes); edges.extend(semantic_edges)

        if level >= 3:
            if recursive_hierarchy is not None:
                edges.extend(self._recursive_hierarchy_edges(
                    memory_id, recursive_hierarchy, session_cards,
                    scene_nodes, event_nodes))
            else:
                edges.extend(self._hierarchy_edges(
                    memory_id, memory_card, l2_cards, session_cards, scene_nodes, event_nodes
                ))
            edges.extend(self._typed_edges(memory_id, event_nodes, entity_nodes))

        gated_plan: GatedRelationPlan | None = None
        relation_semantic_vector_count = 0
        rare_lexical_terms: Mapping[str, frozenset[str]] = {}
        if (recursive_hierarchy is not None
                and profile.edges.parent_gated_relations and level >= 3):
            node_map = {node.node_id: node for node in nodes}
            child_map: dict[str, list[str]] = {
                parent: list(children)
                for parent, children in recursive_hierarchy.children.items()
            }
            for scene in scene_nodes.values():
                session_id = str(scene.attributes.get("session_id", ""))
                card = session_cards.get(session_id)
                if card is not None:
                    child_map.setdefault(card.node_id, []).append(scene.node_id)
            for edge in edges:
                if edge.relation == RelationType.SCENE_CONTAINS:
                    child_map.setdefault(edge.src_id, []).append(edge.dst_id)
            # Relation candidates need proposition-level geometry.  Reusing a
            # supporting turn vector collapses every fact extracted from that
            # turn to one point and loses predicate/value distinctions.  An
            # explicit provider therefore embeds atomic summaries at their
            # native granularity.  These vectors feed a typed-only side channel
            # and never alter routing/scene coarse-edge scores.
            semantic_vectors: Mapping[str, Sequence[float]] = {}
            if (profile.edges.relation_candidate_method == "hnsw"
                    and self.relation_vector_provider is not None):
                relation_vector_nodes = tuple(
                    node for node in node_map.values()
                    if node.node_type in ATOMIC_RELATION_NODE_TYPES)
                semantic_vectors = self.relation_vector_provider(
                    memory_id, relation_vector_nodes)
                relation_semantic_vector_count = len(semantic_vectors)
            if (profile.edges.relation_mask_propagation
                    and profile.edges.rare_lexical_relation
                    and str(RelationSignal.LEXICAL_RARE)
                    in profile.edges.enabled_relation_signals):
                rare_lexical_terms = build_rare_lexical_node_terms(
                    tuple(node_map.values()), turns,
                    df_share=profile.edges.rare_lexical_df_share)
            gated_plan = build_parent_gated_relations(
                memory_id, recursive_hierarchy, node_map, child_map,
                embedding_k=profile.edges.embedding_k,
                max_candidates_per_node=profile.edges.max_candidates_per_node,
                low_threshold=profile.edges.low_threshold,
                high_threshold=profile.edges.high_threshold,
                refine_mode=profile.edges.refine_mode,
                candidate_method=profile.edges.relation_candidate_method,
                vectors=recursive_hierarchy.vectors,
                hnsw_dimension=profile.coarsen.hnsw_dimension,
                hnsw_m=profile.coarsen.hnsw_m,
                hnsw_ef_construction=profile.coarsen.hnsw_ef_construction,
                cross_session_quota=profile.edges.cross_session_neighbor_quota,
                typed_restoration=profile.edges.typed_relation_restoration,
                typed_min_confidence=profile.edges.typed_relation_min_confidence,
                max_refine_candidates_per_node=(
                    profile.edges.max_refine_candidates_per_node),
                max_refine_candidates_per_1000_nodes=(
                    profile.edges.max_refine_candidates_per_1000_nodes),
                atomic_vector_channels=((semantic_vectors,)
                                        if semantic_vectors else ()),
                relation_mask_propagation=(
                    profile.edges.relation_mask_propagation),
                atomic_relation_multiview=(
                    profile.edges.atomic_relation_multiview),
                relation_view_quotas=profile.edges.relation_view_quotas,
                lexical_rare_terms=rare_lexical_terms,
                rare_lexical_min_shared=(
                    profile.edges.rare_lexical_min_shared),
                enabled_signals=profile.edges.enabled_relation_signals,
                predicate_family_state_relations=(
                    profile.edges.predicate_family_state_relations),
            )
            for left_id, right_id, score, _gate_level in gated_plan.accepted_pairs:
                left, right = node_map[left_id], node_map[right_id]
                relation_group_ids = tuple(dict.fromkeys((
                    left.evidence_group_id, right.evidence_group_id)))
                signals = gated_plan.accepted_pair_signals.get(
                    (left_id, right_id), ())
                relation_source = ("relation_mask:" + ",".join(signals)
                                   if signals else "cir_high_confidence")
                witnesses = gated_plan.accepted_pair_witnesses.get(
                    (left_id, right_id), {})
                if witnesses:
                    relation_source += "|relation_witness:" + canonical_json(
                        witnesses)
                edges.append(GraphEdge(
                    stable_id("edge", memory_id, left_id,
                              RelationType.COARSE_RELATED, right_id),
                    memory_id, left_id, RelationType.COARSE_RELATED, right_id,
                    relation_group_ids[0], False, score,
                    relation_source, relation_group_ids[1:]))
            for (left_id, right_id, relation, confidence, _gate_level,
                 source) in gated_plan.typed_pairs:
                left, right = node_map[left_id], node_map[right_id]
                relation_group_ids = tuple(dict.fromkeys((
                    left.evidence_group_id, right.evidence_group_id)))
                typed_witnesses = gated_plan.typed_pair_witnesses.get(
                    (left_id, right_id, str(relation)), {})
                if typed_witnesses:
                    source += "|relation_witness:" + canonical_json(
                        typed_witnesses)
                edges.append(GraphEdge(
                    stable_id("edge", memory_id, left_id, relation, right_id),
                    memory_id, left_id, relation, right_id,
                    relation_group_ids[0],
                    relation in DIRECTIONAL_REFINED_RELATIONS,
                    confidence, source,
                    relation_group_ids[1:]))

        refine_tokens = {"cached_input_tokens": 0, "uncached_input_tokens": 0,
                         "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        truncated: tuple[str, ...] = ()
        if level >= 4 and profile.edges.refine_mode != "none" and self.refiner:
            candidates = (gated_plan.refine_candidates if gated_plan is not None
                          else self._ambiguous_candidates(
                              memory_id, scenes, event_nodes, entity_nodes))
            decisions, truncated = self.refiner.refine(memory_id, candidates)
            by_candidate = {candidate.candidate_id: candidate for candidate in candidates}
            node_map = {node.node_id: node for node in nodes}
            for decision in decisions:
                if decision.decision == "NONE" or decision.candidate_id not in by_candidate:
                    continue
                candidate = by_candidate[decision.candidate_id]
                relation = RelationType(decision.decision)
                left_id, right_id = candidate.left_id, candidate.right_id
                if not admit_llm_refined_relation(
                        relation, node_map[left_id], node_map[right_id],
                        decision.confidence,
                        min_confidence=(
                            profile.edges.typed_relation_min_confidence)):
                    continue
                if (decision.inverse
                        and relation in DIRECTIONAL_REFINED_RELATIONS):
                    left_id, right_id = right_id, left_id
                left = node_map[left_id]; right = node_map[right_id]
                evidence_groups = tuple(dict.fromkeys((
                    left.evidence_group_id, right.evidence_group_id)))
                refine_source = (
                    decision.source + "|relation_mask:" + ",".join(
                        gated_plan.refine_candidate_signals.get(
                            decision.candidate_id, ()))
                    if gated_plan is not None and
                    gated_plan.refine_candidate_signals.get(
                        decision.candidate_id) else decision.source)
                refine_witnesses = (
                    gated_plan.refine_candidate_witnesses.get(
                        decision.candidate_id, {})
                    if gated_plan is not None else {})
                if refine_witnesses:
                    refine_source += "|relation_witness:" + canonical_json(
                        refine_witnesses)
                edges.append(GraphEdge(
                    stable_id("edge", memory_id, left_id, relation, right_id),
                    memory_id, left_id, relation, right_id,
                    evidence_groups[0], relation in DIRECTIONAL_REFINED_RELATIONS,
                    decision.confidence,
                    refine_source,
                    evidence_groups[1:],
                ))
            usage_after = self._usage(memory_id)
            refine_tokens = {key: usage_after[key] - usage_before[key] for key in usage_after}

        if level >= 5:
            if profile.coarsen.cross_session_merge and not packets:
                edges.extend(self._portal_edges(memory_id, session_cards, scene_nodes))

        nodes = self._dedup_nodes(nodes)
        edges = self._bounded_edges(self._dedup_edges(edges), profile)
        if lean_graph:
            nodes = self._propagate_time_ranges(nodes, edges)
        method_diagnostics: dict[str, Any] = {
            "recursive_hierarchy_enabled": recursive_hierarchy is not None,
            "parent_gated_relations_enabled": gated_plan is not None,
            "relation_semantic_vector_count": relation_semantic_vector_count,
            "relation_vector_granularity": (
                "atomic_summary" if relation_semantic_vector_count
                else "deterministic_lexical_fallback"),
            "enabled_relation_signals": tuple(
                profile.edges.enabled_relation_signals),
        }
        if recursive_hierarchy is not None:
            method_diagnostics["coarsening"] = dataclass_dict(
                recursive_hierarchy.stats)
        if gated_plan is not None:
            method_diagnostics["cir"] = {
                "coarse_candidate_pairs": gated_plan.coarse_candidate_pairs,
                "gated_child_pairs": gated_plan.gated_child_pairs,
                "score_comparisons": gated_plan.score_comparisons,
                "accepted_pairs": len(gated_plan.accepted_pairs),
                "refine_candidates": len(gated_plan.refine_candidates),
                "refine_candidates_generated": (
                    gated_plan.refine_candidates_generated),
                "refine_candidates_dropped": (
                    gated_plan.refine_candidates_dropped),
                "atomic_relation_candidates_generated": (
                    gated_plan.atomic_relation_candidates_generated),
                "atomic_relation_pairs_proposed": (
                    gated_plan.atomic_relation_pairs_proposed),
                "relation_mask_pairs": gated_plan.relation_mask_pairs,
                "relation_mask_counts": dict(
                    gated_plan.relation_mask_counts),
                "rare_lexical_feature_nodes": len(rare_lexical_terms),
                "atomic_candidate_source_counts": dict(
                    gated_plan.atomic_candidate_source_counts),
                "atomic_candidate_signal_counts": dict(
                    gated_plan.atomic_candidate_signal_counts),
                "levels_with_relations": gated_plan.levels_with_relations,
                "typed_pairs": len(gated_plan.typed_pairs),
                "candidate_method": gated_plan.candidate_method,
            }
        version = self.store.replace_graph(memory_id, nodes, edges, groups)
        checksum = logical_graph_checksum(nodes, edges)
        artifact_id = stable_id("graph-artifact", memory_id, self.dataset_hash,
                                config_hash(profile), checksum, version)
        usage_after = self._usage(memory_id)
        token_usage = {key: usage_after[key] - usage_before[key] for key in usage_after}
        token_usage.update({
            "truncated_candidates": len(truncated),
            "wall_time_ms": round((time.perf_counter() - started) * 1000),
            "coarsen_candidate_comparisons": (
                recursive_hierarchy.stats.cluster_candidate_comparisons
                if recursive_hierarchy is not None else 0),
            "coarse_relation_candidates": (
                gated_plan.coarse_candidate_pairs if gated_plan is not None else 0),
            "gated_child_candidates": (
                gated_plan.gated_child_pairs if gated_plan is not None else 0),
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
            build_diagnostics=self._build_diagnostics(
                memory_id, nodes, edges, packets, method_diagnostics),
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

    @staticmethod
    def _turn_evidence_ref(memory_id, scene_id, turn, group):
        raw_tokens = WORD_RE.findall(turn.raw_text)
        priority = []
        media = re.findall(r"caption:\s*([^\]]+)", turn.raw_text, re.I)
        quoted = re.findall(r'["“]([^"”]{2,80})["”]', turn.raw_text)
        for phrase in (*media, *quoted):
            priority.extend(WORD_RE.findall(phrase))
        priority.extend(token for token in raw_tokens if any(char.isdigit() for char in token))
        # Head-only sketches systematically hid late-turn payloads in LoCoMo.
        # Interleave salient caption/number tokens with the head and tail while
        # keeping this lossless terminal reference compact.
        ordered_tokens = [*priority, *raw_tokens[:12], *raw_tokens[-12:]]
        terms = []
        for token in ordered_tokens:
            normalized = token.casefold()
            if normalized in EVIDENCE_REF_STOPWORDS or len(normalized) < 2:
                continue
            if normalized not in terms:
                terms.append(normalized)
            if len(terms) >= 18:
                break
        explicit_time = extract_time_expression(turn.raw_text)
        interval = normalize_time(explicit_time, turn.timestamp, turn.turn_id) if explicit_time else None
        value_type = ("currency" if re.search(r"[$£€¥]\s*\d", turn.raw_text)
                      else "number" if re.search(r"\d", turn.raw_text)
                      else "time" if explicit_time else "text")
        return GraphNode(
            stable_id("node", memory_id, "evidence-ref", turn.turn_id), memory_id,
            NodeType.EVIDENCE_GROUP_REF, 0,
            " ".join((turn.speaker, *terms)), group.evidence_group_id,
            event_time=interval.start if interval and interval.start else None,
            attributes={"scene_id": scene_id, "session_id": turn.session_id,
                        "turn_id": turn.turn_id, "turn_index": turn.turn_index,
                        "value_type": value_type,
                        "time_interval": dataclass_dict(interval) if interval else None,
                        "roles": ("evidence_turn", "terminal"),
                        "provenance_scope": "terminal"})

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
                    **({"provenance_scope": "terminal"} if profile.edges.graph_variant == "g5" else {}),
                },
            ))
        return result

    def _entities(self, memory_id, scenes, event_nodes, group_by_turn, *, route_only=False):
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
                            "roles": ("entity", "route") if route_only else ("entity",),
                            **({"provenance_scope": "route"} if route_only else {})},
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

    def _semantic_graph(self, memory_id, packets, turns, group_by_turn, profile,
                        scene_nodes, hierarchy_cards=()):
        turn_map = {turn.turn_id: turn for turn in turns}
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        variant = profile.edges.graph_variant
        if variant == "g5":
            return self._lean_semantic_graph(
                memory_id, packets, turn_map, group_by_turn, profile, scene_nodes)
        facts: list[tuple[GraphNode, str, str, str, int]] = []
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
                    "information_unit_ids": fact.information_unit_ids,
                    "evidence_spans": tuple({"turn_id": turn_id, "start": start, "end": end}
                                            for turn_id, start, end in fact.evidence),
                    "roles": ("fact", "predicate", "object")})
                source_turn = turn_map[refs[0][0]]
                nodes[fact_id] = fact_node; facts.append((fact_node, owner_id, value_id,
                    source_turn.session_id, source_turn.turn_index))
                edges.append(self._edge(memory_id, nodes[owner_id], RelationType.HAS_FACT, fact_node, "semantic"))
                if variant == "g0":
                    edges.append(self._edge(memory_id, fact_node, RelationType.FACT_VALUE, nodes[value_id], "semantic"))
        by_value: dict[str, list[tuple[GraphNode, str]]] = defaultdict(list)
        by_activity: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
        by_state: dict[tuple[str, str], list[tuple[GraphNode, str, int]]] = defaultdict(list)
        for fact, owner_id, value_id, session_id, turn_index in facts:
            by_value[value_id].append((fact, session_id))
            by_activity[(owner_id, str(fact.attributes["predicate"]))].append(fact)
            by_state[(owner_id, str(fact.attributes["predicate"]))].append(
                (fact, session_id, turn_index))
            if variant in {"g2", "g3", "g4"} and fact.event_time:
                time_id = stable_id("node", memory_id, "semantic-time", fact.event_time)
                nodes.setdefault(time_id, GraphNode(time_id, memory_id, NodeType.TIME_ANCHOR, 0,
                    fact.event_time, fact.evidence_group_id, fact.evidence_group_ids,
                    event_time=fact.event_time, attributes={"roles": ("time", "anchor")}))
                edges.append(self._edge(memory_id, fact, RelationType.AT_TIME, nodes[time_id], "temporal"))
        if variant in {"g0", "g4"} and profile.coarsen.cross_session_merge:
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
        if variant == "g0":
            for rows in by_activity.values():
                for left, right in zip(rows, rows[1:]):
                    edges.append(self._edge(memory_id, left, RelationType.SAME_ACTIVITY, right, "semantic"))
        if variant in {"g2", "g3", "g4"}:
            for rows in by_state.values():
                ordered = sorted(rows, key=lambda row: (row[0].event_time or "", row[1], row[2], row[0].node_id))
                for (left, _, _), (right, _, _) in zip(ordered, ordered[1:]):
                    if left.attributes.get("value_id") != right.attributes.get("value_id"):
                        edges.append(self._edge(memory_id, left, RelationType.STATE_NEXT, right, "temporal_state"))
                    if left.event_time and right.event_time and left.event_time != right.event_time:
                        edges.append(self._edge(memory_id, left, RelationType.TEMPORAL_BEFORE, right, "temporal"))
        if variant in {"g3", "g4"}:
            for (owner_id, predicate), rows in by_state.items():
                values = {row[0].attributes.get("value_id") for row in rows}
                if len(values) < 2:
                    continue
                evidence = tuple(dict.fromkeys(group for fact, _, _ in rows
                                                for group in fact.all_evidence_group_ids))
                collection_id = stable_id("node", memory_id, "semantic-collection", owner_id, predicate)
                collection = GraphNode(collection_id, memory_id, NodeType.COLLECTION_SCOPE, 1,
                    f"{nodes[owner_id].summary} {predicate}", evidence[0], evidence[1:],
                    attributes={"owner_id": owner_id, "predicate": predicate,
                                "roles": ("collection_scope", "route")})
                nodes[collection_id] = collection
                for fact, _, _ in rows:
                    edges.append(self._edge(memory_id, collection, RelationType.COLLECTION_CO_MEMBER,
                                            fact, "semantic_collection"))
        values_by_label = {self._normal(node.summary): node for node in nodes.values()
                           if node.node_type == NodeType.CANONICAL_VALUE}
        seen_aliases = set()
        for card in hierarchy_cards if variant in {"g0", "g4"} else ():
            for group in card.attributes.get("aliases", ()):
                candidates = [values_by_label.get(self._normal(value)) for value in group]
                candidates = [node for node in candidates if node]
                for left, right in zip(candidates, candidates[1:]):
                    key = tuple(sorted((left.node_id, right.node_id)))
                    if key in seen_aliases: continue
                    seen_aliases.add(key)
                    edges.append(self._edge(memory_id, left, RelationType.COREFERENCE, right, "semantic_alias"))
        return list(nodes.values()), edges

    def _lean_semantic_graph(self, memory_id, packets, turn_map, group_by_turn,
                             profile, scene_nodes):
        nodes: dict[str, GraphNode] = {}; edges: list[GraphEdge] = []
        chains: dict[tuple[str, str, str, str, str, str], list[
            tuple[GraphNode, str, int, Any, Any]
        ]] = defaultdict(list)
        predicate_keys = []
        for packet in packets:
            for fact in packet.facts:
                if not self._keep_lean_fact(fact):
                    continue
                owner_key = self._normal(fact.owner)
                predicate_key = self._predicate_key(fact)
                scope_key = self._scope_key(fact)
                predicate_keys.append((owner_key, predicate_key, scope_key,
                                       self._normal(fact.value_type), fact.polarity))
        predicate_map = (self.predicate_canonicalizer.canonicalize(memory_id, predicate_keys)
                         if self.predicate_canonicalizer and predicate_keys else
                         {key: key[1] for key in predicate_keys})
        for packet in packets:
            for fact in packet.facts:
                if not self._keep_lean_fact(fact):
                    continue
                refs = [ref for ref in fact.evidence if ref[0] in group_by_turn]
                if not refs:
                    continue
                groups = tuple(dict.fromkeys(group_by_turn[turn_id].evidence_group_id
                                             for turn_id, _, _ in refs))
                source_turn = turn_map[refs[0][0]]
                owner_key = self._normal(fact.owner)
                raw_predicate = self._predicate_key(fact)
                value_key = self._normal(fact.value)
                scope_key = self._scope_key(fact)
                predicate_key = predicate_map[(owner_key, raw_predicate, scope_key,
                                               self._normal(fact.value_type), fact.polarity)]
                owner_id = stable_id("node", memory_id, "semantic-owner", owner_key)
                nodes.setdefault(owner_id, GraphNode(
                    owner_id, memory_id, NodeType.CANONICAL_ENTITY, 0, fact.owner,
                    groups[0], groups[1:], entity_id=owner_id,
                    attributes={"aliases": (fact.owner,),
                                "entities": (owner_key,),
                                "relation_entity_roles": {"owner": (owner_key,)},
                                "roles": ("entity", "owner", "route"),
                                "provenance_scope": "route"}))
                observed = observed_interval(source_turn.timestamp, source_turn.turn_id)
                evidence_text = source_turn.raw_text[refs[0][1]:refs[0][2]]
                relation_entities: dict[str, set[str]] = {
                    "owner": {owner_key} if owner_key else set(),
                    "object": set(),
                }
                fact_surface = " ".join((
                    str(fact.predicate), str(fact.value), evidence_text))
                normalized_surface = f" {self._normal(fact_surface)} "
                for entity in packet.entities:
                    entity_key = self._normal(entity)
                    if (entity_key and entity_key != owner_key
                            and entity_key not in EVIDENCE_REF_STOPWORDS
                            and entity_key not in {"am", "pm"}
                            and f" {entity_key} " in normalized_surface):
                        relation_entities["object"].add(entity_key)
                # Frozen extraction caches may predate the scene-entity field.
                # Recover only source-grounded named phrases from the cited
                # span/value; never promote an arbitrary full value to an entity.
                for surface in (str(fact.value), evidence_text):
                    for match in ENTITY_MENTION_RE.finditer(surface):
                        mention = match.group(0)
                        entity_key = self._normal(mention)
                        if (not entity_key or entity_key == owner_key
                                or entity_key in EVIDENCE_REF_STOPWORDS
                                or entity_key in {"am", "pm"}
                                or (match.end() < len(surface)
                                    and surface[match.end()] in {"'", "\u2019"})
                                or (" " not in mention and len(mention) <= 2
                                    and not mention.isupper())
                                or len(entity_key) > 64):
                            continue
                        relation_entities["object"].add(entity_key)
                object_entities = tuple(sorted(
                    relation_entities["object"])[:16])
                # Lower-case object phrases are common in LoCoMo (for example
                # ``dance studio`` and ``cooking class``) and were previously
                # invisible to entity relation construction.  Promote only the
                # same conservative short-value contract used by recoarsening,
                # and retain the source role in node provenance.
                promoted_object = promotable_object_value(value_key)
                if promoted_object:
                    relation_entities["object"].add(promoted_object)
                    object_entities = tuple(sorted(
                        relation_entities["object"])[:16])
                # Modality must describe this fact, not a nearby sentence in the
                # same turn. Neighbour context caused completed events to inherit
                # unrelated phrases such as "would love to".
                modality = self._fact_modality(
                    f"{fact.predicate} {fact.value} {evidence_text}")
                explicit_time = (fact.time or extract_time_expression(evidence_text)
                                 or extract_time_expression(source_turn.raw_text))
                interval = (normalize_time(explicit_time, source_turn.timestamp, source_turn.turn_id)
                            if profile.edges.temporal_normalization and explicit_time else None)
                fact_id = stable_id("node", memory_id, "lean-fact", owner_key, predicate_key,
                                    value_key, scope_key, fact.polarity, tuple(refs))
                attrs = {"owner_id": owner_id,
                         "entities": tuple(filter(None, (
                             owner_key, *object_entities))),
                         "relation_entity_roles": {
                             role: tuple(sorted(values))
                             for role, values in relation_entities.items()
                             if values},
                         "predicate": predicate_key, "value": fact.value,
                         "value_key": value_key, "value_type": fact.value_type, "scope": scope_key,
                         "polarity": fact.polarity, "scene_id": packet.scene_id,
                         "information_unit_ids": fact.information_unit_ids,
                         "evidence_spans": tuple({"turn_id": turn_id, "start": start, "end": end}
                                                 for turn_id, start, end in fact.evidence),
                         "modality": modality,
                         "session_id": source_turn.session_id, "turn_index": source_turn.turn_index,
                         "roles": ("fact", "predicate", "object") +
                                  (("negative_scope",) if fact.polarity == "negative" else ()) +
                                  ((modality,) if modality != "asserted" else ()),
                         "provenance_scope": "terminal",
                         "observed_at": dataclass_dict(observed) if observed else None,
                         "time_interval": dataclass_dict(interval) if interval else None}
                fact_node = GraphNode(
                    fact_id, memory_id, NodeType.CANONICAL_FACT, 0,
                    f"{fact.owner} " + (f"{modality} " if modality != "asserted" else "") +
                    f"{fact.predicate} {fact.value}" +
                    (f" {explicit_time}" if explicit_time else ""), groups[0], groups[1:],
                    event_time=interval.start if interval and interval.start else None,
                    state=fact.value, confidence=fact.confidence, attributes=attrs)
                nodes[fact_id] = fact_node
                edges.append(self._edge(memory_id, nodes[owner_id], RelationType.HAS_FACT,
                                        fact_node, "lean_semantic"))
                if packet.scene_id in scene_nodes:
                    edges.append(self._edge(memory_id, scene_nodes[packet.scene_id],
                                            RelationType.SCENE_CONTAINS, fact_node, "lean_semantic"))
                    # Query seeds are scenes. This explicit owner link makes
                    # related facts reachable as Scene -> Entity -> Fact within
                    # the fixed two-hop budget, without fuzzy person merging.
                    edges.append(self._edge(memory_id, scene_nodes[packet.scene_id],
                                            RelationType.PARTICIPATES_IN, nodes[owner_id],
                                            "lean_scene_owner"))
                if interval and interval.start:
                    time_id = stable_id("node", memory_id, "time-interval", interval.start,
                                        interval.end, interval.precision)
                    nodes.setdefault(time_id, GraphNode(
                        time_id, memory_id, NodeType.TIME_ANCHOR, 0, interval.raw_text,
                        groups[0], groups[1:], event_time=interval.start,
                        attributes={"interval": dataclass_dict(interval),
                                    "roles": ("time", "temporal_left", "temporal_right"),
                                    "provenance_scope": "terminal"}))
                    edges.append(self._edge(memory_id, fact_node, RelationType.AT_TIME,
                                            nodes[time_id], "normalized_temporal"))
                collection_key = self._collection_key(fact)
                attrs["collection_key"] = collection_key
                # attrs changed after GraphNode creation; preserve the value on
                # the immutable node used by downstream chain compilation.
                fact_node = replace(fact_node, attributes=attrs)
                nodes[fact_id] = fact_node
                chains[(owner_id, predicate_key, scope_key, collection_key,
                        fact.polarity, modality)].append(
                    (fact_node, source_turn.session_id, source_turn.turn_index, interval, observed))

        for (owner_id, predicate, scope, collection_key, polarity, modality), rows in sorted(chains.items()):
            values = {str(row[0].attributes["value_key"]) for row in rows}
            # Read the aspect back off the members rather than carrying it in the
            # chain key, so this matches projection/manifest.py exactly -- it
            # derives occurrence the same way.
            occurrence_collection = any(row[0].attributes.get("event_instance_id") for row in rows)
            if len(rows) < 2 or (len(values) < 2 and not occurrence_collection):
                continue
            ordered = sorted(rows, key=lambda row: (
                row[3].start if row[3] and row[3].start else
                row[4].start if row[4] and row[4].start else "",
                row[1], row[2], row[0].node_id))
            evidence = tuple(dict.fromkeys(group for fact, *_ in ordered
                                            for group in fact.all_evidence_group_ids))
            collection_id = stable_id("node", memory_id, "lean-collection", owner_id,
                                      predicate, scope, collection_key, polarity, modality)
            collection = GraphNode(
                collection_id, memory_id, NodeType.COLLECTION_SCOPE, 1,
                f"{nodes[owner_id].summary} " +
                ("not " if polarity == "negative" else "") +
                (f"{modality} " if modality != "asserted" else "") + f"{predicate} " +
                " ".join(str(row[0].attributes["value"]) for row in ordered[:8]),
                evidence[0], evidence[1:], attributes={
                    "owner_id": owner_id,
                    "entities": (self._normal(nodes[owner_id].summary),),
                    "relation_entity_roles": {
                        "owner": (self._normal(nodes[owner_id].summary),)},
                    "predicate": predicate, "scope": scope,
                    "collection_key": collection_key,
                    "polarity": polarity, "modality": modality,
                    "member_count": len(ordered),
                    "collection_semantics": ("event_instances" if occurrence_collection
                                             else "distinct_values"),
                    "roles": ("collection_scope", "route"), "provenance_scope": "route"})
            nodes[collection_id] = collection
            edges.append(self._edge(memory_id, nodes[owner_id], RelationType.HAS_FACT,
                                    collection, "lean_collection_route"))
            if not occurrence_collection:
                state_id = stable_id("node", memory_id, "state-head", owner_id, predicate,
                                     scope, collection_key, polarity, modality)
                state = GraphNode(
                    state_id, memory_id, NodeType.STATE_HEAD, 0,
                    f"{nodes[owner_id].summary} {predicate} {ordered[-1][0].attributes['value']}",
                    evidence[0], evidence[1:], state=str(ordered[-1][0].attributes["value"]),
                    attributes={"owner_id": owner_id,
                                "entities": (self._normal(nodes[owner_id].summary),),
                                "relation_entity_roles": {
                                    "owner": (self._normal(nodes[owner_id].summary),)},
                                "predicate": predicate, "scope": scope,
                                "roles": ("prior_state", "current_state"),
                                "polarity": polarity, "modality": modality,
                                "provenance_scope": "terminal",
                                "event_time_range": self._range_from_intervals(
                                    [row[3] for row in ordered]),
                                "observation_time_range": self._range_from_intervals(
                                    [row[4] for row in ordered])})
                nodes[state_id] = state
                edges.append(self._edge(memory_id, nodes[owner_id], RelationType.HAS_FACT,
                                        state, "lean_state"))
            for fact, *_ in ordered[:32]:
                edges.append(self._edge(memory_id, collection, RelationType.COLLECTION_CO_MEMBER,
                                        fact, "lean_collection"))
            # The collection hub path needs three hops from a Scene. Compile a
            # bounded member projection so list/count evidence is navigable in
            # two hops while retaining CollectionScope as the authority node.
            members = [row[0] for row in ordered[:32]]
            member_cap = int(profile.edges.relation_degree_caps.get(
                str(RelationType.COLLECTION_CO_MEMBER), profile.edges.max_degree_per_relation))
            member_degree: dict[str, int] = defaultdict(int)
            for left_index, left in enumerate(members):
                for right in members[left_index + 1:]:
                    if (member_degree[left.node_id] >= member_cap
                            or member_degree[right.node_id] >= member_cap):
                        continue
                    edges.append(self._edge(memory_id, left,
                                            RelationType.COLLECTION_CO_MEMBER, right,
                                            "lean_collection_projection"))
                    member_degree[left.node_id] += 1
                    member_degree[right.node_id] += 1
            for left, right in zip(ordered, ordered[1:]):
                if (not occurrence_collection
                        and left[0].attributes["value_key"] == right[0].attributes["value_key"]):
                    continue
                edges.append(self._edge(memory_id, left[0], RelationType.STATE_NEXT,
                                        right[0], "normalized_state"))
                if (left[3] and right[3] and left[3].end and right[3].start
                        and left[3].end < right[3].start):
                    edges.append(self._edge(memory_id, left[0], RelationType.TEMPORAL_BEFORE,
                                            right[0], "normalized_temporal"))
            if profile.edges.cross_session_portals:
                degree: dict[str, int] = defaultdict(int)
                for left, right in zip(ordered, ordered[1:]):
                    if left[1] == right[1]:
                        continue
                    if (degree[left[0].node_id] >= profile.edges.portal_degree_cap or
                            degree[right[0].node_id] >= profile.edges.portal_degree_cap):
                        continue
                    edges.append(self._edge(memory_id, left[0], RelationType.PORTAL,
                                            right[0], "bounded_exact_portal"))
                    degree[left[0].node_id] += 1; degree[right[0].node_id] += 1
        return list(nodes.values()), edges

    @staticmethod
    def _range_from_intervals(intervals):
        starts = [row.start for row in intervals if row and row.start]
        ends = [row.end for row in intervals if row and row.end]
        return (min(starts), max(ends)) if starts and ends else None

    def _propagate_time_ranges(self, nodes, edges):
        node_map = {node.node_id: node for node in nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.relation in {RelationType.REFINES_TO, RelationType.SCENE_CONTAINS,
                                 RelationType.HAS_FACT, RelationType.COLLECTION_CO_MEMBER}:
                children[edge.src_id].append(edge.dst_id)

        def own_ranges(node):
            interval = node.attributes.get("time_interval") or node.attributes.get("interval")
            observed = node.attributes.get("observed_at")
            event_range = ((interval.get("start"), interval.get("end"))
                           if isinstance(interval, dict) and interval.get("start") and interval.get("end")
                           else node.attributes.get("event_time_range"))
            observed_range = ((observed.get("start"), observed.get("end"))
                              if isinstance(observed, dict) and observed.get("start") and observed.get("end")
                              else node.attributes.get("observation_time_range"))
            return event_range, observed_range

        def aggregate(node_id):
            event_rows = []; observed_rows = []
            own_event, own_observed = own_ranges(node_map[node_id])
            if own_event: event_rows.append(own_event)
            if own_observed: observed_rows.append(own_observed)
            for child_id in children.get(node_id, ()):
                if child_id not in node_map:
                    continue
                child_event, child_observed = own_ranges(node_map[child_id])
                if child_event: event_rows.append(child_event)
                if child_observed: observed_rows.append(child_observed)
            event_range = ((min(row[0] for row in event_rows), max(row[1] for row in event_rows))
                           if event_rows else None)
            observed_range = ((min(row[0] for row in observed_rows), max(row[1] for row in observed_rows))
                              if observed_rows else None)
            attrs = dict(node_map[node_id].attributes)
            attrs["event_time_range"] = event_range
            attrs["observation_time_range"] = observed_range
            node_map[node_id] = replace(node_map[node_id], attributes=attrs,
                                        event_time=event_range[0] if event_range else node_map[node_id].event_time)

        first = [node for node in nodes if node.node_type in {
            NodeType.CANONICAL_ENTITY, NodeType.SCENE, NodeType.COLLECTION_SCOPE, NodeType.STATE_HEAD}]
        for node in sorted(first, key=lambda row: (row.level, row.node_id)):
            aggregate(node.node_id)
        routes = [node for node in nodes if node.node_type == NodeType.ROUTING_CARD]
        for node in sorted(routes, key=lambda row: (row.level, row.node_id)):
            aggregate(node.node_id)
        return [node_map[node.node_id] for node in nodes]

    @staticmethod
    def _normal(value):
        return " ".join(re.findall(r"[\w'-]+", value.casefold()))

    # The three ladders that used to live here -- _canonical_predicate (win /
    # participate / write / receive / visit / read / attend / buy),
    # _canonical_scope (competition / writing / travel / reading / acquisition)
    # and _collection_key (tournament / game / screenplay / letter / book / poem
    # / pet) -- were regexes hand-fitted to a handful of LongMemEval questions.
    # They are gone for two reasons.  They are dataset constants sitting in
    # general build config, and they did not generalize: _collection_key matched
    # none of model kit, fitness class, film festival, health device or delivery
    # service, so it fell through to value_type (4.4 distinct per memory) and
    # left the collection chain keyed on the predicate instead -- 369.3 distinct
    # per memory, which is why 91.5% of an aggregation question's gold facts land
    # in different chains and a count has no set to range over.  Extraction now
    # supplies all three as fields, uniformly for every memory.

    @classmethod
    def _predicate_key(cls, fact):
        return cls._normal(str(fact.predicate).replace("_", " "))

    @classmethod
    def _scope_key(cls, fact):
        return cls._normal(fact.scope) or "general"

    @classmethod
    def _collection_key(cls, fact):
        """The class of thing this fact is about -- the key a count ranges over."""
        return cls._normal(fact.value_type) or "value"

    @classmethod
    def _keep_lean_fact(cls, fact):
        predicate = cls._normal(str(fact.predicate).replace("_", " "))
        value = cls._normal(fact.value)
        if re.search(r"\b(?:shared|showed|sent)\b.*\b(?:media|photo|picture|image|caption)\b",
                     predicate):
            return False
        if value in {"yes", "true", "it", "that"} and len(predicate.split()) <= 2:
            return False
        return True

    @staticmethod
    def _fact_modality(text):
        normalized = " ".join(str(text).casefold().split())
        if re.search(r"\b(?:currently|right now|at the moment)\b", normalized):
            return "current"
        if re.search(
            r"\b(?:plan(?:s|ned|ning)? to|planning on|want(?:s|ed)? to|hope(?:s|d)? to|"
            r"would love to|going to|will|next (?:week|month|year)|to-do list|dream trip|"
            r"can't wait to|cannot wait to|upcoming)\b", normalized
        ):
            return "planned"
        return "asserted"

    @staticmethod
    def _packet_record(packet):
        return {"child_id": packet.scene_id, "summary": " ".join(packet.summary.split()[:48]),
                "owners": tuple(dict.fromkeys(
                    tuple(fact.owner for fact in packet.facts) + tuple(packet.entities))),
                "predicates": tuple(dict.fromkeys(fact.predicate for fact in packet.facts)),
                "values": tuple(dict.fromkeys(fact.value for fact in packet.facts)),
                "scopes": tuple(dict.fromkeys(fact.scope for fact in packet.facts)),
                "times": tuple(dict.fromkeys(fact.time for fact in packet.facts if fact.time))}

    def _compile_routing(self, children, limit, aliases=None):
        fields = ("owners", "predicates", "values", "scopes", "times")
        aliases = aliases or {}
        collected = {field: [] for field in fields}; postings: dict[str, list[str]] = defaultdict(list)
        summary_parts: list[str] = []
        seen_summaries: set[str] = set()
        for child in children:
            child_id = str(child["child_id"])
            for field in fields:
                for value in child.get(field, ()):
                    text = " ".join(str(value).split())
                    if not text:
                        continue
                    collected[field].append(text)
                    keys = {self._normal(text), *self._terms(text)} - {""}
                    # A merged surface also indexes under its siblings, so a
                    # query naming the referent one way reaches the children that
                    # named it another -- which, since the merge only keeps keys
                    # spanning two or more sessions, is a cross-session hop.
                    keys |= {alias for key in tuple(keys) for alias in aliases.get(key, ())}
                    for key in keys:
                        postings[key].append(child_id)
            # Index the summary too.  Postings were built only from the fact
            # fields, so a question word that appears in a child's summary but in
            # none of its predicate/value strings could not reach that child at
            # all -- which is most question words once the summary is a sentence
            # rather than a concatenation of those same fields.
            child_summary = str(child.get("summary", ""))
            for key in self._terms(child_summary):
                if key:
                    postings[key].append(child_id)
            # Sibling children restate each other -- adjacent scenes in a session
            # carry overlapping facts -- and the card is truncated to `limit`
            # words, so a repeated sentence displaces a distinct one.  Skip a
            # child summary already contributed verbatim; postings above are
            # unaffected, since every child still indexes its own terms.
            normalized = " ".join(child_summary.split()).casefold()
            if normalized and normalized in seen_summaries:
                continue
            seen_summaries.add(normalized)
            summary_parts.extend(child_summary.split())
        if not summary_parts:
            for field in fields:
                summary_parts.extend(collected[field])
        return {
            "summary": " ".join(summary_parts[:limit]),
            **{field: tuple(dict.fromkeys(values))[:64] for field, values in collected.items()},
            "child_postings": {key: tuple(dict.fromkeys(ids)) for key, ids in sorted(postings.items())},
        }

    @staticmethod
    def _node_record(node):
        return {"child_id": node.node_id, "summary": node.summary,
                "owners": node.attributes.get("owners", ()),
                "predicates": node.attributes.get("predicates", ()),
                "values": node.attributes.get("values", ()), "scopes": node.attributes.get("scopes", ()),
                "times": node.attributes.get("times", ())}

    def _recursive_hierarchy_edges(self, memory_id, hierarchy, session_cards,
                                   scene_nodes, event_nodes):
        """Materialize a compact, strictly parent-to-child routing tree."""
        nodes = {
            node.node_id: node
            for node in (*session_cards.values(), *hierarchy.parent_cards,
                         *scene_nodes.values(),
                         *(event for rows in event_nodes.values() for event in rows))
        }

        def edge(left, relation, right):
            groups = tuple(dict.fromkeys((left.evidence_group_id,
                                         right.evidence_group_id)))
            return GraphEdge(
                stable_id("edge", memory_id, left.node_id, relation, right.node_id),
                memory_id, left.node_id, relation, right.node_id,
                groups[0], True, 1.0, "recursive_coarsening", groups[1:])

        edges = []
        for parent_id, child_ids in hierarchy.children.items():
            parent = nodes[parent_id]
            for child_id in child_ids:
                edges.append(edge(parent, RelationType.REFINES_TO, nodes[child_id]))
        for scene in scene_nodes.values():
            card = session_cards[str(scene.attributes["session_id"])]
            edges.append(edge(card, RelationType.REFINES_TO, scene))
            for event in event_nodes.get(scene.node_id, ()):
                edges.append(edge(scene, RelationType.SCENE_CONTAINS, event))
        return edges

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
        degree: dict[tuple[str, RelationType], int] = defaultdict(int)
        result: list[GraphEdge] = []
        for edge in sorted(edges, key=lambda item: (-item.confidence, item.edge_id)):
            cap = int(profile.edges.relation_degree_caps.get(
                str(edge.relation), profile.edges.max_degree_per_relation
            ))
            left = (edge.src_id, edge.relation)
            right = (edge.dst_id, edge.relation)
            if degree[left] >= cap:
                continue
            # An undirected edge consumes capacity at both endpoints.  The old
            # source-only cap permitted a coreference hub to receive unbounded
            # incoming edges even though every edge is traversable both ways.
            if not edge.directed and degree[right] >= cap:
                continue
            result.append(edge)
            degree[left] += 1
            if not edge.directed:
                degree[right] += 1
        return result

    def _usage(self, memory_id):
        # Through ``_read`` so the store lock is held.  Going at
        # ``_connection.execute`` directly races the other memory workers and
        # raises "bad parameter or other API misuse" intermittently -- the same
        # defect already fixed in the storage read helpers, missed here because
        # this is the one caller that reaches past them.
        rows = self.store._read(
            "SELECT usage_json,cached FROM llm_calls WHERE memory_id=?", (memory_id,))
        totals = {"cached_input_tokens": 0, "uncached_input_tokens": 0,
                  "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        import json
        for row in rows:
            usage = json.loads(row["usage_json"])
            for key in totals:
                totals[key] += int(usage.get(key, 0))
        return totals

    def _build_diagnostics(self, memory_id, nodes, edges, packets,
                           method_diagnostics: Mapping[str, Any] | None = None):
        import json
        scopes = defaultdict(int); terminal_groups = set(); relations = defaultdict(int)
        semantic_groups = set(); collection_semantics = defaultdict(int)
        degrees: dict[tuple[str, str], int] = defaultdict(int)
        for node in nodes:
            scope = str(node.attributes.get("provenance_scope", "terminal"))
            scopes[scope] += 1
            if scope == "terminal":
                terminal_groups.update(node.all_evidence_group_ids)
            if node.node_type in {NodeType.CANONICAL_FACT, NodeType.EVENT_SKELETON}:
                semantic_groups.update(node.all_evidence_group_ids)
            if node.node_type == NodeType.COLLECTION_SCOPE:
                collection_semantics[str(node.attributes.get(
                    "collection_semantics", "distinct_values"))] += 1
        for edge in edges:
            relations[str(edge.relation)] += 1; degrees[(edge.src_id, str(edge.relation))] += 1
        terminal_turns = set()
        for group_id in terminal_groups:
            group = self.store.evidence_group(group_id)
            if group:
                terminal_turns.update(member.turn_id for member in group.members)
        semantic_turns = set()
        for group_id in semantic_groups:
            group = self.store.evidence_group(group_id)
            if group:
                semantic_turns.update(member.turn_id for member in group.members)
        source_turn_count = max(1, len(self.store.turns(memory_id)))
        facts_per_scene = sorted(len(packet.facts) for packet in packets)
        unit_total = sum(len(packet.information_units) for packet in packets)
        unit_covered = sum(len(packet.covered_unit_ids) for packet in packets)
        unit_unresolved = sum(len(packet.unresolved_unit_ids) for packet in packets)
        unit_missing = sum(len(packet.missing_unit_ids) for packet in packets)
        unit_implicit = sum(len(packet.implicitly_covered_unit_ids) for packet in packets)
        raw_fallback_turns = {
            turn_id for packet in packets for turn_id in packet.raw_fallback_turn_ids
        }
        fact_caps = sorted(packet.fact_cap for packet in packets if packet.fact_cap)
        stage_usage: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in self.store._read(
            "SELECT c.stage,c.cache_key,k.usage_json FROM llm_calls c JOIN llm_cache k "
            "ON k.cache_key=c.cache_key WHERE c.memory_id=?", (memory_id,)):
            stage_usage[str(row["stage"])][str(row["cache_key"])] = json.loads(row["usage_json"])
        degree_values = sorted(degrees.values())
        retries = self.store._read(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND stage='scene_semantic_retry'",
            (memory_id,))[0][0]
        return {
            "extraction_scenes": len(packets),
            "extraction_success_scenes": sum(bool(packet.facts) and not packet.fallback for packet in packets),
            "extraction_fallback_scenes": sum(packet.fallback for packet in packets),
            "extraction_retry_calls": int(retries),
            "terminal_turn_coverage": len(terminal_turns) / source_turn_count,
            "semantic_terminal_turn_coverage": len(semantic_turns) / source_turn_count,
            "semantic_fact_turns": len(semantic_turns),
            "facts_per_scene_mean": (sum(facts_per_scene) / max(1, len(facts_per_scene))),
            "facts_per_scene_p95": (facts_per_scene[max(0, int(len(facts_per_scene) * 0.95) - 1)]
                                    if facts_per_scene else 0),
            "information_unit_coverage": {
                "total": unit_total,
                "covered_by_fact": unit_covered,
                "explicitly_unresolved": unit_unresolved,
                "missing": unit_missing,
                "implicitly_linked": unit_implicit,
                "accounted_ratio": ((unit_covered + unit_unresolved) / unit_total
                                    if unit_total else 1.0),
                "fact_coverage_ratio": (unit_covered / unit_total if unit_total else 1.0),
                "raw_fallback_turns": len(raw_fallback_turns),
            },
            "adaptive_fact_cap": {
                "mean": (sum(fact_caps) / len(fact_caps) if fact_caps else 0),
                "max": (max(fact_caps) if fact_caps else 0),
            },
            "collection_semantics": dict(sorted(collection_semantics.items())),
            "semantic_prompt_hash": (self.distiller.prompt_hash if self.distiller else None),
            "provenance_scope_nodes": dict(sorted(scopes.items())),
            "relation_counts": dict(sorted(relations.items())),
            "degree_p95": degree_values[max(0, int(len(degree_values) * 0.95) - 1)] if degree_values else 0,
            "cold_equivalent_stage_tokens": {stage: sum(int(usage.get("total_tokens", 0))
                for usage in keys.values()) for stage, keys in sorted(stage_usage.items())},
            # A degraded or truncated extraction must be visible in the manifest:
            # a silent degradation looks exactly like an unexplained accuracy drop.
            "build_token_budget": (dict(self.distiller.ledger.snapshot())
                                   if getattr(self.distiller, "ledger", None) else None),
            "method": dict(method_diagnostics or {}),
        }

    @staticmethod
    def _terms(text):
        return {token.casefold() for token in WORD_RE.findall(text) if len(token) > 2}

    @staticmethod
    def _merge_surface(text: str) -> str:
        """Fold the spelling variations that split one referent into many keys.

        Extraction runs per scene, so the same referent arrives as "the dance
        studio", "Dance Studio", "dance studios" and "dance studio's" from four
        different calls with no shared vocabulary.  Folding is deliberately
        shallow -- case, articles, possessives, a trailing plural -- because
        anything more aggressive merges referents that are genuinely distinct,
        and a key that merges too much routes nowhere.
        """
        words = re.findall(r"[\w'-]+", text.casefold())
        if words and words[0] in {"the", "a", "an", "my", "his", "her", "their", "our"}:
            words = words[1:]
        folded = []
        for word in words:
            word = word.removesuffix("'s").removesuffix("s'")
            if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
                word = word[:-1]
            if word:
                folded.append(word)
        return " ".join(folded)

    def _merge_aliases(self, scenes, packet_by_scene, profile):
        """Surface -> the sibling surfaces that name the same thing elsewhere.

        The graph audit measured 124 of 21,117 edges (0.59%) leaving their
        session, and of 1,305 entity names only 60 reached two sessions -- the
        eight widest of those being speaker names, which are constant within a
        memory and so discriminate nothing.  Nothing in the graph joined two
        sessions.  This is the second pass that gives it one: it reads the whole
        memory at once, which no single extraction call can do, and it costs no
        LLM call because every string it folds is already in the packets.

        The result is consumed by `_compile_routing`, not by traversal.  The
        routing channel keys postings on *query* terms, so the merge pays off by
        making any surface of a referent reach every child that mentions any
        other surface of it -- across sessions, which is the whole point.
        """
        merge = getattr(profile.coarsen, "entity_merge", False)
        if not merge:
            return {}
        sessions_by_surface: dict[str, set[str]] = defaultdict(set)
        forms_by_key: dict[str, set[str]] = defaultdict(set)
        speakers = {turn.speaker.casefold() for scene in scenes for turn in scene.turns}
        for scene in scenes:
            packet = packet_by_scene.get(scene.scene_id)
            if packet is None:
                continue
            surfaces = {fact.value for fact in packet.facts}
            surfaces |= {fact.owner for fact in packet.facts}
            for surface in surfaces:
                key = self._merge_surface(str(surface))
                if len(key) < profile.coarsen.entity_merge_min_chars or key in speakers:
                    continue
                sessions_by_surface[key].add(scene.session_id)
                forms_by_key[key].add(self._normal(str(surface)))
        total = len({scene.session_id for scene in scenes}) or 1
        ceiling = max(profile.coarsen.entity_merge_min_sessions,
                      int(total * profile.coarsen.entity_merge_max_session_share))
        aliases: dict[str, set[str]] = defaultdict(set)
        for key, sessions in sessions_by_surface.items():
            # Both ends are cut: a key inside one session joins nothing, and a
            # key spanning most of the memory routes nowhere.
            if not profile.coarsen.entity_merge_min_sessions <= len(sessions) <= ceiling:
                continue
            forms = {form for form in forms_by_key[key] if form}
            if len(forms) < 2:
                continue
            for form in forms:
                aliases[form] |= forms - {form}
        return {form: tuple(sorted(others)) for form, others in aliases.items() if others}

    @staticmethod
    def _names(text):
        return {name.casefold() for name in CAPITAL_RE.findall(text)
                if name.casefold() not in NON_ENTITY_NAMES}

    @staticmethod
    def _bounded_summary(text, words):
        return " ".join(text.split()[:words])
