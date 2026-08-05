from __future__ import annotations

import heapq
import itertools
import math
import re
import time
from collections import defaultdict
from dataclasses import replace
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from ..domain import (
    CandidateScore,
    EvidenceCertificate,
    NavigationResult,
    NodeType,
    ProofStep,
    QueryBudget,
    RelationType,
    SourceTurn,
    stable_id,
)
from ..runtime import SQLiteSnapshotRuntime
from ..storage import SQLiteGraphStore
from ..tokenization import TokenCounter, resolve_token_counter
from .algebra import evaluate as evaluate_algebra
from .bindings import bind_facts
from .certificate import evaluate_certificate
from .packer import build_proof_units, pack as pack_proof_units
from .query_ir import compile_query
from .scheduler import ScheduleResult, execute as schedule_relations
from .facts import (CHANNELS as FACT_CHANNELS, build_fact_reservoir,
                    select_active_facts, turn_group_index)
from .seeding import seed_operands


DEFAULT_BACKBONE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "that", "the", "their", "they", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
})
NEGATIVE_TERMS = frozenset({"not", "never", "no", "neither", "without", "didn't", "don't"})
TIME_TERMS = frozenset({
    "before", "after", "first", "last", "later", "earlier", "during", "until",
    "since", "when", "date", "day", "week", "month", "year", "long", "duration",
})


class NavigatorVariant(StrEnum):
    N0_LEGACY = "n0_legacy"
    N1_RAW_FUSION = "n1_raw_fusion"
    N2_PROVENANCE = "n2_provenance"
    N3_PRIORITY = "n3_priority"
    N4_CERTIFICATE = "n4_certificate"
    N5_SET_COVER = "n5_set_cover"


class HarnessProfile(StrEnum):
    H0_N5 = "h0"
    H1_TELEMETRY = "h1"
    H2_POSTINGS = "h2"
    H3_MULTI_ANCHOR = "h3"
    H4_SCHEDULER = "h4"
    H5_ALGEBRA = "h5"
    H6_PROOF_PACKING = "h6"
    # The feedback's H-series: H7 is proof-unit packing, H8 the safe-superset
    # reservoir, H9 the operator algebra.  They are added in dependency order,
    # which is why H8 lands before H7.
    H8_RESERVOIR = "h8"
    # PR4a: the reservoir principle applied to CanonicalFacts.  The proof funnel
    # showed the fact exists on 68.5% of questions but is reached on 21.0%.
    H9_FACT_RESERVOIR = "h9"


VARIANT_RANK = {variant: index for index, variant in enumerate(NavigatorVariant)}


DenseSearch = Callable[[str, str, int], Sequence[tuple[str, float]]]


def terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_RE.findall(text))


def content_terms(text: str) -> frozenset[str]:
    return frozenset(token for token in terms(text) if token not in STOPWORDS and len(token) > 1)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(terms(text)) * 1.3))


def question_slots(query: str) -> tuple[str, tuple[str, ...], bool]:
    lowered = query.casefold()
    negative = any(term in terms(lowered) for term in NEGATIVE_TERMS)
    if "how many" in lowered or lowered.startswith("count"):
        return "count", ("collection_scope", "member_1", "member_2"), negative
    if any(word in lowered for word in ("before", "after", "earlier", "later", "first", "last")):
        return "temporal_comparison", ("temporal_left", "temporal_right", "ordering"), negative
    if "how long" in lowered or "duration" in lowered:
        return "duration", ("temporal_start", "temporal_end", "duration"), negative
    if any(word in lowered for word in ("what are", "which", "list", "what do")):
        return "list", ("collection_scope", "member_1", "member_2"), negative
    if any(word in lowered for word in ("change", "became", "now", "currently", "replaced")):
        return "state_change", ("prior_state", "current_state"), negative
    if any(word in lowered for word in TIME_TERMS):
        return "temporal", ("event", "time"), negative
    return "fact", ("subject", "predicate", "object"), negative


class GraphNavigator:
    def __init__(
        self,
        store: SQLiteGraphStore,
        *,
        variant: NavigatorVariant | str = NavigatorVariant.N5_SET_COVER,
        dense_search: DenseSearch | None = None,
        harness_profile: HarnessProfile | str | None = None,
        token_counter: TokenCounter | None = None,
        fact_channels: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.runtime = SQLiteSnapshotRuntime(store)
        self.variant = NavigatorVariant(variant)
        self.dense_search = dense_search
        self.harness_profile = HarnessProfile(harness_profile) if harness_profile else None
        # Evidence budgets are only meaningful against the backbone's own
        # vocabulary.  The word-count estimate stays available as a labelled
        # fallback so a run without a local tokenizer still completes, but the
        # manifest then has to say the numbers are estimates.
        self.token_counter = token_counter or resolve_token_counter(DEFAULT_BACKBONE_MODEL)
        # Channel set for the fact reservoir, so F0-F5 can be ablated.
        self.fact_channels = tuple(fact_channels) if fact_channels is not None else FACT_CHANNELS
        self._turn_group_cache: dict[str, dict[str, tuple[str, ...]]] = {}

    def _turn_tokens(self, turn: SourceTurn) -> int:
        return self.token_counter.count(turn.raw_text)

    def _turn_groups(self, memory_id: str) -> dict[str, tuple[str, ...]]:
        if memory_id not in self._turn_group_cache:
            self._turn_group_cache = {memory_id: turn_group_index(self.store, memory_id)}
        return self._turn_group_cache[memory_id]

    def navigate(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        if self.harness_profile in {None, HarnessProfile.H0_N5, HarnessProfile.H1_TELEMETRY}:
            result = self._navigate_legacy(memory_id, query, budget)
            if self.harness_profile == HarnessProfile.H1_TELEMETRY:
                trace = dict(result.trace)
                trace["exhaustion_telemetry"] = {
                    "search_budget_exhausted": bool(result.budget_exhausted and result.visited_nodes >= budget.max_visited_nodes),
                    "node_cap_reached": result.visited_nodes >= budget.max_visited_nodes,
                    "edge_cap_reached": result.visited_edges >= budget.max_visited_edges,
                    "hop_cap_reached": False,
                    "frontier_truncated": result.frontier_peak >= budget.max_frontier,
                    "pack_turn_cap_reached": len(result.packed_turn_ids) >= budget.max_evidence_turns,
                    "pack_token_cap_reached": result.evidence_tokens >= budget.max_evidence_tokens,
                    "normal_candidate_drop": len(result.dropped_turn_ids) > 0,
                }
                return replace(result, trace=trace)
            return result
        return self._navigate_harness(memory_id, query, budget)

    def _navigate_legacy(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        started = time.perf_counter()
        stage_times: dict[str, float] = {}
        all_turns = list(self.store.turns(memory_id))
        by_id = {turn.turn_id: turn for turn in all_turns}
        query_terms = content_terms(query)

        tick = time.perf_counter()
        scores = self._raw_candidates(memory_id, query, query_terms, all_turns)
        stage_times["seed_fusion"] = (time.perf_counter() - tick) * 1000
        top_sessions = self._top_sessions(scores, by_id, limit=8)
        candidate_ids = set(scores)
        for turn in all_turns:
            if turn.session_id in top_sessions:
                candidate_ids.add(turn.turn_id)

        view = self.runtime.view(memory_id)
        node_relevance = self._node_relevance(view.nodes, query_terms)
        seeds = tuple(node_id for node_id, _ in sorted(
            node_relevance.items(), key=lambda item: (-item[1], item[0])
        )[: min(12, budget.max_frontier)])

        visited: list[str] = []
        proof: list[ProofStep] = []
        frontier_peak = len(seeds)
        budget_exhausted = False
        tick = time.perf_counter()
        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N3_PRIORITY]:
            visited, proof, frontier_peak, budget_exhausted = self._priority_expand(
                view, seeds, node_relevance, query, budget,
                certificate_guided=(
                    VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N4_CERTIFICATE]
                ),
            )
        elif VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N2_PROVENANCE]:
            visited = list(seeds[: budget.max_visited_nodes])
        stage_times["graph_read_view"] = (time.perf_counter() - tick) * 1000

        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N2_PROVENANCE]:
            tick = time.perf_counter()
            group_ids = view.evidence_group_ids_for_nodes(visited or seeds)
            for group_id in group_ids:
                group = self.store.evidence_group(group_id)
                if group:
                    candidate_ids.update(member.turn_id for member in group.members)
            stage_times["provenance_closure"] = (time.perf_counter() - tick) * 1000

        kind, required_slots, negative_required = question_slots(query)
        candidate_rows = self._candidate_rows(
            candidate_ids, by_id, scores, query_terms, visited, view, proof
        )
        tick = time.perf_counter()
        if VARIANT_RANK[self.variant] >= VARIANT_RANK[NavigatorVariant.N5_SET_COVER]:
            packed, dropped, coverage = self._set_cover(
                candidate_rows, by_id, kind, required_slots, budget
            )
        else:
            packed, dropped, coverage = self._rank_pack(candidate_rows, by_id, budget)
        stage_times["evidence_pack"] = (time.perf_counter() - tick) * 1000

        covered = tuple(slot for slot in required_slots if coverage.get(slot))
        if negative_required and not any(
            content_terms(by_id[turn_id].raw_text) & NEGATIVE_TERMS for turn_id in packed
        ):
            missing = tuple(slot for slot in required_slots if slot not in covered) + ("negative_scope",)
        else:
            missing = tuple(slot for slot in required_slots if slot not in covered)
        certificate = EvidenceCertificate(
            question_kind=kind,
            required_slots=required_slots + (("negative_scope",) if negative_required else ()),
            covered_slots=covered + (("negative_scope",) if negative_required and "negative_scope" not in missing else ()),
            missing_slots=missing,
            complete=not missing,
            iterations=min(budget.max_iterations, max(1, len(visited) // max(1, budget.max_frontier) + 1)),
            negative_scope_required=negative_required,
        )
        evidence_tokens = sum(estimate_tokens(by_id[item].raw_text) for item in packed)
        retrieved_sessions = tuple(dict.fromkeys(by_id[item].session_id for item in packed))
        stage_times["total"] = (time.perf_counter() - started) * 1000
        graph_id = stable_id(
            "graph-artifact", memory_id, self.store.graph_version(memory_id),
            self.store.graph_checksum(memory_id),
        )
        return NavigationResult(
            question_id=stable_id("question", memory_id, query),
            memory_id=memory_id,
            graph_artifact_id=graph_id,
            retrieved_session_ids=retrieved_sessions,
            retrieved_turn_ids=packed,
            proof=tuple(proof),
            visited_nodes=len(visited),
            visited_edges=len(proof),
            frontier_peak=frontier_peak,
            evidence_tokens=evidence_tokens,
            budget_exhausted=budget_exhausted or len(candidate_rows) > len(packed),
            trace={
                "variant": str(self.variant),
                "top_sessions": top_sessions,
                "candidate_count": len(candidate_rows),
                "semantic_navigation_excludes_provenance_edges": True,
            },
            seed_node_ids=seeds,
            visited_path_node_ids=tuple(visited),
            slot_coverage={key: tuple(value) for key, value in coverage.items()},
            certificate=certificate,
            candidate_scores=tuple(candidate_rows),
            packed_turn_ids=packed,
            dropped_turn_ids=dropped,
            stage_latency_ms=stage_times,
            graph_only_candidate_turn_ids=tuple(row.turn_id for row in candidate_rows
                                                if row.graph_score > 0 and not any((row.exact_score,
                                                   row.bm25_score, row.dense_score))),
            relation_trace=tuple(proof),
            first_hit_relations={row.turn_id: row.relation_contributions[0]
                                 for row in candidate_rows if row.relation_contributions},
        )

    def _navigate_harness(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
        """V5.5 query-only Graph Harness over an immutable V5.4 graph."""
        started = time.perf_counter(); stage_times: dict[str, float] = {}
        all_turns = list(self.store.turns(memory_id)); by_id = {row.turn_id: row for row in all_turns}
        view = self.runtime.view(memory_id)
        tick = time.perf_counter(); ir = compile_query(query, view)
        stage_times["query_compile"] = (time.perf_counter() - tick) * 1000
        profile = self.harness_profile
        use_postings = profile in {HarnessProfile.H2_POSTINGS, HarnessProfile.H3_MULTI_ANCHOR,
                                   HarnessProfile.H4_SCHEDULER, HarnessProfile.H5_ALGEBRA,
                                   HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                                   HarnessProfile.H9_FACT_RESERVOIR}
        use_rrf = profile in {HarnessProfile.H3_MULTI_ANCHOR, HarnessProfile.H4_SCHEDULER,
                              HarnessProfile.H5_ALGEBRA, HarnessProfile.H6_PROOF_PACKING,
                              HarnessProfile.H8_RESERVOIR, HarnessProfile.H9_FACT_RESERVOIR}
        # H7 is the first profile with the wide reservoir; H2-H6 keep the narrow
        # V5.5 seeding so the ablation ladder stays interpretable.
        wide_reservoir = profile in {HarnessProfile.H8_RESERVOIR,
                                     HarnessProfile.H9_FACT_RESERVOIR}
        fact_reservoir_enabled = profile is HarnessProfile.H9_FACT_RESERVOIR
        tick = time.perf_counter()
        seeded = seed_operands(self.store, view, memory_id, ir, all_turns, dense_search=self.dense_search,
                               use_rrf=use_rrf, use_postings=use_postings,
                               reservoir_limit=budget.max_candidate_reservoir,
                               max_views_per_operand=budget.max_query_views_per_operand,
                               node_budget=budget.max_visited_nodes,
                               wide_reservoir=wide_reservoir)
        stage_times["seed_fusion"] = (time.perf_counter() - tick) * 1000
        route_closure = (self._route_terminal_closure(view, seeded.semantic_node_ids)
                         if profile in {HarnessProfile.H2_POSTINGS, HarnessProfile.H3_MULTI_ANCHOR} else ())
        semantic_seeds = tuple(dict.fromkeys((*seeded.semantic_node_ids, *route_closure)))[:budget.max_visited_nodes]
        direct_owner_terminals = self._owner_terminal_postings(
            view, semantic_seeds, seeded.raw_scores, query
        ) if profile in {HarnessProfile.H4_SCHEDULER, HarnessProfile.H5_ALGEBRA,
                         HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                         HarnessProfile.H9_FACT_RESERVOIR} else ()
        owners = {
            operand.operand_id: set().union(*(set(view.owner_alias_index.get(alias.casefold(), ()))
                                               for alias in operand.owner_aliases))
            for operand in ir.operands
        }
        initial_bindings = bind_facts(view, owners, ir.operands, semantic_seeds)
        initial_closure = evaluate_algebra(ir.operator, initial_bindings, (item.operand_id for item in ir.operands),
                                           distinct_by=ir.distinct_by, collection_complete=False)
        initial_certificate = evaluate_certificate(ir, initial_closure, no_progress=not initial_bindings)
        tick = time.perf_counter()
        if profile in {HarnessProfile.H5_ALGEBRA, HarnessProfile.H6_PROOF_PACKING,
                       HarnessProfile.H8_RESERVOIR,
                       HarnessProfile.H9_FACT_RESERVOIR} and initial_certificate.complete:
            schedule = ScheduleResult(semantic_seeds, (), {}, {
                "node_cap_reached": False, "edge_cap_reached": False,
                "hop_cap_reached": False, "frontier_truncated": False,
            })
        else:
            schedule = schedule_relations(view, ir, semantic_seeds, budget,
                                          structured=profile in {HarnessProfile.H4_SCHEDULER,
                                                                 HarnessProfile.H5_ALGEBRA,
                                                                 HarnessProfile.H6_PROOF_PACKING,
                                                                 HarnessProfile.H8_RESERVOIR,
                                                                 HarnessProfile.H9_FACT_RESERVOIR})
        stage_times["graph_read_view"] = (time.perf_counter() - tick) * 1000
        fact_ids = set(semantic_seeds) | set(schedule.visited_node_ids) | set(direct_owner_terminals)
        fact_reservoir = None
        if fact_reservoir_enabled:
            # Facts the narrow path already reached are the precision core and are
            # never displaced; the reservoir only adds rescues, and only the
            # bounded active shortlist reaches binding.
            tick = time.perf_counter()
            core_facts = tuple(node_id for node_id in fact_ids if node_id in view.nodes
                               and view.nodes[node_id].node_type == NodeType.CANONICAL_FACT)
            fact_reservoir = select_active_facts(
                build_fact_reservoir(view, ir, seeded, self._turn_groups(memory_id),
                                     core_fact_ids=core_facts,
                                     channels=self.fact_channels),
                view, ir,
                per_operand=budget.max_active_facts_per_operand,
                total=budget.max_active_facts)
            fact_ids |= set(fact_reservoir.active)
            stage_times["fact_reservoir"] = (time.perf_counter() - tick) * 1000
        paths: dict[str, tuple[str, ...]] = {}
        for step in schedule.proof:
            paths.setdefault(step.dst_id, (step.edge_id,))
        bindings = bind_facts(view, owners, ir.operands, fact_ids, paths)
        collection_ops = {"union_distinct", "intersection_distinct", "group_by_owner", "count_distinct"}
        if str(ir.operator) in collection_ops and any(item.predicate_candidates for item in ir.operands):
            # Owner equality alone is insufficient proof for a list/count.  It
            # previously made every fact spoken by an owner mandatory and
            # crowded out lossless terminal rescue evidence.
            bindings = tuple(item for item in bindings if item.confidence >= 0.70)
        manifests = [view.nodes[item] for item in fact_ids if item in view.nodes and view.nodes[item].node_type in {
            NodeType.COLLECTION_SCOPE, NodeType.COLLECTION_MANIFEST
        }]
        collection_complete = str(ir.operator) not in collection_ops or all(
            any((not owners.get(operand.operand_id) or str(node.attributes.get("owner_id", "")) in owners[operand.operand_id])
                and (not operand.predicate_candidates or any(
                    content_terms(str(candidate)) & content_terms(str(node.attributes.get("predicate", "")))
                    for candidate in operand.predicate_candidates))
                for node in manifests)
            for operand in ir.operands
        )
        closure = evaluate_algebra(ir.operator, bindings, (item.operand_id for item in ir.operands),
                                   distinct_by=ir.distinct_by, collection_complete=collection_complete)
        no_progress = not bindings
        exhausted = any(schedule.exhaustion.values())
        certificate = evaluate_certificate(ir, closure, exhausted=exhausted, no_progress=no_progress)
        tick = time.perf_counter()
        group_turns: dict[str, tuple[str, ...]] = {}
        for group_id in view.terminal_groups_for_nodes(tuple(fact_ids)):
            group = self.store.evidence_group(group_id)
            if group:
                group_turns[group_id] = tuple(member.turn_id for member in group.members)
        units = build_proof_units(closure.bindings, group_turns)
        # Hydrate every terminal reached by the structured route, not only the
        # subset already accepted as an algebra result.  The latter is a proof
        # preference, while the former is the lossless CandidatePool contract.
        hydrated_turn_ids = tuple(dict.fromkeys(turn_id for turn_ids in group_turns.values() for turn_id in turn_ids))
        proof_turn_ids = tuple(dict.fromkeys(turn_id for unit in units for turn_id in unit.source_turn_ids))
        direct_owner_turn_ids = tuple(dict.fromkeys(
            turn_id for group_id in view.terminal_groups_for_nodes(direct_owner_terminals)
            for turn_id in group_turns.get(group_id, ())
        ))
        candidate_ids = tuple(dict.fromkeys((*seeded.source_turn_ids, *hydrated_turn_ids)))
        mandatory = set(proof_turn_ids)
        # A wider pool only helps if it is ranked at least as well as the narrow
        # one was.  The legacy scorer's lexical, session and adjacency terms are
        # what let it pick 16 useful turns out of 278; without them a 474-turn
        # pool ranks worse than the 45-turn pool it replaced.
        query_terms = content_terms(query)
        session_best: dict[str, float] = defaultdict(float)
        by_session_index = {(row.session_id, row.turn_index): row.turn_id for row in all_turns}
        base_score: dict[str, float] = {}
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn: continue
            channels = seeded.raw_scores.get(turn_id, {})
            value = (float(channels.get("exact", 0.0)) * 1.2 + float(channels.get("bm25", 0.0))
                     + float(channels.get("dense", 0.0)))
            base_score[turn_id] = value
            session_best[turn.session_id] = max(session_best[turn.session_id], value)
        adjacency: dict[str, float] = defaultdict(float)
        for turn_id, value in base_score.items():
            turn = by_id[turn_id]
            for distance in (1, 2):
                for index in (turn.turn_index - distance, turn.turn_index + distance):
                    neighbour = by_session_index.get((turn.session_id, index))
                    if neighbour:
                        adjacency[neighbour] = max(adjacency[neighbour], value * (0.35 / distance))
        rows: list[CandidateScore] = []
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn: continue
            channels = seeded.raw_scores.get(turn_id, {})
            exact = float(channels.get("exact", 0.0)); bm25 = float(channels.get("bm25", 0.0)); dense = float(channels.get("dense", 0.0))
            graph = 1.0 if turn_id in hydrated_turn_ids else 0.0
            text_terms = content_terms(turn.raw_text)
            slot_gain = min(1.0, len(query_terms & text_terms) / max(1, len(query_terms)))
            role_gain = (0.3 if text_terms & TIME_TERMS else 0.0) + (0.3 if text_terms & NEGATIVE_TERMS else 0.0)
            session_score = session_best.get(turn.session_id, 0.0)
            adjacency_score = adjacency.get(turn_id, 0.0)
            operand_ids = set(binding.operand_id for binding in closure.bindings
                                        if turn_id in set().union(*(set(group_turns.get(group_id, ()))
                                                                    for group_id in binding.evidence_group_ids)))
            if turn_id in direct_owner_turn_ids:
                operand_ids.update(item.operand_id for item in ir.operands if item.owner_aliases)
            operand_ids = tuple(sorted(operand_ids))
            bscore = max((binding.confidence for binding in closure.bindings if binding.operand_id in operand_ids), default=0.0)
            # The richer lexical/session/adjacency terms exist because a 545-turn
            # pool needs them to rank as well as the legacy scorer ranked 278.
            # They stay off for H2-H6 so those rungs keep their frozen V5.5 values.
            fused = ((1.2 * exact + bm25 + dense + 0.8 * graph + 0.7 * bscore
                      + 0.4 * len(operand_ids) + 0.25 * role_gain + 0.5 * slot_gain
                      + 0.12 * session_score + adjacency_score) if wide_reservoir else
                     (exact + bm25 + dense + 0.8 * graph + 0.7 * bscore + 0.4 * len(operand_ids)))
            rows.append(CandidateScore(turn_id, turn.session_id, exact, bm25, dense, graph,
                                       role_gain, slot_gain,
                                       self._turn_tokens(turn), fused,
                                       tuple(name for name, value in (("exact", exact), ("bm25", bm25), ("dense", dense), ("graph", graph)) if value),
                                       session_score, adjacency_score,
                                       operand_ids=operand_ids, binding_score=bscore, obligation_gain=len(operand_ids),
                                       mandatory=turn_id in mandatory,
                                       proof_unit_ids=tuple(unit.unit_id for unit in units if turn_id in unit.source_turn_ids)))
        rows.sort(key=lambda row: (-row.mandatory, -row.fused_score, row.turn_id))
        if profile in {HarnessProfile.H6_PROOF_PACKING, HarnessProfile.H8_RESERVOIR,
                       HarnessProfile.H9_FACT_RESERVOIR}:
            packed, dropped, pack_exhaustion = pack_proof_units(units, rows, by_id,
                                                                 max_turns=budget.max_evidence_turns,
                                                                 max_tokens=budget.max_evidence_tokens,
                                                                 token_cost=self._turn_tokens)
        else:
            packed, dropped, _ = self._rank_pack(rows, by_id, budget)
            pack_exhaustion = {"turn_cap_reached": len(packed) >= budget.max_evidence_turns,
                               "token_cap_reached": sum(self._turn_tokens(by_id[item]) for item in packed) >= budget.max_evidence_tokens}
        stage_times["evidence_pack"] = (time.perf_counter() - tick) * 1000
        stage_times["total"] = (time.perf_counter() - started) * 1000
        graph_id = stable_id("graph-artifact", memory_id, self.store.graph_version(memory_id), self.store.graph_checksum(memory_id))
        evidence_tokens = sum(self._turn_tokens(by_id[item]) for item in packed)
        operand_coverage = {
            operand.operand_id: {
                "seed_nodes": seeded.operand_nodes.get(operand.operand_id, ()),
                "seed_turns": seeded.operand_turns.get(operand.operand_id, ()),
                "bindings": tuple(row.binding_id for row in bindings if row.operand_id == operand.operand_id),
                "packed_turns": tuple(item for item in packed if item in mandatory),
            } for operand in ir.operands
        }
        return NavigationResult(
            question_id=stable_id("question", memory_id, query), memory_id=memory_id, graph_artifact_id=graph_id,
            retrieved_session_ids=tuple(dict.fromkeys(by_id[item].session_id for item in packed)),
            retrieved_turn_ids=packed, proof=schedule.proof, visited_nodes=len(schedule.visited_node_ids),
            visited_edges=len(schedule.proof), frontier_peak=min(budget.max_frontier, len(seeded.semantic_node_ids)),
            evidence_tokens=evidence_tokens, budget_exhausted=any(schedule.exhaustion.values()) or any(pack_exhaustion.values()),
            trace={"variant": str(profile), "query_operator": str(ir.operator), "candidate_count": len(rows),
                   "semantic_navigation_excludes_provenance_edges": True, "relation_counts": schedule.relation_counts,
                   # The legacy H0/H1 path keeps the historical word-count estimate so it stays
                   # a faithful frozen baseline; the harness budgets against real tokens.
                   "token_counter": self.token_counter.describe(),
                   "seeding": dict(seeded.stats),
                   # The AST is compiled but not executed yet.  Recording it here,
                   # together with where it disagrees with the legacy classifier,
                   # measures the compiler before anything depends on it.
                   "operator_ast": ir.describe_ast(),
                   "ast_operator": str(ir.ast_operator) if ir.ast_operator else "",
                   "ast_diverges": ir.ast_diverges,
                   "ast_operands": [item.operand_id for item in ir.ast_operands],
                   "ast_obligations": [f"{item.kind}:{item.operand_id or 'root'}"
                                       for item in ir.ast_obligations],
                   "parse_warnings": list(ir.parse_warnings),
                   "fact_reservoir": dict(fact_reservoir.stats) if fact_reservoir else {}},
            seed_node_ids=semantic_seeds, visited_path_node_ids=tuple(dict.fromkeys(
                (*schedule.visited_node_ids, *direct_owner_terminals))),
            slot_coverage={item.operand_id: tuple(row.binding_id for row in bindings if row.operand_id == item.operand_id)
                           for item in ir.operands}, certificate=certificate, candidate_scores=tuple(rows),
            packed_turn_ids=packed, dropped_turn_ids=dropped, stage_latency_ms=stage_times,
            graph_only_candidate_turn_ids=hydrated_turn_ids, relation_trace=schedule.proof,
            first_hit_relations={item.operand_id: (paths.get(item.fact_node_id, ("posting",))[0]) for item in bindings},
            search_exhaustion=schedule.exhaustion, pack_exhaustion=pack_exhaustion,
            operand_coverage=operand_coverage, proof_units=units, stop_reason=certificate.stop_reason,
            reservoir_fact_node_ids=tuple(row.fact_id for row in fact_reservoir.entries)
                                     if fact_reservoir else (),
            reached_fact_node_ids=tuple(sorted(
                node_id for node_id in fact_ids
                if node_id in view.nodes and view.nodes[node_id].node_type == NodeType.CANONICAL_FACT)),
            bound_fact_node_ids=tuple(dict.fromkeys(row.fact_node_id for row in bindings)),
            selected_fact_node_ids=tuple(dict.fromkeys(row.fact_node_id for row in closure.bindings)),
        )

    @staticmethod
    def _route_terminal_closure(view, seed_ids: Sequence[str], *, limit: int = 48) -> tuple[str, ...]:
        """Bounded route-card refinement before terminal provenance hydration."""
        queue = [(item, 0) for item in seed_ids]; seen: set[str] = set(); result: list[str] = []
        while queue and len(seen) < limit:
            node_id, depth = queue.pop(0)
            if node_id in seen or node_id not in view.nodes:
                continue
            seen.add(node_id); node = view.nodes[node_id]
            if node.attributes.get("provenance_scope", "terminal") == "terminal":
                result.append(node_id); continue
            if depth >= 3:
                continue
            for row in view.neighbors(node_id, (RelationType.REFINES_TO, RelationType.SCENE_CONTAINS),
                                      semantic_only=True):
                queue.append((row.next_node_id, depth + 1))
        return tuple(dict.fromkeys(result))

    @staticmethod
    def _owner_terminal_postings(view, seed_ids: Sequence[str], raw_scores, query: str,
                                 *, per_owner_limit: int = 256) -> tuple[str, ...]:
        """Lossless rescue for the matched speaker/owner, without session flooding."""
        query_terms = content_terms(query); result: list[str] = []
        for node_id in seed_ids:
            node = view.nodes.get(node_id)
            if not node or node.node_type != NodeType.CANONICAL_ENTITY:
                continue
            candidates = []
            for row in view.neighbors(node_id, (RelationType.HAS_FACT,), semantic_only=True):
                target = view.nodes.get(row.next_node_id)
                if not target or target.node_type != NodeType.EVIDENCE_GROUP_REF:
                    continue
                turn_id = str(target.attributes.get("turn_id", ""))
                score = sum(float(value) for value in raw_scores.get(turn_id, {}).values())
                score += len(query_terms & content_terms(target.summary)) / max(1, len(query_terms))
                candidates.append((score, target.node_id))
            result.extend(node_id for _, node_id in sorted(candidates, key=lambda item: (-item[0], item[1]))[:per_owner_limit])
        return tuple(dict.fromkeys(result))

    def _raw_candidates(
        self, memory_id: str, query: str, query_terms: frozenset[str], turns_: Sequence[SourceTurn]
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {"exact": 0.0, "bm25": 0.0, "dense": 0.0}
        )
        for turn in turns_:
            turn_terms = content_terms(turn.raw_text)
            overlap = len(query_terms & turn_terms)
            if overlap:
                scores[turn.turn_id]["exact"] = overlap / max(1, len(query_terms))
                if " ".join(query_terms) in turn.raw_text.casefold():
                    scores[turn.turn_id]["exact"] += 0.5
        for turn_id, score in self.store.search_turns(memory_id, query, limit=96):
            scores[turn_id]["bm25"] = max(0.0, score)
        if self.dense_search:
            for turn_id, score in self.dense_search(memory_id, query, 96):
                scores[turn_id]["dense"] = float(score)
        bm25_max = max((row["bm25"] for row in scores.values()), default=0.0)
        dense_values = [row["dense"] for row in scores.values() if row["dense"]]
        dense_min = min(dense_values, default=0.0)
        dense_max = max(dense_values, default=0.0)
        for row in scores.values():
            if bm25_max > 0:
                row["bm25"] /= bm25_max
            if row["dense"] and dense_max > dense_min:
                row["dense"] = (row["dense"] - dense_min) / (dense_max - dense_min)
        return scores

    @staticmethod
    def _top_sessions(scores: Mapping[str, Mapping[str, float]], by_id: Mapping[str, SourceTurn], limit: int) -> tuple[str, ...]:
        result: dict[str, float] = defaultdict(float)
        for turn_id, channels in scores.items():
            if turn_id not in by_id:
                continue
            fused = channels["exact"] * 1.4 + channels["bm25"] + channels["dense"]
            result[by_id[turn_id].session_id] = max(result[by_id[turn_id].session_id], fused)
        return tuple(key for key, _ in sorted(result.items(), key=lambda item: (-item[1], item[0]))[:limit])

    @staticmethod
    def _node_relevance(nodes: Mapping[str, object], query_terms: frozenset[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for node_id, node in nodes.items():
            node_terms = content_terms(getattr(node, "summary", ""))
            overlap = len(query_terms & node_terms)
            if overlap:
                result[node_id] = overlap / max(1, len(query_terms)) + float(getattr(node, "confidence", 1.0)) * 0.05
        return result

    def _priority_expand(self, view, seeds, relevance, query, budget, *, certificate_guided=False):
        kind, required, _ = question_slots(query)
        uncovered = set(required)
        queue: list[tuple[float, int, str, int, tuple[ProofStep, ...]]] = []
        sequence = itertools.count()
        for node_id in seeds:
            heapq.heappush(queue, (-relevance.get(node_id, 0.0), 0, node_id, next(sequence), ()))
        visited: list[str] = []
        visited_set: set[str] = set()
        used_edges: set[str] = set()
        proof: list[ProofStep] = []
        peak = len(queue)
        while queue and len(visited) < budget.max_visited_nodes and len(used_edges) < budget.max_visited_edges:
            _, hop, node_id, _, path = heapq.heappop(queue)
            if node_id in visited_set:
                continue
            visited_set.add(node_id)
            visited.append(node_id)
            if certificate_guided:
                uncovered -= set(view.role_bitset.get(node_id, ()))
            proof.extend(step for step in path if step.edge_id not in used_edges)
            used_edges.update(step.edge_id for step in path)
            if hop >= budget.max_hops:
                continue
            for row in view.neighbors(node_id, semantic_only=True):
                if row.next_node_id in visited_set:
                    continue
                roles = view.role_bitset.get(row.next_node_id, ())
                target_roles = uncovered if certificate_guided else set(required)
                role_gain = len(target_roles & set(roles)) / max(1, len(target_roles))
                relation_gain = 0.25 if row.edge.relation in {
                    RelationType.PORTAL, RelationType.TEMPORAL_BEFORE,
                    RelationType.TEMPORAL_AFTER, RelationType.STATE_TRANSITION,
                    RelationType.MEMBER_OF,
                } else 0.0
                priority = relevance.get(row.next_node_id, 0.0) + role_gain + relation_gain - 0.08 * (hop + 1)
                step = ProofStep(row.edge.edge_id, row.edge.src_id, row.edge.relation,
                                 row.edge.dst_id, row.edge.evidence_group_id)
                heapq.heappush(queue, (-priority, hop + 1, row.next_node_id,
                                       next(sequence), (*path, step)))
            if len(queue) > budget.max_frontier:
                queue = heapq.nsmallest(budget.max_frontier, queue)
                heapq.heapify(queue)
            peak = max(peak, len(queue))
        exhausted = bool(queue) or len(visited) >= budget.max_visited_nodes or len(used_edges) >= budget.max_visited_edges
        unique_proof = {step.edge_id: step for step in proof}
        return visited, list(unique_proof.values())[: budget.max_visited_edges], peak, exhausted

    def _candidate_rows(self, candidate_ids, by_id, raw_scores, query_terms, visited, view, proof=()):
        graph_turns: set[str] = set()
        path_by_turn: dict[str, list[str]] = defaultdict(list)
        relation_by_turn: dict[str, list[str]] = defaultdict(list)
        proof_by_node = defaultdict(list)
        for step in proof:
            proof_by_node[step.dst_id].append(step)
        for group_id in view.evidence_group_ids_for_nodes(visited):
            group = self.store.evidence_group(group_id)
            if group:
                graph_turns.update(member.turn_id for member in group.members)
        for node_id in visited:
            for group_id in view.evidence_group_ids_for_nodes((node_id,)):
                group = self.store.evidence_group(group_id)
                if not group:
                    continue
                for member in group.members:
                    for step in proof_by_node.get(node_id, ()):
                        path_by_turn[member.turn_id].append(step.edge_id)
                        relation_by_turn[member.turn_id].append(str(step.relation))
        base_by_turn = {
            turn_id: channels.get("exact", 0.0) * 1.2 + channels.get("bm25", 0.0)
            + channels.get("dense", 0.0)
            for turn_id, channels in raw_scores.items() if turn_id in by_id
        }
        session_max: dict[str, float] = defaultdict(float)
        by_session_index: dict[tuple[str, int], str] = {
            (turn.session_id, turn.turn_index): turn_id for turn_id, turn in by_id.items()
        }
        for turn_id, score in base_by_turn.items():
            turn = by_id[turn_id]
            session_max[turn.session_id] = max(session_max[turn.session_id], score)
        adjacency: dict[str, float] = defaultdict(float)
        for turn_id, score in base_by_turn.items():
            turn = by_id[turn_id]
            for distance in (1, 2):
                for index in (turn.turn_index - distance, turn.turn_index + distance):
                    neighbor = by_session_index.get((turn.session_id, index))
                    if neighbor:
                        adjacency[neighbor] = max(adjacency[neighbor], score * (0.35 / distance))
        rows: list[CandidateScore] = []
        for turn_id in candidate_ids:
            turn = by_id.get(turn_id)
            if not turn:
                continue
            channels = raw_scores.get(turn_id, {})
            exact = float(channels.get("exact", 0.0))
            bm25 = float(channels.get("bm25", 0.0))
            dense = float(channels.get("dense", 0.0))
            graph = 1.0 if turn_id in graph_turns else 0.0
            text_terms = content_terms(turn.raw_text)
            temporal_gain = 0.3 if text_terms & TIME_TERMS else 0.0
            negative_gain = 0.3 if text_terms & NEGATIVE_TERMS else 0.0
            role_gain = temporal_gain + negative_gain
            slot_gain = min(1.0, len(query_terms & text_terms) / max(1, len(query_terms)))
            session_score = session_max.get(turn.session_id, 0.0)
            adjacency_score = adjacency.get(turn_id, 0.0)
            fused = (1.2 * exact + bm25 + dense + 0.55 * graph + 0.25 * role_gain
                     + 0.5 * slot_gain + 0.12 * session_score + adjacency_score)
            channels_used = tuple(name for name, value in (
                ("exact", exact), ("bm25", bm25), ("dense", dense), ("graph", graph)
            ) if value > 0)
            rows.append(CandidateScore(
                turn_id, turn.session_id, exact, bm25, dense, graph,
                role_gain, slot_gain, estimate_tokens(turn.raw_text), fused, channels_used,
                session_score, adjacency_score,
                tuple(dict.fromkeys(path_by_turn.get(turn_id, ()))),
                tuple(dict.fromkeys(relation_by_turn.get(turn_id, ()))),
            ))
        return sorted(rows, key=lambda row: (-row.fused_score, row.turn_id))

    @staticmethod
    def _slot_matches(kind: str, required: Sequence[str], turn: SourceTurn) -> set[str]:
        lowered = turn.raw_text.casefold()
        tokens_ = content_terms(lowered)
        matched: set[str] = set()
        if kind in {"count", "list"}:
            matched.add("collection_scope")
            if any(char.isdigit() for char in lowered) or "," in lowered or " and " in lowered:
                matched.update(slot for slot in required if slot.startswith("member_"))
            else:
                matched.add("member_1")
        elif kind.startswith("temporal") or kind == "duration":
            if tokens_ & TIME_TERMS or re.search(r"\b\d{1,4}([:/-]\d{1,2})?\b", lowered):
                matched.update(required[:2])
            if any(word in lowered for word in ("before", "after", "since", "until", "later", "earlier")):
                matched.add(required[-1])
        elif kind == "state_change":
            if any(word in lowered for word in ("was", "used to", "before", "previous")):
                matched.add("prior_state")
            if any(word in lowered for word in ("now", "currently", "became", "is")):
                matched.add("current_state")
        else:
            matched.update(required)
        return matched

    def _set_cover(self, rows, by_id, kind, required, budget):
        uncovered = set(required)
        remaining = list(rows)
        packed: list[str] = []
        coverage: dict[str, list[str]] = defaultdict(list)
        tokens_used = 0
        slot_matches = {
            row.turn_id: self._slot_matches(kind, required, by_id[row.turn_id]) for row in rows
        }
        while remaining and len(packed) < budget.max_evidence_turns:
            ranked = []
            for row in remaining:
                matched = slot_matches[row.turn_id]
                gain = len(uncovered & matched)
                diversity = (0.9 if kind in {"count", "list"} else 0.2) if by_id[row.turn_id].session_id not in {
                    by_id[item].session_id for item in packed
                } else 0.0
                utility = row.fused_score + gain * 1.25 + diversity - 0.0005 * row.token_cost
                ranked.append((utility, row, matched))
            _, selected, matched = max(ranked, key=lambda item: (item[0], item[1].fused_score, item[1].turn_id))
            remaining = [row for row in remaining if row.turn_id != selected.turn_id]
            if tokens_used + selected.token_cost > budget.max_evidence_tokens:
                continue
            packed.append(selected.turn_id)
            tokens_used += selected.token_cost
            for slot in matched:
                coverage[slot].append(selected.turn_id)
            uncovered -= matched
            # A heuristic certificate is a routing hint, not permission to throw
            # away high-scoring source evidence. Continue filling the fixed turn/
            # token budget so false-positive closure cannot reduce recall.
        dropped = tuple(row.turn_id for row in rows if row.turn_id not in packed)
        return tuple(packed), dropped, coverage

    @staticmethod
    def _rank_pack(rows, by_id, budget):
        packed: list[str] = []
        tokens_used = 0
        for row in rows:
            if len(packed) >= budget.max_evidence_turns:
                break
            if tokens_used + row.token_cost > budget.max_evidence_tokens:
                continue
            packed.append(row.turn_id)
            tokens_used += row.token_cost
        dropped = tuple(row.turn_id for row in rows if row.turn_id not in packed)
        return tuple(packed), dropped, {}
