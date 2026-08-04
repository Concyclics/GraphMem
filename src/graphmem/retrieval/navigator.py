from __future__ import annotations

import heapq
import itertools
import math
import re
import time
from collections import defaultdict
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from ..domain import (
    CandidateScore,
    EvidenceCertificate,
    NavigationResult,
    ProofStep,
    QueryBudget,
    RelationType,
    SourceTurn,
    stable_id,
)
from ..runtime import SQLiteSnapshotRuntime
from ..storage import SQLiteGraphStore


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
    ) -> None:
        self.store = store
        self.runtime = SQLiteSnapshotRuntime(store)
        self.variant = NavigatorVariant(variant)
        self.dense_search = dense_search

    def navigate(self, memory_id: str, query: str, budget: QueryBudget) -> NavigationResult:
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
            candidate_ids, by_id, scores, query_terms, visited, view
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
        )

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

    def _candidate_rows(self, candidate_ids, by_id, raw_scores, query_terms, visited, view):
        graph_turns: set[str] = set()
        for group_id in view.evidence_group_ids_for_nodes(visited):
            group = self.store.evidence_group(group_id)
            if group:
                graph_turns.update(member.turn_id for member in group.members)
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
