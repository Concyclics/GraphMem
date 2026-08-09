"""Near-linear recursive coarsening and parent-gated relation candidates."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..domain import GraphNode, NodeType, RelationType, canonical_json, stable_id
from ..text import content_terms, normalize_key
from .refine import RefineCandidate


@dataclass(frozen=True, slots=True)
class CoarsenStats:
    leaf_cards: int
    parent_cards: int
    levels: int
    cluster_candidate_comparisons: int
    max_fanout: int
    overflow_root_fanout: int
    assignment_method: str = "bounded_semantic_partition"
    vector_dimension: int = 0
    ann_queries: int = 0


@dataclass(frozen=True, slots=True)
class RecursiveHierarchy:
    parent_cards: tuple[GraphNode, ...]
    root: GraphNode
    children: Mapping[str, tuple[str, ...]]
    levels: Mapping[int, tuple[str, ...]]
    stats: CoarsenStats
    vectors: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GatedRelationPlan:
    accepted_pairs: tuple[tuple[str, str, float, int], ...]
    refine_candidates: tuple[RefineCandidate, ...]
    coarse_candidate_pairs: int
    gated_child_pairs: int
    score_comparisons: int
    levels_with_relations: int
    typed_pairs: tuple[tuple[str, str, RelationType, float, int, str], ...] = ()
    candidate_method: str = "bounded_sparse"
    refine_candidates_generated: int = 0
    refine_candidates_dropped: int = 0
    atomic_relation_candidates_generated: int = 0
    atomic_relation_pairs_proposed: int = 0


ATOMIC_RELATION_NODE_TYPES = frozenset({
    NodeType.CANONICAL_FACT, NodeType.EVENT_FRAME, NodeType.EVENT_SKELETON,
    NodeType.STATE_HEAD, NodeType.STATE_VALUE,
})

COARSE_NAVIGATION_NODE_TYPES = frozenset({
    NodeType.ROUTING_CARD, NodeType.SCENE,
})

def _relation_context(node: GraphNode) -> str:
    """Compact endpoint contract for the bounded relation refiner.

    A bare routing-card summary does not expose polarity, predicate or time, so
    the model cannot distinguish continuation from contradiction.  Keep only
    fields needed by the relation vocabulary; candidate count remains O(kN).
    """
    attrs = node.attributes
    return canonical_json({
        "type": str(node.node_type),
        "summary": node.summary,
        "owner": attrs.get("owner_id", attrs.get("owners", ())),
        "predicate": attrs.get("predicate", attrs.get("predicates", ())),
        "value": attrs.get("value", attrs.get("values", ())),
        "scope": attrs.get("scope", attrs.get("scopes", ())),
        "polarity": attrs.get("polarity", ""),
        "observation_time": attrs.get(
            "observed_at", attrs.get("observation_time_range", ())),
        "event_time": attrs.get(
            "time_interval", attrs.get("event_time_range", attrs.get("times", ()))),
        "session": attrs.get("session_id", attrs.get("session_ids", ())),
    })


def _relation_field(node: GraphNode, field: str):
    return node.attributes.get(field, node.attributes.get(f"{field}s", ""))


def _field_overlap(left: GraphNode, right: GraphNode, field: str) -> bool:
    return bool(content_terms(str(_relation_field(left, field)))
                & content_terms(str(_relation_field(right, field))))


def _field_containment(left: GraphNode, right: GraphNode, field: str) -> float:
    left_terms = content_terms(str(_relation_field(left, field)))
    right_terms = content_terms(str(_relation_field(right, field)))
    return len(left_terms & right_terms) / max(
        1, min(len(left_terms), len(right_terms)))


def _owner_terms(node: GraphNode) -> frozenset[str]:
    value = node.attributes.get("owner_id", node.attributes.get("owners", ()))
    values = ((value,) if isinstance(value, (str, bytes))
              else tuple(value or ()))
    return frozenset(normalize_key(str(item)) for item in values
                     if normalize_key(str(item)))


def _observation_start(node: GraphNode) -> str:
    for field in ("observed_at", "observation_time_range", "time_interval",
                  "event_time_range"):
        value = node.attributes.get(field)
        if isinstance(value, Mapping):
            start = value.get("start")
            if start:
                return str(start)
        elif isinstance(value, str) and value:
            return value
    return ""


def structurally_allowed_refined_relations(
    left: GraphNode, right: GraphNode,
) -> tuple[RelationType, ...]:
    """Relations that could pass the deterministic materialization contract.

    This is deliberately direction-agnostic.  It runs before an LLM call to
    remove endpoint pairs that every possible label would reject; directional
    time ordering is checked after the refiner chooses LR/RL.
    """
    if (left.node_type not in ATOMIC_RELATION_NODE_TYPES
            or right.node_type not in ATOMIC_RELATION_NODE_TYPES):
        return ()
    same_owner = bool(_owner_terms(left) & _owner_terms(right))
    allowed: list[RelationType] = []
    same_summary = (bool(normalize_key(left.summary))
                    and normalize_key(left.summary)
                    == normalize_key(right.summary))
    left_value = normalize_key(str(_relation_field(left, "value")))
    right_value = normalize_key(str(_relation_field(right, "value")))
    high_precision_coreference = (
        same_summary or (
            bool(left_value) and left_value == right_value
            and _field_containment(left, right, "predicate") >= 0.75))
    if same_owner and high_precision_coreference:
        allowed.append(RelationType.COREFERENCE)
    shared_proposition = (
        same_owner and _field_overlap(left, right, "predicate")
        and _field_overlap(left, right, "scope"))
    left_value = normalize_key(str(_relation_field(left, "value")))
    right_value = normalize_key(str(_relation_field(right, "value")))
    if (shared_proposition
            and _field_containment(left, right, "value") >= 0.5
            and (left.attributes.get("polarity")
                 != right.attributes.get("polarity")
                 or left_value != right_value)):
        allowed.append(RelationType.CONTRADICTION_UPDATE)
    if (shared_proposition
            and left.attributes.get("polarity")
            == right.attributes.get("polarity")
            and _field_containment(left, right, "value") >= 0.25):
        allowed.append(RelationType.TEMPORAL_CONTINUATION)
    # The complete directional audit measured temporal/update/causal at
    # 80%/0%/50% precision, below the online-edge gate.  Keep their structural
    # logic above as the contract for a future second-stage verifier, but only
    # coreference may enter the current one-stage LLM path.  Deterministic
    # AT_TIME/TEMPORAL_BEFORE/STATE_NEXT edges continue to carry time queries.
    return tuple(relation for relation in allowed
                 if relation == RelationType.COREFERENCE)


def admit_llm_refined_relation(
    relation: RelationType,
    left: GraphNode,
    right: GraphNode,
    confidence: float,
    *,
    min_confidence: float,
) -> bool:
    """Precision gate for materializing bounded LLM relation decisions.

    Self-reported confidence was not calibrated: a five-memory audit measured
    66% typed precision, including invalid edges at 0.98 confidence.  Structural
    agreement is therefore required in addition to the model score.  Relations
    without a validated discriminant stay deferred instead of polluting the
    navigation graph.
    """
    if confidence < min_confidence:
        return False
    if (left.node_type not in ATOMIC_RELATION_NODE_TYPES
            or right.node_type not in ATOMIC_RELATION_NODE_TYPES):
        return False

    if relation not in structurally_allowed_refined_relations(left, right):
        return False
    if relation == RelationType.COREFERENCE and confidence < 0.88:
        return False
    if relation == RelationType.TEMPORAL_CONTINUATION:
        left_time = _observation_start(left)
        right_time = _observation_start(right)
        return not (left_time and right_time and left_time > right_time)
    if relation in {
            RelationType.COREFERENCE, RelationType.CONTRADICTION_UPDATE}:
        return True
    if relation == RelationType.CAUSAL:
        # The full cross-session audit reached only 50% causal type precision;
        # shared domains and an LLM confidence are not proof that one endpoint
        # causes the other.  Keep decisions in the deferred ledger until a
        # second-stage directional verifier is enabled.
        return False
    # SAME_ENTITY_STATE needs an explicit entity+state-domain key; shared topic
    # words produced only 54.7% precision.  COLLECTION_CO_MEMBER belongs to the
    # deterministic closed-manifest projection.  LLM COARSE_RELATED would lower
    # the HNSW high-threshold gate using an uncalibrated model score.
    return False


def _feature_vectors(nodes: Sequence[GraphNode], dimension: int = 256) -> dict[str, np.ndarray]:
    """Deterministic sparse semantic vectors used when model vectors are absent.

    This is a real vector/HNSW path, not an id-order fill.  Production callers
    can supply Qwen vectors; the feature fallback keeps tests, offline builds and
    recovery rebuilds deterministic without introducing a model dependency.
    """

    if dimension < 16:
        raise ValueError("HNSW feature dimension must be at least 16")
    result: dict[str, np.ndarray] = {}
    for node in nodes:
        vector = np.zeros(dimension, dtype=np.float32)
        terms = sorted(_terms(node))
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
            slot = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[slot] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        result[node.node_id] = vector
    return result


def _normalized_vectors(
    nodes: Sequence[GraphNode],
    vectors: Mapping[str, Sequence[float]] | None,
    *,
    dimension: int,
) -> dict[str, np.ndarray]:
    supplied_dimensions = {
        int(np.asarray(vectors[node.node_id]).size)
        for node in nodes if vectors and node.node_id in vectors
        and np.asarray(vectors[node.node_id]).ndim == 1
        and np.asarray(vectors[node.node_id]).size
    }
    if len(supplied_dimensions) > 1:
        raise ValueError(
            "all supplied HNSW vectors must use one embedding dimension; "
            f"received {sorted(supplied_dimensions)}")
    effective_dimension = (next(iter(supplied_dimensions))
                           if len(supplied_dimensions) == 1 else dimension)
    fallback = _feature_vectors(nodes, effective_dimension)
    result: dict[str, np.ndarray] = {}
    for node in nodes:
        raw = vectors.get(node.node_id) if vectors else None
        vector = np.asarray(raw, dtype=np.float32) if raw is not None else fallback[node.node_id]
        if (vector.ndim != 1 or not vector.size
                or vector.size != effective_dimension):
            vector = fallback[node.node_id]
        norm = float(np.linalg.norm(vector))
        result[node.node_id] = vector / max(norm, 1e-12)
    return result


def _hnsw_pairs(
    nodes: Sequence[GraphNode],
    *,
    vectors: Mapping[str, Sequence[float]] | None = None,
    per_node_k: int,
    max_candidates: int,
    dimension: int = 256,
    hnsw_m: int = 16,
    ef_construction: int = 100,
    cross_session_quota: int = 0,
) -> tuple[list[tuple[str, str, float]], int, dict[str, np.ndarray]]:
    """Return bounded cosine candidates from an actual HNSW index."""

    ordered = tuple(sorted(nodes, key=lambda row: row.node_id))
    if len(ordered) < 2:
        mapped = _normalized_vectors(ordered, vectors, dimension=dimension)
        return [], 0, mapped
    try:
        import hnswlib
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("assignment_method=hnsw requires hnswlib") from error
    mapped = _normalized_vectors(ordered, vectors, dimension=dimension)
    matrix = np.stack([mapped[node.node_id] for node in ordered]).astype(np.float32)
    index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    index.init_index(
        max_elements=len(ordered), ef_construction=max(16, ef_construction),
        M=max(4, hnsw_m), random_seed=42)
    labels = np.arange(len(ordered), dtype=np.int32)
    index.add_items(matrix, labels, num_threads=1)
    query_k = min(len(ordered), max(2, max_candidates + 1, per_node_k + 1))
    index.set_ef(max(32, query_k * 2))
    neighbours, distances = index.knn_query(matrix, k=query_k, num_threads=1)
    pairs: dict[tuple[str, str], float] = {}
    for source_index, (candidate_labels, candidate_distances) in enumerate(
            zip(neighbours, distances)):
        source = ordered[source_index]
        ranked: list[tuple[float, int]] = []
        for target_label, distance in zip(candidate_labels, candidate_distances):
            target_index = int(target_label)
            if target_index == source_index:
                continue
            ranked.append((max(-1.0, min(1.0, 1.0 - float(distance))),
                           target_index))
        ranked.sort(key=lambda row: (-row[0], ordered[row[1]].node_id))
        source_session = str(source.attributes.get("session_id", ""))
        cross = [row for row in ranked if source_session and str(
            ordered[row[1]].attributes.get("session_id", "")) not in {"", source_session}]
        selected = list(cross[:min(cross_session_quota, per_node_k)])
        selected_keys = {row[1] for row in selected}
        selected.extend(row for row in ranked if row[1] not in selected_keys)
        for score, target_index in selected[:per_node_k]:
            pair = tuple(sorted((source.node_id, ordered[target_index].node_id)))
            pairs[pair] = max(score, pairs.get(pair, -1.0))
    return ([(left, right, score) for (left, right), score in sorted(pairs.items())],
            len(ordered) * (query_k - 1), mapped)


def _hnsw_semantic_clusters(
    nodes: Sequence[GraphNode],
    *,
    fanout: int,
    max_candidates: int,
    vectors: Mapping[str, Sequence[float]] | None,
    dimension: int,
    hnsw_m: int,
    ef_construction: int,
) -> tuple[list[tuple[GraphNode, ...]], int, dict[str, np.ndarray], int]:
    """Balanced HNSW graph coarsening without arbitrary id-order fill."""

    ordered = tuple(sorted(nodes, key=lambda row: row.node_id))
    mapped = _normalized_vectors(ordered, vectors, dimension=dimension)
    clusters: list[tuple[GraphNode, ...]] = [(node,) for node in ordered]
    comparisons = 0
    queries = 0
    while len(clusters) > 1:
        active = [(index, cluster) for index, cluster in enumerate(clusters)
                  if len(cluster) < fanout]
        if len(active) < 2:
            break
        pseudo_nodes = tuple(GraphNode(
            node_id=min(row.node_id for row in cluster),
            memory_id=cluster[0].memory_id,
            node_type=cluster[0].node_type,
            level=cluster[0].level,
            summary=" ".join(row.summary for row in cluster),
            evidence_group_id=cluster[0].evidence_group_id,
            attributes={"session_id": "|".join(sorted(str(
                row.attributes.get("session_id", "")) for row in cluster))},
        ) for _index, cluster in active)
        pseudo_vectors = {
            pseudo.node_id: np.mean(
                [mapped[row.node_id] for row in cluster], axis=0)
            for pseudo, (_index, cluster) in zip(pseudo_nodes, active)
        }
        pairs, compared, _ = _hnsw_pairs(
            pseudo_nodes, vectors=pseudo_vectors,
            per_node_k=min(max_candidates, max(2, fanout * 2)),
            max_candidates=max_candidates, dimension=dimension,
            hnsw_m=hnsw_m, ef_construction=ef_construction)
        comparisons += compared
        queries += len(pseudo_nodes)
        position = {node.node_id: cluster_index
                    for node, (cluster_index, _cluster) in zip(pseudo_nodes, active)}
        parent = list(range(len(clusters)))
        sizes = [len(cluster) for cluster in clusters]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        merges = 0
        for left_id, right_id, score in sorted(
                pairs, key=lambda row: (-row[2], row[0], row[1])):
            left_root, right_root = find(position[left_id]), find(position[right_id])
            if left_root == right_root or sizes[left_root] + sizes[right_root] > fanout:
                continue
            # Deterministic union: the component with the lexically smaller
            # minimum node id is the representative; score only chooses edges.
            left_key = min(row.node_id for row in clusters[left_root])
            right_key = min(row.node_id for row in clusters[right_root])
            keep, drop = ((left_root, right_root) if left_key < right_key
                          else (right_root, left_root))
            parent[drop] = keep
            sizes[keep] += sizes[drop]
            merges += 1
        if not merges:
            # Remaining components are capacity-incompatible (for example
            # 4+4+2 with fanout 4).  They are valid sibling clusters; the next
            # hierarchy level coarsens their parent centroids.
            break
        grouped: dict[int, list[GraphNode]] = defaultdict(list)
        for index, cluster in enumerate(clusters):
            grouped[find(index)].extend(cluster)
        next_clusters = [tuple(sorted(rows, key=lambda row: row.node_id))
                         for rows in grouped.values()]
        next_clusters.sort(key=lambda rows: rows[0].node_id)
        if len(next_clusters) == len(clusters):
            raise RuntimeError("HNSW coarsening failed to reduce cluster count")
        clusters = next_clusters
        if all(len(cluster) >= fanout for cluster in clusters):
            break
    return clusters, comparisons, mapped, queries


def _terms(node: GraphNode) -> frozenset[str]:
    attrs = node.attributes
    def values(key: str) -> tuple[str, ...]:
        value = attrs.get(key, ())
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            return (str(value),)
        return tuple(map(str, value))

    surface = " ".join((
        node.summary,
        " ".join(values("owners")),
        " ".join(values("predicates")),
        " ".join(values("values")),
        " ".join(values("scopes")),
        " ".join(values("times")),
    ))
    return content_terms(surface)


def _similarity(left: GraphNode, right: GraphNode) -> float:
    a, b = _terms(left), _terms(right)
    if not a or not b:
        return 0.0
    # Overlap coefficient is less hostile to a short child summary than Jaccard,
    # while remaining bounded and deterministic.
    return len(a & b) / min(len(a), len(b))


def _semantic_clusters(nodes: Sequence[GraphNode], *, fanout: int,
                       max_candidates: int) -> tuple[list[tuple[GraphNode, ...]], int]:
    """Greedy balanced clustering with O(N * max_candidates) comparisons."""
    ordered = sorted(nodes, key=lambda row: row.node_id)
    by_id = {row.node_id: row for row in ordered}
    postings: dict[str, list[str]] = defaultdict(list)
    terms_by_id = {row.node_id: _terms(row) for row in ordered}
    for row in ordered:
        for term in terms_by_id[row.node_id]:
            postings[term].append(row.node_id)
    unassigned = set(by_id)
    fallback = deque(row.node_id for row in ordered)
    result: list[tuple[GraphNode, ...]] = []
    comparisons = 0
    for anchor in ordered:
        if anchor.node_id not in unassigned:
            continue
        candidates: set[str] = set()
        for term in sorted(terms_by_id[anchor.node_id]):
            for node_id in postings[term][:max_candidates]:
                if node_id != anchor.node_id and node_id in unassigned:
                    candidates.add(node_id)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        scored = []
        for node_id in candidates:
            comparisons += 1
            scored.append((_similarity(anchor, by_id[node_id]), node_id))
        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen = [anchor.node_id, *(node_id for _, node_id in scored[:fanout - 1])]
        chosen = list(dict.fromkeys(item for item in chosen if item in unassigned))
        while len(chosen) < fanout and unassigned - set(chosen):
            while fallback and fallback[0] not in unassigned:
                fallback.popleft()
            if not fallback:
                break
            candidate = fallback.popleft()
            if candidate not in chosen:
                chosen.append(candidate)
        for node_id in chosen:
            unassigned.discard(node_id)
        result.append(tuple(by_id[node_id] for node_id in chosen))
    return result, comparisons


def _parent_node(memory_id: str, level: int, children: Sequence[GraphNode], *,
                 summary_words: int, root: bool,
                 coarsen_method: str = "bounded_semantic_partition") -> GraphNode:
    child_ids = tuple(row.node_id for row in children)
    node_id = stable_id("node", memory_id, "recursive-card", level, child_ids)
    words = " ".join(row.summary for row in children).split()
    summary = " ".join(words[:summary_words])
    postings: dict[str, list[str]] = defaultdict(list)
    for child in children:
        for term in sorted(_terms(child)):
            postings[term].append(child.node_id)
    # Bound key count so routing metadata itself remains lightweight.  Terms
    # appearing in fewer children are more discriminating and kept first.
    posting_rows = sorted(postings.items(), key=lambda row: (len(set(row[1])), row[0]))[:512]
    sessions = tuple(dict.fromkeys(
        session_id for child in children
        for session_id in (
            tuple(child.attributes.get("session_ids", ()))
            or tuple(child.attributes.get("child_session_ids", ()))
            or ((str(child.attributes.get("session_id")),)
                if child.attributes.get("session_id") else ()))
    ))
    return GraphNode(
        node_id=node_id,
        memory_id=memory_id,
        node_type=NodeType.ROUTING_CARD,
        level=level,
        summary=summary,
        # Structural provenance is a pointer into the child tree, not a copy of
        # every descendant EvidenceGroup id.
        evidence_group_id=children[0].evidence_group_id,
        evidence_group_ids=(),
        attributes={
            "child_ids": child_ids,
            "session_ids": sessions,
            "roles": ("route", "memory") if root else ("route", "cross_session"),
            "provenance_scope": "route",
            "provenance_ref_children": child_ids,
            "provenance_compact": True,
            "coarsen_method": coarsen_method,
            "child_postings": {
                key: tuple(dict.fromkeys(ids)) for key, ids in posting_rows
            },
        },
    )


def build_recursive_hierarchy(
    memory_id: str,
    leaves: Sequence[GraphNode],
    *,
    fanout: int,
    max_levels: int,
    summary_words: int,
    max_candidates: int,
    assignment_method: str = "bounded_semantic_partition",
    vectors: Mapping[str, Sequence[float]] | None = None,
    hnsw_dimension: int = 256,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 100,
) -> RecursiveHierarchy:
    if not leaves:
        raise ValueError("recursive coarsening requires at least one leaf card")
    if fanout < 2:
        raise ValueError("recursive coarsening fanout must be at least 2")
    if max_levels < 2:
        raise ValueError("recursive coarsening needs at least two levels")
    current = tuple(sorted(leaves, key=lambda row: row.node_id))
    parents: list[GraphNode] = []
    children: dict[str, tuple[str, ...]] = {}
    levels: dict[int, list[str]] = defaultdict(list)
    for leaf in current:
        levels[leaf.level].append(leaf.node_id)
    comparisons = 0
    ann_queries = 0
    maximum_fanout = 0
    overflow = 0
    vector_map: dict[str, np.ndarray] = {}
    if assignment_method == "hnsw":
        vector_map.update(_normalized_vectors(
            current, vectors, dimension=hnsw_dimension))
    elif assignment_method != "bounded_semantic_partition":
        raise ValueError("assignment_method must be bounded_semantic_partition or hnsw")
    level = max(row.level for row in current) + 1
    while True:
        if len(current) == 1:
            # Even a single-session memory gets a distinct root snapshot node.
            clusters = [current]
        elif level >= max_levels:
            clusters = [current]
            overflow = max(0, len(current) - fanout)
        elif assignment_method == "hnsw":
            clusters, compared, current_vectors, queried = _hnsw_semantic_clusters(
                current, fanout=fanout, max_candidates=max_candidates,
                vectors=vector_map, dimension=hnsw_dimension,
                hnsw_m=hnsw_m, ef_construction=hnsw_ef_construction)
            comparisons += compared
            ann_queries += queried
            vector_map.update(current_vectors)
            if len(clusters) >= len(current):
                raise RuntimeError("HNSW hierarchy level did not coarsen")
        else:
            clusters, compared = _semantic_clusters(
                current, fanout=fanout, max_candidates=max_candidates)
            comparisons += compared
        next_level: list[GraphNode] = []
        is_final = len(clusters) == 1
        for cluster in clusters:
            if assignment_method == "hnsw" and len(cluster) == 1 and not is_final:
                # Carry an unmatched component upward unchanged.  Wrapping it
                # in a one-child parent adds depth and metadata but performs no
                # graph coarsening.
                next_level.append(cluster[0])
                continue
            parent = _parent_node(
                memory_id, level, cluster, summary_words=summary_words,
                root=is_final, coarsen_method=assignment_method)
            parents.append(parent)
            next_level.append(parent)
            children[parent.node_id] = tuple(row.node_id for row in cluster)
            levels[level].append(parent.node_id)
            maximum_fanout = max(maximum_fanout, len(cluster))
            if assignment_method == "hnsw":
                parent_vector = np.mean(
                    [vector_map[row.node_id] for row in cluster], axis=0)
                parent_vector /= max(float(np.linalg.norm(parent_vector)), 1e-12)
                vector_map[parent.node_id] = parent_vector.astype(np.float32)
        current = tuple(next_level)
        if len(current) == 1:
            break
        level += 1
    root = current[0]
    return RecursiveHierarchy(
        parent_cards=tuple(parents),
        root=root,
        children={key: tuple(value) for key, value in children.items()},
        levels={key: tuple(sorted(value)) for key, value in levels.items()},
        stats=CoarsenStats(
            leaf_cards=len(leaves), parent_cards=len(parents),
            levels=len(levels), cluster_candidate_comparisons=comparisons,
            max_fanout=maximum_fanout, overflow_root_fanout=overflow,
            assignment_method=assignment_method,
            vector_dimension=(len(next(iter(vector_map.values()))) if vector_map else 0),
            ann_queries=ann_queries,
        ),
        vectors={key: tuple(map(float, value)) for key, value in vector_map.items()},
    )


def _bounded_sparse_pairs(nodes: Sequence[GraphNode], *, max_candidates: int,
                          per_node_k: int) -> tuple[list[tuple[str, str, float]], int]:
    ordered = sorted(nodes, key=lambda row: row.node_id)
    by_id = {row.node_id: row for row in ordered}
    postings: dict[str, list[str]] = defaultdict(list)
    terms_by_id = {row.node_id: _terms(row) for row in ordered}
    for row in ordered:
        for term in terms_by_id[row.node_id]:
            postings[term].append(row.node_id)
    comparisons = 0
    pairs: dict[tuple[str, str], float] = {}
    for row in ordered:
        candidates: set[str] = set()
        for term in sorted(terms_by_id[row.node_id]):
            for candidate in postings[term][:max_candidates]:
                if candidate != row.node_id:
                    candidates.add(candidate)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        scored = []
        for candidate in candidates:
            comparisons += 1
            scored.append((_similarity(row, by_id[candidate]), candidate))
        for score, candidate in sorted(scored, key=lambda item: (-item[0], item[1]))[:per_node_k]:
            pair = tuple(sorted((row.node_id, candidate)))
            pairs[pair] = max(score, pairs.get(pair, 0.0))
    return [(left, right, score) for (left, right), score in sorted(pairs.items())], comparisons


def _attribute_phrases(node: GraphNode, *keys: str) -> frozenset[str]:
    values: list[str] = []
    for key in keys:
        value = node.attributes.get(key, ())
        if isinstance(value, (str, bytes)):
            values.append(str(value))
        elif value is not None:
            values.extend(map(str, value))
    return frozenset(normalized for item in values
                     if (normalized := normalize_key(item)))


def _node_sessions(node: GraphNode) -> frozenset[str]:
    values = node.attributes.get("session_ids", ())
    sessions = ({str(values)} if isinstance(values, (str, bytes)) else set(map(str, values)))
    if node.attributes.get("session_id"):
        sessions.add(str(node.attributes["session_id"]))
    return frozenset(item for item in sessions if item)


def classify_typed_relation(
    left: GraphNode,
    right: GraphNode,
    similarity: float,
) -> tuple[RelationType, float, str] | None:
    """High-precision, zero-token relation classifier for cross-session gates.

    It intentionally declines weak cases.  Ambiguous pairs remain eligible for
    the bounded LLM refiner instead of receiving a generic semantic label.
    """

    if (left.node_type not in ATOMIC_RELATION_NODE_TYPES
            or right.node_type not in ATOMIC_RELATION_NODE_TYPES):
        return None
    left_sessions, right_sessions = _node_sessions(left), _node_sessions(right)
    if (not left_sessions or not right_sessions
            or not left_sessions.isdisjoint(right_sessions)):
        return None
    generic_principals = frozenset({
        "user", "assistant", "speaker", "listener", "person", "people",
        "friend", "friends", "family", "they", "them", "i", "me", "you",
    })
    left_owner = (_attribute_phrases(left, "owners", "owner_id", "entities")
                  - generic_principals)
    right_owner = (_attribute_phrases(right, "owners", "owner_id", "entities")
                   - generic_principals)
    shared_owner = left_owner & right_owner
    base = max(0.0, min(1.0, similarity))

    # Dev-set judge results: zero-token state/temporal/causal materialization
    # reached only 0/2, 3/6 and 0/1 typed precision respectively.  Those labels
    # remain in the refine vocabulary below, but the deterministic fast path is
    # intentionally limited to the class that cleared the 85% precision gate.
    if (shared_owner and base >= 0.82
            and normalize_key(left.summary) == normalize_key(right.summary)):
        return (RelationType.COREFERENCE,
                min(0.94, 0.80 + 0.15 * base), "typed_shared_referent")
    return None


def build_parent_gated_relations(
    memory_id: str,
    hierarchy: RecursiveHierarchy,
    nodes: Mapping[str, GraphNode],
    child_map: Mapping[str, Sequence[str]],
    *,
    embedding_k: int,
    max_candidates_per_node: int,
    low_threshold: float,
    high_threshold: float,
    refine_mode: str,
    candidate_method: str = "bounded_sparse",
    vectors: Mapping[str, Sequence[float]] | None = None,
    hnsw_dimension: int = 256,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 100,
    cross_session_quota: int = 0,
    typed_restoration: bool = False,
    typed_min_confidence: float = 0.82,
    max_refine_candidates_per_node: int = 0,
    max_refine_candidates_per_1000_nodes: int = 0,
    atomic_vector_channels: Sequence[
        Mapping[str, Sequence[float]]] = (),
) -> GatedRelationPlan:
    """Generate fine candidates only below a surviving coarse candidate edge.

    Every parent is a bounded local candidate scope.  Cross-scope fine pairs
    are generated only when their parent pair survives the low threshold.  The
    sum of sibling scope sizes is linear in the number of hierarchy nodes, and
    each sparse search admits at most ``embedding_k`` neighbours per child.
    """
    if candidate_method not in {"bounded_sparse", "hnsw"}:
        raise ValueError("candidate_method must be bounded_sparse or hnsw")
    if max_refine_candidates_per_node < 0:
        raise ValueError("max_refine_candidates_per_node cannot be negative")
    if max_refine_candidates_per_1000_nodes < 0:
        raise ValueError(
            "max_refine_candidates_per_1000_nodes cannot be negative")
    all_nodes = tuple(nodes.values())
    ann_vectors = (_normalized_vectors(all_nodes, vectors, dimension=hnsw_dimension)
                   if candidate_method == "hnsw" else {})

    def similarity(left: GraphNode, right: GraphNode) -> float:
        if candidate_method != "hnsw":
            return _similarity(left, right)
        return max(-1.0, min(1.0, float(np.dot(
            ann_vectors[left.node_id], ann_vectors[right.node_id]))))

    seed_gates: dict[tuple[str, str], tuple[str, str, float]] = {}
    comparisons = 0
    coarse_candidates = 0
    # A relation can originate inside any coarse partition, not only between
    # the root's immediate children.  Seeding each bounded sibling scope avoids
    # missing within-cluster relations while retaining O(Nk) candidate work.
    for parent_id in sorted(child_map):
        siblings = [nodes[item] for item in child_map.get(parent_id, ())
                    if item in nodes]
        if len(siblings) < 2:
            continue
        if candidate_method == "hnsw":
            local_pairs, local_comparisons, _local_vectors = _hnsw_pairs(
                siblings, vectors=ann_vectors, per_node_k=embedding_k,
                max_candidates=max_candidates_per_node,
                dimension=hnsw_dimension, hnsw_m=hnsw_m,
                ef_construction=hnsw_ef_construction,
                cross_session_quota=cross_session_quota)
        else:
            local_pairs, local_comparisons = _bounded_sparse_pairs(
                siblings, max_candidates=max_candidates_per_node,
                per_node_k=embedding_k)
        comparisons += local_comparisons
        coarse_candidates += len(local_pairs)
        for left, right, score in local_pairs:
            if score < low_threshold:
                continue
            pair = (left, right)
            previous = seed_gates.get(pair)
            if previous is None or score > previous[2]:
                seed_gates[pair] = (left, right, score)
    gates = list(seed_gates.values())
    gated_children = 0
    accepted: dict[tuple[str, str], tuple[str, str, float, int]] = {}
    typed: dict[tuple[str, str, RelationType], tuple[
        str, str, RelationType, float, int, str]] = {}
    refine: dict[str, RefineCandidate] = {}
    relation_levels: set[int] = set()

    processed: dict[tuple[str, str], float] = {}
    while gates:
        next_gates: dict[tuple[str, str], tuple[str, str, float]] = {}
        for left_id, right_id, score in gates:
            gate_key = tuple(sorted((left_id, right_id)))
            if processed.get(gate_key, -1.0) >= score:
                continue
            processed[gate_key] = score
            left, right = nodes[left_id], nodes[right_id]
            level = max(left.level, right.level)
            relation_levels.add(level)
            pair = tuple(sorted((left_id, right_id)))
            atomic_pair = (
                left.node_type in ATOMIC_RELATION_NODE_TYPES
                and right.node_type in ATOMIC_RELATION_NODE_TYPES)
            # Once typed restoration owns the atomic layer, a high semantic
            # cosine is only a candidate signal, not a generic relation.  The
            # Qwen-vector smoke otherwise more than doubled coarse edges because
            # a threshold calibrated on hashed lexical vectors was applied in a
            # different vector space.  Only routing/scene regions receive coarse
            # edges; terminal facts/evidence must earn a typed relation or abstain.
            if score >= high_threshold and not (
                    typed_restoration and (
                        left.node_type not in COARSE_NAVIGATION_NODE_TYPES
                        or right.node_type not in COARSE_NAVIGATION_NODE_TYPES)):
                accepted[pair] = (pair[0], pair[1], score, level)
            typed_decision = (classify_typed_relation(left, right, score)
                              if typed_restoration else None)
            if typed_decision is not None:
                relation, confidence, source = typed_decision
                if confidence >= typed_min_confidence:
                    typed[(pair[0], pair[1], relation)] = (
                        pair[0], pair[1], relation, confidence, level, source)
            ambiguous = low_threshold <= score < high_threshold
            left_sessions = _node_sessions(left)
            right_sessions = _node_sessions(right)
            cross_session_pair = bool(
                left_sessions and right_sessions
                and left_sessions.isdisjoint(right_sessions))
            should_refine = (
                (refine_mode == "ambiguous_only" and ambiguous)
                or (refine_mode == "high_value_only" and ambiguous
                    and bool(child_map.get(left_id)) and bool(child_map.get(right_id)))
                or (refine_mode == "all_bounded_candidates" and score >= low_threshold)
                # Generic coarse-edge confidence and atomic relation confidence
                # answer different questions.  In particular, highly similar
                # facts from different sessions are often the best candidates
                # for coreference, continuation or update.  Do not discard them
                # merely because they sit above the coarse ambiguity band; the
                # endpoint-degree and global O(|V|) budgets below still bound
                # the token and materialized-edge cost.
                or (typed_restoration and atomic_pair and cross_session_pair
                    and score >= low_threshold)
            )
            structurally_allowed = (
                structurally_allowed_refined_relations(left, right)
                if atomic_pair and cross_session_pair else ())
            if (typed_restoration and atomic_pair and cross_session_pair
                    and not structurally_allowed):
                should_refine = False
            if should_refine:
                candidate_id = stable_id("candidate", memory_id, "coarse", *pair, level)
                estimated_pairs = (
                    len(child_map.get(left_id, ())) * len(child_map.get(right_id, ())))
                midpoint = (low_threshold + high_threshold) / 2
                radius = max((high_threshold - low_threshold) / 2, 1e-9)
                uncertainty = max(0.0, 1.0 - abs(score - midpoint) / radius)
                bridge_value = 1.0 + math.log1p(max(1, estimated_pairs))
                # Coarse cards are routing regions, not factual propositions.
                # Giving a card pair atomic labels produced thousands of
                # routing_card->routing_card "causal" and "update" edges whose
                # direction and semantics were unverifiable.  Typed restoration
                # is therefore an atomic-layer operation; higher levels may only
                # accept/reject a generic coarse gate.
                left_scene = str(left.attributes.get("scene_id", ""))
                right_scene = str(right.attributes.get("scene_id", ""))
                cross_scene_pair = bool(
                    left_scene and right_scene and left_scene != right_scene)
                allowed = ((*map(str, structurally_allowed), "NONE")
                           if atomic_pair and cross_session_pair
                           else (str(RelationType.COARSE_RELATED), "NONE"))
                refine[candidate_id] = RefineCandidate(
                    candidate_id, "coarse_edge", pair[0], pair[1],
                    _relation_context(nodes[pair[0]]),
                    _relation_context(nodes[pair[1]]),
                    allowed,
                    min(abs(score - low_threshold), abs(high_threshold - score)),
                    cross_scene_pair, cross_session_pair, cross_session_pair,
                    similarity=score, gate_level=level,
                    estimated_child_pairs=estimated_pairs,
                    priority=uncertainty * bridge_value / 448.0,
                )

            left_children = [nodes[item] for item in child_map.get(left_id, ()) if item in nodes]
            right_children = [nodes[item] for item in child_map.get(right_id, ()) if item in nodes]
            if not left_children or not right_children:
                continue
            # Cross-product is bounded by fanout in the normal case.  For an
            # overflow root, keep at most max_candidates_per_node candidates per
            # left child so the implementation stays near-linear.
            for left_child in left_children:
                scored_children = []
                for right_child in right_children:
                    comparisons += 1
                    child_score = similarity(left_child, right_child)
                    if child_score >= low_threshold:
                        scored_children.append((child_score, right_child.node_id))
                for child_score, right_child_id in sorted(
                        scored_children, key=lambda row: (-row[0], row[1]))[
                            :max_candidates_per_node]:
                    gated_children += 1
                    child_pair = tuple(sorted((left_child.node_id, right_child_id)))
                    previous = next_gates.get(child_pair)
                    if (processed.get(child_pair, -1.0) < child_score
                            and (previous is None or child_score > previous[2])):
                        next_gates[child_pair] = (
                            child_pair[0], child_pair[1], child_score)
        # A child may be proposed by several surviving parent gates.  Applying
        # ``k`` independently inside each parent pair still permits a dense
        # union, so cap the complete next-level frontier by endpoint degree.
        # This is the graph analogue of HNSW's neighbour-selection heuristic:
        # at most O(k|V_l|) gates survive at level l.
        next_degree: dict[str, int] = defaultdict(int)
        bounded_next: list[tuple[str, str, float]] = []
        for child_gate in sorted(
                next_gates.values(), key=lambda row: (-row[2], row[0], row[1])):
            left_id, right_id, _child_score = child_gate
            if (next_degree[left_id] >= embedding_k
                    or next_degree[right_id] >= embedding_k):
                continue
            bounded_next.append(child_gate)
            next_degree[left_id] += 1
            next_degree[right_id] += 1
        gates = bounded_next

    # Atomic relations need a proposition-level safety channel across coarse
    # partitions.  Keep it separate from the coarse graph: lexical and each
    # supplied atomic-summary vector channel independently propose a bounded
    # HNSW neighbourhood, but only cross-session atomic pairs enter the typed
    # refiner.  They can never materialize as generic COARSE_RELATED edges.
    atomic_relation_candidates_generated = 0
    atomic_relation_pairs_proposed = 0
    if typed_restoration and atomic_vector_channels:
        atomic_nodes = tuple(node for node in all_nodes
                             if node.node_type in ATOMIC_RELATION_NODE_TYPES)
        channel_pairs: dict[tuple[str, str], tuple[float, str]] = {}
        channels: tuple[tuple[str, Mapping[str, Sequence[float]] | None], ...] = (
            ("lexical", None),
            *((f"atomic_summary_{index}", channel)
              for index, channel in enumerate(atomic_vector_channels)),
        )
        for channel_name, channel_vectors in channels:
            pairs, channel_comparisons, _ = _hnsw_pairs(
                atomic_nodes, vectors=channel_vectors,
                per_node_k=embedding_k,
                max_candidates=max_candidates_per_node,
                dimension=hnsw_dimension, hnsw_m=hnsw_m,
                ef_construction=hnsw_ef_construction,
                cross_session_quota=max(1, cross_session_quota))
            comparisons += channel_comparisons
            for left_id, right_id, score in pairs:
                left = nodes[left_id]; right = nodes[right_id]
                left_sessions = _node_sessions(left)
                right_sessions = _node_sessions(right)
                if (not left_sessions or not right_sessions
                        or not left_sessions.isdisjoint(right_sessions)):
                    continue
                pair = tuple(sorted((left_id, right_id)))
                previous = channel_pairs.get(pair)
                if previous is None or score > previous[0]:
                    channel_pairs[pair] = (score, channel_name)
        atomic_relation_pairs_proposed = len(channel_pairs)
        for pair, (score, channel_name) in sorted(channel_pairs.items()):
            allowed_relations = structurally_allowed_refined_relations(
                nodes[pair[0]], nodes[pair[1]])
            if not allowed_relations:
                continue
            candidate_id = stable_id(
                "candidate", memory_id, "atomic_relation", *pair)
            refine[candidate_id] = RefineCandidate(
                candidate_id, "atomic_relation_edge", pair[0], pair[1],
                _relation_context(nodes[pair[0]]),
                _relation_context(nodes[pair[1]]),
                (*map(str, allowed_relations), "NONE"),
                0.0, True, True, True,
                similarity=score, gate_level=0,
                estimated_child_pairs=1,
                # Atomic relation restoration is the purpose of the refiner;
                # route-card ambiguity candidates must not consume its bounded
                # per-memory call budget first.
                priority=2.0 + max(-1.0, min(1.0, score)),
            )
        atomic_relation_candidates_generated = sum(
            candidate.kind == "atomic_relation_edge"
            for candidate in refine.values())

    # Generation over-samples the ambiguous band; admission then keeps the
    # highest expected information gain under endpoint and global budgets.
    # The latter is proportional to |V|, so LLM decision tokens are O(|V|).
    generated_refine = len(refine)
    ordered_refine = sorted(
        refine.values(), key=lambda row: (-row.priority, row.candidate_id))
    admitted_refine: list[RefineCandidate] = []
    refine_degree: dict[str, int] = defaultdict(int)
    total_refine_cap = 0
    if max_refine_candidates_per_1000_nodes:
        total_refine_cap = max(1, math.ceil(
            len(all_nodes) * max_refine_candidates_per_1000_nodes / 1000))
    for candidate in ordered_refine:
        if total_refine_cap and len(admitted_refine) >= total_refine_cap:
            break
        if (max_refine_candidates_per_node
                and (refine_degree[candidate.left_id]
                     >= max_refine_candidates_per_node
                     or refine_degree[candidate.right_id]
                     >= max_refine_candidates_per_node)):
            continue
        admitted_refine.append(candidate)
        refine_degree[candidate.left_id] += 1
        refine_degree[candidate.right_id] += 1

    return GatedRelationPlan(
        accepted_pairs=tuple(accepted[key] for key in sorted(accepted)),
        refine_candidates=tuple(admitted_refine),
        coarse_candidate_pairs=coarse_candidates,
        gated_child_pairs=gated_children,
        score_comparisons=comparisons,
        levels_with_relations=len(relation_levels),
        typed_pairs=tuple(typed[key] for key in sorted(
            typed, key=lambda row: (row[0], row[1], str(row[2])))),
        candidate_method=candidate_method,
        refine_candidates_generated=generated_refine,
        refine_candidates_dropped=generated_refine - len(admitted_refine),
        atomic_relation_candidates_generated=(
            atomic_relation_candidates_generated),
        atomic_relation_pairs_proposed=atomic_relation_pairs_proposed,
    )
