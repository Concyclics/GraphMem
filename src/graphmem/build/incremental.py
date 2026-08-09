"""Affected-path planning and copy-on-write routing-card publication."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping, Sequence

from ..domain import (
    EvidenceGroup, GraphEdge, GraphNode, NodeType, RelationType, Session,
    SourceTurn, stable_id,
)
from ..runtime.read_view import GraphReadView
from ..storage.sqlite import GraphDeltaResult, IncrementalJobRecord, SQLiteGraphStore
from ..text import content_terms


class IncrementalWriteState(StrEnum):
    RECEIVED = "received"
    RAW_DURABLE = "raw_durable"
    FACT_INDEXED = "fact_indexed"
    RELATION_INDEXED = "relation_indexed"
    ROUTE_PUBLISHED = "route_published"


@dataclass(frozen=True, slots=True)
class AffectedPathPlan:
    memory_id: str
    source_version: int
    changed_session_ids: tuple[str, ...]
    session_card_ids: tuple[str, ...]
    ancestor_card_ids: tuple[str, ...]
    branch_node_ids: tuple[str, ...]
    incident_edge_ids: tuple[str, ...]
    missing_session_ids: tuple[str, ...]
    total_nodes: int
    total_edges: int

    @property
    def recomputed_nodes(self) -> int:
        return len(set(self.session_card_ids) | set(self.ancestor_card_ids))

    @property
    def recompute_fraction(self) -> float:
        return self.recomputed_nodes / max(1, self.total_nodes)


@dataclass(frozen=True, slots=True)
class NewSessionInsertionPlan:
    """Bounded local insertion of one new level-1 session partition."""

    affected: AffectedPathPlan
    parent_card_id: str
    hierarchy_edge: GraphEdge
    previous_fanout: int
    needs_background_rebalance: bool


def plan_affected_paths(
    view: GraphReadView,
    *,
    memory_id: str,
    source_version: int,
    changed_session_ids: Sequence[str],
) -> AffectedPathPlan:
    """Resolve changed session leaves, descendants, ancestors and incident edges."""
    requested = tuple(dict.fromkeys(str(item) for item in changed_session_ids))
    by_session = {
        str(node.attributes.get("session_id")): node.node_id
        for node in view.nodes.values()
        if (node.node_type == NodeType.ROUTING_CARD
            and node.attributes.get("session_id") is not None)
    }
    leaves = tuple(by_session[item] for item in requested if item in by_session)
    missing = tuple(item for item in requested if item not in by_session)

    ancestors: set[str] = set()
    queue = deque(leaves)
    while queue:
        node_id = queue.popleft()
        for parent_id in view.hierarchy_parents(node_id):
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                queue.append(parent_id)

    branch: set[str] = set(leaves)
    queue = deque(leaves)
    while queue:
        node_id = queue.popleft()
        for child_id in view.hierarchy_children(node_id):
            if child_id not in branch:
                branch.add(child_id)
                queue.append(child_id)

    touched = branch | ancestors
    incident = tuple(sorted(
        edge.edge_id for edge in view.edges.values()
        if edge.src_id in touched or edge.dst_id in touched
    ))
    return AffectedPathPlan(
        memory_id=memory_id,
        source_version=source_version,
        changed_session_ids=requested,
        session_card_ids=tuple(sorted(leaves)),
        ancestor_card_ids=tuple(sorted(
            ancestors, key=lambda item: (view.nodes[item].level, item))),
        branch_node_ids=tuple(sorted(branch)),
        incident_edge_ids=incident,
        missing_session_ids=missing,
        total_nodes=len(view.nodes),
        total_edges=len(view.edges),
    )


def _route_terms(node: GraphNode) -> frozenset[str]:
    attrs = node.attributes
    fields = [node.summary]
    for key in ("owners", "predicates", "values", "scopes", "times"):
        value = attrs.get(key, ())
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            fields.append(str(value))
        else:
            fields.extend(map(str, value))
    return content_terms(" ".join(fields))


def recompile_route_ancestors(
    view: GraphReadView,
    plan: AffectedPathPlan,
    replacement_nodes: Sequence[GraphNode],
    *,
    summary_words: int = 320,
    posting_terms: int = 512,
    additional_children: Mapping[str, Sequence[str]] | None = None,
) -> tuple[GraphNode, ...]:
    """Rebuild only ancestors of replacement nodes, bottom-up.

    Node IDs and child membership stay stable, so readers can keep using the
    old immutable view while the new route summaries are assembled.
    """
    if any(node.memory_id != plan.memory_id for node in replacement_nodes):
        raise ValueError("replacement nodes must belong to the affected memory")
    replacement_by_id = {node.node_id: node for node in replacement_nodes}
    unknown = set(replacement_by_id) - set(view.nodes)
    extra = {
        str(parent): tuple(dict.fromkeys(map(str, children)))
        for parent, children in dict(additional_children or {}).items()
    }
    declared_new_children = {
        child_id for children in extra.values() for child_id in children
    }
    if unknown - declared_new_children:
        raise ValueError(
            "new branch nodes require explicit hierarchy edges before route recompilation: "
            + ", ".join(sorted(unknown - declared_new_children)))
    working = dict(view.nodes)
    working.update(replacement_by_id)
    rebuilt: list[GraphNode] = []
    # Parent levels increase towards the root.  Updating level 2 before level 3
    # ensures every parent reads already-recompiled children.
    for parent_id in sorted(
            plan.ancestor_card_ids,
            key=lambda item: (view.nodes[item].level, item)):
        parent = working[parent_id]
        child_ids = tuple(dict.fromkeys((
            *view.hierarchy_children(parent_id), *extra.get(parent_id, ()),
        )))
        children = [working[item] for item in child_ids if item in working]
        if not children:
            continue
        words = " ".join(child.summary for child in children).split()
        summary = " ".join(words[:summary_words])
        postings: dict[str, list[str]] = defaultdict(list)
        for child in children:
            for term in sorted(_route_terms(child)):
                postings[term].append(child.node_id)
        posting_rows = sorted(
            postings.items(), key=lambda row: (len(set(row[1])), row[0]))[:posting_terms]
        sessions = tuple(dict.fromkeys(
            session_id for child in children
            for session_id in (
                tuple(child.attributes.get("session_ids", ()))
                or ((str(child.attributes.get("session_id")),)
                    if child.attributes.get("session_id") else ()))
        ))
        attrs = dict(parent.attributes)
        attrs.update({
            "child_ids": tuple(child_ids),
            "session_ids": sessions,
            "child_postings": {
                key: tuple(dict.fromkeys(ids)) for key, ids in posting_rows
            },
            "provenance_ref_children": tuple(child_ids),
            "provenance_compact": True,
            "incremental_recompiled": True,
        })
        updated = replace(parent, summary=summary, attributes=attrs)
        working[parent_id] = updated
        rebuilt.append(updated)
    # Preserve caller order, then append ancestors not explicitly replaced.
    result = list(replacement_nodes)
    explicit = set(replacement_by_id)
    result.extend(node for node in rebuilt if node.node_id not in explicit)
    return tuple(result)


def publish_affected_path(
    store: SQLiteGraphStore,
    view: GraphReadView,
    plan: AffectedPathPlan,
    *,
    replacement_nodes: Sequence[GraphNode],
    upsert_edges: Sequence[GraphEdge] = (),
    upsert_evidence_groups: Sequence[EvidenceGroup] = (),
    delete_node_ids: Sequence[str] = (),
    delete_edge_ids: Sequence[str] = (),
    delete_evidence_group_ids: Sequence[str] = (),
    summary_words: int = 320,
    incremental_job_id: str | None = None,
    expected_job_state: IncrementalWriteState | str | None = None,
    next_job_state: IncrementalWriteState | str | None = None,
) -> GraphDeltaResult:
    """Compile changed route ancestors and commit the whole delta atomically."""
    if plan.missing_session_ids:
        raise ValueError(
            "new sessions need a local partition insertion before affected-path publish: "
            + ", ".join(plan.missing_session_ids))
    nodes = recompile_route_ancestors(
        view, plan, replacement_nodes, summary_words=summary_words)
    return store.apply_graph_delta(
        plan.memory_id,
        upsert_nodes=nodes,
        upsert_edges=upsert_edges,
        upsert_evidence_groups=upsert_evidence_groups,
        delete_node_ids=delete_node_ids,
        delete_edge_ids=delete_edge_ids,
        delete_evidence_group_ids=delete_evidence_group_ids,
        expected_version=plan.source_version,
        event_type="affected_path_publish",
        incremental_job_id=incremental_job_id,
        expected_job_state=(str(expected_job_state)
                            if expected_job_state is not None else None),
        next_job_state=(str(next_job_state) if next_job_state is not None else None),
    )


def plan_new_session_insertion(
    view: GraphReadView,
    *,
    memory_id: str,
    source_version: int,
    session_card: GraphNode,
    target_fanout: int = 8,
) -> NewSessionInsertionPlan:
    """Choose one semantic-nearest existing partition and its immediate parent.

    The foreground path remains O(number of lexical postings touched + tree
    height).  A full HNSW rebuild is deliberately not performed here.  If the
    selected parent exceeds ``target_fanout``, the insertion is still correct
    and visible, while ``needs_background_rebalance`` schedules split/merge on
    the asynchronous maintenance path.
    """
    if session_card.memory_id != memory_id:
        raise ValueError("new session card must belong to the target memory")
    if session_card.node_type != NodeType.ROUTING_CARD or session_card.level != 1:
        raise ValueError("new session insertion requires a level-1 routing card")
    if session_card.node_id in view.nodes:
        raise ValueError(f"session card {session_card.node_id!r} already exists")
    session_id = str(session_card.attributes.get("session_id", ""))
    if not session_id:
        raise ValueError("new session card must declare attributes.session_id")
    leaves = tuple(
        node for node in view.nodes.values()
        if node.node_type == NodeType.ROUTING_CARD and node.level == 1
    )
    if not leaves:
        raise ValueError("cannot incrementally attach a session to an empty hierarchy")
    terms = _route_terms(session_card)

    def similarity(node: GraphNode) -> tuple[float, str]:
        other = _route_terms(node)
        score = len(terms & other) / max(1, len(terms | other))
        return score, node.node_id

    nearest = max(leaves, key=similarity)
    parents = view.hierarchy_parents(nearest.node_id)
    if not parents:
        raise ValueError("existing session card has no routing parent")
    parent_id = min(parents, key=lambda item: (view.nodes[item].level, item))
    ancestors: set[str] = {parent_id}
    queue = deque((parent_id,))
    while queue:
        node_id = queue.popleft()
        for ancestor_id in view.hierarchy_parents(node_id):
            if ancestor_id not in ancestors:
                ancestors.add(ancestor_id)
                queue.append(ancestor_id)
    ordered_ancestors = tuple(sorted(
        ancestors, key=lambda item: (view.nodes[item].level, item)))
    group_ids = tuple(dict.fromkeys((
        view.nodes[parent_id].evidence_group_id,
        session_card.evidence_group_id,
    )))
    hierarchy_edge = GraphEdge(
        stable_id("edge", memory_id, parent_id, RelationType.REFINES_TO,
                  session_card.node_id),
        memory_id, parent_id, RelationType.REFINES_TO, session_card.node_id,
        group_ids[0], True, 1.0, "incremental_local_partition", group_ids[1:],
    )
    previous_fanout = len(view.hierarchy_children(parent_id))
    affected = AffectedPathPlan(
        memory_id=memory_id,
        source_version=source_version,
        changed_session_ids=(session_id,),
        session_card_ids=(session_card.node_id,),
        ancestor_card_ids=ordered_ancestors,
        branch_node_ids=(session_card.node_id,),
        incident_edge_ids=(),
        missing_session_ids=(),
        total_nodes=len(view.nodes),
        total_edges=len(view.edges),
    )
    return NewSessionInsertionPlan(
        affected=affected, parent_card_id=parent_id,
        hierarchy_edge=hierarchy_edge, previous_fanout=previous_fanout,
        needs_background_rebalance=previous_fanout + 1 > target_fanout,
    )


def publish_new_session_partition(
    store: SQLiteGraphStore,
    view: GraphReadView,
    plan: NewSessionInsertionPlan,
    *,
    session_card: GraphNode,
    upsert_nodes: Sequence[GraphNode] = (),
    upsert_edges: Sequence[GraphEdge] = (),
    upsert_evidence_groups: Sequence[EvidenceGroup] = (),
    summary_words: int = 320,
    incremental_job_id: str | None = None,
    expected_job_state: IncrementalWriteState | str = IncrementalWriteState.RELATION_INDEXED,
    next_job_state: IncrementalWriteState | str = IncrementalWriteState.ROUTE_PUBLISHED,
) -> GraphDeltaResult:
    """Atomically attach a new session leaf, rebuild its route, and publish."""
    replacements = (session_card, *upsert_nodes)
    nodes = recompile_route_ancestors(
        view, plan.affected, replacements, summary_words=summary_words,
        additional_children={plan.parent_card_id: (session_card.node_id,)},
    )
    return store.apply_graph_delta(
        plan.affected.memory_id,
        upsert_nodes=nodes,
        upsert_edges=(plan.hierarchy_edge, *upsert_edges),
        upsert_evidence_groups=upsert_evidence_groups,
        expected_version=plan.affected.source_version,
        event_type="new_session_route_publish",
        incremental_job_id=incremental_job_id,
        expected_job_state=(str(expected_job_state)
                            if incremental_job_id is not None else None),
        next_job_state=(str(next_job_state)
                        if incremental_job_id is not None else None),
    )


class IncrementalWriter:
    """Small durable coordinator for asynchronous, idempotent stage workers."""

    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store

    def receive(
        self, *, job_id: str, session: Session, turns: Sequence[SourceTurn],
        source_offset: int, payload: Mapping[str, Any] | None = None,
    ) -> IncrementalJobRecord:
        return self.store.append_incremental_raw(
            job_id=job_id, session=session, turns=turns,
            source_offset=source_offset, payload=payload,
        )

    def pending(
        self, state: IncrementalWriteState, *, limit: int = 100,
    ) -> tuple[IncrementalJobRecord, ...]:
        return self.store.incremental_jobs(state=str(state), limit=limit)

    def publish_stage(
        self,
        job: IncrementalJobRecord,
        *,
        expected_state: IncrementalWriteState,
        next_state: IncrementalWriteState,
        upsert_nodes: Sequence[GraphNode] = (),
        upsert_edges: Sequence[GraphEdge] = (),
        upsert_evidence_groups: Sequence[EvidenceGroup] = (),
        delete_node_ids: Sequence[str] = (),
        delete_edge_ids: Sequence[str] = (),
        delete_evidence_group_ids: Sequence[str] = (),
        expected_version: int | None = None,
    ) -> GraphDeltaResult:
        """Commit graph rows and the job-state CAS as one SQLite transaction."""
        return self.store.apply_graph_delta(
            job.memory_id,
            upsert_nodes=upsert_nodes, upsert_edges=upsert_edges,
            upsert_evidence_groups=upsert_evidence_groups,
            delete_node_ids=delete_node_ids, delete_edge_ids=delete_edge_ids,
            delete_evidence_group_ids=delete_evidence_group_ids,
            expected_version=(job.expected_version if expected_version is None
                              else expected_version),
            event_type=f"incremental_{next_state}",
            incremental_job_id=job.job_id,
            expected_job_state=str(expected_state),
            next_job_state=str(next_state),
        )

    def record_failure(self, job_id: str, error: BaseException | str) -> None:
        self.store.mark_incremental_attempt(job_id, str(error))
