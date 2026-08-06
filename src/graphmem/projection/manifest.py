"""Authoritative collection manifests, derived deterministically from facts.

The navigator asks ``collection_complete`` before it will let a count or a list
close, and that check reads ``NodeType.COLLECTION_MANIFEST``.  The frozen corpus
contains **zero** such nodes: ``pipeline.py`` only builds ``COLLECTION_SCOPE``,
and only when a chain has at least 2 rows *and* at least 2 distinct values.  So
every single-member collection is invisible, and the completeness check has
never had anything to read.

Aggregation questions are 40% of the development set and its worst category at
51.9%, so this is the missing datum behind the largest error class.

Rules here are question-independent, use no LLM and no gold, and depend only on
the fact attributes the build already wrote.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..domain import (
    GraphEdge, GraphNode, NodeType, RelationType, stable_id,
)
from .config import ProjectionConfig


#: Attributes that jointly identify one collection chain.  Mirrors the grouping
#: ``pipeline._lean_semantic_graph`` already uses, so a manifest lines up with
#: the COLLECTION_SCOPE node when one exists.
CHAIN_KEYS = ("owner_id", "predicate", "scope", "collection_key", "polarity", "modality")


@dataclass(frozen=True, slots=True)
class ManifestRow:
    manifest_id: str
    chain: tuple[str, ...]
    member_ids: tuple[str, ...]
    value_keys: tuple[str, ...]
    member_count: int
    distinct_values: int
    occurrence: bool


def _attribute(node: GraphNode, key: str) -> str:
    return str(node.attributes.get(key, "") or "")


def chain_key(node: GraphNode, config: ProjectionConfig | None = None) -> tuple[str, ...]:
    config = config or ProjectionConfig()
    values = []
    for key in CHAIN_KEYS:
        value = _attribute(node, key)
        if key == "predicate":
            value = "" if not config.chain_includes_predicate else config.normalize_predicate(value)
        if key == "scope" and not config.chain_includes_scope:
            value = ""
        values.append(value)
    return tuple(values)


def collect_chains(nodes: Iterable[GraphNode],
                   config: ProjectionConfig | None = None) -> dict[tuple[str, ...], list[GraphNode]]:
    """Group canonical facts into collection chains, deterministically ordered."""
    chains: dict[tuple[str, ...], list[GraphNode]] = defaultdict(list)
    for node in nodes:
        if node.node_type != NodeType.CANONICAL_FACT:
            continue
        if not _attribute(node, "owner_id") or not _attribute(node, "predicate"):
            continue
        chains[chain_key(node, config)].append(node)
    for rows in chains.values():
        # Session/turn order first so a manifest's member list reads in
        # conversation order; node_id breaks ties so the projection is stable.
        rows.sort(key=lambda row: (_attribute(row, "session_id"),
                                   int(row.attributes.get("turn_index", -1) or -1),
                                   row.node_id))
    return chains


def build_manifests(memory_id: str, nodes: Sequence[GraphNode],
                    config: ProjectionConfig) -> tuple[list[GraphNode], list[GraphEdge], list[ManifestRow]]:
    """Emit one COLLECTION_MANIFEST per chain, plus MEMBER_OF edges.

    ``manifest_min_members=1`` is the point of this module: a collection with a
    single member is still a collection, and "how many X" over one X is exactly
    the question the frozen build could not answer.
    """
    if not config.collection_manifest:
        return [], [], []
    manifest_nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    rows: list[ManifestRow] = []
    for chain, members in sorted(collect_chains(nodes, config).items()):
        if len(members) < config.manifest_min_members:
            continue
        owner_id, predicate, scope, collection_key, polarity, modality = chain
        value_keys = tuple(dict.fromkeys(_attribute(row, "value_key") for row in members))
        occurrence = any(_attribute(row, "event_instance_id") for row in members)
        evidence = tuple(dict.fromkeys(
            group for row in members for group in row.all_evidence_group_ids))
        if not evidence:
            continue
        manifest_id = stable_id("node", memory_id, "collection-manifest", *chain)
        summary = " ".join(filter(None, [
            owner_id, "not" if polarity == "negative" else "",
            modality if modality and modality != "asserted" else "",
            # collection_key carries the class name once the predicate is out of
            # the chain, and it is the field a question's noun actually matches.
            predicate or collection_key,
            ", ".join(str(row.attributes.get("value", "")) for row in members[:8]),
        ]))
        manifest_nodes.append(GraphNode(
            manifest_id, memory_id, NodeType.COLLECTION_MANIFEST, 1, summary,
            evidence[0], evidence[1:],
            attributes={
                "owner_id": owner_id, "predicate": predicate, "scope": scope,
                "collection_key": collection_key, "polarity": polarity, "modality": modality,
                "member_ids": tuple(row.node_id for row in members),
                "member_count": len(members),
                "distinct_values": len(value_keys),
                "value_keys": value_keys,
                # The manifest enumerates every fact the graph holds for this
                # chain, so within the graph it is closed by construction.  It
                # says nothing about facts extraction never produced.
                "closed": True,
                "collection_semantics": "event_instances" if occurrence else "distinct_values",
                "roles": ("collection_manifest", "route"),
                "provenance_scope": "route",
            }))
        for member in members:
            edges.append(GraphEdge(
                stable_id("edge", memory_id, "member-of", member.node_id, manifest_id),
                memory_id, member.node_id, RelationType.MEMBER_OF, manifest_id,
                member.evidence_group_id, True, 1.0, "projection_manifest",
                member.evidence_group_ids))
        rows.append(ManifestRow(manifest_id, chain, tuple(row.node_id for row in members),
                                value_keys, len(members), len(value_keys), occurrence))
    return manifest_nodes, edges, rows


def manifest_stats(rows: Sequence[ManifestRow]) -> Mapping[str, float | int]:
    if not rows:
        return {"manifests": 0}
    sizes = sorted(row.member_count for row in rows)
    return {
        "manifests": len(rows),
        "single_member_manifests": sum(1 for row in rows if row.member_count == 1),
        # The frozen build required >=2 members and >=2 distinct values, so this
        # is the share of collections it could not see at all.
        "invisible_to_frozen_build": sum(
            1 for row in rows
            if row.member_count < 2 or (row.distinct_values < 2 and not row.occurrence)),
        "members_mean": sum(sizes) / len(sizes),
        "members_max": sizes[-1],
        "occurrence_manifests": sum(1 for row in rows if row.occurrence),
    }
