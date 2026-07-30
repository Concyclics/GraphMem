from __future__ import annotations

from collections import defaultdict

from ..v36.dialogue_topology import infer_dialogue_topology, is_memory_source
from ..v36.schema import V36Index
from .schema import CapabilityViewV4


def _frame_capabilities(frame: object) -> set[str]:
    capabilities = {"fact"}
    kind = str(getattr(frame, "frame_kind", ""))
    if kind:
        capabilities.add(kind)
    if str(getattr(frame, "state_op", "none")) != "none":
        capabilities.add("state")
    quantity = getattr(frame, "quantity", None)
    if quantity is not None and getattr(quantity, "value", None) is not None:
        capabilities.update({"quantity", "collection"})
    temporal = getattr(frame, "temporal", None)
    if temporal is not None and any(
        getattr(temporal, name, None)
        for name in ("event_time", "start", "end")
    ):
        capabilities.add("temporal")
    if str(getattr(frame, "polarity", "positive")) == "negative":
        capabilities.add("negative")
    if str(getattr(frame, "lifecycle_status", "unknown")) in {
        "planned", "ongoing", "completed", "cancelled",
    }:
        capabilities.add("lifecycle")
    return capabilities


def build_capability_view(index: V36Index) -> CapabilityViewV4:
    """Derive V2-like state views and V3-like dialogue views without node copies."""
    topology = infer_dialogue_topology(index.turns)
    turns_by_speaker: dict[str, list[str]] = defaultdict(list)
    memory_sources: list[str] = []
    dialogue_context: list[str] = []
    for turn in index.turns:
        turns_by_speaker[turn.speaker_key].append(turn.node_id)
        if is_memory_source(turn, topology):
            memory_sources.append(turn.node_id)
        else:
            dialogue_context.append(turn.node_id)

    frames_by_capability: dict[str, list[str]] = defaultdict(list)
    for frame in index.frames:
        for capability in sorted(_frame_capabilities(frame)):
            frames_by_capability[capability].append(frame.frame_id)

    groups_by_kind: dict[str, list[str]] = defaultdict(list)
    for group in index.evidence_groups:
        groups_by_kind[group.group_kind].append(group.group_id)

    valid_turn_ids = {turn.node_id for turn in index.turns}
    sourced_frame_count = sum(
        bool(frame.source_turn_ids)
        and all(source_id in valid_turn_ids for source_id in frame.source_turn_ids)
        for frame in index.frames
    )
    source_coverage_complete = sourced_frame_count == len(index.frames)
    return CapabilityViewV4(
        topology_mode=topology.mode,
        speaker_keys=list(topology.speaker_keys),
        memory_source_turn_ids=memory_sources,
        dialogue_context_turn_ids=dialogue_context,
        frame_ids_by_capability=dict(frames_by_capability),
        group_ids_by_kind=dict(groups_by_kind),
        turn_ids_by_speaker=dict(turns_by_speaker),
        state_chain_ids=[chain.chain_id for chain in index.state_chains],
        source_coverage_complete=source_coverage_complete,
        diagnostics={
            "physical_fact_node_count": len(index.frames),
            "projected_frame_reference_count": sum(
                len(values) for values in frames_by_capability.values()
            ),
            "duplicated_physical_nodes": 0,
            "sourced_frame_count": sourced_frame_count,
            "explicit_listener_ratio": topology.explicit_listener_ratio,
        },
    )


def validate_capability_view(
    index: V36Index, view: CapabilityViewV4,
) -> list[str]:
    errors: list[str] = []
    turn_ids = {turn.node_id for turn in index.turns}
    frame_ids = {frame.frame_id for frame in index.frames}
    group_ids = {group.group_id for group in index.evidence_groups}
    chain_ids = {chain.chain_id for chain in index.state_chains}
    if set(view.memory_source_turn_ids) - turn_ids:
        errors.append("memory_source_turn_ids contain unknown turns")
    if set(view.dialogue_context_turn_ids) - turn_ids:
        errors.append("dialogue_context_turn_ids contain unknown turns")
    for capability, ids in view.frame_ids_by_capability.items():
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate frame reference in capability {capability}")
        if set(ids) - frame_ids:
            errors.append(f"unknown frame in capability {capability}")
    for kind, ids in view.group_ids_by_kind.items():
        if set(ids) - group_ids:
            errors.append(f"unknown group in kind {kind}")
    if set(view.state_chain_ids) - chain_ids:
        errors.append("unknown state chain")
    if not view.source_coverage_complete:
        errors.append("role-frame provenance is incomplete")
    return errors
