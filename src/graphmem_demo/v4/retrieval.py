from __future__ import annotations

from ..models import QuestionCase, RetrievedContext
from ..v36.retrieval import (
    answer_messages as v36_answer_messages,
    build_query_ir,
    query_views,
    retrieve as retrieve_v36,
)
from ..v36.schema import V36Index
from .capability_retrieval import supplement_capability_gaps
from .schema import CapabilityViewV4


def _requested_capabilities(query_ir: object) -> list[str]:
    capabilities = {"fact"}
    value_type = str(getattr(query_ir, "requested_value_type", "span"))
    if value_type in {"state"}:
        capabilities.update({"state", "lifecycle"})
    if value_type in {"count", "list", "aggregate"}:
        capabilities.update({"collection", "quantity", "lifecycle"})
    if value_type in {"date", "duration", "temporal_order"}:
        capabilities.add("temporal")
    if value_type in {"preference", "recommendation"}:
        capabilities.add("preference")
    roles = set(getattr(query_ir, "required_roles", []) or [])
    if roles.intersection({"question", "reply", "answer_content"}):
        capabilities.add("dialogue_answer")
    if roles.intersection({"old_state", "new_state", "change_operation"}):
        capabilities.update({"state", "lifecycle"})
    return sorted(capabilities)


def retrieve(
    *,
    case: QuestionCase,
    variant: str,
    index: V36Index,
    capability_view: CapabilityViewV4,
    query_vectors: list[list[float]],
    token_budget: int = 8800,
) -> RetrievedContext:
    """Run one generic navigator and expose the capability policy it selected."""
    query_ir = build_query_ir(case.question)
    requested = _requested_capabilities(query_ir)
    available = [
        name for name in requested
        if capability_view.frame_ids_by_capability.get(name)
    ]
    result = retrieve_v36(
        case=case,
        variant=variant,
        index=index,
        query_vectors=query_vectors,
        token_budget=token_budget,
    )
    supplements = supplement_capability_gaps(
        result=result, index=index, capability_view=capability_view,
        requested=requested, query_vectors=query_vectors,
        question=case.question, token_budget=token_budget,
    )
    result.schema_version = "graphmem_v4_0"
    result.retrieval_trace["v4_capability_policy"] = {
        "dialogue_topology": capability_view.topology_mode,
        "requested_capabilities": requested,
        "available_capabilities": available,
        "state_projection_enabled": bool(
            set(available).intersection({"state", "collection", "quantity", "lifecycle"})
        ),
        "dialogue_navigation_enabled": (
            capability_view.topology_mode == "peer_dialogue"
            or "dialogue_answer" in available
        ),
        "single_physical_role_graph": True,
        "source_coverage_complete": capability_view.source_coverage_complete,
        "supplement_count": len(supplements),
    }
    result.retrieval_trace["v4_capability_supplements"] = supplements
    result.retrieval_trace["v4_capability_counts"] = {
        name: len(ids)
        for name, ids in capability_view.frame_ids_by_capability.items()
    }
    result.retrieval_trace["v4_group_counts"] = {
        name: len(ids)
        for name, ids in capability_view.group_ids_by_kind.items()
    }
    return result


def answer_messages(
    case: QuestionCase,
    retrieval: RetrievedContext,
) -> list[dict[str, str]]:
    messages = v36_answer_messages(case, retrieval)
    policy = (
        "GraphMem V4 policy: use state/collection/temporal projections only as "
        "candidate organization; verify every value against cited lossless source "
        "turns. In peer dialogue, preserve speaker ownership and question/reply "
        "pairing. In assistant-mediated dialogue, distinguish user memory from "
        "assistant-provided results. Never infer from benchmark identity or topic."
    )
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": f"{messages[0].get('content', '')}\n\n{policy}",
        }
    else:
        messages.insert(0, {"role": "system", "content": policy})
    return messages


__all__ = ["answer_messages", "build_query_ir", "query_views", "retrieve"]
