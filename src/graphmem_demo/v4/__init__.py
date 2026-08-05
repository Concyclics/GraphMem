"""GraphMem V4: one role graph with topology-aware capability projections."""

from .build import build_capability_view, validate_capability_view
from .retrieval import answer_messages, build_query_ir, query_views, retrieve
from .schema import (
    GRAPHMEM_V4_SCHEMA,
    V4_BUILD_VERSION,
    V4_RETRIEVAL_VERSION,
    CapabilityViewV4,
    capability_view_from_dict,
)

__all__ = [
    "GRAPHMEM_V4_SCHEMA",
    "V4_BUILD_VERSION",
    "V4_RETRIEVAL_VERSION",
    "CapabilityViewV4",
    "answer_messages",
    "build_capability_view",
    "build_query_ir",
    "capability_view_from_dict",
    "query_views",
    "retrieve",
    "validate_capability_view",
]
