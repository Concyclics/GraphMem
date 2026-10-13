"""GraphMem persistent Memory SDK and versioned graph contracts."""

__version__ = "5.81.1"

from .config import (
    CacheIdentity,
    GraphMemRuntimeConfig,
    GraphMemV5Config,
    RetrievalRuntimeConfig,
    ServingConfig,
    config_hash,
    load_config,
    load_runtime_config,
    runtime_config_hash,
)
from .domain import (
    Conversation,
    CollectionScope,
    CanonicalEntity,
    CanonicalFact,
    CanonicalValue,
    CandidateScore,
    EntityMention,
    FactMention,
    EvidenceCertificate,
    EvidenceGroup,
    EventSkeleton,
    GraphArtifactManifest,
    GraphEdge,
    GraphNode,
    NavigationResult,
    QueryBudget,
    RunManifest,
    SemanticExtractionManifest,
    RoutingCard,
    Scene,
    Session,
    StateHead,
    TemporalInterval,
    SourceTurn,
    stable_id,
)

__all__ = [
    "EvidenceGroup",
    "EvidenceCertificate",
    "CandidateScore",
    "CanonicalEntity",
    "CanonicalFact",
    "CanonicalValue",
    "CollectionScope",
    "CacheIdentity",
    "Conversation",
    "EntityMention",
    "FactMention",
    "EventSkeleton",
    "GraphArtifactManifest",
    "GraphEdge",
    "GraphMemV5Config",
    "GraphMemRuntimeConfig",
    "GraphNode",
    "NavigationResult",
    "QueryBudget",
    "RetrievalRuntimeConfig",
    "RunManifest",
    "SemanticExtractionManifest",
    "RoutingCard",
    "Scene",
    "Session",
    "SourceTurn",
    "ServingConfig",
    "StateHead",
    "TemporalInterval",
    "config_hash",
    "load_config",
    "load_runtime_config",
    "runtime_config_hash",
    "stable_id",
    "MemoryService",
    "ServiceConfig",
]


def __getattr__(name):
    # Domain-only users need not import model clients or the service platform.
    if name in {"MemoryService", "ServiceConfig"}:
        from .service import MemoryService, ServiceConfig
        return {"MemoryService": MemoryService, "ServiceConfig": ServiceConfig}[name]
    raise AttributeError(name)
