"""GraphMem 5.0 contracts.

The V5 package intentionally coexists with ``graphmem_demo``.  Gate A exposes
stable domain/configuration interfaces and a read-only legacy adapter without
changing the V4/V4.1 execution path.
"""

from .config import CacheIdentity, GraphMemV5Config, config_hash, load_config
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
    "GraphNode",
    "NavigationResult",
    "QueryBudget",
    "RunManifest",
    "SemanticExtractionManifest",
    "RoutingCard",
    "Scene",
    "Session",
    "SourceTurn",
    "StateHead",
    "config_hash",
    "load_config",
    "stable_id",
]
