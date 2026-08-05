from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .config import GraphMemV5Config
from .domain import (
    Conversation,
    EvidenceGroup,
    GraphArtifactManifest,
    GraphEdge,
    GraphNode,
    NavigationResult,
    QueryBudget,
    RelationType,
    RunManifest,
    Session,
    SourceTurn,
)


class RawStore(Protocol):
    def ingest_conversation(
        self, conversation: Conversation, sessions: Sequence[Session],
        turns: Sequence[SourceTurn],
    ) -> int: ...
    def ingest_turns(self, turns: Sequence[SourceTurn]) -> int: ...
    def conversation(self, memory_id: str) -> Conversation | None: ...
    def sessions(self, memory_id: str) -> Sequence[Session]: ...
    def turns(self, memory_id: str) -> Sequence[SourceTurn]: ...
    def delete_turn(self, turn_id: str) -> None: ...
    def delete_session(self, session_id: str) -> None: ...
    def delete_memory(self, memory_id: str) -> None: ...


class GraphStore(Protocol):
    def replace_graph(
        self,
        memory_id: str,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        evidence_groups: Sequence[EvidenceGroup],
    ) -> int: ...
    def nodes(self, memory_id: str) -> Sequence[GraphNode]: ...
    def edges(self, memory_id: str) -> Sequence[GraphEdge]: ...
    def evidence_groups(self, memory_id: str) -> Sequence[EvidenceGroup]: ...
    def graph_version(self, memory_id: str) -> int: ...


class GraphProjector(Protocol):
    def project_pending(self, *, max_events: int | None = None) -> int: ...
    def rebuild(self, memory_id: str) -> str: ...


class GraphRuntime(Protocol):
    @property
    def mode(self) -> str: ...
    def nodes(self, memory_id: str, node_ids: Sequence[str]) -> Sequence[GraphNode]: ...
    def expand(
        self,
        memory_id: str,
        frontier: Sequence[str],
        relations: Sequence[RelationType],
        *,
        limit: int,
    ) -> Sequence[GraphEdge]: ...


class VectorIndex(Protocol):
    def upsert(self, rows: Iterable[tuple[str, Sequence[float]]]) -> None: ...
    def search(self, vector: Sequence[float], *, k: int) -> Sequence[tuple[str, float]]: ...


class BuildPipeline(Protocol):
    def build(self, memory_id: str, profile: GraphMemV5Config) -> GraphArtifactManifest: ...


class Navigator(Protocol):
    def navigate(
        self, memory_id: str, query: str, budget: QueryBudget
    ) -> NavigationResult: ...


class AblationRunner(Protocol):
    def run(self, matrix: Path, dataset: Path) -> RunManifest: ...
