"""Serving primitives for multi-user GraphMem deployments."""

from .compiled_lifecycle import (
    CompiledSidecarMaintainer,
    sync_compiled_sidecars,
)
from .dense_lifecycle import DenseSidecarMaintainer, sync_dense_sidecars

from .process_pool import (
    AdmissionRejected,
    BoundedAdmissionController,
    ProcessShardedNavigator,
    RequestDeadlineExceeded,
    WorkerCacheSnapshot,
    WorkerSnapshot,
)
from .replication import (
    ReplicaCorruptionError,
    ReplicaStaleError,
    SQLiteSnapshotReplicator,
    SnapshotReplicaManifest,
)

__all__ = [
    "AdmissionRejected", "BoundedAdmissionController",
    "CompiledSidecarMaintainer",
    "DenseSidecarMaintainer",
    "ProcessShardedNavigator", "RequestDeadlineExceeded", "WorkerSnapshot",
    "WorkerCacheSnapshot",
    "ReplicaCorruptionError", "ReplicaStaleError", "SQLiteSnapshotReplicator",
    "SnapshotReplicaManifest",
    "sync_compiled_sidecars",
    "sync_dense_sidecars",
]
