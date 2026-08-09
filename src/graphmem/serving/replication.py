"""Checksum-verified immutable SQLite snapshots for a local read follower."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from ..domain import canonical_json, logical_graph_checksum
from ..storage.sqlite import MIGRATION_VERSION, SQLiteGraphStore


class ReplicaCorruptionError(RuntimeError):
    """The immutable follower artifact does not match its durable manifest."""


class ReplicaStaleError(RuntimeError):
    """The follower is farther behind the authority than promotion permits."""


@dataclass(frozen=True, slots=True)
class SnapshotReplicaManifest:
    memory_id: str
    graph_version: int
    graph_checksum: str
    source_offset: int
    schema_version: int
    config_hash: str
    snapshot_file: str
    snapshot_bytes: int
    file_sha256: str
    created_unix_ms: int
    copy_latency_ms: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class SQLiteSnapshotReplicator:
    """Replicate the whole WAL authority into versioned, immutable artifacts.

    SQLite's online backup API reads one transactionally consistent image.  A
    completed image is checksum-verified before an atomic ``LATEST.json``
    pointer swap.  A crash can therefore leave an unreferenced file, but never
    a pointer to a half-written follower snapshot.
    """

    def __init__(
        self, source: SQLiteGraphStore, target_dir: str | Path, *,
        config_hash: str = "unknown",
    ) -> None:
        if source.read_only:
            raise ValueError("replication source must be the writable authority")
        self.source = source
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.latest_path = self.target_dir / "LATEST.json"

    def replicate(
        self,
        memory_id: str,
        *,
        max_snapshot_bytes: int | None = None,
        fail_after_copy: bool = False,
    ) -> SnapshotReplicaManifest:
        started = time.perf_counter()
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".graphmem-replica-", suffix=".sqlite", dir=self.target_dir)
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            destination = sqlite3.connect(temp_path)
            try:
                # The authority lock also serializes its shared connection; the
                # online backup still permits WAL readers in serving workers.
                with self.source._lock:
                    self.source._connection.backup(destination)
                destination.commit()
            finally:
                destination.close()
            size = temp_path.stat().st_size
            if max_snapshot_bytes is not None and size > max_snapshot_bytes:
                raise OSError(
                    f"snapshot requires {size} bytes, quota is {max_snapshot_bytes}")

            replica = SQLiteGraphStore(temp_path, read_only=True)
            try:
                version, checksum, nodes, edges = replica.graph_snapshot(memory_id)
                logical = logical_graph_checksum(nodes, edges)
                source_offset = replica.incremental_high_watermark(memory_id)
            finally:
                replica.close()
            if not version or not checksum or checksum != logical:
                raise ReplicaCorruptionError(
                    "replica graph version/checksum does not match canonical rows")
            file_digest = _file_sha256(temp_path)
            final_name = f"snapshot-v{version:08d}-{checksum[:12]}.sqlite"
            final_path = self.target_dir / final_name
            if final_path.exists():
                if _file_sha256(final_path) != file_digest:
                    raise ReplicaCorruptionError(
                        f"immutable snapshot name collision for {final_name}")
                temp_path.unlink()
            else:
                os.replace(temp_path, final_path)
            manifest = SnapshotReplicaManifest(
                memory_id=memory_id, graph_version=version,
                graph_checksum=checksum, source_offset=source_offset,
                schema_version=MIGRATION_VERSION, config_hash=self.config_hash,
                snapshot_file=final_name, snapshot_bytes=size,
                file_sha256=file_digest, created_unix_ms=int(time.time() * 1000),
                copy_latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            if fail_after_copy:
                raise RuntimeError("injected crash after immutable copy, before pointer swap")
            pointer_temp = self.target_dir / ".LATEST.json.tmp"
            pointer_temp.write_text(canonical_json(asdict(manifest)), encoding="utf-8")
            os.replace(pointer_temp, self.latest_path)
            return manifest
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def latest(self, *, verify_file: bool = True) -> SnapshotReplicaManifest:
        if not self.latest_path.exists():
            raise FileNotFoundError("replica has no committed LATEST manifest")
        manifest = SnapshotReplicaManifest(**json.loads(
            self.latest_path.read_text(encoding="utf-8")))
        snapshot = self.target_dir / manifest.snapshot_file
        if not snapshot.is_file() or snapshot.stat().st_size != manifest.snapshot_bytes:
            raise ReplicaCorruptionError("replica snapshot is missing or has the wrong size")
        if verify_file and _file_sha256(snapshot) != manifest.file_sha256:
            raise ReplicaCorruptionError("replica file SHA-256 mismatch")
        store = SQLiteGraphStore(snapshot, read_only=True)
        try:
            version, checksum, nodes, edges = store.graph_snapshot(manifest.memory_id)
        finally:
            store.close()
        if ((version, checksum) != (manifest.graph_version, manifest.graph_checksum)
                or logical_graph_checksum(nodes, edges) != manifest.graph_checksum):
            raise ReplicaCorruptionError("replica manifest does not match graph rows")
        return manifest

    def lag_versions(self, memory_id: str) -> int:
        manifest = self.latest(verify_file=False)
        if manifest.memory_id != memory_id:
            raise ValueError("LATEST belongs to another memory")
        return max(0, self.source.graph_version(memory_id) - manifest.graph_version)

    def promote(
        self,
        target_path: str | Path,
        *,
        authority_version: int | None = None,
        max_lag_versions: int = 0,
    ) -> SQLiteGraphStore:
        """Copy a verified follower image into a writable promoted authority."""
        manifest = self.latest()
        reference = (self.source.graph_version(manifest.memory_id)
                     if authority_version is None else authority_version)
        lag = max(0, reference - manifest.graph_version)
        if lag > max_lag_versions:
            raise ReplicaStaleError(
                f"replica is {lag} graph versions behind; limit is {max_lag_versions}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".promoting", dir=target.parent)
        os.close(descriptor)
        temp = Path(temp_name)
        try:
            shutil.copy2(self.target_dir / manifest.snapshot_file, temp)
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        promoted = SQLiteGraphStore(target)
        if ((promoted.graph_version(manifest.memory_id),
             promoted.graph_checksum(manifest.memory_id))
                != (manifest.graph_version, manifest.graph_checksum)):
            promoted.close()
            raise ReplicaCorruptionError("promoted authority failed version/checksum validation")
        return promoted
