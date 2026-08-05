#!/usr/bin/env python3
"""Prove a V5.6 run consumed the frozen V5.4 authority graph, unmodified.

Track A must never mutate the authority database.  Comparing the per-memory
``graph_checksum`` digest is far cheaper than re-hashing a 2.5 GB file, so the
digest is the default check and the full file hash is opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from graphmem.storage import SQLiteGraphStore


def freeze_expectations(path: Path) -> dict[str, str]:
    """Read the checksum table out of ``docs/V5_6_FREEZE.md``."""
    rows = re.findall(r"^\| `([0-9a-z:._-]+)` \| (\d+) \| \d+ \| \d+ \| \d+ \| `([0-9a-f]{64})` \|$",
                      path.read_text(encoding="utf-8"), re.MULTILINE)
    if not rows:
        raise ValueError(f"no per-memory checksum rows found in {path}")
    return {memory_id: f"{memory_id}:{version}:{checksum}" for memory_id, version, checksum in rows}


def observed_triples(store: SQLiteGraphStore, memory_ids: list[str]) -> dict[str, str]:
    return {memory_id: f"{memory_id}:{store.graph_version(memory_id)}:"
                       f"{store.graph_checksum(memory_id)}" for memory_id in memory_ids}


def digest(triples: dict[str, str]) -> str:
    return hashlib.sha256("\n".join(value for _, value in sorted(triples.items())).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--freeze-doc", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/V5_6_FREEZE.md")
    parser.add_argument("--full-file-hash", action="store_true",
                        help="also re-hash the whole SQLite file (slow)")
    parser.add_argument("--json", type=Path, help="write the verification record here")
    args = parser.parse_args()

    expected = freeze_expectations(args.freeze_doc)
    store = SQLiteGraphStore(args.source_db, read_only=True)
    try:
        observed = observed_triples(store, sorted(expected))
    finally:
        store.close()

    mismatched = sorted(key for key, value in expected.items() if observed.get(key) != value)
    record = {
        "source_db": str(args.source_db),
        "freeze_doc": str(args.freeze_doc),
        "memories": len(expected),
        "expected_digest": digest(expected),
        "observed_digest": digest(observed),
        "mismatched_memories": mismatched,
    }
    if args.full_file_hash:
        record["source_db_sha256"] = file_sha256(args.source_db)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True))
    if mismatched:
        print(f"FAIL: {len(mismatched)} memories diverge from the freeze record", file=sys.stderr)
        return 1
    print("OK: authority graph matches the freeze record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
