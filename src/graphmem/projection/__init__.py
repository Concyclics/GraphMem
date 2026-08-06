"""Deterministic graph projection.

Rules here run after extraction, are identical for every memory, use no LLM and
no gold label, and depend only on attributes the build already wrote.  A
projection arm can therefore be re-run over a frozen graph without spending a
single token or invalidating the extraction cache.
"""
from __future__ import annotations

from .config import ARMS, ProjectionConfig
from .manifest import ManifestRow, build_manifests, chain_key, collect_chains, manifest_stats

__all__ = [
    "ARMS", "ManifestRow", "ProjectionConfig", "build_manifests", "chain_key",
    "collect_chains", "manifest_stats",
]
