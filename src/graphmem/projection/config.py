"""Which deterministic relations a projection arm builds, and how widely.

Every field is a knob in the P-series graph-index ablation.  Defaults reproduce
the frozen V5.4 graph exactly (everything off), so ``ProjectionConfig()`` is the
P0 control and each arm turns on one field.

This config is deliberately separate from ``GraphMemV5Config``: projection runs
after extraction and cannot change a single LLM call, so it must not perturb
``config_hash`` and invalidate the extraction cache.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from ..domain import canonical_json


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    # --- features -------------------------------------------------------
    #: Authoritative CollectionManifest nodes plus MEMBER_OF edges.  The
    #: navigator's ``collection_complete`` check is unreachable without these:
    #: no COLLECTION_MANIFEST node has ever been built.
    collection_manifest: bool = False
    #: DIALOGUE_PAIR across adjacent turns whose role alternates.  Assistant
    #: turns are otherwise reachable only via a fact that happens to cite them.
    dialogue_pair: bool = False
    #: TEMPORAL_BEFORE/TEMPORAL_AFTER from TemporalKey ordering, and REPLACEMENT
    #: split out of STATE_NEXT.  The frozen graph holds 150 temporal edges in
    #: total because the build requires both intervals fully resolved.
    temporal_closure: bool = False
    #: Re-derive quote spans by locating a fact's value in its source turn.
    #: Extraction computed spans and the build discarded them: all 55,323 stored
    #: evidence members cover the whole turn.
    fact_spans: bool = False
    #: CANONICAL_VALUE nodes with FACT_VALUE/SHARED_VALUE, turning intersection
    #: from a post-hoc string comparison into a graph join.
    value_lattice: bool = False
    #: EVENT_FRAME per event_instance_id, so occurrence counting has nodes to
    #: count rather than an attribute nothing carries.
    event_frames: bool = False

    # --- tunables -------------------------------------------------------
    #: 1 makes single-member collections visible; the build requires 2 rows and
    #: 2 distinct values, so single-member collections are invisible today.
    manifest_min_members: int = 1
    #: How many turns apart a dialogue pair may be.
    dialogue_pair_window: int = 1
    #: Cap on temporal edges emitted per node, bounding the O(n^2) closure.
    temporal_edge_cap: int = 32
    #: "value" matches the fact value alone; "value_predicate" also requires a
    #: predicate token nearby, trading recall for precision.
    span_derivation: str = "value"
    #: Minimum characters a derived span may cover before it is discarded as
    #: too short to be a useful quote.
    span_min_chars: int = 4
    #: Cap on SHARED_VALUE edges per canonical value, bounding value-hub cliques.
    shared_value_cap: int = 16

    def __post_init__(self) -> None:
        if self.manifest_min_members < 1:
            raise ValueError("manifest_min_members must be at least 1")
        if self.dialogue_pair_window < 1:
            raise ValueError("dialogue_pair_window must be at least 1")
        if self.temporal_edge_cap < 1 or self.shared_value_cap < 1:
            raise ValueError("edge caps must be positive")
        if self.span_derivation not in {"value", "value_predicate"}:
            raise ValueError("span_derivation must be 'value' or 'value_predicate'")
        if self.span_min_chars < 1:
            raise ValueError("span_min_chars must be positive")

    @property
    def any_enabled(self) -> bool:
        return any((self.collection_manifest, self.dialogue_pair, self.temporal_closure,
                    self.fact_spans, self.value_lattice, self.event_frames))

    def digest(self) -> str:
        """Stable identity for caching an arm's projected graph."""
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


#: The P-series arms.  P0 is the frozen control; P7 turns everything on.
ARMS: dict[str, ProjectionConfig] = {
    "P0": ProjectionConfig(),
    "P1": ProjectionConfig(collection_manifest=True),
    "P2": ProjectionConfig(value_lattice=True),
    "P3": ProjectionConfig(dialogue_pair=True),
    "P4": ProjectionConfig(temporal_closure=True),
    "P5": ProjectionConfig(fact_spans=True),
    "P6": ProjectionConfig(event_frames=True),
    "P7": ProjectionConfig(collection_manifest=True, value_lattice=True, dialogue_pair=True,
                           temporal_closure=True, fact_spans=True, event_frames=True),
}
