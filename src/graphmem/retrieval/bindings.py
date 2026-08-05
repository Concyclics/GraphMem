"""Operand binding discriminant.

V5.5 scored a binding with three additive terms and one dominant default: an
operand that resolved no owner gave *every* canonical fact 0.55, so the score
lattice was {0.55, 0.73, 0.90} and the 0.70 threshold downstream was just an
alias for "at least one predicate term overlapped".  ``scope_candidates`` was
never populated, so its term was dead code.

That was survivable while only 21% of the right facts were ever reached.  With
the fact reservoir it is not: a wide candidate pool scored by a permissive rule
manufactures bindings, which would inflate recall while destroying precision.

So the decision is decomposed into named dimensions, conflicts veto outright
rather than being outvoted, and an ownerless operand has to earn its binding
from at least two independent signals.

**This is measured but not promoted.**  On the fixed 200 it halves binding
coverage (10% -> 5%), because the owner signal it leans on is unusable today:
`_query_owners` resolves nothing for first-person questions and otherwise
matches junk entities minted from common words, so 0/8 sampled questions had
the gold fact's owner among the operand's owners.  It stays behind
``GraphNavigator(binding_discriminant=True)`` until owner resolution is fixed.
See docs/V5_6_PROGRESS.md for the full measurement.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..domain import FactBinding, OperandSpec, TemporalKey, stable_id
from ..text import content_terms, normalize_key


# Soft weights.  These only order candidates that already cleared every veto.
WEIGHTS = {
    "owner": 0.40, "predicate": 0.28, "scope": 0.10, "value_type": 0.08,
    "temporal": 0.08, "source_projection": 0.14, "graph_path": 0.06, "session": 0.04,
}
# An ownerless operand needs corroboration from at least this many dimensions.
OWNERLESS_MIN_SIGNALS = 2
ACCEPT_THRESHOLD = 0.30
# Owner aliases that are really extraction noise: the graph mints a
# CanonicalEntity for common words, and matching one of those against the
# question resolves an "owner" that owns nothing relevant.
UNTRUSTED_OWNER_ALIASES = frozenset({
    "have", "has", "had", "how", "what", "when", "where", "which", "who", "why",
    "did", "does", "do", "was", "were", "is", "are", "the", "this", "that",
    "there", "here", "it", "one", "time", "day", "thing", "things", "people",
})


def trusted_owner(operand: OperandSpec) -> bool:
    """Whether the operand's owner alias is name-like enough to veto on."""
    aliases = [normalize_key(item) for item in operand.owner_aliases if item]
    if not aliases:
        return False
    return all(alias not in UNTRUSTED_OWNER_ALIASES and len(alias) > 2 for alias in aliases)


@dataclass(frozen=True, slots=True)
class BindingEvidence:
    """Why a fact bound, dimension by dimension, so a score is explainable."""

    owner_match: bool = False
    predicate_match: bool = False
    scope_match: bool = False
    value_type_match: bool = False
    temporal_match: bool = False
    polarity_match: bool = True
    modality_match: bool = True
    session_match: bool = False
    source_projection_match: bool = False
    graph_path_match: bool = False
    veto: str = ""

    @property
    def matched(self) -> tuple[str, ...]:
        return tuple(name for name, value in (
            ("owner", self.owner_match), ("predicate", self.predicate_match),
            ("scope", self.scope_match), ("value_type", self.value_type_match),
            ("temporal", self.temporal_match), ("session", self.session_match),
            ("source_projection", self.source_projection_match),
            ("graph_path", self.graph_path_match),
        ) if value)

    @property
    def independent_signals(self) -> int:
        """Corroborating dimensions, counting owner separately from the rest."""
        return len([name for name in self.matched if name != "owner"])

    def score(self) -> float:
        return sum(WEIGHTS.get(name, 0.0) for name in self.matched)


def _predicate_overlap(predicate: str, candidates: Sequence[str]) -> bool:
    predicate_terms = set(content_terms(predicate))
    if not predicate_terms:
        return False
    for candidate in candidates:
        candidate_terms = set(content_terms(candidate))
        if candidate_terms and (candidate_terms <= predicate_terms
                                or predicate_terms <= candidate_terms):
            return True
    return False


def _temporal_conflict(fact_time: TemporalKey | None, constraint: TemporalKey | None) -> bool:
    """Only a resolved pair that provably cannot overlap is a conflict."""
    if fact_time is None or constraint is None:
        return False
    if not (fact_time.resolved and constraint.resolved):
        return False
    return not fact_time.overlaps(constraint)


def evaluate_binding(node, operand: OperandSpec, owner_ids: frozenset[str], *,
                     query_terms: frozenset[str] = frozenset(),
                     source_projected: bool = False,
                     graph_path: bool = False,
                     session_ids: frozenset[str] = frozenset(),
                     temporal_constraint: TemporalKey | None = None,
                     owner_trusted: bool = False) -> BindingEvidence:
    attrs = node.attributes
    owner_id = str(attrs.get("owner_id", "") or "")
    predicate = normalize_key(str(attrs.get("predicate", "") or ""))
    scope = normalize_key(str(attrs.get("scope", "") or ""))
    value_type = str(attrs.get("value_type", "") or "")
    polarity = str(attrs.get("polarity", "positive") or "positive")
    modality = str(attrs.get("modality", "asserted") or "asserted")
    session_id = str(attrs.get("session_id", "") or "")
    fact_time = TemporalKey.from_attribute(attrs.get("time_interval"))

    # --- hard conflicts: a mismatch here is disqualifying, not outvoted -------
    # Owner is deliberately *not* a veto.  Measured on the fixed 200, the
    # resolved operand owners are frequently junk entities extracted from common
    # words ("have", "how"), and first-person questions resolve no owner at all,
    # so 0/8 sampled questions had the gold fact's owner among them.  Vetoing on
    # that signal removes correct bindings; it only demotes here.
    if owner_ids and owner_id and owner_id not in owner_ids and owner_trusted:
        return BindingEvidence(veto="owner_conflict")
    if operand.polarity and polarity != operand.polarity:
        return BindingEvidence(veto="polarity_conflict")
    if operand.modalities and modality not in operand.modalities:
        return BindingEvidence(veto="modality_conflict")
    if operand.value_type and value_type and value_type != operand.value_type:
        return BindingEvidence(veto="value_type_conflict")
    if _temporal_conflict(fact_time, temporal_constraint):
        return BindingEvidence(veto="temporal_conflict")

    scope_keys = {normalize_key(item) for item in (operand.scope_candidates or ())}
    evidence = BindingEvidence(
        owner_match=bool(owner_ids) and owner_id in owner_ids,
        predicate_match=_predicate_overlap(predicate, operand.predicate_candidates),
        scope_match=bool(scope_keys) and scope in scope_keys,
        value_type_match=bool(operand.value_type) and value_type == operand.value_type,
        temporal_match=bool(temporal_constraint) and fact_time is not None
                       and fact_time.resolved and temporal_constraint.resolved
                       and fact_time.overlaps(temporal_constraint),
        polarity_match=not operand.polarity or polarity == operand.polarity,
        modality_match=not operand.modalities or modality in operand.modalities,
        session_match=bool(session_ids) and session_id in session_ids,
        # Being projected from a retrieved turn is corroboration, not proof: the
        # turn may carry several facts and only some belong to this operand.
        source_projection_match=source_projected and bool(
            query_terms & (set(content_terms(predicate))
                           | set(content_terms(str(attrs.get("value", "") or ""))))),
        graph_path_match=graph_path,
    )
    return evidence


def accepts(evidence: BindingEvidence, operand: OperandSpec) -> bool:
    if evidence.veto:
        return False
    if evidence.owner_match:
        return True
    # No free ride for a missing owner, but no automatic rejection either: the
    # operand's owner may simply be unresolvable (first-person questions) or
    # wrong (a junk alias), so corroboration from other dimensions decides.
    return evidence.independent_signals >= OWNERLESS_MIN_SIGNALS


def binding_score(node, operand: OperandSpec, owner_ids: set[str]) -> float:
    """Backwards-compatible scalar used by the frozen H2-H6 rungs."""
    evidence = evaluate_binding(node, operand, frozenset(owner_ids))
    if evidence.veto:
        return 0.0
    score = 0.0
    if not operand.owner_aliases or evidence.owner_match:
        score += 0.55
    if operand.predicate_candidates:
        predicate = normalize_key(str(node.attributes.get("predicate", "")))
        overlap = max((len(content_terms(predicate) & content_terms(value))
                       for value in operand.predicate_candidates), default=0)
        score += min(0.35, overlap * 0.18)
    if evidence.scope_match:
        score += 0.10
    return score


def _text(attributes, key: str) -> str | None:
    """Read an optional string attribute without inventing the literal 'None'."""
    value = attributes.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _binding(operand_id: str, node, score: float, path: tuple[str, ...]) -> FactBinding:
    attrs = node.attributes
    return FactBinding(
        stable_id("binding", operand_id, node.node_id), operand_id, node.node_id,
        _text(attrs, "owner_id"), str(attrs.get("predicate", "")), str(attrs.get("scope", "")),
        normalize_key(str(attrs.get("value_key", attrs.get("value", "")))),
        _text(attrs, "event_instance_id"),
        TemporalKey.from_attribute(attrs.get("time_interval")),
        tuple(node.all_evidence_group_ids), path, score,
        value=str(attrs.get("value", "")),
        value_type=str(attrs.get("value_type", "text") or "text"),
        polarity=str(attrs.get("polarity", "positive") or "positive"),
        modality=str(attrs.get("modality", "asserted") or "asserted"),
        session_id=str(attrs.get("session_id", "") or ""),
        turn_index=int(attrs.get("turn_index", -1) or -1),
        collection_key=normalize_key(str(attrs.get("collection_key", "") or "")),
        node_type=node.node_type.value,
    )


def bind_facts(view, operand_owners: dict[str, set[str]], operands: Iterable[OperandSpec],
               fact_node_ids: Iterable[str], paths: dict[str, tuple[str, ...]] | None = None,
               ) -> tuple[FactBinding, ...]:
    paths = paths or {}
    rows: list[FactBinding] = []
    for operand in operands:
        owners = operand_owners.get(operand.operand_id, set())
        for node_id in fact_node_ids:
            node = view.nodes.get(node_id)
            if not node or node.node_type.value != "canonical_fact":
                continue
            score = binding_score(node, operand, owners)
            if score <= 0.0:
                continue
            rows.append(_binding(operand.operand_id, node, score, paths.get(node_id, ())))
    dedup = {row.binding_id: row for row in rows}
    return tuple(sorted(dedup.values(), key=lambda row: (-row.confidence, row.binding_id)))


def bind_facts_discriminant(view, operand_owners: dict[str, set[str]],
                            operands: Iterable[OperandSpec], fact_node_ids: Iterable[str],
                            paths: dict[str, tuple[str, ...]] | None = None, *,
                            query_terms: frozenset[str] = frozenset(),
                            source_projected: Mapping[str, bool] | None = None,
                            temporal_constraint: TemporalKey | None = None,
                            ) -> tuple[tuple[FactBinding, ...], dict[str, int]]:
    """Dimension-based binding, with vetoes and per-reason telemetry."""
    paths = paths or {}
    projected = source_projected or {}
    rows: list[FactBinding] = []
    reasons: dict[str, int] = defaultdict(int)
    node_ids = list(fact_node_ids)
    for operand in operands:
        owners = frozenset(operand_owners.get(operand.operand_id, set()))
        for node_id in node_ids:
            node = view.nodes.get(node_id)
            if not node or node.node_type.value != "canonical_fact":
                continue
            evidence = evaluate_binding(
                node, operand, owners, query_terms=query_terms,
                source_projected=bool(projected.get(node_id)),
                graph_path=node_id in paths, temporal_constraint=temporal_constraint,
                owner_trusted=trusted_owner(operand))
            if evidence.veto:
                reasons["veto:" + evidence.veto] += 1
                continue
            if not accepts(evidence, operand):
                reasons["rejected:insufficient_signals"] += 1
                continue
            score = evidence.score()
            if score < ACCEPT_THRESHOLD:
                reasons["rejected:below_threshold"] += 1
                continue
            reasons["accepted"] += 1
            for name in evidence.matched:
                reasons["matched:" + name] += 1
            rows.append(_binding(operand.operand_id, node, score, paths.get(node_id, ())))
    dedup = {row.binding_id: row for row in rows}
    ordered = tuple(sorted(dedup.values(), key=lambda row: (-row.confidence, row.binding_id)))
    reasons["bindings"] = len(ordered)
    reasons["candidate_facts"] = len(node_ids)
    return ordered, dict(reasons)


def by_operand(rows: Iterable[FactBinding]) -> dict[str, tuple[FactBinding, ...]]:
    grouped: dict[str, list[FactBinding]] = defaultdict(list)
    for row in rows:
        grouped[row.operand_id].append(row)
    return {key: tuple(sorted(value, key=lambda row: (-row.confidence, row.binding_id)))
            for key, value in grouped.items()}
