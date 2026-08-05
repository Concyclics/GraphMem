from __future__ import annotations

from graphmem.domain import GraphNode, NodeType, OperandSpec, TemporalKey
from graphmem.retrieval.bindings import (
    ACCEPT_THRESHOLD,
    OWNERLESS_MIN_SIGNALS,
    accepts,
    bind_facts,
    bind_facts_discriminant,
    binding_score,
    evaluate_binding,
    trusted_owner,
)
from graphmem.runtime import GraphReadView


def _fact(node_id: str = "fact:1", **attrs) -> GraphNode:
    base = {"owner_id": "owner-alice", "predicate": "volunteer", "value": "homeless shelter",
            "value_key": "homeless shelter", "scope": "community", "polarity": "positive",
            "modality": "asserted", "session_id": "s1"}
    base.update(attrs)
    return GraphNode(node_id, "m", NodeType.CANONICAL_FACT, 0,
                     f"{base['owner_id']} {base['predicate']} {base['value']}", "group:1",
                     attributes=base)


def _operand(**kwargs) -> OperandSpec:
    base = {"operand_id": "only", "owner_aliases": ("alice",),
            "predicate_candidates": ("volunteer",)}
    base.update(kwargs)
    return OperandSpec(**base)


def test_polarity_conflict_vetoes() -> None:
    evidence = evaluate_binding(_fact(polarity="negative"),
                                _operand(polarity="positive"), frozenset({"owner-alice"}))

    assert evidence.veto == "polarity_conflict"
    assert not accepts(evidence, _operand(polarity="positive"))


def test_modality_conflict_vetoes() -> None:
    """A planned trip is not evidence for a completed one."""
    evidence = evaluate_binding(_fact(modality="planned"),
                                _operand(modalities=("asserted",)), frozenset({"owner-alice"}))

    assert evidence.veto == "modality_conflict"


def test_value_type_conflict_vetoes() -> None:
    evidence = evaluate_binding(_fact(value_type="number"),
                                _operand(value_type="location"), frozenset({"owner-alice"}))

    assert evidence.veto == "value_type_conflict"


def test_disjoint_resolved_intervals_veto() -> None:
    fact = _fact(time_interval={"start": "2023-01-01T00:00:00", "end": "2023-01-02T00:00:00",
                                "precision": "day", "kind": "absolute", "confidence": 1.0,
                                "raw_text": "1 Jan 2023"})
    constraint = TemporalKey("2024-06-01T00:00:00", "2024-06-02T00:00:00", "day", "absolute", 1.0)

    evidence = evaluate_binding(fact, _operand(), frozenset({"owner-alice"}),
                                temporal_constraint=constraint)

    assert evidence.veto == "temporal_conflict"


def test_unresolved_time_is_not_a_conflict() -> None:
    """Only a provably disjoint pair conflicts; missing time must not veto."""
    constraint = TemporalKey("2024-06-01T00:00:00", "2024-06-02T00:00:00", "day", "absolute", 1.0)

    evidence = evaluate_binding(_fact(), _operand(), frozenset({"owner-alice"}),
                                temporal_constraint=constraint)

    assert evidence.veto == ""


def test_owner_mismatch_does_not_veto_when_the_alias_is_untrusted() -> None:
    """Measured: resolved owners are often junk entities like 'have' or 'how'.

    Vetoing on that signal deletes correct bindings, so it only demotes.
    """
    operand = _operand(owner_aliases=("have",))

    assert trusted_owner(operand) is False
    evidence = evaluate_binding(_fact(owner_id="owner-someone-else"), operand,
                                frozenset({"owner-have"}), owner_trusted=trusted_owner(operand))
    assert evidence.veto == ""


def test_owner_mismatch_vetoes_for_a_name_like_alias() -> None:
    operand = _operand(owner_aliases=("maria",))

    assert trusted_owner(operand) is True
    evidence = evaluate_binding(_fact(owner_id="owner-someone-else"), operand,
                                frozenset({"owner-maria"}), owner_trusted=True)
    assert evidence.veto == "owner_conflict"


def test_ownerless_operand_needs_corroboration() -> None:
    """V5.5 handed every fact 0.55 when no owner resolved; that is gone."""
    operand = _operand(owner_aliases=(), predicate_candidates=("unrelated",))

    evidence = evaluate_binding(_fact(), operand, frozenset())

    assert evidence.independent_signals < OWNERLESS_MIN_SIGNALS
    assert not accepts(evidence, operand)


def test_ownerless_operand_accepts_two_independent_signals() -> None:
    operand = _operand(owner_aliases=(), scope_candidates=("community",))

    evidence = evaluate_binding(_fact(), operand, frozenset())

    assert evidence.predicate_match and evidence.scope_match
    assert accepts(evidence, operand)
    assert evidence.score() >= ACCEPT_THRESHOLD


def test_evidence_names_the_dimensions_that_matched() -> None:
    """A binding must be explainable, not a single opaque float."""
    evidence = evaluate_binding(_fact(), _operand(scope_candidates=("community",)),
                                frozenset({"owner-alice"}))

    assert set(evidence.matched) >= {"owner", "predicate", "scope"}


def test_legacy_binding_score_is_unchanged_for_the_frozen_rungs() -> None:
    """H2-H6 keep the V5.5 lattice, so their numbers stay comparable.

    Owner match contributes 0.55 and each overlapping predicate term 0.18 up to
    0.35, so a one-word predicate match lands on 0.73.  The downstream 0.70
    collection filter therefore means exactly "the owner matched and at least
    one predicate term overlapped" -- which is why removing it doubles binding
    coverage and costs precision.
    """
    assert binding_score(_fact(), _operand(), {"owner-alice"}) == 0.73
    assert binding_score(_fact(), _operand(predicate_candidates=()), {"owner-alice"}) == 0.55
    assert binding_score(_fact(owner_id="other"), _operand(), {"owner-alice"}) == 0.18
    two_terms = binding_score(_fact(predicate="volunteer shelter"),
                              _operand(predicate_candidates=("volunteer shelter",)),
                              {"owner-alice"})
    assert two_terms == 0.9


def test_discriminant_reports_why_each_candidate_was_rejected() -> None:
    view = GraphReadView([_fact("fact:1"), _fact("fact:2", polarity="negative")], [])

    bindings, reasons = bind_facts_discriminant(
        view, {"only": {"owner-alice"}}, [_operand(polarity="positive")],
        ["fact:1", "fact:2"])

    assert len(bindings) == 1
    assert reasons["veto:polarity_conflict"] == 1
    assert reasons["accepted"] == 1
    assert reasons["candidate_facts"] == 2


def test_both_binders_agree_on_a_clean_match() -> None:
    view = GraphReadView([_fact("fact:1")], [])

    legacy = bind_facts(view, {"only": {"owner-alice"}}, [_operand()], ["fact:1"])
    strict, _ = bind_facts_discriminant(view, {"only": {"owner-alice"}}, [_operand()], ["fact:1"])

    assert {row.fact_node_id for row in legacy} == {row.fact_node_id for row in strict}
