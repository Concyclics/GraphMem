from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from graphmem.domain import Conversation, Session, SourceTurn, stable_id
from graphmem.principals import (
    BLOCKED_ALIASES,
    build_principal_registry,
    resolution_stats,
    resolve_query_owners,
)
from graphmem.storage import SQLiteGraphStore


@dataclass
class _View:
    owner_alias_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def _turn(memory_id: str, session_id: str, index: int, speaker: str, role: str,
          text: str = "hello") -> SourceTurn:
    return SourceTurn(stable_id("turn", memory_id, session_id, index), memory_id, session_id,
                      index, speaker, "", role, None, text,
                      hashlib.sha256(text.encode()).hexdigest())


def _store(path, memory_id: str, rows: Sequence[tuple[str, str]]) -> SQLiteGraphStore:
    store = SQLiteGraphStore(path)
    turns = [_turn(memory_id, "s1", index, speaker, role)
             for index, (speaker, role) in enumerate(rows)]
    store.ingest_conversation(Conversation(memory_id, "test", memory_id, "hash"),
                              [Session("s1", memory_id, 0, None, "sh")], turns)
    return store


def _generic_store(path):
    """LongMemEval shape: generic 'user'/'assistant' speaker labels."""
    return _store(path, "m", [("user", "user"), ("assistant", "assistant")] * 5)


def _named_store(path):
    """LoCoMo shape: real names, with role marking which side speaks."""
    return _store(path, "m", [("Caroline", "user"), ("Melanie", "assistant")] * 5)


# --- registry construction -----------------------------------------------------

def test_generic_labels_produce_user_and_assistant_principals(tmp_path) -> None:
    store = _generic_store(tmp_path / "g.sqlite")

    registry = build_principal_registry(store, "m", _View({"user": ("entity:user",),
                                                           "assistant": ("entity:assistant",)}))

    roles = {row.role for row in registry.principals}
    assert roles == {"memory_user", "assistant"}
    user = registry.memory_user
    assert user is not None and user.is_memory_user and not user.named
    assert user.canonical_entity_ids == ("entity:user",)
    store.close()


def test_named_speakers_become_participants_with_a_user_side(tmp_path) -> None:
    store = _named_store(tmp_path / "n.sqlite")

    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",),
                                                           "melanie": ("entity:m",)}))

    assert {row.role for row in registry.principals} == {"conversation_participant"}
    user = registry.memory_user
    assert user is not None and user.speaker_labels == ("Caroline",) and user.named
    assert user.canonical_entity_ids == ("entity:c",)
    store.close()


def test_generic_labels_never_become_matchable_aliases(tmp_path) -> None:
    """'user' identifies a principal but must not match the word in a question."""
    store = _generic_store(tmp_path / "g.sqlite")

    registry = build_principal_registry(store, "m", _View({"user": ("entity:user",)}))

    assert registry.alias_index == {}
    store.close()


def test_common_word_entities_are_excluded_from_strong_matching(tmp_path) -> None:
    """Measured pollution: 'have', 'how', 'life', 'right', 'sounds'."""
    store = _named_store(tmp_path / "n.sqlite")
    noisy = {"caroline": ("entity:c",), "melanie": ("entity:m",),
             "life": ("entity:life",), "right": ("entity:right",),
             "sounds": ("entity:sounds",), "have": ("entity:have",), "how": ("entity:how",)}

    registry = build_principal_registry(store, "m", _View(noisy))

    assert set(registry.alias_index) == {"caroline", "melanie"}
    for junk in ("life", "right", "sounds", "have", "how"):
        assert junk not in registry.entity_alias_index
    store.close()


# --- resolution ----------------------------------------------------------------

def test_first_person_resolves_to_the_memory_user(tmp_path) -> None:
    """The whole LongMemEval fix: 'I' never matched any alias before."""
    store = _generic_store(tmp_path / "g.sqlite")
    registry = build_principal_registry(store, "m", _View({"user": ("entity:user",)}))

    resolved, warnings = resolve_query_owners("What did I order at the restaurant?", registry)

    assert len(resolved) == 1
    owner = resolved[0]
    assert owner.resolution_kind == "first_person"
    assert owner.confidence == 1.0 and owner.strong
    assert owner.canonical_entity_ids == ("entity:user",)
    assert "no_owner_resolved" not in warnings
    store.close()


def test_first_person_against_a_named_speaker_is_weak_and_says_so(tmp_path) -> None:
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",)}))

    resolved, warnings = resolve_query_owners("What did I do last week?", registry)

    assert resolved[0].resolution_kind == "first_person"
    assert resolved[0].confidence < 0.9 and not resolved[0].strong
    assert "first_person_mapped_to_named_speaker" in warnings
    store.close()


def test_named_owners_resolve_separately_for_a_shared_question(tmp_path) -> None:
    """'both A and B' must stay two operands, not one merged alias."""
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",),
                                                           "melanie": ("entity:m",)}))

    resolved, _ = resolve_query_owners("Where did Caroline and Melanie both go?", registry)

    assert len(resolved) == 2
    assert {row.mention_text for row in resolved} == {"caroline", "melanie"}
    assert all(row.resolution_kind == "explicit_principal" and row.strong for row in resolved)
    store.close()


def test_role_word_resolves_the_assistant_principal(tmp_path) -> None:
    store = _generic_store(tmp_path / "g.sqlite")
    registry = build_principal_registry(store, "m", _View({"assistant": ("entity:a",)}))

    resolved, _ = resolve_query_owners("What did the assistant recommend?", registry)

    assert [row.resolution_kind for row in resolved] == ["explicit_principal"]
    assert resolved[0].canonical_entity_ids == ("entity:a",)
    store.close()


def test_second_person_is_recorded_rather_than_guessed(tmp_path) -> None:
    """'you' may address the assistant or the other participant; do not choose."""
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",)}))

    _resolved, warnings = resolve_query_owners("What did you say about the trip?", registry)

    assert "second_person_unresolved" in warnings
    store.close()


def test_first_person_plural_is_not_narrowed_to_the_user(tmp_path) -> None:
    store = _generic_store(tmp_path / "g.sqlite")
    registry = build_principal_registry(store, "m", _View({"user": ("entity:user",)}))

    _resolved, warnings = resolve_query_owners("Where did we go together?", registry)

    assert "first_person_plural_unresolved_group" in warnings
    store.close()


def test_ambiguous_question_resolves_nothing_rather_than_guessing(tmp_path) -> None:
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",)}))

    resolved, warnings = resolve_query_owners("What happened at the tournament?", registry)

    assert resolved == ()
    assert "no_owner_resolved" in warnings
    store.close()


def test_no_blocked_word_can_ever_resolve_as_an_owner(tmp_path) -> None:
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(
        store, "m", _View({word: (f"entity:{word}",) for word in BLOCKED_ALIASES}))

    for word in sorted(BLOCKED_ALIASES)[:20]:
        resolved, _ = resolve_query_owners(f"What did {word} do?", registry)
        assert all(row.mention_text != word or row.resolution_kind == "first_person"
                   for row in resolved), word
    store.close()


def test_resolution_is_deterministic(tmp_path) -> None:
    store = _named_store(tmp_path / "n.sqlite")
    registry = build_principal_registry(store, "m", _View({"caroline": ("entity:c",),
                                                           "melanie": ("entity:m",)}))
    query = "Where did Caroline and Melanie both go?"

    assert resolve_query_owners(query, registry) == resolve_query_owners(query, registry)
    store.close()


def test_stats_report_the_resolution_shape(tmp_path) -> None:
    store = _generic_store(tmp_path / "g.sqlite")
    registry = build_principal_registry(store, "m", _View({"user": ("entity:user",)}))

    resolved, warnings = resolve_query_owners("What did I buy?", registry)
    stats = resolution_stats(resolved, warnings)

    assert stats["first_person"] is True
    assert stats["strong_owners"] == 1
    assert stats["entity_ids"] == ["entity:user"]
    store.close()
