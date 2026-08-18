"""Memory-scoped principal registry and principal-aware owner resolution.

V5.5 resolved a question's owner by scanning ``owner_alias_index`` for any alias
occurring as a substring of the question.  Measured on the fixed 200 that is
close to unusable:

* LongMemEval facts are owned by the ``user`` and ``assistant`` principals
  (133 and 159 facts in one memory), but a first-person question never contains
  the word "user", so it resolves nothing at all;
* the same index holds 125 aliases for that memory, including ``asics`` and
  ``advertisers`` -- *value* entities competing to be the owner;
* the aliases that do match are things like ``have`` and ``how``, minted from
  common words, and 0 of 8 sampled questions had the gold fact's owner among
  the resolved set.

Identity is therefore rebuilt from authority rather than guessed from the
extraction graph: turn speaker roles define the principals, principals link to
canonical entities, and only that linked set is eligible for strong owner
matching.  Pronouns are resolved against the registry instead of being dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from .domain import stable_id
from .text import normalize_key, terms


PrincipalRole = Literal["memory_user", "assistant", "conversation_participant", "mentioned_entity"]

# Speaker labels that name a role rather than a person.  They identify the
# principal but must never be matched as a name inside a question.
GENERIC_SPEAKER_LABELS = frozenset({"user", "assistant", "system", "speaker", "questioner", "bot"})

FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})
FIRST_PERSON_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves"})
SECOND_PERSON = frozenset({"you", "your", "yours", "yourself"})

# Never eligible for strong owner matching.  Pronouns are handled by the deictic
# layer rather than discarded.
BLOCKED_ALIASES = frozenset({
    # question words
    "how", "what", "which", "who", "whom", "when", "where", "why", "whose",
    # auxiliaries and common verbs
    "have", "has", "had", "do", "did", "does", "be", "is", "are", "was", "were",
    "am", "been", "being", "will", "would", "can", "could", "should", "may",
    "might", "must", "get", "got", "go", "went", "say", "said", "make", "made",
    # pronouns and determiners
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "it", "its", "this", "that", "these", "those", "there", "here", "some", "any",
    # generic nouns that extraction mints entities from
    "user", "assistant", "person", "people", "thing", "things", "event", "events",
    "activity", "activities", "time", "times", "day", "days", "life", "right",
    "sounds", "way", "one", "ones", "someone", "something", "everyone", "stuff",
    "place", "places", "year", "years", "week", "month", "today", "yesterday",
    # function words occasionally emitted as entities by noisy extraction
    "a", "an", "the", "and", "or", "but", "if", "as", "at", "by", "for",
    "from", "in", "into", "of", "off", "on", "onto", "out", "over", "per",
    "than", "then", "through", "to", "under", "up", "with", "within", "without",
})
MIN_ALIAS_LENGTH = 3


@dataclass(frozen=True, slots=True)
class MemoryPrincipal:
    """An identity in one memory, derived from turn authority."""

    principal_id: str
    role: PrincipalRole
    speaker_labels: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    canonical_entity_ids: tuple[str, ...] = ()
    turn_count: int = 0
    is_memory_user: bool = False
    named: bool = False
    confidence: float = 1.0
    source: str = "speaker_role"


@dataclass(frozen=True, slots=True)
class ResolvedOwner:
    """One owner mention in a question, resolved to principals and entities."""

    mention_text: str
    resolution_kind: str
    principal_id: str | None = None
    canonical_entity_ids: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def strong(self) -> bool:
        """Whether a mismatch against this owner is safe to treat as a veto."""
        return (self.confidence >= 0.9
                and self.resolution_kind in {"first_person", "explicit_principal"}
                and bool(self.canonical_entity_ids))


@dataclass(frozen=True, slots=True)
class PrincipalRegistry:
    memory_id: str
    principals: tuple[MemoryPrincipal, ...] = ()
    alias_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    entity_alias_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def principal(self, principal_id: str) -> MemoryPrincipal | None:
        return next((row for row in self.principals if row.principal_id == principal_id), None)

    @property
    def memory_user(self) -> MemoryPrincipal | None:
        return next((row for row in self.principals if row.is_memory_user), None)

    @property
    def participants(self) -> tuple[MemoryPrincipal, ...]:
        return tuple(row for row in self.principals
                     if row.role in {"memory_user", "conversation_participant"})

    def stats(self) -> dict[str, Any]:
        return {
            "principals": len(self.principals),
            "named_principals": sum(1 for row in self.principals if row.named),
            "linked_principals": sum(1 for row in self.principals if row.canonical_entity_ids),
            "alias_entries": len(self.alias_index),
            "entity_alias_entries": len(self.entity_alias_index),
            "memory_user": self.memory_user.principal_id if self.memory_user else None,
            "warnings": list(self.warnings),
        }


def _eligible_alias(alias: str) -> bool:
    key = normalize_key(alias)
    if not key or len(key) < MIN_ALIAS_LENGTH:
        return False
    tokens = terms(key)
    if len(tokens) == 1 and key in BLOCKED_ALIASES:
        return False
    # A multi-word alias made only of blocked words is still noise.
    return not all(token in BLOCKED_ALIASES for token in tokens)


def build_principal_registry(store, memory_id: str, view=None) -> PrincipalRegistry:
    """Derive principals from turn speaker roles, then link them to entities.

    Speaker role is authority; the extraction graph is only consulted to attach
    canonical entity ids to an identity that already exists.
    """
    turns = list(store.turns(memory_id))
    by_speaker: dict[tuple[str, str], int] = {}
    for turn in turns:
        key = (str(turn.role or ""), str(turn.speaker or ""))
        by_speaker[key] = by_speaker.get(key, 0) + 1

    warnings: list[str] = []
    principals: list[MemoryPrincipal] = []
    for (role, speaker), count in sorted(by_speaker.items(), key=lambda row: (-row[1], row[0])):
        if not speaker:
            continue
        generic = normalize_key(speaker) in GENERIC_SPEAKER_LABELS
        named = not generic
        is_user_side = role == "user"
        if generic and normalize_key(speaker) == "assistant":
            principal_role: PrincipalRole = "assistant"
        elif generic:
            principal_role = "memory_user"
        else:
            principal_role = "conversation_participant"
        principals.append(MemoryPrincipal(
            principal_id=stable_id("principal", memory_id, role, normalize_key(speaker)),
            role=principal_role,
            speaker_labels=(speaker,),
            # A generic label identifies the principal but is not a name, so it
            # must not become a matchable alias.
            aliases=() if generic else (speaker,),
            turn_count=count,
            is_memory_user=is_user_side,
            named=named,
            confidence=1.0 if generic else 0.6,
            source="speaker_role",
        ))

    if not any(row.is_memory_user for row in principals):
        warnings.append("no_user_side_speaker")
    if sum(1 for row in principals if row.is_memory_user) > 1:
        warnings.append("multiple_user_side_speakers")

    # Link principals to canonical entities by speaker label.
    linked: list[MemoryPrincipal] = []
    entity_alias_index: dict[str, tuple[str, ...]] = dict(
        getattr(view, "owner_alias_index", {}) or {})
    for principal in principals:
        entity_ids: list[str] = []
        for label in principal.speaker_labels:
            entity_ids.extend(entity_alias_index.get(normalize_key(label), ()))
        linked.append(MemoryPrincipal(
            **{**{field_name: getattr(principal, field_name)
                  for field_name in principal.__slots__},
               "canonical_entity_ids": tuple(dict.fromkeys(entity_ids))}))
    principals = linked
    if not any(row.canonical_entity_ids for row in principals):
        warnings.append("no_principal_linked_to_entities")

    # Strong alias index: principal names only.
    alias_index: dict[str, list[str]] = {}
    for principal in principals:
        for alias in principal.aliases:
            key = normalize_key(alias)
            if _eligible_alias(key):
                alias_index.setdefault(key, []).append(principal.principal_id)

    # Weak index: other entities whose alias survives hygiene.  These can be
    # mentioned participants ("Alice's father") but never the memory user.
    weak: dict[str, tuple[str, ...]] = {
        key: value for key, value in entity_alias_index.items()
        if _eligible_alias(key) and key not in alias_index
    }
    return PrincipalRegistry(
        memory_id, tuple(principals),
        {key: tuple(value) for key, value in alias_index.items()},
        weak, tuple(warnings),
    )


def _longest_alias_matches(query: str, index: Mapping[str, tuple[str, ...]]) -> list[tuple[str, tuple[str, ...]]]:
    """Token-boundary matches, longest alias first, deduplicated by target."""
    lowered = f" {normalize_key(query)} "
    matches = [(alias, targets) for alias, targets in index.items()
               if alias and f" {alias} " in lowered]
    matches.sort(key=lambda row: (-len(row[0].split()), -len(row[0]), row[0]))
    selected: list[tuple[str, tuple[str, ...]]] = []
    claimed: set[str] = set()
    for alias, targets in matches:
        key = tuple(sorted(targets))
        if key in claimed:
            continue
        claimed.add(key)
        selected.append((alias, targets))
    return selected


def resolve_query_owners(query: str, registry: PrincipalRegistry,
                         ) -> tuple[tuple[ResolvedOwner, ...], tuple[str, ...]]:
    """Resolve owner mentions in three layers: deictic, principal, entity."""
    tokens = frozenset(terms(query))
    resolved: list[ResolvedOwner] = []
    warnings: list[str] = []

    # --- layer 1: deictic ------------------------------------------------------
    if tokens & FIRST_PERSON:
        user = registry.memory_user
        if user is None:
            warnings.append("first_person_without_memory_user")
        else:
            # A generic "user" label is unambiguous; a named user-side speaker is
            # an inference, so it carries lower confidence and says so.
            confidence = 1.0 if not user.named else 0.6
            if user.named:
                warnings.append("first_person_mapped_to_named_speaker")
            resolved.append(ResolvedOwner(
                mention_text=next(iter(sorted(tokens & FIRST_PERSON))),
                resolution_kind="first_person", principal_id=user.principal_id,
                canonical_entity_ids=user.canonical_entity_ids, confidence=confidence))
    # A question may name the role instead of the person ("what did the
    # assistant suggest").  The role word is not a matchable alias, but it does
    # identify a principal, so resolve it here rather than in the name layer.
    for role_word, role in (("assistant", "assistant"), ("user", "memory_user")):
        if role_word not in tokens:
            continue
        principal = next((row for row in registry.principals if row.role == role), None)
        if principal is None or any(row.principal_id == principal.principal_id
                                    for row in resolved):
            continue
        resolved.append(ResolvedOwner(
            mention_text=role_word, resolution_kind="explicit_principal",
            principal_id=principal.principal_id,
            canonical_entity_ids=principal.canonical_entity_ids, confidence=0.95))

    if tokens & FIRST_PERSON_PLURAL:
        # "we" is a group, not the user alone; do not silently narrow it.
        warnings.append("first_person_plural_unresolved_group")
    if tokens & SECOND_PERSON:
        # "you" may address the assistant or the other conversation participant.
        # Guessing here was judged unsafe, so it is recorded and left alone.
        warnings.append("second_person_unresolved")

    # --- layer 2: explicit principals -----------------------------------------
    for alias, principal_ids in _longest_alias_matches(query, registry.alias_index):
        entity_ids: list[str] = []
        for principal_id in principal_ids:
            principal = registry.principal(principal_id)
            if principal:
                entity_ids.extend(principal.canonical_entity_ids)
        resolved.append(ResolvedOwner(
            mention_text=alias, resolution_kind="explicit_principal",
            principal_id=principal_ids[0] if len(principal_ids) == 1 else None,
            canonical_entity_ids=tuple(dict.fromkeys(entity_ids)),
            confidence=0.95 if len(principal_ids) == 1 else 0.6))

    # --- layer 3: other named entities ----------------------------------------
    if not resolved:
        for alias, entity_ids in _longest_alias_matches(query, registry.entity_alias_index):
            resolved.append(ResolvedOwner(
                mention_text=alias, resolution_kind="explicit_entity",
                canonical_entity_ids=tuple(entity_ids), confidence=0.5))

    if not resolved:
        warnings.append("no_owner_resolved")
    return tuple(resolved), tuple(dict.fromkeys(warnings))


def resolution_stats(resolved: Sequence[ResolvedOwner], warnings: Iterable[str]) -> dict[str, Any]:
    kinds = [row.resolution_kind for row in resolved]
    return {
        "resolved_owners": len(resolved),
        "resolution_kinds": sorted(set(kinds)),
        "first_person": "first_person" in kinds,
        "strong_owners": sum(1 for row in resolved if row.strong),
        "entity_ids": sorted({item for row in resolved for item in row.canonical_entity_ids}),
        "max_confidence": max((row.confidence for row in resolved), default=0.0),
        "warnings": sorted(set(warnings)),
    }
