"""A controlled predicate vocabulary, compiled once per memory by the backbone.

Extraction emits ~433 distinct predicates per memory for ~498 facts -- sentences
like *"plans a monthly family game night"* rather than relations -- so the
collection chain key is nearly a primary key and 96.1% of collections are
singletons.  Embedding clustering is the free alternative; this is the
generative one, and it is cheap because only the predicate strings are sent, not
the summaries: ~433 short labels in, ~433 short labels out, one call per memory.

The whole risk of this pass is **over-merging**.  Collapsing "visited Kyoto"
into "wants to visit Kyoto" turns a plan into a fact and would raise aggregation
accuracy while corrupting every modality-sensitive answer.  The prompt therefore
forbids merging across polarity, modality, completion and distinct actions, and
``apply_vocabulary`` re-checks those axes in code rather than trusting the model
-- a family that would merge two different modalities is rejected on the way in.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config
from ..domain import canonical_json, stable_id
from ..storage import SQLiteGraphStore

PROMPT_VERSION = "graphmem-v5.6-predicate-vocabulary-v1"

SYSTEM_PROMPT = (
    "Group predicate labels that describe the SAME action into families, so that facts "
    "about one activity can be counted together. Input is a numbered list. Return compact "
    'JSON {"f":[{"c":canonical_short_verb_phrase,"m":[index,index,...]}]} where each index is '
    "the number of an input label. Return INDICES ONLY, never label text: the labels are long "
    "propositions and echoing them does not fit in the answer. "
    "Rules, in priority order:\n"
    "1. NEVER group labels that differ in polarity: 'likes' and 'dislikes' stay apart.\n"
    "2. NEVER group labels that differ in modality: something done, something planned, "
    "something wanted, something recommended and something considered are four families, "
    "not one. 'visited Paris' and 'wants to visit Paris' must stay apart.\n"
    "3. NEVER group labels that differ in completion: 'finished the book', 'is reading the "
    "book' and 'started the book' stay apart.\n"
    "4. NEVER group different actions that merely share an object: 'bought a bike' and "
    "'repaired a bike' stay apart.\n"
    "5. ONLY group surface variants of one action: 'visited', 'went to', 'traveled to'.\n"
    "6. A label that has no clear partner forms its own family; leaving a label alone is "
    "always correct and always safe.\n"
    "6b. OUTPUT ONLY families that have two or more members. Never emit a family with a "
    "single member -- unlisted labels are kept as they are. Most labels will be unlisted, "
    "and that is the expected result.\n"
    "7. The canonical label must be a short verb phrase of one to three words, copied or "
    "shortened from the members, never invented.\n"
    "Copy member labels exactly. No markdown, no explanation, no reasoning."
)

#: Axes a family may never straddle.  Checked in code after the model answers,
#: because a prompt rule is a request and this is a guarantee.
_MODALITY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intent", ("want", "wants", "wanted", "plan", "plans", "planned", "planning",
                "hope", "hopes", "hoping", "intend", "intends", "would like", "going to",
                "thinking of", "considering", "consider", "may", "might")),
    ("recommend", ("recommend", "recommends", "recommended", "suggest", "suggests",
                   "suggested", "advise", "advised")),
    ("obligation", ("must", "should", "needs to", "has to", "have to")),
    ("progress", ("starting", "started", "began", "is reading", "currently")),
    ("complete", ("finished", "completed", "done with", "ended")),
)
_NEGATION = ("not", "never", "no longer", "without", "n't", "stopped", "quit", "cancelled")


def _axis(label: str) -> tuple[str, bool]:
    """Modality bucket and polarity of a predicate label."""
    lowered = " " + " ".join(label.casefold().split()) + " "
    modality = "asserted"
    for name, markers in _MODALITY_MARKERS:
        if any(f" {marker} " in lowered or lowered.startswith(f" {marker} ")
               for marker in markers):
            modality = name
            break
    negative = any(f" {marker} " in lowered or marker in lowered for marker in _NEGATION)
    return modality, negative


@dataclass(frozen=True, slots=True)
class VocabularyResult:
    mapping: Mapping[str, str]
    families: int
    merged_labels: int
    rejected_families: tuple[str, ...]
    tokens: int
    cached: bool
    #: "length" means the answer was cut off and the result is not a measurement.
    finish_reason: str = ""

    @property
    def largest_family(self) -> int:
        counts: dict[str, int] = {}
        for canonical in self.mapping.values():
            counts[canonical] = counts.get(canonical, 0) + 1
        return max(counts.values(), default=0)


class PredicateVocabulary:
    """Compiles raw predicate labels into families with one bounded call."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, client: Any | None = None,
                 max_family_size: int = 24) -> None:
        self.store, self.config, self.dataset_hash = store, config, dataset_hash
        self.max_family_size = max_family_size
        self.prompt_hash = hashlib.sha256(
            (PROMPT_VERSION + SYSTEM_PROMPT).encode("utf-8")).hexdigest()
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
        self.client = client

    def compile(self, memory_id: str, predicates: Sequence[str]) -> VocabularyResult:
        labels = tuple(sorted({" ".join(str(row).split()) for row in predicates if str(row).strip()}))
        if len(labels) < 2:
            return VocabularyResult({label: label for label in labels}, len(labels), 0, (), 0, True)
        # Numbered so the answer can reference members by index; echoing the
        # label text overflowed the output on 6 of 6 memories, because these
        # predicates are whole propositions averaging 4.7 words.
        payload = {"p": {str(index): label for index, label in enumerate(labels)}}
        response, usage, cached = self._call(memory_id, payload)
        families = self._parse(response)
        result = self._apply(labels, families, usage, cached)
        return replace(result, finish_reason=str(response.get("finish_reason") or ""))

    # -- family admission -------------------------------------------------

    def _apply(self, labels: Sequence[str], families: Sequence[Mapping[str, Any]],
               usage: Mapping[str, int], cached: bool) -> VocabularyResult:
        known = set(labels)
        mapping = {label: label for label in labels}
        rejected: list[str] = []
        merged = 0
        claimed: set[str] = set()
        canonical_axis: dict[str, tuple[str, bool]] = {}
        for family in families:
            canonical = " ".join(str(family.get("c", "")).split())
            members = []
            for item in family.get("m", ()):
                # Indices are the contract; a label echoed anyway still resolves.
                try:
                    members.append(labels[int(item)])
                except (TypeError, ValueError, IndexError):
                    text = " ".join(str(item).split())
                    if text in known:
                        members.append(text)
            members = [item for item in dict.fromkeys(members) if item not in claimed]
            if not canonical or len(members) < 2:
                continue
            if len(members) > self.max_family_size:
                # A family of hundreds is vocabulary collapse, not compression.
                rejected.append(f"{canonical}:oversize:{len(members)}")
                continue
            # The canonical label is what every member's fact gets relabelled
            # with, so it has to sit on the same axis as the members it replaces.
            # Checking members alone let 9 families through whose canonical label
            # carried a different modality -- e.g. asserted members collapsed
            # under a "wants to" label, which turns facts into intentions.
            axes = {_axis(item) for item in [*members, canonical]}
            if len(axes) > 1:
                # The model was asked not to straddle modality or polarity; this
                # is the guarantee that it did not.
                rejected.append(f"{canonical}:mixed_axis:{sorted(str(a) for a in axes)}")
                continue
            # Two families may be admitted separately yet share a canonical
            # label, and they then collapse into one group in the mapping.  Each
            # was internally consistent, so the per-family axis check passed
            # while the resulting group straddled modalities -- measured as 9
            # unsafe merges on one memory.  Reject the colliding second family.
            if canonical in canonical_axis and canonical_axis[canonical] != _axis(canonical):
                rejected.append(f"{canonical}:canonical_collision")
                continue
            canonical_axis[canonical] = _axis(canonical)
            for item in members:
                mapping[item] = canonical
                claimed.add(item)
            merged += len(members)
        return VocabularyResult(
            mapping=mapping, families=len({value for value in mapping.values()}),
            merged_labels=merged, rejected_families=tuple(rejected),
            tokens=int(usage.get("total_tokens", 0)), cached=cached)

    # -- plumbing ---------------------------------------------------------

    @staticmethod
    def _parse(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        text = str(response.get("content", "")).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            return []
        rows = row.get("f", ()) if isinstance(row, Mapping) else ()
        return [item for item in rows if isinstance(item, Mapping)]

    def _call(self, memory_id: str, payload: Any) -> tuple[Mapping[str, Any], Mapping[str, int], bool]:
        serialized = canonical_json(payload)
        identity = CacheIdentity(
            self.dataset_hash, self.config.models.llm_model, self.prompt_hash,
            self.config.schema_version,
            hashlib.sha256(canonical_json({
                "max_family_size": self.max_family_size}).encode()).hexdigest(),
            "predicate_vocabulary:" + hashlib.sha256(serialized.encode()).hexdigest())
        key = identity.key()
        request = {
            "model": self.config.models.llm_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": serialized}],
            # Truncation is silent: a cut-off answer parses as zero families and
            # looks exactly like "nothing should merge".  Three of six memories
            # hit finish_reason=length at 8192 and reported no merges at all.
            "temperature": 0, "max_tokens": 16384,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        started = time.perf_counter()
        cached_row = self.store.cache_get(key)
        if cached_row:
            response, old = cached_row["response"], cached_row["usage"]
            usage = {"cached_input_tokens": int(old.get("uncached_input_tokens", 0)),
                     "uncached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                     "total_tokens": int(old.get("uncached_input_tokens", 0))}
            is_cached = True
        else:
            result = self.client.chat.completions.create(**request)
            message = result.choices[0].message
            response = {"content": message.content or "", "model": getattr(result, "model", ""),
                        "finish_reason": getattr(result.choices[0], "finish_reason", None)}
            raw = getattr(result, "usage", None)
            prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
            output = int(getattr(raw, "completion_tokens", 0) or 0)
            usage = {"cached_input_tokens": 0, "uncached_input_tokens": prompt,
                     "output_tokens": output, "reasoning_tokens": 0,
                     "total_tokens": prompt + output}
            self.store.cache_put(key, "predicate_vocabulary", request, response, usage,
                                 self.prompt_hash)
            is_cached = False
        occurrence = self.store._read_one(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?",
            (memory_id, key))[0]
        self.store.log_llm_call(
            call_id=stable_id("llm-call", memory_id, key, is_cached, occurrence),
            memory_id=memory_id, stage="predicate_vocabulary", cache_key=key, cached=is_cached,
            request=request, response=response, usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000, retry_count=0, batch_size=1,
            prompt_hash=self.prompt_hash)
        return response, usage, is_cached
