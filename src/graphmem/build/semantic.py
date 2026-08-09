from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config, config_hash
from ..domain import SourceTurn, canonical_json, stable_id
from ..storage import SQLiteGraphStore
from .atomic_extractor import (
    InformationUnit,
    adaptive_fact_cap,
    scan_information_units,
    sentence_chunks,
    units_for_span,
)
from .budget import BuildTokenLedger


#: Measured fixed input cost of one extraction call on the frozen V5.4 build:
#: system prompt + JSON schema + payload scaffold, over 4,000 sampled calls.
CALL_OVERHEAD_TOKENS = 316
#: Measured characters per token for this corpus under the Qwen3 vocabulary.
CHARS_PER_TOKEN = 3.84
#: Margin applied to every reservation so a call that costs more than its
#: estimate cannot carry the memory past its ceiling.
ESTIMATE_SAFETY = 1.10

PROMPT_VERSION = "graphmem-v5.1-scene-semantic-v1"
STRICT_PROMPT_VERSION = "graphmem-v5.8-strict-scene-facts-v7"
ATOMIC_PROMPT_VERSION = "graphmem-v5.10-lossless-atomic-facts-v1"
SYSTEM_PROMPT = """Extract grounded memory facts. Return compact JSON {"s":[{"i":scene_id,"m":summary,"f":[{"o":owner,"p":predicate,"v":value,"y":value_type,"g":scope,"n":"positive|negative","t":time_or_null,"c":confidence,"e":[{"i":turn_id,"a":start,"b":end}]}],"u":[]}]}. Every fact must cite exact supplied offsets. No invention, markdown, explanations, or reasoning. Summary <=64 tokens. Omit empty optional fields."""
STRICT_PROMPT = """Extract durable facts that help route later questions to exact source turns. Return only schema JSON. Each scene supplies its turns in order, and every turn shows only who spoke, when, and what was said. Fact keys: o=owner entity; p=short canonical verb phrase; v=the concrete value including names and ordinals; g=short domain; n=positive or negative; r=one or two 0-based positions of the cited turns within this scene's turn array; q=one exact quote containing the value and any explicit time. Cover every informative turn before adding a second fact from any turn. Highest priority: quantities and ordinals, named people/places/books/pets, acquisitions and state changes, participation or wins, preferences, explicit or relative time, and factual media-caption details. Preserve words such as first/second/third and numeric caption facts. Never emit generic facts such as shared/showed/sent a photo when the caption supports a more concrete fact. Omit greetings, acknowledgements, emotions without a durable state, advice, and filler. Input d is observation metadata and may only anchor relative time; never copy it as event time. Do not invent, explain, reason, summarize, or create aliases."""
HIERARCHY_PROMPT = """Compress supplied child semantic records for routing. Return compact JSON with summary, owners, predicates, values, scopes, times, child_postings mapping keys to child ID arrays, and aliases as arrays of equivalent value strings found in the children. Only copy supported values and IDs. Never put IDs in scopes or aliases. No markdown or reasoning."""
#: The input protocol labels scenes `s1` and turns `s1t0`, and the model copies
#: those labels into any free-text field it is given.  `scope` has been filtered
#: for this since V5.4; the V5.7 category field `k` leaked them at 2.9%, and the
#: V5.8 scene summary and entity list leaked them at 68.5% and across the entire
#: top of the entity frequency table.  Every new free-text field needs this guard.
ALIAS_RE = re.compile(r"s?\d+(?:t\d+)?", re.I)


def strip_aliases(text: str) -> str:
    """Empty when the model echoed a scene or turn label instead of writing prose."""
    cleaned = " ".join(str(text).split())
    return "" if ALIAS_RE.fullmatch(cleaned) else cleaned


EXPLICIT_TIME_RE = re.compile(
    r"\b(?:\d{1,4}[:/\-]\d{1,2}|\d{4}|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|today|yesterday|tomorrow|last|next|ago|week|month|year)\b", re.I)


def strict_scene_schema(max_scenes: int, max_facts: int, *,
                        quote_evidence: bool = True,
                        predicate_max_chars: int = 0,
                        scene_summary_chars: int = 0,
                        scene_entities: bool = False,
                        atomic_coverage: bool = False,
                        max_information_units: int = 0,
                        max_turn_index: int = 63) -> Mapping[str, Any]:
    """Schema for one strict extraction call.

    ``quote_evidence`` emits the exact-quote field ``q``.  It is 26% of
    extraction output bytes and only narrows the span inside a turn that ``r``
    already cites, which the projection re-derives from the fact value, so a
    budgeted run may drop it.

    """
    properties: dict[str, Any] = {
        "o": {"type": "string", "minLength": 1},
        "p": ({"type": "string", "minLength": 1, "maxLength": predicate_max_chars}
              if predicate_max_chars else {"type": "string", "minLength": 1}),
        "v": {"type": "string", "minLength": 1},
        "g": {"type": "string", "minLength": 1},
        "n": {"type": "string", "enum": ["positive", "negative"]},
        # A 0-based position in the scene's turn array, not a copyable label.
        # The maximum is load-bearing, not decoration: an unbounded integer let
        # guided decoding emit a 32,727-digit number that consumed the whole
        # output budget and failed JSON parsing outright.  Scenes hold
        # `scenes.max_turns` turns, so any index past this bound is already
        # unresolvable.
        "r": {"type": "array", "minItems": 1, "maxItems": 2,
              "items": {"type": "integer", "minimum": 0, "maximum": max_turn_index}}}
    required = ["o", "p", "v", "g", "n", "r"]
    if quote_evidence:
        properties["q"] = {"type": "string", "minLength": 1, "maxLength": 160}
        required.append("q")
    if atomic_coverage:
        # Confidence is part of the strict contract.  Before V5.10 it was named
        # in the legacy prompt but absent from the strict schema, so guided
        # decoding could never emit it and every fact became the same synthetic
        # 0.5 downstream.
        properties["c"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        properties["z"] = {
            "type": "array", "minItems": 1 if max_information_units else 0,
            "maxItems": max(1, max_information_units),
            "items": {"type": "integer", "minimum": 0,
                      "maximum": max(0, max_information_units - 1)},
        }
        required.extend(("c", "z"))
    fact = {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}
    scene_properties: dict[str, Any] = {
        "i": {"type": "string"},
        "f": {"type": "array", "maxItems": max_facts, "items": fact}}
    scene_required = ["i", "f"]
    if scene_summary_chars:
        # A readable standalone sentence, not the concatenation of fact triples
        # the build compiles today ("Obsess has website URL https://obsessvr.com/
        # Vertebrae has website URL ...").  Both mem0's FACT_RETRIEVAL_PROMPT and
        # LightMem's METADATA_GENERATE_PROMPT return standalone sentences, and a
        # sentence is what a question embedding can match against.
        scene_properties["m"] = {"type": "string", "minLength": 1,
                                 "maxLength": scene_summary_chars}
        scene_required.append("m")
    if scene_entities:
        # Named entities carry a multi-session question across sessions: LoCoMo
        # cat1 ("What did Caroline research?") spreads its evidence over 2.68
        # sessions and is the worst-routed category at session_all_hit 0.592.
        scene_properties["e"] = {"type": "array", "maxItems": 8,
                                 "items": {"type": "string", "minLength": 1, "maxLength": 40}}
        scene_required.append("e")
    if atomic_coverage:
        scene_properties["u"] = {
            "type": "array", "maxItems": max(0, max_information_units),
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["i", "r"],
                "properties": {
                    "i": {"type": "integer", "minimum": 0,
                          "maximum": max(0, max_information_units - 1)},
                    "r": {"type": "string", "minLength": 1, "maxLength": 80},
                },
            },
        }
        scene_required.append("u")
    scene = {"type": "object", "additionalProperties": False,
             "required": scene_required, "properties": scene_properties}
    return {"type": "object", "additionalProperties": False, "required": ["s"],
            "properties": {"s": {"type": "array", "minItems": max_scenes, "maxItems": max_scenes,
                                     "items": scene}}}


@dataclass(frozen=True, slots=True)
class SemanticFact:
    owner: str
    predicate: str
    value: str
    value_type: str
    scope: str
    polarity: str
    time: str | None
    confidence: float
    evidence: tuple[tuple[str, int, int], ...]
    information_unit_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ScenePacket:
    scene_id: str
    summary: str
    facts: tuple[SemanticFact, ...]
    unresolved: tuple[str, ...]
    fallback: bool = False
    #: Named entities the scene mentions.  Empty when the config does not ask for
    #: `e`, which is what every graph before V5.8 carries.
    entities: tuple[str, ...] = ()
    information_units: tuple[InformationUnit, ...] = ()
    covered_unit_ids: tuple[int, ...] = ()
    unresolved_unit_ids: tuple[int, ...] = ()
    missing_unit_ids: tuple[int, ...] = ()
    raw_fallback_turn_ids: tuple[str, ...] = ()
    fact_cap: int = 0
    implicitly_covered_unit_ids: tuple[int, ...] = ()

    @property
    def unit_coverage(self) -> float:
        if not self.information_units:
            return 1.0
        accounted = set(self.covered_unit_ids) | set(self.unresolved_unit_ids)
        return len(accounted) / len(self.information_units)


class QwenSemanticDistiller:
    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, client: Any | None = None, *,
                 request_gate: threading.BoundedSemaphore | None = None,
                 worker_limit: int = 16) -> None:
        self.store, self.config, self.dataset_hash = store, config, dataset_hash
        if worker_limit < 1:
            raise ValueError("worker_limit must be positive")
        # A full build creates one distiller per Memory.  Without a process-wide
        # gate, ``memory_workers * 16`` requests can reach vLLM even though
        # ModelConfig.max_concurrency is meant to be the global service limit.
        self.request_gate = request_gate
        self.worker_limit = worker_limit
        strict_prompt = STRICT_PROMPT + (
            f" Return at most {config.models.semantic_max_facts_per_scene} highest-routing-value facts per scene.")
        if config.models.semantic_predicate_max_chars:
            # The schema caps `p` by guided decoding; say so in the prompt too so
            # the model puts the specifics in `v` rather than getting truncated.
            strict_prompt += (
                f" p must be a bare relation of one to three words and at most "
                f"{config.models.semantic_predicate_max_chars} characters, such as visited, bought, "
                "recommends, lives in. Never put objects, names, quantities, adjectives or clauses "
                "in p; those belong in v. Reuse the same p across facts that share a relation.")
        if config.models.semantic_scene_summary_chars:
            # Phrased after mem0's FACT_RETRIEVAL_PROMPT and LightMem's
            # METADATA_GENERATE_PROMPT, both of which return standalone sentences
            # that read without the surrounding turns.  The summary is what a
            # question is matched against, so it has to carry the specifics --
            # LightMem's LoCoMo prompt is explicit that names, places and
            # quantities must survive into the fact text.
            strict_prompt += (
                " m is one standalone sentence stating what this scene is about, readable without "
                "the conversation. Name the people, places, objects, organizations and quantities "
                "involved rather than referring to them as he, she, it or they. State it as prose, "
                "never as a list of subject-predicate-object fragments. Never write a scene or turn "
                "label such as s1 or s1t0 into m, and never copy a turn verbatim: write your own "
                "sentence about what happened.")
        if config.models.semantic_scene_entities:
            strict_prompt += (
                " e lists the named entities the scene mentions -- people, places, organizations, "
                "products, works and events -- each written the same way every time it appears "
                "anywhere in the input, so the same entity in two sessions yields the same string. "
                "Scene and turn labels such as s1 or s1t0 are not entities and must never appear "
                "in e.")
        if config.models.semantic_atomic_coverage:
            strict_prompt += (
                " Each input scene provides k, its fact budget, and compact information-unit entries "
                "u=[unit_id,type,verbatim_surface] beside the source chunk that contains them. Every "
                "unit must be accounted for exactly: put its integer id in z on at least one fact "
                "grounded in that same source turn, or put {i:unit_id,r:short_reason} in scene-level u "
                "when the surface cannot support a durable fact. A fact may cover several units. "
                "c is calibrated confidence in [0,1]: use high confidence only when owner, relation, "
                "value, polarity and citation are explicit. Never mark a unit covered merely because "
                "it was listed in the input. Return no more than the scene's k facts. k is a ceiling, "
                "not a target: never fill it with paraphrases, progressively longer versions of the "
                "same proposition, or relations inferred from a place name. Questions and requests "
                "are not evidence for their presupposed answers: record only that the speaker asked "
                "or wanted something, never invent the requested movie technique, recommendation, "
                "event detail, or explanation. q must be a byte-for-byte substring of one input t, "
                "including its quotation marks; choose a shorter exact clause instead of rewriting it. "
                "For every number_unit, date, duration, negation, or modality id, the linked fact must "
                "preserve that exact surface in p, v, or q. Put dates and durations in v even when d "
                "also supplies an observation anchor. Resolve pronouns to the nearest type-compatible "
                "explicit antecedent; do not default a third-person pronoun to the speaker, and keep "
                "giver, recipient, owner and object roles distinct. Never replace a quantity or object "
                "with a person named in a later question in the same turn.")
        self.strict_prompt = strict_prompt
        strict_version = (ATOMIC_PROMPT_VERSION if config.models.semantic_atomic_coverage
                          else STRICT_PROMPT_VERSION)
        prompt_material = (strict_version + strict_prompt if
                           config.models.semantic_extraction_mode != "legacy_batch" else
                           PROMPT_VERSION + SYSTEM_PROMPT + HIERARCHY_PROMPT)
        self.prompt_hash = hashlib.sha256(prompt_material.encode()).hexdigest()
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
        self.client = client

    def extract(self, memory_id: str, scenes: Sequence[Any]) -> tuple[ScenePacket, ...]:
        batches: list[list[Any]] = []
        current: list[Any] = []
        estimate = 0
        mode = self.config.models.semantic_extraction_mode
        # "strict_batch" keeps the strict schema but honours semantic_batch_scenes,
        # so the measured 316-token fixed cost per call can be amortized over more
        # scenes.  Larger batches push harder against semantic_batch_output_tokens,
        # so this is an ablation knob rather than a default.
        batch_limit = {"strict_single": 1, "strict_pair": 2}.get(
            mode, self.config.models.semantic_batch_scenes)
        for scene in scenes:
            scene_tokens = sum(max(1, len(turn.raw_text.split()) * 13 // 10) for turn in scene.turns)
            if current and (len(current) >= batch_limit
                            or estimate + scene_tokens > self.config.models.semantic_batch_input_tokens):
                batches.append(current); current = []; estimate = 0
            current.append(scene); estimate += scene_tokens
        if current:
            batches.append(current)
        packets: list[ScenePacket] = []
        # One ledger per memory, shared by every worker, so the ceiling holds
        # across the fan-out rather than per call.
        self.ledger = BuildTokenLedger(
            memory_id, self.config.models.semantic_max_tokens_per_memory,
            self.config.models.semantic_budget_degrade_at,
            fallback_on_overrun=self.config.models.semantic_fallback_on_overrun)
        workers = min(self.worker_limit, self.config.models.max_concurrency,
                      max(1, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for rows in executor.map(lambda batch: self._extract_batch(memory_id, batch), batches):
                packets.extend(rows)
        return tuple(packets)

    def _ledger(self) -> BuildTokenLedger:
        """The current memory's ledger, or an unenforced one for direct calls."""
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            ledger = self.ledger = BuildTokenLedger("", 0)
        return ledger

    def _batch_estimate(self, batch: Sequence[Any], max_tokens: int) -> int:
        """Upper bound on what one extraction call will cost.

        Input is the turn text plus the measured 316-token fixed overhead
        (system prompt, JSON schema, payload scaffold); output is bounded by
        ``max_tokens``.  Estimating high is the safe direction: it makes the
        ledger stop early rather than overshoot.
        """
        text = sum(len(turn.raw_text) for scene in batch for turn in scene.turns)
        # Reserve what the call is expected to cost, not what it is allowed to
        # cost.  Those were the same number until the output ceiling was raised
        # to stop truncation, at which point the ceiling silently became the
        # budget and starved extraction.
        expected = self.config.models.semantic_expected_output_tokens or max_tokens
        estimate = int(text / CHARS_PER_TOKEN) + CALL_OVERHEAD_TOKENS + expected
        # CHARS_PER_TOKEN is a corpus mean, so individual calls tokenize both
        # better and worse than it.  A call that costs more than its estimate
        # pushes ``spent`` past the ceiling with no way to take it back, so the
        # reservation carries a margin: overshooting the estimate is recoverable,
        # overshooting the budget is not.
        return int(estimate * ESTIMATE_SAFETY)

    def compress_many(self, memory_id: str, level: int,
                      requests: Sequence[tuple[str, Sequence[Mapping[str, Any]], int]]) -> tuple[Mapping[str, Any], ...]:
        workers = min(self.worker_limit, self.config.models.max_concurrency,
                      max(1, len(requests)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(executor.map(
                lambda row: self.compress(memory_id, level, row[0], row[1], row[2]), requests
            ))

    def compress(self, memory_id: str, level: int, parent_id: str,
                 children: Sequence[Mapping[str, Any]], limit: int) -> Mapping[str, Any]:
        payload = {"level": level, "parent_id": parent_id, "children": list(children), "summary_limit": limit}
        response, _ = self._call(memory_id, f"hierarchy_l{level}", HIERARCHY_PROMPT, payload,
                                 len(children), max_tokens=1536 if level == 3 else 768)
        parsed = self._parse_objects(str(response.get("content", "")))
        row = parsed[0] if parsed else {}
        summary = str(row.get("summary", "")).strip()
        if not summary:
            summary = " ".join(str(child.get("summary", "")) for child in children)
        valid_ids = {str(child["child_id"]) for child in children}
        postings = {}
        for key, ids in dict(row.get("child_postings", {})).items() if isinstance(row.get("child_postings"), dict) else ():
            if isinstance(ids, list):
                postings[str(key)] = tuple(item for item in map(str, ids) if item in valid_ids)
        available_values = {str(value) for child in children for value in child.get("values", ())}
        aliases = []
        for group in row.get("aliases", ()) if isinstance(row.get("aliases"), list) else ():
            if isinstance(group, list):
                valid = tuple(dict.fromkeys(str(value) for value in group if str(value) in available_values))
                if len(valid) >= 2: aliases.append(valid)
        return {
            "summary": " ".join(summary.split()[:limit]),
            "owners": self._strings(row.get("owners")), "predicates": self._strings(row.get("predicates")),
            "values": self._strings(row.get("values")), "scopes": self._strings(row.get("scopes")),
            "times": self._strings(row.get("times")), "child_postings": postings,
            "aliases": tuple(aliases),
        }

    def _extract_batch(self, memory_id: str, batch: Sequence[Any]) -> list[ScenePacket]:
        if self.config.models.semantic_extraction_mode != "legacy_batch":
            return self._extract_strict_batch(memory_id, batch)
        # The ceiling must hold in every extraction mode; a budget that silently
        # does not apply to one path is worse than no budget at all.
        output_cap = self.config.models.semantic_batch_output_tokens
        estimate = self._batch_estimate(batch, output_cap)
        if not self._ledger().reserve(estimate)[0]:
            return [self._fallback(scene) for scene in batch]
        payload = {"s": [{"i": scene.scene_id, "r": [
            {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp, "l": len(turn.raw_text),
             "t": turn.raw_text}
            for turn in scene.turns]} for scene in batch]}
        system = self._scene_prompt()
        response, usage = self._call(memory_id, "scene_semantic", system, payload, len(batch))
        self._ledger().settle(estimate, int(usage.get("total_tokens", 0)))
        objects = self._parse_objects(str(response.get("content", "")))
        root = objects[0] if objects else {}
        rows = root.get("s", root.get("scenes", [])) if isinstance(root, dict) else []
        if not isinstance(rows, list):
            rows = []
        by_scene = {str(row.get("i", row.get("scene_id"))): row for row in rows if isinstance(row, dict)}
        missing = [scene for scene in batch if scene.scene_id not in by_scene]
        if missing:
            repair_batches = [[scene] for scene in missing] if self.config.models.semantic_individual_repair else [missing]
            repair_cap = self.config.models.semantic_repair_output_tokens
            for repair_batch in repair_batches:
                repair_estimate = self._batch_estimate(repair_batch, repair_cap)
                if not self._ledger().reserve(repair_estimate)[0]:
                    continue
                repair_payload = {"repair": "Return complete valid JSON for only these missing scenes.",
                "s": [{"i": scene.scene_id, "r": [
                    {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp,
                     "l": len(turn.raw_text), "t": turn.raw_text}
                    for turn in scene.turns]} for scene in repair_batch]}
                repaired, repair_usage = self._call(memory_id, "scene_semantic_repair", system,
                    repair_payload, len(repair_batch), max_tokens=repair_cap)
                self._ledger().settle(repair_estimate,
                                      int(repair_usage.get("total_tokens", 0)))
                repaired_objects = self._parse_objects(str(repaired.get("content", "")))
                repaired_root = repaired_objects[0] if repaired_objects else {}
                for item in repaired_root.get("s", repaired_root.get("scenes", ())) if isinstance(repaired_root, dict) else ():
                    if isinstance(item, dict):
                        by_scene[str(item.get("i", item.get("scene_id")))] = item
        result = []
        for scene in batch:
            row = by_scene.get(scene.scene_id)
            result.append(self._validate_scene(scene, row) if row else self._fallback(scene))
        return result

    def _extract_strict_batch(self, memory_id: str, batch: Sequence[Any]) -> list[ScenePacket]:
        units_by_scene = {
            scene.scene_id: (scan_information_units(scene.turns)
                             if self.config.models.semantic_atomic_coverage else ())
            for scene in batch
        }
        fact_caps = {scene.scene_id: self._scene_fact_cap(units_by_scene[scene.scene_id])
                     for scene in batch}
        payload, scene_aliases, turn_aliases = self._strict_payload(
            batch, units_by_scene, fact_caps)
        output_cap = self.config.models.semantic_batch_output_tokens
        allowed, degrade = self._ledger().reserve(self._batch_estimate(batch, output_cap))
        if not allowed:
            # Budget exhausted: keep the scenes in the graph with deterministic
            # summaries rather than dropping the tail of the conversation.
            return [self._fallback(scene, units_by_scene[scene.scene_id],
                                   fact_caps[scene.scene_id]) for scene in batch]
        max_facts = max(fact_caps.values(), default=self.config.models.semantic_max_facts_per_scene)
        if degrade:
            fact_caps = {key: max(1, value // 2) for key, value in fact_caps.items()}
            max_facts = max(fact_caps.values())
            payload, scene_aliases, turn_aliases = self._strict_payload(
                batch, units_by_scene, fact_caps)
        strict_base = (self.strict_prompt if self.config.models.semantic_atomic_coverage
                       else STRICT_PROMPT)
        strict_prompt = strict_base + (
            f" This call's schema permits at most {max_facts} facts per scene; obey each scene's k.")
        response, usage = self._call(
            memory_id, "scene_semantic", strict_prompt, payload, len(batch),
            max_facts=max_facts,
            max_information_units=max(
                (len(value) for value in units_by_scene.values()), default=0),
            max_turn_index=max(
                (len(value) for value in turn_aliases.values()), default=1) - 1)
        self._ledger().settle(self._batch_estimate(batch, output_cap),
                              int(usage.get("total_tokens", 0)))
        rows = self._strict_rows(response, scene_aliases, turn_aliases)
        by_scene = {str(row.get("i")): row for row in rows}
        packets = {
            scene.scene_id: (
                self._validate_scene(
                    scene, by_scene[scene.scene_id],
                    units=units_by_scene[scene.scene_id],
                    fact_cap=fact_caps[scene.scene_id])
                if scene.scene_id in by_scene else
                self._fallback(scene, units_by_scene[scene.scene_id],
                               fact_caps[scene.scene_id]))
            for scene in batch
        }
        retry_scenes = [
            scene for scene in batch
            if (scene.scene_id not in by_scene
                or (self.config.models.semantic_atomic_coverage
                    and packets[scene.scene_id].unit_coverage
                    < self.config.models.semantic_min_unit_coverage))
        ]
        if retry_scenes and self.config.models.semantic_max_retries:
            retry_cap = self.config.models.semantic_retry_output_tokens
            for scene in retry_scenes:
                # Retries must be ledgered too.  Leaving them out let a memory
                # finish at 221,305 tokens against a 220,000 ceiling: ~11 retry
                # calls per memory were spending outside the budget entirely.
                retry_estimate = self._batch_estimate((scene,), retry_cap)
                if not self._ledger().reserve(retry_estimate)[0]:
                    continue
                retry_payload, retry_aliases, retry_turns = self._strict_payload(
                    (scene,), units_by_scene,
                    {scene.scene_id: fact_caps[scene.scene_id]})
                if self.config.models.semantic_atomic_coverage:
                    current = packets[scene.scene_id]
                    retry_payload = dict(retry_payload)
                    retry_payload["repair"] = {
                        "instruction": "Return one complete replacement scene; cover or reject every unit.",
                        "missing_unit_ids": list(current.missing_unit_ids),
                    }
                repaired, retry_usage = self._call(
                    memory_id, "scene_semantic_retry", strict_prompt, retry_payload, 1,
                    max_tokens=retry_cap, retry_count=1,
                    max_facts=fact_caps[scene.scene_id],
                    max_information_units=len(units_by_scene[scene.scene_id]),
                    max_turn_index=max(0, len(retry_turns.get("0", ())) - 1))
                self._ledger().settle(retry_estimate, int(retry_usage.get("total_tokens", 0)))
                for row in self._strict_rows(repaired, retry_aliases, retry_turns):
                    candidate = self._validate_scene(
                        scene, row, units=units_by_scene[scene.scene_id],
                        fact_cap=fact_caps[scene.scene_id])
                    previous = packets[scene.scene_id]
                    candidate_score = (
                        -len(candidate.missing_unit_ids), len(candidate.covered_unit_ids),
                        len(candidate.facts))
                    previous_score = (
                        -len(previous.missing_unit_ids), len(previous.covered_unit_ids),
                        len(previous.facts))
                    if candidate_score >= previous_score:
                        packets[scene.scene_id] = candidate
        return [packets[scene.scene_id] for scene in batch]

    def _scene_fact_cap(self, units: Sequence[InformationUnit]) -> int:
        base = self.config.models.semantic_max_facts_per_scene
        if not self.config.models.semantic_adaptive_fact_cap:
            return base
        return adaptive_fact_cap(
            units,
            floor=base,
            ceiling=self.config.models.semantic_adaptive_fact_cap_max,
            alpha=self.config.models.semantic_fact_cap_alpha,
            beta=self.config.models.semantic_fact_cap_beta,
            gamma=self.config.models.semantic_fact_cap_gamma,
        )

    def _strict_payload(
        self,
        batch: Sequence[Any],
        units_by_scene: Mapping[str, Sequence[InformationUnit]] | None = None,
        fact_caps: Mapping[str, int] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, str], Mapping[str, Sequence[Any]]]:
        scene_aliases: dict[str, str] = {}
        turn_aliases: dict[str, Sequence[Any]] = {}
        rows = []
        # A turn shows the model three things: who spoke, when, and what was said.
        # Everything else is our plumbing, and plumbing that is shaped like prose
        # gets copied into prose: string aliases ("s0", "s0t1") were echoed back
        # as 68.5% of scene summaries and as the six most frequent "entities" in
        # the corpus.  Citations are therefore positions in the arrays we already
        # send -- an integer cannot be mistaken for a summary or an entity name,
        # and it costs fewer output tokens than "s0t1".
        for scene_index, scene in enumerate(batch):
            scene_alias = str(scene_index); scene_aliases[scene_alias] = scene.scene_id
            turns = []
            aliases = []
            scene_units = tuple((units_by_scene or {}).get(scene.scene_id, ()))
            for turn in scene.turns:
                if (self.config.models.semantic_sentence_chunking
                        and self.config.models.semantic_turn_input_chars):
                    chunks = sentence_chunks(
                        turn.turn_id, turn.raw_text,
                        self.config.models.semantic_turn_input_chars,
                        tuple((unit.start, unit.end) for unit in scene_units
                              if unit.turn_id == turn.turn_id))
                else:
                    compacted = self._compact_turn(turn.raw_text)
                    chunks = sentence_chunks(turn.turn_id, compacted, 0)
                for chunk in chunks:
                    item: dict[str, Any] = {
                        "s": turn.speaker, "d": turn.timestamp, "t": chunk.text}
                    if self.config.models.semantic_atomic_coverage:
                        item["u"] = [
                            [unit.unit_id, unit.kind, unit.text]
                            for unit in units_for_span(
                                scene_units, turn.turn_id, chunk.start, chunk.end)
                        ]
                    turns.append(item)
                    aliases.append((turn.turn_id, chunk.start, chunk.end))
            # Citations resolve against this scene's own turn order, so the model
            # never needs -- and never sees -- an identifier it could copy.
            turn_aliases[scene_alias] = tuple(aliases)
            scene_row: dict[str, Any] = {"i": scene_alias, "r": turns}
            if self.config.models.semantic_atomic_coverage:
                scene_row["k"] = int((fact_caps or {}).get(
                    scene.scene_id, self._scene_fact_cap(scene_units)))
            rows.append(scene_row)
        return {"s": rows}, scene_aliases, turn_aliases

    def _strict_rows(self, response: Mapping[str, Any], scene_aliases: Mapping[str, str],
                     turn_aliases: Mapping[str, Sequence[Any]]) -> list[Mapping[str, Any]]:
        content = str(response.get("content", ""))
        objects = self._parse_objects(content)
        root = objects[0] if objects else {}
        rows = root.get("s", ()) if isinstance(root, dict) else ()
        if not isinstance(rows, list) or not rows:
            rows = self._parse_complete_scene_rows(content)
        result = []
        source_rows = rows if isinstance(rows, list) else ()
        expected_aliases = tuple(scene_aliases)
        raw_aliases = tuple(str(row.get("i")) for row in source_rows if isinstance(row, dict))
        aliases_are_permutation = (len(raw_aliases) == len(expected_aliases)
                                   and len(set(raw_aliases)) == len(expected_aliases)
                                   and set(raw_aliases) == set(expected_aliases))
        used_aliases: set[str] = set()
        for position, source in enumerate(source_rows):
            if not isinstance(source, dict):
                continue
            alias = str(source.get("i"))
            # Guided JSON guarantees the array shape but cannot express that each
            # scene alias is unique. Qwen occasionally emits s0 twice for a
            # two-scene batch while preserving row order. Recover that row
            # deterministically instead of paying for a redundant single-scene
            # retry. Positional recovery is only safe at exact cardinality.
            if (not aliases_are_permutation and len(source_rows) == len(expected_aliases)
                    and position < len(expected_aliases)):
                alias = expected_aliases[position]
            if alias not in scene_aliases:
                continue
            used_aliases.add(alias)
            row = dict(source); row["i"] = scene_aliases[alias]
            facts = []
            for source_fact in source.get("f", ()) if isinstance(source.get("f"), list) else ():
                if not isinstance(source_fact, dict):
                    continue
                fact = dict(source_fact); refs = []
                quote = " ".join(str(source_fact.get("q", "")).split())
                # `r` is a 0-based position in this scene's turn array.  A model
                # that answers with a 1-based index or a stray string simply
                # fails to resolve, exactly as an unknown alias used to.
                scene_turns = turn_aliases.get(alias, ())
                for cited in source_fact.get("r", ()) if isinstance(source_fact.get("r"), list) else ():
                    try:
                        index = int(cited)
                    except (TypeError, ValueError):
                        continue
                    if not 0 <= index < len(scene_turns):
                        continue
                    target = scene_turns[index]
                    if isinstance(target, (list, tuple)) and len(target) >= 3:
                        turn_id, chunk_start, chunk_end = target[:3]
                        refs.append({"i": turn_id, "q": quote,
                                     "h": chunk_start, "j": chunk_end})
                    else:  # Backward-compatible direct tests and cached rows.
                        refs.append({"i": target, "q": quote})
                fact["e"] = refs; facts.append(fact)
            row["f"] = facts; result.append(row)
        return result

    def _compact_turn(self, text: str) -> str:
        limit = int(self.config.models.semantic_turn_input_chars)
        if not limit or len(text) <= limit:
            return text
        head = int(limit * 0.72)
        tail = limit - head
        return text[:head].rstrip() + "\n[...middle omitted...]\n" + text[-tail:].lstrip()

    @staticmethod
    def _parse_complete_scene_rows(text: str) -> list[Mapping[str, Any]]:
        """Recover complete scene objects from a length-truncated JSON root."""
        rows = []
        for match in re.finditer(r'\{\s*"i"\s*:', text):
            start = match.start(); depth = 0; quoted = False; escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                    continue
                if char == '"':
                    quoted = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            row = json.loads(text[start:index + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(row, dict) and isinstance(row.get("f"), list):
                            rows.append(row)
                        break
        return rows

    def _validate_scene(
        self,
        scene: Any,
        row: Mapping[str, Any],
        *,
        units: Sequence[InformationUnit] = (),
        fact_cap: int | None = None,
    ) -> ScenePacket:
        turns = {turn.turn_id: turn for turn in scene.turns}
        facts = []
        fact_rows = row.get("f", row.get("facts", ()))
        cap = fact_cap or self.config.models.semantic_max_facts_per_scene
        valid_unit_ids = {unit.unit_id for unit in units}
        for item in (fact_rows[:cap]
                     if isinstance(fact_rows, list) else ()):
            if not isinstance(item, dict):
                continue
            evidence = []
            for ref in item.get("e", item.get("evidence", ())):
                try:
                    turn = turns[str(ref.get("i", ref.get("turn_id")))]
                except (KeyError, TypeError):
                    continue
                quote = " ".join(str(ref.get("q", "")).split())
                if quote:
                    try:
                        hint = max(0, int(ref.get("h", 0)))
                        chunk_end = min(len(turn.raw_text), int(ref.get("j", len(turn.raw_text))))
                    except (TypeError, ValueError):
                        hint, chunk_end = 0, len(turn.raw_text)
                    relative = turn.raw_text[hint:chunk_end].casefold().find(quote.casefold())
                    start = hint + relative if relative >= 0 else -1
                    end = start + len(quote) if start >= 0 else -1
                elif ref.get("a") is not None or ref.get("start") is not None:
                    try:
                        start = int(ref.get("a", ref.get("start")))
                        end = int(ref.get("b", ref.get("end")))
                    except (TypeError, ValueError):
                        continue
                else:
                    # Quote-free extraction: locate the value inside the turn the
                    # fact already cites.  Falling through to the whole-memory
                    # value scan below would keep the fact but lose the turn it
                    # was actually grounded in.
                    value_text = str(item.get("v", item.get("value", ""))).strip()
                    start = turn.raw_text.casefold().find(value_text.casefold()) if value_text else -1
                    end = start + len(value_text) if start >= 0 else -1
                if 0 <= start < end <= len(turn.raw_text):
                    evidence.append((turn.turn_id, start, end))
            owner = str(item.get("o", item.get("owner", ""))).strip()
            predicate = str(item.get("p", item.get("predicate", ""))).strip()
            value = str(item.get("v", item.get("value", ""))).strip()
            if not evidence or not owner or not predicate or not value:
                if owner and predicate and value:
                    for turn in turns.values():
                        start = turn.raw_text.casefold().find(value.casefold())
                        if start >= 0:
                            evidence.append((turn.turn_id, start, start + len(value))); break
                if not evidence or not owner or not predicate or not value:
                    continue
            cited_turns = [turns[turn_id] for turn_id, _, _ in evidence]
            if owner.casefold() in {"user", "assistant", "speaker", "questioner"} and cited_turns:
                owner = cited_turns[0].speaker
            scope = str(item.get("g", item.get("scope", "scene"))).strip()
            if re.fullmatch(r"s?\d+t\d+|t\d+", scope.casefold()):
                scope = "scene"
            # Event time is derived from the exact cited quote downstream.  The
            # observation timestamp remains available as its normalization
            # anchor but is deliberately not generated by the model.
            raw_time = None
            value_type = str(item.get("y", item.get("value_type", ""))).strip() \
                or self._infer_value_type(value)
            raw_confidence = item.get("c", item.get("confidence"))
            if self.config.models.semantic_atomic_coverage and raw_confidence is None:
                # A missing confidence is a schema/contract violation, not a
                # calibrated 0.5.  Silently manufacturing one made confidence
                # unusable for scheduling and certification in every strict
                # graph before V5.10.
                continue
            try:
                confidence = float(0.5 if raw_confidence is None else raw_confidence)
            except (TypeError, ValueError):
                continue
            if not 0.0 <= confidence <= 1.0:
                continue
            information_unit_ids = tuple(dict.fromkeys(
                int(unit_id) for unit_id in item.get("z", ())
                if isinstance(unit_id, int) and int(unit_id) in valid_unit_ids
            ))
            facts.append(SemanticFact(owner, predicate, value, value_type,
                                      scope,
                                      str(item.get("n", item.get("polarity", "positive"))),
                                      raw_time,
                                      confidence,
                                      tuple(evidence), information_unit_ids))
        facts = self._dedup_facts(facts)
        summary = " ".join(strip_aliases(row.get("m", row.get("summary", ""))).split()[
            :self.config.models.semantic_summary_tokens])
        # `semantic_compile_summary` replaces the model's sentence with a
        # concatenation of fact triples.  That is what routing cards are built
        # from today, and it is why they read as duplicated term soup
        # ("user track spending on luxury items vs. budget-friendly ones luxury
        # items vs. budget-friendly ones").  When the model is asked for a real
        # summary, keep it.
        if self.config.models.semantic_compile_summary and not (
                self.config.models.semantic_scene_summary_chars and summary):
            summary = self._compiled_summary(facts)
        if not summary:
            summary = " ".join(scene.summary.split()[:96])
        entities = tuple(dict.fromkeys(
            filter(None, (strip_aliases(item) for item in (row.get("e") or ())))))[:8]
        unresolved_rows = row.get("u", row.get("unresolved", ()))
        unresolved: list[str] = []
        unresolved_ids: list[int] = []
        if isinstance(unresolved_rows, list):
            for item in unresolved_rows:
                if isinstance(item, dict):
                    try:
                        unit_id = int(item.get("i"))
                    except (TypeError, ValueError):
                        continue
                    reason = " ".join(str(item.get("r", "unresolved")).split())[:80]
                    if unit_id in valid_unit_ids and unit_id not in unresolved_ids:
                        unresolved_ids.append(unit_id)
                        unresolved.append(f"{unit_id}:{reason}")
                elif str(item).strip():
                    unresolved.append(str(item).strip())

        units_by_id = {unit.unit_id: unit for unit in units}
        covered: list[int] = []
        for fact in facts:
            fact_text = self._normal_unit_text(
                f"{fact.owner} {fact.predicate} {fact.value} {fact.scope}")
            for unit_id in fact.information_unit_ids:
                unit = units_by_id[unit_id]
                grounded = False
                for turn_id, start, end in fact.evidence:
                    if turn_id != unit.turn_id:
                        continue
                    overlaps = max(start, unit.start) < min(end, unit.end)
                    unit_in_fact = self._normal_unit_text(unit.text) in fact_text
                    if overlaps or unit_in_fact:
                        grounded = True
                        break
                if grounded and unit_id not in covered:
                    covered.append(unit_id)
        explicitly_covered = set(covered)
        implicitly_covered: list[int] = []
        # z is a compact model-produced accounting link, not the source of
        # truth.  Independently recover a missed link when a grounded fact
        # literally preserves the unit surface.  This makes the verifier robust
        # to an otherwise correct fact omitting one id, while still exposing the
        # omission as ``implicitly_covered_unit_ids``.
        for unit in units:
            if unit.unit_id in explicitly_covered:
                continue
            unit_text = self._normal_unit_text(unit.text)
            if not unit_text:
                continue
            needle = f" {unit_text} "
            for fact in facts:
                if not any(turn_id == unit.turn_id for turn_id, _, _ in fact.evidence):
                    continue
                fact_text = " " + self._normal_unit_text(
                    f"{fact.owner} {fact.predicate} {fact.value} {fact.scope}") + " "
                if needle in fact_text:
                    covered.append(unit.unit_id)
                    implicitly_covered.append(unit.unit_id)
                    break
        # A unit explicitly covered by a fact wins over an unresolved entry.
        unresolved_ids = [unit_id for unit_id in unresolved_ids if unit_id not in covered]
        unresolved = [item for item in unresolved
                      if not item.split(":", 1)[0].isdigit()
                      or int(item.split(":", 1)[0]) not in covered]
        missing = sorted(valid_unit_ids - set(covered) - set(unresolved_ids))
        raw_fallback_ids = set(missing) | set(unresolved_ids)
        raw_fallback_turns = tuple(dict.fromkeys(
            unit.turn_id for unit in units if unit.unit_id in raw_fallback_ids
        )) if self.config.models.semantic_raw_fallback_on_low_coverage else ()
        return ScenePacket(
            scene.scene_id, summary, tuple(facts), tuple(unresolved), False,
            entities, tuple(units), tuple(sorted(covered)), tuple(unresolved_ids),
            tuple(missing), raw_fallback_turns, cap,
            tuple(sorted(implicitly_covered)))

    @staticmethod
    def _normal_unit_text(value: str) -> str:
        return " ".join(re.findall(r"[\w$%£€¥.-]+", value.casefold()))

    @staticmethod
    def _dedup_facts(facts: Sequence[SemanticFact]) -> list[SemanticFact]:
        merged: dict[tuple[Any, ...], SemanticFact] = {}
        order: list[tuple[Any, ...]] = []
        for fact in facts:
            key = (
                " ".join(fact.owner.casefold().split()),
                " ".join(fact.predicate.casefold().replace("_", " ").split()),
                " ".join(fact.value.casefold().split()),
                fact.polarity.casefold(), fact.evidence,
            )
            if key not in merged:
                merged[key] = fact
                order.append(key)
                continue
            previous = merged[key]
            merged[key] = replace(
                previous,
                confidence=max(previous.confidence, fact.confidence),
                information_unit_ids=tuple(dict.fromkeys(
                    previous.information_unit_ids + fact.information_unit_ids)),
            )
        return [merged[key] for key in order]

    @staticmethod
    def _compiled_summary(facts: Sequence[SemanticFact]) -> str:
        # One scene routinely yields the same triple twice -- the model restates
        # a fact it already emitted -- and the summary is capped at 48 words, so
        # a repeat costs the words a distinct fact would have used.  Measured on
        # the B1 arm graph, 24.9% of the words in a scene summary and 35.1% of
        # the words in a routing card were duplicates.  Deduplicate whole facts,
        # not words: dropping a repeated *word* would shred the phrasing.
        parts, seen = [], set()
        for fact in facts:
            text = f"{fact.owner} {fact.predicate} {fact.value}"
            if fact.time:
                text += f" {fact.time}"
            key = " ".join(text.split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.extend(text.split())
        return " ".join(parts[:48])

    @staticmethod
    def _infer_value_type(value: str) -> str:
        normalized = " ".join(value.split()).casefold()
        if re.search(r"(?:[$£€¥]\s*\d|\d\s*(?:usd|dollars?|euros?|pounds?|yen)\b)", normalized):
            return "currency"
        if re.fullmatch(r"(?:yes|no|true|false)", normalized):
            return "boolean"
        if re.search(r"(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|monday|tuesday|wednesday|thursday|"
                     r"friday|saturday|sunday|today|yesterday|tomorrow|\b(?:week|month|year)s?\b)",
                     normalized):
            return "time"
        if re.search(r"\d", normalized):
            return "number"
        return "text"

    def _scene_prompt(self) -> str:
        if (self.config.models.semantic_max_facts_per_scene == 12
                and self.config.models.semantic_summary_tokens == 64):
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + (
            f" Hard limits: at most {self.config.models.semantic_max_facts_per_scene} facts per scene; "
            f"summary at most {self.config.models.semantic_summary_tokens} tokens. Prioritize state, event, "
            "time, preference and cross-session routing facts; omit greetings and acknowledgements."
        )

    @staticmethod
    def _fallback(
        scene: Any,
        units: Sequence[InformationUnit] = (),
        fact_cap: int = 0,
    ) -> ScenePacket:
        return ScenePacket(
            scene.scene_id, " ".join(scene.summary.split()[:96]), (),
            ("llm_parse_fallback",), True, (), tuple(units), (), (),
            tuple(unit.unit_id for unit in units),
            tuple(dict.fromkeys(unit.turn_id for unit in units)), fact_cap)

    def _call(self, memory_id: str, stage: str, system: str, payload: Any,
              batch_size: int, max_tokens: int | None = None,
              retry_count: int = 0,
              max_facts: int | None = None,
              max_information_units: int = 0,
              max_turn_index: int | None = None) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        serialized = canonical_json(payload)
        semantic_settings = {
            "schema": self.config.schema_version, "model": self.config.models.llm_model,
            "batch_scenes": self.config.models.semantic_batch_scenes,
            "batch_input": self.config.models.semantic_batch_input_tokens,
            "scene_input": self.config.models.semantic_scene_input_tokens,
            "turn_input_chars": self.config.models.semantic_turn_input_chars,
            "batch_output": self.config.models.semantic_batch_output_tokens,
        }
        if self.config.models.semantic_extraction_mode != "legacy_batch":
            semantic_settings.update({
                "extraction_mode": self.config.models.semantic_extraction_mode,
                "max_retries": self.config.models.semantic_max_retries,
                "retry_output": self.config.models.semantic_retry_output_tokens,
                "compile_summary": self.config.models.semantic_compile_summary,
            })
        if self.config.models.semantic_atomic_coverage:
            semantic_settings.update({
                "atomic_coverage": True,
                "adaptive_fact_cap": self.config.models.semantic_adaptive_fact_cap,
                "adaptive_fact_cap_max": self.config.models.semantic_adaptive_fact_cap_max,
                "fact_cap_coefficients": (
                    self.config.models.semantic_fact_cap_alpha,
                    self.config.models.semantic_fact_cap_beta,
                    self.config.models.semantic_fact_cap_gamma,
                ),
                "min_unit_coverage": self.config.models.semantic_min_unit_coverage,
                "sentence_chunking": self.config.models.semantic_sentence_chunking,
                "max_information_units": max_information_units,
                "max_turn_index": max_turn_index,
            })
        if (self.config.models.semantic_max_facts_per_scene,
                self.config.models.semantic_summary_tokens,
                self.config.models.semantic_repair_output_tokens,
                self.config.models.semantic_constrained_json,
                self.config.models.semantic_individual_repair) != (12, 64, 4096, False, False):
            semantic_settings.update({
                "max_facts": self.config.models.semantic_max_facts_per_scene,
                "summary_tokens": self.config.models.semantic_summary_tokens,
                "repair_output": self.config.models.semantic_repair_output_tokens,
                "constrained_json": self.config.models.semantic_constrained_json,
                "individual_repair": self.config.models.semantic_individual_repair,
            })
        # A degraded call asks for fewer facts, and a quote-free run emits a
        # different schema; both must key differently or a throttled answer would
        # be served from a full-budget cache entry.
        effective_facts = (self.config.models.semantic_max_facts_per_scene
                           if max_facts is None else max_facts)
        if effective_facts != self.config.models.semantic_max_facts_per_scene:
            semantic_settings["degraded_max_facts"] = effective_facts
        if not self.config.models.semantic_quote_evidence:
            semantic_settings["quote_evidence"] = False
        if self.config.models.semantic_predicate_max_chars:
            semantic_settings["predicate_max_chars"] = self.config.models.semantic_predicate_max_chars
        semantic_config = hashlib.sha256(canonical_json(semantic_settings).encode()).hexdigest()
        identity = CacheIdentity(self.dataset_hash, self.config.models.llm_model, self.prompt_hash,
                                 self.config.schema_version, semantic_config,
                                 stage + ":" + hashlib.sha256(serialized.encode()).hexdigest())
        key = identity.key(); request = {"model": self.config.models.llm_model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": serialized}],
            "temperature": 0, "max_tokens": max_tokens or self.config.models.semantic_batch_output_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        if self.config.models.semantic_extraction_mode != "legacy_batch":
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "graphmem_scene_facts", "strict": True,
                "schema": strict_scene_schema(
                    batch_size, effective_facts,
                    quote_evidence=self.config.models.semantic_quote_evidence,
                    predicate_max_chars=self.config.models.semantic_predicate_max_chars,
                    scene_summary_chars=self.config.models.semantic_scene_summary_chars,
                    scene_entities=self.config.models.semantic_scene_entities,
                    atomic_coverage=self.config.models.semantic_atomic_coverage,
                    max_information_units=max_information_units,
                    max_turn_index=(max(0, self.config.scenes.max_turns - 1)
                                    if max_turn_index is None else max(0, max_turn_index)))}}
        elif self.config.models.semantic_constrained_json:
            request["response_format"] = {"type": "json_object"}
        cached = self.store.cache_get(key); started = time.perf_counter()
        if cached:
            response = cached["response"]; old = cached["usage"]
            usage = {"cached_input_tokens": int(old.get("uncached_input_tokens", 0)),
                     "uncached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                     "total_tokens": int(old.get("uncached_input_tokens", 0))}
            is_cached = True
        else:
            if self.request_gate is not None:
                self.request_gate.acquire()
            try:
                result = self.client.chat.completions.create(**request)
            finally:
                if self.request_gate is not None:
                    self.request_gate.release()
            message = result.choices[0].message
            if getattr(message, "reasoning_content", None):
                raise RuntimeError("semantic distillation returned reasoning content")
            choice = result.choices[0]
            response = {"content": message.content or "", "model": getattr(result, "model", ""),
                        "finish_reason": getattr(choice, "finish_reason", None)}
            raw = getattr(result, "usage", None); prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
            output = int(getattr(raw, "completion_tokens", 0) or 0)
            usage = {"cached_input_tokens": 0, "uncached_input_tokens": prompt,
                     "output_tokens": output, "reasoning_tokens": 0, "total_tokens": prompt + output}
            self.store.cache_put(key, stage, request, response, usage, self.prompt_hash); is_cached = False
        occurrence = self.store._read_one(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?", (memory_id, key))[0]
        self.store.log_llm_call(call_id=stable_id("llm-call", memory_id, key, is_cached, occurrence),
            memory_id=memory_id, stage=stage, cache_key=key, cached=is_cached, request=request,
            response=response, usage=usage, latency_ms=(time.perf_counter()-started)*1000,
            retry_count=retry_count, batch_size=batch_size, prompt_hash=self.prompt_hash)
        return response, usage

    @staticmethod
    def _strings(value: Any) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip())) if isinstance(value, list) else ()

    @staticmethod
    def _parse_objects(text: str) -> list[Mapping[str, Any]]:
        clean = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            row = json.loads(clean)
            return [row] if isinstance(row, dict) else [item for item in row if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            decoder = json.JSONDecoder(); rows = []; index = 0
            while index < len(clean):
                while index < len(clean) and clean[index] in " \r\n\t,": index += 1
                try: row, index = decoder.raw_decode(clean, index)
                except json.JSONDecodeError: break
                if isinstance(row, dict): rows.append(row)
            return rows
