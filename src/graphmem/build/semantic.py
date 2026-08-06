from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config, config_hash
from ..domain import SourceTurn, canonical_json, stable_id
from ..storage import SQLiteGraphStore
from .budget import BuildTokenLedger


#: Measured fixed input cost of one extraction call on the frozen V5.4 build:
#: system prompt + JSON schema + payload scaffold, over 4,000 sampled calls.
CALL_OVERHEAD_TOKENS = 316
#: Measured characters per token for this corpus under the Qwen3 vocabulary.
CHARS_PER_TOKEN = 3.84

PROMPT_VERSION = "graphmem-v5.1-scene-semantic-v1"
STRICT_PROMPT_VERSION = "graphmem-v5.4-strict-scene-facts-v5"
SYSTEM_PROMPT = """Extract grounded memory facts. Return compact JSON {"s":[{"i":scene_id,"m":summary,"f":[{"o":owner,"p":predicate,"v":value,"y":value_type,"g":scope,"n":"positive|negative","t":time_or_null,"c":confidence,"e":[{"i":turn_id,"a":start,"b":end}]}],"u":[]}]}. Every fact must cite exact supplied offsets. No invention, markdown, explanations, or reasoning. Summary <=64 tokens. Omit empty optional fields."""
STRICT_PROMPT = """Extract durable facts that help route later questions to exact source turns. Return only schema JSON and copy local aliases exactly. Fact keys: o=owner entity; p=short canonical verb phrase; v=the concrete value including names and ordinals; g=short domain; n=positive or negative; r=one or two cited turn aliases; q=one exact quote containing the value and any explicit time. Cover every informative turn before adding a second fact from any turn. Highest priority: quantities and ordinals, named people/places/books/pets, acquisitions and state changes, participation or wins, preferences, explicit or relative time, and factual media-caption details. Preserve words such as first/second/third and numeric caption facts. Never emit generic facts such as shared/showed/sent a photo when the caption supports a more concrete fact. Omit greetings, acknowledgements, emotions without a durable state, advice, and filler. Input d is observation metadata and may only anchor relative time; never copy it as event time. Do not invent, explain, reason, summarize, or create aliases."""
HIERARCHY_PROMPT = """Compress supplied child semantic records for routing. Return compact JSON with summary, owners, predicates, values, scopes, times, child_postings mapping keys to child ID arrays, and aliases as arrays of equivalent value strings found in the children. Only copy supported values and IDs. Never put IDs in scopes or aliases. No markdown or reasoning."""
EXPLICIT_TIME_RE = re.compile(
    r"\b(?:\d{1,4}[:/\-]\d{1,2}|\d{4}|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|today|yesterday|tomorrow|last|next|ago|week|month|year)\b", re.I)


def strict_scene_schema(max_scenes: int, max_facts: int, *,
                        quote_evidence: bool = True) -> Mapping[str, Any]:
    """Schema for one strict extraction call.

    ``quote_evidence`` emits the exact-quote field ``q``.  It is 26% of
    extraction output bytes and only narrows the span inside a turn that ``r``
    already cites, which the projection re-derives from the fact value, so a
    budgeted run may drop it.
    """
    properties: dict[str, Any] = {
        "o": {"type": "string", "minLength": 1}, "p": {"type": "string", "minLength": 1},
        "v": {"type": "string", "minLength": 1},
        "g": {"type": "string", "minLength": 1},
        "n": {"type": "string", "enum": ["positive", "negative"]},
        "r": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string"}}}
    required = ["o", "p", "v", "g", "n", "r"]
    if quote_evidence:
        properties["q"] = {"type": "string", "minLength": 1, "maxLength": 160}
        required.append("q")
    fact = {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}
    scene = {"type": "object", "additionalProperties": False,
             "required": ["i", "f"], "properties": {
                 "i": {"type": "string"},
                 "f": {"type": "array", "maxItems": max_facts, "items": fact}}}
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


@dataclass(frozen=True, slots=True)
class ScenePacket:
    scene_id: str
    summary: str
    facts: tuple[SemanticFact, ...]
    unresolved: tuple[str, ...]
    fallback: bool = False


class QwenSemanticDistiller:
    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, client: Any | None = None) -> None:
        self.store, self.config, self.dataset_hash = store, config, dataset_hash
        strict_prompt = STRICT_PROMPT + (
            f" Return at most {config.models.semantic_max_facts_per_scene} highest-routing-value facts per scene.")
        prompt_material = (STRICT_PROMPT_VERSION + strict_prompt if
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
            self.config.models.semantic_budget_degrade_at)
        workers = min(16, self.config.models.max_concurrency, max(1, len(batches)))
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
        return int(text / CHARS_PER_TOKEN) + CALL_OVERHEAD_TOKENS + max_tokens

    def compress_many(self, memory_id: str, level: int,
                      requests: Sequence[tuple[str, Sequence[Mapping[str, Any]], int]]) -> tuple[Mapping[str, Any], ...]:
        workers = min(16, self.config.models.max_concurrency, max(1, len(requests)))
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
        payload = {"s": [{"i": scene.scene_id, "r": [
            {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp, "l": len(turn.raw_text),
             "t": turn.raw_text}
            for turn in scene.turns]} for scene in batch]}
        system = self._scene_prompt()
        response, _ = self._call(memory_id, "scene_semantic", system, payload, len(batch))
        objects = self._parse_objects(str(response.get("content", "")))
        root = objects[0] if objects else {}
        rows = root.get("s", root.get("scenes", [])) if isinstance(root, dict) else []
        if not isinstance(rows, list):
            rows = []
        by_scene = {str(row.get("i", row.get("scene_id"))): row for row in rows if isinstance(row, dict)}
        missing = [scene for scene in batch if scene.scene_id not in by_scene]
        if missing:
            repair_batches = [[scene] for scene in missing] if self.config.models.semantic_individual_repair else [missing]
            for repair_batch in repair_batches:
                repair_payload = {"repair": "Return complete valid JSON for only these missing scenes.",
                "s": [{"i": scene.scene_id, "r": [
                    {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp,
                     "l": len(turn.raw_text), "t": turn.raw_text}
                    for turn in scene.turns]} for scene in repair_batch]}
                repaired, _ = self._call(memory_id, "scene_semantic_repair", system,
                    repair_payload, len(repair_batch),
                    max_tokens=self.config.models.semantic_repair_output_tokens)
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
        payload, scene_aliases, turn_aliases = self._strict_payload(batch)
        output_cap = self.config.models.semantic_batch_output_tokens
        allowed, degrade = self._ledger().reserve(self._batch_estimate(batch, output_cap))
        if not allowed:
            # Budget exhausted: keep the scenes in the graph with deterministic
            # summaries rather than dropping the tail of the conversation.
            return [self._fallback(scene) for scene in batch]
        max_facts = self.config.models.semantic_max_facts_per_scene
        if degrade:
            max_facts = max(1, max_facts // 2)
        strict_prompt = STRICT_PROMPT + (
            f" Return at most {max_facts} highest-routing-value facts per scene.")
        response, usage = self._call(memory_id, "scene_semantic", strict_prompt, payload,
                                     len(batch), max_facts=max_facts)
        self._ledger().settle(self._batch_estimate(batch, output_cap),
                              int(usage.get("total_tokens", 0)))
        rows = self._strict_rows(response, scene_aliases, turn_aliases)
        by_scene = {str(row.get("i")): row for row in rows}
        missing = [scene for scene in batch if scene.scene_id not in by_scene]
        if missing and self.config.models.semantic_max_retries:
            for scene in missing:
                retry_payload, retry_scenes, retry_turns = self._strict_payload((scene,))
                repaired, _ = self._call(
                    memory_id, "scene_semantic_retry", strict_prompt, retry_payload, 1,
                    max_tokens=self.config.models.semantic_retry_output_tokens, retry_count=1)
                for row in self._strict_rows(repaired, retry_scenes, retry_turns):
                    by_scene[str(row.get("i"))] = row
        result = []
        for scene in batch:
            row = by_scene.get(scene.scene_id)
            result.append(self._validate_scene(scene, row) if row else self._fallback(scene))
        return result

    def _strict_payload(self, batch: Sequence[Any]) -> tuple[Mapping[str, Any], Mapping[str, str], Mapping[str, str]]:
        scene_aliases: dict[str, str] = {}
        turn_aliases: dict[str, str] = {}
        rows = []
        for scene_index, scene in enumerate(batch):
            scene_alias = f"s{scene_index}"; scene_aliases[scene_alias] = scene.scene_id
            turns = []
            for turn_index, turn in enumerate(scene.turns):
                alias = f"s{scene_index}t{turn_index}"; turn_aliases[alias] = turn.turn_id
                turns.append({"i": alias, "s": turn.speaker, "d": turn.timestamp,
                              "t": self._compact_turn(turn.raw_text)})
            rows.append({"i": scene_alias, "r": turns})
        return {"s": rows}, scene_aliases, turn_aliases

    def _strict_rows(self, response: Mapping[str, Any], scene_aliases: Mapping[str, str],
                     turn_aliases: Mapping[str, str]) -> list[Mapping[str, Any]]:
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
            if alias not in scene_aliases and alias in turn_aliases:
                alias = alias.split("t", 1)[0]
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
                for alias in source_fact.get("r", ()) if isinstance(source_fact.get("r"), list) else ():
                    turn_id = turn_aliases.get(str(alias))
                    if not turn_id:
                        continue
                    refs.append({"i": turn_id, "q": quote})
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

    def _validate_scene(self, scene: Any, row: Mapping[str, Any]) -> ScenePacket:
        turns = {turn.turn_id: turn for turn in scene.turns}
        facts = []
        fact_rows = row.get("f", row.get("facts", ()))
        for item in (fact_rows[:self.config.models.semantic_max_facts_per_scene]
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
                    start = turn.raw_text.casefold().find(quote.casefold())
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
            facts.append(SemanticFact(owner, predicate, value, value_type,
                                      scope,
                                      str(item.get("n", item.get("polarity", "positive"))),
                                      raw_time,
                                      min(1.0, max(0.0, float(item.get("c", item.get("confidence", 0.5))))), tuple(evidence)))
        summary = " ".join(str(row.get("m", row.get("summary", ""))).split()[
            :self.config.models.semantic_summary_tokens])
        if self.config.models.semantic_compile_summary:
            summary = self._compiled_summary(facts)
        if not summary:
            summary = " ".join(scene.summary.split()[:96])
        return ScenePacket(scene.scene_id, summary, tuple(facts),
                           self._strings(row.get("u", row.get("unresolved"))), False)

    @staticmethod
    def _compiled_summary(facts: Sequence[SemanticFact]) -> str:
        parts = []
        for fact in facts:
            text = f"{fact.owner} {fact.predicate} {fact.value}"
            if fact.time:
                text += f" {fact.time}"
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
    def _fallback(scene: Any) -> ScenePacket:
        return ScenePacket(scene.scene_id, " ".join(scene.summary.split()[:96]), (), ("llm_parse_fallback",), True)

    def _call(self, memory_id: str, stage: str, system: str, payload: Any,
              batch_size: int, max_tokens: int | None = None,
              retry_count: int = 0,
              max_facts: int | None = None) -> tuple[Mapping[str, Any], Mapping[str, int]]:
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
                    quote_evidence=self.config.models.semantic_quote_evidence)}}
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
            result = self.client.chat.completions.create(**request); message = result.choices[0].message
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
